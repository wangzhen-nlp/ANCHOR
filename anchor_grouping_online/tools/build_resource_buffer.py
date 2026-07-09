#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成单个资源缓冲文件（JSONL：每行一个 {"resource_type": ..., "data": ...} 资源，
紧凑 JSON、resource_type 置于行首便于加载方按需跳过），按 resource_type 区分内容：

- site_graph:         站点邻接图
- ne_graph:           NE 邻接图
- link_peer_index:    端口对端索引
- site_device_counts: 每个站点各 domain 的设备数量（由 ne_graph 派生）
- site_chains:        预计算站点上下游 hop 索引

默认读取过程：

- SYS_NE 读取 1 次（得到完整 NE 信息，并据此派生 nativeId->site_id 映射）
- SYS_SITE 读取 1 次（站点信息驻留内存，供 site 图与 ne 图共用）
- SYS_LINK 由 NE-NE、端口-端口、NE-端口三个生成器分别派生；当前 CSV/zip
  适配会分别读取链路输入，后续可替换成其它数据源直接生成三类关系
"""

import json
import os
import sys
import csv
import io
import zipfile
import argparse

from dataclasses import asdict, dataclass
from collections import defaultdict
from types import SimpleNamespace

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.peer_index_keys import make_key
from anchor_grouping_online.tools.progress_utils import ProgressBar
from anchor_grouping_online.tools.topology_resources import (
    NE_FILE,
    NE_NE_FILE,
    NE_PORT_FILE,
    PORT_PORT_FILE,
    RESOURCE_BUFFER_JSONL,
    SITE_FILE,
    SYS_LINK_DIR,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)
from anchor_grouping_online.tools.generate_site_chains import (
    build_site_chains_from_data,
    verify_cross_domain_constraints,
)
from anchor_grouping_online.tools.generate_site_pair_order_pairwise import build_pairwise_prediction

# 单文件缓冲产物的默认输出路径（JSONL：每行一个资源）
DEFAULT_BUFFER_OUTPUT = RESOURCE_BUFFER_JSONL

# 行级进度的刷新间隔（行数）；计数模式下每次 set 都会重绘，需要批量节流
PROGRESS_ROW_STEP = 5000

CSV_FILE_SUFFIXES = ('.csv', '.zip')

# site_chains 计算阶段实际会从 ne_graph 节点读取的字段（site_id/domain/link）。
_SITE_CHAINS_NE_FIELDS = ('site_id', 'domain', 'link')

# site_graph 中缺省站点信息（NE 引用了某站点但 SYS_SITE 中无该站点时使用）
_DEFAULT_SITE = {
    'site_name': '',
    'longitude': '',
    'latitude': '',
    'is_hub': False,
}


def _resource_buffer_pairwise_args():
    """固定 build_resource_buffer 的 pairwise 行为，不暴露或拼装 CLI flags。"""
    return SimpleNamespace(
        ne_graph="<resource_buffer: ne_graph>",
        output="<resource_buffer: site_pair_order_pairwise>",
        direction_margin=2.5,
        core_distance_penalty=2.0,
        non_bridge_margin_bonus=2.0,
        shared_neighbor_margin_bonus=0.5,
        max_shared_neighbor_bonus_count=3,
        anchor_score_ratio=0.85,
        max_anchor_sites_per_component=3,
        data_site_bonus=6.0,
        data_ne_weight=1.5,
        neighbor_weight=0.8,
        external_edge_weight=0.6,
        data_edge_weight=1.2,
        transmission_edge_weight=0.4,
        core_distance_weight=2.0,
        level_score_weight=0.8,
        base_score_weight=0.5,
        data_presence_weight=2.0,
        pair_data_weight=1.0,
        neighbor_direction_weight=0.4,
        leaf_bias_weight=1.2,
        min_level_gap=0.5,
        min_base_score_gap=1.0,
        min_neighbor_gap=1,
        leaf_neighbor_threshold=1,
        max_core_distance_delta=3,
        max_level_score_delta=6.0,
        max_base_score_delta=6.0,
        max_neighbor_delta=4,
        max_pair_domain_delta=4,
        smooth_alpha=0.5,
        smooth_iters=100,
        smooth_tol=0.0001,
        global_gap_threshold=1.0,
        global_gap_nonbridge_bonus=2.0,
        global_gap_shared_neighbor_bonus=0.5,
        # 跨类型连边方向约束（Data > Transmission > Ran 优先级高者为上行）：
        # 注入层级平滑、解除含约束端点的严格环块、强制直连对判向
        cross_domain_priority_constraint=True,
        constraint_level_gap=1.0,
        # 约束触发的环块解除：默认关闭（环块会把共边环/双归下游一并放开，
        # 解除范围无法正确圈定）；约束仅经直连对硬覆盖与势场投影生效
        constraint_ring_release=False,
        # 直连约束对硬覆盖：强制 upstream->downstream，去掉平行/反向
        constraint_hard_override=True,
        # 势场投影：约束注入层级平滑（抬上行/压下行+扩散），多跳约束的唯一作用通道
        constraint_level_projection=True,
        # 误连接预处理：Data 站点与 Trans+Ran(无 Data) 站点之间只有传输连边、
        # 无 Data-Ran 佐证时，剔除两者间全部传输类连边
        transmission_misconnection_filter=True,
        full_output=False,
        no_progress=True,
    )


# --------------------------------------------------------------------------- #
# 去重 / 校验辅助
# --------------------------------------------------------------------------- #
def _report_duplicate_detail(duplicates: dict, label: str) -> None:
    """打印重复 key 的明细（按重复次数降序）；duplicates 为空或 None 时不输出。"""
    if not duplicates:
        return
    items = sorted(duplicates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  检测到 {len(items)} 个{label}存在重复（已按出现顺序保留最后一条记录）:")
    for key, extra in items:
        print(f"    {key}: {extra + 1} 条")


def _parse_bool(value) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "是"}


# --------------------------------------------------------------------------- #
# 文件迭代：CSV / zip(内含CSV)
# --------------------------------------------------------------------------- #
def _iter_zip_csv(zip_path: str):
    """迭代 zip 包内所有 CSV 的记录。"""
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith('.csv'):
                continue
            with zf.open(name) as member:
                text = io.TextIOWrapper(
                    member, encoding='utf-8-sig', errors='replace', newline=''
                )
                yield from csv.DictReader(text)


def _iter_csv_or_zip_file(file_path: str):
    """迭代单个 .csv / .zip(内含 CSV) 文件的记录。"""
    lower = file_path.lower()
    if lower.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
            yield from csv.DictReader(f)
    elif lower.endswith('.zip'):
        yield from _iter_zip_csv(file_path)
    else:
        raise SystemExit(f"不支持的文件格式: {file_path}（支持 .csv/.zip）")


def iter_csv_records(input_path: str, progress: ProgressBar = None):
    """迭代 .csv / .zip(内含 CSV) 记录。

    input_path 可以是单个文件，也可以是包含这些文件的目录。目录内无匹配文件时
    报错退出。
    """
    if os.path.isdir(input_path):
        file_names = sorted(
            name for name in os.listdir(input_path)
            if name.lower().endswith(CSV_FILE_SUFFIXES)
        )
        if not file_names:
            raise SystemExit(f"目录中未找到 .csv/.zip 文件: {input_path}")
        for name in file_names:
            if progress is not None:
                progress.set_extra_text(name, force=True)
            yield from _iter_csv_or_zip_file(os.path.join(input_path, name))
    else:
        if progress is not None:
            progress.set_extra_text(os.path.basename(input_path), force=True)
        yield from _iter_csv_or_zip_file(input_path)


# --------------------------------------------------------------------------- #
# 记录生成器：把“从哪读取原始记录”与“如何去重构图”解耦。
# 下面生成器基于当前 CSV/zip 构造，也可换成其它数据源（DB / 接口等）。
# 各类 vid / linkLayer 的归一化必须在生成器阶段完成，消费端不再兜底归一化。
# --------------------------------------------------------------------------- #
def _iter_csv_records_with_progress(input_path: str, label: str):
    """在 iter_csv_records 之上叠加行级进度显示，逐条产出原始记录。"""
    progress = ProgressBar(0, label)
    row_count = 0
    for row in iter_csv_records(input_path, progress):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        yield row
    progress.set(row_count)
    progress.close()


def _normalize_ne_vid_for_generator(value) -> str:
    return str(value or '').strip().upper()


def _normalize_site_vid_for_generator(value) -> str:
    return str(value or '').strip().upper()


def _normalize_link_layer_for_generator(value) -> str:
    return str(value or '').strip().upper()


def iter_ne_csv_records(data_dir: str = SYS_NE_DIR):
    """从当前 SYS_NE CSV 目录构造 NE 记录生成器，供 load_ne_info 消费。

    仅产出规范化后的字段（其余原始列被丢弃）：
        vid / domain / typeId / networkType / name / vender / siteId
    """
    for row in _iter_csv_records_with_progress(data_dir, "  读取NE记录"):
        yield {
            'vid': _normalize_ne_vid_for_generator(row.get('nativeId', '')),
            'domain': row.get('domain', ''),
            'typeId': row.get('typeId', ''),
            'networkType': row.get('network_type', ''),
            'name': row.get('name', ''),
            'vender': row.get('manufacturer', ''),
            'siteId': _normalize_site_vid_for_generator(row.get('ne_site_id', '')),
        }


def iter_site_csv_records(data_dir: str = SYS_SITE_DIR):
    """从当前 SYS_SITE CSV 目录构造站点记录生成器，供 load_site_info 消费。

    仅产出规范化后的字段（其余原始列被丢弃）：
        vid / name / longitude / latitude / isHub
    """
    for row in _iter_csv_records_with_progress(data_dir, "  读取站点记录"):
        yield {
            'vid': _normalize_site_vid_for_generator(row.get('site_id', '')),
            'name': row.get('name', ''),
            'longitude': row.get('longitude', ''),
            'latitude': row.get('latitude', ''),
            'isHub': row.get('is_hub', ''),
        }


def _make_port_vid(ne_id: str, port_name: str) -> str:
    """用 NE ID + 端口名构造全局唯一端口 VID；格式与对端索引 key 保持一致。"""
    return make_key(ne_id, port_name)


def iter_ne_ne_link(link_input: str = SYS_LINK_DIR):
    """从 SYS_LINK 构造 NE-NE 链路记录。

    输出字段：
        src_vid / dst_vid / linkLayer
    """
    for row in _iter_csv_records_with_progress(link_input, "  读取NE-NE链路"):
        yield {
            'src_vid': _normalize_ne_vid_for_generator(
                _get_record_value(row, 'a_end_ne_nativeId', "a_end_ne_nativeId(')")
            ),
            'dst_vid': _normalize_ne_vid_for_generator(
                _get_record_value(row, 'z_end_ne_nativeId', "z_end_ne_nativeId(')")
            ),
            'linkLayer': _normalize_link_layer_for_generator(row.get('link_layer', '')),
        }


def iter_port_port(link_input: str = SYS_LINK_DIR):
    """从 SYS_LINK 构造端口-端口链路记录。

    输出字段：
        src_vid / dst_vid / linkLayer

    src_vid 和 dst_vid 使用 NE ID + 端口名构造的端口 VID，而非裸端口名，避免不同
    NE 上的同名端口互相冲突。两端 NE 与端口名均非空时才产出记录。
    """
    for row in _iter_csv_records_with_progress(link_input, "  读取端口-端口链路"):
        a_ne = _get_record_value(row, 'a_end_ne_nativeId', "a_end_ne_nativeId(')")
        z_ne = _get_record_value(row, 'z_end_ne_nativeId', "z_end_ne_nativeId(')")
        a_port = _get_record_value(row, 'a_end_port_name')
        z_port = _get_record_value(row, 'z_end_port_name')
        if not (a_ne and z_ne and a_port and z_port):
            continue
        yield {
            'src_vid': _make_port_vid(a_ne, a_port),
            'dst_vid': _make_port_vid(z_ne, z_port),
            'linkLayer': _normalize_link_layer_for_generator(row.get('link_layer', '')),
        }


def iter_ne_port_link(link_input: str = SYS_LINK_DIR):
    """从 SYS_LINK 构造 NE-端口归属关系记录。

    输出字段：
        src_vid / dst_vid

    src_vid 为 NE VID；dst_vid 为由 NE ID + 端口名构造的端口 VID。一条 SYS_LINK
    记录最多产出 a/z 两条归属关系。
    """
    for row in _iter_csv_records_with_progress(link_input, "  读取NE-端口关系"):
        a_ne = _get_record_value(row, 'a_end_ne_nativeId', "a_end_ne_nativeId(')")
        z_ne = _get_record_value(row, 'z_end_ne_nativeId', "z_end_ne_nativeId(')")
        a_port = _get_record_value(row, 'a_end_port_name')
        z_port = _get_record_value(row, 'z_end_port_name')
        if a_ne and a_port:
            yield {
                'src_vid': _normalize_ne_vid_for_generator(a_ne),
                'dst_vid': _make_port_vid(a_ne, a_port),
            }
        if z_ne and z_port:
            yield {
                'src_vid': _normalize_ne_vid_for_generator(z_ne),
                'dst_vid': _make_port_vid(z_ne, z_port),
            }


# --------------------------------------------------------------------------- #
# online 版记录生成器：CSV/zip 原始行原样透传，不做任何改动（不筛选列、不归一化、
# 不构造 VID），假定数据源已直接产出消费端期望的字段与值。
# --------------------------------------------------------------------------- #
def iter_ne_records_online(input_path: str = NE_FILE):
    """从 online CSV 原样透传 NE 记录，供 load_ne_info 消费。

    期望列：vid / domain / typeId / networkType / name / vender / siteId
    """
    yield from _iter_csv_records_with_progress(input_path, "  读取NE记录")


def iter_site_records_online(input_path: str = SITE_FILE):
    """从 online CSV 原样透传站点记录，供 load_site_info 消费。

    期望列：vid / name / longitude / latitude / isHub
    """
    yield from _iter_csv_records_with_progress(input_path, "  读取站点记录")


def iter_ne_ne_link_online(input_path: str = NE_NE_FILE):
    """从 online CSV 原样透传 NE-NE 链路记录（src_vid / dst_vid 为两端 NE VID）。

    期望列：src_vid / dst_vid / linkLayer
    """
    yield from _iter_csv_records_with_progress(input_path, "  读取NE-NE链路")


def iter_port_port_online(input_path: str = PORT_PORT_FILE):
    """从 online CSV 原样透传端口-端口链路记录（src_vid / dst_vid 为两端端口 VID）。

    期望列：src_vid / dst_vid / linkLayer
    """
    yield from _iter_csv_records_with_progress(input_path, "  读取端口-端口链路")


def iter_ne_port_link_online(input_path: str = NE_PORT_FILE):
    """从 online CSV 原样透传 NE-端口归属关系记录（src_vid 为 NE VID，dst_vid 为端口 VID）。

    期望列：src_vid / dst_vid
    """
    yield from _iter_csv_records_with_progress(input_path, "  读取NE-端口关系")


# --------------------------------------------------------------------------- #
# 基础数据加载（去重取最新）
# --------------------------------------------------------------------------- #
def load_ne_info(records, report_duplicates: bool = False) -> dict:
    """从记录生成器加载 NE 信息；同 vid 冲突时按出现顺序后者直接覆盖前者，不做字段合并。

    records 为逐条产出规范化 NE 记录（dict）的可迭代对象，可由 iter_ne_csv_records 从当前
    SYS_NE CSV 构造，也可替换成其它数据源；期望字段见 iter_ne_csv_records（NE ID 为 vid）。

    Returns:
        {vid: {domain, type, network_type, name, manufacturer, site_id}}
    """
    result = {}
    duplicates = defaultdict(int) if report_duplicates else None

    row_count = 0
    for row in records:
        row_count += 1
        vid = row.get('vid') or ''
        if not vid:
            continue
        vid = sys.intern(vid)
        if duplicates is not None and vid in result:
            duplicates[vid] += 1
        # 低基数/高重复字段做 intern：同值只留一份对象，大幅压缩百万级记录的重复占用。
        # name 为高基数（设备名各异），不 intern 以免污染 intern 表。
        result[vid] = {
            'domain': sys.intern((row.get('domain') or '').strip()),
            'type': sys.intern((row.get('typeId') or '').strip()),
            'network_type': sys.intern((row.get('networkType') or '').strip()),
            'name': (row.get('name') or '').strip(),
            'manufacturer': sys.intern((row.get('vender') or '').strip()),
            'site_id': sys.intern(row.get('siteId') or ''),
        }

    print(f"  读取 {row_count} 行，去重后 {len(result)} 个NE")
    _report_duplicate_detail(duplicates, 'vid')
    return result


def load_site_info(records, report_duplicates: bool = False) -> dict:
    """从记录生成器加载站点信息；同 site_id 冲突时按出现顺序后者直接替换前者，不做字段合并。

    records 为逐条产出规范化站点记录（dict）的可迭代对象，可由 iter_site_csv_records 从当前
    SYS_SITE CSV 构造，也可替换成其它数据源；期望字段见 iter_site_csv_records（站点 ID 为 vid）。

    Returns:
        {site_id: {site_name, longitude, latitude, is_hub}}
    """
    result = {}
    duplicates = defaultdict(int) if report_duplicates else None

    row_count = 0
    for row in records:
        row_count += 1
        site_id = row.get('vid') or ''
        if not site_id:
            continue
        site_id = sys.intern(site_id)
        if duplicates is not None and site_id in result:
            duplicates[site_id] += 1
        # site_name 与经纬度高基数不 intern。
        result[site_id] = {
            'site_name': (row.get('name') or '').strip(),
            'longitude': (row.get('longitude') or '').strip(),
            'latitude': (row.get('latitude') or '').strip(),
            'is_hub': _parse_bool(row.get('isHub', '')),
        }

    print(f"  读取 {row_count} 行，去重后 {len(result)} 个站点")
    _report_duplicate_detail(duplicates, 'site_id')
    return result


def _get_record_value(record, *field_names):
    for field_name in field_names:
        value = str(record.get(field_name, "") or "").strip()
        if value:
            return value
    return ""


# --------------------------------------------------------------------------- #
# 端口对端索引
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PeerDevice:
    ne_native_id: str
    port_vid: str = ""


def _json_default(obj):
    """json.dump 的回退序列化器：把 PeerDevice 即时转 dict，避免预先整体物化一份。"""
    if isinstance(obj, PeerDevice):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# 图构建
# --------------------------------------------------------------------------- #
def _add_bidirectional_edge(graph: dict, a: str, b: str, link_type: str) -> None:
    """在 graph 中登记 a->b 的有向边，并维护 b 侧的反向标记。

    首次出现记 '->'/'<-'，两个方向都出现则升级为 '<->'。
    """
    fwd = graph[a][b]
    if link_type in fwd:
        if fwd[link_type] == '<-':
            fwd[link_type] = '<->'
    else:
        fwd[link_type] = '->'
    rev = graph[b][a]
    if link_type in rev:
        if rev[link_type] == '->':
            rev[link_type] = '<->'
    else:
        rev[link_type] = '<-'


def build_graphs_from_relations(ne_ne_links, port_port_links, ne_port_links, ne_info: dict):
    """消费三类拓扑关系，构建 site 邻接图、NE 邻接图与端口对端索引。

    port_vid 是不透明的全局端口 ID。当前 SYS_LINK 生成器用 NE+port 构造它，
    迁移到其它数据源时只要三类生成器对同一端口使用同一个 port_vid 即可。
    输入记录的各类 vid / linkLayer 必须已在生成器阶段完成归一化。

    Returns:
        (site_links, ne_links, peer_index, stats)
    """
    site_links = defaultdict(lambda: defaultdict(dict))
    ne_links = defaultdict(lambda: defaultdict(dict))
    peer_index = {}

    stats = {
        'ne_link_count': 0,
        'ne_mapped_count': 0,
        'site_link_count': 0,
        'site_mapped_count': 0,
        'port_link_count': 0,
        'port_mapped_count': 0,
    }

    for record in ne_ne_links:
        src_ne = record.get('src_vid') or ''
        dst_ne = record.get('dst_vid') or ''
        link_type = record.get('linkLayer') or ''

        if not (src_ne and dst_ne):
            continue

        # NE 邻接图
        stats['ne_link_count'] += 1
        stats['ne_mapped_count'] += 1
        _add_bidirectional_edge(ne_links, src_ne, dst_ne, link_type)

        # 站点邻接图：两端 NE 都能映射到站点时才登记
        stats['site_link_count'] += 1
        src_site = ne_info.get(src_ne, {}).get('site_id')
        dst_site = ne_info.get(dst_ne, {}).get('site_id')
        if src_site and dst_site:
            stats['site_mapped_count'] += 1
            _add_bidirectional_edge(site_links, src_site, dst_site, link_type)

    port_to_ne = {}
    for record in ne_port_links:
        ne_vid = record.get('src_vid') or ''
        port_vid = record.get('dst_vid') or ''
        if ne_vid and port_vid:
            port_to_ne[port_vid] = sys.intern(ne_vid)

    for record in port_port_links:
        src_port_vid = record.get('src_vid') or ''
        dst_port_vid = record.get('dst_vid') or ''
        if not (src_port_vid and dst_port_vid):
            continue

        stats['port_link_count'] += 1
        src_ne = port_to_ne.get(src_port_vid)
        dst_ne = port_to_ne.get(dst_port_vid)
        if not (src_ne and dst_ne):
            continue

        stats['port_mapped_count'] += 1
        peer_index[src_port_vid] = PeerDevice(
            ne_native_id=dst_ne,
            port_vid=dst_port_vid,
        )
        peer_index[dst_port_vid] = PeerDevice(
            ne_native_id=src_ne,
            port_vid=src_port_vid,
        )

    return site_links, ne_links, peer_index, stats


def assemble_site_graph(site_info: dict, site_links: dict) -> dict:
    """原地把站点信息补成 site_graph：给每个站点挂 link 字段，返回 site_info 自身。

    直接复用 site_info 外层字典、并把 site_links 的邻接字典逐站点 pop 后直接挂载，
    避免拷贝外层字典与邻接字典。顶层站点保持 site_info 插入顺序，缺省站点追加在后。
    """
    for site_id, site_data in site_info.items():
        site_data['link'] = site_links.pop(site_id, {})
    # 仅出现在邻接里、SYS_SITE 中没有的站点，用缺省信息补齐
    for site_id in list(site_links):
        site_data = dict(_DEFAULT_SITE)
        site_data['link'] = site_links.pop(site_id)
        site_info[site_id] = site_data
    return site_info


def assemble_ne_graph(ne_info: dict, site_info: dict, ne_links: dict) -> dict:
    """原地把 NE 信息补成 ne_graph：补站点字段与 link 字段，返回 ne_info 自身。

    直接复用 ne_info 外层字典与各 NE 内层字典，把 ne_links 的邻接字典逐节点 pop 后
    直接挂载，避免创建 ne_graph 外层字典和 set 并集临时集合。调用方此后不应再使用
    ne_info。顶层 NE 保持 ne_info 插入顺序，缺省 NE 追加在后。
    """
    for ne_id, ne_data in ne_info.items():
        site_id = ne_data.get('site_id', '')
        site_data = site_info.get(site_id, {})
        ne_data['site_id'] = site_id
        ne_data['site_name'] = site_data.get('site_name', '')
        ne_data['longitude'] = site_data.get('longitude', '')
        ne_data['latitude'] = site_data.get('latitude', '')
        ne_data['link'] = ne_links.pop(ne_id, {})
    # 仅出现在邻接里、SYS_NE 中没有的 NE，用缺省字段补齐
    for ne_id in list(ne_links):
        ne_info[ne_id] = {
            'site_id': '',
            'site_name': '',
            'longitude': '',
            'latitude': '',
            'link': ne_links.pop(ne_id),
        }
    return ne_info


def build_site_device_counts(ne_graph: dict) -> dict:
    """由 ne_graph 派生每个站点各 domain 的设备数量（站点画像）。

    与匹配运行时构建 ne->domain 的口径一致：site_id 与 domain 都非空才计数。

    Returns:
        {site_id: {domain: count, ...}}
    """
    counts = defaultdict(lambda: defaultdict(int))
    for ne_data in ne_graph.values():
        site_id = ne_data.get('site_id', '')
        domain = ne_data.get('domain', '')
        if site_id and domain:
            counts[site_id][domain] += 1
    return {site_id: dict(domains) for site_id, domains in counts.items()}


def build_site_chains_field(ne_graph: dict) -> dict:
    """在内存中生成 site_chains.json 对应的数据，产出可作为缓冲字段的站点链路。

    等价命令：
        generate_site_pair_order_pairwise.py
        generate_site_chains.py --restrict-relation --ne-graph <ne_graph>

    两步都直接吃内存里的 ne_graph，省去中间落盘再读盘。
    """
    prediction = build_pairwise_prediction(
        ne_graph,
        _resource_buffer_pairwise_args(),
        show_progress=False,
    )
    site_chains = build_site_chains_from_data(
        prediction,
        ne_graph=ne_graph,
        prediction_label="<resource_buffer: pairwise prediction>",
        ne_graph_label="<resource_buffer: ne_graph>",
        restrict_relation=True,
        show_progress=False,
    )
    # 跨类型方向约束校验：只统计不修复，结果并入 meta 供落盘后核对
    constraints = prediction.get("cross_domain_constraints")
    if constraints:
        site_chains["meta"]["cross_domain_constraint_check"] = (
            verify_cross_domain_constraints(site_chains.get("sites", {}), constraints)
        )
    # 误连接预处理计数随 site_chains meta 落盘，便于核对剔除规模
    if prediction["meta"].get("transmission_misconnection_filter"):
        site_chains["meta"]["transmission_misconnection_pair_count"] = (
            prediction["meta"].get("transmission_misconnection_pair_count", 0)
        )
    return site_chains


# 紧凑 JSON 分隔符：去掉默认的空格，进一步压缩体积
_COMPACT_SEPARATORS = (',', ':')


def _write_resource_line(output_file, resource_type: str, data) -> None:
    """写出一行资源记录：{"resource_type": <type>, "data": <data>}\\n（紧凑 JSON）。

    resource_type 置于行首，加载方可先按行前缀判断类型、跳过不需要的资源，
    无需解析整行 data，提升按需加载效率。
    """
    output_file.write('{"resource_type":')
    json.dump(resource_type, output_file, ensure_ascii=False)
    output_file.write(',"data":')
    json.dump(
        data,
        output_file,
        ensure_ascii=False,
        separators=_COMPACT_SEPARATORS,
        default=_json_default,
    )
    output_file.write('}\n')


def build_resource_buffer(ne_records, site_records, ne_ne_records, port_port_records,
                          ne_port_records, output_path, report_duplicates: bool = False):
    """消费资源记录生成器，产出单个资源缓冲文件（JSONL）。

    ne_records / site_records / ne_ne_records / port_port_records / ne_port_records
    均为逐条产出规范化记录（dict）的可迭代对象，是本工具与“数据从哪来”之间的
    唯一耦合点。默认由当前 CSV 构造，后续可整体替换成 DB / 接口等其它数据源。
    消费端假定这些记录中的 vid / linkLayer 已归一化。
    """
    # 1) NE 读取一次，得到完整 NE 信息（其中含 site_id）
    print("加载NE信息...")
    ne_info = load_ne_info(ne_records, report_duplicates)
    print(f"  NE数量: {len(ne_info)}")

    # 2) 站点读取一次
    print("\n加载站点信息...")
    site_info = load_site_info(site_records, report_duplicates)

    # 3) 消费三类拓扑关系，产出三张图
    print("\n生成传播图与对端索引...")
    site_links, ne_links, peer_index, stats = build_graphs_from_relations(
        ne_ne_records,
        port_port_records,
        ne_port_records,
        ne_info=ne_info,
    )
    print(f"  站点图: 处理 {stats['site_link_count']} 条链路，成功映射 {stats['site_mapped_count']} 条")
    print(f"  NE图:   处理 {stats['ne_link_count']} 条链路，成功映射 {stats['ne_mapped_count']} 条")
    print(f"  端口对端: 处理 {stats['port_link_count']} 条链路，成功映射 {stats['port_mapped_count']} 条")

    # 4) 组装为单个缓冲对象，用字段区分不同内容后一次性写出
    site_graph = assemble_site_graph(site_info, site_links)
    ne_graph = assemble_ne_graph(ne_info, site_info, ne_links)
    site_device_counts = build_site_device_counts(ne_graph)

    # 保存汇总值后释放组装输入；ne_info/ne_graph 共享内层 NE dict，但外层 dict 可回收。
    site_graph_count = len(site_graph)
    site_neighbor_total = sum(len(v['link']) for v in site_graph.values())
    site_neighbor_max = max((len(v['link']) for v in site_graph.values()), default=0)
    ne_graph_count = len(ne_graph)
    ne_with_site_count = sum(1 for ne in ne_graph.values() if ne.get('site_id'))
    peer_index_count = len(peer_index)
    site_device_count = len(site_device_counts)
    del site_links, ne_links, ne_info, site_info

    # 逐资源写 JSONL（每行一个 {"resource_type", "data"}）：site_chains 只依赖 ne_graph，
    # 先写并释放其余大资源，避免 pairwise/site_chains 的临时结构与之叠加成峰值。
    with open(output_path, 'w', encoding='utf-8') as f:
        _write_resource_line(f, 'site_graph', site_graph)
        del site_graph

        _write_resource_line(f, 'ne_graph', ne_graph)

        _write_resource_line(f, 'link_peer_index', peer_index)
        del peer_index

        _write_resource_line(f, 'site_device_counts', site_device_counts)
        del site_device_counts

        # ne_graph 已写盘；site_chains 仅读 site_id/domain/link。删除字段不会让 Python dict
        # 的哈希表缩容，因此逐节点重建小字典，真正释放 name/经纬度等值及空槽容量。
        for ne_id, ne_data in ne_graph.items():
            ne_graph[ne_id] = {
                field: ne_data[field]
                for field in _SITE_CHAINS_NE_FIELDS
                if field in ne_data
            }

        # 由 ne_graph 派生站点链路（pairwise 方向先验 + restrict-relation 裁剪）
        print("\n生成站点链路(site_chains)...")
        site_chains = build_site_chains_field(ne_graph)
        _write_resource_line(f, 'site_chains', site_chains)

    # 汇总
    print(f"\n生成文件: {output_path}")

    print(f"  [site_graph] 站点数: {site_graph_count}")
    if site_graph_count:
        print(f"    平均邻居站点数: {site_neighbor_total/site_graph_count:.1f}")
        print(f"    最大邻居站点数: {site_neighbor_max}")

    print(f"  [ne_graph] NE数: {ne_graph_count}")
    if ne_graph_count:
        print(
            f"    有站点信息的NE: {ne_with_site_count} "
            f"({ne_with_site_count/ne_graph_count*100:.1f}%)"
        )

    print(f"  [link_peer_index] 对端索引记录数: {peer_index_count}")

    print(f"  [site_device_counts] 站点画像数: {site_device_count}")

    site_chains_meta = site_chains.get('meta', {})
    print(f"  [site_chains] 站点数: {site_chains_meta.get('site_count', len(site_chains.get('sites', {})))}")
    print(f"    下游可达关系数: {site_chains_meta.get('total_downstream_relations', 0)}")
    print(f"    双向直接边数: {site_chains_meta.get('total_bidirectional_edges', 0)}")

    if 'transmission_misconnection_pair_count' in site_chains_meta:
        print(
            f"    误连接剔除站点对数: "
            f"{site_chains_meta['transmission_misconnection_pair_count']}"
        )

    constraint_check_stats = (
        site_chains_meta.get('cross_domain_constraint_check') or {}
    ).get('stats') or {}
    if constraint_check_stats:
        print(
            f"    跨类型约束: {constraint_check_stats.get('constraint_count', 0)} 条 "
            f"(直连满足 {constraint_check_stats.get('satisfied_direct_count', 0)}, "
            f"多跳满足 {constraint_check_stats.get('satisfied_multi_hop_count', 0)}, "
            f"不可达 {constraint_check_stats.get('unreachable_count', 0)})"
        )
        print(
            f"    约束违例: 反向 {constraint_check_stats.get('reverse_violation_count', 0)}, "
            f"平行 {constraint_check_stats.get('bidirectional_violation_count', 0)}"
        )
        reverse_violations = [
            violation
            for violation in (
                site_chains_meta.get('cross_domain_constraint_check') or {}
            ).get('violations', [])
            if violation.get('type') == 'reverse'
        ]
        if reverse_violations:
            print("    反向违例明细:")
            for violation in reverse_violations:
                print(
                    f"      {violation.get('upstream_site')} -> "
                    f"{violation.get('downstream_site')} "
                    f"(最终反向 hop={violation.get('hop')})"
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "一次性生成单个资源缓冲文件（site_graph / ne_graph / "
            "link_peer_index / site_device_counts / site_chains）"
        )
    )
    parser.add_argument(
        "--ne-dir",
        default=SYS_NE_DIR,
        help=f"SYS_NE 数据目录，默认: {resource_display('SYS_NE_20260525')}",
    )
    parser.add_argument(
        "--site-dir",
        default=SYS_SITE_DIR,
        help=f"SYS_SITE 数据目录，默认: {resource_display('SYS_SITE_20260525')}",
    )
    parser.add_argument(
        "--link-input",
        default=SYS_LINK_DIR,
        help=(
            "链路输入，支持 .csv/.zip(内含CSV) 文件或包含这些文件的目录，"
            f"默认: {resource_display('SYS_LINK_20260525')}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_BUFFER_OUTPUT,
        help=f"输出缓冲文件（JSONL，每行一个 {{resource_type,data}} 资源），默认: {resource_display('resource_buffer.jsonl')}",
    )
    parser.add_argument(
        "--report-duplicates",
        action="store_true",
        help="打印 NE/站点/链路 中重复 ID 的明细（默认仅汇总，不打印明细）",
    )
    args = parser.parse_args()

    # 唯一耦合点：从当前 CSV 构造资源记录生成器。要换数据源，只需在这里换成
    # 产出同结构 dict 记录的生成器，build_resource_buffer 无需改动。
    build_resource_buffer(
        ne_records=iter_ne_csv_records(args.ne_dir),
        site_records=iter_site_csv_records(args.site_dir),
        ne_ne_records=iter_ne_ne_link(args.link_input),
        port_port_records=iter_port_port(args.link_input),
        ne_port_records=iter_ne_port_link(args.link_input),
        output_path=args.output,
        report_duplicates=args.report_duplicates,
    )


if __name__ == "__main__":
    main()
