#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成单个资源缓冲文件（JSONL：每行一个 {"resource_type": ..., "data": ...} 资源，
紧凑 JSON、resource_type 置于行首便于加载方按需跳过），按 resource_type 区分内容：

- site_graph:         站点邻接图
- ne_graph:           NE 邻接图
- link_peer_index:    端口对端索引
- site_device_counts: 每个站点各 domain 的设备数量（由 ne_graph 派生）

读取过程共享，避免重复 IO：

- SYS_NE 读取 1 次（得到完整 NE 信息，并据此派生 nativeId->site_id 映射）
- SYS_SITE 读取 1 次（站点信息驻留内存，供 site 图与 ne 图共用）
- 链路读取 + 去重 1 次（单趟遍历同时聚合 site 邻接、ne 邻接与端口对端索引）
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

from fault_grouping_official.peer_index_keys import make_key
from fault_grouping_official.tools.progress_utils import ProgressBar
from fault_grouping_official.tools.topology_resources import (
    RESOURCE_BUFFER_JSONL,
    SYS_LINK_DIR,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)
from fault_grouping_official.tools.generate_site_chains import build_site_chains_from_data
from fault_grouping_official.tools.generate_site_pair_order_pairwise import build_pairwise_prediction

# 单文件缓冲产物的默认输出路径（JSONL：每行一个资源）
DEFAULT_BUFFER_OUTPUT = RESOURCE_BUFFER_JSONL

# 行级进度的刷新间隔（行数）；计数模式下每次 set 都会重绘，需要批量节流
PROGRESS_ROW_STEP = 5000

CSV_FILE_SUFFIXES = ('.csv', '.zip')

# 链路记录的去重键字段，按优先级取第一个非空值
LINK_KEY_FIELDS = ('nativeId', "nativeId(')", 'resId', 'source_uuid')

# site_chains 计算阶段实际会从 ne_graph 节点读取的字段（site_id/domain/link）。
_SITE_CHAINS_NE_FIELDS = ('site_id', 'domain', 'link')

# site_graph 中缺省站点信息（NE 引用了某站点但 SYS_SITE 中无该站点时使用）
_DEFAULT_SITE = {
    'site_name': '',
    'site_type': '',
    'longitude': '',
    'latitude': '',
    'region_id': '',
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
        full_output=False,
        no_progress=True,
    )


# --------------------------------------------------------------------------- #
# 去重 / 校验辅助
# --------------------------------------------------------------------------- #
def _require_last_modified(record: dict, row_number: int, source: str) -> int:
    """校验并解析 last_Modified（毫秒级 Unix 时间戳，形如 1758256413325）；缺失或格式不符直接报错退出。"""
    raw = record.get('last_Modified')
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in ("", None):
        raise SystemExit(f"{source} 第 {row_number} 行缺少 last_Modified 字段: {record}")
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise SystemExit(
            f"{source} 第 {row_number} 行 last_Modified 格式错误: {raw!r}"
            f"（要求为毫秒级 Unix 时间戳，形如 1758256413325）"
        )


def _report_duplicate_detail(duplicates: dict, label: str) -> None:
    """打印重复 key 的明细（按重复次数降序）；duplicates 为空或 None 时不输出。"""
    if not duplicates:
        return
    items = sorted(duplicates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  检测到 {len(items)} 个{label}存在重复（已保留 last_Modified 最新记录）:")
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


def iter_csv_records(input_path: str, progress: ProgressBar = None, *,
                     name_keyword: str = None, require: bool = False):
    """迭代 .csv / .zip(内含 CSV) 记录。

    input_path 可以是单个文件，也可以是包含这些文件的目录；目录模式下 name_keyword
    非空时只取文件名含该关键字的文件。目录内无匹配文件时：require=True 报错退出，
    否则仅打印警告并跳过。
    """
    if os.path.isdir(input_path):
        file_names = sorted(
            name for name in os.listdir(input_path)
            if name.lower().endswith(CSV_FILE_SUFFIXES)
            and (name_keyword is None or name_keyword in name)
        )
        if not file_names:
            scope = f"文件名含 {name_keyword} 的 " if name_keyword else ""
            message = f"目录中未找到{scope}.csv/.zip 文件: {input_path}"
            if require:
                raise SystemExit(message)
            print(f"警告: {message}")
            return
        for name in file_names:
            if progress is not None:
                progress.set_extra_text(name, force=True)
            yield from _iter_csv_or_zip_file(os.path.join(input_path, name))
    else:
        if progress is not None:
            progress.set_extra_text(os.path.basename(input_path), force=True)
        yield from _iter_csv_or_zip_file(input_path)


# --------------------------------------------------------------------------- #
# 基础数据加载（去重取最新）
# --------------------------------------------------------------------------- #
def load_ne_from_csv(data_dir: str = SYS_NE_DIR, report_duplicates: bool = False) -> dict:
    """从 SYS_NE 加载 NE 信息；同 nativeId 按 last_Modified 取最新记录，不做字段合并。

    Returns:
        {nativeId: {domain, type, network_type, name, manufacturer, region_id, site_id, running_status}}
    """
    records = {}
    duplicates = defaultdict(int) if report_duplicates else None

    progress = ProgressBar(0, "  读取NE记录")
    row_count = 0
    for row in iter_csv_records(data_dir, progress, name_keyword='SYS_NE'):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        last_modified = _require_last_modified(row, row_count, 'SYS_NE')
        nativeId = sys.intern((row.get('nativeId') or '').strip().upper())
        if not nativeId:
            continue
        existing = records.get(nativeId)
        if existing is not None:
            if duplicates is not None:
                duplicates[nativeId] += 1
            if last_modified <= existing[0]:
                continue
        # 低基数/高重复字段做 intern：同值只留一份对象，大幅压缩百万级记录的重复占用。
        # name 为高基数（设备名各异），不 intern 以免污染 intern 表。
        incoming = {
            'domain': sys.intern((row.get('domain') or '').strip()),
            'type': sys.intern((row.get('typeId') or '').strip()),
            'network_type': sys.intern((row.get('network_type') or '').strip()),
            'name': (row.get('name') or '').strip(),
            'manufacturer': sys.intern((row.get('manufacturer') or '').strip()),
            'region_id': sys.intern((row.get('regionId1') or '').strip()),
            'site_id': sys.intern((row.get('ne_site_id') or '').strip().upper()),
            'running_status': sys.intern((row.get('running_status') or '').strip())
        }
        records[nativeId] = (last_modified, incoming)
    progress.set(row_count)
    progress.close()

    # 只替换 value 不改变 key 与插入顺序，避免同时驻留第二个等大的外层 dict。
    for key, (_, payload) in records.items():
        records[key] = payload
    print(f"  读取 {row_count} 行，去重后 {len(records)} 个NE")
    _report_duplicate_detail(duplicates, 'nativeId')
    return records


def load_site_info(data_dir: str = SYS_SITE_DIR, report_duplicates: bool = False) -> dict:
    """从 SYS_SITE 加载站点信息；同 site_id 按 last_Modified 取最新记录，不做字段合并。

    Returns:
        {site_id: {site_name, site_type, longitude, latitude, region_id, is_hub}}
    """
    records = {}
    duplicates = defaultdict(int) if report_duplicates else None

    progress = ProgressBar(0, "  读取站点记录")
    row_count = 0
    for row in iter_csv_records(data_dir, progress, name_keyword='SYS_SITE'):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        last_modified = _require_last_modified(row, row_count, 'SYS_SITE')
        site_id = sys.intern((row.get('site_id') or '').strip().upper())
        if not site_id:
            continue
        existing = records.get(site_id)
        if existing is not None:
            if duplicates is not None:
                duplicates[site_id] += 1
            if last_modified <= existing[0]:
                continue
        # site_type/region_id 低基数做 intern；site_name 与经纬度高基数不 intern。
        incoming = {
            'site_name': (row.get('name') or '').strip(),
            'site_type': sys.intern((row.get('site_type') or '').strip()),
            'longitude': (row.get('longitude') or '').strip(),
            'latitude': (row.get('latitude') or '').strip(),
            'region_id': sys.intern((row.get('region_id') or '').strip()),
            'is_hub': _parse_bool(row.get('is_hub', '')),
        }
        records[site_id] = (last_modified, incoming)
    progress.set(row_count)
    progress.close()

    # 只替换 value 不改变 key 与插入顺序，避免同时驻留第二个等大的外层 dict。
    for key, (_, payload) in records.items():
        records[key] = payload
    print(f"  读取 {row_count} 行，去重后 {len(records)} 个站点")
    _report_duplicate_detail(duplicates, 'site_id')
    return records


def _link_dedup_key(record) -> str:
    """取链路去重键：按 LINK_KEY_FIELDS 优先级返回第一个非空值，全空返回 ''。"""
    for field in LINK_KEY_FIELDS:
        key = (record.get(field) or '').strip()
        if key:
            return key
    return ''


def _get_record_value(record, *field_names):
    for field_name in field_names:
        value = str(record.get(field_name, "") or "").strip()
        if value:
            return value
    return ""


@dataclass(slots=True)
class LatestLink:
    """去重阶段驻留的紧凑链路记录；字段均已按构图口径完成解析。"""

    last_modified: int
    a_ne: str
    z_ne: str
    link_type: str
    a_port: str
    z_port: str


def _parse_latest_link(record: dict, last_modified: int) -> LatestLink:
    """把原始 CSV/JSON 记录压缩成 build_graphs 实际使用的语义字段。"""
    # NE 端点(基数=设备数，跨链路高度重复)、link_type 做 intern，与 NE/站点加载共用
    # 同一 intern 表，端点对象可与 ne_graph 的 key 共享。端口名基数高，不 intern。
    return LatestLink(
        last_modified=last_modified,
        a_ne=sys.intern(_get_record_value(
            record, "a_end_ne_nativeId", "a_end_ne_nativeId(')"
        ).upper()),
        z_ne=sys.intern(_get_record_value(
            record, "z_end_ne_nativeId", "z_end_ne_nativeId(')"
        ).upper()),
        link_type=sys.intern((record.get('link_layer') or '').strip().upper()),
        a_port=_get_record_value(record, "a_end_port_name"),
        z_port=_get_record_value(record, "z_end_port_name"),
    )


def _consume_latest_links(keyed_records: list, no_key_records: list):
    """保持既有顺序逐条产出，弹出后即置空槽位，使已构图记录尽早回收。

    入参均为普通 list（去重字典在调用前已转 list 并清空），因此消费阶段不再驻留
    去重哈希表与链路 ID 字符串，只剩一段随消费逐步缩小的指针数组。
    """
    for records in (keyed_records, no_key_records):
        for index, payload in enumerate(records):
            records[index] = None
            yield payload
        records.clear()


def load_latest_link_records(link_input: str, report_duplicates: bool = False):
    """读取链路记录并按 last_Modified 去重：同一链路 ID 只保留最新记录。

    无法确定链路 ID 的记录不参与去重，原样保留。

    去重时把原始记录解析为 slots 紧凑对象，不在内存中驻留字段字典；返回惰性可迭代
    对象，调用方单趟遍历并逐条从去重表弹出，避免完整去重表与完整图长期重叠。

    按链路 ID 首次出现的顺序给出其最新记录，再接所有无 ID 记录，以保证
    build_graphs 中“同 (ne,port) 键后写覆盖”等顺序相关行为稳定。
    """
    records = {}
    no_key_records = []
    duplicates = defaultdict(int) if report_duplicates else None

    progress = ProgressBar(0, "  读取链路记录")
    row_count = 0
    for record in iter_csv_records(link_input, progress, require=True):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        last_modified = _require_last_modified(record, row_count, '链路输入')
        key = _link_dedup_key(record)
        if not key:
            no_key_records.append(_parse_latest_link(record, last_modified))
            continue
        existing = records.get(key)
        if existing is not None:
            if duplicates is not None:
                duplicates[key] += 1
            if last_modified <= existing.last_modified:
                continue
        records[key] = _parse_latest_link(record, last_modified)
    progress.set(row_count)
    progress.close()

    unique_count = len(records) + len(no_key_records)
    print(f"  读取 {row_count} 行，去重后 {unique_count} 条链路记录")
    _report_duplicate_detail(duplicates, '链路ID')

    # 转成 value 列表并立即清空字典：在构图开始前就释放去重哈希表与全部链路 ID 字符串
    keyed_records = list(records.values())
    records.clear()
    # last_Modified 只参与去重；构图阶段统一指向小整数单例，释放每条记录独占的大整数。
    for payload in keyed_records:
        payload.last_modified = 0
    for payload in no_key_records:
        payload.last_modified = 0
    return _consume_latest_links(keyed_records, no_key_records)


# --------------------------------------------------------------------------- #
# 端口对端索引
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PeerDevice:
    ne_native_id: str
    port_name: str = ""


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


def build_graphs(latest_links, ne_info: dict):
    """单趟遍历去重后的链路，同时构建 site 邻接图、ne 邻接图与端口对端索引。

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
    }

    for record in latest_links:
        a_ne = record.a_ne
        z_ne = record.z_ne
        link_type = record.link_type

        if a_ne and z_ne:
            # NE 邻接图
            stats['ne_link_count'] += 1
            stats['ne_mapped_count'] += 1
            _add_bidirectional_edge(ne_links, a_ne, z_ne, link_type)

            # 站点邻接图：两端 NE 都能映射到站点时才登记
            stats['site_link_count'] += 1
            a_site = ne_info.get(a_ne, {}).get('site_id')
            z_site = ne_info.get(z_ne, {}).get('site_id')
            if a_site and z_site:
                stats['site_mapped_count'] += 1
                _add_bidirectional_edge(site_links, a_site, z_site, link_type)

            # 端口对端索引：还需两端端口名
            a_port = record.a_port
            z_port = record.z_port
            if a_port and z_port:
                peer_index[make_key(a_ne, a_port)] = PeerDevice(
                    # a_ne/z_ne 已按构图口径归一化；直接复用，避免每条索引再复制 NE ID。
                    ne_native_id=z_ne,
                    port_name=z_port,
                )
                peer_index[make_key(z_ne, z_port)] = PeerDevice(
                    ne_native_id=a_ne,
                    port_name=a_port,
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
        ne_data['site_type'] = site_data.get('site_type', '')
        ne_data['longitude'] = site_data.get('longitude', '')
        ne_data['latitude'] = site_data.get('latitude', '')
        ne_data['region_id'] = ne_data.get('region_id', '') or site_data.get('region_id', '')
        ne_data['link'] = ne_links.pop(ne_id, {})
    # 仅出现在邻接里、SYS_NE 中没有的 NE，用缺省字段补齐
    for ne_id in list(ne_links):
        ne_info[ne_id] = {
            'site_id': '',
            'site_name': '',
            'site_type': '',
            'longitude': '',
            'latitude': '',
            'region_id': '',
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
    return build_site_chains_from_data(
        prediction,
        ne_graph=ne_graph,
        prediction_label="<resource_buffer: pairwise prediction>",
        ne_graph_label="<resource_buffer: ne_graph>",
        restrict_relation=True,
        show_progress=False,
    )


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


def main():
    parser = argparse.ArgumentParser(
        description="一次性生成单个资源缓冲文件（site_graph / ne_graph / link_peer_index / site_device_counts 各占一个字段）"
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

    # 1) SYS_NE 读取一次，得到完整 NE 信息（其中含 site_id）
    print("加载NE信息...")
    ne_info = load_ne_from_csv(args.ne_dir, args.report_duplicates)
    print(f"  NE数量: {len(ne_info)}")

    # 2) SYS_SITE 读取一次
    print("\n加载站点信息...")
    site_info = load_site_info(args.site_dir, args.report_duplicates)

    # 3) 链路读取+去重一次，单趟遍历产出三张图
    print("\n生成传播图与对端索引...")
    latest_links = load_latest_link_records(args.link_input, args.report_duplicates)
    site_links, ne_links, peer_index, stats = build_graphs(latest_links, ne_info=ne_info)
    del latest_links  # 释放去重表（slim 记录）
    print(f"  站点图: 处理 {stats['site_link_count']} 条链路，成功映射 {stats['site_mapped_count']} 条")
    print(f"  NE图:   处理 {stats['ne_link_count']} 条链路，成功映射 {stats['ne_mapped_count']} 条")

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
    with open(args.output, 'w', encoding='utf-8') as f:
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
    print(f"\n生成文件: {args.output}")

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


if __name__ == "__main__":
    main()
