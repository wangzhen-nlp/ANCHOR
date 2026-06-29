#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成单个资源缓冲文件，用字段区分不同内容：

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
import csv
import io
import zipfile
import argparse
import stat

from dataclasses import asdict, dataclass
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import (
    NE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)
from topology_tools.generate_site_chains import build_site_chains_from_data
from topology_tools.generate_site_pair_order_pairwise import (
    build_pairwise_prediction,
    parse_args as parse_pairwise_args,
)

# site_chains 字段的生成口径：等价于先跑
#   generate_site_pair_order_pairwise.py --smooth-level --global-gap-first --strict-ring-bidirectional
# 再用其产物跑 generate_site_chains.py --restrict-relation（ne_graph 取本缓冲内存版）。
SITE_PAIR_PAIRWISE_FLAGS = [
    "--smooth-level",
    "--global-gap-first",
    "--strict-ring-bidirectional",
    "--no-progress",
]

# 单文件缓冲产物的默认输出路径；与默认资源同目录，避免额外引入路径常量
DEFAULT_BUFFER_OUTPUT = os.path.join(os.path.dirname(NE_GRAPH_JSON), "resource_buffer.json")

# 行级进度的刷新间隔（行数）；计数模式下每次 set 都会重绘，需要批量节流
PROGRESS_ROW_STEP = 5000

LINK_FILE_SUFFIXES = ('.jsonl', '.csv', '.zip')
CSV_FILE_SUFFIXES = ('.csv', '.zip')

# 链路记录的去重键字段，按优先级取第一个非空值
LINK_KEY_FIELDS = ('nativeId', "nativeId(')", 'resId', 'source_uuid')

# site_graph 中缺省站点信息（NE 引用了某站点但 SYS_SITE 中无该站点时使用）
_DEFAULT_SITE = {
    'site_name': '',
    'site_type': '',
    'longitude': '',
    'latitude': '',
    'region_id': '',
    'is_hub': False,
}


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
# 文件迭代：CSV / zip(内含CSV) / JSONL
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


def _iter_link_file(file_path: str):
    """迭代单个链路文件中的记录，支持 .jsonl / .csv / .zip（内含 CSV）。"""
    lower = file_path.lower()
    if lower.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif lower.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
            yield from csv.DictReader(f)
    elif lower.endswith('.zip'):
        yield from _iter_zip_csv(file_path)
    else:
        raise SystemExit(f"不支持的链路文件格式: {file_path}（支持 .jsonl/.csv/.zip）")


def iter_csv_dir_records(data_dir: str, name_keyword: str, progress: ProgressBar = None):
    """迭代目录中文件名含 name_keyword 的 .csv / .zip(内含CSV) 文件的记录。"""
    file_names = sorted(
        name for name in os.listdir(data_dir)
        if name_keyword in name and name.lower().endswith(CSV_FILE_SUFFIXES)
    )
    if not file_names:
        print(f"警告: 目录中未找到文件名含 {name_keyword} 的 .csv/.zip 文件: {data_dir}")
        return
    for name in file_names:
        if progress is not None:
            progress.set_extra_text(name, force=True)
        path = os.path.join(data_dir, name)
        if name.lower().endswith('.csv'):
            with open(path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
                yield from csv.DictReader(f)
        else:
            yield from _iter_zip_csv(path)


def iter_link_records(link_input: str, progress: ProgressBar = None):
    """迭代链路记录；link_input 可以是单个文件，也可以是包含链路文件的目录。"""
    if os.path.isdir(link_input):
        file_names = sorted(
            name for name in os.listdir(link_input)
            if name.lower().endswith(LINK_FILE_SUFFIXES)
        )
        if not file_names:
            raise SystemExit(f"目录中未找到链路文件(.jsonl/.csv/.zip): {link_input}")
        for name in file_names:
            if progress is not None:
                progress.set_extra_text(name, force=True)
            yield from _iter_link_file(os.path.join(link_input, name))
    else:
        if progress is not None:
            progress.set_extra_text(os.path.basename(link_input), force=True)
        yield from _iter_link_file(link_input)


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
    for row in iter_csv_dir_records(data_dir, 'SYS_NE', progress):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        last_modified = _require_last_modified(row, row_count, 'SYS_NE')
        nativeId = (row.get('nativeId') or '').strip().upper()
        if not nativeId:
            continue
        existing = records.get(nativeId)
        if existing is not None:
            if duplicates is not None:
                duplicates[nativeId] += 1
            if last_modified <= existing[0]:
                continue
        incoming = {
            'domain': (row.get('domain') or '').strip(),
            'type': (row.get('typeId') or '').strip(),
            'network_type': (row.get('network_type') or '').strip(),
            'name': (row.get('name') or '').strip(),
            'manufacturer': (row.get('manufacturer') or '').strip(),
            'region_id': (row.get('regionId1') or '').strip(),
            'site_id': (row.get('ne_site_id') or '').strip().upper(),
            'running_status': (row.get('running_status') or '').strip()
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
    for row in iter_csv_dir_records(data_dir, 'SYS_SITE', progress):
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        last_modified = _require_last_modified(row, row_count, 'SYS_SITE')
        site_id = (row.get('site_id') or '').strip().upper()
        if not site_id:
            continue
        existing = records.get(site_id)
        if existing is not None:
            if duplicates is not None:
                duplicates[site_id] += 1
            if last_modified <= existing[0]:
                continue
        incoming = {
            'site_name': (row.get('name') or '').strip(),
            'site_type': (row.get('site_type') or '').strip(),
            'longitude': (row.get('longitude') or '').strip(),
            'latitude': (row.get('latitude') or '').strip(),
            'region_id': (row.get('region_id') or '').strip(),
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


@dataclass(frozen=True, slots=True)
class LatestLink:
    """去重阶段驻留的紧凑链路记录；字段均已按构图口径完成解析。"""

    last_modified: int
    a_ne: str
    z_ne: str
    link_type: str
    a_port: str
    z_port: str
    a_ip: str
    z_ip: str
    a_manager: str
    z_manager: str


def _parse_latest_link(record: dict, last_modified: int) -> LatestLink:
    """把原始 CSV/JSON 记录压缩成 build_graphs 实际使用的语义字段。"""
    return LatestLink(
        last_modified=last_modified,
        a_ne=_get_record_value(
            record, "a_end_ne_nativeId", "a_end_ne_nativeId(')"
        ).upper(),
        z_ne=_get_record_value(
            record, "z_end_ne_nativeId", "z_end_ne_nativeId(')"
        ).upper(),
        link_type=(record.get('link_layer') or '').strip().upper(),
        a_port=_get_record_value(record, "a_end_port_name"),
        z_port=_get_record_value(record, "z_end_port_name"),
        a_ip=_get_record_value(record, "a_end_port_ip"),
        z_ip=_get_record_value(record, "z_end_port_ip"),
        a_manager=_get_record_value(record, "a_end_ne_manager_name"),
        z_manager=_get_record_value(record, "z_end_ne_manager_name"),
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

    产出顺序与原始实现完全一致（按链路 ID 首次出现的顺序给出其最新记录，再接所有
    无 ID 记录），以保证 build_graphs 中“同 (ne,port) 键后写覆盖”等顺序相关行为不变。
    """
    records = {}
    no_key_records = []
    duplicates = defaultdict(int) if report_duplicates else None

    progress = ProgressBar(0, "  读取链路记录")
    row_count = 0
    for record in iter_link_records(link_input, progress):
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
    return _consume_latest_links(keyed_records, no_key_records)


# --------------------------------------------------------------------------- #
# 端口对端索引
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PeerDevice:
    ne_native_id: str
    port_name: str = ""
    port_ip: str = ""
    manager_name: str = ""


def _normalize_ne_key(ne_id):
    return str(ne_id or "").strip().upper()


def _normalize_port_key(port_name):
    return str(port_name or "").strip()


def _make_key(ne_id, port_name):
    return f"{_normalize_ne_key(ne_id)}|{_normalize_port_key(port_name)}"


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
        if isinstance(record, LatestLink):
            a_ne = record.a_ne
            z_ne = record.z_ne
            link_type = record.link_type
        else:
            # 兼容直接调用 build_graphs 的既有 dict 输入。
            a_ne = _get_record_value(
                record, "a_end_ne_nativeId", "a_end_ne_nativeId(')"
            ).upper()
            z_ne = _get_record_value(
                record, "z_end_ne_nativeId", "z_end_ne_nativeId(')"
            ).upper()
            link_type = (record.get('link_layer') or '').strip().upper()

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
            if isinstance(record, LatestLink):
                a_port = record.a_port
                z_port = record.z_port
                a_ip = record.a_ip
                z_ip = record.z_ip
                a_manager = record.a_manager
                z_manager = record.z_manager
            else:
                a_port = _get_record_value(record, "a_end_port_name")
                z_port = _get_record_value(record, "z_end_port_name")
                a_ip = _get_record_value(record, "a_end_port_ip")
                z_ip = _get_record_value(record, "z_end_port_ip")
                a_manager = _get_record_value(record, "a_end_ne_manager_name")
                z_manager = _get_record_value(record, "z_end_ne_manager_name")
            if a_port and z_port:
                peer_index[_make_key(a_ne, a_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(z_ne),
                    port_name=z_port,
                    port_ip=z_ip,
                    manager_name=z_manager,
                )
                peer_index[_make_key(z_ne, z_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(a_ne),
                    port_name=a_port,
                    port_ip=a_ip,
                    manager_name=a_manager,
                )

    return site_links, ne_links, peer_index, stats


def assemble_site_graph(site_info: dict, site_links: dict) -> dict:
    """原地把站点信息补成 site_graph：给每个站点挂 link 字段，返回 site_info 自身。

    直接复用 site_info 外层字典、并把 site_links 的邻接字典逐站点 pop 后直接挂载
    （不再拷贝外层字典与邻接字典）。输出值与原实现完全一致，仅顶层站点的 key 顺序
    由原来的集合并集顺序变为 site_info 插入顺序、缺省站点追加在后。
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
    直接挂载（不再额外创建 ne_graph 外层字典，也不再产生 set 并集临时集合）。
    调用方此后不应再使用 ne_info。输出值与原实现完全一致，仅顶层 NE 的 key 顺序
    由集合并集顺序变为 ne_info 插入顺序、缺省 NE 追加在后。
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
    """在内存中复刻 site_chains.json 的生成链路，产出可作为缓冲字段的站点链路。

    等价命令：
        generate_site_pair_order_pairwise.py --smooth-level --global-gap-first --strict-ring-bidirectional
        generate_site_chains.py --restrict-relation --ne-graph <ne_graph>

    两步都直接吃内存里的 ne_graph，省去中间落盘再读盘。
    """
    pairwise_args = parse_pairwise_args(SITE_PAIR_PAIRWISE_FLAGS)
    prediction = build_pairwise_prediction(ne_graph, pairwise_args, show_progress=False)
    return build_site_chains_from_data(
        prediction,
        ne_graph=ne_graph,
        prediction_label="<resource_buffer: pairwise prediction>",
        ne_graph_label="<resource_buffer: ne_graph>",
        restrict_relation=True,
        show_progress=False,
    )


class _NestedIndentWriter:
    """为单个顶层字段增加一级缩进，使分字段写出与 json.dump 格式一致。"""

    def __init__(self, raw_file):
        self.raw_file = raw_file

    def write(self, text):
        return self.raw_file.write(text.replace('\n', '\n  '))


def _write_buffer_field(output_file, name: str, value, *, first: bool) -> None:
    """按 json.dump(..., indent=2) 的格式写一个顶层字段。"""
    if not first:
        output_file.write(',\n')
    output_file.write('  ')
    json.dump(name, output_file, ensure_ascii=False)
    output_file.write(': ')
    json.dump(
        value,
        _NestedIndentWriter(output_file),
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )


def main():
    parser = argparse.ArgumentParser(
        description="一次性生成单个资源缓冲文件（site_graph / ne_graph / link_peer_index / site_device_counts 各占一个字段）"
    )
    parser.add_argument(
        "--ne-dir",
        default=SYS_NE_DIR,
        help=f"SYS_NE 数据目录，默认: {resource_display('SYS_NE_0306')}",
    )
    parser.add_argument(
        "--site-dir",
        default=SYS_SITE_DIR,
        help=f"SYS_SITE 数据目录，默认: {resource_display('SYS_SITE_0306')}",
    )
    parser.add_argument(
        "--link-input",
        default=SYS_LINK_JSONL,
        help=(
            "链路输入，支持 .jsonl/.csv/.zip(内含CSV) 文件或包含这些文件的目录，"
            f"默认: {resource_display('sys_link_1231.jsonl')}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_BUFFER_OUTPUT,
        help=f"输出缓冲文件（单文件，字段区分内容），默认: {resource_display('resource_buffer.json')}",
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

    # 分字段写临时文件：site_chains 只依赖 ne_graph，先写并释放其余大字段，
    # 避免 pairwise/site_chains 的临时结构与 site_graph、peer_index 叠加成峰值。
    output_path = os.path.realpath(args.output)
    temp_output = f"{output_path}.tmp.{os.getpid()}"
    existing_mode = (
        stat.S_IMODE(os.stat(output_path).st_mode)
        if os.path.exists(output_path)
        else None
    )
    try:
        with open(temp_output, 'w', encoding='utf-8') as f:
            f.write('{\n')
            _write_buffer_field(f, 'site_graph', site_graph, first=True)
            del site_graph

            _write_buffer_field(f, 'ne_graph', ne_graph, first=False)

            _write_buffer_field(f, 'link_peer_index', peer_index, first=False)
            del peer_index

            _write_buffer_field(f, 'site_device_counts', site_device_counts, first=False)
            del site_device_counts

            # 由 ne_graph 派生站点链路（pairwise 方向先验 + restrict-relation 裁剪）
            print("\n生成站点链路(site_chains)...")
            site_chains = build_site_chains_field(ne_graph)
            _write_buffer_field(f, 'site_chains', site_chains, first=False)
            f.write('\n}')

        if existing_mode is not None:
            os.chmod(temp_output, existing_mode)
        os.replace(temp_output, output_path)
    except BaseException:
        try:
            os.remove(temp_output)
        except FileNotFoundError:
            pass
        raise

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
