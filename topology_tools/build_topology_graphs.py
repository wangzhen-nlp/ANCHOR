#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成 site_graph.json / ne_graph.json / link_peer_index.json 三个产物。

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

from dataclasses import asdict, dataclass
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import (
    LINK_PEER_INDEX_JSON,
    NE_GRAPH_JSON,
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)

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


def _keep_latest(records: dict, key: str, last_modified: int, payload, duplicates: dict = None) -> None:
    """同 key 去重：只保留 last_Modified 更晚的记录，不做字段合并；时间相同保留先读到的。

    传入 duplicates(dict) 时，会累加重复 key 的额外出现次数，供明细报告使用。
    """
    existing = records.get(key)
    if existing is not None and duplicates is not None:
        duplicates[key] += 1
    if existing is None or last_modified > existing[0]:
        records[key] = (last_modified, payload)


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
        _keep_latest(records, nativeId, last_modified, incoming, duplicates)
    progress.set(row_count)
    progress.close()

    ne_info = {key: payload for key, (_, payload) in records.items()}
    print(f"  读取 {row_count} 行，去重后 {len(ne_info)} 个NE")
    _report_duplicate_detail(duplicates, 'nativeId')
    return ne_info


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
        incoming = {
            'site_name': (row.get('name') or '').strip(),
            'site_type': (row.get('site_type') or '').strip(),
            'longitude': (row.get('longitude') or '').strip(),
            'latitude': (row.get('latitude') or '').strip(),
            'region_id': (row.get('region_id') or '').strip(),
            'is_hub': _parse_bool(row.get('is_hub', '')),
        }
        _keep_latest(records, site_id, last_modified, incoming, duplicates)
    progress.set(row_count)
    progress.close()

    site_info = {key: payload for key, (_, payload) in records.items()}
    print(f"  读取 {row_count} 行，去重后 {len(site_info)} 个站点")
    _report_duplicate_detail(duplicates, 'site_id')
    return site_info


def load_latest_link_records(link_input: str, report_duplicates: bool = False) -> list:
    """读取链路记录并按 last_Modified 去重：同一链路 ID 只保留最新记录。

    无法确定链路 ID 的记录不参与去重，原样保留。
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
        key = ''
        for field in LINK_KEY_FIELDS:
            key = (record.get(field) or '').strip()
            if key:
                break
        if not key:
            no_key_records.append(record)
            continue
        _keep_latest(records, key, last_modified, record, duplicates)
    progress.set(row_count)
    progress.close()

    latest_records = [payload for _, payload in records.values()] + no_key_records
    print(f"  读取 {row_count} 行，去重后 {len(latest_records)} 条链路记录")
    _report_duplicate_detail(duplicates, '链路ID')
    return latest_records


# --------------------------------------------------------------------------- #
# 端口对端索引
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
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


def _get_record_value(record, *field_names):
    for field_name in field_names:
        value = str(record.get(field_name, "") or "").strip()
        if value:
            return value
    return ""


def save_peer_index(peer_index: dict, output_path: str) -> None:
    data = {
        key: asdict(value) if isinstance(value, PeerDevice) else dict(value)
        for key, value in peer_index.items()
    }
    with open(output_path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2, sort_keys=True)


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


def build_graphs(latest_links: list, ne_site_map: dict):
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
        a_ne = _get_record_value(record, "a_end_ne_nativeId", "a_end_ne_nativeId(')").upper()
        z_ne = _get_record_value(record, "z_end_ne_nativeId", "z_end_ne_nativeId(')").upper()
        link_type = (record.get('link_layer') or '').strip().upper()

        if a_ne and z_ne:
            # NE 邻接图
            stats['ne_link_count'] += 1
            stats['ne_mapped_count'] += 1
            _add_bidirectional_edge(ne_links, a_ne, z_ne, link_type)

            # 站点邻接图：两端 NE 都能映射到站点时才登记
            stats['site_link_count'] += 1
            a_site = ne_site_map.get(a_ne)
            z_site = ne_site_map.get(z_ne)
            if a_site and z_site:
                stats['site_mapped_count'] += 1
                _add_bidirectional_edge(site_links, a_site, z_site, link_type)

            # 端口对端索引：还需两端端口名
            a_port = _get_record_value(record, "a_end_port_name")
            z_port = _get_record_value(record, "z_end_port_name")
            if a_port and z_port:
                peer_index[_make_key(a_ne, a_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(z_ne),
                    port_name=z_port,
                    port_ip=_get_record_value(record, "z_end_port_ip"),
                    manager_name=_get_record_value(record, "z_end_ne_manager_name"),
                )
                peer_index[_make_key(z_ne, z_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(a_ne),
                    port_name=a_port,
                    port_ip=_get_record_value(record, "a_end_port_ip"),
                    manager_name=_get_record_value(record, "a_end_ne_manager_name"),
                )

    return site_links, ne_links, peer_index, stats


def assemble_site_graph(site_info: dict, site_links: dict) -> dict:
    """合并站点信息与站点邻接关系，生成 site_graph。"""
    result = {}
    for site_id in (set(site_info) | set(site_links)):
        site_data = dict(site_info.get(site_id, _DEFAULT_SITE))
        site_data['link'] = dict(site_links.get(site_id, {}))
        result[site_id] = site_data
    return result


def assemble_ne_graph(ne_info: dict, site_info: dict, ne_links: dict) -> dict:
    """合并 NE 信息、站点信息与 NE 邻接关系，生成 ne_graph。"""
    ne_graph = {}
    for ne_id in (set(ne_links) | set(ne_info)):
        ne_data = dict(ne_info.get(ne_id, {}))
        neighbors = ne_links.get(ne_id, {})
        site_id = ne_data.get('site_id', '')
        site_data = site_info.get(site_id, {})

        ne_data.update(
            {
                'site_id': site_id,
                'site_name': site_data.get('site_name', ''),
                'site_type': site_data.get('site_type', ''),
                'longitude': site_data.get('longitude', ''),
                'latitude': site_data.get('latitude', ''),
            }
        )
        ne_data['region_id'] = ne_data.get('region_id', '') or site_data.get('region_id', '')

        ne_graph[ne_id] = ne_data
        ne_graph[ne_id]['link'] = {}
        for neighbor_id, links in neighbors.items():
            ne_graph[ne_id]['link'][neighbor_id] = links
    return ne_graph


def main():
    parser = argparse.ArgumentParser(
        description="一次性生成 site_graph.json / ne_graph.json / link_peer_index.json"
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
        "--site-graph-output",
        default=SITE_GRAPH_JSON,
        help=f"输出 site_graph.json，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--ne-graph-output",
        default=NE_GRAPH_JSON,
        help=f"输出 ne_graph.json，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--peer-index-output",
        default=LINK_PEER_INDEX_JSON,
        help=f"输出 link_peer_index.json，默认: {resource_display('link_peer_index.json')}",
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

    # 由 NE 信息派生 nativeId->site_id 映射，无需再读一遍 SYS_NE
    ne_site_map = {
        native_id: info['site_id']
        for native_id, info in ne_info.items()
        if info.get('site_id')
    }

    # 2) SYS_SITE 读取一次
    print("\n加载站点信息...")
    site_info = load_site_info(args.site_dir, args.report_duplicates)

    # 3) 链路读取+去重一次，单趟遍历产出三张图
    print("\n生成传播图与对端索引...")
    latest_links = load_latest_link_records(args.link_input, args.report_duplicates)
    site_links, ne_links, peer_index, stats = build_graphs(latest_links, ne_site_map)
    print(f"  站点图: 处理 {stats['site_link_count']} 条链路，成功映射 {stats['site_mapped_count']} 条")
    print(f"  NE图:   处理 {stats['ne_link_count']} 条链路，成功映射 {stats['ne_mapped_count']} 条")

    # 4) 组装并写出
    site_graph = assemble_site_graph(site_info, site_links)
    with open(args.site_graph_output, 'w', encoding='utf-8') as f:
        json.dump(site_graph, f, ensure_ascii=False, indent=2)

    ne_graph = assemble_ne_graph(ne_info, site_info, ne_links)
    with open(args.ne_graph_output, 'w', encoding='utf-8') as f:
        json.dump(ne_graph, f, ensure_ascii=False, indent=2)

    save_peer_index(peer_index, args.peer_index_output)

    # 汇总
    print(f"\n生成文件: {args.site_graph_output}")
    print(f"  站点数: {len(site_graph)}")
    site_neighbor_counts = [len(v['link']) for v in site_graph.values()]
    if site_neighbor_counts:
        print(f"  平均邻居站点数: {sum(site_neighbor_counts)/len(site_neighbor_counts):.1f}")
        print(f"  最大邻居站点数: {max(site_neighbor_counts)}")

    print(f"生成文件: {args.ne_graph_output}")
    print(f"  NE数: {len(ne_graph)}")
    if ne_graph:
        with_site = sum(1 for ne in ne_graph.values() if ne.get('site_id'))
        print(f"  有站点信息的NE: {with_site} ({with_site/len(ne_graph)*100:.1f}%)")

    print(f"生成文件: {args.peer_index_output}")
    print(f"  对端索引记录数: {len(peer_index)}")


if __name__ == "__main__":
    main()
