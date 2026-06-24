#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据sys_link_1231.jsonl和SYS_NE_0306生成站点传播图

逻辑:
1. 从SYS_NE_0306中提取 nativeId -> ne_site_id 的映射
2. 从SYS_SITE_0306中提取站点信息 (longitude, latitude, site_type, region_id)
3. 从链路输入(JSONL/CSV/zip压缩CSV文件或其所在目录)中提取链路信息
4. 对于每条链路，根据ne所属站点生成站点邻接关系
"""

import json
import os
import csv
import io
import zipfile
import argparse
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import (
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)

# 行级进度的刷新间隔（行数）；计数模式下每次 set 都会重绘，需要批量节流
PROGRESS_ROW_STEP = 5000


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


def load_ne_site_mapping(data_dir: str = SYS_NE_DIR, report_duplicates: bool = False) -> dict:
    """
    从SYS_NE中加载nativeId -> ne_site_id的映射；同 nativeId 按 last_Modified 取最新记录

    Returns:
        {nativeId: ne_site_id}
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
        ne_site_id = (row.get('ne_site_id') or '').strip().upper()
        if not (nativeId and ne_site_id):
            continue
        _keep_latest(records, nativeId, last_modified, ne_site_id, duplicates)
    progress.set(row_count)
    progress.close()

    mapping = {key: payload for key, (_, payload) in records.items()}
    print(f"  读取 {row_count} 行，去重后 {len(mapping)} 条映射")
    _report_duplicate_detail(duplicates, 'nativeId')
    return mapping


def load_site_info(data_dir: str = SYS_SITE_DIR, report_duplicates: bool = False) -> dict:
    """
    从SYS_SITE中加载站点信息；同 site_id 按 last_Modified 取最新记录，不做字段合并

    Returns:
        {site_id: {longitude, latitude, site_type, region_id, is_hub}}
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


LINK_FILE_SUFFIXES = ('.jsonl', '.csv', '.zip')
CSV_FILE_SUFFIXES = ('.csv', '.zip')


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


# 链路记录的去重键字段，按优先级取第一个非空值
LINK_KEY_FIELDS = ('nativeId', "nativeId(')", 'resId', 'source_uuid')


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


def build_site_graph(link_input: str, ne_site_map: dict, report_duplicates: bool = False) -> dict:
    """
    根据链路信息生成站点邻接图

    Returns:
        {
            site_id: {
                neighbor_site_id: {
                    link_layer: link_direction,
                    ...
                }
            },
            ...
        }
    """
    site_links = defaultdict(lambda: defaultdict(dict))

    link_count = 0
    mapped_count = 0

    for record in load_latest_link_records(link_input, report_duplicates):
        a_ne = (record.get('a_end_ne_nativeId') or '').strip().upper()
        z_ne = (record.get('z_end_ne_nativeId') or '').strip().upper()
        a_ne = a_ne or (record.get("a_end_ne_nativeId(')") or '').strip().upper()
        z_ne = z_ne or (record.get("z_end_ne_nativeId(')") or '').strip().upper()
        link_type = (record.get('link_layer') or '').strip().upper()

        if not (a_ne and z_ne):
            continue

        link_count += 1

        a_site = ne_site_map.get(a_ne)
        z_site = ne_site_map.get(z_ne)

        if not (a_site and z_site):
            continue

        mapped_count += 1
        if link_type in site_links[a_site][z_site]:
            if site_links[a_site][z_site][link_type] == '<-':
                site_links[a_site][z_site][link_type] = '<->'
        else:
            site_links[a_site][z_site][link_type] = '->'
        if link_type in site_links[z_site][a_site]:
            if site_links[z_site][a_site][link_type] == '->':
                site_links[z_site][a_site][link_type] = '<->'
        else:
            site_links[z_site][a_site][link_type] = '<-'

    print(f"  处理 {link_count} 条链路")
    print(f"  成功映射 {mapped_count} 条")

    return site_links


def main():
    parser = argparse.ArgumentParser(description="根据链路文件和站点/NE基础数据生成站点传播图")
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
        "-o",
        "--output",
        default=SITE_GRAPH_JSON,
        help=f"输出站点图 JSON，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--report-duplicates",
        action="store_true",
        help="打印 NE/站点/链路 中重复 ID 的明细（默认仅汇总，不打印明细）",
    )
    args = parser.parse_args()

    # 加载NE到站点的映射
    print("加载NE站点映射...")
    ne_site_map = load_ne_site_mapping(args.ne_dir, args.report_duplicates)

    # 加载站点信息
    print("\n加载站点信息...")
    site_info = load_site_info(args.site_dir, args.report_duplicates)

    # 生成站点传播图
    print("\n生成站点传播图...")
    site_links = build_site_graph(args.link_input, ne_site_map, args.report_duplicates)

    # 合并站点信息和链路信息
    result = {}
    for site_id in (set(site_info) | set(site_links)):
        neighbors = site_links.get(site_id, {})
        site_data = site_info.get(site_id, {
            'site_name': '',
            'site_type': '',
            'longitude': '',
            'latitude': '',
            'region_id': '',
            'is_hub': False,
        })
        site_data['link'] = {}
        for neighbor_id, links in neighbors.items():
            site_data['link'][neighbor_id] = links
        result[site_id] = site_data

    # 输出结果
    output_file = args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n生成文件: {output_file}")
    print(f"站点数: {len(result)}")

    neighbor_counts = [len(v['link']) for v in result.values()]
    if neighbor_counts:
        print(f"平均邻居站点数: {sum(neighbor_counts)/len(neighbor_counts):.1f}")
        print(f"最大邻居站点数: {max(neighbor_counts)}")


if __name__ == "__main__":
    main()
