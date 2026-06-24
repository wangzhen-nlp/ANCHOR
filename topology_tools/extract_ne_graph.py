
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成ne_graph.json

结合SYS_NE_0306和site_graph.json的信息:
- 从SYS_NE_0306获取: domain, name, manufacturer, region_id, site_id
- 从site_graph.json获取: site_name, site_type, longitude, latitude, region_id
"""

import json
import os
import argparse

from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    resource_display,
)
from topology_tools.extract_site_graph import (
    PROGRESS_ROW_STEP,
    _keep_latest,
    _report_duplicate_detail,
    _require_last_modified,
    iter_csv_dir_records,
    load_latest_link_records,
)


def load_ne_from_csv(data_dir: str = SYS_NE_DIR, report_duplicates: bool = False) -> dict:
    """
    从SYS_NE加载NE信息；同 nativeId 按 last_Modified 取最新记录，不做字段合并

    Returns:
        {nativeId: {domain, name, manufacturer, region_id, site_id, ...}}
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


def load_site_info(site_graph_file: str = SITE_GRAPH_JSON) -> dict:
    """
    从site_graph.json加载站点信息

    Returns:
        {site_id: {site_name, site_type, longitude, latitude, region_id}}
    """
    site_info = {}

    if not os.path.exists(site_graph_file):
        print(f"警告: 文件不存在 {site_graph_file}")
        return site_info

    with open(site_graph_file, 'r', encoding='utf-8') as f:
        site_graph = json.load(f)

    for site_id, data in site_graph.items():
        site_info[site_id] = {
            'site_name': data.get('site_name', ''),
            'site_type': data.get('site_type', ''),
            'longitude': data.get('longitude', ''),
            'latitude': data.get('latitude', ''),
            'region_id': data.get('region_id', ''),
        }

    return site_info


def build_ne_graph(link_input: str, report_duplicates: bool = False) -> dict:
    """
    根据链路信息生成NE邻接图

    Returns:
        {
            ne_id: {
                link: {
                    neighbor_ne_id: {
                        link_layer: link_direction,
                        ...
                    }
                }
            },
            ...
        }
    """
    ne_links = defaultdict(lambda: defaultdict(dict))

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

        mapped_count += 1
        if link_type in ne_links[a_ne][z_ne]:
            if ne_links[a_ne][z_ne][link_type] == '<-':
                ne_links[a_ne][z_ne][link_type] = '<->'
        else:
            ne_links[a_ne][z_ne][link_type] = '->'
        if link_type in ne_links[z_ne][a_ne]:
            if ne_links[z_ne][a_ne][link_type] == '->':
                ne_links[z_ne][a_ne][link_type] = '<->'
        else:
            ne_links[z_ne][a_ne][link_type] = '<-'

    print(f"  处理 {link_count} 条链路")
    print(f"  成功映射 {mapped_count} 条")

    return ne_links


def main():
    parser = argparse.ArgumentParser(description="结合链路和站点图生成 ne_graph.json")
    parser.add_argument(
        "--ne-dir",
        default=SYS_NE_DIR,
        help=f"SYS_NE 数据目录，默认: {resource_display('SYS_NE_0306')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"site_graph.json 文件，默认: {resource_display('site_graph.json')}",
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
        default=NE_GRAPH_JSON,
        help=f"输出 ne_graph.json，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--report-duplicates",
        action="store_true",
        help="打印 NE/链路 中重复 ID 的明细（默认仅汇总，不打印明细）",
    )
    args = parser.parse_args()

    # 加载NE信息
    print("加载NE信息...")
    ne_info = load_ne_from_csv(args.ne_dir, args.report_duplicates)
    print(f"  NE数量: {len(ne_info)}")

    # 加载站点信息
    print("加载站点信息...")
    site_info = load_site_info(args.site_graph)
    print(f"  站点数量: {len(site_info)}")

    # 生成节点传播图
    print("\n生成节点传播图...")
    ne_links = build_ne_graph(args.link_input, args.report_duplicates)

    # 合并站点信息和链路信息
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

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(ne_graph, f, ensure_ascii=False, indent=2)

    print(f"\n生成文件: {args.output}")
    print(f"NE数: {len(ne_graph)}")

    # 统计有站点信息的NE
    with_site = sum(1 for ne in ne_graph.values() if ne.get('site_id'))
    print(f"有站点信息的NE: {with_site} ({with_site/len(ne_graph)*100:.1f}%)")


if __name__ == "__main__":
    main()
