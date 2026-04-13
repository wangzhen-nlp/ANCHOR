
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
import csv
import argparse

from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import (
    NE_GRAPH_JSON,
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    resource_display,
)


def _merge_fields_preserve_first(existing: dict, incoming: dict, entity_label: str, entity_id: str, source_label: str) -> dict:
    """按字段合并记录；冲突时优先保留字符更长的非空值。"""
    merged = dict(existing)
    for field, incoming_value in incoming.items():
        if incoming_value in ("", None):
            continue

        existing_value = merged.get(field, "")
        if existing_value in ("", None):
            merged[field] = incoming_value
            continue

        if existing_value != incoming_value:
            if len(str(incoming_value)) > len(str(existing_value)):
                merged[field] = incoming_value
    return merged


def load_ne_from_csv(data_dir: str = SYS_NE_DIR) -> dict:
    """
    从SYS_NE_0306加载NE信息

    Returns:
        {nativeId: {domain, name, manufacturer, region_id, site_id, ...}}
    """
    ne_info = {}

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv') and 'SYS_NE' in f)
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nativeId = row.get('nativeId', '').strip().upper()
                if not nativeId:
                    continue
                incoming = {
                    'domain': row.get('domain', '').strip(),
                    'type': row.get('typeId', '').strip(),
                    'network_type': row.get('network_type', '').strip(),
                    'name': row.get('name', '').strip(),
                    'manufacturer': row.get('manufacturer', '').strip(),
                    'region_id': row.get('regionId1', '').strip(),
                    'site_id': row.get('ne_site_id', '').strip().upper(),
                    'running_status': row.get('running_status', '').strip()
                }
                ne_info[nativeId] = _merge_fields_preserve_first(
                    ne_info.get(nativeId, {}),
                    incoming,
                    entity_label="NE",
                    entity_id=nativeId,
                    source_label=csv_file,
                )

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


def build_ne_graph(jsonl_file: str) -> dict:
    """
    根据链路信息生成站点邻接图

    Returns:
        {
            site_id: {
                link: {
                    neighbor_site_id: {
                        link_type: link_direction,
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

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            a_ne = record.get('a_end_ne_nativeId', '').upper()
            z_ne = record.get('z_end_ne_nativeId', '').upper()
            a_ne = a_ne or record.get("a_end_ne_nativeId(')", "").upper()
            z_ne = z_ne or record.get("z_end_ne_nativeId(')", "").upper()
            link_type = record.get('link_type', '').upper()

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
        help=f"链路 JSONL 输入，默认: {resource_display('sys_link_1231.jsonl')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=NE_GRAPH_JSON,
        help=f"输出 ne_graph.json，默认: {resource_display('ne_graph.json')}",
    )
    args = parser.parse_args()

    # 加载NE信息
    print("加载NE信息...")
    ne_info = load_ne_from_csv(args.ne_dir)
    print(f"  NE数量: {len(ne_info)}")

    # 加载站点信息
    print("加载站点信息...")
    site_info = load_site_info(args.site_graph)
    print(f"  站点数量: {len(site_info)}")

    # 生成节点传播图
    print("\n生成节点传播图...")
    ne_links = build_ne_graph(args.link_input)

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
