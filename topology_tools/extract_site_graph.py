#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据sys_link_1231.jsonl和SYS_NE_0306生成站点传播图

逻辑:
1. 从SYS_NE_0306中提取 nativeId -> ne_site_id 的映射
2. 从SYS_SITE_0306中提取站点信息 (longitude, latitude, site_type, region_id)
3. 从sys_link_1231.jsonl中提取链路信息
4. 对于每条链路，根据ne所属站点生成站点邻接关系
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
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    SYS_SITE_DIR,
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


def _merge_scalar_preserve_first(mapping: dict, key: str, value: str, entity_label: str, field_name: str, source_label: str) -> None:
    """合并单字段映射；冲突时优先保留字符更长的非空值。"""
    if not key or value in ("", None):
        return

    existing_value = mapping.get(key, "")
    if existing_value in ("", None):
        mapping[key] = value
        return

    if existing_value != value:
        if len(str(value)) > len(str(existing_value)):
            mapping[key] = value


def _parse_bool(value) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "是"}


def load_ne_site_mapping(data_dir: str = SYS_NE_DIR) -> dict:
    """
    从SYS_NE_0306中加载nativeId -> ne_site_id的映射

    Returns:
        {nativeId: ne_site_id}
    """
    mapping = {}

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv') and 'SYS_NE' in f)
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"  处理: {csv_file}")
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nativeId = row.get('nativeId', '').strip().upper()
                ne_site_id = row.get('ne_site_id', '').strip().upper()
                _merge_scalar_preserve_first(
                    mapping,
                    nativeId,
                    ne_site_id,
                    entity_label="NE",
                    field_name="ne_site_id",
                    source_label=csv_file,
                )

    print(f"  共 {len(mapping)} 条映射")
    return mapping


def load_site_info(data_dir: str = SYS_SITE_DIR) -> dict:
    """
    从SYS_SITE_0306中加载站点信息

    Returns:
        {site_id: {longitude, latitude, site_type, region_id, is_hub}}
    """
    site_info = {}

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv') and 'SYS_SITE' in f)
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"  处理: {csv_file}")
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                site_id = row.get('site_id', '').strip().upper()
                if not site_id:
                    continue
                incoming = {
                    'site_name': row.get('name', '').strip(),
                    'site_type': row.get('site_type', '').strip(),
                    'longitude': row.get('longitude', '').strip(),
                    'latitude': row.get('latitude', '').strip(),
                    'region_id': row.get('region_id', '').strip(),
                    'is_hub': _parse_bool(row.get('is_hub', '')),
                }
                site_info[site_id] = _merge_fields_preserve_first(
                    site_info.get(site_id, {}),
                    incoming,
                    entity_label="SITE",
                    entity_id=site_id,
                    source_label=csv_file,
                )

    print(f"  共 {len(site_info)} 个站点")
    return site_info


def build_site_graph(jsonl_file: str, ne_site_map: dict) -> dict:
    """
    根据链路信息生成站点邻接图

    Returns:
        {
            site_id: {
                neighbor_site_id: {
                    link_type: link_direction,
                    ...
                }
            },
            ...
        }
    """
    site_links = defaultdict(lambda: defaultdict(dict))

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
        help=f"链路 JSONL 输入，默认: {resource_display('sys_link_1231.jsonl')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=SITE_GRAPH_JSON,
        help=f"输出站点图 JSON，默认: {resource_display('site_graph.json')}",
    )
    args = parser.parse_args()

    # 加载NE到站点的映射
    print("加载NE站点映射...")
    ne_site_map = load_ne_site_mapping(args.ne_dir)

    # 加载站点信息
    print("\n加载站点信息...")
    site_info = load_site_info(args.site_dir)

    # 生成站点传播图
    print("\n生成站点传播图...")
    site_links = build_site_graph(args.link_input, ne_site_map)

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
