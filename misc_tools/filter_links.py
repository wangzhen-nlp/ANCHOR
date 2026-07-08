#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据站点ID列表筛选相关的NE信息，输出格式与 ne_propagation.json 一致
"""

import json
import argparse


def parse_site_ids(id_string: str) -> list:
    """解析逗号分隔的站点ID字符串为集合"""
    return [s.strip() for s in id_string.split(',') if s.strip()]


def convert_link_format(old_link: dict) -> dict:
    """
    将 ne_graph.json 的 link 格式转换为 ne_propagation.json 格式

    Args:
        old_link: ne_graph.json 中的 link 格式

    Returns:
        ne_propagation.json 格式的 link
    """
    new_link = {}
    for ne_name, link_info in old_link.items():
        new_link[ne_name] = {
            "connection_type": f"one_hop_link:{','.join(link_info.keys())}",
            "distance": -1,
            "topology": "CrossNE",
            "time_window": 7,
            "left_alarm": {},
            "right_alarm": {}
        }
    return new_link


def filter_links_in_sites(old_link: dict, valid_ne_names: set) -> dict:
    """
    只保留两端 NE 都在指定站点集合中的连边

    Args:
        old_link: 转换后的 link 格式
        valid_ne_names: 有效的 NE 名称集合

    Returns:
        过滤后的 link
    """
    filtered_link = {}
    for ne_name in old_link.keys():
        if ne_name in valid_ne_names:
            filtered_link[ne_name] = old_link[ne_name]
    return filtered_link


def get_nes_by_site_ids(graph_file: str, site_ids: set) -> dict:
    """
    根据 site_id 列表返回 NE 信息（ne_propagation 格式）
    只保留提供的站点之间的连边

    Args:
        graph_file: ne_graph.json 文件路径
        site_ids: 站点 ID 集合

    Returns:
        包含 ne_info 和 group_info 的字典
    """
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # 第一步：收集所有在目标站点中的 NE
    raw_ne_info = {}
    site_set = set()
    ne_list = []

    for ne_name, ne_data in graph_data.items():
        ne_site_id = ne_data.get('site_id', '')
        if ne_site_id in site_ids:
            raw_ne_info[ne_name] = {
                "ne_data": ne_data,
                "site_id": ne_site_id
            }
            site_set.add(ne_site_id)
            ne_list.append(ne_name)

    # 构建有效的 NE 名称集合（用于过滤连边）
    valid_ne_names = set(raw_ne_info.keys())

    # 第二步：构建 NE 信息，并过滤连边
    ne_info = {}
    for ne_name, info in raw_ne_info.items():
        ne_data = info["ne_data"]
        old_link = ne_data.get('link', {})
        converted_link = convert_link_format(old_link)
        filtered_link = filter_links_in_sites(converted_link, valid_ne_names)

        ne_info[ne_name] = {
            "alarm": ne_data.get('alarm', []),
            "link": filtered_link,
            "group": 1,
            "type": ne_data.get('type', ne_data.get('network_type', '')),
            "network_type": ne_data.get('network_type', ''),
            "manufacturer": ne_data.get('manufacturer', ''),
            "domain": ne_data.get('domain', ''),
            "name": ne_data.get('name', ne_name),
            "site_id": ne_data.get('site_id', ''),
            "site_name": ne_data.get('site_name', ''),
            "longitude": ne_data.get('longitude', ''),
            "latitude": ne_data.get('latitude', ''),
            "region_id": ne_data.get('region_id', '')
        }

    # 构建 group_info
    group_info = {
        "1": {
            "ne_list": ne_list,
            "site_list": list(site_set)
        }
    }

    return {
        "ne_info": ne_info,
        "group_info": group_info
    }


def main():
    parser = argparse.ArgumentParser(
        description='根据站点ID列表获取NE信息，输出格式与 propagation.json 一致',
        epilog='示例: python filter_links.py "14BJN0009,14BJN0010" output.json'
    )
    parser.add_argument('site_ids', type=str, help='站点ID（逗号分隔）')
    parser.add_argument('output', type=str, help='输出JSON文件')
    parser.add_argument('--graph-file', type=str, default='ne_graph.json',
                        help='ne_graph.json 文件路径')

    args = parser.parse_args()

    # 解析站点ID
    site_ids = parse_site_ids(args.site_ids)
    print(f"输入站点数: {len(site_ids)}")
    site_ids = set(site_ids)

    result = get_nes_by_site_ids(args.graph_file, site_ids)

    ne_count = len(result['ne_info'])
    site_count = len(result['group_info']['1']['site_list'])

    print(f"匹配站点: {site_count}")
    print(f"NE 数量: {ne_count}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()

