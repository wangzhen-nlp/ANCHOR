#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据两个站点获取其 n 阶邻居站点，并生成与 filter_links.py 一致格式的输出文件
站点之间的连接通过站点内 NE 的连接实现
"""

import json
import argparse
from collections import defaultdict
from filter_links import get_nes_by_site_ids


def build_site_graph(ne_graph_file: str = "ne_graph.json") -> dict:
    """
    构建站点之间的连接图
    返回: site -> {connected_sites}
    """
    with open(ne_graph_file, 'r', encoding='utf-8') as f:
        ne_graph = json.load(f)

    site_connections = defaultdict(set)
    ne_to_site = {}

    # 建立 ne -> site 映射
    for ne_name, ne_info in ne_graph.items():
        site_id = ne_info.get('site_id', '')
        if site_id:
            ne_to_site[ne_name] = site_id

    # 遍历所有 NE 的连接，建立站点间的连接
    for ne_name, ne_info in ne_graph.items():
        source_site = ne_to_site.get(ne_name)
        if not source_site:
            continue

        links = ne_info.get('link', {})
        for target_ne in links.keys():
            target_site = ne_to_site.get(target_ne)
            if target_site and target_site != source_site:
                site_connections[source_site].add(target_site)
                site_connections[target_site].add(source_site)

    return site_connections


def get_nth_order_neighbors(start_sites: list, n: int, site_graph: dict) -> set:
    """
    获取起始站点的 n 阶邻居

    Args:
        start_sites: 起始站点列表
        n: 阶数
        site_graph: 站点连接图

    Returns:
        所有相关站点的集合（包括起始站点）
    """
    if n < 1:
        return set(start_sites)

    visited = set(start_sites)
    queue = list(start_sites)
    current_level = 0

    while queue and current_level < n:
        level_size = len(queue)
        for _ in range(level_size):
            current_site = queue.pop(0)
            for neighbor in site_graph.get(current_site, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        current_level += 1

    return visited


def main():
    parser = argparse.ArgumentParser(
        description='根据站点获取其 n 阶邻居站点，输出格式与 filter_links.py 一致',
        epilog='示例: python find_site_neighbors.py "07OKU0065,07OKU0066" 2 output.json'
    )
    parser.add_argument('sites', type=str, help='起始站点ID（逗号分隔）')
    parser.add_argument('output', type=str, help='输出JSON文件')
    parser.add_argument('--n', type=int, default=5, help='阶数')
    parser.add_argument('--graph-file', type=str, default='topology_resources/ne_graph.json',
                        help='ne_graph.json 文件路径')

    args = parser.parse_args()

    # 解析站点列表
    start_sites = [s.strip() for s in args.sites.split(',') if s.strip()]
    print(f"起始站点数: {len(start_sites)}")

    # 构建站点连接图
    print("正在构建站点连接图...")
    site_graph = build_site_graph(args.graph_file)
    print(f"站点数量: {len(site_graph)}")

    # 获取 n 阶邻居
    neighbor_sites = get_nth_order_neighbors(start_sites, args.n, site_graph)
    print(f"获取 {args.n} 阶邻居，总站点数: {len(neighbor_sites)}")

    # 调用 filter_links 的函数生成输出
    result = get_nes_by_site_ids(args.graph_file, neighbor_sites)

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {args.output}")
    print(f"总 NE 数: {len(result['ne_info'])}")
    print(f"总站点数: {len(result['group_info']['1']['site_list'])}")


if __name__ == "__main__":
    main()