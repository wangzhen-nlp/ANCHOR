#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成站点的偏序关系
逻辑：
1. 不需要看站点内的 RAN 设备
2. 所有节点根据 Data 设备的入度排序，放入 data_list
3. 从 data_list[0] 放入 site_list，依次 BFS
4. 先取 Data 设备有连接的站点，后取 Transmission 设备有连接的站点
"""

import json
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_tools.site_pair_order_common import (
    _get_site_id,
    build_site_role_counts,
    normalize_domain,
    should_include_cross_site_link,
)

NE_GRAPH_FILE = "ne_graph.json"


def build_site_graph_no_ran(ne_graph_file: str = NE_GRAPH_FILE) -> tuple:
    """
    构建站点之间的连接图
    返回:
        site_connections: site -> {connected_sites}  (按 Data/Transmission 分开)
        site_degree: site -> 设备入度（来自其他站点的连接数）
        site_to_nes: site -> {domain: [nes]}
    """
    with open(ne_graph_file, 'r', encoding='utf-8') as f:
        ne_graph = json.load(f)

    # ne -> site
    ne_to_site = {}
    # site -> domain -> [nes]
    site_to_nes = defaultdict(lambda: defaultdict(list))

    for ne_name, ne_info in ne_graph.items():
        site_id = _get_site_id(ne_info)
        site_name = ne_info.get('site_name', '')
        domain = normalize_domain(ne_info.get('domain', ''))
        if site_id and site_name and domain:
            ne_to_site[ne_name] = site_id
            site_to_nes[site_id][domain].append(ne_name)

    # 站点连接: site -> {connected_site: {Data: count, Transmission: count}}
    site_connections = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # 站点 设备入度（来自其他站点的连接）
    site_degree = {}
    site_role_counts = build_site_role_counts(ne_graph)

    domain_weights = {'Data': 0, 'Transmission': 1, 'Ran': 2}

    for ne_name, ne_info in ne_graph.items():
        source_site = ne_to_site.get(ne_name)
        source_domain = normalize_domain(ne_info.get('domain', ''))
        if not source_site or not source_domain:
            continue

        links = ne_info.get('link', {})
        for target_ne in links.keys():
            target_site = ne_to_site.get(target_ne)
            if not target_site or target_site == source_site:
                continue

            target_info = ne_graph.get(target_ne, {})
            target_domain = normalize_domain(target_info.get('domain', ''))
            if not should_include_cross_site_link(
                source_site,
                source_domain,
                target_site,
                target_domain,
                site_role_counts,
            ):
                continue

            if source_domain not in domain_weights or target_domain not in domain_weights:
                continue

            if source_site not in site_degree:
                site_degree[source_site] = [0 for _ in range(9)]
            index = domain_weights[source_domain] * len(domain_weights) + domain_weights[target_domain]
            site_degree[source_site][index] += 1

            if target_domain != 'Ran':
                site_connections[source_site][target_site][target_domain] += 1

    return site_connections, site_degree, site_to_nes


def generate_site_order(ne_graph_file: str = NE_GRAPH_FILE) -> list:
    """
    生成站点的偏序列表
    """
    site_connections, site_degree, site_to_nes = build_site_graph_no_ran(ne_graph_file)

    # 按 Data 入度排序站点
    data_list = sorted(site_degree.keys(), key=lambda x: tuple(site_degree[x]), reverse=True)

    # 已加入 site_list 的站点
    visited = set()
    site_list = []
    processed_in_bfs = set()  # 已经通过 BFS 处理过的站点

    # 遍历 data_list
    for start_site in data_list:
        if start_site in visited:
            continue

        # BFS
        queue = [start_site]
        visited.add(start_site)
        site_list.append(start_site)
        processed_in_bfs.add(start_site)

        while queue:
            current_site = queue.pop(0)

            # 获取当前站点连接的其他站点
            connected = site_connections[current_site]

            # 先按 Data 排序，后按 Transmission 排序
            other_sites = []
            for other_site, domains in connected.items():
                if other_site in visited:
                    continue

                data_count = domains.get('Data', 0)
                transmission_count = domains.get('Transmission', 0)
                other_sites.append((other_site, data_count, transmission_count))

            # 排序：Data 多的优先，其次 Transmission 多的优先
            other_sites.sort(key=lambda x: (x[1], x[2]), reverse=True)

            for other_site, _, _ in other_sites:
                if other_site not in visited:
                    visited.add(other_site)
                    site_list.append(other_site)
                if other_site not in processed_in_bfs:
                    queue.append(other_site)
                    processed_in_bfs.add(other_site)

    print(f"site_list 长度: {len(site_list)}, visited 长度: {len(visited)}")
    return site_list


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--ne_graph_file", default=NE_GRAPH_FILE, help='ne graph file')
    parser.add_argument("--output", default='site_order.json', help='output file')
    args = parser.parse_args()

    site_list = generate_site_order(args.ne_graph_file)
    site_order = {site: i for i, site in enumerate(site_list)}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(site_order, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
