#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成 site_graph_by_ne.json
NE 之间有连边，则它们所属的站点之间也添加连边
连边方向：根据 site_order.json 中 site 的 index 确定，从小指向大
"""

import json
from collections import defaultdict
from argparse import ArgumentParser

NE_GRAPH_FILE = "ne_graph.json"
SITE_ORDER_FILE = "site_order.json"
OUTPUT_FILE = "site_graph_by_ne.json"


def generate_site_graph_by_ne(
    ne_graph_file: str = NE_GRAPH_FILE,
    site_order_file: str = SITE_ORDER_FILE,
    output_file: str = OUTPUT_FILE
):
    # 读取 site_order
    with open(site_order_file, 'r', encoding='utf-8') as f:
        site_order = json.load(f)

    # 读取 ne_graph
    with open(ne_graph_file, 'r', encoding='utf-8') as f:
        ne_graph = json.load(f)

    # ne -> site
    ne_to_site = {}
    for ne_name, ne_info in ne_graph.items():
        site_id = ne_info.get('site_id', '')
        site_name = ne_info.get('site_name', '')
        if site_id and site_name:
            ne_to_site[ne_name] = site_id

    # 站点连接: site -> {connected_site: count}
    site_links = defaultdict(set)

    # 遍历所有 NE 的连接
    for ne_name, ne_info in ne_graph.items():
        source_site = ne_to_site.get(ne_name)
        if not source_site:
            continue

        links = ne_info.get('link', {})
        for target_ne in links.keys():
            target_site = ne_to_site.get(target_ne)
            if not target_site or target_site == source_site:
                continue

            # 根据 site_order 确定方向
            source_idx = site_order.get(source_site)
            target_idx = site_order.get(target_site)

            if source_idx is None or target_idx is None:
                continue

            if source_idx < target_idx:
                site_links[source_site].add(target_site)
            elif target_idx < source_idx:
                site_links[target_site].add(source_site)

    site_links = {site: list(site_links[site]) for site in site_links}
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(site_links, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ne-graph", default=NE_GRAPH_FILE, help='ne_graph.json 文件')
    parser.add_argument("--site-order", default=SITE_ORDER_FILE, help='site_order.json 文件')
    parser.add_argument("--output", default=OUTPUT_FILE, help='输出文件')
    args = parser.parse_args()

    generate_site_graph_by_ne(args.ne_graph, args.site_order, args.output)