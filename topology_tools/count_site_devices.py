#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过 ne_graph.json 统计每个站点包含的设备类型数量
输出: {site_id: {domain: count, ...}, ...}
"""

import json
from collections import defaultdict
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, SITE_DEVICE_COUNTS_JSON, resource_display


def count_site_devices(ne_graph_file: str = NE_GRAPH_JSON) -> dict:
    """统计每个站点的设备类型数量"""
    site_devices = defaultdict(lambda: defaultdict(int))

    with open(ne_graph_file, 'r', encoding='utf-8') as f:
        ne_graph = json.load(f)

    for ne_name, ne_info in ne_graph.items():
        site_id = ne_info.get('site_id', '')
        domain = ne_info.get('domain', '')
        if site_id and domain:
            site_devices[site_id][domain] += 1

    # 转换为普通 dict
    result = {site: dict(domains) for site, domains in site_devices.items()}
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=f'ne_graph.json 文件，默认: {resource_display("ne_graph.json")}')
    parser.add_argument("--output", "-o", default=SITE_DEVICE_COUNTS_JSON, help=f'输出文件，默认: {resource_display("site_device_counts.json")}')
    args = parser.parse_args()

    result = count_site_devices(args.ne_graph)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"站点数: {len(result)}")
    print(f"已保存到: {args.output}")

    # 示例
    sample = list(result.items())[0]
    print(f"示例: {sample}")


if __name__ == "__main__":
    main()

