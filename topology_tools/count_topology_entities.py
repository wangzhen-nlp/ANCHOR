#!/usr/bin/env python3
"""Count device and site entries in topology JSON resources."""

import json
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def count_top_level_entries(data, entity_label, path):
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    raise ValueError(
        f"{path} 的顶层结构是 {type(data).__name__}，"
        f"无法按顶层条目统计{entity_label}"
    )


def main():
    parser = ArgumentParser(description="统计 ne_graph.json 中的设备数和 site_graph.json 中的站点数")
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件路径，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"site_graph.json 文件路径，默认: {resource_display('site_graph.json')}",
    )
    args = parser.parse_args()

    device_count = count_top_level_entries(load_json(args.ne_graph), "设备", args.ne_graph)
    site_count = count_top_level_entries(load_json(args.site_graph), "站点", args.site_graph)

    print(f"ne_graph.json 设备数: {device_count}")
    print(f"site_graph.json 站点数: {site_count}")


if __name__ == "__main__":
    main()
