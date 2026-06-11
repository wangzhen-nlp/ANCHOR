#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 site_graph.json 的站点数量和非空字段数量。"""

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import SITE_GRAPH_JSON, resource_display


def _is_non_empty(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def count_non_empty_fields(value):
    if isinstance(value, dict):
        total = 0
        for child in value.values():
            if isinstance(child, (dict, list, tuple, set)):
                total += count_non_empty_fields(child)
            elif _is_non_empty(child):
                total += 1
        return total
    if isinstance(value, (list, tuple, set)):
        return sum(count_non_empty_fields(child) for child in value)
    return 1 if _is_non_empty(value) else 0


def count_non_empty_top_level_fields(record):
    if not isinstance(record, dict):
        return 1 if _is_non_empty(record) else 0
    return sum(1 for value in record.values() if _is_non_empty(value))


def summarize_site_graph(site_graph_path):
    path = Path(site_graph_path)
    with path.open("r", encoding="utf-8") as fr:
        site_graph = json.load(fr)
    if not isinstance(site_graph, dict):
        raise ValueError(f"site_graph 顶层必须是对象: {path}")

    non_empty_field_count = 0
    non_empty_leaf_field_count = 0
    for site_info in site_graph.values():
        non_empty_field_count += count_non_empty_top_level_fields(site_info)
        non_empty_leaf_field_count += count_non_empty_fields(site_info)

    return {
        "input": str(path),
        "site_count": len(site_graph),
        "non_empty_field_count": non_empty_field_count,
        "non_empty_leaf_field_count": non_empty_leaf_field_count,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="统计 site_graph.json 的站点数量和非空字段数量")
    parser.add_argument(
        "site_graph",
        nargs="?",
        default=SITE_GRAPH_JSON,
        help=f"site_graph.json 输入文件，默认: {resource_display('site_graph.json')}",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = summarize_site_graph(args.site_graph)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
