#!/usr/bin/env python3
"""Count device and site entries in topology JSON resources."""

from collections import Counter
import json
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


REGION_KEYS = (
    "region_id",
    "regionId",
    "regionId1",
    "region",
    "area_id",
    "area",
    "区域",
    "地市",
)
UNKNOWN_REGION = "未填区域"


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


def iter_top_level_records(data, entity_label, path):
    if isinstance(data, dict):
        return data.values()
    if isinstance(data, list):
        return data
    raise ValueError(
        f"{path} 的顶层结构是 {type(data).__name__}，"
        f"无法按顶层条目统计{entity_label}"
    )


def get_region(record, unknown_label=UNKNOWN_REGION):
    if not isinstance(record, dict):
        return unknown_label
    for key in REGION_KEYS:
        value = record.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return unknown_label


def count_entries_by_region(data, entity_label, path):
    counts = Counter()
    for record in iter_top_level_records(data, entity_label, path):
        counts[get_region(record)] += 1
    return counts


def sorted_regions(*counters):
    regions = set()
    for counter in counters:
        regions.update(counter)
    return sorted(regions, key=lambda value: (value == UNKNOWN_REGION, value))


def print_region_counts(site_counts, device_counts):
    regions = sorted_regions(site_counts, device_counts)
    if not regions:
        print("\n按区域统计: 无数据")
        return

    region_width = max(len("区域"), *(len(region) for region in regions))
    site_width = max(len("站点数"), *(len(str(site_counts.get(region, 0))) for region in regions))
    device_width = max(len("设备数"), *(len(str(device_counts.get(region, 0))) for region in regions))

    print("\n按区域统计:")
    print(
        f"{'区域':<{region_width}}  "
        f"{'站点数':>{site_width}}  "
        f"{'设备数':>{device_width}}"
    )
    print(
        f"{'-' * region_width}  "
        f"{'-' * site_width}  "
        f"{'-' * device_width}"
    )
    for region in regions:
        print(
            f"{region:<{region_width}}  "
            f"{site_counts.get(region, 0):>{site_width}}  "
            f"{device_counts.get(region, 0):>{device_width}}"
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

    ne_graph = load_json(args.ne_graph)
    site_graph = load_json(args.site_graph)

    device_count = count_top_level_entries(ne_graph, "设备", args.ne_graph)
    site_count = count_top_level_entries(site_graph, "站点", args.site_graph)
    device_counts_by_region = count_entries_by_region(ne_graph, "设备", args.ne_graph)
    site_counts_by_region = count_entries_by_region(site_graph, "站点", args.site_graph)

    print(f"ne_graph.json 设备数: {device_count}")
    print(f"site_graph.json 站点数: {site_count}")
    print_region_counts(site_counts_by_region, device_counts_by_region)


if __name__ == "__main__":
    main()
