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
from topology_tools.region_utils import REGION_KEYS, get_region as _get_region


UNKNOWN_REGION = "未填区域"


def load_json(path):
    """
    从指定路径加载 JSON 文件并返回解析后的数据。
    如果文件不存在，则抛出 FileNotFoundError 异常。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def count_top_level_entries(data, entity_label, path):
    """
    统计顶层数据的条目数量。
    支持 dict（统计键值对数量）和 list（统计元素个数）。
    """
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    raise ValueError(
        f"{path} 的顶层结构是 {type(data).__name__}，"
        f"无法按顶层条目统计{entity_label}"
    )


def iter_top_level_records(data, entity_label, path):
    """
    获取一个用于遍历顶层数据记录的迭代器。
    如果 data 是 dict，则遍历其 values；如果是 list，则直接遍历其元素。
    """
    if isinstance(data, dict):
        return data.values()
    if isinstance(data, list):
        return data
    raise ValueError(
        f"{path} 的顶层结构是 {type(data).__name__}，"
        f"无法按顶层条目统计{entity_label}"
    )


def get_region(record, unknown_label=UNKNOWN_REGION):
    """
    从单条记录 (dict) 中提取区域 (Region) 信息。
    通过遍历 REGION_KEYS 来匹配可能的键名，若未找到或值为空，则返回 unknown_label。
    """
    return _get_region(record, default=unknown_label)


def count_entries_by_region(data, entity_label, path):
    """
    按区域统计实体数量。
    遍历所有记录，提取区域字段并使用 Counter 进行分类计数。
    """
    counts = Counter()
    for record in iter_top_level_records(data, entity_label, path):
        counts[get_region(record)] += 1
    return counts


def sorted_regions(*counters):
    """
    合并多个 Counter 中的所有区域名称并进行排序。
    排序规则：普通区域按字典序排列，未知区域 (UNKNOWN_REGION) 放在最后。
    """
    regions = set()
    for counter in counters:
        regions.update(counter)
    return sorted(regions, key=lambda value: (value == UNKNOWN_REGION, value))


def print_region_counts(site_counts, device_counts):
    """
    以表格形式格式化并打印按区域统计的站点数和设备数。
    自动计算对齐宽度，保证中英文混合时的输出美观性。
    """
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
    parser.add_argument(
        "--details",
        action="store_true",
        help="额外按区域统计站点数和设备数",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 1. 加载设备与站点拓扑数据
    ne_graph = load_json(args.ne_graph)
    site_graph = load_json(args.site_graph)

    # 2. 统计全局数量
    device_count = count_top_level_entries(ne_graph, "设备", args.ne_graph)
    site_count = count_top_level_entries(site_graph, "站点", args.site_graph)
    
    # 3. 统计各区域的设备与站点数量
    device_counts_by_region = count_entries_by_region(ne_graph, "设备", args.ne_graph)
    site_counts_by_region = count_entries_by_region(site_graph, "站点", args.site_graph)

    # 4. 打印全局统计结果
    print(f"ne_graph.json 设备数: {device_count}")
    print(f"site_graph.json 站点数: {site_count}")
    
    # 5. 如果启用了 --details 参数，则打印按区域的详细统计
    if args.details:
        print_region_counts(site_counts_by_region, device_counts_by_region)


if __name__ == "__main__":
    main()
