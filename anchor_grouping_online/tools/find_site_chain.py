#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入一个站点，查找其下游可达站点。"""

import argparse
import json

from collections import defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.tools.topology_resources import resource_display, resource_path


DEFAULT_PREDICTION_CANDIDATES = [
    "site_pair_order_global_path_optimized_ring.json",
    "site_pair_order_global_path_voting_ring.json",
    "site_pair_order_global_no_path_ring.json",
    "site_pair_order_pairwise_ring.json",
]


def normalize_site_id(site_id):
    return str(site_id or "").strip().upper()


def default_prediction_json():
    for name in DEFAULT_PREDICTION_CANDIDATES:
        path = Path(resource_path(name))
        if path.exists():
            return str(path)
    return resource_path(DEFAULT_PREDICTION_CANDIDATES[0])


def add_adjacency_edge(adjacency, all_sites, source_site, target_site):
    source_site = normalize_site_id(source_site)
    target_site = normalize_site_id(target_site)
    if not source_site or not target_site or source_site == target_site:
        return
    adjacency[source_site].add(target_site)
    all_sites.add(source_site)
    all_sites.add(target_site)


def normalize_downstream_map(raw_map):
    adjacency = defaultdict(set)
    all_sites = set()

    if not isinstance(raw_map, dict):
        raise ValueError("downstream_map 必须是 dict")

    for source_site, downstream_sites in raw_map.items():
        source_site = normalize_site_id(source_site)
        if not source_site:
            continue
        all_sites.add(source_site)
        if not isinstance(downstream_sites, (list, tuple, set)):
            continue
        for target_site in downstream_sites:
            add_adjacency_edge(adjacency, all_sites, source_site, target_site)

    return adjacency, all_sites


def parse_prediction_direction(edge):
    return (
        normalize_site_id(edge.get("upstream_site")),
        normalize_site_id(edge.get("downstream_site")),
    )


def build_adjacency_from_edges(edges, include_bidirectional=True):
    adjacency = defaultdict(set)
    all_sites = set()
    edge_stats = {
        "edge_count": 0,
        "directed_edge_count": 0,
        "bidirectional_edge_count": 0,
        "ignored_edge_count": 0,
    }

    if not isinstance(edges, list):
        raise ValueError("edges 必须是 list")

    for edge in edges:
        if not isinstance(edge, dict):
            edge_stats["ignored_edge_count"] += 1
            continue

        site_a = normalize_site_id(edge.get("site_a"))
        site_b = normalize_site_id(edge.get("site_b"))
        if site_a:
            all_sites.add(site_a)
        if site_b:
            all_sites.add(site_b)

        edge_stats["edge_count"] += 1
        prediction = edge.get("prediction")
        if prediction == "bidirectional":
            edge_stats["bidirectional_edge_count"] += 1
            if include_bidirectional and site_a and site_b:
                add_adjacency_edge(adjacency, all_sites, site_a, site_b)
                add_adjacency_edge(adjacency, all_sites, site_b, site_a)
            continue

        upstream_site, downstream_site = parse_prediction_direction(edge)
        if upstream_site and downstream_site:
            edge_stats["directed_edge_count"] += 1
            add_adjacency_edge(adjacency, all_sites, upstream_site, downstream_site)
        else:
            edge_stats["ignored_edge_count"] += 1

    return adjacency, all_sites, edge_stats


def build_adjacency(data, directed_only=False):
    warnings = []
    source = "unknown"
    edge_stats = None
    first_hop_adjacency = None

    if isinstance(data, dict) and directed_only and isinstance(data.get("edges"), list):
        adjacency, all_sites, edge_stats = build_adjacency_from_edges(
            data["edges"],
            include_bidirectional=False,
        )
        first_hop_adjacency = adjacency
        source = "edges_directed_only"
    elif isinstance(data, dict) and isinstance(data.get("downstream_map"), dict):
        if not isinstance(data.get("edges"), list):
            raise ValueError("downstream_map 输入必须同时包含 edges，用于第一跳排除双向边")
        adjacency, all_sites = normalize_downstream_map(data["downstream_map"])
        source = "downstream_map"
        first_hop_adjacency, edge_sites, edge_stats = build_adjacency_from_edges(
            data["edges"],
            include_bidirectional=False,
        )
        all_sites.update(edge_sites)
    elif isinstance(data, dict) and isinstance(data.get("edges"), list):
        adjacency, all_sites, edge_stats = build_adjacency_from_edges(
            data["edges"],
            include_bidirectional=not directed_only,
        )
        if directed_only:
            first_hop_adjacency = adjacency
        else:
            first_hop_adjacency, first_hop_sites, _ = build_adjacency_from_edges(
                data["edges"],
                include_bidirectional=False,
            )
            all_sites.update(first_hop_sites)
        source = "edges"
    else:
        raise ValueError("输入 JSON 需要包含 downstream_map 或 edges")

    return adjacency, first_hop_adjacency, all_sites, source, edge_stats, warnings


def find_downstream_sites(adjacency, source_site, max_depth=None, first_hop_adjacency=None):
    source_site = normalize_site_id(source_site)
    first_hop_adjacency = first_hop_adjacency or adjacency
    visited = {source_site}
    parent = {}
    depth = {source_site: 0}
    queue = deque([source_site])

    while queue:
        current_site = queue.popleft()
        current_depth = depth[current_site]
        if max_depth is not None and current_depth >= max_depth:
            continue

        next_sites = (
            first_hop_adjacency.get(current_site, ())
            if current_depth == 0
            else adjacency.get(current_site, ())
        )
        for next_site in sorted(next_sites):
            if next_site in visited:
                continue
            visited.add(next_site)
            parent[next_site] = current_site
            depth[next_site] = current_depth + 1
            queue.append(next_site)

    downstream_sites = sorted(site for site in visited if site != source_site)
    return downstream_sites, depth, parent


def reconstruct_path(source_site, target_site, parent):
    path = [target_site]
    current_site = target_site
    while current_site != source_site and current_site in parent:
        current_site = parent[current_site]
        path.append(current_site)
    path.reverse()
    return path


def group_sites_by_hop(downstream_sites, depth):
    grouped = defaultdict(list)
    for site_id in downstream_sites:
        grouped[depth.get(site_id, -1)].append(site_id)
    return [
        {
            "hop": hop,
            "count": len(sites),
            "sites": sorted(sites),
        }
        for hop, sites in sorted(grouped.items())
        if hop >= 0
    ]


def build_result(prediction_path, source_site, args):
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    adjacency, first_hop_adjacency, all_sites, adjacency_source, edge_stats, warnings = build_adjacency(
        data,
        directed_only=args.directed_only,
    )
    source_site = normalize_site_id(source_site)
    downstream_sites, depth, parent = find_downstream_sites(
        adjacency,
        source_site,
        max_depth=args.max_depth,
        first_hop_adjacency=first_hop_adjacency,
    )

    result = {
        "meta": {
            "prediction_json": str(prediction_path),
            "source_site": source_site,
            "adjacency_source": adjacency_source,
            "directed_only": args.directed_only,
            "first_hop_downstream_only": True,
            "max_depth": args.max_depth,
            "site_count": len(all_sites),
            "downstream_count": len(downstream_sites),
            "direct_child_count": len(first_hop_adjacency.get(source_site, set())),
            "site_present": source_site in all_sites,
            "warnings": warnings,
        },
        "direct_children": sorted(first_hop_adjacency.get(source_site, set())),
        "downstream_sites": downstream_sites,
        "downstream_by_hop": group_sites_by_hop(downstream_sites, depth),
    }

    if edge_stats:
        result["meta"]["edge_stats"] = edge_stats

    if args.include_paths:
        result["paths"] = {
            site_id: reconstruct_path(source_site, site_id, parent)
            for site_id in downstream_sites
        }

    return result


def parse_args():
    default_json = default_prediction_json()
    parser = argparse.ArgumentParser(
        description="输入一个站点，基于站点上下行预测 JSON 查找其下游可达站点"
    )
    parser.add_argument("site", help="站点 ID")
    parser.add_argument(
        "--prediction-json",
        default=default_json,
        help=(
            "站点上下行预测 JSON，需包含 downstream_map 或 edges；"
            f"默认优先使用 {resource_display(DEFAULT_PREDICTION_CANDIDATES[0])}"
        ),
    )
    parser.add_argument("-o", "--output", help="输出完整 JSON 文件")
    parser.add_argument("--max-depth", type=int, help="最多向下游遍历多少跳；默认不限制")
    parser.add_argument(
        "--directed-only",
        action="store_true",
        help="所有 hop 都只沿显式 upstream_site -> downstream_site 边遍历，忽略 bidirectional 边",
    )
    parser.add_argument(
        "--include-paths",
        action="store_true",
        help="在 JSON 中输出从输入站点到每个下游站点的一条代表路径",
    )
    parser.add_argument("--max-print", type=int, default=200, help="摘要中最多打印多少个下游站点")
    args = parser.parse_args()

    if args.max_depth is not None and args.max_depth < 0:
        parser.error("max-depth 不能小于 0")
    if args.max_print < 0:
        parser.error("max-print 不能小于 0")
    return args


def print_summary(result, max_print):
    meta = result["meta"]
    print(f"输入文件: {meta['prediction_json']}")
    print(f"站点: {meta['source_site']}")
    print(f"站点是否存在于输入图: {'是' if meta['site_present'] else '否'}")
    print(f"邻接来源: {meta['adjacency_source']}")
    print("第一跳约束: 必须走显式下游边")
    print(f"后续遍历模式: {'只走显式有向边' if meta['directed_only'] else '按 downstream_map/双向边可双向传播'}")
    print(f"图内站点数: {meta['site_count']}")
    print(f"直接下游数: {meta['direct_child_count']}")
    print(f"全部下游站点数: {meta['downstream_count']}")

    for warning in meta.get("warnings", []):
        print(f"警告: {warning}")

    for group in result["downstream_by_hop"]:
        print(f"{group['hop']} hop: {group['count']} 个站点")

    sites = result["downstream_sites"]
    if max_print and sites:
        shown_sites = sites[:max_print]
        print("下游站点:")
        for site_id in shown_sites:
            print(f"  {site_id}")
        if len(sites) > max_print:
            print(f"  ... 还有 {len(sites) - max_print} 个未打印，可用 -o 输出完整 JSON")


def main():
    args = parse_args()
    prediction_path = Path(args.prediction_json)
    if not prediction_path.exists():
        raise SystemExit(f"未找到站点上下行预测 JSON: {args.prediction_json}")

    result = build_result(prediction_path, args.site, args)
    print_summary(result, args.max_print)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
