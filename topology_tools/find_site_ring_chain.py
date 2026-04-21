#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入两个站点，查找同时包含二者的站点环链。"""

import argparse
import json

from collections import defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display
from topology_tools.site_pair_order_common import (
    _get_site_id,
    find_bridges,
    iter_unique_cross_site_links,
)


def normalize_site_id(site_id):
    return str(site_id or "").strip().upper()


def edge_key(site_a, site_b):
    return tuple(sorted((site_a, site_b)))


def build_site_graph_from_ne(ne_graph):
    """用和 site_pair_order 脚本一致的跨站链路过滤逻辑构建无向站点图。"""
    adjacency = defaultdict(set)
    site_edges = {}
    all_sites = set()

    for ne_info in ne_graph.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = _get_site_id(ne_info)
        if site_id:
            all_sites.add(site_id)
            adjacency[site_id]

    for link in iter_unique_cross_site_links(ne_graph):
        source_site = link["source_site"]
        target_site = link["target_site"]
        key = edge_key(source_site, target_site)

        adjacency[source_site].add(target_site)
        adjacency[target_site].add(source_site)

        edge = site_edges.setdefault(key, {
            "site_a": key[0],
            "site_b": key[1],
            "link_count": 0,
            "link_types": set(),
            "ne_pairs": [],
        })
        edge["link_count"] += 1
        edge["link_types"].add(link.get("link_type", "__unknown__"))
        edge["ne_pairs"].append({
            "source_ne": link["source_ne"],
            "target_ne": link["target_ne"],
            "source_site": source_site,
            "target_site": target_site,
            "source_domain": link["source_domain"],
            "target_domain": link["target_domain"],
            "link_type": link.get("link_type", "__unknown__"),
        })

    return all_sites, adjacency, site_edges


def find_non_bridge_component(adjacency, bridge_edges, site_a, site_b):
    """查找移除桥边后同时包含两个站点的环块。"""
    non_bridge_adjacency = defaultdict(set)
    bridge_neighbors = defaultdict(set)

    for source_site, neighbors in adjacency.items():
        for target_site in neighbors:
            if source_site > target_site:
                continue
            key = edge_key(source_site, target_site)
            if key in bridge_edges:
                bridge_neighbors[source_site].add(target_site)
                bridge_neighbors[target_site].add(source_site)
            else:
                non_bridge_adjacency[source_site].add(target_site)
                non_bridge_adjacency[target_site].add(source_site)

    visited = set()
    for start_site in sorted(non_bridge_adjacency):
        if start_site in visited:
            continue

        queue = deque([start_site])
        visited.add(start_site)
        component = []
        while queue:
            current_site = queue.popleft()
            component.append(current_site)
            for neighbor_site in sorted(non_bridge_adjacency.get(current_site, ())):
                if neighbor_site in visited:
                    continue
                visited.add(neighbor_site)
                queue.append(neighbor_site)

        component_set = set(component)
        if site_a not in component_set or site_b not in component_set:
            continue

        external_start_candidates = sorted(
            site_id
            for site_id in component_set
            if any(neighbor not in component_set for neighbor in bridge_neighbors.get(site_id, ()))
        )
        start_site = external_start_candidates[0] if len(external_start_candidates) == 1 else None
        return {
            "sites": sorted(component_set),
            "site_count": len(component_set),
            "external_start_candidates": external_start_candidates,
            "entry_exit_sites": external_start_candidates,
            "start_site": start_site,
        }

    return None


def build_entry_exit_edges(component_sites, site_edges, bridge_edges, include_ne_pairs=False):
    """输出环块通过桥边连到外部的出入口边。"""
    component_set = set(component_sites)
    output = []

    for key in sorted(bridge_edges):
        site_a, site_b = key
        site_a_in = site_a in component_set
        site_b_in = site_b in component_set
        if site_a_in == site_b_in:
            continue

        ring_site = site_a if site_a_in else site_b
        outside_site = site_b if site_a_in else site_a
        edge = site_edges.get(key, {
            "link_count": 0,
            "link_types": set(),
            "ne_pairs": [],
        })
        rec = {
            "ring_site": ring_site,
            "outside_site": outside_site,
            "site_a": site_a,
            "site_b": site_b,
            "link_count": edge["link_count"],
            "link_types": sorted(edge["link_types"]),
        }
        if include_ne_pairs:
            rec["ne_pairs"] = edge["ne_pairs"]
        output.append(rec)

    return output


def component_edges(component_sites, site_edges, include_ne_pairs=False):
    component_set = set(component_sites)
    output = []
    for key in sorted(site_edges):
        site_a, site_b = key
        if site_a not in component_set or site_b not in component_set:
            continue
        edge = site_edges[key]
        rec = {
            "site_a": site_a,
            "site_b": site_b,
            "link_count": edge["link_count"],
            "link_types": sorted(edge["link_types"]),
        }
        if include_ne_pairs:
            rec["ne_pairs"] = edge["ne_pairs"]
        output.append(rec)
    return output


def shortest_path(adjacency, start_site, end_site, allowed_sites, banned_nodes=None, banned_edges=None, max_depth=None):
    banned_nodes = set(banned_nodes or ())
    banned_edges = set(banned_edges or ())
    allowed_sites = set(allowed_sites)
    queue = deque([[start_site]])

    while queue:
        path = queue.popleft()
        current_site = path[-1]
        if max_depth is not None and len(path) - 1 >= max_depth:
            continue

        for neighbor_site in sorted(adjacency.get(current_site, ())):
            if neighbor_site not in allowed_sites:
                continue
            if neighbor_site in banned_nodes and neighbor_site != end_site:
                continue
            if edge_key(current_site, neighbor_site) in banned_edges:
                continue
            if neighbor_site in path:
                continue

            next_path = path + [neighbor_site]
            if neighbor_site == end_site:
                return next_path
            queue.append(next_path)

    return None


def enumerate_base_paths(adjacency, start_site, end_site, allowed_sites, max_depth, max_paths, max_expanded):
    allowed_sites = set(allowed_sites)
    queue = deque([[start_site]])
    paths = []
    expanded = 0

    while queue and len(paths) < max_paths and expanded < max_expanded:
        path = queue.popleft()
        expanded += 1
        current_site = path[-1]

        if len(path) - 1 >= max_depth:
            continue

        for neighbor_site in sorted(adjacency.get(current_site, ())):
            if neighbor_site not in allowed_sites or neighbor_site in path:
                continue

            next_path = path + [neighbor_site]
            if neighbor_site == end_site:
                paths.append(next_path)
                if len(paths) >= max_paths:
                    break
            else:
                queue.append(next_path)

    return paths, expanded


def path_edges(path):
    return {
        edge_key(path[index], path[index + 1])
        for index in range(len(path) - 1)
    }


def cycle_from_paths(path_one, path_two):
    return path_one + list(reversed(path_two))[1:]


def find_cycles_containing_sites(adjacency, component_sites, site_a, site_b, max_depth, max_base_paths, max_cycles, max_expanded):
    base_paths, expanded = enumerate_base_paths(
        adjacency,
        site_a,
        site_b,
        component_sites,
        max_depth=max_depth,
        max_paths=max_base_paths,
        max_expanded=max_expanded,
    )
    cycles = []
    seen_cycle_edges = set()

    for path_one in base_paths:
        banned_nodes = set(path_one[1:-1])
        banned_edges = path_edges(path_one)
        path_two = shortest_path(
            adjacency,
            site_a,
            site_b,
            component_sites,
            banned_nodes=banned_nodes,
            banned_edges=banned_edges,
            max_depth=max_depth,
        )
        if not path_two:
            continue

        cycle_sites = cycle_from_paths(path_one, path_two)
        cycle_edges = frozenset(path_edges(cycle_sites))
        if cycle_edges in seen_cycle_edges:
            continue
        seen_cycle_edges.add(cycle_edges)

        cycles.append({
            "cycle_id": len(cycles),
            "site_count": len(set(cycle_sites)),
            "edge_count": len(cycle_sites) - 1,
            "path_a_to_b_1": path_one,
            "path_a_to_b_2": path_two,
            "cycle_sites": cycle_sites,
        })
        if len(cycles) >= max_cycles:
            break

    return cycles, {
        "base_path_count": len(base_paths),
        "expanded_path_count": expanded,
    }


def build_result(ne_graph_path, site_a, site_b, args):
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)

    all_sites, adjacency, site_edges = build_site_graph_from_ne(ne_graph)
    bridge_edges = find_bridges(adjacency)

    result = {
        "meta": {
            "ne_graph": str(ne_graph_path),
            "site_a": site_a,
            "site_b": site_b,
            "site_count": len(all_sites),
            "edge_count": len(site_edges),
            "bridge_edge_count": len(bridge_edges),
            "max_depth": args.max_depth,
            "max_base_paths": args.max_base_paths,
            "max_cycles": args.max_cycles,
        },
        "found": False,
        "reason": "",
        "ring_component": None,
        "cycles": [],
        "search_stats": {},
    }

    missing_sites = [site for site in (site_a, site_b) if site not in all_sites]
    if missing_sites:
        result["reason"] = f"站点不在 ne_graph 中: {', '.join(missing_sites)}"
        return result

    ring_component = find_non_bridge_component(adjacency, bridge_edges, site_a, site_b)
    if not ring_component:
        result["reason"] = "两个站点不在同一个非桥边环块中，未找到同时包含二者的环链"
        return result

    ring_component["entry_exit_edges"] = build_entry_exit_edges(
        ring_component["sites"],
        site_edges,
        bridge_edges,
        include_ne_pairs=args.include_ne_pairs,
    )
    ring_component["edges"] = component_edges(
        ring_component["sites"],
        site_edges,
        include_ne_pairs=args.include_ne_pairs,
    )
    result["ring_component"] = ring_component

    cycles, search_stats = find_cycles_containing_sites(
        adjacency,
        ring_component["sites"],
        site_a,
        site_b,
        max_depth=args.max_depth,
        max_base_paths=args.max_base_paths,
        max_cycles=args.max_cycles,
        max_expanded=args.max_expanded,
    )
    result["cycles"] = cycles
    result["search_stats"] = search_stats
    result["found"] = bool(cycles)
    if cycles:
        result["reason"] = "已找到同时包含两个站点的环链"
    else:
        result["reason"] = "两个站点在同一环块中，但在当前搜索限制内未枚举到具体 simple cycle"
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="输入两个站点，基于 ne_graph.json 查找同时包含二者的站点环链"
    )
    parser.add_argument("site_a", help="站点 A")
    parser.add_argument("site_b", help="站点 B")
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument("-o", "--output", help="输出完整 JSON 文件；不指定则只打印摘要")
    parser.add_argument("--max-depth", type=int, default=30, help="单条 A-B 路径的最大 hop 数")
    parser.add_argument("--max-base-paths", type=int, default=200, help="最多枚举多少条 A-B 基础路径")
    parser.add_argument("--max-cycles", type=int, default=20, help="最多输出多少个环")
    parser.add_argument("--max-expanded", type=int, default=100000, help="枚举路径时最多展开多少个状态")
    parser.add_argument(
        "--include-ne-pairs",
        action="store_true",
        help="在环块边信息中输出底层 NE 对明细，可能会让 JSON 很大",
    )
    args = parser.parse_args()

    if args.max_depth < 2:
        parser.error("max-depth 至少为 2")
    if args.max_base_paths <= 0:
        parser.error("max-base-paths 必须大于 0")
    if args.max_cycles <= 0:
        parser.error("max-cycles 必须大于 0")
    if args.max_expanded <= 0:
        parser.error("max-expanded 必须大于 0")
    return args


def print_summary(result):
    print(f"站点数: {result['meta']['site_count']}")
    print(f"站点边数: {result['meta']['edge_count']}")
    print(f"桥边数: {result['meta']['bridge_edge_count']}")
    print(f"是否找到环链: {'是' if result['found'] else '否'}")
    print(f"原因: {result['reason']}")

    component = result.get("ring_component")
    if component:
        print(f"环块站点数: {component['site_count']}")
        print(f"环块边数: {len(component['edges'])}")
        print(f"环块出入口数: {len(component.get('entry_exit_edges', []))}")
        if component.get("start_site"):
            print(f"唯一起始点: {component['start_site']}")
        elif component.get("external_start_candidates"):
            print(f"外部接入候选: {', '.join(component['external_start_candidates'])}")
        for edge in component.get("entry_exit_edges", []):
            print(
                "  出入口: "
                f"{edge['ring_site']} <-> {edge['outside_site']} "
                f"links={edge['link_count']} types={','.join(edge['link_types'])}"
            )

    for cycle in result.get("cycles", []):
        print(
            f"[cycle {cycle['cycle_id']}] "
            f"sites={cycle['site_count']} edges={cycle['edge_count']} "
            f"path={' -> '.join(cycle['cycle_sites'])}"
        )


def main():
    args = parse_args()
    ne_graph_path = Path(args.ne_graph)
    if not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    site_a = normalize_site_id(args.site_a)
    site_b = normalize_site_id(args.site_b)
    if site_a == site_b:
        raise SystemExit("两个输入站点不能相同")

    result = build_result(ne_graph_path, site_a, site_b, args)
    print_summary(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
