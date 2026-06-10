#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为每个站点生成双向邻居、下游集合、上游集合。"""

import argparse
import json

from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import resource_display, resource_path
from topology_tools.find_site_chain import (
    DEFAULT_PREDICTION_CANDIDATES,
    add_adjacency_edge,
    build_adjacency,
    default_prediction_json,
    looks_like_downstream_map,
    normalize_downstream_map,
    normalize_site_id,
)
from topology_tools.site_pair_order_common import (
    ProgressReporter,
    _get_site_id,
    normalize_domain,
)


def build_bidirectional_neighbors(data):
    """提取直接双向边连接的站点集合。"""
    neighbors = defaultdict(set)
    all_sites = set()

    if isinstance(data, dict) and isinstance(data.get("edges"), list):
        for edge in data["edges"]:
            if not isinstance(edge, dict):
                continue
            site_a = normalize_site_id(edge.get("site_a"))
            site_b = normalize_site_id(edge.get("site_b"))
            if site_a:
                all_sites.add(site_a)
            if site_b:
                all_sites.add(site_b)
            if edge.get("prediction") != "bidirectional":
                continue
            if not site_a or not site_b or site_a == site_b:
                continue
            neighbors[site_a].add(site_b)
            neighbors[site_b].add(site_a)
        return neighbors, all_sites, "edges"

    if isinstance(data, dict) and isinstance(data.get("downstream_map"), dict):
        adjacency, all_sites = normalize_downstream_map(data["downstream_map"])
        source = "downstream_map_mutual"
    elif looks_like_downstream_map(data):
        adjacency, all_sites = normalize_downstream_map(data)
        source = "plain_downstream_map_mutual"
    else:
        return neighbors, all_sites, "none"

    for site_id, downstream_sites in adjacency.items():
        for downstream_site in downstream_sites:
            if site_id in adjacency.get(downstream_site, set()):
                neighbors[site_id].add(downstream_site)
                neighbors[downstream_site].add(site_id)
    return neighbors, all_sites, source


def site_pair_key(site_a, site_b):
    return tuple(sorted((normalize_site_id(site_a), normalize_site_id(site_b))))


def collect_prediction_pairs(data):
    """收集 prediction/downstream_map 中已经存在的站点对。"""
    pairs = set()

    if isinstance(data, dict) and isinstance(data.get("edges"), list):
        for edge in data["edges"]:
            if not isinstance(edge, dict):
                continue
            site_a = normalize_site_id(edge.get("site_a"))
            site_b = normalize_site_id(edge.get("site_b"))
            if site_a and site_b and site_a != site_b:
                pairs.add(site_pair_key(site_a, site_b))
        return pairs

    raw_map = None
    if isinstance(data, dict) and isinstance(data.get("downstream_map"), dict):
        raw_map = data["downstream_map"]
    elif looks_like_downstream_map(data):
        raw_map = data

    if raw_map:
        for site_id, downstream_sites in raw_map.items():
            if not isinstance(downstream_sites, (list, tuple, set)):
                continue
            for downstream_site in downstream_sites:
                site_a = normalize_site_id(site_id)
                site_b = normalize_site_id(downstream_site)
                if site_a and site_b and site_a != site_b:
                    pairs.add(site_pair_key(site_a, site_b))
    return pairs


def domain_tuple_index(domain):
    domain = normalize_domain(domain)
    domain = str(domain or "").strip().lower()
    if domain == "data":
        return 0
    if domain == "transmission":
        return 1
    if domain == "ran":
        return 2
    return None


def iter_raw_unique_cross_site_links(ne_graph, show_progress=False):
    """不做 domain 过滤，按 NE 对 + link_type 去重遍历原始跨站连边。"""
    seen = set()

    with ProgressReporter(len(ne_graph), "site_chains: 扫描原始 ne_graph 连边", show_progress) as progress:
        for source_ne, source_info in ne_graph.items():
            progress.update()
            if not isinstance(source_info, dict):
                continue

            source_site = normalize_site_id(_get_site_id(source_info))
            if not source_site:
                continue

            raw_links = source_info.get("link", {})
            if not isinstance(raw_links, dict):
                continue

            source_domain = normalize_domain(source_info.get("domain", ""))
            for target_ne, link_meta in raw_links.items():
                target_info = ne_graph.get(target_ne)
                if not isinstance(target_info, dict):
                    continue

                target_site = normalize_site_id(_get_site_id(target_info))
                if not target_site or target_site == source_site:
                    continue

                target_domain = normalize_domain(target_info.get("domain", ""))
                link_types = (
                    sorted(link_meta.keys())
                    if isinstance(link_meta, dict) and link_meta
                    else ["__unknown__"]
                )

                for link_type in link_types:
                    key = tuple(sorted((source_ne, target_ne))) + (str(link_type),)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {
                        "source_ne": source_ne,
                        "target_ne": target_ne,
                        "source_site": source_site,
                        "target_site": target_site,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "link_type": str(link_type),
                    }


def collect_missing_ne_graph_pair_domain_counts(ne_graph, prediction_pairs, show_progress=False):
    """只为 prediction 未覆盖的站点对统计原始连边两端设备 domain 数量。"""
    relation_data = collect_ne_graph_relation_data(
        ne_graph,
        prediction_pairs=prediction_pairs,
        collect_missing_counts=True,
        show_progress=show_progress,
    )
    return relation_data["missing_pair_counts"], relation_data["stats"]


def collect_ne_graph_relation_data(
    ne_graph,
    prediction_pairs=None,
    *,
    collect_missing_counts=False,
    show_progress=False,
):
    """收集 ne_graph 中真实出现过跨站连边的站点对，可选统计缺失关系补边证据。"""
    prediction_pairs = prediction_pairs or set()
    ne_graph_pairs = set()
    missing_pair_counts = {}
    skipped_prediction_link_count = 0
    raw_cross_site_link_count = 0

    for link in iter_raw_unique_cross_site_links(ne_graph, show_progress=show_progress):
        source_site = normalize_site_id(link["source_site"])
        target_site = normalize_site_id(link["target_site"])
        if not source_site or not target_site or source_site == target_site:
            continue

        key = site_pair_key(source_site, target_site)
        ne_graph_pairs.add(key)
        raw_cross_site_link_count += 1

        if not collect_missing_counts:
            continue
        if key in prediction_pairs:
            skipped_prediction_link_count += 1
            continue

        rec = missing_pair_counts.setdefault(key, {
            "site_counts": {
                key[0]: [0, 0, 0],
                key[1]: [0, 0, 0],
            },
            "link_count": 0,
        })
        rec["link_count"] += 1

        source_index = domain_tuple_index(link.get("source_domain"))
        target_index = domain_tuple_index(link.get("target_domain"))
        if source_index is not None:
            rec["site_counts"][source_site][source_index] += 1
        if target_index is not None:
            rec["site_counts"][target_site][target_index] += 1

    return {
        "ne_graph_pairs": ne_graph_pairs,
        "missing_pair_counts": missing_pair_counts,
        "stats": {
            "ne_graph_pair_count": len(ne_graph_pairs),
            "raw_cross_site_link_count": raw_cross_site_link_count,
            "skipped_prediction_link_count": skipped_prediction_link_count,
        },
    }


def apply_ne_graph_augmentation_from_counts(
    ne_graph_path,
    prediction_pair_count,
    missing_pair_counts,
    collect_stats,
    adjacency,
    first_hop_adjacency,
    bidirectional_neighbors,
    all_sites,
    *,
    directed_only=False,
    show_progress=True,
):
    """用 ne_graph 中 prediction 未覆盖的站点连边补充方向关系。"""
    stats = {
        "ne_graph": str(ne_graph_path),
        "prediction_pair_count": prediction_pair_count,
        "ne_graph_pair_count": collect_stats["ne_graph_pair_count"],
        "skipped_prediction_link_count": collect_stats["skipped_prediction_link_count"],
        "augmented_pair_count": 0,
        "augmented_directed_pair_count": 0,
        "augmented_bidirectional_pair_count": 0,
        "domain_tuple_semantics": "raw_inter_site_link_endpoint_domains",
    }

    with ProgressReporter(len(missing_pair_counts), "site_chains: 应用 ne_graph 补边", show_progress) as progress:
        for pair_key, rec in sorted(missing_pair_counts.items()):
            progress.update()
            site_a, site_b = pair_key
            all_sites.update(pair_key)

            tuple_a = tuple(rec["site_counts"].get(site_a, [0, 0, 0]))
            tuple_b = tuple(rec["site_counts"].get(site_b, [0, 0, 0]))
            stats["augmented_pair_count"] += 1

            if tuple_a > tuple_b:
                add_adjacency_edge(adjacency, all_sites, site_a, site_b)
                add_adjacency_edge(first_hop_adjacency, all_sites, site_a, site_b)
                stats["augmented_directed_pair_count"] += 1
            elif tuple_b > tuple_a:
                add_adjacency_edge(adjacency, all_sites, site_b, site_a)
                add_adjacency_edge(first_hop_adjacency, all_sites, site_b, site_a)
                stats["augmented_directed_pair_count"] += 1
            else:
                bidirectional_neighbors[site_a].add(site_b)
                bidirectional_neighbors[site_b].add(site_a)
                if not directed_only:
                    add_adjacency_edge(adjacency, all_sites, site_a, site_b)
                    add_adjacency_edge(adjacency, all_sites, site_b, site_a)
                stats["augmented_bidirectional_pair_count"] += 1

    return stats


def apply_ne_graph_augmentation(
    data,
    ne_graph_path,
    adjacency,
    first_hop_adjacency,
    bidirectional_neighbors,
    all_sites,
    *,
    directed_only=False,
    show_progress=True,
):
    """兼容旧调用：加载 ne_graph 后补充 prediction 未覆盖的站点关系。"""
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)

    prediction_pairs = collect_prediction_pairs(data)
    relation_data = collect_ne_graph_relation_data(
        ne_graph,
        prediction_pairs=prediction_pairs,
        collect_missing_counts=True,
        show_progress=show_progress,
    )
    return apply_ne_graph_augmentation_from_counts(
        ne_graph_path,
        len(prediction_pairs),
        relation_data["missing_pair_counts"],
        relation_data["stats"],
        adjacency,
        first_hop_adjacency,
        bidirectional_neighbors,
        all_sites,
        directed_only=directed_only,
        show_progress=show_progress,
    )


def reachable_downstream_sites(adjacency, first_hop_adjacency, source_site, max_depth=None):
    """和 find_site_chain.py 保持一致：第一跳走 first_hop_adjacency，后续走 adjacency。"""
    source_site = normalize_site_id(source_site)
    visited = {source_site}
    depth = {source_site: 0}
    queue = [source_site]
    head = 0

    while head < len(queue):
        current_site = queue[head]
        head += 1
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
            depth[next_site] = current_depth + 1
            queue.append(next_site)

    return {
        site_id: hop
        for site_id, hop in depth.items()
        if site_id != source_site
    }


def build_site_chains(
    prediction_path,
    *,
    ne_graph_path=None,
    enrich_relation=False,
    restrict_relation=False,
    directed_only=False,
    max_depth=None,
    show_progress=True,
):
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    adjacency, first_hop_adjacency, all_sites, adjacency_source, edge_stats, warnings = build_adjacency(
        data,
        directed_only=directed_only,
    )
    warnings = list(warnings or [])
    bidirectional_neighbors, bidirectional_sites, bidirectional_source = build_bidirectional_neighbors(data)
    all_sites.update(bidirectional_sites)
    relation_data = None
    augmentation_stats = None
    restriction_stats = None

    if ne_graph_path and (enrich_relation or restrict_relation):
        with open(ne_graph_path, "r", encoding="utf-8") as f:
            ne_graph = json.load(f)

        prediction_pairs = collect_prediction_pairs(data)
        relation_data = collect_ne_graph_relation_data(
            ne_graph,
            prediction_pairs=prediction_pairs,
            collect_missing_counts=enrich_relation,
            show_progress=show_progress,
        )
        if enrich_relation:
            augmentation_stats = apply_ne_graph_augmentation_from_counts(
                ne_graph_path,
                len(prediction_pairs),
                relation_data["missing_pair_counts"],
                relation_data["stats"],
                adjacency,
                first_hop_adjacency,
                bidirectional_neighbors,
                all_sites,
                directed_only=directed_only,
                show_progress=show_progress,
            )
    elif enrich_relation or restrict_relation:
        warnings.append("--enrich-relation/--restrict-relation 未生效：未提供 --ne-graph")

    if ne_graph_path and not enrich_relation and not restrict_relation:
        warnings.append("--ne-graph 已提供，但未开启 --enrich-relation/--restrict-relation；不会影响站点关系")

    if restrict_relation and not ne_graph_path:
        restriction_stats = {
            "enabled": False,
            "reason": "missing_ne_graph",
        }
    elif restrict_relation:
        restriction_stats = {
            "enabled": True,
            "mode": "filter_final_downstream_site_hops",
            "ne_graph": str(ne_graph_path),
            "ne_graph_pair_count": len(relation_data["ne_graph_pairs"]) if relation_data else 0,
            "downstream_relation_count_before": 0,
            "downstream_relation_count_after": 0,
            "removed_downstream_relation_count": 0,
        }

    all_sites.update(adjacency.keys())
    all_sites.update(first_hop_adjacency.keys())
    for downstream_sites in adjacency.values():
        all_sites.update(downstream_sites)
    for downstream_sites in first_hop_adjacency.values():
        all_sites.update(downstream_sites)

    downstream_hops_by_site = {}
    upstream_hops_by_site = {site_id: {} for site_id in all_sites}

    sorted_sites = sorted(all_sites)
    total_sites = len(sorted_sites)
    with ProgressReporter(total_sites, "site_chains: 生成每站点链路集合", show_progress) as progress:
        for site_id in sorted_sites:
            progress.update()
            downstream_site_hops = reachable_downstream_sites(
                adjacency,
                first_hop_adjacency,
                site_id,
                max_depth=max_depth,
            )
            if restriction_stats and restriction_stats.get("enabled"):
                before_count = len(downstream_site_hops)
                ne_graph_pairs = relation_data["ne_graph_pairs"] if relation_data else set()
                downstream_site_hops = {
                    downstream_site: hop
                    for downstream_site, hop in downstream_site_hops.items()
                    if site_pair_key(site_id, downstream_site) in ne_graph_pairs
                }
                after_count = len(downstream_site_hops)
                restriction_stats["downstream_relation_count_before"] += before_count
                restriction_stats["downstream_relation_count_after"] += after_count
                restriction_stats["removed_downstream_relation_count"] += before_count - after_count
            downstream_hops_by_site[site_id] = downstream_site_hops
            for downstream_site, hop in downstream_site_hops.items():
                upstream_hops_by_site.setdefault(downstream_site, {})[site_id] = hop

    site_chains = {}
    for site_id in sorted_sites:
        downstream_site_hops = downstream_hops_by_site.get(site_id, {})
        upstream_site_hops = upstream_hops_by_site.get(site_id, {})
        site_chains[site_id] = {
            "bidirectional_sites": sorted(bidirectional_neighbors.get(site_id, set())),
            "downstream_site_hops": {
                downstream_site: downstream_site_hops[downstream_site]
                for downstream_site in sorted(downstream_site_hops)
            },
            "upstream_site_hops": {
                upstream_site: upstream_site_hops[upstream_site]
                for upstream_site in sorted(upstream_site_hops)
            },
        }

    edge_stats = edge_stats or {}
    meta = {
        "input_config": {
            "prediction_json": str(prediction_path),
            "ne_graph": str(ne_graph_path) if ne_graph_path else None,
            "max_depth": max_depth,
            "directed_only": directed_only,
            "enrich_relation": enrich_relation,
            "restrict_relation": restrict_relation,
        },
        "adjacency_source": adjacency_source,
        "bidirectional_source": bidirectional_source,
        "first_hop_downstream_only": True,
        "site_count": len(sorted_sites),
        "warning_count": len(warnings),
        "warnings": warnings,
        "edge_stats": edge_stats,
        "relation_options": {
            "ne_graph_provided": bool(ne_graph_path),
            "enrich_relation_requested": enrich_relation,
            "restrict_relation_requested": restrict_relation,
            "enrich_relation_effective": bool(ne_graph_path and enrich_relation),
            "restrict_relation_effective": bool(ne_graph_path and restrict_relation),
        },
    }
    if augmentation_stats:
        meta["ne_graph_augmentation"] = augmentation_stats
    if restriction_stats:
        meta["ne_graph_restriction"] = restriction_stats

    total_downstream_relations = sum(
        len(info["downstream_site_hops"])
        for info in site_chains.values()
    )
    total_upstream_relations = sum(
        len(info["upstream_site_hops"])
        for info in site_chains.values()
    )
    total_bidirectional_relations = sum(
        len(info["bidirectional_sites"])
        for info in site_chains.values()
    )
    meta.update({
        "total_downstream_relations": total_downstream_relations,
        "total_upstream_relations": total_upstream_relations,
        "total_bidirectional_directed_relations": total_bidirectional_relations,
        "total_bidirectional_edges": total_bidirectional_relations // 2,
    })

    return {
        "meta": meta,
        "sites": site_chains,
    }


def parse_args():
    default_json = default_prediction_json()
    parser = argparse.ArgumentParser(
        description="为每个站点生成双向邻居集合、下游集合、上游集合"
    )
    parser.add_argument(
        "--prediction-json",
        default=default_json,
        help=(
            "站点上下行预测 JSON，需包含 downstream_map 或 edges；"
            f"默认优先使用 {resource_display(DEFAULT_PREDICTION_CANDIDATES[0])}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=resource_path("site_chains.json"),
        help=f"输出 JSON，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument("--max-depth", type=int, help="最多向下游遍历多少跳；默认不限制")
    parser.add_argument(
        "--ne-graph",
        help=(
            "可选 ne_graph.json；仅作为 --enrich-relation/--restrict-relation 的数据源，"
            "单独提供不会改变站点关系"
        ),
    )
    parser.add_argument(
        "--enrich-relation",
        action="store_true",
        help=(
            "开启后，如果 prediction 未覆盖某站点对，则基于 ne_graph 连边两端设备 "
            "(data_num, transmission_num, ran_num) 三元组补充方向；未提供 --ne-graph 时失效"
        ),
    )
    parser.add_argument(
        "--restrict-relation",
        action="store_true",
        help=(
            "开启后，在生成每个站点的 downstream_site_hops 后，只保留与当前站点在 "
            "ne_graph 中存在跨站连边的下游站点；未提供 --ne-graph 时失效"
        ),
    )
    parser.add_argument(
        "--directed-only",
        action="store_true",
        help="所有 hop 都只沿显式 upstream_site -> downstream_site 边遍历，忽略 bidirectional 边",
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭进度显示")
    args = parser.parse_args()

    if args.max_depth is not None and args.max_depth < 0:
        parser.error("max-depth 不能小于 0")
    return args


def print_summary(result, output_path):
    meta = result["meta"]
    input_config = meta.get("input_config", {})
    print(f"输入文件: {input_config.get('prediction_json')}")
    print(f"输出文件: {output_path}")
    print(f"站点数: {meta['site_count']}")
    print(f"邻接来源: {meta['adjacency_source']}")
    print(f"双向边来源: {meta['bidirectional_source']}")
    print("第一跳约束: 必须走显式下游边")
    print(f"后续遍历模式: {'只走显式有向边' if input_config.get('directed_only') else '按 downstream_map/双向边可双向传播'}")
    print(f"双向直接边数: {meta['total_bidirectional_edges']}")
    print(f"下游可达关系数: {meta['total_downstream_relations']}")
    print(f"上游可达关系数: {meta['total_upstream_relations']}")
    relation_options = meta.get("relation_options", {})
    if relation_options:
        print(f"ne_graph关系补充: {'开启' if relation_options.get('enrich_relation_effective') else '关闭'}")
        print(f"ne_graph下游结果裁剪: {'开启' if relation_options.get('restrict_relation_effective') else '关闭'}")
    augmentation_stats = meta.get("ne_graph_augmentation")
    if augmentation_stats:
        print(f"ne_graph补充候选站点对数: {augmentation_stats['ne_graph_pair_count']}")
        print(f"prediction已覆盖跳过连边数: {augmentation_stats['skipped_prediction_link_count']}")
        print(f"ne_graph补充站点对数: {augmentation_stats['augmented_pair_count']}")
        print(f"ne_graph补充有向边数: {augmentation_stats['augmented_directed_pair_count']}")
        print(f"ne_graph补充双向边数: {augmentation_stats['augmented_bidirectional_pair_count']}")
    restriction_stats = meta.get("ne_graph_restriction")
    if restriction_stats and restriction_stats.get("enabled"):
        print(f"ne_graph裁剪站点对数: {restriction_stats['ne_graph_pair_count']}")
        print(f"裁剪前下游关系数: {restriction_stats['downstream_relation_count_before']}")
        print(f"裁剪后下游关系数: {restriction_stats['downstream_relation_count_after']}")
        print(f"移除下游关系数: {restriction_stats['removed_downstream_relation_count']}")
    for warning in meta.get("warnings", []):
        print(f"警告: {warning}")


def main():
    args = parse_args()
    prediction_path = Path(args.prediction_json)
    if not prediction_path.exists():
        raise SystemExit(f"未找到站点上下行预测 JSON: {args.prediction_json}")
    ne_graph_path = Path(args.ne_graph) if args.ne_graph else None
    if ne_graph_path and not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    result = build_site_chains(
        prediction_path,
        ne_graph_path=ne_graph_path,
        enrich_relation=args.enrich_relation,
        restrict_relation=args.restrict_relation,
        directed_only=args.directed_only,
        max_depth=args.max_depth,
        show_progress=not args.no_progress,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_summary(result, output_path)


if __name__ == "__main__":
    main()
