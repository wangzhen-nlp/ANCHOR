#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为每个站点生成双向邻居、下游集合、上游集合。"""

import argparse
import json

from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.tools.find_site_chain import (
    DEFAULT_PREDICTION_CANDIDATES,
    add_adjacency_edge,
    build_adjacency,
    default_prediction_json,
    normalize_site_id,
)
from anchor_grouping_online.tools.site_pair_order_common import (
    ProgressReporter,
    _get_site_id,
    is_transmission_domain,
    normalize_domain,
)
from anchor_grouping_online.tools.topology_resources import resource_display, resource_path

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

    return neighbors, all_sites, "none"


def site_pair_key(site_a, site_b):
    return tuple(sorted((normalize_site_id(site_a), normalize_site_id(site_b))))


def collect_prediction_pairs(data):
    """收集 prediction edges 中已经存在的站点对。"""
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


def iter_raw_unique_cross_site_links(
    ne_graph,
    show_progress=False,
    transmission_misconnection_pairs=None,
):
    """不做 domain 过滤，按 NE 对 + link_type 去重遍历原始跨站连边。

    transmission_misconnection_pairs（pair_key 集合）非空时，命中站点对之间
    任一端为传输类 domain 的连边被视为误连接，直接跳过。
    """
    seen = set()

    with ProgressReporter(len(ne_graph), "site_chains: 扫描原始 ne_graph 连边", show_progress) as progress:
        for source_ne, source_info in ne_graph.items():
            progress.update()
            if not isinstance(source_info, dict):
                continue
            yield from _iter_ne_cross_site_links(
                ne_graph, source_ne, source_info, seen,
                transmission_misconnection_pairs,
            )


def _iter_ne_cross_site_links(
    ne_graph, source_ne, source_info, seen, transmission_misconnection_pairs
):
    """产出单个源 NE 的去重跨站连边记录。"""
    source_site = normalize_site_id(_get_site_id(source_info))
    if not source_site:
        return
    raw_links = source_info.get("link", {})
    if not isinstance(raw_links, dict):
        return

    source_domain = normalize_domain(source_info.get("domain", ""))
    for target_ne, link_meta in raw_links.items():
        target_info = ne_graph.get(target_ne)
        if not isinstance(target_info, dict):
            continue

        target_site = normalize_site_id(_get_site_id(target_info))
        if not target_site or target_site == source_site:
            continue

        target_domain = normalize_domain(target_info.get("domain", ""))
        if _is_transmission_misconnection(
            source_site, target_site, source_domain, target_domain,
            transmission_misconnection_pairs,
        ):
            continue
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


def _is_transmission_misconnection(
    source_site, target_site, source_domain, target_domain,
    transmission_misconnection_pairs,
):
    """站点对命中误连接名单且任一端为传输类 domain 时视为误连接。"""
    if not transmission_misconnection_pairs:
        return False
    if not (
        is_transmission_domain(source_domain)
        or is_transmission_domain(target_domain)
    ):
        return False
    misconnection_key = (
        (source_site, target_site)
        if source_site <= target_site
        else (target_site, source_site)
    )
    return misconnection_key in transmission_misconnection_pairs


def collect_ne_graph_relation_data(
    ne_graph,
    prediction_pairs=None,
    *,
    collect_missing_counts=False,
    show_progress=False,
    transmission_misconnection_pairs=None,
):
    """收集 ne_graph 中真实出现过跨站连边的站点对，可选统计缺失关系补边证据。"""
    prediction_pairs = prediction_pairs or set()
    ne_graph_pairs = set()
    missing_pair_counts = {}
    skipped_prediction_link_count = 0
    raw_cross_site_link_count = 0

    for link in iter_raw_unique_cross_site_links(
        ne_graph,
        show_progress=show_progress,
        transmission_misconnection_pairs=transmission_misconnection_pairs,
    ):
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
        _record_missing_pair_evidence(
            missing_pair_counts, key, source_site, target_site, link
        )

    return {
        "ne_graph_pairs": ne_graph_pairs,
        "missing_pair_counts": missing_pair_counts,
        "stats": {
            "ne_graph_pair_count": len(ne_graph_pairs),
            "raw_cross_site_link_count": raw_cross_site_link_count,
            "skipped_prediction_link_count": skipped_prediction_link_count,
        },
    }


def _record_missing_pair_evidence(
    missing_pair_counts, key, source_site, target_site, link
):
    """记录 prediction 未覆盖站点对的连边证据（端点 domain 计数）。"""
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
            all_sites.update(pair_key)
            _apply_augmented_pair(
                rec, pair_key, adjacency, first_hop_adjacency,
                bidirectional_neighbors, all_sites, directed_only, stats,
            )

    return stats


def _apply_augmented_pair(
    rec, pair_key, adjacency, first_hop_adjacency, bidirectional_neighbors,
    all_sites, directed_only, stats,
):
    """按端点 domain 计数比较决定补边方向，计数相同记为平行边。"""
    site_a, site_b = pair_key
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


def _downstream_bfs_parents(
    adjacency,
    first_hop_adjacency,
    source_site,
    max_depth=None,
    stop_targets=None,
):
    """与 reachable_downstream_sites 同语义的 BFS，额外记录父指针用于还原路径。

    stop_targets 非空时，找齐全部目标即提前结束。返回 {site: parent}。
    """
    source_site = normalize_site_id(source_site)
    parents = {source_site: None}
    depth = {source_site: 0}
    remaining = set(stop_targets or ()) - {source_site}
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
            if next_site in parents:
                continue
            parents[next_site] = current_site
            depth[next_site] = current_depth + 1
            queue.append(next_site)
            if remaining:
                remaining.discard(next_site)
                if not remaining:
                    return parents

    return parents


def _complete_data_upstream_chains(
    site_chains, adjacency, first_hop_adjacency, data_sites,
    max_depth, show_progress, allowed_pairs=None,
):
    """后处理：非 Data 站点若有 Data 站点作为多跳上行，则把传播路径上的
    非 Data 中间站点依次补为该站点的上行（hop 取路径距离）。

    与新增上行关系冲突的原有平行/反向关系被删除；已存在的同向上行关系
    只做 hop 取小，不重复计数。路径按与 reachable_downstream_sites 相同
    的 BFS 语义重建（第一跳走显式有向边，后续可经双向边），补出的 hop
    与既有多跳 hop 口径一致。allowed_pairs 非 None 时（restrict_relation
    开启，传入 ne_graph 物理直连站点对集合），补全与 restrict 保持同一
    口径：中间站点与目标站点不直连的关系不补（计入
    skipped_by_restrict_count），保证「输出关系必有跨站连边」的不变量
    整体成立。
    """
    stats = {
        "target_site_count": 0,
        "added_upstream_relation_count": 0,
        "replaced_bidirectional_pair_count": 0,
        "replaced_reverse_relation_count": 0,
        "skipped_by_restrict_count": 0,
        "path_not_found_count": 0,
    }
    pending = _collect_pending_data_upstreams(site_chains, data_sites)

    label = "site_chains: 补全Data多跳上行链路"
    with ProgressReporter(len(pending), label, show_progress) as progress:
        for data_site in sorted(pending):
            progress.update()
            targets = pending[data_site]
            _complete_data_site_targets(
                site_chains, adjacency, first_hop_adjacency,
                data_site, targets, data_sites, allowed_pairs,
                max_depth, stats,
            )

    return stats


def _complete_data_site_targets(
    site_chains, adjacency, first_hop_adjacency, data_site, targets,
    data_sites, allowed_pairs, max_depth, stats,
):
    """为一个 Data 站点重建 BFS 路径并补全全部目标。"""
    bfs_depth = max(targets.values())
    if max_depth is not None:
        bfs_depth = min(bfs_depth, max_depth)
    parents = _downstream_bfs_parents(
        adjacency,
        first_hop_adjacency,
        data_site,
        max_depth=bfs_depth,
        stop_targets=set(targets),
    )
    for target_site in sorted(targets):
        if target_site not in parents:
            stats["path_not_found_count"] += 1
            continue
        _complete_target_upstream_path(
            site_chains, data_site, target_site, parents,
            data_sites, allowed_pairs, stats,
        )


def _collect_pending_data_upstreams(site_chains, data_sites):
    """按 Data 上行站点分组：{data_site: {target_site: hop}}，一个源只跑一次 BFS。"""
    pending = defaultdict(dict)
    for site_id, info in site_chains.items():
        if site_id in data_sites:
            continue
        for upstream_site, hop in info["upstream_site_hops"].items():
            if hop > 1 and upstream_site in data_sites:
                pending[upstream_site][site_id] = hop
    return pending


def _complete_target_upstream_path(
    site_chains, data_site, target_site, parents, data_sites, allowed_pairs, stats
):
    """沿 data_site -> target_site 的最短路径补全非 Data 中间站点上行。"""
    # 还原 data_site -> ... -> target_site 的最短路径
    path = [target_site]
    while path[-1] != data_site:
        path.append(parents[path[-1]])
    path.reverse()

    target_info = site_chains[target_site]
    target_changed = False
    path_hop_total = len(path) - 1
    for position in range(1, path_hop_total):
        mid_site = path[position]
        if mid_site in data_sites:
            continue
        # restrict 口径一致：不与目标物理直连的中间站不补
        if allowed_pairs is not None and (
            site_pair_key(mid_site, target_site) not in allowed_pairs
        ):
            stats["skipped_by_restrict_count"] += 1
            continue
        mid_info = site_chains[mid_site]
        hop_to_target = path_hop_total - position
        _replace_conflicting_relations(
            target_info, mid_info, target_site, mid_site, stats
        )
        existing_hop = target_info["upstream_site_hops"].get(mid_site)
        if existing_hop is None:
            stats["added_upstream_relation_count"] += 1
            target_changed = True
        if existing_hop is None or hop_to_target < existing_hop:
            target_info["upstream_site_hops"][mid_site] = hop_to_target
            mid_info["downstream_site_hops"][target_site] = hop_to_target
    if target_changed:
        stats["target_site_count"] += 1


def _replace_conflicting_relations(target_info, mid_info, target_site, mid_site, stats):
    """删除与新上行关系冲突的原有反向/平行关系。"""
    if mid_site in target_info["downstream_site_hops"]:
        del target_info["downstream_site_hops"][mid_site]
        mid_info["upstream_site_hops"].pop(target_site, None)
        stats["replaced_reverse_relation_count"] += 1
    if target_site in mid_info["bidirectional_sites"]:
        mid_info["bidirectional_sites"] = [
            s for s in mid_info["bidirectional_sites"] if s != target_site
        ]
        target_info["bidirectional_sites"] = [
            s for s in target_info["bidirectional_sites"] if s != mid_site
        ]
        stats["replaced_bidirectional_pair_count"] += 1


def _apply_relation_options(
    data, ne_graph, ne_graph_label, enrich_relation, restrict_relation,
    adjacency, first_hop_adjacency, bidirectional_neighbors, all_sites,
    warnings, directed_only, show_progress,
    transmission_misconnection_pairs=None,
):
    """按 ne_graph 关系选项做增强/限制准备，原地补充 warnings。

    返回 (relation_data, augmentation_stats, restriction_stats)。
    """
    has_ne_graph = ne_graph is not None
    relation_data = None
    augmentation_stats = None

    if has_ne_graph and (enrich_relation or restrict_relation):
        prediction_pairs = collect_prediction_pairs(data)
        relation_data = collect_ne_graph_relation_data(
            ne_graph,
            prediction_pairs=prediction_pairs,
            collect_missing_counts=enrich_relation,
            show_progress=show_progress,
            transmission_misconnection_pairs=transmission_misconnection_pairs,
        )
        if enrich_relation:
            augmentation_stats = apply_ne_graph_augmentation_from_counts(
                ne_graph_label,
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

    if has_ne_graph and not enrich_relation and not restrict_relation:
        warnings.append(
            "--ne-graph 已提供，但未开启 "
            "--enrich-relation/--restrict-relation；不会影响站点关系"
        )

    restriction_stats = _build_restriction_stats(
        restrict_relation, has_ne_graph, ne_graph_label, relation_data
    )
    return relation_data, augmentation_stats, restriction_stats


def _build_restriction_stats(restrict_relation, has_ne_graph, ne_graph_label, relation_data):
    """restrict_relation 开启时构造裁剪统计骨架，未开启返回 None。"""
    if not restrict_relation:
        return None
    if not has_ne_graph:
        return {
            "enabled": False,
            "reason": "missing_ne_graph",
        }
    return {
        "enabled": True,
        "mode": "filter_final_downstream_site_hops",
        "ne_graph": ne_graph_label,
        "ne_graph_pair_count": len(relation_data["ne_graph_pairs"]) if relation_data else 0,
        "downstream_relation_count_before": 0,
        "downstream_relation_count_after": 0,
        "removed_downstream_relation_count": 0,
    }


def _populate_site_chains(
    sorted_sites, adjacency, first_hop_adjacency, bidirectional_neighbors,
    relation_data, restriction_stats, max_depth, show_progress,
):
    """生成每站点链路集合；开启限制时原地累加 restriction_stats 计数。"""
    site_chains = {
        site_id: {
            "bidirectional_sites": sorted(bidirectional_neighbors.get(site_id, set())),
            "downstream_site_hops": {},
            "upstream_site_hops": {},
        }
        for site_id in sorted_sites
    }
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
                downstream_site_hops = _apply_downstream_restriction(
                    site_id, downstream_site_hops, relation_data, restriction_stats
                )
            # 直接填充最终结果，避免同时保留 downstream/upstream 中间表及其完整副本。
            site_chains[site_id]["downstream_site_hops"] = {
                downstream_site: downstream_site_hops[downstream_site]
                for downstream_site in sorted(downstream_site_hops)
            }
            # site_id 按 sorted_sites 递增处理，因此每个 upstream dict 的插入顺序稳定。
            for downstream_site, hop in downstream_site_hops.items():
                site_chains[downstream_site]["upstream_site_hops"][site_id] = hop
    return site_chains


def _apply_downstream_restriction(
    site_id, downstream_site_hops, relation_data, restriction_stats
):
    """restrict 开启时按 ne_graph 物理直连站点对过滤下游关系并计数。"""
    before_count = len(downstream_site_hops)
    ne_graph_pairs = relation_data["ne_graph_pairs"] if relation_data else set()
    filtered = {
        downstream_site: hop
        for downstream_site, hop in downstream_site_hops.items()
        if site_pair_key(site_id, downstream_site) in ne_graph_pairs
    }
    after_count = len(filtered)
    restriction_stats["downstream_relation_count_before"] += before_count
    restriction_stats["downstream_relation_count_after"] += after_count
    restriction_stats["removed_downstream_relation_count"] += before_count - after_count
    return filtered


def build_site_chains_from_data(
    data, *, ne_graph=None, prediction_label=None, ne_graph_label=None,
    enrich_relation=False, restrict_relation=False, directed_only=False,
    max_depth=None, complete_data_upstream_chains=False, show_progress=True,
):
    """从 prediction 与可选 ne_graph 在内存中生成站点链路。"""
    context = _build_site_chain_context(data, directed_only)
    adjacency = context["adjacency"]
    first_hop_adjacency = context["first_hop_adjacency"]
    all_sites = context["all_sites"]
    warnings = context["warnings"]
    relation_data, augmentation_stats, restriction_stats = _apply_relation_options(
        data, ne_graph, ne_graph_label, enrich_relation, restrict_relation,
        adjacency, first_hop_adjacency, context["bidirectional_neighbors"], all_sites,
        warnings, directed_only, show_progress,
        transmission_misconnection_pairs=_load_misconnection_pairs(data),
    )

    _collect_all_sites(all_sites, adjacency, first_hop_adjacency)
    sorted_sites = sorted(all_sites)
    site_chains = _populate_site_chains(
        sorted_sites, adjacency, first_hop_adjacency,
        context["bidirectional_neighbors"],
        relation_data, restriction_stats, max_depth, show_progress,
    )

    completion_stats = None
    if complete_data_upstream_chains:
        completion_stats = _run_data_upstream_completion(
            site_chains, ne_graph, adjacency, first_hop_adjacency,
            max_depth, show_progress, restriction_stats, relation_data,
            warnings,
        )

    input_config = _site_chain_input_config(
        prediction_label, ne_graph_label, max_depth, directed_only,
        enrich_relation, restrict_relation, complete_data_upstream_chains,
    )
    return _finalize_site_chains_result(
        site_chains, sorted_sites, warnings, context["edge_stats"],
        context["adjacency_source"], context["bidirectional_source"],
        augmentation_stats, restriction_stats,
        completion_stats,
        has_ne_graph=ne_graph is not None,
        enrich_relation=enrich_relation,
        restrict_relation=restrict_relation,
        input_config=input_config,
    )


def _site_chain_input_config(
    prediction_label, ne_graph_label, max_depth, directed_only,
    enrich_relation, restrict_relation, complete_data_upstream_chains,
):
    return {
        "prediction_json": prediction_label,
        "ne_graph": ne_graph_label,
        "max_depth": max_depth,
        "directed_only": directed_only,
        "enrich_relation": enrich_relation,
        "restrict_relation": restrict_relation,
        "complete_data_upstream_chains": complete_data_upstream_chains,
    }


def _build_site_chain_context(data, directed_only):
    """构造生成站点链路所需的基础邻接与双向关系。"""
    adjacency_result = build_adjacency(data, directed_only=directed_only)
    (
        adjacency, first_hop_adjacency, all_sites,
        adjacency_source, edge_stats, warnings,
    ) = adjacency_result
    bidirectional_result = build_bidirectional_neighbors(data)
    neighbors, bidirectional_sites, bidirectional_source = bidirectional_result
    all_sites.update(bidirectional_sites)
    return {
        "adjacency": adjacency,
        "first_hop_adjacency": first_hop_adjacency,
        "all_sites": all_sites,
        "adjacency_source": adjacency_source,
        "edge_stats": edge_stats,
        "warnings": list(warnings or []),
        "bidirectional_neighbors": neighbors,
        "bidirectional_source": bidirectional_source,
    }


def _load_misconnection_pairs(data):
    """prediction 携带的疑似误连接站点对：restrict 裁剪与 pairwise 拓扑同一口径。"""
    return {
        tuple(pair)
        for pair in data.get("transmission_misconnection_pairs") or ()
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }


def _collect_all_sites(all_sites, adjacency, first_hop_adjacency):
    """把两张邻接表两端的站点全部并入 all_sites。"""
    all_sites.update(adjacency.keys())
    all_sites.update(first_hop_adjacency.keys())
    for downstream_sites in adjacency.values():
        all_sites.update(downstream_sites)
    for downstream_sites in first_hop_adjacency.values():
        all_sites.update(downstream_sites)


def _run_data_upstream_completion(
    site_chains, ne_graph, adjacency, first_hop_adjacency,
    max_depth, show_progress, restriction_stats, relation_data, warnings,
):
    """识别 Data 站点并执行多跳上行补全后处理，返回统计或 None。"""
    if ne_graph is None:
        warnings.append("complete_data_upstream_chains 未生效：未提供 ne_graph")
        return None
    data_sites = {
        normalize_site_id(_get_site_id(ne_info))
        for ne_info in ne_graph.values()
        if isinstance(ne_info, dict)
        and normalize_domain(ne_info.get("domain", "")) == "Data"
        and _get_site_id(ne_info)
    }
    # restrict 开启时补全走同一物理直连口径（ne_graph_pairs 已在
    # _apply_relation_options 中算好）；关闭时不限制
    completion_allowed_pairs = None
    if restriction_stats and restriction_stats.get("enabled") and relation_data:
        completion_allowed_pairs = relation_data["ne_graph_pairs"]
    return _complete_data_upstream_chains(
        site_chains,
        adjacency,
        first_hop_adjacency,
        data_sites,
        max_depth,
        show_progress,
        allowed_pairs=completion_allowed_pairs,
    )


def _finalize_site_chains_result(
    site_chains, sorted_sites, warnings, edge_stats, adjacency_source,
    bidirectional_source, augmentation_stats, restriction_stats,
    completion_stats, *, has_ne_graph, enrich_relation, restrict_relation,
    input_config,
):
    """组装 meta 与最终返回结构。"""
    meta = _build_site_chains_meta(
        site_chains,
        input_config=input_config,
        adjacency_source=adjacency_source,
        bidirectional_source=bidirectional_source,
        site_count=len(sorted_sites),
        warnings=warnings,
        edge_stats=edge_stats or {},
        has_ne_graph=has_ne_graph,
        enrich_relation=enrich_relation,
        restrict_relation=restrict_relation,
        augmentation_stats=augmentation_stats,
        restriction_stats=restriction_stats,
    )
    if completion_stats is not None:
        meta["data_upstream_chain_completion"] = completion_stats
    return {
        "meta": meta,
        "sites": site_chains,
    }


def _build_site_chains_meta(
    site_chains, *, input_config, adjacency_source, bidirectional_source,
    site_count, warnings, edge_stats, has_ne_graph, enrich_relation,
    restrict_relation, augmentation_stats, restriction_stats,
):
    """组装输出 meta：来源标注、关系选项、统计与告警列表。"""
    meta = {
        "input_config": input_config,
        "adjacency_source": adjacency_source,
        "bidirectional_source": bidirectional_source,
        "first_hop_downstream_only": True,
        "site_count": site_count,
        "warning_count": len(warnings),
        "warnings": warnings,
        "edge_stats": edge_stats,
        "relation_options": {
            "ne_graph_provided": has_ne_graph,
            "enrich_relation_requested": enrich_relation,
            "restrict_relation_requested": restrict_relation,
            "enrich_relation_effective": bool(has_ne_graph and enrich_relation),
            "restrict_relation_effective": bool(has_ne_graph and restrict_relation),
        },
    }
    if augmentation_stats:
        meta["ne_graph_augmentation"] = augmentation_stats
    if restriction_stats:
        meta["ne_graph_restriction"] = restriction_stats

    meta.update(_count_relation_totals(site_chains))
    return meta


def _count_relation_totals(site_chains):
    """统计上下行/平行关系总数。"""
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
    return {
        "total_downstream_relations": total_downstream_relations,
        "total_upstream_relations": total_upstream_relations,
        "total_bidirectional_directed_relations": total_bidirectional_relations,
        "total_bidirectional_edges": total_bidirectional_relations // 2,
    }


def verify_cross_domain_constraints(site_chains, constraints):
    """校验最终 site_chains 是否满足跨类型方向约束；只统计不修复。

    对每条约束 upstream->downstream 检查上行站点视角的三类关系：
    - downstream 出现在其 bidirectional_sites  -> 平行违例
    - downstream 出现在其 upstream_site_hops   -> 反向违例
    - downstream 出现在其 downstream_site_hops -> 满足（按 hop 区分直连/多跳）
    - 三者皆无                                  -> 不可达（只计数，不注入）

    site_chains 为 {site_id: {...}}（即产物的 sites 字段）。
    """
    stats = {
        "constraint_count": len(constraints),
        "satisfied_direct_count": 0,
        "satisfied_multi_hop_count": 0,
        "unreachable_count": 0,
        "reverse_violation_count": 0,
        "bidirectional_violation_count": 0,
    }
    violations = []
    for constraint in constraints:
        _check_cross_domain_constraint(site_chains, constraint, stats, violations)
    return {
        "stats": stats,
        "violations": violations[:100],
        "violation_detail_truncated": len(violations) > 100,
    }


def _check_cross_domain_constraint(site_chains, constraint, stats, violations):
    """检查单条 upstream->downstream 约束并累加统计/违例。"""
    upstream_site = normalize_site_id(constraint.get("upstream_site"))
    downstream_site = normalize_site_id(constraint.get("downstream_site"))
    upstream_info = site_chains.get(upstream_site) or {}
    downstream_hops = upstream_info.get("downstream_site_hops") or {}
    upstream_hops = upstream_info.get("upstream_site_hops") or {}
    has_violation = False
    if downstream_site in (upstream_info.get("bidirectional_sites") or ()):
        stats["bidirectional_violation_count"] += 1
        has_violation = True
        violations.append({
            "type": "bidirectional",
            "upstream_site": upstream_site,
            "downstream_site": downstream_site,
        })
    if downstream_site in upstream_hops:
        stats["reverse_violation_count"] += 1
        has_violation = True
        violations.append({
            "type": "reverse",
            "upstream_site": upstream_site,
            "downstream_site": downstream_site,
            "hop": upstream_hops[downstream_site],
        })
    if downstream_site in downstream_hops:
        if downstream_hops[downstream_site] <= 1:
            stats["satisfied_direct_count"] += 1
        else:
            stats["satisfied_multi_hop_count"] += 1
    elif not has_violation:
        stats["unreachable_count"] += 1


def build_site_chains(
    prediction_path,
    *,
    ne_graph_path=None,
    enrich_relation=False,
    restrict_relation=False,
    directed_only=False,
    max_depth=None,
    complete_data_upstream_chains=False,
    show_progress=True,
):
    """从文件读取 prediction 与可选 ne_graph 后生成站点链路（CLI 入口的薄封装）。"""
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ne_graph = None
    if ne_graph_path:
        with open(ne_graph_path, "r", encoding="utf-8") as f:
            ne_graph = json.load(f)

    return build_site_chains_from_data(
        data,
        ne_graph=ne_graph,
        prediction_label=str(prediction_path),
        ne_graph_label=str(ne_graph_path) if ne_graph_path else None,
        enrich_relation=enrich_relation,
        restrict_relation=restrict_relation,
        directed_only=directed_only,
        max_depth=max_depth,
        complete_data_upstream_chains=complete_data_upstream_chains,
        show_progress=show_progress,
    )


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
    _add_relation_arguments(parser)
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


def _add_relation_arguments(parser):
    """注册 ne_graph 关系增强/裁剪/补全相关的 CLI 选项。"""
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
        "--complete-data-upstream-chains",
        action="store_true",
        help=(
            "后处理：非 Data 站点若有 Data 站点作为多跳上行，把传播路径上的"
            "非 Data 中间站点依次补为其上行（冲突的平行/反向关系删除）；"
            "需要提供 --ne-graph"
        ),
    )


def print_summary(result, output_path):
    meta = result["meta"]
    input_config = meta.get("input_config", {})
    print(f"输入文件: {input_config.get('prediction_json')}")
    print(f"输出文件: {output_path}")
    print(f"站点数: {meta['site_count']}")
    print(f"邻接来源: {meta['adjacency_source']}")
    print(f"双向边来源: {meta['bidirectional_source']}")
    print("第一跳约束: 必须走显式下游边")
    traversal_mode = (
        "只走显式有向边"
        if input_config.get("directed_only")
        else "按 downstream_map/双向边可双向传播"
    )
    print(f"后续遍历模式: {traversal_mode}")
    print(f"双向直接边数: {meta['total_bidirectional_edges']}")
    print(f"下游可达关系数: {meta['total_downstream_relations']}")
    print(f"上游可达关系数: {meta['total_upstream_relations']}")
    relation_options = meta.get("relation_options", {})
    if relation_options:
        print(f"ne_graph关系补充: {'开启' if relation_options.get('enrich_relation_effective') else '关闭'}")
        restriction_enabled = relation_options.get("restrict_relation_effective")
        print(f"ne_graph下游结果裁剪: {'开启' if restriction_enabled else '关闭'}")
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
        complete_data_upstream_chains=args.complete_data_upstream_chains,
        show_progress=not args.no_progress,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_summary(result, output_path)


if __name__ == "__main__":
    main()
