#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成相邻站点对的方向先验（pairwise 证据融合版）。

核心思路：
1. 先从 ne_graph 提取站点级无向邻接，只回答“站点是否相连”；
2. 再按连通分量挑选更像汇聚层的 anchor 站点；
3. 计算每个站点到 anchor 的 hops 与层级分数；
4. 只对相邻站点对输出 A->B / B->A / <-> 三态方向关系。

这里不再强行给所有站点排全序；当局部与全局证据都不够时，保留双向边。
"""

import argparse
import json

from collections import Counter, defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from fault_grouping_official.tools.topology_resources import (
    NE_GRAPH_JSON,
    resource_display,
    resource_path,
)
from fault_grouping_official.tools.site_pair_order_common import (
    ProgressReporter,
    _get_site_id,
    apply_strict_ring_pairwise_override,
    build_strict_ring_context,
    format_direction_count_summary,
    iter_unique_cross_site_links,
    normalize_domain,
)


def compact_pairwise_prediction(pair_result):
    """把 pairwise 内部方向结果转换成统一的上下行预测格式。"""
    if pair_result.get("relation") == "<->":
        prediction = "bidirectional"
        upstream_site = None
        downstream_site = None
    else:
        upstream_site = pair_result.get("preferred_source")
        downstream_site = pair_result.get("preferred_target")
        prediction = (
            f"{upstream_site}->{downstream_site}"
            if downstream_site and upstream_site
            else None
        )

    return {
        "site_a": pair_result.get("site_a"),
        "site_b": pair_result.get("site_b"),
        "prediction": prediction,
        "upstream_site": upstream_site,
        "downstream_site": downstream_site,
    }


def build_site_pair_inputs(ne_graph, show_progress=False):
    site_domain_counts = defaultdict(Counter)
    site_neighbors = defaultdict(set)
    site_external_edge_count = Counter()
    site_data_edge_count = Counter()
    site_transmission_edge_count = Counter()
    pair_edge_count = Counter()
    pair_site_domain_counts = defaultdict(lambda: defaultdict(Counter))
    all_sites = set()

    with ProgressReporter(len(ne_graph), "pairwise: 聚合站点设备", show_progress) as progress:
        for ne_info in ne_graph.values():
            progress.update()
            if not isinstance(ne_info, dict):
                continue
            site_id = _get_site_id(ne_info)
            if not site_id:
                continue
            all_sites.add(site_id)
            site_domain_counts[site_id][normalize_domain(ne_info.get("domain", ""))] += 1

    with ProgressReporter(0, "pairwise: 扫描跨站链路", show_progress) as progress:
        for link in iter_unique_cross_site_links(ne_graph):
            progress.update()
            left_site = link["source_site"]
            right_site = link["target_site"]
            pair_key = tuple(sorted((left_site, right_site)))

            site_neighbors[left_site].add(right_site)
            site_neighbors[right_site].add(left_site)
            pair_edge_count[pair_key] += 1

            left_domain = link["source_domain"]
            right_domain = link["target_domain"]
            pair_site_domain_counts[pair_key][left_site][left_domain] += 1
            pair_site_domain_counts[pair_key][right_site][right_domain] += 1

            site_external_edge_count[left_site] += 1
            site_external_edge_count[right_site] += 1
            if left_domain == "Data":
                site_data_edge_count[left_site] += 1
            elif left_domain == "Transmission":
                site_transmission_edge_count[left_site] += 1
            if right_domain == "Data":
                site_data_edge_count[right_site] += 1
            elif right_domain == "Transmission":
                site_transmission_edge_count[right_site] += 1

    with ProgressReporter(len(all_sites), "pairwise: 初始化孤立站点邻接", show_progress) as progress:
        for site_id in all_sites:
            progress.update()
            site_neighbors[site_id]

    return {
        "all_sites": all_sites,
        "site_domain_counts": site_domain_counts,
        "site_neighbors": site_neighbors,
        "site_external_edge_count": site_external_edge_count,
        "site_data_edge_count": site_data_edge_count,
        "site_transmission_edge_count": site_transmission_edge_count,
        "pair_edge_count": pair_edge_count,
        "pair_site_domain_counts": pair_site_domain_counts,
    }


def compute_connected_components(all_sites, site_neighbors, show_progress=False):
    visited = set()
    components = []

    # 全程不排序：分量划分与成员集合是图不变量，遍历/输出顺序不影响边方向判断
    # （anchor 选择内部自带 site_id 全序，core_distance 为 BFS 最短路，均与顺序无关）。
    # 代价：component_id 编号与 full-output 顺序不再跨运行稳定，但二者都不参与决策。
    with ProgressReporter(len(all_sites), "pairwise: 计算连通分量", show_progress) as progress:
        for start_site in all_sites:
            progress.update()
            if start_site in visited:
                continue
            queue = deque([start_site])
            visited.add(start_site)
            component = []

            while queue:
                current_site = queue.popleft()
                component.append(current_site)
                for neighbor_site in site_neighbors.get(current_site, ()):
                    if neighbor_site in visited:
                        continue
                    visited.add(neighbor_site)
                    queue.append(neighbor_site)

            components.append(component)

    return components


def find_bridge_pairs(all_sites, site_neighbors, show_progress=False):
    """非递归 Tarjan 算法识别无向站点图中的桥边。"""
    discovery_time = {}
    low_link = {}
    parent = {}
    bridges = set()
    current_time = 0

    # 不排序：桥集是图不变量，DFS 根/邻居顺序不改变返回的桥边集合（is_bridge 因此稳定）
    with ProgressReporter(len(all_sites), "pairwise: 识别桥边", show_progress) as progress:
        for start_site in all_sites:
            if start_site in discovery_time:
                continue

            parent[start_site] = None
            current_time += 1
            discovery_time[start_site] = low_link[start_site] = current_time
            progress.update()
            # 邻居无需排序：桥集是图不变量，Tarjan 对任意 DFS 邻居顺序返回相同结果
            stack = [(start_site, iter(site_neighbors.get(start_site, ())))]

            while stack:
                site_id, neighbors_iter = stack[-1]

                try:
                    neighbor_site = next(neighbors_iter)
                except StopIteration:
                    stack.pop()
                    parent_site = parent.get(site_id)
                    if parent_site is not None:
                        low_link[parent_site] = min(low_link[parent_site], low_link[site_id])
                        if low_link[site_id] > discovery_time[parent_site]:
                            bridges.add(tuple(sorted((parent_site, site_id))))
                    continue

                if neighbor_site == parent.get(site_id):
                    continue

                if neighbor_site not in discovery_time:
                    parent[neighbor_site] = site_id
                    current_time += 1
                    discovery_time[neighbor_site] = low_link[neighbor_site] = current_time
                    progress.update()
                    stack.append((neighbor_site, iter(site_neighbors.get(neighbor_site, ()))))
                else:
                    low_link[site_id] = min(low_link[site_id], discovery_time[neighbor_site])

    return bridges


def build_pairwise_graph_metrics(inputs, show_progress=False):
    bridge_pairs = find_bridge_pairs(
        inputs["all_sites"],
        inputs["site_neighbors"],
        show_progress=show_progress,
    )
    pair_graph_metrics = {}

    with ProgressReporter(len(inputs["pair_edge_count"]), "pairwise: 计算站点对图指标", show_progress) as progress:
        for left_site, right_site in sorted(inputs["pair_edge_count"].keys()):
            progress.update()
            pair_key = tuple(sorted((left_site, right_site)))
            left_neighbors = set(inputs["site_neighbors"].get(left_site, ()))
            right_neighbors = set(inputs["site_neighbors"].get(right_site, ()))
            left_neighbors.discard(right_site)
            right_neighbors.discard(left_site)
            shared_neighbors = sorted(left_neighbors & right_neighbors)

            is_bridge = pair_key in bridge_pairs
            pair_graph_metrics[pair_key] = {
                "is_bridge": is_bridge,
                "has_alternative_path": not is_bridge,
                "shared_neighbor_count": len(shared_neighbors),
                "shared_neighbors": shared_neighbors,
            }

    return pair_graph_metrics


def compute_base_core_score(site_id, inputs, args):
    domain_counts = inputs["site_domain_counts"].get(site_id, Counter())
    has_data = 1.0 if domain_counts.get("Data", 0) > 0 else 0.0
    data_ne_count = float(domain_counts.get("Data", 0))
    neighbor_count = float(len(inputs["site_neighbors"].get(site_id, ())))
    external_edge_count = float(inputs["site_external_edge_count"].get(site_id, 0))
    data_edge_count = float(inputs["site_data_edge_count"].get(site_id, 0))
    transmission_edge_count = float(
        inputs["site_transmission_edge_count"].get(site_id, 0)
    )

    score = 0.0
    score += args.data_site_bonus * has_data
    score += args.data_ne_weight * data_ne_count
    score += args.neighbor_weight * neighbor_count
    score += args.external_edge_weight * external_edge_count
    score += args.data_edge_weight * data_edge_count
    score += args.transmission_edge_weight * transmission_edge_count
    return score


def select_component_anchors(component_sites, inputs, base_scores, args):
    data_sites = [
        site_id
        for site_id in component_sites
        if inputs["site_domain_counts"].get(site_id, Counter()).get("Data", 0) > 0
    ]
    ranked_candidates = sorted(
        data_sites if data_sites else component_sites,
        key=lambda site_id: (
            base_scores[site_id],
            len(inputs["site_neighbors"].get(site_id, ())),
            site_id,
        ),
        reverse=True,
    )

    if not ranked_candidates:
        return []

    anchors = []
    max_score = base_scores[ranked_candidates[0]]
    threshold = max_score * args.anchor_score_ratio

    for site_id in ranked_candidates:
        if len(anchors) >= args.max_anchor_sites_per_component:
            break
        if data_sites:
            if base_scores[site_id] + 1e-9 >= threshold:
                anchors.append(site_id)
        elif not anchors:
            anchors.append(site_id)

    if not anchors:
        anchors.append(ranked_candidates[0])

    return anchors


def compute_component_core_distance(component_sites, anchors, site_neighbors):
    distance_map = {site_id: None for site_id in component_sites}
    if not anchors:
        return distance_map

    queue = deque()
    for anchor_site in anchors:
        if anchor_site not in distance_map:
            continue
        distance_map[anchor_site] = 0
        queue.append(anchor_site)

    while queue:
        current_site = queue.popleft()
        current_distance = distance_map[current_site]
        for neighbor_site in site_neighbors.get(current_site, ()):
            if neighbor_site not in distance_map:
                continue
            if distance_map[neighbor_site] is not None:
                continue
            distance_map[neighbor_site] = current_distance + 1
            queue.append(neighbor_site)

    return distance_map


def smooth_level_scores(
    site_metrics,
    site_neighbors,
    anchor_sites,
    alpha,
    max_iter,
    tol,
    show_progress=False,
):
    """对站点 level_score 做标签传播平滑，得到全局一致的层级场。

    - anchor 站点被钉住（保持原始 level_score），充当稳定参照系；
    - 其余站点按 new = alpha * 自身原始分 + (1-alpha) * 邻居均值 迭代，直到收敛。

    注意自锚项用的是**原始 level_score**（固定值，软 Dirichlet 约束），而非当前迭代值；
    否则不动点退化为纯调和场，非 anchor 站点会全部塌缩到最近 anchor，抹掉层级落差。

    复杂度 O(max_iter * E)，纯线性迭代，无路径枚举/优化器。
    """
    base_level = {site_id: metrics["level_score"] for site_id, metrics in site_metrics.items()}
    scores = dict(base_level)
    anchors = set(anchor_sites)

    # 预先算好每个非 anchor 站点的有效邻居列表，避免每轮重复过滤/成员判断
    pending = {}
    for site_id in scores:
        if site_id in anchors:
            continue
        neighbors = [n for n in site_neighbors.get(site_id, ()) if n in scores]
        if neighbors:
            pending[site_id] = neighbors

    with ProgressReporter(max_iter, "pairwise: 平滑层级分", show_progress) as progress:
        for _ in range(max_iter):
            progress.update()
            new_scores = dict(scores)
            max_delta = 0.0
            for site_id, neighbors in pending.items():
                neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
                value = alpha * base_level[site_id] + (1.0 - alpha) * neighbor_avg
                new_scores[site_id] = value
                max_delta = max(max_delta, abs(value - scores[site_id]))
            scores = new_scores
            if max_delta < tol:
                break

    return scores


def build_pairwise_site_metrics(inputs, args, show_progress=False):
    base_scores = {
        site_id: compute_base_core_score(site_id, inputs, args)
        for site_id in inputs["all_sites"]
    }
    components = compute_connected_components(
        inputs["all_sites"],
        inputs["site_neighbors"],
        show_progress=show_progress,
    )

    site_metrics = {}
    component_summaries = []
    with ProgressReporter(len(components), "pairwise: 计算分量层级指标", show_progress) as progress:
        for component_index, component_sites in enumerate(components):
            progress.update()
            anchors = select_component_anchors(component_sites, inputs, base_scores, args)
            distance_map = compute_component_core_distance(
                component_sites,
                anchors,
                inputs["site_neighbors"],
            )
            component_summaries.append(
                {
                    "component_id": component_index,
                    "site_count": len(component_sites),
                    "anchor_sites": anchors,
                }
            )

            for site_id in component_sites:
                core_distance = distance_map.get(site_id)
                level_score = base_scores[site_id]
                if core_distance is not None:
                    level_score -= args.core_distance_penalty * core_distance
                site_metrics[site_id] = {
                    "component_id": component_index,
                    "domain_counts": dict(inputs["site_domain_counts"].get(site_id, {})),
                    "neighbor_count": len(inputs["site_neighbors"].get(site_id, ())),
                    "external_edge_count": int(
                        inputs["site_external_edge_count"].get(site_id, 0)
                    ),
                    "data_edge_count": int(inputs["site_data_edge_count"].get(site_id, 0)),
                    "transmission_edge_count": int(
                        inputs["site_transmission_edge_count"].get(site_id, 0)
                    ),
                    "base_core_score": round(base_scores[site_id], 6),
                    "core_distance": core_distance,
                    "level_score": round(level_score, 6),
                }

    # 标签传播平滑：得到全局一致的层级场，供 gap-first 定向与严格环入口排序使用
    anchor_sites = set()
    for summary in component_summaries:
        anchor_sites.update(summary["anchor_sites"])
    smoothed = smooth_level_scores(
        site_metrics,
        inputs["site_neighbors"],
        anchor_sites,
        args.smooth_alpha,
        args.smooth_iters,
        args.smooth_tol,
        show_progress=show_progress,
    )
    for site_id, metrics in site_metrics.items():
        metrics["level_score_smoothed"] = round(smoothed[site_id], 6)

    return site_metrics, component_summaries


def _add_direction_evidence(score_container, winner, loser, feature, amount, detail):
    score_container[winner]["score"] += amount
    score_container[winner]["breakdown"].append(
        {
            "feature": feature,
            "amount": round(amount, 6),
            "detail": detail,
            "towards": f"{winner}->{loser}",
        }
    )


def build_global_gap_result(
    left_site, right_site, left_level, right_level, gap, effective_threshold, graph_metrics, args
):
    """全局层级差决定方向时的结果，字段对齐常规 evaluate 输出。"""
    if gap > 0:
        preferred_source, preferred_target = left_site, right_site
    else:
        preferred_source, preferred_target = right_site, left_site
    return {
        "site_a": left_site,
        "site_b": right_site,
        "relation": "->",
        "preferred_source": preferred_source,
        "preferred_target": preferred_target,
        "score_a_to_b": round(max(gap, 0.0), 6),
        "score_b_to_a": round(max(-gap, 0.0), 6),
        "score_gap": round(gap, 6),
        "decision_margin": round(effective_threshold, 6),
        "base_global_gap_threshold": args.global_gap_threshold,
        "base_direction_margin": args.direction_margin,
        "decision_method": "global_level_gap",
        "level_smoothed_a": round(left_level, 6),
        "level_smoothed_b": round(right_level, 6),
        "is_bridge": graph_metrics["is_bridge"],
        "has_alternative_path": graph_metrics["has_alternative_path"],
        "shared_neighbor_count": graph_metrics["shared_neighbor_count"],
        "shared_neighbors": graph_metrics["shared_neighbors"],
        "uncertainty_adjustments": [],
        "score_breakdown_a_to_b": [],
        "score_breakdown_b_to_a": [],
    }


def evaluate_pair_direction(left_site, right_site, site_metrics, inputs, pair_graph_metrics, args):
    left_metrics = site_metrics[left_site]
    right_metrics = site_metrics[right_site]
    pair_key = tuple(sorted((left_site, right_site)))
    pair_domain_counts = inputs["pair_site_domain_counts"].get(pair_key, {})
    graph_metrics = pair_graph_metrics.get(
        pair_key,
        {
            "is_bridge": False,
            "has_alternative_path": True,
            "shared_neighbor_count": 0,
            "shared_neighbors": [],
        },
    )

    # 全局层级场清晰时直接按层级差定向，跳过局部 7 特征投票。
    # 短路阈值与投票的 effective_margin 一样对非桥边/共享邻居更保守：层级差不够大时
    # 不短路，交回投票，由其保守 margin 把不确定的环路/多路径边判成双向。
    left_level = left_metrics.get("level_score_smoothed", left_metrics["level_score"])
    right_level = right_metrics.get("level_score_smoothed", right_metrics["level_score"])
    gap = left_level - right_level
    effective_threshold = args.global_gap_threshold
    if not graph_metrics["is_bridge"]:
        effective_threshold += args.global_gap_nonbridge_bonus
    effective_threshold += (
        min(graph_metrics["shared_neighbor_count"], args.max_shared_neighbor_bonus_count)
        * args.global_gap_shared_neighbor_bonus
    )
    if gap != 0.0 and abs(gap) >= effective_threshold:
        return build_global_gap_result(
            left_site, right_site, left_level, right_level, gap,
            effective_threshold, graph_metrics, args,
        )

    score_container = {
        left_site: {"score": 0.0, "breakdown": []},
        right_site: {"score": 0.0, "breakdown": []},
    }

    left_distance = left_metrics.get("core_distance")
    right_distance = right_metrics.get("core_distance")
    if left_distance is not None and right_distance is not None and left_distance != right_distance:
        delta = min(abs(left_distance - right_distance), args.max_core_distance_delta)
        amount = delta * args.core_distance_weight
        if left_distance < right_distance:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "core_distance",
                amount,
                f"{left_site} 更接近汇聚 anchor ({left_distance} < {right_distance})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "core_distance",
                amount,
                f"{right_site} 更接近汇聚 anchor ({right_distance} < {left_distance})",
            )

    level_gap = left_metrics["level_score"] - right_metrics["level_score"]
    if abs(level_gap) >= args.min_level_gap:
        amount = min(abs(level_gap), args.max_level_score_delta) * args.level_score_weight
        if level_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "level_score",
                amount,
                f"{left_site} 层级分更高 ({left_metrics['level_score']:.3f} > {right_metrics['level_score']:.3f})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "level_score",
                amount,
                f"{right_site} 层级分更高 ({right_metrics['level_score']:.3f} > {left_metrics['level_score']:.3f})",
            )

    base_score_gap = left_metrics["base_core_score"] - right_metrics["base_core_score"]
    if abs(base_score_gap) >= args.min_base_score_gap:
        amount = min(abs(base_score_gap), args.max_base_score_delta) * args.base_score_weight
        if base_score_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "base_core_score",
                amount,
                f"{left_site} 汇聚候选分更高 ({left_metrics['base_core_score']:.3f} > {right_metrics['base_core_score']:.3f})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "base_core_score",
                amount,
                f"{right_site} 汇聚候选分更高 ({right_metrics['base_core_score']:.3f} > {left_metrics['base_core_score']:.3f})",
            )

    left_has_data = left_metrics["domain_counts"].get("Data", 0) > 0
    right_has_data = right_metrics["domain_counts"].get("Data", 0) > 0
    if left_has_data != right_has_data:
        if left_has_data:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "data_presence",
                args.data_presence_weight,
                f"{left_site} 存在 Data 设备而 {right_site} 不存在",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "data_presence",
                args.data_presence_weight,
                f"{right_site} 存在 Data 设备而 {left_site} 不存在",
            )

    left_pair_data = pair_domain_counts.get(left_site, Counter()).get("Data", 0)
    right_pair_data = pair_domain_counts.get(right_site, Counter()).get("Data", 0)
    if left_pair_data != right_pair_data:
        amount = min(abs(left_pair_data - right_pair_data), args.max_pair_domain_delta)
        amount *= args.pair_data_weight
        if left_pair_data > right_pair_data:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "pair_data_exposure",
                amount,
                f"{left_site} 在该站点对连接中出现更多 Data 设备 ({left_pair_data} > {right_pair_data})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "pair_data_exposure",
                amount,
                f"{right_site} 在该站点对连接中出现更多 Data 设备 ({right_pair_data} > {left_pair_data})",
            )

    neighbor_gap = left_metrics["neighbor_count"] - right_metrics["neighbor_count"]
    if abs(neighbor_gap) >= args.min_neighbor_gap:
        amount = min(abs(neighbor_gap), args.max_neighbor_delta) * args.neighbor_direction_weight
        if neighbor_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "neighbor_count",
                amount,
                f"{left_site} 的邻接站点更多 ({left_metrics['neighbor_count']} > {right_metrics['neighbor_count']})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "neighbor_count",
                amount,
                f"{right_site} 的邻接站点更多 ({right_metrics['neighbor_count']} > {left_metrics['neighbor_count']})",
            )

    left_is_leaf = left_metrics["neighbor_count"] <= args.leaf_neighbor_threshold
    right_is_leaf = right_metrics["neighbor_count"] <= args.leaf_neighbor_threshold
    if left_is_leaf != right_is_leaf:
        if right_is_leaf:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "leaf_bias",
                args.leaf_bias_weight,
                f"{right_site} 更像叶子站点 (neighbor_count <= {args.leaf_neighbor_threshold})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "leaf_bias",
                args.leaf_bias_weight,
                f"{left_site} 更像叶子站点 (neighbor_count <= {args.leaf_neighbor_threshold})",
            )

    left_score = score_container[left_site]["score"]
    right_score = score_container[right_site]["score"]
    score_gap = left_score - right_score

    effective_margin = args.direction_margin
    uncertainty_reasons = []
    if not graph_metrics["is_bridge"]:
        effective_margin += args.non_bridge_margin_bonus
        uncertainty_reasons.append(
            {
                "feature": "non_bridge",
                "amount": round(args.non_bridge_margin_bonus, 6),
                "detail": "该站点对存在替代路径，环路/多路径下方向判断需更保守",
            }
        )
    shared_neighbor_bonus = (
        min(
            graph_metrics["shared_neighbor_count"],
            args.max_shared_neighbor_bonus_count,
        )
        * args.shared_neighbor_margin_bonus
    )
    if shared_neighbor_bonus > 0:
        effective_margin += shared_neighbor_bonus
        uncertainty_reasons.append(
            {
                "feature": "shared_neighbors",
                "amount": round(shared_neighbor_bonus, 6),
                "detail": f"两端共享 {graph_metrics['shared_neighbor_count']} 个邻居站点，局部多路径更明显",
            }
        )

    if abs(score_gap) < effective_margin:
        relation = "<->"
        preferred_source = None
        preferred_target = None
    elif score_gap > 0:
        relation = "->"
        preferred_source = left_site
        preferred_target = right_site
    else:
        relation = "->"
        preferred_source = right_site
        preferred_target = left_site

    return {
        "site_a": left_site,
        "site_b": right_site,
        "relation": relation,
        "preferred_source": preferred_source,
        "preferred_target": preferred_target,
        "score_a_to_b": round(left_score, 6),
        "score_b_to_a": round(right_score, 6),
        "score_gap": round(score_gap, 6),
        "decision_margin": effective_margin,
        "base_direction_margin": args.direction_margin,
        "pair_edge_count": int(inputs["pair_edge_count"].get(pair_key, 0)),
        "is_bridge": graph_metrics["is_bridge"],
        "has_alternative_path": graph_metrics["has_alternative_path"],
        "shared_neighbor_count": graph_metrics["shared_neighbor_count"],
        "shared_neighbors": graph_metrics["shared_neighbors"],
        "uncertainty_adjustments": uncertainty_reasons,
        "score_breakdown_a_to_b": score_container[left_site]["breakdown"],
        "score_breakdown_b_to_a": score_container[right_site]["breakdown"],
    }


def build_pairwise_orders(
    inputs,
    site_metrics,
    pair_graph_metrics,
    args,
    show_progress=False,
    compact_output=False,
):
    pair_orders = {}
    compact_edges = [] if compact_output else None
    downstream_map = defaultdict(set)
    before_directed_pair_count = 0
    before_bidirectional_pair_count = 0
    directed_pair_count = 0
    bidirectional_pair_count = 0
    strict_ring_context = {"pair_context": {}, "components": []}
    strict_ring_forced_pair_count = 0
    strict_ring_entry_direction_pair_count = 0
    strict_ring_changed_pair_count = 0

    strict_ring_context = build_strict_ring_context(
        inputs["pair_edge_count"].keys(),
        [
            pair_key
            for pair_key, graph_metrics in pair_graph_metrics.items()
            if graph_metrics.get("is_bridge")
        ],
        site_scores={
            # 与 gap-first 一致地使用平滑层级
            site_id: metrics.get(
                "level_score_smoothed", metrics.get("level_score", 0.0)
            )
            for site_id, metrics in site_metrics.items()
        },
    )
    strict_ring_pair_context = strict_ring_context["pair_context"]

    with ProgressReporter(len(inputs["pair_edge_count"]), "pairwise: 判断站点对方向", show_progress) as progress:
        for left_site, right_site in sorted(inputs["pair_edge_count"].keys()):
            progress.update()
            pair_key = tuple(sorted((left_site, right_site)))
            pair_result = evaluate_pair_direction(
                left_site,
                right_site,
                site_metrics,
                inputs,
                pair_graph_metrics,
                args,
            )
            if pair_result.get("relation") == "<->":
                before_bidirectional_pair_count += 1
            else:
                before_directed_pair_count += 1
            ring_pair_context = strict_ring_pair_context.get(pair_key)
            pair_result, strict_ring_changed = apply_strict_ring_pairwise_override(
                pair_result,
                ring_pair_context,
            )
            if ring_pair_context:
                if ring_pair_context.get("force_bidirectional"):
                    strict_ring_forced_pair_count += 1
                elif ring_pair_context.get("force_entry_direction"):
                    strict_ring_entry_direction_pair_count += 1
                if strict_ring_changed:
                    strict_ring_changed_pair_count += 1
            if compact_output:
                compact_edges.append(compact_pairwise_prediction(pair_result))
            else:
                pair_orders[f"{left_site}||{right_site}"] = pair_result

            relation = pair_result["relation"]
            if relation == "<->":
                bidirectional_pair_count += 1
                downstream_map[left_site].add(right_site)
                downstream_map[right_site].add(left_site)
            else:
                directed_pair_count += 1
                downstream_map[pair_result["preferred_source"]].add(
                    pair_result["preferred_target"]
                )

    output = {
        "pair_orders": pair_orders,
        "downstream_map": {
            site_id: sorted(neighbors)
            for site_id, neighbors in sorted(downstream_map.items())
        },
        "before_directed_pair_count": before_directed_pair_count,
        "before_bidirectional_pair_count": before_bidirectional_pair_count,
        "directed_pair_count": directed_pair_count,
        "bidirectional_pair_count": bidirectional_pair_count,
        "strict_ring_components": strict_ring_context["components"],
        "strict_ring_forced_pair_count": strict_ring_forced_pair_count,
        "strict_ring_entry_direction_pair_count": strict_ring_entry_direction_pair_count,
        "strict_ring_changed_pair_count": strict_ring_changed_pair_count,
    }
    if compact_output:
        output["compact_edges"] = compact_edges
    return output


def build_pairwise_meta(args, inputs, component_summaries, pair_outputs, bridge_pair_count, pair_graph_metrics):
    """组装 pairwise 输出的 meta（main 与内存调用共用，避免字段漂移）。"""
    return {
        "algorithm": "pairwise_evidence",
        "ne_graph": args.ne_graph,
        "site_count": len(inputs["all_sites"]),
        "adjacent_pair_count": len(inputs["pair_edge_count"]),
        "component_count": len(component_summaries),
        "strict_ring_before_directed_pair_count": pair_outputs["before_directed_pair_count"],
        "strict_ring_before_bidirectional_pair_count": pair_outputs["before_bidirectional_pair_count"],
        "directed_pair_count": pair_outputs["directed_pair_count"],
        "bidirectional_pair_count": pair_outputs["bidirectional_pair_count"],
        "bridge_pair_count": bridge_pair_count,
        "non_bridge_pair_count": len(pair_graph_metrics) - bridge_pair_count,
        "smooth_alpha": args.smooth_alpha,
        "smooth_iters": args.smooth_iters,
        "smooth_tol": args.smooth_tol,
        "global_gap_threshold": args.global_gap_threshold,
        "global_gap_nonbridge_bonus": args.global_gap_nonbridge_bonus,
        "global_gap_shared_neighbor_bonus": args.global_gap_shared_neighbor_bonus,
        "strict_ring_component_count": len(pair_outputs["strict_ring_components"]),
        "strict_ring_forced_pair_count": pair_outputs["strict_ring_forced_pair_count"],
        "strict_ring_entry_direction_pair_count": pair_outputs["strict_ring_entry_direction_pair_count"],
        "strict_ring_changed_pair_count": pair_outputs["strict_ring_changed_pair_count"],
    }


def build_pairwise_prediction(ne_graph, args, show_progress=False):
    """在内存中跑完整 pairwise 流程，返回 compact 输出（meta / edges / downstream_map）。

    与 main 的非 full-output 路径产物一致，可直接作为下游 site_chains 的 prediction 输入，
    省去落盘再读盘。
    """
    inputs = build_site_pair_inputs(ne_graph, show_progress=show_progress)
    pair_graph_metrics = build_pairwise_graph_metrics(inputs, show_progress=show_progress)
    bridge_pair_count = sum(
        1 for graph_metrics in pair_graph_metrics.values() if graph_metrics["is_bridge"]
    )
    site_metrics, component_summaries = build_pairwise_site_metrics(
        inputs, args, show_progress=show_progress
    )
    pair_outputs = build_pairwise_orders(
        inputs,
        site_metrics,
        pair_graph_metrics,
        args,
        show_progress=show_progress,
        compact_output=True,
    )
    meta = build_pairwise_meta(
        args, inputs, component_summaries, pair_outputs, bridge_pair_count, pair_graph_metrics
    )
    return {
        "meta": meta,
        "edges": pair_outputs["compact_edges"],
        "downstream_map": pair_outputs["downstream_map"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="根据 ne_graph.json 生成相邻站点对的方向先验（pairwise 证据融合版）"
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=resource_path("site_pair_order_pairwise.json"),
        help=f"输出 JSON，默认: {resource_display('site_pair_order_pairwise.json')}",
    )
    parser.add_argument(
        "--direction-margin",
        type=float,
        default=2.5,
        help="相邻站点对判成单向边所需的最小分差；不足则保留双向",
    )
    parser.add_argument(
        "--core-distance-penalty",
        type=float,
        default=2.0,
        help="站点层级分里，到汇聚 anchor 每多 1 hop 的惩罚",
    )
    parser.add_argument(
        "--non-bridge-margin-bonus",
        type=float,
        default=2.0,
        help="对存在替代路径的非桥边，额外提高多少定向门槛",
    )
    parser.add_argument(
        "--shared-neighbor-margin-bonus",
        type=float,
        default=0.5,
        help="每个共享邻居为该站点对增加多少定向门槛",
    )
    parser.add_argument(
        "--max-shared-neighbor-bonus-count",
        type=int,
        default=3,
        help="共享邻居用于增加门槛时的计数上限",
    )
    parser.add_argument(
        "--anchor-score-ratio",
        type=float,
        default=0.85,
        help="连通分量内 anchor 候选保留阈值 = 最高分 * ratio",
    )
    parser.add_argument(
        "--max-anchor-sites-per-component",
        type=int,
        default=3,
        help="每个连通分量最多保留多少个 anchor 站点",
    )

    parser.add_argument("--data-site-bonus", type=float, default=6.0)
    parser.add_argument("--data-ne-weight", type=float, default=1.5)
    parser.add_argument("--neighbor-weight", type=float, default=0.8)
    parser.add_argument("--external-edge-weight", type=float, default=0.6)
    parser.add_argument("--data-edge-weight", type=float, default=1.2)
    parser.add_argument("--transmission-edge-weight", type=float, default=0.4)

    parser.add_argument("--core-distance-weight", type=float, default=2.0)
    parser.add_argument("--level-score-weight", type=float, default=0.8)
    parser.add_argument("--base-score-weight", type=float, default=0.5)
    parser.add_argument("--data-presence-weight", type=float, default=2.0)
    parser.add_argument("--pair-data-weight", type=float, default=1.0)
    parser.add_argument("--neighbor-direction-weight", type=float, default=0.4)
    parser.add_argument("--leaf-bias-weight", type=float, default=1.2)

    parser.add_argument("--min-level-gap", type=float, default=0.5)
    parser.add_argument("--min-base-score-gap", type=float, default=1.0)
    parser.add_argument("--min-neighbor-gap", type=int, default=1)
    parser.add_argument("--leaf-neighbor-threshold", type=int, default=1)

    parser.add_argument("--max-core-distance-delta", type=int, default=3)
    parser.add_argument("--max-level-score-delta", type=float, default=6.0)
    parser.add_argument("--max-base-score-delta", type=float, default=6.0)
    parser.add_argument("--max-neighbor-delta", type=int, default=4)
    parser.add_argument("--max-pair-domain-delta", type=int, default=4)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="平滑时非 anchor 站点保留自身分的权重，(1-alpha) 给邻居均值，默认 0.5",
    )
    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=100,
        help="层级平滑最大迭代轮数，默认 100（带 tol 提前收敛）",
    )
    parser.add_argument(
        "--smooth-tol",
        type=float,
        default=1e-4,
        help="层级平滑收敛阈值（最大变化小于此值即停止），默认 1e-4",
    )
    parser.add_argument(
        "--global-gap-threshold",
        type=float,
        default=1.0,
        help="层级差直接定向的基础判定阈值：|层级差| 大于等于该值才直接定向，默认 1.0",
    )
    parser.add_argument(
        "--global-gap-nonbridge-bonus",
        type=float,
        default=2.0,
        help=(
            "非桥边（环路/有替代路径）做层级差直接定向时额外要求的层级差，"
            "对应投票的 non-bridge-margin-bonus，默认 2.0"
        ),
    )
    parser.add_argument(
        "--global-gap-shared-neighbor-bonus",
        type=float,
        default=0.5,
        help=(
            "层级差直接定向时每个共享邻居额外要求的层级差（按 "
            "max-shared-neighbor-bonus-count 封顶），对应投票的 shared-neighbor-margin-bonus，默认 0.5"
        ),
    )
    parser.add_argument("--full-output", action="store_true", help="输出完整调试信息")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条显示")

    args = parser.parse_args(argv)

    if args.direction_margin < 0:
        parser.error("direction-margin 不能小于 0")
    if args.anchor_score_ratio <= 0:
        parser.error("anchor-score-ratio 必须大于 0")
    if args.max_anchor_sites_per_component <= 0:
        parser.error("max-anchor-sites-per-component 必须大于 0")
    if args.non_bridge_margin_bonus < 0:
        parser.error("non-bridge-margin-bonus 不能小于 0")
    if args.shared_neighbor_margin_bonus < 0:
        parser.error("shared-neighbor-margin-bonus 不能小于 0")
    if args.max_shared_neighbor_bonus_count < 0:
        parser.error("max-shared-neighbor-bonus-count 不能小于 0")
    if not 0.0 <= args.smooth_alpha <= 1.0:
        parser.error("smooth-alpha 必须在 [0, 1] 之间")
    if args.smooth_iters <= 0:
        parser.error("smooth-iters 必须大于 0")
    if args.smooth_tol < 0:
        parser.error("smooth-tol 不能小于 0")
    if args.global_gap_threshold < 0:
        parser.error("global-gap-threshold 不能小于 0")
    if args.global_gap_nonbridge_bonus < 0:
        parser.error("global-gap-nonbridge-bonus 不能小于 0")
    if args.global_gap_shared_neighbor_bonus < 0:
        parser.error("global-gap-shared-neighbor-bonus 不能小于 0")
    return args


def main():
    args = parse_args()

    ne_graph_path = Path(args.ne_graph)
    if not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    print(f"加载 ne_graph: {args.ne_graph}")
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)
    show_progress = not args.no_progress

    print("构建站点级输入特征...")
    inputs = build_site_pair_inputs(ne_graph, show_progress=show_progress)
    print(f"站点数: {len(inputs['all_sites'])}")
    print(f"相邻站点对数: {len(inputs['pair_edge_count'])}")
    pair_graph_metrics = build_pairwise_graph_metrics(inputs, show_progress=show_progress)
    bridge_pair_count = sum(
        1
        for graph_metrics in pair_graph_metrics.values()
        if graph_metrics["is_bridge"]
    )
    print(f"桥边站点对数: {bridge_pair_count}")
    print(f"存在替代路径的站点对数: {len(pair_graph_metrics) - bridge_pair_count}")

    print("计算站点层级与汇聚 anchor...")
    site_metrics, component_summaries = build_pairwise_site_metrics(
        inputs,
        args,
        show_progress=show_progress,
    )
    print(f"连通分量数: {len(component_summaries)}")

    print("生成相邻站点对方向判断...")
    pair_outputs = build_pairwise_orders(
        inputs,
        site_metrics,
        pair_graph_metrics,
        args,
        show_progress=show_progress,
    )
    print(f"单向站点对数: {pair_outputs['directed_pair_count']}")
    print(f"双向站点对数: {pair_outputs['bidirectional_pair_count']}")
    total_pair_count = pair_outputs["directed_pair_count"] + pair_outputs["bidirectional_pair_count"]
    print(format_direction_count_summary(
        total_pair_count,
        pair_outputs["directed_pair_count"],
        pair_outputs["bidirectional_pair_count"],
        unit="站点对",
    ))
    before_total_pair_count = (
        pair_outputs["before_directed_pair_count"]
        + pair_outputs["before_bidirectional_pair_count"]
    )
    print(format_direction_count_summary(
        before_total_pair_count,
        pair_outputs["before_directed_pair_count"],
        pair_outputs["before_bidirectional_pair_count"],
        unit="站点对",
        label="strict-ring作用前",
    ))
    print(format_direction_count_summary(
        total_pair_count,
        pair_outputs["directed_pair_count"],
        pair_outputs["bidirectional_pair_count"],
        unit="站点对",
        label="strict-ring作用后",
    ))
    print(f"严格环组件数: {len(pair_outputs['strict_ring_components'])}")
    print(f"严格环强制双向站点对数: {pair_outputs['strict_ring_forced_pair_count']}")
    print(f"严格环入口定向站点对数: {pair_outputs['strict_ring_entry_direction_pair_count']}")
    print(f"严格环实际改写站点对数: {pair_outputs['strict_ring_changed_pair_count']}")

    meta = build_pairwise_meta(
        args, inputs, component_summaries, pair_outputs, bridge_pair_count, pair_graph_metrics
    )

    compact_edges = [
        compact_pairwise_prediction(pair_result)
        for pair_result in pair_outputs["pair_orders"].values()
    ]
    output_data = {
        "meta": meta,
        "edges": compact_edges,
        "downstream_map": pair_outputs["downstream_map"],
    }

    if args.full_output:
        output_data = {
            "meta": {
                **meta,
                "direction_margin": args.direction_margin,
                "core_distance_penalty": args.core_distance_penalty,
                "non_bridge_margin_bonus": args.non_bridge_margin_bonus,
                "shared_neighbor_margin_bonus": args.shared_neighbor_margin_bonus,
            },
            "components": component_summaries,
            "strict_ring_components": pair_outputs["strict_ring_components"],
            "site_metrics": site_metrics,
            "pair_orders": pair_outputs["pair_orders"],
            "downstream_map": pair_outputs["downstream_map"],
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
