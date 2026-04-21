#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成相邻站点对的方向先验（global 路径约束优化版）。

与 generate_site_pair_order_global_no_path.py 的主要区别：
1. 使用增强版站点图，保留跨站边上的角色证据；
2. 生成 access -> core 候选路径，并将其作为软约束；
3. 在站点层级 h(v) 上做全局优化；
4. 最终融合站点分数、边角色先验和路径票来决定边方向。
"""

import argparse
import json

from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display, resource_path
from topology_tools.site_pair_order_common import (
    ProgressReporter,
    apply_strict_ring_edge_override,
    build_candidate_paths,
    build_downstream_map,
    build_site_topology_enhanced,
    build_strict_ring_context,
    collect_path_votes,
    compute_distance_scores,
    compute_site_priors_enhanced,
    counter_to_json_dict,
    edge_prior_vote,
    extract_primary_upstream_map,
    find_bridges,
    score_to_level,
    select_anchor_sites_enhanced,
    smooth_site_scores,
)


def optimize_site_heights(
    site_stats,
    site_edges,
    adjacency,
    candidate_paths,
    *,
    init_scores=None,
    lr=0.03,
    max_iter=300,
    tol=1e-5,
    path_margin=0.18,
    edge_margin=0.10,
    lambda_prior=1.20,
    lambda_smooth=0.08,
    lambda_path=1.80,
    lambda_edge=0.90,
    clip_grad=3.0,
    show_progress=False,
):
    """
    在站点层级 h(v) 上做软约束优化。
    支持有环，因为：
    - 只对候选主路径施加单调约束
    - 对所有边只施加软先验，不要求严格可满足
    """
    sites = list(site_stats.keys())
    if not sites:
        return {}, {"final_loss": 0.0, "history": []}

    if init_scores is None:
        heights = {site: site_stats[site].get("base_score", 0.0) for site in sites}
    else:
        heights = {site: float(init_scores.get(site, 0.0)) for site in sites}

    anchor_strength = {
        site: float(site_stats[site].get("anchor_strength", 0.5))
        for site in sites
    }
    base_score = {
        site: float(site_stats[site].get("base_score", 0.0))
        for site in sites
    }

    edge_weight = {}
    for key, rec in site_edges.items():
        edge_weight[key] = 1.0 + 0.15 * min(rec.get("link_count", 1), 4)

    history = []

    with ProgressReporter(max_iter, "path_optimized: 优化站点高度", show_progress) as progress:
        for step in range(max_iter):
            progress.update()
            grad = {site: 0.0 for site in sites}
            loss = 0.0

            for site in sites:
                weight = anchor_strength[site]
                diff = heights[site] - base_score[site]
                loss += lambda_prior * weight * diff * diff
                grad[site] += 2.0 * lambda_prior * weight * diff

            for (site_a, site_b), weight in edge_weight.items():
                diff = heights[site_a] - heights[site_b]
                loss += lambda_smooth * weight * diff * diff
                g = 2.0 * lambda_smooth * weight * diff
                grad[site_a] += g
                grad[site_b] -= g

            for path in candidate_paths:
                if len(path) < 2:
                    continue
                for index in range(len(path) - 1):
                    source_site = path[index]
                    target_site = path[index + 1]
                    violation = heights[source_site] - heights[target_site] + path_margin
                    if violation > 0:
                        loss += lambda_path * violation * violation
                        g = 2.0 * lambda_path * violation
                        grad[source_site] += g
                        grad[target_site] -= g

            for site_a, site_b in site_edges:
                prior_ab, prior_ba = edge_prior_vote(site_a, site_b, site_edges)
                prior_gap = prior_ab - prior_ba
                if abs(prior_gap) < 1e-9:
                    continue

                weight = abs(prior_gap)
                if prior_gap > 0:
                    violation = heights[site_a] - heights[site_b] + edge_margin
                    if violation > 0:
                        loss += lambda_edge * weight * violation * violation
                        g = 2.0 * lambda_edge * weight * violation
                        grad[site_a] += g
                        grad[site_b] -= g
                else:
                    violation = heights[site_b] - heights[site_a] + edge_margin
                    if violation > 0:
                        loss += lambda_edge * weight * violation * violation
                        g = 2.0 * lambda_edge * weight * violation
                        grad[site_b] += g
                        grad[site_a] -= g

            history.append(loss)
            progress.set_extra_text(f"iter={step + 1}, loss={loss:.6g}")
            if step > 3 and abs(history[-1] - history[-2]) < tol:
                break

            max_update = 0.0
            for site in sites:
                g = grad[site]
                if g > clip_grad:
                    g = clip_grad
                elif g < -clip_grad:
                    g = -clip_grad

                update = lr * g
                heights[site] -= update
                max_update = max(max_update, abs(update))

            mean_height = sum(heights.values()) / len(heights)
            for site in sites:
                heights[site] -= mean_height

            if max_update < tol:
                break

    stats = {
        "final_loss": history[-1] if history else 0.0,
        "history": history,
        "iterations": len(history),
    }
    return heights, stats


def normalize_vote_pair(x, y, eps=1e-9):
    total = x + y
    if total <= eps:
        return 0.0, 0.0, 0.0
    return x / total, y / total, total


def predict_site_directions_global_path_optimized(
    ne_graph,
    *,
    score_margin=0.22,
    cycle_vote_gap_margin=0.18,
    same_level_vote_gap_margin=0.15,
    lr=0.03,
    max_iter=300,
    path_margin=0.18,
    edge_margin=0.10,
    lambda_prior=1.20,
    lambda_smooth=0.08,
    lambda_path=1.80,
    lambda_edge=0.90,
    strict_ring_bidirectional=False,
    show_progress=False,
):
    """
    路径约束优化版：
    1) 先构站点图
    2) 生成接入->核心候选路径
    3) 优化站点层级 h
    4) 用优化后的 h + 路径票 + 边先验输出边方向
    """
    site_stats, site_edges, adjacency = build_site_topology_enhanced(
        ne_graph,
        show_progress=show_progress,
    )
    if not site_stats:
        return {
            "sites": {},
            "edges": [],
            "candidate_paths": [],
            "optimization": {},
            "strict_ring_components": [],
            "strict_ring_stats": {
                "forced_edge_count": 0,
                "changed_edge_count": 0,
            },
        }

    compute_site_priors_enhanced(site_stats, show_progress=show_progress)
    core_anchors, access_anchors = select_anchor_sites_enhanced(site_stats)
    if show_progress:
        print("path_optimized: 计算 anchor 距离分...")
    distance_scores = compute_distance_scores(
        site_stats, adjacency, core_anchors, access_anchors
    )

    for site_id, rec in site_stats.items():
        base_score = 0.65 * rec["raw_prior"] + 1.00 * distance_scores.get(site_id, 0.0)
        rec["distance_score"] = distance_scores.get(site_id, 0.0)
        rec["base_score"] = base_score

    init_scores = {site_id: rec["base_score"] for site_id, rec in site_stats.items()}
    candidate_paths = build_candidate_paths(
        site_stats,
        site_edges,
        adjacency,
        access_anchors,
        core_anchors,
        init_scores,
        show_progress=show_progress,
    )

    optimized_scores, opt_stats = optimize_site_heights(
        site_stats,
        site_edges,
        adjacency,
        candidate_paths,
        init_scores=init_scores,
        lr=lr,
        max_iter=max_iter,
        path_margin=path_margin,
        edge_margin=edge_margin,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
        lambda_path=lambda_path,
        lambda_edge=lambda_edge,
        show_progress=show_progress,
    )

    path_votes = collect_path_votes(candidate_paths, optimized_scores, show_progress=show_progress)
    if show_progress:
        print("path_optimized: 识别桥边...")
    bridges = find_bridges(adjacency)
    strict_ring_context = {"pair_context": {}, "components": []}
    strict_ring_forced_edge_count = 0
    strict_ring_changed_edge_count = 0
    if strict_ring_bidirectional:
        strict_ring_context = build_strict_ring_context(site_edges.keys(), bridges)
        strict_ring_pair_context = strict_ring_context["pair_context"]
    else:
        strict_ring_pair_context = {}

    site_output = {}
    with ProgressReporter(len(site_stats), "path_optimized: 生成站点输出", show_progress) as progress:
        for site_id, rec in site_stats.items():
            progress.update()
            score = optimized_scores[site_id]
            site_output[site_id] = {
                "score": round(score, 6),
                "level": score_to_level(score),
                "predominant_role": rec["predominant_role"],
                "role_counts": dict(rec["role_counts"]),
                "degree": rec["degree"],
                "raw_prior": round(rec["raw_prior"], 6),
                "distance_score": round(rec["distance_score"], 6),
                "anchor_strength": round(rec["anchor_strength"], 6),
                "base_score": round(rec["base_score"], 6),
                "is_core_anchor": site_id in core_anchors,
                "is_access_anchor": site_id in access_anchors,
                "neighbors": sorted(rec["neighbors"]),
            }

    edges_output = []
    with ProgressReporter(len(site_edges), "path_optimized: 预测边方向", show_progress) as progress:
        for key in sorted(site_edges.keys()):
            progress.update()
            site_a, site_b = key
            edge = site_edges[key]

            score_a = optimized_scores[site_a]
            score_b = optimized_scores[site_b]
            diff = score_b - score_a

            score_ab = max(0.0, diff)
            score_ba = max(0.0, -diff)

            prior_ab, prior_ba = edge_prior_vote(site_a, site_b, site_edges)

            path_vote = path_votes.get(key, {"ab": 0.0, "ba": 0.0, "support_paths": 0})
            path_ab = path_vote["ab"]
            path_ba = path_vote["ba"]

            vote_ab = 0.45 * score_ab + 0.25 * prior_ab + 0.30 * path_ab
            vote_ba = 0.45 * score_ba + 0.25 * prior_ba + 0.30 * path_ba

            normalized_ab, normalized_ba, vote_total = normalize_vote_pair(vote_ab, vote_ba)
            vote_gap = abs(normalized_ab - normalized_ba)

            level_a = site_output[site_a]["level"]
            level_b = site_output[site_b]["level"]
            same_level = level_a == level_b
            same_role = (
                site_output[site_a]["predominant_role"] ==
                site_output[site_b]["predominant_role"]
                and site_output[site_a]["predominant_role"] != "unknown"
            )
            is_bridge = key in bridges
            in_cycle = not is_bridge

            reasons = []
            bidirectional = False

            if vote_total == 0:
                bidirectional = True
                reasons.append("no_directional_evidence")
            elif abs(diff) < score_margin and in_cycle and vote_gap < cycle_vote_gap_margin:
                bidirectional = True
                reasons.append("cycle_edge_low_score_gap_low_vote_gap")
            elif same_level and vote_gap < same_level_vote_gap_margin:
                bidirectional = True
                reasons.append(f"same_level={level_a}")
            elif same_role and in_cycle and vote_gap < cycle_vote_gap_margin:
                bidirectional = True
                reasons.append("same_role_cycle_edge")

            if bidirectional:
                prediction = "bidirectional"
                upstream_site = None
                downstream_site = None
                confidence = max(0.05, min(0.55, vote_gap))
            else:
                if vote_ab >= vote_ba:
                    prediction = f"{site_a}->{site_b}"
                    downstream_site = site_a
                    upstream_site = site_b
                else:
                    prediction = f"{site_b}->{site_a}"
                    downstream_site = site_b
                    upstream_site = site_a

                confidence = min(0.99, max(vote_gap, min(1.0, abs(diff) / 1.5)))
                reasons.append(f"score_diff={diff:.3f}")
                reasons.append(f"vote_ab={vote_ab:.3f}")
                reasons.append(f"vote_ba={vote_ba:.3f}")

            edge_result = {
                "site_a": site_a,
                "site_b": site_b,
                "prediction": prediction,
                "upstream_site": upstream_site,
                "downstream_site": downstream_site,
                "confidence": round(confidence, 6),
                "score_a": round(score_a, 6),
                "score_b": round(score_b, 6),
                "level_a": level_a,
                "level_b": level_b,
                "same_level": same_level,
                "same_role": same_role,
                "is_bridge": is_bridge,
                "in_cycle": in_cycle,
                "link_types": sorted(edge["link_types"]),
                "link_count": edge["link_count"],
                "role_pair_counter": counter_to_json_dict(edge["role_pair_counter"]),
                "path_vote_ab": round(path_ab, 6),
                "path_vote_ba": round(path_ba, 6),
                "prior_vote_ab": round(prior_ab, 6),
                "prior_vote_ba": round(prior_ba, 6),
                "normalized_vote_ab": round(normalized_ab, 6),
                "normalized_vote_ba": round(normalized_ba, 6),
                "reasons": reasons,
            }
            ring_pair_context = strict_ring_pair_context.get(key)
            edge_result, strict_ring_changed = apply_strict_ring_edge_override(
                edge_result,
                ring_pair_context,
            )
            if ring_pair_context and ring_pair_context.get("force_bidirectional"):
                strict_ring_forced_edge_count += 1
                if strict_ring_changed:
                    strict_ring_changed_edge_count += 1
            edges_output.append(edge_result)

    return {
        "sites": site_output,
        "edges": edges_output,
        "candidate_paths": candidate_paths,
        "optimization": {
            "core_anchors": core_anchors,
            "access_anchors": access_anchors,
            **opt_stats,
        },
        "strict_ring_components": strict_ring_context["components"],
        "strict_ring_stats": {
            "forced_edge_count": strict_ring_forced_edge_count,
            "changed_edge_count": strict_ring_changed_edge_count,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 ne_graph.json 生成相邻站点对的方向先验（global 路径约束优化版）"
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=resource_path("site_pair_order_global_path_optimized.json"),
        help=f"输出 JSON，默认: {resource_display('site_pair_order_global_path_optimized.json')}",
    )
    parser.add_argument("--score-margin", type=float, default=0.22, help="低分差时转双向的阈值")
    parser.add_argument(
        "--cycle-vote-gap-margin",
        type=float,
        default=0.18,
        help="环内低票差时转双向的阈值",
    )
    parser.add_argument(
        "--same-level-vote-gap-margin",
        type=float,
        default=0.15,
        help="同层边低票差时转双向的阈值",
    )
    parser.add_argument("--lr", type=float, default=0.03, help="优化学习率")
    parser.add_argument("--max-iter", type=int, default=300, help="优化最大轮数")
    parser.add_argument("--path-margin", type=float, default=0.18, help="路径单调约束间隔")
    parser.add_argument("--edge-margin", type=float, default=0.10, help="边先验约束间隔")
    parser.add_argument("--lambda-prior", type=float, default=1.20, help="站点先验项权重")
    parser.add_argument("--lambda-smooth", type=float, default=0.08, help="图平滑项权重")
    parser.add_argument("--lambda-path", type=float, default=1.80, help="路径项权重")
    parser.add_argument("--lambda-edge", type=float, default=0.90, help="边先验项权重")
    parser.add_argument(
        "--strict-ring-bidirectional",
        action="store_true",
        help="严格环模式：环块内部除唯一起始点相关连接外，其余边强制输出双向",
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条显示")
    args = parser.parse_args()

    if args.max_iter <= 0:
        parser.error("max-iter 必须大于 0")
    return args


def main():
    args = parse_args()

    ne_graph_path = Path(args.ne_graph)
    if not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    print(f"加载 ne_graph: {args.ne_graph}")
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)

    prediction_result = predict_site_directions_global_path_optimized(
        ne_graph,
        score_margin=args.score_margin,
        cycle_vote_gap_margin=args.cycle_vote_gap_margin,
        same_level_vote_gap_margin=args.same_level_vote_gap_margin,
        lr=args.lr,
        max_iter=args.max_iter,
        path_margin=args.path_margin,
        edge_margin=args.edge_margin,
        lambda_prior=args.lambda_prior,
        lambda_smooth=args.lambda_smooth,
        lambda_path=args.lambda_path,
        lambda_edge=args.lambda_edge,
        strict_ring_bidirectional=args.strict_ring_bidirectional,
        show_progress=not args.no_progress,
    )

    primary_upstream_map = extract_primary_upstream_map(prediction_result)
    downstream_map = build_downstream_map(prediction_result)

    bidirectional_edge_count = sum(
        1
        for edge in prediction_result["edges"]
        if edge.get("prediction") == "bidirectional"
    )
    directed_edge_count = len(prediction_result["edges"]) - bidirectional_edge_count
    bridge_edge_count = sum(
        1
        for edge in prediction_result["edges"]
        if edge.get("is_bridge")
    )

    print(f"站点数: {len(prediction_result['sites'])}")
    print(f"边数: {len(prediction_result['edges'])}")
    print(f"单向边数: {directed_edge_count}")
    print(f"双向边数: {bidirectional_edge_count}")
    print(f"桥边数: {bridge_edge_count}")
    print(f"候选路径数: {len(prediction_result['candidate_paths'])}")
    print(f"优化迭代轮数: {prediction_result['optimization'].get('iterations', 0)}")
    if args.strict_ring_bidirectional:
        strict_ring_stats = prediction_result["strict_ring_stats"]
        print(f"严格环组件数: {len(prediction_result['strict_ring_components'])}")
        print(f"严格环强制双向边数: {strict_ring_stats['forced_edge_count']}")
        print(f"严格环实际改写边数: {strict_ring_stats['changed_edge_count']}")

    output_data = {
        "meta": {
            "algorithm": "global_path_optimized",
            "ne_graph": args.ne_graph,
            "site_count": len(prediction_result["sites"]),
            "edge_count": len(prediction_result["edges"]),
            "directed_edge_count": directed_edge_count,
            "bidirectional_edge_count": bidirectional_edge_count,
            "bridge_edge_count": bridge_edge_count,
            "candidate_path_count": len(prediction_result["candidate_paths"]),
            "score_margin": args.score_margin,
            "cycle_vote_gap_margin": args.cycle_vote_gap_margin,
            "same_level_vote_gap_margin": args.same_level_vote_gap_margin,
            "lr": args.lr,
            "max_iter": args.max_iter,
            "path_margin": args.path_margin,
            "edge_margin": args.edge_margin,
            "lambda_prior": args.lambda_prior,
            "lambda_smooth": args.lambda_smooth,
            "lambda_path": args.lambda_path,
            "lambda_edge": args.lambda_edge,
            "strict_ring_bidirectional": args.strict_ring_bidirectional,
            "strict_ring_component_count": len(prediction_result["strict_ring_components"]),
            "strict_ring_forced_edge_count": prediction_result["strict_ring_stats"]["forced_edge_count"],
            "strict_ring_changed_edge_count": prediction_result["strict_ring_stats"]["changed_edge_count"],
        },
        "sites": prediction_result["sites"],
        "edges": prediction_result["edges"],
        "candidate_paths": prediction_result["candidate_paths"],
        "optimization": prediction_result["optimization"],
        "strict_ring_components": prediction_result["strict_ring_components"],
        "primary_upstream_map": primary_upstream_map,
        "downstream_map": downstream_map,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
