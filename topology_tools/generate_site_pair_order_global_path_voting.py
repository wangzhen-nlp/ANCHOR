#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成相邻站点对的方向先验（global 路径投票融合版）。

核心思路：
1. 使用增强版站点图，保留跨站边上的角色证据；
2. 用站点先验 + distance score 得到平滑后的全局站点分数；
3. 生成 access -> core 候选路径，并为边收集路径票；
4. 融合分数差、边角色先验和路径票，输出边方向。
"""

import argparse
import json

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
    compact_prediction_edges,
    compute_distance_scores,
    compute_site_priors_enhanced,
    counter_to_json_dict,
    edge_prior_vote,
    extract_primary_upstream_map,
    find_bridges,
    format_direction_count_summary,
    score_to_level,
    select_anchor_sites_enhanced,
    smooth_site_scores,
)


def predict_site_directions_global_path_voting(
    ne_graph,
    *,
    score_margin=0.28,
    path_vote_weight=0.55,
    edge_prior_weight=0.30,
    score_diff_weight=0.15,
    cycle_bidirectional_margin=0.90,
    same_level_bidirectional_margin=0.70,
    strict_ring_bidirectional=False,
    show_progress=False,
):
    """
    路径投票融合版站点上下行预测。

    返回:
    {
        "sites": {...},
        "edges": [...],
        "candidate_paths": [...],
    }

    约定:
      prediction = "A->B"  表示 A 向 B 上行
      prediction = "bidirectional" 表示双向/不确定
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
            "strict_ring_components": [],
            "strict_ring_stats": {
                "before_directed_edge_count": 0,
                "before_bidirectional_edge_count": 0,
                "forced_edge_count": 0,
                "entry_direction_edge_count": 0,
                "changed_edge_count": 0,
            },
        }

    compute_site_priors_enhanced(site_stats, show_progress=show_progress)
    core_anchors, access_anchors = select_anchor_sites_enhanced(site_stats)

    if show_progress:
        print("path_voting: 计算 anchor 距离分...")
    distance_scores = compute_distance_scores(
        site_stats, adjacency, core_anchors, access_anchors
    )

    for site_id, rec in site_stats.items():
        base_score = 0.65 * rec["raw_prior"] + 1.00 * distance_scores.get(site_id, 0.0)
        rec["distance_score"] = distance_scores.get(site_id, 0.0)
        rec["base_score"] = base_score

    final_scores = smooth_site_scores(site_stats, adjacency, show_progress=show_progress)

    candidate_paths = build_candidate_paths(
        site_stats,
        site_edges,
        adjacency,
        access_anchors,
        core_anchors,
        final_scores,
        show_progress=show_progress,
    )

    path_votes = collect_path_votes(candidate_paths, final_scores, show_progress=show_progress)
    if show_progress:
        print("path_voting: 识别桥边...")
    bridges = find_bridges(adjacency)
    strict_ring_context = {"pair_context": {}, "components": []}
    strict_ring_before_directed_edge_count = 0
    strict_ring_before_bidirectional_edge_count = 0
    strict_ring_forced_edge_count = 0
    strict_ring_entry_direction_edge_count = 0
    strict_ring_changed_edge_count = 0
    if strict_ring_bidirectional:
        strict_ring_context = build_strict_ring_context(
            site_edges.keys(),
            bridges,
            site_scores=final_scores,
        )
        strict_ring_pair_context = strict_ring_context["pair_context"]
    else:
        strict_ring_pair_context = {}

    site_output = {}
    with ProgressReporter(len(site_stats), "path_voting: 生成站点输出", show_progress) as progress:
        for site_id, rec in site_stats.items():
            progress.update()
            score = final_scores[site_id]
            site_output[site_id] = {
                "score": round(score, 6),
                "level": score_to_level(score),
                "predominant_role": rec["predominant_role"],
                "role_counts": dict(rec["role_counts"]),
                "degree": rec["degree"],
                "raw_prior": round(rec["raw_prior"], 6),
                "distance_score": round(rec["distance_score"], 6),
                "anchor_strength": round(rec["anchor_strength"], 6),
                "is_core_anchor": site_id in core_anchors,
                "is_access_anchor": site_id in access_anchors,
                "neighbors": sorted(rec["neighbors"]),
            }

    edges_output = []
    with ProgressReporter(len(site_edges), "path_voting: 预测边方向", show_progress) as progress:
        for key in sorted(site_edges.keys()):
            progress.update()
            site_a, site_b = key
            edge = site_edges[key]

            score_a = final_scores[site_a]
            score_b = final_scores[site_b]
            diff = score_b - score_a

            vote_ab_from_score = max(0.0, diff)
            vote_ba_from_score = max(0.0, -diff)

            prior_ab, prior_ba = edge_prior_vote(site_a, site_b, site_edges)

            path_vote = path_votes.get(key, {"ab": 0.0, "ba": 0.0, "support_paths": 0})
            path_ab = path_vote["ab"]
            path_ba = path_vote["ba"]

            vote_ab = (
                score_diff_weight * vote_ab_from_score +
                edge_prior_weight * prior_ab +
                path_vote_weight * path_ab
            )
            vote_ba = (
                score_diff_weight * vote_ba_from_score +
                edge_prior_weight * prior_ba +
                path_vote_weight * path_ba
            )

            total_vote = vote_ab + vote_ba
            vote_gap = abs(vote_ab - vote_ba)

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

            if total_vote == 0:
                bidirectional = True
                reasons.append("no_directional_evidence")
            elif abs(diff) < score_margin and vote_gap < cycle_bidirectional_margin:
                if in_cycle:
                    bidirectional = True
                    reasons.append("low_score_gap_and_low_vote_gap_in_cycle")
            elif same_level and vote_gap < same_level_bidirectional_margin:
                bidirectional = True
                reasons.append(f"same_level={level_a}")
            elif same_role and in_cycle and vote_gap < cycle_bidirectional_margin:
                bidirectional = True
                reasons.append("same_role_cycle_edge_low_vote_gap")

            if bidirectional:
                prediction = "bidirectional"
                upstream_site = None
                downstream_site = None
                confidence = max(0.05, min(0.55, vote_gap / max(1.0, total_vote + 1e-6)))
            else:
                if vote_ab >= vote_ba:
                    prediction = f"{site_a}->{site_b}"
                    downstream_site = site_a
                    upstream_site = site_b
                else:
                    prediction = f"{site_b}->{site_a}"
                    downstream_site = site_b
                    upstream_site = site_a

                confidence = min(0.99, vote_gap / max(0.5, total_vote))
                reasons.append(f"vote_ab={vote_ab:.3f}")
                reasons.append(f"vote_ba={vote_ba:.3f}")
                reasons.append(f"score_diff={diff:.3f}")

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
                "reasons": reasons,
            }
            if edge_result.get("prediction") == "bidirectional":
                strict_ring_before_bidirectional_edge_count += 1
            else:
                strict_ring_before_directed_edge_count += 1
            ring_pair_context = strict_ring_pair_context.get(key)
            edge_result, strict_ring_changed = apply_strict_ring_edge_override(
                edge_result,
                ring_pair_context,
            )
            if ring_pair_context:
                if ring_pair_context.get("force_bidirectional"):
                    strict_ring_forced_edge_count += 1
                elif ring_pair_context.get("force_entry_direction"):
                    strict_ring_entry_direction_edge_count += 1
                if strict_ring_changed:
                    strict_ring_changed_edge_count += 1
            edges_output.append(edge_result)

    return {
        "sites": site_output,
        "edges": edges_output,
        "candidate_paths": candidate_paths,
        "strict_ring_components": strict_ring_context["components"],
        "strict_ring_stats": {
            "before_directed_edge_count": strict_ring_before_directed_edge_count,
            "before_bidirectional_edge_count": strict_ring_before_bidirectional_edge_count,
            "forced_edge_count": strict_ring_forced_edge_count,
            "entry_direction_edge_count": strict_ring_entry_direction_edge_count,
            "changed_edge_count": strict_ring_changed_edge_count,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 ne_graph.json 生成相邻站点对的方向先验（global 路径投票融合版）"
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=resource_path("site_pair_order_global_path_voting.json"),
        help=f"输出 JSON，默认: {resource_display('site_pair_order_global_path_voting.json')}",
    )
    parser.add_argument("--score-margin", type=float, default=0.28, help="低分差时转双向的阈值")
    parser.add_argument("--path-vote-weight", type=float, default=0.55, help="路径票权重")
    parser.add_argument("--edge-prior-weight", type=float, default=0.30, help="边角色先验权重")
    parser.add_argument("--score-diff-weight", type=float, default=0.15, help="站点分数差权重")
    parser.add_argument(
        "--cycle-bidirectional-margin",
        type=float,
        default=0.90,
        help="环内低票差时转双向的阈值",
    )
    parser.add_argument(
        "--same-level-bidirectional-margin",
        type=float,
        default=0.70,
        help="同层边低票差时转双向的阈值",
    )
    parser.add_argument(
        "--strict-ring-bidirectional",
        action="store_true",
        help="严格环模式：入口相关边强制为入口指向环内站点，其余环内边强制输出双向",
    )
    parser.add_argument("--full-output", action="store_true", help="输出完整调试信息")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条显示")
    return parser.parse_args()


def main():
    args = parse_args()

    ne_graph_path = Path(args.ne_graph)
    if not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    print(f"加载 ne_graph: {args.ne_graph}")
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)

    prediction_result = predict_site_directions_global_path_voting(
        ne_graph,
        score_margin=args.score_margin,
        path_vote_weight=args.path_vote_weight,
        edge_prior_weight=args.edge_prior_weight,
        score_diff_weight=args.score_diff_weight,
        cycle_bidirectional_margin=args.cycle_bidirectional_margin,
        same_level_bidirectional_margin=args.same_level_bidirectional_margin,
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
    print(format_direction_count_summary(
        len(prediction_result["edges"]),
        directed_edge_count,
        bidirectional_edge_count,
        unit="边",
    ))
    print(f"桥边数: {bridge_edge_count}")
    print(f"候选路径数: {len(prediction_result['candidate_paths'])}")
    if args.strict_ring_bidirectional:
        strict_ring_stats = prediction_result["strict_ring_stats"]
        before_directed_edge_count = strict_ring_stats["before_directed_edge_count"]
        before_bidirectional_edge_count = strict_ring_stats["before_bidirectional_edge_count"]
        before_total_edge_count = before_directed_edge_count + before_bidirectional_edge_count
        print(format_direction_count_summary(
            before_total_edge_count,
            before_directed_edge_count,
            before_bidirectional_edge_count,
            unit="边",
            label="strict-ring作用前",
        ))
        print(format_direction_count_summary(
            len(prediction_result["edges"]),
            directed_edge_count,
            bidirectional_edge_count,
            unit="边",
            label="strict-ring作用后",
        ))
        print(f"严格环组件数: {len(prediction_result['strict_ring_components'])}")
        print(f"严格环强制双向边数: {strict_ring_stats['forced_edge_count']}")
        print(f"严格环入口定向边数: {strict_ring_stats['entry_direction_edge_count']}")
        print(f"严格环实际改写边数: {strict_ring_stats['changed_edge_count']}")

    meta = {
        "algorithm": "global_path_voting",
        "ne_graph": args.ne_graph,
        "site_count": len(prediction_result["sites"]),
        "edge_count": len(prediction_result["edges"]),
        "directed_edge_count": directed_edge_count,
        "bidirectional_edge_count": bidirectional_edge_count,
        "strict_ring_before_directed_edge_count": prediction_result["strict_ring_stats"]["before_directed_edge_count"],
        "strict_ring_before_bidirectional_edge_count": prediction_result["strict_ring_stats"]["before_bidirectional_edge_count"],
        "bridge_edge_count": bridge_edge_count,
        "candidate_path_count": len(prediction_result["candidate_paths"]),
        "strict_ring_bidirectional": args.strict_ring_bidirectional,
        "strict_ring_component_count": len(prediction_result["strict_ring_components"]),
        "strict_ring_forced_edge_count": prediction_result["strict_ring_stats"]["forced_edge_count"],
        "strict_ring_entry_direction_edge_count": prediction_result["strict_ring_stats"]["entry_direction_edge_count"],
        "strict_ring_changed_edge_count": prediction_result["strict_ring_stats"]["changed_edge_count"],
    }
    output_data = {
        "meta": meta,
        "edges": compact_prediction_edges(prediction_result),
        "downstream_map": downstream_map,
    }

    if args.full_output:
        output_data = {
            "meta": {
                **meta,
                "score_margin": args.score_margin,
                "path_vote_weight": args.path_vote_weight,
                "edge_prior_weight": args.edge_prior_weight,
                "score_diff_weight": args.score_diff_weight,
                "cycle_bidirectional_margin": args.cycle_bidirectional_margin,
                "same_level_bidirectional_margin": args.same_level_bidirectional_margin,
            },
            "sites": prediction_result["sites"],
            "edges": prediction_result["edges"],
            "candidate_paths": prediction_result["candidate_paths"],
            "strict_ring_components": prediction_result["strict_ring_components"],
            "primary_upstream_map": primary_upstream_map,
            "downstream_map": downstream_map,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
