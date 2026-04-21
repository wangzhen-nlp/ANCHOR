#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已有站点上下行预测结果应用严格环后处理。"""

import argparse
import json

from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_tools.site_pair_order_common import (
    apply_strict_ring_edge_override,
    build_downstream_map,
    build_strict_ring_context,
    compact_prediction_edges,
    find_bridges,
    format_direction_count_summary,
)


def build_adjacency(edges):
    adjacency = defaultdict(set)
    edge_keys = []
    for edge in edges:
        site_a = edge.get("site_a")
        site_b = edge.get("site_b")
        if not site_a or not site_b or site_a == site_b:
            continue
        adjacency[site_a].add(site_b)
        adjacency[site_b].add(site_a)
        edge_keys.append(tuple(sorted((site_a, site_b))))
    return adjacency, sorted(set(edge_keys))


def extract_site_scores(data):
    """从 full output 里尽量提取站点分数；compact output 通常没有。"""
    scores = {}
    for site_id, info in data.get("sites", {}).items():
        if not isinstance(info, dict):
            continue
        score = info.get("score")
        if isinstance(score, (int, float)):
            scores[site_id] = float(score)
    return scores


def infer_site_scores_from_predictions(edges):
    """从 compact edges 的上下行预测中推断站点层级分。"""
    scores = defaultdict(float)

    for edge in edges:
        site_a = edge.get("site_a")
        site_b = edge.get("site_b")
        if site_a:
            scores[site_a] += 0.0
        if site_b:
            scores[site_b] += 0.0

        upstream_site = edge.get("upstream_site")
        downstream_site = edge.get("downstream_site")
        prediction = edge.get("prediction")

        if (not upstream_site or not downstream_site) and isinstance(prediction, str):
            if prediction != "bidirectional" and "->" in prediction:
                left_site, right_site = prediction.split("->", 1)
                upstream_site = upstream_site or left_site
                downstream_site = downstream_site or right_site

        if upstream_site and downstream_site:
            scores[upstream_site] += 1.0
            scores[downstream_site] -= 1.0

    return dict(scores)


def get_site_scores(data, edges):
    full_scores = extract_site_scores(data)
    if full_scores:
        return full_scores, "site_scores"

    inferred_scores = infer_site_scores_from_predictions(edges)
    has_directional_signal = any(abs(score) > 1e-9 for score in inferred_scores.values())
    if has_directional_signal:
        return inferred_scores, "prediction_edges"
    return {}, "none"


def count_direction_edges(edges):
    bidirectional_count = sum(
        1
        for edge in edges
        if edge.get("prediction") == "bidirectional"
    )
    directed_count = len(edges) - bidirectional_count
    return directed_count, bidirectional_count


def apply_strict_ring(data, include_components=False):
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("输入 JSON 中 edges 必须是 list")

    before_directed_edge_count, before_bidirectional_edge_count = count_direction_edges(edges)
    adjacency, edge_keys = build_adjacency(edges)
    bridge_edges = find_bridges(adjacency)
    site_scores, score_source = get_site_scores(data, edges)
    strict_ring_context = build_strict_ring_context(
        edge_keys,
        bridge_edges,
        site_scores=site_scores,
    )
    pair_context = strict_ring_context["pair_context"]

    output_edges = []
    forced_edge_count = 0
    entry_direction_edge_count = 0
    changed_edge_count = 0
    for edge in edges:
        site_a = edge.get("site_a")
        site_b = edge.get("site_b")
        if not site_a or not site_b:
            output_edges.append(edge)
            continue

        pair_key = tuple(sorted((site_a, site_b)))
        ring_pair_context = pair_context.get(pair_key)
        updated_edge, changed = apply_strict_ring_edge_override(edge, ring_pair_context)
        if ring_pair_context:
            if ring_pair_context.get("force_bidirectional"):
                forced_edge_count += 1
            elif ring_pair_context.get("force_entry_direction"):
                entry_direction_edge_count += 1
            if changed:
                changed_edge_count += 1
        output_edges.append(updated_edge)

    prediction_result = {"edges": output_edges}
    compact_edges = compact_prediction_edges(prediction_result)
    after_directed_edge_count, after_bidirectional_edge_count = count_direction_edges(compact_edges)
    downstream_map = build_downstream_map({"edges": compact_edges})

    meta = dict(data.get("meta", {}))
    meta.update({
        "strict_ring_bidirectional": True,
        "strict_ring_source": "postprocess",
        "strict_ring_before_directed_edge_count": before_directed_edge_count,
        "strict_ring_before_bidirectional_edge_count": before_bidirectional_edge_count,
        "directed_edge_count": after_directed_edge_count,
        "bidirectional_edge_count": after_bidirectional_edge_count,
        "strict_ring_component_count": len(strict_ring_context["components"]),
        "strict_ring_forced_edge_count": forced_edge_count,
        "strict_ring_entry_direction_edge_count": entry_direction_edge_count,
        "strict_ring_changed_edge_count": changed_edge_count,
        "bridge_edge_count": len(bridge_edges),
        "strict_ring_score_source": score_source,
    })

    output = {
        "meta": meta,
        "edges": compact_edges,
        "downstream_map": downstream_map,
    }
    if include_components:
        output["strict_ring_components"] = strict_ring_context["components"]
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="对已有 global 站点上下行预测 JSON 应用严格环后处理：入口边定向，其余环内边双向"
    )
    parser.add_argument("input", help="未开启 strict-ring 的预测 JSON")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON")
    parser.add_argument("--include-components", action="store_true", help="输出命中的严格环组件明细")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"未找到输入文件: {args.input}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = apply_strict_ring(data, include_components=args.include_components)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"输入边数: {len(data.get('edges', []))}")
    output_edges = output.get("edges", [])
    before_directed_edge_count = output["meta"]["strict_ring_before_directed_edge_count"]
    before_bidirectional_edge_count = output["meta"]["strict_ring_before_bidirectional_edge_count"]
    before_total_edge_count = before_directed_edge_count + before_bidirectional_edge_count
    directed_edge_count = output["meta"]["directed_edge_count"]
    bidirectional_edge_count = output["meta"]["bidirectional_edge_count"]
    print(format_direction_count_summary(
        len(output_edges),
        directed_edge_count,
        bidirectional_edge_count,
        unit="边",
    ))
    print(format_direction_count_summary(
        before_total_edge_count,
        before_directed_edge_count,
        before_bidirectional_edge_count,
        unit="边",
        label="strict-ring作用前",
    ))
    print(format_direction_count_summary(
        len(output_edges),
        directed_edge_count,
        bidirectional_edge_count,
        unit="边",
        label="strict-ring作用后",
    ))
    print(f"严格环组件数: {output['meta']['strict_ring_component_count']}")
    print(f"严格环强制双向边数: {output['meta']['strict_ring_forced_edge_count']}")
    print(f"严格环入口定向边数: {output['meta']['strict_ring_entry_direction_edge_count']}")
    print(f"严格环实际改写边数: {output['meta']['strict_ring_changed_edge_count']}")
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
