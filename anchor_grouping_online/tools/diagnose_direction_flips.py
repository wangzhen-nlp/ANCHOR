#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方向翻转归因诊断：对比开关组合，拆解“双向 -> 有向”翻转的成因。

以两个开关全关为基线，分别跑 约束only / 误连接only / 全开 三个变体，
对每个变体统计从双向翻成有向的站点对，并按翻转机制归类：

- constraint_forced      跨类型约束硬覆盖直接强制
- ring_force_lost        基线里是严格环强制双向，变体中环强制消失
                         （约束解除环块，或删边导致环解体）
- gap_first_directed     基线投票判双向，变体中平滑层级差过线（势场被投影
                         抻大，或结构变化改变层级场）
- voting_directed        基线投票判双向，变体中投票改判有向（证据/margin 变化）

每类同时统计其中“非桥变桥”的数量（删边导致判向阈值降档的规模）。

用法：
    python3 diagnose_direction_flips.py --resource-buffer resources/resource_buffer.jsonl
    python3 diagnose_direction_flips.py --ne-graph ne_graph.json [--sample 5]
"""

import argparse
import json

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.tools.build_resource_buffer import (
    _resource_buffer_pairwise_args,
)
from anchor_grouping_online.tools.generate_site_pair_order_pairwise import (
    build_cross_domain_constraints,
    build_pairwise_graph_metrics,
    build_pairwise_orders,
    build_pairwise_site_metrics,
    build_site_pair_inputs,
)
from anchor_grouping_online.tools.site_pair_order_common import (
    build_transmission_misconnection_pairs,
)


def load_ne_graph(resource_buffer_path=None, ne_graph_path=None):
    if resource_buffer_path:
        prefix = '{"resource_type":"ne_graph"'
        with open(resource_buffer_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(prefix):
                    return json.loads(line)["data"]
        raise SystemExit(f"未在 {resource_buffer_path} 中找到 ne_graph 资源行")
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_variant(ne_graph, constraint_on, misconnection_on):
    """跑一遍 pairwise（非 compact，保留完整判定明细），返回诊断所需产物。"""
    args = _resource_buffer_pairwise_args()
    args.cross_domain_priority_constraint = constraint_on
    args.transmission_misconnection_filter = misconnection_on

    misconnection_pairs, misconnection_stats = _build_misconnections(
        ne_graph, misconnection_on
    )
    inputs = build_site_pair_inputs(
        ne_graph,
        collect_cross_domain=constraint_on,
        transmission_misconnection_pairs=misconnection_pairs,
    )
    constraints = {}
    constraint_stats = {}
    if constraint_on:
        constraints, constraint_stats = build_cross_domain_constraints(
            inputs["cross_domain_pair_evidence"],
            inputs["pair_edge_count"],
        )
    pair_graph_metrics = build_pairwise_graph_metrics(inputs)
    site_metrics, _, smoothing_stats = build_pairwise_site_metrics(
        inputs, args, constraints=constraints
    )
    pair_outputs = build_pairwise_orders(
        inputs,
        site_metrics,
        pair_graph_metrics,
        args,
        compact_output=False,
        constraints=constraints,
    )
    return _build_variant_result(
        pair_outputs, constraints, constraint_stats,
        misconnection_stats, smoothing_stats,
    )


def _build_misconnections(ne_graph, enabled):
    if not enabled:
        return set(), {}
    return build_transmission_misconnection_pairs(ne_graph)


def _build_variant_result(
    pair_outputs, constraints, constraint_stats,
    misconnection_stats, smoothing_stats,
):
    return {
        "pair_orders": pair_outputs["pair_orders"],
        "counts": {
            "directed": pair_outputs["directed_pair_count"],
            "bidirectional": pair_outputs["bidirectional_pair_count"],
            "ring_forced": pair_outputs["strict_ring_forced_pair_count"],
            "ring_entry": pair_outputs["strict_ring_entry_direction_pair_count"],
            "ring_released_pairs": pair_outputs["strict_ring_released_pair_count"],
            "ring_released_components": pair_outputs["strict_ring_released_component_count"],
            "constraint_forced": pair_outputs["cross_domain_constraint_forced_pair_count"],
        },
        "constraint_stats": constraint_stats,
        "constraint_count": len(constraints),
        "misconnection_stats": misconnection_stats,
        "smoothing_stats": smoothing_stats,
    }


def attribute_flips(base_orders, variant_orders, sample_limit):
    """归因：基线双向、变体有向的站点对分类计数（附非桥变桥子计数与样例）。"""
    categories = {}

    def record(category, key, bridge_flip):
        bucket = categories.setdefault(
            category, {"count": 0, "bridge_flip_count": 0, "samples": []}
        )
        bucket["count"] += 1
        if bridge_flip:
            bucket["bridge_flip_count"] += 1
        if len(bucket["samples"]) < sample_limit:
            bucket["samples"].append(key)

    removed_pair_count = 0
    for key, base_result in base_orders.items():
        variant_result = variant_orders.get(key)
        if variant_result is None:
            removed_pair_count += 1
            continue
        if base_result.get("relation") != "<->" or variant_result.get("relation") != "->":
            continue

        bridge_flip = (
            not base_result.get("is_bridge") and variant_result.get("is_bridge")
        )
        if variant_result.get("cross_domain_constraint"):
            record("constraint_forced", key, bridge_flip)
        elif base_result.get("strict_ring_bidirectional") and not (
            variant_result.get("strict_ring_bidirectional")
            or variant_result.get("strict_ring_entry_direction")
        ):
            record("ring_force_lost", key, bridge_flip)
        elif variant_result.get("decision_method") == "global_level_gap":
            record("gap_first_directed", key, bridge_flip)
        else:
            record("voting_directed", key, bridge_flip)

    return categories, removed_pair_count


def main():
    args = _parse_args()
    print("加载 ne_graph...")
    ne_graph = load_ne_graph(args.resource_buffer, args.ne_graph)
    print(f"  NE 数: {len(ne_graph)}")
    results = _run_variants(ne_graph)
    _print_flip_comparisons(results, args.sample)


def _parse_args():
    parser = argparse.ArgumentParser(description="方向翻转归因诊断")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--resource-buffer",
        help="resource_buffer.jsonl 路径（读取其中 ne_graph 行）",
    )
    source.add_argument("--ne-graph", help="ne_graph.json 路径")
    parser.add_argument("--sample", type=int, default=5, help="每类翻转打印的样例站点对数，默认 5")
    return parser.parse_args()


def _run_variants(ne_graph):
    variants = [
        ("baseline(全关)", False, False),
        ("constraint(仅约束)", True, False),
        ("misconnection(仅误连接)", False, True),
        ("both(全开)", True, True),
    ]
    results = {}
    for name, constraint_on, misconnection_on in variants:
        print(f"\n运行 {name} ...")
        results[name] = run_variant(ne_graph, constraint_on, misconnection_on)
        _print_variant_result(results[name])
    return results


def _print_variant_result(result):
    counts = result["counts"]
    print(
        f"  有向 {counts['directed']} / 双向 {counts['bidirectional']} | "
        f"环强制 {counts['ring_forced']} 环入口 {counts['ring_entry']} "
        f"环解除边 {counts['ring_released_pairs']}"
        f"(块 {counts['ring_released_components']}) | "
        f"约束强制 {counts['constraint_forced']}"
    )
    if result["constraint_count"]:
        constraint_stats = result["constraint_stats"]
        smoothing_stats = result["smoothing_stats"]
        print(
            f"  约束数 {result['constraint_count']} "
            f"(tie {constraint_stats.get('tie_pair_count')}, "
            f"消圈 {constraint_stats.get('cycle_dropped_pair_count')}, "
            f"势场未满足 {smoothing_stats.get('unsatisfied_constraint_count')})"
        )
    if result["misconnection_stats"]:
        count = result["misconnection_stats"].get("misconnection_pair_count")
        print(f"  误连接剔除对数 {count}")


def _print_flip_comparisons(results, sample_limit):
    base_orders = results["baseline(全关)"]["pair_orders"]
    print(f"\n基线站点对总数: {len(base_orders)}")
    for name in ("constraint(仅约束)", "misconnection(仅误连接)", "both(全开)"):
        categories, removed = attribute_flips(
            base_orders, results[name]["pair_orders"], sample_limit
        )
        total_flips = sum(bucket["count"] for bucket in categories.values())
        print(
            f"\n===== {name} vs 基线：双向->有向 "
            f"翻转 {total_flips} 对，剔除消失 {removed} 对 ====="
        )
        _print_flip_categories(categories)


def _print_flip_categories(categories):
    for category in (
        "constraint_forced", "ring_force_lost",
        "gap_first_directed", "voting_directed",
    ):
        bucket = categories.get(category)
        if not bucket:
            continue
        print(
            f"  {category}: {bucket['count']} 对"
            f"（其中非桥变桥 {bucket['bridge_flip_count']}）"
        )
        for key in bucket["samples"]:
            print(f"    样例: {key}")


if __name__ == "__main__":
    main()
