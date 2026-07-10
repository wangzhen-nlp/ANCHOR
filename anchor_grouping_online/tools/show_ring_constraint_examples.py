#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出“环上有约束”的最小示例，供人工核查环块解除是否符合预期。

按当前生产口径跑一遍 pairwise（含跨类型约束与误连接过滤），找出所有
“同一条约束的两个端点都落在同一环块内”的环块，按站点数升序取前 N 个，
逐块打印：

- 块概要：站点数 / 块内边数 / 出入口站点 / 是否被解除
- 命中约束：上行 -> 下行、证据链路数、有无直连拓扑边
- 每个站点：设备 domain 构成、平滑层级分、邻居站点数
- 块内每条边：两端连边的 domain 构成、当前方向判定及判定来源

用法：
    python3 show_ring_constraint_examples.py --resource-buffer resources/resource_buffer.jsonl
    python3 show_ring_constraint_examples.py --ne-graph ne_graph.json --limit 8 --max-sites 12
"""

import argparse

from anchor_grouping_online.tools.build_resource_buffer import (
    _resource_buffer_pairwise_args,
)
from anchor_grouping_online.tools.diagnose_direction_flips import load_ne_graph
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

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()


def _decision_label(pair_result):
    """把一条 pair_result 的最终判定来源翻译成短标签。"""
    if pair_result.get("cross_domain_constraint"):
        return "约束强制"
    if pair_result.get("strict_ring_bidirectional"):
        return "环强制双向"
    if pair_result.get("strict_ring_entry_direction"):
        return "环入口强制"
    if pair_result.get("decision_method") == "global_level_gap":
        return "平滑层级差"
    return "特征投票"


def _relation_text(pair_result):
    if pair_result.get("relation") == "<->":
        return "<->"
    return f"{pair_result.get('preferred_source')} -> {pair_result.get('preferred_target')}"


def _domain_text(domain_counter):
    if not domain_counter:
        return "(无)"
    return ",".join(
        f"{domain}={count}"
        for domain, count in sorted(domain_counter.items(), key=lambda kv: -kv[1])
    )


def main():
    args = _parse_args()
    context = _build_example_context(args)
    examples = _collect_examples(
        context["components"], context["constraints"], args.max_sites
    )
    _print_example_summary(context, examples, args.limit)
    for component, inside_constraints in examples[: args.limit]:
        _print_component_example(component, inside_constraints, context)


def _parse_args():
    parser = argparse.ArgumentParser(description="导出环上有约束的最小环块示例")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--resource-buffer",
        help="resource_buffer.jsonl 路径（读取其中 ne_graph 行）",
    )
    source.add_argument("--ne-graph", help="ne_graph.json 路径")
    parser.add_argument("--limit", type=int, default=5, help="输出的环块示例数，默认 5")
    parser.add_argument("--max-sites", type=int, help="只看站点数不超过该值的环块")
    return parser.parse_args()


def _build_example_context(args):

    print("加载 ne_graph...")
    ne_graph = load_ne_graph(args.resource_buffer, args.ne_graph)
    print(f"  NE 数: {len(ne_graph)}")

    pairwise_args = _resource_buffer_pairwise_args()
    misconnection_pairs = set()
    if getattr(pairwise_args, "transmission_misconnection_filter", False):
        misconnection_pairs, _ = build_transmission_misconnection_pairs(ne_graph)
    inputs = build_site_pair_inputs(
        ne_graph,
        collect_cross_domain=True,
        transmission_misconnection_pairs=misconnection_pairs,
    )
    constraints, _ = build_cross_domain_constraints(
        inputs["cross_domain_pair_evidence"],
        inputs["pair_edge_count"],
    )
    pair_graph_metrics = build_pairwise_graph_metrics(inputs)
    site_metrics, _, _ = build_pairwise_site_metrics(
        inputs, pairwise_args, constraints=constraints
    )
    pair_outputs = build_pairwise_orders(
        inputs,
        site_metrics,
        pair_graph_metrics,
        pairwise_args,
        compact_output=False,
        constraints=constraints,
    )
    return {
        "inputs": inputs,
        "constraints": constraints,
        "pair_graph_metrics": pair_graph_metrics,
        "site_metrics": site_metrics,
        "pair_orders": pair_outputs["pair_orders"],
        "components": pair_outputs["strict_ring_components"],
    }


def _collect_examples(components, constraints, max_sites):
    examples = []
    for component in components:
        component_sites = set(component["sites"])
        inside_constraints = [
            constraint
            for constraint in constraints.values()
            if constraint["upstream_site"] in component_sites
            and constraint["downstream_site"] in component_sites
        ]
        if not inside_constraints:
            continue
        if max_sites and component["site_count"] > max_sites:
            continue
        examples.append((component, inside_constraints))
    examples.sort(key=lambda item: (item[0]["site_count"], item[0]["component_id"]))
    return examples


def _print_example_summary(context, examples, limit):
    components = context["components"]
    constraints = context["constraints"]
    print(f"\n环块总数: {len(components)}")
    print(f"约束总数: {len(constraints)}")
    print(f"两端同块的环块数: {len(examples)}（按站点数升序取前 {limit} 个）")
    size_list = sorted(component["site_count"] for component, _ in examples)
    if size_list:
        median = size_list[len(size_list) // 2]
        print(
            f"命中环块站点数分布: 最小 {size_list[0]} / "
            f"中位 {median} / 最大 {size_list[-1]}"
        )


def _print_component_example(component, inside_constraints, context):
    component_sites = set(component["sites"])
    released = component.get("released_by_constraint", False)
    print(
        f"\n===== 环块 #{component['component_id']} "
        f"| 站点数 {component['site_count']} "
        f"| 块内边数 {component['internal_pair_count']} "
        f"| 出入口 {component['entry_exit_sites'] or '(无,孤立环)'} "
        f"| {'已解除' if released else '未解除'} ====="
    )
    _print_constraints(inside_constraints)
    _print_component_sites(component_sites, context)
    _print_component_edges(component_sites, context)


def _print_constraints(constraints):
    print(f"  命中约束 ({len(constraints)} 条):")
    for constraint in sorted(
        constraints, key=lambda c: (c["upstream_site"], c["downstream_site"])
    ):
        topology = "有" if constraint["has_topology_edge"] else "无(多跳)"
        print(
            f"    {constraint['upstream_site']} -> {constraint['downstream_site']}"
            f" | 证据链路数 {constraint['evidence_link_count']}"
            f"/{constraint['total_cross_link_count']} | 直连拓扑边: {topology}"
        )


def _print_component_sites(component_sites, context):
    print("  站点明细:")
    inputs = context["inputs"]
    for site_id in sorted(component_sites):
        metrics = context["site_metrics"].get(site_id, {})
        domains = _domain_text(inputs["site_domain_counts"].get(site_id))
        print(
            f"    {site_id} | 设备: {domains}"
            f" | 平滑层级 {metrics.get('level_score_smoothed', '?')}"
            f" | 邻居站点数 {metrics.get('neighbor_count', '?')}"
        )


def _print_component_edges(component_sites, context):
    print("  块内边判定:")
    inputs = context["inputs"]
    for pair_key in sorted(inputs["pair_edge_count"]):
        left_site, right_site = pair_key
        if not {left_site, right_site}.issubset(component_sites):
            continue
        if context["pair_graph_metrics"].get(pair_key, {}).get("is_bridge"):
            continue
        pair_result = context["pair_orders"].get(f"{left_site}||{right_site}")
        if pair_result is None:
            continue
        pair_domains = inputs["pair_site_domain_counts"].get(pair_key, {})
        relation = _relation_text(pair_result)
        decision = _decision_label(pair_result)
        left_domains = _domain_text(pair_domains.get(left_site))
        right_domains = _domain_text(pair_domains.get(right_site))
        print(
            f"    {left_site} - {right_site} | 判定: {relation} ({decision})"
            f" | 连边构成: {left_site}[{left_domains}]"
            f" {right_site}[{right_domains}]"
        )


if __name__ == "__main__":
    main()
