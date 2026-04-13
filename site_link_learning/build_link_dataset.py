#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display
from ne_link_learning.core import load_ne_graph, summarize_samples, write_json, write_jsonl
from site_link_learning.core import build_site_pair_context, generate_site_link_learning_samples


def _derive_summary_path(output_file):
    if output_file.endswith(".jsonl"):
        return output_file[:-6] + ".summary.json"
    return output_file + ".summary.json"


def main():
    parser = ArgumentParser(
        description="基于 ne_graph.json 构造站点对级别的有向拓扑补边学习样本"
    )
    parser.add_argument(
        "ne_graph",
        nargs="?",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件路径，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="site_topology_link_dataset.jsonl",
        help="输出样本 JSONL，默认: site_topology_link_dataset.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="样本统计输出 JSON；默认与输出同名前缀",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument(
        "--max-negative-per-positive",
        type=float,
        default=4.0,
        help="最多保留多少个负样本 / 正样本，默认: 4.0",
    )
    parser.add_argument(
        "--same-source-region-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同源 region 替代负样本，默认: 1",
    )
    parser.add_argument(
        "--same-target-region-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同目标 region 替代负样本，默认: 1",
    )
    parser.add_argument(
        "--same-source-domain-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同源 dominant domain 替代负样本，默认: 1",
    )
    parser.add_argument(
        "--same-target-domain-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同目标 dominant domain 替代负样本，默认: 1",
    )
    parser.add_argument(
        "--two-hop-target-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个两跳目标负样本，默认: 1",
    )
    parser.add_argument(
        "--two-hop-source-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个两跳源头负样本，默认: 1",
    )
    parser.add_argument(
        "--reverse-direction-negatives",
        type=int,
        default=1,
        help="是否补充反向缺失站点对作为负样本（>0 表示开启），默认: 1",
    )
    parser.add_argument(
        "--random-hard-negative-ratio",
        type=float,
        default=1.0,
        help="额外随机硬负样本比例（相对正样本数），默认: 1.0",
    )
    args = parser.parse_args()

    print(f"加载 ne_graph: {args.ne_graph}")
    ne_graph_data = load_ne_graph(args.ne_graph)
    print("构建站点上下文...")
    context = build_site_pair_context(ne_graph_data)
    print("生成站点对学习样本...")
    samples = generate_site_link_learning_samples(
        context=context,
        max_negative_per_positive=args.max_negative_per_positive,
        seed=args.seed,
        same_source_region_negatives=args.same_source_region_negatives,
        same_target_region_negatives=args.same_target_region_negatives,
        same_source_domain_negatives=args.same_source_domain_negatives,
        same_target_domain_negatives=args.same_target_domain_negatives,
        two_hop_target_negatives=args.two_hop_target_negatives,
        two_hop_source_negatives=args.two_hop_source_negatives,
        reverse_direction_negatives=args.reverse_direction_negatives,
        random_hard_negative_ratio=args.random_hard_negative_ratio,
        show_progress=True,
    )

    print(f"写出样本文件: {args.output}")
    write_jsonl(args.output, samples)

    summary_output = args.summary_output or _derive_summary_path(args.output)
    summary = summarize_samples(samples)
    summary.update(
        {
            "sample_granularity": "site_pair",
            "ne_graph": args.ne_graph,
            "output": args.output,
            "seed": args.seed,
            "node_count": len(context.base_context.node_infos),
            "site_count": len(context.site_infos),
            "config": {
                "max_negative_per_positive": args.max_negative_per_positive,
                "same_source_region_negatives": args.same_source_region_negatives,
                "same_target_region_negatives": args.same_target_region_negatives,
                "same_source_domain_negatives": args.same_source_domain_negatives,
                "same_target_domain_negatives": args.same_target_domain_negatives,
                "two_hop_target_negatives": args.two_hop_target_negatives,
                "two_hop_source_negatives": args.two_hop_source_negatives,
                "reverse_direction_negatives": args.reverse_direction_negatives,
                "random_hard_negative_ratio": args.random_hard_negative_ratio,
            },
        }
    )
    print(f"写出统计文件: {summary_output}")
    write_json(summary_output, summary)

    print(f"NE 节点数: {len(context.base_context.node_infos)}")
    print(f"站点数: {len(context.site_infos)}")
    print(f"样本数: {summary['sample_count']}")
    print(f"  正样本: {summary['positive_count']}")
    print(f"  负样本: {summary['negative_count']}")
    print(f"特征数: {summary['feature_name_count']}")
    print(f"样本已输出到: {args.output}")
    print(f"统计已输出到: {summary_output}")


if __name__ == "__main__":
    main()
