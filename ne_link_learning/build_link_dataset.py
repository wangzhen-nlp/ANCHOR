#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display
from ne_link_learning.core import (
    build_graph_context,
    generate_link_learning_samples,
    load_ne_graph,
    summarize_samples,
    write_json,
    write_jsonl,
)


def _derive_summary_path(output_file):
    if output_file.endswith(".jsonl"):
        return output_file[:-6] + ".summary.json"
    return output_file + ".summary.json"


def main():
    parser = ArgumentParser(
        description="基于 ne_graph.json 构造 NE 跨站点有向连边学习样本（正样本=已观测连边，负样本=局部硬负样本）"
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
        default="topology_link_dataset.jsonl",
        help="输出样本 JSONL，默认: topology_link_dataset.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="样本统计输出 JSON；默认与输出同名前缀",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认: 42",
    )
    parser.add_argument(
        "--max-negative-per-positive",
        type=float,
        default=4.0,
        help="最多保留多少个负样本 / 正样本，默认: 4.0",
    )
    parser.add_argument(
        "--same-source-site-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同源站点替代负样本，默认: 1",
    )
    parser.add_argument(
        "--same-target-site-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同目标站点替代负样本，默认: 1",
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
        "--site-pair-negatives",
        type=int,
        default=1,
        help="每条正样本额外采多少个同站点对未连通负样本，默认: 1",
    )
    parser.add_argument(
        "--reverse-direction-negatives",
        type=int,
        default=1,
        help="是否补充反向缺失边作为负样本（>0 表示开启），默认: 1",
    )
    parser.add_argument(
        "--random-hard-negative-ratio",
        type=float,
        default=1.0,
        help="额外随机硬负样本比例（相对正样本数），默认: 1.0",
    )
    args = parser.parse_args()

    ne_graph_data = load_ne_graph(args.ne_graph)
    context = build_graph_context(ne_graph_data)
    samples = generate_link_learning_samples(
        context=context,
        max_negative_per_positive=args.max_negative_per_positive,
        seed=args.seed,
        same_source_site_negatives=args.same_source_site_negatives,
        same_target_site_negatives=args.same_target_site_negatives,
        two_hop_target_negatives=args.two_hop_target_negatives,
        two_hop_source_negatives=args.two_hop_source_negatives,
        site_pair_negatives=args.site_pair_negatives,
        reverse_direction_negatives=args.reverse_direction_negatives,
        random_hard_negative_ratio=args.random_hard_negative_ratio,
    )

    write_jsonl(args.output, samples)

    summary_output = args.summary_output or _derive_summary_path(args.output)
    summary = summarize_samples(samples)
    summary.update(
        {
            "ne_graph": args.ne_graph,
            "output": args.output,
            "seed": args.seed,
            "node_count": len(context.node_infos),
            "site_count": len(context.site_to_nodes),
            "config": {
                "max_negative_per_positive": args.max_negative_per_positive,
                "same_source_site_negatives": args.same_source_site_negatives,
                "same_target_site_negatives": args.same_target_site_negatives,
                "two_hop_target_negatives": args.two_hop_target_negatives,
                "two_hop_source_negatives": args.two_hop_source_negatives,
                "site_pair_negatives": args.site_pair_negatives,
                "reverse_direction_negatives": args.reverse_direction_negatives,
                "random_hard_negative_ratio": args.random_hard_negative_ratio,
            },
        }
    )
    write_json(summary_output, summary)

    print(f"NE 节点数: {len(context.node_infos)}")
    print(f"站点数: {len(context.site_to_nodes)}")
    print(f"样本数: {summary['sample_count']}")
    print(f"  正样本: {summary['positive_count']}")
    print(f"  负样本: {summary['negative_count']}")
    print(f"特征数: {summary['feature_name_count']}")
    print(f"样本已输出到: {args.output}")
    print(f"统计已输出到: {summary_output}")


if __name__ == "__main__":
    main()
