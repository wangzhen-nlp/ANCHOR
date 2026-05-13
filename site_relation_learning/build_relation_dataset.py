#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import SITE_CHAINS_JSON, SITE_DEVICE_COUNTS_JSON, SITE_GRAPH_JSON, resource_display
from site_relation_learning.core import (
    RELATION_CLASSES,
    build_site_relation_context,
    generate_relation_learning_samples,
    summarize_samples,
    write_json,
    write_jsonl,
)


def _derive_summary_path(output_file):
    if output_file.endswith(".jsonl"):
        return output_file[:-6] + ".summary.json"
    return output_file + ".summary.json"


def _format_label_counts(summary):
    counts = summary.get("label_counts", {})
    return ", ".join(
        f"{label}={counts.get(label, 0)}"
        for label in RELATION_CLASSES
    )


def main():
    parser = ArgumentParser(
        description=(
            "基于 generate_site_chains.py 输出的一跳站点关系标签，"
            "构造站点关系四分类样本: bidirection/upstream/downstream/none"
        )
    )
    parser.add_argument(
        "--labels",
        default=SITE_CHAINS_JSON,
        help=f"generate_site_chains.py 输出 JSON，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"site_graph.json，提供站点经纬度/区域信息，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--site-device-counts",
        default=SITE_DEVICE_COUNTS_JSON,
        help=f"site_device_counts.json，提供站点设备类型统计，默认: {resource_display('site_device_counts.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="site_relation_dataset.jsonl",
        help="输出样本 JSONL，默认: site_relation_dataset.jsonl",
    )
    parser.add_argument("--summary-output", default="", help="样本统计 JSON；默认与输出同名前缀")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--none-per-positive", type=float, default=2.0, help="none 样本数 / 非 none 样本数，默认: 2.0")
    parser.add_argument(
        "--same-region-negatives",
        type=int,
        default=1,
        help="同区域 hard none 每侧采样数；源侧和目标侧会分别尝试，默认: 1",
    )
    parser.add_argument(
        "--same-domain-negatives",
        type=int,
        default=1,
        help="同主 domain hard none 每侧采样数；源侧和目标侧会分别尝试，默认: 1",
    )
    parser.add_argument("--nearest-negatives", type=int, default=1, help="近距离 hard none 采样数，默认: 1")
    parser.add_argument("--random-negative-ratio", type=float, default=1.0, help="随机 none 补充比例，默认: 1.0")
    parser.add_argument("--none-max-rounds", type=int, default=3, help="hard none 候选最多扫描轮数，默认: 3")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    print(f"加载标签: {args.labels}")
    print("注意: upstream/downstream 标签只使用 site_chains 中 hop=1 的一跳关系，多跳 downstream/upstream 不作为监督标签。")
    context = build_site_relation_context(args.labels, args.site_graph, args.site_device_counts)
    samples = generate_relation_learning_samples(
        context,
        seed=args.seed,
        none_per_positive=args.none_per_positive,
        same_region_negatives=args.same_region_negatives,
        same_domain_negatives=args.same_domain_negatives,
        nearest_negatives=args.nearest_negatives,
        random_negative_ratio=args.random_negative_ratio,
        none_max_rounds=args.none_max_rounds,
        show_progress=not args.no_progress,
    )

    write_jsonl(args.output, samples)
    summary_output = args.summary_output or _derive_summary_path(args.output)
    summary = summarize_samples(samples)
    summary.update(
        {
            "labels": args.labels,
            "site_graph": args.site_graph,
            "site_device_counts": args.site_device_counts,
            "output": args.output,
            "seed": args.seed,
            "site_count": len(context.site_infos),
            "label_source": "site_chains.sites only; upstream/downstream labels require hop == 1",
            "config": {
                "none_per_positive": args.none_per_positive,
                "same_region_negatives": args.same_region_negatives,
                "same_domain_negatives": args.same_domain_negatives,
                "nearest_negatives": args.nearest_negatives,
                "random_negative_ratio": args.random_negative_ratio,
                "none_max_rounds": args.none_max_rounds,
            },
        }
    )
    write_json(summary_output, summary)

    print(f"站点数: {len(context.site_infos)}")
    print(f"样本数: {summary['sample_count']}")
    print(f"类别分布: {_format_label_counts(summary)}")
    print(f"特征数: {summary['feature_name_count']}")
    print(f"样本已输出到: {args.output}")
    print(f"统计已输出到: {summary_output}")


if __name__ == "__main__":
    main()
