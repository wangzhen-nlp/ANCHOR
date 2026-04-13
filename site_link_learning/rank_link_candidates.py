#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import NE_GRAPH_JSON, resource_display
from ne_link_learning.core import load_json, load_ne_graph, vectorize_samples, write_json, write_jsonl
from site_link_learning.core import build_site_pair_context, generate_candidate_site_link_samples_for_scoring


def _predict_probability(weights, bias, dense_vector):
    score = bias
    for weight, feature_value in zip(weights, dense_vector):
        score += weight * feature_value
    if score >= 0:
        exp_term = math.exp(-score)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(score)
    return exp_term / (1.0 + exp_term)


def _derive_output_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".site_ranked_candidates.jsonl"
    return model_file + ".site_ranked_candidates.jsonl"


def _derive_summary_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".site_ranked_candidates.summary.json"
    return model_file + ".site_ranked_candidates.summary.json"


def main():
    parser = ArgumentParser(description="用训练好的模型对当前站点对候选缺失边进行打分排序")
    parser.add_argument("--model", required=True, help="模型 JSON")
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件路径，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument("--output", default="", help="排序后候选输出 JSONL")
    parser.add_argument("--summary-output", default="", help="摘要输出 JSON")
    parser.add_argument("--top-k", type=int, default=1000, help="最多输出前 K 条候选，默认: 1000")
    parser.add_argument("--min-score", type=float, default=0.0, help="最低保留分数，默认: 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--max-candidate-count", type=int, default=20000, help="候选池上限，默认: 20000")
    parser.add_argument(
        "--same-source-region-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个同源 region 候选，默认: 2",
    )
    parser.add_argument(
        "--same-target-region-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个同目标 region 候选，默认: 2",
    )
    parser.add_argument(
        "--same-source-domain-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个同源 dominant domain 候选，默认: 2",
    )
    parser.add_argument(
        "--same-target-domain-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个同目标 dominant domain 候选，默认: 2",
    )
    parser.add_argument(
        "--two-hop-target-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个两跳目标候选，默认: 2",
    )
    parser.add_argument(
        "--two-hop-source-negatives",
        type=int,
        default=2,
        help="每条正样本参考边额外采多少个两跳源头候选，默认: 2",
    )
    parser.add_argument(
        "--reverse-direction-negatives",
        type=int,
        default=1,
        help="是否补充反向缺失站点对候选（>0 表示开启），默认: 1",
    )
    parser.add_argument(
        "--random-hard-negative-ratio",
        type=float,
        default=2.0,
        help="额外随机硬候选比例（相对正样本数），默认: 2.0",
    )
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    model_payload = load_json(args.model)
    feature_names = model_payload["feature_names"]
    standardizer = model_payload["standardizer"]
    weights = [model_payload["weights"].get(feature_name, 0.0) for feature_name in feature_names]
    bias = float(model_payload.get("bias", 0.0))

    print(f"加载 ne_graph: {args.ne_graph}")
    ne_graph_data = load_ne_graph(args.ne_graph)
    print("构建站点上下文...")
    context = build_site_pair_context(ne_graph_data)
    print("生成候选站点对...")
    candidate_samples_raw = generate_candidate_site_link_samples_for_scoring(
        context=context,
        max_candidate_count=args.max_candidate_count,
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

    print("整理候选样本元数据...")
    candidate_samples = []
    candidate_progress = ProgressBar(len(candidate_samples_raw), "整理候选样本")
    try:
        for index, item in enumerate(candidate_samples_raw, start=1):
            candidate_samples.append(
                {
                    "sample_id": item["sample_id"],
                    "label": 0,
                    "features": item["features"],
                    "meta": {
                        key: value
                        for key, value in item.items()
                        if key not in {"sample_id", "label", "features"}
                    },
                }
            )
            candidate_progress.set(index)
    finally:
        candidate_progress.close()

    print("向量化候选样本...")
    dense_samples = vectorize_samples(
        candidate_samples,
        feature_names,
        standardizer,
        show_progress=True,
        progress_label="向量化候选样本",
    )

    scored_rows = []
    print("候选样本打分...")
    score_progress = ProgressBar(len(candidate_samples_raw), "候选样本打分")
    try:
        for index, (raw_item, dense_item) in enumerate(zip(candidate_samples_raw, dense_samples), start=1):
            probability = _predict_probability(weights, bias, dense_item["x"])
            if probability >= args.min_score:
                scored_rows.append({**raw_item, "score": probability})
            score_progress.set(index)
            score_progress.set_extra_text(f"保留 {len(scored_rows)} 条")
    finally:
        score_progress.close()

    scored_rows.sort(key=lambda item: (-item["score"], item["sample_id"]))
    if args.top_k > 0:
        scored_rows = scored_rows[: args.top_k]

    output_file = args.output or _derive_output_path(args.model)
    summary_output = args.summary_output or _derive_summary_path(args.model)
    print(f"写出排序结果: {output_file}")
    write_jsonl(output_file, scored_rows)
    print(f"写出摘要: {summary_output}")
    write_json(
        summary_output,
        {
            "model": args.model,
            "sample_granularity": "site_pair",
            "ne_graph": args.ne_graph,
            "candidate_pool_size": len(candidate_samples_raw),
            "retained_candidate_count": len(scored_rows),
            "top_k": args.top_k,
            "min_score": args.min_score,
            "output": output_file,
        },
    )

    print(f"候选池大小: {len(candidate_samples_raw)}")
    print(f"保留候选数: {len(scored_rows)}")
    print(f"排序结果已输出到: {output_file}")
    print(f"摘要已输出到: {summary_output}")


if __name__ == "__main__":
    main()
