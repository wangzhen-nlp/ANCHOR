#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import SITE_CHAINS_JSON, SITE_DEVICE_COUNTS_JSON, SITE_GRAPH_JSON, resource_display
from site_relation_learning.core import (
    RELATION_CLASSES,
    build_pair_level_prediction_rows,
    build_site_relation_context,
    generate_candidate_relation_samples,
    vectorize_samples,
    write_json,
    write_jsonl,
)
from site_relation_learning.test_model import _load_model


def _derive_output_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".ranked_site_relations.jsonl"
    return model_file + ".ranked_site_relations.jsonl"


def _derive_summary_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".ranked_site_relations.summary.json"
    return model_file + ".ranked_site_relations.summary.json"


def main():
    parser = ArgumentParser(description="用站点关系模型对候选站点对四分类打分排序")
    parser.add_argument("--model", required=True, help="模型 JSON")
    parser.add_argument(
        "--labels",
        default=SITE_CHAINS_JSON,
        help=f"generate_site_chains.py 输出 JSON，用于构造上下文并排除已知一跳关系，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=f"site_graph.json，默认: {resource_display('site_graph.json')}")
    parser.add_argument(
        "--site-device-counts",
        default=SITE_DEVICE_COUNTS_JSON,
        help=f"site_device_counts.json，默认: {resource_display('site_device_counts.json')}",
    )
    parser.add_argument("--output", default="", help="排序结果 JSONL")
    parser.add_argument("--summary-output", default="", help="摘要 JSON")
    parser.add_argument("--top-k", type=int, default=1000, help="最多输出前 K 条，默认: 1000")
    parser.add_argument("--min-score", type=float, default=0.0, help="最低预测概率，默认: 0")
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="默认只输出非 none 预测；开启后也输出 none",
    )
    parser.add_argument("--max-candidate-count", type=int, default=50000, help="候选池上限，默认: 50000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    model_payload, feature_names, weights, biases = _load_model(args.model)
    context = build_site_relation_context(args.labels, args.site_graph, args.site_device_counts)
    candidate_samples = generate_candidate_relation_samples(
        context,
        max_candidate_count=args.max_candidate_count,
        seed=args.seed,
    )
    dense = vectorize_samples(
        candidate_samples,
        feature_names,
        model_payload["standardizer"],
        show_progress=not args.no_progress,
        progress_label="向量化候选站点对",
    )

    probabilities = []
    from site_relation_learning.core import predict_probabilities

    for item in dense:
        probabilities.append(predict_probabilities(weights, biases, item["x"]))
    rows = build_pair_level_prediction_rows(dense, probabilities)
    retained = []
    for row in rows:
        predicted_label = row["predicted_relation"]
        score = row["predicted_score"]
        if score < args.min_score:
            continue
        if not args.include_none and predicted_label == "none":
            continue
        retained.append(row)

    retained.sort(key=lambda item: (-item["predicted_score"], item["predicted_relation"], item["sample_id"]))
    if args.top_k > 0:
        retained = retained[: args.top_k]

    output = args.output or _derive_output_path(args.model)
    summary_output = args.summary_output or _derive_summary_path(args.model)
    write_jsonl(output, retained)
    write_json(
        summary_output,
        {
            "model": args.model,
            "labels": args.labels,
            "candidate_count": len(candidate_samples),
            "pair_candidate_count": len(rows),
            "retained_count": len(retained),
            "classes": list(RELATION_CLASSES),
            "top_k": args.top_k,
            "min_score": args.min_score,
            "include_none": args.include_none,
            "output": output,
        },
    )

    print(f"候选池大小: {len(candidate_samples)}")
    print(f"保留候选数: {len(retained)}")
    print(f"排序结果已输出到: {output}")
    print(f"摘要已输出到: {summary_output}")


if __name__ == "__main__":
    main()
