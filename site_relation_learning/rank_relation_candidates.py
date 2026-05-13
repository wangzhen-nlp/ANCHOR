#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import SITE_CHAINS_JSON, SITE_DEVICE_COUNTS_JSON, SITE_GRAPH_JSON, resource_display
from site_relation_learning.core import (
    RELATION_CLASSES,
    build_pair_level_prediction_rows,
    build_site_relation_context,
    iter_candidate_relation_sample_chunks,
    predict_probabilities_batch,
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


def _row_sort_key(row):
    return (-row["predicted_score"], row["predicted_relation"], row["sample_id"])


def _filter_rows(rows, min_score, include_none):
    retained = []
    for row in rows:
        predicted_label = row["predicted_relation"]
        score = row["predicted_score"]
        if score < min_score:
            continue
        if not include_none and predicted_label == "none":
            continue
        retained.append(row)
    return retained


def _append_top_rows(retained, rows, top_k):
    retained.extend(rows)
    if top_k > 0 and len(retained) > max(top_k * 2, top_k + 1000):
        retained.sort(key=_row_sort_key)
        del retained[top_k:]


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
    parser.add_argument(
        "--candidate-max-samples-per-chunk",
        type=int,
        default=20000,
        help="候选每批最多 ordered samples 数，默认: 20000",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    model_payload, feature_names, weights, biases = _load_model(args.model)
    context = build_site_relation_context(args.labels, args.site_graph, args.site_device_counts)

    output = args.output or _derive_output_path(args.model)
    summary_output = args.summary_output or _derive_summary_path(args.model)
    stream_output = args.top_k == 0
    retained = []
    candidate_count = 0
    pair_candidate_count = 0
    retained_count = 0
    chunk_count = 0
    output_file = open(output, "w", encoding="utf-8") if stream_output else None
    try:
        chunks = iter_candidate_relation_sample_chunks(
            context,
            max_candidate_count=args.max_candidate_count,
            seed=args.seed,
            max_samples_per_chunk=args.candidate_max_samples_per_chunk,
            show_progress=not args.no_progress,
            progress_label="扫描候选站点对",
        )
        for samples in chunks:
            chunk_count += 1
            candidate_count += len(samples)
            dense = vectorize_samples(
                samples,
                feature_names,
                model_payload["standardizer"],
                show_progress=False,
                progress_label="向量化候选站点对",
            )
            probabilities = (
                predict_probabilities_batch(weights, biases, [item["x"] for item in dense]).tolist()
                if dense else []
            )
            rows = build_pair_level_prediction_rows(dense, probabilities)
            pair_candidate_count += len(rows)
            chunk_retained = _filter_rows(rows, args.min_score, args.include_none)
            if stream_output:
                chunk_retained.sort(key=_row_sort_key)
                for row in chunk_retained:
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                if chunk_retained:
                    output_file.flush()
                retained_count += len(chunk_retained)
            else:
                _append_top_rows(retained, chunk_retained, args.top_k)
                retained_count = len(retained)
            if not args.no_progress and (chunk_count == 1 or chunk_count % 10 == 0):
                print(
                    f"已处理候选: chunks={chunk_count}, ordered_samples={candidate_count}, "
                    f"pairs={pair_candidate_count}, retained={retained_count}"
                )
    finally:
        if output_file is not None:
            output_file.close()

    if not stream_output:
        retained.sort(key=_row_sort_key)
        if args.top_k > 0:
            retained = retained[: args.top_k]
        retained_count = len(retained)
        write_jsonl(output, retained)

    write_json(
        summary_output,
        {
            "model": args.model,
            "labels": args.labels,
            "candidate_count": candidate_count,
            "pair_candidate_count": pair_candidate_count,
            "retained_count": retained_count,
            "classes": list(RELATION_CLASSES),
            "top_k": args.top_k,
            "min_score": args.min_score,
            "include_none": args.include_none,
            "candidate_max_ordered_samples_per_chunk": args.candidate_max_samples_per_chunk,
            "global_sort": not stream_output,
            "output": output,
        },
    )

    print(f"候选有序样本数: {candidate_count}")
    print(f"候选 pair 数: {pair_candidate_count}")
    print(f"保留候选数: {retained_count}")
    print(f"排序结果已输出到: {output}")
    print(f"摘要已输出到: {summary_output}")


if __name__ == "__main__":
    main()
