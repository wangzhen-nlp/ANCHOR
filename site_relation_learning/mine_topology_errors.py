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
    build_relation_sample,
    build_site_relation_context,
    iter_candidate_relation_sample_chunks,
    vectorize_samples,
    write_json,
)
from site_relation_learning.test_model import _load_model


def _derive_summary_path(output_file):
    if output_file.endswith(".jsonl"):
        return output_file[:-6] + ".summary.json"
    return output_file + ".summary.json"


def _known_relation_samples(context):
    seen_pairs = set()
    samples = []
    for left_site_id, right_site_id in sorted(context.label_relation_map):
        pair_key = tuple(sorted((left_site_id, right_site_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        relation = context.label_relation_map[(pair_key[0], pair_key[1])]
        samples.append(
            build_relation_sample(
                context,
                pair_key[0],
                pair_key[1],
                relation,
                {"known_site_chain_relation"},
                "known_relation",
            )
        )
        samples.append(
            build_relation_sample(
                context,
                pair_key[1],
                pair_key[0],
                context.label_relation_map[(pair_key[1], pair_key[0])],
                {"known_site_chain_relation"},
                "known_relation",
            )
        )
    return samples


def _predict_pair_rows(samples, model_payload, feature_names, weights, biases, no_progress=False, label="候选"):
    from site_relation_learning.core import predict_probabilities

    dense = vectorize_samples(
        samples,
        feature_names,
        model_payload["standardizer"],
        show_progress=not no_progress,
        progress_label=f"向量化{label}",
    )
    probabilities = [
        predict_probabilities(weights, biases, item["x"])
        for item in dense
    ]
    return build_pair_level_prediction_rows(dense, probabilities)


def _predict_missing_error_rows_streaming(
    context,
    model_payload,
    feature_names,
    weights,
    biases,
    threshold,
    max_candidate_count,
    chunk_size,
    seed,
    top_k=0,
    output_file=None,
    output_state=None,
    no_progress=False,
):
    sample_count = 0
    pair_count = 0
    chunk_count = 0

    chunks = iter_candidate_relation_sample_chunks(
        context,
        max_candidate_count=max_candidate_count,
        seed=seed,
        chunk_size=chunk_size,
        show_progress=not no_progress,
        progress_label="扫描潜在缺边候选源站点",
    )
    for samples in chunks:
        chunk_count += 1
        sample_count += len(samples)
        rows = _predict_pair_rows(
            samples,
            model_payload,
            feature_names,
            weights,
            biases,
            no_progress=True,
            label="缺边候选样本",
        )
        pair_count += len(rows)
        chunk_error_rows = []
        for row in rows:
            error_type = _classify_missing_relation_error(row, threshold)
            if error_type:
                chunk_error_rows.append(_format_error_row(row, error_type))
        if output_file is not None and output_state is not None and chunk_error_rows:
            _write_error_rows(output_file, chunk_error_rows, output_state, top_k=top_k)
        if not no_progress and (chunk_count == 1 or chunk_count % 10 == 0):
            print(
                f"已处理缺边候选: chunks={chunk_count}, ordered_samples={sample_count}, "
                f"pairs={pair_count}, retained={output_state['retained_error_count'] if output_state else 0}"
            )

    if not no_progress:
        print(
            f"缺边候选处理完成: chunks={chunk_count}, ordered_samples={sample_count}, "
            f"pairs={pair_count}, retained={output_state['retained_error_count'] if output_state else 0}"
        )
    return sample_count, pair_count


def _classify_known_relation_error(row, min_score):
    gold = row.get("gold_relation", "none")
    pred = row.get("predicted_relation", "none")
    score = float(row.get("predicted_score", 0.0) or 0.0)
    if gold == pred:
        return ""
    if score < min_score:
        return ""
    if gold != "none" and pred == "none":
        return "extra"
    if gold != "none" and pred != "none":
        return "wrong_direction"
    return ""


def _classify_missing_relation_error(row, min_score):
    pred = row.get("predicted_relation", "none")
    score = float(row.get("predicted_score", 0.0) or 0.0)
    if pred == "none" or score < min_score:
        return ""
    return "missing"


def _format_error_row(row, error_type):
    output = dict(row)
    output["error_type"] = error_type
    output["score"] = output.get("predicted_score", 0.0)
    return output


def _error_sort_key(row):
    return (-float(row.get("score", 0.0) or 0.0), row["error_type"], row["sample_id"])


def _new_output_state():
    return {
        "retained_error_count": 0,
        "error_type_counts": {error_type: 0 for error_type in ("missing", "wrong_direction", "extra")},
        "predicted_relation_counts": {label: 0 for label in RELATION_CLASSES},
    }


def _write_error_rows(output_file, rows, state, top_k=0):
    rows = sorted(rows, key=_error_sort_key)
    written = 0
    for row in rows:
        if top_k > 0 and state["retained_error_count"] >= top_k:
            break
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        state["retained_error_count"] += 1
        state["error_type_counts"][row["error_type"]] = state["error_type_counts"].get(row["error_type"], 0) + 1
        predicted_relation = row.get("predicted_relation", "none")
        if predicted_relation not in state["predicted_relation_counts"]:
            state["predicted_relation_counts"][predicted_relation] = 0
        state["predicted_relation_counts"][predicted_relation] += 1
        written += 1
    if written:
        output_file.flush()
    return written


def main():
    parser = ArgumentParser(description="基于站点关系模型挖掘 site_chains.json 中的拓扑错例: 缺/错/多")
    parser.add_argument("--model", required=True, help="site_relation_learning/train_model.py 输出模型 JSON")
    parser.add_argument(
        "--site-chains",
        "--labels",
        dest="site_chains",
        default=SITE_CHAINS_JSON,
        help=f"generate_site_chains.py 输出 JSON，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=f"site_graph.json，默认: {resource_display('site_graph.json')}")
    parser.add_argument(
        "--site-device-counts",
        default=SITE_DEVICE_COUNTS_JSON,
        help=f"site_device_counts.json，默认: {resource_display('site_device_counts.json')}",
    )
    parser.add_argument("-o", "--output", default="site_topology_error_candidates.jsonl", help="错例候选输出 JSONL")
    parser.add_argument("--summary-output", default="", help="摘要输出 JSON")
    parser.add_argument("--min-score", type=float, default=0.95, help="统一高置信阈值，默认: 0.95")
    parser.add_argument("--missing-min-score", type=float, default=-1.0, help="缺边阈值；<0 使用 --min-score")
    parser.add_argument("--wrong-min-score", type=float, default=-1.0, help="错方向阈值；<0 使用 --min-score")
    parser.add_argument("--extra-min-score", type=float, default=-1.0, help="多边阈值；<0 使用 --min-score")
    parser.add_argument("--max-candidate-count", type=int, default=50000, help="missing 候选池上限，默认: 50000")
    parser.add_argument("--candidate-chunk-size", type=int, default=500, help="missing 候选每批扫描的源站点数，默认: 500")
    parser.add_argument("--top-k", type=int, default=0, help="最多输出前 K 条；0 表示不限制")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    missing_threshold = args.missing_min_score if args.missing_min_score >= 0 else args.min_score
    wrong_threshold = args.wrong_min_score if args.wrong_min_score >= 0 else args.min_score
    extra_threshold = args.extra_min_score if args.extra_min_score >= 0 else args.min_score

    print(f"加载模型: {args.model}")
    model_payload, feature_names, weights, biases = _load_model(args.model)
    print(f"加载站点关系上下文: {args.site_chains}")
    context = build_site_relation_context(args.site_chains, args.site_graph, args.site_device_counts)

    print("构造已知一跳关系候选...")
    known_samples = _known_relation_samples(context)
    print(f"已知关系有序样本数: {len(known_samples)}")
    known_rows = _predict_pair_rows(
        known_samples,
        model_payload,
        feature_names,
        weights,
        biases,
        no_progress=args.no_progress,
        label="已知关系样本",
    )

    output_state = _new_output_state()
    known_error_rows = []
    for row in known_rows:
        error_type = _classify_known_relation_error(
            row,
            extra_threshold if row.get("predicted_relation") == "none" else wrong_threshold,
        )
        if error_type:
            known_error_rows.append(_format_error_row(row, error_type))

    print(f"写出已知关系错例: {args.output}")
    with open(args.output, "w", encoding="utf-8") as output_file:
        _write_error_rows(output_file, known_error_rows, output_state, top_k=args.top_k)
        print(f"已知关系错例已写出: {output_state['retained_error_count']}")

        print("构造并流式预测潜在缺边候选...")
        missing_sample_count, missing_pair_count = _predict_missing_error_rows_streaming(
            context,
            model_payload,
            feature_names,
            weights,
            biases,
            threshold=missing_threshold,
            max_candidate_count=args.max_candidate_count,
            chunk_size=args.candidate_chunk_size,
            seed=args.seed,
            top_k=args.top_k,
            output_file=output_file,
            output_state=output_state,
            no_progress=args.no_progress,
        )

    summary_output = args.summary_output or _derive_summary_path(args.output)
    counts = output_state["error_type_counts"]
    relation_counts = output_state["predicted_relation_counts"]

    write_json(
        summary_output,
        {
            "model": args.model,
            "site_chains": args.site_chains,
            "site_graph": args.site_graph,
            "site_device_counts": args.site_device_counts,
            "known_pair_count": len(known_rows),
            "missing_candidate_ordered_sample_count": missing_sample_count,
            "missing_candidate_pair_count": missing_pair_count,
            "candidate_source_site_chunk_size": args.candidate_chunk_size,
            "retained_error_count": output_state["retained_error_count"],
            "error_type_counts": counts,
            "predicted_relation_counts": relation_counts,
            "thresholds": {
                "missing": missing_threshold,
                "wrong_direction": wrong_threshold,
                "extra": extra_threshold,
            },
            "output": args.output,
        },
    )

    print(f"已知关系 pair 数: {len(known_rows)}")
    print(f"缺边候选有序样本数: {missing_sample_count}")
    print(f"缺边候选 pair 数: {missing_pair_count}")
    print(f"错例候选数: {output_state['retained_error_count']}")
    print(f"错例类型分布: {counts}")
    print(f"输出: {args.output}")
    print(f"摘要: {summary_output}")


if __name__ == "__main__":
    main()
