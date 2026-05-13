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
    predict_probabilities_batch,
    vectorize_samples,
    write_json,
)
from site_relation_learning.test_model import _load_model


def _derive_rank_output_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".ranked_site_relations.jsonl"
    return model_file + ".ranked_site_relations.jsonl"


def _derive_rank_summary_path(model_file):
    if model_file.endswith(".json"):
        return model_file[:-5] + ".ranked_site_relations.summary.json"
    return model_file + ".ranked_site_relations.summary.json"


def _derive_error_summary_path(output_file):
    if output_file.endswith(".jsonl"):
        return output_file[:-6] + ".summary.json"
    return output_file + ".summary.json"


def _row_sort_key(row):
    return (-row["predicted_score"], row["predicted_relation"], row["sample_id"])


def _filter_rank_rows(rows, min_score, include_none):
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


def _known_relation_samples(context):
    seen_pairs = set()
    samples = []
    for left_site_id, right_site_id in sorted(context.label_relation_map):
        pair_key = tuple(sorted((left_site_id, right_site_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        samples.append(
            build_relation_sample(
                context,
                pair_key[0],
                pair_key[1],
                context.label_relation_map[(pair_key[0], pair_key[1])],
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
    dense = vectorize_samples(
        samples,
        feature_names,
        model_payload["standardizer"],
        show_progress=not no_progress,
        progress_label=f"向量化{label}",
    )
    probabilities = (
        predict_probabilities_batch(weights, biases, [item["x"] for item in dense]).tolist()
        if dense else []
    )
    return build_pair_level_prediction_rows(dense, probabilities)


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


def _write_error_rows(output_file, rows, state):
    rows = sorted(rows, key=_error_sort_key)
    written = 0
    for row in rows:
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


def _predict_rank_rows_streaming(args, model_payload, feature_names, weights, biases, context, output, summary_output):
    candidate_count = 0
    pair_candidate_count = 0
    retained_count = 0
    chunk_count = 0
    output_file = open(output, "w", encoding="utf-8")
    try:
        chunks = iter_candidate_relation_sample_chunks(
            context,
            max_candidate_count=args.max_candidate_count,
            seed=args.seed,
            max_samples_per_chunk=args.candidate_max_samples_per_chunk,
            same_region_limit=args.candidate_same_region_limit,
            same_domain_limit=args.candidate_same_domain_limit,
            topology_neighbor_limit=args.candidate_topology_neighbor_limit,
            nearest_limit=args.candidate_nearest_limit,
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
            chunk_retained = _filter_rank_rows(rows, args.min_score, args.include_none)
            chunk_retained.sort(key=_row_sort_key)
            for row in chunk_retained:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if chunk_retained:
                output_file.flush()
            retained_count += len(chunk_retained)
            if not args.no_progress and (chunk_count == 1 or chunk_count % 10 == 0):
                print(
                    f"已处理候选: chunks={chunk_count}, ordered_samples={candidate_count}, "
                    f"pairs={pair_candidate_count}, retained={retained_count}"
                )
    finally:
        output_file.close()

    write_json(
        summary_output,
        {
            "mode": "rank",
            "model": args.model,
            "labels": args.site_chains,
            "candidate_count": candidate_count,
            "pair_candidate_count": pair_candidate_count,
            "retained_count": retained_count,
            "classes": list(RELATION_CLASSES),
            "min_score": args.min_score,
            "include_none": args.include_none,
            "candidate_max_ordered_samples_per_chunk": args.candidate_max_samples_per_chunk,
            "candidate_limits": {
                "same_region": args.candidate_same_region_limit,
                "same_domain": args.candidate_same_domain_limit,
                "topology_neighbor": args.candidate_topology_neighbor_limit,
                "nearest": args.candidate_nearest_limit,
            },
            "global_sort": False,
            "output": output,
        },
    )

    print(f"候选有序样本数: {candidate_count}")
    print(f"候选 pair 数: {pair_candidate_count}")
    print(f"保留候选数: {retained_count}")
    print(f"排序结果已输出到: {output}")
    print(f"摘要已输出到: {summary_output}")


def _predict_missing_error_rows_streaming(
    args,
    model_payload,
    feature_names,
    weights,
    biases,
    context,
    threshold,
    output_file,
    output_state,
):
    sample_count = 0
    pair_count = 0
    chunk_count = 0

    chunks = iter_candidate_relation_sample_chunks(
        context,
        max_candidate_count=args.max_candidate_count,
        seed=args.seed,
        max_samples_per_chunk=args.candidate_max_samples_per_chunk,
        same_region_limit=args.candidate_same_region_limit,
        same_domain_limit=args.candidate_same_domain_limit,
        topology_neighbor_limit=args.candidate_topology_neighbor_limit,
        nearest_limit=args.candidate_nearest_limit,
        show_progress=not args.no_progress,
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
        if chunk_error_rows:
            _write_error_rows(output_file, chunk_error_rows, output_state)
        if not args.no_progress and (chunk_count == 1 or chunk_count % 10 == 0):
            print(
                f"已处理缺边候选: chunks={chunk_count}, ordered_samples={sample_count}, "
                f"pairs={pair_count}, retained={output_state['retained_error_count']}"
            )

    if not args.no_progress:
        print(
            f"缺边候选处理完成: chunks={chunk_count}, ordered_samples={sample_count}, "
            f"pairs={pair_count}, retained={output_state['retained_error_count']}"
        )
    return sample_count, pair_count


def _predict_topology_error_rows(args, model_payload, feature_names, weights, biases, context, output, summary_output):
    missing_threshold = args.missing_min_score if args.missing_min_score >= 0 else args.min_score
    wrong_threshold = args.wrong_min_score if args.wrong_min_score >= 0 else args.min_score
    extra_threshold = args.extra_min_score if args.extra_min_score >= 0 else args.min_score

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

    print(f"写出已知关系错例: {output}")
    with open(output, "w", encoding="utf-8") as output_file:
        _write_error_rows(output_file, known_error_rows, output_state)
        print(f"已知关系错例已写出: {output_state['retained_error_count']}")

        print("构造并流式预测潜在缺边候选...")
        missing_sample_count, missing_pair_count = _predict_missing_error_rows_streaming(
            args,
            model_payload,
            feature_names,
            weights,
            biases,
            context,
            missing_threshold,
            output_file,
            output_state,
        )

    counts = output_state["error_type_counts"]
    relation_counts = output_state["predicted_relation_counts"]
    write_json(
        summary_output,
        {
            "mode": "topology-errors",
            "model": args.model,
            "site_chains": args.site_chains,
            "site_graph": args.site_graph,
            "site_device_counts": args.site_device_counts,
            "known_pair_count": len(known_rows),
            "missing_candidate_ordered_sample_count": missing_sample_count,
            "missing_candidate_pair_count": missing_pair_count,
            "candidate_max_ordered_samples_per_chunk": args.candidate_max_samples_per_chunk,
            "candidate_limits": {
                "same_region": args.candidate_same_region_limit,
                "same_domain": args.candidate_same_domain_limit,
                "topology_neighbor": args.candidate_topology_neighbor_limit,
                "nearest": args.candidate_nearest_limit,
            },
            "retained_error_count": output_state["retained_error_count"],
            "error_type_counts": counts,
            "predicted_relation_counts": relation_counts,
            "thresholds": {
                "missing": missing_threshold,
                "wrong_direction": wrong_threshold,
                "extra": extra_threshold,
            },
            "output": output,
        },
    )

    print(f"已知关系 pair 数: {len(known_rows)}")
    print(f"缺边候选有序样本数: {missing_sample_count}")
    print(f"缺边候选 pair 数: {missing_pair_count}")
    print(f"错例候选数: {output_state['retained_error_count']}")
    print(f"错例类型分布: {counts}")
    print(f"输出: {output}")
    print(f"摘要: {summary_output}")


def _derive_output_path(args):
    if args.output:
        return args.output
    if args.mode == "rank":
        return _derive_rank_output_path(args.model)
    return "site_topology_error_candidates.jsonl"


def _derive_summary_output_path(args, output):
    if args.summary_output:
        return args.summary_output
    if args.mode == "rank":
        return _derive_rank_summary_path(args.model)
    return _derive_error_summary_path(output)


def main():
    parser = ArgumentParser(description="站点关系模型统一推理入口：候选排序或拓扑错例挖掘")
    parser.add_argument("--mode", choices=("rank", "topology-errors"), required=True, help="推理模式")
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
    parser.add_argument("-o", "--output", default="", help="输出 JSONL")
    parser.add_argument("--summary-output", default="", help="摘要输出 JSON")
    parser.add_argument("--min-score", type=float, default=None, help="最低预测概率；rank 默认 0，topology-errors 默认 0.95")
    parser.add_argument("--max-candidate-count", type=int, default=None, help="候选池上限；0 表示不限制；默认: 0")
    parser.add_argument(
        "--candidate-max-samples-per-chunk",
        type=int,
        default=20000,
        help="候选每批最多 ordered samples 数，默认: 20000",
    )
    parser.add_argument("--candidate-same-region-limit", type=int, default=10, help="每个站点最多加入同 region 候选数；-1 不限制，0 关闭，默认: 10")
    parser.add_argument("--candidate-same-domain-limit", type=int, default=10, help="每个站点最多加入同 dominant domain 候选数；-1 不限制，0 关闭，默认: 10")
    parser.add_argument("--candidate-topology-neighbor-limit", type=int, default=10, help="每个站点最多加入已知拓扑邻居候选数；-1 不限制，0 关闭，默认: 10")
    parser.add_argument("--candidate-nearest-limit", type=int, default=10, help="每个站点最多加入近距离候选数；0 关闭，默认: 10")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")

    rank_group = parser.add_argument_group("rank 模式")
    rank_group.add_argument("--include-none", action="store_true", help="rank 模式默认只输出非 none 预测；开启后也输出 none")

    error_group = parser.add_argument_group("topology-errors 模式")
    error_group.add_argument("--missing-min-score", type=float, default=-1.0, help="缺边阈值；<0 使用 --min-score")
    error_group.add_argument("--wrong-min-score", type=float, default=-1.0, help="错方向阈值；<0 使用 --min-score")
    error_group.add_argument("--extra-min-score", type=float, default=-1.0, help="多边阈值；<0 使用 --min-score")
    args = parser.parse_args()

    if args.min_score is None:
        args.min_score = 0.95 if args.mode == "topology-errors" else 0.0
    if args.max_candidate_count is None:
        args.max_candidate_count = 0

    output = _derive_output_path(args)
    summary_output = _derive_summary_output_path(args, output)

    print(f"加载模型: {args.model}")
    model_payload, feature_names, weights, biases = _load_model(args.model)
    print(f"加载站点关系上下文: {args.site_chains}")
    context = build_site_relation_context(args.site_chains, args.site_graph, args.site_device_counts)

    if args.mode == "rank":
        _predict_rank_rows_streaming(args, model_payload, feature_names, weights, biases, context, output, summary_output)
    else:
        _predict_topology_error_rows(args, model_payload, feature_names, weights, biases, context, output, summary_output)


if __name__ == "__main__":
    main()
