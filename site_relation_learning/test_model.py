#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from site_relation_learning.core import (
    RELATION_CLASSES,
    build_prediction_rows,
    build_pair_level_prediction_rows,
    evaluate_dense_samples,
    evaluate_pair_level_prediction_rows,
    load_dataset_samples,
    load_json,
    safe_ratio,
    vectorize_samples,
    write_json,
    write_jsonl,
)


# 合并 downstream / upstream 为一个"有向边"类别，因为 pair-level 已按
# canonical_pair 字典序去重，方向只取决于站点 ID 排序，分开统计意义不大。
MERGED_RELATION_CLASSES = ("downstream/upstream", "bidirection", "none")


def _merge_relation(label):
    if label in ("downstream", "upstream"):
        return "downstream/upstream"
    if label in MERGED_RELATION_CLASSES:
        return label
    return "none"


def _evaluate_pair_rows_merged(rows):
    """与 evaluate_pair_level_prediction_rows 同形态，但用合并后的 3 类。"""
    classes = MERGED_RELATION_CLASSES
    class_count = len(classes)
    confusion = [[0 for _ in classes] for _ in classes]
    for row in rows:
        gold = _merge_relation(row.get("gold_relation", "none"))
        pred = _merge_relation(row.get("predicted_relation", "none"))
        confusion[classes.index(gold)][classes.index(pred)] += 1

    total = sum(sum(r) for r in confusion)
    correct = sum(confusion[idx][idx] for idx in range(class_count))
    per_class = {}
    macro_f1 = 0.0
    macro_class_count = 0
    for idx, label in enumerate(classes):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(class_count) if row != idx)
        fn = sum(confusion[idx][col] for col in range(class_count) if col != idx)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        support = sum(confusion[idx])
        if support > 0:
            macro_f1 += f1
            macro_class_count += 1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = safe_ratio(macro_f1, macro_class_count)
    return {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "classes": list(classes),
        "per_class": per_class,
        "pair_count": total,
    }


def _domain_pair_key(row):
    left_domain = str(row.get("site_a_domain") or "MISSING")
    right_domain = str(row.get("site_b_domain") or "MISSING")
    return f"{left_domain}__{right_domain}"


def _evaluate_pair_rows_by_domain_pair(pair_rows):
    buckets = {}
    for row in pair_rows:
        buckets.setdefault(_domain_pair_key(row), []).append(row)
    return {
        key: evaluate_pair_level_prediction_rows(rows)
        for key, rows in sorted(buckets.items())
    }


def _evaluate_pair_rows_by_relation(pair_rows):
    buckets = {label: [] for label in MERGED_RELATION_CLASSES}
    for row in pair_rows:
        buckets[_merge_relation(row.get("gold_relation", "none"))].append(row)
    return {
        label: _evaluate_pair_rows_merged(rows)
        for label, rows in buckets.items()
    }


def _print_domain_pair_metrics(metrics_by_domain_pair):
    if not metrics_by_domain_pair:
        return
    print("pair-level 按 dominant domain pair 分桶指标:")
    for key, metrics in sorted(
        metrics_by_domain_pair.items(),
        key=lambda item: (-item[1].get("pair_count", 0), item[0]),
    ):
        print(
            f"  {key}: pair_count={metrics['pair_count']}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )


def _print_relation_metrics(metrics_by_relation):
    if not metrics_by_relation:
        return
    print("pair-level 按 gold relation（合并 downstream/upstream 后）分桶指标:")
    for label in MERGED_RELATION_CLASSES:
        metrics = metrics_by_relation.get(label)
        if not metrics or metrics.get("pair_count", 0) == 0:
            print(f"  {label}: pair_count=0 (无样本)")
            continue
        per_class = metrics.get("per_class", {}).get(label, {})
        precision = per_class.get("precision", 0.0)
        recall = per_class.get("recall", 0.0)
        f1 = per_class.get("f1", 0.0)
        print(
            f"  {label}: pair_count={metrics['pair_count']}, "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )


def _derive_output_base(model_file, test_file):
    model_path = Path(model_file)
    return model_path.parent / f"{model_path.stem}.{Path(test_file).stem}"


def _load_model(model_file):
    payload = load_json(model_file)
    classes = tuple(payload.get("classes") or RELATION_CLASSES)
    if tuple(classes) != RELATION_CLASSES:
        raise ValueError(f"模型类别不兼容: {classes}")
    feature_names = payload["feature_names"]
    model_type = payload.get("model_type", "site_relation_softmax_v1")
    if model_type == "site_relation_gbdt_v1":
        import lightgbm as lgb

        weights = {
            "model_type": "gbdt",
            "model_string": payload["gbdt_model_string"],
            "booster": lgb.Booster(model_str=payload["gbdt_model_string"]),
            "best_iteration": payload.get("gbdt_best_iteration", 0),
        }
        return payload, feature_names, weights, None
    if model_type == "site_relation_mlp_v1":
        weights = {
            "model_type": "mlp",
            "hidden_weights": payload["hidden_weights"],
            "hidden_biases": payload["hidden_biases"],
            "output_weights": payload["output_weights"],
            "output_biases": payload["output_biases"],
        }
        return payload, feature_names, weights, None
    weights = [
        [payload["weights"].get(label, {}).get(feature_name, 0.0) for feature_name in feature_names]
        for label in RELATION_CLASSES
    ]
    biases = [payload.get("biases", {}).get(label, 0.0) for label in RELATION_CLASSES]
    return payload, feature_names, weights, biases


def main():
    parser = ArgumentParser(description="评估站点关系四分类模型")
    parser.add_argument("--model", required=True, help="模型 JSON")
    parser.add_argument("--test", required=True, help="测试集 JSONL")
    parser.add_argument("--output", default="", help="评估指标输出 JSON")
    parser.add_argument("--predictions-output", default="", help="逐样本预测输出 JSONL")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    args = parser.parse_args()

    model_payload, feature_names, weights, biases = _load_model(args.model)
    test_samples = load_dataset_samples(args.test)
    dense = vectorize_samples(
        test_samples,
        feature_names,
        model_payload["standardizer"],
        show_progress=not args.no_progress,
        progress_label="向量化测试样本",
    )
    metrics, probabilities = evaluate_dense_samples(dense, weights, biases)
    prediction_rows = build_prediction_rows(dense, probabilities)
    pair_prediction_rows = build_pair_level_prediction_rows(dense, probabilities)
    pair_metrics = evaluate_pair_level_prediction_rows(pair_prediction_rows)
    pair_metrics_by_domain_pair = _evaluate_pair_rows_by_domain_pair(pair_prediction_rows)
    pair_metrics_by_relation = _evaluate_pair_rows_by_relation(pair_prediction_rows)

    output_base = _derive_output_base(args.model, args.test)
    output_file = args.output or str(output_base) + ".eval.json"
    predictions_output = args.predictions_output or str(output_base) + ".predictions.jsonl"
    pair_predictions_output = str(output_base) + ".pair_predictions.jsonl"
    write_json(
        output_file,
        {
            "model": args.model,
            "test": args.test,
            "metrics": metrics,
            "pair_level_metrics": pair_metrics,
            "pair_level_metrics_by_dominant_domain_pair": pair_metrics_by_domain_pair,
            "pair_level_metrics_by_gold_relation": pair_metrics_by_relation,
        },
    )
    write_jsonl(predictions_output, prediction_rows)
    write_jsonl(pair_predictions_output, pair_prediction_rows)

    print(f"test: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(
        f"pair-level test: accuracy={pair_metrics['accuracy']:.4f}, "
        f"macro_f1={pair_metrics['macro_f1']:.4f}, pair_count={pair_metrics['pair_count']}"
    )
    _print_relation_metrics(pair_metrics_by_relation)
    _print_domain_pair_metrics(pair_metrics_by_domain_pair)
    print(f"评估结果已输出到: {output_file}")
    print(f"逐样本预测已输出到: {predictions_output}")
    print(f"pair-level预测已输出到: {pair_predictions_output}")


if __name__ == "__main__":
    main()
