#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path
import sys
import unittest

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from ne_link_learning.core import (
    build_prediction_rows,
    evaluate_dense_samples,
    load_dataset_samples,
    load_json,
    vectorize_samples,
    write_json,
    write_jsonl,
)


def _derive_output_base(model_file, test_file):
    model_path = Path(model_file)
    model_dir = model_path.parent
    model_stem = model_path.stem
    test_stem = Path(test_file).stem
    return model_dir / f"{model_stem}.{test_stem}"


def _derive_eval_path(model_file, test_file):
    return str(_derive_output_base(model_file, test_file)) + ".eval.json"


def _derive_prediction_path(model_file, test_file):
    return str(_derive_output_base(model_file, test_file)) + ".predictions.jsonl"


class ModelEvaluationPathTests(unittest.TestCase):
    def test_output_paths_are_derived_from_model_and_test_stems(self):
        self.assertEqual(
            _derive_eval_path("/tmp/models/ne-link.json", "/data/holdout.jsonl"),
            "/tmp/models/ne-link.holdout.eval.json",
        )
        self.assertEqual(
            _derive_prediction_path("/tmp/models/ne-link.json", "/data/holdout.jsonl"),
            "/tmp/models/ne-link.holdout.predictions.jsonl",
        )

    def test_script_entrypoint_runs_cli_only_for_model_arguments(self):
        self.assertFalse(_should_run_cli(["test_model.py"]))
        self.assertFalse(_should_run_cli(["test_model.py", "-v"]))
        self.assertTrue(_should_run_cli(["test_model.py", "--model", "model.json"]))
        self.assertTrue(_should_run_cli(["test_model.py", "--model=model.json"]))


def main():
    parser = ArgumentParser(description="在测试集上评估 topology link 模型")
    parser.add_argument("--model", required=True, help="模型 JSON")
    parser.add_argument("--test", required=True, help="测试集 JSONL")
    parser.add_argument("--output", default="", help="评估指标输出 JSON")
    parser.add_argument("--predictions-output", default="", help="逐样本预测输出 JSONL")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="覆盖模型内置 threshold；<0 表示使用模型 threshold",
    )
    args = parser.parse_args()

    model_payload = load_json(args.model)
    feature_names = model_payload["feature_names"]
    standardizer = model_payload["standardizer"]
    weights = [model_payload["weights"].get(feature_name, 0.0) for feature_name in feature_names]
    bias = float(model_payload.get("bias", 0.0))
    threshold = args.threshold if args.threshold >= 0 else float(model_payload.get("threshold", 0.5))

    test_samples = load_dataset_samples(args.test)
    test_dense = vectorize_samples(test_samples, feature_names, standardizer)
    metrics, probabilities = evaluate_dense_samples(test_dense, weights, bias, threshold=threshold)
    prediction_rows = build_prediction_rows(test_dense, probabilities, threshold)

    output_file = args.output or _derive_eval_path(args.model, args.test)
    predictions_output = args.predictions_output or _derive_prediction_path(args.model, args.test)

    write_json(
        output_file,
        {
            "model": args.model,
            "test": args.test,
            "threshold": threshold,
            "metrics": metrics,
        },
    )
    write_jsonl(predictions_output, prediction_rows)

    print(
        f"test: precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"auc={metrics['roc_auc']:.4f}, "
        f"ap={metrics['average_precision']:.4f}"
    )
    print(f"评估结果已输出到: {output_file}")
    print(f"逐样本预测已输出到: {predictions_output}")


def _should_run_cli(argv):
    cli_options = {
        "--model",
        "--test",
        "--output",
        "--predictions-output",
        "--threshold",
        "--help",
        "-h",
    }
    return any(arg in cli_options or arg.split("=", 1)[0] in cli_options for arg in argv[1:])


if __name__ == "__main__":
    if _should_run_cli(sys.argv):
        main()
    else:
        unittest.main()
