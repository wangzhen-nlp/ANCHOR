#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from site_relation_learning.core import (
    RELATION_CLASSES,
    load_dataset_samples,
    split_samples_by_group,
    summarize_samples,
    write_json,
    write_jsonl,
)


def _derive_output_path(input_file, suffix):
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}.{suffix}.jsonl"))


def _format_label_counts(summary):
    counts = summary.get("label_counts", {})
    return ", ".join(
        f"{label}={counts.get(label, 0)}"
        for label in RELATION_CLASSES
    )


def main():
    parser = ArgumentParser(description="切分站点关系四分类数据集")
    parser.add_argument("dataset", help="build_relation_dataset.py 输出 JSONL")
    parser.add_argument("--train-output", default="", help="训练集输出")
    parser.add_argument("--valid-output", default="", help="验证集输出")
    parser.add_argument("--test-output", default="", help="测试集输出")
    parser.add_argument("--summary-output", default="", help="切分统计输出 JSON")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例，默认: 0.8")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="验证集比例，默认: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    args = parser.parse_args()

    samples = load_dataset_samples(args.dataset)
    buckets = split_samples_by_group(
        samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    train_output = args.train_output or _derive_output_path(args.dataset, "train")
    valid_output = args.valid_output or _derive_output_path(args.dataset, "valid")
    test_output = args.test_output or _derive_output_path(args.dataset, "test")
    summary_output = args.summary_output or str(Path(args.dataset).with_suffix(".split.summary.json"))

    write_jsonl(train_output, buckets["train"])
    write_jsonl(valid_output, buckets["valid"])
    write_jsonl(test_output, buckets["test"])
    write_json(
        summary_output,
        {
            "input": args.dataset,
            "seed": args.seed,
            "train": summarize_samples(buckets["train"]),
            "valid": summarize_samples(buckets["valid"]),
            "test": summarize_samples(buckets["test"]),
        },
    )

    train_summary = summarize_samples(buckets["train"])
    valid_summary = summarize_samples(buckets["valid"])
    test_summary = summarize_samples(buckets["test"])

    print(f"train: {len(buckets['train'])} -> {train_output}")
    print(f"  类别分布: {_format_label_counts(train_summary)}")
    print(f"valid: {len(buckets['valid'])} -> {valid_output}")
    print(f"  类别分布: {_format_label_counts(valid_summary)}")
    print(f"test: {len(buckets['test'])} -> {test_output}")
    print(f"  类别分布: {_format_label_counts(test_summary)}")
    print(f"统计已输出到: {summary_output}")


if __name__ == "__main__":
    main()
