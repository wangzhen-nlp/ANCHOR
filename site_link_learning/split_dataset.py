#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from ne_link_learning.core import read_jsonl, split_samples_by_group, summarize_samples, write_json


def _derive_split_path(input_file, split_name):
    if input_file.endswith(".jsonl"):
        return input_file[:-6] + f".{split_name}.jsonl"
    return input_file + f".{split_name}.jsonl"


def _derive_summary_path(input_file):
    if input_file.endswith(".jsonl"):
        return input_file[:-6] + ".split_summary.json"
    return input_file + ".split_summary.json"


def _write_jsonl_with_progress(filepath, items, label):
    print(f"⏳ {label}...")
    progress = ProgressBar(len(items), label)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for index, item in enumerate(items, start=1):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                progress.set(index)
    finally:
        progress.close()


def main():
    parser = ArgumentParser(description="按 group key 将 topology link 样本拆分为 train/valid/test")
    parser.add_argument("dataset", help="输入样本 JSONL")
    parser.add_argument("--train-output", default="", help="训练集输出 JSONL")
    parser.add_argument("--valid-output", default="", help="验证集输出 JSONL")
    parser.add_argument("--test-output", default="", help="测试集输出 JSONL")
    parser.add_argument(
        "--summary-output",
        default="",
        help="拆分统计 JSON；默认与输入同名前缀",
    )
    parser.add_argument(
        "--group-field",
        default="unordered_site_pair_key",
        choices=["unordered_site_pair_key", "ordered_site_pair_key"],
        help="按哪个 key 分组拆分，默认: unordered_site_pair_key",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例，默认: 0.8")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="验证集比例，默认: 0.1")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例，默认: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    args = parser.parse_args()

    print(f"加载样本文件: {args.dataset}")
    samples = read_jsonl(args.dataset)
    print("按 group key 拆分样本...")
    split_buckets = split_samples_by_group(
        samples=samples,
        group_field=args.group_field,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        show_progress=True,
        progress_label="拆分样本",
    )

    train_output = args.train_output or _derive_split_path(args.dataset, "train")
    valid_output = args.valid_output or _derive_split_path(args.dataset, "valid")
    test_output = args.test_output or _derive_split_path(args.dataset, "test")
    summary_output = args.summary_output or _derive_summary_path(args.dataset)

    _write_jsonl_with_progress(train_output, split_buckets["train"], "写出训练集")
    _write_jsonl_with_progress(valid_output, split_buckets["valid"], "写出验证集")
    _write_jsonl_with_progress(test_output, split_buckets["test"], "写出测试集")

    summary = {
        "dataset": args.dataset,
        "group_field": args.group_field,
        "seed": args.seed,
        "splits": {
            "train": summarize_samples(split_buckets["train"]),
            "valid": summarize_samples(split_buckets["valid"]),
            "test": summarize_samples(split_buckets["test"]),
        },
        "outputs": {
            "train": train_output,
            "valid": valid_output,
            "test": test_output,
        },
    }
    print(f"写出拆分统计: {summary_output}")
    write_json(summary_output, summary)

    for split_name in ("train", "valid", "test"):
        split_summary = summary["splits"][split_name]
        print(
            f"{split_name}: 样本 {split_summary['sample_count']} "
            f"(正 {split_summary['positive_count']}, 负 {split_summary['negative_count']})"
        )
    print(f"拆分统计已输出到: {summary_output}")


if __name__ == "__main__":
    main()
