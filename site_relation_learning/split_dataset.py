#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from site_relation_learning.core import (
    RELATION_CLASSES,
    SiteInfo,
    build_site_relation_context_from_relation_map,
    load_dataset_samples,
    load_site_infos,
    rebuild_site_relation_sample_features,
    summarize_samples,
    write_json,
)
from topology_resources import SITE_DEVICE_COUNTS_JSON, SITE_GRAPH_JSON, resource_display


def _derive_output_path(input_file, suffix):
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}.{suffix}.jsonl"))


def _format_label_counts(summary):
    counts = summary.get("label_counts", {})
    return ", ".join(
        f"{label}={counts.get(label, 0)}"
        for label in RELATION_CLASSES
    )


def _split_samples_by_group_with_progress(samples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    from site_relation_learning.core import stable_hash_fraction

    train_boundary = train_ratio
    valid_boundary = train_ratio + valid_ratio
    buckets = {"train": [], "valid": [], "test": []}
    progress = ProgressBar(len(samples), "切分样本")
    try:
        for index, sample in enumerate(samples, start=1):
            group_key = sample.get("unordered_site_pair_key") or sample.get("sample_id", "")
            value = stable_hash_fraction(group_key, seed=seed)
            if value < train_boundary:
                buckets["train"].append(sample)
            elif value < valid_boundary:
                buckets["valid"].append(sample)
            else:
                buckets["test"].append(sample)
            progress.set(index)
            if index % 1000 == 0 or index == len(samples):
                progress.set_extra_text(
                    f"train={len(buckets['train'])}, valid={len(buckets['valid'])}, test={len(buckets['test'])}"
                )
    finally:
        progress.close()
    return buckets


def _write_jsonl_with_progress(output_path, samples, label):
    progress = ProgressBar(len(samples), label)
    try:
        with open(output_path, "w", encoding="utf-8") as file_obj:
            for index, item in enumerate(samples, start=1):
                file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
                progress.set(index)
    finally:
        progress.close()


def _collect_train_positive_relation_map(samples):
    relation_map = {}
    for sample in samples:
        relation = str(sample.get("label", "none") or "none")
        if relation == "none" or relation not in RELATION_CLASSES:
            continue
        left_site_id = str(sample.get("u_site_id", "") or "")
        right_site_id = str(sample.get("v_site_id", "") or "")
        if not left_site_id or not right_site_id or left_site_id == right_site_id:
            continue
        relation_map[(left_site_id, right_site_id)] = relation
    return relation_map


def _ensure_sample_sites(site_infos, buckets):
    for samples in buckets.values():
        for sample in samples:
            for key in ("u_site_id", "v_site_id"):
                site_id = str(sample.get(key, "") or "")
                if not site_id or site_id in site_infos:
                    continue
                site_infos[site_id] = SiteInfo(
                    site_id=site_id,
                    site_name=site_id,
                    region_id="MISSING",
                    city_id="MISSING",
                    latitude=None,
                    longitude=None,
                    device_counts=Counter(),
                    device_total=0,
                    dominant_domain="MISSING",
                )


def _rebuild_features_with_train_context(buckets, site_graph_file, site_device_counts_file):
    from site_relation_learning.core import _set_relation_pair

    print(f"加载站点信息: {site_graph_file}")
    site_infos = load_site_infos(site_graph_file, site_device_counts_file)
    _ensure_sample_sites(site_infos, buckets)
    train_relation_map = {}
    for (left_site_id, right_site_id), relation in _collect_train_positive_relation_map(buckets["train"]).items():
        _set_relation_pair(train_relation_map, left_site_id, right_site_id, relation)
    print(f"train 正关系有序边数: {len(train_relation_map)}")
    print("基于 train 正关系重建站点关系上下文并重算特征...")
    train_context = build_site_relation_context_from_relation_map(site_infos, train_relation_map)
    for split_name in ("train", "valid", "test"):
        buckets[split_name] = rebuild_site_relation_sample_features(
            buckets[split_name],
            train_context,
            show_progress=True,
            progress_label=f"重算 {split_name} 特征",
        )
    return len(train_relation_map)


def main():
    parser = ArgumentParser(description="切分站点关系四分类数据集")
    parser.add_argument("dataset", help="build_relation_dataset.py 输出 JSONL")
    parser.add_argument("--train-output", default="", help="训练集输出")
    parser.add_argument("--valid-output", default="", help="验证集输出")
    parser.add_argument("--test-output", default="", help="测试集输出")
    parser.add_argument("--summary-output", default="", help="切分统计输出 JSON")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例，默认: 0.8")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="验证集比例，默认: 0.1")
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=f"strict inductive 重算特征时使用的 site_graph.json，默认: {resource_display('site_graph.json')}")
    parser.add_argument(
        "--site-device-counts",
        default=SITE_DEVICE_COUNTS_JSON,
        help=f"strict inductive 重算特征时使用的 site_device_counts.json，默认: {resource_display('site_device_counts.json')}",
    )
    parser.add_argument(
        "--strict-inductive",
        action="store_true",
        help="开启严格 inductive 评测: 只用 train 正关系重建拓扑上下文并重算所有 split 特征",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认: 42")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    samples = load_dataset_samples(args.dataset)
    print(f"样本数: {len(samples)}")
    buckets = _split_samples_by_group_with_progress(
        samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    train_positive_relation_count = 0
    if args.strict_inductive:
        train_positive_relation_count = _rebuild_features_with_train_context(
            buckets,
            args.site_graph,
            args.site_device_counts,
        )
    else:
        print("保留原始特征: 使用 build_relation_dataset.py 生成的 transductive / leave-one-pair-out 特征。")

    train_output = args.train_output or _derive_output_path(args.dataset, "train")
    valid_output = args.valid_output or _derive_output_path(args.dataset, "valid")
    test_output = args.test_output or _derive_output_path(args.dataset, "test")
    summary_output = args.summary_output or str(Path(args.dataset).with_suffix(".split.summary.json"))

    _write_jsonl_with_progress(train_output, buckets["train"], "写出 train")
    _write_jsonl_with_progress(valid_output, buckets["valid"], "写出 valid")
    _write_jsonl_with_progress(test_output, buckets["test"], "写出 test")
    write_json(
        summary_output,
        {
            "input": args.dataset,
            "seed": args.seed,
            "feature_rebuild": {
                "enabled": bool(args.strict_inductive),
                "topology_scope": "train_positive_only" if args.strict_inductive else "original_full_graph_leave_one_pair_out",
                "site_graph": args.site_graph if args.strict_inductive else "",
                "site_device_counts": args.site_device_counts if args.strict_inductive else "",
                "train_positive_relation_count": train_positive_relation_count,
                "strict_inductive": bool(args.strict_inductive),
            },
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
