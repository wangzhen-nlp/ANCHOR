#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 match_rules.py 输出的故障组站点数分布。

用法:
    python fault_grouping/analyze_group_site_distribution.py \
        fault_groups.jsonl \
        --output group_site_dist.json
"""

import argparse
import json
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _add_site_id(site_ids, raw_site_id):
    site_id = str(raw_site_id or "").strip()
    if site_id:
        site_ids.add(site_id)


def _add_site_ids(site_ids, raw_site_ids):
    if isinstance(raw_site_ids, (list, tuple, set)):
        for site_id in raw_site_ids:
            _add_site_id(site_ids, site_id)


def extract_site_ids(record):
    """
    提取一个故障组覆盖的站点集合。

    优先使用 match_rules.py 当前输出的 group_info[*].site_list；
    若缺失，则兼容 compact/旧格式中的 role_mapping、symptoms、ne_info。
    """
    site_ids = set()

    group_info = record.get("group_info") or {}
    if isinstance(group_info, dict):
        for gmeta in group_info.values():
            if isinstance(gmeta, dict):
                _add_site_ids(site_ids, gmeta.get("site_list"))

    _add_site_ids(site_ids, record.get("site_list"))

    role_mapping_sources = [
        record.get("role_mapping"),
        (record.get("match_info") or {}).get("role_mapping"),
    ]
    for role_mapping in role_mapping_sources:
        if not isinstance(role_mapping, dict):
            continue
        for role_sites in role_mapping.values():
            _add_site_ids(site_ids, role_sites)

    symptom_sources = [
        record.get("symptoms"),
        (record.get("match_info") or {}).get("symptoms"),
    ]
    for symptoms in symptom_sources:
        if not isinstance(symptoms, list):
            continue
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            _add_site_id(site_ids, symptom.get("node") or symptom.get("site_id"))

    ne_info = record.get("ne_info") or {}
    if isinstance(ne_info, dict):
        for ne_meta in ne_info.values():
            if isinstance(ne_meta, dict):
                _add_site_id(site_ids, ne_meta.get("site_id"))

    return site_ids


def load_groups(jsonl_path):
    """逐行加载 JSONL 故障组，yield 每个 group 的 (uuid, rule, site_count)。"""
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {jsonl_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ 跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue

            match_info = record.get("match_info") or {}
            uuid = match_info.get("uuid", "")
            rule = match_info.get("rule", "")
            if not uuid and record.get("uuid"):
                uuid = record["uuid"]
            if not rule and record.get("rule"):
                rule = record["rule"]

            yield uuid, rule, len(extract_site_ids(record))


def compute_percentiles(values, percentiles=(50, 90, 95, 99, 99.9)):
    """计算百分位数。"""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        idx = int((p / 100.0) * (n - 1))
        idx = max(0, min(idx, n - 1))
        result[f"p{p}"] = sorted_vals[idx]
    return result


def build_histogram(site_counts, max_explicit=20):
    """
    构建站点数直方图。
    - 1~max_explicit: 每值一档
    - >max_explicit: 合并为 ">{max_explicit}"
    """
    hist = Counter()
    for c in site_counts:
        if c <= max_explicit:
            hist[c] += 1
        else:
            hist[f">{max_explicit}"] += 1
    return dict(sorted(hist.items(), key=lambda x: (isinstance(x[0], str), x[0])))


def analyze(jsonl_path, max_explicit=20):
    uuid_list = []
    rule_list = []
    site_counts = []
    rule_to_counts = defaultdict(list)

    for uuid, rule, site_count in load_groups(jsonl_path):
        uuid_list.append(uuid)
        rule_list.append(rule)
        site_counts.append(site_count)
        rule_to_counts[rule].append(site_count)

    total = len(site_counts)
    if total == 0:
        raise SystemExit("未解析到任何故障组")

    overall_counter = Counter(site_counts)
    most_common = overall_counter.most_common(5)

    # 按规则分组统计
    rule_stats = {}
    for rule, counts in sorted(rule_to_counts.items()):
        rule_stats[rule] = {
            "count": len(counts),
            "mean": round(sum(counts) / len(counts), 2),
            "min": min(counts),
            "max": max(counts),
            "median": statistics.median(counts),
            "histogram": build_histogram(counts, max_explicit),
        }

    result = {
        "meta": {
            "source_file": str(jsonl_path),
            "total_groups": total,
        },
        "overall": {
            "mean": round(sum(site_counts) / total, 2),
            "min": min(site_counts),
            "max": max(site_counts),
            "median": statistics.median(site_counts),
            "percentiles": compute_percentiles(site_counts),
            "histogram": build_histogram(site_counts, max_explicit),
            "top_modes": [
                {"site_count": sc, "group_count": cnt, "ratio": round(cnt / total, 4)}
                for sc, cnt in most_common
            ],
        },
        "by_rule": rule_stats,
        "detail": [
            {"uuid": u, "rule": r, "site_count": c}
            for u, r, c in zip(uuid_list, rule_list, site_counts)
        ],
    }
    return result


def print_summary(result):
    """在终端打印可读摘要。"""
    o = result["overall"]
    print(f"\n故障组站点数分布摘要")
    print(f"{'=' * 50}")
    print(f"故障组总数: {result['meta']['total_groups']}")
    print(f"平均站点数: {o['mean']}")
    print(f"中位数: {o['median']}")
    print(f"最小: {o['min']}, 最大: {o['max']}")
    print(f"\n百分位数:")
    for k, v in o["percentiles"].items():
        print(f"  {k}: {v}")
    print(f"\n最常见站点数 Top-5:")
    for item in o["top_modes"]:
        print(f"  {item['site_count']:>3} 个站点: {item['group_count']:>5} 组 ({item['ratio']*100:.2f}%)")
    print(f"\n按规则统计:")
    for rule, stats in result["by_rule"].items():
        print(f"  {rule:<40s}: 共 {stats['count']:>5} 组, 平均 {stats['mean']:.2f}, 中位 {stats['median']}, 最大 {stats['max']}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="分析故障组站点数分布")
    parser.add_argument("jsonl", help="match_rules.py 输出的 JSONL 文件")
    parser.add_argument("-o", "--output", default="", help="输出 JSON 文件路径；为空则只打印终端摘要")
    parser.add_argument("--max-explicit", type=int, default=20, help="直方图显式展示的最大站点数，超过合并；默认 20")
    args = parser.parse_args()

    result = analyze(args.jsonl, max_explicit=args.max_explicit)
    print_summary(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已保存详细结果: {args.output}")


if __name__ == "__main__":
    main()
