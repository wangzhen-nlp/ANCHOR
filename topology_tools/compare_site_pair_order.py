#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较任意两个 site_pair_order_*.json（pairwise / global_no_path /
global_path_voting / global_path_optimized，含其 _ring 变体）的方向预测差异统计。

四个生成脚本输出同一外层结构：
    {"meta": {...}, "edges": [{site_a, site_b, prediction, upstream_site, downstream_site}, ...], "downstream_map": {...}}

注意：edges 里的 `prediction` 字符串方向约定在 pairwise 与 global 系列之间并不一致
（pairwise 写 upstream->downstream，global 写 downstream->upstream），因此本脚本
一律以 `upstream_site` / `downstream_site` 字段判定方向，跨算法可比。

每条无向边 {a,b}（按 site_id 排序得 (s0, s1)）归一化为三态关系标签：
    "s0_up"  —— s0 是上行端（更靠核心）
    "s1_up"  —— s1 是上行端
    "bidir"  —— 双向 / 无方向

统计内容：
1. 各文件的边数 / 有向边数 / 双向边数 / 有向占比；
2. 边集合对比：交集 / 仅左 / 仅右；
3. 共同边上的 3x3 混淆矩阵与一致率、Cohen's kappa；
4. 方向反转（一方 s0_up、另一方 s1_up）数量与样例；
5. 双向<->有向 的转换数量；
6. downstream_map（有向关系对）层面的 Jaccard 相似度。
"""

import argparse
import json

from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import resource_path
from topology_tools.site_pair_order_common import build_downstream_map


RELATION_LABELS = ("s0_up", "s1_up", "bidir")


def resolve_input_path(value):
    """优先按原样路径；不存在时再尝试 topology_resources/ 下同名文件。"""
    path = Path(value)
    if path.exists():
        return path
    fallback = Path(resource_path(value))
    if fallback.exists():
        return fallback
    return path


def load_prediction(value):
    path = resolve_input_path(value)
    if not path.exists():
        raise SystemExit(f"未找到预测文件: {value}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "edges" not in data:
        raise SystemExit(f"文件结构不含 edges，无法比较: {path}")
    return data, path


def edge_relation(edge):
    """把一条 edge 归一化为 (pair_key, relation_label)。

    pair_key = (s0, s1)，s0 < s1；relation 取 RELATION_LABELS 之一。
    无效边（缺端点）返回 None。
    """
    site_a = edge.get("site_a")
    site_b = edge.get("site_b")
    if not site_a or not site_b or site_a == site_b:
        return None

    s0, s1 = sorted((site_a, site_b))
    upstream = edge.get("upstream_site")
    downstream = edge.get("downstream_site")

    if edge.get("prediction") == "bidirectional" or not upstream or not downstream:
        relation = "bidir"
    elif upstream == s0:
        relation = "s0_up"
    elif upstream == s1:
        relation = "s1_up"
    else:
        # upstream_site 不在该边两端，视为数据异常，按双向处理
        relation = "bidir"
    return (s0, s1), relation


def build_relation_map(prediction):
    """edges -> {pair_key: relation}；重复边以最后一次为准并计数。"""
    relation_map = {}
    duplicate_count = 0
    invalid_count = 0
    for edge in prediction.get("edges", []):
        result = edge_relation(edge)
        if result is None:
            invalid_count += 1
            continue
        pair_key, relation = result
        if pair_key in relation_map:
            duplicate_count += 1
        relation_map[pair_key] = relation
    return relation_map, duplicate_count, invalid_count


def relation_counts(relation_map):
    counts = Counter(relation_map.values())
    directed = counts["s0_up"] + counts["s1_up"]
    bidirectional = counts["bidir"]
    total = directed + bidirectional
    return {
        "edge_count": total,
        "directed_count": directed,
        "bidirectional_count": bidirectional,
        "directed_ratio": directed / total if total else 0.0,
        "bidirectional_ratio": bidirectional / total if total else 0.0,
    }


def cohen_kappa(confusion, labels):
    """根据混淆矩阵 confusion[(left,right)] 计算 Cohen's kappa。"""
    total = sum(confusion.values())
    if total == 0:
        return 0.0
    observed = sum(confusion.get((label, label), 0) for label in labels) / total

    left_marginal = Counter()
    right_marginal = Counter()
    for (left_label, right_label), count in confusion.items():
        left_marginal[left_label] += count
        right_marginal[right_label] += count

    expected = sum(
        (left_marginal[label] / total) * (right_marginal[label] / total)
        for label in labels
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def directed_relation_pairs(prediction):
    """downstream_map 展开成有向关系对集合 {(upstream, downstream), ...}。

    downstream_map 语义：upstream -> [downstream...]；双向边会在两端互相登记，
    因此一条双向边贡献两个有序对，与有向边天然区分。
    """
    downstream_map = prediction.get("downstream_map")
    if not isinstance(downstream_map, dict):
        downstream_map = build_downstream_map(prediction)
    pairs = set()
    for upstream, downstream_sites in downstream_map.items():
        for downstream in downstream_sites or ():
            if upstream and downstream and upstream != downstream:
                pairs.add((upstream, downstream))
    return pairs


def compare(left_map, right_map):
    left_keys = set(left_map)
    right_keys = set(right_map)
    common = left_keys & right_keys

    confusion = Counter()
    agree = 0
    reversals = []
    bidir_to_directed = []
    directed_to_bidir = []

    for pair_key in common:
        left_relation = left_map[pair_key]
        right_relation = right_map[pair_key]
        confusion[(left_relation, right_relation)] += 1

        if left_relation == right_relation:
            agree += 1
            continue

        left_directed = left_relation != "bidir"
        right_directed = right_relation != "bidir"
        if left_directed and right_directed:
            # 两边都有向但方向相反
            reversals.append((pair_key, left_relation, right_relation))
        elif left_relation == "bidir" and right_directed:
            bidir_to_directed.append((pair_key, left_relation, right_relation))
        else:
            directed_to_bidir.append((pair_key, left_relation, right_relation))

    return {
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "common_count": len(common),
        "agree_count": agree,
        "agreement_rate": agree / len(common) if common else 0.0,
        "kappa": cohen_kappa(confusion, RELATION_LABELS),
        "confusion": confusion,
        "reversals": sorted(reversals),
        "bidir_to_directed": sorted(bidir_to_directed),
        "directed_to_bidir": sorted(directed_to_bidir),
    }


def format_pair(pair_key):
    return f"{pair_key[0]} -- {pair_key[1]}"


def relation_arrow(pair_key, relation):
    s0, s1 = pair_key
    if relation == "s0_up":
        return f"{s1} -> {s0}"  # downstream -> upstream
    if relation == "s1_up":
        return f"{s0} -> {s1}"
    return "<->"


def print_report(left_name, right_name, left_stats, right_stats, result, args):
    print("=" * 72)
    print(f"对比: [L] {left_name}")
    print(f"      [R] {right_name}")
    print("=" * 72)

    print("\n[各文件方向分布]")
    header = f"  {'指标':<16}{'L':>12}{'R':>12}"
    print(header)
    for label, key in (
        ("边数", "edge_count"),
        ("有向边", "directed_count"),
        ("双向边", "bidirectional_count"),
    ):
        print(f"  {label:<16}{left_stats[key]:>12}{right_stats[key]:>12}")
    print(
        f"  {'有向占比':<16}{left_stats['directed_ratio']:>11.2%}"
        f"{right_stats['directed_ratio']:>12.2%}"
    )

    print("\n[边集合对比]")
    print(f"  共同边:   {result['common_count']}")
    print(f"  仅 L 有:  {len(result['left_only'])}")
    print(f"  仅 R 有:  {len(result['right_only'])}")

    print("\n[共同边一致性]")
    print(f"  完全一致:        {result['agree_count']} / {result['common_count']}")
    print(f"  一致率:          {result['agreement_rate']:.2%}")
    print(f"  Cohen's kappa:   {result['kappa']:.4f}")
    print(f"  方向反转:        {len(result['reversals'])}")
    print(f"  L双向->R有向:    {len(result['bidir_to_directed'])}")
    print(f"  L有向->R双向:    {len(result['directed_to_bidir'])}")

    print("\n[3x3 混淆矩阵]  行=L  列=R  (s0_up/s1_up/bidir，s0<s1)")
    confusion = result["confusion"]
    col_header = "        " + "".join(f"{label:>10}" for label in RELATION_LABELS)
    print(col_header)
    for left_label in RELATION_LABELS:
        row = f"  {left_label:>6}" + "".join(
            f"{confusion.get((left_label, right_label), 0):>10}"
            for right_label in RELATION_LABELS
        )
        print(row)

    print("\n[downstream_map 有向关系对 Jaccard]")
    print(f"  L 关系对: {result['left_pairs_count']}")
    print(f"  R 关系对: {result['right_pairs_count']}")
    print(f"  交集:     {result['pairs_intersection']}")
    print(f"  并集:     {result['pairs_union']}")
    print(f"  Jaccard:  {result['pairs_jaccard']:.4f}")

    sample = args.sample
    if sample > 0:
        _print_sample("方向反转样例 (L / R)", result["reversals"], sample, both=True)
        _print_sample("仅 L 有的边样例", result["left_only"], sample, both=False)
        _print_sample("仅 R 有的边样例", result["right_only"], sample, both=False)


def _print_sample(title, items, sample, both):
    if not items:
        return
    print(f"\n[{title}]  (前 {min(sample, len(items))}/{len(items)})")
    for item in items[:sample]:
        if both:
            pair_key, left_relation, right_relation = item
            left_arrow = relation_arrow(pair_key, left_relation)
            right_arrow = relation_arrow(pair_key, right_relation)
            print(
                f"  {format_pair(pair_key):<28}"
                f"L: {left_arrow:<24} | R: {right_arrow}"
            )
        else:
            print(f"  {format_pair(item)}")


def build_json_report(left_name, right_name, left_stats, right_stats, result):
    return {
        "left": {"name": left_name, **left_stats},
        "right": {"name": right_name, **right_stats},
        "edge_set": {
            "common": result["common_count"],
            "left_only": len(result["left_only"]),
            "right_only": len(result["right_only"]),
        },
        "agreement": {
            "agree_count": result["agree_count"],
            "agreement_rate": round(result["agreement_rate"], 6),
            "kappa": round(result["kappa"], 6),
            "reversal_count": len(result["reversals"]),
            "bidir_to_directed_count": len(result["bidir_to_directed"]),
            "directed_to_bidir_count": len(result["directed_to_bidir"]),
        },
        "confusion_matrix": {
            f"{left_label}|{right_label}": result["confusion"].get(
                (left_label, right_label), 0
            )
            for left_label in RELATION_LABELS
            for right_label in RELATION_LABELS
        },
        "downstream_pairs": {
            "left": result["left_pairs_count"],
            "right": result["right_pairs_count"],
            "intersection": result["pairs_intersection"],
            "union": result["pairs_union"],
            "jaccard": round(result["pairs_jaccard"], 6),
        },
        "reversals": [
            {
                "site_a": pair_key[0],
                "site_b": pair_key[1],
                "left": relation_arrow(pair_key, left_relation),
                "right": relation_arrow(pair_key, right_relation),
            }
            for pair_key, left_relation, right_relation in result["reversals"]
        ],
        "left_only_edges": [list(pair_key) for pair_key in result["left_only"]],
        "right_only_edges": [list(pair_key) for pair_key in result["right_only"]],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较两个 site_pair_order_*.json 的方向预测差异统计"
    )
    parser.add_argument("left", help="左侧预测 JSON（路径或 topology_resources/ 下文件名）")
    parser.add_argument("right", help="右侧预测 JSON（路径或 topology_resources/ 下文件名）")
    parser.add_argument(
        "-o",
        "--output",
        help="可选：把完整差异报告写入 JSON 文件",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="人类可读报告中每类样例最多展示多少条；0 表示不展示，默认 10",
    )
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample 不能小于 0")
    return args


def main():
    args = parse_args()

    left_data, left_path = load_prediction(args.left)
    right_data, right_path = load_prediction(args.right)

    left_map, left_dup, left_invalid = build_relation_map(left_data)
    right_map, right_dup, right_invalid = build_relation_map(right_data)

    for name, dup, invalid in (
        (left_path, left_dup, left_invalid),
        (right_path, right_dup, right_invalid),
    ):
        if dup:
            print(f"提示: {name} 存在 {dup} 条重复无向边，已按最后一次为准")
        if invalid:
            print(f"提示: {name} 跳过 {invalid} 条缺端点的无效边")

    left_stats = relation_counts(left_map)
    right_stats = relation_counts(right_map)

    result = compare(left_map, right_map)

    left_pairs = directed_relation_pairs(left_data)
    right_pairs = directed_relation_pairs(right_data)
    intersection = left_pairs & right_pairs
    union = left_pairs | right_pairs
    result["left_pairs_count"] = len(left_pairs)
    result["right_pairs_count"] = len(right_pairs)
    result["pairs_intersection"] = len(intersection)
    result["pairs_union"] = len(union)
    result["pairs_jaccard"] = len(intersection) / len(union) if union else 0.0

    left_name = str(left_path)
    right_name = str(right_path)
    print_report(left_name, right_name, left_stats, right_stats, result, args)

    if args.output:
        report = build_json_report(left_name, right_name, left_stats, right_stats, result)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n完整报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
