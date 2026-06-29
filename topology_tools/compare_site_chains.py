#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较任意两个 site_chains 结果的差异统计。

兼容两种输入结构：
1. generate_site_chains.py 的产物：顶层即 {"meta": {...}, "sites": {site_id: {...}}}；
2. build_resource_buffer.py 的产物：站点链路位于 buffer["site_chains"] 字段下，
   其内部仍是 {"meta": {...}, "sites": {...}}。

每个站点条目结构：
    {
        "bidirectional_sites": [site_id, ...],          # 直接双向邻居（无向）
        "downstream_site_hops": {site_id: hop, ...},     # 下游可达站点 -> 最短跳数
        "upstream_site_hops":   {site_id: hop, ...},     # 上游可达站点 -> 最短跳数
    }

统计内容：
1. 各文件的站点数 / 双向边数 / 下游可达关系数 / 上游可达关系数；
2. 站点集合对比：交集 / 仅左 / 仅右；
3. 双向边集合（无向）对比：交集 / 仅左 / 仅右 / Jaccard；
4. 下游可达关系（有向 src->dst）对比：交集 / 仅左 / 仅右 / Jaccard；
5. 共同下游关系上的跳数差异：同跳 / 异跳数量、跳数差分布与样例；
6. 方向反转：左 src->dst 而右 dst->src 的关系对数量与样例。

下游与上游关系互为对偶（A 的下游含 B@h <=> B 的上游含 A@h），因此方向层面的
集合 / 跳数 / 反转统计统一以下游关系为准，避免重复口径。
"""

import argparse
import json

from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import resource_path


def resolve_input_path(value):
    """优先按原样路径；不存在时再尝试 topology_resources/ 下同名文件。"""
    path = Path(value)
    if path.exists():
        return path
    fallback = Path(resource_path(value))
    if fallback.exists():
        return fallback
    return path


def extract_site_chains(data, path):
    """从原始 JSON 中取出 {"meta", "sites"} 站点链路对象，兼容 buffer 嵌套结构。"""
    if not isinstance(data, dict):
        raise SystemExit(f"文件结构不是对象，无法比较: {path}")

    if isinstance(data.get("sites"), dict):
        return data

    nested = data.get("site_chains")
    if isinstance(nested, dict) and isinstance(nested.get("sites"), dict):
        return nested

    raise SystemExit(
        f"文件结构既不含顶层 sites，也不含 site_chains.sites，无法比较: {path}"
    )


def load_site_chains(value):
    path = resolve_input_path(value)
    if not path.exists():
        raise SystemExit(f"未找到 site_chains 文件: {value}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return extract_site_chains(data, path), path


def normalize_sites(site_chains):
    """把 sites 归一化为 {site_id: {bidirectional:set, downstream:dict, upstream:dict}}。"""
    normalized = {}
    for site_id, info in site_chains.get("sites", {}).items():
        if not isinstance(info, dict):
            continue
        bidirectional = {
            str(other)
            for other in info.get("bidirectional_sites", []) or []
            if other and other != site_id
        }
        downstream = {
            str(other): hop
            for other, hop in (info.get("downstream_site_hops", {}) or {}).items()
            if other and other != site_id
        }
        upstream = {
            str(other): hop
            for other, hop in (info.get("upstream_site_hops", {}) or {}).items()
            if other and other != site_id
        }
        normalized[str(site_id)] = {
            "bidirectional": bidirectional,
            "downstream": downstream,
            "upstream": upstream,
        }
    return normalized


def bidirectional_edges(normalized):
    """归一化为无向边集合 {(s0, s1), ...}，s0 < s1。"""
    edges = set()
    for site_id, info in normalized.items():
        for other in info["bidirectional"]:
            edges.add(tuple(sorted((site_id, other))))
    return edges


def downstream_pairs_with_hops(normalized):
    """下游可达关系展开为 {(src, dst): hop}。"""
    pairs = {}
    for site_id, info in normalized.items():
        for dst, hop in info["downstream"].items():
            pairs[(site_id, dst)] = hop
    return pairs


def chain_stats(normalized):
    downstream_pairs = downstream_pairs_with_hops(normalized)
    upstream_count = sum(len(info["upstream"]) for info in normalized.values())
    return {
        "site_count": len(normalized),
        "bidirectional_edge_count": len(bidirectional_edges(normalized)),
        "downstream_relation_count": len(downstream_pairs),
        "upstream_relation_count": upstream_count,
    }


def jaccard(left_set, right_set):
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def compare_sets(left_set, right_set):
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection": len(left_set & right_set),
        "left_only": sorted(left_set - right_set),
        "right_only": sorted(right_set - left_set),
        "jaccard": jaccard(left_set, right_set),
    }


def compare_downstream(left_pairs, right_pairs):
    """对比下游有向关系：集合差、共同关系跳数差、方向反转。"""
    left_keys = set(left_pairs)
    right_keys = set(right_pairs)
    common = left_keys & right_keys

    hop_same = 0
    hop_diff = []          # (src, dst, left_hop, right_hop)
    hop_delta_counter = Counter()
    for key in common:
        left_hop = left_pairs[key]
        right_hop = right_pairs[key]
        if left_hop == right_hop:
            hop_same += 1
        else:
            hop_diff.append((key[0], key[1], left_hop, right_hop))
            try:
                hop_delta_counter[int(right_hop) - int(left_hop)] += 1
            except (TypeError, ValueError):
                hop_delta_counter["non_numeric"] += 1

    # 方向反转：左 src->dst，右 dst->src（统计无序对，避免左右各记一次）
    reversals = set()
    for src, dst in left_keys:
        if (dst, src) in right_keys:
            reversals.add(tuple(sorted((src, dst))))

    return {
        "set": compare_sets(left_keys, right_keys),
        "common_count": len(common),
        "hop_same_count": hop_same,
        "hop_diff": sorted(hop_diff),
        "hop_delta_distribution": hop_delta_counter,
        "reversals": sorted(reversals),
    }


def compare_per_site(left_norm, right_norm):
    """逐站点对比三类集合，返回有差异的站点明细。"""
    common_sites = sorted(set(left_norm) & set(right_norm))
    diffs = []
    bidir_diff_count = 0
    downstream_diff_count = 0
    upstream_diff_count = 0

    for site_id in common_sites:
        left_info = left_norm[site_id]
        right_info = right_norm[site_id]

        bidir_changed = left_info["bidirectional"] != right_info["bidirectional"]
        downstream_changed = set(left_info["downstream"]) != set(right_info["downstream"])
        upstream_changed = set(left_info["upstream"]) != set(right_info["upstream"])

        if bidir_changed:
            bidir_diff_count += 1
        if downstream_changed:
            downstream_diff_count += 1
        if upstream_changed:
            upstream_diff_count += 1

        if bidir_changed or downstream_changed or upstream_changed:
            diffs.append({
                "site_id": site_id,
                "bidirectional_added": sorted(right_info["bidirectional"] - left_info["bidirectional"]),
                "bidirectional_removed": sorted(left_info["bidirectional"] - right_info["bidirectional"]),
                "downstream_added": sorted(set(right_info["downstream"]) - set(left_info["downstream"])),
                "downstream_removed": sorted(set(left_info["downstream"]) - set(right_info["downstream"])),
                "upstream_added": sorted(set(right_info["upstream"]) - set(left_info["upstream"])),
                "upstream_removed": sorted(set(left_info["upstream"]) - set(right_info["upstream"])),
            })

    return {
        "common_site_count": len(common_sites),
        "changed_site_count": len(diffs),
        "bidir_diff_count": bidir_diff_count,
        "downstream_diff_count": downstream_diff_count,
        "upstream_diff_count": upstream_diff_count,
        "diffs": diffs,
    }


def format_pair(pair_key):
    return f"{pair_key[0]} -- {pair_key[1]}"


def print_report(left_name, right_name, left_stats, right_stats,
                 site_set, bidir, downstream, per_site, args):
    print("=" * 72)
    print(f"对比: [L] {left_name}")
    print(f"      [R] {right_name}")
    print("=" * 72)

    print("\n[各文件规模]")
    print(f"  {'指标':<18}{'L':>12}{'R':>12}")
    for label, key in (
        ("站点数", "site_count"),
        ("双向边数", "bidirectional_edge_count"),
        ("下游关系数", "downstream_relation_count"),
        ("上游关系数", "upstream_relation_count"),
    ):
        print(f"  {label:<18}{left_stats[key]:>12}{right_stats[key]:>12}")

    print("\n[站点集合对比]")
    print(f"  共同站点: {site_set['intersection']}")
    print(f"  仅 L 有:  {len(site_set['left_only'])}")
    print(f"  仅 R 有:  {len(site_set['right_only'])}")
    print(f"  Jaccard:  {site_set['jaccard']:.4f}")

    print("\n[双向边集合对比(无向)]")
    print(f"  共同:     {bidir['intersection']}")
    print(f"  仅 L 有:  {len(bidir['left_only'])}")
    print(f"  仅 R 有:  {len(bidir['right_only'])}")
    print(f"  Jaccard:  {bidir['jaccard']:.4f}")

    ds_set = downstream["set"]
    print("\n[下游可达关系对比(有向 src->dst)]")
    print(f"  L 关系数: {ds_set['left_count']}")
    print(f"  R 关系数: {ds_set['right_count']}")
    print(f"  共同:     {ds_set['intersection']}")
    print(f"  仅 L 有:  {len(ds_set['left_only'])}")
    print(f"  仅 R 有:  {len(ds_set['right_only'])}")
    print(f"  Jaccard:  {ds_set['jaccard']:.4f}")
    print(f"  方向反转: {len(downstream['reversals'])}")

    print("\n[共同下游关系跳数一致性]")
    print(f"  同跳:     {downstream['hop_same_count']} / {downstream['common_count']}")
    print(f"  异跳:     {len(downstream['hop_diff'])}")
    if downstream["hop_delta_distribution"]:
        deltas = sorted(
            downstream["hop_delta_distribution"].items(),
            key=lambda kv: (isinstance(kv[0], str), kv[0]),
        )
        rendered = ", ".join(f"{delta:+}: {count}" if isinstance(delta, int) else f"{delta}: {count}"
                             for delta, count in deltas)
        print(f"  跳数差(R-L)分布: {rendered}")

    print("\n[逐站点差异]")
    print(f"  共同站点:        {per_site['common_site_count']}")
    print(f"  有差异站点:      {per_site['changed_site_count']}")
    print(f"  双向集合变化:    {per_site['bidir_diff_count']}")
    print(f"  下游集合变化:    {per_site['downstream_diff_count']}")
    print(f"  上游集合变化:    {per_site['upstream_diff_count']}")

    sample = args.sample
    if sample > 0:
        _print_pair_sample("仅 L 有的下游关系样例", ds_set["left_only"], sample)
        _print_pair_sample("仅 R 有的下游关系样例", ds_set["right_only"], sample)
        _print_hop_sample("跳数变化样例 (L跳 / R跳)", downstream["hop_diff"], sample)
        _print_reversal_sample("方向反转样例", downstream["reversals"], sample)
        _print_site_diff_sample("逐站点差异样例", per_site["diffs"], sample)


def _print_pair_sample(title, items, sample):
    if not items:
        return
    print(f"\n[{title}]  (前 {min(sample, len(items))}/{len(items)})")
    for src, dst in items[:sample]:
        print(f"  {src} -> {dst}")


def _print_hop_sample(title, items, sample):
    if not items:
        return
    print(f"\n[{title}]  (前 {min(sample, len(items))}/{len(items)})")
    for src, dst, left_hop, right_hop in items[:sample]:
        print(f"  {src} -> {dst:<20} L:{left_hop}  R:{right_hop}")


def _print_reversal_sample(title, items, sample):
    if not items:
        return
    print(f"\n[{title}]  (前 {min(sample, len(items))}/{len(items)})")
    for pair_key in items[:sample]:
        print(f"  {format_pair(pair_key)}")


def _print_site_diff_sample(title, items, sample):
    if not items:
        return
    print(f"\n[{title}]  (前 {min(sample, len(items))}/{len(items)})")
    for diff in items[:sample]:
        parts = []
        for label, key in (
            ("双向+", "bidirectional_added"), ("双向-", "bidirectional_removed"),
            ("下游+", "downstream_added"), ("下游-", "downstream_removed"),
            ("上游+", "upstream_added"), ("上游-", "upstream_removed"),
        ):
            if diff[key]:
                parts.append(f"{label}{len(diff[key])}")
        print(f"  {diff['site_id']:<20} {' '.join(parts)}")


def build_json_report(left_name, right_name, left_stats, right_stats,
                      site_set, bidir, downstream, per_site):
    ds_set = downstream["set"]
    return {
        "left": {"name": left_name, **left_stats},
        "right": {"name": right_name, **right_stats},
        "site_set": {
            "common": site_set["intersection"],
            "left_only": len(site_set["left_only"]),
            "right_only": len(site_set["right_only"]),
            "jaccard": round(site_set["jaccard"], 6),
            "left_only_sites": site_set["left_only"],
            "right_only_sites": site_set["right_only"],
        },
        "bidirectional_edges": {
            "common": bidir["intersection"],
            "left_only": len(bidir["left_only"]),
            "right_only": len(bidir["right_only"]),
            "jaccard": round(bidir["jaccard"], 6),
            "left_only_edges": [list(pair) for pair in bidir["left_only"]],
            "right_only_edges": [list(pair) for pair in bidir["right_only"]],
        },
        "downstream_relations": {
            "left_count": ds_set["left_count"],
            "right_count": ds_set["right_count"],
            "common": ds_set["intersection"],
            "left_only": len(ds_set["left_only"]),
            "right_only": len(ds_set["right_only"]),
            "jaccard": round(ds_set["jaccard"], 6),
            "hop_same_count": downstream["hop_same_count"],
            "hop_diff_count": len(downstream["hop_diff"]),
            "reversal_count": len(downstream["reversals"]),
            "hop_delta_distribution": {
                str(delta): count
                for delta, count in downstream["hop_delta_distribution"].items()
            },
            "left_only_pairs": [list(pair) for pair in ds_set["left_only"]],
            "right_only_pairs": [list(pair) for pair in ds_set["right_only"]],
            "hop_diffs": [
                {"src": src, "dst": dst, "left_hop": left_hop, "right_hop": right_hop}
                for src, dst, left_hop, right_hop in downstream["hop_diff"]
            ],
            "reversals": [list(pair) for pair in downstream["reversals"]],
        },
        "per_site": {
            "common_site_count": per_site["common_site_count"],
            "changed_site_count": per_site["changed_site_count"],
            "bidir_diff_count": per_site["bidir_diff_count"],
            "downstream_diff_count": per_site["downstream_diff_count"],
            "upstream_diff_count": per_site["upstream_diff_count"],
            "diffs": per_site["diffs"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="比较两个 site_chains 结果的差异（兼容 build_resource_buffer.py 的 site_chains 字段）"
    )
    parser.add_argument("left", help="左侧 site_chains JSON（路径或 topology_resources/ 下文件名）")
    parser.add_argument("right", help="右侧 site_chains JSON（路径或 topology_resources/ 下文件名）")
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

    left_chains, left_path = load_site_chains(args.left)
    right_chains, right_path = load_site_chains(args.right)

    left_norm = normalize_sites(left_chains)
    right_norm = normalize_sites(right_chains)

    left_stats = chain_stats(left_norm)
    right_stats = chain_stats(right_norm)

    site_set = compare_sets(set(left_norm), set(right_norm))
    bidir = compare_sets(bidirectional_edges(left_norm), bidirectional_edges(right_norm))
    downstream = compare_downstream(
        downstream_pairs_with_hops(left_norm),
        downstream_pairs_with_hops(right_norm),
    )
    per_site = compare_per_site(left_norm, right_norm)

    left_name = str(left_path)
    right_name = str(right_path)
    print_report(
        left_name, right_name, left_stats, right_stats,
        site_set, bidir, downstream, per_site, args,
    )

    if args.output:
        report = build_json_report(
            left_name, right_name, left_stats, right_stats,
            site_set, bidir, downstream, per_site,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n完整报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
