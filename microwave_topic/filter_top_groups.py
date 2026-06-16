#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 (站点数, 设备数, 告警数) 元组从大到小保留 top-k 故障组，并打乱输出顺序。

输入/输出与 complete_group_topology.py 对应：
- 文件输入（多行 JSONL）-> 文件输出（多行 JSONL）；
- 文件夹输入（每组一个单行 jsonl，--per-file 的产物）-> 文件夹输出（每组一个单行 jsonl）。
输入类型自动按路径是文件还是目录判断，输出类型与输入保持一致。
"""

import argparse
import json
import random
import sys
from pathlib import Path


def _normalize_text(value):
    return str(value or "").strip()


def _group_id(group):
    return (
        _normalize_text(group.get("uuid"))
        or _normalize_text((group.get("match_info") or {}).get("uuid"))
        or _normalize_text(group.get("故障组ID"))
    )


def _alarm_stats(group):
    """统计带告警的实体：返回 (有告警的站点集合, 有告警的设备集合, 告警数)。

    只统计 alarm 列表非空的设备；其所在站点计入站点集合；告警数为这些设备 alarm 长度之和。
    """
    ne_info = group.get("ne_info") if isinstance(group.get("ne_info"), dict) else {}
    alarm_site_ids = set()
    alarm_ne_ids = set()
    alarm_count = 0
    for ne_id, entry in ne_info.items():
        if not isinstance(entry, dict):
            continue
        alarms = entry.get("alarm")
        if not (isinstance(alarms, list) and alarms):
            continue
        alarm_ne_ids.add(_normalize_text(ne_id))
        alarm_count += len(alarms)
        site_id = _normalize_text(entry.get("site_id", ""))
        if site_id:
            alarm_site_ids.add(site_id)
    return alarm_site_ids, alarm_ne_ids, alarm_count


def _group_metrics(group):
    """返回排序元组 (有告警的站点数, 有告警的设备数, 告警数)，越大越靠前。"""
    site_ids, ne_ids, alarm_count = _alarm_stats(group)
    return (len(site_ids), len(ne_ids), alarm_count)


def _site_signature(group):
    """返回该组带告警站点集合的可哈希签名，用于判断两组站点是否完全相同。"""
    site_ids, _, _ = _alarm_stats(group)
    return frozenset(site_ids)


def _load_groups_from_file(input_path):
    """读取多行 JSONL 文件，返回 [group, ...]。"""
    groups = []
    with open(input_path, "r", encoding="utf-8") as fr:
        for line_num, raw_line in enumerate(fr, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path} 第 {line_num} 行 JSON 解析失败: {exc}") from exc
            if isinstance(record, dict):
                groups.append(record)
    return groups


def _load_groups_from_dir(input_dir):
    """读取目录下每个单行 jsonl 文件，返回 [(源文件名, group), ...]，按文件名排序保证稳定。"""
    items = []
    for path in sorted(Path(input_dir).glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fr:
            line = fr.readline().strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} JSON 解析失败: {exc}") from exc
        if isinstance(record, dict):
            items.append((path.name, record))
    return items


def _select_top_k(groups, top_k, key):
    """按 metrics 元组从大到小排序取 top-k，平手时用原始下标（保证稳定可复现）破除。

    去重原则：若多组带告警的站点集合完全相同，只保留告警数最多的那个（平手取靠前的），
    其余丢弃，不再补足 top-k（最终数量可能少于 k）。
    """
    # 先按站点签名去重：同一站点集合只保留告警数最多的（平手取原始下标靠前的）。
    best_by_sites = {}
    for idx, item in enumerate(groups):
        group = key(item)
        signature = _site_signature(group)
        _, _, alarm_count = _alarm_stats(group)
        # 选择键越大越优先：告警数大优先，平手时下标小优先（-idx 大）。
        rank = (alarm_count, -idx)
        if signature not in best_by_sites or rank > best_by_sites[signature][0]:
            best_by_sites[signature] = (rank, idx, item)

    survivors = [(idx, item) for _, idx, item in best_by_sites.values()]
    # 再按 metrics 元组从大到小排序，平手用原始下标破除，保证稳定可复现。
    survivors.sort(key=lambda pair: (_group_metrics(key(pair[1])), -pair[0]), reverse=True)

    if top_k is not None and top_k >= 0:
        survivors = survivors[:top_k]
    return [pair[1] for pair in survivors]


def filter_groups(input_path, output_path, top_k, seed=None):
    input_path = Path(input_path)
    rng = random.Random(seed)

    if input_path.is_dir():
        items = _load_groups_from_dir(input_path)
        total = len(items)
        selected = _select_top_k(items, top_k, key=lambda item: item[1])
        rng.shuffle(selected)

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, group in selected:
            line = json.dumps(group, ensure_ascii=False, separators=(",", ":"))
            (out_dir / name).write_text(line + "\n", encoding="utf-8")
        kept = len(selected)
        output_kind = "dir"
    elif input_path.is_file():
        groups = _load_groups_from_file(input_path)
        total = len(groups)
        selected = _select_top_k(groups, top_k, key=lambda group: group)
        rng.shuffle(selected)

        out_path = Path(output_path)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fw:
            for group in selected:
                fw.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
                fw.write("\n")
        kept = len(selected)
        output_kind = "file"
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "output_kind": output_kind,
        "input_group_count": total,
        "output_group_count": kept,
        "top_k": top_k,
        "seed": seed,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="按 (站点数, 设备数, 告警数) 从大到小保留 top-k 故障组并打乱输出顺序"
    )
    parser.add_argument("input", help="输入：故障组多行 JSONL 文件，或每组一个单行 jsonl 的目录")
    parser.add_argument("output", help="输出：文件输入对应文件，目录输入对应目录（不存在则新建）")
    parser.add_argument("-k", "--top-k", type=int, required=True, help="保留的故障组数量（top-k）")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="打乱顺序用的随机种子，省略则每次随机；指定后结果可复现",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k 不能为负数")
    stats = filter_groups(args.input, args.output, args.top_k, seed=args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
