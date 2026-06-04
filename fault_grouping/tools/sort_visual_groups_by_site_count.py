#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sort stream visual JSONL groups by covered site count and keep large groups.

Usage:
    python fault_grouping/tools/sort_visual_groups_by_site_count.py \
        stream_visual.jsonl stream_visual_ge10_sorted.jsonl --min-site-count 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fault_grouping.tools.analyze_group_site_distribution import extract_site_ids


def iter_jsonl_records(path):
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"文件不存在: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"跳过第 {line_num} 行：JSON 顶层不是对象", file=sys.stderr)
                continue
            yield line_num, record


def group_id(record):
    match_info = record.get("match_info") if isinstance(record.get("match_info"), dict) else {}
    return str(
        record.get("uuid")
        or record.get("group_id")
        or record.get("cascade_id")
        or match_info.get("uuid")
        or ""
    )


def sort_and_filter_visual_groups(
    input_path,
    output_path,
    *,
    min_site_count: int,
    add_site_count: bool = False,
):
    rows = []
    total = 0
    max_site_count = 0

    for ordinal, (_line_num, record) in enumerate(iter_jsonl_records(input_path)):
        total += 1
        sites = sorted(extract_site_ids(record))
        site_count = len(sites)
        max_site_count = max(max_site_count, site_count)
        if site_count < min_site_count:
            continue
        if add_site_count:
            record = dict(record)
            record["site_count"] = site_count
            record["site_list_sorted"] = sites
        rows.append((site_count, group_id(record), ordinal, record))

    rows.sort(key=lambda item: (-item[0], item[1], item[2]))

    output = sys.stdout if str(output_path) == "-" else Path(output_path).open("w", encoding="utf-8")
    try:
        for _site_count, _gid, _ordinal, record in rows:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    finally:
        if output is not sys.stdout:
            output.close()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "min_site_count": min_site_count,
        "total_groups": total,
        "kept_groups": len(rows),
        "dropped_groups": total - len(rows),
        "max_seen_site_count": max_site_count,
        "sort": "site_count_desc",
        "add_site_count": add_site_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="按覆盖站点数降序排序 stream visual JSONL，并保留站点数 >= 阈值的 group"
    )
    parser.add_argument("input", help="输入 stream visual JSONL")
    parser.add_argument("output", help="输出 JSONL；使用 '-' 输出到 stdout")
    parser.add_argument(
        "--min-site-count",
        "-n",
        type=int,
        required=True,
        help="只保留覆盖站点数 >= N 的 group",
    )
    parser.add_argument(
        "--add-site-count",
        action="store_true",
        help="在输出记录中附加 site_count 和 site_list_sorted，便于人工检查",
    )
    args = parser.parse_args()

    if args.min_site_count < 0:
        parser.error("--min-site-count 不能小于 0")

    stats = sort_and_filter_visual_groups(
        args.input,
        args.output,
        min_site_count=args.min_site_count,
        add_site_count=args.add_site_count,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr if args.output == "-" else sys.stdout)


if __name__ == "__main__":
    main()
