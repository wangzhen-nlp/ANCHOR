#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按故障组覆盖站点数过滤 match_rules.py 输出的 JSONL。

默认行为：过滤掉站点数大于 n 的故障组，只保留 site_count <= n 的记录。

用法:
    python fault_grouping/tools/filter_groups_by_site_count.py fault_groups.jsonl filtered.jsonl --max-site-count 20
    python fault_grouping/tools/filter_groups_by_site_count.py fault_groups.jsonl large_groups.jsonl --max-site-count 20 --keep-large
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fault_grouping.tools.analyze_group_site_distribution import extract_site_ids


def iter_group_lines(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {jsonl_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ 跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"⚠️ 跳过第 {line_num} 行：JSON 顶层不是对象", file=sys.stderr)
                continue
            yield line_num, raw_line.rstrip("\n"), record


def should_keep_group(site_count, max_site_count, keep_large=False):
    if keep_large:
        return site_count > max_site_count
    return site_count <= max_site_count


def filter_groups(input_path, output_path, max_site_count, keep_large=False):
    total_count = 0
    kept_count = 0
    dropped_count = 0
    max_seen_site_count = 0

    with open(output_path, "w", encoding="utf-8") as fw:
        for _line_num, raw_line, record in iter_group_lines(input_path):
            total_count += 1
            site_count = len(extract_site_ids(record))
            max_seen_site_count = max(max_seen_site_count, site_count)
            if should_keep_group(site_count, max_site_count, keep_large=keep_large):
                fw.write(raw_line)
                fw.write("\n")
                kept_count += 1
            else:
                dropped_count += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "max_site_count": max_site_count,
        "mode": "keep_site_count_gt_n" if keep_large else "drop_site_count_gt_n",
        "total_groups": total_count,
        "kept_groups": kept_count,
        "dropped_groups": dropped_count,
        "max_seen_site_count": max_seen_site_count,
    }


def main():
    parser = argparse.ArgumentParser(description="按站点数过滤 match_rules.py 输出的故障组 JSONL")
    parser.add_argument("input", help="输入故障组 JSONL")
    parser.add_argument("output", help="输出过滤后的 JSONL")
    parser.add_argument("--max-site-count", "-n", type=int, required=True, help="站点数阈值 n")
    parser.add_argument("--keep-large", action="store_true", help="反向过滤：只保留站点数大于 n 的故障组")
    args = parser.parse_args()

    if args.max_site_count < 0:
        parser.error("--max-site-count 不能小于 0")

    stats = filter_groups(
        args.input,
        args.output,
        args.max_site_count,
        keep_large=args.keep_large,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
