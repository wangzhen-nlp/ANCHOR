#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 sys_link 链路输入的所有字段

读取 .jsonl/.csv/.zip(内含CSV) 文件或包含这些文件的目录，统计所有出现过的字段名
（取所有记录字段名的并集），并打印每个字段的出现次数与非空次数。
"""

import argparse
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.progress_utils import ProgressBar
from topology_resources import SYS_LINK_JSONL, resource_display
from topology_tools.extract_site_graph import PROGRESS_ROW_STEP, iter_link_records


def collect_link_fields(link_input: str, limit: int = None):
    """遍历链路记录，统计字段名的并集及各字段出现/非空次数。

    Args:
        limit: 仅读取前 limit 条记录；None 表示读取全部。

    Returns:
        (row_count, total_count, nonempty_count)
        - total_count[field]:    该字段出现的记录数
        - nonempty_count[field]: 该字段值非空的记录数
    """
    total_count = defaultdict(int)
    nonempty_count = defaultdict(int)

    progress = ProgressBar(0, "  读取链路记录")
    row_count = 0
    for record in iter_link_records(link_input, progress):
        if limit is not None and row_count >= limit:
            break
        row_count += 1
        if row_count % PROGRESS_ROW_STEP == 0:
            progress.set(row_count)
        if not isinstance(record, dict):
            continue
        for field, value in record.items():
            total_count[field] += 1
            if str(value if value is not None else "").strip():
                nonempty_count[field] += 1
    progress.set(row_count)
    progress.close()

    return row_count, total_count, nonempty_count


def main():
    parser = argparse.ArgumentParser(description="打印 sys_link 链路输入的所有字段")
    parser.add_argument(
        "--link-input",
        default=SYS_LINK_JSONL,
        help=(
            "链路输入，支持 .jsonl/.csv/.zip(内含CSV) 文件或包含这些文件的目录，"
            f"默认: {resource_display('sys_link_1231.jsonl')}"
        ),
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="仅读取前 N 条记录用于统计；默认读取全部",
    )
    args = parser.parse_args()

    print(f"读取链路输入: {args.link_input}")
    row_count, total_count, nonempty_count = collect_link_fields(args.link_input, args.limit)

    # 按非空次数降序排序，次数相同按字段名升序
    fields = sorted(total_count, key=lambda f: (-nonempty_count[f], f))
    print(f"\n共读取 {row_count} 条记录，发现 {len(fields)} 个字段:\n")

    name_width = max((len(f) for f in fields), default=len("字段"))
    name_width = max(name_width, len("字段"))
    print(f"{'字段':<{name_width}}  {'出现次数':>10}  {'非空次数':>10}")
    print(f"{'-' * name_width}  {'-' * 10}  {'-' * 10}")
    for field in fields:
        print(f"{field:<{name_width}}  {total_count[field]:>10}  {nonempty_count[field]:>10}")


if __name__ == "__main__":
    main()
