#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计告警输入的条数，支持 jsonl / csv / zip / 目录。"""

import argparse
import json

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import list_alarm_filepaths, stream_alarm_file


SORTED_ALARM_CACHE_TYPE = "fault_grouping.sorted_alarms.v1"


def _is_sorted_alarm_cache_header(record):
    return isinstance(record, dict) and record.get("cache_type") == SORTED_ALARM_CACHE_TYPE


def count_alarms(alarms_input, show_progress=True):
    filepaths = list_alarm_filepaths(alarms_input)
    total_files = len(filepaths)
    file_counts = []
    total_count = 0
    cache_header_count = 0

    for file_index, filepath in enumerate(filepaths, start=1):
        file_count = 0
        for record in stream_alarm_file(filepath, show_progress=show_progress,
                                        file_index=file_index, total_files=total_files):
            if _is_sorted_alarm_cache_header(record):
                cache_header_count += 1
                continue
            file_count += 1
        file_counts.append({"file": str(filepath), "alarm_count": file_count})
        total_count += file_count

    return {
        "input": str(alarms_input),
        "file_count": total_files,
        "alarm_count": total_count,
        "cache_header_count": cache_header_count,
        "files": file_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="统计告警输入的条数")
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整统计（含每个文件的条数）")
    parser.add_argument("--no-progress", action="store_true", help="关闭读取进度显示")
    args = parser.parse_args()

    result = count_alarms(args.alarms, show_progress=not args.no_progress)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"输入路径: {result['input']}")
    print(f"文件数: {result['file_count']}")
    for item in result["files"]:
        print(f"  {item['file']}: {item['alarm_count']} 条")
    if result["cache_header_count"]:
        print(f"已排除缓存头记录: {result['cache_header_count']} 条")
    print(f"告警总条数: {result['alarm_count']}")


if __name__ == "__main__":
    main()
