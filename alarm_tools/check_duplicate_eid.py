#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查告警里是否存在相同 eid 的情况，支持 jsonl / csv / zip / 目录，可选时间段过滤。"""

import argparse
import json
from datetime import datetime

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import stream_alarm_inputs


# 与 alarm_flow_mhp/aggregator.py 的 _event_id 保持一致的 eid 候选字段
DEFAULT_EID_FIELDS = ["告警编码ID", "alarm_id", "event_id", "id"]
DEFAULT_TIME_FIELD = "告警首次发生时间"
SORTED_ALARM_CACHE_TYPE = "fault_grouping.sorted_alarms.v1"

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d")


def _is_sorted_alarm_cache_header(record):
    return isinstance(record, dict) and record.get("cache_type") == SORTED_ALARM_CACHE_TYPE


def _parse_time(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_dt(dt_obj):
    return "-" if dt_obj is None else dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _extract_eid(alarm, eid_fields):
    """提取告警的 eid，兼容嵌套在 alarm 字段下的情况。"""
    nested = alarm.get("alarm") if isinstance(alarm.get("alarm"), dict) else {}
    for key in eid_fields:
        value = nested.get(key) if key in nested else alarm.get(key, "")
        value = str(value or "").strip()
        if value:
            return value
    return ""


def check_duplicate_eid(alarms_input, eid_fields=None, time_field=DEFAULT_TIME_FIELD,
                        start=None, end=None, show_progress=True):
    eid_fields = eid_fields or DEFAULT_EID_FIELDS

    # eid -> 命中告警数
    eid_counts = {}
    # eid -> 首条命中告警的时间（字符串，用于展示）
    eid_first_time = {}

    processed_count = 0
    cache_header_count = 0
    out_of_range_count = 0
    no_eid_count = 0
    counted_count = 0

    for alarm in stream_alarm_inputs(alarms_input, show_progress=show_progress):
        if _is_sorted_alarm_cache_header(alarm):
            cache_header_count += 1
            continue
        processed_count += 1

        # 时间段过滤（仅在指定了 start/end 时生效）
        if start is not None or end is not None:
            dt_obj = _parse_time(alarm.get(time_field))
            if dt_obj is None:
                out_of_range_count += 1
                continue
            if start is not None and dt_obj < start:
                out_of_range_count += 1
                continue
            if end is not None and dt_obj > end:
                out_of_range_count += 1
                continue

        eid = _extract_eid(alarm, eid_fields)
        if not eid:
            no_eid_count += 1
            continue

        counted_count += 1
        eid_counts[eid] = eid_counts.get(eid, 0) + 1
        if eid not in eid_first_time:
            eid_first_time[eid] = str(alarm.get(time_field) or "").strip()

    duplicates = [
        {"eid": eid, "count": count, "first_time": eid_first_time.get(eid, "")}
        for eid, count in eid_counts.items()
        if count > 1
    ]
    duplicates.sort(key=lambda item: (-item["count"], item["eid"]))
    duplicate_alarm_count = sum(item["count"] for item in duplicates)

    return {
        "input": str(alarms_input),
        "eid_fields": eid_fields,
        "time_field": time_field,
        "start": _format_dt(start) if start else None,
        "end": _format_dt(end) if end else None,
        "processed_count": processed_count,
        "counted_count": counted_count,
        "unique_eid_count": len(eid_counts),
        "duplicate_eid_count": len(duplicates),
        "duplicate_alarm_count": duplicate_alarm_count,
        "no_eid_count": no_eid_count,
        "out_of_range_count": out_of_range_count,
        "cache_header_count": cache_header_count,
        "duplicates": duplicates,
    }


def _parse_time_arg(parser, value, name):
    if value is None:
        return None
    dt_obj = _parse_time(value)
    if dt_obj is None:
        parser.error(f"{name} 时间格式无法解析: {value!r}（支持 'YYYY-MM-DD HH:MM:SS' 等）")
    return dt_obj


def main():
    parser = argparse.ArgumentParser(description="检查告警里是否存在相同 eid 的情况")
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    parser.add_argument("--start-time", help="时间段起点（含），如 '2026-06-01 00:00:00'")
    parser.add_argument("--end-time", help="时间段终点（含），如 '2026-06-02 00:00:00'")
    parser.add_argument(
        "--time-field",
        default=DEFAULT_TIME_FIELD,
        help=f"用于时间过滤的字段名，默认: {DEFAULT_TIME_FIELD}",
    )
    parser.add_argument(
        "--eid-fields",
        nargs="+",
        default=DEFAULT_EID_FIELDS,
        help="eid 候选字段（按顺序取第一个非空），默认: " + " ".join(DEFAULT_EID_FIELDS),
    )
    parser.add_argument("--top", type=int, default=20, help="文本输出时展示的重复 eid 数量，默认 20")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    parser.add_argument("--no-progress", action="store_true", help="关闭读取进度显示")
    args = parser.parse_args()

    start = _parse_time_arg(parser, args.start_time, "--start-time")
    end = _parse_time_arg(parser, args.end_time, "--end-time")
    if start is not None and end is not None and start > end:
        parser.error("--start-time 不能晚于 --end-time")

    result = check_duplicate_eid(
        args.alarms,
        eid_fields=args.eid_fields,
        time_field=args.time_field,
        start=start,
        end=end,
        show_progress=not args.no_progress,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"输入路径: {result['input']}")
    if result["start"] or result["end"]:
        print(f"时间段: {result['start'] or '-'} ~ {result['end'] or '-'}（字段: {result['time_field']}）")
        print(f"超出时间段/无时间被跳过: {result['out_of_range_count']} 条")
    print(f"处理告警数: {result['processed_count']}")
    print(f"参与统计的告警数: {result['counted_count']}")
    if result["no_eid_count"]:
        print(f"无 eid 被跳过: {result['no_eid_count']} 条")
    if result["cache_header_count"]:
        print(f"已排除缓存头记录: {result['cache_header_count']} 条")
    print(f"去重后 eid 数: {result['unique_eid_count']}")
    print()

    if not result["duplicates"]:
        print("✅ 未发现相同 eid 的告警")
        return

    print(f"⚠️ 发现 {result['duplicate_eid_count']} 个重复 eid，"
          f"共涉及 {result['duplicate_alarm_count']} 条告警:")
    for item in result["duplicates"][: args.top]:
        first_time = f"（首次: {item['first_time']}）" if item["first_time"] else ""
        print(f"  eid={item['eid']}: {item['count']} 条{first_time}")
    remaining = len(result["duplicates"]) - args.top
    if remaining > 0:
        print(f"  ... 还有 {remaining} 个重复 eid（使用 --json 或 --top 查看全部）")


if __name__ == "__main__":
    main()
