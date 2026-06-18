#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查告警里是否存在相同 eid 的情况，支持 jsonl / csv / zip / 目录，可选时间段过滤。

对于 eid 相同的告警：
- 如果它们的所有字段完全相同，则视为同一条告警的重复记录，不算作重复；
- 如果存在字段取了多个不同的值，则算作重复，并打印出这些有差异的字段及其取值。
"""

import argparse
import json
from datetime import datetime

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import stream_alarm_inputs
from fault_grouping.alarm_events.sorted_cache import consume_sorted_alarm_cache_header


# 与 alarm_flow_mhp/aggregator.py 的 _event_id 保持一致的 eid 候选字段
DEFAULT_EID_FIELDS = ["告警编码ID", "alarm_id", "event_id", "id"]
DEFAULT_TIME_FIELD = "告警首次发生时间"
MISSING = "<缺失>"

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d")


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


def _passes_time_filter(alarm, time_field, start, end):
    if start is None and end is None:
        return True
    dt_obj = _parse_time(alarm.get(time_field))
    if dt_obj is None:
        return False
    if start is not None and dt_obj < start:
        return False
    if end is not None and dt_obj > end:
        return False
    return True


def _extract_eid(alarm, eid_fields):
    """提取告警的 eid，兼容嵌套在 alarm 字段下的情况。"""
    nested = alarm.get("alarm") if isinstance(alarm.get("alarm"), dict) else {}
    for key in eid_fields:
        value = nested.get(key) if key in nested else alarm.get(key, "")
        value = str(value or "").strip()
        if value:
            return value
    return ""


def _value_repr(value):
    """把字段值规整成可比较 / 可展示的字符串。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _record_fingerprint(alarm):
    """整条记录的规范化指纹，用于判断两条记录是否完全相同。"""
    return json.dumps(alarm, ensure_ascii=False, sort_keys=True, default=str)


def check_duplicate_eid(alarms_input, eid_fields=None, time_field=DEFAULT_TIME_FIELD,
                        start=None, end=None, show_progress=True):
    eid_fields = eid_fields or DEFAULT_EID_FIELDS

    # ---- Pass 1：流式统计每个 eid 的命中数（内存只保存按 eid 聚合的轻量信息）----
    eid_counts = {}
    eid_first_time = {}

    processed_count = 0
    cache_header_count = 0
    out_of_range_count = 0
    no_eid_count = 0
    counted_count = 0

    for alarm in stream_alarm_inputs(alarms_input, show_progress=show_progress):
        if consume_sorted_alarm_cache_header(alarm):
            cache_header_count += 1
            continue
        processed_count += 1

        if not _passes_time_filter(alarm, time_field, start, end):
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

    candidate_eids = {eid for eid, count in eid_counts.items() if count > 1}

    # ---- Pass 2：仅对重复出现的 eid 收集字段级明细（内存仅与重复 eid 数相关）----
    details = {
        eid: {"count": 0, "fingerprints": set(), "fields": {}}
        for eid in candidate_eids
    }
    if candidate_eids:
        for alarm in stream_alarm_inputs(alarms_input, show_progress=show_progress):
            if consume_sorted_alarm_cache_header(alarm):
                continue
            if not _passes_time_filter(alarm, time_field, start, end):
                continue
            eid = _extract_eid(alarm, eid_fields)
            entry = details.get(eid)
            if entry is None:
                continue
            entry["count"] += 1
            entry["fingerprints"].add(_record_fingerprint(alarm))
            for key, value in alarm.items():
                field_entry = entry["fields"].setdefault(key, {"values": set(), "present": 0})
                field_entry["values"].add(_value_repr(value))
                field_entry["present"] += 1

    # ---- 分类：完全相同 vs 字段有差异 ----
    duplicates = []
    identical_eid_count = 0
    identical_alarm_count = 0

    for eid, entry in details.items():
        count = entry["count"]
        # 所有记录指纹一致 => 完全相同，不算重复
        if len(entry["fingerprints"]) <= 1:
            identical_eid_count += 1
            identical_alarm_count += count
            continue

        varying_fields = {}
        for field, field_entry in entry["fields"].items():
            values = set(field_entry["values"])
            if field_entry["present"] < count:
                values.add(MISSING)  # 部分记录缺失该字段，也算一种差异
            if len(values) > 1:
                varying_fields[field] = sorted(values)

        duplicates.append({
            "eid": eid,
            "count": count,
            "first_time": eid_first_time.get(eid, ""),
            "varying_fields": varying_fields,
        })

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
        "identical_eid_count": identical_eid_count,
        "identical_alarm_count": identical_alarm_count,
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


def _format_values(values, max_show):
    shown = values[:max_show]
    text = " | ".join(shown)
    if len(values) > max_show:
        text += f" ...(共 {len(values)} 种)"
    return text


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
    parser.add_argument("--max-values", type=int, default=6, help="每个差异字段最多展示的取值数量，默认 6")
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
    if result["identical_eid_count"]:
        print(f"字段完全相同的 eid（不计为重复）: {result['identical_eid_count']} 个，"
              f"涉及 {result['identical_alarm_count']} 条告警")
    print()

    if not result["duplicates"]:
        print("✅ 未发现字段存在差异的相同 eid 告警")
        return

    print(f"⚠️ 发现 {result['duplicate_eid_count']} 个重复 eid（字段存在差异），"
          f"共涉及 {result['duplicate_alarm_count']} 条告警:")
    for item in result["duplicates"][: args.top]:
        first_time = f"，首次: {item['first_time']}" if item["first_time"] else ""
        print(f"  eid={item['eid']}: {item['count']} 条{first_time}")
        for field, values in item["varying_fields"].items():
            print(f"      - {field}: {_format_values(values, args.max_values)}")
    remaining = len(result["duplicates"]) - args.top
    if remaining > 0:
        print(f"  ... 还有 {remaining} 个重复 eid（使用 --json 或 --top 查看全部）")


if __name__ == "__main__":
    main()
