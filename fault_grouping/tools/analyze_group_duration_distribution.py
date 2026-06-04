#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze alarm time-span distribution for group JSONL / stream visual JSONL.

The duration of one group is:

    max(first occurrence ts of alarms) - min(first occurrence ts of alarms)

Usage:
    python fault_grouping/tools/analyze_group_duration_distribution.py stream_visual.jsonl
    python fault_grouping/tools/analyze_group_duration_distribution.py stream_visual.jsonl -o duration_dist.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import statistics
import sys
from pathlib import Path


DEFAULT_BUCKETS_SEC = (
    0,
    60,
    5 * 60,
    15 * 60,
    30 * 60,
    60 * 60,
    2 * 60 * 60,
    6 * 60 * 60,
    12 * 60 * 60,
    24 * 60 * 60,
    48 * 60 * 60,
    7 * 24 * 60 * 60,
)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _parse_time_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(text[:26], fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _coerce_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts if ts > 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = float(text)
        return ts if ts > 0 else None
    except ValueError:
        return _parse_time_text(text)


def _symptom_ts(symptom):
    if not isinstance(symptom, dict):
        return None
    for key in ("ts", "_segment_start_ts", "first_ts", "first_occurrence_ts"):
        ts = _coerce_ts(symptom.get(key))
        if ts is not None:
            return ts
    for key in ("alarm_time", "time_str", "告警首次发生时间", "first_occurrence_time"):
        ts = _parse_time_text(symptom.get(key))
        if ts is not None:
            return ts
    return None


def extract_alarm_timestamps(record):
    """Extract first-occurrence timestamps from one group record."""
    if not isinstance(record, dict):
        return []

    timestamps = []
    symptom_sources = [
        record.get("symptoms"),
        _as_dict(record.get("match_info")).get("symptoms"),
    ]
    for symptoms in symptom_sources:
        for symptom in _as_list(symptoms):
            ts = _symptom_ts(symptom)
            if ts is not None:
                timestamps.append(ts)

    # Fallback for visual records where alarms are nested under ne_info.
    if not timestamps:
        ne_info = _as_dict(record.get("ne_info"))
        for ne_meta in ne_info.values():
            if not isinstance(ne_meta, dict):
                continue
            for alarm in _as_list(ne_meta.get("alarm")):
                ts = _symptom_ts(alarm)
                if ts is not None:
                    timestamps.append(ts)

    # Last-resort fallback: some group outputs persist aggregate bounds only.
    if not timestamps:
        start_ts = _coerce_ts(record.get("start_ts"))
        end_ts = _coerce_ts(record.get("end_ts"))
        if start_ts is not None and end_ts is not None:
            timestamps.extend([start_ts, end_ts])

    return timestamps


def group_id(record):
    match_info = _as_dict(record.get("match_info"))
    return str(
        record.get("uuid")
        or record.get("group_id")
        or record.get("cascade_id")
        or match_info.get("uuid")
        or ""
    )


def group_rule(record):
    match_info = _as_dict(record.get("match_info"))
    return str(record.get("rule") or match_info.get("rule") or "")


def compute_percentiles(values, percentiles=(50, 90, 95, 99, 99.9)):
    if not values:
        return {}
    vals = sorted(values)
    n = len(vals)
    out = {}
    for p in percentiles:
        idx = int((p / 100.0) * (n - 1))
        idx = max(0, min(idx, n - 1))
        out[f"p{p}"] = vals[idx]
    return out


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def bucket_label(upper_sec, previous_sec=None):
    if upper_sec == 0:
        return "0s"
    if previous_sec is None or previous_sec <= 0:
        return f"(0,{format_duration(upper_sec)}]"
    return f"({format_duration(previous_sec)},{format_duration(upper_sec)}]"


def build_histogram(durations, buckets_sec=DEFAULT_BUCKETS_SEC):
    buckets = sorted(float(b) for b in buckets_sec)
    hist = Counter()
    for duration in durations:
        placed = False
        prev = None
        for upper in buckets:
            if duration <= upper:
                hist[bucket_label(upper, prev)] += 1
                placed = True
                break
            prev = upper
        if not placed:
            hist[f">{format_duration(buckets[-1])}"] += 1
    ordered = []
    prev = None
    for upper in buckets:
        label = bucket_label(upper, prev)
        ordered.append({"bucket": label, "group_count": hist.get(label, 0)})
        prev = upper
    tail = f">{format_duration(buckets[-1])}"
    ordered.append({"bucket": tail, "group_count": hist.get(tail, 0)})
    return ordered


def parse_buckets(raw):
    if not raw:
        return DEFAULT_BUCKETS_SEC
    vals = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        vals.append(float(text))
    if not vals:
        raise SystemExit("--bucket-sec 至少需要一个有效数字")
    if any(v < 0 for v in vals):
        raise SystemExit("--bucket-sec 不能包含负数")
    return tuple(sorted(set(vals)))


def iter_records(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {jsonl_path}")
    with path.open("r", encoding="utf-8") as handle:
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


def analyze(jsonl_path, *, buckets_sec=DEFAULT_BUCKETS_SEC):
    durations = []
    details = []
    rule_to_durations = defaultdict(list)
    total = 0
    skipped_no_time = 0

    for line_num, record in iter_records(jsonl_path):
        total += 1
        timestamps = extract_alarm_timestamps(record)
        if not timestamps:
            skipped_no_time += 1
            continue
        start_ts = min(timestamps)
        end_ts = max(timestamps)
        duration_sec = max(0.0, end_ts - start_ts)
        durations.append(duration_sec)
        rule = group_rule(record)
        rule_to_durations[rule].append(duration_sec)
        details.append({
            "line_num": line_num,
            "uuid": group_id(record),
            "rule": rule,
            "alarm_count_with_time": len(timestamps),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_sec": duration_sec,
            "duration_human": format_duration(duration_sec),
        })

    if not durations:
        raise SystemExit("未解析到任何带时间戳的 group")

    durations_sorted = sorted(durations)
    by_rule = {}
    for rule, vals in sorted(rule_to_durations.items()):
        by_rule[rule] = {
            "count": len(vals),
            "mean_sec": round(sum(vals) / len(vals), 3),
            "median_sec": statistics.median(vals),
            "max_sec": max(vals),
            "percentiles_sec": compute_percentiles(vals),
            "histogram": build_histogram(vals, buckets_sec),
        }

    return {
        "meta": {
            "source_file": str(jsonl_path),
            "total_groups": total,
            "groups_with_time": len(durations),
            "skipped_no_time": skipped_no_time,
            "bucket_upper_bounds_sec": list(buckets_sec),
        },
        "overall": {
            "mean_sec": round(sum(durations) / len(durations), 3),
            "min_sec": durations_sorted[0],
            "max_sec": durations_sorted[-1],
            "median_sec": statistics.median(durations),
            "percentiles_sec": compute_percentiles(durations),
            "histogram": build_histogram(durations, buckets_sec),
        },
        "by_rule": by_rule,
        "detail": details,
    }


def print_summary(result):
    overall = result["overall"]
    meta = result["meta"]
    print("\nGroup 告警时长跨度分布摘要")
    print("=" * 60)
    print(f"总 group 数: {meta['total_groups']}")
    print(f"带时间戳 group 数: {meta['groups_with_time']}")
    print(f"跳过无时间戳 group 数: {meta['skipped_no_time']}")
    print(f"平均跨度: {format_duration(overall['mean_sec'])}")
    print(f"中位跨度: {format_duration(overall['median_sec'])}")
    print(f"最小/最大: {format_duration(overall['min_sec'])} / {format_duration(overall['max_sec'])}")
    print("\n百分位数:")
    for key, value in overall["percentiles_sec"].items():
        print(f"  {key}: {format_duration(value)} ({value:.0f}s)")
    print("\n直方图:")
    total = max(1, meta["groups_with_time"])
    for item in overall["histogram"]:
        count = item["group_count"]
        print(f"  {item['bucket']:>14}: {count:>8} ({count / total * 100:>6.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="统计 group 内告警首次发生时间跨度分布")
    parser.add_argument("jsonl", help="输入 group/visual JSONL")
    parser.add_argument("-o", "--output", default="", help="输出统计 JSON；为空则只打印摘要")
    parser.add_argument(
        "--bucket-sec",
        default="",
        help="自定义直方图桶上界秒数，逗号分隔；默认覆盖 0s 到 7d",
    )
    args = parser.parse_args()

    result = analyze(args.jsonl, buckets_sec=parse_buckets(args.bucket_sec))
    print_summary(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"已保存详细结果: {args.output}")


if __name__ == "__main__":
    main()
