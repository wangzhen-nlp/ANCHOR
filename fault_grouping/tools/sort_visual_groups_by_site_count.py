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


ALARM_TYPE_CHOICES = frozenset({"offline", "power", "link"})


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


def normalize_alarm_type(value):
    return str(value or "").strip().lower()


def parse_alarm_types(raw_values):
    selected = set()
    for raw in raw_values or []:
        for part in str(raw).split(","):
            value = normalize_alarm_type(part)
            if value:
                selected.add(value)
    invalid = sorted(selected - ALARM_TYPE_CHOICES)
    if invalid:
        raise SystemExit(
            "不支持的 alarm_type: "
            + ", ".join(invalid)
            + "；可选值: "
            + ", ".join(sorted(ALARM_TYPE_CHOICES))
        )
    return selected


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def extract_alarm_types(record):
    """Extract coarse alarm types from a visual/group record."""
    if not isinstance(record, dict):
        return set()

    out = set()

    for key in ("alarm_type_counts",):
        for alarm_type in _as_dict(record.get(key)).keys():
            value = normalize_alarm_type(alarm_type)
            if value:
                out.add(value)

    cascade_info = _as_dict(record.get("cascade_info"))
    for alarm_type in _as_dict(cascade_info.get("alarm_type_counts")).keys():
        value = normalize_alarm_type(alarm_type)
        if value:
            out.add(value)

    for symptoms in (record.get("symptoms"), _as_dict(record.get("match_info")).get("symptoms")):
        for symptom in _as_list(symptoms):
            if not isinstance(symptom, dict):
                continue
            for key in ("alarm_type", "alarm"):
                value = normalize_alarm_type(symptom.get(key))
                if value:
                    out.add(value)

    ne_info = _as_dict(record.get("ne_info"))
    for ne_meta in ne_info.values():
        if not isinstance(ne_meta, dict):
            continue
        for alarm in _as_list(ne_meta.get("alarm")):
            if not isinstance(alarm, dict):
                continue
            for key in ("alarm_type", "alarm"):
                value = normalize_alarm_type(alarm.get(key))
                if value:
                    out.add(value)

    return out


def sort_and_filter_visual_groups(
    input_path,
    output_path,
    *,
    min_site_count: int,
    alarm_types: set[str] | None = None,
):
    rows = []
    total = 0
    max_site_count = 0
    dropped_by_site_count = 0
    dropped_by_alarm_type = 0
    alarm_types = set(alarm_types or [])

    for ordinal, (_line_num, record) in enumerate(iter_jsonl_records(input_path)):
        total += 1
        sites = sorted(extract_site_ids(record))
        site_count = len(sites)
        max_site_count = max(max_site_count, site_count)
        if site_count < min_site_count:
            dropped_by_site_count += 1
            continue
        record_alarm_types = extract_alarm_types(record)
        if alarm_types and record_alarm_types.isdisjoint(alarm_types):
            dropped_by_alarm_type += 1
            continue
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
        "alarm_types": sorted(alarm_types),
        "total_groups": total,
        "kept_groups": len(rows),
        "dropped_groups": total - len(rows),
        "dropped_by_site_count": dropped_by_site_count,
        "dropped_by_alarm_type": dropped_by_alarm_type,
        "max_seen_site_count": max_site_count,
        "sort": "site_count_desc",
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
        "--alarm-type",
        action="append",
        default=[],
        help=(
            "只保留至少包含一个指定告警类型的 group。可重复传入或用逗号分隔；"
            "可选: offline,power,link。多个类型按并集保留。"
        ),
    )
    args = parser.parse_args()

    if args.min_site_count < 0:
        parser.error("--min-site-count 不能小于 0")

    stats = sort_and_filter_visual_groups(
        args.input,
        args.output,
        min_site_count=args.min_site_count,
        alarm_types=parse_alarm_types(args.alarm_type),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr if args.output == "-" else sys.stdout)


if __name__ == "__main__":
    main()
