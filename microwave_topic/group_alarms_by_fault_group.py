#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按告警流中的故障组ID聚合告警，并输出 JSONL。"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_inputs import stream_alarm_inputs
from topology_resources import NE_GRAPH_JSON, resource_display


DEFAULT_GROUP_FIELD = "故障组ID"
DEFAULT_DEVICE_FIELDS = (
    "domain",
    "Domain",
    "专业",
    "设备类型",
    "设备类型名称",
    "设备专业",
    "网络类型",
    "网元类型",
)


def _normalize_text(value):
    return str(value or "").strip()


def _normalize_key(value):
    return _normalize_text(value).lower()


def _split_csv_values(values):
    result = []
    for value in values or []:
        for part in str(value).replace("，", ",").split(","):
            text = part.strip()
            if text:
                result.append(text)
    return result


def _parse_device_fields(raw_fields):
    fields = _split_csv_values(raw_fields)
    return fields or list(DEFAULT_DEVICE_FIELDS)


def _load_ne_domain_map(ne_graph_path):
    if not ne_graph_path:
        return {}

    if not Path(ne_graph_path).exists():
        print(f"⚠️ ne_graph 文件不存在，回退到告警字段判断设备类型: {ne_graph_path}", file=sys.stderr)
        return {}

    with open(ne_graph_path, "r", encoding="utf-8") as fr:
        ne_graph = json.load(fr)

    if not isinstance(ne_graph, dict):
        raise ValueError(f"ne_graph 顶层必须是对象: {ne_graph_path}")

    domain_map = {}
    for ne_id, ne_info in ne_graph.items():
        if not isinstance(ne_info, dict):
            continue
        domain = _normalize_text(ne_info.get("domain", ""))
        if domain:
            domain_map[_normalize_text(ne_id)] = domain
    return domain_map


def _is_sorted_alarm_cache_header(record):
    return isinstance(record, dict) and record.get("cache_type") == "fault_grouping.sorted_alarms.v1"


def _unwrap_alarm(record):
    if isinstance(record, dict) and isinstance(record.get("alarm"), dict):
        return record["alarm"], record
    return record, None


def _first_non_empty(record, fields):
    if not isinstance(record, dict):
        return ""

    for field in fields:
        value = _normalize_text(record.get(field, ""))
        if value:
            return value
    return ""


def _get_group_id(alarm, wrapper, group_field):
    group_id = _normalize_text(alarm.get(group_field, "")) if isinstance(alarm, dict) else ""
    if group_id:
        return group_id
    if isinstance(wrapper, dict):
        return _normalize_text(wrapper.get(group_field, ""))
    return ""


def _get_alarm_source(alarm, wrapper):
    alarm_source = _normalize_text(alarm.get("告警源", "")) if isinstance(alarm, dict) else ""
    if alarm_source:
        return alarm_source
    if isinstance(wrapper, dict):
        return _normalize_text(wrapper.get("alarm_source", ""))
    return ""


def _get_device_type(alarm, wrapper, device_fields, ne_domain_map):
    alarm_source = _get_alarm_source(alarm, wrapper)
    if alarm_source and ne_domain_map:
        device_type = ne_domain_map.get(alarm_source, "")
        if device_type:
            return device_type

    device_type = _first_non_empty(alarm, device_fields)
    if device_type:
        return device_type
    if isinstance(wrapper, dict):
        device_type = _first_non_empty(wrapper, device_fields)
        if device_type:
            return device_type

    return ""


def _append_unique(values, value):
    if value and value not in values:
        values.append(value)


def _build_group_record(group_id):
    return {
        "故障组ID": group_id,
        "alarms": [],
        "_device_type_keys": set(),
        "_device_types": [],
        "_site_ids": [],
        "_alarm_sources": [],
    }


def group_alarms(
    alarms_input,
    *,
    group_field=DEFAULT_GROUP_FIELD,
    device_fields=None,
    excluded_device_types=None,
    ne_graph=NE_GRAPH_JSON,
    show_progress=False,
    progress_interval=100000,
):
    device_fields = list(device_fields or DEFAULT_DEVICE_FIELDS)
    excluded_device_type_keys = {
        _normalize_key(value) for value in (excluded_device_types or []) if _normalize_text(value)
    }
    ne_domain_map = _load_ne_domain_map(ne_graph)

    groups = OrderedDict()
    stats = {
        "input_count": 0,
        "grouped_alarm_count": 0,
        "skipped_no_group_id": 0,
        "group_count": 0,
        "excluded_group_count": 0,
        "output_group_count": 0,
    }
    progress_interval = max(int(progress_interval or 0), 0)
    last_progress_time = time.time()
    if show_progress:
        print(f"⏳ 开始读取告警输入: {alarms_input}", file=sys.stderr, flush=True)

    def _maybe_print_progress():
        nonlocal last_progress_time
        if not show_progress:
            return

        now = time.time()
        should_print_by_count = progress_interval > 0 and stats["input_count"] % progress_interval == 0
        should_print_by_time = now - last_progress_time >= 5.0
        if should_print_by_count or should_print_by_time:
            print(
                "  已读取 "
                f"{stats['input_count']} 条，"
                f"有效分组告警 {stats['grouped_alarm_count']} 条，"
                f"当前故障组 {len(groups)} 个",
                file=sys.stderr,
                flush=True,
            )
            last_progress_time = now

    for record in stream_alarm_inputs(alarms_input, show_progress=show_progress):
        stats["input_count"] += 1
        if _is_sorted_alarm_cache_header(record):
            _maybe_print_progress()
            continue

        alarm, wrapper = _unwrap_alarm(record)
        if not isinstance(alarm, dict):
            stats["skipped_no_group_id"] += 1
            _maybe_print_progress()
            continue

        group_id = _get_group_id(alarm, wrapper, group_field)
        if not group_id:
            stats["skipped_no_group_id"] += 1
            _maybe_print_progress()
            continue

        group = groups.setdefault(group_id, _build_group_record(group_id))
        group["alarms"].append(alarm)
        stats["grouped_alarm_count"] += 1

        device_type = _get_device_type(alarm, wrapper, device_fields, ne_domain_map)
        if device_type:
            group["_device_type_keys"].add(_normalize_key(device_type))
            _append_unique(group["_device_types"], device_type)

        site_id = _normalize_text(alarm.get("站点ID", ""))
        if not site_id and isinstance(wrapper, dict):
            site_id = _normalize_text(wrapper.get("site_id", ""))
        _append_unique(group["_site_ids"], site_id)

        alarm_source = _get_alarm_source(alarm, wrapper)
        _append_unique(group["_alarm_sources"], alarm_source)

        _maybe_print_progress()

    stats["group_count"] = len(groups)
    if show_progress:
        print(
            "✅ 告警读取完成: "
            f"读取 {stats['input_count']} 条，"
            f"有效分组告警 {stats['grouped_alarm_count']} 条，"
            f"跳过无故障组ID {stats['skipped_no_group_id']} 条，"
            f"形成故障组 {stats['group_count']} 个",
            file=sys.stderr,
            flush=True,
        )

    output_groups = []
    for group in groups.values():
        if excluded_device_type_keys and group["_device_type_keys"] & excluded_device_type_keys:
            stats["excluded_group_count"] += 1
            continue

        record = {
            "故障组ID": group["故障组ID"],
            "alarm_count": len(group["alarms"]),
            "alarms": group["alarms"],
        }
        if group["_device_types"]:
            record["device_types"] = group["_device_types"]
        if group["_site_ids"]:
            record["site_ids"] = group["_site_ids"]
        if group["_alarm_sources"]:
            record["alarm_sources"] = group["_alarm_sources"]
        output_groups.append(record)

    stats["output_group_count"] = len(output_groups)
    return output_groups, stats


def write_jsonl(groups, output_path):
    with open(output_path, "w", encoding="utf-8") as fw:
        for group in groups:
            fw.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
            fw.write("\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="按告警流中的非空故障组ID聚合告警，输出每行一个故障组的 JSONL"
    )
    parser.add_argument("alarms", help="告警流输入，支持 jsonl/csv/zip/目录，与 fault_grouping/match_rules.py 一致")
    parser.add_argument("output", help="输出 JSONL 文件")
    parser.add_argument(
        "--group-field",
        default=DEFAULT_GROUP_FIELD,
        help=f"分组字段名，默认: {DEFAULT_GROUP_FIELD}",
    )
    parser.add_argument(
        "--device-field",
        action="append",
        default=None,
        help=(
            "告警字段兜底用的设备类型字段名；可重复传入，也支持逗号分隔。"
            "默认自动尝试 domain/专业/设备类型/网络类型/网元类型等字段。"
            "脚本会优先使用 ne_graph 中 告警源 -> domain 的结果"
        ),
    )
    parser.add_argument(
        "--exclude-device-type",
        "--forbid-device-type",
        dest="exclude_device_types",
        action="append",
        default=[],
        help="如果故障组中出现该设备类型则不输出；可重复传入，也支持逗号分隔，如 Data,Ran,Microwave",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=(
            "ne_graph.json 文件；默认: "
            f"{resource_display('ne_graph.json')}。优先用 告警源 -> domain 判断设备类型，找不到再回退到告警字段"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭读取输入时的进度显示",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100000,
        help="进度日志按读取条数输出的间隔，默认 100000；设为 0 表示只按时间输出",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device_fields = _parse_device_fields(args.device_field)
    excluded_device_types = _split_csv_values(args.exclude_device_types)

    groups, stats = group_alarms(
        args.alarms,
        group_field=args.group_field,
        device_fields=device_fields,
        excluded_device_types=excluded_device_types,
        ne_graph=args.ne_graph,
        show_progress=not args.no_progress,
        progress_interval=args.progress_interval,
    )
    write_jsonl(groups, args.output)

    stats["output"] = args.output
    stats["group_field"] = args.group_field
    stats["ne_graph"] = args.ne_graph
    stats["device_fields"] = device_fields
    stats["excluded_device_types"] = excluded_device_types
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
