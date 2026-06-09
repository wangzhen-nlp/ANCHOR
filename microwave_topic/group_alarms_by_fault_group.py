#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按告警流中的故障组ID聚合告警，并输出 JSONL。"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_inputs import stream_alarm_inputs


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
    device_type = _first_non_empty(alarm, device_fields)
    if device_type:
        return device_type

    if isinstance(wrapper, dict):
        device_type = _first_non_empty(wrapper, device_fields)
        if device_type:
            return device_type

    alarm_source = _get_alarm_source(alarm, wrapper)
    if alarm_source and ne_domain_map:
        return ne_domain_map.get(alarm_source, "")
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
    ne_graph=None,
    show_progress=False,
):
    device_fields = list(device_fields or DEFAULT_DEVICE_FIELDS)
    excluded_device_type_keys = {
        _normalize_key(value) for value in (excluded_device_types or []) if _normalize_text(value)
    }
    ne_domain_map = _load_ne_domain_map(ne_graph)

    groups = OrderedDict()
    stats = {
        "input_count": 0,
        "skipped_no_group_id": 0,
        "group_count": 0,
        "excluded_group_count": 0,
        "output_group_count": 0,
    }

    for record in stream_alarm_inputs(alarms_input, show_progress=show_progress):
        stats["input_count"] += 1
        if _is_sorted_alarm_cache_header(record):
            continue

        alarm, wrapper = _unwrap_alarm(record)
        if not isinstance(alarm, dict):
            stats["skipped_no_group_id"] += 1
            continue

        group_id = _get_group_id(alarm, wrapper, group_field)
        if not group_id:
            stats["skipped_no_group_id"] += 1
            continue

        group = groups.setdefault(group_id, _build_group_record(group_id))
        group["alarms"].append(alarm)

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

    stats["group_count"] = len(groups)

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
            "设备类型字段名；可重复传入，也支持逗号分隔。"
            "默认自动尝试 domain/专业/设备类型/网络类型/网元类型等字段"
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
        default="",
        help="可选 ne_graph.json；当告警本身没有设备类型字段时，用 告警源 -> domain 补齐过滤依据",
    )
    parser.add_argument("--show-progress", action="store_true", help="读取输入时显示进度")
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
        show_progress=args.show_progress,
    )
    write_jsonl(groups, args.output)

    stats["output"] = args.output
    stats["group_field"] = args.group_field
    stats["device_fields"] = device_fields
    stats["excluded_device_types"] = excluded_device_types
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
