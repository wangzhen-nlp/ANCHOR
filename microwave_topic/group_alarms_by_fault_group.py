#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按告警流中的故障组ID聚合告警，并输出 JSONL。"""

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_inputs import list_alarm_filepaths, stream_alarm_file
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


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
    ne_graph = _load_json_object(ne_graph_path, "ne_graph", warn_if_missing=True)
    return _build_ne_domain_map(ne_graph)


def _load_json_object(path, label, warn_if_missing=False):
    if not path:
        return {}

    if not Path(path).exists():
        if warn_if_missing:
            print(f"⚠️ {label} 文件不存在，跳过对应补充信息: {path}", file=sys.stderr)
        return {}

    with open(path, "r", encoding="utf-8") as fr:
        data = json.load(fr)

    if not isinstance(data, dict):
        raise ValueError(f"{label} 顶层必须是对象: {path}")
    return data


def _build_ne_domain_map(ne_graph):
    if not isinstance(ne_graph, dict):
        return {}

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


def _get_site_id(alarm, wrapper, alarm_source, ne_graph_data):
    site_id = _normalize_text(alarm.get("站点ID", "")) if isinstance(alarm, dict) else ""
    if site_id:
        return site_id
    if isinstance(wrapper, dict):
        site_id = _normalize_text(wrapper.get("site_id", ""))
        if site_id:
            return site_id
    ne_info = ne_graph_data.get(alarm_source, {}) if alarm_source and isinstance(ne_graph_data, dict) else {}
    if isinstance(ne_info, dict):
        return _normalize_text(ne_info.get("site_id", ""))
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
        "_events": [],
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
    min_alarm_count=1,
    min_device_count=0,
    min_site_count=0,
    ne_graph=NE_GRAPH_JSON,
    site_graph=SITE_GRAPH_JSON,
    visual_output=True,
    show_progress=False,
):
    device_fields = list(device_fields or DEFAULT_DEVICE_FIELDS)
    excluded_device_type_keys = {
        _normalize_key(value) for value in (excluded_device_types or []) if _normalize_text(value)
    }
    ne_graph_data = _load_json_object(ne_graph, "ne_graph", warn_if_missing=True)
    ne_domain_map = _build_ne_domain_map(ne_graph_data)
    site_graph_data = (
        _load_json_object(site_graph, "site_graph", warn_if_missing=True)
        if visual_output
        else {}
    )

    groups = OrderedDict()
    stats = {
        "input_count": 0,
        "grouped_alarm_count": 0,
        "skipped_no_group_id": 0,
        "group_count": 0,
        "excluded_group_count": 0,
        "below_min_alarm_count_group_count": 0,
        "below_min_device_count_group_count": 0,
        "below_min_site_count_group_count": 0,
        "output_group_count": 0,
    }

    filepaths = list_alarm_filepaths(alarms_input)
    total_files = len(filepaths)
    for file_index, filepath in enumerate(filepaths, start=1):
        before_input_count = stats["input_count"]
        before_grouped_alarm_count = stats["grouped_alarm_count"]
        before_skipped_no_group_id = stats["skipped_no_group_id"]
        if show_progress:
            print(f"⏳ [{file_index}/{total_files}] 开始读取文件: {filepath}", file=sys.stderr, flush=True)

        for record in stream_alarm_file(filepath, show_progress=False):
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
            group["_events"].append({"alarm": alarm, "wrapper": wrapper})
            stats["grouped_alarm_count"] += 1

            device_type = _get_device_type(alarm, wrapper, device_fields, ne_domain_map)
            if device_type:
                group["_device_type_keys"].add(_normalize_key(device_type))
                _append_unique(group["_device_types"], device_type)

            alarm_source = _get_alarm_source(alarm, wrapper)
            site_id = _get_site_id(alarm, wrapper, alarm_source, ne_graph_data)
            _append_unique(group["_site_ids"], site_id)

            _append_unique(group["_alarm_sources"], alarm_source)

        if show_progress:
            print(
                f"✅ [{file_index}/{total_files}] 文件读取完成: {filepath}，"
                f"本文件读取 {stats['input_count'] - before_input_count} 条，"
                f"有效分组告警 {stats['grouped_alarm_count'] - before_grouped_alarm_count} 条，"
                f"跳过无故障组ID {stats['skipped_no_group_id'] - before_skipped_no_group_id} 条，"
                f"累计故障组 {len(groups)} 个",
                file=sys.stderr,
                flush=True,
            )

    stats["group_count"] = len(groups)
    if show_progress:
        print(
            "✅ 全部文件读取完成: "
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

        if len(group["alarms"]) < min_alarm_count:
            stats["below_min_alarm_count_group_count"] += 1
            continue

        if min_device_count > 0 and len(group["_alarm_sources"]) < min_device_count:
            stats["below_min_device_count_group_count"] += 1
            continue

        if min_site_count > 0 and len(group["_site_ids"]) < min_site_count:
            stats["below_min_site_count_group_count"] += 1
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
        if visual_output:
            record.update(_build_visual_fields(group, ne_graph_data, ne_domain_map, site_graph_data))
        output_groups.append(record)

    stats["output_group_count"] = len(output_groups)
    return output_groups, stats


def _parse_datetime_ts(value):
    text = _normalize_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("T", " ")).timestamp()
    except ValueError:
        return None


def _format_ts(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _get_alarm_ts(alarm, wrapper):
    if isinstance(wrapper, dict):
        raw_ts = wrapper.get("ts")
        if raw_ts not in (None, ""):
            try:
                return float(raw_ts)
            except (TypeError, ValueError):
                pass
    for field_name in (
        "告警首次发生时间",
        "告警发生时间",
        "发生时间",
        "首次发生时间",
        "alarm_time",
        "time",
    ):
        ts = _parse_datetime_ts(alarm.get(field_name, ""))
        if ts is not None:
            return ts
    return None


def _first_alarm_field(alarm, fields):
    for field_name in fields:
        value = _normalize_text(alarm.get(field_name, ""))
        if value:
            return value
    return ""


def _get_site_context(site_id, ne_info, site_graph_data):
    site_info = site_graph_data.get(site_id, {}) if site_id and isinstance(site_graph_data, dict) else {}
    if not isinstance(site_info, dict):
        site_info = {}
    return {
        "site_name": (
            _normalize_text(ne_info.get("site_name", ""))
            or _normalize_text(site_info.get("site_name", ""))
            or _normalize_text(site_info.get("name", ""))
        ),
        "site_type": _normalize_text(ne_info.get("site_type", "")) or _normalize_text(site_info.get("site_type", "")),
        "region_id": _normalize_text(ne_info.get("region_id", "")) or _normalize_text(site_info.get("region_id", "")),
        "longitude": ne_info.get("longitude", site_info.get("longitude", site_info.get("lon", site_info.get("lng", "")))),
        "latitude": ne_info.get("latitude", site_info.get("latitude", site_info.get("lat", ""))),
    }


def _build_visual_link_info(ne_id, group_ne_ids, ne_graph_data):
    raw_ne_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    raw_links = raw_ne_info.get("link", {}) if isinstance(raw_ne_info, dict) else {}
    if not isinstance(raw_links, dict):
        return {}

    group_ne_id_set = set(group_ne_ids)
    links = {}
    for target_ne_id, link_meta in raw_links.items():
        if target_ne_id == ne_id or target_ne_id not in group_ne_id_set:
            continue
        if isinstance(link_meta, dict):
            connection_types = sorted(str(key) for key in link_meta.keys())
            topologies = sorted({str(value) for value in link_meta.values() if value})
        else:
            connection_types = [str(link_meta)]
            topologies = []
        links[target_ne_id] = {
            "connection_type": ",".join(connection_types),
            "distance": "",
            "topology": ",".join(topologies),
            "time_window": "",
            "left_alarm": {},
            "right_alarm": {},
        }
    return links


def _build_visual_fields(group, ne_graph_data, ne_domain_map, site_graph_data):
    group_id = group["故障组ID"]
    symptoms = []
    ne_alarms = OrderedDict()
    site_ids = []
    ne_ids = []

    for index, event in enumerate(group.get("_events", []), start=1):
        alarm = event.get("alarm", {})
        wrapper = event.get("wrapper")
        alarm_source = _get_alarm_source(alarm, wrapper)
        site_id = _get_site_id(alarm, wrapper, alarm_source, ne_graph_data)
        ts = _get_alarm_ts(alarm, wrapper)
        alarm_title = _first_alarm_field(alarm, ("告警标题", "alarm_title", "alarm_type", "title"))
        alarm_id = _first_alarm_field(alarm, ("告警编码ID", "告警ID", "alarm_id", "id")) or f"{group_id}-{index}"
        device_type = _get_device_type(alarm, wrapper, DEFAULT_DEVICE_FIELDS, ne_domain_map)

        symptom = {
            "node": site_id,
            "alarm": alarm_title,
            "alarm_source": alarm_source,
            "ts": ts,
            "eid": alarm_id,
            "matched_role": "alarm_group",
            "matched_rule": "alarm_group_id_rule",
            "matched_role_key": "alarm_group",
            "故障组ID": group_id,
            "工单号": _normalize_text(alarm.get("工单号", "")),
            "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
            "domain": device_type,
        }
        symptoms.append(symptom)

        if site_id:
            _append_unique(site_ids, site_id)
        if alarm_source:
            _append_unique(ne_ids, alarm_source)
            node_alarm = {
                "alarm_id": alarm_id,
                "alarm_type": alarm_title,
                "alarm_time": _format_ts(ts),
                "alarm_clear_time": symptom["告警清除时间"],
                "domain": device_type,
                "site_id": site_id,
                "matched_role": "alarm_group",
                "matched_rule": "alarm_group_id_rule",
                "matched_role_key": "alarm_group",
                "工单号": symptom["工单号"],
                "故障组ID": group_id,
                "ts": ts,
            }
            ne_alarms.setdefault(alarm_source, []).append(node_alarm)

    ne_info_output = OrderedDict()
    for ne_id in ne_ids:
        raw_ne_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
        if not isinstance(raw_ne_info, dict):
            raw_ne_info = {}
        alarms = ne_alarms.get(ne_id, [])
        site_id = (
            _normalize_text(raw_ne_info.get("site_id", ""))
            or (_normalize_text(alarms[0].get("site_id", "")) if alarms else "")
        )
        site_context = _get_site_context(site_id, raw_ne_info, site_graph_data)
        ne_info_output[ne_id] = {
            "link": _build_visual_link_info(ne_id, ne_ids, ne_graph_data),
            "group": group_id,
            "name": raw_ne_info.get("name", ne_id),
            "site_id": site_id,
            "site_name": site_context["site_name"],
            "site_type": site_context["site_type"],
            "type": str(raw_ne_info.get("type", "")).upper(),
            "network_type": str(raw_ne_info.get("network_type", "")).upper(),
            "manufacturer": str(raw_ne_info.get("manufacturer", "")).upper(),
            "running_status": raw_ne_info.get("running_status", raw_ne_info.get("status", "")),
            "domain": str(raw_ne_info.get("domain", "") or (alarms[0].get("domain", "") if alarms else "")).upper(),
            "region_id": site_context["region_id"],
            "longitude": site_context["longitude"],
            "latitude": site_context["latitude"],
            "alarm": alarms,
        }

    timestamps = [symptom["ts"] for symptom in symptoms if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None
    role_mapping = {"associated_site": sorted(site_ids)}
    match_info = {
        "uuid": group_id,
        "rule": "alarm_group_id_rule",
        "merged_rules": ["alarm_group_id_rule"],
        "related_group_uuids": [],
        "inferred_roots": {},
        "role_mapping": role_mapping,
        "uses_missing_topology": False,
        "missing_topology_edges": [],
    }
    return {
        "uuid": group_id,
        "rule": "alarm_group_id_rule",
        "merged_rules": ["alarm_group_id_rule"],
        "related_group_uuids": [],
        "role_mapping": role_mapping,
        "symptoms": symptoms,
        "match_info": match_info,
        "ne_info": ne_info_output,
        "group_info": {
            group_id: {
                "ne_list": sorted(ne_ids),
                "site_list": sorted(site_ids),
            }
        },
        "group_anchor_ts": group_anchor_ts,
        "group_anchor_time": _format_ts(group_anchor_ts),
    }


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
        "--min-alarm-count",
        dest="min_alarm_count",
        type=int,
        default=1,
        help="故障组内告警数量至少达到该值才输出，默认 1；--min-count 是兼容旧参数的别名",
    )
    parser.add_argument(
        "--min-device-count",
        type=int,
        default=0,
        help="故障组内非空唯一告警源数量至少达到该值才输出，默认 0 表示不限制",
    )
    parser.add_argument(
        "--min-site-count",
        type=int,
        default=0,
        help="故障组内非空唯一站点数量至少达到该值才输出，默认 0 表示不限制",
    )
    parser.add_argument(
        "--min-count",
        dest="min_alarm_count",
        type=int,
        help=argparse.SUPPRESS,
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
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=(
            "site_graph.json 文件；默认: "
            f"{resource_display('site_graph.json')}。用于补充可视化输出中的站点名称和经纬度"
        ),
    )
    parser.add_argument(
        "--no-visual-output",
        action="store_true",
        help="不追加故障组总览页和 NE 传播图可识别的可视化字段",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭读取输入时的进度显示",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device_fields = _parse_device_fields(args.device_field)
    excluded_device_types = _split_csv_values(args.exclude_device_types)
    if args.min_alarm_count < 1:
        parser.error("--min-alarm-count 必须大于等于 1")
    if args.min_device_count < 0:
        parser.error("--min-device-count 不能小于 0")
    if args.min_site_count < 0:
        parser.error("--min-site-count 不能小于 0")

    groups, stats = group_alarms(
        args.alarms,
        group_field=args.group_field,
        device_fields=device_fields,
        excluded_device_types=excluded_device_types,
        min_alarm_count=args.min_alarm_count,
        min_device_count=args.min_device_count,
        min_site_count=args.min_site_count,
        ne_graph=args.ne_graph,
        site_graph=args.site_graph,
        visual_output=not args.no_visual_output,
        show_progress=not args.no_progress,
    )
    write_jsonl(groups, args.output)

    stats["output"] = args.output
    stats["group_field"] = args.group_field
    stats["ne_graph"] = args.ne_graph
    stats["site_graph"] = args.site_graph
    stats["visual_output"] = not args.no_visual_output
    stats["device_fields"] = device_fields
    stats["excluded_device_types"] = excluded_device_types
    stats["min_alarm_count"] = args.min_alarm_count
    stats["min_device_count"] = args.min_device_count
    stats["min_site_count"] = args.min_site_count
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
