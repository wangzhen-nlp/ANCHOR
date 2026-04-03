import json
import os

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

from alarm_inputs import build_ne_to_site_map, stream_alarm_inputs


TIME_FIELDS = (
    "告警首次发生时间",
    "告警最后发生时间",
    "告警清除时间",
    "告警首次采集时间",
    "告警首次入库时间",
)


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def _normalize_unique_list(values):
    seen = set()
    result = []
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _parse_group_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_parse_group_ids(item))
        return _normalize_unique_list(result)

    text = _normalize_text(value)
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                parsed = parsed.values()
            return _parse_group_ids(parsed)

    normalized_text = (
        text.replace("，", ",")
        .replace(";", ",")
        .replace("；", ",")
        .replace("|", ",")
    )
    return _normalize_unique_list(normalized_text.split(","))


def _parse_time(value):
    text = _normalize_text(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_time(dt_obj):
    if dt_obj is None:
        return ""
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field):
    site_id = _normalize_text(alarm.get(site_field, ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get(source_field, ""))
    if not alarm_source:
        return ""

    return _normalize_text(ne_to_site.get(alarm_source, ""))


def _alarm_sort_key(alarm):
    first_occurrence = _parse_time(alarm.get("告警首次发生时间"))
    last_occurrence = _parse_time(alarm.get("告警最后发生时间"))
    return (
        first_occurrence is None,
        first_occurrence or datetime.max,
        last_occurrence is None,
        last_occurrence or datetime.max,
        _normalize_text(alarm.get("告警编码ID", "")),
        _normalize_text(alarm.get("告警标题", "")),
        _normalize_text(alarm.get("告警源", "")),
    )


def _build_group_record(group_id, state):
    alarms = sorted(state["告警列表"], key=_alarm_sort_key)
    record = {
        "故障组ID": group_id,
        "告警数": len(alarms),
        "工单号数": len(state["工单号集合"]),
        "工单号列表": sorted(state["工单号集合"]),
        "站点数": len(state["站点ID集合"]),
        "站点ID列表": sorted(state["站点ID集合"]),
        "告警源数": len(state["告警源集合"]),
        "告警源列表": sorted(state["告警源集合"]),
        "告警标题数": len(state["告警标题集合"]),
        "告警标题列表": sorted(state["告警标题集合"]),
    }

    for field_name in TIME_FIELDS:
        parsed_times = sorted(
            dt_obj
            for dt_obj in (
                _parse_time(alarm.get(field_name))
                for alarm in alarms
            )
            if dt_obj is not None
        )
        record[f"{field_name}最早"] = _format_time(parsed_times[0]) if parsed_times else ""
        record[f"{field_name}最晚"] = _format_time(parsed_times[-1]) if parsed_times else ""

    # 保留完整告警列表，便于页面直接加载后查看完整字段。
    record["告警列表"] = alarms
    return record


def build_alarm_group_reference_json(
    alarm_input,
    group_field="故障组ID",
    ticket_field="工单号",
    site_field="站点ID",
    source_field="告警源",
    alarm_name_field="告警标题",
    ne_graph_file=None,
    show_progress=True,
):
    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    grouped_state = defaultdict(
        lambda: {
            "工单号集合": set(),
            "站点ID集合": set(),
            "告警源集合": set(),
            "告警标题集合": set(),
            "告警列表": [],
        }
    )

    for alarm in stream_alarm_inputs(alarm_input, show_progress=show_progress):
        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        if not group_ids:
            continue

        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        alarm_source = _normalize_text(alarm.get(source_field, ""))
        alarm_name = _normalize_text(alarm.get(alarm_name_field, ""))

        enriched_alarm = dict(alarm)
        if resolved_site_id:
            enriched_alarm["关联站点ID"] = resolved_site_id

        for group_id in group_ids:
            state = grouped_state[group_id]
            if ticket_id:
                state["工单号集合"].add(ticket_id)
            if resolved_site_id:
                state["站点ID集合"].add(resolved_site_id)
            if alarm_source:
                state["告警源集合"].add(alarm_source)
            if alarm_name:
                state["告警标题集合"].add(alarm_name)
            state["告警列表"].append(enriched_alarm)

    return {
        "故障组ID": {
            group_id: _build_group_record(group_id, state)
            for group_id, state in sorted(grouped_state.items())
        }
    }


def _load_base_reference_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("基础 JSON 顶层必须是对象")
    return data


def main():
    parser = ArgumentParser(
        description="从告警流中提取“故障组ID -> 多条关联告警详情”的 JSON，供页面直接加载"
    )
    parser.add_argument(
        "alarms",
        help="告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="alarm_group_reference.json",
        help="输出 JSON 文件，默认: alarm_group_reference.json",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警中的故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="告警中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--site-field",
        default="站点ID",
        help="告警中的站点字段名，默认: 站点ID",
    )
    parser.add_argument(
        "--source-field",
        default="告警源",
        help="告警中的设备/告警源字段名，默认: 告警源",
    )
    parser.add_argument(
        "--alarm-name-field",
        default="告警标题",
        help="告警名称字段名，默认: 告警标题",
    )
    parser.add_argument(
        "--ne-graph",
        default="ne_graph.json",
        help="用于通过告警源回填站点ID的 ne_graph 文件，默认: ne_graph.json",
    )
    parser.add_argument(
        "--base-json",
        help="已有的关联明细 JSON；提供后会把提取出的“故障组ID”信息并入该 JSON 再输出",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭读取进度显示",
    )

    args = parser.parse_args()

    result = build_alarm_group_reference_json(
        alarm_input=args.alarms,
        group_field=args.group_field,
        ticket_field=args.ticket_field,
        site_field=args.site_field,
        source_field=args.source_field,
        alarm_name_field=args.alarm_name_field,
        ne_graph_file=args.ne_graph,
        show_progress=not args.no_progress,
    )

    output_data = {}
    if args.base_json:
        output_data = _load_base_reference_json(args.base_json)
    output_data["故障组ID"] = result["故障组ID"]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"已输出 {len(result['故障组ID'])} 个故障组ID 的关联明细到: {args.output}")


if __name__ == "__main__":
    main()
