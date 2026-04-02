import json

from argparse import ArgumentParser
from collections import defaultdict

from alarm_inputs import stream_alarm_inputs
from compute_ticket_site_recall import (
    _compute_ticket_recalls,
    _load_ticket_sites,
    _normalize_site_list,
    _normalize_text,
)


def _extract_group_id(group_record):
    match_info = group_record.get("match_info", {})
    return _normalize_text(match_info.get("uuid") or group_record.get("uuid", ""))


def _extract_group_sites(group_record, group_id):
    group_info = group_record.get("group_info", {})
    collected_sites = []

    if isinstance(group_info, dict):
        if group_id and group_id in group_info and isinstance(group_info[group_id], dict):
            collected_sites.extend(group_info[group_id].get("site_list", []))
        else:
            for group_entry in group_info.values():
                if isinstance(group_entry, dict):
                    collected_sites.extend(group_entry.get("site_list", []))

    # 兜底：如果 group_info 缺失或不完整，就从 symptoms 里的 node 再补一遍。
    if not collected_sites:
        for symptom in group_record.get("symptoms", []):
            collected_sites.append(symptom.get("node", ""))

    return _normalize_site_list(collected_sites)


def _extract_ticket_ids(group_record, ticket_field):
    ticket_ids = []
    for symptom in group_record.get("symptoms", []):
        ticket_id = _normalize_text(symptom.get(ticket_field, ""))
        if ticket_id:
            ticket_ids.append(ticket_id)
    return _normalize_site_list(ticket_ids)


def _count_ticket_occurrences_in_group(group_record, ticket_field):
    counts = defaultdict(int)
    for symptom in group_record.get("symptoms", []):
        ticket_id = _normalize_text(symptom.get(ticket_field, ""))
        if ticket_id:
            counts[ticket_id] += 1
    return counts


def _build_group_output_indexes(group_output_input, ticket_field):
    ticket_to_groups = defaultdict(set)
    group_to_sites = defaultdict(set)
    ticket_occurrence_counts = defaultdict(int)

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id:
            continue

        group_sites = _extract_group_sites(group_record, group_id)
        if group_sites:
            group_to_sites[group_id].update(group_sites)

        ticket_ids = _extract_ticket_ids(group_record, ticket_field)
        for ticket_id in ticket_ids:
            ticket_to_groups[ticket_id].add(group_id)
        for ticket_id, count in _count_ticket_occurrences_in_group(group_record, ticket_field).items():
            ticket_occurrence_counts[ticket_id] += count

    return ticket_to_groups, group_to_sites, ticket_occurrence_counts


def _count_ticket_occurrences_in_alarms(alarm_input, ticket_field):
    ticket_alarm_counts = defaultdict(int)
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id:
            ticket_alarm_counts[ticket_id] += 1
    return ticket_alarm_counts


def main():
    parser = ArgumentParser(description="基于 match_rules.py 输出故障组计算工单站点召回率")
    parser.add_argument(
        "group_output",
        help="故障组输出输入，支持 jsonl/zip/目录，与 match_rules.py 输出格式一致",
    )
    parser.add_argument(
        "--ticket-sites",
        required=True,
        help="工单站点映射 JSON，格式为 {工单号: [站点列表]}",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="故障组 symptoms 中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--alarms",
        help="原始告警输入（可选）。提供后，工单是否纳入分母将与 compute_ticket_site_recall.py 一致，按原始告警中是否出现来判断。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="group_output_ticket_recall.json",
        help="输出明细 JSON 文件，默认: group_output_ticket_recall.json",
    )

    args = parser.parse_args()

    ticket_sites = _load_ticket_sites(args.ticket_sites)
    if not ticket_sites:
        print("❌ 工单站点映射为空，无法计算召回率")
        return

    ticket_to_groups, group_to_sites, ticket_occurrence_counts = _build_group_output_indexes(
        args.group_output,
        args.ticket_field,
    )

    if args.alarms:
        ticket_alarm_counts = _count_ticket_occurrences_in_alarms(args.alarms, args.ticket_field)
        denominator_source = "alarms"
    else:
        ticket_alarm_counts = dict(ticket_occurrence_counts)
        denominator_source = "group_output"

    details, average_recall, evaluated_count = _compute_ticket_recalls(
        ticket_sites,
        ticket_to_groups,
        group_to_sites,
        ticket_alarm_counts,
    )

    for detail in details:
        ticket_id = detail.get("ticket_id", "")
        detail["ticket_occurrence_count"] = ticket_occurrence_counts.get(ticket_id, 0)
        detail.pop("ticket_alarm_count", None)

    result = {
        "ticket_count": evaluated_count,
        "average_recall": average_recall,
        "denominator_source": denominator_source,
        "details": details,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"工单数: {evaluated_count}")
    print(f"平均召回率: {average_recall:.6f}")
    print(f"分母口径来源: {denominator_source}")
    print(f"明细已输出到: {args.output}")


if __name__ == "__main__":
    main()
