import json
import os

from argparse import ArgumentParser
from collections import defaultdict

from alarm_types import OFFLINE_ALARMS
from alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from compute_ticket_site_recall import (
    _build_ticket_sites_from_alarms,
    _load_ticket_sites,
    _normalize_text,
    _parse_group_ids,
    _resolve_alarm_site_id,
)
from ticket_recall_v2_utils import (
    build_site_alarm_map_for_sites,
    build_ticket_site_count_distribution,
    build_unrecalled_visualization_cases,
    dedupe_alarm_records,
    derive_case_jsonl_output_path,
    load_upper_bound_index,
    write_jsonl_records,
)


OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)


def _build_alarm_record(alarm, resolved_site_id, group_id):
    record = {
        "故障组ID": group_id,
        "告警编码ID": _normalize_text(alarm.get("告警编码ID", "")),
        "告警标题": _normalize_text(alarm.get("告警标题", "")),
        "工单号": _normalize_text(alarm.get("工单号", "")),
        "站点ID": _normalize_text(alarm.get("站点ID", "")),
        "关联站点ID": resolved_site_id,
        "告警源": _normalize_text(alarm.get("告警源", "")),
        "告警首次发生时间": _normalize_text(alarm.get("告警首次发生时间", "")),
        "告警最后发生时间": _normalize_text(alarm.get("告警最后发生时间", "")),
        "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
    }
    return {key: value for key, value in record.items() if value}


def _build_ticket_group_index_for_eligible(alarm_input, eligible_ticket_ids, ticket_field, group_field):
    ticket_to_groups = defaultdict(set)
    ticket_alarm_counts = defaultdict(int)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id not in eligible_ticket_ids:
            continue

        ticket_alarm_counts[ticket_id] += 1
        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        if group_ids:
            ticket_to_groups[ticket_id].update(group_ids)

    return ticket_to_groups, ticket_alarm_counts


def _build_group_alarm_indexes(alarm_input, relevant_group_ids, ne_to_site, group_field):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))

    if not relevant_group_ids:
        return group_to_sites, group_to_site_alarms

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        group_ids = [
            group_id
            for group_id in _parse_group_ids(alarm.get(group_field, ""))
            if group_id in relevant_group_ids
        ]
        if not group_ids:
            continue

        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site)
        if not resolved_site_id:
            continue

        for group_id in group_ids:
            group_to_sites[group_id].add(resolved_site_id)
            group_to_site_alarms[group_id][resolved_site_id].append(
                _build_alarm_record(alarm, resolved_site_id, group_id)
            )

    return group_to_sites, group_to_site_alarms


def _merge_group_site_alarms(group_ids, group_to_site_alarms):
    merged = defaultdict(list)
    for group_id in group_ids:
        for site_id, alarms in group_to_site_alarms.get(group_id, {}).items():
            merged[site_id].extend(alarms)
    return {
        site_id: dedupe_alarm_records(alarms)
        for site_id, alarms in sorted(merged.items())
    }


def _alarm_record_is_offline(record):
    if not isinstance(record, dict):
        return False
    alarm_name = _normalize_text(record.get("alarm", ""))
    if alarm_name and alarm_name in OFFLINE_ALARM_SET:
        return True
    alarm_title = _normalize_text(record.get("告警标题", ""))
    return bool(alarm_title and alarm_title in OFFLINE_ALARM_SET)


def _site_alarm_map_contains_offline(site_alarm_map):
    for alarms in site_alarm_map.values():
        if not isinstance(alarms, list):
            continue
        for record in alarms:
            if _alarm_record_is_offline(record):
                return True
    return False


def compute_ticket_site_recall_v2(
    alarm_input,
    upper_bound_file,
    ticket_sites_file=None,
    ticket_field="工单号",
    group_field="故障组ID",
    ne_graph_file=None,
    output_file=None,
    case_jsonl_output_file=None,
    strict=False,
):
    upper_bound_index = load_upper_bound_index(upper_bound_file)
    eligible_ticket_ids = {
        ticket_id
        for ticket_id, item in upper_bound_index.items()
        if item.get("fully_associable")
    }
    if not eligible_ticket_ids:
        raise ValueError("召回率上限结果里没有“可完整关联”的工单")

    if ticket_sites_file:
        ticket_sites = _load_ticket_sites(ticket_sites_file)
        ticket_site_source = "ticket_sites"
    else:
        ticket_sites = _build_ticket_sites_from_alarms(alarm_input, ticket_field, ne_graph_file)
        ticket_site_source = "alarms"

    ticket_sites = {
        ticket_id: site_list
        for ticket_id, site_list in ticket_sites.items()
        if ticket_id in eligible_ticket_ids
    }
    if not ticket_sites:
        raise ValueError("没有可用于计算的工单站点映射")

    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    print("阶段 1/2：提取 eligible 工单的故障组索引...")
    ticket_to_groups, ticket_alarm_counts = _build_ticket_group_index_for_eligible(
        alarm_input,
        eligible_ticket_ids=set(ticket_sites.keys()),
        ticket_field=ticket_field,
        group_field=group_field,
    )

    relevant_group_ids = {
        group_id
        for group_ids in ticket_to_groups.values()
        for group_id in group_ids
    }

    print("阶段 2/2：提取相关故障组覆盖到的站点和站点告警...")
    group_to_sites, group_to_site_alarms = _build_group_alarm_indexes(
        alarm_input,
        relevant_group_ids=relevant_group_ids,
        ne_to_site=ne_to_site,
        group_field=group_field,
    )

    details = []
    total_recall = 0.0

    for ticket_id in sorted(ticket_sites.keys()):
        if ticket_alarm_counts.get(ticket_id, 0) <= 0:
            continue

        target_sites = set(ticket_sites[ticket_id])
        fault_groups = sorted(ticket_to_groups.get(ticket_id, set()))
        merged_site_alarms = _merge_group_site_alarms(fault_groups, group_to_site_alarms)

        recalled_sites = set()
        for group_id in fault_groups:
            recalled_sites.update(group_to_sites.get(group_id, set()))
        recalled_sites &= target_sites

        unrecalled_sites = target_sites - recalled_sites
        upper_info = upper_bound_index.get(ticket_id, {})
        upper_site_evidence = upper_info.get("site_evidence", {})
        associated_site_alarms = build_site_alarm_map_for_sites(merged_site_alarms, recalled_sites)
        missing_site_alarms = {
            site_id: upper_site_evidence.get(site_id, [])
            for site_id in sorted(unrecalled_sites)
        }

        if strict and not _site_alarm_map_contains_offline(upper_site_evidence):
            continue

        recall = len(recalled_sites) / len(target_sites) if target_sites else 0.0
        total_recall += recall

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_alarm_count": ticket_alarm_counts.get(ticket_id, 0),
            "fault_group_count": len(fault_groups),
            "fault_groups": fault_groups,
            "associated_site_count": len(recalled_sites),
            "associated_sites": sorted(recalled_sites),
            "associated_site_alarms": associated_site_alarms,
            "missing_site_count": len(unrecalled_sites),
            "missing_sites": sorted(unrecalled_sites),
            "missing_site_alarms": missing_site_alarms,
            "recall": recall,
        })

    details.sort(
        key=lambda item: (
            -item.get("ticket_site_count", 0),
            item.get("ticket_id", ""),
        )
    )
    site_count_distribution = build_ticket_site_count_distribution(details)
    average_recall = total_recall / len(details) if details else 0.0

    result = {
        "method": "alarm_stream_group_field",
        "ticket_count": len(details),
        "final_sample_count": len(details),
        "ticket_site_count_distribution": site_count_distribution,
        "average_recall": average_recall,
        "denominator_source": "alarms",
        "ticket_site_source": ticket_site_source,
        "upper_bound_source": upper_bound_file,
        "strict_mode": strict,
        "details": details,
    }

    case_records = build_unrecalled_visualization_cases(details, result["method"])
    if output_file and not case_jsonl_output_file:
        case_jsonl_output_file = derive_case_jsonl_output_path(output_file)
    if case_jsonl_output_file:
        write_jsonl_records(case_jsonl_output_file, case_records)
        result["case_jsonl_output"] = case_jsonl_output_file
        result["unrecalled_case_count"] = len(case_records)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = ArgumentParser(
        description="v2：只针对上限结果里可完整关联的工单，输出当前方法召回到的站点/告警和未召回站点/告警"
    )
    parser.add_argument("alarms", help="告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致")
    parser.add_argument(
        "--upper-bound",
        required=True,
        help="compute_ticket_site_recall_upper_bound.py 的输出 JSON",
    )
    parser.add_argument(
        "--ticket-sites",
        help="工单站点映射 JSON；不提供时会退化为从 alarms 中回推工单站点",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="告警中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警中的故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ne-graph",
        default="ne_graph.json",
        help="用于通过告警源回填 site_id 的 ne_graph 文件，默认: ne_graph.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ticket_site_recall_v2.json",
        help="输出 JSON 文件，默认: ticket_site_recall_v2.json",
    )
    parser.add_argument(
        "--case-jsonl-output",
        help="额外输出召回率 < 100%% 的样本为可视化 jsonl；默认随主输出生成同名 .cases.jsonl",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：若 upper bound 文件里该工单的 evidence 中未出现 OFFLINE_ALARMS，则跳过该工单样本",
    )

    args = parser.parse_args()

    try:
        result = compute_ticket_site_recall_v2(
            alarm_input=args.alarms,
            upper_bound_file=args.upper_bound,
            ticket_sites_file=args.ticket_sites,
            ticket_field=args.ticket_field,
            group_field=args.group_field,
            ne_graph_file=args.ne_graph,
            output_file=args.output,
            case_jsonl_output_file=args.case_jsonl_output,
            strict=args.strict,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(f"工单数: {result['ticket_count']}")
    print(f"最终统计样本数: {result['final_sample_count']}")
    print(f"样本 site 个数分布: {result['ticket_site_count_distribution']}")
    print(f"平均召回率: {result['average_recall']:.6f}")
    print(f"明细已输出到: {args.output}")
    if result.get("case_jsonl_output"):
        print(f"未满召回样本 jsonl: {result['case_jsonl_output']} ({result.get('unrecalled_case_count', 0)} 条)")


if __name__ == "__main__":
    main()
