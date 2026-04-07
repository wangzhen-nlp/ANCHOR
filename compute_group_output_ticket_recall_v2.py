import json
from argparse import ArgumentParser
from collections import defaultdict

from alarm_types import OFFLINE_ALARMS
from alarm_inputs import stream_alarm_inputs
from compute_group_output_ticket_recall import (
    _count_ticket_occurrences_in_alarms,
    _count_ticket_occurrences_in_group,
    _extract_group_id,
    _extract_group_sites,
    _extract_ticket_ids,
)
from compute_ticket_site_recall import (
    _build_ticket_sites_from_alarms,
    _load_ticket_sites,
    _normalize_text,
)
from compute_ticket_site_recall_upper_bound import _should_skip_alarm
from ticket_recall_v2_utils import (
    build_alarm_to_group_index,
    build_group_site_time_index,
    build_site_alarm_map_for_sites,
    build_site_to_group_index,
    build_ticket_site_count_distribution,
    build_unrecalled_visualization_cases,
    collect_groups_by_evidence,
    dedupe_alarm_records,
    derive_case_jsonl_output_path,
    expand_groups_by_time_window,
    load_upper_bound_index,
    load_upper_bound_settings,
    load_ne_graph_data,
    select_best_group_by_target_sites,
    write_jsonl_records,
)


OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)


def _build_group_output_ticket_index_for_eligible(group_output_input, eligible_ticket_ids, ticket_field):
    ticket_to_groups = defaultdict(set)
    ticket_occurrence_counts = defaultdict(int)

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id:
            continue

        ticket_ids = [
            ticket_id
            for ticket_id in _extract_ticket_ids(group_record, ticket_field)
            if ticket_id in eligible_ticket_ids
        ]
        if not ticket_ids:
            continue

        for ticket_id in ticket_ids:
            ticket_to_groups[ticket_id].add(group_id)
        for ticket_id, count in _count_ticket_occurrences_in_group(group_record, ticket_field).items():
            if ticket_id in eligible_ticket_ids:
                ticket_occurrence_counts[ticket_id] += count

    return ticket_to_groups, ticket_occurrence_counts


def _build_group_output_alarm_indexes(group_output_input, relevant_group_ids):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))

    if not relevant_group_ids:
        return group_to_sites, group_to_site_alarms

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id or group_id not in relevant_group_ids:
            continue

        for site_id in _extract_group_sites(group_record, group_id):
            group_to_sites[group_id].add(site_id)

        for symptom in group_record.get("symptoms", []):
            if not isinstance(symptom, dict):
                continue
            if _should_skip_alarm({"告警标题": symptom.get("alarm", "")}):
                continue
            site_id = _normalize_text(symptom.get("node", ""))
            if not site_id:
                continue
            evidence_record = dict(symptom)
            evidence_record["来源故障组UUID"] = group_id
            group_to_site_alarms[group_id][site_id].append(evidence_record)

    return group_to_sites, group_to_site_alarms


def _build_group_output_alarm_indexes_for_sites(group_output_input, allowed_site_ids):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    if not allowed_site_ids:
        return group_to_sites, group_to_site_alarms

    allowed_site_ids = {_normalize_text(site_id) for site_id in allowed_site_ids if _normalize_text(site_id)}

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id:
            continue

        extracted_sites = {
            _normalize_text(site_id)
            for site_id in _extract_group_sites(group_record, group_id)
            if _normalize_text(site_id)
        }
        for site_id in extracted_sites:
            if site_id in allowed_site_ids:
                group_to_sites[group_id].add(site_id)

        for symptom in group_record.get("symptoms", []):
            if not isinstance(symptom, dict):
                continue
            if _should_skip_alarm({"告警标题": symptom.get("alarm", "")}):
                continue
            site_id = _normalize_text(symptom.get("node", ""))
            if not site_id or site_id not in allowed_site_ids:
                continue
            evidence_record = dict(symptom)
            evidence_record["来源故障组UUID"] = group_id
            group_to_site_alarms[group_id][site_id].append(evidence_record)

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


def compute_group_output_ticket_recall_v2(
    group_output_input,
    upper_bound_file,
    ticket_sites_file=None,
    ticket_field="工单号",
    alarms_input=None,
    ne_graph_file=None,
    output_file=None,
    case_jsonl_output_file=None,
    only_offline=False,
    loose=False,
    potential=False,
    only_one=False,
):
    upper_bound_index = load_upper_bound_index(upper_bound_file)
    upper_bound_settings = load_upper_bound_settings(upper_bound_file)
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
        if not alarms_input:
            raise ValueError("未提供 ticket-sites 时，必须提供 alarms 以便从告警中回推工单站点")
        ticket_sites = _build_ticket_sites_from_alarms(alarms_input, ticket_field, ne_graph_file)
        ticket_site_source = "alarms"

    ticket_sites = {
        ticket_id: site_list
        for ticket_id, site_list in ticket_sites.items()
        if ticket_id in eligible_ticket_ids
    }
    if not ticket_sites:
        raise ValueError("没有可用于计算的工单站点映射")

    stage_total = 3 if (loose or potential) else 2
    print(f"阶段 1/{stage_total}：提取 eligible 工单和故障组输出的关联关系...")
    ticket_to_groups, ticket_occurrence_counts = _build_group_output_ticket_index_for_eligible(
        group_output_input,
        eligible_ticket_ids=set(ticket_sites.keys()),
        ticket_field=ticket_field,
    )
    ticket_to_base_groups = {
        ticket_id: set(group_ids)
        for ticket_id, group_ids in ticket_to_groups.items()
    }
    relevant_group_ids = {
        group_id
        for group_ids in ticket_to_base_groups.values()
        for group_id in group_ids
    }

    loose_ticket_to_groups = defaultdict(set)
    potential_ticket_to_groups = defaultdict(set)
    if loose or potential:
        allowed_site_ids = {
            _normalize_text(site_id)
            for site_list in ticket_sites.values()
            for site_id in site_list
            if _normalize_text(site_id)
        }
        print(f"阶段 2/{stage_total}：提取工单站点上的所有 group 覆盖站点和症状告警...")
        group_to_sites, group_to_site_alarms = _build_group_output_alarm_indexes_for_sites(
            group_output_input,
            allowed_site_ids=allowed_site_ids,
        )
        print(f"阶段 3/{stage_total}：按 upper bound 口径扩充额外 group...")
        site_to_groups = build_site_to_group_index(group_to_sites) if loose else {}
        group_site_time_index = build_group_site_time_index(group_to_site_alarms) if loose else {}
        alarm_to_groups = build_alarm_to_group_index(group_to_site_alarms) if potential else {}
        for ticket_id, site_list in ticket_sites.items():
            base_group_ids = ticket_to_base_groups.get(ticket_id, set())
            loose_groups = set()
            if loose:
                _, loose_groups = expand_groups_by_time_window(
                    base_group_ids=base_group_ids,
                    target_sites=set(site_list),
                    site_to_groups=site_to_groups,
                    group_site_time_index=group_site_time_index,
                    window_seconds=upper_bound_settings["window_seconds"],
                )
                if loose_groups:
                    loose_ticket_to_groups[ticket_id] = loose_groups

            if potential:
                upper_info = upper_bound_index.get(ticket_id, {})
                upper_site_evidence = upper_info.get("site_evidence", {})
                potential_groups = collect_groups_by_evidence(
                    site_evidence=upper_site_evidence,
                    alarm_to_groups=alarm_to_groups,
                    excluded_group_ids=set(base_group_ids) | set(loose_groups),
                )
                if potential_groups:
                    potential_ticket_to_groups[ticket_id] = potential_groups
    else:
        print("阶段 2/2：提取相关故障组覆盖到的站点和症状告警...")
        group_to_sites, group_to_site_alarms = _build_group_output_alarm_indexes(
            group_output_input,
            relevant_group_ids=relevant_group_ids,
        )

    if alarms_input:
        ticket_alarm_counts = _count_ticket_occurrences_in_alarms(alarms_input, ticket_field)
        denominator_source = "alarms"
    else:
        ticket_alarm_counts = dict(ticket_occurrence_counts)
        denominator_source = "group_output"

    details = []
    total_recall = 0.0

    for ticket_id in sorted(ticket_sites.keys()):
        if ticket_alarm_counts.get(ticket_id, 0) <= 0:
            continue

        target_sites = set(ticket_sites[ticket_id])
        base_fault_groups = sorted(ticket_to_base_groups.get(ticket_id, set()))
        loose_fault_groups = sorted(loose_ticket_to_groups.get(ticket_id, set()))
        potential_fault_groups = sorted(potential_ticket_to_groups.get(ticket_id, set()))
        fault_groups = sorted(set(base_fault_groups) | set(loose_fault_groups) | set(potential_fault_groups))
        if only_one:
            selected_fault_group = select_best_group_by_target_sites(
                group_ids=fault_groups,
                group_to_sites=group_to_sites,
                target_sites=target_sites,
            )
            effective_fault_groups = [selected_fault_group] if selected_fault_group else []
        else:
            selected_fault_group = ""
            effective_fault_groups = list(fault_groups)

        merged_site_alarms = _merge_group_site_alarms(effective_fault_groups, group_to_site_alarms)

        recalled_sites = set()
        for group_id in effective_fault_groups:
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

        if only_offline and not _site_alarm_map_contains_offline(upper_site_evidence):
            continue

        recall = len(recalled_sites) / len(target_sites) if target_sites else 0.0
        total_recall += recall

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_occurrence_count": ticket_occurrence_counts.get(ticket_id, 0),
            "fault_group_count": len(fault_groups),
            "base_fault_groups": base_fault_groups,
            "loose_fault_groups": loose_fault_groups,
            "potential_fault_groups": potential_fault_groups,
            "fault_groups": fault_groups,
            "effective_fault_group_count": len(effective_fault_groups),
            "effective_fault_groups": effective_fault_groups,
            "selected_fault_group": selected_fault_group,
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
        "method": "group_output",
        "ticket_count": len(details),
        "final_sample_count": len(details),
        "ticket_site_count_distribution": site_count_distribution,
        "average_recall": average_recall,
        "denominator_source": denominator_source,
        "ticket_site_source": ticket_site_source,
        "upper_bound_source": upper_bound_file,
        "only_offline_mode": only_offline,
        "loose_mode": loose,
        "potential_mode": potential,
        "only_one_mode": only_one,
        "details": details,
    }

    ne_graph_data = load_ne_graph_data(ne_graph_file)
    case_records = build_unrecalled_visualization_cases(details, result["method"], ne_graph_data=ne_graph_data)
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
        description="v2：只针对上限结果里可完整关联的工单，输出故障组方法召回到的站点/告警和未召回站点/告警"
    )
    parser.add_argument(
        "group_output",
        help="故障组输出输入，支持 jsonl/zip/目录，与 match_rules.py 输出格式一致",
    )
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
        help="故障组 symptoms 中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--alarms",
        help="原始告警输入。未提供 ticket-sites 时必填；提供后，分母口径也按原始告警来判断。",
    )
    parser.add_argument(
        "--ne-graph",
        default="ne_graph.json",
        help="未提供 ticket-sites 时，用于通过告警源回推 site_id 的 ne_graph 文件，默认: ne_graph.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="group_output_ticket_recall_v2.json",
        help="输出 JSON 文件，默认: group_output_ticket_recall_v2.json",
    )
    parser.add_argument(
        "--case-jsonl-output",
        help="额外输出召回率 < 100%% 的样本为可视化 jsonl；默认随主输出生成同名 .cases.jsonl",
    )
    parser.add_argument(
        "--only-offline",
        action="store_true",
        help="仅保留 upper bound evidence 中出现过 OFFLINE_ALARMS 的工单样本",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许用 upper bound 同口径时间窗，在工单站点上的其它 group 进一步扩充关联",
    )
    parser.add_argument(
        "--potential",
        action="store_true",
        help="允许用 upper bound evidence 中出现过的告警，直接吸附这些告警所在的额外 group",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖该工单目标站点最多的单个 group，用它的站点计算召回率",
    )

    args = parser.parse_args()

    try:
        result = compute_group_output_ticket_recall_v2(
            group_output_input=args.group_output,
            upper_bound_file=args.upper_bound,
            ticket_sites_file=args.ticket_sites,
            ticket_field=args.ticket_field,
            alarms_input=args.alarms,
            ne_graph_file=args.ne_graph,
            output_file=args.output,
            case_jsonl_output_file=args.case_jsonl_output,
            only_offline=args.only_offline,
            loose=args.loose,
            potential=args.potential,
            only_one=args.only_one,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(f"工单数: {result['ticket_count']}")
    print(f"最终统计样本数: {result['final_sample_count']}")
    print(f"样本 site 个数分布: {result['ticket_site_count_distribution']}")
    print(f"平均召回率: {result['average_recall']:.6f}")
    print(f"分母口径来源: {result['denominator_source']}")
    print(f"明细已输出到: {args.output}")
    if result.get("case_jsonl_output"):
        print(f"未满召回样本 jsonl: {result['case_jsonl_output']} ({result.get('unrecalled_case_count', 0)} 条)")


if __name__ == "__main__":
    main()
