import json
import os

from argparse import ArgumentParser
from collections import defaultdict

from alarm_types import OFFLINE_ALARMS
from alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from compute_ticket_site_recall import (
    _build_ticket_sites_from_alarms,
    _compute_site_metrics,
    _load_ticket_sites,
    _normalize_text,
    _parse_group_ids,
    _resolve_alarm_site_id,
)
from ticket_recall_v2_utils import (
    build_alarm_to_group_index,
    build_ne_to_domain_map,
    build_site_has_domain_map,
    build_group_site_time_index,
    build_site_alarm_map_for_sites,
    build_site_to_group_index,
    build_ticket_site_count_distribution,
    build_unrecalled_visualization_cases,
    collect_groups_by_evidence,
    dedupe_alarm_records,
    derive_case_jsonl_output_path,
    expand_groups_by_time_window,
    extract_nonempty_alarm_sites,
    filter_ticket_sites_by_site_flag,
    load_upper_bound_index,
    load_upper_bound_settings,
    load_ne_graph_data,
    select_best_group_by_target_sites,
    site_alarm_map_contains_domain,
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


def _build_group_alarm_indexes_for_sites(alarm_input, allowed_site_ids, ne_to_site, group_field):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    if not allowed_site_ids:
        return group_to_sites, group_to_site_alarms

    allowed_site_ids = {_normalize_text(site_id) for site_id in allowed_site_ids if _normalize_text(site_id)}

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        if not group_ids:
            continue

        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site)
        if not resolved_site_id or resolved_site_id not in allowed_site_ids:
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
    only_offline=False,
    no_data_alarm=False,
    no_data_site=False,
    require_transmission_per_site=False,
    loose=False,
    potential=False,
    only_one=False,
    min_site_num=0,
    upper_bound_associated_as_gold=False,
):
    upper_bound_index = load_upper_bound_index(upper_bound_file)
    upper_bound_settings = load_upper_bound_settings(upper_bound_file)
    if upper_bound_associated_as_gold:
        eligible_ticket_ids = {
            ticket_id
            for ticket_id, item in upper_bound_index.items()
            if int(item.get("associated_site_count", 0) or 0) > 0
        }
        if not eligible_ticket_ids:
            raise ValueError("召回率上限结果里没有“已关联站点”的工单")

        ticket_sites = {
            ticket_id: list(upper_bound_index[ticket_id].get("associated_sites", []))
            for ticket_id in sorted(eligible_ticket_ids)
        }
        ticket_site_source = "upper_bound_associated_sites"
    else:
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

    ne_graph_data = load_ne_graph_data(ne_graph_file)
    if no_data_site:
        if not ne_graph_data:
            raise ValueError("开启 no-data-site 时，必须提供有效的 ne_graph 文件")
        site_has_data = build_site_has_domain_map(ne_graph_data, "DATA")
        ticket_sites = {
            ticket_id: site_list
            for ticket_id, site_list in ticket_sites.items()
            if not any(site_has_data.get(_normalize_text(site_id), False) for site_id in site_list)
        }
    if require_transmission_per_site:
        if not ne_graph_data:
            raise ValueError("开启 require-transmission-per-site 时，必须提供有效的 ne_graph 文件")
        site_has_transmission = build_site_has_domain_map(ne_graph_data, "TRANSMISSION")
        ticket_sites = filter_ticket_sites_by_site_flag(ticket_sites, site_has_transmission)
    if min_site_num > 0:
        ticket_sites = {
            ticket_id: site_list
            for ticket_id, site_list in ticket_sites.items()
            if len(site_list) >= min_site_num
        }
    if not ticket_sites:
        raise ValueError("没有可用于计算的工单站点映射")

    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    stage_total = 4 if (loose or potential) else 2
    print(f"阶段 1/{stage_total}：提取 eligible 工单的故障组索引...")
    ticket_to_groups, ticket_alarm_counts = _build_ticket_group_index_for_eligible(
        alarm_input,
        eligible_ticket_ids=set(ticket_sites.keys()),
        ticket_field=ticket_field,
        group_field=group_field,
    )
    ticket_to_base_groups = {
        ticket_id: set(group_ids)
        for ticket_id, group_ids in ticket_to_groups.items()
    }

    loose_ticket_to_groups = defaultdict(set)
    potential_ticket_to_groups = defaultdict(set)
    relevant_group_ids = {
        group_id
        for group_ids in ticket_to_base_groups.values()
        for group_id in group_ids
    }

    if loose or potential:
        allowed_site_ids = {
            _normalize_text(site_id)
            for site_list in ticket_sites.values()
            for site_id in site_list
            if _normalize_text(site_id)
        }
        print(f"阶段 2/{stage_total}：提取工单站点上的候选故障组ID覆盖站点和站点告警...")
        scoped_group_to_sites, scoped_group_to_site_alarms = _build_group_alarm_indexes_for_sites(
            alarm_input,
            allowed_site_ids=allowed_site_ids,
            ne_to_site=ne_to_site,
            group_field=group_field,
        )
        print(f"阶段 3/{stage_total}：按 upper bound 口径扩充额外故障组ID...")
        site_to_groups = build_site_to_group_index(scoped_group_to_sites) if loose else {}
        group_site_time_index = build_group_site_time_index(scoped_group_to_site_alarms) if loose else {}
        alarm_to_groups = build_alarm_to_group_index(scoped_group_to_site_alarms) if potential else {}
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

        relevant_group_ids = (
            {
                group_id
                for group_ids in ticket_to_base_groups.values()
                for group_id in group_ids
            }
            | {
                group_id
                for group_ids in loose_ticket_to_groups.values()
                for group_id in group_ids
            }
            | {
                group_id
                for group_ids in potential_ticket_to_groups.values()
                for group_id in group_ids
            }
        )
        print(f"阶段 4/{stage_total}：提取最终相关故障组ID的全量覆盖站点和站点告警...")
        group_to_sites, group_to_site_alarms = _build_group_alarm_indexes(
            alarm_input,
            relevant_group_ids=relevant_group_ids,
            ne_to_site=ne_to_site,
            group_field=group_field,
        )
    else:
        print("阶段 2/2：提取相关故障组覆盖到的站点和站点告警...")
        group_to_sites, group_to_site_alarms = _build_group_alarm_indexes(
            alarm_input,
            relevant_group_ids=relevant_group_ids,
            ne_to_site=ne_to_site,
            group_field=group_field,
        )

    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    ne_to_domain = build_ne_to_domain_map(ne_graph_data)

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
        predicted_sites = extract_nonempty_alarm_sites(merged_site_alarms)
        true_positive_sites, recall, precision, f1 = _compute_site_metrics(target_sites, predicted_sites)
        recalled_sites = set(true_positive_sites)

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
        if no_data_alarm and site_alarm_map_contains_domain(upper_site_evidence, ne_to_domain, "DATA"):
            continue

        total_recall += recall
        total_precision += precision
        total_f1 += f1

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_alarm_count": ticket_alarm_counts.get(ticket_id, 0),
            "fault_group_count": len(fault_groups),
            "base_fault_groups": base_fault_groups,
            "loose_fault_groups": loose_fault_groups,
            "potential_fault_groups": potential_fault_groups,
            "fault_groups": fault_groups,
            "effective_fault_group_count": len(effective_fault_groups),
            "effective_fault_groups": effective_fault_groups,
            "selected_fault_group": selected_fault_group,
            "group_site_count": len(predicted_sites),
            "group_sites": sorted(predicted_sites),
            "associated_site_count": len(recalled_sites),
            "associated_sites": sorted(recalled_sites),
            "associated_site_alarms": associated_site_alarms,
            "missing_site_count": len(unrecalled_sites),
            "missing_sites": sorted(unrecalled_sites),
            "missing_site_alarms": missing_site_alarms,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

    details.sort(
        key=lambda item: (
            -item.get("ticket_site_count", 0),
            item.get("ticket_id", ""),
        )
    )
    site_count_distribution = build_ticket_site_count_distribution(details)
    average_recall = total_recall / len(details) if details else 0.0
    average_precision = total_precision / len(details) if details else 0.0
    average_f1 = total_f1 / len(details) if details else 0.0

    result = {
        "method": "alarm_stream_group_field",
        "ticket_count": len(details),
        "final_sample_count": len(details),
        "ticket_site_count_distribution": site_count_distribution,
        "average_recall": average_recall,
        "average_precision": average_precision,
        "average_f1": average_f1,
        "denominator_source": "alarms",
        "ticket_site_source": ticket_site_source,
        "upper_bound_source": upper_bound_file,
        "only_offline_mode": only_offline,
        "no_data_alarm_mode": no_data_alarm,
        "no_data_site_mode": no_data_site,
        "require_transmission_per_site_mode": require_transmission_per_site,
        "loose_mode": loose,
        "potential_mode": potential,
        "only_one_mode": only_one,
        "min_site_num": min_site_num,
        "upper_bound_associated_as_gold_mode": upper_bound_associated_as_gold,
        "details": details,
    }

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
        description="v2：以上限结果里的 associated_sites 作为 gold，输出当前方法召回到的站点/告警和未召回站点/告警"
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
        help="用于通过告警源回填 site_id 以及做 site/domain 过滤的 ne_graph 文件，默认: ne_graph.json",
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
        "--only-offline",
        action="store_true",
        help="仅保留 upper bound evidence 中出现过 OFFLINE_ALARMS 的工单样本",
    )
    parser.add_argument(
        "--no-data-alarm",
        action="store_true",
        help="如果 upper bound evidence 中存在来自 Data 设备的告警，则跳过该工单样本",
    )
    parser.add_argument(
        "--no-data-site",
        action="store_true",
        help="如果当前工单站点里存在包含 Data 设备的站点，则跳过该工单样本",
    )
    parser.add_argument(
        "--require-transmission-per-site",
        action="store_true",
        help="先从工单站点里剔除不包含 Transmission 设备的站点；过滤后若站点数不足 min-site-num，则跳过该工单",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许用 upper bound 同口径时间窗，在工单站点上的其它故障组ID 进一步扩充关联",
    )
    parser.add_argument(
        "--potential",
        action="store_true",
        help="允许用 upper bound evidence 中出现过的告警，直接吸附这些告警所在的额外故障组ID",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖该工单目标站点最多的单个故障组ID，用它的站点计算召回率",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="仅统计工单站点数 >= 该值的工单；默认: 0（不过滤）",
    )
    parser.add_argument(
        "--upper-bound-associated-as-gold",
        action="store_true",
        help="改用 upper bound 的 associated_sites 作为 gold；不开时保持原来的 fully_associable + 原工单站点口径",
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
            only_offline=args.only_offline,
            no_data_alarm=args.no_data_alarm,
            no_data_site=args.no_data_site,
            require_transmission_per_site=args.require_transmission_per_site,
            loose=args.loose,
            potential=args.potential,
            only_one=args.only_one,
            min_site_num=args.min_site_num,
            upper_bound_associated_as_gold=args.upper_bound_associated_as_gold,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    print(f"工单数: {result['ticket_count']}")
    print(f"最终统计样本数: {result['final_sample_count']}")
    print(f"样本 site 个数分布: {result['ticket_site_count_distribution']}")
    print(f"平均召回率: {result['average_recall']:.6f}")
    print(f"平均准确率: {result['average_precision']:.6f}")
    print(f"平均F1: {result['average_f1']:.6f}")
    print(f"明细已输出到: {args.output}")
    if result.get("case_jsonl_output"):
        print(f"未满召回样本 jsonl: {result['case_jsonl_output']} ({result.get('unrecalled_case_count', 0)} 条)")


if __name__ == "__main__":
    main()
