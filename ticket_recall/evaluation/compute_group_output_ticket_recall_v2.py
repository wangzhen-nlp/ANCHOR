import json
from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

from alarm_tools.alarm_types import OFFLINE_ALARMS
from alarm_tools.alarm_inputs import stream_alarm_inputs
from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.compute_group_output_ticket_recall import (
    _count_ticket_occurrences_in_alarms,
    _count_ticket_occurrences_in_group,
    _extract_group_id,
    _extract_group_sites,
    _extract_ticket_ids,
)
from ticket_recall.evaluation.compute_ticket_site_recall import (
    _build_ticket_sites_from_alarms,
    _compute_site_metrics,
    _load_ticket_sites,
    _normalize_text,
)
from ticket_recall.evaluation.compute_ticket_site_recall_upper_bound import _should_skip_alarm
from ticket_recall.ticket_recall_v2_utils import (
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
    extract_alarm_record_id,
    select_best_group_by_target_sites,
    site_alarm_map_contains_domain,
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


def _collect_referenced_group_uuids(group_output_input):
    referenced_group_ids = set()

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        match_info = group_record.get("match_info", {})
        related_group_ids = []
        if isinstance(match_info, dict):
            related_group_ids = match_info.get("related_group_uuids", [])
        if not isinstance(related_group_ids, list):
            continue
        for group_id in related_group_ids:
            normalized_group_id = _normalize_text(group_id)
            if normalized_group_id:
                referenced_group_ids.add(normalized_group_id)

    return referenced_group_ids


def _build_group_output_alarm_indexes(group_output_input, relevant_group_ids, excluded_group_ids=None):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    excluded_group_ids = set(excluded_group_ids or ())

    if not relevant_group_ids:
        return group_to_sites, group_to_site_alarms

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id or group_id not in relevant_group_ids or group_id in excluded_group_ids:
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


def _build_group_output_alarm_indexes_for_sites(group_output_input, allowed_site_ids, excluded_group_ids=None):
    group_to_sites = defaultdict(set)
    group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    excluded_group_ids = set(excluded_group_ids or ())
    if not allowed_site_ids:
        return group_to_sites, group_to_site_alarms

    allowed_site_ids = {_normalize_text(site_id) for site_id in allowed_site_ids if _normalize_text(site_id)}

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id or group_id in excluded_group_ids:
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


def _normalize_debug_ticket_ids(values):
    return {
        normalized_ticket_id
        for normalized_ticket_id in (_normalize_text(value) for value in (values or []))
        if normalized_ticket_id
    }


def _append_debug_example(example_map, key, ticket_id, limit):
    if limit <= 0:
        return
    normalized_ticket_id = _normalize_text(ticket_id)
    if not normalized_ticket_id:
        return
    bucket = example_map.setdefault(key, [])
    if normalized_ticket_id in bucket or len(bucket) >= limit:
        return
    bucket.append(normalized_ticket_id)


def _build_group_site_union(group_ids, group_to_sites):
    merged_sites = set()
    for group_id in group_ids:
        merged_sites.update(group_to_sites.get(group_id, set()))
    return sorted({
        normalized_site_id
        for normalized_site_id in (_normalize_text(site_id) for site_id in merged_sites)
        if normalized_site_id
    })


def _build_potential_evidence_debug_info(site_evidence, alarm_to_groups, excluded_group_ids):
    excluded_groups = set(excluded_group_ids or ())
    matched_groups = set()
    evidence_alarm_hits = []
    if not isinstance(site_evidence, dict):
        return [], []

    for site_id in sorted(site_evidence):
        alarms = site_evidence.get(site_id, [])
        if not isinstance(alarms, list):
            continue
        for record in alarms:
            if not isinstance(record, dict):
                continue
            alarm_id = extract_alarm_record_id(record)
            if not alarm_id:
                continue
            raw_groups = sorted({
                _normalize_text(group_id)
                for group_id in alarm_to_groups.get(alarm_id, ())
                if _normalize_text(group_id) and _normalize_text(group_id) not in excluded_groups
            })
            if not raw_groups:
                continue
            matched_groups.update(raw_groups)
            evidence_alarm_hits.append({
                "site_id": _normalize_text(site_id),
                "alarm_id": alarm_id,
                "alarm_title": _normalize_text(record.get("告警标题", "")) or _normalize_text(record.get("alarm", "")),
                "matched_groups": raw_groups,
            })

    return sorted(matched_groups), evidence_alarm_hits


def _normalize_debug_alarm_ids(values):
    return {
        normalized_alarm_id
        for normalized_alarm_id in (_normalize_text(value) for value in (values or []))
        if normalized_alarm_id
    }


def _normalize_debug_site_ids(values):
    return {
        normalized_site_id
        for normalized_site_id in (_normalize_text(value) for value in (values or []))
        if normalized_site_id
    }


def _normalize_debug_group_ids(values):
    return {
        normalized_group_id
        for normalized_group_id in (_normalize_text(value) for value in (values or []))
        if normalized_group_id
    }


def _build_debug_alarm_group_lookup(debug_alarm_ids, ticket_sites, upper_bound_index, alarm_to_groups):
    if not debug_alarm_ids:
        return {}

    alarm_presence = {alarm_id: [] for alarm_id in sorted(debug_alarm_ids)}
    for ticket_id in sorted(ticket_sites):
        upper_site_evidence = upper_bound_index.get(ticket_id, {}).get("site_evidence", {})
        if not isinstance(upper_site_evidence, dict):
            continue
        for site_id in sorted(upper_site_evidence):
            alarms = upper_site_evidence.get(site_id, [])
            if not isinstance(alarms, list):
                continue
            for record in alarms:
                alarm_id = extract_alarm_record_id(record)
                if alarm_id not in debug_alarm_ids:
                    continue
                alarm_presence[alarm_id].append({
                    "ticket_id": ticket_id,
                    "site_id": _normalize_text(site_id),
                    "alarm_title": _normalize_text(record.get("告警标题", "")) or _normalize_text(record.get("alarm", "")),
                })

    result = {}
    for alarm_id in sorted(debug_alarm_ids):
        result[alarm_id] = {
            "matched_groups": sorted({
                _normalize_text(group_id)
                for group_id in alarm_to_groups.get(alarm_id, ())
                if _normalize_text(group_id)
            }),
            "evidence_hits": alarm_presence.get(alarm_id, []),
        }
    return result


def _print_debug_alarm_group_lookup(debug_alarm_group_lookup):
    if not debug_alarm_group_lookup:
        return

    for alarm_id in sorted(debug_alarm_group_lookup):
        item = debug_alarm_group_lookup[alarm_id]
        print(f"=== DEBUG EVIDENCE ALARM {alarm_id} ===")
        print(f"- matched_groups: {item.get('matched_groups', [])}")
        print(f"- evidence_hits: {item.get('evidence_hits', [])}")


def _build_debug_site_group_lookup(debug_site_ids, site_to_groups):
    if not debug_site_ids:
        return {}

    result = {}
    for site_id in sorted(debug_site_ids):
        result[site_id] = {
            "matched_groups": sorted({
                _normalize_text(group_id)
                for group_id in site_to_groups.get(site_id, ())
                if _normalize_text(group_id)
            }),
        }
    return result


def _print_debug_site_group_lookup(debug_site_group_lookup):
    if not debug_site_group_lookup:
        return

    for site_id in sorted(debug_site_group_lookup):
        item = debug_site_group_lookup[site_id]
        print(f"=== DEBUG SITE {site_id} ===")
        print(f"- matched_groups: {item.get('matched_groups', [])}")


def _build_debug_group_site_lookup(group_output_input, debug_group_ids, referenced_group_ids=None, allowed_site_ids=None):
    if not debug_group_ids:
        return {}

    normalized_group_ids = {
        _normalize_text(group_id)
        for group_id in debug_group_ids
        if _normalize_text(group_id)
    }
    referenced_group_ids = {
        _normalize_text(group_id)
        for group_id in (referenced_group_ids or ())
        if _normalize_text(group_id)
    }
    allowed_site_ids = {
        _normalize_text(site_id)
        for site_id in (allowed_site_ids or ())
        if _normalize_text(site_id)
    }

    result = {
        group_id: {
            "present_in_output": False,
            "present_in_top_level_uuid": False,
            "present_in_match_info_uuid": False,
            "present_in_related_group_uuids": False,
            "is_referenced_group": group_id in referenced_group_ids,
            "group_sites": [],
            "matched_allowed_sites": [],
            "symptom_nodes": [],
            "related_group_carriers": [],
            "matched_records": [],
        }
        for group_id in sorted(normalized_group_ids)
    }

    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        top_level_uuid = _normalize_text(group_record.get("uuid", ""))
        match_info = group_record.get("match_info", {})
        match_info_uuid = ""
        related_group_uuids = []
        if isinstance(match_info, dict):
            match_info_uuid = _normalize_text(match_info.get("uuid", ""))
            raw_related_group_uuids = match_info.get("related_group_uuids", [])
            if isinstance(raw_related_group_uuids, list):
                related_group_uuids = sorted({
                    _normalize_text(group_id)
                    for group_id in raw_related_group_uuids
                    if _normalize_text(group_id)
                })

        effective_group_id = _extract_group_id(group_record)
        matched_group_ids = {
            group_id
            for group_id in normalized_group_ids
            if group_id in {top_level_uuid, match_info_uuid}
        }
        related_only_group_ids = {
            group_id
            for group_id in normalized_group_ids
            if group_id in related_group_uuids
        }
        if not matched_group_ids and not related_only_group_ids:
            continue

        group_sites = _extract_group_sites(group_record, effective_group_id)
        symptom_nodes = {
            _normalize_text(symptom.get("node", ""))
            for symptom in group_record.get("symptoms", [])
            if isinstance(symptom, dict) and _normalize_text(symptom.get("node", ""))
        }
        for group_id in sorted(related_only_group_ids):
            result[group_id]["present_in_related_group_uuids"] = True
            carrier = effective_group_id or top_level_uuid or match_info_uuid
            if carrier and carrier not in result[group_id]["related_group_carriers"]:
                result[group_id]["related_group_carriers"].append(carrier)

        for group_id in sorted(matched_group_ids):
            matched_by = []
            if group_id == top_level_uuid:
                matched_by.append("top_level_uuid")
                result[group_id]["present_in_top_level_uuid"] = True
            if group_id == match_info_uuid:
                matched_by.append("match_info_uuid")
                result[group_id]["present_in_match_info_uuid"] = True

            result[group_id]["present_in_output"] = True
            result[group_id]["group_sites"] = sorted({
                _normalize_text(site_id)
                for site_id in (result[group_id]["group_sites"] + group_sites)
                if _normalize_text(site_id)
            })
            result[group_id]["matched_allowed_sites"] = sorted({
                site_id
                for site_id in result[group_id]["group_sites"]
                if site_id in allowed_site_ids
            })
            result[group_id]["symptom_nodes"] = sorted({
                _normalize_text(site_id)
                for site_id in (result[group_id]["symptom_nodes"] + list(symptom_nodes))
                if _normalize_text(site_id)
            })
            result[group_id]["matched_records"].append({
                "matched_by": matched_by,
                "top_level_uuid": top_level_uuid,
                "match_info_uuid": match_info_uuid,
                "effective_group_id": effective_group_id,
                "related_group_uuids": related_group_uuids,
            })

    return result


def _print_debug_group_site_lookup(debug_group_site_lookup):
    if not debug_group_site_lookup:
        return

    for group_id in sorted(debug_group_site_lookup):
        item = debug_group_site_lookup[group_id]
        print(f"=== DEBUG GROUP {group_id} ===")
        print(f"- present_in_output: {'是' if item.get('present_in_output') else '否'}")
        print(f"- present_in_top_level_uuid: {'是' if item.get('present_in_top_level_uuid') else '否'}")
        print(f"- present_in_match_info_uuid: {'是' if item.get('present_in_match_info_uuid') else '否'}")
        print(f"- present_in_related_group_uuids: {'是' if item.get('present_in_related_group_uuids') else '否'}")
        print(f"- is_referenced_group: {'是' if item.get('is_referenced_group') else '否'}")
        print(f"- group_sites: {item.get('group_sites', [])}")
        print(f"- matched_allowed_sites: {item.get('matched_allowed_sites', [])}")
        print(f"- symptom_nodes: {item.get('symptom_nodes', [])}")
        print(f"- related_group_carriers: {item.get('related_group_carriers', [])}")
        print(f"- matched_records: {item.get('matched_records', [])}")


def _print_v2_debug_summary(debug_summary):
    if not debug_summary:
        return

    print("=== DEBUG SUMMARY ===")
    if debug_summary.get("method"):
        print(f"- 方法: {debug_summary['method']}")
    if debug_summary.get("ticket_site_source"):
        print(f"- ticket-sites 来源: {debug_summary['ticket_site_source']}")
    if debug_summary.get("denominator_source"):
        print(f"- 分母口径来源: {debug_summary['denominator_source']}")
    if "upper_bound_associated_as_gold_mode" in debug_summary:
        print(
            "- upper-bound-associated-as-gold: "
            f"{'开' if debug_summary['upper_bound_associated_as_gold_mode'] else '关'}"
        )
    if "ultimate_only_mode" in debug_summary:
        print(f"- ultimate-only: {'开' if debug_summary['ultimate_only_mode'] else '关'}")
    if "allowed_site_count" in debug_summary:
        print(f"- allowed_site_ids 数量: {debug_summary['allowed_site_count']}")
    if "allowed_site_ids" in debug_summary:
        print(f"- allowed_site_ids: {debug_summary['allowed_site_ids']}")

    for label, key in (
        ("upper bound 工单数", "upper_bound_ticket_count"),
        ("满足 upper bound 口径的工单数", "upper_bound_eligible_ticket_count"),
        ("ticket-sites 原始工单数", "ticket_site_source_count"),
        ("upper bound 过滤后工单数", "after_upper_bound_filter_count"),
        ("no-data-site 后工单数", "after_no_data_site_count"),
        ("require-transmission-per-site 后工单数", "after_require_transmission_count"),
        ("min-site-num 后工单数", "after_min_site_num_count"),
        ("进入 group 评估循环的工单数", "candidate_ticket_count"),
        ("有 base group 的工单数", "ticket_with_base_group_count"),
        ("有任意候选 group 的工单数", "ticket_with_any_fault_group_count"),
        ("有有效 group 的工单数", "ticket_with_effective_fault_group_count"),
        ("预测到非空告警站点的工单数", "ticket_with_predicted_sites_count"),
        ("有召回站点的工单数", "ticket_with_recalled_sites_count"),
        ("被 only-offline 过滤的工单数", "filtered_by_only_offline_count"),
        ("被 no-data-alarm 过滤的工单数", "filtered_by_no_data_alarm_count"),
        ("最终输出工单数", "final_output_count"),
    ):
        if key in debug_summary:
            print(f"- {label}: {debug_summary[key]}")

    examples = debug_summary.get("example_ticket_ids", {})
    for label, key in (
        ("被 upper bound 过滤样例", "filtered_by_upper_bound"),
        ("被 no-data-site 过滤样例", "filtered_by_no_data_site"),
        ("被 require-transmission-per-site 过滤样例", "filtered_by_require_transmission"),
        ("被 min-site-num 过滤样例", "filtered_by_min_site_num"),
        ("没有候选 group 的样例", "no_fault_groups"),
        ("有 group 但没有预测站点的样例", "no_predicted_sites"),
        ("召回为 0 的样例", "zero_recall"),
        ("被 only-offline 过滤样例", "filtered_by_only_offline"),
        ("被 no-data-alarm 过滤样例", "filtered_by_no_data_alarm"),
    ):
        if examples.get(key):
            print(f"  * {label}: {examples[key]}")


def _print_v2_debug_tickets(debug_ticket_details, count_field_name, count_label):
    if not debug_ticket_details:
        return

    for ticket_id in sorted(debug_ticket_details):
        item = debug_ticket_details[ticket_id]
        print(f"=== DEBUG TICKET {ticket_id} ===")
        print(f"- 在 ticket-sites 来源中: {'是' if item.get('present_in_ticket_site_source') else '否'}")
        print(f"- 在 upper bound 中: {'是' if item.get('present_in_upper_bound') else '否'}")
        print(f"- 满足 upper bound 口径: {'是' if item.get('upper_bound_eligible') else '否'}")
        print(f"- upper bound fully_associable: {'是' if item.get('upper_bound_fully_associable') else '否'}")
        print(f"- upper bound associated_sites: {item.get('upper_bound_associated_sites', [])}")
        print(f"- upper bound evidence sites: {item.get('upper_bound_site_evidence_sites', [])}")
        print(f"- 来源站点: {item.get('source_ticket_sites', [])}")
        print(f"- upper bound 过滤后站点: {item.get('sites_after_upper_bound_filter', [])}")
        print(f"- no-data-site 命中站点: {item.get('data_sites_in_ticket', [])}")
        print(f"- no-data-site 后站点: {item.get('sites_after_no_data_site', [])}")
        print(f"- require-transmission 去掉的站点: {item.get('transmission_removed_sites', [])}")
        print(f"- require-transmission 后站点: {item.get('sites_after_require_transmission', [])}")
        print(f"- min-site-num 之前站点: {item.get('sites_before_min_site_num', [])}")
        print(f"- 最终 gold 站点: {item.get('final_ticket_sites', [])}")
        print(f"- {count_label}: {item.get(count_field_name, 0)}")
        print(f"- base_fault_groups: {item.get('base_fault_groups', [])}")
        print(f"- loose_fault_groups: {item.get('loose_fault_groups', [])}")
        print(f"- potential_fault_groups: {item.get('potential_fault_groups', [])}")
        print(f"- potential_evidence_groups: {item.get('potential_evidence_groups', [])}")
        print(f"- potential_evidence_alarm_hits: {item.get('potential_evidence_alarm_hits', [])}")
        print(f"- fault_groups: {item.get('fault_groups', [])}")
        print(f"- effective_fault_groups: {item.get('effective_fault_groups', [])}")
        print(f"- selected_fault_group: {item.get('selected_fault_group', '')}")
        print(f"- group_sites_from_index: {item.get('group_sites_from_index', [])}")
        print(f"- predicted_sites_with_alarms: {item.get('predicted_sites', [])}")
        print(f"- associated_sites: {item.get('associated_sites', [])}")
        print(f"- missing_sites: {item.get('missing_sites', [])}")
        print(f"- upper bound 有 offline evidence: {'是' if item.get('upper_bound_has_offline_evidence') else '否'}")
        print(f"- upper bound 有 data alarm: {'是' if item.get('upper_bound_has_data_alarm') else '否'}")
        print(f"- 最终进入输出: {'是' if item.get('included_in_final_output') else '否'}")
        print(f"- 过滤原因: {item.get('excluded_reasons', [])}")
        if "recall" in item:
            print(
                "- recall/precision/f1: "
                f"{item.get('recall', 0.0):.6f} / "
                f"{item.get('precision', 0.0):.6f} / "
                f"{item.get('f1', 0.0):.6f}"
            )


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
    no_data_alarm=False,
    no_data_site=False,
    require_transmission_per_site=False,
    loose=False,
    potential=False,
    only_one=False,
    ultimate_only=False,
    min_site_num=0,
    upper_bound_associated_as_gold=False,
    debug=False,
    debug_ticket_ids=None,
    debug_sample_limit=5,
    debug_evidence_alarm_ids=None,
    debug_site_ids=None,
    debug_group_ids=None,
):
    upper_bound_index = load_upper_bound_index(upper_bound_file)
    upper_bound_settings = load_upper_bound_settings(upper_bound_file)
    debug_ticket_id_set = _normalize_debug_ticket_ids(debug_ticket_ids)
    debug_alarm_id_set = _normalize_debug_alarm_ids(debug_evidence_alarm_ids)
    debug_site_id_set = _normalize_debug_site_ids(debug_site_ids)
    debug_group_id_set = _normalize_debug_group_ids(debug_group_ids)
    debug_enabled = bool(debug or debug_ticket_id_set or debug_alarm_id_set or debug_site_id_set or debug_group_id_set)
    debug_summary = None
    debug_ticket_details = {}
    debug_alarm_group_lookup = {}
    debug_site_group_lookup = {}
    debug_group_site_lookup = {}

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
        source_ticket_sites = {
            ticket_id: list(site_list)
            for ticket_id, site_list in ticket_sites.items()
        }
    else:
        eligible_ticket_ids = {
            ticket_id
            for ticket_id, item in upper_bound_index.items()
            if item.get("fully_associable")
        }
        if not eligible_ticket_ids:
            raise ValueError("召回率上限结果里没有“可完整关联”的工单")

        if ticket_sites_file:
            source_ticket_sites = _load_ticket_sites(ticket_sites_file)
            ticket_site_source = "ticket_sites"
        else:
            if not alarms_input:
                raise ValueError("未提供 ticket-sites 时，必须提供 alarms 以便从告警中回推工单站点")
            source_ticket_sites = _build_ticket_sites_from_alarms(alarms_input, ticket_field, ne_graph_file)
            ticket_site_source = "alarms"

        ticket_sites = {
            ticket_id: site_list
            for ticket_id, site_list in source_ticket_sites.items()
            if ticket_id in eligible_ticket_ids
        }

    if debug_enabled:
        debug_summary = {
            "method": "group_output",
            "ticket_site_source": ticket_site_source,
            "upper_bound_associated_as_gold_mode": upper_bound_associated_as_gold,
            "ultimate_only_mode": ultimate_only,
            "upper_bound_ticket_count": len(upper_bound_index),
            "upper_bound_eligible_ticket_count": len(eligible_ticket_ids),
            "ticket_site_source_count": len(source_ticket_sites),
            "after_upper_bound_filter_count": len(ticket_sites),
            "example_ticket_ids": {},
        }
        removed_by_upper_bound = set(source_ticket_sites) - set(ticket_sites)
        for ticket_id in sorted(removed_by_upper_bound):
            _append_debug_example(
                debug_summary["example_ticket_ids"],
                "filtered_by_upper_bound",
                ticket_id,
                debug_sample_limit,
            )

    if debug_ticket_id_set:
        for debug_ticket_id in sorted(debug_ticket_id_set):
            upper_info = upper_bound_index.get(debug_ticket_id, {})
            debug_ticket_details[debug_ticket_id] = {
                "ticket_id": debug_ticket_id,
                "present_in_ticket_site_source": debug_ticket_id in source_ticket_sites,
                "present_in_upper_bound": debug_ticket_id in upper_bound_index,
                "upper_bound_eligible": debug_ticket_id in eligible_ticket_ids,
                "upper_bound_fully_associable": bool(upper_info.get("fully_associable")),
                "upper_bound_associated_sites": list(upper_info.get("associated_sites", [])),
                "upper_bound_site_evidence_sites": sorted(upper_info.get("site_evidence", {}).keys()),
                "source_ticket_sites": list(source_ticket_sites.get(debug_ticket_id, [])),
                "sites_after_upper_bound_filter": list(ticket_sites.get(debug_ticket_id, [])),
                "data_sites_in_ticket": [],
                "sites_after_no_data_site": [],
                "transmission_removed_sites": [],
                "sites_after_require_transmission": [],
                "sites_before_min_site_num": [],
                "final_ticket_sites": [],
                "ticket_occurrence_count": 0,
                "base_fault_groups": [],
                "loose_fault_groups": [],
                "potential_fault_groups": [],
                "potential_evidence_groups": [],
                "potential_evidence_alarm_hits": [],
                "fault_groups": [],
                "effective_fault_groups": [],
                "selected_fault_group": "",
                "group_sites_from_index": [],
                "predicted_sites": [],
                "associated_sites": [],
                "missing_sites": [],
                "upper_bound_has_offline_evidence": False,
                "upper_bound_has_data_alarm": False,
                "included_in_final_output": False,
                "excluded_reasons": [],
            }
            if debug_ticket_id not in source_ticket_sites:
                debug_ticket_details[debug_ticket_id]["excluded_reasons"].append("not_in_ticket_site_source")
            if debug_ticket_id not in eligible_ticket_ids:
                debug_ticket_details[debug_ticket_id]["excluded_reasons"].append(
                    "filtered_by_upper_bound_eligibility"
                )

        def _snapshot_debug_sites(field_name, mapping):
            for debug_ticket_id, item in debug_ticket_details.items():
                item[field_name] = list(mapping.get(debug_ticket_id, []))

        _snapshot_debug_sites("sites_after_upper_bound_filter", ticket_sites)

    ne_graph_data = load_ne_graph_data(ne_graph_file)
    if no_data_site:
        if not ne_graph_data:
            raise ValueError("开启 no-data-site 时，必须提供有效的 ne_graph 文件")
        site_has_data = build_site_has_domain_map(ne_graph_data, "DATA")
        filtered_ticket_sites = {}
        for ticket_id, site_list in ticket_sites.items():
            data_sites = sorted({
                normalized_site_id
                for normalized_site_id in (_normalize_text(site_id) for site_id in site_list)
                if normalized_site_id and site_has_data.get(normalized_site_id, False)
            })
            if ticket_id in debug_ticket_details:
                debug_ticket_details[ticket_id]["data_sites_in_ticket"] = data_sites
            if data_sites:
                if debug_enabled:
                    _append_debug_example(
                        debug_summary["example_ticket_ids"],
                        "filtered_by_no_data_site",
                        ticket_id,
                        debug_sample_limit,
                    )
                if ticket_id in debug_ticket_details:
                    debug_ticket_details[ticket_id]["excluded_reasons"].append("filtered_by_no_data_site")
                continue
            filtered_ticket_sites[ticket_id] = site_list
        ticket_sites = filtered_ticket_sites
    if debug_enabled:
        debug_summary["after_no_data_site_count"] = len(ticket_sites)
    if debug_ticket_id_set:
        _snapshot_debug_sites("sites_after_no_data_site", ticket_sites)

    if require_transmission_per_site:
        if not ne_graph_data:
            raise ValueError("开启 require-transmission-per-site 时，必须提供有效的 ne_graph 文件")
        site_has_transmission = build_site_has_domain_map(ne_graph_data, "TRANSMISSION")
        filtered_ticket_sites = filter_ticket_sites_by_site_flag(ticket_sites, site_has_transmission)
        for ticket_id, site_list in ticket_sites.items():
            filtered_site_list = filtered_ticket_sites.get(ticket_id, [])
            filtered_site_set = set(filtered_site_list)
            removed_sites = [
                normalized_site_id
                for normalized_site_id in (_normalize_text(site_id) for site_id in site_list)
                if normalized_site_id and normalized_site_id not in filtered_site_set
            ]
            if ticket_id in debug_ticket_details:
                debug_ticket_details[ticket_id]["transmission_removed_sites"] = removed_sites
            if not filtered_site_list:
                if debug_enabled:
                    _append_debug_example(
                        debug_summary["example_ticket_ids"],
                        "filtered_by_require_transmission",
                        ticket_id,
                        debug_sample_limit,
                    )
                if ticket_id in debug_ticket_details:
                    debug_ticket_details[ticket_id]["excluded_reasons"].append(
                        "filtered_by_require_transmission"
                    )
        ticket_sites = filtered_ticket_sites
    if debug_enabled:
        debug_summary["after_require_transmission_count"] = len(ticket_sites)
    if debug_ticket_id_set:
        _snapshot_debug_sites("sites_after_require_transmission", ticket_sites)
        _snapshot_debug_sites("sites_before_min_site_num", ticket_sites)

    if min_site_num > 0:
        filtered_ticket_sites = {}
        for ticket_id, site_list in ticket_sites.items():
            if len(site_list) < min_site_num:
                if debug_enabled:
                    _append_debug_example(
                        debug_summary["example_ticket_ids"],
                        "filtered_by_min_site_num",
                        ticket_id,
                        debug_sample_limit,
                    )
                if ticket_id in debug_ticket_details:
                    debug_ticket_details[ticket_id]["excluded_reasons"].append("filtered_by_min_site_num")
                continue
            filtered_ticket_sites[ticket_id] = site_list
        ticket_sites = filtered_ticket_sites
    if debug_enabled:
        debug_summary["after_min_site_num_count"] = len(ticket_sites)
    if debug_ticket_id_set:
        _snapshot_debug_sites("final_ticket_sites", ticket_sites)

    if not ticket_sites:
        raise ValueError("没有可用于计算的工单站点映射")

    referenced_group_ids = set()
    if ultimate_only:
        print("预处理：提取被其它 group 引用的关联 group...")
        referenced_group_ids = _collect_referenced_group_uuids(group_output_input)

    stage_total = 4 if (loose or potential) else 2
    allowed_site_ids = {
        _normalize_text(site_id)
        for site_list in ticket_sites.values()
        for site_id in site_list
        if _normalize_text(site_id)
    }
    if debug_enabled:
        debug_summary["allowed_site_count"] = len(allowed_site_ids)
        debug_summary["allowed_site_ids"] = sorted(allowed_site_ids)
    if debug_group_id_set:
        debug_group_site_lookup = _build_debug_group_site_lookup(
            group_output_input=group_output_input,
            debug_group_ids=debug_group_id_set,
            referenced_group_ids=referenced_group_ids,
            allowed_site_ids=allowed_site_ids,
        )
    print(f"阶段 1/{stage_total}：提取 eligible 工单和故障组输出的关联关系...")
    ticket_to_groups, ticket_occurrence_counts = _build_group_output_ticket_index_for_eligible(
        group_output_input,
        eligible_ticket_ids=set(ticket_sites.keys()),
        ticket_field=ticket_field,
    )
    ticket_to_base_groups = {
        ticket_id: set(group_id for group_id in group_ids if group_id not in referenced_group_ids)
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
        print(f"阶段 2/{stage_total}：提取工单站点上的候选 group 覆盖站点和症状告警...")
        scoped_group_to_sites, scoped_group_to_site_alarms = _build_group_output_alarm_indexes_for_sites(
            group_output_input,
            allowed_site_ids=allowed_site_ids,
            excluded_group_ids=referenced_group_ids,
        )
        print(f"阶段 3/{stage_total}：按 upper bound 口径扩充额外 group...")
        site_to_groups = build_site_to_group_index(scoped_group_to_sites) if loose else {}
        if debug_site_id_set:
            debug_site_group_lookup = _build_debug_site_group_lookup(
                debug_site_ids=debug_site_id_set,
                site_to_groups=site_to_groups if loose else build_site_to_group_index(scoped_group_to_sites),
            )
        group_site_time_index = build_group_site_time_index(scoped_group_to_site_alarms) if loose else {}
        alarm_to_groups = build_alarm_to_group_index(scoped_group_to_site_alarms) if potential else {}
        if debug_alarm_id_set:
            debug_alarm_group_lookup = _build_debug_alarm_group_lookup(
                debug_alarm_ids=debug_alarm_id_set,
                ticket_sites=ticket_sites,
                upper_bound_index=upper_bound_index,
                alarm_to_groups=alarm_to_groups,
            )
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
                potential_evidence_groups = []
                potential_evidence_alarm_hits = []
                if ticket_id in debug_ticket_details:
                    potential_evidence_groups, potential_evidence_alarm_hits = _build_potential_evidence_debug_info(
                        site_evidence=upper_site_evidence,
                        alarm_to_groups=alarm_to_groups,
                        excluded_group_ids=set(base_group_ids) | set(loose_groups),
                    )
                    debug_ticket_details[ticket_id]["potential_evidence_groups"] = potential_evidence_groups
                    debug_ticket_details[ticket_id]["potential_evidence_alarm_hits"] = potential_evidence_alarm_hits
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
        print(f"阶段 4/{stage_total}：提取最终相关 group 的全量覆盖站点和症状告警...")
        group_to_sites, group_to_site_alarms = _build_group_output_alarm_indexes(
            group_output_input,
            relevant_group_ids=relevant_group_ids,
            excluded_group_ids=referenced_group_ids,
        )
    else:
        print("阶段 2/2：提取相关故障组覆盖到的站点和症状告警...")
        group_to_sites, group_to_site_alarms = _build_group_output_alarm_indexes(
            group_output_input,
            relevant_group_ids=relevant_group_ids,
            excluded_group_ids=referenced_group_ids,
        )
        if debug_site_id_set:
            debug_site_group_lookup = _build_debug_site_group_lookup(
                debug_site_ids=debug_site_id_set,
                site_to_groups=build_site_to_group_index(group_to_sites),
            )

    if alarms_input:
        ticket_alarm_counts = _count_ticket_occurrences_in_alarms(alarms_input, ticket_field)
        denominator_source = "alarms"
    else:
        ticket_alarm_counts = dict(ticket_occurrence_counts)
        denominator_source = "group_output"

    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    ne_to_domain = build_ne_to_domain_map(ne_graph_data)
    if debug_enabled:
        debug_summary["denominator_source"] = denominator_source
        debug_summary["candidate_ticket_count"] = len(ticket_sites)
        debug_summary["ticket_with_base_group_count"] = 0
        debug_summary["ticket_with_any_fault_group_count"] = 0
        debug_summary["ticket_with_effective_fault_group_count"] = 0
        debug_summary["ticket_with_predicted_sites_count"] = 0
        debug_summary["ticket_with_recalled_sites_count"] = 0
        debug_summary["filtered_by_only_offline_count"] = 0
        debug_summary["filtered_by_no_data_alarm_count"] = 0
        debug_summary["final_output_count"] = 0

    for ticket_id in sorted(ticket_sites.keys()):
        target_sites = set(ticket_sites[ticket_id])
        base_fault_groups = sorted(ticket_to_base_groups.get(ticket_id, set()))
        loose_fault_groups = sorted(loose_ticket_to_groups.get(ticket_id, set()))
        potential_fault_groups = sorted(potential_ticket_to_groups.get(ticket_id, set()))
        fault_groups = sorted(set(base_fault_groups) | set(loose_fault_groups) | set(potential_fault_groups))
        if debug_enabled:
            if base_fault_groups:
                debug_summary["ticket_with_base_group_count"] += 1
            if fault_groups:
                debug_summary["ticket_with_any_fault_group_count"] += 1
            else:
                _append_debug_example(
                    debug_summary["example_ticket_ids"],
                    "no_fault_groups",
                    ticket_id,
                    debug_sample_limit,
                )
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
        if debug_enabled and effective_fault_groups:
            debug_summary["ticket_with_effective_fault_group_count"] += 1

        merged_site_alarms = _merge_group_site_alarms(effective_fault_groups, group_to_site_alarms)
        predicted_sites = extract_nonempty_alarm_sites(merged_site_alarms)
        if debug_enabled:
            if predicted_sites:
                debug_summary["ticket_with_predicted_sites_count"] += 1
            else:
                _append_debug_example(
                    debug_summary["example_ticket_ids"],
                    "no_predicted_sites",
                    ticket_id,
                    debug_sample_limit,
                )
        true_positive_sites, recall, precision, f1 = _compute_site_metrics(target_sites, predicted_sites)
        recalled_sites = set(true_positive_sites)
        if debug_enabled:
            if recalled_sites:
                debug_summary["ticket_with_recalled_sites_count"] += 1
            else:
                _append_debug_example(
                    debug_summary["example_ticket_ids"],
                    "zero_recall",
                    ticket_id,
                    debug_sample_limit,
                )

        unrecalled_sites = target_sites - recalled_sites
        upper_info = upper_bound_index.get(ticket_id, {})
        upper_site_evidence = upper_info.get("site_evidence", {})
        associated_site_alarms = build_site_alarm_map_for_sites(merged_site_alarms, recalled_sites)
        missing_site_alarms = {
            site_id: upper_site_evidence.get(site_id, [])
            for site_id in sorted(unrecalled_sites)
        }
        upper_bound_has_offline_evidence = _site_alarm_map_contains_offline(upper_site_evidence)
        upper_bound_has_data_alarm = site_alarm_map_contains_domain(upper_site_evidence, ne_to_domain, "DATA")

        if ticket_id in debug_ticket_details:
            debug_ticket_details[ticket_id].update({
                "ticket_occurrence_count": ticket_occurrence_counts.get(ticket_id, 0),
                "base_fault_groups": base_fault_groups,
                "loose_fault_groups": loose_fault_groups,
                "potential_fault_groups": potential_fault_groups,
                "fault_groups": fault_groups,
                "effective_fault_groups": effective_fault_groups,
                "selected_fault_group": selected_fault_group,
                "group_sites_from_index": _build_group_site_union(effective_fault_groups, group_to_sites),
                "predicted_sites": sorted(predicted_sites),
                "associated_sites": sorted(recalled_sites),
                "missing_sites": sorted(unrecalled_sites),
                "upper_bound_has_offline_evidence": upper_bound_has_offline_evidence,
                "upper_bound_has_data_alarm": upper_bound_has_data_alarm,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            })

        if only_offline and not upper_bound_has_offline_evidence:
            if debug_enabled:
                debug_summary["filtered_by_only_offline_count"] += 1
                _append_debug_example(
                    debug_summary["example_ticket_ids"],
                    "filtered_by_only_offline",
                    ticket_id,
                    debug_sample_limit,
                )
            if ticket_id in debug_ticket_details:
                debug_ticket_details[ticket_id]["excluded_reasons"].append("filtered_by_only_offline")
            continue
        if no_data_alarm and upper_bound_has_data_alarm:
            if debug_enabled:
                debug_summary["filtered_by_no_data_alarm_count"] += 1
                _append_debug_example(
                    debug_summary["example_ticket_ids"],
                    "filtered_by_no_data_alarm",
                    ticket_id,
                    debug_sample_limit,
                )
            if ticket_id in debug_ticket_details:
                debug_ticket_details[ticket_id]["excluded_reasons"].append("filtered_by_no_data_alarm")
            continue

        total_recall += recall
        total_precision += precision
        total_f1 += f1
        if debug_enabled:
            debug_summary["final_output_count"] += 1
        if ticket_id in debug_ticket_details:
            debug_ticket_details[ticket_id]["included_in_final_output"] = True

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
        "method": "group_output",
        "ticket_count": len(details),
        "final_sample_count": len(details),
        "ticket_site_count_distribution": site_count_distribution,
        "average_recall": average_recall,
        "average_precision": average_precision,
        "average_f1": average_f1,
        "denominator_source": denominator_source,
        "ticket_site_source": ticket_site_source,
        "upper_bound_source": upper_bound_file,
        "only_offline_mode": only_offline,
        "no_data_alarm_mode": no_data_alarm,
        "no_data_site_mode": no_data_site,
        "require_transmission_per_site_mode": require_transmission_per_site,
        "loose_mode": loose,
        "potential_mode": potential,
        "only_one_mode": only_one,
        "ultimate_only_mode": ultimate_only,
        "min_site_num": min_site_num,
        "upper_bound_associated_as_gold_mode": upper_bound_associated_as_gold,
        "details": details,
    }
    if debug_enabled:
        result["debug_summary"] = debug_summary
    if debug_ticket_details:
        result["debug_tickets"] = {
            ticket_id: debug_ticket_details[ticket_id]
            for ticket_id in sorted(debug_ticket_details)
        }
    if debug_alarm_group_lookup:
        result["debug_evidence_alarm_groups"] = debug_alarm_group_lookup
    if debug_site_group_lookup:
        result["debug_site_groups"] = debug_site_group_lookup
    if debug_group_site_lookup:
        result["debug_groups"] = debug_group_site_lookup

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
        description="v2：以上限结果里的 associated_sites 作为 gold，输出故障组方法召回到的站点/告警和未召回站点/告警"
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
        default=NE_GRAPH_JSON,
        help=f"未提供 ticket-sites 时，用于通过告警源回推 site_id，同时也用于 site/domain 过滤的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
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
    parser.add_argument(
        "--ultimate-only",
        action="store_true",
        help="只考虑不作为关联 group 出现的最终 group（即未出现在其它 group 的 related_group_uuids 中）",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印 v2 评测的阶段摘要，帮助定位样本为何被过滤、为何没有召回",
    )
    parser.add_argument(
        "--debug-ticket",
        action="append",
        default=[],
        help="打印指定工单的详细调试信息；可重复传入多个工单号",
    )
    parser.add_argument(
        "--debug-sample-limit",
        type=int,
        default=5,
        help="每类调试样例最多打印多少个工单号，默认: 5",
    )
    parser.add_argument(
        "--debug-evidence-alarm-id",
        action="append",
        default=[],
        help="直接查看指定 upper_site_evidence 告警ID 在当前 potential 索引下能关联到哪些 group；可重复传入多个",
    )
    parser.add_argument(
        "--debug-site",
        action="append",
        default=[],
        help="直接查看指定 site 在当前候选范围里能关联到哪些 group；可重复传入多个",
    )
    parser.add_argument(
        "--debug-group",
        action="append",
        default=[],
        help="直接查看指定 group 在 match_rules 输出里关联到哪些站点；可重复传入多个",
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
            no_data_alarm=args.no_data_alarm,
            no_data_site=args.no_data_site,
            require_transmission_per_site=args.require_transmission_per_site,
            loose=args.loose,
            potential=args.potential,
            only_one=args.only_one,
            ultimate_only=args.ultimate_only,
            min_site_num=args.min_site_num,
            upper_bound_associated_as_gold=args.upper_bound_associated_as_gold,
            debug=args.debug,
            debug_ticket_ids=args.debug_ticket,
            debug_sample_limit=args.debug_sample_limit,
            debug_evidence_alarm_ids=args.debug_evidence_alarm_id,
            debug_site_ids=args.debug_site,
            debug_group_ids=args.debug_group,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    if args.debug or args.debug_ticket:
        _print_v2_debug_summary(result.get("debug_summary"))
        if args.debug_ticket:
            _print_v2_debug_tickets(
                result.get("debug_tickets", {}),
                count_field_name="ticket_occurrence_count",
                count_label="ticket_occurrence_count",
            )
        else:
            print("如需单工单明细，请追加 --debug-ticket 工单号")
    if args.debug_evidence_alarm_id:
        _print_debug_alarm_group_lookup(result.get("debug_evidence_alarm_groups", {}))
    if args.debug_site:
        _print_debug_site_group_lookup(result.get("debug_site_groups", {}))
    if args.debug_group:
        _print_debug_group_site_lookup(result.get("debug_groups", {}))

    print(f"工单数: {result['ticket_count']}")
    print(f"最终统计样本数: {result['final_sample_count']}")
    print(f"样本 site 个数分布: {result['ticket_site_count_distribution']}")
    print(f"平均召回率: {result['average_recall']:.6f}")
    print(f"平均准确率: {result['average_precision']:.6f}")
    print(f"平均F1: {result['average_f1']:.6f}")
    print(f"分母口径来源: {result['denominator_source']}")
    print(f"明细已输出到: {args.output}")
    if result.get("case_jsonl_output"):
        print(f"未满召回样本 jsonl: {result['case_jsonl_output']} ({result.get('unrecalled_case_count', 0)} 条)")


if __name__ == "__main__":
    main()
