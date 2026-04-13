import json
import os

from argparse import ArgumentParser
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

from alarm_tools.alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from topology_resources import NE_GRAPH_JSON, resource_display


EXCLUDED_ALARM_TITLES = {"FAN FAIL"}


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def _normalize_site_list(values):
    seen = set()
    result = []
    for value in values:
        site_id = _normalize_text(value)
        if not site_id or site_id in seen:
            continue
        seen.add(site_id)
        result.append(site_id)
    return result


def _truncate_debug_samples(records, sample_limit):
    if sample_limit <= 0:
        return []
    return records[:sample_limit]


def _load_ticket_sites(ticket_sites_file):
    with open(ticket_sites_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("工单站点映射 JSON 顶层必须是对象")

    ticket_sites = {}
    ticket_recorded_times = {}
    for ticket_id, payload in data.items():
        normalized_ticket_id = _normalize_text(ticket_id)
        if not normalized_ticket_id:
            continue

        normalized_sites = []
        normalized_times = []
        if isinstance(payload, list):
            normalized_sites = _normalize_site_list(payload)
        elif isinstance(payload, dict):
            normalized_sites = _normalize_site_list(payload.get("site_ids", []))
            raw_times = payload.get("extracted_times", [])
            if isinstance(raw_times, list):
                normalized_times = _normalize_site_list(raw_times)
        else:
            continue

        if normalized_sites:
            ticket_sites[normalized_ticket_id] = normalized_sites
        if normalized_times:
            ticket_recorded_times[normalized_ticket_id] = normalized_times
    return ticket_sites, ticket_recorded_times


def _build_ticket_site_sets(ticket_sites):
    return {
        ticket_id: set(site_list)
        for ticket_id, site_list in ticket_sites.items()
    }


def _build_effective_associated_sites(base_associated_sites, extra_site_sets, ticket_site_sets):
    effective_associated_sites = {}
    for ticket_id, target_sites in ticket_site_sets.items():
        merged_sites = set(base_associated_sites.get(ticket_id, set())) & target_sites
        merged_sites.update(set(extra_site_sets.get(ticket_id, set())) & target_sites)
        effective_associated_sites[ticket_id] = merged_sites
    return effective_associated_sites


def _build_ticket_recorded_time_ranges(ticket_recorded_times):
    ticket_time_ranges = {}
    normalized_recorded_times = {}

    for ticket_id, raw_times in ticket_recorded_times.items():
        parsed_times = []
        for raw_time in raw_times:
            normalized_time = _normalize_text(raw_time)
            if not normalized_time:
                continue
            ts = _parse_time_to_ts(normalized_time)
            if ts is None:
                continue
            parsed_times.append((ts, normalized_time))

        if not parsed_times:
            continue

        parsed_times.sort(key=lambda item: (item[0], item[1]))
        normalized_recorded_times[ticket_id] = [time_text for _, time_text in parsed_times]
        ticket_time_ranges[ticket_id] = {
            "min_ts": parsed_times[0][0],
            "max_ts": parsed_times[-1][0],
            "min_time": parsed_times[0][1],
            "max_time": parsed_times[-1][1],
        }

    return normalized_recorded_times, ticket_time_ranges


def _parse_time_to_ts(value):
    text = _normalize_text(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return None


def _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field):
    site_id = _normalize_text(alarm.get(site_field, ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get(source_field, ""))
    if not alarm_source:
        return ""

    return _normalize_text(ne_to_site.get(alarm_source, ""))


def _should_skip_alarm(alarm):
    alarm_title = _normalize_text(alarm.get("告警标题", ""))
    return alarm_title in EXCLUDED_ALARM_TITLES


def _build_alarm_evidence_record(alarm, resolved_site_id, ticket_field, source_field):
    record = {
        "告警编码ID": _normalize_text(alarm.get("告警编码ID", "")),
        "告警标题": _normalize_text(alarm.get("告警标题", "")),
        "工单号": _normalize_text(alarm.get(ticket_field, "")),
        "站点ID": _normalize_text(alarm.get("站点ID", "")),
        "关联站点ID": resolved_site_id,
        "告警源": _normalize_text(alarm.get(source_field, "")),
        "告警首次发生时间": _normalize_text(alarm.get("告警首次发生时间", "")),
        "告警最后发生时间": _normalize_text(alarm.get("告警最后发生时间", "")),
        "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
    }
    return {key: value for key, value in record.items() if value}


def _build_ticket_time_windows(ticket_alarm_times, window_seconds):
    ticket_windows = {}
    for ticket_id, raw_times in ticket_alarm_times.items():
        if not raw_times:
            continue

        times = sorted(set(raw_times))
        merged = []
        for ts in times:
            start = ts - window_seconds
            end = ts + window_seconds
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        starts = [item[0] for item in merged]
        ends = [item[1] for item in merged]
        ticket_windows[ticket_id] = (starts, ends)
    return ticket_windows


def _timestamp_in_windows(ts, windows):
    if ts is None or windows is None:
        return False

    starts, ends = windows
    idx = bisect_right(starts, ts) - 1
    return idx >= 0 and ts <= ends[idx]


def _collect_ticket_alarm_stats(
    alarm_input,
    ticket_sites,
    ticket_site_sets,
    ticket_field,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    target_ticket_ids = set(ticket_sites.keys())
    ticket_alarm_counts = defaultdict(int)
    ticket_alarm_times = defaultdict(list)
    direct_sites = defaultdict(set)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id not in target_ticket_ids:
            continue

        ticket_alarm_counts[ticket_id] += 1

        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        if ts is not None:
            ticket_alarm_times[ticket_id].append(ts)

        site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if site_id and site_id in ticket_site_sets[ticket_id]:
            direct_sites[ticket_id].add(site_id)

    return ticket_alarm_counts, ticket_alarm_times, direct_sites


def _build_site_pending_tickets(ticket_sites, ticket_site_sets, direct_sites, valid_ticket_ids):
    site_pending_tickets = defaultdict(set)
    associated_sites = {}

    for ticket_id, target_site_list in ticket_sites.items():
        if ticket_id not in valid_ticket_ids:
            continue
        target_sites = ticket_site_sets[ticket_id]
        current_direct_sites = set(direct_sites.get(ticket_id, set())) & target_sites
        associated_sites[ticket_id] = set(current_direct_sites)
        for site_id in target_sites - current_direct_sites:
            site_pending_tickets[site_id].add(ticket_id)

    return associated_sites, site_pending_tickets


def _build_site_window_bounds(site_pending_tickets, ticket_windows):
    site_window_bounds = {}
    for site_id, pending_tickets in site_pending_tickets.items():
        min_start = None
        max_end = None
        for ticket_id in pending_tickets:
            windows = ticket_windows.get(ticket_id)
            if not windows:
                continue
            starts, ends = windows
            if not starts or not ends:
                continue
            site_min = starts[0]
            site_max = ends[-1]
            if min_start is None or site_min < min_start:
                min_start = site_min
            if max_end is None or site_max > max_end:
                max_end = site_max
        if min_start is not None and max_end is not None:
            site_window_bounds[site_id] = (min_start, max_end)
    return site_window_bounds


def _associate_missing_sites_by_time_window(
    alarm_input,
    ticket_windows,
    associated_sites,
    site_pending_tickets,
    site_window_bounds,
    inferred_sites,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    if not site_pending_tickets:
        return

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if not site_id:
            continue

        pending_tickets = site_pending_tickets.get(site_id)
        if not pending_tickets:
            continue

        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        if ts is None:
            continue

        bounds = site_window_bounds.get(site_id)
        if bounds is None:
            continue
        if ts < bounds[0] or ts > bounds[1]:
            continue

        matched_tickets = []
        for ticket_id in tuple(pending_tickets):
            if _timestamp_in_windows(ts, ticket_windows.get(ticket_id)):
                associated_sites[ticket_id].add(site_id)
                inferred_sites[ticket_id].add(site_id)
                matched_tickets.append(ticket_id)

        if not matched_tickets:
            continue

        for ticket_id in matched_tickets:
            pending_tickets.discard(ticket_id)
        if not pending_tickets:
            site_pending_tickets.pop(site_id, None)
            site_window_bounds.pop(site_id, None)


def _compute_upper_bound_recalls(
    ticket_sites,
    ticket_site_sets,
    ticket_alarm_counts,
    direct_sites,
    inferred_sites,
    associated_sites,
    output_ticket_ids=None,
):
    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    ticket_count = 0
    normalized_output_ticket_ids = None if output_ticket_ids is None else set(output_ticket_ids)

    for ticket_id in sorted(ticket_sites.keys()):
        if normalized_output_ticket_ids is not None and ticket_id not in normalized_output_ticket_ids:
            continue

        target_sites = ticket_site_sets[ticket_id]
        direct_site_set = set(direct_sites.get(ticket_id, set())) & target_sites
        inferred_site_set = set(inferred_sites.get(ticket_id, set())) & target_sites
        associated_site_set = set(associated_sites.get(ticket_id, set())) & target_sites

        recall = len(associated_site_set) / len(target_sites) if target_sites else 0.0
        precision = len(associated_site_set) / len(associated_site_set) if associated_site_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_alarm_count": ticket_alarm_counts.get(ticket_id, 0),
            "direct_site_count": len(direct_site_set),
            "direct_sites": sorted(direct_site_set),
            "inferred_site_count": len(inferred_site_set),
            "inferred_sites": sorted(inferred_site_set),
            "associated_site_count": len(associated_site_set),
            "associated_sites": sorted(associated_site_set),
            "recall_upper_bound": recall,
            "precision_upper_bound": precision,
            "f1_upper_bound": f1,
        })
        total_recall += recall
        total_precision += precision
        total_f1 += f1
        ticket_count += 1

    details.sort(
        key=lambda item: (
            -item.get("ticket_site_count", 0),
            item.get("ticket_id", ""),
        )
    )
    average_recall = total_recall / ticket_count if ticket_count else 0.0
    average_precision = total_precision / ticket_count if ticket_count else 0.0
    average_f1 = total_f1 / ticket_count if ticket_count else 0.0
    return details, average_recall, average_precision, average_f1


def _collect_association_evidence(
    alarm_input,
    ticket_sites,
    ticket_site_sets,
    direct_sites,
    inferred_sites,
    ticket_windows,
    ticket_recorded_time_ranges,
    ticket_field,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    direct_site_tickets = defaultdict(set)
    inferred_site_tickets = defaultdict(set)
    evidence = {}

    for ticket_id in ticket_sites:
        evidence[ticket_id] = {
            "direct_site_alarms": defaultdict(list),
            "inferred_site_alarms": defaultdict(list),
            "ticket_recorded_range_site_alarms": defaultdict(list),
        }

    for ticket_id, site_ids in direct_sites.items():
        for site_id in site_ids:
            direct_site_tickets[site_id].add(ticket_id)

    for ticket_id, site_ids in inferred_sites.items():
        for site_id in site_ids:
            inferred_site_tickets[site_id].add(ticket_id)

    recorded_range_site_tickets = defaultdict(set)
    for ticket_id, site_ids in ticket_site_sets.items():
        if ticket_id not in ticket_recorded_time_ranges:
            continue
        for site_id in site_ids:
            recorded_range_site_tickets[site_id].add(ticket_id)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if not resolved_site_id:
            continue

        normalized_ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        evidence_record = None

        direct_ticket_candidates = direct_site_tickets.get(resolved_site_id)
        if direct_ticket_candidates and normalized_ticket_id:
            for ticket_id in direct_ticket_candidates:
                if normalized_ticket_id != ticket_id:
                    continue
                if evidence_record is None:
                    evidence_record = _build_alarm_evidence_record(
                        alarm,
                        resolved_site_id,
                        ticket_field,
                        source_field,
                    )
                evidence[ticket_id]["direct_site_alarms"][resolved_site_id].append(evidence_record)

        inferred_ticket_candidates = inferred_site_tickets.get(resolved_site_id)
        if inferred_ticket_candidates and ts is not None:
            for ticket_id in inferred_ticket_candidates:
                if not _timestamp_in_windows(ts, ticket_windows.get(ticket_id)):
                    continue
                if evidence_record is None:
                    evidence_record = _build_alarm_evidence_record(
                        alarm,
                        resolved_site_id,
                        ticket_field,
                        source_field,
                    )
                evidence[ticket_id]["inferred_site_alarms"][resolved_site_id].append(evidence_record)

        recorded_range_ticket_candidates = recorded_range_site_tickets.get(resolved_site_id)
        if recorded_range_ticket_candidates and ts is not None:
            for ticket_id in recorded_range_ticket_candidates:
                time_range = ticket_recorded_time_ranges.get(ticket_id)
                if not time_range:
                    continue
                if ts < time_range["min_ts"] or ts > time_range["max_ts"]:
                    continue
                if evidence_record is None:
                    evidence_record = _build_alarm_evidence_record(
                        alarm,
                        resolved_site_id,
                        ticket_field,
                        source_field,
                    )
                evidence[ticket_id]["ticket_recorded_range_site_alarms"][resolved_site_id].append(evidence_record)

    normalized_evidence = {}
    for ticket_id, payload in evidence.items():
        time_range = ticket_recorded_time_ranges.get(ticket_id)
        normalized_evidence[ticket_id] = {
            "ticket_recorded_time_range": {
                "min_time": time_range["min_time"],
                "max_time": time_range["max_time"],
            } if time_range else {},
            "direct_site_alarms": {
                site_id: alarms
                for site_id, alarms in sorted(payload["direct_site_alarms"].items())
            },
            "inferred_site_alarms": {
                site_id: alarms
                for site_id, alarms in sorted(payload["inferred_site_alarms"].items())
            },
            "ticket_recorded_range_site_alarms": {
                site_id: alarms
                for site_id, alarms in sorted(payload["ticket_recorded_range_site_alarms"].items())
            },
        }
    return normalized_evidence


def _build_recorded_range_site_sets(association_evidence):
    recorded_range_site_sets = {}
    for ticket_id, payload in (association_evidence or {}).items():
        recorded_range_sites = payload.get("ticket_recorded_range_site_alarms", {})
        recorded_range_site_sets[ticket_id] = set(recorded_range_sites.keys())
    return recorded_range_site_sets


def _build_debug_alarm_brief(alarm, resolved_site_id, ticket_field, source_field, time_field):
    record = {
        "告警编码ID": _normalize_text(alarm.get("告警编码ID", "")),
        "告警标题": _normalize_text(alarm.get("告警标题", "")),
        "工单号": _normalize_text(alarm.get(ticket_field, "")),
        "关联站点ID": resolved_site_id,
        "告警源": _normalize_text(alarm.get(source_field, "")),
        time_field: _normalize_text(alarm.get(time_field, "")),
    }
    return {key: value for key, value in record.items() if value}


def _append_debug_sample(samples, record, sample_limit):
    if sample_limit <= 0 or len(samples) >= sample_limit:
        return
    samples.append(record)


def _build_recorded_range_debug_info(
    alarm_input,
    debug_ticket_ids,
    ticket_site_sets,
    ticket_recorded_times,
    normalized_recorded_times,
    ticket_recorded_time_ranges,
    ticket_alarm_counts,
    direct_sites,
    inferred_sites,
    base_associated_sites,
    recorded_range_site_sets,
    effective_associated_sites,
    output_ticket_ids,
    include_ticket_recorded_range_sites,
    ticket_field,
    site_field,
    source_field,
    time_field,
    ne_to_site,
    sample_limit,
):
    selected_ticket_ids = sorted({
        _normalize_text(ticket_id)
        for ticket_id in (debug_ticket_ids or [])
        if _normalize_text(ticket_id)
    })
    if not selected_ticket_ids:
        return {}

    debug_info = {}
    site_to_debug_tickets = defaultdict(set)
    for ticket_id in selected_ticket_ids:
        target_sites = set(ticket_site_sets.get(ticket_id, set()))
        for site_id in target_sites:
            site_to_debug_tickets[site_id].add(ticket_id)

        time_range = ticket_recorded_time_ranges.get(ticket_id)
        site_debug = {}
        for site_id in sorted(target_sites):
            site_debug[site_id] = {
                "in_range_count": 0,
                "before_range_count": 0,
                "after_range_count": 0,
                "unparsable_time_count": 0,
                "no_recorded_range_count": 0,
                "in_range_samples": [],
                "before_range_samples": [],
                "after_range_samples": [],
                "unparsable_time_samples": [],
                "no_recorded_range_samples": [],
            }

        debug_info[ticket_id] = {
            "present_in_ticket_sites": ticket_id in ticket_site_sets,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "raw_recorded_times": list(ticket_recorded_times.get(ticket_id, [])),
            "parsed_recorded_times": list(normalized_recorded_times.get(ticket_id, [])),
            "ticket_recorded_time_range": {
                "min_time": time_range["min_time"],
                "max_time": time_range["max_time"],
            } if time_range else {},
            "ticket_alarm_count": int(ticket_alarm_counts.get(ticket_id, 0) or 0),
            "valid_for_output": ticket_id in set(output_ticket_ids or ()),
            "include_ticket_recorded_range_sites_for_recall": include_ticket_recorded_range_sites,
            "direct_sites": sorted(set(direct_sites.get(ticket_id, set())) & target_sites),
            "inferred_sites": sorted(set(inferred_sites.get(ticket_id, set())) & target_sites),
            "base_associated_sites": sorted(set(base_associated_sites.get(ticket_id, set())) & target_sites),
            "recorded_range_sites": sorted(set(recorded_range_site_sets.get(ticket_id, set())) & target_sites),
            "effective_associated_sites": sorted(set(effective_associated_sites.get(ticket_id, set())) & target_sites),
            "site_debug": site_debug,
        }

    for ticket_id, payload in debug_info.items():
        recorded_range_sites = set(payload["recorded_range_sites"])
        base_sites = set(payload["base_associated_sites"])
        payload["recorded_range_added_sites"] = sorted(recorded_range_sites - base_sites)
        output_reasons = []
        if payload["ticket_alarm_count"] > 0:
            output_reasons.append("ticket_alarm_count>0")
        if payload["effective_associated_sites"]:
            output_reasons.append("effective_associated_sites_nonempty")
        payload["output_reasons"] = output_reasons

    if not site_to_debug_tickets:
        return debug_info

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if not resolved_site_id:
            continue

        candidate_tickets = site_to_debug_tickets.get(resolved_site_id)
        if not candidate_tickets:
            continue

        alarm_ts = _parse_time_to_ts(alarm.get(time_field, ""))
        alarm_brief = _build_debug_alarm_brief(
            alarm,
            resolved_site_id,
            ticket_field,
            source_field,
            time_field,
        )

        for ticket_id in candidate_tickets:
            ticket_entry = debug_info[ticket_id]
            site_entry = ticket_entry["site_debug"].setdefault(
                resolved_site_id,
                {
                    "in_range_count": 0,
                    "before_range_count": 0,
                    "after_range_count": 0,
                    "unparsable_time_count": 0,
                    "no_recorded_range_count": 0,
                    "in_range_samples": [],
                    "before_range_samples": [],
                    "after_range_samples": [],
                    "unparsable_time_samples": [],
                    "no_recorded_range_samples": [],
                },
            )

            time_range = ticket_recorded_time_ranges.get(ticket_id)
            if not time_range:
                site_entry["no_recorded_range_count"] += 1
                _append_debug_sample(site_entry["no_recorded_range_samples"], alarm_brief, sample_limit)
                continue

            if alarm_ts is None:
                site_entry["unparsable_time_count"] += 1
                _append_debug_sample(site_entry["unparsable_time_samples"], alarm_brief, sample_limit)
                continue

            if alarm_ts < time_range["min_ts"]:
                site_entry["before_range_count"] += 1
                _append_debug_sample(site_entry["before_range_samples"], alarm_brief, sample_limit)
                continue

            if alarm_ts > time_range["max_ts"]:
                site_entry["after_range_count"] += 1
                _append_debug_sample(site_entry["after_range_samples"], alarm_brief, sample_limit)
                continue

            site_entry["in_range_count"] += 1
            _append_debug_sample(site_entry["in_range_samples"], alarm_brief, sample_limit)

    return debug_info


def _print_debug_summary(
    ticket_sites,
    ticket_recorded_time_ranges,
    valid_ticket_ids,
    base_associated_sites,
    recorded_range_site_sets,
    effective_associated_sites,
    output_ticket_ids,
):
    total_ticket_count = len(ticket_sites)
    recorded_range_ticket_count = len(ticket_recorded_time_ranges)
    valid_ticket_count = len(valid_ticket_ids)
    output_ticket_count = len(set(output_ticket_ids or ()))
    recorded_range_hit_ticket_count = 0
    recorded_range_added_ticket_count = 0
    total_added_site_count = 0

    for ticket_id, target_site_list in ticket_sites.items():
        target_sites = set(target_site_list)
        base_sites = set(base_associated_sites.get(ticket_id, set())) & target_sites
        recorded_sites = set(recorded_range_site_sets.get(ticket_id, set())) & target_sites
        effective_sites = set(effective_associated_sites.get(ticket_id, set())) & target_sites
        added_sites = effective_sites - base_sites

        if recorded_sites:
            recorded_range_hit_ticket_count += 1
        if added_sites:
            recorded_range_added_ticket_count += 1
            total_added_site_count += len(added_sites)

    print("=== DEBUG SUMMARY ===")
    print(f"- 工单站点映射数: {total_ticket_count}")
    print(f"- 有可解析记录时间范围的工单数: {recorded_range_ticket_count}")
    print(f"- 在告警流中真实出现过的工单数: {valid_ticket_count}")
    print(f"- 最终进入结果输出的工单数: {output_ticket_count}")
    print(f"- 命中过记录时间范围站点告警的工单数: {recorded_range_hit_ticket_count}")
    print(f"- 因记录时间范围而新增 associated_sites 的工单数: {recorded_range_added_ticket_count}")
    print(f"- 因记录时间范围新增的站点总数: {total_added_site_count}")


def _print_recorded_range_debug_info(debug_info):
    if not debug_info:
        return

    for ticket_id in sorted(debug_info):
        item = debug_info[ticket_id]
        print(f"=== DEBUG TICKET {ticket_id} ===")
        print(f"- 在 ticket-sites 中: {'是' if item['present_in_ticket_sites'] else '否'}")
        print(f"- 目标站点数: {item['ticket_site_count']}")
        print(f"- 目标站点: {item['ticket_sites']}")
        print(f"- 原始 extracted_times: {item['raw_recorded_times']}")
        print(f"- 成功解析时间: {item['parsed_recorded_times']}")
        print(f"- 记录时间范围: {item['ticket_recorded_time_range']}")
        print(f"- 工单号命中告警数: {item['ticket_alarm_count']}")
        print(f"- 会进入详情输出: {'是' if item['valid_for_output'] else '否'}")
        print(f"- 进入输出原因: {item['output_reasons']}")
        print(f"- direct_sites: {item['direct_sites']}")
        print(f"- inferred_sites: {item['inferred_sites']}")
        print(f"- 原 associated_sites: {item['base_associated_sites']}")
        print(f"- recorded_range_sites: {item['recorded_range_sites']}")
        print(f"- recorded_range_added_sites: {item['recorded_range_added_sites']}")
        print(f"- 最终 associated_sites: {item['effective_associated_sites']}")

        for site_id in item["ticket_sites"]:
            site_debug = item["site_debug"].get(site_id, {})
            print(
                f"  * 站点 {site_id}: "
                f"in_range={site_debug.get('in_range_count', 0)}, "
                f"before={site_debug.get('before_range_count', 0)}, "
                f"after={site_debug.get('after_range_count', 0)}, "
                f"unparsable={site_debug.get('unparsable_time_count', 0)}, "
                f"no_range={site_debug.get('no_recorded_range_count', 0)}"
            )
            for label, key in (
                ("命中范围样例", "in_range_samples"),
                ("早于范围样例", "before_range_samples"),
                ("晚于范围样例", "after_range_samples"),
                ("时间解析失败样例", "unparsable_time_samples"),
                ("无记录时间范围样例", "no_recorded_range_samples"),
            ):
                samples = site_debug.get(key, [])
                if samples:
                    print(f"    - {label}: {samples}")


def main():
    parser = ArgumentParser(
        description="基于工单标注告警和站点时间窗，估算工单站点召回率的理论上限"
    )
    parser.add_argument(
        "alarms",
        help="告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致",
    )
    parser.add_argument(
        "--ticket-sites",
        required=True,
        help="工单站点映射 JSON，支持旧格式 {工单号: [站点列表]}，也支持 filter_incident_tickets.py 输出的 {工单号: {site_ids, extracted_times, ...}}",
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
        "--time-field",
        default="告警首次发生时间",
        help="用于做时间窗关联的告警时间字段，默认: 告警首次发生时间",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=900,
        help="缺失站点告警与工单告警允许的时间窗，单位秒，默认: 900",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于通过告警源回填站点ID的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--include-ticket-recorded-range-sites",
        action="store_true",
        help="将工单记录时间范围内命中的站点也并入可关联站点，并据此计算 recall_upper_bound",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印记录时间范围扩站点相关的调试摘要",
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
        help="每类调试样例最多打印多少条，默认: 5",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ticket_site_recall_upper_bound.json",
        help="输出 JSON 文件，默认: ticket_site_recall_upper_bound.json",
    )

    args = parser.parse_args()

    ticket_sites, ticket_recorded_times = _load_ticket_sites(args.ticket_sites)
    if not ticket_sites:
        print("❌ 工单站点映射为空，无法计算召回率上限")
        return
    ticket_site_sets = _build_ticket_site_sets(ticket_sites)
    normalized_recorded_times, ticket_recorded_time_ranges = _build_ticket_recorded_time_ranges(ticket_recorded_times)

    ne_to_site = {}
    if args.ne_graph and os.path.exists(args.ne_graph):
        ne_to_site = build_ne_to_site_map(args.ne_graph)

    print("阶段 1/2：提取每个工单的标注告警时间和直接命中站点...")
    ticket_alarm_counts, ticket_alarm_times, direct_sites = _collect_ticket_alarm_stats(
        alarm_input=args.alarms,
        ticket_sites=ticket_sites,
        ticket_site_sets=ticket_site_sets,
        ticket_field=args.ticket_field,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )
    valid_ticket_ids = {
        ticket_id
        for ticket_id, count in ticket_alarm_counts.items()
        if count > 0
    }

    ticket_windows = _build_ticket_time_windows(ticket_alarm_times, args.window_seconds)
    associated_sites, site_pending_tickets = _build_site_pending_tickets(
        ticket_sites,
        ticket_site_sets,
        direct_sites,
        valid_ticket_ids,
    )
    site_window_bounds = _build_site_window_bounds(site_pending_tickets, ticket_windows)
    inferred_sites = defaultdict(set)

    print("阶段 2/2：用站点告警时间窗补齐缺失站点...")
    _associate_missing_sites_by_time_window(
        alarm_input=args.alarms,
        ticket_windows=ticket_windows,
        associated_sites=associated_sites,
        site_pending_tickets=site_pending_tickets,
        site_window_bounds=site_window_bounds,
        inferred_sites=inferred_sites,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )

    print("阶段 3/3：收集工单关联站点的告警证据...")
    association_evidence = _collect_association_evidence(
        alarm_input=args.alarms,
        ticket_sites=ticket_sites,
        ticket_site_sets=ticket_site_sets,
        direct_sites=direct_sites,
        inferred_sites=inferred_sites,
        ticket_windows=ticket_windows,
        ticket_recorded_time_ranges=ticket_recorded_time_ranges,
        ticket_field=args.ticket_field,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )

    base_associated_sites = {
        ticket_id: set(site_ids)
        for ticket_id, site_ids in associated_sites.items()
    }
    recorded_range_site_sets = _build_recorded_range_site_sets(association_evidence)
    effective_associated_sites = base_associated_sites
    if args.include_ticket_recorded_range_sites:
        effective_associated_sites = _build_effective_associated_sites(
            base_associated_sites=base_associated_sites,
            extra_site_sets=recorded_range_site_sets,
            ticket_site_sets=ticket_site_sets,
        )
    output_ticket_ids = {
        ticket_id
        for ticket_id in ticket_sites
        if ticket_alarm_counts.get(ticket_id, 0) > 0
        or bool(effective_associated_sites.get(ticket_id, set()))
    }

    details, average_recall, average_precision, average_f1 = _compute_upper_bound_recalls(
        ticket_sites=ticket_sites,
        ticket_site_sets=ticket_site_sets,
        ticket_alarm_counts=ticket_alarm_counts,
        direct_sites=direct_sites,
        inferred_sites=inferred_sites,
        associated_sites=effective_associated_sites,
        output_ticket_ids=output_ticket_ids,
    )

    for item in details:
        ticket_id = item["ticket_id"]
        time_range = ticket_recorded_time_ranges.get(ticket_id)
        base_associated_site_set = set(base_associated_sites.get(ticket_id, set())) & ticket_site_sets.get(ticket_id, set())
        recorded_range_site_set = set(recorded_range_site_sets.get(ticket_id, set())) & ticket_site_sets.get(ticket_id, set())
        recorded_range_added_site_set = recorded_range_site_set - base_associated_site_set
        item["ticket_recorded_time_count"] = len(normalized_recorded_times.get(ticket_id, []))
        item["ticket_recorded_times"] = normalized_recorded_times.get(ticket_id, [])
        item["ticket_recorded_time_range"] = {
            "min_time": time_range["min_time"],
            "max_time": time_range["max_time"],
        } if time_range else {}
        item["ticket_recorded_range_site_count"] = len(recorded_range_site_set)
        item["ticket_recorded_range_sites"] = sorted(recorded_range_site_set)
        item["ticket_recorded_range_added_site_count"] = len(recorded_range_added_site_set)
        item["ticket_recorded_range_added_sites"] = sorted(recorded_range_added_site_set)
        item["evidence"] = association_evidence.get(item["ticket_id"], {
            "ticket_recorded_time_range": {},
            "direct_site_alarms": {},
            "inferred_site_alarms": {},
            "ticket_recorded_range_site_alarms": {},
        })

    result = {
        "ticket_count": len(details),
        "window_seconds": args.window_seconds,
        "time_field": args.time_field,
        "site_field": args.site_field,
        "source_field": args.source_field,
        "include_ticket_recorded_range_sites_for_recall": args.include_ticket_recorded_range_sites,
        "average_recall_upper_bound": average_recall,
        "average_precision_upper_bound": average_precision,
        "average_f1_upper_bound": average_f1,
        "details": details,
    }

    debug_enabled = args.debug or bool(args.debug_ticket)
    if debug_enabled:
        _print_debug_summary(
            ticket_sites=ticket_sites,
            ticket_recorded_time_ranges=ticket_recorded_time_ranges,
            valid_ticket_ids=valid_ticket_ids,
            base_associated_sites=base_associated_sites,
            recorded_range_site_sets=recorded_range_site_sets,
            effective_associated_sites=effective_associated_sites,
            output_ticket_ids=output_ticket_ids,
        )
        debug_info = _build_recorded_range_debug_info(
            alarm_input=args.alarms,
            debug_ticket_ids=args.debug_ticket,
            ticket_site_sets=ticket_site_sets,
            ticket_recorded_times=ticket_recorded_times,
            normalized_recorded_times=normalized_recorded_times,
            ticket_recorded_time_ranges=ticket_recorded_time_ranges,
            ticket_alarm_counts=ticket_alarm_counts,
            direct_sites=direct_sites,
            inferred_sites=inferred_sites,
            base_associated_sites=base_associated_sites,
            recorded_range_site_sets=recorded_range_site_sets,
            effective_associated_sites=effective_associated_sites,
            output_ticket_ids=output_ticket_ids,
            include_ticket_recorded_range_sites=args.include_ticket_recorded_range_sites,
            ticket_field=args.ticket_field,
            site_field=args.site_field,
            source_field=args.source_field,
            time_field=args.time_field,
            ne_to_site=ne_to_site,
            sample_limit=args.debug_sample_limit,
        )
        _print_recorded_range_debug_info(debug_info)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"工单数: {len(details)}")
    print(f"平均召回率上限: {average_recall:.6f}")
    print(f"平均准确率上限: {average_precision:.6f}")
    print(f"平均F1上限: {average_f1:.6f}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
