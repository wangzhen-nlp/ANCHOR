import json
import os

from collections import defaultdict

from alarm_tools.alarm_inputs import build_ne_to_site_map, stream_alarm_inputs


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_site_list(values):
    seen = set()
    normalized = []
    for value in values:
        site_id = _normalize_text(value)
        if not site_id or site_id in seen:
            continue
        seen.add(site_id)
        normalized.append(site_id)
    return normalized


def _parse_group_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_parse_group_ids(item))
        return _normalize_site_list(result)

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

    parts = [text]
    for delimiter in [",", "，", ";", "；", "|"]:
        if delimiter in text:
            parts = [
                segment
                for segment in text.replace("，", ",")
                .replace(";", ",")
                .replace("；", ",")
                .replace("|", ",")
                .split(",")
            ]
            break

    return _normalize_site_list(parts)


def _load_ticket_sites(ticket_sites_file):
    with open(ticket_sites_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ticket_sites = {}
    for ticket_id, sites in data.items():
        normalized_ticket_id = _normalize_text(ticket_id)
        if not normalized_ticket_id:
            continue
        if not isinstance(sites, list):
            continue
        normalized_sites = _normalize_site_list(sites)
        if not normalized_sites:
            continue
        ticket_sites[normalized_ticket_id] = normalized_sites
    return ticket_sites


def _resolve_alarm_site_id(alarm, ne_to_site):
    site_id = _normalize_text(alarm.get("站点ID", ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get("告警源", ""))
    if not alarm_source:
        return ""

    return _normalize_text(ne_to_site.get(alarm_source, ""))


def _build_ticket_sites_from_alarms(alarm_input, ticket_field, ne_graph_file=None):
    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    ticket_sites = defaultdict(set)
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if not ticket_id:
            continue

        site_id = _resolve_alarm_site_id(alarm, ne_to_site)
        if site_id:
            ticket_sites[ticket_id].add(site_id)

    return {
        ticket_id: sorted(site_ids)
        for ticket_id, site_ids in ticket_sites.items()
        if site_ids
    }


def _compute_site_metrics(target_sites, predicted_sites):
    target_site_set = set(target_sites)
    predicted_site_set = set(predicted_sites)
    true_positive_sites = target_site_set & predicted_site_set

    recall = len(true_positive_sites) / len(target_site_set) if target_site_set else 0.0
    precision = len(true_positive_sites) / len(predicted_site_set) if predicted_site_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return true_positive_sites, recall, precision, f1


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


def _count_ticket_occurrences_in_alarms(alarm_input, ticket_field):
    ticket_alarm_counts = defaultdict(int)
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id:
            ticket_alarm_counts[ticket_id] += 1
    return ticket_alarm_counts
