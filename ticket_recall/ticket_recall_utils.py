import json
import os
from bisect import bisect_right
from datetime import datetime
from collections import defaultdict

from topology_resources import SITE_GRAPH_JSON
from fault_grouping.alarm_events.identity import require_alarm_identity
from ticket_recall.evaluation.recall_common import _normalize_text


def normalize_text(value):
    return _normalize_text(value)


def _alarm_record_context(record):
    return (
        normalize_text(record.get("site_id", ""))
        or normalize_text(record.get("node", ""))
        or normalize_text(record.get("关联站点ID", ""))
        or normalize_text(record.get("站点ID", "")),
        normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", "")),
        normalize_text(record.get("alarm_id", ""))
        or normalize_text(record.get("eid", ""))
        or normalize_text(record.get("告警编码ID", "")),
        normalize_text(record.get("alarm_time", ""))
        or normalize_text(record.get("告警首次发生时间", ""))
        or normalize_text(record.get("time_str", ""))
        or normalize_text(record.get("ts", "")),
        normalize_text(record.get("alarm", ""))
        or normalize_text(record.get("alarm_type", ""))
        or normalize_text(record.get("告警标题", "")),
    )


UPPER_BOUND_EVIDENCE_BUCKETS = (
    "direct_site_alarms",
    "inferred_site_alarms",
    "ticket_recorded_range_site_alarms",
)


def dedupe_alarm_records(records):
    seen = set()
    keyed_records = {}
    result = []

    def identity_key(record):
        return require_alarm_identity(record)

    def append_unique_text(record, field_name, value):
        incoming_values = [
            normalize_text(item)
            for item in str(value or "").replace(";", ",").split(",")
            if normalize_text(item)
        ]
        if not incoming_values:
            return
        existing_values = [
            normalize_text(item)
            for item in str(record.get(field_name, "") or "").replace(";", ",").split(",")
            if normalize_text(item)
        ]
        for incoming_value in incoming_values:
            if incoming_value not in existing_values:
                existing_values.append(incoming_value)
        if existing_values:
            record[field_name] = ",".join(existing_values)

    def merge_duplicate_record(target, incoming):
        for field_name in ("故障组ID", "alarm_group_id", "mhp_group_id", "来源故障组UUID"):
            append_unique_text(target, field_name, incoming.get(field_name, ""))
        role_values = []
        for raw_role in target.get("matched_role_list", []) or []:
            role = normalize_text(raw_role)
            if role and role not in role_values:
                role_values.append(role)
        for raw_role in (target.get("matched_role", ""), incoming.get("matched_role", "")):
            role = normalize_text(raw_role)
            if role and role not in role_values:
                role_values.append(role)
        if role_values:
            target["matched_role"] = role_values[0]
            target["matched_role_list"] = role_values

    for record in records:
        if not isinstance(record, dict):
            continue
        key = identity_key(record)
        existing = keyed_records.get(key)
        if existing is not None:
            merge_duplicate_record(existing, record)
            continue
        keyed_records[key] = record
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def build_merged_windows_from_timestamps(raw_times, window_seconds):
    normalized_times = sorted({int(ts) for ts in raw_times if ts is not None})
    if not normalized_times:
        return None

    merged = []
    for ts in normalized_times:
        start = ts - window_seconds
        end = ts + window_seconds
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    starts = [item[0] for item in merged]
    ends = [item[1] for item in merged]
    return starts, ends


def timestamp_in_windows(ts, windows):
    if ts is None or windows is None:
        return False

    starts, ends = windows
    idx = bisect_right(starts, ts) - 1
    return idx >= 0 and ts <= ends[idx]


def extract_alarm_record_id(record):
    if not isinstance(record, dict):
        return ""
    return (
        normalize_text(record.get("eid", ""))
        or normalize_text(record.get("alarm_id", ""))
        or normalize_text(record.get("告警编码ID", ""))
    )


def alarm_record_identity_key(record):
    if not isinstance(record, dict):
        return None
    return require_alarm_identity(record)


def load_upper_bound_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("召回率上限结果 JSON 顶层必须是对象")

    try:
        window_seconds = int(data.get("window_seconds", 900) or 900)
    except (TypeError, ValueError):
        window_seconds = 900

    ticket_index = {}
    for item in data.get("details", []):
        if not isinstance(item, dict):
            continue

        ticket_id = normalize_text(item.get("ticket_id", ""))
        if not ticket_id:
            continue

        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
        associated_site_count = int(item.get("associated_site_count", 0) or 0)
        associated_sites = sorted(
            {
                normalized_site_id
                for normalized_site_id in (
                    normalize_text(site_id)
                    for site_id in item.get("associated_sites", [])
                )
                if normalized_site_id
            }
        )

        merged_site_evidence = defaultdict(list)
        direct_anchor_times = []
        merged_evidence_times = []
        evidence = item.get("evidence", {})
        if isinstance(evidence, dict):
            for bucket_name in UPPER_BOUND_EVIDENCE_BUCKETS:
                bucket = evidence.get(bucket_name, {})
                if not isinstance(bucket, dict):
                    continue
                for site_id, alarms in bucket.items():
                    normalized_site_id = normalize_text(site_id)
                    if not normalized_site_id or not isinstance(alarms, list):
                        continue
                    valid_alarms = [alarm for alarm in alarms if isinstance(alarm, dict)]
                    merged_site_evidence[normalized_site_id].extend(valid_alarms)
                    for alarm in valid_alarms:
                        ts = parse_alarm_record_ts(alarm)
                        if ts is None:
                            continue
                        merged_evidence_times.append(ts)
                        if bucket_name == "direct_site_alarms":
                            direct_anchor_times.append(ts)

        anchor_times = direct_anchor_times or merged_evidence_times
        ticket_windows = build_merged_windows_from_timestamps(anchor_times, window_seconds)

        ticket_index[ticket_id] = {
            "ticket_site_count": ticket_site_count,
            "associated_site_count": associated_site_count,
            "associated_sites": associated_sites,
            "fully_associable": ticket_site_count > 0 and associated_site_count == ticket_site_count,
            "window_seconds": window_seconds,
            "ticket_windows": ticket_windows,
            "site_evidence": {
                site_id: dedupe_alarm_records(alarms)
                for site_id, alarms in sorted(merged_site_evidence.items())
            },
        }

    return ticket_index


def load_upper_bound_settings(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("召回率上限结果 JSON 顶层必须是对象")

    try:
        window_seconds = int(data.get("window_seconds", 900) or 900)
    except (TypeError, ValueError):
        window_seconds = 900

    return {
        "window_seconds": window_seconds,
        "time_field": normalize_text(data.get("time_field", "")) or "告警首次发生时间",
        "site_field": normalize_text(data.get("site_field", "")) or "站点ID",
        "source_field": normalize_text(data.get("source_field", "")) or "告警源",
    }


def upper_bound_site_diff(item):
    if not isinstance(item, dict):
        return 0
    try:
        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
    except (TypeError, ValueError):
        ticket_site_count = 0
    try:
        associated_site_count = int(item.get("associated_site_count", 0) or 0)
    except (TypeError, ValueError):
        associated_site_count = 0
    return max(ticket_site_count - associated_site_count, 0)


def upper_bound_matches_site_diff(item, expected_diff):
    if not isinstance(item, dict):
        return False
    try:
        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
    except (TypeError, ValueError):
        ticket_site_count = 0
    if ticket_site_count <= 0:
        return False
    return upper_bound_site_diff(item) == expected_diff


def build_site_alarm_map_for_sites(site_alarm_map, site_ids):
    result = {}
    for site_id in sorted(site_ids):
        result[site_id] = dedupe_alarm_records(site_alarm_map.get(site_id, []))
    return result


def extract_nonempty_alarm_sites(site_alarm_map):
    if not isinstance(site_alarm_map, dict):
        return set()

    result = set()
    for site_id, alarms in site_alarm_map.items():
        normalized_site_id = normalize_text(site_id)
        if not normalized_site_id or not isinstance(alarms, list):
            continue
        if any(isinstance(record, dict) for record in alarms):
            result.add(normalized_site_id)
    return result


def build_ticket_site_count_distribution(details):
    counts = defaultdict(int)
    for item in details:
        try:
            site_count = int(item.get("ticket_site_count", 0) or 0)
        except (TypeError, ValueError):
            continue
        counts[site_count] += 1

    return {
        str(site_count): counts[site_count]
        for site_count in sorted(counts)
    }


def select_best_group_by_target_sites(group_ids, group_to_sites, target_sites, group_to_site_alarms=None):
    normalized_target_sites = {
        normalize_text(site_id) for site_id in target_sites if normalize_text(site_id)
    }
    normalized_group_ids = sorted({
        normalize_text(group_id) for group_id in group_ids if normalize_text(group_id)
    })
    if not normalized_group_ids:
        return ""

    best_group_id = ""
    best_score = (-1, -1)
    group_to_site_alarms = group_to_site_alarms or {}
    for group_id in normalized_group_ids:
        covered_count = len(set(group_to_sites.get(group_id, set())) & normalized_target_sites)
        alarm_recalled_count = len(
            extract_nonempty_alarm_sites(group_to_site_alarms.get(group_id, {})) & normalized_target_sites
        )
        score = (covered_count, alarm_recalled_count)
        if score > best_score:
            best_group_id = group_id
            best_score = score
    return best_group_id


def build_site_to_group_index(group_to_sites):
    site_to_groups = defaultdict(set)
    for group_id, site_ids in group_to_sites.items():
        for site_id in site_ids:
            normalized_site_id = normalize_text(site_id)
            if normalized_site_id:
                site_to_groups[normalized_site_id].add(group_id)
    return site_to_groups


def build_group_site_time_index(group_to_site_alarms):
    result = {}
    for group_id, site_alarm_map in group_to_site_alarms.items():
        if not isinstance(site_alarm_map, dict):
            continue
        site_time_map = {}
        for site_id, alarms in site_alarm_map.items():
            if not isinstance(alarms, list):
                continue
            timestamps = sorted({
                ts for ts in (parse_alarm_record_ts(alarm) for alarm in alarms)
                if ts is not None
            })
            if timestamps:
                site_time_map[normalize_text(site_id)] = timestamps
        if site_time_map:
            result[group_id] = site_time_map
    return result


def build_alarm_to_group_index(group_to_site_alarms):
    alarm_to_groups = defaultdict(set)
    for group_id, site_alarm_map in group_to_site_alarms.items():
        if not isinstance(site_alarm_map, dict):
            continue
        for alarms in site_alarm_map.values():
            if not isinstance(alarms, list):
                continue
            for record in alarms:
                alarm_key = alarm_record_identity_key(record)
                if alarm_key is not None:
                    alarm_to_groups[alarm_key].add(group_id)
    return alarm_to_groups


def expand_groups_by_time_window(base_group_ids, target_sites, site_to_groups, group_site_time_index, window_seconds):
    normalized_target_sites = {
        normalize_text(site_id) for site_id in target_sites if normalize_text(site_id)
    }
    if not normalized_target_sites:
        return set(base_group_ids), set()

    candidate_groups = set()
    for site_id in normalized_target_sites:
        candidate_groups.update(site_to_groups.get(site_id, set()))

    if not candidate_groups:
        return set(base_group_ids), set()

    group_candidate_times = {}
    for group_id in candidate_groups | set(base_group_ids):
        site_time_map = group_site_time_index.get(group_id, {})
        merged_times = []
        for site_id in normalized_target_sites:
            merged_times.extend(site_time_map.get(site_id, []))
        if merged_times:
            group_candidate_times[group_id] = sorted(set(merged_times))

    expanded_groups = set(base_group_ids)
    current_times = []
    for group_id in expanded_groups:
        current_times.extend(group_candidate_times.get(group_id, []))

    if not current_times:
        return expanded_groups, set()

    loose_groups = set()
    pending_groups = {
        group_id for group_id in candidate_groups - expanded_groups
        if group_candidate_times.get(group_id)
    }

    while pending_groups:
        current_windows = build_merged_windows_from_timestamps(current_times, window_seconds)
        if current_windows is None:
            break

        added_groups = []
        for group_id in list(pending_groups):
            candidate_times = group_candidate_times.get(group_id, [])
            matched = any(timestamp_in_windows(ts, current_windows) for ts in candidate_times)
            if matched:
                added_groups.append(group_id)

        if not added_groups:
            break

        for group_id in added_groups:
            pending_groups.discard(group_id)
            expanded_groups.add(group_id)
            loose_groups.add(group_id)
            current_times.extend(group_candidate_times.get(group_id, []))

    return expanded_groups, loose_groups


def collect_groups_by_windows(anchor_windows, candidate_sites, site_to_groups, group_site_time_index, excluded_group_ids=None):
    normalized_candidate_sites = {
        normalize_text(site_id) for site_id in candidate_sites if normalize_text(site_id)
    }
    if not normalized_candidate_sites or anchor_windows is None:
        return set()

    excluded_groups = set(excluded_group_ids or ())
    candidate_groups = set()
    for site_id in normalized_candidate_sites:
        candidate_groups.update(site_to_groups.get(site_id, set()))

    if not candidate_groups:
        return set()

    matched_groups = set()
    for group_id in candidate_groups:
        if group_id in excluded_groups:
            continue
        site_time_map = group_site_time_index.get(group_id, {})
        for site_id in normalized_candidate_sites:
            candidate_times = site_time_map.get(site_id, [])
            if candidate_times and any(timestamp_in_windows(ts, anchor_windows) for ts in candidate_times):
                matched_groups.add(group_id)
                break

    return matched_groups


def collect_groups_by_evidence(site_evidence, alarm_to_groups, excluded_group_ids=None):
    if not isinstance(site_evidence, dict) or not site_evidence:
        return set()

    excluded_groups = set(excluded_group_ids or ())
    matched_groups = set()
    for alarms in site_evidence.values():
        if not isinstance(alarms, list):
            continue
        for record in alarms:
            alarm_key = alarm_record_identity_key(record)
            if alarm_key is None:
                continue
            for group_id in alarm_to_groups.get(alarm_key, ()):
                if group_id not in excluded_groups:
                    matched_groups.add(group_id)
    return matched_groups


def parse_alarm_record_ts(record):
    if not isinstance(record, dict):
        return None

    raw_ts = record.get("ts")
    if raw_ts is not None:
        try:
            return int(raw_ts)
        except (TypeError, ValueError):
            pass

    for field_name in ("告警首次发生时间", "alarm_time", "time_str"):
        text = normalize_text(record.get(field_name, ""))
        if not text:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return int(datetime.strptime(text, fmt).timestamp())
            except ValueError:
                continue

    return None


def load_ne_graph_data(ne_graph_file):
    if not ne_graph_file or not os.path.exists(ne_graph_file):
        return {}
    with open(ne_graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def build_ne_to_domain_map(ne_graph_data):
    ne_to_domain = {}
    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        normalized_ne_id = normalize_text(ne_id)
        if not normalized_ne_id:
            continue
        domain = (
            normalize_text(ne_info.get("domain", ""))
            or normalize_text(ne_info.get("Domain", ""))
            or normalize_text(ne_info.get("DOMAIN", ""))
        ).upper()
        if domain:
            ne_to_domain[normalized_ne_id] = domain
    return ne_to_domain


def build_site_has_domain_map(ne_graph_data, target_domain):
    normalized_target_domain = normalize_text(target_domain).upper()
    if not normalized_target_domain or not isinstance(ne_graph_data, dict):
        return {}

    site_has_domain = defaultdict(bool)
    for ne_info in ne_graph_data.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = normalize_text(ne_info.get("site_id", ""))
        if not site_id:
            continue
        domain = (
            normalize_text(ne_info.get("domain", ""))
            or normalize_text(ne_info.get("Domain", ""))
            or normalize_text(ne_info.get("DOMAIN", ""))
        ).upper()
        if domain == normalized_target_domain:
            site_has_domain[site_id] = True

    return dict(site_has_domain)

def filter_ticket_sites_by_site_flag(ticket_sites, site_flag_map):
    filtered_ticket_sites = {}
    site_flag_map = site_flag_map or {}

    for ticket_id, site_list in ticket_sites.items():
        filtered_site_list = []
        seen_sites = set()
        for site_id in site_list:
            normalized_site_id = normalize_text(site_id)
            if (
                not normalized_site_id
                or normalized_site_id in seen_sites
                or not site_flag_map.get(normalized_site_id, False)
            ):
                continue
            filtered_site_list.append(normalized_site_id)
            seen_sites.add(normalized_site_id)
        if filtered_site_list:
            filtered_ticket_sites[ticket_id] = filtered_site_list

    return filtered_ticket_sites


def site_alarm_map_contains_domain(site_alarm_map, ne_to_domain, target_domain):
    normalized_target_domain = normalize_text(target_domain).upper()
    if not normalized_target_domain or not isinstance(site_alarm_map, dict) or not site_alarm_map:
        return False
    for alarms in site_alarm_map.values():
        if not isinstance(alarms, list):
            continue
        for record in alarms:
            if not isinstance(record, dict):
                continue
            alarm_source = (
                normalize_text(record.get("alarm_source", ""))
                or normalize_text(record.get("告警源", ""))
            )
            if alarm_source and ne_to_domain.get(alarm_source, "") == normalized_target_domain:
                return True
    return False


def build_site_to_ne_ids(ne_graph_data):
    site_to_ne_ids = defaultdict(list)
    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        site_id = normalize_text(ne_info.get("site_id", ""))
        if not site_id:
            continue
        site_to_ne_ids[site_id].append(ne_id)
    return {
        site_id: sorted(ne_ids)
        for site_id, ne_ids in site_to_ne_ids.items()
    }


def build_site_coord_index(ne_graph_data):
    site_coords = {}
    for ne_info in ne_graph_data.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = normalize_text(ne_info.get("site_id", ""))
        if not site_id or site_id in site_coords:
            continue
        latitude = ne_info.get("latitude", ne_info.get("lat"))
        longitude = ne_info.get("longitude", ne_info.get("lon", ne_info.get("lng")))
        if latitude in (None, "") or longitude in (None, ""):
            continue
        try:
            float(latitude)
            float(longitude)
        except (TypeError, ValueError):
            continue
        site_coords[site_id] = (latitude, longitude)
    return site_coords


def load_site_graph_data(site_graph_file=SITE_GRAPH_JSON):
    if not site_graph_file or not os.path.exists(site_graph_file):
        return {}
    with open(site_graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def build_site_coord_index_from_site_graph(site_graph_data):
    site_coords = {}
    for site_id, site_info in (site_graph_data or {}).items():
        if not isinstance(site_info, dict):
            continue
        normalized_site_id = normalize_text(site_id)
        if not normalized_site_id:
            continue
        latitude = site_info.get("latitude", site_info.get("lat"))
        longitude = site_info.get("longitude", site_info.get("lon", site_info.get("lng")))
        if latitude in (None, "") or longitude in (None, ""):
            continue
        try:
            float(latitude)
            float(longitude)
        except (TypeError, ValueError):
            continue
        site_coords[normalized_site_id] = (latitude, longitude)
    return site_coords


def _build_visual_alarm_entry(record, site_id):
    alarm_type = normalize_text(record.get("alarm", "")) or normalize_text(record.get("alarm_type", "")) or normalize_text(record.get("告警标题", ""))
    alarm_time = (
        normalize_text(record.get("time", ""))
        or normalize_text(record.get("alarm_time", ""))
        or normalize_text(record.get("告警首次发生时间", ""))
        or normalize_text(record.get("time_str", ""))
    )
    if not alarm_time:
        ts = parse_alarm_record_ts(record)
        if ts is not None:
            alarm_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    alarm_clear_time = normalize_text(record.get("alarm_clear_time", "")) or normalize_text(record.get("告警清除时间", ""))
    return {
        "alarm_id": normalize_text(record.get("eid", "")) or normalize_text(record.get("alarm_id", "")) or normalize_text(record.get("告警编码ID", "")),
        "alarm_type": alarm_type,
        "alarm_time": alarm_time,
        "alarm_clear_time": alarm_clear_time,
        "domain": normalize_text(record.get("domain", "")),
        "site_id": site_id,
        "site_name": site_id,
        "matched_role": normalize_text(record.get("matched_role", "")),
        "工单号": normalize_text(record.get("工单号", "")),
        "故障组ID": normalize_text(record.get("故障组ID", "")),
        "来源故障组UUID": normalize_text(record.get("来源故障组UUID", "")),
        "mhp_group_id": normalize_text(record.get("mhp_group_id", "")),
        "alarm_group_id": normalize_text(record.get("alarm_group_id", "")),
        "occurrence_uuid": normalize_text(record.get("occurrence_uuid", "")),
    }


def _build_visual_symptom(record, site_id, ticket_id, matched_role):
    ts = parse_alarm_record_ts(record)
    symptom = {
        "node": site_id,
        "ts": ts,
        "eid": normalize_text(record.get("eid", "")) or normalize_text(record.get("alarm_id", "")) or normalize_text(record.get("告警编码ID", "")),
        "alarm": normalize_text(record.get("alarm", "")) or normalize_text(record.get("alarm_type", "")) or normalize_text(record.get("告警标题", "")),
        "alarm_source": normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", "")) or f"CASE_NE::{site_id}",
        "matched_role": matched_role,
        "time_str": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts is not None else "",
        "工单号": normalize_text(record.get("工单号", "")),
        "故障组ID": normalize_text(record.get("故障组ID", "")),
        "来源故障组UUID": normalize_text(record.get("来源故障组UUID", "")),
        "mhp_group_id": normalize_text(record.get("mhp_group_id", "")),
        "alarm_group_id": normalize_text(record.get("alarm_group_id", "")),
        "occurrence_uuid": normalize_text(record.get("occurrence_uuid", "")),
        "告警清除时间": normalize_text(record.get("alarm_clear_time", "")) or normalize_text(record.get("告警清除时间", "")),
    }
    return symptom


def _visual_symptom_occurrence_key(symptom):
    return require_alarm_identity(symptom)


def _append_visual_symptom_once(symptoms, symptom_index, symptom):
    key = _visual_symptom_occurrence_key(symptom)
    existing = symptom_index.get(key)
    if existing is None:
        symptom_index[key] = symptom
        symptoms.append(symptom)
        return

    for field_name in ("故障组ID", "alarm_group_id", "mhp_group_id", "来源故障组UUID"):
        incoming_values = [
            normalize_text(item)
            for item in str(symptom.get(field_name, "") or "").replace(";", ",").split(",")
            if normalize_text(item)
        ]
        if not incoming_values:
            continue
        existing_values = [
            normalize_text(item)
            for item in str(existing.get(field_name, "") or "").replace(";", ",").split(",")
            if normalize_text(item)
        ]
        for incoming_value in incoming_values:
            if incoming_value not in existing_values:
                existing_values.append(incoming_value)
        if existing_values:
            existing[field_name] = ",".join(existing_values)

    role = normalize_text(symptom.get("matched_role", ""))
    if not role:
        return
    role_list = existing.setdefault(
        "matched_role_list",
        [existing.get("matched_role", "")]
        if existing.get("matched_role")
        else []
    )
    if role not in role_list:
        role_list.append(role)


def _build_visual_link_info(ne_id, group_ne_ids, ne_graph_data, group_ne_set=None):
    link_info = {}
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    raw_links = ne_graph_entry.get("link", {})
    if not isinstance(raw_links, dict):
        return link_info

    if group_ne_set is None:
        group_ne_set = set(group_ne_ids)
    for neighbor_id, link_meta in raw_links.items():
        if neighbor_id not in group_ne_set or neighbor_id == ne_id:
            continue

        if isinstance(link_meta, dict):
            connection_types = sorted(str(link_type) for link_type in link_meta.keys())
            topologies = sorted({str(direction) for direction in link_meta.values() if direction})
        else:
            connection_types = [str(link_meta)]
            topologies = []

        link_info[neighbor_id] = {
            "connection_type": ",".join(connection_types),
            "distance": "",
            "topology": ",".join(topologies),
            "time_window": "",
            "left_alarm": {},
            "right_alarm": {},
        }

    return link_info


def build_visualization_case_record(detail, method, ne_graph_data=None, site_to_ne_ids=None, site_coord_index=None):
    ne_graph_data = ne_graph_data or {}
    if site_to_ne_ids is None:
        site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    if site_coord_index is None:
        site_coord_index = build_site_coord_index(ne_graph_data)
        site_coord_index.update(build_site_coord_index_from_site_graph(load_site_graph_data()))

    ticket_id = normalize_text(detail.get("ticket_id", ""))
    associated_sites = sorted(normalize_text(site_id) for site_id in detail.get("associated_sites", []) if normalize_text(site_id))
    missing_sites = sorted(normalize_text(site_id) for site_id in detail.get("missing_sites", []) if normalize_text(site_id))
    context_sites = sorted(normalize_text(site_id) for site_id in detail.get("context_sites", []) if normalize_text(site_id))
    associated_site_set = set(associated_sites)
    missing_site_set = set(missing_sites)
    context_site_set = set(context_sites)
    ticket_sites = sorted(normalize_text(site_id) for site_id in detail.get("ticket_sites", []) if normalize_text(site_id))
    display_sites = sorted(
        normalize_text(site_id)
        for site_id in detail.get("display_sites", ticket_sites)
        if normalize_text(site_id)
    )
    if not display_sites:
        display_sites = list(ticket_sites)

    associated_site_alarms = detail.get("associated_site_alarms", {}) if isinstance(detail.get("associated_site_alarms", {}), dict) else {}
    missing_site_alarms = detail.get("missing_site_alarms", {}) if isinstance(detail.get("missing_site_alarms", {}), dict) else {}
    context_site_alarms = detail.get("context_site_alarms", {}) if isinstance(detail.get("context_site_alarms", {}), dict) else {}
    note = normalize_text(detail.get("note", ""))

    case_uuid = f"{method}::{ticket_id}"
    ne_info = {}
    symptoms = []
    symptom_index = {}
    ne_list = []
    all_case_ne_ids = []
    per_site_alarm_records = {}
    display_site_set = set(display_sites)

    for site_id in display_sites:
        site_alarm_records = []
        site_alarm_records.extend(associated_site_alarms.get(site_id, []))
        site_alarm_records.extend(missing_site_alarms.get(site_id, []))
        site_alarm_records.extend(context_site_alarms.get(site_id, []))
        per_site_alarm_records[site_id] = [record for record in site_alarm_records if isinstance(record, dict)]

        source_ne_ids = {
            normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", ""))
            for record in per_site_alarm_records[site_id]
        }
        source_ne_ids = {ne_id for ne_id in source_ne_ids if ne_id}

        site_ne_ids = list(site_to_ne_ids.get(site_id, []))
        for ne_id in sorted(source_ne_ids):
            if ne_id not in site_ne_ids:
                site_ne_ids.append(ne_id)

        if not site_ne_ids:
            site_ne_ids = [f"SITE::{site_id}"]

        all_case_ne_ids.extend(site_ne_ids)

    group_ne_ids = sorted(dict.fromkeys(all_case_ne_ids))

    for site_id in display_sites:
        site_records = per_site_alarm_records.get(site_id, [])
        site_ne_ids = list(site_to_ne_ids.get(site_id, []))
        for record in site_records:
            source_ne_id = normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", ""))
            if source_ne_id and source_ne_id not in site_ne_ids:
                site_ne_ids.append(source_ne_id)
        if not site_ne_ids:
            site_ne_ids = [f"SITE::{site_id}"]

        source_to_records = defaultdict(list)
        for record in site_records:
            source_ne_id = normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", ""))
            if source_ne_id and source_ne_id in site_ne_ids:
                source_to_records[source_ne_id].append(record)
            else:
                source_to_records[f"SITE::{site_id}"].append(record)

        if source_to_records.get(f"SITE::{site_id}") and f"SITE::{site_id}" not in site_ne_ids:
            site_ne_ids.append(f"SITE::{site_id}")

        for ne_id in site_ne_ids:
            if ne_id in ne_info:
                continue

            ne_entry = ne_graph_data.get(ne_id, {})
            latitude = ne_entry.get("latitude", ne_entry.get("lat"))
            longitude = ne_entry.get("longitude", ne_entry.get("lon", ne_entry.get("lng")))
            if (latitude in (None, "") or longitude in (None, "")) and site_id in site_coord_index:
                latitude, longitude = site_coord_index[site_id]

            role_tags = []
            if site_id in associated_site_set:
                role_tags.append("ASSOCIATED_SITE")
            if site_id in missing_site_set:
                role_tags.append("MISSING_SITE")
            if site_id in context_site_set:
                role_tags.append("CONTEXT_SITE")
            if not role_tags and site_id in display_site_set:
                role_tags.append("DISPLAY_SITE")

            visual_alarms = [
                _build_visual_alarm_entry(record, site_id)
                for record in source_to_records.get(ne_id, [])
            ]

            fallback_name = ne_id
            if ne_id.startswith("SITE::"):
                fallback_name = site_id
            elif visual_alarms:
                fallback_name = (
                    normalize_text(source_to_records.get(ne_id, [{}])[0].get("告警源", ""))
                    or ne_id
                )

            ne_info[ne_id] = {
                "alarm": visual_alarms,
                "link": {},
                "group": case_uuid,
                "name": normalize_text(ne_entry.get("name", "")) or fallback_name,
                "site_id": site_id,
                "site_name": normalize_text(ne_entry.get("site_name", "")) or site_id,
                "type": normalize_text(ne_entry.get("type", "")) or ",".join(role_tags),
                "network_type": normalize_text(ne_entry.get("network_type", "")),
                "manufacturer": normalize_text(ne_entry.get("manufacturer", "")),
                "running_status": normalize_text(ne_entry.get("running_status", "")) or normalize_text(ne_entry.get("status", "")),
                "domain": normalize_text(ne_entry.get("domain", "")) or "RECALL_CASE",
                "region_id": normalize_text(ne_entry.get("region_id", "")),
                "longitude": longitude if longitude is not None else "",
                "latitude": latitude if latitude is not None else "",
            }

    for matched_role, site_ids, site_alarm_map, site_label in (
        ("associated_site", associated_sites, associated_site_alarms, "ASSOCIATED"),
        ("missing_site", missing_sites, missing_site_alarms, "MISSING"),
        ("context_site", context_sites, context_site_alarms, "CONTEXT"),
    ):
        for site_id in site_ids:
            site_records = [record for record in site_alarm_map.get(site_id, []) if isinstance(record, dict)]
            site_ne_ids = list(site_to_ne_ids.get(site_id, []))
            for record in site_records:
                source_ne_id = normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", ""))
                if source_ne_id and source_ne_id not in site_ne_ids:
                    site_ne_ids.append(source_ne_id)
            source_to_records = defaultdict(list)
            unmapped_records = []
            for record in site_records:
                source_ne_id = normalize_text(record.get("alarm_source", "")) or normalize_text(record.get("告警源", ""))
                if source_ne_id and source_ne_id in site_ne_ids:
                    source_to_records[source_ne_id].append(record)
                else:
                    unmapped_records.append(record)

            if unmapped_records and f"SITE::{site_id}" not in site_ne_ids:
                site_ne_ids.append(f"SITE::{site_id}")
            if unmapped_records:
                source_to_records[f"SITE::{site_id}"].extend(unmapped_records)

            for ne_id in site_ne_ids:
                for record in source_to_records.get(ne_id, []):
                    _append_visual_symptom_once(
                        symptoms,
                        symptom_index,
                        _build_visual_symptom(record, site_id, ticket_id, matched_role),
                    )
                if ne_id in ne_info:
                    existing_alarms = ne_info[ne_id].setdefault("alarm", [])
                    existing_alarms.extend(
                        _build_visual_alarm_entry(record, site_id)
                        for record in source_to_records.get(ne_id, [])
                    )
                    ne_info[ne_id]["alarm"] = dedupe_alarm_records(existing_alarms)

    ne_list = sorted(ne_info.keys())
    group_ne_set = set(ne_list)
    for ne_id in ne_list:
        ne_info[ne_id]["link"] = _build_visual_link_info(
            ne_id,
            ne_list,
            ne_graph_data,
            group_ne_set=group_ne_set,
        )

    timestamps = [symptom["ts"] for symptom in symptoms if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None
    group_anchor_time = datetime.fromtimestamp(group_anchor_ts).strftime("%Y-%m-%d %H:%M:%S") if group_anchor_ts is not None else ""

    return {
        "uuid": case_uuid,
        "note": note,
        "rule": f"{method}_unrecalled_case",
        "merged_rules": [f"{method}_unrecalled_case"],
        "related_group_uuids": list(detail.get("fault_groups", [])),
        "inferred_roots": {
            "associated_site": associated_sites,
            "missing_site": missing_sites,
            "context_site": context_sites,
        },
        "role_mapping": {
            "associated_site": associated_sites,
            "missing_site": missing_sites,
            "context_site": context_sites,
        },
        "symptoms": symptoms,
        "group_anchor_ts": group_anchor_ts,
        "group_anchor_time": group_anchor_time,
        "ticket_id": ticket_id,
        "recall": detail.get("recall", 0.0),
        "ticket_site_count": detail.get("ticket_site_count", 0),
        "associated_site_count": detail.get("associated_site_count", 0),
        "missing_site_count": detail.get("missing_site_count", 0),
        "match_info": {
            "uuid": case_uuid,
            "note": note,
            "rule": f"{method}_unrecalled_case",
            "merged_rules": [f"{method}_unrecalled_case"],
            "related_group_uuids": list(detail.get("fault_groups", [])),
            "inferred_roots": {
                "associated_site": associated_sites,
                "missing_site": missing_sites,
                "context_site": context_sites,
            },
            "role_mapping": {
                "associated_site": associated_sites,
                "missing_site": missing_sites,
                "context_site": context_sites,
            },
        },
        "ne_info": ne_info,
        "group_info": {
            case_uuid: {
                "ne_list": ne_list,
                "site_list": display_sites,
            }
        },
    }


def build_unrecalled_visualization_cases(details, method, ne_graph_data=None):
    ne_graph_data = ne_graph_data or {}
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_coord_index = build_site_coord_index(ne_graph_data)
    site_coord_index.update(build_site_coord_index_from_site_graph(load_site_graph_data()))
    return [
        build_visualization_case_record(
            detail,
            method,
            ne_graph_data=ne_graph_data,
            site_to_ne_ids=site_to_ne_ids,
            site_coord_index=site_coord_index,
        )
        for detail in details
        if float(detail.get("recall", 0.0) or 0.0) < 1.0
    ]


def write_jsonl_records(output_path, records):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def derive_case_jsonl_output_path(output_file):
    base, _ext = os.path.splitext(output_file)
    return f"{base}.cases.jsonl"
