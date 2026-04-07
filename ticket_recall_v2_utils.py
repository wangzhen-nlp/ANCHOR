import json
import os
from datetime import datetime
from collections import defaultdict

from compute_ticket_site_recall import _normalize_text


def normalize_text(value):
    return _normalize_text(value)


def dedupe_alarm_records(records):
    seen = set()
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = (
            normalize_text(record.get("告警编码ID", "")),
            normalize_text(record.get("故障组ID", "")),
            normalize_text(record.get("来源故障组UUID", "")),
            normalize_text(record.get("工单号", "")),
            normalize_text(record.get("站点ID", "")),
            normalize_text(record.get("关联站点ID", "")),
            normalize_text(record.get("告警源", "")),
            normalize_text(record.get("告警标题", "")),
            normalize_text(record.get("告警首次发生时间", "")),
            normalize_text(record.get("告警最后发生时间", "")),
            normalize_text(record.get("告警清除时间", "")),
            normalize_text(record.get("node", "")),
            normalize_text(record.get("matched_role", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def load_upper_bound_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("召回率上限结果 JSON 顶层必须是对象")

    ticket_index = {}
    for item in data.get("details", []):
        if not isinstance(item, dict):
            continue

        ticket_id = normalize_text(item.get("ticket_id", ""))
        if not ticket_id:
            continue

        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
        associated_site_count = int(item.get("associated_site_count", 0) or 0)

        merged_site_evidence = defaultdict(list)
        evidence = item.get("evidence", {})
        if isinstance(evidence, dict):
            for bucket_name in ("direct_site_alarms", "inferred_site_alarms"):
                bucket = evidence.get(bucket_name, {})
                if not isinstance(bucket, dict):
                    continue
                for site_id, alarms in bucket.items():
                    normalized_site_id = normalize_text(site_id)
                    if not normalized_site_id or not isinstance(alarms, list):
                        continue
                    merged_site_evidence[normalized_site_id].extend(
                        alarm for alarm in alarms if isinstance(alarm, dict)
                    )

        ticket_index[ticket_id] = {
            "ticket_site_count": ticket_site_count,
            "associated_site_count": associated_site_count,
            "fully_associable": ticket_site_count > 0 and associated_site_count == ticket_site_count,
            "site_evidence": {
                site_id: dedupe_alarm_records(alarms)
                for site_id, alarms in sorted(merged_site_evidence.items())
            },
        }

    return ticket_index


def build_site_alarm_map_for_sites(site_alarm_map, site_ids):
    result = {}
    for site_id in sorted(site_ids):
        result[site_id] = dedupe_alarm_records(site_alarm_map.get(site_id, []))
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


def parse_alarm_record_ts(record):
    if not isinstance(record, dict):
        return None

    raw_ts = record.get("ts")
    if raw_ts is not None:
        try:
            return int(raw_ts)
        except (TypeError, ValueError):
            pass

    for field_name in ("告警首次发生时间", "alarm_time"):
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


def _build_visual_alarm_entry(record, site_id):
    alarm_type = normalize_text(record.get("alarm", "")) or normalize_text(record.get("alarm_type", "")) or normalize_text(record.get("告警标题", ""))
    alarm_time = normalize_text(record.get("alarm_time", "")) or normalize_text(record.get("告警首次发生时间", ""))
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
        "工单号": normalize_text(record.get("工单号", "")) or ticket_id,
        "故障组ID": normalize_text(record.get("故障组ID", "")),
        "告警清除时间": normalize_text(record.get("alarm_clear_time", "")) or normalize_text(record.get("告警清除时间", "")),
    }
    return symptom


def _build_visual_link_info(ne_id, group_ne_ids, ne_graph_data):
    link_info = {}
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    raw_links = ne_graph_entry.get("link", {})
    if not isinstance(raw_links, dict):
        return link_info

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
    site_to_ne_ids = site_to_ne_ids or {}
    site_coord_index = site_coord_index or {}

    ticket_id = normalize_text(detail.get("ticket_id", ""))
    associated_sites = sorted(normalize_text(site_id) for site_id in detail.get("associated_sites", []) if normalize_text(site_id))
    missing_sites = sorted(normalize_text(site_id) for site_id in detail.get("missing_sites", []) if normalize_text(site_id))
    ticket_sites = sorted(normalize_text(site_id) for site_id in detail.get("ticket_sites", []) if normalize_text(site_id))

    associated_site_alarms = detail.get("associated_site_alarms", {}) if isinstance(detail.get("associated_site_alarms", {}), dict) else {}
    missing_site_alarms = detail.get("missing_site_alarms", {}) if isinstance(detail.get("missing_site_alarms", {}), dict) else {}

    case_uuid = f"{method}::{ticket_id}"
    ne_info = {}
    symptoms = []
    ne_list = []
    all_case_ne_ids = []
    per_site_alarm_records = {}

    for site_id in ticket_sites:
        site_alarm_records = []
        site_alarm_records.extend(associated_site_alarms.get(site_id, []))
        site_alarm_records.extend(missing_site_alarms.get(site_id, []))
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

    for matched_role, site_ids, site_alarm_map, site_label in (
        ("associated_site", associated_sites, associated_site_alarms, "ASSOCIATED"),
        ("missing_site", missing_sites, missing_site_alarms, "MISSING"),
    ):
        for site_id in site_ids:
            site_records = [record for record in site_alarm_map.get(site_id, []) if isinstance(record, dict)]
            site_ne_ids = [
                ne_id for ne_id in group_ne_ids
                if normalize_text(ne_graph_data.get(ne_id, {}).get("site_id", "")) == site_id
                or ne_id == f"SITE::{site_id}"
            ]
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
                if ne_id in ne_info:
                    continue

                ne_entry = ne_graph_data.get(ne_id, {})
                latitude = ne_entry.get("latitude", ne_entry.get("lat"))
                longitude = ne_entry.get("longitude", ne_entry.get("lon", ne_entry.get("lng")))
                if (latitude in (None, "") or longitude in (None, "")) and site_id in site_coord_index:
                    latitude, longitude = site_coord_index[site_id]

                visual_alarms = [
                    _build_visual_alarm_entry(record, site_id)
                    for record in source_to_records.get(ne_id, [])
                ]
                for record in source_to_records.get(ne_id, []):
                    symptoms.append(_build_visual_symptom(record, site_id, ticket_id, matched_role))

                ne_info[ne_id] = {
                    "alarm": visual_alarms,
                    "link": {},
                    "group": case_uuid,
                    "name": normalize_text(ne_entry.get("name", "")) or ne_id,
                    "site_id": site_id,
                    "site_name": normalize_text(ne_entry.get("site_name", "")) or site_id,
                    "type": normalize_text(ne_entry.get("type", "")) or f"{site_label}_SITE",
                    "network_type": normalize_text(ne_entry.get("network_type", "")),
                    "manufacturer": normalize_text(ne_entry.get("manufacturer", "")),
                    "running_status": normalize_text(ne_entry.get("running_status", "")) or normalize_text(ne_entry.get("status", "")),
                    "domain": normalize_text(ne_entry.get("domain", "")) or "RECALL_CASE",
                    "region_id": normalize_text(ne_entry.get("region_id", "")),
                    "longitude": longitude if longitude is not None else "",
                    "latitude": latitude if latitude is not None else "",
                }

    ne_list = sorted(ne_info.keys())
    for ne_id in ne_list:
        ne_info[ne_id]["link"] = _build_visual_link_info(ne_id, ne_list, ne_graph_data)

    timestamps = [symptom["ts"] for symptom in symptoms if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None
    group_anchor_time = datetime.fromtimestamp(group_anchor_ts).strftime("%Y-%m-%d %H:%M:%S") if group_anchor_ts is not None else ""

    return {
        "uuid": case_uuid,
        "rule": f"{method}_unrecalled_case",
        "merged_rules": [f"{method}_unrecalled_case"],
        "related_group_uuids": list(detail.get("fault_groups", [])),
        "inferred_roots": {
            "associated_site": associated_sites,
            "missing_site": missing_sites,
        },
        "role_mapping": {
            "associated_site": associated_sites,
            "missing_site": missing_sites,
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
            "rule": f"{method}_unrecalled_case",
            "merged_rules": [f"{method}_unrecalled_case"],
            "related_group_uuids": list(detail.get("fault_groups", [])),
            "inferred_roots": {
                "associated_site": associated_sites,
                "missing_site": missing_sites,
            },
            "role_mapping": {
                "associated_site": associated_sites,
                "missing_site": missing_sites,
            },
        },
        "ne_info": ne_info,
        "group_info": {
            case_uuid: {
                "ne_list": ne_list,
                "site_list": ticket_sites,
            }
        },
    }


def build_unrecalled_visualization_cases(details, method, ne_graph_data=None):
    ne_graph_data = ne_graph_data or {}
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_coord_index = build_site_coord_index(ne_graph_data)
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
