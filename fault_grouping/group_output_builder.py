from collections import defaultdict
from datetime import datetime


def format_ne_link_info(ne_graph_entry, compact_output=False):
    raw_links = ne_graph_entry.get("link", {}) if isinstance(ne_graph_entry, dict) else {}
    if not isinstance(raw_links, dict):
        return {}

    formatted_links = {}
    for neighbor_id, link_meta in raw_links.items():
        if isinstance(link_meta, dict):
            connection_types = sorted(str(link_type) for link_type in link_meta.keys())
            topologies = sorted({str(direction) for direction in link_meta.values() if direction})
        else:
            connection_types = [str(link_meta)]
            topologies = []

        connection_type = ",".join(connection_types)
        topology = ",".join(topologies)
        if compact_output:
            formatted_link = {}
            if connection_type:
                formatted_link["connection_type"] = connection_type
            if topology:
                formatted_link["topology"] = topology
        else:
            formatted_link = {
                "connection_type": connection_type,
                "distance": "",
                "topology": topology,
                "time_window": "",
                "left_alarm": {},
                "right_alarm": {},
            }

        formatted_links[neighbor_id] = formatted_link
    return formatted_links


def get_cached_ne_link_info(ne_id, ne_graph_data, ne_link_info_cache, compact_output=False):
    if ne_link_info_cache is None:
        return format_ne_link_info(ne_graph_data.get(ne_id, {}), compact_output=compact_output)
    cache_key = (ne_id, compact_output)
    if cache_key not in ne_link_info_cache:
        ne_link_info_cache[cache_key] = format_ne_link_info(
            ne_graph_data.get(ne_id, {}),
            compact_output=compact_output,
        )
    return ne_link_info_cache[cache_key]


def build_group_link_info(ne_id, group_ne_ids, ne_graph_data, ne_link_info_cache=None, compact_output=False):
    formatted_links = get_cached_ne_link_info(
        ne_id,
        ne_graph_data,
        ne_link_info_cache,
        compact_output=compact_output,
    )
    link_info = {}

    if len(group_ne_ids) < len(formatted_links):
        for neighbor_id in group_ne_ids:
            if neighbor_id == ne_id:
                continue
            if neighbor_id in formatted_links:
                link_info[neighbor_id] = formatted_links[neighbor_id]
        return link_info

    for neighbor_id, formatted_link in formatted_links.items():
        if neighbor_id in group_ne_ids:
            link_info[neighbor_id] = formatted_link

    return link_info


def resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data):
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    resolved_site_id = ne_graph_entry.get("site_id", "")

    alarm_site_ids = sorted({
        alarm.get("site_id", "")
        for alarm in alarms
        if alarm.get("site_id")
    })
    if not resolved_site_id and len(alarm_site_ids) == 1:
        resolved_site_id = alarm_site_ids[0]

    site_graph_entry = site_graph_data.get(resolved_site_id, {}) if resolved_site_id else {}
    return {
        "site_id": resolved_site_id,
        "site_name": ne_graph_entry.get("site_name", "") or site_graph_entry.get("site_name", ""),
        "site_type": ne_graph_entry.get("site_type", "") or site_graph_entry.get("site_type", ""),
        "region_id": ne_graph_entry.get("region_id", "") or site_graph_entry.get("region_id", ""),
        "longitude": ne_graph_entry.get("longitude", "") or site_graph_entry.get("longitude", ""),
        "latitude": ne_graph_entry.get("latitude", "") or site_graph_entry.get("latitude", ""),
    }


def build_group_output(
    match,
    ne_graph_data,
    site_graph_data,
    site_to_ne_ids=None,
    ne_link_info_cache=None,
    compact_output=False,
    include_eid_list=False,
):
    group_id = match.get("uuid", "")
    ne_info = {}
    ne_alarms = defaultdict(list)
    group_site_ids = set()

    for nodes in match.get("role_mapping", {}).values():
        for site_id in nodes:
            if site_id:
                group_site_ids.add(site_id)

    for symptom in match.get("symptoms", []):
        site_id = symptom.get("node", "")
        ne_id = symptom.get("alarm_source")
        if not ne_id:
            continue
        eid_list = [
            event_id
            for event_id in (symptom.get("eid_list") or [])
            if event_id not in (None, "")
        ]
        representative_eid = symptom.get("eid", "") or (eid_list[0] if eid_list else "")

        ne_graph_entry = ne_graph_data.get(ne_id, {})
        if site_id:
            group_site_ids.add(site_id)
        site_graph_entry = site_graph_data.get(site_id, {}) if site_id else {}

        ne_alarms[ne_id].append({
            "alarm_id": representative_eid,
            "alarm_type": symptom.get("alarm", ""),
            "alarm_time": datetime.fromtimestamp(symptom["ts"]).strftime("%Y-%m-%d %H:%M:%S") if symptom.get("ts") is not None else "",
            "alarm_clear_time": symptom.get("告警清除时间", ""),
            "domain": ne_graph_entry.get("domain", ""),
            "site_id": site_id,
            "site_name": ne_graph_entry.get("site_name", "") or site_graph_entry.get("site_name", ""),
            "matched_role": symptom.get("matched_role", ""),
            "工单号": symptom.get("工单号", ""),
            "故障组ID": symptom.get("故障组ID", ""),
        })
        if include_eid_list and eid_list:
            ne_alarms[ne_id][-1]["alarm_id_list"] = eid_list

    group_ne_id_set = set(ne_alarms.keys())
    if site_to_ne_ids is None:
        group_ne_id_set.update(
            ne_id
            for ne_id, ne_graph_entry in ne_graph_data.items()
            if ne_graph_entry.get("site_id", "") in group_site_ids
        )
    else:
        for site_id in group_site_ids:
            group_ne_id_set.update(site_to_ne_ids.get(site_id, ()))
    group_ne_ids = sorted(group_ne_id_set)
    group_ne_id_set = set(group_ne_ids)

    for ne_id in group_ne_ids:
        ne_graph_entry = ne_graph_data.get(ne_id, {})
        alarms = sorted(
            ne_alarms.get(ne_id, []),
            key=lambda alarm: (alarm.get("alarm_time", ""), alarm.get("alarm_id", ""))
        )
        site_context = resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data)
        site_id = site_context["site_id"]
        if site_id:
            group_site_ids.add(site_id)

        node_info = {
            "link": build_group_link_info(
                ne_id,
                group_ne_id_set,
                ne_graph_data,
                ne_link_info_cache=ne_link_info_cache,
                compact_output=compact_output,
            ),
            "group": group_id,
            "name": ne_graph_entry.get("name", ne_id),
            "site_id": site_id,
            "site_name": site_context["site_name"],
            "type": str(ne_graph_entry.get("type", "")).upper(),
            "network_type": str(ne_graph_entry.get("network_type", "")).upper(),
            "manufacturer": str(ne_graph_entry.get("manufacturer", "")).upper(),
            "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
            "domain": str(ne_graph_entry.get("domain", "")).upper(),
            "region_id": site_context["region_id"],
            "longitude": site_context["longitude"],
            "latitude": site_context["latitude"],
        }
        if not compact_output:
            node_info["alarm"] = alarms

        ne_info[ne_id] = node_info

    return {
        "match_info": {
            "uuid": match.get("uuid", ""),
            "rule": match.get("rule", ""),
            "merged_rules": match.get("merged_rules", []),
            "related_group_uuids": match.get("related_group_uuids", []),
            "inferred_roots": match.get("inferred_roots", {}),
            "role_mapping": match.get("role_mapping", {}),
        },
        "ne_info": ne_info,
        "group_info": {
            group_id: {
                "ne_list": group_ne_ids,
                "site_list": sorted(group_site_ids),
            }
        }
    }


def build_alarm_metadata_index(valid_alarms):
    alarm_metadata_index = {}
    for item in valid_alarms:
        alarm = item.get("alarm", {})
        event_id = alarm.get("告警编码ID", "")
        if not event_id:
            continue

        existing = alarm_metadata_index.setdefault(event_id, {})
        field_aliases = {
            "工单号": ("工单号",),
            "故障组ID": ("故障组ID",),
            "告警清除时间": ("告警清除时间",),
        }
        for field_name, aliases in field_aliases.items():
            value = ""
            for alias in aliases:
                raw_value = str(alarm.get(alias, "")).strip()
                if raw_value:
                    value = raw_value
                    break
            if value and not existing.get(field_name):
                existing[field_name] = value

    return alarm_metadata_index


def enrich_match_symptoms(match, alarm_metadata_index, include_eid_list=False):
    enriched_symptoms = []
    for symptom in match.get("symptoms", []):
        enriched_symptom = dict(symptom)
        for internal_field in ("_segment_key", "_segment_start_ts", "_segment_end_ts"):
            enriched_symptom.pop(internal_field, None)
        if not include_eid_list:
            enriched_symptom.pop("eid_list", None)
        eid_list = [
            event_id
            for event_id in (enriched_symptom.get("eid_list") or [])
            if event_id not in (None, "")
        ]
        if include_eid_list and eid_list:
            enriched_symptom["eid_list"] = eid_list
        event_id = enriched_symptom.get("eid", "") or (eid_list[0] if eid_list else "")
        if event_id and not enriched_symptom.get("eid"):
            enriched_symptom["eid"] = event_id
        if event_id:
            metadata = alarm_metadata_index.get(event_id, {})
            for field_name in ("工单号", "故障组ID", "告警清除时间"):
                if metadata.get(field_name) and not enriched_symptom.get(field_name):
                    enriched_symptom[field_name] = metadata[field_name]
        enriched_symptoms.append(enriched_symptom)
    return enriched_symptoms


def build_jsonl_match_output(
    match,
    ne_graph_data,
    site_graph_data,
    alarm_metadata_index,
    site_to_ne_ids=None,
    ne_link_info_cache=None,
    compact_output=False,
    include_eid_list=False,
):
    enriched_match = dict(match)
    enriched_match["symptoms"] = enrich_match_symptoms(
        match,
        alarm_metadata_index,
        include_eid_list=include_eid_list,
    )

    group_output = build_group_output(
        enriched_match,
        ne_graph_data,
        site_graph_data,
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
        compact_output=compact_output,
        include_eid_list=include_eid_list,
    )
    timestamps = [symptom["ts"] for symptom in enriched_match.get("symptoms", []) if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None

    enriched_match["group_anchor_ts"] = group_anchor_ts
    enriched_match["group_anchor_time"] = (
        datetime.fromtimestamp(group_anchor_ts).strftime("%Y-%m-%d %H:%M:%S")
        if group_anchor_ts is not None else ""
    )
    enriched_match.update(group_output)
    return enriched_match
