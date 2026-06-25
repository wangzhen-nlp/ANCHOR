import time
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from operator import itemgetter

from fault_grouping.alarm_events.identity import require_alarm_identity, require_occurrence_uuid


_SORT_ALARM_KEY = itemgetter("alarm_time", "alarm_id")
_EMPTY_DICT = {}  # 共享只读空 dict，避免 site_id 为空时每次创建新 dict


# timestamp → "%Y-%m-%d %H:%M:%S"。
# - lru_cache(4096)：上限 4096 个唯一整秒（~70 分钟），内存 ≤600KB，自动 LRU 淘汰，
#   无周期性 clear() 抖动；告警流量再大也不会膨胀。
# - 命中时 ~50ns；未命中走手写 f-string + time.localtime（C 实现），
#   比 datetime.fromtimestamp().strftime() 快 ~2 倍，即使长时间分散数据也不会更慢。
@lru_cache(maxsize=4096)
def _format_int_ts(int_ts):
    lt = time.localtime(int_ts)
    return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d} {lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"


def _format_ts(ts):
    if ts is None:
        return ""
    return _format_int_ts(int(ts))


# 模块级 NE 静态字段缓存：把 build_group_output 内每个 NE 每次都跑一遍的
# .get(...) / str(...).upper() 等静态字段一次性算好。
# key 用 id(ne_graph_data)，不同 engine / reload 自动隔离。
_NE_STATIC_CACHE = {}


def _get_ne_static_info(ne_graph_data, ne_id):
    """返回某 NE 的静态展示字段 dict（name/type/manufacturer/domain/...）。"""
    cache = _NE_STATIC_CACHE.setdefault(id(ne_graph_data), {})
    cached = cache.get(ne_id)
    if cached is not None:
        return cached
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    info = {
        "ne_graph_entry": ne_graph_entry,  # 留一份给 resolve_ne_site_context 等仍要 raw entry 的地方
        "name": ne_graph_entry.get("name", ne_id),
        "type": str(ne_graph_entry.get("type", "")).upper(),
        "network_type": str(ne_graph_entry.get("network_type", "")).upper(),
        "manufacturer": str(ne_graph_entry.get("manufacturer", "")).upper(),
        "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
        "domain_upper": str(ne_graph_entry.get("domain", "")).upper(),
        "domain_raw": ne_graph_entry.get("domain", ""),
        "site_id_from_ne": ne_graph_entry.get("site_id", ""),
        "site_name_from_ne": ne_graph_entry.get("site_name", ""),
    }
    cache[ne_id] = info
    return info


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


# 静态 site context 缓存：当 NE 在 ne_graph 中已有 site_id 时（典型情况），
# resolve_ne_site_context 的结果完全由 (ne_graph_data, site_graph_data, ne_id) 决定，
# 与运行时 alarms 无关，可以一次性缓存。
# key 用 (id(ne_graph_data), id(site_graph_data), ne_id)，不同 engine 自动隔离。
# 内存上限：NE 数 × 引擎实例数（典型 1-2），与告警流量无关。
_NE_STATIC_SITE_CONTEXT_SENTINEL = object()  # 表示"NE 没有静态 site_id，走慢路径"
_NE_STATIC_SITE_CONTEXT_CACHE = {}


def _get_static_site_context(ne_graph_data, site_graph_data, ne_id):
    """若 NE 自身有 site_id，返回缓存的静态 site context；否则返回 None。"""
    cache_key = (id(ne_graph_data), id(site_graph_data), ne_id)
    cached = _NE_STATIC_SITE_CONTEXT_CACHE.get(cache_key)
    if cached is _NE_STATIC_SITE_CONTEXT_SENTINEL:
        return None
    if cached is not None:
        return cached

    ne_graph_entry = ne_graph_data.get(ne_id, {})
    resolved_site_id = ne_graph_entry.get("site_id", "")
    if not resolved_site_id:
        _NE_STATIC_SITE_CONTEXT_CACHE[cache_key] = _NE_STATIC_SITE_CONTEXT_SENTINEL
        return None

    site_graph_entry = site_graph_data.get(resolved_site_id, {})
    result = {
        "site_id": resolved_site_id,
        "site_name": ne_graph_entry.get("site_name", "") or site_graph_entry.get("site_name", ""),
        "site_type": ne_graph_entry.get("site_type", "") or site_graph_entry.get("site_type", ""),
        "region_id": ne_graph_entry.get("region_id", "") or site_graph_entry.get("region_id", ""),
        "longitude": ne_graph_entry.get("longitude", "") or site_graph_entry.get("longitude", ""),
        "latitude": ne_graph_entry.get("latitude", "") or site_graph_entry.get("latitude", ""),
    }
    _NE_STATIC_SITE_CONTEXT_CACHE[cache_key] = result
    return result


def resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data):
    # 快路径：NE 自身有 site_id → 与 alarms 无关，复用静态结果
    static_ctx = _get_static_site_context(ne_graph_data, site_graph_data, ne_id)
    if static_ctx is not None:
        return static_ctx

    # 慢路径：NE 没有静态 site_id，需要看 alarms 单点收敛
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    alarm_site_ids = sorted({
        alarm.get("site_id", "")
        for alarm in alarms
        if alarm.get("site_id")
    })
    resolved_site_id = alarm_site_ids[0] if len(alarm_site_ids) == 1 else ""

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

    # per-call 本地 memoize：同一 match 内多个 symptom 常落在相同 NE / site，
    # 用 dict 直接做内联缓存（不走 closure，避免 Python 调用开销）。
    local_ne_static = {}
    local_site_entry = {}

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
        representative_eid = symptom.get("eid", "")
        if representative_eid in (None, ""):
            representative_eid = eid_list[0] if eid_list else ""

        ne_static = local_ne_static.get(ne_id)
        if ne_static is None:
            ne_static = _get_ne_static_info(ne_graph_data, ne_id)
            local_ne_static[ne_id] = ne_static
        if site_id:
            group_site_ids.add(site_id)
            site_graph_entry = local_site_entry.get(site_id)
            if site_graph_entry is None:
                site_graph_entry = site_graph_data.get(site_id, _EMPTY_DICT)
                local_site_entry[site_id] = site_graph_entry
        else:
            site_graph_entry = _EMPTY_DICT

        alarm_output = {
            "alarm_id": representative_eid,
            "alarm_type": symptom.get("alarm", ""),
            "alarm_time": _format_ts(symptom.get("ts")),
            "alarm_clear_time": symptom.get("告警清除时间", ""),
            "domain": ne_static["domain_raw"],
            "site_id": site_id,
            "site_name": ne_static["site_name_from_ne"] or site_graph_entry.get("site_name", ""),
            "matched_role": symptom.get("matched_role", ""),
            "matched_rule": symptom.get("matched_rule", ""),
            "matched_role_key": symptom.get("matched_role_key", ""),
            "工单号": symptom.get("工单号", ""),
            "故障组ID": symptom.get("故障组ID", ""),
        }
        for field_name in ("matched_rule_list", "matched_role_list", "matched_role_key_list"):
            field_value = symptom.get(field_name)
            if isinstance(field_value, list) and field_value:
                alarm_output[field_name] = field_value
        alarm_output["occurrence_uuid"] = require_occurrence_uuid(symptom)
        mhp_event_index = symptom.get("_mhp_event_index")
        if mhp_event_index not in (None, ""):
            alarm_output["_mhp_event_index"] = mhp_event_index
        ne_alarms[ne_id].append(alarm_output)
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
        ne_static = local_ne_static.get(ne_id)
        if ne_static is None:
            ne_static = _get_ne_static_info(ne_graph_data, ne_id)
            local_ne_static[ne_id] = ne_static
        alarms = sorted(ne_alarms.get(ne_id, []), key=_SORT_ALARM_KEY)
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
            "name": ne_static["name"],
            "site_id": site_id,
            "site_name": site_context["site_name"],
            "type": ne_static["type"],
            "network_type": ne_static["network_type"],
            "manufacturer": ne_static["manufacturer"],
            "running_status": ne_static["running_status"],
            "domain": ne_static["domain_upper"],
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
            "uses_missing_topology": bool(match.get("uses_missing_topology")),
            "missing_topology_edges": match.get("missing_topology_edges", []),
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

    def normalize(value):
        text = str(value or "").strip()
        return "" if text.lower() in {"nan", "none", "null", "undefined"} else text

    def merge_metadata(key, metadata):
        existing = alarm_metadata_index.setdefault(key, {})
        for field_name, value in metadata.items():
            if value and not existing.get(field_name):
                existing[field_name] = value

    for item in valid_alarms:
        alarm = item.get("alarm", {})
        event_id = alarm.get("告警编码ID", "")
        if not event_id:
            continue

        identity = require_alarm_identity({
            "eid": event_id,
            "occurrence_uuid": item.get("occurrence_uuid"),
        })

        field_aliases = {
            "工单号": ("工单号",),
            "故障组ID": ("故障组ID",),
            "告警清除时间": ("告警清除时间",),
        }
        metadata = {}
        for field_name, aliases in field_aliases.items():
            value = ""
            for alias in aliases:
                raw_value = str(alarm.get(alias, "")).strip()
                if raw_value:
                    value = raw_value
                    break
            if value:
                metadata[field_name] = value

        if not metadata:
            continue

        merge_metadata(identity, metadata)

    return alarm_metadata_index


def enrich_match_symptoms(match, alarm_metadata_index, include_eid_list=False):
    enriched_symptoms = []

    def normalize(value):
        text = str(value or "").strip()
        return "" if text.lower() in {"nan", "none", "null", "undefined"} else text

    def lookup_metadata(symptom, event_id):
        return alarm_metadata_index.get(require_alarm_identity({
            "eid": event_id,
            "occurrence_uuid": symptom.get("occurrence_uuid"),
        }), {})

    for symptom in match.get("symptoms", []):
        enriched_symptom = dict(symptom)
        for internal_field in ("_segment_key", "_segment_start_ts", "_segment_end_ts", "alarm_payload"):
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
        event_id = enriched_symptom.get("eid", "")
        if event_id in (None, ""):
            event_id = eid_list[0] if eid_list else ""
        if event_id and not enriched_symptom.get("eid"):
            enriched_symptom["eid"] = event_id
        if event_id:
            metadata = lookup_metadata(enriched_symptom, event_id)
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
    enriched_match["group_anchor_time"] = _format_ts(group_anchor_ts)
    enriched_match.update(group_output)
    return enriched_match
