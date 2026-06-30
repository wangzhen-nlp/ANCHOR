import time
from collections import defaultdict
from functools import lru_cache
from operator import itemgetter

from fault_grouping_official.alarm_events.identity import require_occurrence_uuid


_SORT_ALARM_KEY = itemgetter("alarm_time", "alarm_id")


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
        "name": ne_graph_entry.get("name", ne_id),
        "type": str(ne_graph_entry.get("type", "")).upper(),
        "network_type": str(ne_graph_entry.get("network_type", "")).upper(),
        "manufacturer": str(ne_graph_entry.get("manufacturer", "")).upper(),
        "running_status": ne_graph_entry.get("running_status", ""),
        "domain_upper": str(ne_graph_entry.get("domain", "")).upper(),
        "domain_raw": ne_graph_entry.get("domain", ""),
        "site_name_from_ne": ne_graph_entry.get("site_name", ""),
    }
    cache[ne_id] = info
    return info


def format_ne_link_info(ne_graph_entry):
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
        formatted_link = {}
        if connection_type:
            formatted_link["connection_type"] = connection_type
        if topology:
            formatted_link["topology"] = topology

        formatted_links[neighbor_id] = formatted_link
    return formatted_links


def get_cached_ne_link_info(ne_id, ne_graph_data, ne_link_info_cache):
    if ne_id not in ne_link_info_cache:
        ne_link_info_cache[ne_id] = format_ne_link_info(ne_graph_data.get(ne_id, {}))
    return ne_link_info_cache[ne_id]


def build_group_link_info(ne_id, group_ne_ids, ne_graph_data, ne_link_info_cache):
    formatted_links = get_cached_ne_link_info(
        ne_id,
        ne_graph_data,
        ne_link_info_cache,
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


def resolve_ne_site_context(ne_id, alarms, ne_graph_data):
    """仅使用 ne_graph；NE 未标站点时从当前告警的唯一站点回填。"""
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    resolved_site_id = ne_graph_entry.get("site_id", "")
    if not resolved_site_id:
        alarm_site_ids = sorted({
            alarm.get("site_id", "")
            for alarm in alarms
            if alarm.get("site_id")
        })
        resolved_site_id = alarm_site_ids[0] if len(alarm_site_ids) == 1 else ""
    return {
        "site_id": resolved_site_id,
        "site_name": ne_graph_entry.get("site_name", ""),
        "region_id": ne_graph_entry.get("region_id", ""),
        "longitude": ne_graph_entry.get("longitude", ""),
        "latitude": ne_graph_entry.get("latitude", ""),
    }


def build_group_output(
    match,
    ne_graph_data,
    site_to_ne_ids,
    ne_link_info_cache,
):
    group_id = match["uuid"]
    ne_info = {}
    ne_alarms = defaultdict(list)
    group_site_ids = set()

    # per-call 本地 memoize：同一 match 内多个 symptom 常落在相同 NE / site，
    # 用 dict 直接做内联缓存（不走 closure，避免 Python 调用开销）。
    local_ne_static = {}

    for nodes in match["role_mapping"].values():
        for site_id in nodes:
            if site_id:
                group_site_ids.add(site_id)

    for symptom in match["symptoms"]:
        site_id = symptom.get("node", "")
        ne_id = symptom.get("alarm_source")
        if not ne_id:
            continue
        representative_eid = symptom.get("eid", "")

        ne_static = local_ne_static.get(ne_id)
        if ne_static is None:
            ne_static = _get_ne_static_info(ne_graph_data, ne_id)
            local_ne_static[ne_id] = ne_static
        if site_id:
            group_site_ids.add(site_id)

        alarm_output = {
            "alarm_id": representative_eid,
            "alarm_type": symptom.get("alarm", ""),
            "alarm_time": _format_ts(symptom.get("ts")),
            "alarm_clear_time": symptom.get("告警清除时间", ""),
            "domain": ne_static["domain_raw"],
            "site_id": site_id,
            "site_name": ne_static["site_name_from_ne"],
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

    group_ne_id_set = set(ne_alarms.keys())
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
        site_context = resolve_ne_site_context(ne_id, alarms, ne_graph_data)
        site_id = site_context["site_id"]
        if site_id:
            group_site_ids.add(site_id)

        node_info = {
            "link": build_group_link_info(
                ne_id,
                group_ne_id_set,
                ne_graph_data,
                ne_link_info_cache=ne_link_info_cache,
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

        ne_info[ne_id] = node_info

    return {
        "match_info": {
            "uuid": match["uuid"],
            "rule": match["rule"],
            "merged_rules": match["merged_rules"],
            "related_group_uuids": match.get("related_group_uuids", []),
            "inferred_roots": match["inferred_roots"],
            "role_mapping": match["role_mapping"],
        },
        "ne_info": ne_info,
        "group_info": {
            group_id: {
                "ne_list": group_ne_ids,
                "site_list": sorted(group_site_ids),
            }
        }
    }


def enrich_match_symptoms(match):
    enriched_symptoms = []

    for symptom in match["symptoms"]:
        enriched_symptom = dict(symptom)
        alarm_payload = enriched_symptom.pop("alarm_payload", None)
        if isinstance(alarm_payload, dict):
            for field_name in ("工单号", "故障组ID", "告警清除时间"):
                value = str(alarm_payload.get(field_name, "")).strip()
                if value and not enriched_symptom.get(field_name):
                    enriched_symptom[field_name] = value
        enriched_symptoms.append(enriched_symptom)
    return enriched_symptoms


def build_jsonl_match_output(
    match,
    ne_graph_data,
    site_to_ne_ids,
    ne_link_info_cache,
):
    enriched_match = dict(match)
    enriched_match["symptoms"] = enrich_match_symptoms(match)

    group_output = build_group_output(
        enriched_match,
        ne_graph_data,
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
    )
    timestamps = [symptom["ts"] for symptom in enriched_match["symptoms"]]
    group_anchor_ts = min(timestamps) if timestamps else None

    enriched_match["group_anchor_ts"] = group_anchor_ts
    enriched_match["group_anchor_time"] = _format_ts(group_anchor_ts)
    enriched_match.update(group_output)
    return enriched_match
