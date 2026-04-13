import json
import os
import time
import threading

from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import (
    build_ne_to_site_map,
    load_site_graph,
    stream_alarm_inputs,
)
from alarm_tools.alarm_types import CRITICAL_ALARMS, POWER_ALARMS
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_DEVICE_COUNTS_JSON,
    SITE_GRAPH_BY_NE_JSON,
    SITE_GRAPH_JSON,
    resource_display,
)
from ticket_recall.evaluation.compute_group_output_ticket_recall import compute_group_output_ticket_recall
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.reports import generate_incident_report
from fault_grouping.rule_config import transmission_rule, link_rule, power_rule
from fault_grouping.temporal_graph_engine import TemporalGraphEngine


def _parse_datetime_text(text, field_name="时间"):
    text = str(text).strip()
    if not text:
        raise ValueError(f"{field_name}为空")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("T", " "))
    except ValueError as exc:
        raise ValueError(f"{field_name}格式无法解析: {text}") from exc


def _is_clear_alarm(alarm):
    clear_value = alarm.get("清除告警", None)
    if clear_value is None:
        return False

    return str(clear_value).strip().lower() in {"是", "yes", "true", "1", "y"}


def _stream_alarms_by_ts(valid_alarms, speedup=1.0):
    """按告警 ts 的时间差模拟实时流式输出。"""
    previous_ts = None
    speedup = max(float(speedup), 1e-9)

    for item in valid_alarms:
        current_ts = item["ts"]
        if previous_ts is not None:
            sleep_sec = max(0.0, (current_ts - previous_ts) / speedup)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        yield item
        previous_ts = current_ts


def _append_alarm_event(valid_alarms, alarm, site_id, alarm_title, event_time_str, is_clear=False):
    dt_obj = _parse_datetime_text(event_time_str, "告警时间")
    event_alarm = dict(alarm)
    event_alarm["告警首次发生时间"] = event_time_str
    if is_clear:
        event_alarm["清除告警"] = "是"

    valid_alarms.append({
        "alarm": event_alarm,
        "site_id": site_id,
        "alarm_source": alarm.get("告警源", ""),
        "alarm_title": alarm_title,
        "ts": dt_obj.timestamp()
    })


def _load_valid_alarms(alarm_file_path, valid_alarm_titles, valid_sites, ne_to_site, start_ts=None, end_ts=None):
    processed_count = 0
    valid_alarms = []
    normal_alarm_count = 0
    clear_alarm_count = 0

    for alarm in stream_alarm_inputs(alarm_file_path, show_progress=True):
        processed_count += 1

        alarm_title = alarm.get('告警标题', '')
        if alarm_title not in valid_alarm_titles:
            continue

        site_id = alarm.get('站点ID', '')
        if not site_id or site_id not in valid_sites:
            alarm_source = alarm.get('告警源', '')
            site_id = ne_to_site.get(alarm_source, '')

        if not site_id or site_id not in valid_sites:
            continue

        first_occurrence_str = str(alarm.get("告警首次发生时间", "")).strip()
        first_occurrence_dt = _parse_datetime_text(first_occurrence_str, "告警首次发生时间")
        first_occurrence_ts = first_occurrence_dt.timestamp()
        if start_ts is not None and first_occurrence_ts < start_ts:
            continue
        if end_ts is not None and first_occurrence_ts > end_ts:
            continue

        _append_alarm_event(
            valid_alarms,
            alarm,
            site_id,
            alarm_title,
            first_occurrence_str,
            is_clear=False
        )
        normal_alarm_count += 1

        clear_time_str = str(alarm.get("告警清除时间", "")).strip()
        if clear_time_str:
            _append_alarm_event(
                valid_alarms,
                alarm,
                site_id,
                alarm_title,
                clear_time_str,
                is_clear=True
            )
            clear_alarm_count += 1

    return processed_count, valid_alarms, normal_alarm_count, clear_alarm_count


def _trim_trailing_clear_alarms(valid_alarms):
    """删除尾部仅由清除告警组成的区段。"""
    last_non_clear_index = -1
    for idx, item in enumerate(valid_alarms):
        if not _is_clear_alarm(item.get("alarm", {})):
            last_non_clear_index = idx

    if last_non_clear_index < 0:
        return []

    return valid_alarms[: last_non_clear_index + 1]


def _process_alarm(engine, item, collect_matches=False, register_trigger=True):
    alarm = item["alarm"]
    return engine.process_event(
        node=item["site_id"],
        alarm_source=item.get("alarm_source", ""),
        alarm_type=item["alarm_title"],
        ts=item["ts"],
        event_id=alarm["告警编码ID"],
        is_clear=_is_clear_alarm(alarm),
        collect_matches=collect_matches,
        register_trigger=register_trigger
    )


def _build_simulated_now_ts_getter(valid_alarms, speedup):
    """构造与告警 ts 回放节奏一致的模拟时钟。"""
    if not valid_alarms:
        return time.time

    simulated_start_ts = valid_alarms[0]["ts"]
    real_start_monotonic = time.monotonic()

    def get_now_ts():
        elapsed_real_sec = time.monotonic() - real_start_monotonic
        return simulated_start_ts + elapsed_real_sec * speedup

    return get_now_ts


def _build_group_link_info(ne_id, group_ne_ids, ne_graph_data):
    ne_graph_entry = ne_graph_data.get(ne_id, {})
    raw_links = ne_graph_entry.get("link", {}) if isinstance(ne_graph_entry, dict) else {}
    link_info = {}

    for neighbor_id, link_meta in raw_links.items():
        if neighbor_id not in group_ne_ids:
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


def _resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data):
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


def _build_group_output(match, ne_graph_data, site_graph_data):
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

        ne_graph_entry = ne_graph_data.get(ne_id, {})
        if site_id:
            group_site_ids.add(site_id)
        site_graph_entry = site_graph_data.get(site_id, {}) if site_id else {}

        ne_alarms[ne_id].append({
            "alarm_id": symptom.get("eid", ""),
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

    group_ne_ids = sorted({
        ne_id
        for ne_id, ne_graph_entry in ne_graph_data.items()
        if ne_graph_entry.get("site_id", "") in group_site_ids
    } | set(ne_alarms.keys()))

    for ne_id in group_ne_ids:
        ne_graph_entry = ne_graph_data.get(ne_id, {})
        alarms = sorted(
            ne_alarms.get(ne_id, []),
            key=lambda alarm: (alarm.get("alarm_time", ""), alarm.get("alarm_id", ""))
        )
        site_context = _resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data)
        site_id = site_context["site_id"]
        if site_id:
            group_site_ids.add(site_id)

        ne_info[ne_id] = {
            "alarm": alarms,
            "link": _build_group_link_info(ne_id, set(group_ne_ids), ne_graph_data),
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


def _build_alarm_metadata_index(valid_alarms):
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


def _enrich_match_symptoms(match, alarm_metadata_index):
    enriched_symptoms = []
    for symptom in match.get("symptoms", []):
        enriched_symptom = dict(symptom)
        event_id = enriched_symptom.get("eid", "")
        if event_id:
            metadata = alarm_metadata_index.get(event_id, {})
            for field_name in ("工单号", "故障组ID", "告警清除时间"):
                if metadata.get(field_name) and not enriched_symptom.get(field_name):
                    enriched_symptom[field_name] = metadata[field_name]
        enriched_symptoms.append(enriched_symptom)
    return enriched_symptoms


def _build_jsonl_match_output(match, ne_graph_data, site_graph_data, alarm_metadata_index):
    enriched_match = dict(match)
    enriched_match["symptoms"] = _enrich_match_symptoms(match, alarm_metadata_index)

    group_output = _build_group_output(enriched_match, ne_graph_data, site_graph_data)
    timestamps = [symptom["ts"] for symptom in enriched_match.get("symptoms", []) if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None

    enriched_match["group_anchor_ts"] = group_anchor_ts
    enriched_match["group_anchor_time"] = (
        datetime.fromtimestamp(group_anchor_ts).strftime("%Y-%m-%d %H:%M:%S")
        if group_anchor_ts is not None else ""
    )
    enriched_match.update(group_output)
    return enriched_match


def _match_debug_trigger(match, debug_targets, rules_config):
    merged_rules = match.get("merged_rules", [match.get("rule")])
    trigger_roles = {
        rules_config[rule_name]["trigger_role"]
        for rule_name in merged_rules
        if rule_name in rules_config and rules_config[rule_name].get("trigger_role")
    }
    for symptom in match.get("symptoms", []):
        target = (symptom.get("node"), symptom.get("alarm"))
        if target not in debug_targets:
            continue
        if symptom.get("matched_role") in trigger_roles:
            return True
    return False


def _build_match_time_range(match):
    timestamps = sorted(
        symptom["ts"]
        for symptom in match.get("symptoms", [])
        if symptom.get("ts") is not None
    )
    if not timestamps:
        return "-"

    def format_ts(ts):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    return f"{format_ts(timestamps[0])} ~ {format_ts(timestamps[-1])}"


def _is_debug_trigger_item(item, debug_targets):
    return (
        (item.get("site_id"), item.get("alarm_title")) in debug_targets
        and not _is_clear_alarm(item.get("alarm", {}))
    )


def _parse_debug_targets(args):
    debug_targets = set()

    for raw_value in args.debug_trigger or []:
        if "::" not in raw_value:
            raise ValueError(f"无效的 --debug-trigger 参数: {raw_value}，应为 站点ID::告警名")
        site_id, alarm_name = raw_value.split("::", 1)
        site_id = site_id.strip()
        alarm_name = alarm_name.strip()
        if not site_id or not alarm_name:
            raise ValueError(f"无效的 --debug-trigger 参数: {raw_value}，站点ID和告警名都不能为空")
        debug_targets.add((site_id, alarm_name))

    return debug_targets


def _format_debug_site_events(engine, site_id, limit=50):
    with engine._lock:
        site_events = list(engine.event_cache.get(site_id, []))

    events = site_events[-limit:]
    if not events:
        return json.dumps({"total": 0, "events": []}, ensure_ascii=False)

    formatted = []
    for ts, eid, alarm_type, alarm_source, consumed_trigger_rules in events:
        formatted.append(
            {
                "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "eid": eid,
                "alarm": alarm_type,
                "source": alarm_source,
                "consumed_trigger_rules": sorted(consumed_trigger_rules),
            }
        )
    return json.dumps({"total": len(site_events), "events": formatted}, ensure_ascii=False)


def _format_debug_trigger_index(engine, site_id):
    entries = _snapshot_debug_trigger_index(engine, site_id)
    return json.dumps(entries, ensure_ascii=False)


def _snapshot_debug_trigger_index(engine, site_id):
    with engine._lock:
        trigger_specs = tuple(engine.trigger_specs_by_node.get(site_id, ()))
        trigger_index_snapshot = {
            rule_name: list(engine.trigger_event_index.get((site_id, rule_name), ()))
            for rule_name, _ in trigger_specs
        }

    entries = {}
    for rule_name, _ in trigger_specs:
        trigger_events = trigger_index_snapshot.get(rule_name, [])
        if not trigger_events:
            continue
        entries[rule_name] = [
            {
                "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "eid": eid,
                "seq": seq,
                "alarm": alarm_type,
            }
            for ts, eid, seq, alarm_type in trigger_events
        ]
    return entries


def _print_debug_trigger_changes(engine, debug_sites, previous_snapshots, header, harvest_snapshot=None):
    changed_items = []
    for site_id in sorted(debug_sites):
        current_snapshot = _snapshot_debug_trigger_index(engine, site_id)
        previous_snapshot = previous_snapshots.get(site_id, {})
        if current_snapshot == previous_snapshot:
            continue
        changed_items.append((site_id, previous_snapshot, current_snapshot))

    if not changed_items:
        return False

    if harvest_snapshot is not None:
        _print_debug_harvest_actions(harvest_snapshot, engine)

    for site_id, previous_snapshot, current_snapshot in changed_items:
        print(f"{header}[{site_id}]")
        print(f"   ↳ 变化前: {json.dumps(previous_snapshot, ensure_ascii=False)}")
        print(f"   ↳ 变化后: {json.dumps(current_snapshot, ensure_ascii=False)}")
        previous_snapshots[site_id] = current_snapshot
    return True


def _get_debug_match_alarm_keys(match):
    return {
        symptom.get("eid")
        for symptom in match.get("symptoms", [])
        if symptom.get("eid") not in (None, "")
    }


def _match_present_in_stage(raw_match, stage_matches):
    raw_alarm_keys = _get_debug_match_alarm_keys(raw_match)
    if not raw_alarm_keys:
        return False
    for stage_match in stage_matches:
        stage_alarm_keys = _get_debug_match_alarm_keys(stage_match)
        if raw_alarm_keys.issubset(stage_alarm_keys):
            return True
    return False


def _print_debug_harvest_actions(snapshot, engine):
    mature_items = list(snapshot.get("mature_items", []))
    finalize_profiles = list(snapshot.get("finalize_profiles", []))
    profiles = {
        (profile.get("node"), profile.get("rule"), profile.get("trigger_ts"), profile.get("trigger_seq")): profile
        for profile in snapshot.get("pending_eval_profiles", [])
    }
    if not mature_items:
        return

    print("🔍 本次收割动作")
    print(
        "   ↳ 收割概览: "
        f"mature={len(mature_items)}, "
        f"raw={len(snapshot.get('raw_matches', []))}, "
        f"batch_merged={len(snapshot.get('batch_merged_matches', []))}, "
        f"expanded={len(snapshot.get('expanded_matches', []))}, "
        f"finalized={len(snapshot.get('finalized_matches', []))}"
    )
    for idx, item in enumerate(mature_items, start=1):
        site_id = item.get("node", "")
        rule_name = item.get("rule", "")
        trigger_ts = item.get("trigger_ts")
        trigger_seq = item.get("trigger_seq")
        trigger_time = (
            datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S")
            if trigger_ts is not None else "-"
        )
        trigger_detail = _find_trigger_event_detail(engine, site_id, rule_name, (trigger_ts, trigger_seq))
        profile = profiles.get((site_id, rule_name, trigger_ts, trigger_seq), {})
        raw_match_count = profile.get("raw_match_count", 0)
        print(
            f"   ↳ [{idx}] mature_trigger: "
            f"site={site_id}, rule={rule_name}, alarm={trigger_detail.get('alarm', '')}, "
            f"eid={trigger_detail.get('eid', '')}, trigger_time={trigger_time}, "
            f"trigger_seq={trigger_seq}, raw_match_count={raw_match_count}"
        )
        if raw_match_count == 0:
            debug_trace = profile.get("debug_trace") or {}
            final_reason = debug_trace.get("final_reason", "")
            if final_reason:
                print(f"      未产出原始候选组原因: {final_reason}")
            else:
                print(
                    "      未产出原始候选组，但未拿到 evaluate_rule 的 final_reason；"
                    " 这更像是 debug 记录异常，而不是规则正常行为"
                )
        else:
            print("      该 mature trigger 已产出原始候选组")
            raw_matches = profile.get("raw_matches", [])
            for raw_idx, raw_match in enumerate(raw_matches[:3], start=1):
                present_in_batch = _match_present_in_stage(raw_match, snapshot.get("batch_merged_matches", []))
                present_in_expanded = _match_present_in_stage(raw_match, snapshot.get("expanded_matches", []))
                present_in_finalized = _match_present_in_stage(raw_match, snapshot.get("finalized_matches", []))
                print(
                    f"      raw_match[{raw_idx}] 去向: "
                    f"batch_merged={present_in_batch}, "
                    f"expanded={present_in_expanded}, "
                    f"finalized={present_in_finalized}"
                )
                _print_debug_match_details(raw_match)

    if finalize_profiles:
        print("   ↳ finalize 结果")
        for idx, profile in enumerate(finalize_profiles, start=1):
            print(
                f"      [{idx}] action={profile.get('action', '')}, "
                f"rule={profile.get('rule', '')}, "
                f"uuid={profile.get('uuid', '')}, "
                f"merged_group_count={profile.get('merged_group_count', 0)}, "
                f"related_group_uuids={profile.get('related_group_uuids', [])}"
            )
            reason = profile.get("reason", "")
            if reason == "suppressed_by_fully_containing_history":
                print("         - 最终未输出：当前候选组的 eid 已被历史故障组完全覆盖，只延长历史组停留时间")
            elif reason == "merged_with_related_history":
                print("         - 最终输出：与历史相关组做了合并后重新输出")
            elif reason == "no_related_history":
                print("         - 最终输出：没有命中任何历史相关组")
            elif reason == "no_alarm_keys":
                print("         - 最终阶段异常：候选组没有可用 eid，无法做历史合并判断")

    if snapshot.get("finalized_matches"):
        print(
            "   ↳ 提示: 本次收割进入了 finalize 阶段，后续会执行 "
            "_prune_consumed_alarm_history -> _prune_node_alarm_history_before -> "
            "_refresh_pending_triggers_for_node，"
            "关注站点 trigger 的变化可能来自这里，而不一定是某个 mature trigger 未产出原始候选组"
        )


def _find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor):
    trigger_ts, trigger_seq = trigger_anchor
    with engine._lock:
        trigger_events = list(engine.trigger_event_index.get((site_id, rule_name), ()))

    for event_ts, event_id, event_seq, alarm_type in trigger_events:
        if event_ts == trigger_ts and event_seq == trigger_seq:
            return {
                "alarm": alarm_type,
                "eid": event_id,
                "seq": event_seq,
            }
    return {
        "alarm": "",
        "eid": "",
        "seq": trigger_seq,
    }


def _format_debug_pending(engine, site_id):
    with engine._lock:
        pending_items = [
            ((node, rule_name), trigger_anchor)
            for (node, rule_name), trigger_anchor in engine.pending_triggers.items()
            if node == site_id
        ]

    pending = {}
    for (_node, rule_name), trigger_anchor in pending_items:
        trigger_ts, trigger_seq = trigger_anchor
        trigger_detail = _find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor)
        pending[rule_name] = {
            "site": site_id,
            "alarm": trigger_detail["alarm"],
            "eid": trigger_detail["eid"],
            "trigger_time": datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_seq": trigger_detail["seq"],
            "ready_time": datetime.fromtimestamp(trigger_ts + engine.aggregation_wait_sec).strftime("%Y-%m-%d %H:%M:%S"),
        }
    return json.dumps(pending, ensure_ascii=False)


def _print_debug_pending_items(engine, site_id, header):
    with engine._lock:
        pending_items = [
            ((node, rule_name), trigger_anchor)
            for (node, rule_name), trigger_anchor in engine.pending_triggers.items()
            if node == site_id
        ]

    if not pending_items:
        print(f"{header}: 0 个")
        return

    print(f"{header}: {len(pending_items)} 个")
    for idx, ((_node, rule_name), trigger_anchor) in enumerate(pending_items, start=1):
        trigger_ts, _trigger_seq = trigger_anchor
        trigger_detail = _find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor)
        print(
            f"      [{idx}] pending: "
            f"site={site_id}, "
            f"rule={rule_name}, "
            f"alarm={trigger_detail['alarm']}, "
            f"eid={trigger_detail['eid']}, "
            f"trigger_time={datetime.fromtimestamp(trigger_ts).strftime('%Y-%m-%d %H:%M:%S')}, "
            f"trigger_seq={trigger_detail['seq']}, "
            f"ready_time={datetime.fromtimestamp(trigger_ts + engine.aggregation_wait_sec).strftime('%Y-%m-%d %H:%M:%S')}"
        )


def _print_debug_match_details(match):
    print(
        "   ↳ 命中故障组: "
        f"uuid={match.get('uuid', '')}, "
        f"rules={'+'.join(match.get('merged_rules', [match.get('rule', '')]))}, "
        f"time_range={_build_match_time_range(match)}"
    )
    print(f"      inferred_roots={json.dumps(match.get('inferred_roots', {}), ensure_ascii=False)}")
    print(f"      role_mapping={json.dumps(match.get('role_mapping', {}), ensure_ascii=False)}")
    symptom_preview = [
        {
            "time": datetime.fromtimestamp(symptom["ts"]).strftime("%Y-%m-%d %H:%M:%S") if symptom.get("ts") is not None else "-",
            "node": symptom.get("node", ""),
            "alarm": symptom.get("alarm", ""),
            "matched_role": symptom.get("matched_role", ""),
            "eid": symptom.get("eid", ""),
        }
        for symptom in sorted(
            match.get("symptoms", []),
            key=lambda symptom: (symptom.get("ts", float("inf")), symptom.get("eid", ""))
        )
    ]
    print(f"      symptoms={json.dumps(symptom_preview, ensure_ascii=False)}")


def _print_debug_post_batch_state(engine, debug_sites):
    for site_id in sorted(debug_sites):
        print(f"   ↳ 本批完成后站点状态[{site_id}]")
        print(f"      event_cache={_format_debug_site_events(engine, site_id)}")
        print(f"      trigger_index={_format_debug_trigger_index(engine, site_id)}")
        _print_debug_pending_items(engine, site_id, "      pending")


def _print_debug_pending_eval_profiles(snapshot, debug_sites):
    profiles = [
        profile
        for profile in snapshot.get("pending_eval_profiles", [])
        if profile.get("node") in debug_sites
    ]
    if not profiles:
        return

    def print_debug_trace(trace):
        if not trace:
            return
        trigger_validation = trace.get("trigger_validation") or {}
        if trigger_validation:
            print(
                "      trigger校验: "
                f"{'通过' if trigger_validation.get('valid') else '失败'}; "
                f"{trigger_validation.get('reason', '')}"
            )
        for edge_trace in trace.get("edges", []):
            print(
                f"      边 {edge_trace.get('from_role')} -> {edge_trace.get('to_role')}: "
                f"instances_in={edge_trace.get('instances_in', 0)}, "
                f"instances_out={edge_trace.get('instances_out', 0)}"
            )
            for failure in edge_trace.get("failures", [])[:5]:
                print(f"         - {failure}")
        if trace.get("final_reason"):
            print(f"      最终失败原因: {trace.get('final_reason')}")

    for idx, profile in enumerate(profiles, start=1):
        trigger_ts = profile.get("trigger_ts")
        trigger_time = (
            datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S")
            if trigger_ts is not None else "-"
        )
        print(
            f"   ↳ [{idx}] pending 弹出后: "
            f"site={profile.get('node', '')}, "
            f"rule={profile.get('rule', '')}, "
            f"trigger_time={trigger_time}, "
            f"trigger_seq={profile.get('trigger_seq', '')}"
        )
        print(f"      evaluate_rule 原始候选组数: {profile.get('raw_match_count', 0)}")
        raw_matches = profile.get("raw_matches", [])
        if not raw_matches:
            print("      ↳ 未产出原始候选组")
            print_debug_trace(profile.get("debug_trace"))
            continue
        for match in raw_matches:
            _print_debug_match_details(match)


def _print_debug_collection_snapshot(snapshot, debug_targets, rules_config, engine):
    debug_sites = {site_id for site_id, _alarm_name in debug_targets}
    raw_debug_matches = [
        match for match in snapshot.get("raw_matches", [])
        if _match_debug_trigger(match, debug_targets, rules_config)
    ]
    batch_debug_matches = [
        match for match in snapshot.get("batch_merged_matches", [])
        if _match_debug_trigger(match, debug_targets, rules_config)
    ]
    finalized_debug_matches = [
        match for match in snapshot.get("finalized_matches", [])
        if _match_debug_trigger(match, debug_targets, rules_config)
    ]
    expanded_debug_matches = [
        match for match in snapshot.get("expanded_matches", [])
        if _match_debug_trigger(match, debug_targets, rules_config)
    ]
    mature_triggers = [
        item for item in snapshot.get("mature_items", [])
        if item.get("node") in debug_sites
    ]

    if not mature_triggers and not raw_debug_matches and not batch_debug_matches and not finalized_debug_matches and not expanded_debug_matches:
        return

    watermark = snapshot.get("watermark")
    watermark_str = (
        datetime.fromtimestamp(watermark).strftime("%Y-%m-%d %H:%M:%S")
        if watermark is not None else "-"
    )
    effective_harvest_ts = snapshot.get("effective_harvest_ts")
    effective_harvest_str = (
        datetime.fromtimestamp(effective_harvest_ts).strftime("%Y-%m-%d %H:%M:%S")
        if effective_harvest_ts is not None else "-"
    )
    print(
        f"🔎 收割阶段快照: watermark={watermark_str}, "
        f"effective_harvest_ts={effective_harvest_str}, "
        f"force={snapshot.get('force', False)}"
    )
    if mature_triggers:
        mature_preview = [
            {
                "node": item.get("node", ""),
                "rule": item.get("rule", ""),
                "trigger_time": datetime.fromtimestamp(item["trigger_ts"]).strftime("%Y-%m-%d %H:%M:%S")
                if item.get("trigger_ts") is not None else "-",
                "trigger_seq": item.get("trigger_seq", ""),
            }
            for item in mature_triggers
        ]
        print(f"   ↳ 本轮成熟 trigger: {json.dumps(mature_preview, ensure_ascii=False)}")
        _print_debug_pending_eval_profiles(snapshot, debug_sites)

    stage_mapping = (
        ("原始候选组", raw_debug_matches),
        ("当前批次合并后", batch_debug_matches),
        ("pending 扩充后", expanded_debug_matches),
        ("历史组合并后", finalized_debug_matches),
    )
    for stage_name, stage_matches in stage_mapping:
        print(f"   ↳ {stage_name}: {len(stage_matches)} 个相关故障组")
        for match in stage_matches:
            _print_debug_match_details(match)

    _print_debug_post_batch_state(engine, debug_sites)


def _print_debug_event_removal(payload, debug_sites):
    node = payload.get("node")
    if node not in debug_sites:
        return

    removed_time = payload.get("ts")
    removed_time_str = (
        datetime.fromtimestamp(removed_time).strftime("%Y-%m-%d %H:%M:%S")
        if removed_time is not None else "-"
    )
    if payload.get("reason") == "ttl":
        current_ts = payload.get("current_ts")
        current_time_str = (
            datetime.fromtimestamp(current_ts).strftime("%Y-%m-%d %H:%M:%S")
            if current_ts is not None else "-"
        )
        print(
            f"🗑️ 事件移出缓存(TTL): site={node}, alarm={payload.get('alarm_type', '')}, "
            f"time={removed_time_str}, eid={payload.get('event_id', '')}, current_time={current_time_str}"
        )
    elif payload.get("reason") == "clear":
        print(
            f"🗑️ 事件移出缓存(清除): site={node}, alarm={payload.get('alarm_type', '')}, "
            f"time={removed_time_str}, eid={payload.get('event_id', '')}, "
            f"cleared_event_id={payload.get('cleared_event_id', '')}"
        )


def _run_live_mode(engine, valid_alarms, speedup, real_harvest_interval_sec, on_matches, process_progress):
    """按 ts 差值模拟实时告警流，并由后台定时线程异步收割成熟故障组。"""
    print(
        f"⏱️ 运行模式: live, speedup={speedup:g}x, "
        f"模拟收割周期={real_harvest_interval_sec * speedup:g}s, "
        f"真实收割周期={real_harvest_interval_sec:.3f}s"
    )
    now_ts_getter = _build_simulated_now_ts_getter(valid_alarms, speedup)
    engine.start_periodic_harvest(
        interval_sec=real_harvest_interval_sec,
        on_matches=on_matches,
        now_ts_getter=now_ts_getter
    )

    try:
        for item in _stream_alarms_by_ts(valid_alarms, speedup=speedup):
            _process_alarm(engine, item, collect_matches=False)
            process_progress.update()
    finally:
        process_progress.close()
        engine.stop_periodic_harvest()


def _run_offline_mode(engine, valid_alarms, on_matches, process_progress):
    """按时间排序顺序处理告警，并在每条告警后立即同步收割一次成熟故障组。"""
    print("⏱️ 运行模式: offline, 每条告警到来时直接触发检查")
    try:
        for item in valid_alarms:
            matches = _process_alarm(engine, item, collect_matches=True)
            if matches:
                on_matches(matches)
            process_progress.update()
    finally:
        process_progress.close()


def _run_debug_mode(
    engine,
    valid_alarms,
    on_matches,
    process_progress,
    debug_targets,
    rules_config,
    mode,
    speedup,
    real_harvest_interval_sec,
):
    """不改变原始 trigger 行为，只额外观察指定站点+告警相关的中间过程。"""
    debug_target_text = ", ".join(f"{site} / {alarm}" for site, alarm in sorted(debug_targets))
    print(
        f"🔎 Debug 模式({mode}): 观察 {debug_target_text}，"
        "所有 trigger 仍按原始逻辑正常运行"
    )
    debug_sites = {site_id for site_id, _alarm_name in debug_targets}
    last_trigger_snapshots = {
        site_id: _snapshot_debug_trigger_index(engine, site_id)
        for site_id in debug_sites
    }

    def on_debug_snapshot(snapshot):
        _print_debug_collection_snapshot(snapshot, debug_targets, rules_config, engine)
        _print_debug_trigger_changes(
            engine,
            debug_sites,
            last_trigger_snapshots,
            "🔁 收割后关注站点 trigger 变化",
            harvest_snapshot=snapshot,
        )

    engine.debug_observer = on_debug_snapshot
    engine.debug_event_logger = lambda payload: _print_debug_event_removal(payload, debug_sites)

    def on_debug_matches(matches, source="收割"):
        debug_matches = [
            match for match in matches
            if _match_debug_trigger(match, debug_targets, rules_config)
        ]
        if debug_matches:
            print(f"🔎 {source}命中 {len(debug_matches)} 个故障组")
            for match in debug_matches:
                _print_debug_match_details(match)
        on_matches(matches)

    if mode == 'live':
        now_ts_getter = _build_simulated_now_ts_getter(valid_alarms, speedup)
        engine.start_periodic_harvest(
            interval_sec=real_harvest_interval_sec,
            on_matches=lambda matches: on_debug_matches(matches, "定时收割"),
            now_ts_getter=now_ts_getter
        )
        try:
            for item in _stream_alarms_by_ts(valid_alarms, speedup=speedup):
                is_debug_trigger = _is_debug_trigger_item(item, debug_targets)
                is_debug_site_power_alarm = (
                    item.get("site_id") in debug_sites
                    and item.get("alarm_title") in POWER_ALARMS
                    and not _is_clear_alarm(item.get("alarm", {}))
                )
                is_debug_site_clear = (
                    item.get("site_id") in debug_sites
                    and _is_clear_alarm(item.get("alarm", {}))
                )
                _process_alarm(
                    engine,
                    item,
                    collect_matches=False,
                    register_trigger=True
                )
                _print_debug_trigger_changes(
                    engine,
                    debug_sites,
                    last_trigger_snapshots,
                    "🔁 告警处理后关注站点 trigger 变化",
                )
                if is_debug_trigger:
                    debug_site = item.get("site_id", "")
                    debug_alarm = item.get("alarm_title", "")
                    trigger_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"🔎 Trigger 输入: site={debug_site}, alarm={debug_alarm}, "
                        f"time={trigger_time}, eid={item['alarm'].get('告警编码ID', '')}"
                    )
                    print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
                    print(f"   ↳ 当前 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
                    _print_debug_pending_items(engine, debug_site, "   ↳ 当前 pending")
                elif is_debug_site_power_alarm:
                    debug_site = item.get("site_id", "")
                    power_alarm = item.get("alarm_title", "")
                    power_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"⚡ 电告警进入缓存: site={debug_site}, alarm={power_alarm}, "
                        f"time={power_time}, eid={item['alarm'].get('告警编码ID', '')}"
                    )
                    print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
                elif is_debug_site_clear:
                    debug_site = item.get("site_id", "")
                    clear_alarm = item.get("alarm_title", "")
                    clear_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"🧹 清除告警输入: site={debug_site}, alarm={clear_alarm}, "
                        f"time={clear_time}, eid={item['alarm'].get('告警编码ID', '')}"
                    )
                    print(f"   ↳ 清除后站点最近事件: {_format_debug_site_events(engine, debug_site)}")
                    print(f"   ↳ 清除后 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
                    _print_debug_pending_items(engine, debug_site, "   ↳ 清除后 pending")
                process_progress.update()
        finally:
            process_progress.close()
            engine.stop_periodic_harvest()
            engine.debug_event_logger = None
        return

    try:
        for item in valid_alarms:
            is_debug_trigger = _is_debug_trigger_item(item, debug_targets)
            is_debug_site_power_alarm = (
                item.get("site_id") in debug_sites
                and item.get("alarm_title") in POWER_ALARMS
                and not _is_clear_alarm(item.get("alarm", {}))
            )
            is_debug_site_clear = (
                item.get("site_id") in debug_sites
                and _is_clear_alarm(item.get("alarm", {}))
            )
            matches = _process_alarm(
                engine,
                item,
                collect_matches=True,
                register_trigger=True
            )
            _print_debug_trigger_changes(
                engine,
                debug_sites,
                last_trigger_snapshots,
                "🔁 告警处理后关注站点 trigger 变化",
            )
            if is_debug_trigger:
                debug_site = item.get("site_id", "")
                debug_alarm = item.get("alarm_title", "")
                trigger_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"🔎 Trigger 输入: site={debug_site}, alarm={debug_alarm}, "
                    f"time={trigger_time}, eid={item['alarm'].get('告警编码ID', '')}"
                )
                print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
                print(f"   ↳ 当前 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
                _print_debug_pending_items(engine, debug_site, "   ↳ 当前 pending")
                if not matches:
                    print("   ↳ 当前触发点暂未产出故障组")
            elif is_debug_site_power_alarm:
                debug_site = item.get("site_id", "")
                power_alarm = item.get("alarm_title", "")
                power_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"⚡ 电告警进入缓存: site={debug_site}, alarm={power_alarm}, "
                    f"time={power_time}, eid={item['alarm'].get('告警编码ID', '')}"
                )
                print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
            elif is_debug_site_clear:
                debug_site = item.get("site_id", "")
                clear_alarm = item.get("alarm_title", "")
                clear_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"🧹 清除告警输入: site={debug_site}, alarm={clear_alarm}, "
                    f"time={clear_time}, eid={item['alarm'].get('告警编码ID', '')}"
                )
                print(f"   ↳ 清除后站点最近事件: {_format_debug_site_events(engine, debug_site)}")
                print(f"   ↳ 清除后 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
                _print_debug_pending_items(engine, debug_site, "   ↳ 清除后 pending")
            if matches:
                on_debug_matches(matches, "同步检查")
            process_progress.update()
    finally:
        process_progress.close()
        engine.debug_event_logger = None


def main():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, help='output jsonl file')
    parser.add_argument('--topo', type=str, default=SITE_GRAPH_BY_NE_JSON, help=f'站点拓扑文件，默认: {resource_display("site_graph_by_ne.json")}')
    parser.add_argument('--site-domain', type=str, default=SITE_DEVICE_COUNTS_JSON, help=f'站点画像文件，默认: {resource_display("site_device_counts.json")}')
    parser.add_argument('--site-graph', type=str, default=SITE_GRAPH_JSON, help=f'site_graph.json 文件，默认: {resource_display("site_graph.json")}')
    parser.add_argument('--ne-graph', type=str, default=NE_GRAPH_JSON, help=f'ne_graph.json 文件，默认: {resource_display("ne_graph.json")}')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='live', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=300.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    parser.add_argument('--debug-trigger', action='append', help='debug: 指定一个 trigger，格式为 站点ID::告警名，可重复传多次')
    parser.add_argument('--verbose-groups', action='store_true', help='打印每个故障组的详细报告；默认静默，仅输出进度与汇总')
    parser.add_argument('--start_time', type=str, help='仅处理告警首次发生时间 >= 该时间的告警，格式如 2025-01-01 00:00:00')
    parser.add_argument('--end_time', type=str, help='仅处理告警首次发生时间 <= 该时间的告警，格式如 2025-01-31 23:59:59')
    parser.add_argument('--compute-ticket-recall', action='store_true', help='在主故障组输出完成后额外计算工单站点召回率')
    parser.add_argument('--ticket-sites', type=str, help='工单站点映射 JSON。不提供时，可退化为从 alarms 自身回推工单站点')
    parser.add_argument('--ticket-field', type=str, default='工单号', help='工单字段名，默认: 工单号')
    parser.add_argument('--ticket-recall-output', type=str, help='工单站点召回率输出文件。默认: <output>.ticket_recall.json')
    args = parser.parse_args()

    start_ts = None
    end_ts = None
    if args.start_time:
        start_ts = _parse_datetime_text(args.start_time, "start_time").timestamp()
    if args.end_time:
        end_ts = _parse_datetime_text(args.end_time, "end_time").timestamp()
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        parser.error("start_time 不能晚于 end_time")

    topo_downstream_map = json.load(open(args.topo, 'r', encoding='utf-8'))
    site_domain_map = json.load(open(args.site_domain, 'r', encoding='utf-8'))
    site_graph_data = json.load(open(args.site_graph, 'r', encoding='utf-8'))
    ne_graph_data = json.load(open(args.ne_graph, 'r', encoding='utf-8'))

    print("加载有效站点集合...")
    valid_sites = load_site_graph(args.topo)
    print(f"有效站点数: {len(valid_sites)}")

    print("构建 ne -> site 映射...")
    ne_to_site = build_ne_to_site_map(args.ne_graph)
    print(f"NE 数量: {len(ne_to_site)}")

    valid_alarm_titles = CRITICAL_ALARMS
    print(f"有效告警类型数: {len(valid_alarm_titles)}")
    if args.start_time or args.end_time:
        print(
            "告警首次发生时间过滤: "
            f"start_time={args.start_time or '-'}, "
            f"end_time={args.end_time or '-'}"
        )

    rules_config = {
        "transmission_rule": transmission_rule,
        "link_rule": link_rule,
        "power_rule": power_rule
    }

    print("⏳ 正在初始化时序图引擎与拓扑映射...")
    engine = TemporalGraphEngine(topo_downstream_map, rules_config, site_domain_map)
    print("✅ 引擎启动就绪，开始监听告警流...\n")

    alarm_file_path = args.alarms
    start_time = time.time()

    processed_count, valid_alarms, normal_alarm_count, clear_alarm_count = _load_valid_alarms(
        alarm_file_path,
        valid_alarm_titles,
        valid_sites,
        ne_to_site,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    print("⏳ 正在按时间排序有效告警...")
    sort_start_time = time.time()
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = _trim_trailing_clear_alarms(valid_alarms)
    alarm_metadata_index = _build_alarm_metadata_index(valid_alarms)
    sort_elapsed = time.time() - sort_start_time
    filtered_count = len(valid_alarms)
    print(f"有效告警数: {filtered_count}，排序耗时: {sort_elapsed:.4f} 秒")
    print(f"正常告警数: {normal_alarm_count}，清除告警数: {clear_alarm_count}")

    debug_targets = _parse_debug_targets(args)
    debug_enabled = bool(debug_targets)
    if debug_enabled:
        print("🔎 Debug 模式已开启:")
        for site_id, alarm_name in sorted(debug_targets):
            print(f"   - {site_id} / {alarm_name}")

    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)

    match_count = 0
    process_progress = None
    output_lock = threading.Lock()
    with open(args.output, 'w', encoding='utf-8'):
        pass

    def on_matches(matches):
        # 统一处理一批新产出的故障组：边输出报告，边按 jsonl 落盘。
        nonlocal match_count
        with output_lock:
            with open(args.output, 'a', encoding='utf-8') as fw:
                for match in matches:
                    if args.verbose_groups:
                        generate_incident_report(match)
                    enriched_match = _build_jsonl_match_output(
                        match,
                        ne_graph_data,
                        site_graph_data,
                        alarm_metadata_index
                    )
                    fw.write(json.dumps(enriched_match, ensure_ascii=False) + '\n')
            match_count += len(matches)
            if process_progress is not None:
                process_progress.set_extra_text(f"已汇聚故障组数: {match_count}")

    process_progress = ProgressBar(filtered_count, "处理有效告警")
    process_progress.set_extra_text(f"已汇聚故障组数: {match_count}", force=True)
    if debug_enabled:
        _run_debug_mode(
            engine,
            valid_alarms,
            on_matches,
            process_progress,
            debug_targets,
            rules_config,
            args.mode,
            speedup,
            real_harvest_interval_sec,
        )
    elif args.mode == 'live':
        _run_live_mode(
            engine,
            valid_alarms,
            speedup,
            real_harvest_interval_sec,
            on_matches,
            process_progress
        )
    else:
        _run_offline_mode(engine, valid_alarms, on_matches, process_progress)

    print("⏳ 数据流读取完毕，正在清空并计算延迟聚合队列...")
    final_matches = engine.flush_pending()
    if final_matches:
        if debug_enabled:
            debug_final_matches = [
                match for match in final_matches
                if _match_debug_trigger(match, debug_targets, rules_config)
            ]
            if debug_final_matches:
                print(f"🔎 Flush 阶段额外产出 {len(debug_final_matches)} 个故障组")
                for match in debug_final_matches:
                    _print_debug_match_details(match)
            on_matches(final_matches)
        else:
            on_matches(final_matches)

    if debug_enabled:
        engine.debug_observer = None

    elapsed = time.time() - start_time
    print(f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，生成 {match_count} 个故障组，耗时 {elapsed:.4f} 秒。")

    if args.compute_ticket_recall:
        ticket_recall_output = args.ticket_recall_output or f"{args.output}.ticket_recall.json"
        print("⏳ 正在基于当前故障组输出计算工单站点召回率...")
        try:
            recall_result = compute_group_output_ticket_recall(
                args.output,
                args.ticket_sites,
                ticket_field=args.ticket_field,
                alarms_input=args.alarms,
                ne_graph_file=args.ne_graph,
                output_file=ticket_recall_output,
            )
            print(
                f"✅ 工单站点召回率计算完成。工单数: {recall_result['ticket_count']}，"
                f"平均召回率: {recall_result['average_recall']:.6f}，"
                f"输出: {ticket_recall_output}"
            )
        except ValueError as exc:
            print(f"⚠️ 工单站点召回率计算跳过: {exc}")


if __name__ == "__main__":
    main()
