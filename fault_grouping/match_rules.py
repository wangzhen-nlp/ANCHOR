import json
import os
import time
import threading

from dataclasses import dataclass, field
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_types import POWER_ALARMS
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_DEVICE_COUNTS_JSON,
    SITE_GRAPH_JSON,
    SITE_GRAPH_BY_NE_JSON,
    resource_display,
)
from ticket_recall.evaluation.compute_group_output_ticket_recall import compute_group_output_ticket_recall
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.reports import generate_incident_report
from fault_grouping.match_rules_debug import (
    build_debug_run_context as _build_debug_run_context,
    format_debug_site_events as _format_debug_site_events,
    format_debug_trigger_index as _format_debug_trigger_index,
    match_debug_trigger as _match_debug_trigger,
    parse_debug_targets as _parse_debug_targets,
    print_debug_collection_snapshot as _print_debug_collection_snapshot,
    print_debug_event_removal as _print_debug_event_removal,
    print_debug_match_details as _print_debug_match_details,
    print_debug_pending_items as _print_debug_pending_items,
    print_debug_trigger_changes as _print_debug_trigger_changes,
)
from fault_grouping.match_rules_alarm_io import (
    is_clear_alarm as _is_clear_alarm,
)
from fault_grouping.match_rules_runtime import (
    AlarmLoadResult,
    LoadedStaticContext,
    build_rules_config as _build_rules_config,
    default_valid_alarm_titles as _default_valid_alarm_titles,
    initialize_engine as _initialize_engine,
    load_alarm_data as _load_alarm_data,
    load_static_context as _load_static_context,
    print_alarm_load_summary as _print_alarm_load_summary,
    print_run_configuration as _print_run_configuration,
    validate_main_args as _validate_main_args,
    build_batch_site_merge_helper as _build_batch_site_merge_helper,
)
from fault_grouping.temporal_graph_engine import TemporalGraphEngine


@dataclass
class MatchOutputSession:
    args: object
    engine: TemporalGraphEngine
    output_path: str
    ne_graph_data: dict
    site_graph_data: dict
    alarm_metadata_index: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
    match_count: int = 0
    process_progress: object = None
    output_lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_output_file(self):
        with open(self.output_path, 'w', encoding='utf-8'):
            pass

    def build_progress_extra_text(self):
        merge_stats = self.engine.get_batch_merge_stats_snapshot().get("total", {})
        primary_merge_count = (
            merge_stats.get('alarm_overlap_merge_group_count', 0)
            if self.args.use_alarm_period_cache
            else merge_stats.get('eid_merge_group_count', 0)
        )
        primary_merge_label = "告警时段合并组数" if self.args.use_alarm_period_cache else "eid合并组数"
        return (
            f"已汇聚故障组数: {self.match_count} | "
            f"{primary_merge_label}: {primary_merge_count} | "
            f"hop合并组数: {merge_stats.get('hop_merge_group_count', 0)} | "
            f"距离合并组数: {merge_stats.get('distance_merge_group_count', 0)}"
        )

    def refresh_progress_extra_text(self, force=False):
        if self.process_progress is None:
            return
        self.process_progress.set_extra_text(self.build_progress_extra_text(), force=force)

    def write_matches(self, matches):
        with self.output_lock:
            with open(self.output_path, 'a', encoding='utf-8') as fw:
                output_lines = []
                for match in matches:
                    if self.args.verbose_groups:
                        generate_incident_report(match)
                    enriched_match = _build_jsonl_match_output(
                        match,
                        self.ne_graph_data,
                        self.site_graph_data,
                        self.alarm_metadata_index,
                        site_to_ne_ids=self.site_to_ne_ids,
                        ne_link_info_cache=self.ne_link_info_cache,
                        compact_output=self.args.compact_output,
                        include_eid_list=self.args.use_alarm_period_cache,
                    )
                    output_lines.append(json.dumps(enriched_match, ensure_ascii=False) + '\n')
                fw.writelines(output_lines)
            self.match_count += len(matches)
            self.refresh_progress_extra_text()


@dataclass
class RuntimeExecutionPlan:
    static_context: LoadedStaticContext
    rules_config: dict
    engine: TemporalGraphEngine
    alarm_load_result: AlarmLoadResult
    alarm_metadata_index: dict
    debug_targets: set
    output_session: MatchOutputSession
    start_time: float


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


def _format_ne_link_info(ne_graph_entry, compact_output=False):
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


def _get_cached_ne_link_info(ne_id, ne_graph_data, ne_link_info_cache, compact_output=False):
    if ne_link_info_cache is None:
        return _format_ne_link_info(ne_graph_data.get(ne_id, {}), compact_output=compact_output)
    cache_key = (ne_id, compact_output)
    if cache_key not in ne_link_info_cache:
        ne_link_info_cache[cache_key] = _format_ne_link_info(
            ne_graph_data.get(ne_id, {}),
            compact_output=compact_output,
        )
    return ne_link_info_cache[cache_key]


def _build_group_link_info(ne_id, group_ne_ids, ne_graph_data, ne_link_info_cache=None, compact_output=False):
    formatted_links = _get_cached_ne_link_info(
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


def _build_group_output(
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
        site_context = _resolve_ne_site_context(ne_id, alarms, ne_graph_data, site_graph_data)
        site_id = site_context["site_id"]
        if site_id:
            group_site_ids.add(site_id)

        node_info = {
            "link": _build_group_link_info(
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


def _enrich_match_symptoms(match, alarm_metadata_index, include_eid_list=False):
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


def _build_jsonl_match_output(
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
    enriched_match["symptoms"] = _enrich_match_symptoms(
        match,
        alarm_metadata_index,
        include_eid_list=include_eid_list,
    )

    group_output = _build_group_output(
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


def _is_debug_trigger_item(item, debug_targets):
    return (
        (item.get("site_id"), item.get("alarm_title")) in debug_targets
        and not _is_clear_alarm(item.get("alarm", {}))
    )


def _is_debug_site_power_alarm(item, debug_sites):
    return (
        item.get("site_id") in debug_sites
        and item.get("alarm_title") in POWER_ALARMS
        and not _is_clear_alarm(item.get("alarm", {}))
    )


def _is_debug_site_clear(item, debug_sites):
    return (
        item.get("site_id") in debug_sites
        and _is_clear_alarm(item.get("alarm", {}))
    )


def _get_debug_item_kind(item, debug_context):
    if _is_debug_trigger_item(item, debug_context.debug_targets):
        return "trigger"
    if _is_debug_site_power_alarm(item, debug_context.debug_sites):
        return "power"
    if _is_debug_site_clear(item, debug_context.debug_sites):
        return "clear"
    return None


def _print_debug_item_state(engine, item, item_kind, matches=None):
    debug_site = item.get("site_id", "")
    debug_alarm = item.get("alarm_title", "")
    debug_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
    debug_eid = item['alarm'].get('告警编码ID', '')

    if item_kind == "trigger":
        print(
            f"🔎 Trigger 输入: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
        print(f"   ↳ 当前 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
        _print_debug_pending_items(engine, debug_site, "   ↳ 当前 pending")
        if matches is not None and not matches:
            print("   ↳ 当前触发点暂未产出故障组")
        return

    if item_kind == "power":
        print(
            f"⚡ 电告警进入缓存: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 当前站点最近事件: {_format_debug_site_events(engine, debug_site)}")
        return

    if item_kind == "clear":
        print(
            f"🧹 清除告警输入: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 清除后站点最近事件: {_format_debug_site_events(engine, debug_site)}")
        print(f"   ↳ 清除后 trigger_index: {_format_debug_trigger_index(engine, debug_site)}")
        _print_debug_pending_items(engine, debug_site, "   ↳ 清除后 pending")


def _refresh_process_progress(process_progress):
    if hasattr(process_progress, "_refresh_extra_text"):
        process_progress._refresh_extra_text()
    process_progress.update()


def _build_debug_match_callback(on_matches, debug_targets, rules_config):
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

    return on_debug_matches


def _run_debug_live_loop(
    engine,
    valid_alarms,
    speedup,
    process_progress,
    debug_context,
):
    for item in _stream_alarms_by_ts(valid_alarms, speedup=speedup):
        item_kind = _get_debug_item_kind(item, debug_context)
        _process_alarm(
            engine,
            item,
            collect_matches=False,
            register_trigger=True
        )
        _print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 告警处理后关注站点 trigger 变化",
        )
        if item_kind is not None:
            _print_debug_item_state(engine, item, item_kind)
        _refresh_process_progress(process_progress)


def _run_debug_offline_loop(
    engine,
    valid_alarms,
    process_progress,
    debug_context,
    on_debug_matches,
):
    for item in valid_alarms:
        item_kind = _get_debug_item_kind(item, debug_context)
        matches = _process_alarm(
            engine,
            item,
            collect_matches=True,
            register_trigger=True
        )
        _print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 告警处理后关注站点 trigger 变化",
        )
        if item_kind is not None:
            _print_debug_item_state(engine, item, item_kind, matches=matches)
        if matches:
            on_debug_matches(matches, "同步检查")
        _refresh_process_progress(process_progress)


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
            _refresh_process_progress(process_progress)
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
            _refresh_process_progress(process_progress)
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
    debug_context = _build_debug_run_context(engine, debug_targets)

    def on_debug_snapshot(snapshot):
        _print_debug_collection_snapshot(snapshot, debug_targets, rules_config, engine)
        _print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 收割后关注站点 trigger 变化",
            harvest_snapshot=snapshot,
        )

    engine.debug_observer = on_debug_snapshot
    engine.debug_event_logger = lambda payload: _print_debug_event_removal(payload, debug_context.debug_sites)
    on_debug_matches = _build_debug_match_callback(on_matches, debug_targets, rules_config)

    if mode == 'live':
        now_ts_getter = _build_simulated_now_ts_getter(valid_alarms, speedup)
        engine.start_periodic_harvest(
            interval_sec=real_harvest_interval_sec,
            on_matches=lambda matches: on_debug_matches(matches, "定时收割"),
            now_ts_getter=now_ts_getter
        )
        try:
            _run_debug_live_loop(
                engine,
                valid_alarms,
                speedup,
                process_progress,
                debug_context,
            )
        finally:
            process_progress.close()
            engine.stop_periodic_harvest()
            engine.debug_event_logger = None
        return

    try:
        _run_debug_offline_loop(
            engine,
            valid_alarms,
            process_progress,
            debug_context,
            on_debug_matches,
        )
    finally:
        process_progress.close()
        engine.debug_event_logger = None


def _build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, help='output jsonl file')
    parser.add_argument('--topo', type=str, default=SITE_GRAPH_BY_NE_JSON, help=f'站点拓扑文件，默认: {resource_display("site_graph_by_ne.json")}；若传空值则退回为基于 ne_graph.json 原始连边自动构建')
    parser.add_argument('--site-chains', type=str, default='', help=f'可选 generate_site_chains.py 输出文件；提供后无 path 约束的上下游遍历优先使用预计算 hop，推荐: {resource_display("site_chains.json")}')
    parser.add_argument('--site-domain', type=str, default=SITE_DEVICE_COUNTS_JSON, help=f'站点画像文件，默认: {resource_display("site_device_counts.json")}')
    parser.add_argument('--site-graph', type=str, default=SITE_GRAPH_JSON, help=f'site_graph.json 文件，默认: {resource_display("site_graph.json")}')
    parser.add_argument('--ne-graph', type=str, default=NE_GRAPH_JSON, help=f'ne_graph.json 文件，默认: {resource_display("ne_graph.json")}')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='offline', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=300.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--aggregation-wait-sec', type=float, default=420.0, help='trigger 成熟前的聚合等待时间，单位秒，默认 420')
    parser.add_argument('--clear-delay-sec', type=float, default=420.0, help='清除告警最小延迟时间，清除生效时间=max(clear_delay_sec, 清除时间-发生时间)+发生时间')
    parser.add_argument('--batch-merge-site-hops', type=int, default=0, help='批内候选组额外按站点邻接合并的 hop 数；0 表示关闭，2 表示两跳内可合并')
    parser.add_argument('--batch-merge-density-knn', type=int, default=0, help='批内候选组额外按站点局部密度自适应合并时使用的近邻数；0 表示关闭')
    parser.add_argument('--batch-merge-density-scale', type=float, default=1.0, help='局部密度半径放大倍数，实际阈值=scale * 第k近邻距离')
    parser.add_argument('--batch-merge-density-min-meters', type=float, default=0.0, help='局部密度自适应半径下限，单位米；0 表示不设下限')
    parser.add_argument('--batch-merge-density-max-meters', type=float, default=0.0, help='局部密度自适应半径上限，单位米；0 表示不设上限')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    parser.add_argument('--debug-trigger', action='append', help='debug: 指定一个 trigger，格式为 站点ID::告警名，可重复传多次')
    parser.add_argument('--verbose-groups', action='store_true', help='打印每个故障组的详细报告；默认静默，仅输出进度与汇总')
    parser.add_argument('--start_time', type=str, help='仅处理告警首次发生时间 >= 该时间的告警，格式如 2025-01-01 00:00:00')
    parser.add_argument('--end_time', type=str, help='仅处理告警首次发生时间 <= 该时间的告警，格式如 2025-01-31 23:59:59')
    parser.add_argument('--compute-ticket-recall', action='store_true', help='在主故障组输出完成后额外计算工单站点召回率')
    parser.add_argument('--ticket-sites', type=str, help='工单站点映射 JSON。不提供时，可退化为从 alarms 自身回推工单站点')
    parser.add_argument('--ticket-field', type=str, default='工单号', help='工单字段名，默认: 工单号')
    parser.add_argument('--ticket-recall-output', type=str, help='工单站点召回率输出文件。默认: <output>.ticket_recall.json')
    parser.add_argument('--rule', action='append', default=[], help='仅启用指定规则；可重复传入，也支持逗号分隔，如 --rule transmission_rule --rule link_rule 或 --rule transmission_rule,link_rule')
    parser.add_argument('--sorted-alarms-input', type=str, default='', help='直接加载 prepare_sorted_alarms.py 生成的排序告警缓存(JSONL/ZIP)；若 alarms 本身是该缓存格式，也会自动识别')
    parser.add_argument('--sorted-alarms-output', type=str, default='', help='从原始告警加载并排序后，额外写出排序告警缓存；后缀为 .zip 时写压缩包，供后续快速加载')
    parser.add_argument('--compact-output', action='store_true', help='输出轻量化 JSONL：省略 ne_info 内重复告警列表，并压缩空 link 字段；可视化页会从 symptoms 补回节点告警')
    parser.add_argument('--use-alarm-period-cache', action='store_true', help='可选：把 event_cache 切换为“设备告警时段”模式；默认关闭，保持旧版逐条活跃告警缓存逻辑')
    return parser


def _build_output_session(args, engine, static_context, alarm_metadata_index):
    output_session = MatchOutputSession(
        args=args,
        engine=engine,
        output_path=args.output,
        ne_graph_data=static_context.ne_graph_data,
        site_graph_data=static_context.site_graph_data,
        alarm_metadata_index=alarm_metadata_index,
        site_to_ne_ids=static_context.site_to_ne_ids,
        ne_link_info_cache=static_context.ne_link_info_cache,
    )
    output_session.reset_output_file()
    return output_session


def _run_matching_pipeline(
    args,
    engine,
    valid_alarms,
    output_session,
    debug_targets,
    rules_config,
):
    debug_enabled = bool(debug_targets)
    if debug_enabled:
        print("🔎 Debug 模式已开启:")
        for site_id, alarm_name in sorted(debug_targets):
            print(f"   - {site_id} / {alarm_name}")

    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)
    process_progress = ProgressBar(len(valid_alarms), "处理有效告警")
    process_progress._refresh_extra_text = output_session.refresh_progress_extra_text
    output_session.process_progress = process_progress
    output_session.refresh_progress_extra_text(force=True)

    if debug_enabled:
        _run_debug_mode(
            engine,
            valid_alarms,
            output_session.write_matches,
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
            output_session.write_matches,
            process_progress
        )
    else:
        _run_offline_mode(engine, valid_alarms, output_session.write_matches, process_progress)

    output_session.process_progress = None

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
        output_session.write_matches(final_matches)

    if debug_enabled:
        engine.debug_observer = None


def _print_final_summary(args, engine, processed_count, filtered_count, match_count, elapsed):
    final_merge_stats = engine.get_batch_merge_stats_snapshot().get("total", {})
    primary_merge_count = (
        final_merge_stats.get('alarm_overlap_merge_group_count', 0)
        if args.use_alarm_period_cache
        else final_merge_stats.get('eid_merge_group_count', 0)
    )
    primary_merge_label = "告警时段合并组数" if args.use_alarm_period_cache else "eid合并组数"
    print(
        f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，"
        f"生成 {match_count} 个故障组，"
        f"{primary_merge_label} {primary_merge_count}，"
        f"hop合并组数 {final_merge_stats.get('hop_merge_group_count', 0)}，"
        f"距离合并组数 {final_merge_stats.get('distance_merge_group_count', 0)}，"
        f"耗时 {elapsed:.4f} 秒。"
    )


def _prepare_runtime_execution(parser, args):
    start_ts, end_ts = _validate_main_args(parser, args)
    static_context = _load_static_context(args)
    valid_alarm_titles = _default_valid_alarm_titles()
    _print_run_configuration(args, static_context, valid_alarm_titles)
    rules_config = _build_rules_config(args, parser)
    batch_site_merge_helper = _build_batch_site_merge_helper(args, static_context.topo_downstream_map)
    engine = _initialize_engine(args, static_context, rules_config, batch_site_merge_helper)
    start_time = time.time()
    alarm_load_result = _load_alarm_data(
        args,
        parser,
        static_context,
        valid_alarm_titles,
        start_ts,
        end_ts,
    )
    alarm_metadata_index = _build_alarm_metadata_index(alarm_load_result.valid_alarms)
    _print_alarm_load_summary(alarm_load_result)
    debug_targets = _parse_debug_targets(args)
    output_session = _build_output_session(args, engine, static_context, alarm_metadata_index)
    return RuntimeExecutionPlan(
        static_context=static_context,
        rules_config=rules_config,
        engine=engine,
        alarm_load_result=alarm_load_result,
        alarm_metadata_index=alarm_metadata_index,
        debug_targets=debug_targets,
        output_session=output_session,
        start_time=start_time,
    )


def _maybe_compute_ticket_recall(args):
    if not args.compute_ticket_recall:
        return

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


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    runtime_plan = _prepare_runtime_execution(parser, args)
    _run_matching_pipeline(
        args,
        runtime_plan.engine,
        runtime_plan.alarm_load_result.valid_alarms,
        runtime_plan.output_session,
        runtime_plan.debug_targets,
        runtime_plan.rules_config,
    )
    elapsed = time.time() - runtime_plan.start_time
    _print_final_summary(
        args,
        runtime_plan.engine,
        runtime_plan.alarm_load_result.processed_count,
        runtime_plan.alarm_load_result.filtered_count,
        runtime_plan.output_session.match_count,
        elapsed,
    )
    _maybe_compute_ticket_recall(args)


if __name__ == "__main__":
    main()
