import json

from dataclasses import dataclass
from datetime import datetime

from alarm_tools.alarm_types import POWER_ALARMS
from fault_grouping.alarm_events.io import is_clear_alarm
from fault_grouping.alarm_events.stream import (
    build_simulated_now_ts_getter,
    process_alarm,
    refresh_process_progress,
    stream_alarms_by_ts,
)
from fault_grouping.temporal_engine.utils import get_match_alarm_keys


@dataclass
class DebugRunContext:
    debug_targets: set
    debug_sites: set
    last_trigger_snapshots: dict


def match_debug_trigger(match, debug_targets, rules_config):
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


def build_match_time_range(match):
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


def parse_debug_targets(args):
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


def build_debug_run_context(engine, debug_targets):
    debug_sites = {site_id for site_id, _alarm_name in debug_targets}
    last_trigger_snapshots = {
        site_id: snapshot_debug_trigger_index(engine, site_id)
        for site_id in debug_sites
    }
    return DebugRunContext(
        debug_targets=debug_targets,
        debug_sites=debug_sites,
        last_trigger_snapshots=last_trigger_snapshots,
    )


def format_debug_site_events(engine, site_id, limit=50):
    with engine._lock:
        site_events = list(engine.event_cache.get(site_id, []))

    events = site_events[-limit:]
    if not events:
        return json.dumps({"total": 0, "events": []}, ensure_ascii=False)

    formatted = []
    for cached_event in events:
        if isinstance(cached_event, dict):
            ts = cached_event.get("ts")
            eid = cached_event.get("eid")
            alarm_type = cached_event.get("alarm")
            alarm_source = cached_event.get("alarm_source", "")
            consumed_trigger_rules = cached_event.get("consumed_trigger_rules", ())
            occurrence_id = cached_event.get("occurrence_id") or cached_event.get("_raw_event_occurrence_key")
        else:
            try:
                ts, eid, alarm_type, alarm_source, consumed_trigger_rules, occurrence_id = cached_event
            except (TypeError, ValueError):
                ts, eid, alarm_type, alarm_source, consumed_trigger_rules = cached_event
                occurrence_id = None
        formatted.append(
            {
                "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "eid": eid,
                "occurrence_id": occurrence_id,
                "alarm": alarm_type,
                "source": alarm_source,
                "consumed_trigger_rules": sorted(consumed_trigger_rules),
            }
        )
    return json.dumps({"total": len(site_events), "events": formatted}, ensure_ascii=False)


def format_debug_trigger_index(engine, site_id):
    entries = snapshot_debug_trigger_index(engine, site_id)
    return json.dumps(entries, ensure_ascii=False)


def snapshot_debug_trigger_index(engine, site_id):
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


def print_debug_trigger_changes(engine, debug_sites, previous_snapshots, header, harvest_snapshot=None):
    changed_items = []
    for site_id in sorted(debug_sites):
        current_snapshot = snapshot_debug_trigger_index(engine, site_id)
        previous_snapshot = previous_snapshots.get(site_id, {})
        if current_snapshot == previous_snapshot:
            continue
        changed_items.append((site_id, previous_snapshot, current_snapshot))

    if not changed_items:
        return False

    if harvest_snapshot is not None:
        print_debug_harvest_actions(harvest_snapshot, engine)

    for site_id, previous_snapshot, current_snapshot in changed_items:
        print(f"{header}[{site_id}]")
        print(f"   ↳ 变化前: {json.dumps(previous_snapshot, ensure_ascii=False)}")
        print(f"   ↳ 变化后: {json.dumps(current_snapshot, ensure_ascii=False)}")
        previous_snapshots[site_id] = current_snapshot
    return True


def _get_debug_match_alarm_keys(match):
    return get_match_alarm_keys(match)


def _match_present_in_stage(raw_match, stage_matches):
    raw_alarm_keys = _get_debug_match_alarm_keys(raw_match)
    if not raw_alarm_keys:
        return False
    for stage_match in stage_matches:
        stage_alarm_keys = _get_debug_match_alarm_keys(stage_match)
        if raw_alarm_keys.issubset(stage_alarm_keys):
            return True
    return False


def print_debug_harvest_actions(snapshot, engine):
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
        trigger_detail = find_trigger_event_detail(engine, site_id, rule_name, (trigger_ts, trigger_seq))
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
                print_debug_match_details(raw_match)

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
                print("         - 最终未输出：当前候选组的告警时段已被历史故障组完全覆盖，只延长历史组停留时间")
            elif reason == "merged_with_related_history":
                print("         - 最终输出：与历史相关组做了合并后重新输出")
            elif reason == "no_related_history":
                print("         - 最终输出：没有命中任何历史相关组")
            elif reason == "no_alarm_keys":
                print("         - 最终阶段异常：候选组没有可用告警时段键，无法做历史合并判断")

    if snapshot.get("finalized_matches"):
        print(
            "   ↳ 提示: 本次收割进入了 finalize 阶段，后续会执行 "
            "_prune_consumed_alarm_history -> _prune_node_alarm_history_before -> "
            "_refresh_pending_triggers_for_node，"
            "关注站点 trigger 的变化可能来自这里，而不一定是某个 mature trigger 未产出原始候选组"
        )


def find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor):
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


def format_debug_pending(engine, site_id):
    with engine._lock:
        pending_items = [
            ((node, rule_name), trigger_anchor)
            for (node, rule_name), trigger_anchor in engine.pending_triggers.items()
            if node == site_id
        ]

    pending = {}
    for (_node, rule_name), trigger_anchor in pending_items:
        trigger_ts, trigger_seq = trigger_anchor
        trigger_detail = find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor)
        pending[rule_name] = {
            "site": site_id,
            "alarm": trigger_detail["alarm"],
            "eid": trigger_detail["eid"],
            "trigger_time": datetime.fromtimestamp(trigger_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_seq": trigger_detail["seq"],
            "ready_time": datetime.fromtimestamp(trigger_ts + engine.aggregation_wait_sec).strftime("%Y-%m-%d %H:%M:%S"),
        }
    return json.dumps(pending, ensure_ascii=False)


def print_debug_pending_items(engine, site_id, header):
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
        trigger_detail = find_trigger_event_detail(engine, site_id, rule_name, trigger_anchor)
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


def print_debug_match_details(match):
    print(
        "   ↳ 命中故障组: "
        f"uuid={match.get('uuid', '')}, "
        f"rules={'+'.join(match.get('merged_rules', [match.get('rule', '')]))}, "
        f"time_range={build_match_time_range(match)}"
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


def print_debug_post_batch_state(engine, debug_sites):
    for site_id in sorted(debug_sites):
        print(f"   ↳ 本批完成后站点状态[{site_id}]")
        print(f"      event_cache={format_debug_site_events(engine, site_id)}")
        print(f"      trigger_index={format_debug_trigger_index(engine, site_id)}")
        print_debug_pending_items(engine, site_id, "      pending")


def print_debug_pending_eval_profiles(snapshot, debug_sites):
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
            print_debug_match_details(match)


def print_debug_collection_snapshot(snapshot, debug_targets, rules_config, engine):
    debug_sites = {site_id for site_id, _alarm_name in debug_targets}
    raw_debug_matches = [
        match for match in snapshot.get("raw_matches", [])
        if match_debug_trigger(match, debug_targets, rules_config)
    ]
    batch_debug_matches = [
        match for match in snapshot.get("batch_merged_matches", [])
        if match_debug_trigger(match, debug_targets, rules_config)
    ]
    finalized_debug_matches = [
        match for match in snapshot.get("finalized_matches", [])
        if match_debug_trigger(match, debug_targets, rules_config)
    ]
    expanded_debug_matches = [
        match for match in snapshot.get("expanded_matches", [])
        if match_debug_trigger(match, debug_targets, rules_config)
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
    merge_stats = snapshot.get("merge_stats", {})
    if merge_stats:
        primary_merge_count = (
            merge_stats.get('alarm_overlap_merge_group_count', 0)
            if snapshot.get("use_alarm_period_cache")
            else merge_stats.get('eid_merge_group_count', 0)
        )
        primary_merge_label = "alarm_overlap" if snapshot.get("use_alarm_period_cache") else "eid"
        print(
            "   ↳ 批内合并统计: "
            f"{primary_merge_label}={primary_merge_count}, "
            f"shared_site={merge_stats.get('shared_site_merge_group_count', 0)}, "
            f"hop={merge_stats.get('hop_merge_group_count', 0)}, "
            f"distance={merge_stats.get('distance_merge_group_count', 0)}"
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
        print_debug_pending_eval_profiles(snapshot, debug_sites)

    stage_mapping = (
        ("原始候选组", raw_debug_matches),
        ("当前批次合并后", batch_debug_matches),
        ("pending 扩充后", expanded_debug_matches),
        ("历史组合并后", finalized_debug_matches),
    )
    for stage_name, stage_matches in stage_mapping:
        print(f"   ↳ {stage_name}: {len(stage_matches)} 个相关故障组")
        for match in stage_matches:
            print_debug_match_details(match)

    print_debug_post_batch_state(engine, debug_sites)


def print_debug_event_removal(payload, debug_sites):
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


def is_debug_trigger_item(item, debug_targets):
    return (
        (item.get("site_id"), item.get("alarm_title")) in debug_targets
        and not is_clear_alarm(item.get("alarm", {}))
    )


def is_debug_site_power_alarm(item, debug_sites):
    return (
        item.get("site_id") in debug_sites
        and item.get("alarm_title") in POWER_ALARMS
        and not is_clear_alarm(item.get("alarm", {}))
    )


def is_debug_site_clear(item, debug_sites):
    return (
        item.get("site_id") in debug_sites
        and is_clear_alarm(item.get("alarm", {}))
    )


def get_debug_item_kind(item, debug_context):
    if is_debug_trigger_item(item, debug_context.debug_targets):
        return "trigger"
    if is_debug_site_power_alarm(item, debug_context.debug_sites):
        return "power"
    if is_debug_site_clear(item, debug_context.debug_sites):
        return "clear"
    return None


def print_debug_item_state(engine, item, item_kind, matches=None):
    debug_site = item.get("site_id", "")
    debug_alarm = item.get("alarm_title", "")
    debug_time = datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S")
    debug_eid = item['alarm'].get('告警编码ID', '')

    if item_kind == "trigger":
        print(
            f"🔎 Trigger 输入: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 当前站点最近事件: {format_debug_site_events(engine, debug_site)}")
        print(f"   ↳ 当前 trigger_index: {format_debug_trigger_index(engine, debug_site)}")
        print_debug_pending_items(engine, debug_site, "   ↳ 当前 pending")
        if matches is not None and not matches:
            print("   ↳ 当前触发点暂未产出故障组")
        return

    if item_kind == "power":
        print(
            f"⚡ 电告警进入缓存: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 当前站点最近事件: {format_debug_site_events(engine, debug_site)}")
        return

    if item_kind == "clear":
        print(
            f"🧹 清除告警输入: site={debug_site}, alarm={debug_alarm}, "
            f"time={debug_time}, eid={debug_eid}"
        )
        print(f"   ↳ 清除后站点最近事件: {format_debug_site_events(engine, debug_site)}")
        print(f"   ↳ 清除后 trigger_index: {format_debug_trigger_index(engine, debug_site)}")
        print_debug_pending_items(engine, debug_site, "   ↳ 清除后 pending")


def build_debug_match_callback(on_matches, debug_targets, rules_config):
    def on_debug_matches(matches, source="收割"):
        debug_matches = [
            match for match in matches
            if match_debug_trigger(match, debug_targets, rules_config)
        ]
        if debug_matches:
            print(f"🔎 {source}命中 {len(debug_matches)} 个故障组")
            for match in debug_matches:
                print_debug_match_details(match)
        on_matches(matches)

    return on_debug_matches


def run_debug_live_loop(
    engine,
    valid_alarms,
    speedup,
    process_progress,
    debug_context,
):
    for item in stream_alarms_by_ts(valid_alarms, speedup=speedup):
        item_kind = get_debug_item_kind(item, debug_context)
        process_alarm(
            engine,
            item,
            collect_matches=False,
            register_trigger=True
        )
        print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 告警处理后关注站点 trigger 变化",
        )
        if item_kind is not None:
            print_debug_item_state(engine, item, item_kind)
        refresh_process_progress(process_progress)


def run_debug_offline_loop(
    engine,
    valid_alarms,
    process_progress,
    debug_context,
    on_debug_matches,
):
    for item in valid_alarms:
        item_kind = get_debug_item_kind(item, debug_context)
        matches = process_alarm(
            engine,
            item,
            collect_matches=True,
            register_trigger=True
        )
        print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 告警处理后关注站点 trigger 变化",
        )
        if item_kind is not None:
            print_debug_item_state(engine, item, item_kind, matches=matches)
        if matches:
            on_debug_matches(matches, "同步检查")
        refresh_process_progress(process_progress)


def run_debug_mode(
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
    debug_context = build_debug_run_context(engine, debug_targets)

    def on_debug_snapshot(snapshot):
        print_debug_collection_snapshot(snapshot, debug_targets, rules_config, engine)
        print_debug_trigger_changes(
            engine,
            debug_context.debug_sites,
            debug_context.last_trigger_snapshots,
            "🔁 收割后关注站点 trigger 变化",
            harvest_snapshot=snapshot,
        )

    engine.debug_observer = on_debug_snapshot
    engine.debug_event_logger = lambda payload: print_debug_event_removal(payload, debug_context.debug_sites)
    on_debug_matches = build_debug_match_callback(on_matches, debug_targets, rules_config)

    if mode == 'live':
        now_ts_getter = build_simulated_now_ts_getter(valid_alarms, speedup)
        engine.start_periodic_harvest(
            interval_sec=real_harvest_interval_sec,
            on_matches=lambda matches: on_debug_matches(matches, "定时收割"),
            now_ts_getter=now_ts_getter
        )
        try:
            run_debug_live_loop(
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
        run_debug_offline_loop(
            engine,
            valid_alarms,
            process_progress,
            debug_context,
            on_debug_matches,
        )
    finally:
        process_progress.close()
        engine.debug_event_logger = None
