import json
import time
import threading

from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

from alarm_inputs import (
    build_ne_to_site_map,
    load_site_graph,
    stream_alarm_inputs,
)
from alarm_types import CRITICAL_ALARMS
from progress_utils import ProgressBar
from reports import generate_incident_report
from rule_config import transmission_rule, power_rule
from temporal_graph_engine import TemporalGraphEngine


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
    dt_obj = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
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


def _load_valid_alarms(alarm_file_path, valid_alarm_titles, valid_sites, ne_to_site):
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

        _append_alarm_event(
            valid_alarms,
            alarm,
            site_id,
            alarm_title,
            alarm["告警首次发生时间"],
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


def _build_group_output(match, ne_graph_data):
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

        ne_alarms[ne_id].append({
            "alarm_id": symptom.get("eid", ""),
            "alarm_type": symptom.get("alarm", ""),
            "alarm_time": datetime.fromtimestamp(symptom["ts"]).strftime("%Y-%m-%d %H:%M:%S") if symptom.get("ts") is not None else "",
            "domain": ne_graph_entry.get("domain", ""),
            "site_id": site_id,
            "site_name": ne_graph_entry.get("site_name", ""),
            "matched_role": symptom.get("matched_role", ""),
        })

    group_ne_ids = sorted({
        ne_id
        for ne_id, ne_graph_entry in ne_graph_data.items()
        if ne_graph_entry.get("site_id", "") in group_site_ids
    } | set(ne_alarms.keys()))

    for ne_id in group_ne_ids:
        ne_graph_entry = ne_graph_data.get(ne_id, {})
        site_id = ne_graph_entry.get("site_id", "")
        if site_id:
            group_site_ids.add(site_id)

        alarms = sorted(
            ne_alarms.get(ne_id, []),
            key=lambda alarm: (alarm.get("alarm_time", ""), alarm.get("alarm_id", ""))
        )

        ne_info[ne_id] = {
            "alarm": alarms,
            "link": _build_group_link_info(ne_id, set(group_ne_ids), ne_graph_data),
            "group": group_id,
            "name": ne_graph_entry.get("name", ne_id),
            "site_id": site_id,
            "site_name": ne_graph_entry.get("site_name", ""),
            "type": str(ne_graph_entry.get("type", "")).upper(),
            "network_type": str(ne_graph_entry.get("network_type", "")).upper(),
            "manufacturer": str(ne_graph_entry.get("manufacturer", "")).upper(),
            "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
            "domain": str(ne_graph_entry.get("domain", "")).upper(),
            "region_id": ne_graph_entry.get("region_id", ""),
            "longitude": ne_graph_entry.get("longitude", ""),
            "latitude": ne_graph_entry.get("latitude", ""),
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


def _build_jsonl_match_output(match, ne_graph_data):
    group_output = _build_group_output(match, ne_graph_data)
    timestamps = [symptom["ts"] for symptom in match.get("symptoms", []) if symptom.get("ts") is not None]
    group_anchor_ts = min(timestamps) if timestamps else None

    enriched_match = dict(match)
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
    for ts, eid, alarm_type, alarm_source, consumed_as_trigger in events:
        formatted.append(
            {
                "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "eid": eid,
                "alarm": alarm_type,
                "source": alarm_source,
                "consumed_as_trigger": consumed_as_trigger,
            }
        )
    return json.dumps({"total": len(site_events), "events": formatted}, ensure_ascii=False)


def _format_debug_trigger_index(engine, site_id):
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
    return json.dumps(entries, ensure_ascii=False)


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


def _print_debug_collection_snapshot(snapshot, debug_targets, rules_config):
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
    print(
        f"🔎 收割阶段快照: watermark={watermark_str}, "
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

    stage_mapping = (
        ("原始候选组", raw_debug_matches),
        ("当前批次合并后", batch_debug_matches),
        ("历史组合并后", finalized_debug_matches),
        ("pending 扩充后", expanded_debug_matches),
    )
    for stage_name, stage_matches in stage_mapping:
        print(f"   ↳ {stage_name}: {len(stage_matches)} 个相关故障组")
        for match in stage_matches:
            _print_debug_match_details(match)


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
    engine.debug_observer = lambda snapshot: _print_debug_collection_snapshot(
        snapshot, debug_targets, rules_config
    )

    def on_debug_matches(matches, source="收割"):
        debug_matches = [
            match for match in matches
            if _match_debug_trigger(match, debug_targets, rules_config)
        ]
        if not debug_matches:
            return
        print(f"🔎 {source}命中 {len(debug_matches)} 个故障组")
        for match in debug_matches:
            _print_debug_match_details(match)
        on_matches(debug_matches)

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
                _process_alarm(
                    engine,
                    item,
                    collect_matches=False,
                    register_trigger=True
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
                    print(f"   ↳ 当前 pending: {_format_debug_pending(engine, debug_site)}")
                process_progress.update()
        finally:
            process_progress.close()
            engine.stop_periodic_harvest()
        return

    try:
        for item in valid_alarms:
            is_debug_trigger = _is_debug_trigger_item(item, debug_targets)
            matches = _process_alarm(
                engine,
                item,
                collect_matches=True,
                register_trigger=True
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
                print(f"   ↳ 当前 pending: {_format_debug_pending(engine, debug_site)}")
                if not matches:
                    print("   ↳ 当前触发点暂未产出故障组")
            if matches:
                on_debug_matches(matches, "同步检查")
            process_progress.update()
    finally:
        process_progress.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, help='output jsonl file')
    parser.add_argument('--topo', type=str, default='site_graph_by_ne.json')
    parser.add_argument('--site-domain', type=str, default='site_device_counts.json')
    parser.add_argument('--ne-graph', type=str, default='ne_graph.json', help='ne_graph.json 文件')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='live', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=300.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    parser.add_argument('--debug-trigger', action='append', help='debug: 指定一个 trigger，格式为 站点ID::告警名，可重复传多次')
    args = parser.parse_args()

    topo_downstream_map = json.load(open(args.topo, 'r', encoding='utf-8'))
    site_domain_map = json.load(open(args.site_domain, 'r', encoding='utf-8'))
    ne_graph_data = json.load(open(args.ne_graph, 'r', encoding='utf-8'))

    print("加载有效站点集合...")
    valid_sites = load_site_graph(args.topo)
    print(f"有效站点数: {len(valid_sites)}")

    print("构建 ne -> site 映射...")
    ne_to_site = build_ne_to_site_map(args.ne_graph)
    print(f"NE 数量: {len(ne_to_site)}")

    valid_alarm_titles = CRITICAL_ALARMS
    print(f"有效告警类型数: {len(valid_alarm_titles)}")

    rules_config = {
        "transmission_rule": transmission_rule,
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
        ne_to_site
    )

    print("⏳ 正在按时间排序有效告警...")
    sort_start_time = time.time()
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = _trim_trailing_clear_alarms(valid_alarms)
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
    output_lock = threading.Lock()
    with open(args.output, 'w', encoding='utf-8'):
        pass

    def on_matches(matches):
        # 统一处理一批新产出的故障组：边输出报告，边按 jsonl 落盘。
        nonlocal match_count
        with output_lock:
            with open(args.output, 'a', encoding='utf-8') as fw:
                for match in matches:
                    generate_incident_report(match)
                    enriched_match = _build_jsonl_match_output(match, ne_graph_data)
                    fw.write(json.dumps(enriched_match, ensure_ascii=False) + '\n')
            match_count += len(matches)

    process_progress = ProgressBar(filtered_count, "处理有效告警")
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
                on_matches(debug_final_matches)
        else:
            on_matches(final_matches)

    if debug_enabled:
        engine.debug_observer = None

    elapsed = time.time() - start_time
    print(f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，生成 {match_count} 个故障组，耗时 {elapsed:.4f} 秒。")


if __name__ == "__main__":
    main()
