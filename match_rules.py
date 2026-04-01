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
from rule_config import transmission_rule
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


def _process_alarm(engine, item, collect_matches=False):
    alarm = item["alarm"]
    return engine.process_event(
        node=item["site_id"],
        alarm_source=item.get("alarm_source", ""),
        alarm_type=item["alarm_title"],
        ts=item["ts"],
        event_id=alarm["告警编码ID"],
        is_clear=_is_clear_alarm(alarm),
        collect_matches=collect_matches
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

        ne_info[ne_id] = {
            "alarm": ne_alarms.get(ne_id, []),
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


def _merge_group_output(aggregated_output, group_output):
    aggregated_output["group_info"].update(group_output.get("group_info", {}))

    group_info = next(iter(group_output.get("group_info", {}).values()), {})
    group_ne_ids = set(group_info.get("ne_list", []))
    match_info = group_output.get("match_info", {})
    group_id = match_info.get("uuid", "")

    for ne_id, ne_data in group_output.get("ne_info", {}).items():
        aggregated_ne_id = f"{group_id}::{ne_id}" if group_id else ne_id
        aggregated_ne_data = dict(ne_data)
        aggregated_ne_data["group"] = group_id

        link_info = {}
        for neighbor_id, link_data in ne_data.get("link", {}).items():
            if neighbor_id not in group_ne_ids:
                continue
            aggregated_neighbor_id = f"{group_id}::{neighbor_id}" if group_id else neighbor_id
            link_info[aggregated_neighbor_id] = link_data
        aggregated_ne_data["link"] = link_info

        aggregated_output["ne_info"][aggregated_ne_id] = aggregated_ne_data

    aggregated_output["match_info"][group_id] = match_info


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


def main():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, help='output jsonl file')
    parser.add_argument('--output-format', type=str, choices=('jsonl', 'propagation-json'), default='jsonl',
                        help='jsonl: 每行一个原始故障组; propagation-json: 输出单个传播图 JSON 文件')
    parser.add_argument('--topo', type=str, default='site_graph_by_ne.json')
    parser.add_argument('--site-domain', type=str, default='site_device_counts.json')
    parser.add_argument('--ne-graph', type=str, default='ne_graph.json', help='ne_graph.json 文件')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='live', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=300.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
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
        "transmission_rule": transmission_rule
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
    sort_elapsed = time.time() - sort_start_time
    filtered_count = len(valid_alarms)
    print(f"有效告警数: {filtered_count}，排序耗时: {sort_elapsed:.4f} 秒")
    print(f"正常告警数: {normal_alarm_count}，清除告警数: {clear_alarm_count}")

    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)

    match_count = 0
    output_lock = threading.Lock()
    aggregated_output = {"ne_info": {}, "group_info": {}, "match_info": {}}

    if args.output_format == 'jsonl':
        # 默认输出为 jsonl，避免把全部故障组长期堆在内存里。
        with open(args.output, 'w', encoding='utf-8'):
            pass

    def on_matches(matches):
        # 统一处理一批新产出的故障组：边输出报告，边按指定格式落盘。
        nonlocal match_count
        with output_lock:
            if args.output_format == 'jsonl':
                with open(args.output, 'a', encoding='utf-8') as fw:
                    for match in matches:
                        generate_incident_report(match)
                        fw.write(json.dumps(match, ensure_ascii=False) + '\n')
            else:
                for match in matches:
                    generate_incident_report(match)
                    group_output = _build_group_output(match, ne_graph_data)
                    _merge_group_output(aggregated_output, group_output)
            match_count += len(matches)

    process_progress = ProgressBar(filtered_count, "处理有效告警")
    if args.mode == 'live':
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
        on_matches(final_matches)

    elapsed = time.time() - start_time
    print(f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，生成 {match_count} 个故障组，耗时 {elapsed:.4f} 秒。")

    if args.output_format == 'propagation-json':
        with open(args.output, 'w', encoding='utf-8') as fw:
            json.dump(aggregated_output, fw, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
