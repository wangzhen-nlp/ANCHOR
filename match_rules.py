import json
import time
import threading

from datetime import datetime
from argparse import ArgumentParser

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


def _load_valid_alarms(alarm_file_path, valid_alarm_titles, valid_sites, ne_to_site):
    processed_count = 0
    valid_alarms = []

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

        dt_obj = datetime.strptime(alarm["告警首次发生时间"], "%Y-%m-%d %H:%M:%S")
        valid_alarms.append({
            "alarm": alarm,
            "site_id": site_id,
            "alarm_title": alarm_title,
            "ts": dt_obj.timestamp()
        })

    return processed_count, valid_alarms


def _process_alarm(engine, item, collect_matches=False):
    alarm = item["alarm"]
    return engine.process_event(
        node=item["site_id"],
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
    parser.add_argument('--topo', type=str, default='site_graph_by_ne.json')
    parser.add_argument('--site-domain', type=str, default='site_device_counts.json')
    parser.add_argument('--ne-graph', type=str, default='ne_graph.json', help='ne_graph.json 文件')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='live', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=10.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    args = parser.parse_args()

    topo_downstream_map = json.load(open(args.topo, 'r', encoding='utf-8'))
    site_domain_map = json.load(open(args.site_domain, 'r', encoding='utf-8'))

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

    processed_count, valid_alarms = _load_valid_alarms(
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

    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)

    match_count = 0
    output_lock = threading.Lock()

    # 输出改为 jsonl 追加写，避免把全部故障组长期堆在内存里。
    with open(args.output, 'w', encoding='utf-8'):
        pass

    def on_matches(matches):
        # 统一处理一批新产出的故障组：边输出报告，边按 jsonl 追加落盘。
        nonlocal match_count
        with output_lock:
            with open(args.output, 'a', encoding='utf-8') as fw:
                for match in matches:
                    generate_incident_report(match)
                    fw.write(json.dumps(match, ensure_ascii=False) + '\n')
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


if __name__ == "__main__":
    main()
