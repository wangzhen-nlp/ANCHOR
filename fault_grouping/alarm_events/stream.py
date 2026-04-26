import time

from fault_grouping.alarm_events.io import is_clear_alarm


def stream_alarms_by_ts(valid_alarms, speedup=1.0):
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


def process_alarm(engine, item, collect_matches=False, register_trigger=True):
    alarm = item["alarm"]
    return engine.process_event(
        node=item["site_id"],
        alarm_source=item.get("alarm_source", ""),
        alarm_type=item["alarm_title"],
        ts=item["ts"],
        event_id=alarm["告警编码ID"],
        is_clear=is_clear_alarm(alarm),
        collect_matches=collect_matches,
        register_trigger=register_trigger
    )


def build_simulated_now_ts_getter(valid_alarms, speedup):
    """构造与告警 ts 回放节奏一致的模拟时钟。"""
    if not valid_alarms:
        return time.time

    simulated_start_ts = valid_alarms[0]["ts"]
    real_start_monotonic = time.monotonic()

    def get_now_ts():
        elapsed_real_sec = time.monotonic() - real_start_monotonic
        return simulated_start_ts + elapsed_real_sec * speedup

    return get_now_ts


def refresh_process_progress(process_progress):
    if hasattr(process_progress, "_refresh_extra_text"):
        process_progress._refresh_extra_text()
    process_progress.update()
