import os

from datetime import datetime

from alarm_tools.alarm_inputs import stream_alarm_inputs
from fault_grouping.sorted_alarm_cache import load_sorted_alarm_cache


def parse_datetime_text(text, field_name="时间"):
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


def is_clear_alarm(alarm):
    clear_value = alarm.get("清除告警", None)
    if clear_value is None:
        return False
    return str(clear_value).strip().lower() in {"是", "yes", "true", "1", "y"}


def append_alarm_event(valid_alarms, alarm, site_id, alarm_title, event_time_str, is_clear=False):
    dt_obj = parse_datetime_text(event_time_str, "告警时间")
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


def apply_clear_delay(first_occurrence_str, clear_time_str, clear_delay_sec):
    first_occurrence_dt = parse_datetime_text(first_occurrence_str, "告警首次发生时间")
    clear_time_dt = parse_datetime_text(clear_time_str, "告警清除时间")

    actual_delay_sec = max(0.0, (clear_time_dt - first_occurrence_dt).total_seconds())
    effective_delay_sec = max(float(clear_delay_sec), actual_delay_sec)
    effective_clear_dt = first_occurrence_dt.fromtimestamp(
        first_occurrence_dt.timestamp() + effective_delay_sec
    )
    return effective_clear_dt.strftime("%Y-%m-%d %H:%M:%S")


def load_valid_alarms(
    alarm_file_path,
    valid_alarm_titles,
    valid_sites,
    ne_to_site,
    start_ts=None,
    end_ts=None,
    clear_delay_sec=0.0,
):
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
        first_occurrence_dt = parse_datetime_text(first_occurrence_str, "告警首次发生时间")
        first_occurrence_ts = first_occurrence_dt.timestamp()
        if start_ts is not None and first_occurrence_ts < start_ts:
            continue
        if end_ts is not None and first_occurrence_ts > end_ts:
            continue

        append_alarm_event(
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
            effective_clear_time_str = apply_clear_delay(
                first_occurrence_str,
                clear_time_str,
                clear_delay_sec,
            )
            append_alarm_event(
                valid_alarms,
                alarm,
                site_id,
                alarm_title,
                effective_clear_time_str,
                is_clear=True
            )
            clear_alarm_count += 1

    return processed_count, valid_alarms, normal_alarm_count, clear_alarm_count


def trim_trailing_clear_alarms(valid_alarms):
    """删除尾部仅由清除告警组成的区段。"""
    last_non_clear_index = -1
    for idx, item in enumerate(valid_alarms):
        if not is_clear_alarm(item.get("alarm", {})):
            last_non_clear_index = idx

    if last_non_clear_index < 0:
        return []

    return valid_alarms[: last_non_clear_index + 1]


def count_alarm_event_types(valid_alarms):
    clear_alarm_count = sum(
        1 for item in valid_alarms if is_clear_alarm(item.get("alarm", {}))
    )
    return len(valid_alarms) - clear_alarm_count, clear_alarm_count


def load_sorted_alarm_cache_with_stats(cache_path):
    metadata, valid_alarms = load_sorted_alarm_cache(cache_path, show_progress=True)
    normal_alarm_count = int(
        metadata.get("cached_normal_alarm_count")
        if metadata.get("cached_normal_alarm_count") is not None
        else metadata.get("normal_alarm_count", 0)
    )
    clear_alarm_count = int(
        metadata.get("cached_clear_alarm_count")
        if metadata.get("cached_clear_alarm_count") is not None
        else metadata.get("clear_alarm_count", 0)
    )
    if normal_alarm_count + clear_alarm_count != len(valid_alarms):
        normal_alarm_count, clear_alarm_count = count_alarm_event_types(valid_alarms)
    processed_count = int(metadata.get("processed_count", len(valid_alarms)))
    return processed_count, valid_alarms, normal_alarm_count, clear_alarm_count, metadata


def warn_sorted_alarm_cache_option_mismatch(metadata, args):
    if not metadata:
        return

    mismatches = []
    expected_start_time = str(metadata.get("start_time", "") or "")
    expected_end_time = str(metadata.get("end_time", "") or "")
    expected_clear_delay = float(metadata.get("clear_delay_sec", 0.0) or 0.0)
    expected_topo = str(metadata.get("topo", "") or "")
    expected_ne_graph = str(metadata.get("ne_graph", "") or "")
    current_topo = os.path.abspath(args.topo) if args.topo else ""
    current_ne_graph = os.path.abspath(args.ne_graph)
    if expected_topo and expected_topo != current_topo:
        mismatches.append(f"topo: 缓存={expected_topo}, 当前={current_topo or '-'}")
    if expected_ne_graph and expected_ne_graph != current_ne_graph:
        mismatches.append(f"ne_graph: 缓存={expected_ne_graph}, 当前={current_ne_graph}")
    if (args.start_time or "") != expected_start_time:
        mismatches.append(f"start_time: 缓存={expected_start_time or '-'}, 当前={args.start_time or '-'}")
    if (args.end_time or "") != expected_end_time:
        mismatches.append(f"end_time: 缓存={expected_end_time or '-'}, 当前={args.end_time or '-'}")
    if abs(float(args.clear_delay_sec) - expected_clear_delay) > 1e-9:
        mismatches.append(
            f"clear_delay_sec: 缓存={expected_clear_delay:g}, 当前={float(args.clear_delay_sec):g}"
        )

    if mismatches:
        print(
            "⚠️ 排序告警缓存已预先应用过滤/清除延迟参数，当前参数与缓存元信息不一致："
        )
        for item in mismatches:
            print(f"   - {item}")
        print("   如需使用新参数，请重新生成排序告警缓存。")
