import os

from datetime import datetime

from anchor_grouping_online.alarm_inputs import stream_alarm_inputs
from anchor_grouping_online.alarm_events.identity import alarm_content_uuid, require_alarm_identity
from anchor_grouping_online.alarm_events.sorted_cache import load_sorted_alarm_cache
from anchor_grouping_online.time_config import DEFAULT_CLEAR_DELAY_SEC


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


def append_alarm_event(
    valid_alarms,
    alarm,
    site_id,
    alarm_title,
    event_time_str,
    occurrence_uuid,
    is_clear=False,
):
    dt_obj = parse_datetime_text(event_time_str, "告警时间")
    event_alarm = dict(alarm)
    event_alarm["告警首次发生时间"] = event_time_str
    if is_clear:
        event_alarm["清除告警"] = "是"

    event = {
        "alarm": event_alarm,
        "site_id": site_id,
        "alarm_source": alarm.get("告警源", ""),
        "alarm_title": alarm_title,
        "ts": dt_obj.timestamp(),
        "occurrence_uuid": occurrence_uuid,
    }
    eid, event["occurrence_uuid"] = require_alarm_identity(event)
    event_alarm["告警编码ID"] = eid
    valid_alarms.append(event)


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
    valid_sites=None,
    ne_to_site=None,
    start_ts=None,
    end_ts=None,
    clear_delay_sec=0.0,
    allowed_alarm_sources=None,
    region_filter_stats=None,
    show_progress=True,
):
    processed_count = 0
    valid_alarms = []
    normal_alarm_count = 0
    clear_alarm_count = 0

    for alarm in stream_alarm_inputs(alarm_file_path, show_progress=show_progress):
        processed_count += 1

        alarm_title = alarm.get('告警标题', '')
        if alarm_title not in valid_alarm_titles:
            continue

        alarm_source = str(alarm.get('告警源', '') or '').strip()
        if allowed_alarm_sources is not None:
            if region_filter_stats is not None:
                region_filter_stats["raw_checked_alarm_count"] = (
                    region_filter_stats.get("raw_checked_alarm_count", 0) + 1
                )
            if alarm_source not in allowed_alarm_sources:
                if region_filter_stats is not None:
                    region_filter_stats["raw_dropped_alarm_count"] = (
                        region_filter_stats.get("raw_dropped_alarm_count", 0) + 1
                    )
                continue
            if region_filter_stats is not None:
                region_filter_stats["raw_kept_alarm_count"] = (
                    region_filter_stats.get("raw_kept_alarm_count", 0) + 1
                )

        site_id = str(alarm.get('站点ID', '') or '').strip()
        if valid_sites is not None:
            if not site_id or site_id not in valid_sites:
                site_id = str((ne_to_site or {}).get(alarm_source, '') or '').strip()

            if not site_id or site_id not in valid_sites:
                continue

        first_occurrence_str = str(alarm.get("告警首次发生时间", "")).strip()
        first_occurrence_ts = parse_datetime_text(
            first_occurrence_str, "告警首次发生时间"
        ).timestamp()
        if start_ts is not None and first_occurrence_ts < start_ts:
            continue
        if end_ts is not None and first_occurrence_ts > end_ts:
            continue

        occurrence_uuid = alarm_content_uuid(alarm)
        append_alarm_event(
            valid_alarms,
            alarm,
            site_id,
            alarm_title,
            first_occurrence_str,
            occurrence_uuid,
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
                occurrence_uuid,
                is_clear=True
            )
            clear_alarm_count += 1

    return processed_count, valid_alarms, normal_alarm_count, clear_alarm_count


def resolve_alarm_event_site(item, valid_sites, ne_to_site):
    """按当前拓扑校验缓存事件的站点，必要时通过告警源补全站点。"""
    site_id = str(item.get("site_id", "") or "").strip()
    if not site_id or site_id not in valid_sites:
        alarm_source = str(item.get("alarm_source", "") or "").strip()
        site_id = str(ne_to_site.get(alarm_source, "") or "").strip()
    if not site_id or site_id not in valid_sites:
        return None
    if site_id == item.get("site_id"):
        return item
    resolved = dict(item)
    resolved["site_id"] = site_id
    return resolved


def iter_topology_valid_alarm_events(alarm_events, valid_sites, ne_to_site):
    """过滤并补全缓存事件，同时删除过滤后位于末尾的清除事件。"""
    trailing_clear_events = []
    for item in alarm_events:
        resolved = resolve_alarm_event_site(item, valid_sites, ne_to_site)
        if resolved is None:
            continue
        if is_clear_alarm(resolved.get("alarm", {})):
            trailing_clear_events.append(resolved)
            continue
        yield from trailing_clear_events
        trailing_clear_events.clear()
        yield resolved


def trim_trailing_clear_alarms(valid_alarms):
    """删除尾部仅由清除告警组成的区段。"""
    last_non_clear_index = -1
    for idx, item in enumerate(valid_alarms):
        if not is_clear_alarm(item.get("alarm", {})):
            last_non_clear_index = idx

    if last_non_clear_index < 0:
        return []

    return valid_alarms[: last_non_clear_index + 1]


def load_sorted_alarm_cache_with_stats(cache_path, metadata):
    metadata, valid_alarms = load_sorted_alarm_cache(
        cache_path,
        metadata,
        show_progress=True,
    )
    normal_alarm_count = int(metadata["cached_normal_alarm_count"])
    clear_alarm_count = int(metadata["cached_clear_alarm_count"])
    if normal_alarm_count + clear_alarm_count != len(valid_alarms):
        raise ValueError("排序告警缓存统计数量与实际告警数量不一致")
    processed_count = int(metadata["processed_count"])
    return processed_count, valid_alarms, normal_alarm_count, clear_alarm_count, metadata


def warn_sorted_alarm_cache_option_mismatch(metadata, args):
    mismatches = []
    expected_clear_delay = float(metadata["clear_delay_sec"])
    if abs(float(DEFAULT_CLEAR_DELAY_SEC) - expected_clear_delay) > 1e-9:
        mismatches.append(
            f"clear_delay_sec: 缓存={expected_clear_delay:g}, 当前={float(DEFAULT_CLEAR_DELAY_SEC):g}"
        )

    cached_resource_buffer = str(metadata.get("resource_buffer", "") or "").strip()
    current_resource_buffer = os.path.abspath(args.resource_buffer)
    if cached_resource_buffer and os.path.normcase(os.path.normpath(cached_resource_buffer)) != os.path.normcase(
        os.path.normpath(current_resource_buffer)
    ):
        mismatches.append(
            "resource_buffer: "
            f"缓存={cached_resource_buffer}, 当前={current_resource_buffer}"
        )

    if mismatches:
        print(
            "⚠️ 排序告警缓存已预先应用过滤/清除延迟参数，当前参数与缓存元信息不一致："
        )
        for item in mismatches:
            print(f"   - {item}")
        print("   如需使用新参数，请重新生成排序告警缓存。")
