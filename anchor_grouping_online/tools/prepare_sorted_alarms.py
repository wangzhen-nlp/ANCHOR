import os
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.alarm_types import CRITICAL_ALARMS
from anchor_grouping_online.alarm_events.io import (
    is_clear_alarm,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
)
from anchor_grouping_online.alarm_events.sorted_cache import write_sorted_alarm_cache


def build_sorted_alarms(
    alarm_input,
    *,
    start_time=None,
    end_time=None,
    clear_delay_sec=0.0,
    show_progress=True,
):
    start_ts = parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time 不能晚于 end_time")

    processed_count, valid_alarms, normal_alarm_count, clear_alarm_count = load_valid_alarms(
        alarm_input,
        CRITICAL_ALARMS,
        start_ts=start_ts,
        end_ts=end_ts,
        clear_delay_sec=clear_delay_sec,
        show_progress=show_progress,
    )
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)

    cached_normal_alarm_count = sum(
        1 for item in valid_alarms if not is_clear_alarm(item.get("alarm", {}))
    )
    cached_clear_alarm_count = len(valid_alarms) - cached_normal_alarm_count
    metadata = {
        "source_alarms": os.path.abspath(alarm_input),
        "start_time": start_time or "",
        "end_time": end_time or "",
        "clear_delay_sec": float(clear_delay_sec),
        "processed_count": processed_count,
        "normal_alarm_count": normal_alarm_count,
        "clear_alarm_count": clear_alarm_count,
        "cached_normal_alarm_count": cached_normal_alarm_count,
        "cached_clear_alarm_count": cached_clear_alarm_count,
        "valid_alarm_title_count": len(CRITICAL_ALARMS),
    }
    return valid_alarms, metadata


def main():
    parser = ArgumentParser(description="预处理 match_rules.py 输入，生成已排序告警缓存(JSONL/ZIP，包含清除告警)")
    parser.add_argument("alarms", help="原始告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致")
    parser.add_argument("output", help="排序告警缓存输出；后缀为 .zip 时写压缩包，否则写 JSONL")
    parser.add_argument("--start_time", type=str, help="仅处理告警首次发生时间 >= 该时间")
    parser.add_argument("--end_time", type=str, help="仅处理告警首次发生时间 <= 该时间")
    parser.add_argument(
        "--clear-delay-sec",
        type=float,
        default=0.0,
        help="清除告警最小延迟时间，清除生效时间=max(clear_delay_sec, 清除时间-发生时间)+发生时间",
    )
    args = parser.parse_args()

    start_time = time.time()
    valid_alarms, metadata = build_sorted_alarms(
        args.alarms,
        start_time=args.start_time,
        end_time=args.end_time,
        clear_delay_sec=args.clear_delay_sec,
    )
    header = write_sorted_alarm_cache(args.output, valid_alarms, metadata)
    elapsed = time.time() - start_time
    print(
        f"✅ 排序告警缓存已写入: {args.output}\n"
        f"   缓存告警数: {header['alarm_count']}\n"
        f"   正常告警数: {metadata['cached_normal_alarm_count']}，"
        f"清除告警数: {metadata['cached_clear_alarm_count']}\n"
        f"   耗时: {elapsed:.2f} 秒"
    )


if __name__ == "__main__":
    main()
