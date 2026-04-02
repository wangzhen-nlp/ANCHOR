import argparse
from datetime import datetime

from alarm_inputs import stream_alarm_inputs


DEFAULT_TIME_FIELDS = [
    "告警首次发生时间",
]


def _parse_time(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_dt(dt_obj):
    if dt_obj is None:
        return "-"
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _init_field_stats():
    return {
        "count": 0,
        "min": None,
        "max": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_TIME_FIELDS,
        help="需要统计的时间字段，默认统计常见时间字段"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="读取输入时显示进度（当前默认已开启，仅保留兼容）"
    )
    args = parser.parse_args()

    field_stats = {field: _init_field_stats() for field in args.fields}
    processed_count = 0

    for alarm in stream_alarm_inputs(args.alarms, show_progress=True):
        processed_count += 1
        for field in args.fields:
            dt_obj = _parse_time(alarm.get(field))
            if dt_obj is None:
                continue
            stats = field_stats[field]
            stats["count"] += 1
            if stats["min"] is None or dt_obj < stats["min"]:
                stats["min"] = dt_obj
            if stats["max"] is None or dt_obj > stats["max"]:
                stats["max"] = dt_obj

    print(f"输入路径: {args.alarms}")
    print(f"处理告警数: {processed_count}")
    print()

    for field in args.fields:
        stats = field_stats[field]
        print(f"[{field}]")
        print(f"命中条数: {stats['count']}")
        print(f"最早时间: {_format_dt(stats['min'])}")
        print(f"最晚时间: {_format_dt(stats['max'])}")
        print()


if __name__ == "__main__":
    main()
