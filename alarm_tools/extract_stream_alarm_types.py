import argparse

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import stream_alarm_inputs


DEFAULT_ALARM_FIELD = "告警标题"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    parser.add_argument(
        "--field",
        default=DEFAULT_ALARM_FIELD,
        help=f"告警类型字段名，默认 {DEFAULT_ALARM_FIELD}"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="读取输入时显示进度（当前默认已开启，仅保留兼容）"
    )
    args = parser.parse_args()

    alarm_types = set()
    processed_count = 0

    for alarm in stream_alarm_inputs(args.alarms, show_progress=True):
        processed_count += 1
        alarm_type = alarm.get(args.field)
        if alarm_type is None:
            continue
        alarm_type = str(alarm_type).strip()
        if alarm_type:
            alarm_types.add(alarm_type)

    print(f"输入路径: {args.alarms}")
    print(f"处理告警数: {processed_count}")
    print(f"告警类型数字段: {args.field}")
    print(f"去重后告警类型数: {len(alarm_types)}")
    print()

    for alarm_type in sorted(alarm_types):
        print(alarm_type)


if __name__ == "__main__":
    main()
