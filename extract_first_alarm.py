import argparse
import json

from alarm_inputs import stream_alarm_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    args = parser.parse_args()

    first_alarm = None
    for alarm in stream_alarm_inputs(args.alarms, show_progress=True):
        first_alarm = alarm
        break

    if first_alarm is None:
        print("{}")
        return

    print(json.dumps(first_alarm, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
