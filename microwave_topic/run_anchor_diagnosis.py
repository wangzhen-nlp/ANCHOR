#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑一遍 microwave_anchor_diagnosis：读入拓扑 JSON，打印根因诊断结果。

用法：
    python microwave_topic/run_anchor_diagnosis.py                # 默认读同目录 test.json
    python microwave_topic/run_anchor_diagnosis.py path/to/x.json # 指定输入
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from microwave_anchor_diagnosis import diagnose_root_cause_devices


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("test.json")
    if not input_path.exists():
        sys.exit(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as fr:
        input_json = json.load(fr)

    root_cause_resources, ne_to_alarm_objs = diagnose_root_cause_devices(input_json)

    print(f"输入: {input_path}")
    print(f"告警总数: {len(input_json.get('alarms') or [])}")

    print(f"\n== 根因设备 root_cause_resources ({len(root_cause_resources)}) ==")
    print(json.dumps(root_cause_resources, ensure_ascii=False, indent=2))

    print(f"\n== 告警归并 ne_to_alarm_objs (网元 {len(ne_to_alarm_objs)} 个, "
          f"告警 {sum(len(v) for v in ne_to_alarm_objs.values())} 条) ==")
    for ne, objs in ne_to_alarm_objs.items():
        print(f"  {ne}: {len(objs)} 条")


if __name__ == "__main__":
    main()
