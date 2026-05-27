#!/usr/bin/env python3
"""
将 alarm_types.py 中定义的所有关键告警（CRITICAL_ALARMS）
导出为形如 '("告警标题" 等于 "xxx" 或 "告警标题" 等于 "yyy")' 的规则字符串。
"""

import argparse
import os
import sys

# 确保可以从项目根目录正确导入 alarm_tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alarm_tools.alarm_types import CRITICAL_ALARMS

def generate_rule_string() -> str:
    # 排序以保证每次生成的字符串顺序一致
    sorted_alarms = sorted(CRITICAL_ALARMS)
    
    conditions = [f'"alarmname" = "{alarm}"' for alarm in sorted_alarms]
    rule_body = " or ".join(conditions)
    return f"'({rule_body})'"

def main():
    parser = argparse.ArgumentParser(description="导出关键告警规则字符串")
    parser.add_argument("-o", "--output", help="输出文件路径。如果不指定，则直接打印到控制台。")
    args = parser.parse_args()

    rule_string = generate_rule_string()
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rule_string + "\n")
        print(f"规则已成功导出到: {args.output}")
    else:
        print(rule_string)

if __name__ == "__main__":
    main()