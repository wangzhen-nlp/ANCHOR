import json
from argparse import ArgumentParser

from alarm_inputs import stream_alarm_inputs


DEFAULT_FIELD = "告警标题"


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def _iter_field_values(value):
    if value is None:
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_text(item)
            if normalized:
                yield normalized
        return

    normalized = _normalize_text(value)
    if normalized:
        yield normalized


def main():
    parser = ArgumentParser(description="从 alarms 输入中提取指定字段的去重 value 集合并保存")
    parser.add_argument("alarms", help="告警输入：支持 jsonl / csv / zip / 目录")
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help=f"要提取的字段名，默认: {DEFAULT_FIELD}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="alarm_field_values.json",
        help="输出 JSON 文件，默认: alarm_field_values.json",
    )

    args = parser.parse_args()

    values = set()
    processed_count = 0
    matched_count = 0

    for alarm in stream_alarm_inputs(args.alarms, show_progress=True):
        processed_count += 1
        field_value = alarm.get(args.field)
        current_values = list(_iter_field_values(field_value))
        if not current_values:
            continue
        matched_count += 1
        values.update(current_values)

    sorted_values = sorted(values)
    result = {
        "input": args.alarms,
        "field": args.field,
        "processed_alarm_count": processed_count,
        "matched_alarm_count": matched_count,
        "unique_value_count": len(sorted_values),
        "values": sorted_values,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"输入路径: {args.alarms}")
    print(f"字段名: {args.field}")
    print(f"处理告警数: {processed_count}")
    print(f"命中字段的告警数: {matched_count}")
    print(f"去重后 value 数: {len(sorted_values)}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
