from argparse import ArgumentParser

from alarm_inputs import stream_alarm_inputs


def main():
    parser = ArgumentParser()
    parser.add_argument("inputs", type=str, help="输入文件、目录或 zip")
    parser.add_argument("--field", type=str, default="id", help="要统计的字段名，默认 id")
    args = parser.parse_args()

    field_name = args.field
    unique_ids = set()
    total_records = 0
    missing_field_records = 0

    for item in stream_alarm_inputs(args.inputs, show_progress=True):
        total_records += 1
        value = item.get(field_name)
        if value is None:
            missing_field_records += 1
            continue

        value_str = str(value).strip()
        if not value_str:
            missing_field_records += 1
            continue

        unique_ids.add(value_str)

    print(f"字段名: {field_name}")
    print(f"总记录数: {total_records}")
    print(f"缺少该字段的记录数: {missing_field_records}")
    print(f"不同 {field_name} 的数量: {len(unique_ids)}")


if __name__ == "__main__":
    main()
