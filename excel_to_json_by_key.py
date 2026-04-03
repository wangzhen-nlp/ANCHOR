import argparse
import json
import math
from typing import Any, Dict

import pandas as pd


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(column): _normalize_cell(value)
        for column, value in row.items()
    }


def convert_excel_to_json_by_key(
    input_file: str,
    key_field: str,
    output_file: str,
    sheet_name: str | int | None = 0,
) -> Dict[str, Dict[str, Any]]:
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    if key_field not in df.columns:
        raise ValueError(f"Excel 中未找到作为 key 的字段: {key_field}")

    result: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        record = _row_to_record(row)
        raw_key = record.get(key_field)
        if raw_key is None:
            continue

        key = str(raw_key).strip()
        if not key:
            continue

        result[key] = record

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="将 Excel 转成按指定字段为 key 的 JSON")
    parser.add_argument("input_file", help="输入 Excel 文件")
    parser.add_argument("key_field", help="作为 JSON key 的字段名")
    parser.add_argument(
        "-o",
        "--output",
        default="excel_by_key.json",
        help="输出 JSON 文件，默认: excel_by_key.json",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="sheet 名或 sheet 索引，默认第一个 sheet",
    )
    args = parser.parse_args()

    sheet_name: str | int | None
    if isinstance(args.sheet_name, str) and args.sheet_name.isdigit():
        sheet_name = int(args.sheet_name)
    else:
        sheet_name = args.sheet_name

    result = convert_excel_to_json_by_key(
        args.input_file,
        args.key_field,
        args.output,
        sheet_name=sheet_name,
    )

    print(f"转换完成: {len(result)} 条，输出文件: {args.output}")


if __name__ == "__main__":
    main()
