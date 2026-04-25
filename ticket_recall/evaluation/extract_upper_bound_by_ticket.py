#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 compute_ticket_site_recall_upper_bound.py 的结果 JSON 中按工单号提取单条记录。

输出保持 upper-bound 结果文件格式，只是 details 中仅保留目标工单的一项。

用法:
    python ticket_recall/evaluation/extract_upper_bound_by_ticket.py \
        upper_bound.json TICKET_ID -o selected_upper_bound.json
"""

import argparse
import copy
import json
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)


def _normalize_text(value):
    return str(value or "").strip()


def _load_upper_bound_result(input_file):
    path = Path(input_file)
    if not path.exists():
        raise SystemExit(f"输入文件不存在: {input_file}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("输入文件格式错误: JSON 顶层必须是对象")
    if not isinstance(data.get("details", []), list):
        raise SystemExit("输入文件格式错误: details 必须是数组")
    return data


def _find_detail_by_ticket_id(details, ticket_id):
    normalized_ticket_id = _normalize_text(ticket_id)
    for item in details:
        if not isinstance(item, dict):
            continue
        if _normalize_text(item.get("ticket_id", "")) == normalized_ticket_id:
            return copy.deepcopy(item)
    return None


def _float_field(item, field_name):
    try:
        return float(item.get(field_name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_single_ticket_result(source_data, detail):
    result = copy.deepcopy(source_data)
    result["ticket_count"] = 1
    result["average_recall_upper_bound"] = _float_field(detail, "recall_upper_bound")
    result["average_precision_upper_bound"] = _float_field(detail, "precision_upper_bound")
    result["average_f1_upper_bound"] = _float_field(detail, "f1_upper_bound")
    result["details"] = [detail]
    return result


def main():
    parser = argparse.ArgumentParser(description="按工单号从 upper-bound 结果 JSON 中提取单条 detail")
    parser.add_argument("input", help="compute_ticket_site_recall_upper_bound.py 输出 JSON")
    parser.add_argument("ticket_id", help="目标工单号")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 文件")
    args = parser.parse_args()

    source_data = _load_upper_bound_result(args.input)
    detail = _find_detail_by_ticket_id(source_data.get("details", []), args.ticket_id)
    if detail is None:
        raise SystemExit(f"未找到工单号: {args.ticket_id}")

    result = _build_single_ticket_result(source_data, detail)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"已写出: {args.output}")


if __name__ == "__main__":
    main()
