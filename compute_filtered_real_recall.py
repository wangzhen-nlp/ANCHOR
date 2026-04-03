import json

from argparse import ArgumentParser


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def _load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{filepath} 顶层必须是 JSON 对象")
    return data


def _build_upper_bound_index(upper_bound_result):
    index = {}
    for item in upper_bound_result.get("details", []):
        if not isinstance(item, dict):
            continue
        ticket_id = _normalize_text(item.get("ticket_id", ""))
        if not ticket_id:
            continue
        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
        associated_site_count = int(item.get("associated_site_count", 0) or 0)
        index[ticket_id] = {
            "ticket_site_count": ticket_site_count,
            "associated_site_count": associated_site_count,
            "fully_associable": ticket_site_count > 0 and associated_site_count == ticket_site_count,
        }
    return index


def compute_filtered_real_recall(real_recall_result, upper_bound_result):
    upper_bound_index = _build_upper_bound_index(upper_bound_result)

    eligible_details = []
    total_recall = 0.0

    for item in real_recall_result.get("details", []):
        if not isinstance(item, dict):
            continue

        ticket_id = _normalize_text(item.get("ticket_id", ""))
        if not ticket_id:
            continue

        upper_info = upper_bound_index.get(ticket_id)
        if not upper_info or not upper_info["fully_associable"]:
            continue

        recall = float(item.get("recall", 0.0) or 0.0)
        enriched_item = dict(item)
        enriched_item["upper_bound_ticket_site_count"] = upper_info["ticket_site_count"]
        enriched_item["upper_bound_associated_site_count"] = upper_info["associated_site_count"]
        eligible_details.append(enriched_item)
        total_recall += recall

    eligible_details.sort(
        key=lambda item: (
            -int(item.get("ticket_site_count", 0) or 0),
            item.get("ticket_id", ""),
        )
    )

    eligible_ticket_count = len(eligible_details)
    filtered_average_recall = total_recall / eligible_ticket_count if eligible_ticket_count else 0.0

    return {
        "original_ticket_count": int(real_recall_result.get("ticket_count", 0) or 0),
        "eligible_ticket_count": eligible_ticket_count,
        "filtered_average_recall": filtered_average_recall,
        "details": eligible_details,
    }


def main():
    parser = ArgumentParser(
        description="基于上限关联结果，重新计算真实召回率：只统计能把全部站点关联出来的工单"
    )
    parser.add_argument(
        "real_recall",
        help="真实召回率结果 JSON，来自 compute_ticket_site_recall.py 或 compute_group_output_ticket_recall.py",
    )
    parser.add_argument(
        "upper_bound",
        help="召回率上限结果 JSON，来自 compute_ticket_site_recall_upper_bound.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="filtered_real_recall.json",
        help="输出 JSON 文件，默认: filtered_real_recall.json",
    )

    args = parser.parse_args()

    real_recall_result = _load_json(args.real_recall)
    upper_bound_result = _load_json(args.upper_bound)
    result = compute_filtered_real_recall(real_recall_result, upper_bound_result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"原始工单数: {result['original_ticket_count']}")
    print(f"可完整关联工单数: {result['eligible_ticket_count']}")
    print(f"筛选后的真实平均召回率: {result['filtered_average_recall']:.6f}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
