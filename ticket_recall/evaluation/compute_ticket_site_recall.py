import json
import os

from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

from alarm_tools.alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from topology_resources import NE_GRAPH_JSON, resource_display


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_site_list(values):
    seen = set()
    normalized = []
    for value in values:
        site_id = _normalize_text(value)
        if not site_id or site_id in seen:
            continue
        seen.add(site_id)
        normalized.append(site_id)
    return normalized


def _parse_group_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_parse_group_ids(item))
        return _normalize_site_list(result)

    text = _normalize_text(value)
    if not text:
        return []

    # 兼容故障组字段直接存成 JSON 字符串的情况。
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            if isinstance(parsed, dict):
                parsed = parsed.values()
            return _parse_group_ids(parsed)

    parts = [text]
    for delimiter in [",", "，", ";", "；", "|"]:
        if delimiter in text:
            parts = [segment for segment in text.replace("，", ",").replace(";", ",").replace("；", ",").replace("|", ",").split(",")]
            break

    return _normalize_site_list(parts)


def _load_ticket_sites(ticket_sites_file):
    with open(ticket_sites_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ticket_sites = {}
    for ticket_id, sites in data.items():
        normalized_ticket_id = _normalize_text(ticket_id)
        if not normalized_ticket_id:
            continue
        if not isinstance(sites, list):
            continue
        normalized_sites = _normalize_site_list(sites)
        if not normalized_sites:
            continue
        ticket_sites[normalized_ticket_id] = normalized_sites
    return ticket_sites


def _resolve_alarm_site_id(alarm, ne_to_site):
    # 优先使用告警自带站点；没有时再尝试通过告警源映射到站点。
    site_id = _normalize_text(alarm.get("站点ID", ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get("告警源", ""))
    if not alarm_source:
        return ""

    return _normalize_text(ne_to_site.get(alarm_source, ""))


def _build_ticket_sites_from_alarms(alarm_input, ticket_field, ne_graph_file=None):
    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    ticket_sites = defaultdict(set)
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if not ticket_id:
            continue

        site_id = _resolve_alarm_site_id(alarm, ne_to_site)
        if site_id:
            ticket_sites[ticket_id].add(site_id)

    return {
        ticket_id: sorted(site_ids)
        for ticket_id, site_ids in ticket_sites.items()
        if site_ids
    }


def _build_ticket_group_indexes(alarm_input, ticket_field, group_field, ne_to_site):
    ticket_to_groups = defaultdict(set)
    group_to_sites = defaultdict(set)
    ticket_alarm_counts = defaultdict(int)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id:
            ticket_alarm_counts[ticket_id] += 1

        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        # 第一层索引：工单通过告警字段拿到关联故障组。
        if ticket_id and group_ids:
            ticket_to_groups[ticket_id].update(group_ids)

        site_id = _resolve_alarm_site_id(alarm, ne_to_site)
        # 第二层索引：故障组再通过关联告警反推出覆盖到的站点集合。
        if site_id and group_ids:
            for group_id in group_ids:
                group_to_sites[group_id].add(site_id)

    return ticket_to_groups, group_to_sites, ticket_alarm_counts


def _compute_site_metrics(target_sites, predicted_sites):
    target_site_set = set(target_sites)
    predicted_site_set = set(predicted_sites)
    true_positive_sites = target_site_set & predicted_site_set

    recall = len(true_positive_sites) / len(target_site_set) if target_site_set else 0.0
    precision = len(true_positive_sites) / len(predicted_site_set) if predicted_site_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return true_positive_sites, recall, precision, f1


def _compute_ticket_recalls(ticket_sites, ticket_to_groups, group_to_sites, ticket_alarm_counts):
    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    evaluated_count = 0

    for ticket_id in sorted(ticket_sites.keys()):
        # 只统计在告警数据里实际出现过的工单，避免把完全无告警的工单按 0 计入平均。
        if ticket_alarm_counts.get(ticket_id, 0) <= 0:
            continue

        target_sites = set(ticket_sites[ticket_id])
        if not target_sites:
            continue

        fault_groups = sorted(ticket_to_groups.get(ticket_id, set()))
        predicted_sites = set()
        for group_id in fault_groups:
            predicted_sites.update(group_to_sites.get(group_id, set()))

        # 召回率 / 准确率口径：预测站点来自关联故障组覆盖到的全部站点。
        true_positive_sites, recall, precision, f1 = _compute_site_metrics(target_sites, predicted_sites)
        recalled_target_sites = sorted(true_positive_sites)

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_alarm_count": ticket_alarm_counts.get(ticket_id, 0),
            "fault_group_count": len(fault_groups),
            "fault_groups": fault_groups,
            "recalled_site_count": len(recalled_target_sites),
            "recalled_sites": recalled_target_sites,
            "group_site_count": len(predicted_sites),
            "group_sites": sorted(predicted_sites),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

        total_recall += recall
        total_precision += precision
        total_f1 += f1
        evaluated_count += 1

    average_recall = total_recall / evaluated_count if evaluated_count else 0.0
    average_precision = total_precision / evaluated_count if evaluated_count else 0.0
    average_f1 = total_f1 / evaluated_count if evaluated_count else 0.0
    details.sort(
        key=lambda item: (
            -item.get("ticket_site_count", 0),
            item.get("ticket_id", ""),
        )
    )
    return details, average_recall, average_precision, average_f1, evaluated_count


def main():
    parser = ArgumentParser(description="基于工单-站点映射和告警中的故障组字段计算工单站点召回率")
    parser.add_argument("alarms", help="告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致")
    parser.add_argument(
        "--ticket-sites",
        help="工单站点映射 JSON，格式为 {工单号: [站点列表]}；不提供时会退化为从 alarms 中回推工单站点",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="告警中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警中的故障组字段名，默认: 故障ID",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于通过告警源回填 site_id 的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ticket_site_recall.json",
        help="输出明细 JSON 文件，默认: ticket_site_recall.json",
    )

    args = parser.parse_args()

    if args.ticket_sites:
        ticket_sites = _load_ticket_sites(args.ticket_sites)
        ticket_site_source = "ticket_sites"
    else:
        ticket_sites = _build_ticket_sites_from_alarms(args.alarms, args.ticket_field, args.ne_graph)
        ticket_site_source = "alarms"

    if not ticket_sites:
        print("❌ 工单站点映射为空，无法计算召回率")
        return

    ne_to_site = {}
    if args.ne_graph and os.path.exists(args.ne_graph):
        ne_to_site = build_ne_to_site_map(args.ne_graph)

    ticket_to_groups, group_to_sites, ticket_alarm_counts = _build_ticket_group_indexes(
        args.alarms,
        args.ticket_field,
        args.group_field,
        ne_to_site,
    )

    details, average_recall, average_precision, average_f1, evaluated_count = _compute_ticket_recalls(
        ticket_sites,
        ticket_to_groups,
        group_to_sites,
        ticket_alarm_counts,
    )

    result = {
        "ticket_count": evaluated_count,
        "average_recall": average_recall,
        "average_precision": average_precision,
        "average_f1": average_f1,
        "ticket_site_source": ticket_site_source,
        "details": details,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"工单数: {evaluated_count}")
    print(f"平均召回率: {average_recall:.6f}")
    print(f"平均准确率: {average_precision:.6f}")
    print(f"平均F1: {average_f1:.6f}")
    print(f"工单站点来源: {ticket_site_source}")
    print(f"明细已输出到: {args.output}")


if __name__ == "__main__":
    main()
