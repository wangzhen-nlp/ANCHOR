import json
import os

from argparse import ArgumentParser
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

from alarm_inputs import build_ne_to_site_map, stream_alarm_inputs


EXCLUDED_ALARM_TITLES = {"FAN FAIL"}


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def _normalize_site_list(values):
    seen = set()
    result = []
    for value in values:
        site_id = _normalize_text(value)
        if not site_id or site_id in seen:
            continue
        seen.add(site_id)
        result.append(site_id)
    return result


def _load_ticket_sites(ticket_sites_file):
    with open(ticket_sites_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("工单站点映射 JSON 顶层必须是对象")

    ticket_sites = {}
    for ticket_id, sites in data.items():
        normalized_ticket_id = _normalize_text(ticket_id)
        if not normalized_ticket_id or not isinstance(sites, list):
            continue
        normalized_sites = _normalize_site_list(sites)
        if normalized_sites:
            ticket_sites[normalized_ticket_id] = normalized_sites
    return ticket_sites


def _build_ticket_site_sets(ticket_sites):
    return {
        ticket_id: set(site_list)
        for ticket_id, site_list in ticket_sites.items()
    }


def _parse_time_to_ts(value):
    text = _normalize_text(value)
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return None


def _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field):
    site_id = _normalize_text(alarm.get(site_field, ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get(source_field, ""))
    if not alarm_source:
        return ""

    return _normalize_text(ne_to_site.get(alarm_source, ""))


def _should_skip_alarm(alarm):
    alarm_title = _normalize_text(alarm.get("告警标题", ""))
    return alarm_title in EXCLUDED_ALARM_TITLES


def _build_alarm_evidence_record(alarm, resolved_site_id, ticket_field, source_field):
    record = {
        "告警编码ID": _normalize_text(alarm.get("告警编码ID", "")),
        "告警标题": _normalize_text(alarm.get("告警标题", "")),
        "工单号": _normalize_text(alarm.get(ticket_field, "")),
        "站点ID": _normalize_text(alarm.get("站点ID", "")),
        "关联站点ID": resolved_site_id,
        "告警源": _normalize_text(alarm.get(source_field, "")),
        "告警首次发生时间": _normalize_text(alarm.get("告警首次发生时间", "")),
        "告警最后发生时间": _normalize_text(alarm.get("告警最后发生时间", "")),
        "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
    }
    return {key: value for key, value in record.items() if value}


def _build_ticket_time_windows(ticket_alarm_times, window_seconds):
    ticket_windows = {}
    for ticket_id, raw_times in ticket_alarm_times.items():
        if not raw_times:
            continue

        times = sorted(set(raw_times))
        merged = []
        for ts in times:
            start = ts - window_seconds
            end = ts + window_seconds
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        starts = [item[0] for item in merged]
        ends = [item[1] for item in merged]
        ticket_windows[ticket_id] = (starts, ends)
    return ticket_windows


def _timestamp_in_windows(ts, windows):
    if ts is None or windows is None:
        return False

    starts, ends = windows
    idx = bisect_right(starts, ts) - 1
    return idx >= 0 and ts <= ends[idx]


def _collect_ticket_alarm_stats(
    alarm_input,
    ticket_sites,
    ticket_site_sets,
    ticket_field,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    target_ticket_ids = set(ticket_sites.keys())
    ticket_alarm_counts = defaultdict(int)
    ticket_alarm_times = defaultdict(list)
    direct_sites = defaultdict(set)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        if ticket_id not in target_ticket_ids:
            continue

        ticket_alarm_counts[ticket_id] += 1

        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        if ts is not None:
            ticket_alarm_times[ticket_id].append(ts)

        site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if site_id and site_id in ticket_site_sets[ticket_id]:
            direct_sites[ticket_id].add(site_id)

    return ticket_alarm_counts, ticket_alarm_times, direct_sites


def _build_site_pending_tickets(ticket_sites, ticket_site_sets, direct_sites, valid_ticket_ids):
    site_pending_tickets = defaultdict(set)
    associated_sites = {}

    for ticket_id, target_site_list in ticket_sites.items():
        if ticket_id not in valid_ticket_ids:
            continue
        target_sites = ticket_site_sets[ticket_id]
        current_direct_sites = set(direct_sites.get(ticket_id, set())) & target_sites
        associated_sites[ticket_id] = set(current_direct_sites)
        for site_id in target_sites - current_direct_sites:
            site_pending_tickets[site_id].add(ticket_id)

    return associated_sites, site_pending_tickets


def _build_site_window_bounds(site_pending_tickets, ticket_windows):
    site_window_bounds = {}
    for site_id, pending_tickets in site_pending_tickets.items():
        min_start = None
        max_end = None
        for ticket_id in pending_tickets:
            windows = ticket_windows.get(ticket_id)
            if not windows:
                continue
            starts, ends = windows
            if not starts or not ends:
                continue
            site_min = starts[0]
            site_max = ends[-1]
            if min_start is None or site_min < min_start:
                min_start = site_min
            if max_end is None or site_max > max_end:
                max_end = site_max
        if min_start is not None and max_end is not None:
            site_window_bounds[site_id] = (min_start, max_end)
    return site_window_bounds


def _associate_missing_sites_by_time_window(
    alarm_input,
    ticket_windows,
    associated_sites,
    site_pending_tickets,
    site_window_bounds,
    inferred_sites,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    if not site_pending_tickets:
        return

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if not site_id:
            continue

        pending_tickets = site_pending_tickets.get(site_id)
        if not pending_tickets:
            continue

        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        if ts is None:
            continue

        bounds = site_window_bounds.get(site_id)
        if bounds is None:
            continue
        if ts < bounds[0] or ts > bounds[1]:
            continue

        matched_tickets = []
        for ticket_id in tuple(pending_tickets):
            if _timestamp_in_windows(ts, ticket_windows.get(ticket_id)):
                associated_sites[ticket_id].add(site_id)
                inferred_sites[ticket_id].add(site_id)
                matched_tickets.append(ticket_id)

        if not matched_tickets:
            continue

        for ticket_id in matched_tickets:
            pending_tickets.discard(ticket_id)
        if not pending_tickets:
            site_pending_tickets.pop(site_id, None)
            site_window_bounds.pop(site_id, None)


def _compute_upper_bound_recalls(ticket_sites, ticket_site_sets, ticket_alarm_counts, direct_sites, inferred_sites, associated_sites):
    details = []
    total_recall = 0.0
    ticket_count = 0

    for ticket_id in sorted(ticket_sites.keys()):
        if ticket_alarm_counts.get(ticket_id, 0) <= 0:
            continue

        target_sites = ticket_site_sets[ticket_id]
        direct_site_set = set(direct_sites.get(ticket_id, set())) & target_sites
        inferred_site_set = set(inferred_sites.get(ticket_id, set())) & target_sites
        associated_site_set = set(associated_sites.get(ticket_id, set())) & target_sites

        if len(associated_site_set) <= 1:
            recall = 0.0
        else:
            recall = len(associated_site_set) / len(target_sites) if target_sites else 0.0

        details.append({
            "ticket_id": ticket_id,
            "ticket_site_count": len(target_sites),
            "ticket_sites": sorted(target_sites),
            "ticket_alarm_count": ticket_alarm_counts.get(ticket_id, 0),
            "direct_site_count": len(direct_site_set),
            "direct_sites": sorted(direct_site_set),
            "inferred_site_count": len(inferred_site_set),
            "inferred_sites": sorted(inferred_site_set),
            "associated_site_count": len(associated_site_set),
            "associated_sites": sorted(associated_site_set),
            "recall_upper_bound": recall,
        })
        total_recall += recall
        ticket_count += 1

    details.sort(
        key=lambda item: (
            -item.get("ticket_site_count", 0),
            item.get("ticket_id", ""),
        )
    )
    average_recall = total_recall / ticket_count if ticket_count else 0.0
    return details, average_recall


def _collect_association_evidence(
    alarm_input,
    ticket_sites,
    direct_sites,
    inferred_sites,
    ticket_windows,
    ticket_field,
    site_field,
    source_field,
    time_field,
    ne_to_site,
):
    direct_site_tickets = defaultdict(set)
    inferred_site_tickets = defaultdict(set)
    evidence = {}

    for ticket_id in ticket_sites:
        evidence[ticket_id] = {
            "direct_site_alarms": defaultdict(list),
            "inferred_site_alarms": defaultdict(list),
        }

    for ticket_id, site_ids in direct_sites.items():
        for site_id in site_ids:
            direct_site_tickets[site_id].add(ticket_id)

    for ticket_id, site_ids in inferred_sites.items():
        for site_id in site_ids:
            inferred_site_tickets[site_id].add(ticket_id)

    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if _should_skip_alarm(alarm):
            continue

        resolved_site_id = _resolve_alarm_site_id(alarm, ne_to_site, site_field, source_field)
        if not resolved_site_id:
            continue

        normalized_ticket_id = _normalize_text(alarm.get(ticket_field, ""))
        ts = _parse_time_to_ts(alarm.get(time_field, ""))
        evidence_record = None

        direct_ticket_candidates = direct_site_tickets.get(resolved_site_id)
        if direct_ticket_candidates and normalized_ticket_id:
            for ticket_id in direct_ticket_candidates:
                if normalized_ticket_id != ticket_id:
                    continue
                if evidence_record is None:
                    evidence_record = _build_alarm_evidence_record(
                        alarm,
                        resolved_site_id,
                        ticket_field,
                        source_field,
                    )
                evidence[ticket_id]["direct_site_alarms"][resolved_site_id].append(evidence_record)

        inferred_ticket_candidates = inferred_site_tickets.get(resolved_site_id)
        if inferred_ticket_candidates and ts is not None:
            for ticket_id in inferred_ticket_candidates:
                if not _timestamp_in_windows(ts, ticket_windows.get(ticket_id)):
                    continue
                if evidence_record is None:
                    evidence_record = _build_alarm_evidence_record(
                        alarm,
                        resolved_site_id,
                        ticket_field,
                        source_field,
                    )
                evidence[ticket_id]["inferred_site_alarms"][resolved_site_id].append(evidence_record)

    normalized_evidence = {}
    for ticket_id, payload in evidence.items():
        normalized_evidence[ticket_id] = {
            "direct_site_alarms": {
                site_id: alarms
                for site_id, alarms in sorted(payload["direct_site_alarms"].items())
            },
            "inferred_site_alarms": {
                site_id: alarms
                for site_id, alarms in sorted(payload["inferred_site_alarms"].items())
            },
        }
    return normalized_evidence


def main():
    parser = ArgumentParser(
        description="基于工单标注告警和站点时间窗，估算工单站点召回率的理论上限"
    )
    parser.add_argument(
        "alarms",
        help="告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致",
    )
    parser.add_argument(
        "--ticket-sites",
        required=True,
        help="工单站点映射 JSON，格式为 {工单号: [站点列表]}",
    )
    parser.add_argument(
        "--ticket-field",
        default="工单号",
        help="告警中的工单字段名，默认: 工单号",
    )
    parser.add_argument(
        "--site-field",
        default="站点ID",
        help="告警中的站点字段名，默认: 站点ID",
    )
    parser.add_argument(
        "--source-field",
        default="告警源",
        help="告警中的设备/告警源字段名，默认: 告警源",
    )
    parser.add_argument(
        "--time-field",
        default="告警首次发生时间",
        help="用于做时间窗关联的告警时间字段，默认: 告警首次发生时间",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=600,
        help="缺失站点告警与工单告警允许的时间窗，单位秒，默认: 600",
    )
    parser.add_argument(
        "--ne-graph",
        default="ne_graph.json",
        help="用于通过告警源回填站点ID的 ne_graph 文件，默认: ne_graph.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ticket_site_recall_upper_bound.json",
        help="输出 JSON 文件，默认: ticket_site_recall_upper_bound.json",
    )

    args = parser.parse_args()

    ticket_sites = _load_ticket_sites(args.ticket_sites)
    if not ticket_sites:
        print("❌ 工单站点映射为空，无法计算召回率上限")
        return
    ticket_site_sets = _build_ticket_site_sets(ticket_sites)

    ne_to_site = {}
    if args.ne_graph and os.path.exists(args.ne_graph):
        ne_to_site = build_ne_to_site_map(args.ne_graph)

    print("阶段 1/2：提取每个工单的标注告警时间和直接命中站点...")
    ticket_alarm_counts, ticket_alarm_times, direct_sites = _collect_ticket_alarm_stats(
        alarm_input=args.alarms,
        ticket_sites=ticket_sites,
        ticket_site_sets=ticket_site_sets,
        ticket_field=args.ticket_field,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )
    valid_ticket_ids = {
        ticket_id
        for ticket_id, count in ticket_alarm_counts.items()
        if count > 0
    }

    ticket_windows = _build_ticket_time_windows(ticket_alarm_times, args.window_seconds)
    associated_sites, site_pending_tickets = _build_site_pending_tickets(
        ticket_sites,
        ticket_site_sets,
        direct_sites,
        valid_ticket_ids,
    )
    site_window_bounds = _build_site_window_bounds(site_pending_tickets, ticket_windows)
    inferred_sites = defaultdict(set)

    print("阶段 2/2：用站点告警时间窗补齐缺失站点...")
    _associate_missing_sites_by_time_window(
        alarm_input=args.alarms,
        ticket_windows=ticket_windows,
        associated_sites=associated_sites,
        site_pending_tickets=site_pending_tickets,
        site_window_bounds=site_window_bounds,
        inferred_sites=inferred_sites,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )

    print("阶段 3/3：收集工单关联站点的告警证据...")
    association_evidence = _collect_association_evidence(
        alarm_input=args.alarms,
        ticket_sites=ticket_sites,
        direct_sites=direct_sites,
        inferred_sites=inferred_sites,
        ticket_windows=ticket_windows,
        ticket_field=args.ticket_field,
        site_field=args.site_field,
        source_field=args.source_field,
        time_field=args.time_field,
        ne_to_site=ne_to_site,
    )

    details, average_recall = _compute_upper_bound_recalls(
        ticket_sites=ticket_sites,
        ticket_site_sets=ticket_site_sets,
        ticket_alarm_counts=ticket_alarm_counts,
        direct_sites=direct_sites,
        inferred_sites=inferred_sites,
        associated_sites=associated_sites,
    )

    for item in details:
        item["evidence"] = association_evidence.get(item["ticket_id"], {
            "direct_site_alarms": {},
            "inferred_site_alarms": {},
        })

    result = {
        "ticket_count": len(details),
        "window_seconds": args.window_seconds,
        "time_field": args.time_field,
        "site_field": args.site_field,
        "source_field": args.source_field,
        "average_recall_upper_bound": average_recall,
        "details": details,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"工单数: {len(details)}")
    print(f"平均召回率上限: {average_recall:.6f}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
