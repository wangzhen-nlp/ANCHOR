import json
import os

from argparse import ArgumentParser
from collections import defaultdict

from alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from compute_group_output_ticket_recall import _extract_group_id, _extract_group_sites
from compute_ticket_site_recall import (
    _compute_site_metrics,
    _normalize_text,
    _parse_group_ids,
    _resolve_alarm_site_id,
)
from ticket_recall_v2_utils import (
    build_group_site_time_index,
    build_site_to_group_index,
    expand_groups_by_time_window,
    select_best_group_by_target_sites,
)

ALLOWED_GOLD_DOMAINS = {"RAN", "TRANSMISSION"}


def _extract_alarm_id(record):
    if not isinstance(record, dict):
        return ""
    return (
        _normalize_text(record.get("eid", ""))
        or _normalize_text(record.get("alarm_id", ""))
        or _normalize_text(record.get("告警编码ID", ""))
    )


def _extract_domain(record):
    if not isinstance(record, dict):
        return ""
    return (
        _normalize_text(record.get("domain", ""))
        or _normalize_text(record.get("Domain", ""))
        or _normalize_text(record.get("DOMAIN", ""))
    ).upper()


def _is_allowed_gold_domain(domain):
    return domain in ALLOWED_GOLD_DOMAINS


def _load_latest_group_records(group_output_input):
    latest_records = {}
    for group_record in stream_alarm_inputs(group_output_input, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id:
            continue
        latest_records[group_id] = group_record
    return latest_records


def _collect_referenced_group_ids(group_records):
    referenced_group_ids = set()
    for group_record in group_records.values():
        match_info = group_record.get("match_info", {})
        related_group_ids = []
        if isinstance(match_info, dict):
            related_group_ids = match_info.get("related_group_uuids", [])
        if not isinstance(related_group_ids, list):
            continue
        for group_id in related_group_ids:
            normalized_group_id = _normalize_text(group_id)
            if normalized_group_id:
                referenced_group_ids.add(normalized_group_id)
    return referenced_group_ids


def _build_ultimate_group_indexes(group_records, group_field):
    referenced_group_ids = _collect_referenced_group_ids(group_records)
    ultimate_group_to_sites = {}
    ultimate_group_to_alarm_groups = defaultdict(set)
    ultimate_group_to_alarm_ids = defaultdict(set)
    ultimate_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    ultimate_group_domain_allowed = {}
    alarm_group_to_ultimate_groups = defaultdict(set)
    alarm_id_to_ultimate_groups = defaultdict(set)

    for group_id, group_record in group_records.items():
        if group_id in referenced_group_ids:
            continue

        group_sites = {
            _normalize_text(site_id)
            for site_id in _extract_group_sites(group_record, group_id)
            if _normalize_text(site_id)
        }
        if group_sites:
            ultimate_group_to_sites[group_id] = group_sites

        has_invalid_domain = False
        ne_info = group_record.get("ne_info", {})
        if isinstance(ne_info, dict):
            for ne_entry in ne_info.values():
                if not isinstance(ne_entry, dict):
                    continue
                for alarm in ne_entry.get("alarm", []):
                    domain = _extract_domain(alarm)
                    if not _is_allowed_gold_domain(domain):
                        has_invalid_domain = True
                        break
                if has_invalid_domain:
                    break
        ultimate_group_domain_allowed[group_id] = not has_invalid_domain

        for symptom in group_record.get("symptoms", []):
            if not isinstance(symptom, dict):
                continue
            site_id = _normalize_text(symptom.get("node", ""))
            if site_id:
                evidence_record = dict(symptom)
                evidence_record["来源故障组UUID"] = group_id
                ultimate_group_to_site_alarms[group_id][site_id].append(evidence_record)
            alarm_id = _extract_alarm_id(symptom)
            if alarm_id:
                ultimate_group_to_alarm_ids[group_id].add(alarm_id)
                alarm_id_to_ultimate_groups[alarm_id].add(group_id)
            alarm_group_ids = _parse_group_ids(symptom.get(group_field, ""))
            for alarm_group_id in alarm_group_ids:
                ultimate_group_to_alarm_groups[group_id].add(alarm_group_id)
                alarm_group_to_ultimate_groups[alarm_group_id].add(group_id)

    return (
        ultimate_group_to_sites,
        ultimate_group_to_alarm_groups,
        alarm_group_to_ultimate_groups,
        ultimate_group_to_alarm_ids,
        ultimate_group_to_site_alarms,
        ultimate_group_domain_allowed,
        dict(alarm_id_to_ultimate_groups),
    )


def _build_alarm_group_site_index(alarm_input, ne_graph_file, group_field):
    ne_to_site = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)

    alarm_group_to_sites = defaultdict(set)
    alarm_group_to_alarm_ids = defaultdict(set)
    alarm_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    alarm_group_domain_allowed = {}
    alarm_id_to_alarm_groups = defaultdict(set)
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        if not group_ids:
            continue

        alarm_id = _extract_alarm_id(alarm)
        domain = _extract_domain(alarm)
        is_allowed_domain = _is_allowed_gold_domain(domain)
        site_id = _resolve_alarm_site_id(alarm, ne_to_site)

        for group_id in group_ids:
            if group_id not in alarm_group_domain_allowed:
                alarm_group_domain_allowed[group_id] = True
            if not is_allowed_domain:
                alarm_group_domain_allowed[group_id] = False
            if site_id:
                alarm_group_to_sites[group_id].add(site_id)
                evidence_record = dict(alarm)
                evidence_record["故障组ID"] = group_id
                alarm_group_to_site_alarms[group_id][site_id].append(evidence_record)
            if alarm_id:
                alarm_group_to_alarm_ids[group_id].add(alarm_id)
                alarm_id_to_alarm_groups[alarm_id].add(group_id)

    return (
        dict(alarm_group_to_sites),
        dict(alarm_group_to_alarm_ids),
        alarm_group_to_site_alarms,
        alarm_group_domain_allowed,
        dict(alarm_id_to_alarm_groups),
    )


def _build_potential_groups_by_alarm_id(source_to_alarm_ids, alarm_id_to_target_groups, excluded_groups_map):
    result = defaultdict(set)
    for source_id, alarm_ids in source_to_alarm_ids.items():
        if not alarm_ids:
            continue
        excluded_group_ids = set(excluded_groups_map.get(source_id, set()))
        for alarm_id in alarm_ids:
            for target_group_id in alarm_id_to_target_groups.get(alarm_id, set()):
                if target_group_id not in excluded_group_ids:
                    result[source_id].add(target_group_id)
    return result


def _build_loose_groups_by_time_window(
    gold_to_sites,
    gold_to_base_pred_groups,
    pred_group_to_sites,
    pred_group_to_site_alarms,
    window_seconds,
):
    if window_seconds <= 0:
        return {}

    site_to_groups = build_site_to_group_index(pred_group_to_sites)
    group_site_time_index = build_group_site_time_index(pred_group_to_site_alarms)
    result = {}

    for gold_id, gold_sites in gold_to_sites.items():
        base_group_ids = set(gold_to_base_pred_groups.get(gold_id, set()))
        if not base_group_ids:
            continue
        _, loose_groups = expand_groups_by_time_window(
            base_group_ids=base_group_ids,
            target_sites=set(gold_sites),
            site_to_groups=site_to_groups,
            group_site_time_index=group_site_time_index,
            window_seconds=window_seconds,
        )
        if loose_groups:
            result[gold_id] = loose_groups

    return result


def _compute_direction_metrics(
    gold_to_sites,
    gold_to_pred_groups,
    pred_group_to_sites,
    min_site_num,
    gold_domain_allowed=None,
    only_one=False,
    loose_gold_to_pred_groups=None,
    potential_gold_to_pred_groups=None,
):
    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    gold_domain_allowed = gold_domain_allowed or {}
    loose_gold_to_pred_groups = loose_gold_to_pred_groups or {}
    potential_gold_to_pred_groups = potential_gold_to_pred_groups or {}

    for gold_id in sorted(gold_to_sites.keys()):
        if gold_domain_allowed.get(gold_id) is False:
            continue
        gold_sites = set(gold_to_sites.get(gold_id, set()))
        if not gold_sites:
            continue
        if min_site_num > 0 and len(gold_sites) < min_site_num:
            continue

        base_predicted_groups = sorted(gold_to_pred_groups.get(gold_id, set()))
        loose_predicted_groups = sorted(loose_gold_to_pred_groups.get(gold_id, set()))
        potential_predicted_groups = sorted(potential_gold_to_pred_groups.get(gold_id, set()))
        predicted_groups = sorted(set(base_predicted_groups) | set(loose_predicted_groups) | set(potential_predicted_groups))
        if only_one:
            selected_predicted_group = select_best_group_by_target_sites(
                group_ids=predicted_groups,
                group_to_sites=pred_group_to_sites,
                target_sites=gold_sites,
            )
            effective_predicted_groups = [selected_predicted_group] if selected_predicted_group else []
        else:
            selected_predicted_group = ""
            effective_predicted_groups = list(predicted_groups)

        predicted_sites = set()
        for predicted_group_id in effective_predicted_groups:
            predicted_sites.update(pred_group_to_sites.get(predicted_group_id, set()))

        true_positive_sites, recall, precision, f1 = _compute_site_metrics(gold_sites, predicted_sites)

        details.append({
            "gold_id": gold_id,
            "gold_site_count": len(gold_sites),
            "gold_sites": sorted(gold_sites),
            "predicted_group_count": len(predicted_groups),
            "base_predicted_groups": base_predicted_groups,
            "loose_predicted_groups": loose_predicted_groups,
            "potential_predicted_groups": potential_predicted_groups,
            "predicted_groups": predicted_groups,
            "effective_predicted_group_count": len(effective_predicted_groups),
            "effective_predicted_groups": effective_predicted_groups,
            "selected_predicted_group": selected_predicted_group,
            "predicted_site_count": len(predicted_sites),
            "predicted_sites": sorted(predicted_sites),
            "matched_site_count": len(true_positive_sites),
            "matched_sites": sorted(true_positive_sites),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

        total_recall += recall
        total_precision += precision
        total_f1 += f1

    details.sort(
        key=lambda item: (
            -item.get("gold_site_count", 0),
            item.get("gold_id", ""),
        )
    )

    evaluated_count = len(details)
    return {
        "sample_count": evaluated_count,
        "average_recall": total_recall / evaluated_count if evaluated_count else 0.0,
        "average_precision": total_precision / evaluated_count if evaluated_count else 0.0,
        "average_f1": total_f1 / evaluated_count if evaluated_count else 0.0,
        "details": details,
    }


def compute_ultimate_group_alarm_group_metrics(
    group_output_input,
    alarm_input,
    group_field="故障组ID",
    ne_graph_file=None,
    min_site_num=0,
    only_one=False,
    loose=False,
    window_seconds=600,
    potential=False,
    output_file=None,
):
    stage_total = 3 + int(bool(loose)) + int(bool(potential))
    current_stage = 1
    print(f"阶段 {current_stage}/{stage_total}：加载 group output 最新版本并提取终极 group...")
    group_records = _load_latest_group_records(group_output_input)
    (
        ultimate_group_to_sites,
        ultimate_group_to_alarm_groups,
        alarm_group_to_ultimate_groups,
        ultimate_group_to_alarm_ids,
        ultimate_group_to_site_alarms,
        ultimate_group_domain_allowed,
        alarm_id_to_ultimate_groups,
    ) = _build_ultimate_group_indexes(
        group_records,
        group_field=group_field,
    )
    current_stage += 1

    print(f"阶段 {current_stage}/{stage_total}：从原始告警流提取告警故障组ID覆盖站点...")
    alarm_group_to_sites, alarm_group_to_alarm_ids, alarm_group_to_site_alarms, alarm_group_domain_allowed, alarm_id_to_alarm_groups = _build_alarm_group_site_index(
        alarm_input,
        ne_graph_file=ne_graph_file,
        group_field=group_field,
    )
    current_stage += 1

    ultimate_group_to_loose_alarm_groups = {}
    alarm_group_to_loose_ultimate_groups = {}
    if loose:
        print(f"阶段 {current_stage}/{stage_total}：按时间窗构造 loose 关联...")
        ultimate_group_to_loose_alarm_groups = _build_loose_groups_by_time_window(
            gold_to_sites=ultimate_group_to_sites,
            gold_to_base_pred_groups=ultimate_group_to_alarm_groups,
            pred_group_to_sites=alarm_group_to_sites,
            pred_group_to_site_alarms=alarm_group_to_site_alarms,
            window_seconds=window_seconds,
        )
        alarm_group_to_loose_ultimate_groups = _build_loose_groups_by_time_window(
            gold_to_sites=alarm_group_to_sites,
            gold_to_base_pred_groups=alarm_group_to_ultimate_groups,
            pred_group_to_sites=ultimate_group_to_sites,
            pred_group_to_site_alarms=ultimate_group_to_site_alarms,
            window_seconds=window_seconds,
        )
        current_stage += 1

    ultimate_group_to_potential_alarm_groups = {}
    alarm_group_to_potential_ultimate_groups = {}
    if potential:
        print(f"阶段 {current_stage}/{stage_total}：基于告警ID构造 potential 关联...")
        ultimate_group_to_potential_alarm_groups = _build_potential_groups_by_alarm_id(
            source_to_alarm_ids=ultimate_group_to_alarm_ids,
            alarm_id_to_target_groups=alarm_id_to_alarm_groups,
            excluded_groups_map={
                gold_id: set(ultimate_group_to_alarm_groups.get(gold_id, set())) | set(ultimate_group_to_loose_alarm_groups.get(gold_id, set()))
                for gold_id in ultimate_group_to_sites
            },
        )
        alarm_group_to_potential_ultimate_groups = _build_potential_groups_by_alarm_id(
            source_to_alarm_ids=alarm_group_to_alarm_ids,
            alarm_id_to_target_groups=alarm_id_to_ultimate_groups,
            excluded_groups_map={
                gold_id: set(alarm_group_to_ultimate_groups.get(gold_id, set())) | set(alarm_group_to_loose_ultimate_groups.get(gold_id, set()))
                for gold_id in alarm_group_to_sites
            },
        )
        current_stage += 1

    print(f"阶段 {stage_total}/{stage_total}：分别按正向/反向口径计算平均指标...")
    ultimate_as_gold = _compute_direction_metrics(
        gold_to_sites=ultimate_group_to_sites,
        gold_to_pred_groups=ultimate_group_to_alarm_groups,
        pred_group_to_sites=alarm_group_to_sites,
        min_site_num=min_site_num,
        gold_domain_allowed=ultimate_group_domain_allowed,
        only_one=only_one,
        loose_gold_to_pred_groups=ultimate_group_to_loose_alarm_groups,
        potential_gold_to_pred_groups=ultimate_group_to_potential_alarm_groups,
    )
    alarm_group_as_gold = _compute_direction_metrics(
        gold_to_sites=alarm_group_to_sites,
        gold_to_pred_groups=alarm_group_to_ultimate_groups,
        pred_group_to_sites=ultimate_group_to_sites,
        min_site_num=min_site_num,
        gold_domain_allowed=alarm_group_domain_allowed,
        only_one=only_one,
        loose_gold_to_pred_groups=alarm_group_to_loose_ultimate_groups,
        potential_gold_to_pred_groups=alarm_group_to_potential_ultimate_groups,
    )

    result = {
        "group_field": group_field,
        "min_site_num": min_site_num,
        "only_one_mode": only_one,
        "loose_mode": loose,
        "window_seconds": window_seconds,
        "potential_mode": potential,
        "ultimate_group_count": len(ultimate_group_to_sites),
        "alarm_group_count": len(alarm_group_to_sites),
        "ultimate_group_as_gold": ultimate_as_gold,
        "alarm_group_as_gold": alarm_group_as_gold,
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = ArgumentParser(
        description="基于终极 group 与告警故障组ID 的站点覆盖关系，双向计算平均召回率/准确率/F1"
    )
    parser.add_argument(
        "group_output",
        help="match_rules.py 的输出文件，支持 jsonl/zip/目录",
    )
    parser.add_argument(
        "alarms",
        help="原始告警输入，支持 jsonl/csv/zip/目录",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警/症状中的故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ne-graph",
        default="ne_graph.json",
        help="用于通过告警源回填站点的 ne_graph 文件，默认: ne_graph.json",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="仅统计 gold label 站点数 >= 该值的样本；默认: 0（不过滤）",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖当前 gold 站点最多的单个预测 group，用它的站点计算指标",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许在当前 gold 站点范围内，按时间窗把其它预测 group 做 loose 扩张",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=600,
        help="loose 模式使用的前后对称时间窗，单位秒，默认: 600",
    )
    parser.add_argument(
        "--potential",
        action="store_true",
        help="允许根据告警ID命中关系，把另一侧包含这些告警的额外 group 作为 potential 预测结果并入",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ultimate_group_alarm_group_metrics.json",
        help="输出 JSON 文件，默认: ultimate_group_alarm_group_metrics.json",
    )

    args = parser.parse_args()

    result = compute_ultimate_group_alarm_group_metrics(
        group_output_input=args.group_output,
        alarm_input=args.alarms,
        group_field=args.group_field,
        ne_graph_file=args.ne_graph,
        min_site_num=args.min_site_num,
        only_one=args.only_one,
        loose=args.loose,
        window_seconds=args.window_seconds,
        potential=args.potential,
        output_file=args.output,
    )

    print("【终极 group 作为 gold】")
    print(f"样本数: {result['ultimate_group_as_gold']['sample_count']}")
    print(f"平均召回率: {result['ultimate_group_as_gold']['average_recall']:.6f}")
    print(f"平均准确率: {result['ultimate_group_as_gold']['average_precision']:.6f}")
    print(f"平均F1: {result['ultimate_group_as_gold']['average_f1']:.6f}")

    print("【告警故障组ID 作为 gold】")
    print(f"样本数: {result['alarm_group_as_gold']['sample_count']}")
    print(f"平均召回率: {result['alarm_group_as_gold']['average_recall']:.6f}")
    print(f"平均准确率: {result['alarm_group_as_gold']['average_precision']:.6f}")
    print(f"平均F1: {result['alarm_group_as_gold']['average_f1']:.6f}")
    print(f"结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
