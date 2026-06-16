#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare MHP visual fault groups with alarm-native group ids.

The MHP online stream writes match-rules-compatible visual JSONL. This wrapper
compares two cluster labels directly from its real visual symptoms:

  - MHP visual groups as one clustering
  - the alarms' native group id field (default: 故障组ID) as the other clustering

Virtual/imputed symptoms are excluded from the metric indexes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_flow_mhp.missing_chain_sampler import MHP_VIRTUAL_RULE
from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.recall_common import (
    _extract_group_id,
    _normalize_text,
    _parse_group_ids,
)
from ticket_recall.evaluation.compute_ultimate_group_alarm_group_metrics import (
    _build_case_details_for_direction,
    _build_loose_groups_by_time_window,
    _collect_referenced_group_ids,
    _compute_direction_metrics,
    _extract_alarm_id,
    _is_offline_alarm_record,
    _normalize_domain_arg,
    _resolve_record_domain,
)
from ticket_recall.ticket_recall_utils import (
    build_ne_to_domain_map,
    build_site_has_domain_map,
    build_unrecalled_visualization_cases,
    load_ne_graph_data,
    write_jsonl_records,
)


def _is_disabled(value) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "none", "off"}


def _derive_case_output_path(output_file, suffix):
    base, _ext = os.path.splitext(output_file)
    return f"{base}.{suffix}.cases.jsonl"


def _rename_result(raw_result, *, visual_output, alarms):
    result = dict(raw_result)
    result["method"] = "alarm_flow_mhp_visual_alarm_group_comparison"
    result["mhp_visual_output"] = os.path.abspath(visual_output)
    if alarms:
        result["alarm_input"] = os.path.abspath(alarms)
    return result


def _is_real_symptom(symptom):
    if not isinstance(symptom, dict):
        return False
    if bool(symptom.get("virtual")) or bool(symptom.get("latent")):
        return False
    if _normalize_text(symptom.get("matched_rule", "")) == MHP_VIRTUAL_RULE:
        return False
    return True


def _symptom_site(symptom):
    return _normalize_text(symptom.get("node", "")) or _normalize_text(symptom.get("site_id", ""))


def _load_latest_visual_records(visual_output):
    latest_records = {}
    for group_record in stream_alarm_inputs(visual_output, show_progress=True):
        group_id = _extract_group_id(group_record)
        if not group_id:
            continue
        latest_records[group_id] = group_record
    return latest_records


def _build_visual_indexes(visual_output, *, group_field, ne_to_domain=None):
    group_records = _load_latest_visual_records(visual_output)
    referenced_group_ids = _collect_referenced_group_ids(group_records)

    mhp_group_to_sites = {}
    mhp_group_to_alarm_groups = defaultdict(set)
    mhp_group_to_alarm_ids = defaultdict(set)
    mhp_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    mhp_group_alarm_domains = defaultdict(set)
    mhp_group_has_offline = defaultdict(bool)

    alarm_group_to_sites = defaultdict(set)
    alarm_group_to_mhp_groups = defaultdict(set)
    alarm_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    alarm_group_alarm_domains = defaultdict(set)
    alarm_group_has_offline = defaultdict(bool)

    real_symptom_count = 0
    skipped_virtual_symptom_count = 0
    symptom_with_group_id_count = 0

    for group_id, group_record in group_records.items():
        if group_id in referenced_group_ids:
            continue

        group_sites = set()
        for symptom in group_record.get("symptoms", []):
            if not isinstance(symptom, dict):
                continue
            if not _is_real_symptom(symptom):
                skipped_virtual_symptom_count += 1
                continue

            real_symptom_count += 1
            site_id = _symptom_site(symptom)
            if site_id:
                group_sites.add(site_id)
                evidence_record = dict(symptom)
                evidence_record["来源故障组UUID"] = group_id
                mhp_group_to_site_alarms[group_id][site_id].append(evidence_record)

            domain = _resolve_record_domain(symptom, ne_to_domain)
            if domain:
                mhp_group_alarm_domains[group_id].add(domain)
            if _is_offline_alarm_record(symptom):
                mhp_group_has_offline[group_id] = True

            alarm_id = _extract_alarm_id(symptom)
            if alarm_id:
                mhp_group_to_alarm_ids[group_id].add(alarm_id)

            alarm_group_ids = _parse_group_ids(symptom.get(group_field, ""))
            if alarm_group_ids:
                symptom_with_group_id_count += 1
            for alarm_group_id in alarm_group_ids:
                mhp_group_to_alarm_groups[group_id].add(alarm_group_id)
                alarm_group_to_mhp_groups[alarm_group_id].add(group_id)
                if site_id:
                    alarm_group_to_sites[alarm_group_id].add(site_id)
                    evidence_record = dict(symptom)
                    evidence_record["故障组ID"] = alarm_group_id
                    alarm_group_to_site_alarms[alarm_group_id][site_id].append(evidence_record)
                if domain:
                    alarm_group_alarm_domains[alarm_group_id].add(domain)
                if _is_offline_alarm_record(symptom):
                    alarm_group_has_offline[alarm_group_id] = True

        if group_sites:
            mhp_group_to_sites[group_id] = group_sites

    return {
        "group_records": group_records,
        "referenced_group_ids": referenced_group_ids,
        "mhp_group_to_sites": mhp_group_to_sites,
        "mhp_group_to_alarm_groups": dict(mhp_group_to_alarm_groups),
        "mhp_group_to_alarm_ids": dict(mhp_group_to_alarm_ids),
        "mhp_group_to_site_alarms": mhp_group_to_site_alarms,
        "mhp_group_alarm_domains": {
            group_id: set(domains) for group_id, domains in mhp_group_alarm_domains.items()
        },
        "mhp_group_has_offline": dict(mhp_group_has_offline),
        "alarm_group_to_sites": dict(alarm_group_to_sites),
        "alarm_group_to_mhp_groups": dict(alarm_group_to_mhp_groups),
        "alarm_group_to_site_alarms": alarm_group_to_site_alarms,
        "alarm_group_alarm_domains": {
            group_id: set(domains) for group_id, domains in alarm_group_alarm_domains.items()
        },
        "alarm_group_has_offline": dict(alarm_group_has_offline),
        "real_symptom_count": real_symptom_count,
        "skipped_virtual_symptom_count": skipped_virtual_symptom_count,
        "symptom_with_group_id_count": symptom_with_group_id_count,
    }


def _filter_metric_details_to_unrecalled(metric_result):
    details = metric_result.get("details", [])
    original_detail_count = len(details)
    filtered_details = [
        item
        for item in details
        if float(item.get("recall", 0.0) or 0.0) < 1.0
    ]
    metric_result["details"] = filtered_details
    metric_result["details_filter"] = "recall_lt_1"
    metric_result["details_total_count"] = original_detail_count
    metric_result["details_output_count"] = len(filtered_details)
    return metric_result


def compare_visual_alarm_groups(
    visual_output,
    *,
    group_field="故障组ID",
    ne_graph_file=None,
    min_site_num=0,
    no_domain_alarm="",
    no_domain_site="",
    require_domain_per_site="",
    only_offline=False,
    only_one=False,
    loose=False,
    window_seconds=900,
    only_unrecalled_predictions=False,
    output_file=None,
    mhp_case_jsonl_output_file=None,
    alarm_group_case_jsonl_output_file=None,
):
    ne_graph_data = load_ne_graph_data(ne_graph_file)
    ne_to_domain = build_ne_to_domain_map(ne_graph_data)
    no_domain_alarm = _normalize_domain_arg(no_domain_alarm)
    no_domain_site = _normalize_domain_arg(no_domain_site)
    require_domain_per_site = _normalize_domain_arg(require_domain_per_site)
    site_has_no_domain = build_site_has_domain_map(ne_graph_data, no_domain_site) if no_domain_site else {}
    site_has_required_domain = (
        build_site_has_domain_map(ne_graph_data, require_domain_per_site)
        if require_domain_per_site else {}
    )

    print("阶段 1/2：从 MHP visual 提取真实告警 symptom 并构造两侧 label...")
    indexes = _build_visual_indexes(
        visual_output,
        group_field=group_field,
        ne_to_domain=ne_to_domain,
    )

    print("阶段 2/2：分别按正向/反向口径计算平均指标...")
    mhp_group_to_loose_alarm_groups = {}
    alarm_group_to_loose_mhp_groups = {}
    if loose:
        mhp_group_to_loose_alarm_groups = _build_loose_groups_by_time_window(
            gold_to_sites=indexes["mhp_group_to_sites"],
            gold_to_base_pred_groups=indexes["mhp_group_to_alarm_groups"],
            pred_group_to_sites=indexes["alarm_group_to_sites"],
            pred_group_to_site_alarms=indexes["alarm_group_to_site_alarms"],
            window_seconds=window_seconds,
        )
        alarm_group_to_loose_mhp_groups = _build_loose_groups_by_time_window(
            gold_to_sites=indexes["alarm_group_to_sites"],
            gold_to_base_pred_groups=indexes["alarm_group_to_mhp_groups"],
            pred_group_to_sites=indexes["mhp_group_to_sites"],
            pred_group_to_site_alarms=indexes["mhp_group_to_site_alarms"],
            window_seconds=window_seconds,
        )

    mhp_group_as_gold = _compute_direction_metrics(
        gold_to_sites=indexes["mhp_group_to_sites"],
        gold_to_pred_groups=indexes["mhp_group_to_alarm_groups"],
        pred_group_to_sites=indexes["alarm_group_to_sites"],
        min_site_num=min_site_num,
        gold_alarm_domains=indexes["mhp_group_alarm_domains"],
        gold_has_offline=indexes["mhp_group_has_offline"],
        site_has_no_domain=site_has_no_domain,
        site_has_required_domain=site_has_required_domain,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        only_offline=only_offline,
        only_one=only_one,
        loose_gold_to_pred_groups=mhp_group_to_loose_alarm_groups,
    )
    alarm_group_as_gold = _compute_direction_metrics(
        gold_to_sites=indexes["alarm_group_to_sites"],
        gold_to_pred_groups=indexes["alarm_group_to_mhp_groups"],
        pred_group_to_sites=indexes["mhp_group_to_sites"],
        min_site_num=min_site_num,
        gold_alarm_domains=indexes["alarm_group_alarm_domains"],
        gold_has_offline=indexes["alarm_group_has_offline"],
        site_has_no_domain=site_has_no_domain,
        site_has_required_domain=site_has_required_domain,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        only_offline=only_offline,
        only_one=only_one,
        loose_gold_to_pred_groups=alarm_group_to_loose_mhp_groups,
    )

    if only_unrecalled_predictions:
        _filter_metric_details_to_unrecalled(mhp_group_as_gold)
        _filter_metric_details_to_unrecalled(alarm_group_as_gold)

    result = {
        "method": "alarm_flow_mhp_visual_alarm_group_comparison",
        "mhp_visual_output": os.path.abspath(visual_output),
        "group_field": group_field,
        "min_site_num": min_site_num,
        "no_domain_alarm": no_domain_alarm,
        "no_domain_alarm_mode": bool(no_domain_alarm),
        "no_domain_site": no_domain_site,
        "no_domain_site_mode": bool(no_domain_site),
        "require_domain_per_site": require_domain_per_site,
        "require_domain_per_site_mode": bool(require_domain_per_site),
        "only_offline_mode": only_offline,
        "only_one_mode": only_one,
        "loose_mode": loose,
        "window_seconds": window_seconds,
        "virtual_symptoms_excluded": True,
        "mhp_group_count": len(indexes["mhp_group_to_sites"]),
        "alarm_group_count": len(indexes["alarm_group_to_sites"]),
        "real_symptom_count": indexes["real_symptom_count"],
        "skipped_virtual_symptom_count": indexes["skipped_virtual_symptom_count"],
        "symptom_with_group_id_count": indexes["symptom_with_group_id_count"],
        "mhp_group_as_gold": mhp_group_as_gold,
        "alarm_group_as_gold": alarm_group_as_gold,
    }

    mhp_case_details = _build_case_details_for_direction(
        mhp_group_as_gold["details"],
        indexes["mhp_group_to_site_alarms"],
        indexes["alarm_group_to_site_alarms"],
    )
    alarm_group_case_details = _build_case_details_for_direction(
        alarm_group_as_gold["details"],
        indexes["alarm_group_to_site_alarms"],
        indexes["mhp_group_to_site_alarms"],
    )
    mhp_case_records = build_unrecalled_visualization_cases(
        mhp_case_details,
        "mhp_group_as_gold",
        ne_graph_data=ne_graph_data,
    )
    alarm_group_case_records = build_unrecalled_visualization_cases(
        alarm_group_case_details,
        "alarm_group_as_gold",
        ne_graph_data=ne_graph_data,
    )

    if mhp_case_jsonl_output_file:
        write_jsonl_records(mhp_case_jsonl_output_file, mhp_case_records)
        result["mhp_group_as_gold_case_jsonl_output"] = mhp_case_jsonl_output_file
        result["mhp_group_as_gold_case_count"] = len(mhp_case_records)
    if alarm_group_case_jsonl_output_file:
        write_jsonl_records(alarm_group_case_jsonl_output_file, alarm_group_case_records)
        result["alarm_group_as_gold_case_jsonl_output"] = alarm_group_case_jsonl_output_file
        result["alarm_group_as_gold_case_count"] = len(alarm_group_case_records)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return result


def _print_direction(label, payload):
    print(f"【{label}】")
    print(f"样本数: {payload.get('sample_count', 0)}")
    print(f"gold站点数分布: {payload.get('gold_site_count_distribution', {})}")
    print(f"平均召回率: {payload.get('average_recall', 0.0):.6f}")
    print(f"平均准确率: {payload.get('average_precision', 0.0):.6f}")
    print(f"平均F1: {payload.get('average_f1', 0.0):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="双向比较 alarm_flow_mhp visual 输出与告警自带故障组ID 的站点覆盖结果"
    )
    parser.add_argument("visual_output", help="alarm_flow_mhp 在线输出的 visual JSONL")
    parser.add_argument(
        "alarms",
        nargs="?",
        default="",
        help="兼容旧命令的可选参数；新 visual-only 评估不再读取原始告警输入",
    )
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警/visual symptom 中的原始故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于通过告警源回填站点的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="仅统计 gold label 站点数 >= 该值的样本；默认: 0（不过滤）",
    )
    parser.add_argument(
        "--no-domain-alarm",
        metavar="DOMAIN",
        help="如果当前 gold label 中出现来自指定 domain 的告警，则跳过该样本",
    )
    parser.add_argument(
        "--no-domain-site",
        metavar="DOMAIN",
        help="如果当前 gold label 的任一站点在 ne_graph 中包含指定 domain 设备，则跳过该样本",
    )
    parser.add_argument(
        "--require-domain-per-site",
        metavar="DOMAIN",
        help="先剔除不包含指定 domain 设备的 gold 站点，再做 min-site-num 过滤",
    )
    parser.add_argument(
        "--only-offline",
        action="store_true",
        help="仅统计包含 OFFLINE_ALARMS 的 gold label 样本",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖当前 gold 站点最多的单个预测 group，用它计算指标",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许在当前 gold 站点范围内，按时间窗把其它预测 group 做 loose 扩张",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=900,
        help="loose 模式使用的前后对称时间窗，单位秒，默认: 900",
    )
    parser.add_argument(
        "--no-potential",
        action="store_true",
        help="兼容旧参数；visual-only 评估始终直接使用真实 symptom 里的故障组ID",
    )
    parser.add_argument(
        "--only-unrecalled-predictions",
        action="store_true",
        help="输出 JSON 中两类 details 仅保留召回率不足 100%% 的预测；平均指标仍基于全部样本计算",
    )
    parser.add_argument(
        "--mhp-case-jsonl-output",
        default=None,
        help="MHP group 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成 sidecar，none 关闭",
    )
    parser.add_argument(
        "--alarm-group-case-jsonl-output",
        default=None,
        help="告警故障组ID 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成 sidecar，none 关闭",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mhp_visual_alarm_group_comparison.json",
        help="输出 JSON 文件，默认: mhp_visual_alarm_group_comparison.json",
    )
    args = parser.parse_args()

    mhp_case_output = args.mhp_case_jsonl_output
    alarm_case_output = args.alarm_group_case_jsonl_output
    if mhp_case_output is None and args.output:
        mhp_case_output = _derive_case_output_path(args.output, "mhp_group_as_gold")
    if alarm_case_output is None and args.output:
        alarm_case_output = _derive_case_output_path(args.output, "alarm_group_as_gold")
    if _is_disabled(mhp_case_output):
        mhp_case_output = None
    if _is_disabled(alarm_case_output):
        alarm_case_output = None

    result = compare_visual_alarm_groups(
        visual_output=args.visual_output,
        group_field=args.group_field,
        ne_graph_file=args.ne_graph,
        min_site_num=args.min_site_num,
        no_domain_alarm=args.no_domain_alarm,
        no_domain_site=args.no_domain_site,
        require_domain_per_site=args.require_domain_per_site,
        only_offline=args.only_offline,
        only_one=args.only_one,
        loose=args.loose,
        window_seconds=args.window_seconds,
        only_unrecalled_predictions=args.only_unrecalled_predictions,
        output_file=None,
        mhp_case_jsonl_output_file=mhp_case_output,
        alarm_group_case_jsonl_output_file=alarm_case_output,
    )
    result = _rename_result(result, visual_output=args.visual_output, alarms=args.alarms)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    _print_direction("MHP visual group 作为 gold", result.get("mhp_group_as_gold", {}))
    _print_direction("告警故障组ID 作为 gold", result.get("alarm_group_as_gold", {}))
    if args.output:
        print(f"结果已输出到: {args.output}")
    if result.get("mhp_group_as_gold_case_jsonl_output"):
        print(
            f"MHP group-case jsonl: {result['mhp_group_as_gold_case_jsonl_output']} "
            f"({result.get('mhp_group_as_gold_case_count', 0)} 条)"
        )
    if result.get("alarm_group_as_gold_case_jsonl_output"):
        print(
            f"告警故障组ID-case jsonl: {result['alarm_group_as_gold_case_jsonl_output']} "
            f"({result.get('alarm_group_as_gold_case_count', 0)} 条)"
        )


if __name__ == "__main__":
    main()
