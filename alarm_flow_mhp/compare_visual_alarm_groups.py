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
    _build_alarm_group_site_index,
    _build_alarm_identity_overlap,
    _build_group_site_alarm_ids,
    _extract_alarm_id,
    _merge_group_site_alarms,
    _is_offline_alarm_record,
    _normalize_domain_arg,
    _resolve_record_domain,
)
from alarm_tools.progress_utils import ProgressBar
from ticket_recall.ticket_recall_utils import (
    alarm_record_identity_key,
    build_site_alarm_map_for_sites,
    build_visualization_case_record,
    build_site_coord_index,
    build_site_coord_index_from_site_graph,
    build_site_to_ne_ids,
    build_ne_to_domain_map,
    build_site_has_domain_map,
    load_site_graph_data,
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
    mhp_group_to_alarm_keys = defaultdict(set)
    mhp_group_site_alarm_keys = defaultdict(lambda: defaultdict(set))
    mhp_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    mhp_group_alarm_domains = defaultdict(set)
    mhp_group_has_offline = defaultdict(bool)

    alarm_group_to_sites = defaultdict(set)
    alarm_group_to_mhp_groups = defaultdict(set)
    alarm_group_to_alarm_keys = defaultdict(set)
    alarm_group_site_alarm_keys = defaultdict(lambda: defaultdict(set))
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
                evidence_record["mhp_group_id"] = group_id
                mhp_group_to_site_alarms[group_id][site_id].append(evidence_record)

            domain = _resolve_record_domain(symptom, ne_to_domain)
            if domain:
                mhp_group_alarm_domains[group_id].add(domain)
            if _is_offline_alarm_record(symptom):
                mhp_group_has_offline[group_id] = True

            alarm_id = _extract_alarm_id(symptom)
            if alarm_id:
                mhp_group_to_alarm_ids[group_id].add(alarm_id)

            # 告警级指标用 (eid, occurrence_uuid) 做键：同一 eid 会有多次发生，只用 eid 会把它们混成一条。
            alarm_key = alarm_record_identity_key(symptom)
            if alarm_key is not None:
                mhp_group_to_alarm_keys[group_id].add(alarm_key)
                if site_id:
                    mhp_group_site_alarm_keys[group_id][site_id].add(alarm_key)

            alarm_group_ids = _parse_group_ids(symptom.get(group_field, ""))
            if alarm_group_ids:
                symptom_with_group_id_count += 1
            for alarm_group_id in alarm_group_ids:
                mhp_group_to_alarm_groups[group_id].add(alarm_group_id)
                alarm_group_to_mhp_groups[alarm_group_id].add(group_id)
                if alarm_key is not None:
                    alarm_group_to_alarm_keys[alarm_group_id].add(alarm_key)
                    if site_id:
                        alarm_group_site_alarm_keys[alarm_group_id][site_id].add(alarm_key)
                if site_id:
                    alarm_group_to_sites[alarm_group_id].add(site_id)
                    evidence_record = dict(symptom)
                    evidence_record["故障组ID"] = alarm_group_id
                    evidence_record["alarm_group_id"] = alarm_group_id
                    evidence_record["mhp_group_id"] = group_id
                    evidence_record["来源故障组UUID"] = group_id
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
        "mhp_group_to_alarm_keys": dict(mhp_group_to_alarm_keys),
        "mhp_group_site_alarm_keys": {
            group_id: dict(site_map) for group_id, site_map in mhp_group_site_alarm_keys.items()
        },
        "mhp_group_to_site_alarms": mhp_group_to_site_alarms,
        "mhp_group_alarm_domains": {
            group_id: set(domains) for group_id, domains in mhp_group_alarm_domains.items()
        },
        "mhp_group_has_offline": dict(mhp_group_has_offline),
        "alarm_group_to_sites": dict(alarm_group_to_sites),
        "alarm_group_to_mhp_groups": dict(alarm_group_to_mhp_groups),
        "alarm_group_to_alarm_keys": dict(alarm_group_to_alarm_keys),
        "alarm_group_site_alarm_keys": {
            group_id: dict(site_map) for group_id, site_map in alarm_group_site_alarm_keys.items()
        },
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


def _build_visualization_cases(details, method, *, ne_graph_data=None, case_scope="unrecalled"):
    """两种 case_scope 走同一条路径，只有筛选条件不同，这样都能带上进度。

    unrecalled 的筛选条件与 build_unrecalled_visualization_cases 保持一致（召回率 < 1）。
    """
    if case_scope == "unrecalled":
        selected_details = [
            detail
            for detail in details
            if float(detail.get("recall", 0.0) or 0.0) < 1.0
        ]
    elif case_scope == "all":
        selected_details = list(details)
    else:
        raise ValueError(f"unsupported case_scope: {case_scope}")

    if not selected_details:
        return []

    ne_graph_data = ne_graph_data or {}
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_coord_index = build_site_coord_index(ne_graph_data)
    site_coord_index.update(build_site_coord_index_from_site_graph(load_site_graph_data()))

    progress = ProgressBar(len(selected_details), f"构造 {method} case({case_scope})")
    records = []
    for detail in selected_details:
        progress.update()
        records.append(
            build_visualization_case_record(
                detail,
                method,
                ne_graph_data=ne_graph_data,
                site_to_ne_ids=site_to_ne_ids,
                site_coord_index=site_coord_index,
            )
        )
    progress.close()
    return records


def _build_side_viewer_case(
    group_id,
    method,
    display_sites,
    associated_sites,
    missing_sites,
    context_sites,
    site_alarms,
    recall,
    note,
    group_ids,
    ne_graph_data,
    site_to_ne_ids,
    site_coord_index,
):
    """把一侧的故障组包装成传播图查看器能直接吃的 group 记录。

    站点角色借用查看器已有的三套配色：associated=两侧共有，missing=只有本侧有，
    context=对侧比本侧多出来的。
    """
    detail = {
        "ticket_id": group_id,
        "ticket_site_count": len(display_sites),
        "ticket_sites": list(display_sites),
        "display_sites": list(display_sites),
        "associated_sites": list(associated_sites),
        "associated_site_alarms": build_site_alarm_map_for_sites(site_alarms, associated_sites),
        "missing_sites": list(missing_sites),
        "missing_site_alarms": build_site_alarm_map_for_sites(site_alarms, missing_sites),
        "context_sites": list(context_sites),
        "context_site_alarms": build_site_alarm_map_for_sites(site_alarms, context_sites),
        "associated_site_count": len(associated_sites),
        "missing_site_count": len(missing_sites),
        "fault_groups": list(group_ids),
        "recall": recall,
        "note": note,
    }
    # site_to_ne_ids / site_coord_index 必须由调用方传入并复用：省略时 build_visualization_case_record
    # 会自己重建，其中包含一次 site_graph.json 的磁盘读取，每个样本重复一次会直接卡死。
    return build_visualization_case_record(
        detail,
        method,
        ne_graph_data=ne_graph_data,
        site_to_ne_ids=site_to_ne_ids,
        site_coord_index=site_coord_index,
    )


def _build_comparison_case_records(
    details,
    direction,
    direction_label,
    gold_label,
    pred_label,
    gold_group_to_site_alarms,
    pred_group_to_site_alarms,
    ne_graph_data,
    site_to_ne_ids,
    site_coord_index,
    max_cases=0,
    alarm_scope="",
):
    """每个 gold 组一条记录，带上它自己和对侧命中组的完整内容，供对比浏览器渲染。"""
    ranked = sorted(
        details,
        key=lambda item: (
            float(item.get("f1", 0.0) or 0.0),
            -int(item.get("gold_site_count", 0) or 0),
            item.get("gold_id", ""),
        ),
    )
    if max_cases > 0:
        ranked = ranked[:max_cases]

    progress_label = f"构造对比样本 {alarm_scope}/{direction}".replace("/", "/", 1) if alarm_scope else f"构造对比样本 {direction}"
    progress = ProgressBar(len(ranked), progress_label) if ranked else None
    records = []
    for item in ranked:
        if progress is not None:
            progress.update()
        gold_id = item.get("gold_id", "")
        gold_sites = sorted(item.get("gold_sites", []))
        pred_sites = sorted(item.get("predicted_sites", []))
        shared_sites = sorted(set(gold_sites) & set(pred_sites))
        gold_only_sites = sorted(set(gold_sites) - set(pred_sites))
        pred_only_sites = sorted(set(pred_sites) - set(gold_sites))
        pred_group_ids = list(item.get("effective_predicted_groups", []))

        gold_site_alarms = gold_group_to_site_alarms.get(gold_id, {})
        pred_site_alarms = _merge_group_site_alarms(pred_group_ids, pred_group_to_site_alarms)

        note = (
            f"{gold_label} {gold_id}：{len(gold_sites)} 个站点；"
            f"{pred_label} 命中 {len(pred_group_ids)} 个组、{len(pred_sites)} 个站点；"
            f"共有 {len(shared_sites)}，仅{gold_label}有 {len(gold_only_sites)}，仅{pred_label}有 {len(pred_only_sites)}"
        )

        records.append({
            "alarm_scope": alarm_scope,
            "alarm_scope_label": ALARM_SCOPE_LABELS.get(alarm_scope, alarm_scope),
            "direction": direction,
            "direction_label": direction_label,
            "case_id": gold_id,
            "gold_site_count": len(gold_sites),
            "shared_sites": shared_sites,
            "gold_only_sites": gold_only_sites,
            "pred_only_sites": pred_only_sites,
            "site_recall": item.get("recall", 0.0),
            "site_precision": item.get("precision", 0.0),
            "site_f1": item.get("f1", 0.0),
            "alarm_recall": item.get("alarm_recall", 0.0),
            "alarm_precision": item.get("alarm_precision", 0.0),
            "alarm_f1": item.get("alarm_f1", 0.0),
            "gold_alarm_count": item.get("gold_alarm_count", 0),
            "predicted_alarm_count": item.get("predicted_alarm_count", 0),
            "matched_alarm_count": item.get("matched_alarm_count", 0),
            "gold_alarms_missing_from_pred_universe_count": item.get(
                "gold_alarms_missing_from_pred_universe_count", 0
            ),
            "note": note,
            "gold_side": {
                "present": True,
                "label": gold_label,
                "group_ids": [gold_id],
                "sites": gold_sites,
                "site_count": len(gold_sites),
                "alarm_count": item.get("gold_alarm_count", 0),
                "shared_sites": shared_sites,
                "own_only_sites": gold_only_sites,
                "viewer_case": _build_side_viewer_case(
                    group_id=gold_id,
                    method=f"{direction}::gold",
                    display_sites=gold_sites,
                    associated_sites=shared_sites,
                    missing_sites=gold_only_sites,
                    context_sites=[],
                    site_alarms=gold_site_alarms,
                    recall=item.get("recall", 0.0),
                    note=note,
                    group_ids=[gold_id],
                    ne_graph_data=ne_graph_data,
                    site_to_ne_ids=site_to_ne_ids,
                    site_coord_index=site_coord_index,
                ),
            },
            "pred_side": {
                "present": bool(pred_group_ids),
                "label": pred_label,
                "group_ids": pred_group_ids,
                "sites": pred_sites,
                "site_count": len(pred_sites),
                "alarm_count": item.get("predicted_alarm_count", 0),
                "shared_sites": shared_sites,
                "own_only_sites": pred_only_sites,
                "viewer_case": _build_side_viewer_case(
                    group_id=gold_id,
                    method=f"{direction}::pred",
                    display_sites=pred_sites,
                    associated_sites=shared_sites,
                    missing_sites=[],
                    context_sites=pred_only_sites,
                    site_alarms=pred_site_alarms,
                    recall=item.get("recall", 0.0),
                    note=note,
                    group_ids=pred_group_ids,
                    ne_graph_data=ne_graph_data,
                    site_to_ne_ids=site_to_ne_ids,
                    site_coord_index=site_coord_index,
                ) if pred_group_ids else None,
            },
        })

    if progress is not None:
        progress.close()

    return records



ALARM_SCOPE_LABELS = {
    "visual": "visual-only（只统计 MHP 消费过的告警）",
    "raw": "raw（完整原始告警流，含 MHP 未消费的告警）",
}

ALARM_SIDE_KEYS = (
    "alarm_group_to_sites",
    "alarm_group_to_alarm_keys",
    "alarm_group_site_alarm_keys",
    "alarm_group_to_site_alarms",
    "alarm_group_alarm_domains",
    "alarm_group_has_offline",
)


def _visual_alarm_side(indexes):
    """visual-only 口径：告警组只由 visual 里出现过的真实 symptom 构成。"""
    return {key: indexes[key] for key in ALARM_SIDE_KEYS}


def _build_raw_alarm_side(alarms_input, *, ne_graph_file, group_field):
    """raw 口径：告警组由完整原始告警流构成，包含 MHP 从未消费到的告警。

    注意两侧的 occurrence_uuid 都由 alarm_content_uuid 对原始告警记录取哈希
    （见 fault_grouping/alarm_events/io.py），所以只有传入同一份告警导出时告警级才对得上；
    不一致时 alarm_identity_overlap 会暴露出来。
    """
    (
        alarm_group_to_sites,
        alarm_group_to_alarm_keys,
        alarm_group_to_site_alarms,
        alarm_group_alarm_domains,
        alarm_group_has_offline,
        _alarm_key_to_groups,
    ) = _build_alarm_group_site_index(
        alarms_input,
        ne_graph_file=ne_graph_file,
        group_field=group_field,
    )
    return {
        "alarm_group_to_sites": alarm_group_to_sites,
        "alarm_group_to_alarm_keys": alarm_group_to_alarm_keys,
        "alarm_group_site_alarm_keys": _build_group_site_alarm_ids(alarm_group_to_site_alarms),
        "alarm_group_to_site_alarms": alarm_group_to_site_alarms,
        "alarm_group_alarm_domains": alarm_group_alarm_domains,
        "alarm_group_has_offline": alarm_group_has_offline,
    }


def _alarm_universe(group_to_alarm_keys):
    universe = set()
    for alarm_keys in group_to_alarm_keys.values():
        universe.update(alarm_keys)
    return universe


def _compute_scope_result(
    indexes,
    alarm_side,
    *,
    min_site_num,
    site_has_no_domain,
    site_has_required_domain,
    no_domain_alarm,
    no_domain_site,
    require_domain_per_site,
    only_offline,
    only_one,
    loose,
    window_seconds,
    only_unrecalled_predictions,
    scope_name="",
):
    """MHP 侧固定来自 visual 真实 symptom，只有告警组那一侧随口径切换。"""
    mhp_alarm_universe = _alarm_universe(indexes["mhp_group_to_alarm_keys"])
    alarm_group_alarm_universe = _alarm_universe(alarm_side["alarm_group_to_alarm_keys"])

    mhp_group_to_loose_alarm_groups = {}
    alarm_group_to_loose_mhp_groups = {}
    if loose:
        mhp_group_to_loose_alarm_groups = _build_loose_groups_by_time_window(
            gold_to_sites=indexes["mhp_group_to_sites"],
            gold_to_base_pred_groups=indexes["mhp_group_to_alarm_groups"],
            pred_group_to_sites=alarm_side["alarm_group_to_sites"],
            pred_group_to_site_alarms=alarm_side["alarm_group_to_site_alarms"],
            window_seconds=window_seconds,
        )
        alarm_group_to_loose_mhp_groups = _build_loose_groups_by_time_window(
            gold_to_sites=alarm_side["alarm_group_to_sites"],
            gold_to_base_pred_groups=indexes["alarm_group_to_mhp_groups"],
            pred_group_to_sites=indexes["mhp_group_to_sites"],
            pred_group_to_site_alarms=indexes["mhp_group_to_site_alarms"],
            window_seconds=window_seconds,
        )

    mhp_group_as_gold = _compute_direction_metrics(
        gold_to_sites=indexes["mhp_group_to_sites"],
        gold_to_pred_groups=indexes["mhp_group_to_alarm_groups"],
        pred_group_to_sites=alarm_side["alarm_group_to_sites"],
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
        gold_to_alarm_ids=indexes["mhp_group_to_alarm_keys"],
        gold_to_site_alarm_ids=indexes["mhp_group_site_alarm_keys"],
        pred_group_to_alarm_ids=alarm_side["alarm_group_to_alarm_keys"],
        pred_side_alarm_universe=alarm_group_alarm_universe,
        progress_label=f"计算指标 {scope_name}/mhp_group_as_gold".lstrip("/"),
    )
    alarm_group_as_gold = _compute_direction_metrics(
        gold_to_sites=alarm_side["alarm_group_to_sites"],
        gold_to_pred_groups=indexes["alarm_group_to_mhp_groups"],
        pred_group_to_sites=indexes["mhp_group_to_sites"],
        min_site_num=min_site_num,
        gold_alarm_domains=alarm_side["alarm_group_alarm_domains"],
        gold_has_offline=alarm_side["alarm_group_has_offline"],
        site_has_no_domain=site_has_no_domain,
        site_has_required_domain=site_has_required_domain,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        only_offline=only_offline,
        only_one=only_one,
        loose_gold_to_pred_groups=alarm_group_to_loose_mhp_groups,
        gold_to_alarm_ids=alarm_side["alarm_group_to_alarm_keys"],
        gold_to_site_alarm_ids=alarm_side["alarm_group_site_alarm_keys"],
        pred_group_to_alarm_ids=indexes["mhp_group_to_alarm_keys"],
        pred_side_alarm_universe=mhp_alarm_universe,
        progress_label=f"计算指标 {scope_name}/alarm_group_as_gold".lstrip("/"),
    )

    if only_unrecalled_predictions:
        _filter_metric_details_to_unrecalled(mhp_group_as_gold)
        _filter_metric_details_to_unrecalled(alarm_group_as_gold)

    return {
        "alarm_group_count": len(alarm_side["alarm_group_to_sites"]),
        "alarm_identity_overlap": _build_alarm_identity_overlap(
            mhp_alarm_universe,
            alarm_group_alarm_universe,
        ),
        "mhp_group_as_gold": mhp_group_as_gold,
        "alarm_group_as_gold": alarm_group_as_gold,
    }


def compare_visual_alarm_groups(
    visual_output,
    *,
    alarms_input=None,
    alarm_scope="visual",
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
    case_scope="unrecalled",
    output_file=None,
    mhp_case_jsonl_output_file=None,
    alarm_group_case_jsonl_output_file=None,
    comparison_jsonl_output_file=None,
    comparison_max_cases=0,
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

    alarm_sides = {"visual": _visual_alarm_side(indexes)}
    if alarm_scope in ("raw", "both"):
        if not alarms_input:
            raise ValueError("alarm_scope 为 raw/both 时必须提供原始告警输入")
        print("阶段 2/3：读取完整原始告警流，构造 raw 口径的告警故障组...")
        alarm_sides["raw"] = _build_raw_alarm_side(
            alarms_input,
            ne_graph_file=ne_graph_file,
            group_field=group_field,
        )

    scope_names = ["visual", "raw"] if alarm_scope == "both" else [alarm_scope]
    primary_scope = scope_names[0]

    print("阶段 3/3：分别按正向/反向口径计算平均指标...")
    scopes = {}
    for scope_name in scope_names:
        scopes[scope_name] = _compute_scope_result(
            indexes,
            alarm_sides[scope_name],
            min_site_num=min_site_num,
            site_has_no_domain=site_has_no_domain,
            site_has_required_domain=site_has_required_domain,
            no_domain_alarm=no_domain_alarm,
            no_domain_site=no_domain_site,
            require_domain_per_site=require_domain_per_site,
            only_offline=only_offline,
            only_one=only_one,
            loose=loose,
            window_seconds=window_seconds,
            only_unrecalled_predictions=only_unrecalled_predictions,
            scope_name=scope_name,
        )

    primary = scopes[primary_scope]
    mhp_group_as_gold = primary["mhp_group_as_gold"]
    alarm_group_as_gold = primary["alarm_group_as_gold"]
    primary_alarm_side = alarm_sides[primary_scope]

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
        "case_scope": case_scope,
        "alarm_scope": alarm_scope,
        "primary_alarm_scope": primary_scope,
        "mhp_group_count": len(indexes["mhp_group_to_sites"]),
        "alarm_group_count": primary["alarm_group_count"],
        "real_symptom_count": indexes["real_symptom_count"],
        "skipped_virtual_symptom_count": indexes["skipped_virtual_symptom_count"],
        "symptom_with_group_id_count": indexes["symptom_with_group_id_count"],
        "scopes": {
            scope_name: {
                "alarm_scope": scope_name,
                "alarm_scope_label": ALARM_SCOPE_LABELS[scope_name],
                **scope_result,
            }
            for scope_name, scope_result in scopes.items()
        },
        # 主口径镜像到顶层，保持既有消费方不变
        "alarm_identity_overlap": primary["alarm_identity_overlap"],
        "mhp_group_as_gold": mhp_group_as_gold,
        "alarm_group_as_gold": alarm_group_as_gold,
    }

    mhp_case_details = _build_case_details_for_direction(
        mhp_group_as_gold["details"],
        indexes["mhp_group_to_site_alarms"],
        primary_alarm_side["alarm_group_to_site_alarms"],
    )
    alarm_group_case_details = _build_case_details_for_direction(
        alarm_group_as_gold["details"],
        primary_alarm_side["alarm_group_to_site_alarms"],
        indexes["mhp_group_to_site_alarms"],
    )
    mhp_case_records = _build_visualization_cases(
        mhp_case_details,
        "mhp_group_as_gold",
        ne_graph_data=ne_graph_data,
        case_scope=case_scope,
    )
    alarm_group_case_records = _build_visualization_cases(
        alarm_group_case_details,
        "alarm_group_as_gold",
        ne_graph_data=ne_graph_data,
        case_scope=case_scope,
    )

    if comparison_jsonl_output_file:
        # 索引只建一次，所有样本复用；省略透传会让 build_visualization_case_record 每次重读 site_graph.json。
        comparison_site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
        comparison_site_coord_index = build_site_coord_index(ne_graph_data)
        comparison_site_coord_index.update(
            build_site_coord_index_from_site_graph(load_site_graph_data())
        )
        comparison_records = []
        for scope_name in scope_names:
            scope_alarm_side = alarm_sides[scope_name]
            scope_label = ALARM_SCOPE_LABELS[scope_name]
            comparison_records += _build_comparison_case_records(
                scopes[scope_name]["mhp_group_as_gold"]["details"],
                direction="mhp_group_as_gold",
                direction_label=f"MHP 生成的故障组 作为 gold（{scope_label}）",
                gold_label="MHP 生成的故障组",
                pred_label="原始故障组ID组",
                gold_group_to_site_alarms=indexes["mhp_group_to_site_alarms"],
                pred_group_to_site_alarms=scope_alarm_side["alarm_group_to_site_alarms"],
                ne_graph_data=ne_graph_data,
                site_to_ne_ids=comparison_site_to_ne_ids,
                site_coord_index=comparison_site_coord_index,
                max_cases=comparison_max_cases,
                alarm_scope=scope_name,
            )
            comparison_records += _build_comparison_case_records(
                scopes[scope_name]["alarm_group_as_gold"]["details"],
                direction="alarm_group_as_gold",
                direction_label=f"原始故障组ID组 作为 gold（{scope_label}）",
                gold_label="原始故障组ID组",
                pred_label="MHP 生成的故障组",
                gold_group_to_site_alarms=scope_alarm_side["alarm_group_to_site_alarms"],
                pred_group_to_site_alarms=indexes["mhp_group_to_site_alarms"],
                ne_graph_data=ne_graph_data,
                site_to_ne_ids=comparison_site_to_ne_ids,
                site_coord_index=comparison_site_coord_index,
                max_cases=comparison_max_cases,
                alarm_scope=scope_name,
            )
        write_jsonl_records(comparison_jsonl_output_file, comparison_records)
        result["comparison_case_jsonl_output"] = comparison_jsonl_output_file
        result["comparison_case_count"] = len(comparison_records)

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
    print(f"[站点级] 平均召回率: {payload.get('average_recall', 0.0):.6f}")
    print(f"[站点级] 平均准确率: {payload.get('average_precision', 0.0):.6f}")
    print(f"[站点级] 平均F1: {payload.get('average_f1', 0.0):.6f}")
    print(f"[告警级] 平均召回率: {payload.get('average_alarm_recall', 0.0):.6f}")
    print(f"[告警级] 平均准确率: {payload.get('average_alarm_precision', 0.0):.6f}")
    print(f"[告警级] 平均F1: {payload.get('average_alarm_f1', 0.0):.6f}")
    print(
        f"gold告警中对侧全集不存在的条数: {payload.get('gold_alarms_missing_from_pred_universe_total', 0)}"
        f" / {payload.get('gold_alarm_total', 0)}"
    )


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
        "--case-scope",
        choices=("unrecalled", "all"),
        default="unrecalled",
        help="case JSONL 输出范围：unrecalled 只输出未满召回样本，all 输出全部样本。默认: unrecalled",
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
        "--alarm-scope",
        choices=("visual", "raw", "both"),
        default="visual",
        help=(
            "告警故障组ID 一侧的告警范围。visual: 只统计 MHP 消费过的告警（默认，口径干净）；"
            "raw: 用完整原始告警流，含 MHP 未消费的告警，会系统性压低 MHP 侧召回；"
            "both: 两套指标都算，便于对照。raw/both 需要提供 alarms 位置参数"
        ),
    )
    parser.add_argument(
        "--comparison-jsonl-output",
        default=None,
        help="两侧故障组并排对比的 jsonl，可加载到 visualization/group_comparison_browser.html；默认随主输出生成 sidecar，none 关闭",
    )
    parser.add_argument(
        "--comparison-max-cases",
        type=int,
        default=0,
        help=(
            "对比 jsonl 每个方向/每种口径最多输出的样本数，按站点级 F1 升序取最差的。"
            "默认 0 表示全量不截断；样本多时文件会很大，浏览器一次性解析可能吃不消，"
            "这时可以设个上限只看最差的那批"
        ),
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

    comparison_output = args.comparison_jsonl_output
    if comparison_output is None and args.output:
        comparison_output = _derive_case_output_path(args.output, "comparison")
    if _is_disabled(comparison_output):
        comparison_output = None

    if args.alarm_scope in ("raw", "both") and not args.alarms:
        parser.error("--alarm-scope raw/both 需要提供 alarms 位置参数（原始告警输入）")

    result = compare_visual_alarm_groups(
        visual_output=args.visual_output,
        alarms_input=args.alarms,
        alarm_scope=args.alarm_scope,
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
        case_scope=args.case_scope,
        output_file=None,
        mhp_case_jsonl_output_file=mhp_case_output,
        alarm_group_case_jsonl_output_file=alarm_case_output,
        comparison_jsonl_output_file=comparison_output,
        comparison_max_cases=args.comparison_max_cases,
    )
    result = _rename_result(result, visual_output=args.visual_output, alarms=args.alarms)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    for scope_name, scope_result in result.get("scopes", {}).items():
        print(f"===== 告警范围口径: {scope_result.get('alarm_scope_label', scope_name)} =====")
        overlap = scope_result.get("alarm_identity_overlap", {})
        if scope_name == "raw":
            print(
                f"告警实例键重合度: 两侧共有 {overlap.get('shared_alarm_count', 0)} 条 "
                f"(Jaccard {overlap.get('jaccard', 0.0):.6f})"
            )
            if not overlap.get("shared_alarm_count"):
                print("警告: 两侧没有任何共有告警，告警级指标不可信，请确认 alarms 与跑 visual 时是同一份导出")
        print(f"告警故障组数: {scope_result.get('alarm_group_count', 0)}")
        _print_direction("MHP visual group 作为 gold", scope_result.get("mhp_group_as_gold", {}))
        _print_direction("告警故障组ID 作为 gold", scope_result.get("alarm_group_as_gold", {}))
    if args.output:
        print(f"结果已输出到: {args.output}")
    if result.get("comparison_case_jsonl_output"):
        comparison_path = result["comparison_case_jsonl_output"]
        size_mb = os.path.getsize(comparison_path) / (1024 * 1024) if os.path.exists(comparison_path) else 0.0
        print(
            f"两侧对比 jsonl: {comparison_path} "
            f"({result.get('comparison_case_count', 0)} 条, {size_mb:.1f} MB)，"
            f"用 visualization/group_comparison_browser.html 打开"
        )
        if size_mb > 200:
            print(
                "提示: 文件较大，浏览器一次性解析可能很慢或崩溃；"
                "如需要可用 --comparison-max-cases N 只保留最差的 N 个样本"
            )
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
