import json
import os

from argparse import ArgumentParser
from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(2)

from alarm_tools.alarm_types import OFFLINE_ALARMS
from alarm_tools.alarm_inputs import build_ne_to_site_map, stream_alarm_inputs
from fault_grouping.alarm_events.identity import alarm_content_uuid
from fault_grouping.alarm_events.io import is_clear_alarm, parse_datetime_text
from fault_grouping.alarm_events.sorted_cache import (
    is_sorted_alarm_cache_file,
    iter_sorted_alarm_cache_items,
)
from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.recall_common import _extract_group_id, _extract_group_sites
from ticket_recall.evaluation.recall_common import (
    _compute_site_metrics,
    _normalize_text,
    _parse_group_ids,
    _resolve_alarm_site_id,
)
from alarm_tools.progress_utils import ProgressBar
from ticket_recall.ticket_recall_utils import (
    alarm_record_identity_key,
    build_ne_to_domain_map,
    build_site_alarm_map_for_sites,
    build_visualization_case_record,
    build_site_has_domain_map,
    build_group_site_time_index,
    build_site_to_group_index,
    build_unrecalled_visualization_cases,
    load_ne_graph_data,
    expand_groups_by_time_window,
    select_best_group_by_target_sites,
    write_jsonl_records,
)

OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)


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


def _resolve_record_domain(record, ne_to_domain=None):
    domain = _extract_domain(record)
    if domain:
        return domain
    if not isinstance(record, dict):
        return ""
    alarm_source = (
        _normalize_text(record.get("alarm_source", ""))
        or _normalize_text(record.get("告警源", ""))
    )
    if not alarm_source:
        return ""
    return _normalize_text((ne_to_domain or {}).get(alarm_source, "")).upper()


def _normalize_domain_arg(value):
    return _normalize_text(value).upper()


def _is_offline_alarm_record(record):
    if not isinstance(record, dict):
        return False
    alarm_name = (
        _normalize_text(record.get("alarm", ""))
        or _normalize_text(record.get("alarm_type", ""))
        or _normalize_text(record.get("告警标题", ""))
    )
    return bool(alarm_name and alarm_name in OFFLINE_ALARM_SET)


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


def _build_ultimate_group_indexes(group_records, group_field, ne_to_domain=None):
    referenced_group_ids = _collect_referenced_group_ids(group_records)
    ultimate_group_to_sites = {}
    ultimate_group_to_alarm_groups = defaultdict(set)
    ultimate_group_to_alarm_ids = defaultdict(set)
    ultimate_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    ultimate_group_alarm_domains = defaultdict(set)
    ultimate_group_has_offline = defaultdict(bool)
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

        ne_info = group_record.get("ne_info", {})
        if isinstance(ne_info, dict):
            for ne_entry in ne_info.values():
                if not isinstance(ne_entry, dict):
                    continue
                for alarm in ne_entry.get("alarm", []):
                    domain = _resolve_record_domain(alarm, ne_to_domain)
                    if domain:
                        ultimate_group_alarm_domains[group_id].add(domain)

        for symptom in group_record.get("symptoms", []):
            if not isinstance(symptom, dict):
                continue
            domain = _resolve_record_domain(symptom, ne_to_domain)
            if domain:
                ultimate_group_alarm_domains[group_id].add(domain)
            if _is_offline_alarm_record(symptom):
                ultimate_group_has_offline[group_id] = True
            site_id = _normalize_text(symptom.get("node", ""))
            if site_id:
                evidence_record = dict(symptom)
                evidence_record["来源故障组UUID"] = group_id
                ultimate_group_to_site_alarms[group_id][site_id].append(evidence_record)
            alarm_key = alarm_record_identity_key(symptom)
            if alarm_key is not None:
                ultimate_group_to_alarm_ids[group_id].add(alarm_key)
                alarm_id_to_ultimate_groups[alarm_key].add(group_id)
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
        {group_id: set(domains) for group_id, domains in ultimate_group_alarm_domains.items()},
        dict(ultimate_group_has_offline),
        dict(alarm_id_to_ultimate_groups),
    )


def _stream_raw_alarm_records(alarm_input, start_time=None, end_time=None):
    """统一原始告警输入：JSONL/CSV/ZIP/目录，或推理引擎用的排序告警缓存。

    缓存条目必须复用 prepare 时算好的 occurrence_uuid：缓存里的 alarm 载荷被改写过
    （清除行覆盖了告警首次发生时间，还补了告警编码ID），对它重算 alarm_content_uuid
    会和 visual 侧对不上。同一条原始告警在缓存里有 raise/clear 两行且 identity 相同，
    跳过清除行以还原"一条原始告警一条记录"的口径。

    start_time/end_time 按告警首次发生时间过滤，与 stream_alarm_period_mhp.py
    的时间窗口径一致；缓存的 raise 行 ts 就是首次发生时间，且整体按 ts 有序，
    所以越过 end_time 后可以直接停止读取。
    """
    start_ts = (
        parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    )
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time 不能晚于 end_time")

    if is_sorted_alarm_cache_file(alarm_input):
        for item in iter_sorted_alarm_cache_items(alarm_input, show_progress=True):
            ts = float(item.get("ts", 0.0))
            if end_ts is not None and ts > end_ts:
                break
            payload = item.get("alarm", {})
            if not isinstance(payload, dict) or is_clear_alarm(payload):
                continue
            if start_ts is not None and ts < start_ts:
                continue
            record = dict(payload)
            record["occurrence_uuid"] = item.get("occurrence_uuid", "")
            site_id = _normalize_text(item.get("site_id", ""))
            if site_id:
                # 缓存里的 site_id 是 prepare 时校验过的解析结果，覆盖原字段
                # 才能和引擎实际使用的站点保持一致。
                record["站点ID"] = site_id
            yield record
        return
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        if start_ts is not None or end_ts is not None:
            occurred_ts = parse_datetime_text(
                str(alarm.get("告警首次发生时间", "")).strip(), "告警首次发生时间"
            ).timestamp()
            if start_ts is not None and occurred_ts < start_ts:
                continue
            if end_ts is not None and occurred_ts > end_ts:
                continue
        record = dict(alarm)
        record["occurrence_uuid"] = alarm_content_uuid(record)
        yield record


def _build_alarm_group_site_index(
    alarm_input, ne_graph_file, group_field, start_time=None, end_time=None
):
    ne_to_site = {}
    ne_to_domain = {}
    if ne_graph_file and os.path.exists(ne_graph_file):
        ne_to_site = build_ne_to_site_map(ne_graph_file)
        with open(ne_graph_file, "r", encoding="utf-8") as f:
            ne_graph_data = json.load(f)
        if isinstance(ne_graph_data, dict):
            ne_to_domain = {
                _normalize_text(ne_id): _extract_domain(ne_info)
                for ne_id, ne_info in ne_graph_data.items()
                if _normalize_text(ne_id)
            }

    alarm_group_to_sites = defaultdict(set)
    alarm_group_to_alarm_ids = defaultdict(set)
    alarm_group_to_site_alarms = defaultdict(lambda: defaultdict(list))
    alarm_group_alarm_domains = defaultdict(set)
    alarm_group_has_offline = defaultdict(bool)
    alarm_id_to_alarm_groups = defaultdict(set)
    for alarm in _stream_raw_alarm_records(alarm_input, start_time=start_time, end_time=end_time):
        group_ids = _parse_group_ids(alarm.get(group_field, ""))
        if not group_ids:
            continue

        domain = _extract_domain(alarm)
        if not domain:
            alarm_source = _normalize_text(alarm.get("告警源", ""))
            if alarm_source:
                domain = _normalize_text(ne_to_domain.get(alarm_source, "")).upper()
        site_id = _resolve_alarm_site_id(alarm, ne_to_site)

        for group_id in group_ids:
            if domain:
                alarm_group_alarm_domains[group_id].add(domain)
            if _is_offline_alarm_record(alarm):
                alarm_group_has_offline[group_id] = True
            if site_id:
                alarm_group_to_sites[group_id].add(site_id)
                evidence_record = dict(alarm)
                evidence_record["故障组ID"] = group_id
                evidence_record["关联站点ID"] = site_id
                alarm_group_to_site_alarms[group_id][site_id].append(evidence_record)
            alarm_key = alarm_record_identity_key(evidence_record if site_id else alarm)
            if alarm_key is not None:
                alarm_group_to_alarm_ids[group_id].add(alarm_key)
                alarm_id_to_alarm_groups[alarm_key].add(group_id)

    return (
        dict(alarm_group_to_sites),
        dict(alarm_group_to_alarm_ids),
        alarm_group_to_site_alarms,
        {group_id: set(domains) for group_id, domains in alarm_group_alarm_domains.items()},
        dict(alarm_group_has_offline),
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


def _build_group_site_alarm_ids(group_to_site_alarms):
    """group -> site -> 告警实例键集合，用于 require-domain-per-site 裁剪站点后同步裁剪告警。"""
    result = {}
    for group_id, site_alarm_map in group_to_site_alarms.items():
        per_site = {}
        for site_id, alarms in site_alarm_map.items():
            normalized_site_id = _normalize_text(site_id)
            if not normalized_site_id:
                continue
            alarm_keys = set()
            for record in alarms:
                alarm_key = alarm_record_identity_key(record)
                if alarm_key is not None:
                    alarm_keys.add(alarm_key)
            if alarm_keys:
                per_site[normalized_site_id] = alarm_keys
        result[group_id] = per_site
    return result


def _resolve_gold_alarm_ids(gold_id, gold_to_alarm_ids, gold_to_site_alarm_ids, gold_sites, restrict_to_sites):
    if not restrict_to_sites:
        return set(gold_to_alarm_ids.get(gold_id, set()))

    per_site = gold_to_site_alarm_ids.get(gold_id, {})
    alarm_ids = set()
    for site_id in gold_sites:
        alarm_ids.update(per_site.get(site_id, set()))
    return alarm_ids


def _build_alarm_identity_overlap(ultimate_alarm_universe, alarm_group_alarm_universe):
    """告警实例键跨两侧的重合情况。

    两侧的 occurrence_uuid 都由 alarm_content_uuid 对原始告警记录取值，所以只有在
    group output 和 alarm 输入确实来自同一份告警导出时才会对上。重合度接近 0 通常意味着
    传错了告警文件，此时告警级指标没有意义。
    """
    intersection = ultimate_alarm_universe & alarm_group_alarm_universe
    union = ultimate_alarm_universe | alarm_group_alarm_universe
    return {
        "ultimate_side_alarm_count": len(ultimate_alarm_universe),
        "alarm_group_side_alarm_count": len(alarm_group_alarm_universe),
        "shared_alarm_count": len(intersection),
        "ultimate_side_shared_ratio": (
            len(intersection) / len(ultimate_alarm_universe) if ultimate_alarm_universe else 0.0
        ),
        "alarm_group_side_shared_ratio": (
            len(intersection) / len(alarm_group_alarm_universe) if alarm_group_alarm_universe else 0.0
        ),
        "jaccard": len(intersection) / len(union) if union else 0.0,
    }


def _build_gold_site_count_distribution(details):
    counts = defaultdict(int)
    for item in details:
        try:
            site_count = int(item.get("gold_site_count", 0) or 0)
        except (TypeError, ValueError):
            continue
        counts[site_count] += 1

    return {
        str(site_count): counts[site_count]
        for site_count in sorted(counts)
    }


def _merge_group_site_alarms(group_ids, group_to_site_alarms):
    merged = defaultdict(list)
    for group_id in group_ids:
        for site_id, alarms in group_to_site_alarms.get(group_id, {}).items():
            merged[site_id].extend(alarms)
    return dict(merged)


def _nonempty_alarm_sites(site_alarm_map):
    return sorted(
        _normalize_text(site_id)
        for site_id, alarms in site_alarm_map.items()
        if _normalize_text(site_id) and isinstance(alarms, list) and alarms
    )


def _derive_case_output_path(output_file, suffix):
    base, _ext = os.path.splitext(output_file)
    return f"{base}.{suffix}.cases.jsonl"


def _format_case_sites_note(recalled_site_ids, missing_site_ids):
    recalled_sites = [
        _normalize_text(site_id)
        for site_id in recalled_site_ids
        if _normalize_text(site_id)
    ]
    missing_sites = [
        _normalize_text(site_id)
        for site_id in missing_site_ids
        if _normalize_text(site_id)
    ]
    recalled_text = "，".join(recalled_sites) if recalled_sites else "无"
    missing_text = "，".join(missing_sites) if missing_sites else "无"
    return f"召回的站点列表：{recalled_text}\n未召回的站点列表：{missing_text}"


def _build_case_details_for_direction(details, gold_group_to_site_alarms, pred_group_to_site_alarms):
    case_details = []
    for item in details:
        gold_sites = sorted(item.get("gold_sites", []))
        matched_sites = sorted(item.get("matched_sites", []))
        missing_sites = sorted(set(gold_sites) - set(matched_sites))
        effective_predicted_groups = list(item.get("effective_predicted_groups", []))
        merged_predicted_site_alarms = _merge_group_site_alarms(
            effective_predicted_groups,
            pred_group_to_site_alarms,
        )
        associated_site_alarms = build_site_alarm_map_for_sites(
            merged_predicted_site_alarms,
            matched_sites,
        )
        missing_site_alarms = build_site_alarm_map_for_sites(
            gold_group_to_site_alarms.get(item.get("gold_id", ""), {}),
            missing_sites,
        )
        case_associated_sites = matched_sites
        case_missing_sites = missing_sites
        case_ticket_sites = gold_sites
        case_details.append({
            "ticket_id": item.get("gold_id", ""),
            "ticket_site_count": len(case_ticket_sites),
            "ticket_sites": case_ticket_sites,
            "fault_groups": effective_predicted_groups,
            "effective_fault_groups": effective_predicted_groups,
            "selected_fault_group": item.get("selected_predicted_group", ""),
            "associated_site_count": len(case_associated_sites),
            "associated_sites": case_associated_sites,
            "associated_site_alarms": associated_site_alarms,
            "missing_site_count": len(case_missing_sites),
            "missing_sites": case_missing_sites,
            "missing_site_alarms": missing_site_alarms,
            "recall": item.get("recall", 0.0),
            "note": _format_case_sites_note(case_associated_sites, case_missing_sites),
        })
    return case_details


def _filter_metric_details_to_unrecalled(metric_result):
    """仅过滤输出明细，不改变已经计算好的整体指标。"""
    if not isinstance(metric_result, dict):
        return metric_result
    details = metric_result.get("details", [])
    if not isinstance(details, list):
        return metric_result

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
    gold_alarm_domains=None,
    gold_has_offline=None,
    site_has_no_domain=None,
    site_has_required_domain=None,
    no_domain_alarm="",
    no_domain_site="",
    require_domain_per_site="",
    only_offline=False,
    only_one=False,
    loose_gold_to_pred_groups=None,
    potential_gold_to_pred_groups=None,
    gold_to_alarm_ids=None,
    gold_to_site_alarm_ids=None,
    pred_group_to_alarm_ids=None,
    pred_side_alarm_universe=None,
    progress_label="",
):
    details = []
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_alarm_recall = 0.0
    total_alarm_precision = 0.0
    total_alarm_f1 = 0.0
    gold_alarm_domains = gold_alarm_domains or {}
    gold_has_offline = gold_has_offline or {}
    site_has_no_domain = site_has_no_domain or {}
    site_has_required_domain = site_has_required_domain or {}
    loose_gold_to_pred_groups = loose_gold_to_pred_groups or {}
    potential_gold_to_pred_groups = potential_gold_to_pred_groups or {}
    gold_to_alarm_ids = gold_to_alarm_ids or {}
    gold_to_site_alarm_ids = gold_to_site_alarm_ids or {}
    pred_group_to_alarm_ids = pred_group_to_alarm_ids or {}
    pred_side_alarm_universe = pred_side_alarm_universe or set()

    gold_ids = sorted(gold_to_sites.keys())
    # 每个样本都要对命中的预测组做站点和告警两次集合并集，组大时并不便宜，给它一个进度。
    progress = ProgressBar(len(gold_ids), progress_label) if (progress_label and gold_ids) else None

    for gold_id in gold_ids:
        if progress is not None:
            progress.update()
        if no_domain_alarm and no_domain_alarm in gold_alarm_domains.get(gold_id, set()):
            continue
        gold_sites = set(gold_to_sites.get(gold_id, set()))
        if no_domain_site and any(site_has_no_domain.get(site_id, False) for site_id in gold_sites):
            continue
        if require_domain_per_site:
            gold_sites = {
                site_id
                for site_id in gold_sites
                if site_has_required_domain.get(site_id, False)
            }
        if not gold_sites:
            continue
        if min_site_num > 0 and len(gold_sites) < min_site_num:
            continue
        if only_offline and not gold_has_offline.get(gold_id, False):
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

        # 告警级口径：站点级选组结果(effective_predicted_groups)保持复用，只把比较对象换成告警实例键，
        # 这样两种粒度描述的是同一组配对关系，差异只来自粒度本身。
        gold_alarm_ids = _resolve_gold_alarm_ids(
            gold_id,
            gold_to_alarm_ids,
            gold_to_site_alarm_ids,
            gold_sites,
            bool(require_domain_per_site),
        )
        predicted_alarm_ids = set()
        for predicted_group_id in effective_predicted_groups:
            predicted_alarm_ids.update(pred_group_to_alarm_ids.get(predicted_group_id, set()))

        matched_alarm_ids, alarm_recall, alarm_precision, alarm_f1 = _compute_site_metrics(
            gold_alarm_ids,
            predicted_alarm_ids,
        )
        # 对侧告警全集里根本不存在的 gold 告警：这部分不是分组分歧，而是另一侧压根没见过这条告警
        # （例如 MHP 过滤掉了，或该告警没有故障组ID）。单列出来便于把口径噪声和真实差距分开。
        gold_alarms_missing_from_pred_universe = gold_alarm_ids - pred_side_alarm_universe

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
            "gold_alarm_count": len(gold_alarm_ids),
            "predicted_alarm_count": len(predicted_alarm_ids),
            "matched_alarm_count": len(matched_alarm_ids),
            "gold_alarms_missing_from_pred_universe_count": len(gold_alarms_missing_from_pred_universe),
            "alarm_recall": alarm_recall,
            "alarm_precision": alarm_precision,
            "alarm_f1": alarm_f1,
        })

        total_recall += recall
        total_precision += precision
        total_f1 += f1
        total_alarm_recall += alarm_recall
        total_alarm_precision += alarm_precision
        total_alarm_f1 += alarm_f1

    if progress is not None:
        progress.close()

    details.sort(
        key=lambda item: (
            item.get("recall", 0.0),
            -item.get("gold_site_count", 0),
            item.get("gold_id", ""),
        )
    )

    evaluated_count = len(details)
    return {
        "sample_count": evaluated_count,
        "gold_site_count_distribution": _build_gold_site_count_distribution(details),
        "average_recall": total_recall / evaluated_count if evaluated_count else 0.0,
        "average_precision": total_precision / evaluated_count if evaluated_count else 0.0,
        "average_f1": total_f1 / evaluated_count if evaluated_count else 0.0,
        "average_alarm_recall": total_alarm_recall / evaluated_count if evaluated_count else 0.0,
        "average_alarm_precision": total_alarm_precision / evaluated_count if evaluated_count else 0.0,
        "average_alarm_f1": total_alarm_f1 / evaluated_count if evaluated_count else 0.0,
        "gold_alarms_missing_from_pred_universe_total": sum(
            item.get("gold_alarms_missing_from_pred_universe_count", 0) for item in details
        ),
        "gold_alarm_total": sum(item.get("gold_alarm_count", 0) for item in details),
        "details": details,
    }


def compute_ultimate_group_alarm_group_metrics(
    group_output_input,
    alarm_input,
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
    potential=False,
    only_unrecalled_predictions=False,
    output_file=None,
    ultimate_case_jsonl_output_file=None,
    alarm_group_case_jsonl_output_file=None,
):
    stage_total = 3 + int(bool(loose)) + int(bool(potential))
    current_stage = 1
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
    print(f"阶段 {current_stage}/{stage_total}：加载 group output 最新版本并提取终极 group...")
    group_records = _load_latest_group_records(group_output_input)
    (
        ultimate_group_to_sites,
        ultimate_group_to_alarm_groups,
        alarm_group_to_ultimate_groups,
        ultimate_group_to_alarm_ids,
        ultimate_group_to_site_alarms,
        ultimate_group_alarm_domains,
        ultimate_group_has_offline,
        alarm_id_to_ultimate_groups,
    ) = _build_ultimate_group_indexes(
        group_records,
        group_field=group_field,
        ne_to_domain=ne_to_domain,
    )
    current_stage += 1

    print(f"阶段 {current_stage}/{stage_total}：从原始告警流提取告警故障组ID覆盖站点...")
    alarm_group_to_sites, alarm_group_to_alarm_ids, alarm_group_to_site_alarms, alarm_group_alarm_domains, alarm_group_has_offline, alarm_id_to_alarm_groups = _build_alarm_group_site_index(
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
    ultimate_site_alarm_ids = _build_group_site_alarm_ids(ultimate_group_to_site_alarms)
    alarm_group_site_alarm_ids = _build_group_site_alarm_ids(alarm_group_to_site_alarms)
    ultimate_alarm_universe = set(alarm_id_to_ultimate_groups.keys())
    alarm_group_alarm_universe = set(alarm_id_to_alarm_groups.keys())

    ultimate_as_gold = _compute_direction_metrics(
        gold_to_sites=ultimate_group_to_sites,
        gold_to_pred_groups=ultimate_group_to_alarm_groups,
        pred_group_to_sites=alarm_group_to_sites,
        min_site_num=min_site_num,
        gold_alarm_domains=ultimate_group_alarm_domains,
        gold_has_offline=ultimate_group_has_offline,
        site_has_no_domain=site_has_no_domain,
        site_has_required_domain=site_has_required_domain,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        only_offline=only_offline,
        only_one=only_one,
        loose_gold_to_pred_groups=ultimate_group_to_loose_alarm_groups,
        potential_gold_to_pred_groups=ultimate_group_to_potential_alarm_groups,
        gold_to_alarm_ids=ultimate_group_to_alarm_ids,
        gold_to_site_alarm_ids=ultimate_site_alarm_ids,
        pred_group_to_alarm_ids=alarm_group_to_alarm_ids,
        pred_side_alarm_universe=alarm_group_alarm_universe,
        progress_label="计算指标 ultimate_group_as_gold",
    )
    alarm_group_as_gold = _compute_direction_metrics(
        gold_to_sites=alarm_group_to_sites,
        gold_to_pred_groups=alarm_group_to_ultimate_groups,
        pred_group_to_sites=ultimate_group_to_sites,
        min_site_num=min_site_num,
        gold_alarm_domains=alarm_group_alarm_domains,
        gold_has_offline=alarm_group_has_offline,
        site_has_no_domain=site_has_no_domain,
        site_has_required_domain=site_has_required_domain,
        no_domain_alarm=no_domain_alarm,
        no_domain_site=no_domain_site,
        require_domain_per_site=require_domain_per_site,
        only_offline=only_offline,
        only_one=only_one,
        loose_gold_to_pred_groups=alarm_group_to_loose_ultimate_groups,
        potential_gold_to_pred_groups=alarm_group_to_potential_ultimate_groups,
        gold_to_alarm_ids=alarm_group_to_alarm_ids,
        gold_to_site_alarm_ids=alarm_group_site_alarm_ids,
        pred_group_to_alarm_ids=ultimate_group_to_alarm_ids,
        pred_side_alarm_universe=ultimate_alarm_universe,
        progress_label="计算指标 alarm_group_as_gold",
    )

    if only_unrecalled_predictions:
        _filter_metric_details_to_unrecalled(ultimate_as_gold)
        _filter_metric_details_to_unrecalled(alarm_group_as_gold)

    result = {
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
        "potential_mode": potential,
        "only_unrecalled_predictions_mode": only_unrecalled_predictions,
        "ultimate_group_count": len(ultimate_group_to_sites),
        "alarm_group_count": len(alarm_group_to_sites),
        "alarm_identity_overlap": _build_alarm_identity_overlap(
            ultimate_alarm_universe,
            alarm_group_alarm_universe,
        ),
        "ultimate_group_as_gold": ultimate_as_gold,
        "alarm_group_as_gold": alarm_group_as_gold,
    }

    ultimate_case_details = _build_case_details_for_direction(
        ultimate_as_gold["details"],
        ultimate_group_to_site_alarms,
        alarm_group_to_site_alarms,
    )
    alarm_group_case_details = _build_case_details_for_direction(
        alarm_group_as_gold["details"],
        alarm_group_to_site_alarms,
        ultimate_group_to_site_alarms,
    )
    ultimate_case_records = build_unrecalled_visualization_cases(
        ultimate_case_details,
        "ultimate_group_as_gold",
        ne_graph_data=ne_graph_data,
    )
    alarm_group_case_records = build_unrecalled_visualization_cases(
        alarm_group_case_details,
        "alarm_group_as_gold",
        ne_graph_data=ne_graph_data,
    )

    if output_file and not ultimate_case_jsonl_output_file:
        ultimate_case_jsonl_output_file = _derive_case_output_path(output_file, "ultimate_group_as_gold")
    if output_file and not alarm_group_case_jsonl_output_file:
        alarm_group_case_jsonl_output_file = _derive_case_output_path(output_file, "alarm_group_as_gold")

    if ultimate_case_jsonl_output_file:
        write_jsonl_records(ultimate_case_jsonl_output_file, ultimate_case_records)
        result["ultimate_group_as_gold_case_jsonl_output"] = ultimate_case_jsonl_output_file
        result["ultimate_group_as_gold_case_count"] = len(ultimate_case_records)

    if alarm_group_case_jsonl_output_file:
        write_jsonl_records(alarm_group_case_jsonl_output_file, alarm_group_case_records)
        result["alarm_group_as_gold_case_jsonl_output"] = alarm_group_case_jsonl_output_file
        result["alarm_group_as_gold_case_count"] = len(alarm_group_case_records)

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
        help="match_rules.py 的输出文件，或流式推理的 visual jsonl（如 stream_alarm_period_mhp.py 的 *.visual.jsonl），支持 jsonl/zip/目录",
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
        help="如果当前 gold label 中出现来自指定 domain 的告警，则跳过该样本，例如: --no-domain-alarm DATA",
    )
    parser.add_argument(
        "--no-domain-site",
        metavar="DOMAIN",
        help="如果当前 gold label 的任一站点在 ne_graph.json 中包含指定 domain 设备，则跳过该样本，例如: --no-domain-site DATA",
    )
    parser.add_argument(
        "--require-domain-per-site",
        metavar="DOMAIN",
        help="先从 gold label 站点里剔除不包含指定 domain 设备的站点；过滤后若站点数不足 min-site-num，则跳过该样本，例如: --require-domain-per-site TRANSMISSION",
    )
    parser.add_argument(
        "--only-offline",
        action="store_true",
        help="仅统计包含 OFFLINE_ALARMS 的 gold label 样本",
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
        default=900,
        help="loose 模式使用的前后对称时间窗，单位秒，默认: 900",
    )
    parser.add_argument(
        "--potential",
        action="store_true",
        help="允许根据告警ID命中关系，把另一侧包含这些告警的额外 group 作为 potential 预测结果并入",
    )
    parser.add_argument(
        "--only-unrecalled-predictions",
        action="store_true",
        help="输出 JSON 中两类 details 仅保留召回率不足 100%% 的预测；平均指标仍基于全部样本计算",
    )
    parser.add_argument(
        "--ultimate-case-jsonl-output",
        help="终极 group 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成同名 sidecar",
    )
    parser.add_argument(
        "--alarm-group-case-jsonl-output",
        help="告警故障组ID 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成同名 sidecar",
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
        no_domain_alarm=args.no_domain_alarm,
        no_domain_site=args.no_domain_site,
        require_domain_per_site=args.require_domain_per_site,
        only_offline=args.only_offline,
        only_one=args.only_one,
        loose=args.loose,
        window_seconds=args.window_seconds,
        potential=args.potential,
        only_unrecalled_predictions=args.only_unrecalled_predictions,
        output_file=args.output,
        ultimate_case_jsonl_output_file=args.ultimate_case_jsonl_output,
        alarm_group_case_jsonl_output_file=args.alarm_group_case_jsonl_output,
    )

    overlap = result["alarm_identity_overlap"]
    print("【告警实例键重合度】")
    print(f"终极group侧告警数: {overlap['ultimate_side_alarm_count']}")
    print(f"告警故障组ID侧告警数: {overlap['alarm_group_side_alarm_count']}")
    print(f"两侧共有告警数: {overlap['shared_alarm_count']} (Jaccard {overlap['jaccard']:.6f})")
    if overlap["shared_alarm_count"] == 0:
        print("警告: 两侧没有任何共有告警，告警级指标不可信，请确认 group output 与 alarms 来自同一份告警导出")

    for title, key in (
        ("终极 group 作为 gold", "ultimate_group_as_gold"),
        ("告警故障组ID 作为 gold", "alarm_group_as_gold"),
    ):
        section = result[key]
        print(f"【{title}】")
        print(f"样本数: {section['sample_count']}")
        print(f"gold站点数分布: {section['gold_site_count_distribution']}")
        print(f"[站点级] 平均召回率: {section['average_recall']:.6f}")
        print(f"[站点级] 平均准确率: {section['average_precision']:.6f}")
        print(f"[站点级] 平均F1: {section['average_f1']:.6f}")
        print(f"[告警级] 平均召回率: {section['average_alarm_recall']:.6f}")
        print(f"[告警级] 平均准确率: {section['average_alarm_precision']:.6f}")
        print(f"[告警级] 平均F1: {section['average_alarm_f1']:.6f}")
        print(
            f"gold告警中对侧全集不存在的条数: {section['gold_alarms_missing_from_pred_universe_total']}"
            f" / {section['gold_alarm_total']}"
        )

    print(f"结果已输出到: {args.output}")
    if result.get("ultimate_group_as_gold_case_jsonl_output"):
        print(
            f"终极group-case jsonl: {result['ultimate_group_as_gold_case_jsonl_output']} "
            f"({result.get('ultimate_group_as_gold_case_count', 0)} 条)"
        )
    if result.get("alarm_group_as_gold_case_jsonl_output"):
        print(
            f"告警故障组ID-case jsonl: {result['alarm_group_as_gold_case_jsonl_output']} "
            f"({result.get('alarm_group_as_gold_case_count', 0)} 条)"
        )


if __name__ == "__main__":
    main()
