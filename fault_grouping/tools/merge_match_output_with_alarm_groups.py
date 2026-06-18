#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 match_rules.py 输出与原始告警中的故障组做合并。

逻辑：
1. 读取 match_rules.py 输出（JSONL），提取每个故障组的告警实例键集合；
   若 symptom 中带有 eid_list，也按同一上下文展开纳入索引。
2. 读取原始告警，按故障组ID分组，并提取每个原始故障组的告警实例键集合；
3. 以“输出故障组”和“原始故障组”为两类节点，若二者存在任意重合告警实例键，则连一条边；
   同时将输出故障组之间的 related_group_uuids 也作为连通边；
4. 对这个图求连通分量：
   - 同一连通分量中的多个输出故障组会先彼此合并；
   - 连通分量中的原始故障组再整体并入这个合并结果；
   - 这样可以自然覆盖“一个原始组连接多个输出组”以及“通过多个原始组迭代串联多个输出组”的情况。
   - 若分量内存在未被其它输出组引用的终极输出组，优先用它作为合并后顶层 uuid。
5. 若原始故障组与任何输出故障组都没有重合，默认按输出格式构造一条新记录；
   如果启用 --retain-output，则丢弃这类不包含输出故障组的独立原始组。
6. 最终输出保留：所有包含输出故障组的连通分量合并记录；
   未匹配输出组会原样保留，独立原始组是否保留由 --retain-output 控制。

输出格式与 match_rules.py 保持一致，可直接在 fault_group_browser.html 中加载。

用法：
    python fault_grouping/tools/merge_match_output_with_alarm_groups.py \
        --match-output match_groups.jsonl \
        --alarms alarms.jsonl \
        --ne-graph topology_resources/ne_graph.json \
        --site-graph topology_resources/site_graph.json \
        -o merged_groups.jsonl
"""

import argparse
import copy
import json
from fault_grouping.alarm_events.identity import alarm_content_uuid, require_alarm_identity
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_tools.progress_utils import ProgressBar
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.recall_common import _normalize_text, _parse_group_ids


ALARM_GROUP_RULE_PREFIX = "alarm_group_"
ALARM_GROUP_RULE_SUFFIX = "_rule"


def _build_alarm_group_rule_name(group_id):
    normalized_group_id = _normalize_text(group_id)
    if not normalized_group_id:
        normalized_group_id = "unknown"
    return f"{ALARM_GROUP_RULE_PREFIX}{normalized_group_id}{ALARM_GROUP_RULE_SUFFIX}"


def _load_json_file_if_exists(filepath):
    path = Path(filepath) if filepath else None
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_alarm_site_id(alarm, ne_graph_data):
    site_id = _normalize_text(alarm.get("站点ID", ""))
    if site_id:
        return site_id

    alarm_source = _normalize_text(alarm.get("告警源", ""))
    if not alarm_source:
        return ""

    ne_info = ne_graph_data.get(alarm_source, {}) if isinstance(ne_graph_data, dict) else {}
    return _normalize_text(ne_info.get("site_id", ""))


def _build_site_placeholder_ne_id(site_id):
    return f"SITE::{site_id}"


def _resolve_alarm_ne_id(alarm, site_id):
    alarm_source = _normalize_text(alarm.get("告警源", ""))
    if alarm_source:
        return alarm_source
    if site_id:
        return _build_site_placeholder_ne_id(site_id)
    return ""


def _build_ne_meta_from_context(ne_id, site_id, ne_graph_data, site_graph_data):
    ne_graph_entry = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    site_graph_entry = site_graph_data.get(site_id, {}) if site_id and isinstance(site_graph_data, dict) else {}
    is_placeholder = ne_id.startswith("SITE::")
    return {
        "link": copy.deepcopy(ne_graph_entry.get("link", {})) if isinstance(ne_graph_entry.get("link", {}), dict) else {},
        "group": "",
        "name": ne_graph_entry.get("name", ne_id if not is_placeholder else site_id),
        "site_id": site_id,
        "site_name": ne_graph_entry.get("site_name", "") or site_graph_entry.get("site_name", "") or site_id,
        "type": str(ne_graph_entry.get("type", "")).upper(),
        "network_type": str(ne_graph_entry.get("network_type", "")).upper(),
        "manufacturer": str(ne_graph_entry.get("manufacturer", "")).upper(),
        "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
        "domain": str(ne_graph_entry.get("domain", "")).upper(),
        "region_id": ne_graph_entry.get("region_id", "") or site_graph_entry.get("region_id", ""),
        "longitude": ne_graph_entry.get("longitude", "") or site_graph_entry.get("longitude", ""),
        "latitude": ne_graph_entry.get("latitude", "") or site_graph_entry.get("latitude", ""),
        "alarm": [],
    }


def _parse_datetime_to_ts(text):
    """尝试把告警时间字符串转成 timestamp。"""
    text = _normalize_text(text)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("T", " ")).timestamp()
    except ValueError:
        return None


def _load_match_groups(jsonl_path):
    """加载 match_rules.py 输出，返回列表，每项为原始 dict。"""
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"match-output 文件不存在: {jsonl_path}")

    # 先估算行数用于进度条
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    groups = []
    progress = ProgressBar(total_lines, "加载 match_rules 输出")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            progress.update()
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ 跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            groups.append(record)
    progress.close()
    return groups


def _normalize_alarm_time_value(value):
    text = _normalize_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        number = None
    if number is not None and number == number and number not in (float("inf"), float("-inf")):
        return datetime.fromtimestamp(number).strftime("%Y-%m-%d %H:%M:%S")

    normalized_text = text.replace("T", " ")
    if "." in normalized_text:
        normalized_text = normalized_text.split(".", 1)[0]
    return normalized_text


def _build_alarm_instance_keys(
    alarm_ids,
    *,
    occurrence_uuid,
):
    keys = set()
    for alarm_id in alarm_ids:
        normalized_alarm_id = _normalize_text(alarm_id)
        if not normalized_alarm_id:
            continue
        keys.add(require_alarm_identity({
            "eid": normalized_alarm_id,
            "occurrence_uuid": occurrence_uuid,
        }))
    return keys


def _extract_alarm_keys_from_symptom(symptom):
    alarm_ids = [
        _normalize_text(symptom.get("eid", "") or symptom.get("alarm_id", "") or symptom.get("告警编码ID", ""))
    ]
    alarm_ids.extend(symptom.get("eid_list") or [])
    return _build_alarm_instance_keys(
        alarm_ids,
        occurrence_uuid=symptom.get("occurrence_uuid"),
    )


def _extract_alarm_keys_from_alarm_record(alarm_record, default_ne_id=""):
    return _build_alarm_instance_keys(
        [
            alarm_record.get("alarm_id", "")
            or alarm_record.get("eid", "")
            or alarm_record.get("告警编码ID", "")
        ] + list(alarm_record.get("alarm_id_list") or []),
        occurrence_uuid=alarm_record.get("occurrence_uuid"),
    )


def _extract_alarm_keys_from_match_group(record):
    """从 match_rules 输出中提取所有真实告警实例键。"""
    alarm_keys = set()
    for symptom in record.get("symptoms", []):
        if not isinstance(symptom, dict):
            continue
        alarm_keys.update(_extract_alarm_keys_from_symptom(symptom))

    for ne_id, ne_meta in (record.get("ne_info", {}) or {}).items():
        if not isinstance(ne_meta, dict):
            continue
        for alarm_record in ne_meta.get("alarm", []) or []:
            if not isinstance(alarm_record, dict):
                continue
            alarm_keys.update(
                _extract_alarm_keys_from_alarm_record(alarm_record, default_ne_id=ne_id)
            )
    return alarm_keys


def _extract_match_group_uuid(record):
    match_info = record.get("match_info", {}) or {}
    if not isinstance(match_info, dict):
        return ""
    return _normalize_text(match_info.get("uuid", ""))


def _extract_related_group_uuids(record):
    match_info = record.get("match_info", {}) or {}
    if not isinstance(match_info, dict):
        return set()
    related_group_uuids = match_info.get("related_group_uuids", []) or []
    if not isinstance(related_group_uuids, list):
        return set()
    return {
        normalized_uuid
        for normalized_uuid in (_normalize_text(uuid_text) for uuid_text in related_group_uuids)
        if normalized_uuid
    }


def _extract_alarm_keys_from_alarm_group(alarm_list):
    alarm_keys = set()
    for alarm in alarm_list:
        if not isinstance(alarm, dict):
            continue
        alarm_keys.update(_extract_alarm_keys_from_alarm_record(alarm))
    return alarm_keys


def _load_alarm_groups(alarm_input, group_field="故障组ID"):
    """
    读取原始告警，按 group_field 分组。
    返回: dict[group_id] -> list[alarm_dict]
    """
    alarm_groups = defaultdict(list)
    print("扫描原始告警并分组...")
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        alarm = dict(alarm)
        alarm["occurrence_uuid"] = alarm_content_uuid(alarm)
        raw_group_value = alarm.get(group_field)
        group_ids = _parse_group_ids(raw_group_value)
        if not group_ids:
            continue
        for gid in group_ids:
            alarm_groups[gid].append(alarm)
    return alarm_groups


def _alarm_to_symptom(alarm, ne_graph_data=None):
    """把原始告警转换为 match_rules 输出中的 symptom 格式。"""
    ts = _parse_datetime_to_ts(alarm.get("告警首次发生时间", ""))
    site_id = _resolve_alarm_site_id(alarm, ne_graph_data or {})
    ne_id = _resolve_alarm_ne_id(alarm, site_id)
    return {
        "node": site_id,
        "alarm": _normalize_text(alarm.get("告警标题", "")),
        "ts": ts,
        "eid": _normalize_text(alarm.get("告警编码ID", "")),
        "occurrence_uuid": alarm["occurrence_uuid"],
        "alarm_source": ne_id,
        "matched_role": "",
        "工单号": _normalize_text(alarm.get("工单号", "")),
        "故障组ID": _normalize_text(alarm.get("故障组ID", "")),
        "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
    }


def _extract_alarm_group_meta(alarm_list, ne_graph_data=None):
    """从原始告警列表中提取站点列表、NE 列表等元信息。"""
    ne_graph_data = ne_graph_data or {}
    site_set = set()
    ne_set = set()
    for alarm in alarm_list:
        site_id = _resolve_alarm_site_id(alarm, ne_graph_data)
        ne_id = _resolve_alarm_ne_id(alarm, site_id)
        if site_id:
            site_set.add(site_id)
        if ne_id:
            ne_set.add(ne_id)
    return sorted(site_set), sorted(ne_set)


def _build_ne_info_from_alarms(alarm_list, ne_graph_data=None, site_graph_data=None):
    """根据原始告警构造简化的 ne_info（与 match_rules 输出格式对齐）。"""
    ne_graph_data = ne_graph_data or {}
    site_graph_data = site_graph_data or {}
    ne_info = {}
    for alarm in alarm_list:
        site_id = _resolve_alarm_site_id(alarm, ne_graph_data)
        ne_id = _resolve_alarm_ne_id(alarm, site_id)
        if not ne_id:
            continue
        if ne_id not in ne_info:
            ne_info[ne_id] = _build_ne_meta_from_context(ne_id, site_id, ne_graph_data, site_graph_data)
        site_graph_entry = site_graph_data.get(site_id, {}) if site_id else {}
        ne_graph_entry = ne_graph_data.get(ne_id, {}) if ne_id else {}
        ne_info[ne_id]["alarm"].append({
            "alarm_id": _normalize_text(alarm.get("告警编码ID", "")),
            "occurrence_uuid": alarm["occurrence_uuid"],
            "alarm_type": _normalize_text(alarm.get("告警标题", "")),
            "alarm_time": _normalize_text(alarm.get("告警首次发生时间", "")),
            "alarm_clear_time": _normalize_text(alarm.get("告警清除时间", "")),
            "domain": str(ne_graph_entry.get("domain", "")).upper(),
            "site_id": site_id,
            "site_name": ne_graph_entry.get("site_name", "") or site_graph_entry.get("site_name", "") or site_id,
            "matched_role": "",
            "工单号": _normalize_text(alarm.get("工单号", "")),
            "故障组ID": _normalize_text(alarm.get("故障组ID", "")),
        })
    return ne_info


def _build_group_output_from_alarms(group_id, alarm_list, ne_graph_data=None, site_graph_data=None):
    """当原始告警故障组没有匹配到输出组时，按输出格式构造一条新记录。"""
    ne_graph_data = ne_graph_data or {}
    site_graph_data = site_graph_data or {}
    site_list, ne_list = _extract_alarm_group_meta(alarm_list, ne_graph_data=ne_graph_data)
    symptoms = [_alarm_to_symptom(a, ne_graph_data=ne_graph_data) for a in alarm_list]
    timestamps = [s["ts"] for s in symptoms if s["ts"] is not None]
    anchor_ts = min(timestamps) if timestamps else None
    new_uuid = f"alarm-{group_id}" if group_id else f"alarm-{uuid.uuid4().hex[:12]}"
    alarm_group_rule = _build_alarm_group_rule_name(group_id)

    return {
        "match_info": {
            "uuid": new_uuid,
            "rule": alarm_group_rule,
            "merged_rules": [],
            "related_group_uuids": [],
            "inferred_roots": {},
            "role_mapping": {},
        },
        "ne_info": _build_ne_info_from_alarms(
            alarm_list,
            ne_graph_data=ne_graph_data,
            site_graph_data=site_graph_data,
        ),
        "group_info": {
            new_uuid: {
                "ne_list": ne_list,
                "site_list": site_list,
            }
        },
        "symptoms": symptoms,
        "group_anchor_ts": anchor_ts,
        "group_anchor_time": (
            datetime.fromtimestamp(anchor_ts).strftime("%Y-%m-%d %H:%M:%S")
            if anchor_ts is not None else ""
        ),
    }


def _symptom_merge_key(symptom):
    return require_alarm_identity(symptom)


def _alarm_record_merge_key(alarm_record):
    return require_alarm_identity(alarm_record)


def _merge_alarm_record_lists(existing_list, incoming_list):
    merged = {}
    ordered_keys = []
    for alarm_record in list(existing_list or []) + list(incoming_list or []):
        key = _alarm_record_merge_key(alarm_record)
        if key not in merged:
            merged[key] = copy.deepcopy(alarm_record)
            ordered_keys.append(key)
            continue
        target = merged[key]
        for field, value in alarm_record.items():
            if field == "alarm_id_list":
                merged_ids = [
                    alarm_id
                    for alarm_id in list(target.get("alarm_id_list") or []) + list(value or [])
                    if _normalize_text(alarm_id)
                ]
                deduped_ids = []
                seen = set()
                for alarm_id in merged_ids:
                    normalized_alarm_id = _normalize_text(alarm_id)
                    if normalized_alarm_id in seen:
                        continue
                    seen.add(normalized_alarm_id)
                    deduped_ids.append(normalized_alarm_id)
                if deduped_ids:
                    target["alarm_id_list"] = deduped_ids
            elif not target.get(field) and value not in ("", None, [], {}):
                target[field] = copy.deepcopy(value)
    return [merged[key] for key in ordered_keys]


def _merge_ne_info(existing_ne_info, incoming_ne_info):
    for ne_id, ne_meta in (incoming_ne_info or {}).items():
        if ne_id not in existing_ne_info:
            existing_ne_info[ne_id] = copy.deepcopy(ne_meta)
            continue

        target_meta = existing_ne_info[ne_id]
        for field, value in ne_meta.items():
            if field == "alarm":
                target_meta["alarm"] = _merge_alarm_record_lists(
                    target_meta.get("alarm", []),
                    value or [],
                )
            elif field == "link":
                target_links = target_meta.setdefault("link", {})
                for neighbor_id, link_meta in (value or {}).items():
                    if neighbor_id not in target_links:
                        target_links[neighbor_id] = copy.deepcopy(link_meta)
            elif not target_meta.get(field) and value not in ("", None, [], {}):
                target_meta[field] = copy.deepcopy(value)


def _merge_group_info(existing_group_info, incoming_group_info):
    for group_id, group_meta in (incoming_group_info or {}).items():
        if group_id not in existing_group_info:
            existing_group_info[group_id] = copy.deepcopy(group_meta)
            continue

        target_meta = existing_group_info[group_id]
        if not isinstance(target_meta, dict) or not isinstance(group_meta, dict):
            continue
        target_meta["site_list"] = sorted(
            set(target_meta.get("site_list", [])) | set(group_meta.get("site_list", []))
        )
        target_meta["ne_list"] = sorted(
            set(target_meta.get("ne_list", [])) | set(group_meta.get("ne_list", []))
        )


def _synchronize_primary_group_info(record):
    match_info = record.get("match_info", {}) or {}
    primary_group_id = _normalize_text(match_info.get("uuid", ""))
    if not primary_group_id:
        return

    group_info = record.setdefault("group_info", {})
    all_site_ids = set()
    all_ne_ids = set()
    for group_meta in group_info.values():
        if not isinstance(group_meta, dict):
            continue
        all_site_ids.update(
            _normalize_text(site_id)
            for site_id in group_meta.get("site_list", [])
            if _normalize_text(site_id)
        )
        all_ne_ids.update(
            _normalize_text(ne_id)
            for ne_id in group_meta.get("ne_list", [])
            if _normalize_text(ne_id)
        )

    if not all_site_ids:
        all_site_ids.update(
            _normalize_text(symptom.get("node", ""))
            for symptom in record.get("symptoms", [])
            if isinstance(symptom, dict) and _normalize_text(symptom.get("node", ""))
        )

    if not all_ne_ids:
        all_ne_ids.update(
            _normalize_text(ne_id)
            for ne_id in (record.get("ne_info", {}) or {}).keys()
            if _normalize_text(ne_id)
        )

    group_info[primary_group_id] = {
        "site_list": sorted(all_site_ids),
        "ne_list": sorted(all_ne_ids),
    }


def _merge_match_info(base_record, incoming_record):
    base_info = base_record.setdefault("match_info", {})
    incoming_info = incoming_record.get("match_info", {}) or {}

    related_group_uuids = set(base_info.get("related_group_uuids", []) or [])
    incoming_uuid = _normalize_text(incoming_info.get("uuid", ""))
    base_uuid = _normalize_text(base_info.get("uuid", ""))
    if incoming_uuid and incoming_uuid != base_uuid:
        related_group_uuids.add(incoming_uuid)
    related_group_uuids.update(
        _normalize_text(uuid_text)
        for uuid_text in (incoming_info.get("related_group_uuids", []) or [])
        if _normalize_text(uuid_text)
    )
    related_group_uuids.discard(base_uuid)
    if related_group_uuids:
        base_info["related_group_uuids"] = sorted(related_group_uuids)
    else:
        base_info.pop("related_group_uuids", None)

    merged_rules = []
    seen_rules = set()
    for rule_name in (
        [base_info.get("rule", "")]
        + list(base_info.get("merged_rules", []) or [])
        + [incoming_info.get("rule", "")]
        + list(incoming_info.get("merged_rules", []) or [])
    ):
        normalized_rule = _normalize_text(rule_name)
        if not normalized_rule or normalized_rule in seen_rules:
            continue
        seen_rules.add(normalized_rule)
        merged_rules.append(normalized_rule)
    if merged_rules:
        base_info["merged_rules"] = merged_rules

    for field_name in ("inferred_roots", "role_mapping"):
        target_value = base_info.setdefault(field_name, {})
        incoming_value = incoming_info.get(field_name, {}) or {}
        if isinstance(target_value, dict) and isinstance(incoming_value, dict):
            for key, value in incoming_value.items():
                if key not in target_value:
                    target_value[key] = copy.deepcopy(value)


def _recompute_group_anchor(record):
    timestamps = [symptom.get("ts") for symptom in record.get("symptoms", []) if symptom.get("ts") is not None]
    if not timestamps:
        record["group_anchor_ts"] = None
        record["group_anchor_time"] = ""
        return
    anchor_ts = min(timestamps)
    record["group_anchor_ts"] = anchor_ts
    record["group_anchor_time"] = datetime.fromtimestamp(anchor_ts).strftime("%Y-%m-%d %H:%M:%S")


def _merge_match_group_records(base_record, incoming_record):
    merged_symptoms = {}
    ordered_keys = []
    for symptom in list(base_record.get("symptoms", [])) + list(incoming_record.get("symptoms", [])):
        key = _symptom_merge_key(symptom)
        if key not in merged_symptoms:
            merged_symptoms[key] = copy.deepcopy(symptom)
            ordered_keys.append(key)
            continue
        target = merged_symptoms[key]
        if not target.get("eid") and symptom.get("eid"):
            target["eid"] = symptom.get("eid")
        if not target.get("eid_list") and symptom.get("eid_list"):
            target["eid_list"] = copy.deepcopy(symptom.get("eid_list"))
        for field_name in ("工单号", "故障组ID", "告警清除时间"):
            if not target.get(field_name) and symptom.get(field_name):
                target[field_name] = symptom.get(field_name)

    base_record["symptoms"] = [merged_symptoms[key] for key in ordered_keys]
    base_record["symptoms"].sort(key=lambda s: (s.get("ts") is None, s.get("ts") or float("inf"), s.get("eid", "")))

    _merge_group_info(
        base_record.setdefault("group_info", {}),
        incoming_record.get("group_info", {}),
    )
    _merge_ne_info(
        base_record.setdefault("ne_info", {}),
        incoming_record.get("ne_info", {}),
    )
    _merge_match_info(base_record, incoming_record)
    _synchronize_primary_group_info(base_record)
    _recompute_group_anchor(base_record)
    return base_record


def _merge_alarm_list_into_match_group(record, alarm_group_id, alarm_list, ne_graph_data=None, site_graph_data=None):
    """将原始告警列表合并进已有的 match_rules 输出记录。"""
    ne_graph_data = ne_graph_data or {}
    site_graph_data = site_graph_data or {}
    match_info = record.setdefault("match_info", {})
    alarm_group_rule = _build_alarm_group_rule_name(alarm_group_id)
    merged_rules = []
    seen_rules = set()
    for rule_name in (
        [match_info.get("rule", "")]
        + list(match_info.get("merged_rules", []) or [])
        + [alarm_group_rule]
    ):
        normalized_rule = _normalize_text(rule_name)
        if not normalized_rule or normalized_rule in seen_rules:
            continue
        seen_rules.add(normalized_rule)
        merged_rules.append(normalized_rule)
    if merged_rules:
        match_info["merged_rules"] = merged_rules

    existing_symptoms = list(record.get("symptoms", []))
    for alarm in alarm_list:
        existing_symptoms.append(_alarm_to_symptom(alarm, ne_graph_data=ne_graph_data))

    merged_symptoms = {}
    ordered_keys = []
    for symptom in existing_symptoms:
        key = _symptom_merge_key(symptom)
        if key in merged_symptoms:
            continue
        merged_symptoms[key] = copy.deepcopy(symptom)
        ordered_keys.append(key)
    record["symptoms"] = [merged_symptoms[key] for key in ordered_keys]
    record["symptoms"].sort(key=lambda s: (s.get("ts") is None, s.get("ts") or float("inf"), s.get("eid", "")))

    alarm_group_uuid = f"alarm-{alarm_group_id}" if alarm_group_id else f"alarm-{uuid.uuid4().hex[:12]}"
    site_list, ne_list = _extract_alarm_group_meta(alarm_list, ne_graph_data=ne_graph_data)
    _merge_group_info(
        record.setdefault("group_info", {}),
        {
            alarm_group_uuid: {
                "site_list": site_list,
                "ne_list": ne_list,
            }
        },
    )
    _merge_ne_info(
        record.setdefault("ne_info", {}),
        _build_ne_info_from_alarms(
            alarm_list,
            ne_graph_data=ne_graph_data,
            site_graph_data=site_graph_data,
        ),
    )

    record.setdefault("_merged_alarm_groups", []).append({
        "source": "alarm_group",
        "group_id": alarm_group_id,
        "alarm_count": len(alarm_list),
    })
    _synchronize_primary_group_info(record)
    _recompute_group_anchor(record)
    return record


def _build_match_related_adjacency(match_groups):
    uuid_to_match_indices = defaultdict(set)
    for match_idx, record in enumerate(match_groups):
        group_uuid = _extract_match_group_uuid(record)
        if group_uuid:
            uuid_to_match_indices[group_uuid].add(match_idx)

    related_adjacency = defaultdict(set)
    referenced_group_uuids = set()
    for match_idx, record in enumerate(match_groups):
        for related_uuid in _extract_related_group_uuids(record):
            referenced_group_uuids.add(related_uuid)
            for related_match_idx in uuid_to_match_indices.get(related_uuid, ()):
                if related_match_idx == match_idx:
                    continue
                related_adjacency[match_idx].add(related_match_idx)
                related_adjacency[related_match_idx].add(match_idx)

    return related_adjacency, referenced_group_uuids


def _select_component_base_match_index(match_indices, match_groups, referenced_group_uuids):
    for match_idx in match_indices:
        group_uuid = _extract_match_group_uuid(match_groups[match_idx])
        if group_uuid and group_uuid not in referenced_group_uuids:
            return match_idx
    return match_indices[0]


def _build_connected_components(
    match_group_alarm_keys,
    alarm_group_alarm_keys,
    alarm_key_to_match_indices,
    match_related_adjacency=None,
):
    match_related_adjacency = match_related_adjacency or {}
    alarm_key_to_alarm_group_ids = defaultdict(set)
    for alarm_group_id, alarm_keys in alarm_group_alarm_keys.items():
        for alarm_key in alarm_keys:
            alarm_key_to_alarm_group_ids[alarm_key].add(alarm_group_id)

    visited_match_indices = set()
    visited_alarm_group_ids = set()
    components = []

    for start_idx, alarm_keys in enumerate(match_group_alarm_keys):
        if start_idx in visited_match_indices:
            continue
        queue = [("match", start_idx)]
        visited_match_indices.add(start_idx)
        component_match_indices = set()
        component_alarm_group_ids = set()

        while queue:
            node_type, node_value = queue.pop()
            if node_type == "match":
                component_match_indices.add(node_value)
                current_alarm_keys = match_group_alarm_keys[node_value]
                for alarm_key in current_alarm_keys:
                    for alarm_group_id in alarm_key_to_alarm_group_ids.get(alarm_key, ()):
                        if alarm_group_id in visited_alarm_group_ids:
                            continue
                        visited_alarm_group_ids.add(alarm_group_id)
                        queue.append(("alarm", alarm_group_id))
                for related_match_idx in match_related_adjacency.get(node_value, ()):
                    if related_match_idx in visited_match_indices:
                        continue
                    visited_match_indices.add(related_match_idx)
                    queue.append(("match", related_match_idx))
            else:
                component_alarm_group_ids.add(node_value)
                current_alarm_keys = alarm_group_alarm_keys.get(node_value, set())
                for alarm_key in current_alarm_keys:
                    for match_idx in alarm_key_to_match_indices.get(alarm_key, ()):
                        if match_idx in visited_match_indices:
                            continue
                        visited_match_indices.add(match_idx)
                        queue.append(("match", match_idx))

        components.append({
            "match_indices": sorted(component_match_indices),
            "alarm_group_ids": sorted(component_alarm_group_ids),
        })

    standalone_alarm_group_ids = [
        alarm_group_id
        for alarm_group_id in sorted(alarm_group_alarm_keys)
        if alarm_group_id not in visited_alarm_group_ids
    ]
    return components, standalone_alarm_group_ids


def merge(
    match_output_path,
    alarm_input_path,
    output_path,
    group_field="故障组ID",
    retain_output=False,
    ne_graph_file=NE_GRAPH_JSON,
    site_graph_file=SITE_GRAPH_JSON,
):
    ne_graph_data = _load_json_file_if_exists(ne_graph_file)
    site_graph_data = _load_json_file_if_exists(site_graph_file)

    print("加载 match_rules 输出...")
    match_groups = _load_match_groups(match_output_path)
    print(f"  输出故障组数: {len(match_groups)}")

    print("构建告警实例键 -> 输出组索引...")
    alarm_key_progress = ProgressBar(len(match_groups), "构建告警实例键索引")
    match_group_alarm_keys = []
    alarm_key_to_match_indices = defaultdict(list)
    for idx, record in enumerate(match_groups):
        alarm_key_progress.update()
        current_alarm_keys = _extract_alarm_keys_from_match_group(record)
        match_group_alarm_keys.append(current_alarm_keys)
        for alarm_key in current_alarm_keys:
            alarm_key_to_match_indices[alarm_key].append(idx)
    alarm_key_progress.close()

    print("加载原始告警并按故障组ID分组...")
    alarm_groups = _load_alarm_groups(alarm_input_path, group_field=group_field)
    print(f"  原始告警故障组数: {len(alarm_groups)}")

    alarm_group_alarm_keys = {
        alarm_group_id: _extract_alarm_keys_from_alarm_group(alarm_list)
        for alarm_group_id, alarm_list in alarm_groups.items()
    }

    print("构建输出组-原始组/关联输出组连通关系...")
    match_related_adjacency, referenced_group_uuids = _build_match_related_adjacency(match_groups)
    components, standalone_alarm_group_ids = _build_connected_components(
        match_group_alarm_keys,
        alarm_group_alarm_keys,
        alarm_key_to_match_indices,
        match_related_adjacency=match_related_adjacency,
    )

    final_records = []
    merge_progress = ProgressBar(len(components), "按连通分量合并故障组")
    merged_output_component_count = 0
    merged_alarm_group_count = 0

    for component in components:
        merge_progress.update()
        match_indices = component["match_indices"]
        component_alarm_group_ids = component["alarm_group_ids"]

        if not match_indices:
            continue

        base_idx = _select_component_base_match_index(
            match_indices,
            match_groups,
            referenced_group_uuids,
        )
        merged_record = copy.deepcopy(match_groups[base_idx])

        for match_idx in match_indices:
            if match_idx == base_idx:
                continue
            _merge_match_group_records(merged_record, match_groups[match_idx])

        for alarm_group_id in component_alarm_group_ids:
            _merge_alarm_list_into_match_group(
                merged_record,
                alarm_group_id,
                alarm_groups[alarm_group_id],
                ne_graph_data=ne_graph_data,
                site_graph_data=site_graph_data,
            )

        if component_alarm_group_ids:
            representative_uuid = merged_record.get("match_info", {}).get("uuid", "")
            merge_progress.set_extra_text(
                f"输出组 {representative_uuid[:16]}... "
                f"合并 {len(match_indices)} 个输出组 + {len(component_alarm_group_ids)} 个原始组"
            )

        final_records.append(merged_record)
        if len(match_indices) > 1:
            merged_output_component_count += 1
        merged_alarm_group_count += len(component_alarm_group_ids)

    merge_progress.close()

    standalone_alarm_records = []
    if not retain_output:
        standalone_alarm_records = [
            _build_group_output_from_alarms(
                alarm_group_id,
                alarm_groups[alarm_group_id],
                ne_graph_data=ne_graph_data,
                site_graph_data=site_graph_data,
            )
            for alarm_group_id in standalone_alarm_group_ids
        ]
    final_records.extend(standalone_alarm_records)

    print(
        f"  合并后记录数: {len(final_records)} "
        f"(输出连通分量: {len(components)}, "
        f"跨输出组合并分量: {merged_output_component_count}, "
        f"并入原始组数: {merged_alarm_group_count}, "
        f"独立原始组: {len(standalone_alarm_group_ids)}, "
        f"保留独立原始组: {len(standalone_alarm_records)})"
    )

    # 写出
    write_progress = ProgressBar(len(final_records), "写出合并结果")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in final_records:
            write_progress.update()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_progress.close()

    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并 match_rules 输出与原始告警故障组")
    parser.add_argument("--match-output", required=True, help="match_rules.py 输出的 JSONL 文件")
    parser.add_argument("--alarms", required=True, help="原始告警输入（支持 jsonl/csv/zip/目录）")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--group-field", default="故障组ID", help="告警中的故障组字段名，默认: 故障组ID")
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于补齐原始告警 NE/site 信息的 ne_graph，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"用于补齐站点坐标的 site_graph，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--retain-output",
        action="store_true",
        help="只保留包含 match_rules 输出故障组的记录，丢弃独立原始告警故障组",
    )
    args = parser.parse_args()

    merge(
        args.match_output,
        args.alarms,
        args.output,
        group_field=args.group_field,
        retain_output=args.retain_output,
        ne_graph_file=args.ne_graph,
        site_graph_file=args.site_graph,
    )


if __name__ == "__main__":
    main()
