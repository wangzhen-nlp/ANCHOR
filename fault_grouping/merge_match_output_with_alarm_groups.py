#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 match_rules.py 输出与原始告警中的故障组做合并。

逻辑：
1. 读取 match_rules.py 输出（JSONL），提取每个故障组的 symptom eid 集合；
2. 读取原始告警，按故障组ID分组；
3. 若某原始告警故障组的任意 eid 与输出故障组重合，则把原始告警合并进该输出故障组；
   （按重合 eid 数取最佳匹配；若多个原始组指向同一输出组，累积合并）
4. 若原始告警故障组没有重合的输出故障组，则按输出格式构造一条新记录；
5. 最终输出保留：所有被合并过的输出组、未匹配的输出组、以及按原始告警新建的记录。

输出格式与 match_rules.py 保持一致，可直接在 fault_group_browser.html 中加载。

用法：
    python fault_grouping/merge_match_output_with_alarm_groups.py \
        --match-output match_groups.jsonl \
        --alarms alarms.jsonl \
        -o merged_groups.jsonl
"""

import argparse
import copy
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_tools.progress_utils import ProgressBar
from ticket_recall.evaluation.recall_common import _normalize_text, _parse_group_ids


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


def _extract_eids_from_match_group(record):
    """从 match_rules 输出中提取所有 symptom eid。"""
    eids = set()
    for symptom in record.get("symptoms", []):
        eid = _normalize_text(symptom.get("eid", ""))
        if eid:
            eids.add(eid)
        for raw_eid in symptom.get("eid_list", []) or []:
            normalized_eid = _normalize_text(raw_eid)
            if normalized_eid:
                eids.add(normalized_eid)
    return eids


def _extract_eids_from_alarm_group(alarm_list):
    return {
        _normalize_text(alarm.get("告警编码ID", ""))
        for alarm in alarm_list
        if _normalize_text(alarm.get("告警编码ID", ""))
    }


def _load_alarm_groups(alarm_input, group_field="故障组ID"):
    """
    读取原始告警，按 group_field 分组。
    返回: dict[group_id] -> list[alarm_dict]
    """
    alarm_groups = defaultdict(list)
    print("扫描原始告警并分组...")
    for alarm in stream_alarm_inputs(alarm_input, show_progress=True):
        raw_group_value = alarm.get(group_field)
        group_ids = _parse_group_ids(raw_group_value)
        if not group_ids:
            continue
        for gid in group_ids:
            alarm_groups[gid].append(alarm)
    return alarm_groups


def _alarm_to_symptom(alarm):
    """把原始告警转换为 match_rules 输出中的 symptom 格式。"""
    ts = _parse_datetime_to_ts(alarm.get("告警首次发生时间", ""))
    return {
        "node": _normalize_text(alarm.get("站点ID", "")),
        "alarm": _normalize_text(alarm.get("告警标题", "")),
        "ts": ts,
        "eid": _normalize_text(alarm.get("告警编码ID", "")),
        "alarm_source": _normalize_text(alarm.get("告警源", "")),
        "matched_role": "",
        "工单号": _normalize_text(alarm.get("工单号", "")),
        "故障组ID": _normalize_text(alarm.get("故障组ID", "")),
        "告警清除时间": _normalize_text(alarm.get("告警清除时间", "")),
    }


def _extract_alarm_group_meta(alarm_list):
    """从原始告警列表中提取站点列表、NE 列表等元信息。"""
    site_set = set()
    ne_set = set()
    for alarm in alarm_list:
        site_id = _normalize_text(alarm.get("站点ID", ""))
        ne_id = _normalize_text(alarm.get("告警源", ""))
        if site_id:
            site_set.add(site_id)
        if ne_id:
            ne_set.add(ne_id)
    return sorted(site_set), sorted(ne_set)


def _build_ne_info_from_alarms(alarm_list):
    """根据原始告警构造简化的 ne_info（与 match_rules 输出格式对齐）。"""
    ne_info = {}
    for alarm in alarm_list:
        ne_id = _normalize_text(alarm.get("告警源", ""))
        if not ne_id:
            continue
        if ne_id not in ne_info:
            ne_info[ne_id] = {
                "link": {},
                "group": "",
                "name": ne_id,
                "site_id": _normalize_text(alarm.get("站点ID", "")),
                "site_name": "",
                "type": "",
                "network_type": "",
                "manufacturer": "",
                "running_status": "",
                "domain": "",
                "region_id": "",
                "longitude": "",
                "latitude": "",
                "alarm": [],
            }
        ne_info[ne_id]["alarm"].append({
            "alarm_id": _normalize_text(alarm.get("告警编码ID", "")),
            "alarm_type": _normalize_text(alarm.get("告警标题", "")),
            "alarm_time": _normalize_text(alarm.get("告警首次发生时间", "")),
            "alarm_clear_time": _normalize_text(alarm.get("告警清除时间", "")),
            "domain": "",
            "site_id": _normalize_text(alarm.get("站点ID", "")),
            "site_name": "",
            "matched_role": "",
            "工单号": _normalize_text(alarm.get("工单号", "")),
            "故障组ID": _normalize_text(alarm.get("故障组ID", "")),
        })
    return ne_info


def _build_group_output_from_alarms(group_id, alarm_list):
    """当原始告警故障组没有匹配到输出组时，按输出格式构造一条新记录。"""
    site_list, ne_list = _extract_alarm_group_meta(alarm_list)
    symptoms = [_alarm_to_symptom(a) for a in alarm_list]
    timestamps = [s["ts"] for s in symptoms if s["ts"] is not None]
    anchor_ts = min(timestamps) if timestamps else None
    new_uuid = f"alarm-{group_id}" if group_id else f"alarm-{uuid.uuid4().hex[:12]}"

    return {
        "match_info": {
            "uuid": new_uuid,
            "rule": "alarm_group",
            "merged_rules": [],
            "related_group_uuids": [],
            "inferred_roots": {},
            "role_mapping": {},
        },
        "ne_info": _build_ne_info_from_alarms(alarm_list),
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
    matched_role = _normalize_text(symptom.get("matched_role", ""))
    eid_list = tuple(
        eid for eid in (_normalize_text(x) for x in (symptom.get("eid_list") or []))
        if eid
    )
    if eid_list:
        return ("eid_list", eid_list, matched_role)

    eid = _normalize_text(symptom.get("eid", ""))
    if eid:
        return ("eid", eid, matched_role)

    return (
        "fallback",
        _normalize_text(symptom.get("node", "")),
        _normalize_text(symptom.get("alarm", "")),
        symptom.get("ts"),
        _normalize_text(symptom.get("alarm_source", "")),
        matched_role,
    )


def _alarm_record_merge_key(alarm_record):
    matched_role = _normalize_text(alarm_record.get("matched_role", ""))
    alarm_id_list = tuple(
        alarm_id
        for alarm_id in (_normalize_text(x) for x in (alarm_record.get("alarm_id_list") or []))
        if alarm_id
    )
    if alarm_id_list:
        return ("alarm_id_list", alarm_id_list, matched_role)

    alarm_id = _normalize_text(alarm_record.get("alarm_id", ""))
    if alarm_id:
        return ("alarm_id", alarm_id, matched_role)

    return (
        "fallback",
        _normalize_text(alarm_record.get("alarm_type", "")),
        _normalize_text(alarm_record.get("alarm_time", "")),
        _normalize_text(alarm_record.get("site_id", "")),
        matched_role,
    )


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
    if related_group_uuids:
        base_info["related_group_uuids"] = sorted(related_group_uuids)

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
    _recompute_group_anchor(base_record)
    return base_record


def _merge_alarm_list_into_match_group(record, alarm_group_id, alarm_list):
    """将原始告警列表合并进已有的 match_rules 输出记录。"""
    existing_symptoms = list(record.get("symptoms", []))
    for alarm in alarm_list:
        existing_symptoms.append(_alarm_to_symptom(alarm))

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
    site_list, ne_list = _extract_alarm_group_meta(alarm_list)
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
        _build_ne_info_from_alarms(alarm_list),
    )

    record.setdefault("_merged_alarm_groups", []).append({
        "source": "alarm_group",
        "group_id": alarm_group_id,
        "alarm_count": len(alarm_list),
    })
    _recompute_group_anchor(record)
    return record


def _build_connected_components(match_group_eids, alarm_group_eids, eid_to_match_indices):
    eid_to_alarm_group_ids = defaultdict(set)
    for alarm_group_id, eids in alarm_group_eids.items():
        for eid in eids:
            eid_to_alarm_group_ids[eid].add(alarm_group_id)

    visited_match_indices = set()
    visited_alarm_group_ids = set()
    components = []

    for start_idx, eids in enumerate(match_group_eids):
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
                current_eids = match_group_eids[node_value]
                for eid in current_eids:
                    for alarm_group_id in eid_to_alarm_group_ids.get(eid, ()):
                        if alarm_group_id in visited_alarm_group_ids:
                            continue
                        visited_alarm_group_ids.add(alarm_group_id)
                        queue.append(("alarm", alarm_group_id))
            else:
                component_alarm_group_ids.add(node_value)
                current_eids = alarm_group_eids.get(node_value, set())
                for eid in current_eids:
                    for match_idx in eid_to_match_indices.get(eid, ()):
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
        for alarm_group_id in sorted(alarm_group_eids)
        if alarm_group_id not in visited_alarm_group_ids
    ]
    return components, standalone_alarm_group_ids


def merge(match_output_path, alarm_input_path, output_path, group_field="故障组ID"):
    print("加载 match_rules 输出...")
    match_groups = _load_match_groups(match_output_path)
    print(f"  输出故障组数: {len(match_groups)}")

    print("构建 eid -> 输出组索引...")
    eid_progress = ProgressBar(len(match_groups), "构建 eid 索引")
    match_group_eids = []
    eid_to_match_indices = defaultdict(list)
    for idx, record in enumerate(match_groups):
        eid_progress.update()
        current_eids = _extract_eids_from_match_group(record)
        match_group_eids.append(current_eids)
        for eid in current_eids:
            eid_to_match_indices[eid].append(idx)
    eid_progress.close()

    print("加载原始告警并按故障组ID分组...")
    alarm_groups = _load_alarm_groups(alarm_input_path, group_field=group_field)
    print(f"  原始告警故障组数: {len(alarm_groups)}")

    alarm_group_eids = {
        alarm_group_id: _extract_eids_from_alarm_group(alarm_list)
        for alarm_group_id, alarm_list in alarm_groups.items()
    }

    print("构建输出组-原始组连通关系...")
    components, standalone_alarm_group_ids = _build_connected_components(
        match_group_eids,
        alarm_group_eids,
        eid_to_match_indices,
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

        base_idx = match_indices[0]
        merged_record = copy.deepcopy(match_groups[base_idx])

        for match_idx in match_indices[1:]:
            _merge_match_group_records(merged_record, match_groups[match_idx])

        for alarm_group_id in component_alarm_group_ids:
            _merge_alarm_list_into_match_group(
                merged_record,
                alarm_group_id,
                alarm_groups[alarm_group_id],
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

    standalone_alarm_records = [
        _build_group_output_from_alarms(alarm_group_id, alarm_groups[alarm_group_id])
        for alarm_group_id in standalone_alarm_group_ids
    ]
    final_records.extend(standalone_alarm_records)

    print(
        f"  合并后记录数: {len(final_records)} "
        f"(输出连通分量: {len(components)}, "
        f"跨输出组合并分量: {merged_output_component_count}, "
        f"并入原始组数: {merged_alarm_group_count}, "
        f"独立原始组: {len(standalone_alarm_records)})"
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
    args = parser.parse_args()

    merge(args.match_output, args.alarms, args.output, group_field=args.group_field)


if __name__ == "__main__":
    main()
