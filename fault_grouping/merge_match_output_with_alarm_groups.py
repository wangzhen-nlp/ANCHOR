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
    return eids


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


def _merge_alarm_list_into_match_group(record, alarm_list):
    """将原始告警列表合并进已有的 match_rules 输出记录。"""
    # 1. symptoms 合并（按 eid 去重，已有 eid 保留原 symptom，缺失的追加）
    existing_eids = {s.get("eid") for s in record.get("symptoms", []) if s.get("eid")}
    new_symptoms = []
    for alarm in alarm_list:
        eid = _normalize_text(alarm.get("告警编码ID", ""))
        if eid and eid not in existing_eids:
            new_symptoms.append(_alarm_to_symptom(alarm))
            existing_eids.add(eid)

    if new_symptoms:
        record.setdefault("symptoms", []).extend(new_symptoms)
        # 重新排序 symptoms
        record["symptoms"].sort(key=lambda s: (s.get("ts") or float("inf"), s.get("eid", "")))
        # 更新 anchor_ts
        timestamps = [s["ts"] for s in record["symptoms"] if s.get("ts") is not None]
        if timestamps:
            anchor_ts = min(timestamps)
            record["group_anchor_ts"] = anchor_ts
            record["group_anchor_time"] = datetime.fromtimestamp(anchor_ts).strftime("%Y-%m-%d %H:%M:%S")

    # 2. group_info 合并站点 / NE
    group_info = record.get("group_info", {})
    for gid, gmeta in group_info.items():
        if not isinstance(gmeta, dict):
            continue
        existing_sites = set(gmeta.get("site_list", []))
        existing_nes = set(gmeta.get("ne_list", []))
        for alarm in alarm_list:
            site_id = _normalize_text(alarm.get("站点ID", ""))
            ne_id = _normalize_text(alarm.get("告警源", ""))
            if site_id:
                existing_sites.add(site_id)
            if ne_id:
                existing_nes.add(ne_id)
        gmeta["site_list"] = sorted(existing_sites)
        gmeta["ne_list"] = sorted(existing_nes)

    # 3. ne_info 补充
    existing_ne_info = record.get("ne_info", {})
    new_ne_info = _build_ne_info_from_alarms(alarm_list)
    for ne_id, ne_meta in new_ne_info.items():
        if ne_id not in existing_ne_info:
            existing_ne_info[ne_id] = ne_meta
        else:
            # 已有该 NE，追加 alarm
            existing_ne_info[ne_id].setdefault("alarm", []).extend(ne_meta.get("alarm", []))

    # 4. 标记来源
    record.setdefault("_merged_alarm_groups", []).append({
        "source": "alarm_group",
        "alarm_count": len(alarm_list),
    })

    return record


def merge(match_output_path, alarm_input_path, output_path, group_field="故障组ID"):
    print("加载 match_rules 输出...")
    match_groups = _load_match_groups(match_output_path)
    print(f"  输出故障组数: {len(match_groups)}")

    print("构建 eid -> 输出组索引...")
    eid_progress = ProgressBar(len(match_groups), "构建 eid 索引")
    eid_to_match_indices = defaultdict(list)
    for idx, record in enumerate(match_groups):
        eid_progress.update()
        for eid in _extract_eids_from_match_group(record):
            eid_to_match_indices[eid].append(idx)
    eid_progress.close()

    print("加载原始告警并按故障组ID分组...")
    alarm_groups = _load_alarm_groups(alarm_input_path, group_field=group_field)
    print(f"  原始告警故障组数: {len(alarm_groups)}")

    # idx -> merged_record（深拷贝后的输出组，可能被多次累积合并）
    merged_match_records = {}
    matched_match_indices = set()
    standalone_alarm_records = []

    merge_progress = ProgressBar(len(alarm_groups), "匹配并合并故障组")
    for alarm_group_id, alarm_list in sorted(alarm_groups.items()):
        merge_progress.update()

        # 收集该原始告警组的所有 eid
        alarm_eids = {
            _normalize_text(a.get("告警编码ID", ""))
            for a in alarm_list
            if _normalize_text(a.get("告警编码ID", ""))
        }
        if not alarm_eids:
            # 没有 eid 的告警组无法匹配，直接按新记录输出
            standalone_alarm_records.append(_build_group_output_from_alarms(alarm_group_id, alarm_list))
            continue

        # 统计与每个输出组的重合 eid 数
        match_counter = defaultdict(int)
        for eid in alarm_eids:
            for idx in eid_to_match_indices.get(eid, []):
                match_counter[idx] += 1

        if match_counter:
            # 取重合 eid 数最多的输出组
            best_idx = max(match_counter, key=lambda i: (match_counter[i], i))
            best_record = match_groups[best_idx]
            matched_match_indices.add(best_idx)

            if best_idx not in merged_match_records:
                # 第一次匹配到该输出组：深拷贝并初始化
                merged_match_records[best_idx] = copy.deepcopy(best_record)

            merge_progress.set_extra_text(
                f"原始组 {alarm_group_id} -> 输出组 "
                f"{best_record.get('match_info', {}).get('uuid', '')[:16]}... "
                f"(重合 {match_counter[best_idx]} 个 eid)"
            )
            _merge_alarm_list_into_match_group(merged_match_records[best_idx], alarm_list)
        else:
            # 无重合，按输出格式新建
            standalone_alarm_records.append(_build_group_output_from_alarms(alarm_group_id, alarm_list))
    merge_progress.close()

    # 组装最终输出
    final_records = []

    # 先放已合并的输出组（保持原始顺序）
    for idx, record in enumerate(match_groups):
        if idx in merged_match_records:
            final_records.append(merged_match_records[idx])
        elif idx not in matched_match_indices:
            # 未匹配的输出组原样保留
            final_records.append(record)

    # 再放独立的新建记录
    final_records.extend(standalone_alarm_records)

    print(
        f"  合并后记录数: {len(final_records)} "
        f"(合并输出组: {len(merged_match_records)}, "
        f"未匹配输出组: {len(match_groups) - len(matched_match_indices)}, "
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
