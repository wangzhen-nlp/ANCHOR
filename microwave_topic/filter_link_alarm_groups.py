#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从故障组中筛选出包含 link 告警的故障组，输出格式与输入保持一致。

link 告警的判定：告警记录的告警标题/标准名命中 alarm_types.LINK_ALARMS。

输入/输出与 complete_group_topology.py 对应：
- 文件输入（多行 JSONL）-> 文件输出（多行 JSONL）；
- 文件夹输入（每组一个单行 jsonl，--per-file 的产物）-> 文件夹输出（每组一个单行 jsonl）。
输入类型自动按路径是文件还是目录判断，输出类型与输入保持一致。
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_types import LINK_ALARMS


# 告警记录里可能承载“告警标准名”的字段（与 complete_group_topology 保持一致）。
ALARM_TITLE_FIELDS = ("告警标题", "告警标准名", "alarm", "alarm_type", "title")
# 命中判定同时支持原样与大写，兼容不同来源的大小写差异。
_LINK_ALARM_KEYS = {str(name or "").strip() for name in LINK_ALARMS if str(name or "").strip()}
_LINK_ALARM_KEYS_UPPER = {name.upper() for name in _LINK_ALARM_KEYS}


def _normalize_text(value):
    return str(value or "").strip()


def _group_id(group):
    return (
        _normalize_text(group.get("uuid"))
        or _normalize_text((group.get("match_info") or {}).get("uuid"))
        or _normalize_text(group.get("故障组ID"))
    )


def _is_link_alarm_name(value):
    text = _normalize_text(value)
    if not text:
        return False
    return text in _LINK_ALARM_KEYS or text.upper() in _LINK_ALARM_KEYS_UPPER


def _record_has_link_alarm(record):
    if not isinstance(record, dict):
        return False
    return any(_is_link_alarm_name(record.get(field)) for field in ALARM_TITLE_FIELDS)


def _iter_group_alarm_records(group):
    """遍历故障组内所有告警记录，覆盖 ne_info / alarms / symptoms / match_info.symptoms。"""
    ne_info = group.get("ne_info")
    if isinstance(ne_info, dict):
        for info in ne_info.values():
            if not isinstance(info, dict):
                continue
            for alarm in info.get("alarm") or []:
                if isinstance(alarm, dict):
                    yield alarm

    for alarm in group.get("alarms") or []:
        if isinstance(alarm, dict):
            yield alarm

    for symptom in group.get("symptoms") or []:
        if isinstance(symptom, dict):
            yield symptom

    match_info = group.get("match_info")
    if isinstance(match_info, dict):
        for symptom in match_info.get("symptoms") or []:
            if isinstance(symptom, dict):
                yield symptom


def _group_has_link_alarm(group):
    return any(_record_has_link_alarm(record) for record in _iter_group_alarm_records(group))


def _load_groups_from_file(input_path):
    """读取多行 JSONL 文件，返回 [group, ...]。"""
    groups = []
    with open(input_path, "r", encoding="utf-8") as fr:
        for line_num, raw_line in enumerate(fr, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path} 第 {line_num} 行 JSON 解析失败: {exc}") from exc
            if isinstance(record, dict):
                groups.append(record)
    return groups


def _load_groups_from_dir(input_dir):
    """读取目录下每个单行 jsonl 文件，返回 [(源文件名, group), ...]，按文件名排序保证稳定。"""
    items = []
    for path in sorted(Path(input_dir).glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fr:
            line = fr.readline().strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} JSON 解析失败: {exc}") from exc
        if isinstance(record, dict):
            items.append((path.name, record))
    return items


def filter_groups(input_path, output_path):
    input_path = Path(input_path)

    if input_path.is_dir():
        items = _load_groups_from_dir(input_path)
        total = len(items)
        selected = [(name, group) for name, group in items if _group_has_link_alarm(group)]

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, group in selected:
            line = json.dumps(group, ensure_ascii=False, separators=(",", ":"))
            (out_dir / name).write_text(line + "\n", encoding="utf-8")
        kept = len(selected)
        output_kind = "dir"
    elif input_path.is_file():
        groups = _load_groups_from_file(input_path)
        total = len(groups)
        selected = [group for group in groups if _group_has_link_alarm(group)]

        out_path = Path(output_path)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fw:
            for group in selected:
                fw.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
                fw.write("\n")
        kept = len(selected)
        output_kind = "file"
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "output_kind": output_kind,
        "input_group_count": total,
        "output_group_count": kept,
        "dropped_without_link_alarm": total - kept,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="从故障组中筛选出包含 link 告警的故障组，输出格式与输入保持一致"
    )
    parser.add_argument("input", help="输入：故障组多行 JSONL 文件，或每组一个单行 jsonl 的目录")
    parser.add_argument("output", help="输出：文件输入对应文件，目录输入对应目录（不存在则新建）")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    stats = filter_groups(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
