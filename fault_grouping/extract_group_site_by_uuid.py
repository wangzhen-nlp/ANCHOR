#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 match_rules.py 输出的 JSONL 故障组文件中按 uuid 提取故障组。

用法:
    python fault_grouping/extract_group_site_by_uuid.py fault_groups.jsonl <uuid>
    python fault_grouping/extract_group_site_by_uuid.py fault_groups.jsonl <uuid1>,<uuid2> -o selected_groups.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from fault_grouping.analyze_group_site_distribution import extract_site_ids
except ModuleNotFoundError:
    from analyze_group_site_distribution import extract_site_ids


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _normalize_uuid(value):
    return str(value or "").strip()


def _parse_uuid_args(raw_values):
    uuids = []
    seen = set()
    for raw_value in raw_values:
        for part in str(raw_value or "").split(","):
            uuid = _normalize_uuid(part)
            if uuid and uuid not in seen:
                seen.add(uuid)
                uuids.append(uuid)
    return uuids


def _get_record_uuid(record):
    if not isinstance(record, dict):
        return ""
    match_info = _as_dict(record.get("match_info"))
    return _normalize_uuid(match_info.get("uuid") or record.get("uuid"))


def _get_record_rule(record):
    if not isinstance(record, dict):
        return ""
    match_info = _as_dict(record.get("match_info"))
    return str(match_info.get("rule") or record.get("rule") or "").strip()


def iter_jsonl_records(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {jsonl_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw_line = line.rstrip("\n")
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ 跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"⚠️ 跳过第 {line_num} 行：JSON 顶层不是对象", file=sys.stderr)
                continue
            yield line_num, raw_line, record


def find_groups_by_uuid(jsonl_path, target_uuids, all_matches=False):
    target_uuid_set = set(target_uuids)
    found = []
    found_uuid_set = set()

    for line_num, raw_line, record in iter_jsonl_records(jsonl_path):
        record_uuid = _get_record_uuid(record)
        if record_uuid not in target_uuid_set:
            continue

        found.append({
            "line_num": line_num,
            "uuid": record_uuid,
            "raw_line": raw_line,
            "record": record,
        })
        found_uuid_set.add(record_uuid)

        if not all_matches and found_uuid_set >= target_uuid_set:
            break

    return found


def build_output_payload(matches, sites_only=False, summary=False):
    payload_items = []
    for match in matches:
        record = match["record"]
        site_list = sorted(extract_site_ids(record))
        if sites_only or summary:
            payload_items.append({
                "uuid": match["uuid"],
                "rule": _get_record_rule(record),
                "line_num": match["line_num"],
                "site_count": len(site_list),
                "site_list": site_list,
            })
        else:
            payload_items.append(record)

    if len(payload_items) == 1:
        return payload_items[0]
    return payload_items


def write_payload(payload, output_path):
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if not output_path:
        print(text)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    print(f"已写出: {output_path}")


def write_jsonl_matches(matches, output_path):
    lines = [match["raw_line"].rstrip("\n") + "\n" for match in matches]
    if not output_path:
        sys.stdout.writelines(lines)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"已写出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="按 uuid 从故障组 JSONL 中提取对应故障组")
    parser.add_argument("jsonl", help="match_rules.py 输出的 JSONL 文件")
    parser.add_argument("uuid", nargs="+", help="目标 uuid；支持多个参数，也支持逗号分隔")
    parser.add_argument("-o", "--output", default="", help="输出文件；默认输出 JSONL 原行，为空则打印到 stdout")
    parser.add_argument("--sites-only", action="store_true", help="只输出目标故障组的站点列表摘要")
    parser.add_argument("--summary", action="store_true", help="输出 uuid/rule/行号/站点数/站点列表摘要")
    parser.add_argument("--all-matches", action="store_true", help="继续扫描完整文件，输出所有重复 uuid 命中的记录；默认每个 uuid 找到首个即停止")
    args = parser.parse_args()

    target_uuids = _parse_uuid_args(args.uuid)
    if not target_uuids:
        parser.error("至少需要提供一个有效 uuid")

    matches = find_groups_by_uuid(args.jsonl, target_uuids, all_matches=args.all_matches)
    found_uuids = {match["uuid"] for match in matches}
    missing_uuids = [uuid for uuid in target_uuids if uuid not in found_uuids]
    if missing_uuids:
        print(f"⚠️ 未找到 uuid: {', '.join(missing_uuids)}", file=sys.stderr)
    if not matches:
        raise SystemExit(1)

    if args.sites_only or args.summary:
        payload = build_output_payload(matches, sites_only=args.sites_only, summary=args.summary)
        write_payload(payload, args.output)
    else:
        write_jsonl_matches(matches, args.output)


if __name__ == "__main__":
    main()
