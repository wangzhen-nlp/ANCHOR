import json
from argparse import ArgumentParser

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alarm_tools.progress_utils import ProgressBar


def count_jsonl_records(path):
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def normalize_text(value):
    return str(value or "").strip()


def parse_rule_names(raw_values):
    rule_names = []
    seen = set()
    for raw_value in raw_values or []:
        for part in str(raw_value).replace("，", ",").split(","):
            rule_name = normalize_text(part)
            if not rule_name or rule_name in seen:
                continue
            seen.add(rule_name)
            rule_names.append(rule_name)
    return rule_names


def iter_jsonl_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: 第 {line_no} 行不是 JSON object")
            yield line_no, record


def get_group_uuid(record):
    match_info = record.get("match_info") if isinstance(record.get("match_info"), dict) else {}
    return normalize_text(record.get("uuid") or match_info.get("uuid"))


def get_related_group_uuids(record):
    match_info = record.get("match_info") if isinstance(record.get("match_info"), dict) else {}
    related = record.get("related_group_uuids")
    if related is None:
        related = match_info.get("related_group_uuids", [])
    if not isinstance(related, list):
        return []
    return [normalize_text(value) for value in related if normalize_text(value)]


def iter_rule_values(value):
    if isinstance(value, list):
        for item in value:
            rule_name = normalize_text(item)
            if rule_name:
                yield rule_name
        return
    rule_name = normalize_text(value)
    if rule_name:
        yield rule_name


def get_group_rules(record, include_symptom_rules=False):
    rules = set()
    match_info = record.get("match_info") if isinstance(record.get("match_info"), dict) else {}
    for container in (record, match_info):
        rules.update(iter_rule_values(container.get("rule")))
        rules.update(iter_rule_values(container.get("merged_rules")))

    if include_symptom_rules:
        for symptom in record.get("symptoms") or []:
            if not isinstance(symptom, dict):
                continue
            rules.update(iter_rule_values(symptom.get("matched_rule")))
            rules.update(iter_rule_values(symptom.get("matched_rule_list")))

    return rules


def collect_referenced_group_uuids(input_path, total=None, show_progress=True):
    referenced_group_uuids = set()
    input_count = 0
    progress = ProgressBar(total or 0, "扫描关联故障组") if show_progress else None
    try:
        for _line_no, record in iter_jsonl_records(input_path):
            input_count += 1
            referenced_group_uuids.update(get_related_group_uuids(record))
            if progress is not None:
                progress.update()
    finally:
        if progress is not None:
            progress.close()
    return input_count, referenced_group_uuids


def should_keep_record(
    record,
    selected_rules,
    rule_match_mode,
    ultimate_only,
    referenced_group_uuids,
    include_symptom_rules=False,
):
    if ultimate_only:
        group_uuid = get_group_uuid(record)
        if not group_uuid or group_uuid in referenced_group_uuids:
            return False

    if selected_rules:
        group_rules = get_group_rules(record, include_symptom_rules=include_symptom_rules)
        selected_rule_set = set(selected_rules)
        if rule_match_mode == "all":
            if not selected_rule_set.issubset(group_rules):
                return False
        elif not (group_rules & selected_rule_set):
            return False

    return True


def filter_match_output(
    input_path,
    output_path,
    *,
    selected_rules=None,
    rule_match_mode="any",
    ultimate_only=False,
    include_symptom_rules=False,
    show_progress=True,
):
    selected_rules = list(selected_rules or [])
    total_records = count_jsonl_records(input_path) if show_progress else None
    input_count, referenced_group_uuids = collect_referenced_group_uuids(
        input_path,
        total=total_records,
        show_progress=show_progress,
    )
    output_count = 0
    rule_matched_count = 0
    ultimate_kept_count = 0

    progress = ProgressBar(input_count, "筛选故障组") if show_progress else None
    with open(output_path, "w", encoding="utf-8") as output:
        try:
            for _line_no, record in iter_jsonl_records(input_path):
                group_rules = get_group_rules(record, include_symptom_rules=include_symptom_rules)
                if selected_rules:
                    selected_rule_set = set(selected_rules)
                    if rule_match_mode == "all":
                        rule_matched = selected_rule_set.issubset(group_rules)
                    else:
                        rule_matched = bool(group_rules & selected_rule_set)
                else:
                    rule_matched = True
                if rule_matched:
                    rule_matched_count += 1

                group_uuid = get_group_uuid(record)
                ultimate_kept = bool(group_uuid and group_uuid not in referenced_group_uuids)
                if ultimate_kept:
                    ultimate_kept_count += 1

                if should_keep_record(
                    record,
                    selected_rules,
                    rule_match_mode,
                    ultimate_only,
                    referenced_group_uuids,
                    include_symptom_rules=include_symptom_rules,
                ):
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_count += 1

                if progress is not None:
                    progress.set_extra_text(
                        f"rule匹配: {rule_matched_count} | 终极: {ultimate_kept_count} | 输出: {output_count}"
                    )
                    progress.update()
        finally:
            if progress is not None:
                progress.close()

    return {
        "input_count": input_count,
        "referenced_group_count": len(referenced_group_uuids),
        "ultimate_group_count": ultimate_kept_count,
        "rule_matched_count": rule_matched_count,
        "output_count": output_count,
        "selected_rules": selected_rules,
        "rule_match_mode": rule_match_mode,
        "ultimate_only": ultimate_only,
        "include_symptom_rules": include_symptom_rules,
        "output_path": output_path,
    }


def main():
    parser = ArgumentParser(
        description="从 match_rules.py 输出 JSONL 中筛选包含指定 rule 的故障组，可选只保留终极故障组"
    )
    parser.add_argument("input", help="match_rules.py 输出 JSONL 文件")
    parser.add_argument("output", help="筛选后的输出 JSONL 文件")
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="要保留的规则名；可重复传入，也支持逗号分隔。不传则不按 rule 过滤",
    )
    parser.add_argument(
        "--rule-match-mode",
        choices=("any", "all"),
        default="any",
        help="多个 --rule 的匹配方式：any=包含任意一个，all=必须全部包含。默认 any",
    )
    parser.add_argument(
        "--ultimate-only",
        action="store_true",
        help="只保留未出现在其它故障组 related_group_uuids 中的终极故障组",
    )
    parser.add_argument(
        "--include-symptom-rules",
        action="store_true",
        help="除故障组 rule/merged_rules 外，也把 symptoms[*].matched_rule/list 纳入 rule 判断",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条输出",
    )
    args = parser.parse_args()

    selected_rules = parse_rule_names(args.rule)
    result = filter_match_output(
        args.input,
        args.output,
        selected_rules=selected_rules,
        rule_match_mode=args.rule_match_mode,
        ultimate_only=args.ultimate_only,
        include_symptom_rules=args.include_symptom_rules,
        show_progress=not args.no_progress,
    )
    print(f"输入故障组数: {result['input_count']}")
    print(f"被其它组关联的 uuid 数: {result['referenced_group_count']}")
    print(f"终极故障组数: {result['ultimate_group_count']}")
    print(f"rule 匹配组数: {result['rule_matched_count']}")
    print(f"输出故障组数: {result['output_count']}")
    print(f"输出文件: {result['output_path']}")


if __name__ == "__main__":
    main()
