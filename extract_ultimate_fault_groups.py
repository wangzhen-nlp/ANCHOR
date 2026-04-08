import json
from argparse import ArgumentParser

from compute_group_output_ticket_recall import _extract_group_sites


def stream_jsonl_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc


def get_group_uuid(group):
    return str(group.get("uuid") or group.get("match_info", {}).get("uuid") or "").strip()


def get_related_group_uuids(group):
    match_info = group.get("match_info") or {}
    related = group.get("related_group_uuids")
    if related is None:
        related = match_info.get("related_group_uuids", [])
    if not isinstance(related, list):
        return []
    return [str(value).strip() for value in related if str(value).strip()]


def extract_ultimate_fault_groups(input_path, output_path, min_site_num=0):
    groups = list(stream_jsonl_records(input_path))
    referenced_group_uuids = set()
    for group in groups:
        referenced_group_uuids.update(get_related_group_uuids(group))

    ultimate_groups = [
        group for group in groups
        if (group_uuid := get_group_uuid(group)) and group_uuid not in referenced_group_uuids
    ]
    if min_site_num > 0:
        filtered_ultimate_groups = []
        for group in ultimate_groups:
            group_uuid = get_group_uuid(group)
            group_sites = _extract_group_sites(group, group_uuid)
            if len(group_sites) >= min_site_num:
                filtered_ultimate_groups.append(group)
        ultimate_groups = filtered_ultimate_groups

    with open(output_path, "w", encoding="utf-8") as f:
        for group in ultimate_groups:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")

    return {
        "input_count": len(groups),
        "referenced_group_count": len(referenced_group_uuids),
        "ultimate_group_count": len(ultimate_groups),
        "min_site_num": min_site_num,
        "output_path": output_path,
    }


def main():
    parser = ArgumentParser(description="从 match_rules.py 输出 jsonl 中提取终极故障组（未被其它组关联的故障组）")
    parser.add_argument("input", help="match_rules.py 输出 jsonl 文件")
    parser.add_argument(
        "-o",
        "--output",
        default="ultimate_fault_groups.jsonl",
        help="输出 jsonl 文件，默认: ultimate_fault_groups.jsonl",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="只保留站点数 >= min-site-num 的终极故障组，默认: 0",
    )
    args = parser.parse_args()

    result = extract_ultimate_fault_groups(args.input, args.output, min_site_num=args.min_site_num)
    print(f"输入故障组数: {result['input_count']}")
    print(f"被关联故障组数: {result['referenced_group_count']}")
    print(f"终极故障组数: {result['ultimate_group_count']}")
    print(f"最小站点数过滤: {result['min_site_num']}")
    print(f"输出文件: {result['output_path']}")


if __name__ == "__main__":
    main()
