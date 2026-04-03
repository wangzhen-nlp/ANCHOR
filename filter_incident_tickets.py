
"""
筛选Incident Ticket记录：
1. 行的内容包含至少两个站点ID
2. 涉及的站点只有Ran、Transmission设备，不能有Data设备
"""
import argparse
import json
import os
from collections import deque

import pandas as pd
from progress_utils import ProgressBar

try:
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover - fallback for environments without openpyxl
    load_workbook = None


def load_site_device_mapping(json_file: str) -> dict:
    """加载站点设备映射，返回站点ID -> 设备类型集合的映射"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为：站点ID -> 设备类型集合
    site_devices = {}
    for site_id, devices in data.items():
        if isinstance(devices, dict):
            site_devices[site_id] = set(devices.keys())
        else:
            site_devices[site_id] = set()
    return site_devices


def build_site_matcher(known_site_ids: list) -> dict:
    """基于站点全集构建 Aho-Corasick 自动机，便于按文本一次扫描匹配站点。"""
    root = {"next": {}, "fail": None, "outputs": []}

    for site_id in known_site_ids:
        node = root
        for char in site_id:
            node = node["next"].setdefault(char, {"next": {}, "fail": None, "outputs": []})
        node["outputs"].append(site_id)

    queue = deque()
    for child in root["next"].values():
        child["fail"] = root
        queue.append(child)

    while queue:
        node = queue.popleft()
        for char, child in node["next"].items():
            fail = node["fail"]
            while fail is not None and char not in fail["next"]:
                fail = fail["fail"]
            child["fail"] = fail["next"][char] if fail is not None and char in fail["next"] else root
            child["outputs"].extend(child["fail"]["outputs"])
            queue.append(child)

    return root


def _row_to_text(row_values) -> str:
    return ' '.join(str(value) for value in row_values)


def extract_site_ids_from_text(text: str, site_matcher: dict) -> list:
    """从文本中提取所有站点ID（基于 Aho-Corasick 自动机，去重并按首次出现排序）"""
    if not text:
        return []

    first_positions = {}
    state = site_matcher

    for idx, char in enumerate(text):
        while state is not site_matcher and char not in state["next"]:
            state = state["fail"]

        if char in state["next"]:
            state = state["next"][char]
        else:
            state = site_matcher

        for site_id in state["outputs"]:
            start = idx - len(site_id) + 1
            if site_id not in first_positions:
                first_positions[site_id] = start

    positions = sorted((pos, site_id) for site_id, pos in first_positions.items())
    return [site_id for pos, site_id in positions]


def check_site_devices(site_ids: list, site_device_mapping: dict) -> tuple:
    """
    检查站点列表是否满足条件：
    - 至少2个站点
    - 所有站点只有Ran和Transmission设备（没有Data设备）
    返回: (是否满足条件, 涉及的设备类型)
    """
    if len(site_ids) < 2:
        return False, set()

    all_devices = set()
    valid = True

    for site_id in site_ids:
        if site_id not in site_device_mapping:
            # 如果站点不在映射中，暂时跳过（可能有数据问题）
            continue

        devices = site_device_mapping[site_id]
        all_devices.update(devices)

        # 检查是否有Data设备
        if 'Data' in devices:
            valid = False
            break

    return valid, all_devices


def build_known_site_ids(site_device_mapping: dict) -> list:
    """构建稳定的站点匹配列表，长站点优先，减少短串误匹配。"""
    return sorted(site_device_mapping.keys(), key=lambda site_id: (-len(site_id), site_id))


def _get_ticket_column_index(df: pd.DataFrame) -> int:
    try:
        return df.columns.get_loc('工单ID')
    except KeyError as exc:
        raise KeyError("输入文件缺少 '工单ID' 列") from exc


def _build_ticket_site_json(result_df, matched_sites_by_row):
    json_data = {}
    ticket_col_idx = _get_ticket_column_index(result_df)
    for row_idx, row in enumerate(result_df.itertuples(index=False, name=None)):
        ticket_id = row[ticket_col_idx]
        site_ids = matched_sites_by_row[row_idx]
        json_data[ticket_id] = site_ids
    return json_data


def _normalize_header_row(header_row):
    headers = []
    used = {}
    for idx, value in enumerate(header_row):
        header = str(value).strip() if value is not None and str(value).strip() else f"Unnamed:{idx}"
        count = used.get(header, 0)
        used[header] = count + 1
        headers.append(header if count == 0 else f"{header}.{count}")
    return headers


def _filter_incident_tickets_rows(columns, row_iter, total_rows: int, site_device_mapping: dict, site_matcher: dict, progress_label: str = None):
    """筛选流式行数据，并仅保留命中的记录。"""
    filtered_indices = []
    filtered_rows = []
    matched_sites_by_row = []
    stats = {'total': 0, 'valid': 0, 'only_one_site': 0, 'has_data_device': 0}
    row_progress = ProgressBar(total_rows, progress_label or "处理工单记录", min_interval=0.05)

    try:
        for idx, row in enumerate(row_iter):
            stats['total'] += 1
            if len(row) < len(columns):
                row = tuple(row) + (None,) * (len(columns) - len(row))
            elif len(row) > len(columns):
                row = tuple(row[:len(columns)])

            # 提取站点ID
            site_ids = extract_site_ids_from_text(_row_to_text(row), site_matcher)

            if len(site_ids) < 2:
                stats['only_one_site'] += 1
                row_progress.update()
                continue

            # 检查设备类型
            valid, devices = check_site_devices(site_ids, site_device_mapping)

            if not valid:
                stats['has_data_device'] += 1
                row_progress.update()
                continue

            stats['valid'] += 1
            filtered_indices.append(idx)
            filtered_rows.append(row)
            matched_sites_by_row.append(site_ids)
            row_progress.update()
    finally:
        row_progress.close()

    result_df = pd.DataFrame(filtered_rows, columns=columns) if filtered_rows else None
    return result_df, stats, matched_sites_by_row


def _filter_incident_tickets_df(df, site_device_mapping: dict, site_matcher: dict, progress_label: str = None):
    """筛选单个 DataFrame。"""
    row_iter = df.itertuples(index=False, name=None)
    return _filter_incident_tickets_rows(
        list(df.columns),
        row_iter,
        len(df),
        site_device_mapping,
        site_matcher,
        progress_label=progress_label,
    )


def _filter_incident_tickets_xlsx_stream(input_file: str, site_device_mapping: dict, site_matcher: dict, progress_label: str = None):
    """使用 openpyxl 只读模式流式读取 xlsx，避免一次性把整表读入内存。"""
    if load_workbook is None:
        raise ImportError("openpyxl 不可用，无法启用 xlsx 流式读取")

    workbook = load_workbook(input_file, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[0]
        row_iter = worksheet.iter_rows(values_only=True)
        header_row = next(row_iter, None)
        if header_row is None:
            return None, {'total': 0, 'valid': 0, 'only_one_site': 0, 'has_data_device': 0}, []

        columns = _normalize_header_row(header_row)
        total_rows = max((worksheet.max_row or 1) - 1, 0)
        return _filter_incident_tickets_rows(
            columns,
            row_iter,
            total_rows,
            site_device_mapping,
            site_matcher,
            progress_label=progress_label,
        )
    finally:
        workbook.close()


def _filter_incident_tickets_excel(input_file: str, site_device_mapping: dict, site_matcher: dict, progress_label: str = None):
    """按文件类型选择合适的 Excel 读取方式。"""
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext in {'.xlsx', '.xlsm', '.xltx', '.xltm'} and load_workbook is not None:
        return _filter_incident_tickets_xlsx_stream(
            input_file,
            site_device_mapping,
            site_matcher,
            progress_label=progress_label,
        )

    df = pd.read_excel(input_file)
    print(f"原始记录数: {len(df)}")
    return _filter_incident_tickets_df(
        df,
        site_device_mapping,
        site_matcher,
        progress_label=progress_label,
    )


def _merge_ticket_site_json(existing_json, result_df, matched_sites_by_row):
    if result_df is None:
        return

    ticket_col_idx = _get_ticket_column_index(result_df)
    for row_idx, row in enumerate(result_df.itertuples(index=False, name=None)):
        ticket_id = row[ticket_col_idx]
        site_ids = matched_sites_by_row[row_idx]
        existing_sites = existing_json.setdefault(ticket_id, [])
        seen = set(existing_sites)
        for site_id in site_ids:
            if site_id not in seen:
                existing_sites.append(site_id)
                seen.add(site_id)


def _filter_incident_tickets_file(input_file: str, site_device_mapping: dict, site_matcher: dict, output_file: str, json_output_file: str = None):
    """筛选单个 Excel 文件。"""
    print(f"\n=== 处理文件: {input_file} ===")
    print(f"开始逐行筛选: {os.path.basename(input_file)}")

    result_df, stats, matched_sites_by_row = _filter_incident_tickets_excel(
        input_file,
        site_device_mapping,
        site_matcher,
        progress_label=f"处理记录 {os.path.basename(input_file)}",
    )

    print(f"\n=== 筛选统计 ===")
    print(f"总记录数: {stats['total']}")
    print(f"满足条件（至少2站点+仅Ran/Transmission）: {stats['valid']}")
    print(f"不足2个站点: {stats['only_one_site']}")
    print(f"包含Data设备: {stats['has_data_device']}")

    # 输出结果
    if result_df is not None:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        result_df.to_excel(output_file, index=False)
        print(f"\n已输出 {len(result_df)} 条记录到: {output_file}")

        if json_output_file:
            os.makedirs(os.path.dirname(json_output_file) or '.', exist_ok=True)
            json_data = _build_ticket_site_json(result_df, matched_sites_by_row)
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"已输出JSON到: {json_output_file}")

        return result_df, stats
    else:
        print("\n没有满足条件的记录")
        return None, stats


def filter_incident_tickets(input_file: str, site_device_file: str, output_file: str, json_output_file: str = None):
    """筛选满足条件的Incident Ticket记录"""
    # 加载站点设备映射
    site_device_mapping = load_site_device_mapping(site_device_file)
    known_site_ids = build_known_site_ids(site_device_mapping)
    site_matcher = build_site_matcher(known_site_ids)
    print(f"已加载 {len(site_device_mapping)} 个站点的设备信息")
    return _filter_incident_tickets_file(input_file, site_device_mapping, site_matcher, output_file, json_output_file)


def _iter_incident_input_files(input_path: str):
    if os.path.isdir(input_path):
        for root, dirnames, filenames in os.walk(input_path):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename.lower().endswith(('.xlsx', '.xls')):
                    yield os.path.join(root, filename)
        return

    yield input_path

def main():
    parser = argparse.ArgumentParser(description='筛选Incident Ticket记录')
    parser.add_argument(
        '-i', '--input',
        default='Incident Ticket_20260201-20260318.xlsx',
        help='输入的Excel文件'
    )
    parser.add_argument(
        '-s', '--site-device',
        default='site_device_counts.json',
        help='站点设备映射JSON文件'
    )
    parser.add_argument(
        '-o', '--output',
        default='filtered_incident_tickets.xlsx',
        help='输出的Excel文件'
    )
    parser.add_argument(
        '-j', '--json-output',
        help='JSON输出文件（可选），格式：{工单号: [站点列表]}'
    )

    args = parser.parse_args()

    site_device_mapping = load_site_device_mapping(args.site_device)
    known_site_ids = build_known_site_ids(site_device_mapping)
    site_matcher = build_site_matcher(known_site_ids)
    print(f"已加载 {len(site_device_mapping)} 个站点的设备信息")

    input_files = list(_iter_incident_input_files(args.input))
    if not input_files:
        print("没有找到可处理的 Excel 文件")
        return

    if os.path.isdir(args.input):
        aggregate_stats = {'total': 0, 'valid': 0, 'only_one_site': 0, 'has_data_device': 0}
        processed_files = 0
        aggregated_result_dfs = []
        aggregated_json = {}
        file_progress = ProgressBar(len(input_files), "处理输入文件", min_interval=0.05)

        try:
            for input_file in input_files:
                print(f"\n=== 处理文件: {input_file} ===")
                print(f"开始逐行筛选: {os.path.basename(input_file)}")

                result_df, stats, matched_sites_by_row = _filter_incident_tickets_excel(
                    input_file,
                    site_device_mapping,
                    site_matcher,
                    progress_label=f"处理记录 {os.path.basename(input_file)}",
                )

                print(f"\n=== 筛选统计 ===")
                print(f"总记录数: {stats['total']}")
                print(f"满足条件（至少2站点+仅Ran/Transmission）: {stats['valid']}")
                print(f"不足2个站点: {stats['only_one_site']}")
                print(f"包含Data设备: {stats['has_data_device']}")

                processed_files += 1
                for key in aggregate_stats:
                    aggregate_stats[key] += stats.get(key, 0)
                if result_df is not None:
                    aggregated_result_dfs.append(result_df)
                    _merge_ticket_site_json(aggregated_json, result_df, matched_sites_by_row)
                file_progress.update()
        finally:
            file_progress.close()

        print(f"\n=== 目录处理汇总 ===")
        print(f"处理文件数: {processed_files}")
        print(f"总记录数: {aggregate_stats['total']}")
        print(f"满足条件（至少2站点+仅Ran/Transmission）: {aggregate_stats['valid']}")
        print(f"不足2个站点: {aggregate_stats['only_one_site']}")
        print(f"包含Data设备: {aggregate_stats['has_data_device']}")

        if aggregated_result_dfs:
            final_result_df = pd.concat(aggregated_result_dfs, ignore_index=True)
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            final_result_df.to_excel(args.output, index=False)
            print(f"已汇总输出 {len(final_result_df)} 条记录到: {args.output}")
        else:
            print("没有满足条件的记录")

        if args.json_output:
            os.makedirs(os.path.dirname(args.json_output) or '.', exist_ok=True)
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(aggregated_json, f, ensure_ascii=False, indent=2)
            print(f"已汇总输出JSON到: {args.json_output}")
        return

    filter_incident_tickets(args.input, args.site_device, args.output, args.json_output)


if __name__ == '__main__':
    main()
