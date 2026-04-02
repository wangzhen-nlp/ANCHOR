
"""
筛选Incident Ticket记录：
1. 行的内容包含至少两个站点ID
2. 涉及的站点只有Ran、Transmission设备，不能有Data设备
"""
import argparse
import json
import pandas as pd


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


def extract_site_ids_from_row(row, known_site_ids: list) -> list:
    """从行中提取所有站点ID（基于已知站点全集匹配，去重并按首次出现排序）"""
    all_text = ' '.join([
        str(row.get(col, '')) for col in row.index
    ])
    positions = []
    for site_id in known_site_ids:
        pos = all_text.find(site_id)
        if pos >= 0:
            positions.append((pos, site_id))
    positions.sort(key=lambda item: (item[0], item[1]))
    return [site_id for _pos, site_id in positions]


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


def filter_incident_tickets(input_file: str, site_device_file: str, output_file: str):
    """筛选满足条件的Incident Ticket记录"""
    # 加载站点设备映射
    site_device_mapping = load_site_device_mapping(site_device_file)
    known_site_ids = build_known_site_ids(site_device_mapping)
    print(f"已加载 {len(site_device_mapping)} 个站点的设备信息")

    # 读取Excel
    df = pd.read_excel(input_file)
    print(f"原始记录数: {len(df)}")

    # 筛选
    filtered_rows = []
    stats = {'total': 0, 'valid': 0, 'only_one_site': 0, 'has_data_device': 0}

    for idx, row in df.iterrows():
        stats['total'] += 1

        # 提取站点ID
        site_ids = extract_site_ids_from_row(row, known_site_ids)

        if len(site_ids) < 2:
            stats['only_one_site'] += 1
            continue

        # 检查设备类型
        valid, devices = check_site_devices(site_ids, site_device_mapping)

        if not valid:
            stats['has_data_device'] += 1
            continue

        stats['valid'] += 1
        filtered_rows.append(row)

    print(f"\n=== 筛选统计 ===")
    print(f"总记录数: {stats['total']}")
    print(f"满足条件（至少2站点+仅Ran/Transmission）: {stats['valid']}")
    print(f"不足2个站点: {stats['only_one_site']}")
    print(f"包含Data设备: {stats['has_data_device']}")

    # 输出结果
    if filtered_rows:
        result_df = pd.DataFrame(filtered_rows)
        result_df.to_excel(output_file, index=False)
        print(f"\n已输出 {len(filtered_rows)} 条记录到: {output_file}")

        # 生成JSON格式：{工单号: [站点列表]}
        return result_df, stats, known_site_ids
    else:
        print("\n没有满足条件的记录")
        return None, stats, known_site_ids


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

    result_df, stats, known_site_ids = filter_incident_tickets(args.input, args.site_device, args.output)

    # 输出JSON格式
    if args.json_output and result_df is not None:
        json_data = {}
        for _, row in result_df.iterrows():
            ticket_id = row['工单ID']
            site_ids = extract_site_ids_from_row(row, known_site_ids)
            json_data[ticket_id] = site_ids

        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"已输出JSON到: {args.json_output}")


if __name__ == '__main__':
    main()
