#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串联 extract_site_ids.py 和 find_site_neighbors.py 的流程：
从输入字符串中抽取 site_id，再获取这些站点的 n 阶邻居并生成输出文件
"""

import json
import argparse
from extract_site_ids import extract_site_ids
from find_site_neighbors import build_site_graph, get_nth_order_neighbors
from filter_links import get_nes_by_site_ids


def main():
    parser = argparse.ArgumentParser(
        description='从字符串中抽取 site_id，再获取其 n 阶邻居站点，输出格式与 filter_links.py 一致',
        epilog='示例: python extract_and_find_neighbors.py "07BNS0184: xxx" output.json'
    )
    parser.add_argument('text', type=str, help='待解析的多行字符串（extract_site_ids.py 的输入）')
    parser.add_argument('output', type=str, help='输出JSON文件')
    parser.add_argument('--n', type=int, default=1, help='阶数')
    parser.add_argument('--graph-file', type=str, default='topology_resources/ne_graph.json',
                        help='ne_graph.json 文件路径')

    args = parser.parse_args()

    # 从输入字符串抽取站点列表
    site_ids = extract_site_ids(args.text)
    start_sites = [s.strip() for s in site_ids.split(',') if s.strip()]
    print(f"抽取到起始站点数: {len(start_sites)}")
    if not start_sites:
        print("未从输入中抽取到任何 site_id，退出")
        return

    # 构建站点连接图
    print("正在构建站点连接图...")
    site_graph = build_site_graph(args.graph_file)
    print(f"站点数量: {len(site_graph)}")

    # 获取 n 阶邻居
    neighbor_sites = get_nth_order_neighbors(start_sites, args.n, site_graph)
    print(f"获取 {args.n} 阶邻居，总站点数: {len(neighbor_sites)}")

    # 调用 filter_links 的函数生成输出
    result = get_nes_by_site_ids(args.graph_file, neighbor_sites)

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {args.output}")
    print(f"总 NE 数: {len(result['ne_info'])}")
    print(f"总站点数: {len(result['group_info']['1']['site_list'])}")


if __name__ == "__main__":
    main()
