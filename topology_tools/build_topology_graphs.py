#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性生成 site_graph.json / ne_graph.json / link_peer_index.json 三个产物。

这是 extract_site_graph.py、extract_ne_graph.py、build_link_peer_index.py 的合并版：
三个独立工具各自都要读一遍并去重同一份链路文件，且 SYS_NE 被读取两次、ne_graph
还要把 site_graph.json 从磁盘读回。本工具把这些共享读取合并为：

- SYS_NE 读取 1 次（得到完整 NE 信息，并据此派生 nativeId->site_id 映射）
- SYS_SITE 读取 1 次（站点信息直接驻留内存，供 site 图与 ne 图共用）
- 链路读取 + 去重 1 次（单趟遍历同时聚合 site 邻接、ne 邻接与端口对端索引）

输出与三个独立工具逐字节一致，可直接替代它们。
"""

import json
import argparse

from collections import defaultdict

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import (
    LINK_PEER_INDEX_JSON,
    NE_GRAPH_JSON,
    SITE_GRAPH_JSON,
    SYS_LINK_JSONL,
    SYS_NE_DIR,
    SYS_SITE_DIR,
    resource_display,
)
from topology_tools.extract_site_graph import (
    load_latest_link_records,
    load_site_info,
)
from topology_tools.extract_ne_graph import load_ne_from_csv
from topology_tools.link_peer_index import (
    PeerDevice,
    _get_record_value,
    _make_key,
    _normalize_ne_key,
    save_peer_index,
)

# site_graph 中缺省站点信息（NE 引用了某站点但 SYS_SITE 中无该站点时使用）
_DEFAULT_SITE = {
    'site_name': '',
    'site_type': '',
    'longitude': '',
    'latitude': '',
    'region_id': '',
    'is_hub': False,
}


def _add_bidirectional_edge(graph: dict, a: str, b: str, link_type: str) -> None:
    """在 graph 中登记 a->b 的有向边，并维护 b 侧的反向标记。

    与 extract_site_graph.build_site_graph / extract_ne_graph.build_ne_graph 中的
    逻辑完全一致：首次出现记 '->'/'<-'，两个方向都出现则升级为 '<->'。
    """
    fwd = graph[a][b]
    if link_type in fwd:
        if fwd[link_type] == '<-':
            fwd[link_type] = '<->'
    else:
        fwd[link_type] = '->'
    rev = graph[b][a]
    if link_type in rev:
        if rev[link_type] == '->':
            rev[link_type] = '<->'
    else:
        rev[link_type] = '<-'


def build_graphs(latest_links: list, ne_site_map: dict):
    """单趟遍历去重后的链路，同时构建 site 邻接图、ne 邻接图与端口对端索引。

    Returns:
        (site_links, ne_links, peer_index, stats)
    """
    site_links = defaultdict(lambda: defaultdict(dict))
    ne_links = defaultdict(lambda: defaultdict(dict))
    peer_index = {}

    stats = {
        'ne_link_count': 0,
        'ne_mapped_count': 0,
        'site_link_count': 0,
        'site_mapped_count': 0,
    }

    for record in latest_links:
        a_ne = _get_record_value(record, "a_end_ne_nativeId", "a_end_ne_nativeId(')").upper()
        z_ne = _get_record_value(record, "z_end_ne_nativeId", "z_end_ne_nativeId(')").upper()
        link_type = (record.get('link_layer') or '').strip().upper()

        if a_ne and z_ne:
            # NE 邻接图
            stats['ne_link_count'] += 1
            stats['ne_mapped_count'] += 1
            _add_bidirectional_edge(ne_links, a_ne, z_ne, link_type)

            # 站点邻接图：两端 NE 都能映射到站点时才登记
            stats['site_link_count'] += 1
            a_site = ne_site_map.get(a_ne)
            z_site = ne_site_map.get(z_ne)
            if a_site and z_site:
                stats['site_mapped_count'] += 1
                _add_bidirectional_edge(site_links, a_site, z_site, link_type)

            # 端口对端索引：还需两端端口名
            a_port = _get_record_value(record, "a_end_port_name")
            z_port = _get_record_value(record, "z_end_port_name")
            if a_port and z_port:
                peer_index[_make_key(a_ne, a_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(z_ne),
                    port_name=z_port,
                    port_ip=_get_record_value(record, "z_end_port_ip"),
                    manager_name=_get_record_value(record, "z_end_ne_manager_name"),
                )
                peer_index[_make_key(z_ne, z_port)] = PeerDevice(
                    ne_native_id=_normalize_ne_key(a_ne),
                    port_name=a_port,
                    port_ip=_get_record_value(record, "a_end_port_ip"),
                    manager_name=_get_record_value(record, "a_end_ne_manager_name"),
                )

    return site_links, ne_links, peer_index, stats


def assemble_site_graph(site_info: dict, site_links: dict) -> dict:
    """合并站点信息与站点邻接关系，等价于 extract_site_graph.main 的输出。"""
    result = {}
    for site_id in (set(site_info) | set(site_links)):
        site_data = dict(site_info.get(site_id, _DEFAULT_SITE))
        site_data['link'] = dict(site_links.get(site_id, {}))
        result[site_id] = site_data
    return result


def assemble_ne_graph(ne_info: dict, site_info: dict, ne_links: dict) -> dict:
    """合并 NE 信息、站点信息与 NE 邻接关系，等价于 extract_ne_graph.main 的输出。"""
    ne_graph = {}
    for ne_id in (set(ne_links) | set(ne_info)):
        ne_data = dict(ne_info.get(ne_id, {}))
        neighbors = ne_links.get(ne_id, {})
        site_id = ne_data.get('site_id', '')
        site_data = site_info.get(site_id, {})

        ne_data.update(
            {
                'site_id': site_id,
                'site_name': site_data.get('site_name', ''),
                'site_type': site_data.get('site_type', ''),
                'longitude': site_data.get('longitude', ''),
                'latitude': site_data.get('latitude', ''),
            }
        )
        ne_data['region_id'] = ne_data.get('region_id', '') or site_data.get('region_id', '')

        ne_graph[ne_id] = ne_data
        ne_graph[ne_id]['link'] = {}
        for neighbor_id, links in neighbors.items():
            ne_graph[ne_id]['link'][neighbor_id] = links
    return ne_graph


def main():
    parser = argparse.ArgumentParser(
        description="一次性生成 site_graph.json / ne_graph.json / link_peer_index.json"
    )
    parser.add_argument(
        "--ne-dir",
        default=SYS_NE_DIR,
        help=f"SYS_NE 数据目录，默认: {resource_display('SYS_NE_0306')}",
    )
    parser.add_argument(
        "--site-dir",
        default=SYS_SITE_DIR,
        help=f"SYS_SITE 数据目录，默认: {resource_display('SYS_SITE_0306')}",
    )
    parser.add_argument(
        "--link-input",
        default=SYS_LINK_JSONL,
        help=(
            "链路输入，支持 .jsonl/.csv/.zip(内含CSV) 文件或包含这些文件的目录，"
            f"默认: {resource_display('sys_link_1231.jsonl')}"
        ),
    )
    parser.add_argument(
        "--site-graph-output",
        default=SITE_GRAPH_JSON,
        help=f"输出 site_graph.json，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--ne-graph-output",
        default=NE_GRAPH_JSON,
        help=f"输出 ne_graph.json，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--peer-index-output",
        default=LINK_PEER_INDEX_JSON,
        help=f"输出 link_peer_index.json，默认: {resource_display('link_peer_index.json')}",
    )
    parser.add_argument(
        "--report-duplicates",
        action="store_true",
        help="打印 NE/站点/链路 中重复 ID 的明细（默认仅汇总，不打印明细）",
    )
    args = parser.parse_args()

    # 1) SYS_NE 读取一次，得到完整 NE 信息（其中含 site_id）
    print("加载NE信息...")
    ne_info = load_ne_from_csv(args.ne_dir, args.report_duplicates)
    print(f"  NE数量: {len(ne_info)}")

    # 由 NE 信息派生 nativeId->site_id 映射，无需再读一遍 SYS_NE
    ne_site_map = {
        native_id: info['site_id']
        for native_id, info in ne_info.items()
        if info.get('site_id')
    }

    # 2) SYS_SITE 读取一次
    print("\n加载站点信息...")
    site_info = load_site_info(args.site_dir, args.report_duplicates)

    # 3) 链路读取+去重一次，单趟遍历产出三张图
    print("\n生成传播图与对端索引...")
    latest_links = load_latest_link_records(args.link_input, args.report_duplicates)
    site_links, ne_links, peer_index, stats = build_graphs(latest_links, ne_site_map)
    print(f"  站点图: 处理 {stats['site_link_count']} 条链路，成功映射 {stats['site_mapped_count']} 条")
    print(f"  NE图:   处理 {stats['ne_link_count']} 条链路，成功映射 {stats['ne_mapped_count']} 条")

    # 4) 组装并写出
    site_graph = assemble_site_graph(site_info, site_links)
    with open(args.site_graph_output, 'w', encoding='utf-8') as f:
        json.dump(site_graph, f, ensure_ascii=False, indent=2)

    ne_graph = assemble_ne_graph(ne_info, site_info, ne_links)
    with open(args.ne_graph_output, 'w', encoding='utf-8') as f:
        json.dump(ne_graph, f, ensure_ascii=False, indent=2)

    save_peer_index(peer_index, args.peer_index_output)

    # 汇总
    print(f"\n生成文件: {args.site_graph_output}")
    print(f"  站点数: {len(site_graph)}")
    site_neighbor_counts = [len(v['link']) for v in site_graph.values()]
    if site_neighbor_counts:
        print(f"  平均邻居站点数: {sum(site_neighbor_counts)/len(site_neighbor_counts):.1f}")
        print(f"  最大邻居站点数: {max(site_neighbor_counts)}")

    print(f"生成文件: {args.ne_graph_output}")
    print(f"  NE数: {len(ne_graph)}")
    if ne_graph:
        with_site = sum(1 for ne in ne_graph.values() if ne.get('site_id'))
        print(f"  有站点信息的NE: {with_site} ({with_site/len(ne_graph)*100:.1f}%)")

    print(f"生成文件: {args.peer_index_output}")
    print(f"  对端索引记录数: {len(peer_index)}")


if __name__ == "__main__":
    main()
