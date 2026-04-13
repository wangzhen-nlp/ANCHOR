#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据告警传播规则生成NE传播图和联通分量

与v1的区别:
- 直接从 ne_graph.json 加载NE连接图 (而不是从 sys_link_1231.jsonl 构建)

可选功能:
  - 按距离建边：NE所属站点距离小于K的NE被认为具备等价link的关联
  - --no-topo: 不使用拓扑连接信息，只用距离信息构造传播边

输入:
  - 告警数据JSONL文件
  - CROSS_alarm_propagation.xlsx (规则)
  - ne_graph.json (NE连接关系，包含link信息)
  - site_graph.json (站点连接关系，用于距离计算)
  - SYS_NE_0306/*.csv (NE信息)

输出:
  - 带有link和group字段的NE传播图
"""

import json
import os
import csv
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Set, Optional, Tuple
from geokdtree import GeoKDTree

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_resources import (
    CROSS_ALARM_PROPAGATION_XLSX,
    resource_display as alarm_resource_display,
)
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_GRAPH_JSON,
    SYS_NE_DIR,
    resource_display,
)


class DistanceChecker:
    """
    距离检查器 - 惰性计算，只在需要时查询
    根据NE所属站点之间的距离来判断NE是否"相邻"

    筛选条件: k近邻 且 距离 <= distance_k
    """
    def __init__(self, site_graph_file: str = SITE_GRAPH_JSON, knn: int = 5, distance_k: float = 5000):
        """
        Args:
            site_graph_file: site_graph.json路径
            knn: 数量阈值（个），取最近的k个站点
            distance_k: 距离阈值（米），只保留距离内的站点
        """
        self.site_graph_file = site_graph_file
        self.knn = knn
        self.distance_k = distance_k
        self._tree = None  # 延迟初始化的KD-Tree
        self._record = {}  # 缓存: site_id -> nearby sites

    @property
    def tree(self):
        """延迟加载KD-Tree"""
        if self._tree is None:
            self._tree = GeoKDTree()
            self._tree.build(self.site_graph_file)
        return self._tree

    def is_nearby_sites(self, site1: str, site2: str) -> dict:
        """
        检查两个站点是否满足k近邻且距离条件

        Args:
            site1, site2: 站点ID

        Returns:
            是否满足条件
        """
        if site1 == site2:
            return {site2: 0}

        # 检查缓存
        if site2 in self._record.get(site1, {}):
            return {site2: self._record[site1][site2]}
        if site1 in self._record.get(site2, {}):
            return {site2: self._record[site2][site1]}

        # 查询 site1 的 k近邻且距离内的站点
        neighbors = self.tree.nearest_neighbors(site1, k=self.knn, distance=self.distance_k)
        neighbor_ids = {nb['id']: nb['distance'] for nb in neighbors}
        self._record[site1] = neighbor_ids

        if site2 in neighbor_ids:
            return {site2: neighbor_ids[site2]}
        return {}

    def nearby_sites(self, site_id: str) -> dict:
        """
        获取满足k近邻且距离条件的站点ID

        Args:
            site_id: 站点ID

        Returns:
            满足条件的站点ID集合
        """
        if site_id in self._record:
            return self._record[site_id]

        neighbors = self.tree.nearest_neighbors(site_id, k=self.knn, distance=self.distance_k)
        self._record[site_id] = {nb['id']: nb['distance'] for nb in neighbors}
        return self._record[site_id]


def load_site_graph(graph_file: str = SITE_GRAPH_JSON) -> dict:
    """
    加载站点连接图

    Returns:
        {
          site_id1: {site_id2: {link_type: direction, ...}, ...},
          ...
        }
    """
    if os.path.exists(graph_file):
        with open(graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_ne_graph(graph_file: str = NE_GRAPH_JSON) -> dict:
    """
    从ne_graph.json加载NE连接图

    ne_graph.json 格式:
    {
      "ne_id": {
        "domain": ...,
        "name": ...,
        "manufacturer": ...,
        "region_id": ...,
        "site_id": ...,
        "site_name": ...,
        "site_type": ...,
        "longitude": ...,
        "latitude": ...,
        "link": {
          "neighbor_id": {
            "link_type1": "direction",
            ...
          }
        }
      },
      ...
    }

    Returns:
        {ne_id: {link: {neighbor_id: {link_type: direction, ...}}, ...}}
    """
    if os.path.exists(graph_file):
        with open(graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_ne_info(data_dir: str = SYS_NE_DIR) -> dict:
    """
    从SYS_NE_0306加载NE信息

    Returns:
        {nativeId: {ne_id, name, site_id, site_name, type, manufacturer, ...}}
    """
    ne_info = {}

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'SYS_NE' in f]
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nativeId = row.get('nativeId', '').strip().upper()
                if not nativeId:
                    continue
                if nativeId not in ne_info:
                    ne_info[nativeId] = {
                        'ne_id': row.get('ne_id', '').strip().upper(),
                        'name': row.get('name', '').strip(),
                        'site_id': row.get('ne_site_id', '').strip().upper(),
                        'site_name': row.get('ne_site_name', '').strip(),
                        'type': row.get('typeId', '').strip(),
                        'network_type': row.get('network_type', '').strip(),
                        'class_Name': row.get('class_Name', '').strip(),
                        'manufacturer': row.get('manufacturer', '').strip(),
                        'running_status': row.get('running_status', '').strip(),
                        'region_id': row.get('regionId1', '').strip(),
                        'domain': row.get('domain', '').strip(),
                    }

    return ne_info


def load_rules(excel_file: str = CROSS_ALARM_PROPAGATION_XLSX) -> list:
    """
    加载告警传播规则
    """
    import pandas as pd
    df = pd.read_excel(excel_file)

    rules = []
    for _, row in df.iterrows():
        left_alarms = set()
        for alarm in str(row['*Left Alarm']).split('|'):
            alarm = alarm.strip()
            if alarm:
                left_alarms.add(alarm.upper())

        right_alarms = set()
        for alarm in str(row['*Right Alarm']).split('|'):
            alarm = alarm.strip()
            if alarm:
                right_alarms.add(alarm.upper())

        time_window = int(row['Alarm Aggregation Time Window (min)']) if pd.notna(row['Alarm Aggregation Time Window (min)']) else 0
        topology_type = row['*Alarm Aggregation Topology Type'] if pd.notna(row['*Alarm Aggregation Topology Type']) else ''

        rules.append({
            'left_alarms': left_alarms,
            'right_alarms': right_alarms,
            'time_window': time_window,
            'topology_type': topology_type,
        })

    return rules


def load_alarms(jsonl_file: str, ne_graph: dict) -> list:
    """
    加载告警数据

    Returns:
        [
          {
            'alarm_id': str,
            'ne_id': str,  # 告警源
            'alarm_type': str,
            'alarm_time': datetime,
            ...
          },
          ...
        ]
    """
    alarms = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)

            alarm_type = record.get('告警标题', '') or record.get('告警标准名', '')
            alarm_time_str = record.get('告警首次发生时间', '')

            # 解析时间
            alarm_time = None
            if alarm_time_str:
                try:
                    alarm_time = datetime.strptime(alarm_time_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass

            # 告警源作为NE标识
            ne_id = record.get('告警源', '') or record.get('告警资源标识', '')

            if ne_id not in ne_graph:
                continue

            alarms.append({
                'alarm_id': record.get('告警编码ID', ''),
                'ne_id': ne_id.upper(),
                'site_id': record.get('站点ID', '').upper(),
                'alarm_type': alarm_type.upper(),
                'type': record.get('设备类型', ''),
                'network_type': record.get('网络类型', ''),
                'manufacturer': record.get('设备厂家名称', ''),
                'domain': record.get('设备类型', '') or record.get('网络类型', '') or record.get('网络专业', ''),
                'alarm_time': alarm_time,
                'raw': record,
            })

    return alarms


class Link(object):
    def __init__(self, ne1: str, ne2: str, link_type: str, distance: float):
        self.left_ne = ne1
        self.right_ne = ne2
        self.link_type = link_type
        self.distance = distance


class Path(object):
    def __init__(self, links):
        self.links = links


def check_ne_connection_range(
        ne1: str,
        ne2: str,
        min_hops: int,
        max_hops: int,
        ne_graph: dict,
        distance_checker=None,
        site_to_nes: Optional[Dict[str, Set[str]]] = None,
        use_topo: bool = False,
        use_distance: bool = False,
        record_rule: bool = False,
        find_all: bool = False
) -> List:
    """
    检查两个NE之间是否在指定跳数范围内存在连接

    Args:
        min_hops: 最小跳数（>=1），例如 CrossNE 变体要求 >=2
        max_hops: 最大跳数（>=min_hops）
        find_all: False=找到第一条满足条件的路径立即返回（最快）；
                  True=找全所有满足条件的路径（直到搜索完所有<=max_hops的可能性）

    Returns:
        List[Path]: 若 find_all=False，最多返回1条路径；若 find_all=True，返回所有满足条件的路径
    """
    # 参数校验
    if min_hops > max_hops or max_hops < 1 or ne1 == ne2 or ne1 not in ne_graph or ne2 not in ne_graph:
        return []

    # 数字转英文单词（支持1-20，可扩展）
    def num_to_word(n: int) -> str:
        mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                   6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
                   11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
                   15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
                   19: 'nineteen', 20: 'twenty'}
        return mapping.get(n, f"{n}_hop")

    def get_neighbors(current_ne: str, same_site: bool = False) -> Dict[str, Tuple[str, float]]:
        """获取下一跳候选（拓扑优先去重）"""
        ne_data = ne_graph.get(current_ne, {})
        current_site = ne_data.get('site_id', '')

        neighbors = {}

        # 1. 拓扑连接（优先级高）
        if use_topo:
            for neighbor_id in ne_data.get('link', {}):
                link_types = ','.join(ne_data['link'][neighbor_id]['link_type'].keys())
                if not same_site or current_site == ne_graph.get(neighbor_id, {}).get('site_id'):
                    if neighbor_id != current_ne:
                        neighbors[neighbor_id] = (f'one_hop_link:{link_types}', -1)

        # 2. 距离连接（补充未覆盖的NE）
        if use_distance and distance_checker and current_site and site_to_nes:
            nearby = distance_checker.nearby_sites(current_site)
            if isinstance(nearby, dict):
                for site_id, distance in nearby.items():
                    if site_id in site_to_nes:
                        for target_ne in site_to_nes[site_id]:
                            if target_ne != current_ne and target_ne not in neighbors:
                                neighbors[target_ne] = ('one_hop_distance', distance)
        return neighbors

    def adjust_link_types(links: List, actual_hops: int) -> List:
        """将路径中所有边的 'one_hop' 替换为实际跳数（如 'three_hop'）"""
        if not links:
            return links
        target_word = num_to_word(actual_hops)
        adjusted = []
        for link in links:
            # 将 one_hop_link/distance 替换为 {actual_hops}_hop_link/distance
            new_type = link.link_type.replace('one', target_word, 1)
            adjusted.append(Link(link.left_ne, link.right_ne, new_type, link.distance))
        return adjusted

    def combine_links(links: List, actual_hops: int):
        """合并多跳为单条传播边（用于record_rule）"""
        total_dist = sum(link.distance for link in links)
        has_dist = any('distance' in link.link_type for link in links)
        link_type = f"{num_to_word(actual_hops)}_hop_{'distance' if has_dist else 'link'}"
        return Link(ne1, ne2, link_type, total_dist)

    ne1_site = ne_graph.get(ne1, {}).get('site_id', {})

    # BFS: (当前NE, 路径Link列表, 已访问集合, 当前深度)
    queue = deque([(ne1, [], {ne1}, {ne1_site}, 0)])
    valid_paths = [] if find_all else None

    while queue:
        current_ne, path_links, visited_ne, visited_site, depth = queue.popleft()

        # 剪枝：当前深度已达max_hops，无法继续扩展（该节点只作为终点检查，不再向外扩展）
        if depth >= max_hops:
            continue

        if len(visited_site) > 3:
            continue

        same_site = len(visited_site) == 3

        neighbors = get_neighbors(current_ne, same_site)

        for next_ne, (link_type, distance) in neighbors.items():
            new_depth = depth + 1

            # 超过max_hops不处理
            if new_depth > max_hops:
                continue

            if next_ne in visited_ne:
                continue

            # 构造新边（暂用one_hop标记，最终调整）
            new_link = Link(current_ne, next_ne, link_type, distance)
            new_path_links = path_links + [new_link]

            # 到达目标检查
            if next_ne == ne2:
                # 检查是否满足最小跳数要求
                if new_depth >= min_hops:
                    if record_rule:
                        final_link = combine_links(new_path_links, new_depth)
                        result = Path([final_link])
                    else:
                        # 调整边类型为实际跳数标记
                        adjusted = adjust_link_types(new_path_links, new_depth)
                        result = Path(adjusted)

                    # 关键逻辑：如果不找全，立即返回（提前结束！）
                    if not find_all:
                        return [result]

                    valid_paths.append(result)
                    # 继续搜索其他可能（如还有其他同层节点）
                    continue
            else:
                # 未到达目标，继续BFS扩展（需要未访问过且还有跳数额度）
                new_visited_ne = visited_ne | {next_ne}
                next_site = ne_graph.get(next_ne, {}).get('site_id', {})
                new_visited_site = visited_site | {next_site}
                queue.append((next_ne, new_path_links, new_visited_ne, new_visited_site, new_depth))

    return valid_paths if find_all else []


def check_ne_connection(
        ne1: str,
        ne2: str,
        topology_type: str,
        ne_graph: dict,
        distance_checker=None,
        site_to_nes: dict = None,
        no_topo: bool = False,
        record_rule: bool = False,
) -> List:

    mapping = {
        'CrossNE': (1, 1, 'top,distance'),
        'ConnectNE': (2, 5, 'topo'),
    }

    if topology_type not in mapping:
        return []

    min_h, max_h, link_type = mapping[topology_type]
    return check_ne_connection_range(
        ne1, ne2, min_h, max_h, ne_graph,
        distance_checker, site_to_nes,
        use_topo=not no_topo and 'topo' in link_type,
        use_distance=no_topo or 'distance' in link_type,
        record_rule=record_rule,
        find_all=True
    )


def build_propagation_graph(alarms: list, rules: list, ne_graph: dict,
                            knn: int = None, distance_k: float = None,
                            no_topo: bool = False, record_rule: bool = False,
                            site_graph_file: str = SITE_GRAPH_JSON) -> dict:
    """
    根据告警传播规则构建NE传播图

    Args:
        alarms: 告警列表
        rules: 规则列表
        ne_graph: NE连接图
        knn: 数量阈值（个），如果指定则启用近邻建边
        distance_k: 距离阈值（米），如果指定则启用距离建边
        no_topo: 是否不使用拓扑连接信息，只使用距离信息
        record_rule: 边的方式通过rule构建（传播边），默认是拓扑构建边（拓扑边）
    """
    # 创建距离检查器（延迟初始化）
    distance_checker = None
    if knn or distance_k:
        distance_checker = DistanceChecker(site_graph_file, knn=knn or 5, distance_k=distance_k or 5000)

    # ========== 步骤0: 建立站点到NE的映射（用于快速查找）==========
    site_to_nes = defaultdict(set)
    for ne_id, ne_info in ne_graph.items():
        site_id = ne_info.get('site_id', '')
        if site_id:
            site_to_nes[site_id].add(ne_id)
    # ========== 步骤1: 建立告警类型索引 ==========
    alarm_by_type = defaultdict(list)
    valid_alarms = []

    for i, alarm in enumerate(alarms):
        ne_id = alarm['ne_id']
        alarm_type = alarm['alarm_type']
        alarm_time = alarm['alarm_time']

        if not ne_id or not alarm_type or not alarm_time:
            continue

        valid_alarms.append(i)
        alarm_by_type[alarm_type].append({
            'index': i,
            'alarm_time': alarm_time,
            'ne_id': ne_id,
            'alarm': alarm
        })

        if ne_id not in ne_graph:
            ne_graph[ne_id] = {"site_id": alarm["site_id"], "link": {}}
        elif not ne_graph[ne_id]["site_id"]:
            ne_graph[ne_id]["site_id"] = alarm["site_id"]

    # ========== 步骤2: 按规则匹配告警 ==========
    propagation_edges = defaultdict(dict)  # {ne1: {ne2: {reason: {...}}}, ...}
    processed_pairs = set()

    for rule in tqdm(rules):
        left_types = rule['left_alarms']
        right_types = rule['right_alarms']
        time_window = rule['time_window']
        topology = rule['topology_type']

        if topology not in ['CrossNE', 'ConnectNE']:
            continue

        left_alarms = []
        for lt in left_types:
            left_alarms.extend(alarm_by_type.get(lt, []))

        right_alarms = []
        for rt in right_types:
            right_alarms.extend(alarm_by_type.get(rt, []))

        left_alarms.sort(key=lambda x: x['alarm_time'])
        right_alarms.sort(key=lambda x: x['alarm_time'])

        right_idx = 0
        for left in tqdm(left_alarms):
            left_time = left['alarm_time']
            left_ne = left['ne_id']
            left_idx = left['index']
            left_alarm_info = left.get('alarm', {})

            while right_idx < len(right_alarms):
                time_diff_min = (left_time - right_alarms[right_idx]['alarm_time']).total_seconds() / 60
                if time_diff_min <= time_window:
                    break
                right_idx += 1

            j = right_idx
            while j < len(right_alarms):
                right = right_alarms[j]
                time_diff_min = (right['alarm_time'] - left_time).total_seconds() / 60

                if time_diff_min > time_window:
                    break

                if left_idx == right['index']:
                    j += 1
                    continue

                pair_key = tuple(sorted([left_idx, right['index']]))
                if pair_key in processed_pairs:
                    j += 1
                    continue

                processed_pairs.add(pair_key)

                right_ne = right['ne_id']
                right_alarm_info = right.get('alarm', {})

                if right_ne in propagation_edges[left_ne] or left_ne in propagation_edges[right_ne]:
                    continue

                # 检查连接并记录原因
                paths = check_ne_connection(left_ne, right_ne, topology, ne_graph, distance_checker, site_to_nes, no_topo, record_rule)
                for path in paths:
                    for link in path.links:
                        link_left_ne = link.left_ne
                        link_right_ne = link.right_ne
                        if link_right_ne not in propagation_edges[link_left_ne] and link_left_ne not in propagation_edges[link_right_ne]:
                            # 建立边原因
                            reason = {
                                'topology': topology,
                                'time_window': time_window,
                                'left_alarm': {
                                    'ne_id': left_ne,
                                    'alarm_type': left_alarm_info.get('alarm_type', left.get('alarm_type', '')),
                                    'alarm_time': left.get('alarm_time', '').strftime('%Y-%m-%d %H:%M:%S') if left.get('alarm_time') else '',
                                    'alarm_id': left_alarm_info.get('alarm_id', ''),
                                },
                                'right_alarm': {
                                    'ne_id': right_ne,
                                    'alarm_type': right_alarm_info.get('alarm_type', right.get('alarm_type', '')),
                                    'alarm_time': right.get('alarm_time', '').strftime('%Y-%m-%d %H:%M:%S') if right.get('alarm_time') else '',
                                    'alarm_id': right_alarm_info.get('alarm_id', ''),
                                },
                                'connection_type': link.link_type,
                                'distance': link.distance
                            }

                            propagation_edges[link_right_ne][link_left_ne] = reason
                            propagation_edges[link_left_ne][link_right_ne] = reason

                j += 1

    # ========== 步骤3: 按NE分组告警 ==========
    ne_alarms = defaultdict(list)
    for i in valid_alarms:
        alarm = alarms[i]
        ne_alarms[alarm['ne_id']].append({**alarm, 'index': i})

    # ========== 步骤4: 计算联通分量 ==========
    all_nes = set(ne_alarms.keys()) | set(propagation_edges.keys())
    visited = set()
    component_id = 0
    ne_groups = {}

    for ne in all_nes:
        if ne not in visited:
            component_id += 1
            queue = deque([ne])
            visited.add(ne)

            while queue:
                current = queue.popleft()
                ne_groups[current] = component_id

                for neighbor in propagation_edges.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

    # ========== 步骤5: 构建结果 ==========
    ne_info = {}
    group_nes = defaultdict(list)

    for ne_id, alarms_list in ne_alarms.items():
        group_id = ne_groups.get(ne_id, 0)

        # 将 propagation_edges 转换为 link_info 格式
        link_info = {}
        for neighbor, reason in propagation_edges.get(ne_id, {}).items():
            link_info[neighbor] = {
                'connection_type': reason.get('connection_type', ''),
                'distance': reason.get('distance', ''),
                'topology': reason.get('topology', ''),
                'time_window': reason.get('time_window', 0),
                'left_alarm': reason.get('left_alarm', {}),
                'right_alarm': reason.get('right_alarm', {}),
            }

        ne_info[ne_id] = {
            'alarm': [{
                'alarm_id': a['alarm_id'],
                'alarm_type': a['alarm_type'],
                'alarm_time': a['alarm_time'].strftime('%Y-%m-%d %H:%M:%S') if a['alarm_time'] else '',
                'domain': a['domain'],
            } for a in alarms_list],
            'link': link_info,  # 改为字典格式
            'group': group_id,
        }
        for key in ['type', 'network_type', 'manufacturer', 'domain']:
            values = [alarm[key] for alarm in alarms_list if alarm[key]]
            ne_info[ne_id][key] = values[0].upper() if values else ''

        if group_id > 0:
            group_nes[group_id].append(ne_id)

    for ne_id in propagation_edges:
        if ne_id not in ne_info:
            group_id = ne_groups.get(ne_id, 0)

            # 将 propagation_edges 转换为 link_info 格式
            link_info = {}
            for neighbor, reason in propagation_edges.get(ne_id, {}).items():
                link_info[neighbor] = {
                    'connection_type': reason.get('connection_type', ''),
                    'distance': reason.get('distance', ''),
                    'topology': reason.get('topology', ''),
                    'time_window': reason.get('time_window', 0),
                    'left_alarm': reason.get('left_alarm', {}),
                    'right_alarm': reason.get('right_alarm', {}),
                }

            ne_info[ne_id] = {
                'alarm': [],
                'link': link_info,  # 改为字典格式
                'group': group_id,
            }
            if group_id > 0:
                group_nes[group_id].append(ne_id)

    # 只保留NE数大于1的group，只包含ne_list
    group_info = {}
    for gid, nes in group_nes.items():
        # if len(nes) > 1:
        group_info[str(gid)] = {'ne_list': nes, 'site_list': []}

    return {'ne_info': ne_info, 'group_info': group_info}


def check_and_update(data_dict, new_dict):
    for key in new_dict:
        if new_dict[key]:
            data_dict[key] = new_dict[key]


def add_ne_info(result: dict, ne_info_file: dict, ne_graph: dict, site_graph: dict) -> dict:
    """
    从SYS_NE_0306添加NE基本信息，同时填充group_info中的site_list

    优先使用ne_graph.json中的信息，如果不存在则从SYS_NE补充
    """
    if not isinstance(result, dict) or 'ne_info' not in result:
        return result

    # 收集每个group的site列表
    group_sites = defaultdict(set)

    for ne_id, data in result['ne_info'].items():
        # 优先从ne_graph.json获取基本信息
        if ne_id in ne_graph:
            graph_data = ne_graph[ne_id]
            site_id = graph_data.get('site_id', '')
            check_and_update(
                result['ne_info'][ne_id],
                {
                    'name': graph_data.get('name', ''),
                    'site_id': site_id,
                    'type': graph_data.get('type', '').upper(),
                    'network_type': graph_data.get('network_type', '').upper(),
                    'manufacturer': graph_data.get('manufacturer', '').upper(),
                    'running_status': graph_data.get('running_status', graph_data.get('status', '')),
                    'domain': graph_data.get('domain', '').upper(),
                }
            )
            for key in ['site_name', 'longitude', 'latitude', 'region_id']:
                value = graph_data.get(key, '') or site_graph.get(site_id, {}).get(key, '')
                result['ne_info'][ne_id][key] = value

        # 从SYS_NE补充缺失的信息
        if ne_id in ne_info_file:
            sys_ne = ne_info_file[ne_id]
            for key in ['name', 'site_id', 'site_name', 'type', 'network_type',
                        'manufacturer', 'running_status', 'region_id', 'domain']:
                if not result['ne_info'][ne_id].get(key):
                    result['ne_info'][ne_id][key] = sys_ne.get(key, '')

        # 收集site信息用于group
        site_id = result['ne_info'][ne_id].get('site_id', '')
        group_id = data.get('group', 0)
        if site_id and group_id > 0:
            group_sites[group_id].add(site_id)

    # 填充group_info中的site_list
    if 'group_info' in result:
        for gid, group_data in result['group_info'].items():
            gid_int = int(gid)
            if gid_int in group_sites:
                result['group_info'][gid]['site_list'] = sorted(list(group_sites[gid_int]))

    return result


def main():
    parser = argparse.ArgumentParser(description='根据告警传播规则生成NE传播图')
    parser.add_argument('alarm_file', type=str, help='告警数据JSONL文件')
    parser.add_argument('--rules', type=str, default=CROSS_ALARM_PROPAGATION_XLSX,
                        help=f'规则文件，默认: {alarm_resource_display("CROSS_alarm_propagation.xlsx")}')
    parser.add_argument('--ne-graph', type=str, default=NE_GRAPH_JSON,
                        help=f'NE连接图文件 (包含link信息)，默认: {resource_display("ne_graph.json")}')
    parser.add_argument('--site-graph', type=str, default=SITE_GRAPH_JSON,
                        help=f'站点连接图文件 (用于距离计算)，默认: {resource_display("site_graph.json")}')
    parser.add_argument('--ne-dir', type=str, default=SYS_NE_DIR,
                        help=f'NE数据目录，默认: {resource_display("SYS_NE_0306")}')
    parser.add_argument('--knn', type=int, default=0,
                        help='近邻数量阈值，取最近的k个站点')
    parser.add_argument('--distance-k', type=float, default=0,
                        help='距离阈值（公里），距离内的NE被认为具备等价link的关联')
    parser.add_argument('--no-topo', action='store_true',
                        help='不使用拓扑连接信息，只使用距离信息构造传播边')
    parser.add_argument('--record-rule', action='store_true',
                        help='建立边使用传播边')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='输出JSON文件')

    args = parser.parse_args()

    # 加载NE连接图 (直接从ne_graph.json加载)
    print("加载NE连接图...")
    ne_graph = load_ne_graph(args.ne_graph)
    print(f"  NE数: {len(ne_graph)}")

    # 加载站点图（用于距离计算）
    print("加载站点连接图...")
    site_graph = load_site_graph(args.site_graph)
    print(f"  站点数: {len(site_graph)}")

    # 加载NE信息 (作为补充)
    # print("加载NE信息...")
    # ne_info = load_ne_info(args.ne_dir)
    # print(f"  NE数量: {len(ne_info)}")

    # 加载规则
    print("加载规则...")
    rules = load_rules(args.rules)
    print(f"  共 {len(rules)} 条规则")

    # 加载告警
    print("加载告警...")
    alarms = load_alarms(args.alarm_file, ne_graph)
    print(f"  共 {len(alarms)} 条告警")

    # 转换距离参数
    distance_k = args.distance_k * 1000 if args.distance_k > 0 else None

    # 打印模式信息
    if args.no_topo:
        print(f"\n模式: --no-topo (只使用距离信息，不使用拓扑连接)")
    else:
        print(f"\n模式: 正常模式 (拓扑连接 + 距离信息)")

    # 构建传播图
    print("\n构建传播图...")
    result = build_propagation_graph(alarms, rules, ne_graph,
                                     knn=args.knn or None,
                                     distance_k=distance_k,
                                     no_topo=args.no_topo,
                                     record_rule=args.record_rule,
                                     site_graph_file=args.site_graph)

    # 添加NE信息
    print("添加NE信息...")
    # result = add_ne_info(result, ne_info, ne_graph, site_graph)
    result = add_ne_info(result, {}, ne_graph, site_graph)

    # 保存结果
    if not args.output:
        suffix = ""
        if args.knn > 0:
            suffix += f"_knn{args.knn}"
        if args.distance_k > 0:
            suffix += f"_k{args.distance_k}"
        if args.no_topo:
            suffix += "_notopo"
        args.output = args.alarm_file.replace('.jsonl', f'_ne_propagation{suffix}.json')
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n生成文件: {args.output}")
    print(f"NE数: {len(result.get('ne_info', {}))}")
    # print(f"有效Group数（NE数>1）: {len(result.get('group_info', {}))}")
    print(f"有效Group数: {len(result.get('group_info', {}))}")


if __name__ == "__main__":
    main()
