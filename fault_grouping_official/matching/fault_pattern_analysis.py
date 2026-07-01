"""故障模式分析与记录增强。

主要功能：

  1) 过滤判定：站点连通分量拆分、断站吸收、链/环模式识别（analyze_case_record、
     filter_other_patterns / SiteRelationIndex 等）。
  2) 记录增强：故障模式备注（note）、fault_pattern_* 字段、补充相关站点/网元/链路
     （supplemental_fault_pattern_context，供 ne_propagation_visualizer.html 标红展示）。
"""

import heapq

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

from fault_grouping_official.alarm_types import OFFLINE_ALARMS
from fault_grouping_official.site_topology import (
    build_site_topology_from_ne_graph,
    load_site_chain_index,
)

OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)
ROUTER_DEVICE_DOMAINS = {"DATA"}
OTHER_FAULT_PATTERNS = {"ip_ring_others"}
_EMPTY_NEIGHBORS = frozenset()
PATTERN_PRIORITY = {
    "ip_chain_single_link": 0,
    "ip_chain_multi_link": 1,
    "ip_ring_single_upstream": 2,
    "ip_ring_multi_upstream": 3,
    "ip_ring_others": 4,
    "unknown": 99,
}

# —— 性能保护参数 ——
# longest_path_in_component 对链/环走线性快路径，其余 ≤N 个站的分量用 bitmask 记忆化
# 求精确最长简单链。该问题本身仍是指数级；双 BFS 近似在带环分量上偏差严重
# （哈密顿覆盖几乎 100% 漏判），会把环型模式误判成 unknown，不可用于落盘判定。
#
# 因此这里把阈值做成可配，且 >N 的分量直接返回空链——经 classify_component 的覆盖校验落为
# unknown -> 该故障组被丢弃，而不是给出错误的近似链。
# 默认 18；复杂分量仍需按数据规模权衡精确搜索成本与召回。
LONGEST_PATH_EXACT_MAX_SITES = 18

# 整组站点数上限：故障组站点总数超过该值，直接判定不可保留（在分析前丢弃）。
# 虽然连通分量已按邻接边遍历、断站吸收已改为增量候选堆，但候选规模仍可能达到
# O(n²)，且与 LONGEST_PATH_EXACT_MAX_SITES 无关。emitted_group_store 合并出的巨型组
# （成百上千站）仍会带来明显的内存和 CPU 开销。
# 而这类巨型组几乎不可能满足"单连通分量 + ≤N 路由链/环"的保留条件，故直接丢弃，
# 用一次 O(站点数) 的计数把二次候选规模挡在门外。
# 默认 200；按数据规模和允许的最大单组候选规模调整。
MAX_ANALYSIS_SITES = 200


def normalize_text(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def as_dict(value):
    return value if isinstance(value, dict) else {}


def normalize_site_list(values):
    seen = set()
    result = []
    for value in values or []:
        site_id = normalize_text(value)
        if site_id and site_id not in seen:
            seen.add(site_id)
            result.append(site_id)
    return result


def extract_record_uuid(record):
    match_info = as_dict(record.get("match_info"))
    return normalize_text(match_info.get("uuid") or record.get("uuid"))


def extract_case_sites(record):
    site_ids = set()
    group_info = as_dict(record.get("group_info"))
    for group_meta in group_info.values():
        if isinstance(group_meta, dict):
            site_ids.update(normalize_site_list(group_meta.get("site_list", [])))

    for field_name in ("ticket_sites", "associated_sites", "missing_sites"):
        site_ids.update(normalize_site_list(record.get(field_name, [])))

    match_info = as_dict(record.get("match_info"))
    role_mapping = as_dict(match_info.get("role_mapping") or record.get("role_mapping"))
    for value in role_mapping.values():
        site_ids.update(normalize_site_list(value if isinstance(value, list) else [value]))

    for symptom in record.get("symptoms", []) or []:
        if isinstance(symptom, dict):
            site_id = normalize_text(symptom.get("node"))
            if site_id:
                site_ids.add(site_id)

    return sorted(site_ids)


def extract_alarm_name(record):
    if not isinstance(record, dict):
        return ""
    return (
        normalize_text(record.get("alarm"))
        or normalize_text(record.get("alarm_type"))
        or normalize_text(record.get("告警标题"))
    )


def extract_domain(record):
    if not isinstance(record, dict):
        return ""
    return (
        normalize_text(record.get("domain"))
        or normalize_text(record.get("Domain"))
        or normalize_text(record.get("DOMAIN"))
        or normalize_text(record.get("alarm_source_domain"))
        or normalize_text(record.get("告警源专业"))
    ).upper()


def extract_record_site(record, ne_to_site):
    site_id = normalize_text(record.get("node")) or normalize_text(record.get("site_id")) or normalize_text(record.get("站点ID"))
    if site_id:
        return site_id
    alarm_source = normalize_text(record.get("alarm_source")) or normalize_text(record.get("告警源"))
    return normalize_text(ne_to_site.get(alarm_source, ""))


def build_site_has_router_device_map(ne_graph_data):
    site_has_router_device = defaultdict(bool)
    for ne_info in ne_graph_data.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = normalize_text(ne_info.get("site_id"))
        if not site_id:
            continue
        if extract_domain(ne_info) in ROUTER_DEVICE_DOMAINS:
            site_has_router_device[site_id] = True
    return dict(site_has_router_device)


def extract_case_router_device_sites(record, site_has_router_device, site_ids=None):
    if site_ids is None:
        site_ids = extract_case_sites(record)
    router_sites = {
        site_id
        for site_id in site_ids
        if site_has_router_device.get(site_id, False)
    }

    for ne_meta in as_dict(record.get("ne_info")).values():
        if not isinstance(ne_meta, dict):
            continue
        site_id = normalize_text(ne_meta.get("site_id"))
        if site_id and extract_domain(ne_meta) in ROUTER_DEVICE_DOMAINS:
            router_sites.add(site_id)
        for alarm in ne_meta.get("alarm", []) or []:
            if site_id and isinstance(alarm, dict) and extract_domain(alarm) in ROUTER_DEVICE_DOMAINS:
                router_sites.add(site_id)

    return router_sites


def extract_offline_sites(record, ne_to_site):
    offline_sites = set()

    for symptom in record.get("symptoms", []) or []:
        if not isinstance(symptom, dict):
            continue
        if extract_alarm_name(symptom) in OFFLINE_ALARM_SET:
            site_id = extract_record_site(symptom, ne_to_site)
            if site_id:
                offline_sites.add(site_id)

    for ne_id, ne_meta in as_dict(record.get("ne_info")).items():
        site_id = normalize_text(as_dict(ne_meta).get("site_id")) or normalize_text(ne_to_site.get(ne_id, ""))
        for alarm in as_dict(ne_meta).get("alarm", []) or []:
            if isinstance(alarm, dict) and extract_alarm_name(alarm) in OFFLINE_ALARM_SET and site_id:
                offline_sites.add(site_id)

    return offline_sites


class SiteRelationIndex:
    def __init__(self, ne_graph_data=None, site_chains_path=""):
        self.site_chains = {}
        self.downstream_direct = defaultdict(set)
        self.upstream_direct = defaultdict(set)
        self.bidirectional_direct = defaultdict(set)
        self._undirected_neighbors_cache = {}
        self._ring_neighbors_cache = {}
        self._upstream_distance_cache = {}
        # 只缓存 site_chains 预计算索引中缺失、由 BFS 补出的距离；不复制原 hop dict。
        self._supplemental_upstream_distances_cache = {}
        self.precomputed_upstream_hops_complete = False

        if site_chains_path:
            self.site_chains, _valid_sites = load_site_chain_index(site_chains_path)
            self._load_direct_relations_from_site_chains()
        elif ne_graph_data:
            downstream_map, _valid_sites = build_site_topology_from_ne_graph(ne_graph_data)
            for upstream_site, downstream_sites in downstream_map.items():
                for downstream_site in downstream_sites:
                    self.downstream_direct[upstream_site].add(downstream_site)
                    self.upstream_direct[downstream_site].add(upstream_site)

    def _load_direct_relations_from_site_chains(self):
        self._invalidate_neighbor_caches()
        for site_id, chain_info in self.site_chains.items():
            for downstream_site, hop in chain_info.get("downstream_site_hops", {}).items():
                if hop == 1:
                    self.downstream_direct[site_id].add(downstream_site)
                    self.upstream_direct[downstream_site].add(site_id)
            for upstream_site, hop in chain_info.get("upstream_site_hops", {}).items():
                if hop == 1:
                    self.upstream_direct[site_id].add(upstream_site)
                    self.downstream_direct[upstream_site].add(site_id)
            for neighbor_site in chain_info.get("bidirectional_sites", set()):
                self.bidirectional_direct[site_id].add(neighbor_site)
                self.bidirectional_direct[neighbor_site].add(site_id)

    def _invalidate_neighbor_caches(self):
        self._undirected_neighbors_cache.clear()
        self._ring_neighbors_cache.clear()

    def upstream_distance(self, downstream_site, upstream_site):
        if downstream_site == upstream_site:
            return 0

        chain_info = self.site_chains.get(downstream_site, {})
        upstream_hops = chain_info.get("upstream_site_hops", {})
        if upstream_site in upstream_hops:
            return upstream_hops[upstream_site]

        cache_key = (downstream_site, upstream_site)
        if cache_key in self._upstream_distance_cache:
            return self._upstream_distance_cache[cache_key]

        queue = deque([(downstream_site, 0)])
        visited = {downstream_site}
        while queue:
            site_id, hop = queue.popleft()
            for parent_site in self.upstream_direct.get(site_id, set()):
                if parent_site in visited:
                    continue
                if parent_site == upstream_site:
                    self._upstream_distance_cache[cache_key] = hop + 1
                    return hop + 1
                visited.add(parent_site)
                queue.append((parent_site, hop + 1))

        self._upstream_distance_cache[cache_key] = None
        return None

    def _supplemental_upstream_distances(self, downstream_site, precomputed_hops):
        if self.precomputed_upstream_hops_complete:
            return {}

        cache = self._supplemental_upstream_distances_cache
        if downstream_site not in cache:
            supplemental_distances = {}
            queue = deque([(downstream_site, 0)])
            visited = {downstream_site}
            while queue:
                site_id, hop = queue.popleft()
                for parent_site in self.upstream_direct.get(site_id, set()):
                    if parent_site in visited:
                        continue
                    visited.add(parent_site)
                    if parent_site not in precomputed_hops:
                        supplemental_distances.setdefault(parent_site, hop + 1)
                    queue.append((parent_site, hop + 1))
            cache[downstream_site] = supplemental_distances
        return cache[downstream_site]

    @staticmethod
    def _valid_upstream_hop(downstream_site, upstream_site, hop):
        return (
            upstream_site != downstream_site
            and hop is not None
            and hop > 0
        )

    def iter_upstream_distances(self, downstream_site):
        """迭代某站点的全部可达上游及距离，不复制预计算 hop 字典。"""
        chain_info = self.site_chains.get(downstream_site, {})
        precomputed_hops = chain_info.get("upstream_site_hops", {})
        for upstream_site, hop in precomputed_hops.items():
            if self._valid_upstream_hop(downstream_site, upstream_site, hop):
                yield upstream_site, hop

        supplemental_distances = self._supplemental_upstream_distances(
            downstream_site,
            precomputed_hops,
        )
        yield from supplemental_distances.items()

    def iter_upstream_distances_in(self, downstream_site, allowed_sites):
        """只迭代 allowed_sites 内的可达上游，优先遍历较小集合。

        小故障组无需扫描站点在全局拓扑中的全部上游；预计算表较小时仍顺序遍历
        该表，避免对大故障组做过多随机查表。
        """
        chain_info = self.site_chains.get(downstream_site, {})
        precomputed_hops = chain_info.get("upstream_site_hops", {})
        supplemental_distances = self._supplemental_upstream_distances(
            downstream_site,
            precomputed_hops,
        )
        if isinstance(allowed_sites, (set, frozenset)):
            allowed_site_set = allowed_sites
        else:
            allowed_site_set = set(allowed_sites)
        upstream_count = len(precomputed_hops) + len(supplemental_distances)

        if len(allowed_site_set) <= upstream_count:
            for upstream_site in allowed_site_set:
                if upstream_site in precomputed_hops:
                    hop = precomputed_hops[upstream_site]
                else:
                    hop = supplemental_distances.get(upstream_site)
                if self._valid_upstream_hop(
                    downstream_site,
                    upstream_site,
                    hop,
                ):
                    yield upstream_site, hop
            return

        for upstream_site, hop in precomputed_hops.items():
            if (
                upstream_site in allowed_site_set
                and self._valid_upstream_hop(
                    downstream_site,
                    upstream_site,
                    hop,
                )
            ):
                yield upstream_site, hop
        for upstream_site, hop in supplemental_distances.items():
            if upstream_site in allowed_site_set:
                yield upstream_site, hop

    def directly_connected(self, site_a, site_b):
        if site_a == site_b:
            return False
        return (
            site_b in self.downstream_direct.get(site_a, _EMPTY_NEIGHBORS)
            or site_a in self.downstream_direct.get(site_b, _EMPTY_NEIGHBORS)
            or site_b in self.bidirectional_direct.get(site_a, _EMPTY_NEIGHBORS)
            or site_a in self.bidirectional_direct.get(site_b, _EMPTY_NEIGHBORS)
        )

    def undirected_neighbors(self, site_id):
        cached = self._undirected_neighbors_cache.get(site_id)
        if cached is None:
            cached = frozenset().union(
                self.downstream_direct.get(site_id, _EMPTY_NEIGHBORS),
                self.upstream_direct.get(site_id, _EMPTY_NEIGHBORS),
                self.bidirectional_direct.get(site_id, _EMPTY_NEIGHBORS),
            )
            self._undirected_neighbors_cache[site_id] = cached
        return cached

    def ring_neighbors(self, site_id):
        """环模式使用的双向及直接上游邻居（静态缓存）。"""
        cached = self._ring_neighbors_cache.get(site_id)
        if cached is None:
            cached = frozenset().union(
                self.bidirectional_direct.get(site_id, _EMPTY_NEIGHBORS),
                self.upstream_direct.get(site_id, _EMPTY_NEIGHBORS),
            )
            self._ring_neighbors_cache[site_id] = cached
        return cached

    def bidirectional_neighbor_set(self, site_id):
        return self.bidirectional_direct.get(site_id, _EMPTY_NEIGHBORS)

    def direct_upstream_neighbor_set(self, site_id):
        return self.upstream_direct.get(site_id, _EMPTY_NEIGHBORS)

    def direct_neighbors(self, site_id):
        # 保留原有“返回可变 set”的公开行为；内部热路径直接使用缓存的 frozenset。
        return set(self.undirected_neighbors(site_id))

    def non_downstream_neighbors(self, site_id):
        return sorted(
            self.undirected_neighbors(site_id).difference(
                self.downstream_direct.get(site_id, _EMPTY_NEIGHBORS)
            )
        )

    def bidirectional_neighbors(self, site_id):
        return sorted(self.bidirectional_neighbor_set(site_id))

    def direct_upstream_neighbors(self, site_id):
        return sorted(self.direct_upstream_neighbor_set(site_id))

    def undirected_neighbors_in(self, site_id, site_set):
        return {
            neighbor
            for neighbor in self.undirected_neighbors(site_id)
            if neighbor in site_set
        }


def absorb_unmanaged_downstream_sites(site_ids, initial_unmanaged_sites, relation_index):
    """按最近上游逐步吸收断站，并保持原有 tuple 最小值选择语义。

    旧实现每吸收一个站点都会重新枚举全部 ``unmanaged × remaining`` 候选，最坏
    接近 O(n³)。这里让每个首次成为 unmanaged 的站点只生成一次局部候选堆，
    全局堆仅保留各站点当前最优项，并继续按
    ``(distance, unmanaged_site, upstream_site)`` 选择；站点被移除后按需推进局部堆。
    """
    remaining = set(site_ids)
    unmanaged_sites = set(initial_unmanaged_sites) & remaining
    absorbed_by = {}
    absorb_steps = []

    # 每个 unmanaged 站点维护一个局部上游候选堆；全局堆只放每个站点当前
    # 最优候选，将全局 heap 操作从 O(候选总数) 降为接近 O(站点数)。
    candidates = []
    upstream_candidates_by_site = {}
    seeded_unmanaged_sites = set()

    def push_next_candidate(unmanaged_site):
        site_candidates = upstream_candidates_by_site[unmanaged_site]
        while site_candidates:
            distance, parent_site = heapq.heappop(site_candidates)
            if parent_site not in remaining:
                continue
            heapq.heappush(
                candidates,
                (distance, unmanaged_site, parent_site),
            )
            return

    def seed_candidates(unmanaged_site):
        if unmanaged_site in seeded_unmanaged_sites or unmanaged_site not in remaining:
            return
        seeded_unmanaged_sites.add(unmanaged_site)
        site_candidates = [
            (distance, upstream_site)
            for upstream_site, distance in relation_index.iter_upstream_distances_in(
                unmanaged_site,
                remaining,
            )
        ]
        heapq.heapify(site_candidates)
        upstream_candidates_by_site[unmanaged_site] = site_candidates
        push_next_candidate(unmanaged_site)

    for unmanaged_site in unmanaged_sites:
        seed_candidates(unmanaged_site)

    while candidates:
        distance, unmanaged_site, parent_site = heapq.heappop(candidates)
        if unmanaged_site not in remaining:
            continue
        if parent_site not in remaining:
            push_next_candidate(unmanaged_site)
            continue

        remaining.remove(unmanaged_site)
        # 该站点不会重新进入 remaining，立即释放尚未消费的局部候选。
        upstream_candidates_by_site.pop(unmanaged_site, None)
        absorbed_by[unmanaged_site] = parent_site
        parent_was_unmanaged = parent_site in unmanaged_sites
        unmanaged_sites.add(parent_site)
        if not parent_was_unmanaged:
            seed_candidates(parent_site)
        absorb_steps.append({
            "site": unmanaged_site,
            "absorbed_by": parent_site,
            "upstream_hops": distance,
            "new_unmanaged_site": parent_site,
        })

    return remaining, unmanaged_sites & remaining, absorbed_by, absorb_steps


def iter_connected_components(nodes, relation_index):
    node_set = set(nodes)
    while node_set:
        start = min(node_set)
        queue = deque([start])
        node_set.remove(start)
        component = {start}
        while queue:
            site_id = queue.popleft()
            for neighbor in relation_index.undirected_neighbors_in(site_id, node_set):
                node_set.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        yield component


def connected_components(nodes, relation_index):
    return list(iter_connected_components(nodes, relation_index))


def projected_active_components_by_original_graph(
    original_sites,
    active_sites,
    relation_index,
    max_components=None,
):
    """按原始站点图划分连通分量，再投影出吸收后仍保留的站点。"""
    active_site_set = set(active_sites)
    projected_components = []
    for original_component in iter_connected_components(original_sites, relation_index):
        component_active_sites = set(original_component) & active_site_set
        if component_active_sites:
            projected_components.append(component_active_sites)
            if max_components is not None and len(projected_components) >= max_components:
                break
    return projected_components


@dataclass
class PreparedPatternCase:
    """单条记录的模式分析中间结果，供过滤与分类共享。"""

    site_ids: list
    offline_sites: set
    router_device_sites: set
    active_sites: set
    active_unmanaged_sites: set
    absorbed_by: dict
    absorb_steps: list
    projected_components: list


def prepare_case_record(
    record,
    relation_index,
    ne_to_site,
    site_has_router_device,
    site_ids=None,
    component_limit=None,
):
    """提取并计算一次模式分析公共数据。

    ``site_ids`` 允许调用方传入从原始 match 轻量提取的等价站点集合；
    ``component_limit=2`` 用于 one-component-only 过滤，发现第二个分量即可停止。
    """
    if site_ids is None:
        site_ids = extract_case_sites(record)
    else:
        site_ids = sorted(set(site_ids))
    site_id_set = set(site_ids)
    offline_sites = extract_offline_sites(record, ne_to_site) & site_id_set
    router_device_sites = extract_case_router_device_sites(
        record,
        site_has_router_device,
        site_ids=site_ids,
    )
    active_sites, active_unmanaged_sites, absorbed_by, absorb_steps = (
        absorb_unmanaged_downstream_sites(
            site_ids,
            offline_sites,
            relation_index,
        )
    )
    projected_components = projected_active_components_by_original_graph(
        site_ids,
        active_sites,
        relation_index,
        max_components=component_limit,
    )
    return PreparedPatternCase(
        site_ids=site_ids,
        offline_sites=offline_sites,
        router_device_sites=router_device_sites,
        active_sites=active_sites,
        active_unmanaged_sites=active_unmanaged_sites,
        absorbed_by=absorbed_by,
        absorb_steps=absorb_steps,
        projected_components=projected_components,
    )


def longest_path_in_component(component, relation_index):
    component = set(component)
    if len(component) <= 1:
        return sorted(component)

    adjacency = {
        site_id: sorted(relation_index.undirected_neighbors_in(site_id, component))
        for site_id in component
    }

    def traverse_degree_two_component(start, first_neighbor):
        path = [start]
        previous = None
        current = start
        next_site = first_neighbor
        while next_site is not None and next_site not in path:
            path.append(next_site)
            previous, current = current, next_site
            unvisited_neighbors = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous and neighbor not in path
            ]
            next_site = unvisited_neighbors[0] if unvisited_neighbors else None
        return path

    # 链和环是模式识别的常见输入。度数 ≤2 且连通时可线性构造精确最长链，
    # 无需进入通用的指数级搜索；同时保持旧实现“最长后取字典序最小”的结果。
    if all(len(neighbors) <= 2 for neighbors in adjacency.values()):
        endpoints = sorted(
            site_id for site_id, neighbors in adjacency.items() if len(neighbors) == 1
        )
        if len(endpoints) == 2:
            path = traverse_degree_two_component(
                endpoints[0],
                adjacency[endpoints[0]][0],
            )
            if len(path) == len(component):
                reversed_path = list(reversed(path))
                return min(path, reversed_path)
        elif not endpoints and all(len(neighbors) == 2 for neighbors in adjacency.values()):
            start = min(component)
            candidates = [
                traverse_degree_two_component(start, neighbor)
                for neighbor in adjacency[start]
            ]
            covering_paths = [
                path for path in candidates if len(path) == len(component)
            ]
            if covering_paths:
                return min(covering_paths)

    # 其余 ≤N 分量使用 bitmask 记忆化求精确最长简单链。相同的
    # (末端节点, 已访问集合) 只计算一次，避免旧 DFS 对同一子问题反复穷举。
    if len(component) <= LONGEST_PATH_EXACT_MAX_SITES:
        nodes = sorted(component)
        node_to_index = {site_id: index for index, site_id in enumerate(nodes)}
        adjacency_indexes = [
            tuple(node_to_index[neighbor] for neighbor in adjacency[site_id])
            for site_id in nodes
        ]

        @lru_cache(maxsize=None)
        def best_length(last_index, visited_mask):
            best = 1
            for neighbor_index in adjacency_indexes[last_index]:
                neighbor_bit = 1 << neighbor_index
                if visited_mask & neighbor_bit:
                    continue
                candidate = 1 + best_length(
                    neighbor_index,
                    visited_mask | neighbor_bit,
                )
                if candidate > best:
                    best = candidate
            return best

        start_lengths = [
            best_length(index, 1 << index)
            for index in range(len(nodes))
        ]
        remaining_length = max(start_lengths)
        current_index = next(
            index
            for index, length in enumerate(start_lengths)
            if length == remaining_length
        )
        visited_mask = 1 << current_index
        path = [nodes[current_index]]

        while remaining_length > 1:
            for neighbor_index in adjacency_indexes[current_index]:
                neighbor_bit = 1 << neighbor_index
                if visited_mask & neighbor_bit:
                    continue
                next_mask = visited_mask | neighbor_bit
                if best_length(neighbor_index, next_mask) != remaining_length - 1:
                    continue
                current_index = neighbor_index
                visited_mask = next_mask
                path.append(nodes[current_index])
                remaining_length -= 1
                break
            else:  # pragma: no cover - 防御性兜底，理论上不会发生
                break
        return path

    # >N 的分量不采用误差较大的双 BFS 近似，直接返回空链。classify_component 的
    # `set(chain) != set(unmanaged_component)` 覆盖校验会因此跳过该分量，最终落为
    # unknown -> 故障组被丢弃，避免给出错误的模式分类。
    return []


def classify_chain_uplink(chain, component_sites, relation_index):
    chain = list(chain)
    chain_set = set(chain)
    other_sites = set(component_sites) - chain_set
    external_connected_chain_sites = {
        chain_site
        for chain_site in chain
        if not relation_index.undirected_neighbors(chain_site).isdisjoint(other_sites)
    }
    endpoints = {chain[0], chain[-1]} if chain else set()
    subtype = "single_uplink" if external_connected_chain_sites <= endpoints else "multi_uplink"
    return subtype, sorted(external_connected_chain_sites)


def build_context_edges_for_anchors(anchors, related_sites_by_anchor, relation_type):
    context_edges = []
    seen = set()
    for anchor_site in anchors:
        for related_site in sorted(related_sites_by_anchor.get(anchor_site, set())):
            if related_site == anchor_site:
                continue
            edge_key = (related_site, anchor_site, relation_type)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            context_edges.append({
                "supplemental_site": related_site,
                "anchor_site": anchor_site,
                "relation": relation_type,
            })
    return context_edges


def bidirectional_or_upstream_neighbors(site_id, relation_index):
    return relation_index.ring_neighbors(site_id)


def has_two_bidirectional_or_upstream_neighbors(site_id, relation_index):
    return len(bidirectional_or_upstream_neighbors(site_id, relation_index)) == 2


def classify_ip_ring_chain(chain, relation_index):
    chain = list(chain)
    chain_set = set(chain)
    if len(chain) < 2:
        return "ip_ring_others", [], []

    endpoints = [chain[0], chain[-1]]
    endpoint_set = set(endpoints)
    endpoint_bidir = {
        endpoint: relation_index.bidirectional_neighbor_set(endpoint)
        for endpoint in endpoints
    }

    if all(has_two_bidirectional_or_upstream_neighbors(site_id, relation_index) for site_id in chain):
        context_edges = build_context_edges_for_anchors(
            endpoints,
            {
                endpoint: endpoint_bidir[endpoint] - chain_set
                for endpoint in endpoints
            },
            "bidirectional",
        )
        return "ip_ring_single_upstream", context_edges, []

    internal_sites = [site_id for site_id in chain if site_id not in endpoint_set]
    internal_ring_degree_ok = all(
        has_two_bidirectional_or_upstream_neighbors(site_id, relation_index)
        for site_id in internal_sites
    )
    endpoint_condition_failed = any(
        not has_two_bidirectional_or_upstream_neighbors(endpoint, relation_index)
        for endpoint in endpoints
    )
    endpoint_upstream = {
        endpoint: relation_index.direct_upstream_neighbor_set(endpoint)
        for endpoint in endpoints
    }
    common_bidir_sites = sorted((endpoint_bidir[endpoints[0]] & endpoint_bidir[endpoints[1]]) - chain_set)
    common_upstream_sites = sorted((endpoint_upstream[endpoints[0]] & endpoint_upstream[endpoints[1]]) - chain_set)

    if internal_ring_degree_ok and endpoint_condition_failed and (common_bidir_sites or common_upstream_sites):
        bidir_edges = build_context_edges_for_anchors(
            endpoints,
            {
                endpoint: endpoint_bidir[endpoint] - chain_set
                for endpoint in endpoints
            },
            "bidirectional",
        )
        upstream_edges = build_context_edges_for_anchors(
            endpoints,
            {
                endpoint: endpoint_upstream[endpoint] - chain_set
                for endpoint in endpoints
            },
            "upstream",
        )
        return "ip_ring_multi_upstream", bidir_edges + upstream_edges, [
            {"relation": "common_bidirectional", "sites": common_bidir_sites},
            {"relation": "common_upstream", "sites": common_upstream_sites},
        ]

    return "ip_ring_others", [], []


def has_absorbed_site_for_final(final_site, absorbed_by):
    for absorbed_site in sorted(absorbed_by):
        if absorbed_site == final_site:
            continue
        visited = {absorbed_site}
        parent_site = absorbed_by.get(absorbed_site)
        while parent_site:
            if parent_site == final_site:
                return True
            if parent_site in visited:
                break
            visited.add(parent_site)
            parent_site = absorbed_by.get(parent_site)
    return False


def classify_component(
    component_sites,
    unmanaged_sites,
    relation_index,
    router_device_sites=None,
    absorbed_by=None,
    recognized_patterns_only=False,
    cap_hits=None,
):
    component_sites = set(component_sites)
    component_unmanaged_sites = set(unmanaged_sites) & component_sites
    router_device_sites = set(router_device_sites or [])
    absorbed_by = absorbed_by or {}
    non_router_sites = component_sites - router_device_sites

    if non_router_sites:
        return {
            "pattern": "unknown",
            "sites": sorted(component_sites),
            "unmanaged_sites": sorted(component_unmanaged_sites),
            "managed_sites": [],
            "final_site": "",
            "final_managed_site": "",
            "chains": [],
            "non_router_sites": sorted(non_router_sites),
        }

    if len(component_unmanaged_sites) == 1:
        candidate = next(iter(component_unmanaged_sites))
        non_downstream_neighbors = relation_index.non_downstream_neighbors(candidate)
        if len(non_downstream_neighbors) == 1:
            pattern = "ip_chain_single_link"
        elif len(non_downstream_neighbors) >= 2:
            pattern = "ip_chain_multi_link"
        else:
            pattern = "unknown"
        if pattern == "ip_chain_single_link" and not has_absorbed_site_for_final(candidate, absorbed_by):
            pattern = "unknown"

        if pattern == "unknown":
            return {
                "pattern": "unknown",
                "sites": sorted(component_sites),
                "unmanaged_sites": [candidate],
                "managed_sites": [],
                "final_site": "",
                "final_managed_site": "",
                "chains": [],
                "non_downstream_connected_sites": non_downstream_neighbors,
            }

        return {
            "pattern": pattern,
            "sites": sorted(component_sites),
            "unmanaged_sites": [candidate],
            "managed_sites": [candidate],
            "final_site": candidate if pattern == "ip_chain_single_link" else "",
            "final_managed_site": candidate,
            "non_downstream_connected_sites": non_downstream_neighbors,
            "chains": [],
        }

    if recognized_patterns_only:
        # 多断站分量最终只会保留两类 ring 模式：single 要求所有链节点满足
        # “双向或上游邻居数 == 2”，multi 最多允许两个不满足条件的端点。
        # 超过两个不合格站点时，无论最长链如何都只会落到 others/unknown，
        # 可以在指数级最长路径搜索前安全丢弃。
        ring_degree_mismatch_count = sum(
            not has_two_bidirectional_or_upstream_neighbors(site_id, relation_index)
            for site_id in component_unmanaged_sites
        )
        if ring_degree_mismatch_count > 2:
            return {
                "pattern": "unknown",
                "sites": sorted(component_sites),
                "unmanaged_sites": sorted(component_unmanaged_sites),
                "managed_sites": sorted(component_unmanaged_sites),
                "final_site": "",
                "final_managed_site": "",
                "chains": [],
            }

    unmanaged_components = connected_components(component_unmanaged_sites, relation_index)
    if recognized_patterns_only and len(unmanaged_components) != 1:
        return {
            "pattern": "unknown",
            "sites": sorted(component_sites),
            "unmanaged_sites": sorted(component_unmanaged_sites),
            "managed_sites": sorted(component_unmanaged_sites),
            "final_site": "",
            "final_managed_site": "",
            "chains": [],
        }
    unmanaged_chains = []
    chain_covered_sites = set()
    for unmanaged_component in unmanaged_components:
        chain = longest_path_in_component(unmanaged_component, relation_index)
        if not chain and len(unmanaged_component) > LONGEST_PATH_EXACT_MAX_SITES:
            # longest_path 只在 >LONGEST_PATH_EXACT_MAX_SITES 的非链/环分量上放弃精确
            # 搜索、返回空链（纯链/环走线性快路径不会返回空）。据此统计"因 18 上限被丢弃"。
            if cap_hits is not None:
                cap_hits[0] += 1
        if len(chain) < 2:
            continue
        if set(chain) != set(unmanaged_component):
            continue
        subtype, external_sites = classify_chain_uplink(chain, component_sites, relation_index)
        chain_covered_sites.update(chain)
        unmanaged_chains.append({
            "chain": chain,
            "length": len(chain),
            "uplink_type": subtype,
            "external_connected_chain_sites": external_sites,
        })
    unmanaged_chains.sort(key=lambda item: (-item["length"], item["chain"]))

    if len(unmanaged_chains) == 1 and chain_covered_sites == component_unmanaged_sites:
        ring_chain = unmanaged_chains[0]["chain"]
        ring_pattern, context_edges, ring_common_sites = classify_ip_ring_chain(ring_chain, relation_index)
        return {
            "pattern": ring_pattern,
            "sites": sorted(component_sites),
            "unmanaged_sites": sorted(component_unmanaged_sites),
            "managed_sites": ring_chain,
            "final_site": "",
            "final_managed_site": "",
            "chains": unmanaged_chains,
            "supplemental_context_edges": context_edges,
            "ring_common_sites": ring_common_sites,
        }

    return {
        "pattern": "unknown",
        "sites": sorted(component_sites),
        "unmanaged_sites": sorted(component_unmanaged_sites),
        "managed_sites": sorted(component_unmanaged_sites),
        "final_site": "",
        "final_managed_site": "",
        "chains": unmanaged_chains,
    }


def update_analysis_patterns(analysis, component_records):
    component_records = list(component_records)
    primary_pattern = "none"
    if component_records:
        primary_pattern = min(
            (item["pattern"] for item in component_records),
            key=lambda pattern: PATTERN_PRIORITY.get(pattern, 99),
        )
    managed_sites = sorted({
        site_id
        for component in component_records
        for site_id in component.get("managed_sites", [])
    })
    matched_unmanaged_sites = sorted({
        site_id
        for component in component_records
        for site_id in component.get("unmanaged_sites", [])
    })

    analysis["pattern"] = primary_pattern
    analysis["patterns"] = component_records
    analysis["pattern_count"] = len(component_records)
    analysis["active_unmanaged_sites"] = matched_unmanaged_sites
    analysis["managed_sites"] = managed_sites
    analysis["final_site"] = component_records[0].get("final_site", "") if len(component_records) == 1 else ""
    analysis["final_managed_site"] = (
        component_records[0].get("final_managed_site", "") if len(component_records) == 1 else ""
    )
    analysis["chains"] = [
        chain
        for component in component_records
        for chain in component.get("chains", [])
    ]
    return analysis


def filter_other_patterns(analysis):
    patterns = analysis.get("patterns", [])
    if not isinstance(patterns, list) or not patterns:
        return analysis

    filtered_patterns = [
        pattern_info
        for pattern_info in patterns
        if as_dict(pattern_info).get("pattern") not in OTHER_FAULT_PATTERNS
    ]
    if len(filtered_patterns) == len(patterns):
        return analysis
    return update_analysis_patterns(dict(analysis), filtered_patterns)


def analyze_prepared_case(
    record,
    prepared_case,
    relation_index,
    recognized_patterns_only=False,
    cap_hits=None,
):
    """使用已计算的公共中间结果完成逐分量分类和分析对象构建。

    cap_hits 传入 [int] 单元素列表时，会累计"因断站簇 > LONGEST_PATH_EXACT_MAX_SITES
    放弃精确搜索"的次数，供调用方统计被 18 上限丢弃的故障组。
    """
    site_ids = prepared_case.site_ids
    offline_sites = prepared_case.offline_sites
    router_device_sites = prepared_case.router_device_sites
    active_sites = prepared_case.active_sites
    active_unmanaged_sites = prepared_case.active_unmanaged_sites
    absorbed_by = prepared_case.absorbed_by
    absorb_steps = prepared_case.absorb_steps
    projected_components = prepared_case.projected_components
    component_records = []
    for component_sites in projected_components:
        component_record = classify_component(
            component_sites,
            active_unmanaged_sites,
            relation_index,
            router_device_sites=router_device_sites,
            absorbed_by=absorbed_by,
            recognized_patterns_only=recognized_patterns_only,
            cap_hits=cap_hits,
        )
        if component_record.get("pattern") != "unknown":
            component_records.append(component_record)
    component_records.sort(key=lambda item: (item["pattern"], item["sites"]))

    analysis = {
        "uuid": extract_record_uuid(record),
        "rule": normalize_text(as_dict(record.get("match_info")).get("rule") or record.get("rule")),
        "site_count": len(site_ids),
        "sites": site_ids,
        "component_count": len(projected_components),
        "offline_sites": sorted(offline_sites),
        "site_down_count": len(offline_sites),
        "router_device_sites": sorted(router_device_sites & set(site_ids)),
        "active_sites_after_absorption": sorted(active_sites),
        "absorbed_by": absorbed_by,
        "absorb_steps": absorb_steps,
    }
    return update_analysis_patterns(analysis, component_records)


def analyze_case_record(record, relation_index, ne_to_site, site_has_router_device):
    prepared_case = prepare_case_record(
        record,
        relation_index,
        ne_to_site,
        site_has_router_device,
    )
    return analyze_prepared_case(record, prepared_case, relation_index)


def format_pattern_summary_line(pattern_info, index):
    pattern = pattern_info.get("pattern", "unknown")
    managed_sites = pattern_info.get("managed_sites", []) or []

    if pattern.startswith("ip_ring"):
        chains = pattern_info.get("chains", []) or []
        chain = chains[0].get("chain", []) if chains and isinstance(chains[0], dict) else []
        matched_text = "->".join(chain) if chain else "->".join(managed_sites)
    else:
        matched_text = "->".join(managed_sites)

    return f"模式{index}：{pattern}（{matched_text or '无'}）"


def build_pattern_note(analysis):
    patterns = analysis.get("patterns", []) or []
    if not patterns:
        return ""

    lines = ["故障模式挖掘："]
    lines.extend(
        format_pattern_summary_line(pattern_info, index)
        for index, pattern_info in enumerate(patterns, 1)
    )
    return "\n".join(lines)


def format_link_context(raw_link):
    if isinstance(raw_link, dict):
        return dict(raw_link)
    if raw_link in (None, ""):
        return {}
    return {"connection_type": str(raw_link)}


def build_context_ne_info(ne_id, ne_graph_entry, group_id, site_graph_entry=None):
    site_graph_entry = as_dict(site_graph_entry)
    return {
        "link": {},
        "group": group_id,
        "name": ne_graph_entry.get("name", ne_id),
        "site_id": normalize_text(ne_graph_entry.get("site_id")),
        "site_name": (
            normalize_text(ne_graph_entry.get("site_name"))
            or normalize_text(site_graph_entry.get("site_name"))
            or normalize_text(ne_graph_entry.get("site_id"))
        ),
        "type": normalize_text(ne_graph_entry.get("type")).upper(),
        "network_type": normalize_text(ne_graph_entry.get("network_type")).upper(),
        "manufacturer": normalize_text(ne_graph_entry.get("manufacturer")).upper(),
        "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
        "domain": normalize_text(ne_graph_entry.get("domain")).upper(),
        "region_id": normalize_text(ne_graph_entry.get("region_id")) or normalize_text(site_graph_entry.get("region_id")),
        "longitude": (
            ne_graph_entry.get("longitude")
            or ne_graph_entry.get("lon")
            or ne_graph_entry.get("lng")
            or site_graph_entry.get("longitude", "")
        ),
        "latitude": (
            ne_graph_entry.get("latitude")
            or ne_graph_entry.get("lat")
            or site_graph_entry.get("latitude", "")
        ),
        "alarm": [],
        "supplemental_fault_pattern_context": True,
    }


def add_bidirectional_link(ne_info, source_ne, target_ne, link_context):
    if source_ne not in ne_info or target_ne not in ne_info:
        return
    if not link_context:
        link_context = {"connection_type": "supplemental_fault_pattern_context"}
    source_links = ne_info[source_ne].setdefault("link", {})
    target_links = ne_info[target_ne].setdefault("link", {})
    source_links.setdefault(target_ne, dict(link_context))
    target_links.setdefault(source_ne, dict(link_context))


def find_ne_link_context(source_ne, target_ne, ne_graph_data):
    source_links = as_dict(as_dict(ne_graph_data.get(source_ne)).get("link"))
    if target_ne in source_links:
        return format_link_context(source_links.get(target_ne))
    target_links = as_dict(as_dict(ne_graph_data.get(target_ne)).get("link"))
    if source_ne in target_links:
        return format_link_context(target_links.get(source_ne))
    return {}


def collect_supplemental_fault_pattern_sites(analysis, existing_sites, site_has_router_device=None):
    existing_sites = {normalize_text(site_id) for site_id in existing_sites if normalize_text(site_id)}
    site_has_router_device = site_has_router_device or {}
    supplemental_sites = []
    seen = set()
    for pattern_info in analysis.get("patterns", []) or []:
        pattern = pattern_info.get("pattern")
        if pattern not in {
            "ip_chain_single_link",
            "ip_chain_multi_link",
            "ip_ring_single_upstream",
            "ip_ring_multi_upstream",
        }:
            continue
        for edge_info in pattern_info.get("supplemental_context_edges", []) or []:
            normalized_site_id = normalize_text(as_dict(edge_info).get("supplemental_site"))
            if (
                normalized_site_id
                and normalized_site_id not in existing_sites
                and normalized_site_id not in seen
                and site_has_router_device.get(normalized_site_id, False)
            ):
                seen.add(normalized_site_id)
                supplemental_sites.append(normalized_site_id)
        for site_id in pattern_info.get("non_downstream_connected_sites", []) or []:
            normalized_site_id = normalize_text(site_id)
            if (
                normalized_site_id
                and normalized_site_id not in existing_sites
                and normalized_site_id not in seen
                and site_has_router_device.get(normalized_site_id, False)
            ):
                seen.add(normalized_site_id)
                supplemental_sites.append(normalized_site_id)
    return supplemental_sites


def collect_supplemental_fault_pattern_edges(analysis, existing_sites, supplemental_sites=None):
    existing_sites = {normalize_text(site_id) for site_id in existing_sites if normalize_text(site_id)}
    supplemental_site_set = {
        normalize_text(site_id)
        for site_id in (supplemental_sites or [])
        if normalize_text(site_id)
    }
    supplemental_edges = []
    seen = set()
    for pattern_info in analysis.get("patterns", []) or []:
        pattern = pattern_info.get("pattern")
        if pattern not in {
            "ip_chain_single_link",
            "ip_chain_multi_link",
            "ip_ring_single_upstream",
            "ip_ring_multi_upstream",
        }:
            continue
        for edge_info in pattern_info.get("supplemental_context_edges", []) or []:
            edge_info = as_dict(edge_info)
            supplemental_site = normalize_text(edge_info.get("supplemental_site"))
            anchor_site = normalize_text(edge_info.get("anchor_site"))
            if (
                not supplemental_site
                or not anchor_site
                or supplemental_site in existing_sites
                or supplemental_site not in supplemental_site_set
            ):
                continue
            edge_key = (supplemental_site, anchor_site)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            supplemental_edges.append({
                "supplemental_site": supplemental_site,
                "anchor_site": anchor_site,
                "pattern": pattern,
                "relation": normalize_text(edge_info.get("relation")),
            })
        unmanaged_sites = [
            normalize_text(site_id)
            for site_id in pattern_info.get("unmanaged_sites", []) or []
            if normalize_text(site_id)
        ]
        if len(unmanaged_sites) != 1:
            continue
        anchor_site = unmanaged_sites[0]
        for site_id in pattern_info.get("non_downstream_connected_sites", []) or []:
            supplemental_site = normalize_text(site_id)
            if (
                not supplemental_site
                or supplemental_site in existing_sites
                or supplemental_site not in supplemental_site_set
            ):
                continue
            edge_key = (supplemental_site, anchor_site)
            if edge_key in seen:
                continue
            seen.add(edge_key)
            supplemental_edges.append({
                "supplemental_site": supplemental_site,
                "anchor_site": anchor_site,
                "pattern": pattern,
            })
    return supplemental_edges


def augment_case_with_supplemental_fault_pattern_sites(
    record,
    analysis,
    ne_graph_data,
    site_to_ne_ids,
    site_has_router_device=None,
    site_graph_data=None,
):
    if not ne_graph_data:
        return
    site_graph_data = site_graph_data or {}

    ne_info = record.setdefault("ne_info", {})
    if not isinstance(ne_info, dict):
        return

    group_id = extract_record_uuid(record) or normalize_text(record.get("uuid")) or "fault_pattern_context"
    existing_sites = extract_case_sites(record)
    supplemental_sites = collect_supplemental_fault_pattern_sites(
        analysis,
        existing_sites,
        site_has_router_device=site_has_router_device,
    )
    if not supplemental_sites:
        return

    existing_ne_ids = set(ne_info.keys())
    supplemental_ne_ids = []
    site_to_added_ne_ids = defaultdict(list)
    for site_id in supplemental_sites:
        for ne_id in site_to_ne_ids.get(site_id, ()):
            ne_graph_entry = as_dict(ne_graph_data.get(ne_id))
            if ne_id not in ne_info:
                ne_info[ne_id] = build_context_ne_info(
                    ne_id,
                    ne_graph_entry,
                    group_id,
                    site_graph_entry=site_graph_data.get(site_id),
                )
            ne_info[ne_id]["supplemental_fault_pattern_context"] = True
            supplemental_ne_ids.append(ne_id)
            site_to_added_ne_ids[site_id].append(ne_id)

    supplemental_edges = collect_supplemental_fault_pattern_edges(
        analysis,
        existing_sites,
        supplemental_sites=supplemental_sites,
    )
    for edge in supplemental_edges:
        supplemental_site = edge["supplemental_site"]
        anchor_site = edge["anchor_site"]
        source_ne_ids = site_to_added_ne_ids.get(supplemental_site, [])
        target_ne_ids = [
            ne_id
            for ne_id in site_to_ne_ids.get(anchor_site, ())
            if ne_id in ne_info
        ]
        for source_ne in source_ne_ids:
            for target_ne in target_ne_ids:
                link_context = find_ne_link_context(source_ne, target_ne, ne_graph_data)
                if link_context:
                    link_context["supplemental_fault_pattern_context"] = True
                    add_bidirectional_link(ne_info, source_ne, target_ne, link_context)

    group_info = record.setdefault("group_info", {})
    if isinstance(group_info, dict):
        group_entry = group_info.setdefault(group_id, {})
        if isinstance(group_entry, dict):
            group_entry["site_list"] = sorted(set(group_entry.get("site_list", [])) | set(supplemental_sites))
            group_entry["ne_list"] = sorted(set(group_entry.get("ne_list", [])) | set(existing_ne_ids) | set(supplemental_ne_ids))
            group_entry["supplemental_fault_pattern_sites"] = supplemental_sites

    record["fault_pattern_supplemental_sites"] = supplemental_sites
    record["fault_pattern_supplemental_ne_ids"] = sorted(set(supplemental_ne_ids))
    record["fault_pattern_supplemental_edges"] = supplemental_edges


def append_note(original_note, pattern_note):
    original_note = normalize_text(original_note)
    pattern_note = normalize_text(pattern_note)
    if not pattern_note:
        return original_note
    if not original_note:
        return pattern_note
    if pattern_note in original_note:
        return original_note
    return f"{original_note.rstrip()}\n\n{pattern_note}"
