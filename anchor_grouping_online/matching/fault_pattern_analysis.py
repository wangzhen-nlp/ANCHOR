"""故障模式过滤分析：连通分量拆分、断站吸收及链/环模式识别。"""

import heapq

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache

from anchor_grouping_online.alarm_types import OFFLINE_ALARMS

OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)
ROUTER_DEVICE_DOMAINS = {"DATA"}
_EMPTY_NEIGHBORS = frozenset()

# —— 性能保护参数 ——
# longest_path_in_component 对链/环走线性快路径，其余 ≤N 个站的分量用 bitmask 记忆化
# 求精确最长简单链。该问题本身仍是指数级；双 BFS 近似在带环分量上偏差严重
# （哈密顿覆盖几乎 100% 漏判），会把环型模式误判成 unknown。
#
# 因此这里把阈值做成可配，且 >N 的分量直接返回空链。
# 经 classify_component 的覆盖校验落为
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


def extract_case_sites(record):
    site_ids = set()
    role_mapping = as_dict(record.get("role_mapping"))
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
    return normalize_text(record.get("alarm"))


def extract_domain(record):
    if not isinstance(record, dict):
        return ""
    return normalize_text(record.get("domain")).upper()


def extract_record_site(record, ne_to_site):
    site_id = normalize_text(record.get("node"))
    if site_id:
        return site_id
    alarm_source = normalize_text(record.get("alarm_source"))
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


def extract_case_router_device_sites(site_has_router_device, site_ids):
    return {
        site_id
        for site_id in site_ids
        if site_has_router_device.get(site_id, False)
    }


def extract_offline_sites(record, ne_to_site):
    offline_sites = set()

    for symptom in record.get("symptoms", []) or []:
        if not isinstance(symptom, dict):
            continue
        if extract_alarm_name(symptom) in OFFLINE_ALARM_SET:
            site_id = extract_record_site(symptom, ne_to_site)
            if site_id:
                offline_sites.add(site_id)

    return offline_sites


class SiteRelationIndex:
    def __init__(self):
        self.site_chains = {}
        self.downstream_direct = defaultdict(set)
        self.upstream_direct = defaultdict(set)
        self.bidirectional_direct = defaultdict(set)
        self._undirected_neighbors_cache = {}
        self._ring_neighbors_cache = {}
        # 只缓存 site_chains 预计算索引中缺失、由 BFS 补出的距离；不复制原 hop dict。
        self._supplemental_upstream_distances_cache = {}
        self.precomputed_upstream_hops_complete = False

    def _load_direct_relations_from_site_chains(self):
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

    def non_downstream_neighbors(self, site_id):
        return sorted(
            self.undirected_neighbors(site_id).difference(
                self.downstream_direct.get(site_id, _EMPTY_NEIGHBORS)
            )
        )

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

    # 每个 unmanaged 站点维护一个局部上游候选堆；全局堆只放每个站点当前
    # 最优候选，将全局 heap 操作从 O(候选总数) 降为接近 O(站点数)。
    candidates = []
    upstream_candidates_by_site = {}
    seeded_unmanaged_sites = set()

    def seed_candidates(unmanaged_site):
        _seed_upstream_candidates(
            unmanaged_site, remaining, relation_index,
            candidates, upstream_candidates_by_site, seeded_unmanaged_sites,
        )

    for unmanaged_site in unmanaged_sites:
        seed_candidates(unmanaged_site)

    while candidates:
        _distance, unmanaged_site, parent_site = heapq.heappop(candidates)
        if unmanaged_site not in remaining:
            continue
        if parent_site not in remaining:
            _push_next_upstream_candidate(
                unmanaged_site, remaining, candidates, upstream_candidates_by_site
            )
            continue

        remaining.remove(unmanaged_site)
        # 该站点不会重新进入 remaining，立即释放尚未消费的局部候选。
        upstream_candidates_by_site.pop(unmanaged_site, None)
        absorbed_by[unmanaged_site] = parent_site
        parent_was_unmanaged = parent_site in unmanaged_sites
        unmanaged_sites.add(parent_site)
        if not parent_was_unmanaged:
            seed_candidates(parent_site)
    return remaining, unmanaged_sites & remaining, absorbed_by


def _push_next_upstream_candidate(
    unmanaged_site, remaining, candidates, upstream_candidates_by_site
):
    """把该站点局部堆中下一个仍然存活的上游候选推入全局堆。"""
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


def _seed_upstream_candidates(
    unmanaged_site, remaining, relation_index,
    candidates, upstream_candidates_by_site, seeded_unmanaged_sites,
):
    """为首次成为 unmanaged 的站点建立局部上游候选堆并推入全局最优。"""
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
    _push_next_upstream_candidate(
        unmanaged_site, remaining, candidates, upstream_candidates_by_site
    )


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
    """单条匹配的模式判定中间结果。"""

    router_device_sites: set
    active_unmanaged_sites: set
    absorbed_by: dict
    projected_components: list


def prepare_case_record(
    record,
    relation_index,
    ne_to_site,
    site_has_router_device,
    site_ids,
    component_limit=None,
):
    """提取并计算一次模式分析公共数据。

    ``site_ids`` 是调用方从原始 match 提取的站点集合；
    ``component_limit=2`` 用于 one-component-only 过滤，发现第二个分量即可停止。
    """
    site_ids = sorted(set(site_ids))
    site_id_set = set(site_ids)
    offline_sites = extract_offline_sites(record, ne_to_site) & site_id_set
    router_device_sites = extract_case_router_device_sites(
        site_has_router_device,
        site_ids,
    )
    active_sites, active_unmanaged_sites, absorbed_by = (
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
        router_device_sites=router_device_sites,
        active_unmanaged_sites=active_unmanaged_sites,
        absorbed_by=absorbed_by,
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

    # 链和环是模式识别的常见输入。度数 ≤2 且连通时可线性构造精确最长链，
    # 无需进入通用的指数级搜索；同时保持旧实现“最长后取字典序最小”的结果。
    if all(len(neighbors) <= 2 for neighbors in adjacency.values()):
        path = _degree_two_longest_path(component, adjacency)
        if path is not None:
            return path

    # 其余 ≤N 分量使用 bitmask 记忆化求精确最长简单链。相同的
    # (末端节点, 已访问集合) 只计算一次，避免旧 DFS 对同一子问题反复穷举。
    if len(component) <= LONGEST_PATH_EXACT_MAX_SITES:
        return _exact_longest_path(component, adjacency)

    # >N 的分量不采用误差较大的双 BFS 近似，直接返回空链。classify_component 的
    # `set(chain) != set(unmanaged_component)` 覆盖校验会因此跳过该分量，最终落为
    # unknown -> 故障组被丢弃，避免给出错误的模式分类。
    return []


def _degree_two_longest_path(component, adjacency):
    """度数 ≤2 分量的线性最长链构造；无法整链覆盖时返回 None 回退穷举。"""

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
    return None


def _exact_longest_path(component, adjacency):
    """bitmask 记忆化求精确最长简单链，并按记忆化结果回溯重建路径。"""
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
    return _reconstruct_longest_path(
        nodes, adjacency_indexes, best_length, start_lengths
    )


def _reconstruct_longest_path(nodes, adjacency_indexes, best_length, start_lengths):
    """按记忆化的 best_length 结果回溯重建一条最长简单链。"""
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


def has_two_bidirectional_or_upstream_neighbors(site_id, relation_index):
    return len(relation_index.ring_neighbors(site_id)) == 2


def classify_ip_ring_chain(chain, relation_index):
    chain = list(chain)
    chain_set = set(chain)
    if len(chain) < 2:
        return "ip_ring_others"

    endpoints = [chain[0], chain[-1]]
    endpoint_set = set(endpoints)
    endpoint_bidir = {
        endpoint: relation_index.bidirectional_neighbor_set(endpoint)
        for endpoint in endpoints
    }

    if all(has_two_bidirectional_or_upstream_neighbors(site_id, relation_index) for site_id in chain):
        return "ip_ring_single_upstream"

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
    has_common_bidir_site = bool(
        (endpoint_bidir[endpoints[0]] & endpoint_bidir[endpoints[1]]) - chain_set
    )
    has_common_upstream_site = bool(
        (endpoint_upstream[endpoints[0]] & endpoint_upstream[endpoints[1]]) - chain_set
    )

    if (
        internal_ring_degree_ok
        and endpoint_condition_failed
        and (has_common_bidir_site or has_common_upstream_site)
    ):
        return "ip_ring_multi_upstream"

    return "ip_ring_others"


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
):
    component_sites = set(component_sites)
    component_unmanaged_sites = set(unmanaged_sites) & component_sites
    router_device_sites = set(router_device_sites or [])
    absorbed_by = absorbed_by or {}
    if component_sites - router_device_sites:
        return "unknown"

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
        return pattern

    # 多断站分量只保留两类 ring 模式：single 要求所有链节点满足
    # “双向或上游邻居数 == 2”，multi 最多允许两个不满足条件的端点。
    # 环度数条件不再作为前置早退，统一交给 classify_ip_ring_chain 判定
    # （度数不满足的链会落为 ip_ring_others）。
    unmanaged_components = list(
        iter_connected_components(component_unmanaged_sites, relation_index)
    )
    if len(unmanaged_components) != 1:
        return "unknown"

    unmanaged_component = unmanaged_components[0]
    ring_chain = longest_path_in_component(unmanaged_component, relation_index)
    if len(ring_chain) < 2 or set(ring_chain) != set(unmanaged_component):
        return "unknown"
    return classify_ip_ring_chain(ring_chain, relation_index)


def has_recognized_fault_pattern(prepared_case, relation_index):
    """是否至少命中一个可参与二次汇聚的故障模式。"""
    router_device_sites = prepared_case.router_device_sites
    active_unmanaged_sites = prepared_case.active_unmanaged_sites
    absorbed_by = prepared_case.absorbed_by
    for component_sites in prepared_case.projected_components:
        pattern = classify_component(
            component_sites,
            active_unmanaged_sites,
            relation_index,
            router_device_sites=router_device_sites,
            absorbed_by=absorbed_by,
        )
        if pattern not in {"unknown"}:
            return True
    return False
