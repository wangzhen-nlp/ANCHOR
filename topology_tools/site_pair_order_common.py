#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站点上下行推断脚本的共享工具函数。"""

import heapq

from collections import Counter, defaultdict, deque

from alarm_tools.progress_utils import ProgressBar


CANONICAL_DOMAIN_MAP = {
    "data": "Data",
    "transmission": "Transmission",
    "ran": "Ran",
}


ROLE_SCORE = {
    "wireless": -2.0,
    "microwave": 0.0,
    "router": 2.0,
    "unknown": 0.0,
}


ROLE_ORDER = {
    "wireless": 0,
    "microwave": 1,
    "router": 2,
    "unknown": 1,
}


class ProgressReporter:
    """轻量进度上下文，方便脚本统一打开/关闭进度条。"""

    def __init__(self, total, label, enabled=True):
        self._bar = ProgressBar(total, label) if enabled else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def update(self, step=1):
        if self._bar is not None:
            self._bar.update(step)

    def set(self, current):
        if self._bar is not None:
            self._bar.set(current)

    def set_extra_text(self, text, force=False):
        if self._bar is not None:
            self._bar.set_extra_text(text, force=force)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def normalize_domain(domain):
    text = str(domain or "").strip()
    if not text:
        return "Unknown"
    return CANONICAL_DOMAIN_MAP.get(text.lower(), text)


def _get_site_id(ne_info):
    return str(ne_info.get("site_id", "")).strip().upper()


def classify_device_role(domain: str) -> str:
    """归一化设备类型: wireless / microwave / router / unknown。"""
    text = normalize_domain(domain or "")
    text = str(text).strip().lower()

    if text in {"wireless", "ran", "radio", "无线"}:
        return "wireless"
    if text in {"microwave", "mw", "微波", "transmission", "传输"}:
        return "microwave"
    if text in {"router", "ip", "ipran", "路由", "data", "数据"}:
        return "router"

    wireless_keywords = [
        "wireless", "radio", "ran", "rru", "bbu", "du", "cu",
        "gnb", "enb", "cell", "无线", "基站", "小区",
    ]
    microwave_keywords = [
        "microwave", "mw", "微波", "波道", "中继", "transmission", "传输",
    ]
    router_keywords = [
        "router", "ipran", "pe", "p", "cr", "sr",
        "路由", "核心路由", "汇聚路由", "data", "数据",
    ]

    if any(keyword in text for keyword in wireless_keywords):
        return "wireless"
    if any(keyword in text for keyword in microwave_keywords):
        return "microwave"
    if any(keyword in text for keyword in router_keywords):
        return "router"
    return "unknown"


def build_site_role_counts(ne_graph):
    site_role_counts = defaultdict(Counter)
    for ne_info in ne_graph.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = _get_site_id(ne_info)
        if not site_id:
            continue
        role = classify_device_role(ne_info.get("domain", ""))
        site_role_counts[site_id][role] += 1
    return site_role_counts


def is_wireless_only_site(site_id, site_role_counts):
    role_counts = site_role_counts.get(site_id, Counter())
    total = sum(role_counts.values())
    return total > 0 and role_counts.get("wireless", 0) == total


def should_include_cross_site_link(source_site, source_domain, target_site, target_domain, site_role_counts):
    """
    判断跨站 NE 边是否可用于站点拓扑推断。

    默认忽略不同 domain 的跨站连接，因为这类边大概率是逻辑边。
    例外：如果某一端站点只有无线设备，则允许该端无线设备跨站连接到
    对端无线或路由设备。
    """
    source_domain = normalize_domain(source_domain)
    target_domain = normalize_domain(target_domain)
    if source_domain == target_domain:
        return True

    source_role = classify_device_role(source_domain)
    target_role = classify_device_role(target_domain)
    allowed_peer_roles = {"wireless", "router"}

    if (
        is_wireless_only_site(source_site, site_role_counts)
        and source_role == "wireless"
        and target_role in allowed_peer_roles
    ):
        return True

    if (
        is_wireless_only_site(target_site, site_role_counts)
        and target_role == "wireless"
        and source_role in allowed_peer_roles
    ):
        return True

    return False


def iter_unique_cross_site_links(ne_graph):
    """按 NE 对 + link_type 去重，遍历跨站点链路。"""
    seen = set()
    site_role_counts = build_site_role_counts(ne_graph)

    for source_ne, source_info in ne_graph.items():
        source_site = _get_site_id(source_info)
        if not source_site:
            continue

        source_domain = normalize_domain(source_info.get("domain", ""))
        raw_links = source_info.get("link", {})
        if not isinstance(raw_links, dict):
            continue

        for target_ne, link_meta in raw_links.items():
            target_info = ne_graph.get(target_ne)
            if not isinstance(target_info, dict):
                continue

            target_site = _get_site_id(target_info)
            if not target_site or target_site == source_site:
                continue

            target_domain = normalize_domain(target_info.get("domain", ""))
            if not should_include_cross_site_link(
                source_site,
                source_domain,
                target_site,
                target_domain,
                site_role_counts,
            ):
                continue

            link_types = (
                sorted(link_meta.keys())
                if isinstance(link_meta, dict) and link_meta
                else ["__unknown__"]
            )

            for link_type in link_types:
                key = tuple(sorted((source_ne, target_ne))) + (str(link_type),)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "source_ne": source_ne,
                    "target_ne": target_ne,
                    "source_site": source_site,
                    "target_site": target_site,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "link_type": str(link_type),
                }


def multi_source_bfs(adjacency, nodes, sources):
    inf = 10**9
    dist = {node: inf for node in nodes}
    queue = deque()

    for source in sources:
        if source in dist:
            dist[source] = 0
            queue.append(source)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, ()):
            if dist[neighbor] > dist[current] + 1:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


def compute_distance_scores(site_stats, adjacency, core_anchors, access_anchors):
    nodes = list(site_stats.keys())
    if not nodes:
        return {}

    if not core_anchors or not access_anchors:
        return {node: 0.0 for node in nodes}

    dist_to_core = multi_source_bfs(adjacency, nodes, core_anchors)
    dist_to_access = multi_source_bfs(adjacency, nodes, access_anchors)

    finite_vals = [
        distance
        for distance in list(dist_to_core.values()) + list(dist_to_access.values())
        if distance < 10**9
    ]
    max_dist = max(finite_vals) if finite_vals else 1

    scores = {}
    for node in nodes:
        core_distance = dist_to_core[node] if dist_to_core[node] < 10**9 else max_dist + 1
        access_distance = (
            dist_to_access[node]
            if dist_to_access[node] < 10**9
            else max_dist + 1
        )
        scores[node] = (access_distance - core_distance) / max(1, max_dist + 1)
    return scores


def find_bridges(adjacency):
    """非递归 Tarjan 桥边识别，避免大拓扑上触发 Python 递归深度限制。"""
    visit_time = 0
    discovery = {}
    low_link = {}
    parent = {}
    bridges = set()

    for start_node in adjacency:
        if start_node in discovery:
            continue

        parent[start_node] = None
        visit_time += 1
        discovery[start_node] = low_link[start_node] = visit_time
        stack = [(start_node, iter(adjacency.get(start_node, ())))]

        while stack:
            node, neighbors_iter = stack[-1]

            try:
                neighbor = next(neighbors_iter)
            except StopIteration:
                stack.pop()
                parent_node = parent.get(node)
                if parent_node is not None:
                    low_link[parent_node] = min(low_link[parent_node], low_link[node])
                    if low_link[node] > discovery[parent_node]:
                        bridges.add(tuple(sorted((parent_node, node))))
                continue

            if neighbor == parent.get(node):
                continue

            if neighbor not in discovery:
                parent[neighbor] = node
                visit_time += 1
                discovery[neighbor] = low_link[neighbor] = visit_time
                stack.append((neighbor, iter(adjacency.get(neighbor, ()))))
            else:
                low_link[node] = min(low_link[node], discovery[neighbor])

    return bridges


def build_strict_ring_context(edge_keys, bridge_pairs, site_scores=None):
    """
    构造严格环约束上下文。

    环块定义为移除桥边后的连通块；环块里通过桥边连到外部的站点视为出入口。
    严格模式下，入口站点相关边强制为入口指向环内站点，其它环块内部边全部双向。
    如果传入 site_scores，则分数最高的出入口站点视为更靠上行的 entry，
    分数最低的出入口站点视为更靠端侧的 exit。
    """
    site_scores = site_scores or {}
    normalized_bridge_pairs = {
        tuple(sorted(pair))
        for pair in bridge_pairs
    }
    non_bridge_neighbors = defaultdict(set)
    bridge_neighbors = defaultdict(set)

    for left_site, right_site in edge_keys:
        pair_key = tuple(sorted((left_site, right_site)))
        if pair_key in normalized_bridge_pairs:
            bridge_neighbors[left_site].add(right_site)
            bridge_neighbors[right_site].add(left_site)
        else:
            non_bridge_neighbors[left_site].add(right_site)
            non_bridge_neighbors[right_site].add(left_site)

    visited = set()
    pair_context = {}
    component_summaries = []

    for start_site in sorted(non_bridge_neighbors):
        if start_site in visited:
            continue

        queue = deque([start_site])
        visited.add(start_site)
        component_sites = []

        while queue:
            site_id = queue.popleft()
            component_sites.append(site_id)
            for neighbor_site in sorted(non_bridge_neighbors.get(site_id, ())):
                if neighbor_site in visited:
                    continue
                visited.add(neighbor_site)
                queue.append(neighbor_site)

        component_set = set(component_sites)
        internal_pairs = []
        entry_exit_sites = set()

        for site_id in component_set:
            for neighbor_site in bridge_neighbors.get(site_id, ()):
                if neighbor_site not in component_set:
                    entry_exit_sites.add(site_id)

            for neighbor_site in non_bridge_neighbors.get(site_id, ()):
                if neighbor_site not in component_set:
                    continue
                internal_pairs.append(tuple(sorted((site_id, neighbor_site))))

        internal_pairs = sorted(set(internal_pairs))
        if not internal_pairs:
            continue

        component_id = len(component_summaries)
        entry_exit_sites = sorted(entry_exit_sites)
        start_site = (
            entry_exit_sites[0]
            if len(entry_exit_sites) == 1
            else None
        )
        scored_entry_exit_sites = [
            {
                "site_id": site_id,
                "score": round(float(site_scores.get(site_id, 0.0)), 6),
                "has_score": site_id in site_scores,
            }
            for site_id in entry_exit_sites
        ]
        scored_sites = [
            site_id
            for site_id in entry_exit_sites
            if site_id in site_scores
        ]
        entry_site = None
        exit_site = None
        if len(entry_exit_sites) == 1:
            entry_site = entry_exit_sites[0]
        elif scored_sites:
            ranked_boundary_sites = sorted(
                entry_exit_sites,
                key=lambda site_id: (
                    float(site_scores.get(site_id, 0.0)),
                    str(site_id),
                ),
            )
            exit_site = ranked_boundary_sites[0]
            entry_site = ranked_boundary_sites[-1]

        component_summaries.append({
            "component_id": component_id,
            "sites": sorted(component_set),
            "site_count": len(component_set),
            "internal_pair_count": len(internal_pairs),
            "external_start_candidates": entry_exit_sites,
            "entry_exit_sites": entry_exit_sites,
            "entry_exit_site_scores": scored_entry_exit_sites,
            "entry_site": entry_site,
            "exit_site": exit_site,
            "start_site": start_site,
        })

        for pair_key in internal_pairs:
            entry_related = entry_site is not None and entry_site in pair_key
            force_bidirectional = (
                entry_site is None
                or entry_site not in pair_key
            )
            pair_context[pair_key] = {
                "component_id": component_id,
                "start_site": start_site,
                "entry_exit_sites": entry_exit_sites,
                "entry_site": entry_site,
                "exit_site": exit_site,
                "force_entry_direction": entry_related,
                "force_bidirectional": force_bidirectional,
            }

    return {
        "pair_context": pair_context,
        "components": component_summaries,
    }


def apply_strict_ring_pairwise_override(pair_result, ring_pair_context):
    """按严格环约束覆盖 pairwise 输出。"""
    if not ring_pair_context:
        return pair_result, False

    if ring_pair_context.get("force_entry_direction"):
        entry_site = ring_pair_context.get("entry_site")
        site_a = pair_result.get("site_a")
        site_b = pair_result.get("site_b")
        if entry_site not in {site_a, site_b}:
            return pair_result, False

        other_site = site_b if entry_site == site_a else site_a
        updated = dict(pair_result)
        updated["original_relation"] = pair_result.get("relation")
        updated["original_preferred_source"] = pair_result.get("preferred_source")
        updated["original_preferred_target"] = pair_result.get("preferred_target")
        updated["relation"] = "->"
        updated["preferred_source"] = entry_site
        updated["preferred_target"] = other_site
        updated["strict_ring_entry_direction"] = True
        updated["strict_ring_component_id"] = ring_pair_context.get("component_id")
        updated["strict_ring_start_site"] = ring_pair_context.get("start_site")
        updated["strict_ring_entry_site"] = entry_site
        updated["strict_ring_exit_site"] = ring_pair_context.get("exit_site")
        updated["strict_ring_entry_exit_sites"] = ring_pair_context.get("entry_exit_sites", [])
        updated["uncertainty_adjustments"] = list(
            updated.get("uncertainty_adjustments", [])
        ) + [{
            "feature": "strict_ring_entry_direction",
            "amount": 0.0,
            "detail": "严格环模式：入口相关连接强制为入口指向环内站点",
        }]
        changed = (
            pair_result.get("relation") != "->"
            or pair_result.get("preferred_source") != entry_site
            or pair_result.get("preferred_target") != other_site
        )
        return updated, changed

    if not ring_pair_context.get("force_bidirectional"):
        return pair_result, False

    updated = dict(pair_result)
    updated["original_relation"] = pair_result.get("relation")
    updated["original_preferred_source"] = pair_result.get("preferred_source")
    updated["original_preferred_target"] = pair_result.get("preferred_target")
    updated["relation"] = "<->"
    updated["preferred_source"] = None
    updated["preferred_target"] = None
    updated["strict_ring_bidirectional"] = True
    updated["strict_ring_component_id"] = ring_pair_context.get("component_id")
    updated["strict_ring_start_site"] = ring_pair_context.get("start_site")
    updated["strict_ring_entry_site"] = ring_pair_context.get("entry_site")
    updated["strict_ring_exit_site"] = ring_pair_context.get("exit_site")
    updated["strict_ring_entry_exit_sites"] = ring_pair_context.get("entry_exit_sites", [])
    updated["uncertainty_adjustments"] = list(
        updated.get("uncertainty_adjustments", [])
    ) + [{
        "feature": "strict_ring_bidirectional",
        "amount": 0.0,
        "detail": "严格环模式：环块内部非入口相关连接强制保留双向",
    }]
    return updated, pair_result.get("relation") != "<->"


def apply_strict_ring_edge_override(edge_result, ring_pair_context):
    """按严格环约束覆盖 global/path 输出。"""
    if not ring_pair_context:
        return edge_result, False

    if ring_pair_context.get("force_entry_direction"):
        entry_site = ring_pair_context.get("entry_site")
        site_a = edge_result.get("site_a")
        site_b = edge_result.get("site_b")
        if entry_site not in {site_a, site_b}:
            return edge_result, False

        other_site = site_b if entry_site == site_a else site_a
        updated = dict(edge_result)
        updated["original_prediction"] = edge_result.get("prediction")
        updated["original_upstream_site"] = edge_result.get("upstream_site")
        updated["original_downstream_site"] = edge_result.get("downstream_site")
        updated["original_confidence"] = edge_result.get("confidence")
        updated["prediction"] = f"{entry_site}->{other_site}"
        updated["upstream_site"] = entry_site
        updated["downstream_site"] = other_site
        updated["strict_ring_entry_direction"] = True
        updated["strict_ring_component_id"] = ring_pair_context.get("component_id")
        updated["strict_ring_start_site"] = ring_pair_context.get("start_site")
        updated["strict_ring_entry_site"] = entry_site
        updated["strict_ring_exit_site"] = ring_pair_context.get("exit_site")
        updated["strict_ring_entry_exit_sites"] = ring_pair_context.get("entry_exit_sites", [])
        updated["reasons"] = list(updated.get("reasons", [])) + [
            "strict_ring_entry_direction: 入口相关连接强制为入口指向环内站点"
        ]
        changed = (
            edge_result.get("upstream_site") != entry_site
            or edge_result.get("downstream_site") != other_site
        )
        return updated, changed

    if not ring_pair_context.get("force_bidirectional"):
        return edge_result, False

    updated = dict(edge_result)
    updated["original_prediction"] = edge_result.get("prediction")
    updated["original_upstream_site"] = edge_result.get("upstream_site")
    updated["original_downstream_site"] = edge_result.get("downstream_site")
    updated["original_confidence"] = edge_result.get("confidence")
    updated["prediction"] = "bidirectional"
    updated["upstream_site"] = None
    updated["downstream_site"] = None
    if isinstance(updated.get("confidence"), (int, float)):
        updated["confidence"] = round(min(float(updated["confidence"]), 0.55), 6)
    updated["strict_ring_bidirectional"] = True
    updated["strict_ring_component_id"] = ring_pair_context.get("component_id")
    updated["strict_ring_start_site"] = ring_pair_context.get("start_site")
    updated["strict_ring_entry_site"] = ring_pair_context.get("entry_site")
    updated["strict_ring_exit_site"] = ring_pair_context.get("exit_site")
    updated["strict_ring_entry_exit_sites"] = ring_pair_context.get("entry_exit_sites", [])
    updated["reasons"] = list(updated.get("reasons", [])) + [
        "strict_ring_bidirectional: 环块内部非入口相关连接强制保留双向"
    ]
    return updated, edge_result.get("prediction") != "bidirectional"


def score_to_level(score):
    if score <= -0.75:
        return "access"
    if score >= 0.75:
        return "core"
    return "backhaul"


def counter_to_json_dict(counter):
    """把 Counter/dict 中可能存在的 tuple key 转成 JSON 可序列化的字符串。"""
    output = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            json_key = "||".join(str(part) for part in key)
        else:
            json_key = str(key)
        output[json_key] = value
    return output


def extract_primary_upstream_map(prediction_result):
    upstream_candidates = defaultdict(list)

    for edge in prediction_result.get("edges", []):
        if edge.get("prediction") == "bidirectional":
            continue

        downstream = edge.get("downstream_site")
        upstream = edge.get("upstream_site")
        if downstream and upstream:
            upstream_candidates[downstream].append(
                (edge.get("confidence", 0.0), upstream, edge)
            )

    primary = {}
    for site_id, candidates in upstream_candidates.items():
        candidates.sort(key=lambda item: (-item[0], str(item[1])))
        primary[site_id] = {
            "upstream_site": candidates[0][1],
            "confidence": candidates[0][0],
            "edge": candidates[0][2],
        }
    return primary


def build_downstream_map(prediction_result):
    downstream_map = defaultdict(set)

    for edge in prediction_result.get("edges", []):
        prediction = edge.get("prediction")
        if prediction == "bidirectional":
            site_a = edge.get("site_a")
            site_b = edge.get("site_b")
            if site_a and site_b:
                downstream_map[site_a].add(site_b)
                downstream_map[site_b].add(site_a)
            continue

        upstream_site = edge.get("upstream_site")
        downstream_site = edge.get("downstream_site")
        if upstream_site and downstream_site:
            downstream_map[upstream_site].add(downstream_site)

    return {
        site_id: sorted(neighbors)
        for site_id, neighbors in sorted(downstream_map.items())
    }


def compact_edge_prediction(edge):
    """只保留相邻站点对的上下行预测结果。"""
    upstream_site = edge.get("upstream_site")
    downstream_site = edge.get("downstream_site")
    prediction = edge.get("prediction")
    if upstream_site and downstream_site:
        prediction = f"{upstream_site}->{downstream_site}"
    elif prediction != "bidirectional":
        prediction = "bidirectional"

    return {
        "site_a": edge.get("site_a"),
        "site_b": edge.get("site_b"),
        "prediction": prediction,
        "upstream_site": upstream_site,
        "downstream_site": downstream_site,
    }


def compact_prediction_edges(prediction_result):
    return [
        compact_edge_prediction(edge)
        for edge in prediction_result.get("edges", [])
    ]


def format_direction_count_summary(
    total_count,
    directed_count,
    bidirectional_count,
    unit="边",
    label="上下行预测汇总",
):
    """格式化有向/双向结果数量与比例。"""
    if total_count <= 0:
        directed_ratio = 0.0
        bidirectional_ratio = 0.0
    else:
        directed_ratio = directed_count / total_count
        bidirectional_ratio = bidirectional_count / total_count

    return (
        f"{label}: 有向{unit} {directed_count}/{total_count} "
        f"({directed_ratio:.2%})，双向{unit} {bidirectional_count}/{total_count} "
        f"({bidirectional_ratio:.2%})"
    )


def build_site_topology_enhanced(ne_graph, show_progress=False):
    """
    从 ne_graph 构建站点级无向图，并保留跨站边上的 NE 角色证据。

    返回:
        site_stats: dict[site_id] -> site meta
        site_edges: dict[(site_a, site_b)] -> edge meta
        adjacency: dict[site_id] -> set(neighbor_site)
    """
    site_stats = {}
    site_edges = {}
    adjacency = defaultdict(set)

    with ProgressReporter(len(ne_graph), "增强拓扑: 聚合站点设备", show_progress) as progress:
        for ne_id, ne_info in ne_graph.items():
            progress.update()
            if not isinstance(ne_info, dict):
                continue

            site_id = _get_site_id(ne_info)
            if not site_id:
                continue

            role = classify_device_role(ne_info.get("domain", ""))
            rec = site_stats.setdefault(site_id, {
                "site_id": site_id,
                "nes": set(),
                "role_counts": Counter(),
                "neighbors": set(),
                "degree": 0,
            })
            rec["nes"].add(ne_id)
            rec["role_counts"][role] += 1

    seen_links = set()
    site_role_counts = build_site_role_counts(ne_graph)
    with ProgressReporter(len(ne_graph), "增强拓扑: 扫描跨站链路", show_progress) as progress:
        for source_ne, source_info in ne_graph.items():
            progress.update()
            source_site = _get_site_id(source_info) if isinstance(source_info, dict) else ""
            if not source_site:
                continue

            raw_links = source_info.get("link", {})
            if not isinstance(raw_links, dict):
                continue

            for target_ne, link_meta in raw_links.items():
                target_info = ne_graph.get(target_ne)
                if not isinstance(target_info, dict):
                    continue

                target_site = _get_site_id(target_info)
                if not target_site or target_site == source_site:
                    continue

                source_domain = normalize_domain(source_info.get("domain", ""))
                target_domain = normalize_domain(target_info.get("domain", ""))
                if not should_include_cross_site_link(
                    source_site,
                    source_domain,
                    target_site,
                    target_domain,
                    site_role_counts,
                ):
                    continue

                role_source = classify_device_role(source_domain)
                role_target = classify_device_role(target_domain)
                link_types = (
                    sorted(link_meta.keys())
                    if isinstance(link_meta, dict) and link_meta
                    else ["__unknown__"]
                )

                for link_type in link_types:
                    link_key = tuple(sorted((source_ne, target_ne))) + (str(link_type),)
                    if link_key in seen_links:
                        continue
                    seen_links.add(link_key)

                    key = tuple(sorted((source_site, target_site)))
                    edge = site_edges.setdefault(key, {
                        "site_a": key[0],
                        "site_b": key[1],
                        "link_types": set(),
                        "link_count": 0,
                        "ne_pairs": [],
                        "role_pair_counter": Counter(),
                        "direct_role_evidence": {
                            key[0]: Counter(),
                            key[1]: Counter(),
                        },
                    })

                    edge["link_types"].add(str(link_type))
                    edge["link_count"] += 1
                    edge["ne_pairs"].append((source_ne, target_ne, role_source, role_target))

                    role_pair = tuple(sorted((role_source, role_target)))
                    edge["role_pair_counter"][role_pair] += 1

                    if ROLE_ORDER[role_source] < ROLE_ORDER[role_target]:
                        low_site = source_site
                        high_site = target_site
                    elif ROLE_ORDER[role_target] < ROLE_ORDER[role_source]:
                        low_site = target_site
                        high_site = source_site
                    else:
                        low_site = None
                        high_site = None

                    if low_site and high_site:
                        edge["direct_role_evidence"][low_site]["down_like"] += 1
                        edge["direct_role_evidence"][high_site]["up_like"] += 1
                    else:
                        edge["direct_role_evidence"][source_site]["flat_like"] += 1
                        edge["direct_role_evidence"][target_site]["flat_like"] += 1

                    adjacency[source_site].add(target_site)
                    adjacency[target_site].add(source_site)

    with ProgressReporter(len(site_stats), "增强拓扑: 回填站点度数", show_progress) as progress:
        for site_id, rec in site_stats.items():
            progress.update()
            rec["neighbors"] = set(adjacency.get(site_id, set()))
            rec["degree"] = len(rec["neighbors"])

    return site_stats, site_edges, adjacency


def compute_site_priors_enhanced(site_stats, show_progress=False):
    with ProgressReporter(len(site_stats), "计算增强站点先验", show_progress) as progress:
        for site_id, rec in site_stats.items():
            progress.update()
            counts = rec["role_counts"]
            known_total = counts["wireless"] + counts["microwave"] + counts["router"]

            if known_total > 0:
                raw_prior = (
                    counts["wireless"] * ROLE_SCORE["wireless"] +
                    counts["microwave"] * ROLE_SCORE["microwave"] +
                    counts["router"] * ROLE_SCORE["router"]
                ) / known_total
            else:
                raw_prior = 0.0

            wireless_ratio = counts["wireless"] / known_total if known_total else 0.0
            microwave_ratio = counts["microwave"] / known_total if known_total else 0.0
            router_ratio = counts["router"] / known_total if known_total else 0.0

            predominant_role = "unknown"
            if known_total > 0:
                predominant_role = max(
                    ("wireless", "microwave", "router"),
                    key=lambda role: counts[role],
                )

            degree = rec["degree"]
            if degree <= 1 and wireless_ratio >= 0.5:
                raw_prior -= 0.6
            if degree >= 3 and router_ratio >= 0.5:
                raw_prior += 0.6

            purity = max(wireless_ratio, microwave_ratio, router_ratio) if known_total else 0.0
            if router_ratio >= 0.75 and wireless_ratio == 0:
                anchor_strength = 0.90
            elif wireless_ratio >= 0.75 and router_ratio == 0:
                anchor_strength = 0.90
            elif purity >= 0.60:
                anchor_strength = 0.70
            else:
                anchor_strength = 0.50

            rec["known_total"] = known_total
            rec["wireless_ratio"] = wireless_ratio
            rec["microwave_ratio"] = microwave_ratio
            rec["router_ratio"] = router_ratio
            rec["raw_prior"] = raw_prior
            rec["predominant_role"] = predominant_role
            rec["anchor_strength"] = anchor_strength


def select_anchor_sites_enhanced(site_stats):
    core_anchors = []
    access_anchors = []

    for site_id, rec in site_stats.items():
        if rec["known_total"] == 0:
            continue
        if rec["router_ratio"] >= 0.6 and rec["degree"] >= 1:
            core_anchors.append(site_id)
        if rec["wireless_ratio"] >= 0.6:
            access_anchors.append(site_id)

    all_sites = list(site_stats.keys())
    if not core_anchors and all_sites:
        core_anchors = [max(
            all_sites,
            key=lambda site_id: (
                site_stats[site_id]["router_ratio"],
                site_stats[site_id]["raw_prior"],
                site_stats[site_id]["degree"],
            ),
        )]

    if not access_anchors and all_sites:
        access_anchors = [max(
            all_sites,
            key=lambda site_id: (
                site_stats[site_id]["wireless_ratio"],
                -site_stats[site_id]["raw_prior"],
                -site_stats[site_id]["degree"],
            ),
        )]

    if set(core_anchors) == set(access_anchors) and len(all_sites) > 1:
        for site_id in sorted(
            all_sites,
            key=lambda item: (
                site_stats[item]["wireless_ratio"],
                -site_stats[item]["raw_prior"],
                -site_stats[item]["degree"],
            ),
            reverse=True,
        ):
            if site_id not in core_anchors:
                access_anchors = [site_id]
                break

    return list(dict.fromkeys(core_anchors)), list(dict.fromkeys(access_anchors))


def smooth_site_scores(site_stats, adjacency, max_iter=120, tol=1e-4, show_progress=False):
    scores = {site_id: rec["base_score"] for site_id, rec in site_stats.items()}

    with ProgressReporter(max_iter, "平滑站点分数", show_progress) as progress:
        for iteration in range(max_iter):
            progress.update()
            new_scores = {}
            max_delta = 0.0

            for site_id, rec in site_stats.items():
                neighbors = adjacency.get(site_id, set())
                base = rec["base_score"]
                anchor_strength = rec["anchor_strength"]

                if neighbors:
                    neighbor_avg = sum(scores[neighbor] for neighbor in neighbors) / len(neighbors)
                    structural = 0.75 * neighbor_avg + 0.25 * base
                else:
                    structural = base

                new_score = anchor_strength * base + (1 - anchor_strength) * structural
                new_scores[site_id] = new_score
                max_delta = max(max_delta, abs(new_score - scores[site_id]))

            progress.set_extra_text(f"iter={iteration + 1}, delta={max_delta:.6g}")
            scores = new_scores
            if max_delta < tol:
                break

    return scores


def edge_cost_for_path(source_site, target_site, site_stats, site_edges, score_hint):
    """
    为候选路径生成准备的边代价:
    - 更符合由低层到高层的边，代价更低
    - 明显逆层级的边，代价更高
    - 同层边中性偏高
    """
    key = tuple(sorted((source_site, target_site)))
    edge = site_edges[key]

    source_score = score_hint.get(source_site, 0.0)
    target_score = score_hint.get(target_site, 0.0)
    diff = abs(source_score - target_score)

    role_pair_counter = edge["role_pair_counter"]
    role_signal = 0.0
    role_signal += 0.4 * role_pair_counter.get(("microwave", "router"), 0)
    role_signal += 0.4 * role_pair_counter.get(("router", "wireless"), 0)
    role_signal += 0.5 * role_pair_counter.get(("microwave", "wireless"), 0)

    density_bonus = min(edge["link_count"], 3) * 0.1

    base = 1.0
    cost = (
        base
        - min(diff, 1.2) * 0.35
        - min(role_signal, 2.0) * 0.15
        - density_bonus
    )
    return max(0.15, cost)


def dijkstra_path(adjacency, start, goal, edge_cost_fn):
    queue = [(0.0, start)]
    dist = {start: 0.0}
    parent = {start: None}

    while queue:
        cur_dist, node = heapq.heappop(queue)
        if node == goal:
            break
        if cur_dist > dist.get(node, float("inf")):
            continue

        for neighbor in adjacency.get(node, ()):
            next_dist = cur_dist + edge_cost_fn(node, neighbor)
            if next_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = next_dist
                parent[neighbor] = node
                heapq.heappush(queue, (next_dist, neighbor))

    if goal not in parent:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def build_candidate_paths(
    site_stats,
    site_edges,
    adjacency,
    access_anchors,
    core_anchors,
    score_hint,
    show_progress=False,
):
    """
    为路径投票生成候选路径。
    每个 access 取少量最近/最像核心的 core。
    """
    paths = []
    nodes = set(site_stats.keys())

    with ProgressReporter(len(access_anchors), "生成候选路径", show_progress) as progress:
        for access_site in access_anchors:
            progress.update()
            if access_site not in nodes:
                continue

            ranked_cores = sorted(
                core_anchors,
                key=lambda core_site: (
                    abs(score_hint.get(core_site, 0.0) - score_hint.get(access_site, 0.0)),
                    -site_stats[core_site]["router_ratio"],
                    -site_stats[core_site]["degree"],
                ),
            )[:3]

            for core_site in ranked_cores:
                if core_site == access_site:
                    continue

                path = dijkstra_path(
                    adjacency,
                    access_site,
                    core_site,
                    lambda source, target: edge_cost_for_path(
                        source, target, site_stats, site_edges, score_hint
                    ),
                )
                if path and len(path) >= 2:
                    paths.append(path)

    unique_paths = []
    seen = set()
    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def edge_prior_vote(source_site, target_site, site_edges):
    """
    仅基于边上的设备角色对证据，给出 source->target 与 target->source 的先验票。
    """
    key = tuple(sorted((source_site, target_site)))
    edge = site_edges[key]
    evidence = edge["direct_role_evidence"]

    vote_source_to_target = 0.0
    vote_target_to_source = 0.0

    vote_source_to_target += evidence[source_site]["down_like"] * 1.0
    vote_source_to_target += evidence[target_site]["up_like"] * 1.0

    vote_target_to_source += evidence[target_site]["down_like"] * 1.0
    vote_target_to_source += evidence[source_site]["up_like"] * 1.0

    flat_penalty = 0.15 * (
        evidence[source_site]["flat_like"] + evidence[target_site]["flat_like"]
    )

    vote_source_to_target = max(0.0, vote_source_to_target - flat_penalty)
    vote_target_to_source = max(0.0, vote_target_to_source - flat_penalty)
    return vote_source_to_target, vote_target_to_source


def collect_path_votes(paths, final_scores, show_progress=False):
    """
    路径对边方向投票。
    规则：
    - 默认希望路径上的层级大体单调上升
    - 若路径中某边连接低分->高分，则支持该方向
    - 若两端很接近，则只给弱票
    """
    votes = defaultdict(lambda: {"ab": 0.0, "ba": 0.0, "support_paths": 0})

    with ProgressReporter(len(paths), "统计路径投票", show_progress) as progress:
        for path in paths:
            progress.update()
            if len(path) < 2:
                continue

            for index in range(len(path) - 1):
                source_site = path[index]
                target_site = path[index + 1]
                key = tuple(sorted((source_site, target_site)))
                source_score = final_scores.get(source_site, 0.0)
                target_score = final_scores.get(target_site, 0.0)
                diff = target_score - source_score

                weight = 1.0
                if abs(diff) >= 0.7:
                    weight = 1.25
                elif abs(diff) < 0.2:
                    weight = 0.35

                if key[0] == source_site:
                    if diff >= 0:
                        votes[key]["ab"] += weight
                    else:
                        votes[key]["ba"] += weight
                else:
                    if diff >= 0:
                        votes[key]["ba"] += weight
                    else:
                        votes[key]["ab"] += weight

                votes[key]["support_paths"] += 1

    return votes
