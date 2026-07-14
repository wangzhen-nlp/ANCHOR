#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站点上下行推断脚本的共享工具函数。"""

from collections import Counter, defaultdict, deque

from anchor_grouping_online.tools.progress_utils import ProgressBar


CANONICAL_DOMAIN_MAP = {
    "data": "Data",
    "transmission": "Transmission",
    "ran": "Ran",
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
    return str(ne_info.get("site_id", "")).strip()


# 跨类型连边的 domain 优先级（数值大者为上行侧）：Data > Transmission > Ran。
# 未知/其它 domain 记 0，不参与跨类型方向约束。
CROSS_DOMAIN_PRIORITY = {
    "Data": 3,
    "Transmission": 2,
    "Ran": 1,
}


def cross_domain_priority(domain):
    """返回 domain 在跨类型方向约束中的优先级；未知 domain 返回 0（不参与约束）。"""
    return CROSS_DOMAIN_PRIORITY.get(normalize_domain(domain), 0)


def is_transmission_domain(domain):
    """domain 是否属于传输类（Transmission）；入参需已 normalize。"""
    return normalize_domain(domain) == "Transmission"


def build_transmission_misconnection_pairs(ne_graph):
    """预处理：识别疑似误连接的站点对，返回 (pair_key 集合, stats)。

    规则：站点 X 含 Data 设备，站点 Y 含 Transmission 与 Ran 设备但
    不含 Data（含 Data 的站点间传输连边视为骨干边，不在此列），两站点之间存在
    Transmission 相关连边（任一端为传输类 domain）却没有 Data-Ran 连边佐证时，
    认为这些 Transmission 相关连边是误连接。命中站点对之间的传输类连边应从
    拓扑推断、方向约束与裁剪口径中一并剔除（由消费端按返回集合过滤）。
    """
    site_domains = _collect_site_domains(ne_graph)
    candidate_flags = _scan_misconnection_candidate_flags(ne_graph, site_domains)

    misconnection_pairs = {
        pair_key
        for pair_key, flags in candidate_flags.items()
        if flags and flags["has_transmission_link"] and not flags["has_data_ran_link"]
    }
    stats = {
        "candidate_pair_count": sum(
            1 for flags in candidate_flags.values() if flags is not False
        ),
        "misconnection_pair_count": len(misconnection_pairs),
    }
    return misconnection_pairs, stats


def _collect_site_domains(ne_graph):
    """站点 -> 站内设备的归一化 domain 集合。"""
    site_domains = defaultdict(set)
    for ne_info in ne_graph.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = _get_site_id(ne_info)
        if not site_id:
            continue
        site_domains[site_id].add(normalize_domain(ne_info.get("domain", "")))
    return site_domains


def _misconnection_precondition(site_domains, site_a, site_b):
    """一端含 Data、另一端 Trans+Ran 无 Data 时才是误连接候选站点对。"""
    def _matches(data_side_domains, ran_side_domains):
        return (
            "Data" in data_side_domains
            and "Data" not in ran_side_domains
            and "Ran" in ran_side_domains
            and any(is_transmission_domain(d) for d in ran_side_domains)
        )

    domains_a = site_domains.get(site_a, ())
    domains_b = site_domains.get(site_b, ())
    return _matches(domains_a, domains_b) or _matches(domains_b, domains_a)


def _scan_misconnection_candidate_flags(ne_graph, site_domains):
    """原始连边遍历（不做 domain 过滤——误连接判定要看全部连边），按 NE 对去重。

    返回 {pair_key: False（非候选）或 {"has_transmission_link", "has_data_ran_link"}}。
    """
    candidate_flags = {}
    seen_ne_pairs = set()
    for source_ne, source_info in ne_graph.items():
        if not isinstance(source_info, dict):
            continue
        source_site = _get_site_id(source_info)
        if not source_site:
            continue
        source_domain = normalize_domain(source_info.get("domain", ""))
        raw_links = source_info.get("link", {})
        if not isinstance(raw_links, dict):
            continue
        for target_ne in raw_links:
            _flag_misconnection_edge(
                ne_graph, site_domains, candidate_flags, seen_ne_pairs,
                source_ne, source_site, source_domain, target_ne,
            )
    return candidate_flags


def _flag_misconnection_edge(
    ne_graph, site_domains, candidate_flags, seen_ne_pairs,
    source_ne, source_site, source_domain, target_ne,
):
    """按单条 NE 边更新候选站点对的传输连边/Data-Ran 佐证标记。"""
    target_info = ne_graph.get(target_ne)
    if not isinstance(target_info, dict):
        return
    target_site = _get_site_id(target_info)
    if not target_site or target_site == source_site:
        return
    ne_key = (
        (source_ne, target_ne)
        if source_ne <= target_ne
        else (target_ne, source_ne)
    )
    if ne_key in seen_ne_pairs:
        return
    seen_ne_pairs.add(ne_key)

    pair_key = tuple(sorted((source_site, target_site)))
    flags = candidate_flags.get(pair_key)
    if flags is None:
        if not _misconnection_precondition(site_domains, source_site, target_site):
            candidate_flags[pair_key] = False
            return
        flags = {"has_transmission_link": False, "has_data_ran_link": False}
        candidate_flags[pair_key] = flags
    elif flags is False:
        return

    target_domain = normalize_domain(target_info.get("domain", ""))
    if is_transmission_domain(source_domain) or is_transmission_domain(target_domain):
        flags["has_transmission_link"] = True
    if {source_domain, target_domain} == {"Data", "Ran"}:
        flags["has_data_ran_link"] = True


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


def should_include_cross_site_link(
    source_site,
    source_domain,
    target_site,
    target_domain,
    site_role_counts=None,
    *,
    wireless_only_sites=None,
):
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

    source_wireless_only = (
        source_site in wireless_only_sites
        if wireless_only_sites is not None
        else is_wireless_only_site(source_site, site_role_counts)
    )
    target_wireless_only = (
        target_site in wireless_only_sites
        if wireless_only_sites is not None
        else is_wireless_only_site(target_site, site_role_counts)
    )

    source_allowed = (
        source_wireless_only
        and source_role == "wireless"
        and target_role in allowed_peer_roles
    )
    target_allowed = (
        target_wireless_only
        and target_role == "wireless"
        and source_role in allowed_peer_roles
    )
    return source_allowed or target_allowed


def iter_unique_cross_site_links(
    ne_graph,
    *,
    assume_symmetric=False,
    wireless_only_sites=None,
    include_filtered_cross_domain=False,
    transmission_misconnection_pairs=None,
):
    """按 NE 对 + link_type 去重，遍历跨站点链路。

    assume_symmetric=True 仅用于已知双向存储的 ne_graph：按 NE ID 规范侧遍历，
    避免维护与物理边数同规模的 seen 集合。

    include_filtered_cross_domain=True 时，被 should_include_cross_site_link
    过滤的跨 domain 连边也会产出（included_in_topology=False），供跨类型方向
    约束提取使用；消费端必须自行跳过这些记录，不得计入拓扑。

    transmission_misconnection_pairs（pair_key 集合）非空时，命中站点对之间
    任一端为传输类 domain 的连边被视为误连接，直接跳过（拓扑与约束证据均不产出）。
    """
    seen = None if assume_symmetric else set()
    site_role_counts = None
    if wireless_only_sites is None:
        site_role_counts = build_site_role_counts(ne_graph)

    for source_ne, source_info in ne_graph.items():
        source_site = _get_site_id(source_info)
        if not source_site:
            continue

        yield from _iter_source_cross_site_links(
            ne_graph, source_ne, source_info, source_site,
            site_role_counts, include_filtered_cross_domain,
            transmission_misconnection_pairs, assume_symmetric, seen,
        )


def _iter_source_cross_site_links(
    ne_graph, source_ne, source_info, source_site, site_role_counts,
    include_filtered, misconnection_pairs, assume_symmetric, seen,
):
    """遍历单个 NE 的合法跨站链路。"""
    source_domain = normalize_domain(source_info.get("domain", ""))
    raw_links = source_info.get("link", {})
    if not isinstance(raw_links, dict):
        return
    for target_ne, link_meta in raw_links.items():
        target_info = ne_graph.get(target_ne)
        if not isinstance(target_info, dict):
            continue
        target_site = _get_site_id(target_info)
        if not target_site or target_site == source_site:
            continue
        target_domain = normalize_domain(target_info.get("domain", ""))
        if _is_misconnection(
            source_site, target_site, source_domain, target_domain,
            misconnection_pairs,
        ):
            continue
        included = should_include_cross_site_link(
            source_site, source_domain, target_site, target_domain,
            site_role_counts,
        )
        if not included and not include_filtered:
            continue
        link_types = _get_link_types(link_meta)
        for link_type in link_types:
            if _is_duplicate_link(
                source_ne, target_ne, link_type, assume_symmetric, seen
            ):
                continue
            yield _build_cross_site_link(
                source_ne, target_ne, source_site, target_site,
                source_domain, target_domain, link_type, included,
            )


def _is_misconnection(
    source_site, target_site, source_domain, target_domain, pairs,
):
    if not pairs:
        return False
    if not (
        is_transmission_domain(source_domain)
        or is_transmission_domain(target_domain)
    ):
        return False
    return tuple(sorted((source_site, target_site))) in pairs


def _get_link_types(link_meta):
    if isinstance(link_meta, dict) and link_meta:
        return sorted(link_meta)
    return ["__unknown__"]


def _is_duplicate_link(source_ne, target_ne, link_type, assume_symmetric, seen):
    if assume_symmetric:
        return source_ne > target_ne
    key = tuple(sorted((source_ne, target_ne))) + (str(link_type),)
    if key in seen:
        return True
    seen.add(key)
    return False


def _build_cross_site_link(
    source_ne, target_ne, source_site, target_site,
    source_domain, target_domain, link_type, included,
):
    return {
        "source_ne": source_ne,
        "target_ne": target_ne,
        "source_site": source_site,
        "target_site": target_site,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "link_type": str(link_type),
        "included_in_topology": included,
    }


def _collect_ring_component(start_site, non_bridge_neighbors, visited):
    """BFS 收集一个非桥边连通块的站点列表（原地更新 visited）。"""
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
    return component_sites


def _summarize_ring_component(
    component_set, internal_pairs, entry_exit_sites, site_scores, component_id
):
    """构造环块摘要：出入口打分并选定 entry/exit/start 站点。"""
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
    return {
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
    }


def build_strict_ring_context(edge_keys, bridge_pairs, site_scores=None):
    """
    构造严格环约束上下文。

    环块定义为移除桥边后的连通块；环块里通过桥边连到外部的站点视为出入口。
    严格模式下，入口站点相关边强制为入口指向环内站点，其它环块内部边全部双向。
    如果传入 site_scores，则分数最高的出入口站点视为更靠上行的 entry，
    分数最低的出入口站点视为更靠端侧的 exit。
    """
    site_scores = site_scores or {}
    non_bridge_neighbors, bridge_neighbors = _build_ring_neighbor_maps(
        edge_keys, bridge_pairs
    )

    visited = set()
    pair_context = {}
    component_summaries = []

    for start_site in sorted(non_bridge_neighbors):
        if start_site in visited:
            continue

        component_sites = _collect_ring_component(
            start_site, non_bridge_neighbors, visited
        )
        component_set = set(component_sites)
        internal_pairs, entry_exit_sites = _describe_ring_component(
            component_set, non_bridge_neighbors, bridge_neighbors
        )
        if not internal_pairs:
            continue

        component_id = len(component_summaries)
        entry_exit_sites = sorted(entry_exit_sites)
        summary = _summarize_ring_component(
            component_set, internal_pairs, entry_exit_sites,
            site_scores, component_id,
        )
        component_summaries.append(summary)
        _add_ring_pair_context(pair_context, internal_pairs, summary)

    return {
        "pair_context": pair_context,
        "components": component_summaries,
    }


def _build_ring_neighbor_maps(edge_keys, bridge_pairs):
    normalized_bridges = {tuple(sorted(pair)) for pair in bridge_pairs}
    non_bridge_neighbors = defaultdict(set)
    bridge_neighbors = defaultdict(set)
    for left_site, right_site in edge_keys:
        pair_key = tuple(sorted((left_site, right_site)))
        target = (
            bridge_neighbors
            if pair_key in normalized_bridges
            else non_bridge_neighbors
        )
        target[left_site].add(right_site)
        target[right_site].add(left_site)
    return non_bridge_neighbors, bridge_neighbors


def _describe_ring_component(
    component_set, non_bridge_neighbors, bridge_neighbors,
):
    internal_pairs = set()
    entry_exit_sites = set()
    for site_id in component_set:
        external = set(bridge_neighbors.get(site_id, ())) - component_set
        if external:
            entry_exit_sites.add(site_id)
        for neighbor_site in non_bridge_neighbors.get(site_id, ()):
            if neighbor_site in component_set:
                internal_pairs.add(tuple(sorted((site_id, neighbor_site))))
    return sorted(internal_pairs), sorted(entry_exit_sites)


def _add_ring_pair_context(pair_context, internal_pairs, summary):
    entry_site = summary["entry_site"]
    for pair_key in internal_pairs:
        entry_related = entry_site is not None and entry_site in pair_key
        pair_context[pair_key] = {
            "component_id": summary["component_id"],
            "start_site": summary["start_site"],
            "entry_exit_sites": summary["entry_exit_sites"],
            "entry_site": entry_site,
            "exit_site": summary["exit_site"],
            "force_entry_direction": entry_related,
            "force_bidirectional": not entry_related,
        }


def apply_strict_ring_pairwise_override(pair_result, ring_pair_context):
    """按严格环约束覆盖 pairwise 输出。"""
    if not ring_pair_context:
        return pair_result, False

    if ring_pair_context.get("force_entry_direction"):
        return _apply_ring_entry_override(pair_result, ring_pair_context)

    if not ring_pair_context.get("force_bidirectional"):
        return pair_result, False

    return _apply_ring_bidirectional_override(pair_result, ring_pair_context)


def _copy_ring_override_fields(pair_result, ring_context):
    updated = dict(pair_result)
    updated["original_relation"] = pair_result.get("relation")
    updated["original_preferred_source"] = pair_result.get("preferred_source")
    updated["original_preferred_target"] = pair_result.get("preferred_target")
    updated["strict_ring_component_id"] = ring_context.get("component_id")
    updated["strict_ring_start_site"] = ring_context.get("start_site")
    updated["strict_ring_entry_site"] = ring_context.get("entry_site")
    updated["strict_ring_exit_site"] = ring_context.get("exit_site")
    updated["strict_ring_entry_exit_sites"] = ring_context.get(
        "entry_exit_sites", []
    )
    return updated


def _apply_ring_entry_override(pair_result, ring_context):
    entry_site = ring_context.get("entry_site")
    site_a = pair_result.get("site_a")
    site_b = pair_result.get("site_b")
    if entry_site not in {site_a, site_b}:
        return pair_result, False
    other_site = site_b if entry_site == site_a else site_a
    updated = _copy_ring_override_fields(pair_result, ring_context)
    updated["relation"] = "->"
    updated["preferred_source"] = entry_site
    updated["preferred_target"] = other_site
    updated["strict_ring_entry_direction"] = True
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


def _apply_ring_bidirectional_override(pair_result, ring_context):
    updated = _copy_ring_override_fields(pair_result, ring_context)
    updated["relation"] = "<->"
    updated["preferred_source"] = None
    updated["preferred_target"] = None
    updated["strict_ring_bidirectional"] = True
    updated["uncertainty_adjustments"] = list(
        updated.get("uncertainty_adjustments", [])
    ) + [{
        "feature": "strict_ring_bidirectional",
        "amount": 0.0,
        "detail": "严格环模式：环块内部非入口相关连接强制保留双向",
    }]
    return updated, pair_result.get("relation") != "<->"
