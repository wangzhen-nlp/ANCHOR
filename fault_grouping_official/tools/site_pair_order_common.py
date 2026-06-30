#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站点上下行推断脚本的共享工具函数。"""

from collections import Counter, defaultdict, deque

from fault_grouping_official.tools.progress_utils import ProgressBar


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

    if (
        source_wireless_only
        and source_role == "wireless"
        and target_role in allowed_peer_roles
    ):
        return True

    if (
        target_wireless_only
        and target_role == "wireless"
        and source_role in allowed_peer_roles
    ):
        return True

    return False


def iter_unique_cross_site_links(
    ne_graph,
    *,
    assume_symmetric=False,
    wireless_only_sites=None,
):
    """按 NE 对 + link_type 去重，遍历跨站点链路。

    assume_symmetric=True 仅用于已知双向存储的 ne_graph：按 NE ID 规范侧遍历，
    避免维护与物理边数同规模的 seen 集合。
    """
    seen = None if assume_symmetric else set()
    site_role_counts = (
        None
        if wireless_only_sites is not None
        else build_site_role_counts(ne_graph)
    )

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
                if assume_symmetric:
                    if source_ne > target_ne:
                        continue
                else:
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
