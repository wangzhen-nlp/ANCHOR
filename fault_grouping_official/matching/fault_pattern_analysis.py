"""故障模式分析与记录增强（self-contained 移植）。

从 ticket_recall/evaluation/analyze_case_fault_patterns.py 移植两部分逻辑，使
fault_grouping_official 不依赖 ticket_recall / 旧 fault_grouping / alarm_tools /
topology_resources 等外部包，做到自满足：

  1) 过滤判定：站点连通分量拆分、断站吸收、链/环模式识别（analyze_case_record /
     filter_other_patterns / SiteRelationIndex 等）。
  2) 记录增强：故障模式备注（note）、fault_pattern_* 字段、补充相关站点/网元/链路
     （supplemental_fault_pattern_context，供 ne_propagation_visualizer.html 标红展示）。

不含原脚本的 IO、汇总、CLI 以及 build_augmented_case_record 外壳（该外壳的 27 行
编排在 fault_pattern_filter.py 中按原样实现，但省去 deepcopy——因为传入的是每个故障组
新构建、未被共享的记录，可安全原地改写）。其余函数与原脚本逐行一致；若原脚本行为有变，
请对照同步。OFFLINE_ALARMS 取官方版本（与 alarm_tools 版集合相同）。
"""

from collections import defaultdict, deque

from fault_grouping_official.alarm_types import OFFLINE_ALARMS
from fault_grouping_official.site_topology import (
    build_site_topology_from_ne_graph,
    load_site_chain_index,
)

OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)
ROUTER_DEVICE_DOMAINS = {"DATA"}
OTHER_FAULT_PATTERNS = {"ip_ring_others"}
PATTERN_PRIORITY = {
    "ip_chain_single_link": 0,
    "ip_chain_multi_link": 1,
    "ip_ring_single_upstream": 2,
    "ip_ring_multi_upstream": 3,
    "ip_ring_others": 4,
    "unknown": 99,
}


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


def extract_case_router_device_sites(record, site_has_router_device):
    router_sites = {
        site_id
        for site_id in extract_case_sites(record)
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
        self._upstream_distance_cache = {}

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

    def directly_connected(self, site_a, site_b):
        if site_a == site_b:
            return False
        return (
            site_b in self.downstream_direct.get(site_a, set())
            or site_a in self.downstream_direct.get(site_b, set())
            or site_b in self.bidirectional_direct.get(site_a, set())
            or site_a in self.bidirectional_direct.get(site_b, set())
        )

    def direct_neighbors(self, site_id):
        return (
            set(self.downstream_direct.get(site_id, set()))
            | set(self.upstream_direct.get(site_id, set()))
            | set(self.bidirectional_direct.get(site_id, set()))
        )

    def non_downstream_neighbors(self, site_id):
        return sorted(self.direct_neighbors(site_id) - set(self.downstream_direct.get(site_id, set())))

    def bidirectional_neighbors(self, site_id):
        return sorted(self.bidirectional_direct.get(site_id, set()))

    def direct_upstream_neighbors(self, site_id):
        return sorted(self.upstream_direct.get(site_id, set()))

    def undirected_neighbors_in(self, site_id, site_set):
        return {
            other_site
            for other_site in site_set
            if self.directly_connected(site_id, other_site)
        }


def absorb_unmanaged_downstream_sites(site_ids, initial_unmanaged_sites, relation_index):
    remaining = set(site_ids)
    unmanaged_sites = set(initial_unmanaged_sites) & remaining
    absorbed_by = {}
    absorb_steps = []

    while True:
        candidates = []
        for unmanaged_site in sorted(unmanaged_sites & remaining):
            for upstream_site in sorted(remaining - {unmanaged_site}):
                distance = relation_index.upstream_distance(unmanaged_site, upstream_site)
                if distance is not None and distance > 0:
                    candidates.append((distance, unmanaged_site, upstream_site))
        if not candidates:
            break

        distance, unmanaged_site, parent_site = min(candidates)
        remaining.remove(unmanaged_site)
        absorbed_by[unmanaged_site] = parent_site
        unmanaged_sites.add(parent_site)
        absorb_steps.append({
            "site": unmanaged_site,
            "absorbed_by": parent_site,
            "upstream_hops": distance,
            "new_unmanaged_site": parent_site,
        })

    return remaining, unmanaged_sites & remaining, absorbed_by, absorb_steps


def connected_components(nodes, relation_index):
    node_set = set(nodes)
    components = []
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
        components.append(component)
    return components


def projected_active_components_by_original_graph(original_sites, active_sites, relation_index):
    """按原始站点图划分连通分量，再投影出吸收后仍保留的站点。"""
    active_site_set = set(active_sites)
    projected_components = []
    for original_component in connected_components(original_sites, relation_index):
        component_active_sites = set(original_component) & active_site_set
        if component_active_sites:
            projected_components.append(component_active_sites)
    return projected_components


def longest_path_in_component(component, relation_index):
    component = set(component)
    if len(component) <= 1:
        return sorted(component)

    adjacency = {
        site_id: sorted(relation_index.undirected_neighbors_in(site_id, component))
        for site_id in component
    }

    # case 通常不大；小图用 DFS 找最长简单链，大图退化为双 BFS 直径近似。
    if len(component) <= 18:
        best_path = []

        def dfs(path, visited):
            nonlocal best_path
            if (
                len(path) > len(best_path)
                or (len(path) == len(best_path) and tuple(path) < tuple(best_path))
            ):
                best_path = list(path)
            for neighbor in adjacency.get(path[-1], []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

        for start in sorted(component):
            dfs([start], {start})
        return best_path

    def farthest(start):
        queue = deque([(start, [start])])
        visited = {start}
        best = [start]
        while queue:
            site_id, path = queue.popleft()
            if len(path) > len(best):
                best = path
            for neighbor in adjacency.get(site_id, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return best

    first_path = farthest(min(component))
    return farthest(first_path[-1])


def classify_chain_uplink(chain, component_sites, relation_index):
    chain = list(chain)
    chain_set = set(chain)
    other_sites = set(component_sites) - chain_set
    external_connected_chain_sites = {
        chain_site
        for chain_site in chain
        for other_site in other_sites
        if relation_index.directly_connected(chain_site, other_site)
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
    return set(relation_index.bidirectional_neighbors(site_id)) | set(relation_index.direct_upstream_neighbors(site_id))


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
        endpoint: set(relation_index.bidirectional_neighbors(endpoint))
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
        endpoint: set(relation_index.direct_upstream_neighbors(endpoint))
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


def classify_component(component_sites, unmanaged_sites, relation_index, router_device_sites=None, absorbed_by=None):
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

    unmanaged_components = connected_components(component_unmanaged_sites, relation_index)
    unmanaged_chains = []
    chain_covered_sites = set()
    for unmanaged_component in unmanaged_components:
        chain = longest_path_in_component(unmanaged_component, relation_index)
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


def analyze_case_record(record, relation_index, ne_to_site, site_has_router_device):
    site_ids = extract_case_sites(record)
    offline_sites = extract_offline_sites(record, ne_to_site) & set(site_ids)
    router_device_sites = extract_case_router_device_sites(record, site_has_router_device)
    active_sites, active_unmanaged_sites, absorbed_by, absorb_steps = absorb_unmanaged_downstream_sites(
        site_ids,
        offline_sites,
        relation_index,
    )

    projected_components = projected_active_components_by_original_graph(site_ids, active_sites, relation_index)
    component_records = []
    for component_sites in projected_components:
        component_record = classify_component(
            component_sites,
            active_unmanaged_sites,
            relation_index,
            router_device_sites=router_device_sites,
            absorbed_by=absorbed_by,
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


def build_context_ne_info(ne_id, ne_graph_entry, group_id):
    return {
        "link": {},
        "group": group_id,
        "name": ne_graph_entry.get("name", ne_id),
        "site_id": normalize_text(ne_graph_entry.get("site_id")),
        "site_name": normalize_text(ne_graph_entry.get("site_name")) or normalize_text(ne_graph_entry.get("site_id")),
        "type": normalize_text(ne_graph_entry.get("type")).upper(),
        "network_type": normalize_text(ne_graph_entry.get("network_type")).upper(),
        "manufacturer": normalize_text(ne_graph_entry.get("manufacturer")).upper(),
        "running_status": ne_graph_entry.get("running_status", ne_graph_entry.get("status", "")),
        "domain": normalize_text(ne_graph_entry.get("domain")).upper(),
        "region_id": normalize_text(ne_graph_entry.get("region_id")),
        "longitude": ne_graph_entry.get("longitude", ne_graph_entry.get("lon", ne_graph_entry.get("lng", ""))),
        "latitude": ne_graph_entry.get("latitude", ne_graph_entry.get("lat", "")),
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
):
    if not ne_graph_data:
        return

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
                ne_info[ne_id] = build_context_ne_info(ne_id, ne_graph_entry, group_id)
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
