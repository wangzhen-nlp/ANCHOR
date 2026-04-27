#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挖掘 case JSONL 中的典型站点故障模式。"""

import argparse
import copy
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alarm_tools.alarm_types import OFFLINE_ALARMS
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.site_topology import (
    build_site_topology_from_ne_graph,
    load_site_chain_index,
)
from topology_resources import NE_GRAPH_JSON, SITE_CHAINS_JSON, resource_display

OFFLINE_ALARM_SET = set(OFFLINE_ALARMS)
ROUTER_DEVICE_DOMAINS = {"DATA"}


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


def iter_jsonl_records(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if isinstance(record, dict):
                yield line_num, record


def count_jsonl_lines(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return sum(1 for line in file_obj if line.strip())


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


def classify_component(component_sites, unmanaged_sites, relation_index, router_device_sites=None):
    component_sites = set(component_sites)
    component_unmanaged_sites = set(unmanaged_sites) & component_sites
    router_device_sites = set(router_device_sites or [])
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
            pattern = "ip_chain"
        elif len(non_downstream_neighbors) >= 2:
            pattern = "multi_link"
        else:
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
            "final_site": candidate if pattern == "ip_chain" else "",
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
        return {
            "pattern": "ip_ring",
            "sites": sorted(component_sites),
            "unmanaged_sites": sorted(component_unmanaged_sites),
            "managed_sites": unmanaged_chains[0]["chain"],
            "final_site": "",
            "final_managed_site": "",
            "chains": unmanaged_chains,
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


def analyze_case_record(record, relation_index, ne_to_site, site_has_router_device):
    site_ids = extract_case_sites(record)
    offline_sites = extract_offline_sites(record, ne_to_site) & set(site_ids)
    router_device_sites = extract_case_router_device_sites(record, site_has_router_device)
    active_sites, active_unmanaged_sites, absorbed_by, absorb_steps = absorb_unmanaged_downstream_sites(
        site_ids,
        offline_sites,
        relation_index,
    )

    component_records = []
    for component_sites in projected_active_components_by_original_graph(site_ids, active_sites, relation_index):
        component_record = classify_component(
            component_sites,
            active_unmanaged_sites,
            relation_index,
            router_device_sites=router_device_sites,
        )
        if component_record.get("pattern") != "unknown":
            component_records.append(component_record)
    component_records.sort(key=lambda item: (item["pattern"], item["sites"]))

    primary_pattern = "none"
    if component_records:
        pattern_priority = {"ip_chain": 0, "multi_link": 1, "ip_ring": 2, "unknown": 3}
        primary_pattern = min(
            (item["pattern"] for item in component_records),
            key=lambda pattern: pattern_priority.get(pattern, 99),
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

    return {
        "uuid": extract_record_uuid(record),
        "rule": normalize_text(as_dict(record.get("match_info")).get("rule") or record.get("rule")),
        "pattern": primary_pattern,
        "patterns": component_records,
        "pattern_count": len(component_records),
        "site_count": len(site_ids),
        "sites": site_ids,
        "offline_sites": sorted(offline_sites),
        "router_device_sites": sorted(router_device_sites & set(site_ids)),
        "active_sites_after_absorption": sorted(active_sites),
        "active_unmanaged_sites": matched_unmanaged_sites,
        "managed_sites": managed_sites,
        "absorbed_by": absorbed_by,
        "absorb_steps": absorb_steps,
        "final_site": component_records[0].get("final_site", "") if len(component_records) == 1 else "",
        "final_managed_site": component_records[0].get("final_managed_site", "") if len(component_records) == 1 else "",
        "chains": [
            chain
            for component in component_records
            for chain in component.get("chains", [])
        ],
    }


def load_ne_graph(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def summarize_patterns(results):
    counts = defaultdict(int)
    for result in results:
        patterns = result.get("patterns", [])
        if not isinstance(patterns, list) or not patterns:
            continue
        for pattern_info in patterns:
            if not isinstance(pattern_info, dict):
                continue
            pattern = pattern_info.get("pattern", "")
            if pattern and pattern != "unknown":
                counts[pattern] += 1
    return {pattern: counts[pattern] for pattern in sorted(counts)}


def format_pattern_summary_line(pattern_info, index):
    pattern = pattern_info.get("pattern", "unknown")
    managed_sites = pattern_info.get("managed_sites", []) or []

    if pattern == "ip_ring":
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


def build_augmented_case_record(record, analysis):
    augmented = copy.deepcopy(record)
    pattern_note = build_pattern_note(analysis)

    augmented["note"] = append_note(augmented.get("note", ""), pattern_note)
    match_info = augmented.setdefault("match_info", {})
    if isinstance(match_info, dict):
        match_info["note"] = append_note(match_info.get("note", ""), pattern_note)

    augmented["fault_pattern_analysis"] = analysis
    augmented["fault_patterns"] = analysis.get("patterns", [])
    augmented["fault_pattern_count"] = analysis.get("pattern_count", 0)
    augmented["fault_pattern_managed_sites"] = analysis.get("managed_sites", [])
    augmented["fault_pattern_active_unmanaged_sites"] = analysis.get("active_unmanaged_sites", [])
    return augmented


def main():
    parser = argparse.ArgumentParser(description="挖掘 evaluation case JSONL 中的典型站点故障模式")
    parser.add_argument("cases", help="compute_ultimate_group_alarm_group_metrics.py 输出的 case JSONL")
    parser.add_argument("-o", "--output", required=True, help="保留原 case 结构并追加模式信息后的 JSONL")
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--site-chains",
        default=SITE_CHAINS_JSON,
        help=f"可选 site_chains.json，优先用于上下游和 hop 判断，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument("--summary-output", help="可选输出模式计数摘要 JSON")
    parser.add_argument("--analysis-only", action="store_true", help="只输出模式分析结果，不保留原始 case 结构")
    args = parser.parse_args()

    ne_graph_data = load_ne_graph(args.ne_graph)
    ne_to_site = {
        ne_id: normalize_text(ne_info.get("site_id"))
        for ne_id, ne_info in ne_graph_data.items()
        if isinstance(ne_info, dict) and normalize_text(ne_info.get("site_id"))
    }
    site_has_router_device = build_site_has_router_device_map(ne_graph_data)
    site_chains_path = args.site_chains if args.site_chains and Path(args.site_chains).exists() else ""
    relation_index = SiteRelationIndex(ne_graph_data=ne_graph_data, site_chains_path=site_chains_path)

    total_cases = count_jsonl_lines(args.cases)
    progress = ProgressBar(total_cases, "分析 case 故障模式", min_interval=0.05)
    results = []
    with open(args.output, "w", encoding="utf-8") as output_file:
        try:
            for line_num, record in iter_jsonl_records(args.cases):
                result = analyze_case_record(record, relation_index, ne_to_site, site_has_router_device)
                result["line_num"] = line_num
                results.append(result)
                output_record = result if args.analysis_only else build_augmented_case_record(record, result)
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                progress.update()
        finally:
            progress.close()

    summary = {
        "case_count": len(results),
        "pattern_counts": summarize_patterns(results),
        "site_chains_used": bool(site_chains_path),
    }
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
