import json

from collections import defaultdict


def extract_link_direction_values(link_meta):
    if isinstance(link_meta, dict):
        raw_values = link_meta.values()
    elif isinstance(link_meta, (list, tuple, set)):
        raw_values = link_meta
    else:
        raw_values = [link_meta]

    direction_values = set()
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if text:
            direction_values.add(text)
    return direction_values


def build_site_topology_from_ne_graph(ne_graph_data):
    """基于 ne_graph 原始连边构建站点级 downstream 拓扑。"""
    ne_to_site = {}
    all_sites = set()

    for ne_id, ne_info in ne_graph_data.items():
        site_id = str(ne_info.get("site_id", "")).strip()
        if not site_id:
            continue
        ne_to_site[ne_id] = site_id
        all_sites.add(site_id)

    topo_downstream_map = defaultdict(set)
    for site_id in all_sites:
        topo_downstream_map[site_id]

    for source_ne, source_info in ne_graph_data.items():
        source_site = ne_to_site.get(source_ne)
        if not source_site:
            continue

        raw_links = source_info.get("link", {})
        if not isinstance(raw_links, dict):
            continue

        for target_ne, link_meta in raw_links.items():
            target_site = ne_to_site.get(target_ne)
            if not target_site or target_site == source_site:
                continue

            direction_values = extract_link_direction_values(link_meta)
            if not direction_values:
                continue

            if any("<-" in direction for direction in direction_values):
                topo_downstream_map[source_site].add(target_site)
            if any("->" in direction for direction in direction_values):
                topo_downstream_map[target_site].add(source_site)

    return {
        site_id: sorted(downstream_sites)
        for site_id, downstream_sites in topo_downstream_map.items()
    }, all_sites


def normalize_site_chain_hops(hops_map):
    normalized = {}
    if not isinstance(hops_map, dict):
        return normalized
    for related_site, hop_value in hops_map.items():
        related_site_id = str(related_site).strip()
        if not related_site_id:
            continue
        try:
            hop = int(hop_value)
        except (TypeError, ValueError):
            continue
        if hop <= 0:
            continue
        normalized[related_site_id] = hop
    return normalized


def load_site_chain_index(site_chains_path):
    """加载 generate_site_chains.py 产出的预计算上下游 hop 索引。"""
    data = json.load(open(site_chains_path, 'r', encoding='utf-8'))
    raw_sites = data.get("sites", {}) if isinstance(data, dict) else {}
    site_chain_index = {}
    valid_sites = set()

    for raw_site_id, raw_info in raw_sites.items():
        site_id = str(raw_site_id or "").strip()
        if not site_id or not isinstance(raw_info, dict):
            continue

        downstream_hops = normalize_site_chain_hops(raw_info.get("downstream_site_hops"))
        upstream_hops = normalize_site_chain_hops(raw_info.get("upstream_site_hops"))
        bidirectional_sites = {
            str(neighbor_site or "").strip()
            for neighbor_site in raw_info.get("bidirectional_sites", [])
            if str(neighbor_site or "").strip()
        }

        site_chain_index[site_id] = {
            "downstream_site_hops": downstream_hops,
            "upstream_site_hops": upstream_hops,
            "bidirectional_sites": bidirectional_sites,
        }
        valid_sites.add(site_id)
        valid_sites.update(downstream_hops)
        valid_sites.update(upstream_hops)
        valid_sites.update(bidirectional_sites)

    return site_chain_index, valid_sites


def build_site_to_ne_ids(ne_graph_data):
    site_to_ne_ids = defaultdict(list)
    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        site_id = str(ne_info.get("site_id", "")).strip()
        if site_id:
            site_to_ne_ids[site_id].append(ne_id)
    return {
        site_id: tuple(sorted(ne_ids))
        for site_id, ne_ids in site_to_ne_ids.items()
    }
