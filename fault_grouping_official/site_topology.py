import json

from collections import defaultdict


def normalize_site_chain_hops(hops_map):
    normalized = {}
    if not isinstance(hops_map, dict):
        return normalized
    for related_site, hop_value in hops_map.items():
        related_site_id = str(related_site or "").strip()
        if not related_site_id:
            continue
        try:
            hop = int(hop_value)
        except (TypeError, ValueError):
            continue
        if hop > 0:
            normalized[related_site_id] = hop
    return normalized


def load_site_chain_index(site_chains_path):
    """加载 generate_site_chains.py 产出的预计算上下游 hop 索引。"""
    with open(site_chains_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return build_site_chain_index(data)


def build_site_chain_index(data):
    """从已加载的 site_chains 数据构建上下游 hop 索引与有效站点集合。"""
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


def build_site_domain_map(ne_graph_data):
    """直接从 ne_graph 构建规则所需的站点域画像。"""
    site_domains = defaultdict(lambda: defaultdict(int))
    for ne_info in ne_graph_data.values():
        if not isinstance(ne_info, dict):
            continue
        site_id = str(ne_info.get("site_id", "")).strip()
        domain = str(ne_info.get("domain", "")).strip()
        if site_id and domain:
            site_domains[site_id][domain] += 1
    return {
        site_id: dict(domain_counts)
        for site_id, domain_counts in site_domains.items()
    }


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
