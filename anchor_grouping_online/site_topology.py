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


def build_site_chain_index(data):
    """从已加载的 site_chains 数据构建上下游 hop 索引。"""
    raw_sites = data.get("sites", {}) if isinstance(data, dict) else {}
    site_chain_index = {}

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

    return site_chain_index


def site_chain_upstream_hops_are_complete(data):
    """资源元数据是否保证 upstream_site_hops 未被深度或关系过滤截断。

    旧资源或元数据字段缺失时保守返回 False，由运行时 BFS 补齐。
    """
    if not isinstance(data, dict):
        return False
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return False
    input_config = meta.get("input_config")
    relation_options = meta.get("relation_options")
    if not isinstance(input_config, dict) or not isinstance(relation_options, dict):
        return False
    if "max_depth" not in input_config:
        return False
    if "restrict_relation_effective" not in relation_options:
        return False
    return (
        input_config.get("max_depth") is None
        and relation_options.get("restrict_relation_effective") is False
    )


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
