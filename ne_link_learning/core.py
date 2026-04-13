#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from alarm_tools.progress_utils import ProgressBar

DOMAIN_BUCKETS = ("RAN", "TRANSMISSION", "DATA", "OTHER", "MISSING")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def safe_log1p(value):
    if value <= -1:
        return 0.0
    return math.log1p(value)


def stable_hash_fraction(text, seed=42):
    payload = f"{seed}::{text}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _create_progress_bar(total, label, show_progress):
    if not show_progress:
        return None
    print(f"⏳ {label}...")
    return ProgressBar(total, label)


def _close_progress_bar(progress):
    if progress is not None:
        progress.close()


def deterministic_sample(seq, k, rng):
    seq = list(seq)
    if k <= 0 or not seq:
        return []
    if len(seq) <= k:
        return list(seq)
    return rng.sample(seq, k)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ne_graph(ne_graph_file):
    data = load_json(ne_graph_file)
    if not isinstance(data, dict):
        raise ValueError(f"{ne_graph_file} 顶层必须是对象")
    return data


def write_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(filepath, items):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(filepath):
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            items.append(json.loads(text))
    return items


def _extract_text(ne_info, *field_names):
    if not isinstance(ne_info, dict):
        return ""
    for field_name in field_names:
        value = normalize_text(ne_info.get(field_name, ""))
        if value:
            return value
    return ""


def normalize_domain_bucket(domain):
    normalized_domain = normalize_text(domain)
    if not normalized_domain:
        return "MISSING"
    if normalized_domain == "RAN":
        return "RAN"
    if normalized_domain == "TRANSMISSION":
        return "TRANSMISSION"
    if normalized_domain == "DATA":
        return "DATA"
    return "OTHER"


def _parse_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None

    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return radius_km * c


def _normalize_link_direction(direction):
    text = str(direction).strip()
    if text in {"->", "<-", "<->"}:
        return text
    if "->" in text and "<-" in text:
        return "<->"
    if "->" in text:
        return "->"
    if "<-" in text:
        return "<-"
    return ""


def extract_outgoing_link_types(link_meta):
    outgoing_types = set()
    if isinstance(link_meta, dict):
        for link_type, direction in link_meta.items():
            normalized_direction = _normalize_link_direction(direction)
            if normalized_direction in {"->", "<->"}:
                outgoing_types.add(normalize_text(link_type) or "UNKNOWN")
    elif isinstance(link_meta, str):
        normalized_direction = _normalize_link_direction(link_meta)
        if normalized_direction in {"->", "<->"}:
            outgoing_types.add("UNKNOWN")
    return outgoing_types


def extract_incoming_link_types(link_meta):
    incoming_types = set()
    if isinstance(link_meta, dict):
        for link_type, direction in link_meta.items():
            normalized_direction = _normalize_link_direction(direction)
            if normalized_direction in {"<-", "<->"}:
                incoming_types.add(normalize_text(link_type) or "UNKNOWN")
    elif isinstance(link_meta, str):
        normalized_direction = _normalize_link_direction(link_meta)
        if normalized_direction in {"<-", "<->"}:
            incoming_types.add("UNKNOWN")
    return incoming_types


@dataclass(frozen=True)
class NodeInfo:
    ne_id: str
    site_id: str
    site_name: str
    domain: str
    domain_bucket: str
    ne_type: str
    network_type: str
    manufacturer: str
    region_id: str
    latitude: float | None
    longitude: float | None


@dataclass
class GraphContext:
    node_infos: dict
    site_to_nodes: dict
    out_neighbors: dict
    in_neighbors: dict
    undirected_neighbors: dict
    edge_link_types: dict
    node_out_link_types: dict
    node_in_link_types: dict
    site_coords: dict
    site_domain_bucket_counts: dict
    site_type_counts: dict
    site_network_type_counts: dict
    site_manufacturer_counts: dict
    site_out_sites: dict
    site_in_sites: dict
    site_pair_forward_edge_count: dict
    site_pair_link_type_counts: dict
    site_pair_domain_forward_count: dict
    site_incoming_source_domain_counts: dict
    site_outgoing_target_domain_counts: dict
    region_to_nodes: dict
    domain_bucket_to_nodes: dict


def build_graph_context(ne_graph_data):
    node_infos = {}
    site_to_nodes = defaultdict(list)
    site_coords = {}
    site_domain_bucket_counts = defaultdict(Counter)
    site_type_counts = defaultdict(Counter)
    site_network_type_counts = defaultdict(Counter)
    site_manufacturer_counts = defaultdict(Counter)
    region_to_nodes = defaultdict(list)
    domain_bucket_to_nodes = defaultdict(list)

    for raw_ne_id, raw_ne_info in ne_graph_data.items():
        if not isinstance(raw_ne_info, dict):
            continue

        ne_id = normalize_text(raw_ne_id)
        site_id = _extract_text(raw_ne_info, "site_id")
        if not ne_id or not site_id:
            continue

        domain = _extract_text(raw_ne_info, "domain", "Domain", "DOMAIN")
        domain_bucket = normalize_domain_bucket(domain)
        node_info = NodeInfo(
            ne_id=ne_id,
            site_id=site_id,
            site_name=_extract_text(raw_ne_info, "site_name", "siteName", "name"),
            domain=domain,
            domain_bucket=domain_bucket,
            ne_type=_extract_text(raw_ne_info, "type", "typeId", "TYPE"),
            network_type=_extract_text(raw_ne_info, "network_type", "NETWORK_TYPE"),
            manufacturer=_extract_text(raw_ne_info, "manufacturer", "MANUFACTURER"),
            region_id=_extract_text(raw_ne_info, "region_id", "regionId", "REGION_ID"),
            latitude=_parse_float(raw_ne_info.get("latitude", raw_ne_info.get("lat"))),
            longitude=_parse_float(
                raw_ne_info.get("longitude", raw_ne_info.get("lon", raw_ne_info.get("lng")))
            ),
        )
        node_infos[ne_id] = node_info
        site_to_nodes[site_id].append(ne_id)
        site_domain_bucket_counts[site_id][domain_bucket] += 1
        site_type_counts[site_id][node_info.ne_type or "MISSING"] += 1
        site_network_type_counts[site_id][node_info.network_type or "MISSING"] += 1
        site_manufacturer_counts[site_id][node_info.manufacturer or "MISSING"] += 1
        region_to_nodes[node_info.region_id or "MISSING"].append(ne_id)
        domain_bucket_to_nodes[domain_bucket].append(ne_id)
        if site_id not in site_coords and node_info.latitude is not None and node_info.longitude is not None:
            site_coords[site_id] = (node_info.latitude, node_info.longitude)

    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)
    edge_link_types = defaultdict(set)
    node_out_link_types = defaultdict(set)
    node_in_link_types = defaultdict(set)

    for raw_ne_id, raw_ne_info in ne_graph_data.items():
        if not isinstance(raw_ne_info, dict):
            continue

        ne_id = normalize_text(raw_ne_id)
        if ne_id not in node_infos:
            continue

        links = raw_ne_info.get("link", {})
        if not isinstance(links, dict):
            continue

        for raw_neighbor_id, link_meta in links.items():
            neighbor_id = normalize_text(raw_neighbor_id)
            if not neighbor_id or neighbor_id not in node_infos or neighbor_id == ne_id:
                continue

            outgoing_types = extract_outgoing_link_types(link_meta)
            incoming_types = extract_incoming_link_types(link_meta)

            if outgoing_types:
                out_neighbors[ne_id].add(neighbor_id)
                in_neighbors[neighbor_id].add(ne_id)
                edge_link_types[(ne_id, neighbor_id)].update(outgoing_types)
                node_out_link_types[ne_id].update(outgoing_types)
                node_in_link_types[neighbor_id].update(outgoing_types)

            if incoming_types:
                in_neighbors[ne_id].add(neighbor_id)
                out_neighbors[neighbor_id].add(ne_id)
                edge_link_types[(neighbor_id, ne_id)].update(incoming_types)
                node_in_link_types[ne_id].update(incoming_types)
                node_out_link_types[neighbor_id].update(incoming_types)

    undirected_neighbors = {
        ne_id: set(out_neighbors.get(ne_id, set())) | set(in_neighbors.get(ne_id, set()))
        for ne_id in node_infos
    }

    site_out_sites = defaultdict(set)
    site_in_sites = defaultdict(set)
    site_pair_forward_edge_count = defaultdict(int)
    site_pair_link_type_counts = defaultdict(Counter)
    site_pair_domain_forward_count = defaultdict(Counter)
    site_incoming_source_domain_counts = defaultdict(Counter)
    site_outgoing_target_domain_counts = defaultdict(Counter)

    for (left_ne_id, right_ne_id), link_types in edge_link_types.items():
        left_info = node_infos[left_ne_id]
        right_info = node_infos[right_ne_id]
        if left_info.site_id == right_info.site_id:
            continue

        site_pair_key = (left_info.site_id, right_info.site_id)
        site_pair_forward_edge_count[site_pair_key] += 1
        site_pair_link_type_counts[site_pair_key].update(link_types)
        site_pair_domain_forward_count[site_pair_key][
            (left_info.domain_bucket, right_info.domain_bucket)
        ] += 1
        site_out_sites[left_info.site_id].add(right_info.site_id)
        site_in_sites[right_info.site_id].add(left_info.site_id)
        site_incoming_source_domain_counts[right_info.site_id][left_info.domain_bucket] += 1
        site_outgoing_target_domain_counts[left_info.site_id][right_info.domain_bucket] += 1

    return GraphContext(
        node_infos=node_infos,
        site_to_nodes={site_id: sorted(ne_ids) for site_id, ne_ids in site_to_nodes.items()},
        out_neighbors={ne_id: set(neighbors) for ne_id, neighbors in out_neighbors.items()},
        in_neighbors={ne_id: set(neighbors) for ne_id, neighbors in in_neighbors.items()},
        undirected_neighbors=undirected_neighbors,
        edge_link_types={key: set(value) for key, value in edge_link_types.items()},
        node_out_link_types={ne_id: set(value) for ne_id, value in node_out_link_types.items()},
        node_in_link_types={ne_id: set(value) for ne_id, value in node_in_link_types.items()},
        site_coords=dict(site_coords),
        site_domain_bucket_counts={site_id: Counter(counter) for site_id, counter in site_domain_bucket_counts.items()},
        site_type_counts={site_id: Counter(counter) for site_id, counter in site_type_counts.items()},
        site_network_type_counts={site_id: Counter(counter) for site_id, counter in site_network_type_counts.items()},
        site_manufacturer_counts={site_id: Counter(counter) for site_id, counter in site_manufacturer_counts.items()},
        site_out_sites={site_id: set(value) for site_id, value in site_out_sites.items()},
        site_in_sites={site_id: set(value) for site_id, value in site_in_sites.items()},
        site_pair_forward_edge_count=dict(site_pair_forward_edge_count),
        site_pair_link_type_counts={key: Counter(value) for key, value in site_pair_link_type_counts.items()},
        site_pair_domain_forward_count={key: Counter(value) for key, value in site_pair_domain_forward_count.items()},
        site_incoming_source_domain_counts={site_id: Counter(value) for site_id, value in site_incoming_source_domain_counts.items()},
        site_outgoing_target_domain_counts={site_id: Counter(value) for site_id, value in site_outgoing_target_domain_counts.items()},
        region_to_nodes={region_id: sorted(ne_ids) for region_id, ne_ids in region_to_nodes.items()},
        domain_bucket_to_nodes={bucket: sorted(ne_ids) for bucket, ne_ids in domain_bucket_to_nodes.items()},
    )


def collect_positive_edges(context):
    positives = []
    for left_ne_id, right_ne_id in sorted(context.edge_link_types):
        left_info = context.node_infos[left_ne_id]
        right_info = context.node_infos[right_ne_id]
        if left_info.site_id == right_info.site_id:
            continue
        positives.append((left_ne_id, right_ne_id))
    return positives


def _jaccard(left_set, right_set):
    union_size = len(left_set | right_set)
    if union_size == 0:
        return 0.0
    return float(len(left_set & right_set)) / float(union_size)


def _adamic_adar_score(shared_nodes, undirected_neighbors):
    score = 0.0
    for node_id in shared_nodes:
        degree = len(undirected_neighbors.get(node_id, set()))
        if degree > 1:
            score += 1.0 / math.log(degree)
    return score


def _resource_allocation_score(shared_nodes, undirected_neighbors):
    score = 0.0
    for node_id in shared_nodes:
        degree = len(undirected_neighbors.get(node_id, set()))
        if degree > 0:
            score += 1.0 / float(degree)
    return score


def _count_nodes_on_site(node_ids, context, site_id):
    return sum(1 for node_id in node_ids if context.node_infos[node_id].site_id == site_id)


def _count_neighbor_sites(node_ids, context):
    return len({context.node_infos[node_id].site_id for node_id in node_ids})


def extract_pair_features(context, left_ne_id, right_ne_id):
    left_info = context.node_infos[left_ne_id]
    right_info = context.node_infos[right_ne_id]
    left_site_id = left_info.site_id
    right_site_id = right_info.site_id

    left_out = context.out_neighbors.get(left_ne_id, set())
    left_in = context.in_neighbors.get(left_ne_id, set())
    right_out = context.out_neighbors.get(right_ne_id, set())
    right_in = context.in_neighbors.get(right_ne_id, set())
    left_undirected = context.undirected_neighbors.get(left_ne_id, set())
    right_undirected = context.undirected_neighbors.get(right_ne_id, set())

    left_site_nodes = context.site_to_nodes.get(left_site_id, [])
    right_site_nodes = context.site_to_nodes.get(right_site_id, [])
    left_site_size = len(left_site_nodes)
    right_site_size = len(right_site_nodes)

    common_out = left_out & right_out
    common_in = left_in & right_in
    common_undirected = left_undirected & right_undirected
    mids_left_to_right = left_out & right_in
    mids_right_to_left = right_out & left_in

    left_out_to_right_site_count = _count_nodes_on_site(left_out, context, right_site_id)
    right_in_from_left_site_count = _count_nodes_on_site(right_in, context, left_site_id)

    left_coords = context.site_coords.get(left_site_id)
    right_coords = context.site_coords.get(right_site_id)
    geo_distance_km = None
    if left_coords and right_coords:
        geo_distance_km = haversine_km(
            left_coords[0], left_coords[1],
            right_coords[0], right_coords[1],
        )

    left_type_counts = context.site_type_counts.get(left_site_id, Counter())
    right_type_counts = context.site_type_counts.get(right_site_id, Counter())
    left_network_type_counts = context.site_network_type_counts.get(left_site_id, Counter())
    right_network_type_counts = context.site_network_type_counts.get(right_site_id, Counter())
    left_manufacturer_counts = context.site_manufacturer_counts.get(left_site_id, Counter())
    right_manufacturer_counts = context.site_manufacturer_counts.get(right_site_id, Counter())
    left_domain_counts = context.site_domain_bucket_counts.get(left_site_id, Counter())
    right_domain_counts = context.site_domain_bucket_counts.get(right_site_id, Counter())

    site_pair_key = (left_site_id, right_site_id)
    reverse_site_pair_key = (right_site_id, left_site_id)
    site_pair_forward_edge_count = context.site_pair_forward_edge_count.get(site_pair_key, 0)
    site_pair_reverse_edge_count = context.site_pair_forward_edge_count.get(reverse_site_pair_key, 0)
    site_pair_total_edge_count = site_pair_forward_edge_count + site_pair_reverse_edge_count
    site_pair_link_type_counter = context.site_pair_link_type_counts.get(site_pair_key, Counter())
    site_pair_domain_counter = context.site_pair_domain_forward_count.get(site_pair_key, Counter())

    left_domain_bucket = left_info.domain_bucket
    right_domain_bucket = right_info.domain_bucket
    domain_pair_forward_count = site_pair_domain_counter.get((left_domain_bucket, right_domain_bucket), 0)

    left_site_possible_edge_count = max(1, left_site_size * right_site_size)
    left_out_sites = {context.node_infos[node_id].site_id for node_id in left_out}
    left_in_sites = {context.node_infos[node_id].site_id for node_id in left_in}
    right_out_sites = {context.node_infos[node_id].site_id for node_id in right_out}
    right_in_sites = {context.node_infos[node_id].site_id for node_id in right_in}

    features = {
        "same_site": float(left_site_id == right_site_id),
        "same_region": float(
            bool(left_info.region_id) and left_info.region_id == right_info.region_id
        ),
        "same_domain": float(
            left_info.domain_bucket == right_info.domain_bucket
            and left_info.domain_bucket != "MISSING"
        ),
        "same_type": float(
            bool(left_info.ne_type) and left_info.ne_type == right_info.ne_type
        ),
        "same_network_type": float(
            bool(left_info.network_type) and left_info.network_type == right_info.network_type
        ),
        "same_manufacturer": float(
            bool(left_info.manufacturer) and left_info.manufacturer == right_info.manufacturer
        ),
        "geo_distance_km": geo_distance_km or 0.0,
        "geo_distance_missing": float(geo_distance_km is None),
        "geo_distance_log1p": safe_log1p(geo_distance_km or 0.0),
        "reverse_edge_exists": float(left_ne_id in context.out_neighbors.get(right_ne_id, set())),
        "left_out_degree": float(len(left_out)),
        "left_in_degree": float(len(left_in)),
        "right_out_degree": float(len(right_out)),
        "right_in_degree": float(len(right_in)),
        "left_undirected_degree": float(len(left_undirected)),
        "right_undirected_degree": float(len(right_undirected)),
        "left_cross_site_out_degree": float(
            sum(1 for node_id in left_out if context.node_infos[node_id].site_id != left_site_id)
        ),
        "left_cross_site_in_degree": float(
            sum(1 for node_id in left_in if context.node_infos[node_id].site_id != left_site_id)
        ),
        "right_cross_site_out_degree": float(
            sum(1 for node_id in right_out if context.node_infos[node_id].site_id != right_site_id)
        ),
        "right_cross_site_in_degree": float(
            sum(1 for node_id in right_in if context.node_infos[node_id].site_id != right_site_id)
        ),
        "left_same_site_out_degree": float(
            sum(1 for node_id in left_out if context.node_infos[node_id].site_id == left_site_id)
        ),
        "left_same_site_in_degree": float(
            sum(1 for node_id in left_in if context.node_infos[node_id].site_id == left_site_id)
        ),
        "right_same_site_out_degree": float(
            sum(1 for node_id in right_out if context.node_infos[node_id].site_id == right_site_id)
        ),
        "right_same_site_in_degree": float(
            sum(1 for node_id in right_in if context.node_infos[node_id].site_id == right_site_id)
        ),
        "left_site_size": float(left_site_size),
        "right_site_size": float(right_site_size),
        "left_site_out_site_degree": float(len(context.site_out_sites.get(left_site_id, set()))),
        "left_site_in_site_degree": float(len(context.site_in_sites.get(left_site_id, set()))),
        "right_site_out_site_degree": float(len(context.site_out_sites.get(right_site_id, set()))),
        "right_site_in_site_degree": float(len(context.site_in_sites.get(right_site_id, set()))),
        "left_type_site_ratio": safe_ratio(
            left_type_counts.get(left_info.ne_type or "MISSING", 0), left_site_size
        ),
        "right_type_site_ratio": safe_ratio(
            right_type_counts.get(right_info.ne_type or "MISSING", 0), right_site_size
        ),
        "left_network_type_site_ratio": safe_ratio(
            left_network_type_counts.get(left_info.network_type or "MISSING", 0), left_site_size
        ),
        "right_network_type_site_ratio": safe_ratio(
            right_network_type_counts.get(right_info.network_type or "MISSING", 0), right_site_size
        ),
        "left_manufacturer_site_ratio": safe_ratio(
            left_manufacturer_counts.get(left_info.manufacturer or "MISSING", 0), left_site_size
        ),
        "right_manufacturer_site_ratio": safe_ratio(
            right_manufacturer_counts.get(right_info.manufacturer or "MISSING", 0), right_site_size
        ),
        "site_pair_forward_edge_count": float(site_pair_forward_edge_count),
        "site_pair_reverse_edge_count": float(site_pair_reverse_edge_count),
        "site_pair_total_edge_count": float(site_pair_total_edge_count),
        "site_pair_forward_density": safe_ratio(site_pair_forward_edge_count, left_site_possible_edge_count),
        "site_pair_total_density": safe_ratio(site_pair_total_edge_count, left_site_possible_edge_count),
        "site_pair_link_type_diversity": float(len(site_pair_link_type_counter)),
        "site_pair_domain_pair_count": float(domain_pair_forward_count),
        "site_pair_domain_pair_ratio": safe_ratio(domain_pair_forward_count, site_pair_forward_edge_count),
        "left_out_to_right_site_count": float(left_out_to_right_site_count),
        "right_in_from_left_site_count": float(right_in_from_left_site_count),
        "left_out_to_right_site_ratio": safe_ratio(left_out_to_right_site_count, right_site_size),
        "right_in_from_left_site_ratio": safe_ratio(right_in_from_left_site_count, left_site_size),
        "common_out_count": float(len(common_out)),
        "common_in_count": float(len(common_in)),
        "common_neighbor_count": float(len(common_undirected)),
        "jaccard_out": _jaccard(left_out, right_out),
        "jaccard_in": _jaccard(left_in, right_in),
        "jaccard_neighbor": _jaccard(left_undirected, right_undirected),
        "two_hop_left_to_right_count": float(len(mids_left_to_right)),
        "two_hop_right_to_left_count": float(len(mids_right_to_left)),
        "shared_target_site_count": float(len(left_out_sites & right_out_sites)),
        "shared_source_site_count": float(len(left_in_sites & right_in_sites)),
        "left_out_site_count": float(len(left_out_sites)),
        "left_in_site_count": float(len(left_in_sites)),
        "right_out_site_count": float(len(right_out_sites)),
        "right_in_site_count": float(len(right_in_sites)),
        "left_out_link_type_diversity": float(len(context.node_out_link_types.get(left_ne_id, set()))),
        "left_in_link_type_diversity": float(len(context.node_in_link_types.get(left_ne_id, set()))),
        "right_out_link_type_diversity": float(len(context.node_out_link_types.get(right_ne_id, set()))),
        "right_in_link_type_diversity": float(len(context.node_in_link_types.get(right_ne_id, set()))),
        "left_right_link_type_overlap": float(
            len(context.node_out_link_types.get(left_ne_id, set()) & context.node_in_link_types.get(right_ne_id, set()))
        ),
        "left_site_receives_from_right_domain_count": float(
            context.site_incoming_source_domain_counts.get(left_site_id, Counter()).get(right_domain_bucket, 0)
        ),
        "right_site_receives_from_left_domain_count": float(
            context.site_incoming_source_domain_counts.get(right_site_id, Counter()).get(left_domain_bucket, 0)
        ),
        "left_site_sends_to_right_domain_count": float(
            context.site_outgoing_target_domain_counts.get(left_site_id, Counter()).get(right_domain_bucket, 0)
        ),
        "right_site_sends_to_left_domain_count": float(
            context.site_outgoing_target_domain_counts.get(right_site_id, Counter()).get(left_domain_bucket, 0)
        ),
        "adamic_adar_neighbor": _adamic_adar_score(common_undirected, context.undirected_neighbors),
        "resource_allocation_neighbor": _resource_allocation_score(common_undirected, context.undirected_neighbors),
        "adamic_adar_two_hop_left_to_right": _adamic_adar_score(mids_left_to_right, context.undirected_neighbors),
        "resource_allocation_two_hop_left_to_right": _resource_allocation_score(mids_left_to_right, context.undirected_neighbors),
        "left_domain_missing": float(left_domain_bucket == "MISSING"),
        "right_domain_missing": float(right_domain_bucket == "MISSING"),
        "left_type_missing": float(not left_info.ne_type),
        "right_type_missing": float(not right_info.ne_type),
        "left_network_type_missing": float(not left_info.network_type),
        "right_network_type_missing": float(not right_info.network_type),
        "left_manufacturer_missing": float(not left_info.manufacturer),
        "right_manufacturer_missing": float(not right_info.manufacturer),
        "left_region_missing": float(not left_info.region_id),
        "right_region_missing": float(not right_info.region_id),
    }

    for domain_bucket in DOMAIN_BUCKETS:
        features[f"left_site_domain_ratio__{domain_bucket.lower()}"] = safe_ratio(
            left_domain_counts.get(domain_bucket, 0), left_site_size
        )
        features[f"right_site_domain_ratio__{domain_bucket.lower()}"] = safe_ratio(
            right_domain_counts.get(domain_bucket, 0), right_site_size
        )
        features[f"left_domain_is__{domain_bucket.lower()}"] = float(left_domain_bucket == domain_bucket)
        features[f"right_domain_is__{domain_bucket.lower()}"] = float(right_domain_bucket == domain_bucket)

    for left_bucket in DOMAIN_BUCKETS:
        for right_bucket in DOMAIN_BUCKETS:
            features[
                f"domain_pair__{left_bucket.lower()}__{right_bucket.lower()}"
            ] = float(left_domain_bucket == left_bucket and right_domain_bucket == right_bucket)

    return features


def _make_split_keys(left_site_id, right_site_id):
    ordered_key = f"{left_site_id}__TO__{right_site_id}"
    unordered_key = "__".join(sorted([left_site_id, right_site_id]))
    return ordered_key, unordered_key


def _build_sample(context, left_ne_id, right_ne_id, label, candidate_reasons, sample_role):
    features = extract_pair_features(context, left_ne_id, right_ne_id)
    left_info = context.node_infos[left_ne_id]
    right_info = context.node_infos[right_ne_id]
    ordered_site_pair_key, unordered_site_pair_key = _make_split_keys(
        left_info.site_id, right_info.site_id
    )
    sample_id = f"{left_ne_id}__{right_ne_id}"

    return {
        "sample_id": sample_id,
        "label": int(label),
        "sample_role": sample_role,
        "u_ne_id": left_ne_id,
        "v_ne_id": right_ne_id,
        "u_site_id": left_info.site_id,
        "v_site_id": right_info.site_id,
        "u_site_name": left_info.site_name,
        "v_site_name": right_info.site_name,
        "u_domain": left_info.domain_bucket,
        "v_domain": right_info.domain_bucket,
        "ordered_site_pair_key": ordered_site_pair_key,
        "unordered_site_pair_key": unordered_site_pair_key,
        "candidate_reasons": sorted(candidate_reasons),
        "link_types": sorted(context.edge_link_types.get((left_ne_id, right_ne_id), set())),
        "features": features,
    }


def _try_add_negative_pair(context, negative_reason_map, positive_edge_set, left_ne_id, right_ne_id, reason):
    if left_ne_id == right_ne_id:
        return False
    if left_ne_id not in context.node_infos or right_ne_id not in context.node_infos:
        return False

    left_info = context.node_infos[left_ne_id]
    right_info = context.node_infos[right_ne_id]
    if left_info.site_id == right_info.site_id:
        return False
    if (left_ne_id, right_ne_id) in positive_edge_set:
        return False
    if right_ne_id in context.out_neighbors.get(left_ne_id, set()):
        return False

    negative_reason_map[(left_ne_id, right_ne_id)].add(reason)
    return True


def _sample_site_pair_negatives(context, positive_edge_set, negative_reason_map, rng, left_site_id, right_site_id, sample_count):
    if sample_count <= 0:
        return

    left_site_nodes = context.site_to_nodes.get(left_site_id, [])
    right_site_nodes = context.site_to_nodes.get(right_site_id, [])
    if not left_site_nodes or not right_site_nodes:
        return

    chosen_pairs = set()
    max_attempts = max(20, sample_count * 20)
    attempts = 0
    while len(chosen_pairs) < sample_count and attempts < max_attempts:
        attempts += 1
        left_ne_id = rng.choice(left_site_nodes)
        right_ne_id = rng.choice(right_site_nodes)
        if _try_add_negative_pair(
            context,
            negative_reason_map,
            positive_edge_set,
            left_ne_id,
            right_ne_id,
            "same_site_pair_unlinked",
        ):
            chosen_pairs.add((left_ne_id, right_ne_id))


def _generate_local_negative_pool(
    context,
    positive_edges,
    same_source_site_negatives,
    same_target_site_negatives,
    two_hop_target_negatives,
    two_hop_source_negatives,
    site_pair_negatives,
    reverse_direction_negatives,
    rng,
):
    positive_edge_set = set(positive_edges)
    negative_reason_map = defaultdict(set)

    for left_ne_id, right_ne_id in positive_edges:
        left_info = context.node_infos[left_ne_id]
        right_info = context.node_infos[right_ne_id]
        left_site_id = left_info.site_id
        right_site_id = right_info.site_id

        if reverse_direction_negatives > 0:
            _try_add_negative_pair(
                context,
                negative_reason_map,
                positive_edge_set,
                right_ne_id,
                left_ne_id,
                "reverse_direction_missing",
            )

        if same_target_site_negatives > 0:
            candidate_targets = [
                candidate_ne_id
                for candidate_ne_id in context.site_to_nodes.get(right_site_id, [])
                if candidate_ne_id != right_ne_id
            ]
            sampled_targets = deterministic_sample(candidate_targets, same_target_site_negatives, rng)
            for candidate_ne_id in sampled_targets:
                _try_add_negative_pair(
                    context,
                    negative_reason_map,
                    positive_edge_set,
                    left_ne_id,
                    candidate_ne_id,
                    "same_target_site",
                )

        if same_source_site_negatives > 0:
            candidate_sources = [
                candidate_ne_id
                for candidate_ne_id in context.site_to_nodes.get(left_site_id, [])
                if candidate_ne_id != left_ne_id
            ]
            sampled_sources = deterministic_sample(candidate_sources, same_source_site_negatives, rng)
            for candidate_ne_id in sampled_sources:
                _try_add_negative_pair(
                    context,
                    negative_reason_map,
                    positive_edge_set,
                    candidate_ne_id,
                    right_ne_id,
                    "same_source_site",
                )

        if two_hop_target_negatives > 0:
            candidate_targets = set()
            for mid_ne_id in context.out_neighbors.get(left_ne_id, set()):
                candidate_targets.update(context.out_neighbors.get(mid_ne_id, set()))
            candidate_targets.discard(left_ne_id)
            sampled_targets = deterministic_sample(sorted(candidate_targets), two_hop_target_negatives, rng)
            for candidate_ne_id in sampled_targets:
                _try_add_negative_pair(
                    context,
                    negative_reason_map,
                    positive_edge_set,
                    left_ne_id,
                    candidate_ne_id,
                    "two_hop_target",
                )

        if two_hop_source_negatives > 0:
            candidate_sources = set()
            for mid_ne_id in context.in_neighbors.get(right_ne_id, set()):
                candidate_sources.update(context.in_neighbors.get(mid_ne_id, set()))
            candidate_sources.discard(right_ne_id)
            sampled_sources = deterministic_sample(sorted(candidate_sources), two_hop_source_negatives, rng)
            for candidate_ne_id in sampled_sources:
                _try_add_negative_pair(
                    context,
                    negative_reason_map,
                    positive_edge_set,
                    candidate_ne_id,
                    right_ne_id,
                    "two_hop_source",
                )

        _sample_site_pair_negatives(
            context,
            positive_edge_set,
            negative_reason_map,
            rng,
            left_site_id,
            right_site_id,
            site_pair_negatives,
        )

    return negative_reason_map


def _pick_random_hard_target(context, left_ne_id, rng):
    left_info = context.node_infos[left_ne_id]
    candidate_groups = []

    if left_info.region_id:
        candidate_groups.append(context.region_to_nodes.get(left_info.region_id, []))
    if left_info.domain_bucket and left_info.domain_bucket != "MISSING":
        candidate_groups.append(context.domain_bucket_to_nodes.get(left_info.domain_bucket, []))

    candidate_groups.extend(context.domain_bucket_to_nodes.values())

    for candidate_group in candidate_groups:
        if not candidate_group:
            continue
        candidate_ne_id = rng.choice(candidate_group)
        candidate_info = context.node_infos.get(candidate_ne_id)
        if not candidate_info:
            continue
        if candidate_info.site_id != left_info.site_id:
            return candidate_ne_id

    all_nodes = list(context.node_infos)
    for _ in range(20):
        candidate_ne_id = rng.choice(all_nodes)
        candidate_info = context.node_infos[candidate_ne_id]
        if candidate_info.site_id != left_info.site_id:
            return candidate_ne_id
    return ""


def generate_link_learning_samples(
    context,
    max_negative_per_positive=4.0,
    seed=42,
    same_source_site_negatives=1,
    same_target_site_negatives=1,
    two_hop_target_negatives=1,
    two_hop_source_negatives=1,
    site_pair_negatives=1,
    reverse_direction_negatives=1,
    random_hard_negative_ratio=1.0,
):
    rng = random.Random(seed)
    positive_edges = collect_positive_edges(context)
    positive_edge_set = set(positive_edges)
    negative_reason_map = _generate_local_negative_pool(
        context=context,
        positive_edges=positive_edges,
        same_source_site_negatives=same_source_site_negatives,
        same_target_site_negatives=same_target_site_negatives,
        two_hop_target_negatives=two_hop_target_negatives,
        two_hop_source_negatives=two_hop_source_negatives,
        site_pair_negatives=site_pair_negatives,
        reverse_direction_negatives=reverse_direction_negatives,
        rng=rng,
    )

    target_negative_count = int(math.ceil(len(positive_edges) * max(0.0, float(max_negative_per_positive))))
    if random_hard_negative_ratio > 0:
        extra_random_target = int(math.ceil(len(positive_edges) * float(random_hard_negative_ratio)))
    else:
        extra_random_target = 0

    random_negative_attempts = 0
    max_random_attempts = max(2000, extra_random_target * 50)
    all_left_nodes = list(context.node_infos)
    while (
        len(negative_reason_map) < target_negative_count + extra_random_target
        and random_negative_attempts < max_random_attempts
        and all_left_nodes
    ):
        random_negative_attempts += 1
        left_ne_id = rng.choice(all_left_nodes)
        right_ne_id = _pick_random_hard_target(context, left_ne_id, rng)
        if not right_ne_id:
            continue
        _try_add_negative_pair(
            context,
            negative_reason_map,
            positive_edge_set,
            left_ne_id,
            right_ne_id,
            "random_hard_negative",
    )

    negative_items = list(negative_reason_map.items())
    if target_negative_count <= 0:
        negative_items = []
    elif len(negative_items) > target_negative_count:
        negative_items = rng.sample(negative_items, target_negative_count)

    positive_samples = [
        _build_sample(context, left_ne_id, right_ne_id, 1, {"observed_edge"}, "positive")
        for left_ne_id, right_ne_id in positive_edges
    ]
    negative_samples = [
        _build_sample(context, left_ne_id, right_ne_id, 0, reasons, "negative")
        for (left_ne_id, right_ne_id), reasons in negative_items
    ]

    samples = positive_samples + negative_samples
    rng.shuffle(samples)
    return samples


def generate_candidate_link_samples_for_scoring(
    context,
    max_candidate_count=20000,
    seed=42,
    same_source_site_negatives=2,
    same_target_site_negatives=2,
    two_hop_target_negatives=2,
    two_hop_source_negatives=2,
    site_pair_negatives=2,
    reverse_direction_negatives=1,
    random_hard_negative_ratio=2.0,
):
    rng = random.Random(seed)
    positive_edges = collect_positive_edges(context)
    positive_edge_set = set(positive_edges)
    negative_reason_map = _generate_local_negative_pool(
        context=context,
        positive_edges=positive_edges,
        same_source_site_negatives=same_source_site_negatives,
        same_target_site_negatives=same_target_site_negatives,
        two_hop_target_negatives=two_hop_target_negatives,
        two_hop_source_negatives=two_hop_source_negatives,
        site_pair_negatives=site_pair_negatives,
        reverse_direction_negatives=reverse_direction_negatives,
        rng=rng,
    )

    extra_random_target = int(math.ceil(len(positive_edges) * max(0.0, float(random_hard_negative_ratio))))
    random_negative_attempts = 0
    max_random_attempts = max(2000, extra_random_target * 50)
    all_left_nodes = list(context.node_infos)
    while (
        len(negative_reason_map) < extra_random_target
        and random_negative_attempts < max_random_attempts
        and all_left_nodes
    ):
        random_negative_attempts += 1
        left_ne_id = rng.choice(all_left_nodes)
        right_ne_id = _pick_random_hard_target(context, left_ne_id, rng)
        if not right_ne_id:
            continue
        _try_add_negative_pair(
            context,
            negative_reason_map,
            positive_edge_set,
            left_ne_id,
            right_ne_id,
            "random_hard_negative",
        )

    candidate_items = list(negative_reason_map.items())
    if max_candidate_count > 0 and len(candidate_items) > max_candidate_count:
        candidate_items = rng.sample(candidate_items, max_candidate_count)

    candidate_samples = [
        _build_sample(context, left_ne_id, right_ne_id, 0, reasons, "candidate")
        for (left_ne_id, right_ne_id), reasons in candidate_items
    ]
    candidate_samples.sort(key=lambda item: item["sample_id"])
    return candidate_samples


def summarize_samples(samples):
    label_counter = Counter()
    reason_counter = Counter()
    feature_names = set()
    ordered_site_pair_keys = set()
    unordered_site_pair_keys = set()

    for item in samples:
        label_counter[item.get("label", 0)] += 1
        ordered_site_pair_keys.add(item.get("ordered_site_pair_key", ""))
        unordered_site_pair_keys.add(item.get("unordered_site_pair_key", ""))
        for reason in item.get("candidate_reasons", []):
            reason_counter[reason] += 1
        feature_names.update((item.get("features") or {}).keys())

    return {
        "sample_count": len(samples),
        "positive_count": label_counter.get(1, 0),
        "negative_count": label_counter.get(0, 0),
        "ordered_site_pair_count": len([key for key in ordered_site_pair_keys if key]),
        "unordered_site_pair_count": len([key for key in unordered_site_pair_keys if key]),
        "candidate_reason_counts": dict(sorted(reason_counter.items())),
        "feature_name_count": len(feature_names),
        "feature_names": sorted(feature_names),
    }


def split_samples_by_group(
    samples,
    group_field="unordered_site_pair_key",
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    show_progress=False,
    progress_label="拆分样本",
):
    total_ratio = float(train_ratio) + float(valid_ratio) + float(test_ratio)
    if total_ratio <= 0:
        raise ValueError("train/valid/test ratio 之和必须大于 0")

    train_boundary = float(train_ratio) / total_ratio
    valid_boundary = (float(train_ratio) + float(valid_ratio)) / total_ratio

    split_buckets = {
        "train": [],
        "valid": [],
        "test": [],
    }
    progress = _create_progress_bar(len(samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(samples, start=1):
            group_key = normalize_text(sample.get(group_field, "")) or sample.get("sample_id", "")
            bucket_value = stable_hash_fraction(group_key, seed=seed)
            if bucket_value < train_boundary:
                split_buckets["train"].append(sample)
            elif bucket_value < valid_boundary:
                split_buckets["valid"].append(sample)
            else:
                split_buckets["test"].append(sample)

            if progress is not None:
                progress.set(index)
                progress.set_extra_text(
                    f"train={len(split_buckets['train'])}, valid={len(split_buckets['valid'])}, test={len(split_buckets['test'])}"
                )
    finally:
        _close_progress_bar(progress)

    return split_buckets


def load_dataset_samples(dataset_file):
    raw_items = read_jsonl(dataset_file)
    samples = []
    for raw_item in raw_items:
        features = raw_item.get("features")
        if not isinstance(features, dict):
            continue
        meta = {
            str(key): value
            for key, value in raw_item.items()
            if key not in {"sample_id", "label", "features"}
        }
        samples.append(
            {
                "sample_id": raw_item.get("sample_id", ""),
                "label": int(raw_item.get("label", 0) or 0),
                "features": {str(key): float(value or 0.0) for key, value in features.items()},
                "meta": meta,
            }
        )
    return samples


def infer_feature_names(samples):
    feature_names = set()
    for sample in samples:
        feature_names.update(sample["features"].keys())
    return sorted(feature_names)


def fit_standardizer(samples, feature_names):
    means = {}
    stds = {}
    sample_count = max(1, len(samples))

    for feature_name in feature_names:
        total = 0.0
        for sample in samples:
            total += sample["features"].get(feature_name, 0.0)
        mean = total / float(sample_count)
        means[feature_name] = mean

        variance_sum = 0.0
        for sample in samples:
            diff = sample["features"].get(feature_name, 0.0) - mean
            variance_sum += diff * diff
        std = math.sqrt(variance_sum / float(sample_count))
        stds[feature_name] = std if std > 1e-12 else 1.0

    return {
        "means": means,
        "stds": stds,
    }


def vectorize_samples(samples, feature_names, standardizer, show_progress=False, progress_label="向量化样本"):
    means = standardizer["means"]
    stds = standardizer["stds"]
    dense_samples = []
    progress = _create_progress_bar(len(samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(samples, start=1):
            dense_vector = [
                (sample["features"].get(feature_name, 0.0) - means[feature_name]) / stds[feature_name]
                for feature_name in feature_names
            ]
            dense_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "label": sample["label"],
                    "x": dense_vector,
                    "meta": sample["meta"],
                }
            )
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    return dense_samples


def _sigmoid(value):
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def predict_probability(weights, bias, dense_vector):
    score = bias
    for weight, feature_value in zip(weights, dense_vector):
        score += weight * feature_value
    return _sigmoid(score)


def compute_log_loss(labels, probabilities):
    if not labels:
        return 0.0
    total = 0.0
    for label, probability in zip(labels, probabilities):
        clipped_probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total += (
            -math.log(clipped_probability)
            if label == 1
            else -math.log(1.0 - clipped_probability)
        )
    return total / float(len(labels))


def compute_binary_metrics(labels, probabilities, threshold=0.5):
    tp = fp = tn = fn = 0
    for label, probability in zip(labels, probabilities):
        predicted = 1 if probability >= threshold else 0
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 0 and predicted == 1:
            fp += 1
        elif label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    accuracy = safe_ratio(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "log_loss": compute_log_loss(labels, probabilities),
    }


def compute_roc_auc(labels, probabilities):
    paired = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0

    rank_sum = 0.0
    current_rank = 1
    index = 0
    while index < len(paired):
        next_index = index
        while next_index < len(paired) and paired[next_index][0] == paired[index][0]:
            next_index += 1
        avg_rank = (current_rank + next_index) / 2.0
        positive_in_group = sum(label for _, label in paired[index:next_index])
        rank_sum += avg_rank * positive_in_group
        current_rank = next_index + 1
        index = next_index

    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / float(positive_count * negative_count)


def compute_average_precision(labels, probabilities):
    paired = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(paired, start=1):
        if label == 1:
            hit_count += 1
            precision_sum += hit_count / float(index)
    return precision_sum / float(positive_count)


def evaluate_dense_samples(
    dense_samples,
    weights,
    bias,
    threshold=0.5,
    show_progress=False,
    progress_label="评估样本",
):
    labels = [sample["label"] for sample in dense_samples]
    probabilities = []
    progress = _create_progress_bar(len(dense_samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(dense_samples, start=1):
            probabilities.append(predict_probability(weights, bias, sample["x"]))
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metrics["roc_auc"] = compute_roc_auc(labels, probabilities)
    metrics["average_precision"] = compute_average_precision(labels, probabilities)
    return metrics, probabilities


def choose_best_threshold(
    dense_samples,
    weights,
    bias,
    thresholds=None,
    show_progress=False,
    progress_label="搜索最佳阈值",
):
    if thresholds is None:
        thresholds = [step / 100.0 for step in range(10, 91)]

    best_threshold = 0.5
    best_metrics = None
    best_probabilities = None
    progress = _create_progress_bar(len(thresholds), progress_label, show_progress)
    try:
        for index, threshold in enumerate(thresholds, start=1):
            metrics, probabilities = evaluate_dense_samples(dense_samples, weights, bias, threshold=threshold)
            if best_metrics is None:
                best_threshold = threshold
                best_metrics = metrics
                best_probabilities = probabilities
            elif metrics["f1"] > best_metrics["f1"] + 1e-12:
                best_threshold = threshold
                best_metrics = metrics
                best_probabilities = probabilities
            elif abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and metrics["recall"] > best_metrics["recall"]:
                best_threshold = threshold
                best_metrics = metrics
                best_probabilities = probabilities

            if progress is not None and best_metrics is not None:
                progress.set(index)
                progress.set_extra_text(
                    f"best={best_threshold:.2f}, best_f1={best_metrics['f1']:.4f}"
                )
    finally:
        _close_progress_bar(progress)

    return best_threshold, best_metrics, best_probabilities


def train_logistic_regression(
    train_dense_samples,
    valid_dense_samples=None,
    epochs=20,
    learning_rate=0.03,
    l2=1e-4,
    positive_weight=None,
    seed=42,
    early_stop_patience=5,
    show_progress=False,
    progress_label="训练模型",
):
    if not train_dense_samples:
        raise ValueError("训练集为空")

    rng = random.Random(seed)
    feature_dim = len(train_dense_samples[0]["x"])
    weights = [0.0] * feature_dim
    bias = 0.0

    positive_count = sum(sample["label"] for sample in train_dense_samples)
    negative_count = len(train_dense_samples) - positive_count
    if positive_weight is None:
        positive_weight = safe_ratio(negative_count, positive_count) if positive_count else 1.0
        positive_weight = max(1.0, positive_weight)

    best_state = None
    history = []
    best_score = None
    stale_epochs = 0

    progress = _create_progress_bar(int(epochs), progress_label, show_progress)
    try:
        for epoch in range(1, int(epochs) + 1):
            epoch_samples = list(train_dense_samples)
            rng.shuffle(epoch_samples)

            for sample in epoch_samples:
                label = sample["label"]
                dense_vector = sample["x"]
                probability = predict_probability(weights, bias, dense_vector)
                sample_weight = positive_weight if label == 1 else 1.0
                error = (probability - label) * sample_weight

                for index, feature_value in enumerate(dense_vector):
                    gradient = error * feature_value + l2 * weights[index]
                    weights[index] -= learning_rate * gradient
                bias -= learning_rate * error

            train_metrics, _ = evaluate_dense_samples(train_dense_samples, weights, bias, threshold=0.5)
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
            }

            valid_metrics = None
            valid_threshold = None
            if valid_dense_samples:
                valid_threshold, valid_metrics, _ = choose_best_threshold(valid_dense_samples, weights, bias)
                epoch_record["valid"] = valid_metrics
                epoch_record["valid"]["best_threshold"] = valid_threshold
                current_score = (valid_metrics["f1"], valid_metrics["average_precision"], -valid_metrics["log_loss"])
            else:
                current_score = (train_metrics["f1"], train_metrics["average_precision"], -train_metrics["log_loss"])

            history.append(epoch_record)

            if best_score is None or current_score > best_score:
                best_score = current_score
                best_state = {
                    "weights": list(weights),
                    "bias": bias,
                    "epoch": epoch,
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if valid_dense_samples and stale_epochs >= int(early_stop_patience):
                    if progress is not None:
                        progress.set(epoch)
                        progress.set_extra_text(
                            f"train_f1={train_metrics['f1']:.4f}, valid_f1={valid_metrics['f1']:.4f}, early_stop"
                        )
                    break

            if progress is not None:
                progress.set(epoch)
                extra_text = f"train_f1={train_metrics['f1']:.4f}"
                if valid_metrics is not None:
                    extra_text += f", valid_f1={valid_metrics['f1']:.4f}, best_thr={valid_threshold:.2f}"
                progress.set_extra_text(extra_text)
    finally:
        _close_progress_bar(progress)

    if best_state is None:
        best_state = {
            "weights": list(weights),
            "bias": bias,
            "epoch": len(history),
        }

    return {
        "weights": best_state["weights"],
        "bias": best_state["bias"],
        "best_epoch": best_state["epoch"],
        "positive_weight": positive_weight,
        "history": history,
    }


def build_feature_importance(feature_names, weights, top_k=50):
    importance = [
        {
            "feature": feature_name,
            "weight": weight,
            "abs_weight": abs(weight),
        }
        for feature_name, weight in zip(feature_names, weights)
    ]
    importance.sort(key=lambda item: (-item["abs_weight"], item["feature"]))
    return importance[:top_k]


def build_prediction_rows(
    dense_samples,
    probabilities,
    threshold,
    show_progress=False,
    progress_label="生成预测结果",
):
    prediction_rows = []
    progress = _create_progress_bar(len(dense_samples), progress_label, show_progress)
    try:
        for index, (sample, probability) in enumerate(zip(dense_samples, probabilities), start=1):
            label = sample["label"]
            predicted = 1 if probability >= threshold else 0
            if predicted == label:
                error_type = ""
            elif predicted == 1:
                error_type = "false_positive"
            else:
                error_type = "false_negative"
            prediction_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "label": label,
                    "probability": probability,
                    "predicted_label": predicted,
                    "error_type": error_type,
                    **sample["meta"],
                }
            )
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    return prediction_rows
