#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from alarm_tools.progress_utils import ProgressBar


RELATION_CLASSES = ("downstream", "upstream", "bidirection", "none")
DOMAIN_BUCKETS = ("DATA", "TRANSMISSION", "RAN", "MICROWAVE", "OTHER", "MISSING")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def safe_log1p(value):
    return math.log1p(value) if value > -1 else 0.0


def stable_hash_fraction(text, seed=42):
    payload = f"{seed}::{text}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def read_jsonl(filepath):
    items = []
    with open(filepath, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if text:
                items.append(json.loads(text))
    return items


def write_jsonl(filepath, items):
    with open(filepath, "w", encoding="utf-8") as file_obj:
        for item in items:
            file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")


def _create_progress_bar(total, label, show_progress):
    if not show_progress:
        return None
    print(f"⏳ {label}...")
    return ProgressBar(total, label)


def _close_progress_bar(progress):
    if progress is not None:
        progress.close()


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


def normalize_domain_bucket(domain):
    text = normalize_text(domain)
    if not text:
        return "MISSING"
    if text in {"DATA", "IP", "IPRAN", "ROUTER", "路由", "数据"}:
        return "DATA"
    if text in {"TRANSMISSION", "传输"}:
        return "TRANSMISSION"
    if text in {"RAN", "WIRELESS", "RADIO", "无线"}:
        return "RAN"
    if text in {"MICROWAVE", "MW", "微波"}:
        return "MICROWAVE"
    return "OTHER"


def _pick_counter_mode(counter, default_value="MISSING"):
    if not counter:
        return default_value
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _counter_key_set(counter):
    return {key for key, value in counter.items() if value > 0}


def _counter_cosine_similarity(left_counter, right_counter):
    keys = _counter_key_set(left_counter) | _counter_key_set(right_counter)
    if not keys:
        return 0.0
    dot = sum(float(left_counter.get(key, 0)) * float(right_counter.get(key, 0)) for key in keys)
    left_norm = sum(float(left_counter.get(key, 0)) ** 2 for key in keys)
    right_norm = sum(float(right_counter.get(key, 0)) ** 2 for key in keys)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _jaccard(left_set, right_set):
    union_size = len(left_set | right_set)
    if union_size == 0:
        return 0.0
    return float(len(left_set & right_set)) / float(union_size)


def _adamic_adar_score(shared_nodes, neighbor_map):
    score = 0.0
    for node_id in shared_nodes:
        degree = len(neighbor_map.get(node_id, set()))
        if degree > 1:
            score += 1.0 / math.log(degree)
    return score


def _resource_allocation_score(shared_nodes, neighbor_map):
    score = 0.0
    for node_id in shared_nodes:
        degree = len(neighbor_map.get(node_id, set()))
        if degree > 0:
            score += 1.0 / float(degree)
    return score


@dataclass(frozen=True)
class SiteInfo:
    site_id: str
    site_name: str
    region_id: str
    city_id: str
    latitude: float | None
    longitude: float | None
    device_counts: Counter
    device_total: int
    dominant_domain: str


@dataclass
class SiteRelationContext:
    site_infos: dict
    site_ids: list
    label_relation_map: dict
    downstream_map: dict
    upstream_map: dict
    bidirectional_map: dict
    undirected_map: dict
    region_to_sites: dict
    dominant_domain_to_sites: dict


def _extract_site_id(raw_site_id, site_info=None):
    if raw_site_id:
        return normalize_text(raw_site_id)
    if isinstance(site_info, dict):
        for key in ("site_id", "siteId", "id"):
            value = normalize_text(site_info.get(key))
            if value:
                return value
    return ""


def _extract_site_name(site_id, site_info):
    if isinstance(site_info, dict):
        for key in ("site_name", "siteName", "name", "label"):
            value = str(site_info.get(key, "") or "").strip()
            if value:
                return value
    return site_id


def _extract_region_id(site_info):
    if not isinstance(site_info, dict):
        return "MISSING"
    for key in ("region_id", "regionId", "province_id", "province", "area_id", "area"):
        value = normalize_text(site_info.get(key))
        if value:
            return value
    return "MISSING"


def _extract_city_id(site_info):
    if not isinstance(site_info, dict):
        return "MISSING"
    for key in ("city_id", "cityId", "city", "地市"):
        value = normalize_text(site_info.get(key))
        if value:
            return value
    return "MISSING"


def _extract_lat_lon(site_info):
    if not isinstance(site_info, dict):
        return None, None
    lat = _parse_float(site_info.get("latitude", site_info.get("lat")))
    lon = _parse_float(site_info.get("longitude", site_info.get("lon", site_info.get("lng"))))
    return lat, lon


def _normalize_device_count_entry(value):
    counts = Counter()
    if isinstance(value, dict):
        candidate = value
        for key in ("domain_counts", "device_counts", "counts", "domains"):
            if isinstance(value.get(key), dict):
                candidate = value[key]
                break
        for raw_domain, raw_count in candidate.items():
            bucket = normalize_domain_bucket(raw_domain)
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts[bucket] += count
    return counts


def load_site_device_counts(site_device_counts_file):
    raw_data = load_json(site_device_counts_file)
    result = {}
    if isinstance(raw_data, dict):
        iterable = raw_data.items()
    elif isinstance(raw_data, list):
        iterable = ((item.get("site_id") if isinstance(item, dict) else "", item) for item in raw_data)
    else:
        iterable = []
    for raw_site_id, value in iterable:
        site_id = _extract_site_id(raw_site_id, value if isinstance(value, dict) else None)
        if not site_id:
            continue
        result[site_id] = _normalize_device_count_entry(value)
    return result


def load_site_infos(site_graph_file, site_device_counts_file):
    site_graph = load_json(site_graph_file)
    device_counts_map = load_site_device_counts(site_device_counts_file)
    site_infos = {}

    graph_items = site_graph.items() if isinstance(site_graph, dict) else []
    for raw_site_id, raw_site_info in graph_items:
        site_id = _extract_site_id(raw_site_id, raw_site_info if isinstance(raw_site_info, dict) else None)
        if not site_id:
            continue
        device_counts = Counter(device_counts_map.get(site_id, Counter()))
        device_total = sum(device_counts.values())
        dominant_domain = _pick_counter_mode(device_counts, default_value="MISSING")
        lat, lon = _extract_lat_lon(raw_site_info)
        site_infos[site_id] = SiteInfo(
            site_id=site_id,
            site_name=_extract_site_name(site_id, raw_site_info),
            region_id=_extract_region_id(raw_site_info),
            city_id=_extract_city_id(raw_site_info),
            latitude=lat,
            longitude=lon,
            device_counts=device_counts,
            device_total=device_total,
            dominant_domain=dominant_domain,
        )

    for site_id, device_counts in device_counts_map.items():
        if site_id in site_infos:
            continue
        device_total = sum(device_counts.values())
        site_infos[site_id] = SiteInfo(
            site_id=site_id,
            site_name=site_id,
            region_id="MISSING",
            city_id="MISSING",
            latitude=None,
            longitude=None,
            device_counts=Counter(device_counts),
            device_total=device_total,
            dominant_domain=_pick_counter_mode(device_counts, default_value="MISSING"),
        )
    return site_infos


def _invert_relation(relation):
    if relation == "downstream":
        return "upstream"
    if relation == "upstream":
        return "downstream"
    return relation


def _normalize_prediction_edge_relation(edge, left_site_id, right_site_id):
    prediction = str(edge.get("prediction", "") or "").strip().lower()
    if prediction in {"bidirectional", "bidirection", "<->"}:
        return "bidirection"

    upstream_site = normalize_text(edge.get("upstream_site"))
    downstream_site = normalize_text(edge.get("downstream_site"))
    if upstream_site and downstream_site:
        if left_site_id == upstream_site and right_site_id == downstream_site:
            return "downstream"
        if left_site_id == downstream_site and right_site_id == upstream_site:
            return "upstream"

    if "->" in prediction:
        left_raw, right_raw = prediction.split("->", 1)
        pred_left = normalize_text(left_raw)
        pred_right = normalize_text(right_raw)
        # In site_pair_order outputs, "A->B" means A is downstream and B is upstream.
        if left_site_id == pred_right and right_site_id == pred_left:
            return "downstream"
        if left_site_id == pred_left and right_site_id == pred_right:
            return "upstream"
    return ""


def _set_relation_pair(relation_map, left_site_id, right_site_id, relation):
    if relation not in RELATION_CLASSES or relation == "none":
        return
    if not left_site_id or not right_site_id or left_site_id == right_site_id:
        return

    if relation == "bidirection":
        relation_map[(left_site_id, right_site_id)] = "bidirection"
        relation_map[(right_site_id, left_site_id)] = "bidirection"
        return

    # Direct bidirectional labels are stronger than one-hop upstream/downstream labels.
    existing = relation_map.get((left_site_id, right_site_id))
    if existing == "bidirection":
        return
    if existing and existing != relation:
        relation_map[(left_site_id, right_site_id)] = "bidirection"
        relation_map[(right_site_id, left_site_id)] = "bidirection"
        return
    relation_map[(left_site_id, right_site_id)] = relation
    relation_map[(right_site_id, left_site_id)] = _invert_relation(relation)


def _load_edge_prediction_relation_map(payload):
    edges = payload.get("edges", []) if isinstance(payload, dict) else payload
    relation_map = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        site_a = normalize_text(edge.get("site_a"))
        site_b = normalize_text(edge.get("site_b"))
        if not site_a or not site_b or site_a == site_b:
            continue
        relation_ab = _normalize_prediction_edge_relation(edge, site_a, site_b)
        _set_relation_pair(relation_map, site_a, site_b, relation_ab)
    return relation_map


def _parse_positive_hop(value):
    try:
        hop = int(value)
    except (TypeError, ValueError):
        return None
    return hop if hop > 0 else None


def _iter_hop_items(raw_hops):
    if isinstance(raw_hops, dict):
        return raw_hops.items()
    if isinstance(raw_hops, (list, tuple, set)):
        return ((site_id, 1) for site_id in raw_hops)
    return ()


def _load_site_chains_relation_map(payload):
    """Load one-hop labels from generate_site_chains.py output.

    Only direct hop==1 upstream/downstream entries are treated as labels. The
    transitive downstream/upstream sets are intentionally ignored as labels.
    """
    raw_sites = payload.get("sites", {}) if isinstance(payload, dict) else {}
    relation_map = {}

    for raw_site_id, raw_info in raw_sites.items():
        site_id = normalize_text(raw_site_id)
        if not site_id or not isinstance(raw_info, dict):
            continue

        for raw_neighbor in raw_info.get("bidirectional_sites", []) or []:
            neighbor = normalize_text(raw_neighbor)
            _set_relation_pair(relation_map, site_id, neighbor, "bidirection")

    for raw_site_id, raw_info in raw_sites.items():
        site_id = normalize_text(raw_site_id)
        if not site_id or not isinstance(raw_info, dict):
            continue

        for raw_downstream, raw_hop in _iter_hop_items(raw_info.get("downstream_site_hops")):
            downstream_site = normalize_text(raw_downstream)
            if _parse_positive_hop(raw_hop) == 1:
                _set_relation_pair(relation_map, site_id, downstream_site, "downstream")

        for raw_upstream, raw_hop in _iter_hop_items(raw_info.get("upstream_site_hops")):
            upstream_site = normalize_text(raw_upstream)
            if _parse_positive_hop(raw_hop) == 1:
                _set_relation_pair(relation_map, site_id, upstream_site, "upstream")

    return relation_map


def load_label_relation_map(label_file):
    payload = load_json(label_file)
    if isinstance(payload, dict) and isinstance(payload.get("sites"), dict):
        return _load_site_chains_relation_map(payload)
    return _load_edge_prediction_relation_map(payload)


def build_site_relation_context_from_relation_map(site_infos, relation_map):
    relation_map = dict(relation_map)
    for left_site_id, right_site_id in relation_map:
        if left_site_id not in site_infos:
            site_infos[left_site_id] = SiteInfo(
                left_site_id, left_site_id, "MISSING", "MISSING", None, None, Counter(), 0, "MISSING"
            )
        if right_site_id not in site_infos:
            site_infos[right_site_id] = SiteInfo(
                right_site_id, right_site_id, "MISSING", "MISSING", None, None, Counter(), 0, "MISSING"
            )

    downstream_map = defaultdict(set)
    upstream_map = defaultdict(set)
    bidirectional_map = defaultdict(set)
    undirected_map = defaultdict(set)
    for (left_site_id, right_site_id), relation in relation_map.items():
        if relation == "downstream":
            downstream_map[left_site_id].add(right_site_id)
            upstream_map[right_site_id].add(left_site_id)
        elif relation == "upstream":
            upstream_map[left_site_id].add(right_site_id)
            downstream_map[right_site_id].add(left_site_id)
        elif relation == "bidirection":
            downstream_map[left_site_id].add(right_site_id)
            upstream_map[left_site_id].add(right_site_id)
            bidirectional_map[left_site_id].add(right_site_id)
        undirected_map[left_site_id].add(right_site_id)

    region_to_sites = defaultdict(list)
    dominant_domain_to_sites = defaultdict(list)
    for site_id, info in site_infos.items():
        region_to_sites[info.region_id].append(site_id)
        dominant_domain_to_sites[info.dominant_domain].append(site_id)

    return SiteRelationContext(
        site_infos=site_infos,
        site_ids=sorted(site_infos),
        label_relation_map=relation_map,
        downstream_map={key: set(value) for key, value in downstream_map.items()},
        upstream_map={key: set(value) for key, value in upstream_map.items()},
        bidirectional_map={key: set(value) for key, value in bidirectional_map.items()},
        undirected_map={key: set(value) for key, value in undirected_map.items()},
        region_to_sites={key: sorted(value) for key, value in region_to_sites.items()},
        dominant_domain_to_sites={key: sorted(value) for key, value in dominant_domain_to_sites.items()},
    )


def build_site_relation_context(label_file, site_graph_file, site_device_counts_file):
    site_infos = load_site_infos(site_graph_file, site_device_counts_file)
    relation_map = load_label_relation_map(label_file)
    return build_site_relation_context_from_relation_map(site_infos, relation_map)


def _neighbors_excluding_pair(neighbor_map, site_id, excluded_site_id):
    return set(neighbor_map.get(site_id, set())) - {excluded_site_id}


def extract_site_relation_features(context, left_site_id, right_site_id):
    left_info = context.site_infos[left_site_id]
    right_info = context.site_infos[right_site_id]

    left_down = _neighbors_excluding_pair(context.downstream_map, left_site_id, right_site_id)
    left_up = _neighbors_excluding_pair(context.upstream_map, left_site_id, right_site_id)
    right_down = _neighbors_excluding_pair(context.downstream_map, right_site_id, left_site_id)
    right_up = _neighbors_excluding_pair(context.upstream_map, right_site_id, left_site_id)
    left_undir = left_down | left_up
    right_undir = right_down | right_up
    common_down = left_down & right_down
    common_up = left_up & right_up
    common_undir = left_undir & right_undir
    mids_left_to_right = left_down & right_up
    mids_right_to_left = right_down & left_up

    geo_distance_km = None
    if (
        left_info.latitude is not None
        and left_info.longitude is not None
        and right_info.latitude is not None
        and right_info.longitude is not None
    ):
        geo_distance_km = haversine_km(
            left_info.latitude,
            left_info.longitude,
            right_info.latitude,
            right_info.longitude,
        )

    left_counts = left_info.device_counts
    right_counts = right_info.device_counts
    left_total = left_info.device_total
    right_total = right_info.device_total

    features = {
        "same_region": float(left_info.region_id != "MISSING" and left_info.region_id == right_info.region_id),
        "same_city": float(left_info.city_id != "MISSING" and left_info.city_id == right_info.city_id),
        "same_dominant_domain": float(
            left_info.dominant_domain != "MISSING" and left_info.dominant_domain == right_info.dominant_domain
        ),
        "geo_distance_km": geo_distance_km or 0.0,
        "geo_distance_missing": float(geo_distance_km is None),
        "geo_distance_log1p": safe_log1p(geo_distance_km or 0.0),
        "left_device_total": float(left_total),
        "right_device_total": float(right_total),
        "device_total_ratio_left_right": safe_ratio(left_total, right_total),
        "device_total_ratio_right_left": safe_ratio(right_total, left_total),
        "device_total_diff_left_minus_right": float(left_total - right_total),
        # Label-graph context below is leave-one-site-pair-out: the current
        # pair is removed before computing any degree/common-neighbor feature.
        "left_down_degree_excl_pair": float(len(left_down)),
        "left_up_degree_excl_pair": float(len(left_up)),
        "right_down_degree_excl_pair": float(len(right_down)),
        "right_up_degree_excl_pair": float(len(right_up)),
        "left_undirected_degree_excl_pair": float(len(left_undir)),
        "right_undirected_degree_excl_pair": float(len(right_undir)),
        "common_down_count_excl_pair": float(len(common_down)),
        "common_up_count_excl_pair": float(len(common_up)),
        "common_neighbor_count_excl_pair": float(len(common_undir)),
        "jaccard_down_excl_pair": _jaccard(left_down, right_down),
        "jaccard_up_excl_pair": _jaccard(left_up, right_up),
        "jaccard_neighbor_excl_pair": _jaccard(left_undir, right_undir),
        "two_hop_left_to_right_count_excl_pair": float(len(mids_left_to_right)),
        "two_hop_right_to_left_count_excl_pair": float(len(mids_right_to_left)),
        "adamic_adar_neighbor_excl_pair": _adamic_adar_score(common_undir, context.undirected_map),
        "resource_allocation_neighbor_excl_pair": _resource_allocation_score(common_undir, context.undirected_map),
        "left_region_missing": float(left_info.region_id == "MISSING"),
        "right_region_missing": float(right_info.region_id == "MISSING"),
        "left_city_missing": float(left_info.city_id == "MISSING"),
        "right_city_missing": float(right_info.city_id == "MISSING"),
        "left_domain_missing": float(left_info.dominant_domain == "MISSING"),
        "right_domain_missing": float(right_info.dominant_domain == "MISSING"),
        "device_domain_cosine_similarity": _counter_cosine_similarity(left_counts, right_counts),
        "device_domain_jaccard": _jaccard(_counter_key_set(left_counts), _counter_key_set(right_counts)),
    }

    for domain in DOMAIN_BUCKETS:
        left_count = left_counts.get(domain, 0)
        right_count = right_counts.get(domain, 0)
        domain_key = domain.lower()
        features[f"left_device_count__{domain_key}"] = float(left_count)
        features[f"right_device_count__{domain_key}"] = float(right_count)
        features[f"left_device_ratio__{domain_key}"] = safe_ratio(left_count, left_total)
        features[f"right_device_ratio__{domain_key}"] = safe_ratio(right_count, right_total)
        features[f"left_has_domain__{domain_key}"] = float(left_count > 0)
        features[f"right_has_domain__{domain_key}"] = float(right_count > 0)
        features[f"left_dominant_domain_is__{domain_key}"] = float(left_info.dominant_domain == domain)
        features[f"right_dominant_domain_is__{domain_key}"] = float(right_info.dominant_domain == domain)

    for left_domain in DOMAIN_BUCKETS:
        for right_domain in DOMAIN_BUCKETS:
            features[f"dominant_domain_pair__{left_domain.lower()}__{right_domain.lower()}"] = float(
                left_info.dominant_domain == left_domain and right_info.dominant_domain == right_domain
            )
    return features


def _make_sample_id(left_site_id, right_site_id):
    return f"{left_site_id}__{right_site_id}"


def _make_split_keys(left_site_id, right_site_id):
    return _make_sample_id(left_site_id, right_site_id), "__".join(sorted([left_site_id, right_site_id]))


def build_relation_sample(context, left_site_id, right_site_id, relation, reasons, sample_role):
    left_info = context.site_infos[left_site_id]
    right_info = context.site_infos[right_site_id]
    ordered_key, unordered_key = _make_split_keys(left_site_id, right_site_id)
    return {
        "sample_id": ordered_key,
        "label": relation,
        "label_id": RELATION_CLASSES.index(relation),
        "sample_role": sample_role,
        "sample_granularity": "site_relation_pair",
        "u_site_id": left_site_id,
        "v_site_id": right_site_id,
        "u_site_name": left_info.site_name,
        "v_site_name": right_info.site_name,
        "u_region_id": left_info.region_id,
        "v_region_id": right_info.region_id,
        "u_domain": left_info.dominant_domain,
        "v_domain": right_info.dominant_domain,
        "ordered_site_pair_key": ordered_key,
        "unordered_site_pair_key": unordered_key,
        "candidate_reasons": sorted(reasons),
        "features": extract_site_relation_features(context, left_site_id, right_site_id),
    }


def rebuild_site_relation_sample_features(samples, context, show_progress=False, progress_label="重算站点关系特征"):
    rebuilt = []
    progress = _create_progress_bar(len(samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(samples, start=1):
            item = dict(sample)
            left_site_id = str(item.get("u_site_id", "") or "")
            right_site_id = str(item.get("v_site_id", "") or "")
            if left_site_id in context.site_infos and right_site_id in context.site_infos:
                item["features"] = extract_site_relation_features(context, left_site_id, right_site_id)
            rebuilt.append(item)
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    return rebuilt


def _has_any_labeled_relation(context, left_site_id, right_site_id):
    return (
        (left_site_id, right_site_id) in context.label_relation_map
        or (right_site_id, left_site_id) in context.label_relation_map
    )


def _try_add_none_pair(context, none_reason_map, left_site_id, right_site_id, reason):
    if not left_site_id or not right_site_id or left_site_id == right_site_id:
        return False
    if left_site_id not in context.site_infos or right_site_id not in context.site_infos:
        return False
    if _has_any_labeled_relation(context, left_site_id, right_site_id):
        return False
    none_reason_map[(left_site_id, right_site_id)].add(reason)
    return True


def _none_pool_reached_target(none_reason_map, target_none_count):
    return target_none_count is not None and target_none_count >= 0 and len(none_reason_map) >= target_none_count


def _deterministic_sample(seq, k, rng):
    seq = list(seq)
    if k <= 0 or not seq:
        return []
    if len(seq) <= k:
        return list(seq)
    return rng.sample(seq, k)


def _nearest_sites_by_distance(context, site_id, candidates, limit):
    site_info = context.site_infos.get(site_id)
    if not site_info or site_info.latitude is None or site_info.longitude is None:
        return []
    scored = []
    for candidate_id in candidates:
        cand_info = context.site_infos.get(candidate_id)
        if not cand_info or cand_info.latitude is None or cand_info.longitude is None:
            continue
        distance = haversine_km(site_info.latitude, site_info.longitude, cand_info.latitude, cand_info.longitude)
        if distance is not None:
            scored.append((distance, candidate_id))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [site_id for _, site_id in scored[:limit]]


def _sample_candidates_for_nearest(candidates, rng, max_size=500):
    candidates = list(candidates)
    if len(candidates) <= max_size:
        return candidates
    return rng.sample(candidates, max_size)


def _iter_none_candidate_attempts(context, left_site_id, right_site_id, rng, same_region_negatives, same_domain_negatives, nearest_negatives):
    left_info = context.site_infos[left_site_id]
    right_info = context.site_infos[right_site_id]
    attempt_types = [
        "same_target_region",
        "same_source_region",
        "same_target_domain",
        "same_source_domain",
        "nearest_to_source",
    ]
    rng.shuffle(attempt_types)

    for attempt_type in attempt_types:
        if attempt_type == "same_target_region":
            if same_region_negatives <= 0 or right_info.region_id == "MISSING":
                continue
            candidates = [
                site_id for site_id in context.region_to_sites.get(right_info.region_id, [])
                if site_id != right_site_id
            ]
            for candidate_id in _deterministic_sample(candidates, same_region_negatives, rng):
                yield left_site_id, candidate_id, "same_target_region"
        elif attempt_type == "same_source_region":
            if same_region_negatives <= 0 or left_info.region_id == "MISSING":
                continue
            candidates = [
                site_id for site_id in context.region_to_sites.get(left_info.region_id, [])
                if site_id != left_site_id
            ]
            for candidate_id in _deterministic_sample(candidates, same_region_negatives, rng):
                yield candidate_id, right_site_id, "same_source_region"
        elif attempt_type == "same_target_domain":
            if same_domain_negatives <= 0 or right_info.dominant_domain == "MISSING":
                continue
            candidates = [
                site_id for site_id in context.dominant_domain_to_sites.get(right_info.dominant_domain, [])
                if site_id != right_site_id
            ]
            for candidate_id in _deterministic_sample(candidates, same_domain_negatives, rng):
                yield left_site_id, candidate_id, "same_target_domain"
        elif attempt_type == "same_source_domain":
            if same_domain_negatives <= 0 or left_info.dominant_domain == "MISSING":
                continue
            candidates = [
                site_id for site_id in context.dominant_domain_to_sites.get(left_info.dominant_domain, [])
                if site_id != left_site_id
            ]
            for candidate_id in _deterministic_sample(candidates, same_domain_negatives, rng):
                yield candidate_id, right_site_id, "same_source_domain"
        elif attempt_type == "nearest_to_source":
            if nearest_negatives <= 0:
                continue
            candidates = _sample_candidates_for_nearest(
                (site_id for site_id in context.site_ids if site_id != left_site_id),
                rng,
            )
            for candidate_id in _nearest_sites_by_distance(context, left_site_id, candidates, nearest_negatives):
                yield left_site_id, candidate_id, "nearest_to_source"


def _generate_none_relation_pool(
    context,
    positive_ordered_pairs,
    rng,
    same_region_negatives=1,
    same_domain_negatives=1,
    nearest_negatives=1,
    random_negative_ratio=1.0,
    target_none_count=None,
    max_rounds=3,
    show_progress=False,
):
    none_reason_map = defaultdict(set)
    ordered_pairs = list(positive_ordered_pairs)
    max_rounds = max(1, int(max_rounds)) if target_none_count is not None else 1
    progress = _create_progress_bar(max_rounds * max(1, len(ordered_pairs)), "构造 none 关系候选", show_progress)
    progress_index = 0
    try:
        for round_index in range(max_rounds):
            shuffled_pairs = list(ordered_pairs)
            rng.shuffle(shuffled_pairs)
            added_this_round = 0
            for left_site_id, right_site_id in shuffled_pairs:
                if _none_pool_reached_target(none_reason_map, target_none_count):
                    break
                before_count = len(none_reason_map)
                for cand_left, cand_right, reason in _iter_none_candidate_attempts(
                    context,
                    left_site_id,
                    right_site_id,
                    rng,
                    same_region_negatives,
                    same_domain_negatives,
                    nearest_negatives,
                ):
                    if _try_add_none_pair(context, none_reason_map, cand_left, cand_right, reason):
                        added_this_round += 1
                        break
                if len(none_reason_map) == before_count:
                    continue
                if progress is not None:
                    progress_index += 1
                    progress.set(min(progress_index, max_rounds * max(1, len(ordered_pairs))))
                    progress.set_extra_text(
                        f"round={round_index + 1}, none候选={len(none_reason_map)}"
                    )
            if _none_pool_reached_target(none_reason_map, target_none_count) or added_this_round == 0:
                break
    finally:
        _close_progress_bar(progress)

    if target_none_count is None:
        target_random_count = int(math.ceil(len(positive_ordered_pairs) * max(0.0, random_negative_ratio)))
    else:
        target_random_count = max(0, target_none_count)
    attempts = 0
    max_attempts = max(2000, target_random_count * 50)
    while len(none_reason_map) < target_random_count and attempts < max_attempts and context.site_ids:
        attempts += 1
        left_site_id = rng.choice(context.site_ids)
        right_site_id = rng.choice(context.site_ids)
        _try_add_none_pair(context, none_reason_map, left_site_id, right_site_id, "random_none")
    return none_reason_map


def generate_relation_learning_samples(
    context,
    seed=42,
    none_per_positive=2.0,
    same_region_negatives=1,
    same_domain_negatives=1,
    nearest_negatives=1,
    random_negative_ratio=1.0,
    none_max_rounds=3,
    show_progress=False,
):
    rng = random.Random(seed)
    positive_pairs = sorted(context.label_relation_map)
    target_none_count = int(math.ceil(len(positive_pairs) * max(0.0, none_per_positive)))
    none_reason_map = _generate_none_relation_pool(
        context,
        positive_pairs,
        rng,
        same_region_negatives=same_region_negatives,
        same_domain_negatives=same_domain_negatives,
        nearest_negatives=nearest_negatives,
        random_negative_ratio=random_negative_ratio,
        target_none_count=target_none_count,
        max_rounds=none_max_rounds,
        show_progress=show_progress,
    )
    none_items = list(none_reason_map.items())
    if len(none_items) > target_none_count:
        none_items = rng.sample(none_items, target_none_count)

    samples = []
    progress = _create_progress_bar(len(positive_pairs) + len(none_items), "构造关系样本特征", show_progress)
    try:
        index = 0
        for left_site_id, right_site_id in positive_pairs:
            index += 1
            samples.append(
                build_relation_sample(
                    context,
                    left_site_id,
                    right_site_id,
                    context.label_relation_map[(left_site_id, right_site_id)],
                    {"labeled_site_relation"},
                    "positive",
                )
            )
            if progress is not None:
                progress.set(index)
        for (left_site_id, right_site_id), reasons in none_items:
            index += 1
            samples.append(
                build_relation_sample(context, left_site_id, right_site_id, "none", reasons, "none")
            )
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    rng.shuffle(samples)
    return samples


def summarize_samples(samples):
    label_counter = Counter(sample.get("label", "") for sample in samples)
    feature_names = set()
    for sample in samples:
        feature_names.update((sample.get("features") or {}).keys())
    return {
        "sample_count": len(samples),
        "label_counts": dict(sorted(label_counter.items())),
        "feature_name_count": len(feature_names),
        "feature_names": sorted(feature_names),
    }


def split_samples_by_group(samples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    train_boundary = train_ratio
    valid_boundary = train_ratio + valid_ratio
    buckets = {"train": [], "valid": [], "test": []}
    for sample in samples:
        group_key = sample.get("unordered_site_pair_key") or sample.get("sample_id", "")
        value = stable_hash_fraction(group_key, seed=seed)
        if value < train_boundary:
            buckets["train"].append(sample)
        elif value < valid_boundary:
            buckets["valid"].append(sample)
        else:
            buckets["test"].append(sample)
    return buckets


def load_dataset_samples(filepath):
    samples = read_jsonl(filepath)
    normalized = []
    for item in samples:
        features = item.get("features")
        if not isinstance(features, dict):
            continue
        sample = {
            key: value
            for key, value in item.items()
            if key not in {"features"}
        }
        label = str(sample.get("label", "none") or "none")
        if label not in RELATION_CLASSES:
            label = "none"
        sample["label"] = label
        sample["label_id"] = RELATION_CLASSES.index(label)
        sample["features"] = {str(key): float(value or 0.0) for key, value in features.items()}
        normalized.append(sample)
    return normalized


def infer_feature_names(samples):
    feature_names = set()
    for sample in samples:
        feature_names.update(sample["features"].keys())
    return sorted(feature_names)


def fit_standardizer(samples, feature_names, show_progress=False, progress_label="拟合标准化参数"):
    means = {}
    stds = {}
    count = max(1, len(samples))
    feature_set = set(feature_names)
    sums = {feature_name: 0.0 for feature_name in feature_names}
    sumsq = {feature_name: 0.0 for feature_name in feature_names}
    progress = _create_progress_bar(len(samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(samples, start=1):
            features = sample.get("features", {})
            for feature_name, raw_value in features.items():
                if feature_name not in feature_set:
                    continue
                value = float(raw_value or 0.0)
                sums[feature_name] += value
                sumsq[feature_name] += value * value
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)

    for feature_name in feature_names:
        mean = sums[feature_name] / count
        variance = (sumsq[feature_name] / count) - mean * mean
        if variance < 0 and abs(variance) < 1e-12:
            variance = 0.0
        means[feature_name] = mean
        stds[feature_name] = math.sqrt(variance) if variance > 1e-12 else 1.0
    return {"means": means, "stds": stds}


def vectorize_samples(samples, feature_names, standardizer, show_progress=False, progress_label="向量化样本"):
    means = standardizer["means"]
    stds = standardizer["stds"]
    dense_samples = []
    progress = _create_progress_bar(len(samples), progress_label, show_progress)
    try:
        for index, sample in enumerate(samples, start=1):
            dense_samples.append({
                "x": [
                    (sample["features"].get(feature_name, 0.0) - means.get(feature_name, 0.0))
                    / stds.get(feature_name, 1.0)
                    for feature_name in feature_names
                ],
                "y": int(sample.get("label_id", RELATION_CLASSES.index(sample.get("label", "none")))),
                "sample": sample,
            })
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)
    return dense_samples


def _as_weight_matrix(weights):
    """把 list-of-lists 或 ndarray 统一成 (class_count, feature_dim) ndarray。"""
    return np.asarray(weights, dtype=np.float64)


def _as_bias_vector(biases):
    return np.asarray(biases, dtype=np.float64)


def _softmax(scores):
    """单样本 softmax；保留给少量旧调用者。优先使用 _softmax_batch。"""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - arr.max()
    exps = np.exp(arr)
    total = float(exps.sum()) or 1.0
    return [float(value / total) for value in exps]


def _softmax_batch(scores):
    """对 (n, c) 矩阵做行级 softmax，返回 (n, c) ndarray。"""
    scores = scores - scores.max(axis=1, keepdims=True)
    exps = np.exp(scores)
    return exps / np.maximum(exps.sum(axis=1, keepdims=True), 1e-300)


def predict_probabilities(weights, biases, dense_vector):
    """单样本预测；返回 list[float]，长度为 class_count。

    旧接口保留以兼容外部调用者。性能敏感场景请用 predict_probabilities_batch。
    """
    x = np.asarray(dense_vector, dtype=np.float64)
    return predict_probabilities_batch(weights, biases, x.reshape(1, -1))[0].tolist()


def predict_probabilities_batch(weights, biases, dense_matrix):
    """批量预测；支持 softmax 权重矩阵、MLP 参数 dict 或 GBDT Booster。"""
    X = np.asarray(dense_matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if isinstance(weights, dict) and weights.get("model_type") == "mlp":
        W1 = np.asarray(weights["hidden_weights"], dtype=np.float64)
        b1 = np.asarray(weights["hidden_biases"], dtype=np.float64)
        W2 = np.asarray(weights["output_weights"], dtype=np.float64)
        b2 = np.asarray(weights["output_biases"], dtype=np.float64)
        hidden = np.maximum(0.0, X @ W1.T + b1)
        scores = hidden @ W2.T + b2
        return _softmax_batch(scores)
    if isinstance(weights, dict) and weights.get("model_type") == "gbdt":
        booster = weights.get("booster")
        if booster is None:
            # 兜底：从 model_string 临时构建，但不再回写到 weights，避免污染调用方 dict。
            # 高频调用路径下推荐在加载时（_load_model 或 train_*）就把 booster 放进 dict。
            if "model_string" not in weights:
                raise ValueError("GBDT 权重 dict 既无 booster 也无 model_string")
            import lightgbm as lgb

            booster = lgb.Booster(model_str=weights["model_string"])
        best_iteration = int(weights.get("best_iteration") or 0)
        probs = np.asarray(
            booster.predict(X, num_iteration=best_iteration if best_iteration > 0 else None),
            dtype=np.float64,
        )
        if probs.ndim == 1:
            probs = probs.reshape(-1, len(RELATION_CLASSES))
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-300)
    W = _as_weight_matrix(weights)
    b = _as_bias_vector(biases)
    scores = X @ W.T + b
    return _softmax_batch(scores)


def _stack_dense_samples(dense_samples):
    """把 dense_samples（list of dict with x,y）拼成 (X, y) ndarray。"""
    X = np.asarray([item["x"] for item in dense_samples], dtype=np.float64)
    y = np.asarray([item["y"] for item in dense_samples], dtype=np.int64)
    return X, y


def _compute_metrics_from_confusion(confusion):
    """从 (c, c) ndarray confusion 矩阵计算 accuracy / macro_f1 / per_class。"""
    confusion_list = [[int(v) for v in row] for row in confusion]
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class = {}
    macro_f1 = 0.0
    macro_class_count = 0
    class_count = len(RELATION_CLASSES)
    for idx, label in enumerate(RELATION_CLASSES):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        support = int(confusion[idx, :].sum())
        if support > 0:
            macro_f1 += f1
            macro_class_count += 1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = safe_ratio(macro_f1, macro_class_count)
    return {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_list,
        "classes": list(RELATION_CLASSES),
        "per_class": per_class,
        "sample_count": total,
    }


def evaluate_dense_samples(dense_samples, weights, biases):
    """批量评估；返回 (metrics_dict, probabilities_list_of_list)。"""
    class_count = len(RELATION_CLASSES)
    if not dense_samples:
        empty_confusion = np.zeros((class_count, class_count), dtype=np.int64)
        return _compute_metrics_from_confusion(empty_confusion), []

    X, y = _stack_dense_samples(dense_samples)
    probs = predict_probabilities_batch(weights, biases, X)
    preds = probs.argmax(axis=1)

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (y, preds), 1)
    metrics = _compute_metrics_from_confusion(confusion)
    # 转回 Python list 保持旧接口兼容
    probabilities = probs.tolist()
    return metrics, probabilities


def _evaluate_dense_matrix(X, y, weights, biases):
    """在已拼好的 (X, y) ndarray 上跑批量评估，避免反复构造矩阵。"""
    class_count = len(RELATION_CLASSES)
    if X.size == 0:
        return _compute_metrics_from_confusion(np.zeros((class_count, class_count), dtype=np.int64))
    probs = predict_probabilities_batch(weights, biases, X)
    preds = probs.argmax(axis=1)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (y, preds), 1)
    return _compute_metrics_from_confusion(confusion)


def _compute_class_weight_vector(y_train, class_weight):
    class_count = len(RELATION_CLASSES)
    n_train = int(len(y_train))
    label_counter = Counter(int(label) for label in y_train.tolist())
    if class_weight == "balanced":
        class_weights = {
            idx: safe_ratio(n_train, class_count * label_counter.get(idx, 0))
            for idx in range(class_count)
        }
    else:
        class_weights = {idx: 1.0 for idx in range(class_count)}
    class_weight_vec = np.asarray(
        [class_weights[idx] for idx in range(class_count)],
        dtype=np.float64,
    )
    return class_weights, class_weight_vec


def train_softmax_regression(
    train_dense_samples,
    valid_dense_samples=None,
    epochs=30,
    learning_rate=0.03,
    l2=1e-4,
    class_weight="balanced",
    early_stop_patience=5,
    early_stop_min_delta=1e-8,
    seed=42,
    show_progress=False,
    batch_size=512,
):
    """Softmax 多分类 SGD 训练（mini-batch + numpy 向量化）。

    与原 per-sample SGD 的差异：
    - 批量梯度采用 mean over batch（标准 mini-batch SGD 语义），等效学习率与
      pure SGD 保持一致；
    - L2 在每个 mini-batch 应用一次（标准 weight decay 语义）。
    """
    if not train_dense_samples:
        raise ValueError("训练集为空")

    feature_dim = len(train_dense_samples[0]["x"])
    class_count = len(RELATION_CLASSES)
    batch_size = max(1, int(batch_size))

    X_train, y_train = _stack_dense_samples(train_dense_samples)
    n_train = X_train.shape[0]

    class_weights, class_weight_vec = _compute_class_weight_vector(y_train, class_weight)
    sample_weights = class_weight_vec[y_train]  # (n,)

    if valid_dense_samples:
        X_valid, y_valid = _stack_dense_samples(valid_dense_samples)
    else:
        X_valid = y_valid = None

    weights = np.zeros((class_count, feature_dim), dtype=np.float64)
    biases = np.zeros(class_count, dtype=np.float64)

    rng = np.random.default_rng(seed)
    eye = np.eye(class_count, dtype=np.float64)

    best_state = None
    best_score = -1.0
    no_improve_epochs = 0
    stopped_epoch = epochs
    history = []
    progress = _create_progress_bar(epochs, "训练 softmax 多分类模型", show_progress)
    try:
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n_train)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]
            w_shuf = sample_weights[perm]

            for start in range(0, n_train, batch_size):
                end = start + batch_size
                xb = X_shuf[start:end]            # (b, d)
                yb = y_shuf[start:end]            # (b,)
                wb = w_shuf[start:end]            # (b,)
                b = xb.shape[0]
                if b == 0:
                    continue

                scores = xb @ weights.T + biases  # (b, c)
                probs = _softmax_batch(scores)    # (b, c)
                one_hot = eye[yb]                 # (b, c)
                # 标准 cross-entropy 梯度，按类权重加权
                error = (probs - one_hot) * wb[:, None]  # (b, c)
                # mean-gradient mini-batch
                grad_b = error.mean(axis=0)
                grad_w = error.T @ xb / b + l2 * weights
                biases -= learning_rate * grad_b
                weights -= learning_rate * grad_w

            train_metrics = _evaluate_dense_matrix(X_train, y_train, weights, biases)
            valid_metrics = None
            score = train_metrics["macro_f1"]
            if X_valid is not None:
                valid_metrics = _evaluate_dense_matrix(X_valid, y_valid, weights, biases)
                score = valid_metrics["macro_f1"]
            history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
            if score > best_score + early_stop_min_delta:
                best_score = score
                no_improve_epochs = 0
                best_state = {
                    "weights": [list(row) for row in weights.tolist()],
                    "biases": list(biases.tolist()),
                    "best_epoch": epoch,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                }
            else:
                no_improve_epochs += 1
            if progress is not None:
                progress.set(epoch)
                progress.set_extra_text(f"macro_f1={score:.4f}")
            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                stopped_epoch = epoch
                break
    finally:
        _close_progress_bar(progress)

    if best_state is None:
        # 极端情况下（epochs <= 0）保底返回当前权重
        best_state = {
            "weights": [list(row) for row in weights.tolist()],
            "biases": list(biases.tolist()),
            "best_epoch": 0,
            "train_metrics": _evaluate_dense_matrix(X_train, y_train, weights, biases),
            "valid_metrics": (
                _evaluate_dense_matrix(X_valid, y_valid, weights, biases)
                if X_valid is not None else None
            ),
        }

    best_state["history"] = history
    best_state["class_weights"] = class_weights
    best_state["stopped_epoch"] = stopped_epoch
    best_state["early_stop_patience"] = early_stop_patience
    best_state["early_stop_min_delta"] = early_stop_min_delta
    best_state["batch_size"] = batch_size
    return best_state


def _mlp_param_dict(hidden_weights, hidden_biases, output_weights, output_biases):
    return {
        "model_type": "mlp",
        "hidden_weights": [list(row) for row in hidden_weights.tolist()],
        "hidden_biases": list(hidden_biases.tolist()),
        "output_weights": [list(row) for row in output_weights.tolist()],
        "output_biases": list(output_biases.tolist()),
    }


def train_mlp_classifier(
    train_dense_samples,
    valid_dense_samples=None,
    epochs=50,
    learning_rate=0.01,
    l2=1e-4,
    class_weight="balanced",
    early_stop_patience=8,
    early_stop_min_delta=1e-8,
    seed=42,
    show_progress=False,
    batch_size=512,
    hidden_dim=64,
):
    """单隐层 ReLU MLP，用于学习简单线性模型无法表达的特征交互。"""
    if not train_dense_samples:
        raise ValueError("训练集为空")

    feature_dim = len(train_dense_samples[0]["x"])
    class_count = len(RELATION_CLASSES)
    batch_size = max(1, int(batch_size))
    hidden_dim = max(1, int(hidden_dim))

    X_train, y_train = _stack_dense_samples(train_dense_samples)
    n_train = X_train.shape[0]
    class_weights, class_weight_vec = _compute_class_weight_vector(y_train, class_weight)
    sample_weights = class_weight_vec[y_train]

    if valid_dense_samples:
        X_valid, y_valid = _stack_dense_samples(valid_dense_samples)
    else:
        X_valid = y_valid = None

    rng = np.random.default_rng(seed)
    hidden_weights = rng.normal(0.0, math.sqrt(2.0 / max(1, feature_dim)), size=(hidden_dim, feature_dim))
    hidden_biases = np.zeros(hidden_dim, dtype=np.float64)
    output_weights = rng.normal(0.0, math.sqrt(2.0 / max(1, hidden_dim)), size=(class_count, hidden_dim))
    output_biases = np.zeros(class_count, dtype=np.float64)
    eye = np.eye(class_count, dtype=np.float64)

    best_state = None
    best_score = -1.0
    no_improve_epochs = 0
    stopped_epoch = epochs
    history = []
    progress = _create_progress_bar(epochs, "训练 MLP 多分类模型", show_progress)
    try:
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n_train)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]
            w_shuf = sample_weights[perm]

            for start in range(0, n_train, batch_size):
                end = start + batch_size
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                wb = w_shuf[start:end]
                b = xb.shape[0]
                if b == 0:
                    continue

                hidden_pre = xb @ hidden_weights.T + hidden_biases
                hidden = np.maximum(0.0, hidden_pre)
                scores = hidden @ output_weights.T + output_biases
                probs = _softmax_batch(scores)
                one_hot = eye[yb]
                error = (probs - one_hot) * wb[:, None]

                grad_output_biases = error.mean(axis=0)
                grad_output_weights = error.T @ hidden / b + l2 * output_weights
                hidden_error = error @ output_weights
                hidden_error *= hidden_pre > 0.0
                grad_hidden_biases = hidden_error.mean(axis=0)
                grad_hidden_weights = hidden_error.T @ xb / b + l2 * hidden_weights

                output_biases -= learning_rate * grad_output_biases
                output_weights -= learning_rate * grad_output_weights
                hidden_biases -= learning_rate * grad_hidden_biases
                hidden_weights -= learning_rate * grad_hidden_weights

            params = _mlp_param_dict(hidden_weights, hidden_biases, output_weights, output_biases)
            train_metrics = _evaluate_dense_matrix(X_train, y_train, params, None)
            valid_metrics = None
            score = train_metrics["macro_f1"]
            if X_valid is not None:
                valid_metrics = _evaluate_dense_matrix(X_valid, y_valid, params, None)
                score = valid_metrics["macro_f1"]
            history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
            if score > best_score + early_stop_min_delta:
                best_score = score
                no_improve_epochs = 0
                best_state = {
                    **params,
                    "best_epoch": epoch,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                }
            else:
                no_improve_epochs += 1
            if progress is not None:
                progress.set(epoch)
                progress.set_extra_text(f"macro_f1={score:.4f}")
            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                stopped_epoch = epoch
                break
    finally:
        _close_progress_bar(progress)

    if best_state is None:
        params_fallback = _mlp_param_dict(hidden_weights, hidden_biases, output_weights, output_biases)
        best_state = {
            **params_fallback,
            "best_epoch": 0,
            "train_metrics": _evaluate_dense_matrix(X_train, y_train, params_fallback, None),
            "valid_metrics": (
                _evaluate_dense_matrix(X_valid, y_valid, params_fallback, None)
                if X_valid is not None else None
            ),
        }

    best_state["history"] = history
    best_state["class_weights"] = class_weights
    best_state["stopped_epoch"] = stopped_epoch
    best_state["early_stop_patience"] = early_stop_patience
    best_state["early_stop_min_delta"] = early_stop_min_delta
    best_state["batch_size"] = batch_size
    best_state["hidden_dim"] = hidden_dim
    return best_state


def train_gbdt_classifier(
    train_dense_samples,
    valid_dense_samples=None,
    epochs=200,
    learning_rate=0.05,
    l2=1e-4,
    class_weight="balanced",
    early_stop_patience=20,
    seed=42,
    show_progress=False,
    num_leaves=31,
    min_data_in_leaf=20,
):
    """LightGBM GBDT 多分类模型，适合学习表格特征的非线性交互。"""
    if not train_dense_samples:
        raise ValueError("训练集为空")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("使用 --model-type gbdt 需要安装 lightgbm: pip install lightgbm") from exc

    X_train, y_train = _stack_dense_samples(train_dense_samples)
    class_weights, class_weight_vec = _compute_class_weight_vector(y_train, class_weight)
    sample_weights = class_weight_vec[y_train]
    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weights, free_raw_data=False)

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid_dense_samples:
        X_valid, y_valid = _stack_dense_samples(valid_dense_samples)
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)
        valid_sets.append(valid_set)
        valid_names.append("valid")

    params = {
        "objective": "multiclass",
        "num_class": len(RELATION_CLASSES),
        "metric": "multi_logloss",
        "learning_rate": learning_rate,
        "lambda_l2": l2,
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "seed": int(seed),
        "verbosity": -1,
        "force_col_wise": True,
    }
    evals_result: dict = {}
    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=1 if show_progress else 0),
    ]
    if valid_dense_samples and early_stop_patience > 0:
        callbacks.append(lgb.early_stopping(early_stop_patience, verbose=show_progress))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max(1, int(epochs)),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or booster.current_iteration())
    weights = {
        "model_type": "gbdt",
        "booster": booster,
        "best_iteration": best_iteration,
    }
    train_metrics = _evaluate_dense_matrix(X_train, y_train, weights, None)
    valid_metrics = None
    if valid_dense_samples:
        valid_metrics = _evaluate_dense_matrix(X_valid, y_valid, weights, None)

    # 构造逐轮 history，把 LightGBM record_evaluation 抓到的 multi_logloss 转成
    # 与 softmax/mlp 同形态的列表；只在 best_iteration 那一轮带上完整 metrics。
    train_logloss_per_iter = evals_result.get("train", {}).get("multi_logloss", [])
    valid_logloss_per_iter = evals_result.get("valid", {}).get("multi_logloss", []) if valid_dense_samples else []
    total_iters = max(
        len(train_logloss_per_iter),
        len(valid_logloss_per_iter),
        booster.current_iteration(),
    )
    history = []
    for idx in range(total_iters):
        history.append({
            "epoch": idx + 1,
            "train_logloss": float(train_logloss_per_iter[idx]) if idx < len(train_logloss_per_iter) else None,
            "valid_logloss": float(valid_logloss_per_iter[idx]) if idx < len(valid_logloss_per_iter) else None,
        })
    if 0 < best_iteration <= len(history):
        history[best_iteration - 1]["train"] = train_metrics
        history[best_iteration - 1]["valid"] = valid_metrics

    return {
        "model_type": "gbdt",
        "model_string": booster.model_to_string(num_iteration=best_iteration),
        "booster": booster,
        "best_iteration": best_iteration,
        "best_epoch": best_iteration,
        "stopped_epoch": int(booster.current_iteration()),
        "class_weights": class_weights,
        "history": history,
        "params": params,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }


def build_feature_importance(feature_names, weights, top_k=80):
    rows = []
    if isinstance(weights, dict) and weights.get("model_type") == "gbdt":
        booster = weights.get("booster")
        if booster is None:
            import lightgbm as lgb

            booster = lgb.Booster(model_str=weights["model_string"])
        scores = booster.feature_importance(importance_type="gain")
        for feature_index, feature_name in enumerate(feature_names):
            rows.append({"feature": feature_name, "max_abs_weight": float(scores[feature_index])})
        rows.sort(key=lambda item: (-item["max_abs_weight"], item["feature"]))
        return rows[:top_k]
    if isinstance(weights, dict) and weights.get("model_type") == "mlp":
        hidden_weights = np.asarray(weights["hidden_weights"], dtype=np.float64)
        output_weights = np.asarray(weights["output_weights"], dtype=np.float64)
        # 粗略重要性：输入到隐藏层的绝对权重，经输出层强度加权。
        hidden_strength = np.max(np.abs(output_weights), axis=0)
        scores = np.abs(hidden_weights).T @ hidden_strength
        for feature_index, feature_name in enumerate(feature_names):
            rows.append({"feature": feature_name, "max_abs_weight": float(scores[feature_index])})
        rows.sort(key=lambda item: (-item["max_abs_weight"], item["feature"]))
        return rows[:top_k]
    for feature_index, feature_name in enumerate(feature_names):
        max_abs_weight = max(abs(weights[class_idx][feature_index]) for class_idx in range(len(weights)))
        rows.append({"feature": feature_name, "max_abs_weight": max_abs_weight})
    rows.sort(key=lambda item: (-item["max_abs_weight"], item["feature"]))
    return rows[:top_k]


def build_prediction_rows(dense_samples, probabilities):
    rows = []
    for item, probs in zip(dense_samples, probabilities):
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        sample = item["sample"]
        rows.append({
            **{key: value for key, value in sample.items() if key != "features"},
            "gold_label": sample.get("label", "none"),
            "predicted_label": RELATION_CLASSES[pred_idx],
            "predicted_score": probs[pred_idx],
            "probabilities": {
                label: probs[idx]
                for idx, label in enumerate(RELATION_CLASSES)
            },
        })
    return rows


def invert_relation(relation):
    if relation == "downstream":
        return "upstream"
    if relation == "upstream":
        return "downstream"
    return relation


def canonical_pair(left_site_id, right_site_id):
    return tuple(sorted([str(left_site_id), str(right_site_id)]))


def empty_probability_map():
    return {label: 0.0 for label in RELATION_CLASSES}


def probability_map(probabilities):
    return {
        label: float(probabilities[index])
        for index, label in enumerate(RELATION_CLASSES)
    }


def build_pair_level_prediction_rows(dense_samples, probabilities):
    pair_records = {}
    for item, probs in zip(dense_samples, probabilities):
        sample = item["sample"]
        left_site_id = str(sample.get("u_site_id", ""))
        right_site_id = str(sample.get("v_site_id", ""))
        if not left_site_id or not right_site_id or left_site_id == right_site_id:
            continue

        site_a, site_b = canonical_pair(left_site_id, right_site_id)
        record = pair_records.setdefault(
            (site_a, site_b),
            {
                "site_a": site_a,
                "site_b": site_b,
                "site_a_name": "",
                "site_b_name": "",
                "site_a_domain": "",
                "site_b_domain": "",
                "site_a_region_id": "",
                "site_b_region_id": "",
                "gold_relation": "",
                "ab_probabilities": None,
                "ba_probabilities": None,
                "candidate_reasons": set(),
            },
        )
        sample_label = str(sample.get("label", "none") or "none")
        if left_site_id == site_a:
            record["site_a_name"] = sample.get("u_site_name", record["site_a_name"])
            record["site_b_name"] = sample.get("v_site_name", record["site_b_name"])
            record["site_a_domain"] = sample.get("u_domain", record["site_a_domain"])
            record["site_b_domain"] = sample.get("v_domain", record["site_b_domain"])
            record["site_a_region_id"] = sample.get("u_region_id", record["site_a_region_id"])
            record["site_b_region_id"] = sample.get("v_region_id", record["site_b_region_id"])
            record["gold_relation"] = sample_label if sample_label in RELATION_CLASSES else "none"
            record["ab_probabilities"] = probability_map(probs)
        else:
            record["site_a_name"] = sample.get("v_site_name", record["site_a_name"])
            record["site_b_name"] = sample.get("u_site_name", record["site_b_name"])
            record["site_a_domain"] = sample.get("v_domain", record["site_a_domain"])
            record["site_b_domain"] = sample.get("u_domain", record["site_b_domain"])
            record["site_a_region_id"] = sample.get("v_region_id", record["site_a_region_id"])
            record["site_b_region_id"] = sample.get("u_region_id", record["site_b_region_id"])
            if not record["gold_relation"]:
                record["gold_relation"] = invert_relation(sample_label) if sample_label in RELATION_CLASSES else "none"
            record["ba_probabilities"] = probability_map(probs)
        record["candidate_reasons"].update(sample.get("candidate_reasons", []))

    rows = []
    for (site_a, site_b), record in pair_records.items():
        ab = record["ab_probabilities"] or empty_probability_map()
        ba = record["ba_probabilities"] or empty_probability_map()
        combined_scores = {
            "downstream": (ab["downstream"] + ba["upstream"]) / 2.0,
            "upstream": (ab["upstream"] + ba["downstream"]) / 2.0,
            "bidirection": (ab["bidirection"] + ba["bidirection"]) / 2.0,
            "none": (ab["none"] + ba["none"]) / 2.0,
        }
        predicted_relation = max(
            RELATION_CLASSES,
            key=lambda label: (combined_scores[label], -RELATION_CLASSES.index(label)),
        )
        rows.append(
            {
                "sample_id": f"{site_a}__{site_b}",
                "site_a": site_a,
                "site_b": site_b,
                "site_a_name": record["site_a_name"],
                "site_b_name": record["site_b_name"],
                "site_a_domain": record["site_a_domain"],
                "site_b_domain": record["site_b_domain"],
                "site_a_region_id": record["site_a_region_id"],
                "site_b_region_id": record["site_b_region_id"],
                "gold_relation": record["gold_relation"] or "none",
                "predicted_relation": predicted_relation,
                "predicted_score": combined_scores[predicted_relation],
                "site_a_to_site_b_relation": predicted_relation,
                "site_b_to_site_a_relation": invert_relation(predicted_relation),
                "directional_scores": combined_scores,
                "ab_probabilities": ab,
                "ba_probabilities": ba,
                "has_ab_prediction": record["ab_probabilities"] is not None,
                "has_ba_prediction": record["ba_probabilities"] is not None,
                "candidate_reasons": sorted(record["candidate_reasons"]),
            }
        )
    return rows


def evaluate_pair_level_prediction_rows(rows):
    confusion = [[0 for _ in RELATION_CLASSES] for _ in RELATION_CLASSES]
    for row in rows:
        gold = row.get("gold_relation", "none")
        pred = row.get("predicted_relation", "none")
        if gold not in RELATION_CLASSES:
            gold = "none"
        if pred not in RELATION_CLASSES:
            pred = "none"
        confusion[RELATION_CLASSES.index(gold)][RELATION_CLASSES.index(pred)] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(RELATION_CLASSES)))
    per_class = {}
    macro_f1 = 0.0
    macro_class_count = 0
    for index, label in enumerate(RELATION_CLASSES):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(RELATION_CLASSES)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(RELATION_CLASSES)) if col != index)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        support = sum(confusion[index])
        if support > 0:
            macro_f1 += f1
            macro_class_count += 1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = safe_ratio(macro_f1, macro_class_count)
    return {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "classes": list(RELATION_CLASSES),
        "per_class": per_class,
        "pair_count": total,
    }


def _limit_candidates(candidates, limit, rng=None):
    candidates = list(candidates)
    if limit is None or limit < 0 or len(candidates) <= limit:
        return candidates
    if limit <= 0:
        return []
    if rng is not None:
        return rng.sample(candidates, limit)
    return sorted(candidates)[:limit]


def _candidate_targets_for_site(
    context,
    left_site_id,
    sites,
    rng=None,
    same_region_limit=10,
    same_domain_limit=10,
    topology_neighbor_limit=10,
    nearest_limit=10,
):
    candidate_targets = set()
    info = context.site_infos[left_site_id]
    if info.region_id != "MISSING":
        candidates = [
            site_id for site_id in context.region_to_sites.get(info.region_id, [])
            if site_id != left_site_id
        ]
        candidate_targets.update(_limit_candidates(candidates, same_region_limit, rng))
    if info.dominant_domain != "MISSING":
        candidates = [
            site_id for site_id in context.dominant_domain_to_sites.get(info.dominant_domain, [])
            if site_id != left_site_id
        ]
        candidate_targets.update(_limit_candidates(candidates, same_domain_limit, rng))
    candidate_targets.update(
        _limit_candidates(context.undirected_map.get(left_site_id, set()), topology_neighbor_limit, rng)
    )
    if nearest_limit is None or nearest_limit != 0:
        nearest = _nearest_sites_by_distance(
            context,
            left_site_id,
            [site_id for site_id in sites if site_id != left_site_id],
            10 if nearest_limit is None or nearest_limit < 0 else nearest_limit,
        )
        candidate_targets.update(nearest)
    candidate_targets.discard(left_site_id)
    candidate_targets = list(candidate_targets)
    if rng is not None:
        rng.shuffle(candidate_targets)
    else:
        candidate_targets.sort()
    return candidate_targets


def _iter_candidate_pair_keys(
    context,
    exclude_labeled=True,
    show_progress=False,
    progress_label="扫描候选源站点",
    max_pair_count=0,
    seed=42,
    randomize=False,
    same_region_limit=10,
    same_domain_limit=10,
    topology_neighbor_limit=10,
    nearest_limit=10,
):
    rng = random.Random(seed)
    sites = list(context.site_ids)
    if randomize:
        rng.shuffle(sites)
    emitted_pairs = set() if max_pair_count > 0 else None
    emitted_count = 0
    progress = _create_progress_bar(len(sites), progress_label, show_progress)
    try:
        for index, left_site_id in enumerate(sites, 1):
            for right_site_id in _candidate_targets_for_site(
                context,
                left_site_id,
                sites,
                rng=rng if randomize else None,
                same_region_limit=same_region_limit,
                same_domain_limit=same_domain_limit,
                topology_neighbor_limit=topology_neighbor_limit,
                nearest_limit=nearest_limit,
            ):
                if right_site_id == left_site_id:
                    continue
                pair_key = tuple(sorted((left_site_id, right_site_id)))
                if emitted_pairs is None and pair_key[0] != left_site_id:
                    continue
                if emitted_pairs is not None and pair_key in emitted_pairs:
                    continue
                if exclude_labeled and _has_any_labeled_relation(context, left_site_id, right_site_id):
                    continue
                if emitted_pairs is not None:
                    emitted_pairs.add(pair_key)
                emitted_count += 1
                yield pair_key
                if max_pair_count > 0 and emitted_count >= max_pair_count:
                    if progress is not None:
                        progress.set(index)
                    return
            if progress is not None:
                progress.set(index)
    finally:
        _close_progress_bar(progress)


def iter_candidate_relation_sample_chunks(
    context,
    max_candidate_count=50000,
    seed=42,
    exclude_labeled=True,
    max_samples_per_chunk=20000,
    same_region_limit=10,
    same_domain_limit=10,
    topology_neighbor_limit=10,
    nearest_limit=10,
    show_progress=False,
    progress_label="扫描候选源站点",
):
    max_samples_per_chunk = max(2, int(max_samples_per_chunk or 20000))
    rng = random.Random(seed)
    sites = list(context.site_ids)
    randomize = max_candidate_count > 0
    if randomize:
        rng.shuffle(sites)
    max_pair_count = max(1, max_candidate_count // 2) if max_candidate_count > 0 else 0
    emitted_pairs = set() if max_pair_count > 0 else None
    emitted_count = 0
    chunk = []
    progress = _create_progress_bar(len(sites), progress_label, show_progress)
    try:
        for index, left_site_id in enumerate(sites, 1):
            reached_limit = False
            for right_site_id in _candidate_targets_for_site(
                context,
                left_site_id,
                sites,
                rng=rng if randomize else None,
                same_region_limit=same_region_limit,
                same_domain_limit=same_domain_limit,
                topology_neighbor_limit=topology_neighbor_limit,
                nearest_limit=nearest_limit,
            ):
                if right_site_id == left_site_id:
                    continue
                pair_key = tuple(sorted((left_site_id, right_site_id)))
                if emitted_pairs is None and pair_key[0] != left_site_id:
                    continue
                if emitted_pairs is not None and pair_key in emitted_pairs:
                    continue
                if exclude_labeled and _has_any_labeled_relation(context, left_site_id, right_site_id):
                    continue

                if emitted_pairs is not None:
                    emitted_pairs.add(pair_key)
                emitted_count += 1
                chunk.append(
                    build_relation_sample(context, pair_key[0], pair_key[1], "none", {"candidate"}, "candidate")
                )
                chunk.append(
                    build_relation_sample(context, pair_key[1], pair_key[0], "none", {"candidate"}, "candidate")
                )
                if max_pair_count > 0 and emitted_count >= max_pair_count:
                    reached_limit = True
                    break
                if len(chunk) >= max_samples_per_chunk:
                    yield chunk
                    chunk = []

            if progress is not None:
                progress.set(index)
            if chunk and reached_limit:
                yield chunk
                chunk = []
            if reached_limit:
                return
    finally:
        _close_progress_bar(progress)
    if chunk:
        yield chunk


def generate_candidate_relation_samples(context, max_candidate_count=50000, seed=42, exclude_labeled=True):
    samples = []
    for chunk in iter_candidate_relation_sample_chunks(
        context,
        max_candidate_count=max_candidate_count,
        seed=seed,
        exclude_labeled=exclude_labeled,
        max_samples_per_chunk=max_candidate_count if max_candidate_count > 0 else 20000,
        show_progress=False,
    ):
        samples.extend(chunk)
    return samples
