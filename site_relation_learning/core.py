#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

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


def build_site_relation_context(label_file, site_graph_file, site_device_counts_file):
    site_infos = load_site_infos(site_graph_file, site_device_counts_file)
    relation_map = load_label_relation_map(label_file)
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
        "device_total_ratio_min_max": safe_ratio(min(left_total, right_total), max(left_total, right_total)),
        "device_total_diff_abs": float(abs(left_total - right_total)),
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

    if same_region_negatives > 0 and right_info.region_id != "MISSING":
        candidates = [
            site_id for site_id in context.region_to_sites.get(right_info.region_id, [])
            if site_id != right_site_id
        ]
        for candidate_id in _deterministic_sample(candidates, same_region_negatives, rng):
            yield left_site_id, candidate_id, "same_target_region"

    if same_region_negatives > 0 and left_info.region_id != "MISSING":
        candidates = [
            site_id for site_id in context.region_to_sites.get(left_info.region_id, [])
            if site_id != left_site_id
        ]
        for candidate_id in _deterministic_sample(candidates, same_region_negatives, rng):
            yield candidate_id, right_site_id, "same_source_region"

    if same_domain_negatives > 0 and right_info.dominant_domain != "MISSING":
        candidates = [
            site_id for site_id in context.dominant_domain_to_sites.get(right_info.dominant_domain, [])
            if site_id != right_site_id
        ]
        for candidate_id in _deterministic_sample(candidates, same_domain_negatives, rng):
            yield left_site_id, candidate_id, "same_target_domain"

    if same_domain_negatives > 0 and left_info.dominant_domain != "MISSING":
        candidates = [
            site_id for site_id in context.dominant_domain_to_sites.get(left_info.dominant_domain, [])
            if site_id != left_site_id
        ]
        for candidate_id in _deterministic_sample(candidates, same_domain_negatives, rng):
            yield candidate_id, right_site_id, "same_source_domain"

    if nearest_negatives > 0:
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
    show_progress=False,
):
    none_reason_map = defaultdict(set)
    ordered_pairs = list(positive_ordered_pairs)
    max_rounds = 3 if target_none_count is not None else 1
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


def _softmax(scores):
    max_score = max(scores) if scores else 0.0
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def predict_probabilities(weights, biases, dense_vector):
    scores = []
    for class_index, class_weights in enumerate(weights):
        score = biases[class_index]
        for weight, feature_value in zip(class_weights, dense_vector):
            score += weight * feature_value
        scores.append(score)
    return _softmax(scores)


def evaluate_dense_samples(dense_samples, weights, biases):
    confusion = [[0 for _ in RELATION_CLASSES] for _ in RELATION_CLASSES]
    probabilities = []
    for item in dense_samples:
        probs = predict_probabilities(weights, biases, item["x"])
        pred = max(range(len(probs)), key=lambda idx: probs[idx])
        gold = item["y"]
        confusion[gold][pred] += 1
        probabilities.append(probs)

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[idx][idx] for idx in range(len(RELATION_CLASSES)))
    per_class = {}
    macro_f1 = 0.0
    for idx, label in enumerate(RELATION_CLASSES):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(len(RELATION_CLASSES)) if row != idx)
        fn = sum(confusion[idx][col] for col in range(len(RELATION_CLASSES)) if col != idx)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        macro_f1 += f1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[idx]),
        }
    macro_f1 /= len(RELATION_CLASSES)
    return {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "classes": list(RELATION_CLASSES),
        "per_class": per_class,
        "sample_count": total,
    }, probabilities


def train_softmax_regression(
    train_dense_samples,
    valid_dense_samples=None,
    epochs=30,
    learning_rate=0.03,
    l2=1e-4,
    class_weight="balanced",
    seed=42,
    show_progress=False,
):
    rng = random.Random(seed)
    if not train_dense_samples:
        raise ValueError("训练集为空")
    feature_dim = len(train_dense_samples[0]["x"])
    class_count = len(RELATION_CLASSES)
    weights = [[0.0 for _ in range(feature_dim)] for _ in range(class_count)]
    biases = [0.0 for _ in range(class_count)]
    label_counter = Counter(item["y"] for item in train_dense_samples)
    if class_weight == "balanced":
        class_weights = {
            idx: safe_ratio(len(train_dense_samples), class_count * label_counter.get(idx, 0))
            for idx in range(class_count)
        }
    else:
        class_weights = {idx: 1.0 for idx in range(class_count)}

    best_state = None
    best_score = -1.0
    history = []
    progress = _create_progress_bar(epochs, "训练 softmax 多分类模型", show_progress)
    try:
        for epoch in range(1, epochs + 1):
            shuffled = list(train_dense_samples)
            rng.shuffle(shuffled)
            for item in shuffled:
                probs = predict_probabilities(weights, biases, item["x"])
                gold = item["y"]
                sample_weight = class_weights.get(gold, 1.0)
                for class_idx in range(class_count):
                    target = 1.0 if class_idx == gold else 0.0
                    error = (probs[class_idx] - target) * sample_weight
                    biases[class_idx] -= learning_rate * error
                    class_weights_vec = weights[class_idx]
                    for feat_idx, feature_value in enumerate(item["x"]):
                        grad = error * feature_value + l2 * class_weights_vec[feat_idx]
                        class_weights_vec[feat_idx] -= learning_rate * grad

            train_metrics, _ = evaluate_dense_samples(train_dense_samples, weights, biases)
            valid_metrics = None
            score = train_metrics["macro_f1"]
            if valid_dense_samples:
                valid_metrics, _ = evaluate_dense_samples(valid_dense_samples, weights, biases)
                score = valid_metrics["macro_f1"]
            history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
            if score > best_score:
                best_score = score
                best_state = {
                    "weights": [list(row) for row in weights],
                    "biases": list(biases),
                    "best_epoch": epoch,
                }
            if progress is not None:
                progress.set(epoch)
                progress.set_extra_text(f"macro_f1={score:.4f}")
    finally:
        _close_progress_bar(progress)

    best_state["history"] = history
    best_state["class_weights"] = class_weights
    return best_state


def build_feature_importance(feature_names, weights, top_k=80):
    rows = []
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
    for index, label in enumerate(RELATION_CLASSES):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(RELATION_CLASSES)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(RELATION_CLASSES)) if col != index)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        macro_f1 += f1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[index]),
        }
    macro_f1 /= len(RELATION_CLASSES)
    return {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "classes": list(RELATION_CLASSES),
        "per_class": per_class,
        "pair_count": total,
    }


def generate_candidate_relation_samples(context, max_candidate_count=50000, seed=42, exclude_labeled=True):
    rng = random.Random(seed)
    pair_candidates = set()
    seen = set()
    sites = context.site_ids
    for left_site_id in sites:
        candidate_targets = set()
        info = context.site_infos[left_site_id]
        if info.region_id != "MISSING":
            candidate_targets.update(context.region_to_sites.get(info.region_id, []))
        if info.dominant_domain != "MISSING":
            candidate_targets.update(context.dominant_domain_to_sites.get(info.dominant_domain, []))
        candidate_targets.update(context.undirected_map.get(left_site_id, set()))
        nearest = _nearest_sites_by_distance(
            context,
            left_site_id,
            [site_id for site_id in sites if site_id != left_site_id],
            10,
        )
        candidate_targets.update(nearest)
        for right_site_id in candidate_targets:
            if right_site_id == left_site_id or (left_site_id, right_site_id) in seen:
                continue
            if exclude_labeled and _has_any_labeled_relation(context, left_site_id, right_site_id):
                continue
            seen.add((left_site_id, right_site_id))
            seen.add((right_site_id, left_site_id))
            pair_candidates.add(tuple(sorted((left_site_id, right_site_id))))

    pair_candidates = sorted(pair_candidates)
    if max_candidate_count > 0:
        max_pair_count = max(1, max_candidate_count // 2)
        if len(pair_candidates) > max_pair_count:
            pair_candidates = sorted(rng.sample(pair_candidates, max_pair_count))

    candidates = []
    for left_site_id, right_site_id in pair_candidates:
        candidates.append((left_site_id, right_site_id))
        candidates.append((right_site_id, left_site_id))
    return [
        build_relation_sample(context, left_site_id, right_site_id, "none", {"candidate"}, "candidate")
        for left_site_id, right_site_id in sorted(candidates)
    ]
