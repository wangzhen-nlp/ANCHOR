#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from alarm_tools.progress_utils import ProgressBar
from ne_link_learning.core import (
    DOMAIN_BUCKETS,
    _adamic_adar_score,
    _jaccard,
    _resource_allocation_score,
    build_graph_context,
    deterministic_sample,
    haversine_km,
    normalize_text,
    safe_log1p,
    safe_ratio,
)


@dataclass(frozen=True)
class SiteInfo:
    site_id: str
    site_name: str
    region_id: str
    dominant_domain_bucket: str
    latitude: float | None
    longitude: float | None
    node_count: int


@dataclass
class SitePairContext:
    base_context: object
    site_infos: dict
    site_ids: list
    site_out_sites: dict
    site_in_sites: dict
    site_undirected_sites: dict
    region_to_sites: dict
    dominant_domain_bucket_to_sites: dict
    site_peer_group_keys: dict
    peer_groups: dict


def _create_progress_bar(total, label, show_progress):
    if not show_progress:
        return None
    print(f"⏳ {label}...")
    return ProgressBar(total, label)


def _close_progress_bar(progress):
    if progress is not None:
        progress.close()


def _pick_counter_mode(counter, default_value="MISSING"):
    if not isinstance(counter, Counter) or not counter:
        return default_value
    best_value = default_value
    best_count = -1
    for key, count in counter.items():
        if count > best_count:
            best_value = key
            best_count = count
            continue
        if count == best_count and str(key) < str(best_value):
            best_value = key
    return best_value


def _counter_key_set(counter):
    return {key for key, value in counter.items() if value > 0}


def _counter_cosine_similarity(left_counter, right_counter):
    left_keys = _counter_key_set(left_counter)
    right_keys = _counter_key_set(right_counter)
    all_keys = left_keys | right_keys
    if not all_keys:
        return 0.0

    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for key in all_keys:
        left_value = float(left_counter.get(key, 0))
        right_value = float(right_counter.get(key, 0))
        dot_product += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value

    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot_product / math.sqrt(left_norm * right_norm)


def _site_size_bucket(node_count):
    node_count = int(node_count or 0)
    if node_count <= 1:
        return "1"
    if node_count <= 3:
        return "2_3"
    if node_count <= 7:
        return "4_7"
    return "8_plus"


def _median(values):
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return 0.0
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _resolve_peer_sites(context, site_id):
    site_keys = context.site_peer_group_keys.get(site_id, {})
    for level_name in ("region_domain_size", "domain_size", "domain_only"):
        level_key = site_keys.get(level_name)
        peer_sites = [
            candidate_site_id
            for candidate_site_id in context.peer_groups.get(level_name, {}).get(level_key, [])
            if candidate_site_id != site_id
        ]
        if peer_sites:
            return peer_sites, level_name
    return [], "none"


def build_site_pair_context(ne_graph_data):
    base_context = build_graph_context(ne_graph_data)

    site_infos = {}
    region_to_sites = defaultdict(set)
    dominant_domain_bucket_to_sites = defaultdict(set)
    site_undirected_sites = {}
    peer_groups = {
        "region_domain_size": defaultdict(set),
        "domain_size": defaultdict(set),
        "domain_only": defaultdict(set),
    }
    site_peer_group_keys = {}

    for site_id, node_ids in base_context.site_to_nodes.items():
        if not node_ids:
            continue

        region_counter = Counter()
        site_name = ""
        latitude = longitude = None
        for node_id in node_ids:
            node_info = base_context.node_infos[node_id]
            region_counter[node_info.region_id or "MISSING"] += 1
            if not site_name and node_info.site_name:
                site_name = node_info.site_name
            if latitude is None and longitude is None:
                if node_info.latitude is not None and node_info.longitude is not None:
                    latitude = node_info.latitude
                    longitude = node_info.longitude

        if site_id in base_context.site_coords:
            latitude, longitude = base_context.site_coords[site_id]

        dominant_domain_bucket = _pick_counter_mode(
            base_context.site_domain_bucket_counts.get(site_id, Counter()),
            default_value="MISSING",
        )
        region_id = _pick_counter_mode(region_counter, default_value="MISSING")
        site_infos[site_id] = SiteInfo(
            site_id=site_id,
            site_name=site_name or site_id,
            region_id=region_id,
            dominant_domain_bucket=dominant_domain_bucket,
            latitude=latitude,
            longitude=longitude,
            node_count=len(node_ids),
        )
        region_to_sites[region_id].add(site_id)
        dominant_domain_bucket_to_sites[dominant_domain_bucket].add(site_id)

        size_bucket = _site_size_bucket(len(node_ids))
        site_keys = {
            "region_domain_size": (region_id, dominant_domain_bucket, size_bucket),
            "domain_size": (dominant_domain_bucket, size_bucket),
            "domain_only": (dominant_domain_bucket,),
        }
        site_peer_group_keys[site_id] = site_keys
        for level_name, level_key in site_keys.items():
            peer_groups[level_name][level_key].add(site_id)

    for site_id in site_infos:
        site_undirected_sites[site_id] = set(base_context.site_out_sites.get(site_id, set())) | set(
            base_context.site_in_sites.get(site_id, set())
        )

    return SitePairContext(
        base_context=base_context,
        site_infos=site_infos,
        site_ids=sorted(site_infos),
        site_out_sites={site_id: set(value) for site_id, value in base_context.site_out_sites.items()},
        site_in_sites={site_id: set(value) for site_id, value in base_context.site_in_sites.items()},
        site_undirected_sites=site_undirected_sites,
        region_to_sites={key: sorted(value) for key, value in region_to_sites.items()},
        dominant_domain_bucket_to_sites={key: sorted(value) for key, value in dominant_domain_bucket_to_sites.items()},
        site_peer_group_keys=site_peer_group_keys,
        peer_groups={
            level_name: {key: sorted(value) for key, value in group_map.items()}
            for level_name, group_map in peer_groups.items()
        },
    )


def collect_positive_site_edges(context):
    positive_edges = []
    for left_site_id, right_site_id in sorted(context.base_context.site_pair_forward_edge_count):
        if left_site_id == right_site_id:
            continue
        positive_edges.append((left_site_id, right_site_id))
    return positive_edges


def _count_domain_neighbor_targets(site_ids, context, target_domain_bucket):
    count = 0
    for site_id in site_ids:
        site_info = context.site_infos.get(site_id)
        if site_info and site_info.dominant_domain_bucket == target_domain_bucket:
            count += 1
    return count


def _make_site_split_keys(left_site_id, right_site_id):
    ordered_key = f"{left_site_id}__TO__{right_site_id}"
    unordered_key = "__".join(sorted([left_site_id, right_site_id]))
    return ordered_key, unordered_key


def extract_site_pair_features(context, left_site_id, right_site_id):
    left_info = context.site_infos[left_site_id]
    right_info = context.site_infos[right_site_id]

    left_out_sites = context.site_out_sites.get(left_site_id, set())
    left_in_sites = context.site_in_sites.get(left_site_id, set())
    right_out_sites = context.site_out_sites.get(right_site_id, set())
    right_in_sites = context.site_in_sites.get(right_site_id, set())
    left_undirected_sites = context.site_undirected_sites.get(left_site_id, set())
    right_undirected_sites = context.site_undirected_sites.get(right_site_id, set())

    left_out_sites_excl_pair = set(left_out_sites) - {right_site_id}
    left_in_sites_excl_pair = set(left_in_sites) - {right_site_id}
    right_out_sites_excl_pair = set(right_out_sites) - {left_site_id}
    right_in_sites_excl_pair = set(right_in_sites) - {left_site_id}

    common_out = left_out_sites & right_out_sites
    common_in = left_in_sites & right_in_sites
    common_undirected = left_undirected_sites & right_undirected_sites
    mids_left_to_right = left_out_sites & right_in_sites
    mids_right_to_left = right_out_sites & left_in_sites

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

    base_context = context.base_context
    left_type_counts = base_context.site_type_counts.get(left_site_id, Counter())
    right_type_counts = base_context.site_type_counts.get(right_site_id, Counter())
    left_network_type_counts = base_context.site_network_type_counts.get(left_site_id, Counter())
    right_network_type_counts = base_context.site_network_type_counts.get(right_site_id, Counter())
    left_manufacturer_counts = base_context.site_manufacturer_counts.get(left_site_id, Counter())
    right_manufacturer_counts = base_context.site_manufacturer_counts.get(right_site_id, Counter())
    left_domain_counts = base_context.site_domain_bucket_counts.get(left_site_id, Counter())
    right_domain_counts = base_context.site_domain_bucket_counts.get(right_site_id, Counter())

    left_domain_bucket = left_info.dominant_domain_bucket
    right_domain_bucket = right_info.dominant_domain_bucket
    left_out_degree_excl_pair = len(left_out_sites_excl_pair)
    left_in_degree_excl_pair = len(left_in_sites_excl_pair)
    right_out_degree_excl_pair = len(right_out_sites_excl_pair)
    right_in_degree_excl_pair = len(right_in_sites_excl_pair)

    left_peer_sites, left_peer_level = _resolve_peer_sites(context, left_site_id)
    right_peer_sites, right_peer_level = _resolve_peer_sites(context, right_site_id)

    left_peer_out_median = _median(
        len(context.site_out_sites.get(peer_site_id, set())) for peer_site_id in left_peer_sites
    )
    left_peer_in_median = _median(
        len(context.site_in_sites.get(peer_site_id, set())) for peer_site_id in left_peer_sites
    )
    right_peer_out_median = _median(
        len(context.site_out_sites.get(peer_site_id, set())) for peer_site_id in right_peer_sites
    )
    right_peer_in_median = _median(
        len(context.site_in_sites.get(peer_site_id, set())) for peer_site_id in right_peer_sites
    )

    left_out_gap_to_peer_median = max(0.0, left_peer_out_median - left_out_degree_excl_pair)
    left_in_gap_to_peer_median = max(0.0, left_peer_in_median - left_in_degree_excl_pair)
    right_out_gap_to_peer_median = max(0.0, right_peer_out_median - right_out_degree_excl_pair)
    right_in_gap_to_peer_median = max(0.0, right_peer_in_median - right_in_degree_excl_pair)

    forward_gap_fill_score = safe_ratio(left_out_gap_to_peer_median, left_peer_out_median) + safe_ratio(
        right_in_gap_to_peer_median, right_peer_in_median
    )
    reverse_gap_fill_score = safe_ratio(left_in_gap_to_peer_median, left_peer_in_median) + safe_ratio(
        right_out_gap_to_peer_median, right_peer_out_median
    )

    features = {
        "same_region": float(
            left_info.region_id != "MISSING" and left_info.region_id == right_info.region_id
        ),
        "same_dominant_domain": float(
            left_domain_bucket != "MISSING" and left_domain_bucket == right_domain_bucket
        ),
        "geo_distance_km": geo_distance_km or 0.0,
        "geo_distance_missing": float(geo_distance_km is None),
        "geo_distance_log1p": safe_log1p(geo_distance_km or 0.0),
        "left_site_size": float(left_info.node_count),
        "right_site_size": float(right_info.node_count),
        "site_size_ratio_min_max": safe_ratio(
            min(left_info.node_count, right_info.node_count),
            max(left_info.node_count, right_info.node_count),
        ),
        "site_size_diff_abs": float(abs(left_info.node_count - right_info.node_count)),
        "left_site_out_degree_excl_pair": float(left_out_degree_excl_pair),
        "left_site_in_degree_excl_pair": float(left_in_degree_excl_pair),
        "right_site_out_degree_excl_pair": float(right_out_degree_excl_pair),
        "right_site_in_degree_excl_pair": float(right_in_degree_excl_pair),
        "left_missing_outgoing_excl_pair": float(left_out_degree_excl_pair == 0),
        "left_missing_incoming_excl_pair": float(left_in_degree_excl_pair == 0),
        "right_missing_outgoing_excl_pair": float(right_out_degree_excl_pair == 0),
        "right_missing_incoming_excl_pair": float(right_in_degree_excl_pair == 0),
        "left_has_both_in_out_excl_pair": float(left_out_degree_excl_pair > 0 and left_in_degree_excl_pair > 0),
        "right_has_both_in_out_excl_pair": float(right_out_degree_excl_pair > 0 and right_in_degree_excl_pair > 0),
        "candidate_fills_left_outgoing_gap": float(left_out_degree_excl_pair == 0),
        "candidate_fills_right_incoming_gap": float(right_in_degree_excl_pair == 0),
        "candidate_fills_forward_zero_gap_both": float(
            left_out_degree_excl_pair == 0 and right_in_degree_excl_pair == 0
        ),
        "candidate_fills_reverse_zero_gap_both": float(
            left_in_degree_excl_pair == 0 and right_out_degree_excl_pair == 0
        ),
        "candidate_completes_left_bidirectional_role": float(
            left_out_degree_excl_pair == 0 and left_in_degree_excl_pair > 0
        ),
        "candidate_completes_right_bidirectional_role": float(
            right_in_degree_excl_pair == 0 and right_out_degree_excl_pair > 0
        ),
        "candidate_completes_bidirectional_roles_for_both": float(
            left_out_degree_excl_pair == 0
            and left_in_degree_excl_pair > 0
            and right_in_degree_excl_pair == 0
            and right_out_degree_excl_pair > 0
        ),
        "left_peer_site_count": float(len(left_peer_sites)),
        "right_peer_site_count": float(len(right_peer_sites)),
        "left_peer_out_degree_median": float(left_peer_out_median),
        "left_peer_in_degree_median": float(left_peer_in_median),
        "right_peer_out_degree_median": float(right_peer_out_median),
        "right_peer_in_degree_median": float(right_peer_in_median),
        "left_out_degree_gap_to_peer_median": float(left_out_gap_to_peer_median),
        "left_in_degree_gap_to_peer_median": float(left_in_gap_to_peer_median),
        "right_out_degree_gap_to_peer_median": float(right_out_gap_to_peer_median),
        "right_in_degree_gap_to_peer_median": float(right_in_gap_to_peer_median),
        "left_out_degree_gap_ratio_to_peer_median": safe_ratio(
            left_out_gap_to_peer_median, left_peer_out_median
        ),
        "left_in_degree_gap_ratio_to_peer_median": safe_ratio(
            left_in_gap_to_peer_median, left_peer_in_median
        ),
        "right_out_degree_gap_ratio_to_peer_median": safe_ratio(
            right_out_gap_to_peer_median, right_peer_out_median
        ),
        "right_in_degree_gap_ratio_to_peer_median": safe_ratio(
            right_in_gap_to_peer_median, right_peer_in_median
        ),
        "forward_gap_fill_score": float(forward_gap_fill_score),
        "reverse_gap_fill_score": float(reverse_gap_fill_score),
        "forward_minus_reverse_gap_fill_score": float(forward_gap_fill_score - reverse_gap_fill_score),
        "left_site_out_degree": float(len(left_out_sites)),
        "left_site_in_degree": float(len(left_in_sites)),
        "left_site_undirected_degree": float(len(left_undirected_sites)),
        "right_site_out_degree": float(len(right_out_sites)),
        "right_site_in_degree": float(len(right_in_sites)),
        "right_site_undirected_degree": float(len(right_undirected_sites)),
        "common_out_count": float(len(common_out)),
        "common_in_count": float(len(common_in)),
        "common_neighbor_count": float(len(common_undirected)),
        "jaccard_out": _jaccard(left_out_sites, right_out_sites),
        "jaccard_in": _jaccard(left_in_sites, right_in_sites),
        "jaccard_neighbor": _jaccard(left_undirected_sites, right_undirected_sites),
        "two_hop_left_to_right_count": float(len(mids_left_to_right)),
        "two_hop_right_to_left_count": float(len(mids_right_to_left)),
        "left_neighbor_target_domain_match_count": float(
            _count_domain_neighbor_targets(left_out_sites, context, right_domain_bucket)
        ),
        "right_neighbor_source_domain_match_count": float(
            _count_domain_neighbor_targets(right_in_sites, context, left_domain_bucket)
        ),
        "left_site_type_diversity": float(len(_counter_key_set(left_type_counts))),
        "right_site_type_diversity": float(len(_counter_key_set(right_type_counts))),
        "left_site_network_type_diversity": float(len(_counter_key_set(left_network_type_counts))),
        "right_site_network_type_diversity": float(len(_counter_key_set(right_network_type_counts))),
        "left_site_manufacturer_diversity": float(len(_counter_key_set(left_manufacturer_counts))),
        "right_site_manufacturer_diversity": float(len(_counter_key_set(right_manufacturer_counts))),
        "type_key_jaccard": _jaccard(_counter_key_set(left_type_counts), _counter_key_set(right_type_counts)),
        "network_type_key_jaccard": _jaccard(
            _counter_key_set(left_network_type_counts), _counter_key_set(right_network_type_counts)
        ),
        "manufacturer_key_jaccard": _jaccard(
            _counter_key_set(left_manufacturer_counts), _counter_key_set(right_manufacturer_counts)
        ),
        "domain_ratio_cosine_similarity": _counter_cosine_similarity(left_domain_counts, right_domain_counts),
        "type_ratio_cosine_similarity": _counter_cosine_similarity(left_type_counts, right_type_counts),
        "network_type_ratio_cosine_similarity": _counter_cosine_similarity(
            left_network_type_counts, right_network_type_counts
        ),
        "manufacturer_ratio_cosine_similarity": _counter_cosine_similarity(
            left_manufacturer_counts, right_manufacturer_counts
        ),
        "left_site_receives_from_right_domain_count": float(
            base_context.site_incoming_source_domain_counts.get(left_site_id, Counter()).get(right_domain_bucket, 0)
        ),
        "right_site_receives_from_left_domain_count": float(
            base_context.site_incoming_source_domain_counts.get(right_site_id, Counter()).get(left_domain_bucket, 0)
        ),
        "left_site_sends_to_right_domain_count": float(
            base_context.site_outgoing_target_domain_counts.get(left_site_id, Counter()).get(right_domain_bucket, 0)
        ),
        "right_site_sends_to_left_domain_count": float(
            base_context.site_outgoing_target_domain_counts.get(right_site_id, Counter()).get(left_domain_bucket, 0)
        ),
        "adamic_adar_neighbor": _adamic_adar_score(common_undirected, context.site_undirected_sites),
        "resource_allocation_neighbor": _resource_allocation_score(common_undirected, context.site_undirected_sites),
        "adamic_adar_two_hop_left_to_right": _adamic_adar_score(mids_left_to_right, context.site_undirected_sites),
        "resource_allocation_two_hop_left_to_right": _resource_allocation_score(
            mids_left_to_right, context.site_undirected_sites
        ),
        "left_region_missing": float(left_info.region_id == "MISSING"),
        "right_region_missing": float(right_info.region_id == "MISSING"),
    }

    for level_name in ("region_domain_size", "domain_size", "domain_only", "none"):
        features[f"left_peer_level_is__{level_name}"] = float(left_peer_level == level_name)
        features[f"right_peer_level_is__{level_name}"] = float(right_peer_level == level_name)

    for domain_bucket in DOMAIN_BUCKETS:
        features[f"left_site_domain_ratio__{domain_bucket.lower()}"] = safe_ratio(
            left_domain_counts.get(domain_bucket, 0), left_info.node_count
        )
        features[f"right_site_domain_ratio__{domain_bucket.lower()}"] = safe_ratio(
            right_domain_counts.get(domain_bucket, 0), right_info.node_count
        )
        features[f"left_dominant_domain_is__{domain_bucket.lower()}"] = float(
            left_domain_bucket == domain_bucket
        )
        features[f"right_dominant_domain_is__{domain_bucket.lower()}"] = float(
            right_domain_bucket == domain_bucket
        )

    for left_bucket in DOMAIN_BUCKETS:
        for right_bucket in DOMAIN_BUCKETS:
            features[
                f"dominant_domain_pair__{left_bucket.lower()}__{right_bucket.lower()}"
            ] = float(left_domain_bucket == left_bucket and right_domain_bucket == right_bucket)

    return features


def _build_site_sample(context, left_site_id, right_site_id, label, candidate_reasons, sample_role):
    left_info = context.site_infos[left_site_id]
    right_info = context.site_infos[right_site_id]
    ordered_site_pair_key, unordered_site_pair_key = _make_site_split_keys(left_site_id, right_site_id)
    sample_id = f"{left_site_id}__{right_site_id}"

    supporting_link_types = set()
    pair_key = (left_site_id, right_site_id)
    for left_ne_id in context.base_context.site_to_nodes.get(left_site_id, []):
        for right_ne_id in context.base_context.site_to_nodes.get(right_site_id, []):
            supporting_link_types.update(context.base_context.edge_link_types.get((left_ne_id, right_ne_id), set()))

    return {
        "sample_id": sample_id,
        "label": int(label),
        "sample_role": sample_role,
        "sample_granularity": "site_pair",
        "u_ne_id": "",
        "v_ne_id": "",
        "u_site_id": left_site_id,
        "v_site_id": right_site_id,
        "u_site_name": left_info.site_name,
        "v_site_name": right_info.site_name,
        "u_region_id": left_info.region_id,
        "v_region_id": right_info.region_id,
        "u_domain": left_info.dominant_domain_bucket,
        "v_domain": right_info.dominant_domain_bucket,
        "ordered_site_pair_key": ordered_site_pair_key,
        "unordered_site_pair_key": unordered_site_pair_key,
        "candidate_reasons": sorted(candidate_reasons),
        "supporting_ne_edge_count": int(context.base_context.site_pair_forward_edge_count.get(pair_key, 0)),
        "supporting_link_types": sorted(supporting_link_types),
        "features": extract_site_pair_features(context, left_site_id, right_site_id),
    }


def _try_add_negative_site_pair(context, negative_reason_map, positive_edge_set, left_site_id, right_site_id, reason):
    if left_site_id == right_site_id:
        return False
    if left_site_id not in context.site_infos or right_site_id not in context.site_infos:
        return False
    if (left_site_id, right_site_id) in positive_edge_set:
        return False
    if right_site_id in context.site_out_sites.get(left_site_id, set()):
        return False

    negative_reason_map[(left_site_id, right_site_id)].add(reason)
    return True


def _generate_site_negative_pool(
    context,
    positive_edges,
    same_source_region_negatives,
    same_target_region_negatives,
    same_source_domain_negatives,
    same_target_domain_negatives,
    two_hop_target_negatives,
    two_hop_source_negatives,
    reverse_direction_negatives,
    rng,
    show_progress=False,
):
    positive_edge_set = set(positive_edges)
    negative_reason_map = defaultdict(set)
    progress = _create_progress_bar(len(positive_edges), "构造站点负样本候选", show_progress)
    try:
        for index, (left_site_id, right_site_id) in enumerate(positive_edges, start=1):
            left_info = context.site_infos[left_site_id]
            right_info = context.site_infos[right_site_id]

            if reverse_direction_negatives > 0:
                _try_add_negative_site_pair(
                    context,
                    negative_reason_map,
                    positive_edge_set,
                    right_site_id,
                    left_site_id,
                    "reverse_direction_missing",
                )

            if same_target_region_negatives > 0 and right_info.region_id != "MISSING":
                candidate_targets = [
                    site_id
                    for site_id in context.region_to_sites.get(right_info.region_id, [])
                    if site_id != right_site_id
                ]
                for candidate_site_id in deterministic_sample(candidate_targets, same_target_region_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        left_site_id,
                        candidate_site_id,
                        "same_target_region",
                    )

            if same_source_region_negatives > 0 and left_info.region_id != "MISSING":
                candidate_sources = [
                    site_id
                    for site_id in context.region_to_sites.get(left_info.region_id, [])
                    if site_id != left_site_id
                ]
                for candidate_site_id in deterministic_sample(candidate_sources, same_source_region_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        candidate_site_id,
                        right_site_id,
                        "same_source_region",
                    )

            if same_target_domain_negatives > 0 and right_info.dominant_domain_bucket != "MISSING":
                candidate_targets = [
                    site_id
                    for site_id in context.dominant_domain_bucket_to_sites.get(right_info.dominant_domain_bucket, [])
                    if site_id != right_site_id
                ]
                for candidate_site_id in deterministic_sample(candidate_targets, same_target_domain_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        left_site_id,
                        candidate_site_id,
                        "same_target_domain",
                    )

            if same_source_domain_negatives > 0 and left_info.dominant_domain_bucket != "MISSING":
                candidate_sources = [
                    site_id
                    for site_id in context.dominant_domain_bucket_to_sites.get(left_info.dominant_domain_bucket, [])
                    if site_id != left_site_id
                ]
                for candidate_site_id in deterministic_sample(candidate_sources, same_source_domain_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        candidate_site_id,
                        right_site_id,
                        "same_source_domain",
                    )

            if two_hop_target_negatives > 0:
                candidate_targets = set()
                for mid_site_id in context.site_out_sites.get(left_site_id, set()):
                    candidate_targets.update(context.site_out_sites.get(mid_site_id, set()))
                candidate_targets.discard(left_site_id)
                for candidate_site_id in deterministic_sample(sorted(candidate_targets), two_hop_target_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        left_site_id,
                        candidate_site_id,
                        "two_hop_target",
                    )

            if two_hop_source_negatives > 0:
                candidate_sources = set()
                for mid_site_id in context.site_in_sites.get(right_site_id, set()):
                    candidate_sources.update(context.site_in_sites.get(mid_site_id, set()))
                candidate_sources.discard(right_site_id)
                for candidate_site_id in deterministic_sample(sorted(candidate_sources), two_hop_source_negatives, rng):
                    _try_add_negative_site_pair(
                        context,
                        negative_reason_map,
                        positive_edge_set,
                        candidate_site_id,
                        right_site_id,
                        "two_hop_source",
                    )

            if progress is not None:
                progress.set(index)
                progress.set_extra_text(f"已收集 {len(negative_reason_map)} 条候选")
    finally:
        _close_progress_bar(progress)

    return negative_reason_map


def _pick_random_hard_target_site(context, left_site_id, rng):
    left_info = context.site_infos[left_site_id]
    candidate_groups = []

    if left_info.region_id and left_info.region_id != "MISSING":
        candidate_groups.append(context.region_to_sites.get(left_info.region_id, []))
    if left_info.dominant_domain_bucket and left_info.dominant_domain_bucket != "MISSING":
        candidate_groups.append(
            context.dominant_domain_bucket_to_sites.get(left_info.dominant_domain_bucket, [])
        )

    candidate_groups.extend(context.dominant_domain_bucket_to_sites.values())

    for candidate_group in candidate_groups:
        candidate_group = [site_id for site_id in candidate_group if site_id != left_site_id]
        if not candidate_group:
            continue
        candidate_site_id = rng.choice(candidate_group)
        if candidate_site_id != left_site_id:
            return candidate_site_id

    all_sites = context.site_ids
    for _ in range(20):
        candidate_site_id = rng.choice(all_sites)
        if candidate_site_id != left_site_id:
            return candidate_site_id
    return ""


def generate_site_link_learning_samples(
    context,
    max_negative_per_positive=4.0,
    seed=42,
    same_source_region_negatives=1,
    same_target_region_negatives=1,
    same_source_domain_negatives=1,
    same_target_domain_negatives=1,
    two_hop_target_negatives=1,
    two_hop_source_negatives=1,
    reverse_direction_negatives=1,
    random_hard_negative_ratio=1.0,
    show_progress=False,
):
    rng = random.Random(seed)
    positive_edges = collect_positive_site_edges(context)
    positive_edge_set = set(positive_edges)
    negative_reason_map = _generate_site_negative_pool(
        context=context,
        positive_edges=positive_edges,
        same_source_region_negatives=same_source_region_negatives,
        same_target_region_negatives=same_target_region_negatives,
        same_source_domain_negatives=same_source_domain_negatives,
        same_target_domain_negatives=same_target_domain_negatives,
        two_hop_target_negatives=two_hop_target_negatives,
        two_hop_source_negatives=two_hop_source_negatives,
        reverse_direction_negatives=reverse_direction_negatives,
        rng=rng,
        show_progress=show_progress,
    )

    target_negative_count = int(math.ceil(len(positive_edges) * max(0.0, float(max_negative_per_positive))))
    extra_random_target = (
        int(math.ceil(len(positive_edges) * float(random_hard_negative_ratio)))
        if random_hard_negative_ratio > 0
        else 0
    )

    random_negative_attempts = 0
    max_random_attempts = max(2000, extra_random_target * 50)
    random_target_total = target_negative_count + extra_random_target
    random_progress = None
    if show_progress and context.site_ids and len(negative_reason_map) < random_target_total:
        random_progress = _create_progress_bar(
            random_target_total,
            "补充随机硬负样本",
            show_progress,
        )
        random_progress.set(len(negative_reason_map))
        random_progress.set_extra_text(f"attempts={random_negative_attempts}")
    while (
        len(negative_reason_map) < target_negative_count + extra_random_target
        and random_negative_attempts < max_random_attempts
        and context.site_ids
    ):
        random_negative_attempts += 1
        left_site_id = rng.choice(context.site_ids)
        right_site_id = _pick_random_hard_target_site(context, left_site_id, rng)
        if not right_site_id:
            continue
        _try_add_negative_site_pair(
            context,
            negative_reason_map,
            positive_edge_set,
            left_site_id,
            right_site_id,
            "random_hard_negative",
        )
        if random_progress is not None:
            random_progress.set(min(len(negative_reason_map), random_target_total))
            random_progress.set_extra_text(f"attempts={random_negative_attempts}")
    _close_progress_bar(random_progress)

    negative_items = list(negative_reason_map.items())
    if target_negative_count <= 0:
        negative_items = []
    elif len(negative_items) > target_negative_count:
        negative_items = rng.sample(negative_items, target_negative_count)

    positive_samples = []
    positive_progress = _create_progress_bar(len(positive_edges), "构造正样本特征", show_progress)
    try:
        for index, (left_site_id, right_site_id) in enumerate(positive_edges, start=1):
            positive_samples.append(
                _build_site_sample(
                    context,
                    left_site_id,
                    right_site_id,
                    1,
                    {"observed_site_edge"},
                    "positive",
                )
            )
            if positive_progress is not None:
                positive_progress.set(index)
    finally:
        _close_progress_bar(positive_progress)

    negative_samples = []
    negative_progress = _create_progress_bar(len(negative_items), "构造负样本特征", show_progress)
    try:
        for index, ((left_site_id, right_site_id), reasons) in enumerate(negative_items, start=1):
            negative_samples.append(
                _build_site_sample(
                    context,
                    left_site_id,
                    right_site_id,
                    0,
                    reasons,
                    "negative",
                )
            )
            if negative_progress is not None:
                negative_progress.set(index)
    finally:
        _close_progress_bar(negative_progress)

    samples = positive_samples + negative_samples
    rng.shuffle(samples)
    return samples


def generate_candidate_site_link_samples_for_scoring(
    context,
    max_candidate_count=20000,
    seed=42,
    same_source_region_negatives=2,
    same_target_region_negatives=2,
    same_source_domain_negatives=2,
    same_target_domain_negatives=2,
    two_hop_target_negatives=2,
    two_hop_source_negatives=2,
    reverse_direction_negatives=1,
    random_hard_negative_ratio=2.0,
):
    rng = random.Random(seed)
    positive_edges = collect_positive_site_edges(context)
    positive_edge_set = set(positive_edges)
    negative_reason_map = _generate_site_negative_pool(
        context=context,
        positive_edges=positive_edges,
        same_source_region_negatives=same_source_region_negatives,
        same_target_region_negatives=same_target_region_negatives,
        same_source_domain_negatives=same_source_domain_negatives,
        same_target_domain_negatives=same_target_domain_negatives,
        two_hop_target_negatives=two_hop_target_negatives,
        two_hop_source_negatives=two_hop_source_negatives,
        reverse_direction_negatives=reverse_direction_negatives,
        rng=rng,
    )

    extra_random_target = (
        int(math.ceil(len(positive_edges) * max(0.0, float(random_hard_negative_ratio))))
        if random_hard_negative_ratio > 0
        else 0
    )
    random_negative_attempts = 0
    max_random_attempts = max(2000, extra_random_target * 50)
    while (
        len(negative_reason_map) < extra_random_target
        and random_negative_attempts < max_random_attempts
        and context.site_ids
    ):
        random_negative_attempts += 1
        left_site_id = rng.choice(context.site_ids)
        right_site_id = _pick_random_hard_target_site(context, left_site_id, rng)
        if not right_site_id:
            continue
        _try_add_negative_site_pair(
            context,
            negative_reason_map,
            positive_edge_set,
            left_site_id,
            right_site_id,
            "random_hard_negative",
        )

    candidate_items = list(negative_reason_map.items())
    if max_candidate_count > 0 and len(candidate_items) > max_candidate_count:
        candidate_items = rng.sample(candidate_items, max_candidate_count)

    candidate_samples = [
        _build_site_sample(context, left_site_id, right_site_id, 0, reasons, "candidate")
        for (left_site_id, right_site_id), reasons in candidate_items
    ]
    candidate_samples.sort(key=lambda item: item["sample_id"])
    return candidate_samples
