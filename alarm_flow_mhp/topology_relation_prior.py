"""Inference-time topology relation priors for feature-mode MHP scoring."""

from __future__ import annotations

import math


RELATION_KEYS = (
    "same_device",
    "direct",
    "same_site",
    "indirect",
    "cross_site",
    "unknown",
)

_ALIASES = {
    "same_device": "same_device",
    "same-device": "same_device",
    "same_ne": "same_device",
    "same-ne": "same_device",
    "same_source": "same_device",
    "same-source": "same_device",
    "direct": "direct",
    "direct_topology": "direct",
    "direct-topology": "direct",
    "direct_link": "direct",
    "direct-link": "direct",
    "same_site": "same_site",
    "same-site": "same_site",
    "indirect": "indirect",
    "indirect_topology": "indirect",
    "indirect-topology": "indirect",
    "multi_hop": "indirect",
    "multi-hop": "indirect",
    "cross_site": "cross_site",
    "cross-site": "cross_site",
    "unknown": "unknown",
    "unknown_context": "unknown",
    "unknown-context": "unknown",
    "default": "unknown",
}


def normalize_relation_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    normalized = normalized.replace(" ", "_")
    mapped = _ALIASES.get(normalized)
    if mapped is None:
        valid = ", ".join(RELATION_KEYS)
        raise ValueError(f"unknown topology relation prior key {key!r}; valid keys: {valid}")
    return mapped


def parse_topology_relation_prior(text) -> dict[str, float]:
    """Parse ``k=v,k=v`` relation multipliers.

    Missing keys default to 1.0 downstream, so an empty string means no behavior
    change. Values must be finite and non-negative; 0 disables that relation.
    """
    if text is None:
        return {}
    raw = str(text).strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "topology relation prior entries must be k=v, for example "
                "same_device=1,direct=1,same_site=0.8,indirect=0.5,cross_site=0.2,unknown=0.05"
            )
        key, value = item.split("=", 1)
        rel = normalize_relation_key(key)
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid topology relation prior value for {key!r}: {value!r}") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"topology relation prior for {key!r} must be finite and >= 0")
        out[rel] = weight
    return out


def format_topology_relation_prior(weights: dict | None) -> str:
    weights = weights or {}
    return ",".join(f"{key}={float(weights.get(key, 1.0)):g}" for key in RELATION_KEYS)


def relation_weight(weights: dict | None, relation: str) -> float:
    return float((weights or {}).get(relation, 1.0))


def _site_id(node_infos, ne: str) -> str:
    info = (node_infos or {}).get(ne)
    return str(getattr(info, "site_id", "") or "")


def classify_topology_relation(source_ne: str, target_ne: str, topology_index=None, node_infos=None) -> str:
    """Classify an NE pair using symmetric undirected topology."""
    source_ne = str(source_ne or "").strip()
    target_ne = str(target_ne or "").strip()
    if source_ne and target_ne and source_ne == target_ne:
        return "same_device"

    hop = 0
    if source_ne and target_ne and topology_index is not None:
        undirected = getattr(topology_index, "undirected_hops", {}) or {}
        hop = int(undirected.get(source_ne, {}).get(target_ne, 0) or 0)
        if hop == 1:
            return "direct"

    source_site = _site_id(node_infos, source_ne)
    target_site = _site_id(node_infos, target_ne)
    if source_site and target_site and source_site == target_site:
        return "same_site"
    if hop > 1:
        return "indirect"

    if source_site and target_site and source_site != target_site:
        return "cross_site"
    return "unknown"


def topology_relation_weights(source_nes, target_ne: str, topology_index=None, node_infos=None, priors=None):
    import numpy as np

    if not priors:
        return np.ones(len(source_nes), dtype=np.float64)
    return np.asarray(
        [
            relation_weight(
                priors,
                classify_topology_relation(src_ne, target_ne, topology_index, node_infos),
            )
            for src_ne in source_nes
        ],
        dtype=np.float64,
    )
