"""Learned, indexable candidate policies for AlarmPeriod association plans.

The policy is deliberately a small portfolio of deterministic rules rather
than a pair classifier.  Every rule can enumerate its matches from an inverted
index, so applying a policy does not require scanning the global source
universe first.
"""

from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass
import hashlib
import json
import math
import os
import tempfile

from alarm_flow_mhp.feature_spec import phi_domain_of, topo_node_of


CANDIDATE_POLICY_FORMAT = "alarm_flow_mhp.period_candidate_policy"
CANDIDATE_POLICY_VERSION = 1

RULE_SAME_ENTITY = "same_entity"
RULE_SAME_NODE = "same_node"
RULE_SAME_SITE = "same_site"
RULE_TOPOLOGY = "topology"
RULE_SAME_VENDOR_NETYPE = "same_vendor_ne_type"
RULE_SAME_VENDOR = "same_vendor"
RULE_SAME_NETYPE = "same_ne_type"
RULE_SAME_DOMAIN = "same_domain"
RULE_GEO_NEAR = "geo_near"
RULE_GEO_METRO = "geo_metro"

# New non-local rules are appended so existing RULE_BITS/RELATED_MASK assignments
# stay stable (older policies keep their bit meanings).
RULES = (
    RULE_SAME_ENTITY,
    RULE_SAME_NODE,
    RULE_SAME_SITE,
    RULE_TOPOLOGY,
    RULE_SAME_VENDOR_NETYPE,
    RULE_SAME_VENDOR,
    RULE_SAME_NETYPE,
    RULE_SAME_DOMAIN,
    RULE_GEO_NEAR,
    RULE_GEO_METRO,
)

# Geo blocking: quantize (lat, lon) into a grid and retrieve the 3×3 cell
# neighborhood, a correct superset of "within one cell size" that stays an
# equality-indexable lookup. Degrees are in the rule semantics (bump the
# module version if you change them; policies do not fingerprint the cell size).
GEO_NEAR_DEG = 0.1   # ~11 km of latitude per cell
GEO_METRO_DEG = 0.5  # ~55 km of latitude per cell
_GEO_DEG = {RULE_GEO_NEAR: GEO_NEAR_DEG, RULE_GEO_METRO: GEO_METRO_DEG}


def _geo_cell(coords, deg):
    if not coords:
        return None
    lat, lon = coords
    return (int(math.floor(float(lat) / deg)), int(math.floor(float(lon) / deg)))


def _geo_neighbor_cells(cell):
    if cell is None:
        return ()
    row, col = cell
    return tuple(
        (row + drow, col + dcol) for drow in (-1, 0, 1) for dcol in (-1, 0, 1)
    )


def _geo_adjacent(cell_a, cell_b) -> bool:
    return (
        cell_a is not None
        and cell_b is not None
        and abs(cell_a[0] - cell_b[0]) <= 1
        and abs(cell_a[1] - cell_b[1]) <= 1
    )
RULE_BITS = {rule: 1 << index for index, rule in enumerate(RULES)}
RELATED_RULES = (
    RULE_SAME_ENTITY,
    RULE_SAME_NODE,
    RULE_SAME_SITE,
    RULE_TOPOLOGY,
)
RELATED_MASK = 0
for _rule in RELATED_RULES:
    RELATED_MASK |= RULE_BITS[_rule]
del _rule


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def candidate_policy_fingerprint(
    model_path,
    ne_graph_path,
    site_graph_path,
    plan_config,
    topology_node_field="alarm_source",
) -> dict:
    """Fingerprint every input that changes teacher labels or rule indexes."""
    node_field = str(topology_node_field or "alarm_source")
    topology_graph_path = site_graph_path if node_field == "site_id" else ne_graph_path
    normalized_plan = dict(plan_config)
    normalized_plan.pop("candidate_scope", None)
    return {
        "model_sha256": sha256_file(model_path),
        "ne_graph_sha256": sha256_file(ne_graph_path),
        "topology_graph_sha256": sha256_file(topology_graph_path),
        "topology_node_field": node_field,
        "plan_config": normalized_plan,
    }


@dataclass(frozen=True)
class CandidatePolicy:
    rules_by_alarm_pair: dict
    fallback_rules: tuple = RELATED_RULES
    approved: bool = False
    fingerprint: dict | None = None
    validation: dict | None = None

    def rules_for(self, target_alarm_type, source_alarm_type) -> tuple:
        target_row = self.rules_by_alarm_pair.get(str(target_alarm_type), {})
        rules = target_row.get(str(source_alarm_type), self.fallback_rules)
        return tuple(rules)

    def to_dict(self) -> dict:
        return {
            "format": CANDIDATE_POLICY_FORMAT,
            "version": CANDIDATE_POLICY_VERSION,
            "approved": bool(self.approved),
            "fingerprint": dict(self.fingerprint or {}),
            "fallback_rules": list(self.fallback_rules),
            "rules_by_alarm_pair": {
                str(target): {
                    str(source): list(rules)
                    for source, rules in sorted(row.items())
                }
                for target, row in sorted(self.rules_by_alarm_pair.items())
            },
            "validation": dict(self.validation or {}),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("format") != CANDIDATE_POLICY_FORMAT:
            raise ValueError(
                f"unsupported candidate policy format: {payload.get('format')!r}"
            )
        if int(payload.get("version", -1)) != CANDIDATE_POLICY_VERSION:
            raise ValueError(
                f"unsupported candidate policy version: {payload.get('version')!r}"
            )
        fallback = _validate_rules(payload.get("fallback_rules", RELATED_RULES))
        rows = {}
        for target, row in (payload.get("rules_by_alarm_pair") or {}).items():
            if not isinstance(row, dict):
                raise ValueError("candidate policy alarm-pair row must be an object")
            rows[str(target)] = {
                str(source): _validate_rules(rules)
                for source, rules in row.items()
            }
        return cls(
            rules_by_alarm_pair=rows,
            fallback_rules=fallback,
            approved=bool(payload.get("approved", False)),
            fingerprint=dict(payload.get("fingerprint") or {}),
            validation=dict(payload.get("validation") or {}),
        )


def _validate_rules(rules) -> tuple:
    out = []
    for rule in rules or ():
        rule = str(rule)
        if rule not in RULE_BITS:
            raise ValueError(f"unknown candidate policy rule: {rule!r}")
        if rule not in out:
            out.append(rule)
    return tuple(out)


def load_candidate_policy(path, expected_fingerprint=None, require_approved=True):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            policy = CandidatePolicy.from_dict(json.load(stream))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid candidate policy: {exc}") from exc
    if require_approved and not policy.approved:
        raise ValueError("candidate policy is not approved by validation")
    if expected_fingerprint is not None and policy.fingerprint != expected_fingerprint:
        actual = policy.fingerprint or {}
        changed = sorted(
            key
            for key in set(actual) | set(expected_fingerprint)
            if actual.get(key) != expected_fingerprint.get(key)
        )
        raise ValueError(
            "candidate policy does not match current model/graphs/config; "
            f"changed={','.join(changed) or 'unknown'}"
        )
    return policy


def write_candidate_policy(path, policy: CandidatePolicy):
    output = os.path.abspath(path)
    directory = os.path.dirname(output) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".candidate-policy-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(policy.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, output)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


_EntAttrs = namedtuple(
    "_EntAttrs",
    "node site vendor ne_type domain cell_near cell_metro",
)


def _entity_attributes(entity, scorer):
    node = topo_node_of(entity)
    info = scorer.node_infos.get(node)
    lat = getattr(info, "latitude", None)
    lon = getattr(info, "longitude", None)
    coords = (lat, lon) if (lat is not None and lon is not None) else None
    return _EntAttrs(
        node=node,
        site=str(getattr(info, "site_id", "") or ""),
        vendor=str(getattr(info, "manufacturer", "") or ""),
        ne_type=str(getattr(info, "ne_type", "") or ""),
        domain=str(phi_domain_of(entity, scorer.node_infos) or ""),
        cell_near=_geo_cell(coords, GEO_NEAR_DEG),
        cell_metro=_geo_cell(coords, GEO_METRO_DEG),
    )


def candidate_rule_mask(target, source, scorer) -> int:
    """Rules that can retrieve one directed PeriodType pair."""
    mask = 0
    if target.entity == source.entity:
        mask |= RULE_BITS[RULE_SAME_ENTITY]
    t = _entity_attributes(target.entity, scorer)
    s = _entity_attributes(source.entity, scorer)
    if t.node and t.node == s.node:
        mask |= RULE_BITS[RULE_SAME_NODE]
    if t.site and t.site == s.site:
        mask |= RULE_BITS[RULE_SAME_SITE]
    hops = (
        getattr(scorer.topology_index, "undirected_hops", {}) or {}
        if scorer.topology_index is not None
        else {}
    )
    if t.node and s.node and (
        hops.get(t.node, {}).get(s.node, 0) or hops.get(s.node, {}).get(t.node, 0)
    ):
        mask |= RULE_BITS[RULE_TOPOLOGY]
    if t.vendor and t.ne_type and t.vendor == s.vendor and t.ne_type == s.ne_type:
        mask |= RULE_BITS[RULE_SAME_VENDOR_NETYPE]
    if t.vendor and t.vendor == s.vendor:
        mask |= RULE_BITS[RULE_SAME_VENDOR]
    if t.ne_type and t.ne_type == s.ne_type:
        mask |= RULE_BITS[RULE_SAME_NETYPE]
    if t.domain and t.domain == s.domain:
        mask |= RULE_BITS[RULE_SAME_DOMAIN]
    if _geo_adjacent(t.cell_near, s.cell_near):
        mask |= RULE_BITS[RULE_GEO_NEAR]
    if _geo_adjacent(t.cell_metro, s.cell_metro):
        mask |= RULE_BITS[RULE_GEO_METRO]
    return mask


def policy_pair_allowed(policy, target, source, scorer) -> bool:
    selected = 0
    for rule in policy.rules_for(target.alarm_type, source.alarm_type):
        selected |= RULE_BITS[rule]
    return bool(candidate_rule_mask(target, source, scorer) & selected)


def unrelated_pair_allowed(policy, target, source, scorer) -> bool:
    """Policy-allowed and *not* topology-related.

    The ``unrelated`` branch is disjoint from ``related`` by construction: any
    pair the related predicate would retrieve stays in the related cache, so an
    unrelated candidate must match the policy through some rule while matching
    no related rule at all.
    """
    mask = candidate_rule_mask(target, source, scorer)
    if mask & RELATED_MASK:
        return False
    selected = 0
    for rule in policy.rules_for(target.alarm_type, source.alarm_type):
        selected |= RULE_BITS[rule]
    return bool(mask & selected)


def build_candidate_indices(
    period_types,
    scorer,
    *,
    entities=None,
    alarm_types=None,
):
    """Build reusable inverted indexes for every supported candidate rule."""
    period_types = tuple(
        sorted(period_types or (), key=lambda value: (value.entity, value.alarm_type))
    )
    indexes = {
        rule: defaultdict(list)
        for rule in (
            RULE_SAME_SITE,
            RULE_SAME_VENDOR_NETYPE,
            RULE_SAME_VENDOR,
            RULE_SAME_NETYPE,
            RULE_SAME_DOMAIN,
        )
    }
    geo_cells = {RULE_GEO_NEAR: defaultdict(list), RULE_GEO_METRO: defaultdict(list)}
    node_entities = defaultdict(list)
    attrs = {}
    if alarm_types is None:
        alarm_types = tuple(
            sorted({str(value.alarm_type) for value in period_types})
        )
    else:
        alarm_types = tuple(sorted({str(value) for value in alarm_types}))
    if entities is None:
        entity_values = []
        previous = None
        for value in period_types:
            if value.entity != previous:
                entity_values.append(value.entity)
                previous = value.entity
        entities = tuple(entity_values)
    else:
        entities = tuple(dict.fromkeys(str(value) for value in entities))
    for entity in entities:
        attr = _entity_attributes(entity, scorer)
        attrs[entity] = attr
        keys = {
            RULE_SAME_SITE: attr.site,
            RULE_SAME_VENDOR_NETYPE: (
                (attr.vendor, attr.ne_type) if attr.vendor and attr.ne_type else None
            ),
            RULE_SAME_VENDOR: attr.vendor,
            RULE_SAME_NETYPE: attr.ne_type,
            RULE_SAME_DOMAIN: attr.domain,
        }
        for rule, key in keys.items():
            if key:
                indexes[rule][key].append(entity)
        if attr.node:
            node_entities[attr.node].append(entity)
        if attr.cell_near is not None:
            geo_cells[RULE_GEO_NEAR][attr.cell_near].append(entity)
        if attr.cell_metro is not None:
            geo_cells[RULE_GEO_METRO][attr.cell_metro].append(entity)

    hops = (
        getattr(scorer.topology_index, "undirected_hops", {}) or {}
        if scorer.topology_index is not None
        else {}
    )
    topology_nodes = defaultdict(set)
    for left, row in hops.items():
        for right, distance in (row or {}).items():
            if distance:
                topology_nodes[left].add(right)
                topology_nodes[right].add(left)
    return {
        "period_types": period_types,
        "alarm_types": alarm_types,
        "entities": entities,
        "indexes": indexes,
        "geo_cells": geo_cells,
        "node_entities": node_entities,
        "topology_nodes": topology_nodes,
        "attributes": attrs,
    }


def rule_candidates(target, source_alarm_type, rule, prepared):
    attr = prepared["attributes"].get(target.entity)
    if attr is None:
        return ()
    if rule == RULE_SAME_ENTITY:
        return (target.entity,)
    if rule == RULE_SAME_NODE:
        return prepared["node_entities"].get(attr.node, ())
    if rule == RULE_TOPOLOGY:
        out = set()
        for neighbor in prepared["topology_nodes"].get(attr.node, ()):
            out.update(prepared["node_entities"].get(neighbor, ()))
        return out
    if rule in (RULE_GEO_NEAR, RULE_GEO_METRO):
        cell = attr.cell_near if rule == RULE_GEO_NEAR else attr.cell_metro
        cells = prepared["geo_cells"][rule]
        out = set()
        for neighbor_cell in _geo_neighbor_cells(cell):
            out.update(cells.get(neighbor_cell, ()))
        return out
    key = {
        RULE_SAME_SITE: attr.site,
        RULE_SAME_VENDOR_NETYPE: (
            (attr.vendor, attr.ne_type) if attr.vendor and attr.ne_type else None
        ),
        RULE_SAME_VENDOR: attr.vendor,
        RULE_SAME_NETYPE: attr.ne_type,
        RULE_SAME_DOMAIN: attr.domain,
    }.get(rule)
    if not key:
        return ()
    return prepared["indexes"][rule].get(key, ())


def _related_entity_set(target, prepared):
    """Entities the related predicate would retrieve for ``target``.

    Related rules are entity-level (independent of the source alarm type), so
    this mirrors ``candidate_rule_mask(...) & RELATED_MASK`` and lets the
    enumeration exclude exactly the pairs the related branch already owns.
    """
    related = set()
    for rule in RELATED_RULES:
        related.update(rule_candidates(target, None, rule, prepared))
    return related


def _related_entity_set_cached(target, prepared):
    """``_related_entity_set`` memoized by target entity.

    The related set depends only on ``target.entity`` (the related rules ignore
    the source alarm type), so every alarm type of the same entity shares it.
    The cache lives in ``prepared`` and is populated lazily; it is a pure
    function of the immutable candidate indexes, so results are unchanged.
    """
    cache = prepared.get("_related_cache")
    if cache is None:
        return _related_entity_set(target, prepared)
    related = cache.get(target.entity)
    if related is None:
        related = _related_entity_set(target, prepared)
        cache[target.entity] = related
    return related


def adaptive_candidate_sources(target, policy, prepared, exclude_related=False):
    candidates = set()
    period_type_class = target.__class__
    related = _related_entity_set_cached(target, prepared) if exclude_related else ()
    for source_at in prepared["alarm_types"]:
        candidate_entities = set()
        for rule in policy.rules_for(target.alarm_type, source_at):
            candidate_entities.update(
                rule_candidates(target, source_at, rule, prepared)
            )
        if exclude_related:
            candidate_entities -= related
        candidates.update(
            period_type_class(entity, source_at) for entity in candidate_entities
        )
    return tuple(sorted(candidates, key=lambda value: (value.entity, value.alarm_type)))


def adaptive_candidate_count(target, policy, prepared, exclude_related=False):
    total = 0
    related = _related_entity_set_cached(target, prepared) if exclude_related else ()
    for source_at in prepared["alarm_types"]:
        candidate_entities = set()
        for rule in policy.rules_for(target.alarm_type, source_at):
            candidate_entities.update(
                rule_candidates(target, source_at, rule, prepared)
            )
        if exclude_related:
            candidate_entities -= related
        total += len(candidate_entities)
    return total


def prepare_adaptive_candidates(
    period_types, scorer, policy, count_pairs=True, exclude_related=False
):
    prepared = build_candidate_indices(period_types, scorer)
    prepared["adaptive"] = True
    prepared["global"] = False
    prepared["policy"] = policy
    prepared["exclude_related"] = bool(exclude_related)
    # Memoize the entity-level related set so every alarm type of an entity
    # reuses it instead of recomputing the related-rule union per target.
    prepared["_related_cache"] = {}
    total = None
    if count_pairs:
        total = sum(
            adaptive_candidate_count(
                target, policy, prepared, exclude_related=exclude_related
            )
            for target in prepared["period_types"]
        )
    prepared["total_pair_count"] = total
    return prepared
