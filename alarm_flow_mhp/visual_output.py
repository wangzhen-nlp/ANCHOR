"""MHP-flavoured visualization output (filterable by the missing-chain rule).

Mirrors ``alarm_flow_brunch.visual_output`` but tags every record with the MHP
rules (``alarm_flow_mhp`` / ``alarm_flow_mhp_virtual_event``) instead of the
BRUNCH ones, so a fault-group browser can filter imputed groups/edges by the
MHP missing rule without BRUNCH provenance leaking in.

It deliberately reuses BRUNCH's *rule-agnostic* topology helpers (direct-link /
hop / site lookups) and the shared ``build_jsonl_match_output`` record builder —
only the rule strings, edge provenance, and symptom ``matched_rule`` differ.
"""

from __future__ import annotations

import json
from pathlib import Path

from fault_grouping.matching.group_output_builder import build_jsonl_match_output
from fault_grouping.site_topology import build_site_to_ne_ids

# Reuse BRUNCH's rule-agnostic helpers (topology + virtual-flag mirroring).
from alarm_flow_brunch.visual_output import (
    VISUAL_NE_SCOPES,
    _attach_virtual_alarm_flags,
    _has_direct_ne_link,
    _shortest_ne_hops,
    _symptom_site_id,
    load_json_object,
)
from alarm_flow_mhp.missing_chain_sampler import MHP_RULE, MHP_VIRTUAL_RULE


def _classify_mhp_edge(ne_graph_data, source_symptom, target_symptom):
    """Topology relation of a parent→child edge, MHP-labelled. Same structure as
    BRUNCH's classifier; only the relation strings carry the MHP namespace."""
    source_ne = str(source_symptom.get("alarm_source", "") or "")
    target_ne = str(target_symptom.get("alarm_source", "") or "")
    source_site = _symptom_site_id(ne_graph_data, source_symptom)
    target_site = _symptom_site_id(ne_graph_data, target_symptom)
    if source_ne == target_ne:
        return "mhp_same_device", 0
    if source_site and source_site == target_site:
        return "mhp_same_site", None
    hops = _shortest_ne_hops(ne_graph_data, source_ne, target_ne, max_hops=3)
    if hops is not None and hops > 1:
        return "mhp_indirect_topology", hops
    if source_site and target_site and source_site != target_site:
        return "mhp_cross_site", None
    return "mhp_hawkes_unknown_context", None


def _mhp_propagation_edges(group, ne_graph_data):
    """Parent→child edges without a direct NE link, MHP-tagged. These surface the
    inferred propagation — including edges through imputed missing nodes."""
    if not ne_graph_data:
        return []
    symptoms_by_event_id = {
        str(s.get("event_id", "") or ""): s
        for s in group.get("symptoms") or []
        if s.get("event_id")
    }
    edges = []
    seen = set()
    for edge in group.get("edges") or []:
        src_id = str(edge.get("source_event_id", "") or "")
        tgt_id = str(edge.get("target_event_id", "") or "")
        src = symptoms_by_event_id.get(src_id)
        tgt = symptoms_by_event_id.get(tgt_id)
        if not src or not tgt:
            continue
        src_ne = str(src.get("alarm_source", "") or "")
        tgt_ne = str(tgt.get("alarm_source", "") or "")
        if not src_ne or not tgt_ne:
            continue
        if _has_direct_ne_link(ne_graph_data, src_ne, tgt_ne):
            continue
        relation, hops = _classify_mhp_edge(ne_graph_data, src, tgt)
        key = (src_id, tgt_id, src_ne, tgt_ne, relation)
        if key in seen:
            continue
        seen.add(key)
        src_ts, tgt_ts = src.get("ts"), tgt.get("ts")
        edges.append({
            "source": _symptom_site_id(ne_graph_data, src),
            "target": _symptom_site_id(ne_graph_data, tgt),
            "source_site": _symptom_site_id(ne_graph_data, src),
            "target_site": _symptom_site_id(ne_graph_data, tgt),
            "source_ne": src_ne,
            "target_ne": tgt_ne,
            "source_event_id": src_id,
            "target_event_id": tgt_id,
            "source_alarm": src.get("alarm_title", ""),
            "target_alarm": tgt.get("alarm_title", ""),
            "source_type": edge.get("source_type", ""),
            "target_type": edge.get("target_type", ""),
            "relation": relation,
            "predicted_relation": relation,
            "score": edge.get("score", ""),
            "inferred_hops": hops or 0,
            "dt_sec": (
                float(tgt_ts) - float(src_ts)
                if src_ts is not None and tgt_ts is not None else None
            ),
            # provenance: which side of the edge is an imputed (missing) node
            "source_virtual": bool(edge.get("source_virtual")),
            "target_virtual": bool(edge.get("target_virtual")),
            "edge_source": MHP_RULE,
            "description": "MHP imputed parent-child propagation edge (may pass through an unobserved/missing event)",
        })
    return edges


def _symptom_to_visual_record_mhp(symptom):
    """Per-symptom visual record. ``matched_rule`` reflects the MHP namespace and
    marks imputed nodes with the missing rule so node-level filtering works."""
    is_virtual = bool(symptom.get("virtual", False))
    return {
        "node": symptom.get("site_id", ""),
        "alarm_source": symptom.get("alarm_source", ""),
        "alarm": symptom.get("alarm_title", ""),
        "alarm_type": symptom.get("alarm_type", ""),
        "ts": symptom.get("ts"),
        "eid": symptom.get("event_id", ""),
        "virtual": is_virtual,
        "latent": bool(symptom.get("latent", False)),
        "confidence": symptom.get("confidence", 1.0),
        "virtual_source": symptom.get("virtual_source", ""),
        "matched_rule": MHP_VIRTUAL_RULE if is_virtual else MHP_RULE,
        "matched_role": "cascade",
        "matched_role_key": "cascade",
    }


def group_to_visual_match_mhp(group, ne_graph_data=None):
    root_event = dict(group.get("root_event") or {})
    root_event_id = root_event.get("event_id", "")
    inferred_roots = {"cascade": root_event_id} if root_event_id else {}
    prop_edges = _mhp_propagation_edges(group, ne_graph_data)
    merged_rules = list(group.get("merged_rules") or [])
    if not merged_rules and int(group.get("virtual_event_count", 0) or 0) > 0:
        merged_rules = [MHP_RULE, MHP_VIRTUAL_RULE]
    return {
        "uuid": group.get("group_id", ""),
        "rule": group.get("rule") or MHP_RULE,
        "merged_rules": merged_rules,
        "related_group_uuids": [],
        "inferred_roots": inferred_roots,
        "role_mapping": {"cascade": list(group.get("site_list") or [])},
        "uses_missing_topology": bool(prop_edges),
        "missing_topology_edges": prop_edges,
        "symptoms": [_symptom_to_visual_record_mhp(s) for s in group.get("symptoms") or []],
        "cascade_info": {
            "cascade_id": group.get("cascade_id"),
            "event_count": group.get("event_count", 0),
            "real_event_count": group.get("real_event_count", 0),
            "virtual_event_count": group.get("virtual_event_count", 0),
            "start_ts": group.get("start_ts"),
            "end_ts": group.get("end_ts"),
            "duration_sec": group.get("duration_sec", 0.0),
            "alarm_sources": list(group.get("alarm_source_list") or []),
            "sites": list(group.get("site_list") or []),
            "alarm_title_counts": dict(group.get("alarm_title_counts") or {}),
            "alarm_type_counts": dict(group.get("alarm_type_counts") or {}),
            "root_event_id": root_event_id,
        },
    }


def write_visual_groups_mhp(output_path, groups, *, ne_graph_path, site_graph_path,
                            ne_scope="alarm-only"):
    """Write MHP groups as match-rules visualization JSONL. Same on-disk shape as
    BRUNCH's writer (so the browser reads it unchanged) but MHP-tagged."""
    if ne_scope not in VISUAL_NE_SCOPES:
        raise ValueError(f"unsupported visual NE scope: {ne_scope}")
    ne_graph_data = load_json_object(ne_graph_path)
    site_graph_data = load_json_object(site_graph_path)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data) if ne_scope == "site-context" else {}
    ne_link_info_cache = {}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            match = group_to_visual_match_mhp(group, ne_graph_data=ne_graph_data)
            if not match.get("uuid"):
                continue
            record = build_jsonl_match_output(
                match,
                ne_graph_data,
                site_graph_data,
                alarm_metadata_index={},
                site_to_ne_ids=site_to_ne_ids,
                ne_link_info_cache=ne_link_info_cache,
            )
            _attach_virtual_alarm_flags(record)   # rule-agnostic: copies virtual flags
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
