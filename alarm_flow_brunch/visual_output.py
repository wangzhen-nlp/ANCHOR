from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from fault_grouping.matching.group_output_builder import build_jsonl_match_output
from fault_grouping.site_topology import build_site_to_ne_ids


VISUAL_NE_SCOPES = frozenset({"alarm-only", "site-context"})


def load_json_object(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _symptom_to_visual_record(symptom):
    return {
        "node": symptom.get("site_id", ""),
        "alarm_source": symptom.get("alarm_source", ""),
        "alarm": symptom.get("alarm_title", ""),
        "alarm_type": symptom.get("alarm_type", ""),
        "ts": symptom.get("ts"),
        "eid": symptom.get("event_id", ""),
        "virtual": bool(symptom.get("virtual", False)),
        "latent": bool(symptom.get("latent", False)),
        "confidence": symptom.get("confidence", 1.0),
        "virtual_source": symptom.get("virtual_source", ""),
        "matched_rule": "alarm_flow_brunch",
        "matched_role": "cascade",
        "matched_role_key": "cascade",
    }


def _link_neighbors(ne_graph_data, ne_id):
    entry = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    links = entry.get("link", {}) if isinstance(entry, dict) else {}
    if not isinstance(links, dict):
        return set()
    return set(str(neighbor_id) for neighbor_id in links.keys() if str(neighbor_id))


def _ne_site_id(ne_graph_data, ne_id):
    entry = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    if not isinstance(entry, dict):
        return ""
    for key in ("site_id", "siteId", "site", "site_name", "siteName"):
        value = str(entry.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _symptom_site_id(ne_graph_data, symptom):
    site_id = str(symptom.get("site_id", "") or "").strip()
    if site_id:
        return site_id
    return _ne_site_id(ne_graph_data, str(symptom.get("alarm_source", "") or ""))


def _has_direct_ne_link(ne_graph_data, source_ne, target_ne):
    if not source_ne or not target_ne or source_ne == target_ne:
        return source_ne == target_ne
    return (
        target_ne in _link_neighbors(ne_graph_data, source_ne)
        or source_ne in _link_neighbors(ne_graph_data, target_ne)
    )


def _shortest_ne_hops(ne_graph_data, source_ne, target_ne, max_hops=3):
    if not source_ne or not target_ne or source_ne == target_ne:
        return 0 if source_ne == target_ne else None
    seen = {source_ne}
    queue = deque([(source_ne, 0)])
    while queue:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for neighbor in _link_neighbors(ne_graph_data, node):
            if neighbor == target_ne:
                return hops + 1
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, hops + 1))
    return None


def _classify_brunch_propagation_edge(ne_graph_data, source_symptom, target_symptom):
    source_ne = str(source_symptom.get("alarm_source", "") or "")
    target_ne = str(target_symptom.get("alarm_source", "") or "")
    source_site = _symptom_site_id(ne_graph_data, source_symptom)
    target_site = _symptom_site_id(ne_graph_data, target_symptom)
    if source_ne == target_ne:
        return "brunch_same_device", 0
    if source_site and source_site == target_site:
        return "brunch_same_site", None
    hops = _shortest_ne_hops(ne_graph_data, source_ne, target_ne, max_hops=3)
    if hops is not None and hops > 1:
        return "brunch_indirect_topology", hops
    if source_site and target_site and source_site != target_site:
        return "brunch_hawkes_cross_site", None
    return "brunch_hawkes_unknown_context", None


def _brunch_missing_topology_edges(group, ne_graph_data):
    if not ne_graph_data:
        return []
    symptoms_by_event_id = {
        str(symptom.get("event_id", "") or ""): symptom
        for symptom in group.get("symptoms") or []
        if symptom.get("event_id")
    }
    missing_edges = []
    seen = set()
    for edge in group.get("edges") or []:
        source_event_id = str(edge.get("source_event_id", "") or "")
        target_event_id = str(edge.get("target_event_id", "") or "")
        source_symptom = symptoms_by_event_id.get(source_event_id)
        target_symptom = symptoms_by_event_id.get(target_event_id)
        if not source_symptom or not target_symptom:
            continue
        source_ne = str(source_symptom.get("alarm_source", "") or "")
        target_ne = str(target_symptom.get("alarm_source", "") or "")
        if not source_ne or not target_ne:
            continue
        if _has_direct_ne_link(ne_graph_data, source_ne, target_ne):
            continue

        relation, hops = _classify_brunch_propagation_edge(
            ne_graph_data,
            source_symptom,
            target_symptom,
        )
        key = (source_event_id, target_event_id, source_ne, target_ne, relation)
        if key in seen:
            continue
        seen.add(key)
        source_ts = source_symptom.get("ts")
        target_ts = target_symptom.get("ts")
        source_site = _symptom_site_id(ne_graph_data, source_symptom)
        target_site = _symptom_site_id(ne_graph_data, target_symptom)
        missing_edges.append(
            {
                "source": source_site,
                "target": target_site,
                "source_site": source_site,
                "target_site": target_site,
                "source_ne": source_ne,
                "target_ne": target_ne,
                "source_event_id": source_event_id,
                "target_event_id": target_event_id,
                "source_alarm": source_symptom.get("alarm_title", ""),
                "target_alarm": target_symptom.get("alarm_title", ""),
                "source_type": edge.get("source_type", ""),
                "target_type": edge.get("target_type", ""),
                "relation": relation,
                "predicted_relation": relation,
                "score": "",
                "inferred_hops": hops or 0,
                "dt_sec": (
                    float(target_ts) - float(source_ts)
                    if source_ts is not None and target_ts is not None
                    else None
                ),
                "edge_source": "alarm_flow_brunch",
                "description": "BRUNCH inferred parent-child propagation edge without direct NE topology link",
            }
        )
    return missing_edges


def group_to_visual_match(group, ne_graph_data=None):
    root_event = dict(group.get("root_event") or {})
    root_event_id = root_event.get("event_id", "")
    inferred_roots = {}
    if root_event_id:
        inferred_roots["cascade"] = root_event_id
    missing_topology_edges = _brunch_missing_topology_edges(group, ne_graph_data)
    return {
        "uuid": group.get("group_id", ""),
        "rule": "alarm_flow_brunch",
        "merged_rules": [],
        "related_group_uuids": [],
        "inferred_roots": inferred_roots,
        "role_mapping": {"cascade": list(group.get("site_list") or [])},
        "uses_missing_topology": bool(missing_topology_edges),
        "missing_topology_edges": missing_topology_edges,
        "symptoms": [
            _symptom_to_visual_record(symptom)
            for symptom in group.get("symptoms") or []
        ],
        "cascade_info": {
            "cascade_id": group.get("cascade_id"),
            "event_count": group.get("event_count", 0),
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


def write_visual_groups(
    output_path,
    groups,
    *,
    ne_graph_path,
    site_graph_path,
    ne_scope="alarm-only",
):
    if ne_scope not in VISUAL_NE_SCOPES:
        raise ValueError(f"unsupported visual NE scope: {ne_scope}")
    ne_graph_data = load_json_object(ne_graph_path)
    site_graph_data = load_json_object(site_graph_path)
    site_to_ne_ids = (
        build_site_to_ne_ids(ne_graph_data)
        if ne_scope == "site-context"
        else {}
    )
    ne_link_info_cache = {}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            match = group_to_visual_match(group, ne_graph_data=ne_graph_data)
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
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


class AlarmBRUNCHVisualOutputSession:
    """Append BRUNCH online cascades in match_rules visualization JSONL format."""

    def __init__(self, output_path, ne_graph_data, site_graph_data, ne_scope="alarm-only"):
        if ne_scope not in VISUAL_NE_SCOPES:
            raise ValueError(f"unsupported visual NE scope: {ne_scope}")
        self.output_path = Path(output_path)
        self.ne_graph_data = ne_graph_data
        self.site_graph_data = site_graph_data
        self.ne_scope = ne_scope
        self.site_to_ne_ids = (
            build_site_to_ne_ids(ne_graph_data)
            if ne_scope == "site-context"
            else {}
        )
        self.ne_link_info_cache = {}
        self.emitted_group_ids = set()
        self.emitted_count = 0
        self._handle = None

    @classmethod
    def from_files(cls, output_path, ne_graph_path, site_graph_path, ne_scope="alarm-only"):
        return cls(
            output_path=output_path,
            ne_graph_data=load_json_object(ne_graph_path),
            site_graph_data=load_json_object(site_graph_path),
            ne_scope=ne_scope,
        )

    def reset_output_file(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.close()
        self._handle = self.output_path.open("w", encoding="utf-8")

    def close(self):
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    def emit_groups(self, groups, *, finalization_reason):
        writable_groups = [
            group
            for group in groups
            if group.get("group_id") and group["group_id"] not in self.emitted_group_ids
        ]
        if not writable_groups:
            return 0
        if self._handle is None:
            self.reset_output_file()

        emitted = 0
        for group in writable_groups:
            match = group_to_visual_match(group, ne_graph_data=self.ne_graph_data)
            cascade_info = dict(match.get("cascade_info") or {})
            cascade_info["finalization_reason"] = finalization_reason
            match["cascade_info"] = cascade_info
            record = build_jsonl_match_output(
                match,
                self.ne_graph_data,
                self.site_graph_data,
                alarm_metadata_index={},
                site_to_ne_ids=self.site_to_ne_ids,
                ne_link_info_cache=self.ne_link_info_cache,
            )
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.emitted_group_ids.add(group["group_id"])
            emitted += 1
        self._handle.flush()
        self.emitted_count += emitted
        return emitted
