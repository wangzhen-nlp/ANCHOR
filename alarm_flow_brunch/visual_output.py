from __future__ import annotations

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
        "matched_rule": "alarm_flow_brunch",
        "matched_role": "cascade",
        "matched_role_key": "cascade",
    }


def group_to_visual_match(group):
    root_event = dict(group.get("root_event") or {})
    root_event_id = root_event.get("event_id", "")
    inferred_roots = {}
    if root_event_id:
        inferred_roots["cascade"] = root_event_id
    return {
        "uuid": group.get("group_id", ""),
        "rule": "alarm_flow_brunch",
        "merged_rules": [],
        "related_group_uuids": [],
        "inferred_roots": inferred_roots,
        "role_mapping": {"cascade": list(group.get("site_list") or [])},
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
            match = group_to_visual_match(group)
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
            match = group_to_visual_match(group)
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
