import json

from pathlib import Path

from fault_grouping.matching.group_output_builder import build_jsonl_match_output
from fault_grouping.site_topology import build_site_to_ne_ids


def load_json_object(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_visual_group_record(
    cascade_match,
    ne_graph_data,
    site_graph_data,
    site_to_ne_ids=None,
    ne_link_info_cache=None,
):
    return build_jsonl_match_output(
        cascade_match,
        ne_graph_data,
        site_graph_data,
        alarm_metadata_index={},
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
    )


class CascadeVisualOutputSession:
    """Append finalized cascades in match_rules visualization JSONL format."""

    def __init__(self, output_path, ne_graph_data, site_graph_data):
        self.output_path = Path(output_path)
        self.ne_graph_data = ne_graph_data
        self.site_graph_data = site_graph_data
        self.site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
        self.ne_link_info_cache = {}
        self.emitted_cascade_ids = set()
        self.emitted_count = 0
        self._handle = None

    @classmethod
    def from_files(cls, output_path, ne_graph_path, site_graph_path):
        return cls(
            output_path=output_path,
            ne_graph_data=load_json_object(ne_graph_path),
            site_graph_data=load_json_object(site_graph_path),
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

    def emit_closed(self, engine, now_ts=None):
        return self._emit_matches(
            engine.cascade_visual_matches(now_ts=now_ts, states={"closed"}),
            finalization_reason="closed",
        )

    def emit_remaining(self, engine, now_ts=None):
        return self._emit_matches(
            engine.cascade_visual_matches(now_ts=now_ts),
            finalization_reason="stream_end",
        )

    def _emit_matches(self, matches, finalization_reason):
        writable_matches = [
            match
            for match in matches
            if match.get("uuid") and match["uuid"] not in self.emitted_cascade_ids
        ]
        if not writable_matches:
            return 0

        if self._handle is None:
            self.reset_output_file()

        emitted = 0
        for match in writable_matches:
            output_match = dict(match)
            cascade_info = dict(output_match.get("cascade_info") or {})
            cascade_info["finalization_reason"] = finalization_reason
            output_match["cascade_info"] = cascade_info
            record = build_visual_group_record(
                output_match,
                self.ne_graph_data,
                self.site_graph_data,
                site_to_ne_ids=self.site_to_ne_ids,
                ne_link_info_cache=self.ne_link_info_cache,
            )
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.emitted_cascade_ids.add(match["uuid"])
            emitted += 1
        self._handle.flush()
        self.emitted_count += emitted
        return emitted
