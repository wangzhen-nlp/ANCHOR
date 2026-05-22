from alarm_cascade_dhp.config import AlarmDHPConfig, StreamPolicyConfig
from alarm_cascade_dhp.features import AlarmFeatureBuilder
from alarm_cascade_dhp.model import TopologyPoweredDHP
from alarm_cascade_dhp.streaming import AlarmStreamSanitizer
from alarm_cascade_dhp.topology import TopologyIndex
from alarm_cascade_dhp.event_types import CascadeDecision


class AlarmCascadeEngine:
    """Streaming facade for topology-aware DHP alarm cascade clustering."""

    def __init__(self, model_config=None, stream_config=None, topology=None):
        self.model_config = model_config or AlarmDHPConfig()
        self.stream_config = stream_config or StreamPolicyConfig()
        self.topology = topology or TopologyIndex()
        self.features = AlarmFeatureBuilder(
            topology=self.topology,
            topology_context_hops=self.model_config.topology_context_hops,
            topology_context_limit=self.model_config.topology_context_limit,
        )
        self.sanitizer = AlarmStreamSanitizer(self.stream_config)
        self.model = TopologyPoweredDHP(self.model_config, self.topology)

    @classmethod
    def from_topology_files(
        cls,
        site_graph_path="",
        ne_graph_path="",
        model_config=None,
        stream_config=None,
    ):
        return cls(
            model_config=model_config,
            stream_config=stream_config,
            topology=TopologyIndex.from_files(site_graph_path, ne_graph_path),
        )

    def observe_match_rules_item(self, item):
        """Accept the normalized item shape used by fault_grouping.match_rules."""
        return self.observe_event(self.features.from_match_rules_item(item))

    def observe_alarm_record(self, raw_record):
        """Accept a raw alarm row from CSV, JSONL, or a live alarm source."""
        return self.observe_event(self.features.from_alarm_record(raw_record))

    def observe(self, item):
        """Accept either a raw alarm dict or a match_rules normalized item."""
        if _looks_like_match_rules_item(item):
            return self.observe_match_rules_item(item)
        return self.observe_alarm_record(item)

    def observe_event(self, event):
        return self._consume_sanitized(self.sanitizer.push(event))

    def flush(self):
        """Release events still held in the event-time reorder buffer."""
        return self._consume_sanitized(self.sanitizer.flush())

    def cascade_snapshots(self, now_ts=None):
        return self.model.cascade_snapshots(now_ts=now_ts)

    def progress_snapshot(self):
        return {
            "cascade_count": self.model.cascade_count(),
            "pending_event_count": self.sanitizer.pending_count(),
            "last_clustered_ts": self.model.last_ts,
        }

    def _consume_sanitized(self, sanitized_events):
        decisions = []
        for sanitized in sanitized_events:
            if sanitized.action == "raise":
                decisions.append(self.model.observe_raise(sanitized.event))
            elif sanitized.action == "clear":
                decisions.append(self.model.observe_clear(sanitized.event))
            else:
                decisions.append(
                    CascadeDecision(
                        status="skipped",
                        event=sanitized.event,
                        reason=sanitized.reason or "stream_policy_skip",
                    )
                )
        return decisions


def _looks_like_match_rules_item(item):
    return (
        isinstance(item, dict)
        and isinstance(item.get("alarm"), dict)
        and "ts" in item
    )
