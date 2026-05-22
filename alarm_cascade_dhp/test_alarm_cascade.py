import json
import tempfile
import unittest

from pathlib import Path

from alarm_cascade_dhp.config import AlarmDHPConfig, StreamPolicyConfig
from alarm_cascade_dhp.engine import AlarmCascadeEngine
from alarm_cascade_dhp.features import AlarmFeatureBuilder
from alarm_cascade_dhp.profiling import PhaseTimer, enable_engine_profiling
from alarm_cascade_dhp.run_cascades import _load_sorted_events, _resolve_groups_output
from alarm_cascade_dhp.streaming import AlarmStreamSanitizer
from alarm_cascade_dhp.topology import TopologyIndex
from alarm_cascade_dhp.visual_output import CascadeVisualOutputSession


def _alarm(event_id, ts, title, source, site, **extra):
    alarm = {
        "event_id": event_id,
        "ts": ts,
        "告警标题": title,
        "告警源": source,
        "站点ID": site,
        "告警码": extra.pop("alarm_code", "alarm-code"),
    }
    alarm.update(extra)
    return alarm


class AlarmFeatureTests(unittest.TestCase):
    def test_features_include_alarm_and_topology_tokens(self):
        topology = TopologyIndex(
            site_graph={"site-a": ["site-b"]},
            ne_graph={"ne-a": {"site_id": "site-a", "domain": "wireless", "type": "BBU"}},
        )
        event = AlarmFeatureBuilder(topology=topology).from_alarm_record(
            _alarm("a1", 10, "传输链路中断", "ne-a", "")
        )

        self.assertEqual(event.site_id, "site-a")
        self.assertIn("title:传输链路中断", event.feature_counts)
        self.assertIn("site:site-a", event.feature_counts)
        self.assertIn("device:ne-a", event.feature_counts)
        self.assertIn("device_domain:wireless", event.feature_counts)
        self.assertIn("device_type:BBU", event.feature_counts)
        self.assertIn("topo_site_hop_1:site-b", event.feature_counts)


class TopologyTests(unittest.TestCase):
    def test_relations_distinguish_neighbor_and_disconnected_sites(self):
        topology = TopologyIndex(site_graph={"site-a": ["site-b"], "site-c": []})
        features = AlarmFeatureBuilder(topology=topology)
        left = features.from_alarm_record(_alarm("a1", 10, "A", "ne-a", "site-a"))
        neighbor = features.from_alarm_record(_alarm("a2", 12, "B", "ne-b", "site-b"))
        disconnected = features.from_alarm_record(_alarm("a3", 14, "C", "ne-c", "site-c"))

        self.assertEqual(topology.relation(left, neighbor), "hop_1")
        self.assertEqual(topology.relation(left, disconnected), "disconnected")


class StreamPolicyTests(unittest.TestCase):
    def test_reorder_duplicate_clear_and_flap_controls(self):
        features = AlarmFeatureBuilder()
        sanitizer = AlarmStreamSanitizer(
            StreamPolicyConfig(
                reorder_lag_sec=2,
                late_tolerance_sec=0,
                duplicate_window_sec=10,
                flap_window_sec=10,
            )
        )
        raise_1 = features.from_alarm_record(_alarm("a1", 10, "A", "ne-a", "site-a"))
        raise_2 = features.from_alarm_record(_alarm("a2", 11, "A", "ne-a", "site-a"))
        clear = features.from_alarm_record(
            _alarm("a3", 12, "A", "ne-a", "site-a", **{"清除告警": "是"})
        )
        reopen = features.from_alarm_record(_alarm("a4", 13, "A", "ne-a", "site-a"))
        early = features.from_alarm_record(_alarm("a0", 9, "B", "ne-b", "site-b"))

        self.assertEqual(sanitizer.push(raise_1), [])
        self.assertEqual(sanitizer.push(raise_2), [])
        output = sanitizer.push(early)
        self.assertEqual([item.action for item in output], ["raise"])
        self.assertEqual(output[0].event.event_id, "a0")

        output = sanitizer.push(clear)
        self.assertEqual([item.action for item in output], ["raise"])
        self.assertEqual(output[0].event.event_id, "a1")

        output = sanitizer.push(reopen) + sanitizer.flush()
        self.assertEqual(
            [(item.action, item.reason) for item in output],
            [
                ("skip", "duplicate_raise_compressed"),
                ("clear", ""),
                ("skip", "flap_reopen_compressed"),
            ],
        )


class EngineTests(unittest.TestCase):
    def test_engine_clusters_related_match_rules_items_and_splits_far_alarm(self):
        topology = TopologyIndex(site_graph={"site-a": ["site-b"], "site-c": []})
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(
                particle_count=1,
                assignment_strategy="map",
                base_intensity=0.0001,
                topology_strength=2.0,
                active_window_sec=600,
                cooling_after_sec=600,
                close_after_sec=600,
            ),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
            topology=topology,
        )
        items = [
            {
                "alarm": _alarm("a1", 100, "光路中断", "ne-a", "site-a"),
                "site_id": "site-a",
                "alarm_source": "ne-a",
                "alarm_title": "光路中断",
                "ts": 100,
            },
            {
                "alarm": _alarm("a2", 105, "光路中断", "ne-b", "site-b"),
                "site_id": "site-b",
                "alarm_source": "ne-b",
                "alarm_title": "光路中断",
                "ts": 105,
            },
            {
                "alarm": _alarm("a3", 5000, "电源异常", "ne-c", "site-c"),
                "site_id": "site-c",
                "alarm_source": "ne-c",
                "alarm_title": "电源异常",
                "ts": 5000,
            },
        ]

        decisions = []
        for item in items:
            decisions.extend(engine.observe_match_rules_item(item))
        decisions.extend(engine.flush())

        clustered = [decision for decision in decisions if decision.status == "clustered"]
        self.assertEqual(len(clustered), 3)
        self.assertEqual(clustered[0].cascade_id, clustered[1].cascade_id)
        self.assertNotEqual(clustered[0].cascade_id, clustered[2].cascade_id)

    def test_engine_scores_only_recent_candidate_cascades_when_limited(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(
                particle_count=1,
                assignment_strategy="map",
                base_intensity=100.0,
                active_window_sec=600,
                cooling_after_sec=600,
                close_after_sec=600,
                max_candidate_cascades=1,
            ),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        decisions = []
        for event_id, ts, title, source, site in (
            ("a1", 100, "A", "ne-a", "site-a"),
            ("a2", 101, "B", "ne-b", "site-b"),
            ("a3", 102, "C", "ne-c", "site-c"),
        ):
            decisions.extend(
                engine.observe_alarm_record(_alarm(event_id, ts, title, source, site))
            )

        self.assertEqual(decisions[-1].candidate_count, 2)

    def test_progress_snapshot_tracks_reorder_buffer_and_cascades(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(particle_count=1, assignment_strategy="map"),
            stream_config=StreamPolicyConfig(reorder_lag_sec=10),
        )

        engine.observe_alarm_record(_alarm("a1", 100, "A", "ne-a", "site-a"))
        pending = engine.progress_snapshot()
        self.assertEqual(pending["pending_event_count"], 1)
        self.assertEqual(pending["cascade_count"], 0)

        engine.flush()
        flushed = engine.progress_snapshot()
        self.assertEqual(flushed["pending_event_count"], 0)
        self.assertEqual(flushed["cascade_count"], 1)

    def test_profiling_records_feature_stream_and_model_phases(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(particle_count=1, assignment_strategy="map"),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        timer = PhaseTimer()
        enable_engine_profiling(timer, engine)

        engine.observe_alarm_record(_alarm("a1", 100, "A", "ne-a", "site-a"))
        phases = timer.phase_snapshot()

        self.assertIn("features.from_alarm_record", phases)
        self.assertIn("stream.sanitizer_push", phases)
        self.assertIn("model.observe_raise", phases)
        self.assertIn("update.cluster_add", phases)

    def test_cli_event_loader_sorts_raw_alarm_records_by_event_time(self):
        class _Args:
            alarms = "unused"
            show_progress = False

        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(particle_count=1, assignment_strategy="map"),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        unsorted_records = iter(
            [
                _alarm("a3", 300, "C", "ne-c", "site-c"),
                _alarm("a1", 100, "A", "ne-a", "site-a"),
                _alarm("a2", 200, "B", "ne-b", "site-b"),
            ]
        )

        from unittest.mock import patch

        with patch(
            "alarm_cascade_dhp.run_cascades._iter_input_alarm_records",
            return_value=unsorted_records,
        ):
            events = _load_sorted_events(_Args(), engine)

        self.assertEqual([event.event_id for event in events], ["a1", "a2", "a3"])

    def test_visual_output_disables_implicit_group_snapshot(self):
        class _Args:
            output = "decisions.jsonl"
            groups_output = ""
            visual_output = "groups.jsonl"

        self.assertEqual(_resolve_groups_output(_Args()), "")

        _Args.groups_output = "snapshots.json"
        self.assertEqual(_resolve_groups_output(_Args()), "snapshots.json")

        _Args.groups_output = ""
        _Args.visual_output = ""
        self.assertEqual(_resolve_groups_output(_Args()), "decisions.groups.json")


class VisualOutputTests(unittest.TestCase):
    def test_closed_visual_output_is_match_rules_compatible_and_written_once(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(
                particle_count=1,
                assignment_strategy="map",
                close_after_sec=10,
            ),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        engine.observe_alarm_record(
            _alarm("a1", 100, "链路中断", "ne-a", "site-a", **{"工单号": "ticket-a"})
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "cascade_visual.jsonl"
            session = CascadeVisualOutputSession(
                output_path,
                ne_graph_data={
                    "ne-a": {"site_id": "site-a", "name": "NE A"},
                    "ne-idle": {"site_id": "site-a", "name": "Idle NE"},
                },
                site_graph_data={"site-a": {"site_name": "Site A"}},
            )
            session.reset_output_file()

            self.assertEqual(session.emit_closed(engine, now_ts=105), 0)
            self.assertEqual(session.emit_closed(engine, now_ts=120), 1)
            self.assertEqual(session.emit_closed(engine, now_ts=120), 0)
            session.close()

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["uuid"], "cascade-1")
        self.assertEqual(record["symptoms"][0]["eid"], "a1")
        self.assertEqual(record["symptoms"][0]["工单号"], "ticket-a")
        self.assertEqual(record["match_info"]["uuid"], "cascade-1")
        self.assertIn("cascade-1", record["group_info"])
        self.assertIn("ne-a", record["ne_info"])
        self.assertNotIn("ne-idle", record["ne_info"])
        self.assertEqual(record["group_info"]["cascade-1"]["ne_list"], ["ne-a"])
        self.assertEqual(record["cascade_info"]["finalization_reason"], "closed")

    def test_stream_end_visual_output_emits_unclosed_cascades(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(particle_count=1, assignment_strategy="map"),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        engine.observe_alarm_record(_alarm("a1", 100, "A", "ne-a", "site-a"))

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "cascade_visual.jsonl"
            session = CascadeVisualOutputSession(
                output_path,
                ne_graph_data={"ne-a": {"site_id": "site-a"}},
                site_graph_data={"site-a": {}},
            )
            session.reset_output_file()

            self.assertEqual(session.emit_remaining(engine), 1)
            session.close()
            record = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["cascade_info"]["state"], "active")
        self.assertEqual(record["cascade_info"]["finalization_reason"], "stream_end")

    def test_site_context_visual_output_includes_quiet_devices_at_cascade_sites(self):
        engine = AlarmCascadeEngine(
            model_config=AlarmDHPConfig(particle_count=1, assignment_strategy="map"),
            stream_config=StreamPolicyConfig(reorder_lag_sec=0),
        )
        engine.observe_alarm_record(_alarm("a1", 100, "A", "ne-a", "site-a"))

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "cascade_visual.jsonl"
            session = CascadeVisualOutputSession(
                output_path,
                ne_graph_data={
                    "ne-a": {"site_id": "site-a"},
                    "ne-quiet": {"site_id": "site-a"},
                },
                site_graph_data={"site-a": {}},
                ne_scope="site-context",
            )
            session.reset_output_file()
            session.emit_remaining(engine)
            session.close()
            record = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(
            record["group_info"]["cascade-1"]["ne_list"],
            ["ne-a", "ne-quiet"],
        )
        self.assertIn("ne-quiet", record["ne_info"])


if __name__ == "__main__":
    unittest.main()
