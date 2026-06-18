import collections
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from alarm_flow_brunch.aggregator import summarize_alarm_event as summarize_brunch_alarm_event
from alarm_flow_brunch.visual_output import (
    _brunch_missing_topology_edges,
    _symptom_to_visual_record,
)
from alarm_flow_mhp.aggregator import summarize_alarm_event as summarize_mhp_alarm_event
from alarm_flow_mhp.missing_chain_sampler import MissingChainSampler
from alarm_flow_mhp.visual_output import (
    _mhp_propagation_edges,
    _symptom_to_visual_record_mhp,
)
from alarm_tools.check_duplicate_eid import check_duplicate_eid
from alarm_tools.count_alarms import count_alarms
from fault_grouping.alarm_events.identity import (
    alarm_content_uuid,
    require_alarm_identity,
)
from fault_grouping.alarm_events.io import load_valid_alarms
from fault_grouping.alarm_events.sorted_cache import (
    is_sorted_alarm_cache_file,
    load_sorted_alarm_cache,
    write_sorted_alarm_cache,
)
from fault_grouping.matching.group_output_builder import (
    build_alarm_metadata_index,
    enrich_match_symptoms,
)
from fault_grouping.node_rule_helper import NodeRuleHelper
from fault_grouping.temporal_engine.alarm_period import TemporalGraphEngineAlarmPeriodMixin
from fault_grouping.temporal_engine.engine import TemporalGraphEngine
from fault_grouping.temporal_engine.utils import get_match_alarm_keys, merge_match_batch
from fault_grouping.tools.alarm_group_baseline import alarm_to_baseline_symptom
from fault_csm_claude.engine import IncrementalFaultEngine
from fault_csm_codex.engine import ActiveAlarmIndex, CSMGroupStore
from microwave_topic.group_alarms_by_fault_group import group_alarms
from ticket_recall.evaluation.compute_group_output_ticket_recall import (
    _build_debug_alarm_group_lookup as build_group_output_debug_alarm_lookup,
)
from ticket_recall.evaluation.compute_ticket_site_recall import (
    _build_debug_alarm_group_lookup as build_ticket_site_debug_alarm_lookup,
)
from ticket_recall.ticket_recall_utils import dedupe_alarm_records


def occurrence(label):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"occurrence-contract-test:{label}"))


def symptom(eid, occurrence_uuid, ts=1, **extra):
    return {
        "node": "S1",
        "site_id": "S1",
        "alarm_source": "NE1",
        "alarm": "A",
        "alarm_title": "A",
        "event_id": eid,
        "eid": eid,
        "ts": ts,
        "occurrence_uuid": occurrence_uuid,
        **extra,
    }


class DummyAlarmPeriodEngine(TemporalGraphEngineAlarmPeriodMixin):
    def __init__(self):
        self.use_alarm_period_cache = True
        self.active_alarm_periods = collections.defaultdict(dict)
        self.active_event_to_period = collections.defaultdict(dict)
        self.event_cache = collections.defaultdict(collections.deque)

    def _get_event_ttl(self, _alarm_type):
        return 3600

    def _log_debug_event_removal(self, *_args, **_kwargs):
        return None


class AlarmOccurrenceIdentityContractTest(unittest.TestCase):
    def test_input_uuid_is_stable_for_canonical_alarm_content(self):
        alarm = {"eid": "E", "nested": {"b": 2, "a": 1}}
        reordered = {"nested": {"a": 1, "b": 2}, "eid": "E"}
        first = alarm_content_uuid(alarm)
        self.assertEqual(first, alarm_content_uuid(reordered))
        self.assertEqual(first, alarm_content_uuid({**alarm, "occurrence_uuid": occurrence("ignored")}))
        self.assertNotEqual(first, alarm_content_uuid({**alarm, "eid": "OTHER"}))
        uuid.UUID(first)

    def test_loader_adds_uuid_once_and_clear_reuses_it(self):
        raw = {
            "告警编码ID": "E",
            "告警标题": "A",
            "告警源": "NE1",
            "站点ID": "S1",
            "告警首次发生时间": "2026-01-01 00:00:00",
            "告警清除时间": "2026-01-01 00:01:00",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alarms.jsonl"
            copied_path = Path(tmpdir) / "copied-alarms.jsonl"
            path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            copied_path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            _processed, events, normal_count, clear_count = load_valid_alarms(
                path,
                {"A"},
                {"S1"},
                {},
                show_progress=False,
            )
            _processed, copied_events, _normal_count, _clear_count = load_valid_alarms(
                copied_path,
                {"A"},
                {"S1"},
                {},
                show_progress=False,
            )
        self.assertEqual((normal_count, clear_count), (1, 1))
        self.assertEqual(events[0]["occurrence_uuid"], events[1]["occurrence_uuid"])
        self.assertEqual(events[0]["occurrence_uuid"], copied_events[0]["occurrence_uuid"])
        self.assertEqual(events[0]["occurrence_uuid"], alarm_content_uuid(raw))

    def test_loader_rejects_alarm_without_eid(self):
        raw = {
            "告警标题": "A",
            "告警源": "NE1",
            "站点ID": "S1",
            "告警首次发生时间": "2026-01-01 00:00:00",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alarms.jsonl"
            path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required eid"):
                load_valid_alarms(path, {"A"}, {"S1"}, {}, show_progress=False)

    def test_identity_requires_eid_and_valid_uuid(self):
        uid = occurrence("required")
        self.assertEqual(require_alarm_identity({"eid": "E", "occurrence_uuid": uid}), ("E", uid))
        with self.assertRaises(ValueError):
            require_alarm_identity({"eid": "E"})
        with self.assertRaises(ValueError):
            require_alarm_identity({"eid": "E", "occurrence_uuid": "obs-1"})

    def test_temporal_engine_enforces_and_normalizes_complete_identity_at_ingress(self):
        engine = TemporalGraphEngine({}, {}, {})
        uid = occurrence("engine-ingress")
        with self.assertRaisesRegex(ValueError, "required eid"):
            engine.process_event("S1", "A", 1, None, uid, register_trigger=False)
        self.assertFalse(engine.event_cache)

        engine.process_event("S1", "A", 1, 0, "{" + uid.upper() + "}", register_trigger=False)
        cached = engine.event_cache["S1"][0]
        self.assertEqual((cached[1], cached[5]), ("0", uid))

    def test_incremental_engine_uses_normalized_identity_for_cache_and_clear(self):
        engine = IncrementalFaultEngine({}, {}, {})
        uid = occurrence("incremental-ingress")
        decorated_uuid = "{" + uid.upper() + "}"
        engine.process_event("S1", "A", 1, 0, decorated_uuid, register_trigger=False)
        cached = engine.event_cache["S1"][0]
        self.assertEqual((cached[1], cached[5]), ("0", uid))
        engine.process_event("S1", "A", 2, 0, decorated_uuid, is_clear=True, register_trigger=False)
        self.assertFalse(engine.event_cache.get("S1"))

    def test_baseline_requires_and_normalizes_complete_identity(self):
        uid = occurrence("baseline")
        source = {
            "eid": 0,
            "occurrence_uuid": "{" + uid.upper() + "}",
            "故障组ID": "G",
        }
        result = alarm_to_baseline_symptom(source, group_field="故障组ID", ne_graph_data={})
        self.assertEqual((result["eid"], result["occurrence_uuid"]), ("0", uid))
        with self.assertRaisesRegex(ValueError, "required eid"):
            alarm_to_baseline_symptom(
                {"occurrence_uuid": uid, "故障组ID": "G"},
                group_field="故障组ID",
                ne_graph_data={},
            )

    def test_mhp_and_brunch_preserve_the_same_uuid(self):
        uid = occurrence("summary")
        event = {
            "ts": 1,
            "site_id": "S1",
            "alarm_source": "NE1",
            "alarm_title": "A",
            "occurrence_uuid": uid,
            "alarm": {"告警编码ID": "E"},
        }
        for summarize in (summarize_mhp_alarm_event, summarize_brunch_alarm_event):
            summary = summarize(event, 0)
            self.assertEqual(summary["occurrence_uuid"], uid)
            self.assertEqual(summary["event_id"], "E")

    def test_visual_records_preserve_uuid_without_aliases(self):
        uid = occurrence("visual")
        source = symptom("E", uid)
        for build in (_symptom_to_visual_record, _symptom_to_visual_record_mhp):
            record = build(source)
            self.assertEqual(record["occurrence_uuid"], uid)
            self.assertNotIn("occurrence_id", record)

    def test_visual_edges_reject_invalid_occurrence_uuid(self):
        first = occurrence("edge-1")
        second = occurrence("edge-2")
        group = {
            "symptoms": [symptom("E1", first), symptom("E2", second)],
            "edges": [{
                "source_event_id": "E1",
                "target_event_id": "E2",
                "source_occurrence_uuid": "not-a-uuid",
                "target_occurrence_uuid": second,
            }],
        }
        for build_edges in (_mhp_propagation_edges, _brunch_missing_topology_edges):
            with self.assertRaisesRegex(ValueError, "invalid occurrence_uuid"):
                build_edges(group, {"NE1": {}})

    def test_duplicate_eid_different_uuid_is_not_merged(self):
        left = {"uuid": "L", "rule": "r", "symptoms": [symptom("E", occurrence("left"))]}
        right = {"uuid": "R", "rule": "r", "symptoms": [symptom("E", occurrence("right"))]}
        merged = merge_match_batch([left, right])
        self.assertEqual(len(merged), 2)

    def test_csm_indexes_use_the_complete_pair(self):
        first = occurrence("csm-1")
        second = occurrence("csm-2")
        active = ActiveAlarmIndex()
        active.add("S1", "A", 1, "E", first)
        active.add("S1", "A", 2, "E", second)
        active.remove("E", first)
        self.assertEqual([event["occurrence_uuid"] for event in active.by_site["S1"]], [second])

        store = CSMGroupStore({"r": {"max_stay_time_sec": 100}}, 100)
        left = {"uuid": "L", "rule": "r", "symptoms": [symptom("E", first)]}
        right = {"uuid": "R", "rule": "r", "symptoms": [symptom("E", second)]}
        self.assertEqual(len(store.finalize([left], 1)), 1)
        self.assertEqual(len(store.finalize([right], 2)), 1)

    def test_same_pair_is_deduped_and_role_metadata_is_merged(self):
        uid = occurrence("same")
        first = symptom("E", uid, matched_role="root", matched_role_list=["root"])
        second = symptom("E", uid, matched_role="cascade", matched_role_list=["cascade"])
        merged = merge_match_batch([{"uuid": "G", "rule": "r", "symptoms": [first, second]}])
        self.assertEqual(len(merged[0]["symptoms"]), 1)
        self.assertEqual(set(merged[0]["symptoms"][0]["matched_role_list"]), {"root", "cascade"})

    def test_node_rule_helper_keeps_duplicate_eid_occurrences(self):
        first = occurrence("node-1")
        second = occurrence("node-2")
        cache = [(1, "E", "A", "NE1", frozenset(), first), (2, "E", "A", "NE1", frozenset(), second)]
        helper = NodeRuleHelper({"S1": {}}, set(), lambda _node: cache)
        events = helper.events_in_window("S1", 1.5, 5)
        self.assertEqual(get_match_alarm_keys({"symptoms": events}), {("E", first), ("E", second)})

    def test_alarm_period_keeps_uuid_on_raw_occurrence(self):
        engine = DummyAlarmPeriodEngine()
        uid = occurrence("period")
        engine._register_alarm_period_occurrence("S1", "A", 1, "E", uid, alarm_source="NE1")
        matched = NodeRuleHelper({"S1": {}}, set(), lambda node: engine.event_cache[node]).events_in_window("S1", 1, 1)
        self.assertEqual(require_alarm_identity(matched[0]), ("E", uid))

    def test_clear_removes_only_the_matching_pair(self):
        first = occurrence("clear-1")
        second = occurrence("clear-2")

        period_engine = DummyAlarmPeriodEngine()
        period_engine._register_alarm_period_occurrence("S1", "A", 1, "E", first, alarm_source="NE1")
        period_engine._register_alarm_period_occurrence("S1", "A", 2, "E", second, alarm_source="NE1")
        period_engine._remove_cleared_events("S1", "E", first, alarm_type="A", alarm_source="NE1")
        period_events = NodeRuleHelper({"S1": {}}, set(), lambda node: period_engine.event_cache[node]).events_in_window("S1", 2, 5)
        self.assertEqual(get_match_alarm_keys({"symptoms": period_events}), {("E", second)})

        raw_engine = DummyAlarmPeriodEngine()
        raw_engine.use_alarm_period_cache = False
        raw_engine.event_cache["S1"].extend([
            (1, "E", "A", "NE1", frozenset(), first),
            (2, "E", "A", "NE1", frozenset(), second),
        ])
        raw_engine._remove_cleared_events("S1", "E", first, alarm_type="A", alarm_source="NE1")
        self.assertEqual([event[5] for event in raw_engine.event_cache["S1"]], [second])

    def test_metadata_enrichment_uses_only_exact_pair(self):
        first = occurrence("meta-1")
        second = occurrence("meta-2")
        alarms = [
            {"alarm": {"告警编码ID": "E", "工单号": "T1"}, "occurrence_uuid": first},
            {"alarm": {"告警编码ID": "E", "工单号": "T2"}, "occurrence_uuid": second},
        ]
        index = build_alarm_metadata_index(alarms)
        enriched = enrich_match_symptoms({"symptoms": [symptom("E", second)]}, index)
        self.assertEqual(enriched[0]["工单号"], "T2")

    def test_ticket_dedupe_uses_eid_and_uuid(self):
        first = occurrence("ticket-1")
        second = occurrence("ticket-2")
        rows = [
            {"alarm_id": "E", "occurrence_uuid": first},
            {"alarm_id": "E", "occurrence_uuid": first},
            {"alarm_id": "E", "occurrence_uuid": second},
        ]
        self.assertEqual(len(dedupe_alarm_records(rows)), 2)

    def test_debug_alarm_lookup_resolves_groups_by_complete_identity(self):
        uid = occurrence("debug-lookup")
        upper_bound_index = {
            "T": {"site_evidence": {"S1": [{"alarm_id": "E", "occurrence_uuid": uid}]}}
        }
        alarm_to_groups = {("E", uid): {"G"}}
        for build_lookup in (
            build_group_output_debug_alarm_lookup,
            build_ticket_site_debug_alarm_lookup,
        ):
            result = build_lookup({"E"}, {"T": ["S1"]}, upper_bound_index, alarm_to_groups)
            self.assertEqual(result["E"]["matched_groups"], ["G"])
            self.assertEqual(result["E"]["evidence_hits"][0]["occurrence_uuid"], uid)

    def test_sorted_cache_roundtrip_requires_uuid(self):
        uid = occurrence("cache")
        event = {
            "alarm": {"告警编码ID": "E", "故障组ID": "G"},
            "occurrence_uuid": uid,
            "ts": 1,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alarms.jsonl"
            metadata = write_sorted_alarm_cache(path, [event])
            self.assertEqual(metadata["cache_type"], "fault_grouping.sorted_alarms.v3")
            self.assertEqual(metadata["alarm_identity_scheme"], "eid+canonical-json-uuid5:v1")
            loaded_metadata, loaded = load_sorted_alarm_cache(path)
            self.assertEqual(loaded_metadata["alarm_identity_scheme"], metadata["alarm_identity_scheme"])
            self.assertEqual(loaded[0]["occurrence_uuid"], uid)
            self.assertEqual(count_alarms(path, show_progress=False)["alarm_count"], 1)
            self.assertEqual(
                check_duplicate_eid(path, show_progress=False)["cache_header_count"],
                1,
            )
            grouped, group_stats = group_alarms(
                path,
                ne_graph="",
                site_graph="",
                visual_output=False,
            )
            self.assertEqual(([group["故障组ID"] for group in grouped], group_stats["grouped_alarm_count"]), (["G"], 1))
            with self.assertRaises(ValueError):
                write_sorted_alarm_cache(Path(tmpdir) / "invalid.jsonl", [{"alarm": {"告警编码ID": "E"}}])

            old_path = Path(tmpdir) / "old-v2.jsonl"
            old_path.write_text(
                json.dumps({"cache_type": "fault_grouping.sorted_alarms.v2"}) + "\n"
                + json.dumps(event, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(is_sorted_alarm_cache_file(old_path))
            with self.assertRaisesRegex(ValueError, "当前身份方案"):
                load_sorted_alarm_cache(old_path)
            for consume_old_cache in (
                lambda: count_alarms(old_path, show_progress=False),
                lambda: check_duplicate_eid(old_path, show_progress=False),
                lambda: group_alarms(
                    old_path,
                    ne_graph="",
                    site_graph="",
                    visual_output=False,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "不支持的排序告警缓存格式"):
                    consume_old_cache()

    def test_missing_event_gets_uuid_at_creation(self):
        sampler = MissingChainSampler(object())
        event = sampler._new_event(ts=1, type_id=1, observed=False, meta={}, depth=1)
        uuid.UUID(event.meta["occurrence_uuid"])


if __name__ == "__main__":
    unittest.main()
