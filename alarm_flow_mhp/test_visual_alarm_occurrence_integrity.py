import json
import collections
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from alarm_flow_brunch.aggregator import summarize_alarm_event as summarize_brunch_alarm_event
from alarm_flow_brunch.visual_output import group_to_visual_match as group_to_visual_match_brunch
from alarm_flow_brunch.visual_output import _symptom_to_visual_record
from alarm_flow_mhp.aggregator import summarize_alarm_event as summarize_mhp_alarm_event
from alarm_flow_mhp.compare_visual_alarm_groups import _build_visual_indexes
from alarm_flow_mhp.stream_alarm_mhp import Cascade, OnlineEvent, _cascade_to_group, _summary_of
from alarm_flow_mhp.visual_output import group_to_visual_match_mhp
from fault_grouping.matching.group_output_builder import build_alarm_metadata_index, build_jsonl_match_output
from fault_grouping.node_rule_helper import NodeRuleHelper
from fault_grouping.temporal_engine.alarm_period import TemporalGraphEngineAlarmPeriodMixin
from fault_grouping.temporal_engine.utils import (
    get_match_alarm_keys,
    merge_match_batch,
    merge_overlapping_symptoms,
)
from fault_grouping.tools.alarm_group_baseline import build_baseline_records
from fault_grouping.tools.merge_match_output_with_alarm_groups import (
    _alarm_record_merge_key,
    _build_connected_components,
    _build_group_output_from_alarms,
    _extract_alarm_keys_from_alarm_group,
    _extract_alarm_keys_from_match_group,
    _merge_alarm_record_lists,
    _merge_alarm_list_into_match_group,
    _symptom_merge_key,
)
from ticket_recall.evaluation.compute_ultimate_group_alarm_group_metrics import (
    _build_potential_groups_by_alarm_id,
)
from ticket_recall.ticket_recall_utils import (
    alarm_record_identity_key,
    build_alarm_to_group_index,
    build_visualization_case_record,
    collect_groups_by_evidence,
    dedupe_alarm_records,
    load_upper_bound_index,
)


class _DummyAlarmPeriodEngine(TemporalGraphEngineAlarmPeriodMixin):
    def __init__(self):
        self.active_alarm_periods = collections.defaultdict(dict)
        self.active_event_to_period = collections.defaultdict(dict)
        self.event_cache = collections.defaultdict(collections.deque)
        self.logged = []

    def _get_event_ttl(self, _alarm_type):
        return 3600

    def _log_debug_event_removal(self, *args, **kwargs):
        self.logged.append((args, kwargs))


class VisualAlarmOccurrenceIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.ne_graph = {
            "NE1": {
                "site_id": "S1",
                "site_name": "site1",
                "domain": "D",
                "link": {},
            }
        }
        self.site_graph = {}

    def _raw_alarm_event(self):
        return {
            "ts": 1,
            "site_id": "S1",
            "alarm_source": "NE_RAW",
            "alarm_title": "A",
            "alarm": {
                "告警编码ID": "E_DUP",
                "故障组ID": "AG1",
                "工单号": "T1",
                "告警清除时间": "2026-01-01 00:05:00",
            },
        }

    def test_summary_paths_preserve_alarm_group_fields(self):
        raw = self._raw_alarm_event()
        for summarize in (summarize_brunch_alarm_event, summarize_mhp_alarm_event):
            summary = summarize(raw, 7)
            self.assertEqual(summary["event_id"], "E_DUP")
            self.assertEqual(summary["occurrence_id"], "obs-7")
            self.assertEqual(summary["故障组ID"], "AG1")
            self.assertEqual(summary["工单号"], "T1")
            self.assertEqual(summary["告警清除时间"], "2026-01-01 00:05:00")

        stream_summary = _summary_of(SimpleNamespace(alarm=raw, index=7, ne="NE1"))
        visual_record = _symptom_to_visual_record(stream_summary)
        self.assertEqual(visual_record["eid"], "E_DUP")
        self.assertEqual(visual_record["occurrence_id"], "obs-7")
        self.assertEqual(visual_record["故障组ID"], "AG1")
        self.assertEqual(visual_record["工单号"], "T1")

    def test_mhp_stream_group_edges_use_occurrence_for_duplicate_eid(self):
        parent = OnlineEvent(
            index=0,
            ts=1.0,
            type_id=1,
            type_label="A | NE1",
            alarm={
                "ts": 1.0,
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_title": "A",
                "alarm": {"告警编码ID": "E_DUP"},
            },
            parent_index=-1,
        )
        child = OnlineEvent(
            index=1,
            ts=2.0,
            type_id=1,
            type_label="A | NE1",
            alarm={
                "ts": 2.0,
                "site_id": "S1",
                "alarm_source": "NE2",
                "alarm_title": "A",
                "alarm": {"告警编码ID": "E_DUP"},
            },
            parent_index=0,
        )
        cascade = Cascade(cascade_id=1)
        cascade.add(parent)
        cascade.add(child)

        group = _cascade_to_group(cascade)

        self.assertEqual([s["event_id"] for s in group["symptoms"]], ["E_DUP", "E_DUP"])
        self.assertEqual([s["occurrence_id"] for s in group["symptoms"]], ["obs-0", "obs-1"])
        self.assertEqual(len(group["edges"]), 1)
        self.assertEqual(group["edges"][0]["source_event_id"], "E_DUP")
        self.assertEqual(group["edges"][0]["target_event_id"], "E_DUP")
        self.assertEqual(group["edges"][0]["source_occurrence_id"], "obs-0")
        self.assertEqual(group["edges"][0]["target_occurrence_id"], "obs-1")

        ne_graph = {
            "NE1": {"site_id": "S1", "site_name": "site1", "domain": "D", "link": {}},
            "NE2": {"site_id": "S2", "site_name": "site2", "domain": "D", "link": {}},
        }
        match = group_to_visual_match_mhp(group, ne_graph_data=ne_graph)
        self.assertEqual(len(match["missing_topology_edges"]), 1)
        self.assertEqual(match["missing_topology_edges"][0]["source_occurrence_id"], "obs-0")
        self.assertEqual(match["missing_topology_edges"][0]["target_occurrence_id"], "obs-1")

    def _assert_writer_and_compare_keep_duplicate_eid(self, record):
        self.assertEqual([s["eid"] for s in record["symptoms"]], ["E_DUP", "E_DUP"])
        self.assertEqual([s["occurrence_id"] for s in record["symptoms"]], ["obs-0", "obs-1"])
        self.assertEqual(
            [a["alarm_id"] for a in record["ne_info"]["NE1"]["alarm"]],
            ["E_DUP", "E_DUP"],
        )
        self.assertEqual(
            [a["occurrence_id"] for a in record["ne_info"]["NE1"]["alarm"]],
            ["obs-0", "obs-1"],
        )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as handle:
            visual_path = handle.name
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            indexes = _build_visual_indexes(visual_path, group_field="故障组ID", ne_to_domain={})
        finally:
            Path(visual_path).unlink(missing_ok=True)
        self.assertEqual(indexes["real_symptom_count"], 2)
        self.assertEqual(indexes["symptom_with_group_id_count"], 2)
        self.assertEqual(len(indexes["mhp_group_to_site_alarms"][record["uuid"]]["S1"]), 2)
        self.assertEqual(len(indexes["alarm_group_to_site_alarms"]["AG1"]["S1"]), 2)

    def test_compare_visual_indexes_preserve_duplicate_eid_rows(self):
        record = {
            "uuid": "MG_DUP",
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-0",
                    "故障组ID": "AG1",
                },
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-1",
                    "故障组ID": "AG1",
                },
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl") as handle:
            visual_path = handle.name
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            indexes = _build_visual_indexes(visual_path, group_field="故障组ID", ne_to_domain={})
        finally:
            Path(visual_path).unlink(missing_ok=True)

        self.assertEqual(indexes["real_symptom_count"], 2)
        self.assertEqual(
            [alarm["occurrence_id"] for alarm in indexes["mhp_group_to_site_alarms"]["MG_DUP"]["S1"]],
            ["obs-0", "obs-1"],
        )
        self.assertEqual(
            [alarm["occurrence_id"] for alarm in indexes["alarm_group_to_site_alarms"]["AG1"]["S1"]],
            ["obs-0", "obs-1"],
        )

    def test_brunch_writer_and_compare_keep_duplicate_eid_occurrences(self):
        group = {
            "group_id": "BR1",
            "cascade_id": "BR1",
            "site_list": ["S1"],
            "root_event": {"event_id": "E_DUP"},
            "symptoms": [
                {
                    "index": 0,
                    "site_id": "S1",
                    "alarm_source": "NE1",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 1,
                    "event_id": "E_DUP",
                    "故障组ID": "AG1",
                },
                {
                    "index": 1,
                    "site_id": "S1",
                    "alarm_source": "NE1",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 2,
                    "event_id": "E_DUP",
                    "故障组ID": "AG1",
                },
            ],
            "edges": [],
        }
        match = group_to_visual_match_brunch(group, ne_graph_data=self.ne_graph)
        record = build_jsonl_match_output(match, self.ne_graph, self.site_graph, alarm_metadata_index={})
        self._assert_writer_and_compare_keep_duplicate_eid(record)

    def test_brunch_missing_topology_edges_use_occurrence_for_duplicate_eid(self):
        ne_graph = {
            "NE1": {"site_id": "S1", "site_name": "site1", "domain": "D", "link": {}},
            "NE2": {"site_id": "S2", "site_name": "site2", "domain": "D", "link": {}},
        }
        group = {
            "group_id": "BR_EDGE",
            "cascade_id": "BR_EDGE",
            "site_list": ["S1", "S2"],
            "root_event": {"event_id": "E_DUP"},
            "symptoms": [
                {
                    "index": 0,
                    "site_id": "S1",
                    "alarm_source": "NE1",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 1,
                    "event_id": "E_DUP",
                    "occurrence_id": "obs-0",
                    "故障组ID": "AG1",
                },
                {
                    "index": 1,
                    "site_id": "S2",
                    "alarm_source": "NE2",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 2,
                    "event_id": "E_DUP",
                    "occurrence_id": "obs-1",
                    "故障组ID": "AG1",
                },
            ],
            "edges": [
                {
                    "source_event_id": "E_DUP",
                    "target_event_id": "E_DUP",
                    "source_occurrence_id": "obs-0",
                    "target_occurrence_id": "obs-1",
                    "source_type": "T",
                    "target_type": "T",
                }
            ],
        }
        match = group_to_visual_match_brunch(group, ne_graph_data=ne_graph)
        self.assertEqual(len(match["missing_topology_edges"]), 1)
        edge = match["missing_topology_edges"][0]
        self.assertEqual(edge["source_ne"], "NE1")
        self.assertEqual(edge["target_ne"], "NE2")
        self.assertEqual(edge["source_occurrence_id"], "obs-0")
        self.assertEqual(edge["target_occurrence_id"], "obs-1")

    def test_mhp_writer_and_compare_keep_duplicate_eid_occurrences(self):
        group = {
            "group_id": "MG1",
            "cascade_id": "MG1",
            "site_list": ["S1"],
            "root_event": {"event_id": "E_DUP"},
            "symptoms": [
                {
                    "site_id": "S1",
                    "alarm_source": "NE1",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 1,
                    "event_id": "E_DUP",
                    "occurrence_id": "obs-0",
                    "故障组ID": "AG1",
                    "virtual": False,
                },
                {
                    "site_id": "S1",
                    "alarm_source": "NE1",
                    "alarm_title": "A",
                    "alarm_type": "T",
                    "ts": 2,
                    "event_id": "E_DUP",
                    "occurrence_id": "obs-1",
                    "故障组ID": "AG1",
                    "virtual": False,
                },
            ],
            "edges": [],
        }
        match = group_to_visual_match_mhp(group, ne_graph_data=self.ne_graph)
        record = build_jsonl_match_output(match, self.ne_graph, self.site_graph, alarm_metadata_index={})
        self._assert_writer_and_compare_keep_duplicate_eid(record)

    def test_case_visualization_merges_only_same_occurrence_role_duplicates(self):
        detail = {
            "ticket_id": "T1",
            "ticket_sites": ["S1"],
            "display_sites": ["S1"],
            "target_sites": ["S1"],
            "associated_sites": ["S1"],
            "missing_sites": [],
            "context_sites": ["S1"],
            "associated_site_alarms": {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm_source": "NE1",
                        "alarm_type": "A",
                        "alarm_time": "2026-01-01 00:00:00",
                        "_case_alarm_seq": "G::0",
                        "故障组ID": "AG1,AG2",
                        "mhp_group_id": "MG1",
                    },
                    {
                        "alarm_id": "E_DUP",
                        "alarm_source": "NE1",
                        "alarm_type": "A",
                        "alarm_time": "2026-01-01 00:00:01",
                        "_case_alarm_seq": "G::1",
                        "故障组ID": "AG1",
                        "mhp_group_id": "MG1",
                    },
                ]
            },
            "missing_site_alarms": {},
            "context_site_alarms": {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm_source": "NE1",
                        "alarm_type": "A",
                        "alarm_time": "2026-01-01 00:00:00",
                        "_case_alarm_seq": "G::0",
                        "故障组ID": "AG2;AG3",
                        "mhp_group_id": "MG1",
                    }
                ]
            },
        }
        case = build_visualization_case_record(
            detail,
            "test",
            ne_graph_data=self.ne_graph,
            site_to_ne_ids={"S1": ["NE1"]},
            site_coord_index={},
        )
        self.assertEqual({s["_case_alarm_seq"] for s in case["symptoms"]}, {"G::0", "G::1"})
        seq_to_symptom = {s["_case_alarm_seq"]: s for s in case["symptoms"]}
        self.assertEqual(seq_to_symptom["G::0"]["故障组ID"], "AG1,AG2,AG3")
        self.assertEqual(
            seq_to_symptom["G::0"]["matched_role_list"],
            ["associated_site", "context_site"],
        )
        self.assertEqual(
            {a["_case_alarm_seq"] for a in case["ne_info"]["NE1"]["alarm"]},
            {"G::0", "G::1"},
        )

    def test_case_visualization_preserves_raw_alarm_occurrence_duplicates(self):
        detail = {
            "ticket_id": "T1",
            "ticket_sites": ["S1"],
            "display_sites": ["S1"],
            "target_sites": ["S1"],
            "associated_sites": ["S1"],
            "missing_sites": [],
            "context_sites": [],
            "associated_site_alarms": {
                "S1": [
                    {
                        "告警编码ID": "E_DUP",
                        "告警源": "NE1",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:00",
                        "_raw_alarm_occurrence_id": "raw-alarm-1",
                    },
                    {
                        "告警编码ID": "E_DUP",
                        "告警源": "NE1",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:00",
                        "_raw_alarm_occurrence_id": "raw-alarm-2",
                    },
                ]
            },
            "missing_site_alarms": {},
            "context_site_alarms": {},
        }
        case = build_visualization_case_record(
            detail,
            "test",
            ne_graph_data=self.ne_graph,
            site_to_ne_ids={"S1": ["NE1"]},
            site_coord_index={},
        )
        self.assertEqual(len(case["symptoms"]), 2)
        self.assertEqual(
            [symptom["_raw_alarm_occurrence_id"] for symptom in case["symptoms"]],
            ["raw-alarm-1", "raw-alarm-2"],
        )
        self.assertEqual(len(case["ne_info"]["NE1"]["alarm"]), 2)
        self.assertEqual(
            [alarm["_raw_alarm_occurrence_id"] for alarm in case["ne_info"]["NE1"]["alarm"]],
            ["raw-alarm-1", "raw-alarm-2"],
        )

    def test_fault_grouping_merge_keys_prefer_occurrence_over_duplicate_eid(self):
        left = {
            "uuid": "L",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-0",
                }
            ],
        }
        right = {
            "uuid": "R",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-1",
                }
            ],
        }
        self.assertEqual(len(get_match_alarm_keys(left)), 1)
        self.assertNotEqual(get_match_alarm_keys(left), get_match_alarm_keys(right))

        merged, stats = merge_match_batch([left, right], return_stats=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["eid_merge_group_count"], 0)

    def test_fault_grouping_occurrence_identity_includes_time_context(self):
        left = {
            "uuid": "L",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-1",
                    "ts": 1,
                }
            ],
        }
        right = {
            "uuid": "R",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "occurrence_id": "obs-1",
                    "ts": 2,
                }
            ],
        }
        self.assertNotEqual(get_match_alarm_keys(left), get_match_alarm_keys(right))
        merged, stats = merge_match_batch([left, right], return_stats=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["eid_merge_group_count"], 0)

    def test_fault_grouping_merge_keys_still_merge_same_case_occurrence(self):
        left = {
            "uuid": "L",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "_case_alarm_seq": "G::0",
                }
            ],
        }
        right = {
            "uuid": "R",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "_case_alarm_seq": "G::0",
                }
            ],
        }
        merged, stats = merge_match_batch([left, right], return_stats=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(stats["eid_merge_group_count"], 1)

    def test_fault_grouping_fallback_key_does_not_merge_duplicate_eid_different_occurrences(self):
        left = {
            "uuid": "L",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "ts": 1,
                }
            ],
        }
        right = {
            "uuid": "R",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "ts": 2,
                }
            ],
        }
        merged, stats = merge_match_batch([left, right], return_stats=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["eid_merge_group_count"], 0)

    def test_fault_grouping_fallback_key_still_merges_identical_alarm_record(self):
        left = {
            "uuid": "L",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "ts": 1,
                }
            ],
        }
        right = {
            "uuid": "R",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "ts": 1,
                }
            ],
        }
        merged, stats = merge_match_batch([left, right], return_stats=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(stats["eid_merge_group_count"], 1)

    def test_node_rule_helper_raw_cache_keeps_duplicate_eid_occurrences(self):
        helper = NodeRuleHelper(
            sites_domain_map={},
            critical_alarms=set(),
            event_getter=lambda _node: [
                (1, "E_DUP", "A", "NE1", frozenset(), "raw-1"),
                (2, "E_DUP", "A", "NE1", frozenset(), "raw-2"),
            ],
        )
        matched = helper.events_in_window("S1", reference_ts=1, edge_window=5)
        self.assertEqual([event["eid"] for event in matched], ["E_DUP", "E_DUP"])
        self.assertEqual([event["occurrence_id"] for event in matched], ["raw-1", "raw-2"])
        self.assertNotEqual(
            get_match_alarm_keys({"symptoms": [matched[0]]}),
            get_match_alarm_keys({"symptoms": [matched[1]]}),
        )

    def test_alarm_period_cache_keeps_duplicate_eid_raw_items(self):
        engine = _DummyAlarmPeriodEngine()
        engine._register_alarm_period_occurrence("S1", "A", 1, "E_DUP", alarm_source="NE1")
        engine._register_alarm_period_occurrence("S1", "A", 2, "E_DUP", alarm_source="NE1")

        cached = list(engine.event_cache["S1"])
        self.assertEqual(len(cached), 1)
        self.assertEqual(
            [(raw_id, raw_ts) for raw_id, raw_ts, _occurrence in cached[0]["_raw_event_items"]],
            [("E_DUP", 1), ("E_DUP", 2)],
        )
        helper = NodeRuleHelper(
            sites_domain_map={},
            critical_alarms=set(),
            event_getter=lambda _node: cached,
        )
        matched = helper.events_in_window("S1", reference_ts=1, edge_window=5)
        self.assertEqual([event["eid"] for event in matched], ["E_DUP", "E_DUP"])
        self.assertEqual(len({event["occurrence_id"] for event in matched}), 2)

    def test_alarm_period_overlap_merge_keeps_distinct_occurrences(self):
        symptoms = [
            {
                "node": "S1",
                "alarm_source": "NE1",
                "alarm": "A",
                "eid": "E_DUP",
                "ts": 1,
                "_segment_start_ts": 1,
                "_segment_end_ts": 1,
                "occurrence_id": "raw-1",
            },
            {
                "node": "S1",
                "alarm_source": "NE1",
                "alarm": "A",
                "eid": "E_DUP",
                "ts": 1,
                "_segment_start_ts": 1,
                "_segment_end_ts": 1,
                "occurrence_id": "raw-2",
            },
        ]
        self.assertEqual(len(merge_overlapping_symptoms(symptoms)), 2)

        matches = [
            {
                "uuid": "L",
                "rule": "r",
                "role_mapping": {"cascade": ["S1"]},
                "symptoms": [symptoms[0]],
            },
            {
                "uuid": "R",
                "rule": "r",
                "role_mapping": {"cascade": ["S1"]},
                "symptoms": [symptoms[1]],
            },
        ]
        merged, stats = merge_match_batch(matches, return_stats=True, use_alarm_period_cache=True)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["alarm_overlap_merge_group_count"], 0)

    def test_alarm_group_merge_tool_keeps_duplicate_eid_occurrences(self):
        alarms = [
            {
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:01",
                "故障组ID": "AG1",
                "关联站点ID": "S1",
                "告警源": "NE1",
                "_raw_alarm_occurrence_id": "raw-alarm-1",
            },
            {
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:02",
                "故障组ID": "AG1",
                "关联站点ID": "S1",
                "告警源": "NE1",
                "_raw_alarm_occurrence_id": "raw-alarm-2",
            },
        ]
        record = _build_group_output_from_alarms("AG1", alarms, ne_graph_data=self.ne_graph)

        self.assertEqual([symptom["eid"] for symptom in record["symptoms"]], ["E_DUP", "E_DUP"])
        self.assertEqual(
            [symptom["occurrence_id"] for symptom in record["symptoms"]],
            ["raw-alarm-1", "raw-alarm-2"],
        )
        self.assertEqual(
            [alarm["alarm_id"] for alarm in record["ne_info"]["NE1"]["alarm"]],
            ["E_DUP", "E_DUP"],
        )
        self.assertEqual(
            [alarm["occurrence_id"] for alarm in record["ne_info"]["NE1"]["alarm"]],
            ["raw-alarm-1", "raw-alarm-2"],
        )
        self.assertNotEqual(
            _symptom_merge_key(record["symptoms"][0]),
            _symptom_merge_key(record["symptoms"][1]),
        )
        self.assertNotEqual(
            _alarm_record_merge_key(record["ne_info"]["NE1"]["alarm"][0]),
            _alarm_record_merge_key(record["ne_info"]["NE1"]["alarm"][1]),
        )
        same_occurrence_different_time_symptoms = [
            {
                "node": "S1",
                "alarm_source": "NE1",
                "eid": "E_DUP",
                "alarm": "A",
                "time_str": "2026-01-01 00:00:01",
                "occurrence_id": "raw-alarm-1",
            },
            {
                "node": "S1",
                "alarm_source": "NE1",
                "eid": "E_DUP",
                "alarm": "A",
                "time_str": "2026-01-01 00:00:02",
                "occurrence_id": "raw-alarm-1",
            },
        ]
        self.assertNotEqual(
            _symptom_merge_key(same_occurrence_different_time_symptoms[0]),
            _symptom_merge_key(same_occurrence_different_time_symptoms[1]),
        )
        same_occurrence_different_time_alarms = [
            {
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_id": "E_DUP",
                "alarm_type": "A",
                "alarm_time": "2026-01-01 00:00:01",
                "occurrence_id": "raw-alarm-1",
            },
            {
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_id": "E_DUP",
                "alarm_type": "A",
                "alarm_time": "2026-01-01 00:00:02",
                "occurrence_id": "raw-alarm-1",
            },
        ]
        self.assertNotEqual(
            _alarm_record_merge_key(same_occurrence_different_time_alarms[0]),
            _alarm_record_merge_key(same_occurrence_different_time_alarms[1]),
        )
        no_occurrence_different_time_symptoms = [
            {
                "node": "S1",
                "alarm_source": "NE1",
                "eid": "E_DUP",
                "alarm": "A",
                "time_str": "2026-01-01 00:00:01",
            },
            {
                "node": "S1",
                "alarm_source": "NE1",
                "eid": "E_DUP",
                "alarm": "A",
                "time_str": "2026-01-01 00:00:02",
            },
        ]
        self.assertNotEqual(
            _symptom_merge_key(no_occurrence_different_time_symptoms[0]),
            _symptom_merge_key(no_occurrence_different_time_symptoms[1]),
        )
        no_occurrence_different_time_alarms = [
            {
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_id": "E_DUP",
                "alarm_type": "A",
                "alarm_time": "2026-01-01 00:00:01",
            },
            {
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_id": "E_DUP",
                "alarm_type": "A",
                "alarm_time": "2026-01-01 00:00:02",
            },
        ]
        self.assertNotEqual(
            _alarm_record_merge_key(no_occurrence_different_time_alarms[0]),
            _alarm_record_merge_key(no_occurrence_different_time_alarms[1]),
        )
        self.assertEqual(
            len(_merge_alarm_record_lists(
                [no_occurrence_different_time_alarms[0]],
                [no_occurrence_different_time_alarms[1]],
            )),
            2,
        )

        base = {
            "match_info": {"uuid": "M1", "rule": "r"},
            "symptoms": [],
            "ne_info": {"NE1": {"alarm": []}},
            "group_info": {},
        }
        merged = _merge_alarm_list_into_match_group(base, "AG1", alarms, ne_graph_data=self.ne_graph)
        self.assertEqual(len(merged["symptoms"]), 2)
        self.assertEqual(len(merged["ne_info"]["NE1"]["alarm"]), 2)

        merged_alarm_records = _merge_alarm_record_lists(
            [record["ne_info"]["NE1"]["alarm"][0]],
            [record["ne_info"]["NE1"]["alarm"][1]],
        )
        self.assertEqual(len(merged_alarm_records), 2)

    def test_alarm_group_merge_components_use_context_not_bare_eid(self):
        match_record = {
            "match_info": {"uuid": "M1", "rule": "r"},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "eid": "E_DUP",
                    "time_str": "2026-01-01 00:00:01",
                    "occurrence_id": "obs-1",
                }
            ],
            "ne_info": {},
        }
        match_group_alarm_keys = [_extract_alarm_keys_from_match_group(match_record)]
        alarm_key_to_match_indices = collections.defaultdict(list)
        for match_idx, alarm_keys in enumerate(match_group_alarm_keys):
            for alarm_key in alarm_keys:
                alarm_key_to_match_indices[alarm_key].append(match_idx)

        alarm_groups = {
            "AG1": [
                {
                    "告警编码ID": "E_DUP",
                    "告警标题": "A",
                    "告警首次发生时间": "2026-01-01 00:00:01",
                    "告警源": "NE1",
                    "_raw_alarm_occurrence_id": "raw-alarm-1",
                }
            ],
            "AG2": [
                {
                    "告警编码ID": "E_DUP",
                    "告警标题": "A",
                    "告警首次发生时间": "2026-01-01 00:00:02",
                    "关联站点ID": "S1",
                    "告警源": "NE1",
                    "_raw_alarm_occurrence_id": "raw-alarm-2",
                }
            ],
        }
        alarm_group_alarm_keys = {
            group_id: _extract_alarm_keys_from_alarm_group(alarm_list)
            for group_id, alarm_list in alarm_groups.items()
        }

        components, standalone_alarm_group_ids = _build_connected_components(
            match_group_alarm_keys,
            alarm_group_alarm_keys,
            alarm_key_to_match_indices,
        )

        self.assertEqual(
            components,
            [{"match_indices": [0], "alarm_group_ids": ["AG1"]}],
        )
        self.assertEqual(standalone_alarm_group_ids, ["AG2"])

    def test_alarm_group_baseline_keeps_duplicate_eid_occurrences(self):
        events = [
            {
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:01",
                "故障组ID": "AG1",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
            {
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:02",
                "故障组ID": "AG1",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
        ]
        records = build_baseline_records(
            events,
            group_field="故障组ID",
            ne_graph_data=self.ne_graph,
            min_group_events=1,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual([symptom["eid"] for symptom in records[0]["symptoms"]], ["E_DUP", "E_DUP"])
        self.assertEqual(
            [symptom["occurrence_id"] for symptom in records[0]["symptoms"]],
            ["alarm-baseline-AG1-0", "alarm-baseline-AG1-1"],
        )

    def test_potential_evidence_uses_context_for_duplicate_alarm_id(self):
        group_to_site_alarms = {
            "AG1": {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:01",
                        "site_id": "S1",
                        "alarm_source": "NE1",
                    }
                ]
            },
            "AG2": {
                "S2": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:02",
                        "site_id": "S2",
                        "alarm_source": "NE2",
                    }
                ]
            },
        }
        alarm_to_groups = build_alarm_to_group_index(group_to_site_alarms)

        matched = collect_groups_by_evidence(
            {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:01",
                        "site_id": "S1",
                        "alarm_source": "NE1",
                    }
                ]
            },
            alarm_to_groups,
        )
        self.assertEqual(matched, {"AG1"})
        self.assertEqual(alarm_to_groups[("alarm_id", "E_DUP")], {"AG1", "AG2"})

        chinese_group_to_site_alarms = {
            "AG1": {
                "S1": [
                    {
                        "告警编码ID": "E_DUP",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:01",
                        "关联站点ID": "S1",
                        "告警源": "NE1",
                    }
                ]
            },
            "AG2": {
                "S2": [
                    {
                        "告警编码ID": "E_DUP",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:02",
                        "关联站点ID": "S2",
                        "告警源": "NE2",
                    }
                ]
            },
        }
        chinese_alarm_to_groups = build_alarm_to_group_index(chinese_group_to_site_alarms)
        chinese_matched = collect_groups_by_evidence(
            {
                "S1": [
                    {
                        "告警编码ID": "E_DUP",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:01",
                        "关联站点ID": "S1",
                        "告警源": "NE1",
                    }
                ]
            },
            chinese_alarm_to_groups,
        )
        self.assertEqual(chinese_matched, {"AG1"})

    def test_potential_evidence_occurrence_identity_includes_time_context(self):
        group_to_site_alarms = {
            "AG1": {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:01",
                        "site_id": "S1",
                        "alarm_source": "NE1",
                        "occurrence_id": "obs-1",
                    }
                ]
            },
            "AG2": {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:02",
                        "site_id": "S1",
                        "alarm_source": "NE1",
                        "occurrence_id": "obs-1",
                    }
                ]
            },
        }
        alarm_to_groups = build_alarm_to_group_index(group_to_site_alarms)
        matched = collect_groups_by_evidence(
            {
                "S1": [
                    {
                        "alarm_id": "E_DUP",
                        "alarm": "A",
                        "alarm_time": "2026-01-01 00:00:02",
                        "site_id": "S1",
                        "alarm_source": "NE1",
                        "occurrence_id": "obs-1",
                    }
                ]
            },
            alarm_to_groups,
        )
        self.assertEqual(matched, {"AG2"})

    def test_raw_alarm_occurrence_preserves_upper_bound_duplicate_rows(self):
        records = [
            {
                "_raw_alarm_occurrence_id": "raw-alarm-1",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:01",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
            {
                "_raw_alarm_occurrence_id": "raw-alarm-2",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:01",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
        ]
        self.assertEqual(len(dedupe_alarm_records(records)), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            upper_bound_file = Path(tmpdir) / "upper_bound.json"
            upper_bound_file.write_text(
                json.dumps(
                    {
                        "window_seconds": 900,
                        "details": [
                            {
                                "ticket_id": "T1",
                                "ticket_site_count": 1,
                                "associated_site_count": 1,
                                "associated_sites": ["S1"],
                                "evidence": {
                                    "direct_site_alarms": {
                                        "S1": records,
                                    },
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            index = load_upper_bound_index(str(upper_bound_file))

        evidence_records = index["T1"]["site_evidence"]["S1"]
        self.assertEqual(len(evidence_records), 2)
        self.assertEqual(
            [record["_raw_alarm_occurrence_id"] for record in evidence_records],
            ["raw-alarm-1", "raw-alarm-2"],
        )

        alarm_to_groups = build_alarm_to_group_index({
            "AG1": {
                "S1": [
                    {
                        "告警编码ID": "E_DUP",
                        "告警标题": "A",
                        "告警首次发生时间": "2026-01-01 00:00:01",
                        "关联站点ID": "S1",
                        "告警源": "NE1",
                    }
                ]
            }
        })
        self.assertEqual(
            collect_groups_by_evidence(index["T1"]["site_evidence"], alarm_to_groups),
            {"AG1"},
        )

    def test_raw_alarm_occurrence_dedupe_keeps_different_contexts(self):
        records = [
            {
                "_raw_alarm_occurrence_id": "raw-alarm-1",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:00",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
            {
                "_raw_alarm_occurrence_id": "raw-alarm-1",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:01",
                "关联站点ID": "S1",
                "告警源": "NE1",
            },
            {
                "_raw_alarm_occurrence_id": "raw-alarm-1",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:00",
                "关联站点ID": "S2",
                "告警源": "NE1",
            },
            {
                "_raw_alarm_occurrence_id": "raw-alarm-1",
                "告警编码ID": "E_DUP",
                "告警标题": "A",
                "告警首次发生时间": "2026-01-01 00:00:00",
                "关联站点ID": "S1",
                "告警源": "NE1",
                "故障组ID": "AG1",
            },
        ]
        deduped = dedupe_alarm_records(records)
        self.assertEqual(len(deduped), 3)
        self.assertEqual(
            {
                (
                    record.get("关联站点ID"),
                    record.get("告警源"),
                    record.get("告警首次发生时间"),
                )
                for record in deduped
            },
            {
                ("S1", "NE1", "2026-01-01 00:00:00"),
                ("S1", "NE1", "2026-01-01 00:00:01"),
                ("S2", "NE1", "2026-01-01 00:00:00"),
            },
        )
        s1_record = next(
            record
            for record in deduped
            if record.get("关联站点ID") == "S1"
            and record.get("告警首次发生时间") == "2026-01-01 00:00:00"
        )
        self.assertEqual(s1_record["故障组ID"], "AG1")

    def test_alarm_metadata_index_uses_occurrence_before_duplicate_eid_fallback(self):
        valid_alarms = [
            {
                "alarm": {
                    "告警编码ID": "E_DUP",
                    "告警标题": "A",
                    "故障组ID": "AG1",
                    "工单号": "T1",
                },
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_title": "A",
                "ts": 1.0,
            },
            {
                "alarm": {
                    "告警编码ID": "E_SKIP",
                    "告警标题": "B",
                },
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_title": "B",
                "ts": 2.0,
            },
            {
                "alarm": {
                    "告警编码ID": "E_DUP",
                    "告警标题": "A",
                    "故障组ID": "AG2",
                    "工单号": "T2",
                },
                "site_id": "S1",
                "alarm_source": "NE1",
                "alarm_title": "A",
                "ts": 3.0,
            },
        ]
        metadata_index = build_alarm_metadata_index(valid_alarms)
        match = {
            "uuid": "g1",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "ts": 3.0,
                    "eid": "E_DUP",
                    "occurrence_id": "raw-3",
                    "matched_role": "cascade",
                }
            ],
        }
        record = build_jsonl_match_output(match, self.ne_graph, self.site_graph, metadata_index)
        self.assertEqual(record["symptoms"][0]["故障组ID"], "AG2")
        self.assertEqual(record["symptoms"][0]["工单号"], "T2")
        self.assertEqual(record["ne_info"]["NE1"]["alarm"][0]["故障组ID"], "AG2")

        period_match = {
            "uuid": "g2",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "ts": 3.0,
                    "eid": "E_DUP",
                    "occurrence_id": "E_DUP#2",
                    "matched_role": "cascade",
                }
            ],
        }
        period_record = build_jsonl_match_output(period_match, self.ne_graph, self.site_graph, metadata_index)
        self.assertEqual(period_record["symptoms"][0]["故障组ID"], "AG2")
        self.assertEqual(period_record["symptoms"][0]["工单号"], "T2")

        reset_period_match = {
            "uuid": "g3",
            "rule": "r",
            "role_mapping": {"cascade": ["S1"]},
            "symptoms": [
                {
                    "node": "S1",
                    "alarm_source": "NE1",
                    "alarm": "A",
                    "ts": 3.0,
                    "eid": "E_DUP",
                    "occurrence_id": "E_DUP#1",
                    "matched_role": "cascade",
                }
            ],
        }
        reset_period_record = build_jsonl_match_output(
            reset_period_match,
            self.ne_graph,
            self.site_graph,
            metadata_index,
        )
        self.assertEqual(reset_period_record["symptoms"][0]["故障组ID"], "AG2")
        self.assertEqual(reset_period_record["symptoms"][0]["工单号"], "T2")

    def test_ultimate_metric_potential_groups_use_contextual_alarm_identity(self):
        source_key = alarm_record_identity_key({
            "alarm_id": "E_DUP",
            "alarm": "A",
            "alarm_time": "2026-01-01 00:00:01",
            "site_id": "S1",
            "alarm_source": "NE1",
        })
        target_key = alarm_record_identity_key({
            "alarm_id": "E_DUP",
            "alarm": "A",
            "alarm_time": "2026-01-01 00:00:02",
            "site_id": "S2",
            "alarm_source": "NE2",
        })
        source_to_alarm_ids = {"UG1": {source_key}}
        alarm_id_to_target_groups = {
            source_key: {"AG1"},
            target_key: {"AG2"},
            ("alarm_id", "E_DUP"): {"AG1", "AG2"},
        }
        result = _build_potential_groups_by_alarm_id(
            source_to_alarm_ids,
            alarm_id_to_target_groups,
            excluded_groups_map={},
        )
        self.assertEqual(result["UG1"], {"AG1"})


if __name__ == "__main__":
    unittest.main()
