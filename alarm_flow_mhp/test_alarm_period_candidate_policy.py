from contextlib import redirect_stdout
from collections import defaultdict
import io
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from alarm_flow_mhp.candidate_policy import (
    CandidatePolicy,
    RELATED_MASK,
    adaptive_candidate_sources,
    build_candidate_indices,
    candidate_rule_mask,
    load_candidate_policy,
    prepare_adaptive_candidates,
    unrelated_pair_allowed,
    write_candidate_policy,
)
from alarm_flow_mhp.feature_spec import FeatureLayout, RuntimeFeatureScorer
from alarm_flow_mhp.period_source_imputer import PeriodImputeConfig, PeriodSourceImputer
from alarm_flow_mhp.learn_alarm_period_candidate_policy import (
    _teacher_positive_masks,
)
from alarm_flow_mhp.stream_alarm_period_mhp import (
    CACHE_STATE_LAYOUT_TARGET_ONLY,
    AlarmPeriod,
    AlarmPeriodMHPAssigner,
    CompiledEdge,
    CompiledAssociationPlan,
    CompactAssociationIndex,
    PeriodFaultGroup,
    PeriodSignature,
    PeriodStreamConfig,
    PeriodType,
    RelationEvidence,
    _enable_period_profiling,
    _iter_profiled_events,
    _print_period_profile,
    build_compact_csr_arrays,
)
from alarm_flow_mhp.stream_alarm_mhp import OnlineEvent
from alarm_flow_mhp.visual_output import _symptom_to_visual_record_mhp
from fault_grouping.matching.profiling import PhaseTimer
from fault_grouping.tools.analyze_visual_group_clear_metrics import (
    ClearedAlarm,
    _sample_null_pairs,
)
from mhp.feature_kernel import FeatureKernel


# site, vendor, ne_type. Chosen so the non-local rules carve out pairs that the
# related predicate (same entity/node/site, no topology here) does not:
#   same_ne_type: {A,C} (T1), {B,D} (T2)   same_vendor: {A,B,D} (V1)
#   related same_site: {A,B} (S1), {C,D} (S2)
NODES = {
    "A": ("S1", "V1", "T1"),
    "B": ("S1", "V1", "T2"),
    "C": ("S2", "V2", "T1"),
    "D": ("S2", "V1", "T2"),
}


def _plan(
    *, dynamic="off", seed=0, mu=0.1, at_vocab=("X",), scope="global", policy=None,
    coords=None,
):
    at_vocab = list(at_vocab)
    coords = coords or {}
    infos = {
        node: SimpleNamespace(
            site_id=site,
            manufacturer=vendor,
            ne_type=ne_type,
            domain_bucket="RAN",
            latitude=coords.get(node, (None, None))[0],
            longitude=coords.get(node, (None, None))[1],
        )
        for node, (site, vendor, ne_type) in NODES.items()
    }
    n_dynamic = 6 if dynamic == "source_target" else (3 if dynamic == "target" else 0)
    rng = np.random.default_rng(seed)
    static_count = FeatureLayout(at_vocab).n_features
    weights = rng.normal(0.0, 0.5, static_count + n_dynamic)
    scorer = RuntimeFeatureScorer(
        FeatureKernel(weights),
        at_vocab,
        SimpleNamespace(node_infos=infos),
        None,
        beta=1.0,
        n_dynamic=n_dynamic,
        dynamic_mode=dynamic,
    )
    artifact = SimpleNamespace(
        training_metadata={
            "feature_runtime": {
                "at_vocab": at_vocab,
                "beta": 1.0,
                "mu_default": mu,
            }
        },
        config=SimpleNamespace(
            type_fields=("alarm_source", "alarm_type"),
            topology_node_field="alarm_source",
            dynamic_alpha=dynamic,
        ),
    )
    return CompiledAssociationPlan(
        scorer,
        None,
        artifact,
        PeriodStreamConfig(candidate_scope=scope),
        candidate_policy=policy,
    )


class CandidatePolicyTest(unittest.TestCase):
    def test_policy_round_trip_and_fingerprint(self):
        policy = CandidatePolicy(
            {"X": {"X": ("same_site", "topology")}},
            approved=True,
            fingerprint={"model": "same"},
            validation={"recall": 1.0},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            write_candidate_policy(path, policy)
            self.assertEqual(
                load_candidate_policy(
                    path, expected_fingerprint={"model": "same"}
                ),
                policy,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_candidate_policy(
                    path, expected_fingerprint={"model": "changed"}
                )

    def test_unrelated_pair_allowed_excludes_related(self):
        base = _plan()
        policy = CandidatePolicy({"X": {"X": ("same_ne_type",)}}, approved=True)
        A, B, C = (PeriodType(n, "X") for n in ("A", "B", "C"))
        # A-C: same ne_type T1, not related -> unrelated candidate.
        self.assertTrue(unrelated_pair_allowed(policy, A, C, base.scorer))
        # A-A: related (same entity) even though same ne_type -> excluded.
        self.assertFalse(unrelated_pair_allowed(policy, A, A, base.scorer))
        # A-B: related (same site) and not same ne_type -> excluded.
        self.assertFalse(unrelated_pair_allowed(policy, A, B, base.scorer))
        # B-C: neither related nor same ne_type -> not a candidate.
        self.assertFalse(unrelated_pair_allowed(policy, B, C, base.scorer))

    def test_geo_near_links_close_sites_and_stays_consistent(self):
        # A and C sit in different sites but ~5 km apart, so geo_near links them
        # while no categorical/related rule does; enumeration must equal the
        # predicate, and related pairs stay excluded.
        coords = {
            "A": (30.00, 120.00),
            "C": (30.03, 120.00),  # ~3 km from A -> same near-cell
            "B": (31.00, 121.00),  # far
            "D": (40.00, 130.00),  # far
        }
        base = _plan(coords=coords)
        period_types = tuple(PeriodType(n, "X") for n in NODES)
        policy = CandidatePolicy({"X": {"X": ("geo_near",)}}, approved=True)
        A, B, C = (PeriodType(n, "X") for n in ("A", "B", "C"))
        self.assertTrue(unrelated_pair_allowed(policy, A, C, base.scorer))
        self.assertFalse(unrelated_pair_allowed(policy, A, B, base.scorer))
        self.assertFalse(unrelated_pair_allowed(policy, A, A, base.scorer))
        prepared = prepare_adaptive_candidates(
            period_types, base.scorer, policy, exclude_related=True
        )
        for target in period_types:
            enumerated = set(
                adaptive_candidate_sources(
                    target, policy, prepared, exclude_related=True
                )
            )
            predicted = {
                source
                for source in period_types
                if unrelated_pair_allowed(policy, target, source, base.scorer)
            }
            self.assertEqual(enumerated, predicted)
        self.assertEqual(
            set(adaptive_candidate_sources(A, policy, prepared, exclude_related=True)),
            {C},
        )

    def test_unrelated_enumeration_matches_predicate(self):
        # The offline enumeration and the runtime predicate must agree exactly,
        # including the related exclusion.
        base = _plan()
        period_types = tuple(PeriodType(n, "X") for n in NODES)
        policy = CandidatePolicy(
            {"X": {"X": ("same_ne_type", "same_vendor")}}, approved=True
        )
        prepared = prepare_adaptive_candidates(
            period_types, base.scorer, policy, exclude_related=True
        )
        for target in period_types:
            enumerated = set(
                adaptive_candidate_sources(
                    target, policy, prepared, exclude_related=True
                )
            )
            predicted = {
                source
                for source in period_types
                if unrelated_pair_allowed(policy, target, source, base.scorer)
            }
            self.assertEqual(enumerated, predicted)
            # Nothing enumerated may be related.
            for source in enumerated:
                self.assertFalse(
                    bool(candidate_rule_mask(target, source, base.scorer) & RELATED_MASK)
                )

    def test_unrelated_scalar_compile_matches_global_filtered(self):
        global_plan = _plan(mu=0.01)
        period_types = tuple(PeriodType(n, "X") for n in NODES)
        policy = CandidatePolicy(
            {"X": {"X": ("same_ne_type", "same_vendor")}}, approved=True
        )
        global_plan.precompile_period_types(period_types)
        unrelated = CompiledAssociationPlan(
            global_plan.scorer,
            None,
            global_plan.artifact,
            PeriodStreamConfig(candidate_scope="unrelated"),
            candidate_policy=policy,
        )
        unrelated.precompile_period_types(period_types)
        expected = {
            (target, source)
            for target, sources in global_plan.edges_by_target.items()
            for source in sources
            if unrelated_pair_allowed(
                policy, target.period_type, source.period_type, global_plan.scorer
            )
        }
        actual = {
            (target, source)
            for target, sources in unrelated.edges_by_target.items()
            for source in sources
        }
        self.assertEqual(actual, expected)

    def test_unrelated_batch_disjoint_per_alarm_type(self):
        # Two alarm types with different non-local rules must not share a
        # candidate set in the vectorized batch compiler, and no emitted pair
        # may be related (strict disjointness from the related branch).
        base = _plan(dynamic="target", mu=0.01, at_vocab=("a", "b"))
        period_types = tuple(
            PeriodType(node, at) for node in NODES for at in ("a", "b")
        )
        policy = CandidatePolicy(
            {
                "a": {"a": ("same_ne_type",), "b": ("same_ne_type",)},
                "b": {"a": ("same_vendor",), "b": ("same_vendor",)},
            },
            approved=True,
        )

        def _unrelated_plan():
            return CompiledAssociationPlan(
                base.scorer,
                None,
                base.artifact,
                PeriodStreamConfig(candidate_scope="unrelated"),
                candidate_policy=policy,
            )

        scalar = _unrelated_plan()
        scalar.precompile_period_types(period_types)
        expected = {
            (target.period_type, source.period_type)
            for target, sources in scalar.edges_by_target.items()
            for source in sources
        }
        self.assertTrue(expected)

        batch = _unrelated_plan()
        prepared = batch.prepare_candidate_period_types(period_types)
        emitted = set()

        def sink(
            target_type,
            _target_states,
            source_types,
            source_indices,
            _base_scores,
            _threshold,
            _past_windows,
            _future_windows,
        ):
            for i in np.unique(source_indices):
                emitted.add((target_type, source_types[int(i)]))

        pair_count = batch.precompile_period_types(
            period_types,
            prepared_candidates=prepared,
            edge_batch_sink=sink,
        )

        self.assertEqual(pair_count, prepared["total_pair_count"])
        self.assertEqual(emitted, expected)
        for target, source in emitted:
            self.assertFalse(
                bool(candidate_rule_mask(target, source, base.scorer) & RELATED_MASK)
            )

    def test_multiple_precompiled_indexes_union(self):
        # iter_edges unions every loaded cache index plus in-memory edges.
        plan = _plan()
        target = PeriodSignature(PeriodType("A", "X"), 0)

        class _StubIndex:
            def __init__(self, rows):
                self._rows = rows

            def iter_target(self, sig):
                return iter(self._rows.get(sig, ()))

            def iter_source(self, sig):
                return iter(())

            @property
            def memory_bytes(self):
                return 0

        plan.precompiled_indexes = [
            _StubIndex({target: [("k1", "e1")]}),
            _StubIndex({target: [("k2", "e2")]}),
        ]
        plan.edges_by_target[target][
            PeriodSignature(PeriodType("B", "X"), 0)
        ] = "e3"
        got = dict(plan.iter_edges_by_target(target))
        self.assertEqual(got["k1"], "e1")
        self.assertEqual(got["k2"], "e2")
        self.assertEqual(
            got[PeriodSignature(PeriodType("B", "X"), 0)], "e3"
        )

    def test_teacher_any_state_envelope_matches_brute_force(self):
        period_types = tuple(PeriodType(node, "X") for node in NODES)
        for seed in range(5):
            plan = _plan(dynamic="source_target", seed=seed, mu=0.4)
            masks = _teacher_positive_masks(
                plan,
                ["A"],
                ("X",),
                tuple(NODES),
                2,
                quiet=True,
            )
            teacher_positive_count = sum(masks.get(("X", "X"), {}).values())
            brute_positive_count = 0
            for source_type in period_types:
                # The teacher only records non-related positives now.
                if candidate_rule_mask(
                    period_types[0], source_type, plan.scorer
                ) & RELATED_MASK:
                    continue
                found = False
                for target_state in range(8):
                    for source_state in range(8):
                        edge = plan._compute_edge(
                            PeriodSignature(period_types[0], target_state),
                            PeriodSignature(source_type, source_state),
                        )
                        if edge is not None:
                            found = True
                            break
                    if found:
                        break
                brute_positive_count += int(found)
            self.assertEqual(teacher_positive_count, brute_positive_count)


_PROFILED_ENGINE_METHODS = (
    "process",
    "_open_or_create_period",
    "_handle_clear",
    "_close_idle_periods",
    "_advance_watermark",
    "_harvest_ready",
    "_harvest_period",
    "_collect_relations",
    "_best_for_new_targets",
    "_best_for_new_sources",
    "_apply_relations",
    "_choose_or_create_group",
    "_try_ready_merge_proposals",
    "_merge_groups",
    "_close_inactive_groups",
    "_finalize_group",
    "_evict_expired_periods",
    "_group_record",
    "flush",
)


class AlarmPeriodProfilingTest(unittest.TestCase):
    def test_event_iterator_records_only_returned_events(self):
        timer = PhaseTimer()
        self.assertEqual(list(_iter_profiled_events([1, 2, 3], timer)), [1, 2, 3])
        self.assertEqual(timer.snapshot()["input.read_event"]["count"], 3)

    def test_method_wrapping_preserves_calls_and_refreshes_group_sink(self):
        calls = []
        engine = SimpleNamespace()
        for method_name in _PROFILED_ENGINE_METHODS:
            setattr(
                engine,
                method_name,
                lambda value=None, name=method_name: (name, value),
            )
        engine.plan = SimpleNamespace(
            register_signature=lambda value=None: ("register", value),
            _compute_edge=lambda value=None: ("compute", value),
        )
        output = SimpleNamespace(
            emit_group=lambda value: calls.append(value),
            _write_group_record=lambda value=None: value,
            close=lambda value=None: value,
            visual=None,
        )
        timer = PhaseTimer()

        _enable_period_profiling(timer, engine, output)

        self.assertEqual(engine.process("event"), ("process", "event"))
        engine.closed_group_sink("group")
        self.assertEqual(calls, ["group"])
        phases = timer.snapshot()
        self.assertEqual(phases["ingest.process"]["count"], 1)
        self.assertEqual(phases["output.emit_group"]["count"], 1)

    def test_summary_lists_recorded_phase(self):
        timer = PhaseTimer()
        timer.mark_wall_start()
        with timer.time("harvest.collect_relations"):
            pass
        timer.mark_wall_end()
        output = io.StringIO()
        with redirect_stdout(output):
            _print_period_profile(timer)
        text = output.getvalue()
        self.assertIn("AlarmPeriod MHP 性能分析", text)
        self.assertIn("harvest.collect_relations", text)
        self.assertIn("父阶段包含子阶段", text)


def _eviction_engine():
    engine = object.__new__(AlarmPeriodMHPAssigner)
    engine.config = PeriodStreamConfig(
        history_window_sec=10.0,
        aggregation_wait_sec=5.0,
        time_slack_sec=2.0,
    )
    engine._period_retention_sec = 17.0
    engine.periods = {}
    engine.period_ids_by_signature = {}
    engine.period_ids_by_type = {}
    engine._group_redirect = {}
    engine.groups = {}
    engine.merge_proposals = {}
    engine._ready_merge_proposal_keys = set()
    engine._merge_proposal_keys_by_group = defaultdict(set)
    engine.merge_proposal_peak_count = 0
    engine.merge_proposal_pruned_count = 0
    engine._eviction_heap = []
    engine._heap_seq = 0
    engine.evicted_period_count = 0
    engine.eviction_heap_pop_count = 0
    engine.eviction_stale_entry_count = 0
    engine.eviction_group_deferred_count = 0
    engine.closed_group_sink = None
    engine.closed_group_count = 0
    return engine


def _closed_period(period_id=0, group_id=None):
    return AlarmPeriod(
        period_id=period_id,
        period_type=PeriodType("NE", "X"),
        initial_state=(0, 0, 0),
        initial_state_combo=0,
        first_ts=10.0,
        last_raise_ts=10.0,
        status="closed",
        close_ts=11.0,
        close_reason="clear",
        primary_group_id=group_id,
    )


def _index_period(engine, period):
    engine.periods[period.period_id] = period
    engine.period_ids_by_signature.setdefault(period.signature, set()).add(
        period.period_id
    )
    engine.period_ids_by_type.setdefault(period.period_type, set()).add(
        period.period_id
    )


class AlarmPeriodEvictionHeapTest(unittest.TestCase):
    def test_preserves_strict_legacy_expiry_boundary(self):
        engine = _eviction_engine()
        period = _closed_period()
        _index_period(engine, period)
        engine._schedule_period_eviction(period)

        engine._evict_expired_periods(27.0)
        self.assertIn(period.period_id, engine.periods)

        engine._evict_expired_periods(27.001)
        self.assertNotIn(period.period_id, engine.periods)
        self.assertNotIn(period.signature, engine.period_ids_by_signature)
        self.assertNotIn(period.period_type, engine.period_ids_by_type)
        self.assertEqual(engine.evicted_period_count, 1)

    def test_group_finalizes_when_last_candidate_period_expires(self):
        engine = _eviction_engine()
        period = _closed_period(group_id=1)
        _index_period(engine, period)
        engine.groups[1] = PeriodFaultGroup(
            group_id=1,
            anchor_period_id=period.period_id,
            period_ids={period.period_id},
            active_member_count=1,
        )
        engine._group_record = lambda _group: {"event_count": 0, "real_event_count": 0}
        engine._schedule_period_eviction(period)

        engine._evict_expired_periods(30.0)
        self.assertNotIn(period.period_id, engine.periods)
        self.assertNotIn(1, engine.groups)
        self.assertEqual(engine.eviction_group_deferred_count, 0)

    def test_new_generation_makes_older_heap_entry_stale(self):
        engine = _eviction_engine()
        period = _closed_period()
        _index_period(engine, period)
        engine._schedule_period_eviction(period)
        engine._schedule_period_eviction(period)

        engine._evict_expired_periods(30.0)
        self.assertNotIn(period.period_id, engine.periods)
        self.assertEqual(engine.eviction_stale_entry_count, 1)
        self.assertEqual(engine.evicted_period_count, 1)


class AlarmPeriodOutputFilterTest(unittest.TestCase):
    def test_min_site_num_filters_on_unique_non_empty_sites(self):
        engine = _eviction_engine()
        engine.config.min_site_num = 2
        emitted = []
        engine.closed_group_sink = emitted.append

        def finalize(real_site_list):
            engine.groups[1] = PeriodFaultGroup(group_id=1, anchor_period_id=0)
            engine._group_record = lambda _group: {
                "real_event_count": 2,
                "real_site_list": real_site_list,
            }
            engine._finalize_group(1)

        finalize(["S1"])
        self.assertEqual(emitted, [])
        self.assertEqual(engine.closed_group_count, 0)

        finalize(["S1", "S2"])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(engine.closed_group_count, 1)

    def test_min_site_num_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "min_site_num must be >= 1"):
            PeriodStreamConfig(min_site_num=0).validate()


class AlarmPeriodMergeProposalQueueTest(unittest.TestCase):
    def test_only_newly_ready_proposals_are_examined(self):
        engine = _eviction_engine()
        engine.config.merge_min_evidence = 2
        engine.config.merge_strength_ratio = 2.0
        for gid in range(1, 5):
            engine.groups[gid] = PeriodFaultGroup(group_id=gid, anchor_period_id=gid)

        def evidence(pair, strength):
            return SimpleNamespace(
                period_pair=pair,
                strength=strength,
                score=strength,
            )

        engine._record_merge_proposal(1, 2, evidence((10, 11), 3.0))
        engine._record_merge_proposal(3, 4, evidence((20, 21), 3.0))
        engine._record_merge_proposal(3, 4, evidence((22, 23), 3.0))
        self.assertEqual(engine._ready_merge_proposal_keys, {(3, 4)})

        merged = []
        engine._merge_groups = lambda left, right: merged.append((left, right))
        engine._try_ready_merge_proposals()

        self.assertEqual(merged, [(3, 4)])
        self.assertIn((1, 2), engine.merge_proposals)
        self.assertNotIn((3, 4), engine.merge_proposals)

    def test_group_cleanup_removes_stale_proposals(self):
        engine = _eviction_engine()
        for gid in range(1, 4):
            engine.groups[gid] = PeriodFaultGroup(group_id=gid, anchor_period_id=gid)
        rel = SimpleNamespace(period_pair=(10, 11), strength=1.0, score=1.0)
        engine._record_merge_proposal(1, 2, rel)
        engine._record_merge_proposal(1, 3, rel)

        engine._prune_merge_proposals_for_group(1)

        self.assertEqual(engine.merge_proposals, {})
        self.assertEqual(engine._merge_proposal_keys_by_group, {})
        self.assertEqual(engine.merge_proposal_pruned_count, 2)


class AlarmPeriodSignatureRegistrationTest(unittest.TestCase):
    def test_reopened_period_registers_each_signature_only_once(self):
        engine = object.__new__(AlarmPeriodMHPAssigner)
        engine._next_period_id = 0
        engine.created_periods = 0
        engine.periods = {}
        engine.open_period_by_type = {}
        engine.period_ids_by_signature = defaultdict(set)
        engine.period_ids_by_type = defaultdict(set)
        engine._seen_period_signatures = set()
        registered = []
        engine.plan = SimpleNamespace(register_signature=registered.append)
        period_type = PeriodType("NE-1", "A")

        first = engine._open_or_create_period(
            period_type, SimpleNamespace(ts=1.0), (0, 0, 0)
        )
        first.close(2.0, "test")
        second = engine._open_or_create_period(
            period_type, SimpleNamespace(ts=3.0), (0, 0, 0)
        )
        second.close(4.0, "test")
        engine._open_or_create_period(
            period_type, SimpleNamespace(ts=5.0), (1, 0, 0)
        )

        self.assertEqual(
            registered,
            [
                PeriodSignature(period_type, 0),
                PeriodSignature(period_type, 1),
            ],
        )


class PeriodSourceImputationRegressionTest(unittest.TestCase):
    @staticmethod
    def _virtual_engine(node_field="alarm_source"):
        engine = object.__new__(AlarmPeriodMHPAssigner)
        engine.artifact = SimpleNamespace(
            config=SimpleNamespace(topology_node_field=node_field)
        )
        engine.feature_scorer = SimpleNamespace(
            node_infos={
                "A": SimpleNamespace(site_id="S1"),
                "B": SimpleNamespace(site_id="S2"),
            }
        )
        engine._next_period_id = 0
        engine._next_event_index = 0
        engine.periods = {}
        engine._period_retention_sec = 10.0
        engine._eviction_heap = []
        engine._heap_seq = 0
        return engine

    def test_virtual_summary_preserves_model_alarm_type(self):
        engine = self._virtual_engine()
        period = engine.create_virtual_source_period(
            PeriodSignature(PeriodType("A", "link"), 0), 50.0
        )
        group = PeriodFaultGroup(
            group_id=1,
            anchor_period_id=period.period_id,
            period_ids={period.period_id},
        )

        record = engine._group_record(group)
        symptom = record["symptoms"][0]
        self.assertEqual(symptom["alarm_type"], "link")
        self.assertEqual(symptom["alarm_title"], "")
        self.assertEqual(symptom["alarm_source"], "A")
        self.assertEqual(symptom["site_id"], "S1")

        site_engine = self._virtual_engine(node_field="site_id")
        site_period = site_engine.create_virtual_source_period(
            PeriodSignature(PeriodType("SITE-1\x1fRAN", "power"), 0), 60.0
        )
        site_group = PeriodFaultGroup(
            group_id=2,
            anchor_period_id=site_period.period_id,
            period_ids={site_period.period_id},
        )
        site_symptom = site_engine._group_record(site_group)["symptoms"][0]
        self.assertEqual(site_symptom["alarm_type"], "power")
        self.assertEqual(site_symptom["site_id"], "SITE-1")
        self.assertEqual(site_symptom["alarm_source"], "")
        self.assertEqual(site_period.events[0].alarm["device_domain"], "RAN")

    def test_real_sites_are_topology_enriched_and_exclude_virtual_sites(self):
        engine = self._virtual_engine()
        real = AlarmPeriod(
            period_id=10,
            period_type=PeriodType("A", "link"),
            initial_state=(0, 0, 0),
            initial_state_combo=0,
            first_ts=50.0,
            last_raise_ts=50.0,
        )
        real.append(
            OnlineEvent(
                index=10,
                ts=50.0,
                type_id=-1,
                type_label="link",
                alarm={
                    "eid": "real-10",
                    "occurrence_uuid": "00000000-0000-0000-0000-000000000010",
                    "ts": 50.0,
                    "alarm_source": "A",
                    "alarm_title": "",
                    "alarm": {},
                },
                alarm_type="link",
                ne="A",
            )
        )
        engine.periods[real.period_id] = real
        virtual = engine.create_virtual_source_period(
            PeriodSignature(PeriodType("B", "power"), 0), 49.0
        )
        group = PeriodFaultGroup(
            group_id=3,
            anchor_period_id=real.period_id,
            period_ids={real.period_id, virtual.period_id},
        )
        edge = CompiledEdge(10.0, 1.0, 60.0, 0.0)
        group.evidence_by_pair[(virtual.period_id, real.period_id)] = RelationEvidence(
            target_period_id=real.period_id,
            source_period_id=virtual.period_id,
            target_event=real.events[0],
            source_event=virtual.events[0],
            score=10.0,
            strength=10.0,
            edge=edge,
        )

        record = engine._group_record(group)
        self.assertEqual(record["site_list"], ["S1", "S2"])
        self.assertEqual(record["real_site_list"], ["S1"])
        self.assertEqual(record["real_site_count"], 1)
        virtual_symptom = next(s for s in record["symptoms"] if s.get("virtual"))
        visual = _symptom_to_visual_record_mhp(virtual_symptom)
        self.assertTrue(visual["virtual"])
        self.assertEqual(record["edges"][0]["source_virtual"], True)
        self.assertEqual(record["edges"][0]["target_virtual"], False)

    def test_min_score_ratio_is_a_pre_kappa_guard(self):
        edge = CompiledEdge(
            base_score=10.0,
            threshold=1.0,
            past_window_sec=60.0,
            future_window_sec=0.0,
        )
        target = _closed_period(period_id=1, group_id=1)
        target.period_type = PeriodType("A", "X")
        target.events = [SimpleNamespace(index=1)]
        group = PeriodFaultGroup(
            group_id=1,
            anchor_period_id=target.period_id,
            period_ids={target.period_id},
        )
        engine = SimpleNamespace(
            config=SimpleNamespace(time_slack_sec=1.0),
            groups={1: group},
            plan=SimpleNamespace(
                iter_edges_by_target=lambda _sig: iter(
                    [(PeriodType("B", "Y"), edge)]
                )
            ),
            _resolve_group_id=lambda gid: gid,
            _past_score=lambda _edge, _dt: 10.0,
        )
        imputer = PeriodSourceImputer(
            engine,
            PeriodImputeConfig(
                enabled=True,
                kappa=-2.0,
                min_score_ratio=2.0,
            ),
        )

        best = imputer._best_candidate(target)
        self.assertIsNotNone(best)

    def test_candidate_cap_uses_cached_top_index_and_bounds_scoring(self):
        period_types = [PeriodType("A", "X")] + [
            PeriodType(f"S{i:04d}", "Y") for i in range(1000)
        ]
        edge_count = len(period_types) - 1
        arrays = build_compact_csr_arrays(
            target_signature_ids=np.zeros(edge_count, dtype=np.int64),
            source_signature_ids=np.arange(1, len(period_types), dtype=np.int64),
            base_scores=np.arange(1, edge_count + 1, dtype=np.float64),
            thresholds=np.ones(edge_count, dtype=np.float64),
            past_windows=np.full(edge_count, 60.0, dtype=np.float64),
            future_windows=np.zeros(edge_count, dtype=np.float64),
            signature_count=len(period_types) * 8,
            source_key_count=len(period_types),
        )
        index = CompactAssociationIndex(
            period_types,
            arrays,
            state_layout=CACHE_STATE_LAYOUT_TARGET_ONLY,
        )
        plan = object.__new__(CompiledAssociationPlan)
        plan.precompiled_indexes = [index]
        plan.edges_by_target = {}
        plan._target_edge_versions = {}
        plan._top_edges_by_target_cache = {}
        target = _closed_period(period_id=1, group_id=1)
        target.period_type = PeriodType("A", "X")
        score_calls = []
        engine = SimpleNamespace(
            config=SimpleNamespace(time_slack_sec=1.0),
            plan=plan,
            _past_score=lambda edge, _dt: (
                score_calls.append(edge.base_score) or edge.base_score
            ),
        )
        imputer = PeriodSourceImputer(
            engine,
            PeriodImputeConfig(enabled=True, kappa=-2.0, max_candidates=1),
        )

        first = imputer._best_candidate(target)
        second = imputer._best_candidate(target)
        self.assertEqual(first[1].period_type.entity, "S0999")
        self.assertEqual(second[1].period_type.entity, "S0999")
        self.assertEqual(score_calls, [1000.0, 1000.0])
        self.assertEqual(len(plan._top_edges_by_target_cache), 1)

        dynamic_source = PeriodSignature(PeriodType("DYNAMIC", "Y"), 0)
        plan.edges_by_target[target.signature] = {
            dynamic_source: CompiledEdge(2000.0, 1.0, 60.0, 0.0)
        }
        plan._target_edge_versions[target.signature] = 1
        refreshed = imputer._best_candidate(target)
        self.assertEqual(refreshed[1].period_type.entity, "DYNAMIC")
        self.assertEqual(score_calls, [1000.0, 1000.0, 2000.0])
        self.assertEqual(len(plan._top_edges_by_target_cache), 1)

    def test_cached_top_index_breaks_cutoff_ties_by_source_id(self):
        period_types = [
            PeriodType("TARGET", "X"),
            PeriodType("S1", "Y"),
            PeriodType("S2", "Y"),
            PeriodType("S3", "Y"),
        ]

        def selected_entities(source_ids):
            edge_count = len(source_ids)
            arrays = build_compact_csr_arrays(
                target_signature_ids=np.zeros(edge_count, dtype=np.int64),
                source_signature_ids=np.asarray(source_ids, dtype=np.int64),
                base_scores=np.ones(edge_count, dtype=np.float64),
                thresholds=np.ones(edge_count, dtype=np.float64),
                past_windows=np.full(edge_count, 60.0, dtype=np.float64),
                future_windows=np.zeros(edge_count, dtype=np.float64),
                signature_count=len(period_types) * 8,
                source_key_count=len(period_types),
            )
            index = CompactAssociationIndex(
                period_types,
                arrays,
                state_layout=CACHE_STATE_LAYOUT_TARGET_ONLY,
            )
            target = PeriodSignature(period_types[0], 0)
            return [
                source.entity
                for source, _edge in index.top_target(target, 2)
            ]

        self.assertEqual(selected_entities([3, 1, 2]), ["S1", "S2"])
        self.assertEqual(selected_entities([2, 3, 1]), ["S1", "S2"])

    def test_dynamic_top_edges_break_ties_by_source_signature(self):
        target = PeriodSignature(PeriodType("TARGET", "X"), 0)
        source_a = PeriodSignature(PeriodType("A", "Y"), 0)
        source_b = PeriodSignature(PeriodType("B", "Y"), 0)
        edge = CompiledEdge(10.0, 1.0, 60.0, 0.0)
        plan = object.__new__(CompiledAssociationPlan)
        plan.precompiled_indexes = []
        plan.edges_by_target = {target: {source_b: edge, source_a: edge}}
        plan._target_edge_versions = {target: 2}
        plan._top_edges_by_target_cache = {}

        selected = plan.top_edges_by_target(target, 1)

        self.assertEqual(selected[0][0], source_a)

    def test_overlapping_caches_do_not_consume_multiple_candidate_slots(self):
        target = _closed_period(period_id=1, group_id=1)
        target.period_type = PeriodType("TARGET", "X")
        rejected_source = PeriodSignature(PeriodType("A", "Y"), 0)
        accepted_source = PeriodSignature(PeriodType("B", "Y"), 0)
        rejected_edge = CompiledEdge(100.0, 50.0, 60.0, 0.0)
        accepted_edge = CompiledEdge(90.0, 1.0, 60.0, 0.0)

        class StubIndex:
            def __init__(self, rows):
                self.rows = rows

            def top_target(self, _signature, _limit, _min_past_window):
                return list(self.rows)

        plan = object.__new__(CompiledAssociationPlan)
        plan.precompiled_indexes = [
            StubIndex(
                [
                    (rejected_source, rejected_edge),
                    (accepted_source, accepted_edge),
                ]
            ),
            StubIndex([(rejected_source, rejected_edge)]),
        ]
        plan.edges_by_target = {}
        plan._target_edge_versions = {}
        plan._top_edges_by_target_cache = {}
        engine = SimpleNamespace(
            config=SimpleNamespace(time_slack_sec=1.0),
            plan=plan,
            _past_score=lambda edge, _dt: edge.base_score,
        )
        imputer = PeriodSourceImputer(
            engine,
            PeriodImputeConfig(enabled=True, kappa=-2.0, max_candidates=2),
        )

        best = imputer._best_candidate(target)

        self.assertIsNotNone(best)
        self.assertEqual(best[1], accepted_source)

    def test_zero_lag_uses_zero_map_offset(self):
        edge = CompiledEdge(
            base_score=10.0,
            threshold=1.0,
            past_window_sec=0.5,
            future_window_sec=0.0,
        )
        seen_dt = []
        target = _closed_period(period_id=1, group_id=1)
        target.period_type = PeriodType("A", "X")
        engine = SimpleNamespace(
            config=SimpleNamespace(time_slack_sec=0.0),
            plan=SimpleNamespace(
                iter_edges_by_target=lambda _sig: iter(
                    [(PeriodType("B", "Y"), edge)]
                )
            ),
            _past_score=lambda candidate, dt: (
                seen_dt.append(dt) or candidate.base_score
            ),
        )
        imputer = PeriodSourceImputer(
            engine,
            PeriodImputeConfig(enabled=True, kappa=-2.0, lag_sec=0.0),
        )

        self.assertIsNotNone(imputer._best_candidate(target))
        self.assertEqual(seen_dt, [0.0])
        self.assertEqual(imputer._source_offset_sec(), 0.0)

    def test_config_rejects_non_finite_and_rewarding_values(self):
        invalid = (
            {"kappa": 0.1},
            {"kappa": float("nan")},
            {"kappa": float("inf")},
            {"lag_sec": float("nan")},
            {"lag_sec": float("inf")},
            {"min_score_ratio": float("nan")},
            {"min_score_ratio": float("inf")},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PeriodImputeConfig(enabled=True, **kwargs).validate()

    def test_config_metadata_contains_effective_imputation_knobs(self):
        config = PeriodImputeConfig(
            enabled=True,
            kappa=-3.0,
            max_candidates=7,
            lag_sec=2.5,
            min_score_ratio=1.5,
        )
        self.assertEqual(
            config.to_dict(),
            {
                "enabled": True,
                "kappa": -3.0,
                "max_candidates": 7,
                "lag_sec": 2.5,
                "min_score_ratio": 1.5,
            },
        )


class ClearMetricNullSamplingTest(unittest.TestCase):
    def test_draws_per_alarm_caps_distinct_cross_group_partners(self):
        alarms = [
            ClearedAlarm("anchor", "A", 0.0, 0.0),
            ClearedAlarm("b1", "B", 1.0, 10.0),
            ClearedAlarm("b2", "B", 2.0, 20.0),
            ClearedAlarm("b3", "B", 3.0, 30.0),
            ClearedAlarm("b4", "B", 4.0, 40.0),
            ClearedAlarm("b5", "B", 5.0, 50.0),
        ]

        deltas = _sample_null_pairs(
            alarms,
            occ_window_sec=100.0,
            max_null_pairs=100,
            draws_per_alarm=3,
            rng=random.Random(0),
        )

        self.assertEqual(len(deltas), 3)
        self.assertEqual(len(deltas), len(set(deltas)))


if __name__ == "__main__":
    unittest.main()
