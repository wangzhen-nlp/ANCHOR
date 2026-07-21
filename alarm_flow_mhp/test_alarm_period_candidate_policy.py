from pathlib import Path
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
from alarm_flow_mhp.learn_alarm_period_candidate_policy import (
    _teacher_positive_masks,
)
from alarm_flow_mhp.stream_alarm_period_mhp import (
    CompiledAssociationPlan,
    PeriodSignature,
    PeriodStreamConfig,
    PeriodType,
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
            teacher_positive_count = sum(masks[("X", "X")].values())
            brute_positive_count = 0
            for source_type in period_types:
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


if __name__ == "__main__":
    unittest.main()
