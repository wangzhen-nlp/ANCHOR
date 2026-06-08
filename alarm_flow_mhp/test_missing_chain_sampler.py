#!/usr/bin/env python3
"""Unit tests for the fixed-lag missing-chain sampler.

Self-contained: builds a tiny exp-kernel adapter, no trained artifact / real
data needed. Run directly (``python test_missing_chain_sampler.py``) or via
pytest.
"""

from __future__ import annotations

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

import numpy as np

from alarm_flow_mhp.missing_chain_sampler import (
    MHP_VIRTUAL_RULE,
    ExpKernelAdapter,
    FeatureKernelAdapter,
    MissingChainSampler,
    SamplerConfig,
)
from alarm_flow_mhp.dynamic_state import ObservedStateTimeline


# --- minimal fakes mimicking the feature-mode runtime scorers ---------------


class _FakeTopo:
    """Stand-in for NETopologyIndex: undirected hop adjacency only."""

    def __init__(self, adjacency: dict, max_hops: int = 1):
        # adjacency: {ne: {neighbor: hop}}
        self.undirected_hops = adjacency
        self.max_hops = max_hops


class _FakeNodeInfo:
    def __init__(self, site_id=""):
        self.site_id = site_id


class _FakeFeatureScorer:
    """Mimics RuntimeFeatureScorer.alpha_for_target with a simple rule:

    α((t_at,t_ne) ← (s_at,s_ne)) = trigger[s_at→t_at] · topo(s_ne→t_ne)
    where topo = 1.0 if same NE or a graph neighbour, else 0.0.
    """

    def __init__(self, at_vocab, trigger, topo: _FakeTopo, node_infos, beta=1.0):
        self.at_to_id = {a: i for i, a in enumerate(at_vocab)}
        self.trigger = trigger              # {(s_at, t_at): alpha}
        self.topology_index = topo
        self.node_infos = node_infos
        self.beta = beta

    def _topo_ok(self, s_ne, t_ne):
        if s_ne == t_ne:
            return 1.0
        return 1.0 if s_ne in self.topology_index.undirected_hops.get(t_ne, {}) else 0.0

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None):
        out = np.zeros(len(src_nes), dtype=np.float64)
        for i, (s_at, s_ne) in enumerate(zip(src_ats, src_nes)):
            base = self.trigger.get((s_at, target_at), 0.0)
            out[i] = base * self._topo_ok(s_ne, target_ne)
        return out

    def alpha_for_source(self, source_at, source_ne, tgt_ats, tgt_nes, src_mark=None):
        out = np.zeros(len(tgt_nes), dtype=np.float64)
        for i, (t_at, t_ne) in enumerate(zip(tgt_ats, tgt_nes)):
            base = self.trigger.get((source_at, t_at), 0.0)
            out[i] = base * self._topo_ok(source_ne, t_ne)
        return out


class _FakeDynamicFeatureScorer(_FakeFeatureScorer):
    """Adds source-state gain to the static fake scorer."""

    n_dynamic = 3

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None):
        base = super().alpha_for_target(target_at, target_ne, src_ats, src_nes)
        marks = np.zeros((len(src_nes), self.n_dynamic), dtype=np.float64)
        if src_marks is not None:
            marks = np.asarray(src_marks, dtype=np.float64).reshape(len(src_nes), self.n_dynamic)
        # Only a source with active power state can trigger B in this fake model.
        return base * marks[:, 1]

    def alpha_for_source(self, source_at, source_ne, tgt_ats, tgt_nes, src_mark=None):
        base = super().alpha_for_source(source_at, source_ne, tgt_ats, tgt_nes)
        m = (np.zeros(self.n_dynamic) if src_mark is None
             else np.asarray(src_mark, dtype=np.float64).reshape(self.n_dynamic))
        return base * m[1]


class _FakeSourceTargetFeatureScorer(_FakeFeatureScorer):
    """Requires source power AND target link to activate the edge."""

    n_dynamic = 6

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None, tgt_marks=None):
        base = super().alpha_for_target(target_at, target_ne, src_ats, src_nes)
        marks = np.zeros((len(src_nes), self.n_dynamic), dtype=np.float64)
        if src_marks is not None:
            marks = np.asarray(src_marks, dtype=np.float64).reshape(len(src_nes), self.n_dynamic)
        return base * marks[:, 1] * marks[:, 3]

    def alpha_for_source(self, source_at, source_ne, tgt_ats, tgt_nes, src_mark=None, tgt_marks=None):
        base = super().alpha_for_source(source_at, source_ne, tgt_ats, tgt_nes)
        src = np.zeros(3) if src_mark is None else np.asarray(src_mark, dtype=np.float64).reshape(-1)[:3]
        if tgt_marks is None:
            tgt = np.zeros((len(tgt_nes), 3), dtype=np.float64)
        else:
            tgt = np.asarray(tgt_marks, dtype=np.float64).reshape(len(tgt_nes), -1)[:, :3]
        return base * src[1] * tgt[:, 0]


class _FakeMuScorer:
    def __init__(self, mu_by_at):
        self.mu_by_at = mu_by_at

    def mu_for(self, alarm_type, ne):
        return self.mu_by_at.get(alarm_type, 0.0)


def _build_feature_adapter():
    # NE graph: n1 <-> n2 (neighbours), n3 isolated.
    topo = _FakeTopo({"n1": {"n2": 1}, "n2": {"n1": 1}, "n3": {}}, max_hops=1)
    node_infos = {"n1": _FakeNodeInfo("siteA"), "n2": _FakeNodeInfo("siteA"),
                  "n3": _FakeNodeInfo("siteB")}
    # S triggers B, R triggers S — both strong.
    trigger = {("S", "B"): 3.0, ("R", "S"): 3.0}
    fs = _FakeFeatureScorer(
        at_vocab=["S", "B", "R"], trigger=trigger, topo=topo,
        node_infos=node_infos, beta=1.0,
    )
    mu = _FakeMuScorer({"S": 0.05, "B": 0.0005, "R": 0.05})
    return FeatureKernelAdapter(fs, mu_scorer=mu, time_scale_sec=60.0, alpha_floor=0.0)


def _events_of_type(sampler, type_id):
    return [e for e in sampler.events.values() if e.type_id == type_id]


def test_parent_gibbs_picks_strong_edge():
    """B should bind to a recent A when A→B is strong and μ_B is tiny."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.1, 1: 0.001},
        edges={(1, 0): (2.0, 1.0)},      # target B(1) ← source A(0)
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter, SamplerConfig(lag_sec=1e9, sweeps_per_tick=5, seed=1)
    )
    s.ingest(0.0, 0)        # A
    s.ingest(30.0, 1)       # B, 30s later
    a = _events_of_type(s, 0)[0]
    b = _events_of_type(s, 1)[0]
    assert b.parent == a.eid, f"B should pick A, got parent={b.parent}"
    # No spurious missing events for a well-explained pair.
    assert s._missing_count == 0


def test_birth_explains_orphan_with_missing_parent():
    """An orphan B with a strong hidden source S and tiny μ_B should get a
    missing S parent imputed."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005},
        edges={(1, 0): (3.0, 1.0)},      # B(1) ← S(0), strong
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=10, missing_log_prior=0.0, seed=2),
    )
    s.ingest(100.0, 1)      # only B observed; S never observed
    b = _events_of_type(s, 1)[0]
    missing = [e for e in s.events.values() if e.is_missing()]
    assert missing, "expected a missing parent to be born"
    assert any(m.type_id == 0 for m in missing), "missing parent should be type S(0)"
    assert b.parent != -1, "B should no longer be an immigrant"
    assert not s.events[b.parent].observed, "B's parent should be the missing event"
    assert s.births >= 1


def test_multi_hop_chain():
    """R→S→B: imputing S to explain B, then R to explain S (depth 2)."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005, 2: 0.0005},
        edges={(2, 1): (3.0, 1.0), (1, 0): (3.0, 1.0)},   # B←S, S←R
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(
            lag_sec=1e9, sweeps_per_tick=30, missing_log_prior=0.0,
            max_depth=4, seed=3,
        ),
    )
    s.ingest(100.0, 2)      # only B observed
    depths = {e.type_id: e.depth for e in s.events.values() if e.is_missing()}
    assert any(e.depth >= 2 for e in s.events.values() if e.is_missing()), (
        f"expected a depth>=2 missing event, got {depths}"
    )
    # A depth-2 missing event of type R(0) should exist.
    assert any(e.type_id == 0 and e.is_missing() for e in s.events.values())


def test_max_depth_caps_chain():
    """With max_depth=1, the chain must not grow beyond a single missing hop."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005, 2: 0.0005},
        edges={(2, 1): (3.0, 1.0), (1, 0): (3.0, 1.0)},
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(
            lag_sec=1e9, sweeps_per_tick=30, missing_log_prior=0.0,
            max_depth=1, seed=4,
        ),
    )
    s.ingest(100.0, 2)
    assert all(e.depth <= 1 for e in s.events.values()), "depth cap violated"


def test_death_removes_childless_missing():
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.01},
        edges={(1, 0): (1.0, 1.0)},
        time_scale_sec=60.0,
    )
    # Strong negative prior ⇒ a childless missing event should always be culled.
    s = MissingChainSampler(
        adapter, SamplerConfig(lag_sec=1e9, missing_log_prior=-50.0, seed=5)
    )
    s.now = 200.0
    x = s._new_event(ts=50.0, type_id=0, observed=False, meta={}, depth=1)
    s._insert_ordered(x)
    assert x.eid in s.events
    removed = s._try_death(x)
    assert removed and x.eid not in s.events


def test_death_respects_real_parent_support():
    """A childless missing event with a strong REAL parent must not be over-
    deleted (death uses actual incoming intensity, not μ). The same node as an
    immigrant with tiny μ is readily culled."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 1e-6, 1: 0.01},     # the missing type (0) has tiny μ
        edges={(0, 1): (1e6, 1.0)},        # p(type1) → x(type0) extremely strong
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter, SamplerConfig(lag_sec=1e9, missing_log_prior=0.0, seed=5)
    )
    s.now = 200.0
    p = s._new_event(ts=40.0, type_id=1, observed=True, meta={}, depth=0)
    s._insert_ordered(p)
    x = s._new_event(ts=50.0, type_id=0, observed=False, meta={}, depth=1)
    s._insert_ordered(x)
    s._set_parent(x, p.eid)                # x is strongly supported by parent p
    # High incoming intensity ⇒ death ratio ~ exp(-large) ⇒ rejected.
    assert s._try_death(x) is False
    assert x.eid in s.events
    # Detach → immigrant with tiny μ ⇒ now readily culled.
    s._set_parent(x, -1)
    assert s._try_death(x) is True
    assert x.eid not in s.events


def test_orphan_and_missing_indices_stay_consistent():
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005},
        edges={(1, 0): (3.0, 1.0)},
        time_scale_sec=60.0,
        meta_by_type={0: {"alarm_type": "S"}},
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=100.0, history_window_sec=300.0, sweeps_per_tick=10,
                      missing_log_prior=0.0, seed=21),
    )
    s.ingest(100.0, 1, {"event_id": "B"})

    missing = [e for e in s.events.values() if e.is_missing()]
    assert missing, "expected an imputed missing event"
    assert set(s._missing_set) == {e.eid for e in missing}
    assert set(s._orphan_list) == set(s._orphan_idx)
    for i, eid in enumerate(s._orphan_list):
        assert s._orphan_idx[eid] == i
        ev = s.events[eid]
        assert ev.parent == -1 and not ev.committed

    groups = s.flush()
    assert groups
    assert not s.events
    assert not s._orphan_list and not s._orphan_idx and not s._missing_set


def test_strong_prior_suppresses_births():
    """A very negative κ prior should keep the sampler from inventing events."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005},
        edges={(1, 0): (3.0, 1.0)},
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=10, missing_log_prior=-50.0, seed=6),
    )
    s.ingest(100.0, 1)
    assert s._missing_count == 0, "strong prior should suppress imputation"


def test_close_emits_group_with_parent_linkage():
    """A→B cascade closes as one group with B linked to A."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.1, 1: 0.001},
        edges={(1, 0): (2.0, 1.0)},
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=100.0, history_window_sec=300.0, sweeps_per_tick=5, seed=7),
    )
    s.ingest(0.0, 0, {"event_id": "A"})
    s.ingest(30.0, 1, {"event_id": "B"})
    groups = s.ingest(5000.0, 0, {"event_id": "C"})  # push A,B fully out of reach
    ab = [g for g in groups if g["event_count"] >= 2]
    assert ab, f"expected the A→B group to close, got {groups}"
    g = ab[0]
    assert g["real_event_count"] == 2 and g["virtual_event_count"] == 0
    b = [e for e in g["symptoms"] if e["event_id"] == "B"][0]
    assert b["parent_event_id"] == "A", "B should link to A"
    assert g["rule"] == "alarm_flow_mhp"


def test_group_includes_missing_events_with_markers():
    """A closed group must carry imputed missing events tagged virtual."""
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005},
        edges={(1, 0): (3.0, 1.0)},
        time_scale_sec=60.0,
        meta_by_type={0: {"alarm_type": "S", "alarm_source": "ne0", "type_label": "ne0 | S"}},
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=100.0, history_window_sec=300.0, sweeps_per_tick=10,
                      missing_log_prior=0.0, seed=9),
    )
    s.ingest(100.0, 1, {"event_id": "B", "alarm_source": "ne1"})
    groups = s.flush()
    g = [grp for grp in groups if grp["virtual_event_count"] >= 1]
    assert g, f"expected a group containing an imputed missing event, got {groups}"
    grp = g[0]
    assert MHP_VIRTUAL_RULE in grp["merged_rules"]
    ve = [e for e in grp["symptoms"] if e["virtual"]]
    assert ve and ve[0]["confidence"] >= 0.0 and ve[0]["inferred_virtual"] is True
    # the observed B should point at the missing event
    b = [e for e in grp["symptoms"] if e["event_id"] == "B"][0]
    assert b["parent_virtual"] is True


def test_flush_closes_everything():
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.1, 1: 0.01},
        edges={(1, 0): (1.0, 1.0)},
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(adapter, SamplerConfig(lag_sec=100.0, seed=8))
    s.ingest(0.0, 0, {"event_id": "A"})
    s.ingest(10.0, 1, {"event_id": "B"})
    groups = s.flush()
    total_real = sum(g["real_event_count"] for g in groups)
    assert total_real == 2, f"flush should emit both observed events, got {total_real}"
    assert all("base_group_id" not in g for g in groups)
    assert all("related_group_uuids" not in g for g in groups)


def test_feature_candidate_sources_bounded_by_topology():
    """Candidate missing parents for B@n1 come from {n1, n2}×at_vocab; n3
    (non-neighbour) must not appear, and only real triggers (S→B) survive."""
    adapter = _build_feature_adapter()
    cands = dict(adapter.candidate_sources(("B", "n1")))
    assert ("S", "n1") in cands and ("S", "n2") in cands
    assert all(ne in ("n1", "n2") for (_at, ne) in cands), "n3 should be unreachable"
    # Only S→B has nonzero trigger.
    assert all(at == "S" for (at, _ne) in cands)


def test_feature_birth_imputes_missing_parent():
    adapter = _build_feature_adapter()
    s = MissingChainSampler(
        adapter,
        # max_depth=1 isolates the single-hop intent (the fake adapter also has
        # R→S, which would otherwise legitimately grow a 2nd hop).
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=10, missing_log_prior=0.0,
                      max_depth=1, seed=11),
    )
    s.ingest(100.0, ("B", "n1"))         # orphan B on device n1
    missing = [e for e in s.events.values() if e.is_missing()]
    assert missing, "expected an imputed missing parent in feature mode"
    assert all(mt[0] == "S" for mt in (m.type_id for m in missing)), "parent type should be S"
    b = [e for e in s.events.values() if e.type_id == ("B", "n1")][0]
    assert b.parent != -1 and not s.events[b.parent].observed


def test_feature_multi_hop_chain():
    adapter = _build_feature_adapter()
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=30, missing_log_prior=0.0,
                      max_depth=4, seed=1),
    )
    s.ingest(100.0, ("B", "n1"))
    assert any(e.depth >= 2 for e in s.events.values() if e.is_missing()), \
        "expected R→S→B multi-hop chain"
    assert any(e.type_id[0] == "R" for e in s.events.values() if e.is_missing())


def test_feature_total_compensator_cached_and_positive():
    adapter = _build_feature_adapter()
    # S(n1) can trigger B on {n1,n2} → positive outgoing mass; cached on 2nd call.
    c1 = adapter.total_compensator(("S", "n1"), 600.0)
    assert c1 > 0.0
    assert (("S", "n1"), ()) in adapter._alpha_out_sum
    c2 = adapter.total_compensator(("S", "n1"), 600.0)
    assert c1 == c2
    # Larger horizon ⇒ larger (or equal) survival mass.
    assert adapter.total_compensator(("S", "n1"), 6000.0) >= c1


def test_device_adapter_from_artifact_roundtrip():
    """device_adapter_from_artifact builds a working adapter from numpy edge
    arrays (the stream integration seam)."""
    from alarm_flow_mhp.missing_chain_sampler import device_adapter_from_artifact

    class _P:
        kernel_type = "exp"
        edge_targets = np.array([1])
        edge_sources = np.array([0])
        edge_alpha = np.array([2.0])
        edge_beta = np.array([1.0])
        mu = np.array([0.1, 0.001])

    class _C:
        time_scale_sec = 60.0

    class _A:
        params = _P()
        config = _C()

    adapter = device_adapter_from_artifact(_A())
    assert adapter.kernel_intensity(0, 1, 30.0) > 0
    assert adapter.mu(1) == 0.001
    # outgoing-target index wired → compensator active
    assert adapter.outgoing_targets(0) == [1]
    s = MissingChainSampler(adapter, SamplerConfig(lag_sec=1e9, sweeps_per_tick=5, seed=1))
    s.ingest(0.0, 0)
    s.ingest(30.0, 1)
    b = [e for e in s.events.values() if e.type_id == 1][0]
    a = [e for e in s.events.values() if e.type_id == 0][0]
    assert b.parent == a.eid


def test_group_is_visual_output_compatible():
    """A sampler group (with an imputed missing event) round-trips through
    alarm_flow_brunch.visual_output.group_to_visual_match preserving markers."""
    from alarm_flow_brunch.visual_output import group_to_visual_match

    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.05, 1: 0.0005},
        edges={(1, 0): (3.0, 1.0)},
        time_scale_sec=60.0,
        meta_by_type={0: {"alarm_type": "S", "alarm_source": "ne0", "type_label": "ne0 | S"}},
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=100.0, history_window_sec=300.0, sweeps_per_tick=10,
                      missing_log_prior=0.0, seed=9),
    )
    s.ingest(100.0, 1, {"event_id": "B", "alarm_source": "ne1", "site_id": "siteA",
                        "alarm_title": "B-alarm", "alarm_type": "B"})
    groups = [g for g in s.flush() if g["virtual_event_count"] >= 1]
    assert groups, "expected a group with an imputed missing event"
    vm = group_to_visual_match(groups[0], None)   # no NE graph needed
    assert vm["symptoms"], "visual match should carry symptoms"
    virt = [r for r in vm["symptoms"] if r["virtual"]]
    assert virt, "visual symptoms should include the imputed (virtual) node"
    assert virt[0]["confidence"] >= 0.0
    from alarm_flow_brunch.visual_output import BRUNCH_VIRTUAL_RULE  # noqa
    assert MHP_VIRTUAL_RULE in vm["merged_rules"]


def test_feature_alpha_cache_is_bounded():
    """The per-pair α cache must not grow without limit over a long stream."""
    fs = _FakeFeatureScorer(
        at_vocab=["S", "B"], trigger={("S", "B"): 1.0},
        topo=_FakeTopo({}, max_hops=1), node_infos={}, beta=1.0,
    )
    ad = FeatureKernelAdapter(fs, mu_by_alarm_type={"S": 0.1, "B": 0.1},
                              time_scale_sec=60.0, cache_max_entries=50)
    # Probe many distinct (src,tgt) pairs → far more than the cap.
    for i in range(500):
        ad._alpha(("S", f"n{i}"), ("B", f"m{i}"))
    assert len(ad._pair_alpha) <= 50, f"cache exceeded cap: {len(ad._pair_alpha)}"


def test_dynamic_feature_impute_uses_observed_source_mark():
    topo = _FakeTopo({"n1": {}}, max_hops=1)
    fs = _FakeDynamicFeatureScorer(
        at_vocab=["S", "B"], trigger={("S", "B"): 4.0},
        topo=topo, node_infos={"n1": _FakeNodeInfo("siteA")}, beta=1.0,
    )
    adapter = FeatureKernelAdapter(
        fs,
        mu_by_alarm_type={"S": 0.1, "B": 1e-6},
        time_scale_sec=60.0,
        alpha_floor=0.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=3, max_births_per_sweep=0, seed=31),
    )
    s.ingest(0.0, ("S", "n1"), {"event_id": "S"}, src_mark=(0, 1, 0))
    s.ingest(20.0, ("B", "n1"), {"event_id": "B"}, src_mark=(0, 0, 0))
    src = [e for e in s.events.values() if e.type_id == ("S", "n1")][0]
    tgt = [e for e in s.events.values() if e.type_id == ("B", "n1")][0]
    assert tgt.parent == src.eid


def test_dynamic_feature_missing_sources_use_timeline_mark():
    topo = _FakeTopo({"n1": {}}, max_hops=1)
    fs = _FakeDynamicFeatureScorer(
        at_vocab=["S", "B"], trigger={("S", "B"): 4.0},
        topo=topo, node_infos={"n1": _FakeNodeInfo("siteA")}, beta=1.0,
    )
    timeline = ObservedStateTimeline()
    timeline.ingest(0.0, "n1", "power", False)
    adapter = FeatureKernelAdapter(
        fs,
        mu_by_alarm_type={"S": 0.1, "B": 1e-6},
        time_scale_sec=60.0,
        alpha_floor=0.0,
        source_mark_at=timeline.source_mark_at,
    )
    assert adapter.kernel_intensity(("S", "n1"), ("B", "n1"), 10.0, source_mark=(0, 1, 0)) > 0
    assert adapter.kernel_intensity(("S", "n1"), ("B", "n1"), 10.0, source_mark=(0, 0, 0)) == 0
    assert ("S", "n1") in dict(adapter.candidate_sources(("B", "n1")))
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=10, missing_log_prior=0.0,
                      max_depth=1, seed=41),
    )
    s.ingest(100.0, ("B", "n1"), {"event_id": "B"}, src_mark=(0, 0, 0))
    missing = [e for e in s.events.values() if e.is_missing()]
    assert missing, "expected timeline-backed dynamic missing parent"
    assert any(e.type_id == ("S", "n1") and e.src_mark == (0, 1, 0) for e in missing)


def test_source_target_dynamic_feature_uses_target_mark():
    topo = _FakeTopo({"n1": {}}, max_hops=1)
    fs = _FakeSourceTargetFeatureScorer(
        at_vocab=["S", "B"], trigger={("S", "B"): 5.0},
        topo=topo, node_infos={"n1": _FakeNodeInfo("siteA")}, beta=1.0,
    )
    adapter = FeatureKernelAdapter(
        fs,
        mu_by_alarm_type={"S": 0.1, "B": 1e-6},
        time_scale_sec=60.0,
        alpha_floor=0.0,
    )
    assert adapter.source_mark_dim == 3
    assert adapter.n_dynamic == 6
    assert adapter.kernel_intensity(
        ("S", "n1"), ("B", "n1"), 10.0,
        source_mark=(0, 1, 0), target_mark=(1, 0, 0),
    ) > 0
    assert adapter.kernel_intensity(
        ("S", "n1"), ("B", "n1"), 10.0,
        source_mark=(0, 1, 0), target_mark=(0, 0, 0),
    ) == 0
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=3, max_births_per_sweep=0, seed=51),
    )
    s.ingest(0.0, ("S", "n1"), {"event_id": "S"}, src_mark=(0, 1, 0))
    s.ingest(20.0, ("B", "n1"), {"event_id": "B"}, src_mark=(1, 0, 0))
    src = [e for e in s.events.values() if e.type_id == ("S", "n1")][0]
    tgt = [e for e in s.events.values() if e.type_id == ("B", "n1")][0]
    assert tgt.parent == src.eid


def test_source_target_total_compensator_uses_target_timeline_mark():
    topo = _FakeTopo({"n1": {"n2": 1}, "n2": {"n1": 1}}, max_hops=1)
    fs = _FakeSourceTargetFeatureScorer(
        at_vocab=["S", "B"], trigger={("S", "B"): 2.0},
        topo=topo, node_infos={"n1": _FakeNodeInfo("siteA"), "n2": _FakeNodeInfo("siteA")},
        beta=1.0,
    )
    timeline = ObservedStateTimeline()
    timeline.ingest(0.0, "n2", "link", False)
    adapter = FeatureKernelAdapter(
        fs,
        mu_by_alarm_type={"S": 0.1, "B": 0.1},
        time_scale_sec=60.0,
        alpha_floor=0.0,
        target_mark_at=lambda ne, ts: timeline.state_at(ne, ts),
    )
    assert adapter.total_compensator(
        ("S", "n1"), 60.0, source_mark=(0, 1, 0), source_ts=10.0
    ) > 0
    assert adapter.total_compensator(
        ("S", "n1"), 60.0, source_mark=(0, 0, 0), source_ts=10.0
    ) == 0


def test_observed_state_timeline_prune_keeps_baseline_state():
    timeline = ObservedStateTimeline()
    timeline.ingest(0.0, "n1", "power", False)
    timeline.ingest(10.0, "n1", "link", False)
    timeline.ingest(20.0, "n1", "link", True)
    timeline.ingest(30.0, "n1", "offline", False)
    timeline.prune_before(25.0)
    assert len(timeline._times["n1"]) == 2
    assert timeline.state_at("n1", 26.0) == (0, 1, 0)
    assert timeline.state_at("n1", 35.0) == (0, 1, 1)


def test_visual_snapshots_link_to_previous_versions_without_closing():
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.1, 1: 0.001, 2: 0.001},
        edges={
            (1, 0): (3.0, 1.0),
            (2, 1): (3.0, 1.0),
        },
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=5, max_births_per_sweep=0, seed=43),
    )
    s.ingest(0.0, 0, {"event_id": "A"})
    s.ingest(10.0, 1, {"event_id": "B"})
    first = s.visual_snapshot_groups(5.0)
    assert len(first) == 1
    assert first[0]["group_id"].endswith(".snapshot-0001")
    assert first[0]["related_group_uuids"] == []
    assert s.events, "snapshot must not close/remove live sampler events"
    assert s.visual_snapshot_groups(5.0) == []

    s.ingest(20.0, 2, {"event_id": "C"})
    second = s.visual_snapshot_groups(5.0)
    assert len(second) == 1
    assert second[0]["related_group_uuids"] == [first[0]["group_id"]]

    final = s.flush()
    assert len(final) == 1
    assert final[0]["base_group_id"] == "mhp-online-000000"
    assert final[0]["related_group_uuids"] == [first[0]["group_id"], second[0]["group_id"]]
    assert s._visual_snapshot_history == {}


def test_visual_snapshot_can_use_external_stream_time_without_advancing_sampler():
    adapter = ExpKernelAdapter(
        mu_by_type={0: 0.1},
        edges={},
        time_scale_sec=60.0,
    )
    s = MissingChainSampler(
        adapter,
        SamplerConfig(lag_sec=1e9, sweeps_per_tick=0, max_births_per_sweep=0, seed=44),
    )
    s.ingest(0.0, 0, {"event_id": "A"})
    assert s.visual_snapshot_groups(5.0) == []
    snap = s.visual_snapshot_groups(5.0, now_ts=6.0)
    assert len(snap) == 1
    assert s.now == 0.0
    assert s.events, "visual-only snapshot must not close/remove sampler events"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
