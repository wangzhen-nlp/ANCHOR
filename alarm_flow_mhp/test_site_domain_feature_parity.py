#!/usr/bin/env python3
"""Train→inference parity for the site×domain feature-mode refactor.

Validates that the φ vector a feature-mode model reconstructs at inference (via
RuntimeFeatureScorer over raw events) is byte-identical to the φ it trained on
(build_candidate_features), so the live α matches the trained kernel — the core
correctness guarantee of the (alarm_type, entity) identity with domain features.
Also checks device-mode φ is unchanged (no domain columns) and μ parity.
"""

import os
import sys
import unittest

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alarm_flow_isahp.event_domain import (
    annotate_device_domain,
    filter_and_annotate_device_domain,
)
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import event_type_label, build_alarm_vocabs
from alarm_flow_mhp.aggregator import AlarmMHPConfig, train_alarm_mhp
from alarm_flow_mhp.feature_spec import (
    FeatureLayout,
    RuntimeFeatureScorer,
    RuntimeMuScorer,
    build_candidate_features,
    build_mu_features,
    build_node_context,
    make_entity,
)
from mhp.feature_kernel import FeatureKernel
from alarm_flow_mhp.feature_spec import MuFeatureSpec

LINK = "Physical Port Down"      # -> link
POWER = "DC Low Voltage"         # -> power

# --- synthetic NE graph: 3 NEs across 2 sites + 2 domains, S1<->S2 link ---
NE_GRAPH = {
    "NE_A": {"site_id": "S1", "domain": "RAN", "manufacturer": "V1", "type": "T1",
             "link": {"NE_C": {"MW": "<->"}}},
    "NE_B": {"site_id": "S1", "domain": "TRANSMISSION", "manufacturer": "V2", "type": "T2",
             "link": {}},
    "NE_C": {"site_id": "S2", "domain": "TRANSMISSION", "manufacturer": "V1", "type": "T2",
             "link": {"NE_A": {"MW": "<->"}}},
}
# site graph (same structure, keyed by site)
SITE_GRAPH = {
    "S1": {"site_id": "S1", "link": {"S2": {"MW": "<->"}}},
    "S2": {"site_id": "S2", "link": {"S1": {"MW": "<->"}}},
}


def _events():
    """A repeating cascade NE_A(link)→NE_C(power)→NE_B(link) so pairs co-occur."""
    evs = []
    t = 0.0
    for _ in range(40):
        for src, title in (("NE_A", LINK), ("NE_C", POWER), ("NE_B", LINK)):
            t += 5.0
            evs.append({
                "alarm_source": src, "site_id": NE_GRAPH[src]["site_id"],
                "alarm_title": title, "ts": t, "alarm": {},
            })
        t += 100.0
    return evs


def _train(type_fields, node_field, topo_index, ne_graph, dynamic_alpha="off"):
    events = _events()
    annotate_device_domain(events, ne_graph)
    cfg = AlarmMHPConfig(
        type_fields=type_fields, topology_node_field=node_field,
        edge_mode="feature", history_window_sec=60.0, time_scale_sec=60.0,
        max_iters=5, min_events=2, beta_shared_value=1.0, feature_topo_max_hops=2,
        dynamic_alpha=dynamic_alpha,
    )
    art = train_alarm_mhp(events, cfg, topology_index=topo_index,
                          ne_graph_data=ne_graph, verbose=False)
    return art, events, cfg


def test_site_domain_phi_alpha_parity():
    topo = NETopologyIndex.from_graph(SITE_GRAPH, max_hops=2)
    art, events, cfg = _train(
        ("site_id", "device_domain", "alarm_type"), "site_id", topo, NE_GRAPH)
    rt = art.training_metadata["feature_runtime"]
    assert rt["domain_vocab"], "site mode must produce a non-empty domain vocab"

    # runtime_ne_at must reconstruct the SAME entity (node + domain folded in)
    # that training keyed on — directly from a raw event. This is the path
    # stream/sampler use, so it must match the label-derived entity.
    from alarm_flow_mhp.feature_spec import runtime_ne_at
    for it in events:
        site, dom, at = event_type_label(it, ("site_id", "device_domain", "alarm_type")).split(" | ")
        want = make_entity("" if site == "<empty>" else site, "" if dom == "<empty>" else dom)
        got_ent, got_at = runtime_ne_at(it, ("site_id", "device_domain", "alarm_type"), "site_id")
        assert got_ent == want, f"runtime_ne_at entity {got_ent!r} != {want!r}"
        assert got_at == ("" if at == "<empty>" else at)
    assert "\x1f" in runtime_ne_at(events[0], ("site_id", "device_domain", "alarm_type"), "site_id")[0]
    print("  runtime_ne_at entity reconstruction OK (domain folded in)")

    # Re-derive the exact training candidate φ (same inputs as training).
    from mhp.events import EventCollection
    vocabs, _ = build_alarm_vocabs(events, cfg.sequence_config())
    # Reconstruct EventCollection the way training does, via the public trainer
    # internals would be heavy; instead rebuild candidates from the same vocab +
    # a fresh EventCollection over the modeled events.
    # Build modeled event dims from labels:
    labels = vocabs.type_vocab.labels
    lab_to_id = {l: i for i, l in enumerate(labels)}
    tf = cfg.type_fields
    dims, times = [], []
    from alarm_flow_isahp.sequences import _iter_model_events
    for it in _iter_model_events(events, cfg.sequence_config()):
        lab = event_type_label(it, tf)
        if lab in lab_to_id:
            dims.append(lab_to_id[lab]); times.append(float(it["ts"]))
    ev = EventCollection(np.asarray(times), np.asarray(dims, dtype=np.int64),
                         M=len(labels), T=float(times[-1] + 1.0))
    node_ctx = build_node_context(NE_GRAPH, "site_id")
    cand_t, cand_s, phi, names, at_vocab, _at_id, _topo, domain_vocab = build_candidate_features(
        ev, vocabs, tf, topology_index=topo, graph_context=node_ctx,
        history_window=cfg.history_window_sec,
        max_history_events=cfg.max_history_events, chunk_size=10000,
        topo_max_hops=2, node_field="site_id",
    )
    assert domain_vocab == rt["domain_vocab"]

    # Trained kernel α on training φ.
    kernel = FeatureKernel.from_dict(art.training_metadata["feature_kernel"])
    alpha_train = kernel.alpha(phi)

    # Inference scorer rebuilt exactly as stream does.
    scorer = RuntimeFeatureScorer(
        kernel=kernel, at_vocab=rt["at_vocab"], graph_context=node_ctx,
        topology_index=topo, beta=float(rt["beta"]), n_dynamic=0,
        domain_vocab=rt["domain_vocab"], node_domains=rt["node_domains"],
    )

    def entity_at_of(tid):
        site, dom, at = str(labels[tid]).split(" | ")
        return make_entity("" if site == "<empty>" else site,
                           "" if dom == "<empty>" else dom), at

    max_err = 0.0
    for i in range(len(cand_t)):
        t_ent, t_at = entity_at_of(int(cand_t[i]))
        s_ent, s_at = entity_at_of(int(cand_s[i]))
        a_inf = float(scorer.alpha_for_target(t_at, t_ent, [s_at], [s_ent])[0])
        max_err = max(max_err, abs(a_inf - float(alpha_train[i])))
    assert max_err < 1e-9, f"train/inference α mismatch: max_err={max_err}"
    print(f"  site×domain φ/α parity OK over {len(cand_t)} candidates (max_err={max_err:.2e})")

    # μ parity for a few types
    mu_scorer = RuntimeMuScorer(
        mu_kernel=FeatureKernel.from_dict(rt["mu_kernel"]),
        mu_spec=MuFeatureSpec.from_dict(rt["mu_spec"]), graph_context=node_ctx)
    mu_phi, mu_spec = build_mu_features(vocabs, tf, node_ctx, node_field="site_id")
    mu_kernel = FeatureKernel.from_dict(rt["mu_kernel"])
    mu_train = mu_kernel.alpha(mu_phi)
    mu_err = 0.0
    for tid in range(len(labels)):
        ent, at = entity_at_of(tid)
        mu_err = max(mu_err, abs(float(mu_scorer.mu_for(at, ent)) - float(mu_train[tid])))
    assert mu_err < 1e-9, f"μ mismatch: {mu_err}"
    print(f"  site×domain μ parity OK (max_err={mu_err:.2e})")


def test_device_mode_layout_unchanged():
    """device mode must produce the legacy 13-feature φ (no domain columns)."""
    topo = NETopologyIndex.from_graph(NE_GRAPH, max_hops=2)
    art, events, cfg = _train(("alarm_source", "alarm_type"), "alarm_source", topo, NE_GRAPH)
    rt = art.training_metadata["feature_runtime"]
    assert rt["domain_vocab"] == [], "device mode must have an empty φ domain vocab"
    layout = FeatureLayout(rt["at_vocab"])  # no domain vocab
    assert layout.n_features == FeatureKernel.from_dict(
        art.training_metadata["feature_kernel"]).n_features
    print("  device-mode legacy layout OK (n_features unchanged, no domain cols)")


def test_domain_filter_keeps_only_modeled_domains():
    graph = {
        "ran-ne": {"domain": "RAN"},
        "tx-ne": {"domain": "TRANSMISSION"},
        "data-ne": {"domain": "DATA"},
        "other-ne": {"domain": "CORE"},
        "missing-ne": {"domain": ""},
    }
    events = [{"alarm_source": ne} for ne in graph]
    events.append({"alarm_source": "unknown-ne"})
    kept, stats = filter_and_annotate_device_domain(events, graph)
    assert [e["device_domain"] for e in kept] == ["RAN", "TRANSMISSION", "DATA"]
    assert stats["kept_event_count"] == 3
    assert stats["dropped_by_domain"] == {
        "OTHER": 1,
        "MISSING": 1,
        "UNKNOWN_DEVICE": 1,
    }

    node_ctx = build_node_context(
        {
            "ran-ne": {"site_id": "S1", "domain": "RAN"},
            "tx-ne": {"site_id": "S1", "domain": "TRANSMISSION"},
            "other-ne": {"site_id": "S1", "domain": "CORE"},
            "missing-ne": {"site_id": "S1", "domain": ""},
        },
        "site_id",
    )
    assert node_ctx.node_domains == {"S1": ["RAN", "TRANSMISSION"]}


def test_topology_node_field_is_inferred_compatibly():
    site_cfg = AlarmMHPConfig(
        type_fields=("site_id", "device_domain", "alarm_type")
    )
    assert site_cfg.topology_node_field == "site_id"

    device_cfg = AlarmMHPConfig()
    assert device_cfg.topology_node_field == "alarm_source"

    # Simulates loading an old artifact whose config predates this field.
    old_cfg = AlarmMHPConfig.from_dict(
        {"type_fields": ["site_id", "device_domain", "alarm_type"]}
    )
    assert old_cfg.topology_node_field == "site_id"


def test_site_domain_source_target_dynamic_mode():
    from alarm_flow_mhp.dynamic_state import DeviceStateTracker
    from alarm_flow_mhp.feature_spec import parse_label_entity_at

    type_fields = ("site_id", "device_domain", "alarm_type")
    entity, alarm_type = parse_label_entity_at(
        "S1 | TRANSMISSION | link", type_fields, "site_id"
    )
    assert entity == make_entity("S1", "TRANSMISSION")
    assert alarm_type == "link"

    # Multiple devices in one site+domain share an active-count state: clearing
    # one occurrence must not clear the domain while another remains active.
    tracker = DeviceStateTracker()
    assert tracker.snapshot_then_apply(entity, "power", False).tolist() == [0, 0, 0]
    assert tracker.snapshot_then_apply(entity, "power", False).tolist() == [0, 1, 0]
    tracker.snapshot_then_apply(entity, "power", True)
    assert tracker.state_of(entity).tolist() == [0, 1, 0]
    tracker.snapshot_then_apply(entity, "power", True)
    assert tracker.state_of(entity).tolist() == [0, 0, 0]

    topo = NETopologyIndex.from_graph(SITE_GRAPH, max_hops=2)
    artifact, _events_out, _cfg = _train(
        type_fields,
        "site_id",
        topo,
        NE_GRAPH,
        dynamic_alpha="source_target",
    )
    feature_names = artifact.training_metadata["feature_kernel"]["feature_names"]
    assert feature_names[-6:] == [
        "src_uncleared_link",
        "src_uncleared_power",
        "src_uncleared_offline",
        "tgt_uncleared_link",
        "tgt_uncleared_power",
        "tgt_uncleared_offline",
    ]


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    for name, test_func in sorted(globals().items()):
        if name.startswith("test_") and callable(test_func):
            suite.addTest(unittest.FunctionTestCase(test_func, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
