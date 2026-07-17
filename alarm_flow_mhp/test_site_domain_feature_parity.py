#!/usr/bin/env python3
"""Train→inference parity for the site×domain feature-mode refactor.

Validates that the φ vector a feature-mode model reconstructs at inference (via
RuntimeFeatureScorer over raw events) is byte-identical to the φ it trained on
(build_candidate_features), so the live α matches the trained kernel — the core
correctness guarantee of the (alarm_type, entity) identity with domain features.
Also checks device-mode merged-domain/site/geo columns and μ parity.
"""

import os
import sys
import unittest
from datetime import datetime

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
    GeoStats,
    RuntimeFeatureScorer,
    RuntimeMuScorer,
    SiteStats,
    build_candidate_features,
    build_mu_features,
    build_node_context,
    make_entity,
)
from mhp.feature_kernel import FeatureKernel
from mhp.em import (
    MHPConfig as CoreMHPConfig,
    _apply_clear_time_teacher,
    _run_estep_iteration,
    fit_mhp,
    fit_mhp_feature,
)
from mhp.events import EventCollection
from alarm_flow_mhp.feature_spec import MuFeatureSpec

LINK = "Physical Port Down"      # -> link
POWER = "DC Low Voltage"         # -> power

# --- synthetic NE graph: 3 NEs across 2 sites + 2 domains, S1<->S2 link ---
NE_GRAPH = {
    "NE_A": {"site_id": "S1", "domain": "RAN", "manufacturer": "V1", "type": "T1",
             "latitude": 30.0, "longitude": 120.0,
             "link": {"NE_C": {"MW": "<->"}}},
    "NE_B": {"site_id": "S1", "domain": "TRANSMISSION", "manufacturer": "V2", "type": "T2",
             "latitude": 30.0, "longitude": 120.0,
             "link": {}},
    "NE_C": {"site_id": "S2", "domain": "TRANSMISSION", "manufacturer": "V1", "type": "T2",
             "latitude": 30.09, "longitude": 120.0,
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


def test_geo_stats_proximity_and_missing():
    ctx = build_node_context(NE_GRAPH, "alarm_source")
    geo = GeoStats(ctx.node_infos, ctx.site_coords)
    same, same_missing = geo.pair_features("S1", "S1")
    haversine = geo._haversine_km
    calls = []

    def counted_haversine(*args):
        calls.append(args)
        return haversine(*args)

    geo._haversine_km = counted_haversine
    cross, cross_missing = geo.pair_features("S1", "S2")
    reverse, reverse_missing = geo.pair_features("S2", "S1")
    unknown, unknown_missing = geo.pair_features("S1", "UNKNOWN")
    same_without_coords = geo.pair_features("NO_COORD", "NO_COORD")
    both_sites_unknown = geo.pair_features("", "")
    assert same == 1.0 and same_missing == 0.0
    assert 0.45 < cross < 0.55, cross  # 0.09 latitude degrees is about 10 km
    assert cross_missing == 0.0
    assert (reverse, reverse_missing) == (cross, cross_missing)
    assert len(calls) == 1  # reverse site pair reuses the cached Haversine result
    assert unknown == 0.0 and unknown_missing == 1.0
    assert same_without_coords == (1.0, 0.0)
    assert both_sites_unknown == (0.0, 1.0)


def test_site_and_device_structural_features():
    ctx = build_node_context(NE_GRAPH, "alarm_source")
    topo = NETopologyIndex.from_graph(NE_GRAPH, max_hops=2)
    stats = SiteStats(ctx.node_infos, topo)
    assert stats.site_link_ratio("S1", "S2") == 1.0
    assert stats.site_link_density("S1", "S2") == 0.5
    assert stats.site_link_density("S1", "S1") == 0.0
    assert stats.site_size_balance("S1", "S2") == 0.5
    assert stats.site_size_balance("S1", "S1") == 1.0
    assert abs(stats.site_domain_cosine("S1", "S2") - 1.0 / np.sqrt(2.0)) < 1e-12
    assert stats.site_domain_cosine("S1", "S1") == 1.0
    assert stats.degree_feat("NE_A") == 1.0 / 5.0
    assert stats.degree_feat("NE_C") == 1.0 / 5.0
    assert stats.degree_feat("NE_B") == 0.0
    assert stats.device_link_ratio("NE_A", "NE_C") == 1.0
    assert stats.device_link_ratio("NE_A", "NE_B") == 0.0

    branched = {
        "A": {"site_id": "X", "link": {"B": {"L": "<->"}, "C": {"L": "<->"}}},
        "B": {"site_id": "Y", "link": {"A": {"L": "<->"}}},
        "C": {"site_id": "Z", "link": {"A": {"L": "<->"}}},
    }
    branched_ctx = build_node_context(branched, "alarm_source")
    branched_topo = NETopologyIndex.from_graph(branched, max_hops=2)
    branched_stats = SiteStats(branched_ctx.node_infos, branched_topo)
    assert abs(branched_stats.site_link_ratio("X", "Y") - 2.0 / 3.0) < 1e-12
    assert abs(branched_stats.device_link_ratio("A", "B") - 2.0 / 3.0) < 1e-12
    assert branched_stats.degree_feat("A") == 2.0 / 6.0
    assert branched_stats.degree_feat("B") == 1.0 / 5.0

    # A's edge to U (no site_id) must still count toward the device degree and
    # therefore the direct A-B edge's share: 2 / (degree(A)=2 + degree(B)=1).
    missing_site_neighbor = {
        "A": {"site_id": "X", "link": {"B": {"L": "<->"}, "U": {"L": "<->"}}},
        "B": {"site_id": "Y", "link": {"A": {"L": "<->"}}},
        "U": {"site_id": "", "link": {"A": {"L": "<->"}}},
    }
    missing_ctx = build_node_context(missing_site_neighbor, "alarm_source")
    missing_topo = NETopologyIndex.from_graph(missing_site_neighbor, max_hops=2)
    missing_stats = SiteStats(missing_ctx.node_infos, missing_topo)
    assert missing_stats.node_degrees["A"] == 2
    assert missing_stats.node_degrees["U"] == 1
    assert abs(missing_stats.device_link_ratio("A", "B") - 2.0 / 3.0) < 1e-12

    # A non-empty direction without arrows is still an undirected MHP link.
    # Device and site feature modes must derive their site-link summaries from
    # the same rule even though ne_link_learning itself ignores this direction.
    non_arrow = {
        "A": {"site_id": "X", "link": {"B": {"L": "connected"}}},
        "B": {"site_id": "Y", "link": {}},
    }
    non_arrow_topo = NETopologyIndex.from_graph(non_arrow, max_hops=2)
    device_ctx = build_node_context(non_arrow, "alarm_source")
    device_stats = SiteStats(device_ctx.node_infos, non_arrow_topo)
    site_ctx = build_node_context(non_arrow, "site_id")
    site_stats = SiteStats(
        site_ctx.device_node_infos,
        None,
        undirected_neighbors=site_ctx.device_undirected_neighbors,
    )
    assert device_stats.pair_links[("X", "Y")] == 1
    assert site_stats.pair_links == device_stats.pair_links
    assert (
        site_stats.site_link_ratio("X", "Y")
        == device_stats.site_link_ratio("X", "Y")
        == 1.0
    )
    assert (
        site_stats.site_link_density("X", "Y")
        == device_stats.site_link_density("X", "Y")
        == 1.0
    )


def test_mhp_topology_score_is_strictly_undirected():
    from alarm_flow_mhp.aggregator import _ne_pair_topo_score
    from alarm_flow_mhp.feature_spec import _topo_score
    from alarm_flow_mhp.topology_relation_prior import classify_topology_relation

    # Deliberately mix opposite arrow directions. MHP must ignore both and use
    # only the symmetric undirected shortest-hop distance.
    graph = {
        "A": {"site_id": "X", "link": {"B": {"L": "<-"}}},
        "B": {"site_id": "Y", "link": {"C": {"L": "->"}}},
        "C": {"site_id": "Z", "link": {}},
    }
    topo = NETopologyIndex.from_graph(graph, max_hops=2, undirected_only=True)
    assert topo.directed_hops == {}
    assert topo.direct_edges == set()
    cache = {}
    assert _topo_score("A", "B", topo, cache) == 1.0
    assert _topo_score("B", "A", topo, cache) == 1.0
    assert _topo_score("A", "C", topo, cache) == 0.5
    assert _topo_score("C", "A", topo, cache) == 0.5
    assert _ne_pair_topo_score("A", "C", topo) == 0.5
    assert _ne_pair_topo_score("C", "A", topo) == 0.5
    assert classify_topology_relation("A", "B", topo) == "direct"
    assert classify_topology_relation("B", "A", topo) == "direct"
    assert classify_topology_relation("A", "C", topo) == "indirect"
    assert classify_topology_relation("C", "A", topo) == "indirect"


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
    assert float(np.max(phi[:, names.index("site_link_ratio")])) == 1.0
    assert float(np.max(phi[:, names.index("site_link_density")])) == 0.5
    assert float(np.max(phi[:, names.index("site_size_balance")])) == 1.0
    assert float(np.max(phi[:, names.index("site_domain_cosine")])) == 1.0
    assert float(np.max(phi[:, names.index("tgt_undirected_degree")])) == 0.0
    assert float(np.max(phi[:, names.index("src_undirected_degree")])) == 0.0
    assert float(np.max(phi[:, names.index("device_link_ratio")])) == 0.0

    # Trained kernel α on training φ.
    kernel = FeatureKernel.from_dict(art.training_metadata["feature_kernel"])
    alpha_train = kernel.alpha(phi)

    # Inference scorer rebuilt exactly as stream does.
    scorer = RuntimeFeatureScorer(
        kernel=kernel, at_vocab=rt["at_vocab"], graph_context=node_ctx,
        topology_index=topo, beta=float(rt["beta"]), n_dynamic=0,
        domain_vocab=rt["domain_vocab"], node_domains=rt["node_domains"],
        node_field="site_id",
    )

    def entity_at_of(tid):
        site, dom, at = str(labels[tid]).split(" | ")
        return make_entity("" if site == "<empty>" else site,
                           "" if dom == "<empty>" else dom), at

    max_err = 0.0
    max_err_source = 0.0
    for i in range(len(cand_t)):
        t_ent, t_at = entity_at_of(int(cand_t[i]))
        s_ent, s_at = entity_at_of(int(cand_s[i]))
        a_inf = float(scorer.alpha_for_target(t_at, t_ent, [s_at], [s_ent])[0])
        a_source = float(scorer.alpha_for_source(s_at, s_ent, [t_at], [t_ent])[0])
        max_err = max(max_err, abs(a_inf - float(alpha_train[i])))
        max_err_source = max(max_err_source, abs(a_source - float(alpha_train[i])))
    assert max_err < 1e-9, f"train/inference α mismatch: max_err={max_err}"
    assert max_err_source < 1e-9, (
        f"train/source-oriented inference α mismatch: max_err={max_err_source}"
    )
    print(f"  site×domain φ/α parity OK over {len(cand_t)} candidates (max_err={max_err:.2e})")

    # μ parity for a few types
    mu_scorer = RuntimeMuScorer(
        mu_kernel=FeatureKernel.from_dict(rt["mu_kernel"]),
        mu_spec=MuFeatureSpec.from_dict(rt["mu_spec"]), graph_context=node_ctx)
    mu_phi, mu_spec = build_mu_features(vocabs, tf, node_ctx, node_field="site_id")
    assert mu_spec.numeric_feature_names == [
        "site_size", "site_link_load", "domain_share_in_site"
    ]
    j_mu_size = mu_spec.feature_names.index("site_size")
    j_mu_load = mu_spec.feature_names.index("site_link_load")
    j_mu_share = mu_spec.feature_names.index("domain_share_in_site")
    assert abs(float(np.max(mu_phi[:, j_mu_size])) - 2.0 / 10.0) < 1e-12
    assert abs(float(np.max(mu_phi[:, j_mu_load])) - 1.0 / 9.0) < 1e-12
    assert float(np.max(mu_phi[:, j_mu_share])) == 1.0
    mu_kernel = FeatureKernel.from_dict(rt["mu_kernel"])
    mu_train = mu_kernel.alpha(mu_phi)
    mu_err = 0.0
    for tid in range(len(labels)):
        ent, at = entity_at_of(tid)
        mu_err = max(mu_err, abs(float(mu_scorer.mu_for(at, ent)) - float(mu_train[tid])))
    assert mu_err < 1e-9, f"μ mismatch: {mu_err}"
    print(f"  site×domain μ parity OK (max_err={mu_err:.2e})")


def test_device_mode_carries_merged_domain_block():
    """device mode φ now carries the merged 4-bucket domain block by default:
    the vocab holds phi_node_domain buckets present in the graph, and the
    layout (with the same_domain + dom-pair columns) matches the kernel."""
    topo = NETopologyIndex.from_graph(NE_GRAPH, max_hops=2)
    art, events, cfg = _train(("alarm_source", "alarm_type"), "alarm_source", topo, NE_GRAPH)
    rt = art.training_metadata["feature_runtime"]
    # NE_GRAPH has RAN + TRANSMISSION devices only → merged vocab is exactly those.
    assert rt["domain_vocab"] == ["RAN", "TRANSMISSION"], rt["domain_vocab"]
    layout = FeatureLayout(rt["at_vocab"], rt["domain_vocab"])
    assert "same_domain" in layout.feature_names
    assert layout.n_features == FeatureKernel.from_dict(
        art.training_metadata["feature_kernel"]).n_features
    print("  device-mode merged domain block OK (vocab + layout size)")


def test_device_node_domain_phi_alpha_parity():
    """Device-mode φ gains same_domain + dom-pair columns from the NE graph's
    domain_bucket (OTHER/MISSING merged into OTHER) by default, the entity stays
    the bare NE, and both runtime scorers reconstruct the training α exactly."""
    from alarm_flow_isahp.sequences import _iter_model_events
    from alarm_flow_mhp.aggregator import train_alarm_mhp
    from alarm_flow_mhp.feature_spec import DecomposedFeatureScorer, phi_domain_of
    from mhp.feature_kernel import softplus

    # NE_D's raw domain CORE buckets to OTHER → the merged vocab must carry OTHER.
    graph = dict(NE_GRAPH)
    graph["NE_D"] = {"site_id": "S2", "domain": "CORE", "manufacturer": "V2",
                     "type": "T3", "link": {}}
    topo = NETopologyIndex.from_graph(graph, max_hops=2)
    events = []
    t = 0.0
    for _ in range(40):
        for src, title in (("NE_A", LINK), ("NE_C", POWER), ("NE_B", LINK), ("NE_D", LINK)):
            t += 5.0
            events.append({
                "alarm_source": src, "site_id": graph[src]["site_id"],
                "alarm_title": title, "ts": t, "alarm": {},
            })
        t += 100.0
    annotate_device_domain(events, graph)
    tf = ("alarm_source", "alarm_type")
    cfg = AlarmMHPConfig(
        type_fields=tf, topology_node_field="alarm_source",
        edge_mode="feature", history_window_sec=60.0, time_scale_sec=60.0,
        max_iters=5, min_events=2, beta_shared_value=1.0, feature_topo_max_hops=2,
    )
    art = train_alarm_mhp(events, cfg, topology_index=topo, ne_graph_data=graph, verbose=False)
    rt = art.training_metadata["feature_runtime"]
    assert rt["domain_vocab"] == ["OTHER", "RAN", "TRANSMISSION"], rt["domain_vocab"]

    kernel = FeatureKernel.from_dict(art.training_metadata["feature_kernel"])
    layout = FeatureLayout(rt["at_vocab"], rt["domain_vocab"])
    assert "same_domain" in layout.feature_names and "dom[0->0]" in layout.feature_names
    for col in (
        "tgt_site_size", "src_site_size", "site_link_score",
        "site_link_ratio", "site_link_density", "site_size_balance",
        "site_domain_cosine", "tgt_undirected_degree", "src_undirected_degree",
        "device_link_ratio",
        "geo_proximity", "geo_distance_missing",
    ):
        assert col in layout.feature_names, col
    assert layout.n_features == kernel.n_features

    # Re-derive the exact training candidate φ and compare per-candidate α.
    vocabs, _ = build_alarm_vocabs(events, cfg.sequence_config())
    labels = vocabs.type_vocab.labels
    lab_to_id = {l: i for i, l in enumerate(labels)}
    dims, times = [], []
    for it in _iter_model_events(events, cfg.sequence_config()):
        lab = event_type_label(it, tf)
        if lab in lab_to_id:
            dims.append(lab_to_id[lab]); times.append(float(it["ts"]))
    ev = EventCollection(np.asarray(times), np.asarray(dims, dtype=np.int64),
                         M=len(labels), T=float(times[-1] + 1.0))
    node_ctx = build_node_context(graph, "alarm_source")
    cand_t, cand_s, phi, names, at_vocab, _at_id, _topo_vec, domain_vocab = build_candidate_features(
        ev, vocabs, tf, topology_index=topo, graph_context=node_ctx,
        history_window=cfg.history_window_sec,
        max_history_events=cfg.max_history_events, chunk_size=10000,
        topo_max_hops=2, node_field="alarm_source",
    )
    assert domain_vocab == rt["domain_vocab"]
    alpha_train = kernel.alpha(phi)

    scorer = RuntimeFeatureScorer(
        kernel=kernel, at_vocab=rt["at_vocab"], graph_context=node_ctx,
        topology_index=topo, beta=float(rt["beta"]), n_dynamic=0,
        domain_vocab=rt["domain_vocab"], node_domains=rt["node_domains"],
    )
    decomposed = DecomposedFeatureScorer(scorer)

    def entity_at_of(tid):
        ne, at = str(labels[tid]).split(" | ")
        return ("" if ne == "<empty>" else ne), ("" if at == "<empty>" else at)

    max_err = 0.0
    max_err_source = 0.0
    max_err_decomposed = 0.0
    for i in range(len(cand_t)):
        t_ent, t_at = entity_at_of(int(cand_t[i]))
        s_ent, s_at = entity_at_of(int(cand_s[i]))
        a_inf = float(scorer.alpha_for_target(t_at, t_ent, [s_at], [s_ent])[0])
        a_source = float(scorer.alpha_for_source(s_at, s_ent, [t_at], [t_ent])[0])
        z = decomposed.logits_for_target(t_at, t_ent, [s_at], [s_ent])
        a_decomposed = float(softplus(z)[0]) * decomposed.alpha_scale
        max_err = max(max_err, abs(a_inf - float(alpha_train[i])))
        max_err_source = max(max_err_source, abs(a_source - float(alpha_train[i])))
        max_err_decomposed = max(
            max_err_decomposed, abs(a_decomposed - float(alpha_train[i]))
        )
    assert max_err < 1e-9, f"train/inference α mismatch: max_err={max_err}"
    assert max_err_source < 1e-9, (
        f"train/source-oriented inference α mismatch: max_err={max_err_source}"
    )
    assert max_err_decomposed < 1e-9, (
        f"train/decomposed α mismatch: max_err={max_err_decomposed}"
    )

    # Unknown device at inference → merged bucket OTHER → the OTHER column.
    assert phi_domain_of("NE_UNKNOWN", scorer.node_infos) == "OTHER"
    assert scorer.layout.dom_id(phi_domain_of("NE_UNKNOWN", scorer.node_infos)) == \
        rt["domain_vocab"].index("OTHER")

    # Site columns carry real values in the training φ: NE_A(S1)↔NE_C(S2) is
    # one inter-site NE link → link feat 1/(1+4); S1 hosts 2 NEs → 2/(2+8).
    names = list(layout.feature_names)
    j_link = names.index("site_link_score")
    j_tsize = names.index("tgt_site_size")
    j_site_ratio = names.index("site_link_ratio")
    j_site_density = names.index("site_link_density")
    j_site_balance = names.index("site_size_balance")
    j_site_domain = names.index("site_domain_cosine")
    j_tgt_degree = names.index("tgt_undirected_degree")
    j_src_degree = names.index("src_undirected_degree")
    j_device_ratio = names.index("device_link_ratio")
    j_geo = names.index("geo_proximity")
    j_geo_missing = names.index("geo_distance_missing")
    assert abs(float(np.max(phi[:, j_link])) - 1.0 / 5.0) < 1e-6
    assert abs(float(np.max(phi[:, j_tsize])) - 2.0 / 10.0) < 1e-6
    assert abs(float(np.max(phi[:, j_site_ratio])) - 1.0) < 1e-6
    assert abs(float(np.max(phi[:, j_site_density])) - 1.0 / 4.0) < 1e-6
    assert abs(float(np.max(phi[:, j_site_balance])) - 1.0) < 1e-6
    assert abs(float(np.max(phi[:, j_site_domain])) - 1.0) < 1e-6
    assert abs(float(np.max(phi[:, j_tgt_degree])) - 1.0 / 5.0) < 1e-6
    assert abs(float(np.max(phi[:, j_src_degree])) - 1.0 / 5.0) < 1e-6
    assert abs(float(np.max(phi[:, j_device_ratio])) - 1.0) < 1e-6
    assert abs(float(np.max(phi[:, j_geo])) - 1.0) < 1e-6
    assert float(np.max(phi[:, j_geo_missing])) == 0.0

    # μ parity under the merged 4-bucket ψ domain (CORE/unknown devices → OTHER).
    mu_scorer = RuntimeMuScorer(
        mu_kernel=FeatureKernel.from_dict(rt["mu_kernel"]),
        mu_spec=MuFeatureSpec.from_dict(rt["mu_spec"]), graph_context=node_ctx)
    mu_phi, mu_spec = build_mu_features(vocabs, tf, node_ctx, node_field="alarm_source")
    assert "OTHER" in mu_spec.domain_vocab, mu_spec.domain_vocab
    assert "MISSING" not in mu_spec.domain_vocab
    assert mu_spec.numeric_feature_names == ["site_size", "undirected_degree"]
    j_mu_size = mu_spec.feature_names.index("site_size")
    j_mu_degree = mu_spec.feature_names.index("undirected_degree")
    assert abs(float(np.max(mu_phi[:, j_mu_size])) - 2.0 / 10.0) < 1e-12
    assert abs(float(np.max(mu_phi[:, j_mu_degree])) - 1.0 / 5.0) < 1e-12
    mu_train = FeatureKernel.from_dict(rt["mu_kernel"]).alpha(mu_phi)
    mu_err = 0.0
    for tid in range(len(labels)):
        ent, at = entity_at_of(tid)
        mu_err = max(mu_err, abs(float(mu_scorer.mu_for(at, ent)) - float(mu_train[tid])))
    assert mu_err < 1e-9, f"μ mismatch: {mu_err}"
    print(f"  device node-domain φ/α parity OK over {len(cand_t)} candidates "
          f"(max_err={max_err:.2e}, decomposed={max_err_decomposed:.2e}); "
          f"μ merged-domain "
          f"parity OK (max_err={mu_err:.2e})")


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


def _clear_teacher_fixture():
    events = EventCollection(
        times=np.array([0.0, 0.0, 1.0]),
        dims=np.array([0, 1, 2]),
        M=3,
        T=2.0,
    )
    config = CoreMHPConfig(
        history_window=5.0,
        max_history_events=4,
        beta_mode="shared",
        beta_shared_value=1.0,
        chunk_size=8,
        estep_device="cpu",
        verbose=False,
    )
    alpha = np.zeros((3, 3), dtype=np.float32)
    alpha[2, 0] = 0.5
    alpha[2, 1] = 0.5
    mu = np.array([0.2, 0.2, 0.2])
    # Target clear matches source 0; source 1 is far outside tau.
    clear_times = np.array([100.0, 1000.0, 100.0])
    return events, config, alpha, mu, clear_times


def test_clear_teacher_posterior_modes():
    p_self = np.array([0.4], dtype=np.float64)
    p_parent = np.array([0.3, 0.3], dtype=np.float64)
    target_local = np.array([0, 0], dtype=np.int64)
    target = np.array([2, 2], dtype=np.int64)
    source = np.array([0, 1], dtype=np.int64)
    clear_times = np.array([100.0, 1000.0, 100.0])

    red_self, red_parent = _apply_clear_time_teacher(
        p_self, p_parent, target_local, target, source, clear_times,
        boost=1.0, tau=10.0, mode="redistribute",
    )
    np.testing.assert_allclose(red_self, p_self)
    assert np.isclose(red_parent.sum(), p_parent.sum())
    assert red_parent[0] > red_parent[1]

    full_self, full_parent = _apply_clear_time_teacher(
        p_self, p_parent, target_local, target, source, clear_times,
        boost=1.0, tau=10.0, mode="full",
    )
    assert full_self[0] < p_self[0]
    assert full_parent[0] > full_parent[1]
    assert np.isclose(full_self[0] + full_parent.sum(), 1.0)

    missing = np.full(3, np.nan)
    missing_self, missing_parent = _apply_clear_time_teacher(
        p_self, p_parent, target_local, target, source, missing,
        boost=5.0, tau=10.0, mode="full",
    )
    assert missing_self is p_self
    assert missing_parent is p_parent


def test_clear_teacher_device_estep_keeps_true_ll():
    events, config, alpha, mu, clear_times = _clear_teacher_fixture()
    beta = np.float32(1.0)
    base = _run_estep_iteration(events, alpha, beta, mu, config)
    taught = _run_estep_iteration(
        events,
        alpha,
        beta,
        mu,
        config,
        clear_times=clear_times,
        clear_time_teacher_boost=1.0,
        clear_time_teacher_tau=10.0,
        clear_time_teacher_mode="redistribute",
    )

    base_p_self, base_alpha_num, _, _, base_ll_term = base
    taught_p_self, taught_alpha_num, _, _, taught_ll_term = taught
    np.testing.assert_allclose(taught_p_self, base_p_self)
    assert np.isclose(taught_ll_term, base_ll_term)
    assert np.isclose(base_alpha_num[2, 0], base_alpha_num[2, 1])
    assert taught_alpha_num[2, 0] > taught_alpha_num[2, 1]


def test_clear_teacher_feature_mstep_learns_preferred_feature():
    events, _, _, _, clear_times = _clear_teacher_fixture()
    phi = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    callback_iters = []
    result = fit_mhp_feature(
        events,
        CoreMHPConfig(
            history_window=5.0,
            max_history_events=4,
            max_iters=2,
            tol=0.0,
            beta_mode="shared",
            beta_shared_value=1.0,
            edge_threshold=0.0,
            chunk_size=8,
            estep_device="cpu",
            verbose=False,
        ),
        cand_targets=np.array([2, 2]),
        cand_sources=np.array([0, 1]),
        cand_phi=phi,
        feature_names=["bias", "preferred", "other"],
        l2=1e-3,
        clear_times=clear_times,
        clear_time_teacher_boost=1.0,
        clear_time_teacher_tau=10.0,
        clear_time_teacher_mode="redistribute",
        best_callback=lambda _result, entry: callback_iters.append(entry["iter"]),
        snapshot_callback_every_iter=True,
    )
    edge_alpha = {
        (int(t), int(s)): float(a)
        for t, s, a in zip(
            result.params.edge_targets,
            result.params.edge_sources,
            result.params.edge_alpha,
        )
    }
    assert edge_alpha[(2, 0)] > edge_alpha[(2, 1)]
    assert callback_iters == list(range(result.iterations_run))
    assert all(entry["convergence_metric"] == "parameters" for entry in result.trace)


def test_clear_teacher_nonmonotone_train_ll_still_snapshots_every_iteration():
    rng = np.random.default_rng(0)
    n, m = 120, 5
    times = np.cumsum(rng.exponential(0.3, n))
    dims = rng.integers(0, m, n)
    events = EventCollection(times=times, dims=dims, M=m, T=float(times[-1] + 1.0))
    clear_times = (
        np.floor((times + rng.normal(0.0, 8.0, n)) / 2.0) * 2.0
        + rng.integers(0, 3, n) * 30.0
    )
    callback_iters = []
    result = fit_mhp(
        events,
        CoreMHPConfig(
            history_window=4.0,
            max_history_events=20,
            max_iters=12,
            tol=0.0,
            alpha_prior_strength=1.0,
            beta_shared_value=1.0,
            chunk_size=50,
            edge_threshold=0.0,
            max_active_sources_per_dim=5,
            branching_cap=0.95,
            verbose=False,
        ),
        clear_times=clear_times,
        clear_time_teacher_boost=100.0,
        clear_time_teacher_tau=0.5,
        clear_time_teacher_mode="redistribute",
        best_callback=lambda _result, entry: callback_iters.append(entry["iter"]),
        snapshot_callback_every_iter=True,
    )
    train_ll = np.asarray([entry["log_likelihood"] for entry in result.trace])
    assert np.any(np.diff(train_ll) < 0.0)  # teacher breaks raw-LL monotonicity
    assert callback_iters == list(range(result.iterations_run))
    assert all(entry["convergence_metric"] == "parameters" for entry in result.trace)
    assert all(entry["parameter_delta_rel"] is not None for entry in result.trace)


def test_clear_teacher_alarm_adapter_aligns_and_records_metadata():
    events = _events()
    for start in range(0, len(events), 3):
        group = events[start:start + 3]
        clear_ts = max(event["ts"] for event in group) + 30.0
        clear_text = datetime.fromtimestamp(clear_ts).strftime("%Y-%m-%d %H:%M:%S")
        for event in group:
            event["alarm"]["告警清除时间"] = clear_text

    artifact = train_alarm_mhp(
        events,
        AlarmMHPConfig(
            edge_mode="device",
            history_window_sec=60.0,
            time_scale_sec=60.0,
            max_iters=2,
            min_events=2,
            clear_time_teacher_boost=0.5,
            clear_time_teacher_tau_sec=60.0,
            clear_time_teacher_mode="redistribute",
        ),
        verbose=False,
    )
    teacher = artifact.training_metadata["clear_time_teacher"]
    assert teacher["enabled"] is True
    assert teacher["train_valid_count"] == len(events)
    assert teacher["mode"] == "redistribute"


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    for name, test_func in sorted(globals().items()):
        if name.startswith("test_") and callable(test_func):
            suite.addTest(unittest.FunctionTestCase(test_func, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
