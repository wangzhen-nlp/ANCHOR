#!/usr/bin/env python3
"""Offline compiler for the AlarmPeriod MHP sparse association cache.

The cache covers the ``(feature entity, alarm type)`` values present in the
training artifact and all eight frozen dynamic-state combinations.  It stores
only edges whose peak score reaches the configured immigrant threshold.
The online engine reconstructs the covered signature universe from the same
fingerprinted artifact vocabulary, so zero-edge signatures need no persistent
negative records.

Devices absent from the training artifact are intentionally not synthesized.
``stream_alarm_period_mhp.py`` compiles those signatures incrementally when
they first appear and keeps the additions in process memory only.
"""

from __future__ import annotations

import argparse
import os
import time

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_mhp.stream_alarm_period_mhp import (
    CompiledAssociationPlan,
    PeriodStreamConfig,
    _build_runtime_scorers,
    artifact_period_types,
    association_cache_fingerprint,
    write_association_cache,
)
from alarm_flow_mhp.topology_relation_prior import parse_topology_relation_prior
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compile an offline sparse association cache for AlarmPeriod MHP."
    )
    parser.add_argument("model", help="Trained feature-mode alarm-flow MHP artifact JSON.")
    parser.add_argument("output", help="Association cache output (.json or .json.gz).")
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=resource_display("ne_graph.json"))
    parser.add_argument(
        "--site-graph", default=SITE_GRAPH_JSON, help=resource_display("site_graph.json")
    )
    parser.add_argument("--history-window-sec", type=float, default=None)
    parser.add_argument("--time-slack-sec", type=float, default=None)
    parser.add_argument("--late-penalty-half-life-sec", type=float, default=None)
    parser.add_argument("--immigrant-bias", type=float, default=1.0)
    parser.add_argument("--feature-alpha-floor", type=float, default=None)
    parser.add_argument("--attach-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--candidate-scope", choices=("related", "global"), default="related")
    parser.add_argument(
        "--topology-relation-prior",
        default="",
        help="Comma-separated relation multipliers, same as stream_alarm_period_mhp.py.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Report after this many directed period-type pairs; 0 disables.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        relation_prior = parse_topology_relation_prior(args.topology_relation_prior)
    except ValueError as exc:
        parser.error(str(exc))

    t0 = time.monotonic()
    artifact = load_alarm_mhp_artifact(args.model)
    scorer, mu_scorer, _ne_graph_data = _build_runtime_scorers(
        artifact, args.ne_graph, args.site_graph, quiet=args.quiet
    )
    history = (
        float(args.history_window_sec)
        if args.history_window_sec is not None
        else float(artifact.config.history_window_sec)
    )
    slack = (
        float(args.time_slack_sec)
        if args.time_slack_sec is not None
        else float(getattr(artifact.config, "time_slack_sec", 0.0))
    )
    late_half_life = (
        float(args.late_penalty_half_life_sec)
        if args.late_penalty_half_life_sec is not None
        else float(getattr(artifact.config, "late_penalty_half_life_sec", 1.0))
    )
    floor = (
        float(args.feature_alpha_floor)
        if args.feature_alpha_floor is not None
        else float(getattr(artifact.config, "edge_threshold", 0.0))
    )
    config = PeriodStreamConfig(
        aggregation_wait_sec=max(30.0, slack),
        history_window_sec=history,
        time_slack_sec=slack,
        late_penalty_half_life_sec=late_half_life,
        time_scale_sec=float(artifact.config.time_scale_sec),
        immigrant_bias=args.immigrant_bias,
        feature_alpha_floor=floor,
        attach_threshold_ratio=args.attach_threshold_ratio,
        candidate_scope=args.candidate_scope,
        topology_relation_prior=relation_prior,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    period_types = artifact_period_types(artifact)
    if not period_types:
        parser.error("artifact contains no usable (entity, alarm_type) values")
    if not args.quiet:
        print(
            f"[period-cache] types={len(period_types)}, signatures={len(period_types) * 8}, "
            f"scope={config.candidate_scope}",
            flush=True,
        )

    plan = CompiledAssociationPlan(scorer, mu_scorer, artifact, config)
    last_report = 0

    def report(type_pairs, active_edges, pruned_pairs):
        nonlocal last_report
        if args.quiet or not args.progress_every:
            return
        if type_pairs - last_report < args.progress_every:
            return
        last_report = type_pairs
        elapsed = time.monotonic() - t0
        print(
            f"[period-cache] type_pairs={type_pairs}, active_edges={active_edges}, "
            f"pruned={pruned_pairs}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    type_pair_count = plan.precompile_period_types(period_types, progress=report)
    fingerprint = association_cache_fingerprint(
        args.model,
        args.ne_graph,
        args.site_graph,
        config,
        artifact.config.topology_node_field,
    )
    payload = plan.to_cache_payload(
        fingerprint,
        extra_metadata={
            "period_type_count": len(period_types),
            "directed_period_type_pair_count": type_pair_count,
        },
    )
    write_association_cache(args.output, payload)
    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(
            f"[period-cache] done: signatures={len(payload['signatures'])}, "
            f"edges={len(payload['edges'])}, pruned={plan.pruned_pair_count}, "
            f"elapsed={elapsed:.2f}s; output={os.path.abspath(args.output)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
