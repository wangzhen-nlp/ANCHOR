#!/usr/bin/env python3
"""Train alarm-flow MHP (MAP EM) edge influence parameters."""

from __future__ import annotations

from dataclasses import replace
import os
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import (
    AlarmMHPConfig,
    save_alarm_mhp_artifact,
    train_alarm_mhp,
)
from alarm_flow_brunch.region_filter import (
    filter_ne_graph_by_regions,
    load_ne_graph,
    parse_regions,
)
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import parse_type_fields
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, resource_display


def _progress_enabled(args):
    return not args.quiet


def _print_progress(message, args):
    if _progress_enabled(args):
        print(message, flush=True)


def _training_progress(stage, payload):
    if stage == "region_filter":
        if payload.get("enabled"):
            print(
                "[train] region filter: "
                f"regions={payload.get('regions', [])}, "
                f"events={payload.get('kept_event_count', 0)}/"
                f"{payload.get('input_event_count', 0)}, "
                f"allowed_devices={payload.get('allowed_device_count', 0)}",
                flush=True,
            )
        else:
            print(
                f"[train] region filter: disabled; events={payload.get('input_event_count', 0)}",
                flush=True,
            )
        return
    if stage == "vocab":
        print(
            "[train] vocab: "
            f"events={payload.get('considered_event_count', 0)}, "
            f"types={payload.get('type_count', 0)}",
            flush=True,
        )
        return
    if stage == "sequence":
        print(
            "[train] sequence: "
            f"modeled_events={payload.get('sequence_event_position_count', 0)}",
            flush=True,
        )
        return
    if stage == "fit_start":
        print(
            "[train] MHP fit: "
            f"train_events={payload.get('train_event_count', 0)}, "
            f"val_events={payload.get('val_event_count', 0)}, "
            f"types={payload.get('type_count', 0)}, "
            f"max_iters={payload.get('max_iters', 0)}",
            flush=True,
        )
        return
    if stage == "fit_done":
        val = payload.get("val_log_likelihood")
        val_str = f", val_ll={val:.4f}" if val is not None else ""
        print(
            "[train] fit done: "
            f"iterations={payload.get('iterations_run', 0)}, "
            f"converged={payload.get('converged', False)}, "
            f"active_edges={payload.get('active_edges', 0)}, "
            f"ll={payload.get('log_likelihood', 0.0):.4f}{val_str}",
            flush=True,
        )


def _parse_bucket_edges(text):
    """Parse comma-separated bucket right-edges (real seconds). Empty → default."""
    from alarm_flow_mhp.aggregator import DEFAULT_BUCKET_EDGES_SEC

    parts = [p.strip() for p in str(text or "").split(",") if p.strip()]
    if not parts:
        return DEFAULT_BUCKET_EDGES_SEC
    return tuple(float(p) for p in parts)


def _build_config(args):
    return AlarmMHPConfig(
        type_fields=parse_type_fields(args.type_fields),
        history_window_sec=args.history_window_sec,
        time_slack_sec=args.time_slack_sec,
        late_penalty_half_life_sec=args.late_penalty_half_life_sec,
        max_history_events=args.max_history_events,
        min_events=args.min_events,
        time_scale_sec=args.time_scale_sec,
        include_clear=args.include_clear,
        max_iters=args.max_iters,
        tol=args.tol,
        alpha_prior_strength=args.alpha_prior_strength,
        alpha_prior_mean=args.alpha_prior_mean,
        topology_prior_boost=args.topology_prior_boost,
        topology_prior_max_hops=args.topology_prior_max_hops,
        topology_prior_min_score=args.topology_prior_min_score,
        edge_mode=args.edge_mode,
        feature_l2=args.feature_l2,
        feature_l2_normalize=args.feature_l2_normalize,
        feature_topo_max_hops=args.feature_topo_max_hops,
        feature_topo_min_score=args.feature_topo_min_score,
        feature_topo_prior_boost=args.feature_topo_prior_boost,
        dynamic_alpha=args.dynamic_alpha,
        mu_count_smoothing=args.mu_count_smoothing,
        beta_mode=args.beta_mode,
        beta_shared_value=args.beta_shared_value,
        beta_prior_strength=args.beta_prior_strength,
        beta_prior_mean=args.beta_prior_mean,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        edge_threshold=args.edge_threshold,
        max_active_sources_per_dim=args.max_active_sources_per_dim,
        branching_cap=args.branching_cap,
        stability_radius=args.stability_radius,
        chunk_size=args.chunk_size,
        kernel_type=args.kernel_type,
        bucket_edges_sec=_parse_bucket_edges(args.bucket_edges_sec),
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
        selection_metric=args.selection_metric,
        regions=parse_regions(args.regions),
        min_group_events=args.min_group_events,
        seed=args.seed,
    )


def _adopt_loaded_regions(config, alarm_metadata):
    region_filter = (alarm_metadata or {}).get("region_filter") or {}
    if config.regions or not region_filter.get("enabled"):
        return config
    regions = parse_regions(region_filter.get("regions"))
    return replace(config, regions=regions) if regions else config


def _default_best_output(path: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() == ".json":
        return f"{root}.best{ext}"
    return f"{path}.best.json"


def _json_safe(v):
    """Coerce a CLI arg value to a JSON-serializable form for the run_args snapshot."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return str(v)


def _run_args_snapshot(args) -> dict:
    """JSON-safe snapshot of the full CLI args for reproducibility. ``alarms`` is
    resolved to an absolute path so the input is recoverable from any artifact
    (final or best checkpoint)."""
    snap = {k: _json_safe(v) for k, v in vars(args).items()}
    if snap.get("alarms"):
        snap["alarms"] = os.path.abspath(args.alarms)
    return snap


def main():
    parser = ArgumentParser(description="Train alarm-flow MHP via MAP EM.")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache.")
    parser.add_argument("-o", "--output", required=True, help="Output MHP model artifact JSON.")
    parser.add_argument(
        "--best-output",
        default=None,
        help=(
            "Write a best-so-far checkpoint whenever train LL improves. "
            "Default: output path with .best before .json."
        ),
    )
    parser.add_argument(
        "--no-best-checkpoint",
        action="store_true",
        help="Disable best-so-far checkpoint writing during training.",
    )
    parser.add_argument(
        "--topo",
        default=SITE_GRAPH_BY_NE_JSON,
        help=f"Site topology for raw alarm inputs. Default: {resource_display('site_graph_by_ne.json')}.",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"NE graph for raw alarm inputs. Default: {resource_display('ne_graph.json')}.",
    )
    parser.add_argument("--start-time", default="", help="Lower bound on first occurrence.")
    parser.add_argument("--end-time", default="", help="Upper bound on first occurrence.")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--type-fields",
        default="alarm_source,alarm_type",
        help="Comma-separated alarm fields defining the type.",
    )
    parser.add_argument("--history-window-sec", type=float, default=900.0)
    parser.add_argument(
        "--time-slack-sec",
        type=float,
        default=0.0,
        help=(
            "Training timestamp-jitter tolerance. Candidate parents may be up to "
            "this many seconds later than the target, with a late-parent penalty. "
            "0 keeps the original strictly-past parent set. Default: 0."
        ),
    )
    parser.add_argument(
        "--late-penalty-half-life-sec",
        type=float,
        default=1.0,
        help=(
            "Half-life for discounting parents whose timestamp is later than the "
            "target within --time-slack-sec. Smaller = harsher penalty. Default: 1."
        ),
    )
    parser.add_argument("--max-history-events", type=int, default=128)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--time-scale-sec", type=float, default=60.0)
    parser.add_argument("--include-clear", action="store_true")
    # EM:
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Maximum number of MAP EM iterations. Default: 30.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Relative LL change for convergence. Default: 1e-4.",
    )
    parser.add_argument(
        "--edge-mode",
        choices=("device", "feature"),
        default="device",
        help=(
            "Edge/amplitude model. 'device' (default) learns a free α per "
            "(device-type) pair — transductive, no new-device generalization. "
            "'feature' learns α = softplus(w·φ) over device-agnostic pair "
            "features (alarm-type pair, topology relation, same-site, vendor, "
            "...) — inductive: generalizes to unseen pairs. Needs the NE graph."
        ),
    )
    parser.add_argument(
        "--feature-l2",
        type=float,
        default=1e-3,
        help="Ridge penalty on feature weights (feature mode). Default: 1e-3.",
    )
    parser.add_argument(
        "--feature-l2-normalize",
        action="store_true",
        help="Scale the α ridge by event/data mass N (not raw exposure ΣE) so "
             "--feature-l2 is data-size-independent and actually controls ρ "
             "(λ≈0.01-0.1 bites). OFF by default = legacy raw ridge "
             "(λ negligible at large scale).",
    )
    parser.add_argument(
        "--feature-topo-max-hops",
        type=int,
        default=2,
        help="Topology reach for feature-mode candidate pair generation. Default: 2.",
    )
    parser.add_argument(
        "--feature-topo-min-score",
        type=float,
        default=0.0,
        help="Topology score floor for feature-mode candidates. Default: 0 (keep all reachable).",
    )
    parser.add_argument(
        "--feature-topo-prior-boost",
        type=float,
        default=0.0,
        help=(
            "Feature-mode topology PRIOR (device-parity). Injects a pseudo-count "
            "prior α≈boost·score on topology-related candidate edges: strong where "
            "data is sparse (rare/zero-co-occurrence but physically connected pairs "
            "still form an edge), washed out where data is rich. 0 disables (pure "
            "MLE). Try 0.3. Needs the NE graph (feature mode loads it anyway)."
        ),
    )
    parser.add_argument(
        "--dynamic-alpha",
        choices=("off", "source", "source_target"),
        default="off",
        help=(
            "Feature-mode DYNAMIC (stateful) α: condition excitation on devices' "
            "current uncleared link/power/offline alarms. 'off' (default) static "
            "only; 'source' adds the source device's 3 state booleans (exact, "
            "penalized); 'source_target' adds source + target 3-state booleans "
            "using the B-fast training approximation (target pre-state in E-step, "
            "target state at source_ts in exposure). Needs clears in the input stream."
        ),
    )
    parser.add_argument("--alpha-prior-strength", type=float, default=10.0)
    parser.add_argument("--alpha-prior-mean", type=float, default=0.1)
    parser.add_argument(
        "--topology-prior-boost",
        type=float,
        default=0.0,
        help=(
            "Inject extra MAP prior mass on topologically-related (target, source) "
            "type pairs, so rare or zero-co-occurrence but physically connected "
            "device pairs still form an edge. 0 disables (pure data-driven). "
            "Try 0.3. Requires the NE graph (loaded by default). The prior is "
            "auto-weighted toward rare sources, where data-driven edges are missing."
        ),
    )
    parser.add_argument(
        "--topology-prior-max-hops",
        type=int,
        default=1,
        help="NE-graph hops for the topology prior. 1 = same-NE + direct links only. Default: 1.",
    )
    parser.add_argument(
        "--topology-prior-min-score",
        type=float,
        default=0.6,
        help="Drop topology relations weaker than this (0-1). Default: 0.6 (keeps same-NE/direct).",
    )
    parser.add_argument(
        "--mu-count-smoothing",
        choices=("linear", "log"),
        default="log",
        help="μ_d ∝ count_d (linear) or log(1+count_d) (log). Default: log.",
    )
    parser.add_argument(
        "--beta-mode",
        choices=("shared", "per_edge"),
        default="shared",
        help="Kernel decay rate β: shared scalar or per-edge value. Default: shared.",
    )
    parser.add_argument("--beta-shared-value", type=float, default=1.0)
    parser.add_argument("--beta-prior-strength", type=float, default=5.0)
    parser.add_argument("--beta-prior-mean", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=1e-2)
    parser.add_argument("--beta-max", type=float, default=50.0)
    parser.add_argument(
        "--kernel-type",
        choices=("exp", "piecewise"),
        default="exp",
        help=(
            "Excitation kernel shape. 'exp' = single exponential α·β·exp(-β·dt) "
            "(default, current behavior). 'piecewise' = two-stage: exp fit selects "
            "edges, then box-basis EM learns per-edge per-time-bucket weights θ. "
            "Piecewise is interpretable ('A triggers B in the 3-10min bucket') and "
            "handles delayed-peak cascades that a single exponential cannot."
        ),
    )
    parser.add_argument(
        "--bucket-edges-sec",
        default="",
        help=(
            "Comma-separated right edges (real seconds) for piecewise buckets, "
            "ascending, last <= history-window-sec. Empty uses default "
            "15,60,180,600,1800 (0-15s, 15-60s, 1-3m, 3-10m, 10-30m). "
            "Only used when --kernel-type piecewise."
        ),
    )
    parser.add_argument("--edge-threshold", type=float, default=1e-3)
    parser.add_argument("--max-active-sources-per-dim", type=int, default=16)
    parser.add_argument("--branching-cap", type=float, default=0.9)
    parser.add_argument("--stability-radius", type=float, default=0.95)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help=(
            "Events processed per chunk in E-step. Peak pair memory per chunk "
            "is chunk_size * max_history_events * ~16 bytes. Default 20000 "
            "(~40 MB at K=128). Bump to 50000 for 2M+ events if you have headroom."
        ),
    )
    # Held-out validation (the thing that makes training meaningful):
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help=(
            "Fraction of the event sequence (by time) to hold out for validation. "
            "Final val LL is reported. 0.0 disables. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Patience iterations of no val LL improvement before early stop "
             "(only used with --selection-metric val).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("train", "val"),
        default="train",
        help=(
            "Metric driving model selection + early stop. 'train' (default, "
            "legacy): keep the train-LL-best weights, no val early stop; val LL "
            "is still printed each iter when --val-split>0 (informational only, "
            "model is bit-for-bit identical to a no-val run). 'val': select the "
            "val-LL peak snapshot and early-stop when val LL plateaus."
        ),
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
    )
    parser.add_argument("--min-group-events", type=int, default=1)
    parser.add_argument(
        "--load-topology",
        action="store_true",
        default=True,
        help="Load NE topology graph and report learned-edge consistency. Default: on.",
    )
    parser.add_argument(
        "--no-load-topology",
        action="store_false",
        dest="load_topology",
        help="Skip topology graph loading (faster startup; no topology report).",
    )
    parser.add_argument(
        "--topology-max-hops",
        type=int,
        default=2,
        help="Maximum NE graph hops considered when classifying learned edges. Default: 2.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t_total_start = time.monotonic()
    config = _build_config(args)
    _print_progress("[train] loading alarms...", args)
    alarm_events, alarm_metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
        regions=config.regions,
    )
    config = _adopt_loaded_regions(config, alarm_metadata)
    _print_progress(f"[train] loaded alarm events: {len(alarm_events)}", args)

    topology_index = None
    ne_graph_data = None
    # Feature mode needs the NE graph for device attributes — force-load it.
    need_graph = args.load_topology or config.edge_mode == "feature"
    if need_graph:
        _print_progress(f"[train] loading NE graph: {args.ne_graph}", args)
        ne_graph_data = load_ne_graph(args.ne_graph)
        if config.regions:
            ne_graph_data, _stats = filter_ne_graph_by_regions(ne_graph_data, config.regions)
        # The index must reach at least as far as the topology prior / feature
        # candidate generation needs.
        index_hops = max(args.topology_max_hops, args.topology_prior_max_hops, config.feature_topo_max_hops)
        _print_progress(
            f"[train] building topology index (max_hops={index_hops}) ...",
            args,
        )
        topology_index = NETopologyIndex.from_graph(
            ne_graph_data,
            max_hops=index_hops,
        )

    best_output = None if args.no_best_checkpoint else (args.best_output or _default_best_output(args.output))
    if best_output:
        _print_progress(f"[train] best checkpoint path: {best_output}", args)
    _print_progress("[train] fitting model (MAP EM)...", args)
    artifact = train_alarm_mhp(
        alarm_events,
        config,
        region_filter_stats=(alarm_metadata or {}).get("region_filter"),
        progress_callback=_training_progress if _progress_enabled(args) else None,
        verbose=_progress_enabled(args),
        topology_index=topology_index,
        ne_graph_data=ne_graph_data,
        best_checkpoint_path=best_output,
        # Full CLI argument snapshot (incl. args not in AlarmMHPConfig, e.g.
        # early-stop-patience / ne-graph / output paths) — persisted to both the
        # final artifact and best checkpoints so a run is exactly reproducible.
        # alarms is resolved to an absolute path so the input is recoverable from
        # the best checkpoint too (the post-hoc ["input"] below only lands on the
        # final artifact).
        run_args=_run_args_snapshot(args),
    )
    artifact.training_metadata["input"] = os.path.abspath(args.alarms)
    artifact.training_metadata["alarm_metadata"] = alarm_metadata
    _print_progress(f"[train] saving model artifact: {args.output}", args)
    save_alarm_mhp_artifact(args.output, artifact)
    md = artifact.training_metadata
    val_str = (
        f", val_ll={md['best_val_log_likelihood']:.4f}"
        if md.get("best_val_log_likelihood") is not None
        else ""
    )
    print(
        f"MHP model written to: {args.output}; "
        f"events={md['modeled_event_count']}, "
        f"types={md['type_count']}, "
        f"active_edges={md['active_edge_count']}, "
        f"iters={md['iterations_run']}, "
        f"converged={md['converged']}, "
        f"ll={md['best_log_likelihood']:.4f}{val_str}"
    )

    if _progress_enabled(args):
        _print_cascade_size_distribution(md.get("cascade_size_stats"))
        _print_topology_consistency(md.get("topology_consistency"))
        _print_bucket_mass(md.get("bucket_mass_distribution"))
        total_elapsed = time.monotonic() - t_total_start
        print(f"[train] total wall-clock: {_format_seconds(total_elapsed)}")


def _print_bucket_mass(report):
    """Metric 3 (piecewise): per-bucket excitation mass share."""
    if not report:
        return
    print("[train] piecewise bucket mass distribution (Σθ·width per bucket):")
    for bucket in report["buckets"]:
        bar_len = int(round(bucket["share"] * 40))
        bar = "█" * bar_len
        print(f"  {bucket['label']:>10s} : {bucket['share'] * 100:5.1f}%  {bar}")


def _format_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m{rem:04.1f}s"


def _print_cascade_size_distribution(stats):
    """Metric 1: BRUNCH-comparable cascade size histogram."""
    if not stats:
        return
    print("[train] cascade size distribution (hard parent assignments):")
    for bucket in stats["histogram"]:
        print(
            f"  size={bucket['label']:>5s} : "
            f"{bucket['cascade_count']:>7d} cascades, "
            f"{bucket['event_count']:>7d} events"
        )
    print(
        f"[train] multi(>=2) cascades: "
        f"{stats['multi_event_cascade_count']}/{stats['n_cascades']} "
        f"({stats['multi_event_cascade_share'] * 100:.1f}% of cascades, "
        f"{stats['multi_event_event_share'] * 100:.1f}% of events); "
        f"size mean={stats['mean_size']:.2f}, "
        f"median={stats['median_size']:.1f}, "
        f"max={stats['max_size']}"
    )


def _print_topology_consistency(report):
    """Metric 2: Learned-edge ↔ NE topology alignment."""
    if not report:
        return
    buckets = report["buckets"]
    total = report["total_active_edges"]
    if total == 0:
        return
    print("[train] topology consistency of learned edges:")
    for key in ("same_ne", "direct_link", "indirect_link", "no_topology", "unknown"):
        count = buckets.get(key, 0)
        share = count / total * 100
        print(f"  {key:<14s} : {count:>6d} ({share:5.1f}%)")
    print(
        f"[train] topology-related (same_ne + direct + indirect): "
        f"{report['topology_related_count']}/{total} "
        f"({report['topology_related_share'] * 100:.1f}%)"
    )
    top_edges = report.get("top_edges", [])
    if top_edges:
        print(f"[train] top-{min(len(top_edges), 10)} edges by α:")
        for i, edge in enumerate(top_edges[:10], 1):
            t = edge["target_type"]
            s = edge["source_type"]
            print(
                f"  #{i:2d} α={edge['alpha']:.4f} β={edge['beta']:.3f} "
                f"[{edge['relation']:<14s}]  {s}  →  {t}"
            )


if __name__ == "__main__":
    main()
