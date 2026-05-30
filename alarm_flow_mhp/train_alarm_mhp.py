#!/usr/bin/env python3
"""Train alarm-flow MHP (MAP EM) edge influence parameters."""

from __future__ import annotations

from dataclasses import replace
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import (
    AlarmMHPConfig,
    save_alarm_mhp_artifact,
    train_alarm_mhp,
)
from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
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


def _build_config(args):
    return AlarmMHPConfig(
        type_fields=parse_type_fields(args.type_fields),
        history_window_sec=args.history_window_sec,
        max_history_events=args.max_history_events,
        min_events=args.min_events,
        time_scale_sec=args.time_scale_sec,
        include_clear=args.include_clear,
        max_iters=args.max_iters,
        tol=args.tol,
        alpha_prior_strength=args.alpha_prior_strength,
        alpha_prior_mean=args.alpha_prior_mean,
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
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
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


def main():
    parser = ArgumentParser(description="Train alarm-flow MHP via MAP EM.")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache.")
    parser.add_argument("-o", "--output", required=True, help="Output MHP model artifact JSON.")
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
    parser.add_argument("--alpha-prior-strength", type=float, default=10.0)
    parser.add_argument("--alpha-prior-mean", type=float, default=0.1)
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
    parser.add_argument("--edge-threshold", type=float, default=1e-3)
    parser.add_argument("--max-active-sources-per-dim", type=int, default=16)
    parser.add_argument("--branching-cap", type=float, default=0.9)
    parser.add_argument("--stability-radius", type=float, default=0.95)
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
        help="Patience iterations of no val LL improvement before early stop.",
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
    )
    parser.add_argument("--min-group-events", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

    _print_progress("[train] fitting model (MAP EM)...", args)
    artifact = train_alarm_mhp(
        alarm_events,
        config,
        region_filter_stats=(alarm_metadata or {}).get("region_filter"),
        progress_callback=_training_progress if _progress_enabled(args) else None,
        verbose=_progress_enabled(args),
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


if __name__ == "__main__":
    main()
