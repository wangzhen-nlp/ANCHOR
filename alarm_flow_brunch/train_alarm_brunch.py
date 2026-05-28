#!/usr/bin/env python3
"""Train reusable BRUNCH edge influence parameters from an alarm stream."""

from __future__ import annotations

from dataclasses import replace
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_brunch.aggregator import (
    AlarmBRUNCHConfig,
    save_alarm_brunch_artifact,
    train_alarm_brunch,
)
from alarm_flow_brunch.region_filter import filter_ne_graph_by_regions, load_ne_graph, parse_regions
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import parse_type_fields
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, resource_display


def _progress_enabled(args):
    return not args.quiet


def _print_progress(message, args):
    if _progress_enabled(args):
        print(message, flush=True)


def _format_checkpoint_path(template, payload):
    return template.format(
        sweep=payload["sweep"],
        sweep1=payload["sweep"] + 1,
    )


def _save_checkpoint_atomic(path, artifact):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    save_alarm_brunch_artifact(tmp_path, artifact)
    os.replace(tmp_path, path)


def _training_progress(stage, payload):
    if stage == "region_filter":
        if payload.get("enabled"):
            raw_checked = payload.get("raw_checked_alarm_count")
            raw_kept = payload.get("raw_kept_alarm_count")
            raw_dropped = payload.get("raw_dropped_alarm_count")
            raw_summary = (
                f"raw_kept={raw_kept}/{raw_checked}, raw_dropped={raw_dropped}, "
                if raw_checked is not None
                else ""
            )
            print(
                "[train] region filter: "
                f"regions={payload.get('regions', [])}, "
                f"{raw_summary}"
                f"events={payload.get('kept_event_count', 0)}/"
                f"{payload.get('input_event_count', 0)}, "
                f"allowed_devices={payload.get('allowed_device_count', 0)}",
                flush=True,
            )
        else:
            print(
                "[train] region filter: disabled; "
                f"events={payload.get('input_event_count', 0)}",
                flush=True,
            )
        return
    if stage == "vocab":
        print(
            "[train] vocab: "
            f"events={payload.get('considered_event_count', 0)}, "
            f"types={payload.get('type_count', 0)}, "
            f"alarm_sources={payload.get('alarm_source_count', 0)}, "
            f"alarm_types={payload.get('alarm_type_count', 0)}",
            flush=True,
        )
        return
    if stage == "sequence":
        print(
            "[train] sequence: "
            f"modeled_events={payload.get('sequence_event_position_count', 0)}, "
            f"target_windows={payload.get('target_window_count', 0)}, "
            f"history_pairs={payload.get('history_pair_count', 0)}, "
            f"max_history={payload.get('max_window_history_count', 0)}",
            flush=True,
        )
        return
    if stage == "fit_start":
        print(
            "[train] BRUNCH fit: "
            f"events={payload.get('modeled_event_count', 0)}, "
            f"types={payload.get('type_count', 0)}, "
            f"sweeps={payload.get('n_sweeps', 0)}, "
            f"burn_in={payload.get('burn_in', 0)}",
            flush=True,
        )
        return
    if stage == "fit_done":
        print(
            "[train] fit done: "
            f"active_edges={payload.get('active_edge_count', 0)}, "
            f"best_log_likelihood={payload.get('best_log_likelihood', float('nan')):.4f}",
            flush=True,
        )


def _build_config(args):
    return AlarmBRUNCHConfig(
        type_fields=parse_type_fields(args.type_fields),
        history_window_sec=args.history_window_sec,
        max_history_events=args.max_history_events,
        min_events=args.min_events,
        time_scale_sec=args.time_scale_sec,
        include_clear=args.include_clear,
        n_sweeps=args.n_sweeps,
        burn_in=args.burn_in,
        refit_params=args.refit_params,
        warm_start=not args.no_warm_start,
        seed=args.seed,
        sparse_alpha_threshold=args.sparse_alpha_threshold,
        max_active_sources_per_dim=args.max_active_sources_per_dim,
        min_group_events=args.min_group_events,
        topology_edge_policy=args.topology_edge_policy,
        topology_prefer_multiplier=args.topology_prefer_multiplier,
        topology_fallback_sources_per_dim=args.topology_fallback_sources_per_dim,
        non_topology_alpha_multiplier=args.non_topology_alpha_multiplier,
        regions=parse_regions(args.regions),
        parent_selection=args.parent_selection,
    )


def _adopt_loaded_regions(config, alarm_metadata):
    region_filter = (alarm_metadata or {}).get("region_filter") or {}
    if config.regions or not region_filter.get("enabled"):
        return config
    regions = parse_regions(region_filter.get("regions"))
    return replace(config, regions=regions) if regions else config


def main():
    parser = ArgumentParser(description="Train alarm-flow BRUNCH type-level influence parameters.")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache consumed by match_rules.")
    parser.add_argument("-o", "--output", required=True, help="Output BRUNCH model artifact JSON.")
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
    parser.add_argument("--start-time", default="", help="Raw-input first occurrence lower bound.")
    parser.add_argument("--end-time", default="", help="Raw-input first occurrence upper bound.")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0, help="Raw-input clear delay.")
    parser.add_argument(
        "--type-fields",
        default="alarm_source,alarm_type",
        help="Comma-separated fields forming the BRUNCH event type.",
    )
    parser.add_argument("--history-window-sec", type=float, default=900.0)
    parser.add_argument("--max-history-events", type=int, default=128)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--time-scale-sec", type=float, default=60.0)
    parser.add_argument("--include-clear", action="store_true")
    parser.add_argument("--n-sweeps", type=int, default=30)
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument(
        "--refit-params",
        action="store_true",
        help=(
            "Re-fit μ, α, β by Bayesian MLE between sweeps. Off by default — vanilla MLE "
            "on a single MCMC sample is numerically unstable and tends to collapse into "
            "an all-immigrant attractor on real alarm data."
        ),
    )
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--sparse-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--max-active-sources-per-dim", type=int, default=16)
    parser.add_argument("--min-group-events", type=int, default=1)
    parser.add_argument(
        "--topology-edge-policy",
        choices=("off", "prefer", "require"),
        default="prefer",
        help="Topology candidate edge policy used during parameter training.",
    )
    parser.add_argument("--topology-max-hops", type=int, default=2)
    parser.add_argument("--topology-prefer-multiplier", type=float, default=2.0)
    parser.add_argument("--topology-fallback-sources-per-dim", type=int, default=2)
    parser.add_argument(
        "--non-topology-alpha-multiplier",
        type=float,
        default=0.5,
        help=(
            "Alpha multiplier for non-topology fallback edges in prefer mode. "
            "Default: 0.5, matching the previous hard-coded behavior."
        ),
    )
    parser.add_argument(
        "--parent-selection",
        choices=("sample", "argmax"),
        default="sample",
        help=(
            "How to choose event/cluster parents inside each sweep. "
            "sample keeps stochastic BRUNCH inference; argmax uses deterministic maximum-weight parents."
        ),
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help=(
            "Region values to keep. Repeat this option or pass comma-separated values; "
            "only alarms/devices in these regions are trained."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print BRUNCH sweep progress every N sweeps. Default: 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help=(
            "Print within-sweep progress every N events/clusters. "
            "Use 0 to show only phase start/end. Default: 50000."
        ),
    )
    parser.add_argument(
        "--checkpoint-output",
        default="",
        help=(
            "Optional checkpoint artifact path. Use {sweep} or {sweep1} in the path "
            "to keep per-sweep files; otherwise the same file is overwritten."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N sweeps when --checkpoint-output is set. Default: 0.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final training summary.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.log_every < 1:
        parser.error("--log-every must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.checkpoint_output and args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1 when --checkpoint-output is set")
    if args.checkpoint_every > 0 and not args.checkpoint_output:
        parser.error("--checkpoint-output is required when --checkpoint-every is set")

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
    ne_graph_data = None
    topology_graph = None
    topology_region_stats = None
    if config.topology_edge_policy != "off":
        _print_progress(f"[train] loading NE graph: {args.ne_graph}", args)
        ne_graph_data = load_ne_graph(args.ne_graph)
        topology_graph = ne_graph_data
        if config.regions:
            topology_graph, topology_region_stats = filter_ne_graph_by_regions(
                ne_graph_data,
                config.regions,
            )
            _print_progress(
                "[train] topology region filter: "
                f"devices={topology_region_stats['allowed_device_count']}/"
                f"{topology_region_stats['original_device_count']}, "
                f"links={topology_region_stats['kept_link_count']}/"
                f"{topology_region_stats['original_link_count']}",
                args,
            )
    topology_index = None
    if config.topology_edge_policy != "off":
        _print_progress(
            "[train] building topology index: "
            f"policy={config.topology_edge_policy}, max_hops={args.topology_max_hops}",
            args,
        )
        topology_index = NETopologyIndex.from_graph(topology_graph, max_hops=args.topology_max_hops)

    def checkpoint_callback(checkpoint_artifact, payload):
        if not args.checkpoint_output:
            return
        if (payload["sweep"] + 1) % args.checkpoint_every != 0:
            return
        checkpoint_artifact.training_metadata["input"] = os.path.abspath(args.alarms)
        checkpoint_artifact.training_metadata["alarm_metadata"] = alarm_metadata
        if topology_region_stats is not None:
            checkpoint_artifact.training_metadata["topology_region_filter"] = topology_region_stats
        path = _format_checkpoint_path(args.checkpoint_output, payload)
        _save_checkpoint_atomic(path, checkpoint_artifact)
        _print_progress(
            "[train] checkpoint saved: "
            f"{path}; sweep={payload['sweep']}, "
            f"best_log_likelihood={payload['best_log_likelihood']:.4f}",
            args,
        )

    _print_progress("[train] fitting model...", args)
    artifact = train_alarm_brunch(
        alarm_events,
        config,
        topology_index=topology_index,
        ne_graph_data=ne_graph_data,
        region_filter_stats=(alarm_metadata or {}).get("region_filter"),
        progress_callback=_training_progress if _progress_enabled(args) else None,
        verbose=_progress_enabled(args),
        log_every=args.log_every,
        progress_every=args.progress_every,
        checkpoint_callback=checkpoint_callback if args.checkpoint_output else None,
    )
    artifact.training_metadata["input"] = os.path.abspath(args.alarms)
    artifact.training_metadata["alarm_metadata"] = alarm_metadata
    if topology_region_stats is not None:
        artifact.training_metadata["topology_region_filter"] = topology_region_stats
    _print_progress(f"[train] saving model artifact: {args.output}", args)
    save_alarm_brunch_artifact(args.output, artifact)
    print(
        f"BRUNCH model written to: {args.output}; "
        f"events={artifact.training_metadata['modeled_event_count']}, "
        f"types={artifact.training_metadata['type_count']}, "
        f"active_edges={artifact.training_metadata['active_edge_count']}, "
        f"best_log_likelihood={artifact.training_metadata['best_log_likelihood']:.4f}"
    )


if __name__ == "__main__":
    main()
