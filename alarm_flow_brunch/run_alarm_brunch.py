#!/usr/bin/env python3
"""Aggregate an ordered alarm stream into BRUNCH fault groups."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_brunch.aggregator import AlarmBRUNCHConfig, aggregate_alarm_flow
from alarm_flow_brunch.region_filter import filter_ne_graph_by_regions, load_ne_graph, parse_regions
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import parse_type_fields
from alarm_flow_brunch.visual_output import write_visual_groups
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON, resource_display


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_jsonl(path, records):
    count = 0
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


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
        stability_radius=args.stability_radius,
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
    parser = ArgumentParser(description="Infer BRUNCH fault groups from an ordered alarm stream.")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache consumed by match_rules.")
    parser.add_argument("-o", "--output", required=True, help="Output fault group JSON.")
    parser.add_argument("--edges-output", default="", help="Optional branching edge JSONL.")
    parser.add_argument(
        "--visual-output",
        default="",
        help="Optional visualization JSONL compatible with the fault group browser and propagation visualizer.",
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
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"Site metadata for --visual-output. Default: {resource_display('site_graph.json')}.",
    )
    parser.add_argument(
        "--visual-ne-scope",
        choices=("alarm-only", "site-context"),
        default="alarm-only",
        help="NEs in --visual-output: grouped alarm devices only, or all devices at group sites.",
    )
    parser.add_argument("--start-time", default="", help="Raw-input first occurrence lower bound.")
    parser.add_argument("--end-time", default="", help="Raw-input first occurrence upper bound.")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0, help="Raw-input clear delay.")
    parser.add_argument(
        "--type-fields",
        default="alarm_source,alarm_type",
        help=(
            "Comma-separated alarm event fields forming the BRUNCH dimension. "
            "Supported: alarm_source,alarm_type,alarm_title,site_id."
        ),
    )
    parser.add_argument(
        "--history-window-sec",
        type=float,
        default=900.0,
        help="Only strictly earlier alarms within this window can be candidate parents. Default: 900.",
    )
    parser.add_argument(
        "--max-history-events",
        type=int,
        default=128,
        help="Keep at most this many earlier alarms per target when initializing candidate edges. Default: 128.",
    )
    parser.add_argument("--min-events", type=int, default=2, help="Minimum modeled event count.")
    parser.add_argument("--time-scale-sec", type=float, default=60.0, help="Divide timestamps by this scale.")
    parser.add_argument("--include-clear", action="store_true", help="Include synthetic clear events.")
    parser.add_argument("--n-sweeps", type=int, default=30, help="BRUNCH MEDIA sweeps. Default: 30.")
    parser.add_argument("--burn-in", type=int, default=5, help="BRUNCH burn-in sweeps. Default: 5.")
    parser.add_argument(
        "--refit-params",
        action="store_true",
        help=(
            "Re-fit μ, α, β by Bayesian MLE between sweeps. Off by default — vanilla MLE "
            "on a single MCMC sample is numerically unstable and tends to collapse into "
            "an all-immigrant attractor on real alarm data."
        ),
    )
    parser.add_argument("--no-warm-start", action="store_true", help="Disable nearest-in-window warm start.")
    parser.add_argument(
        "--sparse-alpha-threshold",
        type=float,
        default=1e-4,
        help="Ignore alpha edges at or below this magnitude. Default: 1e-4.",
    )
    parser.add_argument(
        "--max-active-sources-per-dim",
        type=int,
        default=16,
        help="Candidate source alarm dimensions per target dimension. Default: 16.",
    )
    parser.add_argument(
        "--topology-edge-policy",
        choices=("off", "prefer", "require"),
        default="prefer",
        help=(
            "How NE topology constrains active BRUNCH type edges: off ignores topology; "
            "prefer prioritizes topology-related edges while keeping a small fallback; "
            "require keeps only topology-related cross-device candidates. Default: prefer."
        ),
    )
    parser.add_argument(
        "--topology-max-hops",
        type=int,
        default=2,
        help="Maximum NE graph hops used by topology-edge-policy. Default: 2.",
    )
    parser.add_argument(
        "--topology-prefer-multiplier",
        type=float,
        default=2.0,
        help="Alpha multiplier for topology-related candidate edges in prefer/require modes. Default: 2.0.",
    )
    parser.add_argument(
        "--topology-fallback-sources-per-dim",
        type=int,
        default=2,
        help="Non-topology fallback source dimensions kept per target in prefer mode. Default: 2.",
    )
    parser.add_argument(
        "--stability-radius",
        type=float,
        default=0.95,
        help="Stationarity cap for the initial alpha matrix spectral radius. Default: 0.95. Set to 0 or negative to disable.",
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
        "--min-group-events",
        type=int,
        default=1,
        help="Drop inferred groups smaller than this. Default: 1.",
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help=(
            "Region values to keep. Repeat this option or pass comma-separated values; "
            "only alarms/devices in these regions are modeled."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    config = _build_config(args)
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
    ne_graph_data = None
    topology_graph = None
    topology_region_stats = None
    if config.topology_edge_policy != "off":
        ne_graph_data = load_ne_graph(args.ne_graph)
        topology_graph = ne_graph_data
        if config.regions:
            topology_graph, topology_region_stats = filter_ne_graph_by_regions(
                ne_graph_data,
                config.regions,
            )
    topology_index = None
    if config.topology_edge_policy != "off":
        topology_index = NETopologyIndex.from_graph(topology_graph, max_hops=args.topology_max_hops)
    output = aggregate_alarm_flow(
        alarm_events,
        config,
        topology_index=topology_index,
        ne_graph_data=ne_graph_data,
        region_filter_stats=(alarm_metadata or {}).get("region_filter"),
    )
    payload = output.to_json_payload()
    payload["metadata"]["input"] = os.path.abspath(args.alarms)
    payload["metadata"]["alarm_metadata"] = alarm_metadata
    if topology_region_stats is not None:
        payload["metadata"]["topology_region_filter"] = topology_region_stats
    _write_json(args.output, payload)
    print(
        f"BRUNCH fault groups written to: {args.output}; "
        f"groups={output.metadata['group_count']}, "
        f"events={output.metadata['modeled_event_count']}, "
        f"types={output.metadata['type_count']}, "
        f"active_edges={output.metadata['active_edge_count']}"
    )
    if args.edges_output:
        edge_count = _write_jsonl(args.edges_output, output.edges)
        print(f"branching edges written to: {args.edges_output}; edges={edge_count}")
    if args.visual_output:
        visual_count = write_visual_groups(
            args.visual_output,
            output.groups,
            ne_graph_path=args.ne_graph,
            site_graph_path=args.site_graph,
            ne_scope=args.visual_ne_scope,
        )
        print(f"visual groups written to: {args.visual_output}; groups={visual_count}")


if __name__ == "__main__":
    main()
