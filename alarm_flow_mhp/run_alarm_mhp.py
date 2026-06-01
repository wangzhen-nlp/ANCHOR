#!/usr/bin/env python3
"""Apply a trained MHP artifact to an alarm stream offline and emit groups."""

from __future__ import annotations

import json
import os
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_brunch.visual_output import write_visual_groups
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_mhp.aggregator import (
    AlarmMHPOutput,
    infer_alarm_mhp,
    load_alarm_mhp_artifact,
)
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


def main():
    parser = ArgumentParser(description="Infer MHP fault groups using a trained model artifact.")
    parser.add_argument("model", help="Trained MHP artifact JSON (produced by train_alarm_mhp.py).")
    parser.add_argument("alarms", help="Raw alarms or prepare_sorted_alarms cache.")
    parser.add_argument("-o", "--output", required=True, help="Output fault group JSON.")
    parser.add_argument("--edges-output", default="", help="Optional branching edge JSONL.")
    parser.add_argument(
        "--visual-output",
        default="",
        help="Optional visualization JSONL compatible with the fault group browser.",
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
    parser.add_argument("--start-time", default="", help="Lower bound on first occurrence.")
    parser.add_argument("--end-time", default="", help="Upper bound on first occurrence.")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help="Override artifact regions. Omit to reuse the model's regions.",
    )
    parser.add_argument(
        "--min-group-events",
        type=int,
        default=None,
        help="Override artifact min_group_events: only emit groups of at least this size.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    t_total_start = time.monotonic()
    if not args.quiet:
        print(f"[run] loading MHP artifact: {args.model}", flush=True)
    artifact = load_alarm_mhp_artifact(args.model)
    if not args.quiet:
        print(
            f"[run] artifact: events_trained={artifact.training_metadata.get('train_event_count', 'n/a')}, "
            f"types={artifact.training_metadata.get('type_count', len(artifact.vocabs.type_vocab))}, "
            f"active_edges={len(artifact.params.edge_alpha)}",
            flush=True,
        )

    # Region override: explicit --regions takes precedence, else artifact's regions
    regions = parse_regions(args.regions) if args.regions is not None else artifact.config.regions
    if not args.quiet:
        print(
            f"[run] region filter: {sorted(regions) if regions else '<none — disabled>'}",
            flush=True,
        )

    if not args.quiet:
        print(f"[run] loading alarms: {args.alarms}", flush=True)
    alarm_events, alarm_metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
        regions=regions,
    )
    if not args.quiet:
        print(f"[run] loaded alarm events: {len(alarm_events)}", flush=True)

    # If the user overrode regions, we need to make a config copy with those
    # regions; otherwise reuse artifact.config as-is.
    if args.regions is not None:
        from dataclasses import replace as _replace
        config = _replace(artifact.config, regions=regions)
    else:
        config = artifact.config

    output: AlarmMHPOutput = infer_alarm_mhp(
        alarm_events,
        artifact,
        config=config,
        region_filter_stats=(alarm_metadata or {}).get("region_filter"),
        verbose=not args.quiet,
        min_group_events=args.min_group_events,
    )

    payload = output.to_json_payload()
    payload["metadata"]["input"] = os.path.abspath(args.alarms)
    payload["metadata"]["alarm_metadata"] = alarm_metadata
    payload["metadata"]["model"] = os.path.abspath(args.model)
    _write_json(args.output, payload)
    md = output.metadata
    print(
        f"MHP fault groups written to: {args.output}; "
        f"groups={md['group_count']}, "
        f"events={md['modeled_event_count']}, "
        f"types={md['type_count']}, "
        f"active_edges={md['active_edge_count']}, "
        f"branching_edges={md['branching_edge_count']}"
    )

    if args.edges_output:
        n = _write_jsonl(args.edges_output, output.edges)
        print(f"branching edges written to: {args.edges_output}; edges={n}")

    if args.visual_output:
        visual_count = write_visual_groups(
            args.visual_output,
            output.groups,
            ne_graph_path=args.ne_graph,
            site_graph_path=args.site_graph,
            ne_scope=args.visual_ne_scope,
        )
        print(f"visual groups written to: {args.visual_output}; groups={visual_count}")

    cascade_stats = md.get("cascade_size_stats")
    if cascade_stats and not args.quiet:
        print("[run] cascade size distribution:")
        for bucket in cascade_stats["histogram"]:
            print(
                f"  size={bucket['label']:>5s} : "
                f"{bucket['cascade_count']:>7d} cascades, "
                f"{bucket['event_count']:>7d} events"
            )
        print(
            f"[run] multi(>=2) cascades: "
            f"{cascade_stats['multi_event_cascade_count']}/{cascade_stats['n_cascades']} "
            f"({cascade_stats['multi_event_cascade_share'] * 100:.1f}% of cascades, "
            f"{cascade_stats['multi_event_event_share'] * 100:.1f}% of events); "
            f"mean={cascade_stats['mean_size']:.2f}, max={cascade_stats['max_size']}"
        )

    total = time.monotonic() - t_total_start
    if not args.quiet:
        if total < 60:
            print(f"[run] total wall-clock: {total:.1f}s")
        else:
            print(f"[run] total wall-clock: {int(total // 60)}m{total % 60:04.1f}s")


if __name__ == "__main__":
    main()
