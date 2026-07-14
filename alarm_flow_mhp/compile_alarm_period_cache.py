#!/usr/bin/env python3
"""Offline compiler for the AlarmPeriod MHP sparse association cache.

The cache covers every feature entity in the supplied NE graph crossed with
every alarm type learned by the feature artifact, plus all eight frozen
dynamic-state combinations. It stores only edges whose peak score reaches the
configured immigrant threshold. The online engine reconstructs this universe
from the same fingerprinted graph/model inputs, so zero-edge signatures need
no persistent negative records.

Devices added after the graph snapshot (or alarm types absent from the model
vocabulary) are compiled incrementally by ``stream_alarm_period_mhp.py`` and
kept in process memory only.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_mhp.stream_alarm_period_mhp import (
    ASSOCIATION_CACHE_FORMAT,
    ASSOCIATION_CACHE_VERSION,
    CompiledAssociationPlan,
    PeriodStreamConfig,
    _build_runtime_scorers,
    association_cache_fingerprint,
    build_compact_csr_arrays,
    graph_period_types,
    write_association_cache,
)
from alarm_flow_mhp.topology_relation_prior import parse_topology_relation_prior
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


def _format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class _BinaryEdgeSpool:
    """Bounded-memory raw edge writer used before the final CSR archive."""

    _FLOAT_FIELDS = ("base_scores", "thresholds", "past_windows", "future_windows")

    def __init__(self, directory, period_types, buffer_size=100_000):
        self.period_types = tuple(period_types)
        self.type_to_id = {value: index for index, value in enumerate(self.period_types)}
        self.signature_count = len(self.period_types) * 8
        self.id_dtype = (
            np.uint32
            if self.signature_count <= np.iinfo(np.uint32).max
            else np.uint64
        )
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.paths = {
            name: os.path.join(directory, f"{name}.bin")
            for name in ("target_signature_ids", "source_signature_ids") + self._FLOAT_FIELDS
        }
        self.streams = {name: open(path, "wb") for name, path in self.paths.items()}
        self.buffers = {name: [] for name in self.paths}

    def append(self, target, source, edge):
        self.buffers["target_signature_ids"].append(
            self.type_to_id[target.period_type] * 8 + target.initial_state
        )
        self.buffers["source_signature_ids"].append(
            self.type_to_id[source.period_type] * 8 + source.initial_state
        )
        self.buffers["base_scores"].append(edge.base_score)
        self.buffers["thresholds"].append(edge.threshold)
        self.buffers["past_windows"].append(edge.past_window_sec)
        self.buffers["future_windows"].append(edge.future_window_sec)
        self.count += 1
        if len(self.buffers["base_scores"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        size = len(self.buffers["base_scores"])
        if not size:
            return
        for name in ("target_signature_ids", "source_signature_ids"):
            np.asarray(self.buffers[name], dtype=self.id_dtype).tofile(self.streams[name])
        for name in self._FLOAT_FIELDS:
            np.asarray(self.buffers[name], dtype=np.float64).tofile(self.streams[name])
        for values in self.buffers.values():
            values.clear()

    def arrays(self):
        self.flush()
        for stream in self.streams.values():
            stream.close()
        if self.count == 0:
            empty_ids = np.empty(0, dtype=self.id_dtype)
            empty_values = np.empty(0, dtype=np.float64)
            return build_compact_csr_arrays(
                empty_ids,
                empty_ids,
                empty_values,
                empty_values,
                empty_values,
                empty_values,
                self.signature_count,
            )
        raw = {
            "target_signature_ids": np.memmap(
                self.paths["target_signature_ids"], dtype=self.id_dtype, mode="r", shape=(self.count,)
            ),
            "source_signature_ids": np.memmap(
                self.paths["source_signature_ids"], dtype=self.id_dtype, mode="r", shape=(self.count,)
            ),
        }
        for name in self._FLOAT_FIELDS:
            raw[name] = np.memmap(
                self.paths[name], dtype=np.float64, mode="r", shape=(self.count,)
            )
        return build_compact_csr_arrays(
            raw["target_signature_ids"],
            raw["source_signature_ids"],
            raw["base_scores"],
            raw["thresholds"],
            raw["past_windows"],
            raw["future_windows"],
            self.signature_count,
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compile an offline sparse association cache for AlarmPeriod MHP."
    )
    parser.add_argument("model", help="Trained feature-mode alarm-flow MHP artifact JSON.")
    parser.add_argument("output", help="Compact binary association cache output (.npz).")
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
        help="Refresh the progress bar after this many directed type pairs; 0 disables.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Build the candidate index, print total_type_pairs, then exit without scoring/writing.",
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

    period_types, graph_entity_count, alarm_type_count = graph_period_types(
        artifact, scorer
    )
    if not period_types:
        parser.error("NE graph × model alarm-type vocabulary produced no period types")
    if not args.quiet:
        print(
            f"[period-cache] graph_entities={graph_entity_count}, "
            f"model_alarm_types={alarm_type_count}, types={len(period_types)}, "
            f"signatures={len(period_types) * 8}, "
            f"scope={config.candidate_scope}",
            flush=True,
        )

    plan = CompiledAssociationPlan(scorer, mu_scorer, artifact, config)
    candidate_t0 = time.monotonic()
    prepared_candidates = plan.prepare_candidate_period_types(period_types)
    period_types = prepared_candidates["period_types"]
    total_type_pairs = prepared_candidates["total_pair_count"]
    if not args.quiet:
        print(
            f"[period-cache] total_type_pairs={total_type_pairs}, "
            f"max_state_edges={total_type_pairs * 64}, "
            f"candidate_index_elapsed={time.monotonic() - candidate_t0:.1f}s, "
            "estimated_active_edges=pending, ETA=pending",
            flush=True,
        )
    if args.count_only:
        return
    if not str(args.output).lower().endswith(".npz"):
        parser.error("binary association cache output must end with .npz")

    last_report = 0
    compile_t0 = time.monotonic()
    interactive_progress = bool(getattr(sys.stdout, "isatty", lambda: False)())
    last_line_width = 0

    def render_progress(type_pairs, active_edges, pruned_pairs, final=False):
        nonlocal last_line_width
        elapsed = time.monotonic() - compile_t0
        rate = type_pairs / max(elapsed, 1e-12)
        remaining = max(total_type_pairs - type_pairs, 0)
        eta = remaining / max(rate, 1e-12)
        estimated_active_edges = (
            round(active_edges / type_pairs * total_type_pairs) if type_pairs else 0
        )
        active_ratio = active_edges / max(active_edges + pruned_pairs, 1)
        fraction = type_pairs / max(total_type_pairs, 1)
        bar_width = 28
        filled = min(bar_width, int(fraction * bar_width))
        if final and type_pairs >= total_type_pairs:
            filled = bar_width
        bar = "█" * filled + "░" * (bar_width - filled)
        line = (
            f"[period-cache] [{bar}] {fraction * 100:6.2f}% "
            f"{type_pairs:,}/{total_type_pairs:,} pairs | "
            f"edges={active_edges:,} ({active_ratio:.1%}, est={estimated_active_edges:,}) | "
            f"{rate:.1f} pairs/s | ETA {_format_duration(eta)}"
        )
        if interactive_progress:
            padded = line.ljust(last_line_width)
            print(f"\r{padded}", end="\n" if final else "", flush=True)
            last_line_width = len(line)
        else:
            print(line, flush=True)

    def report(type_pairs, active_edges, pruned_pairs):
        nonlocal last_report
        if args.quiet or not args.progress_every:
            return
        if type_pairs - last_report < args.progress_every:
            return
        last_report = type_pairs
        render_progress(type_pairs, active_edges, pruned_pairs)

    with tempfile.TemporaryDirectory(prefix="alarm-period-cache-") as spool_dir:
        spool = _BinaryEdgeSpool(spool_dir, period_types)
        type_pair_count = plan.precompile_period_types(
            period_types,
            progress=report,
            prepared_candidates=prepared_candidates,
            edge_sink=spool.append,
        )
        if not args.quiet and args.progress_every:
            render_progress(
                type_pair_count,
                plan.compiled_pair_count,
                plan.pruned_pair_count,
                final=True,
            )
        if not args.quiet:
            print(
                f"[period-cache] scoring complete; building CSR and compressing "
                f"{spool.count:,} positive edges ...",
                flush=True,
            )
        arrays = spool.arrays()
        fingerprint = association_cache_fingerprint(
            args.model,
            args.ne_graph,
            args.site_graph,
            config,
            artifact.config.topology_node_field,
        )
        payload = {
            "format": ASSOCIATION_CACHE_FORMAT,
            "version": ASSOCIATION_CACHE_VERSION,
            "fingerprint": fingerprint,
            "arrays": arrays,
            "metadata": {
                "type_universe": "graph",
                "period_type_count": len(period_types),
                "signature_count": len(period_types) * 8,
                "edge_count": spool.count,
                "pruned_pair_count": plan.pruned_pair_count,
                "graph_entity_count": graph_entity_count,
                "model_alarm_type_count": alarm_type_count,
                "directed_period_type_pair_count": type_pair_count,
            },
        }
        write_association_cache(args.output, payload)
    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(
            f"[period-cache] done: signatures={payload['metadata']['signature_count']}, "
            f"edges={payload['metadata']['edge_count']}, pruned={plan.pruned_pair_count}, "
            f"size={os.path.getsize(args.output) / (1024 * 1024):.1f}MiB, "
            f"elapsed={elapsed:.2f}s; output={os.path.abspath(args.output)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
