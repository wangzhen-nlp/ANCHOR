#!/usr/bin/env python3
"""Offline compiler for the AlarmPeriod MHP sparse association cache.

The cache covers every feature entity in the supplied NE graph crossed with
every alarm type learned by the feature artifact. It stores only edges whose
peak score reaches the configured immigrant threshold. In target-dynamic mode,
only the target's eight frozen states are materialized because source state
cannot affect alpha. The online engine reconstructs this universe from the
same fingerprinted graph/model inputs, so zero-edge signatures need no
persistent negative records.

Devices added after the graph snapshot (or alarm types absent from the model
vocabulary) are compiled incrementally by ``stream_alarm_period_mhp.py`` and
kept in process memory only.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_mhp.candidate_policy import (
    candidate_policy_fingerprint,
    load_candidate_policy,
)
from alarm_flow_mhp.stream_alarm_period_mhp import (
    ASSOCIATION_CACHE_FORMAT,
    ASSOCIATION_CACHE_VERSION,
    CACHE_STATE_LAYOUT_TARGET_ONLY,
    CompiledAssociationPlan,
    PeriodStreamConfig,
    _association_plan_config,
    _build_runtime_scorers,
    association_cache_fingerprint,
    association_cache_state_layout,
    build_compact_csr_arrays,
    graph_period_types,
    write_association_cache,
)
from alarm_flow_mhp.topology_relation_prior import parse_topology_relation_prior
from alarm_tools.progress_utils import ProgressBar
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


class _BinaryEdgeSpool:
    """Bounded-memory raw edge writer used before the final CSR archive."""

    _FLOAT_FIELDS = ("base_scores", "thresholds", "past_windows", "future_windows")

    def __init__(
        self,
        directory,
        period_types,
        state_layout,
        buffer_size=100_000,
    ):
        self.period_types = tuple(period_types)
        self.type_to_id = {value: index for index, value in enumerate(self.period_types)}
        self.state_layout = str(state_layout)
        self.signature_count = len(self.period_types) * 8
        self.source_key_count = (
            len(self.period_types)
            if self.state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else self.signature_count
        )
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
        self.batch_buffers = {name: [] for name in self.paths}
        self.batch_count = 0
        self.memmaps = []

    def append(self, target, source, edge):
        if self.batch_count:
            self.flush()
        self.buffers["target_signature_ids"].append(
            self.type_to_id[target.period_type] * 8 + target.initial_state
        )
        self.buffers["source_signature_ids"].append(
            self.type_to_id[source.period_type]
            if self.state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else self.type_to_id[source.period_type] * 8 + source.initial_state
        )
        self.buffers["base_scores"].append(edge.base_score)
        self.buffers["thresholds"].append(edge.threshold)
        self.buffers["past_windows"].append(edge.past_window_sec)
        self.buffers["future_windows"].append(edge.future_window_sec)
        self.count += 1
        if len(self.buffers["base_scores"]) >= self.buffer_size:
            self.flush()

    def append_batch(
        self,
        target_type,
        target_states,
        source_types,
        source_indices,
        base_scores,
        threshold,
        past_windows,
        future_windows,
    ):
        """Append vectorized target-dynamic rows without per-edge objects."""
        if self.buffers["base_scores"]:
            self.flush()
        target_states = np.asarray(target_states, dtype=self.id_dtype)
        source_indices = np.asarray(source_indices, dtype=np.int64)
        size = len(target_states)
        if not size:
            return
        source_type_ids = np.fromiter(
            (self.type_to_id[value] for value in source_types),
            dtype=self.id_dtype,
            count=len(source_types),
        )
        values = {
            "target_signature_ids": (
                self.type_to_id[target_type] * 8 + target_states
            ).astype(self.id_dtype, copy=False),
            "source_signature_ids": source_type_ids[source_indices],
            "base_scores": np.asarray(base_scores, dtype=np.float64),
            "thresholds": np.full(size, float(threshold), dtype=np.float64),
            "past_windows": np.asarray(past_windows, dtype=np.float64),
            "future_windows": np.asarray(future_windows, dtype=np.float64),
        }
        for name, value in values.items():
            self.batch_buffers[name].append(value)
        self.batch_count += size
        self.count += size
        if self.batch_count >= self.buffer_size:
            self.flush()

    def flush(self):
        size = len(self.buffers["base_scores"])
        if size:
            for name in ("target_signature_ids", "source_signature_ids"):
                np.asarray(self.buffers[name], dtype=self.id_dtype).tofile(self.streams[name])
            for name in self._FLOAT_FIELDS:
                np.asarray(self.buffers[name], dtype=np.float64).tofile(self.streams[name])
            for values in self.buffers.values():
                values.clear()
        if self.batch_count:
            for name in ("target_signature_ids", "source_signature_ids"):
                np.concatenate(self.batch_buffers[name]).astype(
                    self.id_dtype, copy=False
                ).tofile(self.streams[name])
            for name in self._FLOAT_FIELDS:
                np.concatenate(self.batch_buffers[name]).astype(
                    np.float64, copy=False
                ).tofile(self.streams[name])
            for values in self.batch_buffers.values():
                values.clear()
            self.batch_count = 0

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
                source_key_count=self.source_key_count,
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
        self.memmaps = list(raw.values())
        return build_compact_csr_arrays(
            raw["target_signature_ids"],
            raw["source_signature_ids"],
            raw["base_scores"],
            raw["thresholds"],
            raw["past_windows"],
            raw["future_windows"],
            self.signature_count,
            source_key_count=self.source_key_count,
        )

    def release_mmaps(self):
        """Release spool mappings before Windows removes the temp directory."""
        for stream in self.streams.values():
            if not stream.closed:
                stream.close()
        for array in self.memmaps:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
        self.memmaps.clear()


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
    parser.add_argument(
        "--candidate-scope",
        choices=("related", "global", "unrelated"),
        default="related",
    )
    parser.add_argument(
        "--candidate-policy",
        default="",
        help=(
            "Approved candidate policy JSON; required for "
            "--candidate-scope unrelated."
        ),
    )
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

    if config.candidate_scope == "unrelated" and not args.candidate_policy:
        parser.error("--candidate-policy is required for --candidate-scope unrelated")
    if args.candidate_policy and config.candidate_scope != "unrelated":
        parser.error("--candidate-policy requires --candidate-scope unrelated")
    candidate_policy = None
    if args.candidate_policy:
        try:
            expected_policy_fingerprint = candidate_policy_fingerprint(
                args.model,
                args.ne_graph,
                args.site_graph,
                _association_plan_config(config),
                artifact.config.topology_node_field,
            )
            candidate_policy = load_candidate_policy(
                args.candidate_policy,
                expected_fingerprint=expected_policy_fingerprint,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load --candidate-policy: {exc}")

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

    plan = CompiledAssociationPlan(
        scorer,
        mu_scorer,
        artifact,
        config,
        candidate_policy=candidate_policy,
    )
    state_layout = association_cache_state_layout(
        getattr(artifact.config, "dynamic_alpha", "off")
    )
    state_expansion = 8 if state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY else 64
    compile_backend = (
        "cpu-vectorized"
        if state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
        else "cpu-scalar"
    )
    candidate_t0 = time.monotonic()
    prepared_candidates = plan.prepare_candidate_period_types(period_types)
    period_types = prepared_candidates["period_types"]
    total_type_pairs = prepared_candidates["total_pair_count"]
    if not args.quiet:
        print(
            f"[period-cache] total_type_pairs={total_type_pairs}, "
            f"state_layout={state_layout}, "
            f"backend={compile_backend}, "
            f"states_per_type_pair={state_expansion}, "
            f"max_state_edges={total_type_pairs * state_expansion}, "
            f"candidate_index_elapsed={time.monotonic() - candidate_t0:.1f}s, "
            "estimated_active_edges=pending, ETA=pending",
            flush=True,
        )
    if args.count_only:
        return
    if not str(args.output).lower().endswith(".npz"):
        parser.error("binary association cache output must end with .npz")

    last_report = 0
    progress_bar = (
        ProgressBar(total_type_pairs, "编译 AlarmPeriod 关联缓存")
        if not args.quiet and args.progress_every
        else None
    )

    def update_progress(type_pairs, active_edges, pruned_pairs):
        if progress_bar is None:
            return
        estimated_active_edges = (
            round(active_edges / type_pairs * total_type_pairs) if type_pairs else 0
        )
        active_ratio = active_edges / max(active_edges + pruned_pairs, 1)
        progress_bar.extra_text = (
            f"edges={active_edges:,} ({active_ratio:.1%}, "
            f"est={estimated_active_edges:,})"
        )
        progress_bar.set(type_pairs)

    def report(type_pairs, active_edges, pruned_pairs):
        nonlocal last_report
        if args.quiet or not args.progress_every:
            return
        if type_pairs - last_report < args.progress_every:
            return
        last_report = type_pairs
        update_progress(type_pairs, active_edges, pruned_pairs)

    with tempfile.TemporaryDirectory(prefix="alarm-period-cache-") as spool_dir:
        spool = _BinaryEdgeSpool(spool_dir, period_types, state_layout)
        arrays = None
        payload = None
        try:
            type_pair_count = plan.precompile_period_types(
                period_types,
                progress=report,
                prepared_candidates=prepared_candidates,
                edge_sink=spool.append,
                edge_batch_sink=spool.append_batch,
            )
            if progress_bar is not None:
                update_progress(
                    type_pair_count,
                    plan.compiled_pair_count,
                    plan.pruned_pair_count,
                )
                progress_bar.close()
                progress_bar = None
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
                candidate_policy_path=args.candidate_policy,
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
                    "source_key_count": spool.source_key_count,
                    "state_layout": state_layout,
                    "edge_count": spool.count,
                    "pruned_pair_count": plan.pruned_pair_count,
                    "graph_entity_count": graph_entity_count,
                    "model_alarm_type_count": alarm_type_count,
                    "directed_period_type_pair_count": type_pair_count,
                    "candidate_policy_validation": (
                        dict(candidate_policy.validation or {})
                        if candidate_policy is not None
                        else {}
                    ),
                },
            }
            write_association_cache(args.output, payload)
        finally:
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            # np.asarray(memmap) keeps the underlying Windows file mapping alive.
            # Drop every view before closing mappings and leaving TemporaryDirectory.
            if payload is not None:
                payload.pop("arrays", None)
            arrays = None
            spool.release_mmaps()
    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(
            f"[period-cache] done: signatures={payload['metadata']['signature_count']}, "
            f"edges={payload['metadata']['edge_count']}, pruned={plan.pruned_pair_count}, "
            f"prescreen_dropped_pairs={plan.prescreen_dropped_pair_count}, "
            f"size={os.path.getsize(args.output) / (1024 * 1024):.1f}MiB, "
            f"elapsed={elapsed:.2f}s; output={os.path.abspath(args.output)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
