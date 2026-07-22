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
import contextlib
import multiprocessing as mp
import os
import shutil
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


# Forked workers inherit these via copy-on-write; the parent sets them before
# creating the pool so the heavy plan/candidate index need not be pickled.
_PARALLEL_CTX: dict = {}


def _thread_limit():
    """Cap native (BLAS/OpenMP) threads per worker to avoid oversubscription."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except Exception:  # threadpoolctl is optional; fall back to a no-op.
        return contextlib.nullcontext()


def _compile_chunk(task):
    """Score one contiguous target-entity chunk into its own spool shard.

    Runs inside a forked worker. Counters are reset per chunk so the returned
    values are deltas the parent can sum. Only raw edge ``.bin`` files are
    produced; the parent concatenates shards in chunk order and builds the CSR.
    """
    idx, start, stop = task
    plan = _PARALLEL_CTX["plan"]
    prepared = _PARALLEL_CTX["prepared"]
    period_types = _PARALLEL_CTX["period_types"]
    state_layout = _PARALLEL_CTX["state_layout"]
    spool_root = _PARALLEL_CTX["spool_root"]
    plan.compiled_pair_count = 0
    plan.pruned_pair_count = 0
    plan.prescreen_dropped_pair_count = 0
    shard_dir = os.path.join(spool_root, f"shard-{idx:05d}")
    os.makedirs(shard_dir, exist_ok=True)
    spool = _BinaryEdgeSpool(shard_dir, period_types, state_layout)
    chunk_prepared = dict(prepared)
    chunk_prepared["period_types"] = period_types[start:stop]
    with _thread_limit():
        type_pair_count = plan._precompile_target_only_batches(
            chunk_prepared, None, spool.append_batch
        )
    spool.flush()
    for stream in spool.streams.values():
        stream.close()
    return {
        "idx": idx,
        "paths": dict(spool.paths),
        "count": spool.count,
        "type_pair_count": type_pair_count,
        "target_count": stop - start,
        "compiled_pair_count": plan.compiled_pair_count,
        "pruned_pair_count": plan.pruned_pair_count,
        "prescreen_dropped_pair_count": plan.prescreen_dropped_pair_count,
    }


def _entity_aligned_chunks(period_types, workers):
    """Split globally sorted period types into contiguous entity-aligned chunks.

    Chunks never straddle a target entity (so each entity's shared prescreen
    tables are built once), and oversampling relative to ``workers`` lets the
    pool load-balance skewed candidate-set sizes dynamically.
    """
    boundaries = [0]
    for i in range(1, len(period_types)):
        if period_types[i].entity != period_types[i - 1].entity:
            boundaries.append(i)
    boundaries.append(len(period_types))
    n_groups = len(boundaries) - 1
    if n_groups == 0:
        return []
    n_chunks = min(n_groups, max(workers * 4, workers))
    tasks = []
    for c in range(n_chunks):
        g0 = c * n_groups // n_chunks
        g1 = (c + 1) * n_groups // n_chunks
        if g0 >= g1:
            continue
        tasks.append((len(tasks), boundaries[g0], boundaries[g1]))
    return tasks


def _merge_shards(spool_dir, ordered_results, period_types, state_layout):
    """Concatenate shard edge files in chunk order into one merged spool.

    Chunks are ascending contiguous target-signature ranges, so byte-appending
    them in order yields a single target-sorted archive that ``arrays()`` can
    turn into the compact CSR — identical to the single-process spool.
    """
    merged = _BinaryEdgeSpool(spool_dir, period_types, state_layout)
    for stream in merged.streams.values():
        stream.close()
    for name in merged.paths:
        with open(merged.paths[name], "wb") as out:
            for res in ordered_results:
                with open(res["paths"][name], "rb") as src:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
    merged.count = sum(res["count"] for res in ordered_results)
    return merged


def _parallel_init(plan, prepared, period_types, state_layout, spool_root):
    """Worker initializer for the ``spawn`` start method (e.g. Windows).

    ``spawn`` gives each worker a fresh interpreter that cannot inherit the
    parent's memory, so the plan and candidate index are shipped once per worker
    via pickled init args instead of copy-on-write.
    """
    global _PARALLEL_CTX
    _PARALLEL_CTX = {
        "plan": plan,
        "prepared": prepared,
        "period_types": period_types,
        "state_layout": state_layout,
        "spool_root": spool_root,
    }


def _run_parallel_compile(
    plan, prepared, period_types, state_layout, spool_dir, workers, progress_cb
):
    """Process-parallel target-dynamic compile; returns (type_pair_count, spool).

    Uses ``fork`` (copy-on-write, no pickling) where available and falls back to
    ``spawn`` on platforms without fork (Windows), shipping state to workers via
    a pickled initializer.
    """
    global _PARALLEL_CTX
    tasks = _entity_aligned_chunks(period_types, workers)
    use_fork = "fork" in mp.get_all_start_methods()
    ctx = mp.get_context("fork" if use_fork else "spawn")
    pool_kwargs = {"processes": min(workers, len(tasks)) or 1}
    if use_fork:
        # Warm the target-independent static table before forking so every
        # worker inherits it copy-on-write instead of rebuilding it per chunk.
        if prepared.get("adaptive") and prepared.get("entities"):
            plan.adaptive_static_table(prepared["entities"])
        _PARALLEL_CTX = {
            "plan": plan,
            "prepared": prepared,
            "period_types": period_types,
            "state_layout": state_layout,
            "spool_root": spool_dir,
        }
    else:
        # spawn: each worker gets the (picklable) plan/index via init args and
        # warms its own static table lazily on the first chunk.
        pool_kwargs["initializer"] = _parallel_init
        pool_kwargs["initargs"] = (
            plan,
            prepared,
            period_types,
            state_layout,
            spool_dir,
        )
    results = {}
    agg_pairs = agg_compiled = agg_pruned = agg_targets = 0
    try:
        with ctx.Pool(**pool_kwargs) as pool:
            for res in pool.imap_unordered(_compile_chunk, tasks):
                results[res["idx"]] = res
                agg_pairs += res["type_pair_count"]
                agg_compiled += res["compiled_pair_count"]
                agg_pruned += res["pruned_pair_count"]
                agg_targets += res["target_count"]
                if progress_cb is not None:
                    progress_cb(agg_pairs, agg_compiled, agg_pruned, agg_targets)
    finally:
        if use_fork:
            _PARALLEL_CTX = {}
    ordered = [results[i] for i in sorted(results)]
    plan.compiled_pair_count = sum(r["compiled_pair_count"] for r in ordered)
    plan.pruned_pair_count = sum(r["pruned_pair_count"] for r in ordered)
    plan.prescreen_dropped_pair_count = sum(
        r["prescreen_dropped_pair_count"] for r in ordered
    )
    type_pair_count = sum(r["type_pair_count"] for r in ordered)
    spool = _merge_shards(spool_dir, ordered, period_types, state_layout)
    return type_pair_count, spool


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
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel target-entity workers for the target-dynamic vectorized "
            "compiler (fork pool). 1 keeps the single-process path. Output is "
            "bit-identical regardless of worker count; only the target-dynamic "
            "path is parallelized (other layouts fall back to one process)."
        ),
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
    if args.workers < 1:
        parser.error("--workers must be >= 1")
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
    # The exact directed-pair total requires enumerating every target's
    # candidate set. That enumeration is the same work the compile pass already
    # does, so we skip it for the normal path (progress is driven by the known
    # target count instead) and only pay it for --count-only, whose sole job is
    # to report the total.
    prepared_candidates = plan.prepare_candidate_period_types(
        period_types, count_pairs=args.count_only
    )
    period_types = prepared_candidates["period_types"]
    total_type_pairs = prepared_candidates["total_pair_count"]
    total_targets = len(period_types)
    if not args.quiet:
        pairs_field = (
            f"total_type_pairs={total_type_pairs}, "
            f"max_state_edges={total_type_pairs * state_expansion}, "
            if total_type_pairs is not None
            else "total_type_pairs=counted-during-compile, "
        )
        print(
            f"[period-cache] {pairs_field}"
            f"target_types={total_targets}, "
            f"state_layout={state_layout}, "
            f"backend={compile_backend}, "
            f"states_per_type_pair={state_expansion}, "
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
        ProgressBar(total_targets, "编译 AlarmPeriod 关联缓存")
        if not args.quiet and args.progress_every
        else None
    )

    def update_progress(type_pairs, active_edges, pruned_pairs, targets_done):
        if progress_bar is None:
            return
        estimated_active_edges = (
            round(active_edges / targets_done * total_targets) if targets_done else 0
        )
        active_ratio = active_edges / max(active_edges + pruned_pairs, 1)
        progress_bar.extra_text = (
            f"pairs={type_pairs:,} edges={active_edges:,} ({active_ratio:.1%}, "
            f"est={estimated_active_edges:,})"
        )
        progress_bar.set(targets_done)

    def report(type_pairs, active_edges, pruned_pairs, targets_done):
        nonlocal last_report
        if args.quiet or not args.progress_every:
            return
        if type_pairs - last_report < args.progress_every:
            return
        last_report = type_pairs
        update_progress(type_pairs, active_edges, pruned_pairs, targets_done)

    use_parallel = args.workers > 1 and state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
    if args.workers > 1 and not use_parallel and not args.quiet:
        print(
            "[period-cache] --workers>1 仅支持 target-dynamic 向量化路径；回退单进程",
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix="alarm-period-cache-") as spool_dir:
        spool = None if use_parallel else _BinaryEdgeSpool(
            spool_dir, period_types, state_layout
        )
        arrays = None
        payload = None
        try:
            if use_parallel:
                type_pair_count, spool = _run_parallel_compile(
                    plan,
                    prepared_candidates,
                    period_types,
                    state_layout,
                    spool_dir,
                    args.workers,
                    None if progress_bar is None else update_progress,
                )
            else:
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
                    total_targets,
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
            if spool is not None:
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
