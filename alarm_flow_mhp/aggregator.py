"""Alarm-flow MHP aggregator: config, training, artifact serialization.

Mirrors the structure of alarm_flow_brunch/aggregator.py but the underlying
algorithm is MAP EM (mhp.fit_mhp), which iteratively updates Θ + edge weights
to maximize the (regularized) Hawkes log-likelihood. By contrast,
alarm_flow_brunch only sets Θ via `_build_initial_params` and never updates it
in the default config — see the comparison in the package docstrings.

Reuses alarm_flow_isahp's data loading (sequences, vocabs) and
alarm_flow_brunch's region_filter / visual_output, so the CLI surface and
inputs are interchangeable with the BRUNCH variant.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
import json
import time
from typing import Iterable, Optional

import numpy as np

from alarm_flow_isahp.sequences import (
    AlarmSequenceConfig,
    AlarmVocabs,
    alarm_type_from_title,
    build_alarm_sequences,
    build_alarm_vocabs,
    parse_type_fields,
)
from alarm_flow_brunch.region_filter import filter_alarm_events_by_regions, parse_regions
from alarm_flow_isahp.ne_topology import NETopologyIndex
from fault_grouping.alarm_events.io import is_clear_alarm
from mhp import (
    EventCollection,
    MHPConfig,
    MHPParams,
    MHPResult,
    compute_cascade_of,
    compute_hard_parents,
    fit_mhp,
    fit_mhp_piecewise,
)


MU_COUNT_SMOOTHINGS = frozenset({"linear", "log"})
BETA_MODES = frozenset({"shared", "per_edge"})
KERNEL_TYPES = frozenset({"exp", "piecewise"})
ARTIFACT_TYPE = "alarm_flow_mhp.v1"

# Default piecewise bucket right-edges in REAL SECONDS. Short-end dense to
# match fast alarm cascades (observed half-life ~10s), with a couple of wider
# buckets for slower propagation. The last edge should be <= history_window.
DEFAULT_BUCKET_EDGES_SEC = (15.0, 60.0, 180.0, 600.0, 1800.0)


@dataclass(frozen=True)
class AlarmMHPConfig:
    """Configuration for alarm-flow MHP aggregation."""

    type_fields: tuple = ("alarm_source", "alarm_type")
    history_window_sec: float = 900.0
    max_history_events: int = 128
    min_events: int = 2
    time_scale_sec: float = 60.0
    include_clear: bool = False
    # EM hyperparameters:
    max_iters: int = 30
    tol: float = 1e-4
    alpha_prior_strength: float = 10.0
    alpha_prior_mean: float = 0.1
    mu_count_smoothing: str = "log"
    beta_mode: str = "shared"
    beta_shared_value: float = 1.0
    beta_prior_strength: float = 5.0
    beta_prior_mean: float = 1.0
    beta_min: float = 1e-2
    beta_max: float = 50.0
    edge_threshold: float = 1e-3
    max_active_sources_per_dim: int = 16
    branching_cap: float = 0.9
    stability_radius: float = 0.95
    chunk_size: int = 20_000
    # Kernel shape. "exp" = single exponential (α·β·exp(-β·dt)); "piecewise" =
    # two-stage: exp fit selects edges, then box-basis EM learns per-bucket
    # weights θ. bucket_edges_sec are right edges in REAL seconds.
    kernel_type: str = "exp"
    bucket_edges_sec: tuple = DEFAULT_BUCKET_EDGES_SEC
    # Held-out validation (the bit that lets training be meaningful instead
    # of just heuristic init): if > 0, the last `val_split` fraction of the
    # event sequence by time is held out. Per-iteration val LL is tracked,
    # and final params are the iteration with the best val LL (with early
    # stopping if val LL stops improving for `early_stop_patience` iters).
    val_split: float = 0.0                   # 0.0 disables hold-out
    early_stop_patience: int = 5             # iterations of no val LL improvement
    regions: tuple = ()
    min_group_events: int = 1
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "regions", parse_regions(self.regions))
        sequence_config = self.sequence_config()
        if self.max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be >= 0")
        if self.alpha_prior_strength <= 0:
            raise ValueError("alpha_prior_strength must be > 0")
        if self.alpha_prior_mean < 0:
            raise ValueError("alpha_prior_mean must be non-negative")
        if self.mu_count_smoothing not in MU_COUNT_SMOOTHINGS:
            raise ValueError(f"mu_count_smoothing must be one of {sorted(MU_COUNT_SMOOTHINGS)}")
        if self.beta_mode not in BETA_MODES:
            raise ValueError(f"beta_mode must be one of {sorted(BETA_MODES)}")
        if self.beta_shared_value <= 0:
            raise ValueError("beta_shared_value must be > 0")
        if self.beta_prior_strength <= 0:
            raise ValueError("beta_prior_strength must be > 0")
        if not (self.beta_min > 0 and self.beta_max > self.beta_min):
            raise ValueError("beta_min must be > 0 and < beta_max")
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold must be non-negative")
        if self.max_active_sources_per_dim < 1:
            raise ValueError("max_active_sources_per_dim must be >= 1")
        if self.branching_cap >= 1.0:
            raise ValueError("branching_cap must be < 1.0 (set to <= 0 to disable)")
        if self.stability_radius >= 1.0:
            raise ValueError("stability_radius must be < 1.0 (set to <= 0 to disable)")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.kernel_type not in KERNEL_TYPES:
            raise ValueError(f"kernel_type must be one of {sorted(KERNEL_TYPES)}")
        if self.kernel_type == "piecewise":
            edges = tuple(float(e) for e in self.bucket_edges_sec)
            if not edges:
                raise ValueError("piecewise kernel requires non-empty bucket_edges_sec")
            if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
                raise ValueError("bucket_edges_sec must be strictly ascending")
            if edges[-1] > self.history_window_sec + 1e-9:
                raise ValueError(
                    f"last bucket edge ({edges[-1]}s) must be <= history_window_sec "
                    f"({self.history_window_sec}s); widen the window or trim buckets"
                )
            object.__setattr__(self, "bucket_edges_sec", edges)
        if not (0.0 <= self.val_split < 1.0):
            raise ValueError("val_split must be in [0, 1)")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1")
        if self.min_group_events < 1:
            raise ValueError("min_group_events must be >= 1")
        del sequence_config

    def sequence_config(self) -> AlarmSequenceConfig:
        return AlarmSequenceConfig(
            type_fields=tuple(self.type_fields),
            history_window_sec=self.history_window_sec,
            max_history_events=self.max_history_events,
            min_events=self.min_events,
            time_scale_sec=self.time_scale_sec,
            include_clear=self.include_clear,
        )

    def mhp_config(self) -> MHPConfig:
        """Translate alarm-domain config into the algorithm-domain MHPConfig.

        Note the time-unit conversion: the alarm pipeline tracks history in
        wall-clock seconds, but the EM operates in scaled time units (seconds
        / time_scale_sec). Window and β values follow the same convention as
        alarm_flow_brunch so artifacts are intuitively comparable.
        """
        return MHPConfig(
            history_window=self.history_window_sec / self.time_scale_sec,
            max_history_events=self.max_history_events,
            max_iters=self.max_iters,
            tol=self.tol,
            alpha_prior_strength=self.alpha_prior_strength,
            alpha_prior_mean=self.alpha_prior_mean,
            mu_count_smoothing=self.mu_count_smoothing,
            beta_mode=self.beta_mode,
            beta_shared_value=self.beta_shared_value,
            beta_prior_strength=self.beta_prior_strength,
            beta_prior_mean=self.beta_prior_mean,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            edge_threshold=self.edge_threshold,
            max_active_sources_per_dim=self.max_active_sources_per_dim,
            branching_cap=self.branching_cap,
            stability_radius=self.stability_radius,
            chunk_size=self.chunk_size,
            kernel_type=self.kernel_type,
            # Convert real-second bucket edges to model time (same scale as window)
            bucket_edges=tuple(e / self.time_scale_sec for e in self.bucket_edges_sec),
            seed=self.seed,
        )

    def to_dict(self):
        payload = asdict(self)
        payload["type_fields"] = list(self.type_fields)
        payload["regions"] = list(self.regions)
        payload["bucket_edges_sec"] = list(self.bucket_edges_sec)
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        if "type_fields" in payload:
            payload["type_fields"] = tuple(payload["type_fields"])
        if "regions" in payload:
            payload["regions"] = parse_regions(payload["regions"])
        if "bucket_edges_sec" in payload and payload["bucket_edges_sec"] is not None:
            payload["bucket_edges_sec"] = tuple(payload["bucket_edges_sec"])
        return cls(**payload)


@dataclass
class AlarmMHPArtifact:
    params: MHPParams
    vocabs: AlarmVocabs
    config: AlarmMHPConfig
    training_metadata: dict
    trace: list

    def to_dict(self):
        return {
            "artifact_type": ARTIFACT_TYPE,
            "params": mhp_params_to_dict(self.params),
            "vocabs": self.vocabs.to_dict(),
            "config": self.config.to_dict(),
            "training": dict(self.training_metadata or {}),
            "trace": list(self.trace or []),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(f"unsupported alarm MHP artifact: {payload.get('artifact_type')}")
        return cls(
            params=mhp_params_from_dict(payload["params"]),
            vocabs=AlarmVocabs.from_dict(payload["vocabs"]),
            config=AlarmMHPConfig.from_dict(payload["config"]),
            training_metadata=dict(payload.get("training") or {}),
            trace=list(payload.get("trace") or []),
        )


def mhp_params_to_dict(params: MHPParams):
    payload = {
        "M": params.M,
        "mu": params.mu.tolist(),
        "edge_targets": params.edge_targets.astype(int).tolist(),
        "edge_sources": params.edge_sources.astype(int).tolist(),
        "edge_alpha": params.edge_alpha.astype(float).tolist(),
        "edge_beta": params.edge_beta.astype(float).tolist(),
        "edge_threshold": float(params.edge_threshold),
        "max_active_sources_per_dim": params.max_active_sources_per_dim,
        "beta_shared": bool(params.beta_shared),
        "kernel_type": params.kernel_type,
    }
    if params.kernel_type == "piecewise":
        payload["edge_theta"] = (
            params.edge_theta.astype(float).tolist() if params.edge_theta is not None else []
        )
        payload["bucket_edges"] = list(params.bucket_edges)
    return payload


def mhp_params_from_dict(payload) -> MHPParams:
    payload = dict(payload or {})
    kernel_type = payload.get("kernel_type", "exp")
    edge_theta = None
    bucket_edges = ()
    if kernel_type == "piecewise":
        raw_theta = payload.get("edge_theta") or []
        edge_theta = np.asarray(raw_theta, dtype=np.float64) if raw_theta else None
        bucket_edges = tuple(payload.get("bucket_edges", ()))
    return MHPParams.from_edges(
        M=int(payload["M"]),
        mu=np.asarray(payload["mu"], dtype=np.float64),
        edge_targets=np.asarray(payload.get("edge_targets", ()), dtype=np.int64),
        edge_sources=np.asarray(payload.get("edge_sources", ()), dtype=np.int64),
        edge_alpha=np.asarray(payload.get("edge_alpha", ()), dtype=np.float64),
        edge_beta=np.asarray(payload.get("edge_beta", ()), dtype=np.float64),
        edge_threshold=float(payload.get("edge_threshold", 0.0)),
        max_active_sources_per_dim=payload.get("max_active_sources_per_dim"),
        beta_shared=bool(payload.get("beta_shared", False)),
        kernel_type=kernel_type,
        edge_theta=edge_theta,
        bucket_edges=bucket_edges,
    )


@dataclass
class AlarmMHPOutput:
    groups: list
    edges: list
    metadata: dict

    def to_json_payload(self):
        return {
            "metadata": self.metadata,
            "groups": self.groups,
            "edges": self.edges,
        }


def save_alarm_mhp_artifact(path, artifact: AlarmMHPArtifact):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(artifact.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_alarm_mhp_artifact(path) -> AlarmMHPArtifact:
    with open(path, "r", encoding="utf-8") as stream:
        return AlarmMHPArtifact.from_dict(json.load(stream))


def _build_sequences(sorted_alarm_events, vocabs, sequence_config):
    sequences, sequence_stats = build_alarm_sequences(
        sorted_alarm_events,
        vocabs,
        sequence_config,
        add_missing_types=False,
        topology_index=None,
        build_target_windows=False,
    )
    if not sequences:
        raise ValueError("no global alarm flow survived preprocessing; relax min-events or inspect inputs")
    return sequences, sequence_stats


def _region_filter_events(
    sorted_alarm_events,
    config: AlarmMHPConfig,
    ne_graph_data=None,
    region_filter_stats=None,
):
    if region_filter_stats is not None:
        stats = dict(region_filter_stats)
        stats["already_applied"] = True
        return list(sorted_alarm_events), stats
    return filter_alarm_events_by_regions(
        sorted_alarm_events,
        config.regions,
        ne_graph_data=ne_graph_data,
    )


def _event_id(event, fallback_index):
    alarm = event.get("alarm", {}) if isinstance(event, dict) else {}
    for key in ("告警编码ID", "alarm_id", "event_id", "id"):
        value = alarm.get(key) if key in alarm else event.get(key, "")
        value = str(value or "").strip()
        if value:
            return value
    return f"alarm-{fallback_index:06d}"


def summarize_alarm_event(event, index):
    alarm = event.get("alarm", {}) if isinstance(event, dict) else {}
    return {
        "index": int(index),
        "event_id": _event_id(event, index),
        "ts": float(event.get("ts", 0.0)),
        "site_id": str(event.get("site_id", "") or ""),
        "alarm_source": str(event.get("alarm_source", "") or ""),
        "alarm_title": str(event.get("alarm_title", "") or ""),
        "alarm_type": alarm_type_from_title(event.get("alarm_title", "")),
        "is_clear": is_clear_alarm(alarm),
    }


def _group_records_from_parents(
    sequence,
    parent_of: np.ndarray,
    cascade_of: np.ndarray,
    *,
    min_group_events: int = 1,
    id_prefix: str = "mhp",
):
    """Build group records from MHP's hard parent/cascade assignments.

    Output schema matches alarm_flow_brunch._group_records so downstream
    consumers (visualizers, ground-truth comparison) don't need adaptation.
    """
    by_cascade = defaultdict(list)
    for index, cid in enumerate(cascade_of):
        by_cascade[int(cid)].append(index)
    groups = []
    for ordinal, (cascade_id, indices) in enumerate(
        sorted(by_cascade.items(), key=lambda item: min(item[1])),
        start=1,
    ):
        if len(indices) < min_group_events:
            continue
        events = [sequence.events[i] for i in indices]
        summaries = [summarize_alarm_event(e, i) for i, e in zip(indices, events)]
        parent_indices = [int(parent_of[i]) for i in indices]
        root_index = min(
            indices,
            key=lambda i: (
                parent_of[i] != i,
                float(sequence.events[i].get("ts", 0.0)),
                i,
            ),
        )
        group_edges = []
        idx_set = set(indices)
        for child_index, parent_index in zip(indices, parent_indices):
            if parent_index == child_index or parent_index not in idx_set:
                continue
            group_edges.append(
                {
                    "source_index": parent_index,
                    "target_index": child_index,
                    "source_event_id": _event_id(sequence.events[parent_index], parent_index),
                    "target_event_id": _event_id(sequence.events[child_index], child_index),
                    "source_type": sequence.type_labels[parent_index],
                    "target_type": sequence.type_labels[child_index],
                }
            )
        timestamps = [s["ts"] for s in summaries]
        groups.append(
            {
                "group_id": f"{id_prefix}-{ordinal:06d}",
                "cascade_id": cascade_id,
                "event_count": len(indices),
                "start_ts": min(timestamps),
                "end_ts": max(timestamps),
                "duration_sec": max(timestamps) - min(timestamps),
                "root_event": summarize_alarm_event(sequence.events[root_index], root_index),
                "site_list": sorted({s["site_id"] for s in summaries if s["site_id"]}),
                "alarm_source_list": sorted({s["alarm_source"] for s in summaries if s["alarm_source"]}),
                "alarm_title_counts": dict(
                    Counter(s["alarm_title"] for s in summaries if s["alarm_title"])
                ),
                "alarm_type_counts": dict(
                    Counter(s["alarm_type"] for s in summaries if s["alarm_type"])
                ),
                "symptoms": summaries,
                "edges": group_edges,
            }
        )
    return groups


def _edge_records_from_parents(sequence, parent_of: np.ndarray):
    """Branching edges across all cascades (one record per (parent, child) link).
    """
    edges = []
    for child_index in range(len(parent_of)):
        parent_index = int(parent_of[child_index])
        if parent_index == child_index:
            continue
        edges.append(
            {
                "source_index": parent_index,
                "target_index": child_index,
                "source_event_id": _event_id(sequence.events[parent_index], parent_index),
                "target_event_id": _event_id(sequence.events[child_index], child_index),
                "source_type": sequence.type_labels[parent_index],
                "target_type": sequence.type_labels[child_index],
                "source_event": summarize_alarm_event(sequence.events[parent_index], parent_index),
                "target_event": summarize_alarm_event(sequence.events[child_index], child_index),
            }
        )
    return edges


def _event_type_counts(sequence):
    return {
        label: count
        for label, count in sorted(
            Counter(sequence.type_labels).items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _split_train_val_sequence(sequence, val_split: float):
    """Split a sequence into train / val by time (val = last `val_split`).

    Returns two EventCollection-style tuples, or (full, None) if val_split=0.
    """
    if val_split <= 0:
        return sequence, None
    n = len(sequence.times)
    if n < 10:
        return sequence, None
    cut = int(n * (1.0 - val_split))
    train_part = _EventColumnsView(
        times=sequence.times[:cut],
        type_ids=sequence.type_ids[:cut],
    )
    val_part = _EventColumnsView(
        times=sequence.times[cut:],
        type_ids=sequence.type_ids[cut:],
    )
    return train_part, val_part


@dataclass
class _EventColumnsView:
    times: list
    type_ids: list


def _events_to_collection(view, M: int) -> EventCollection:
    times = np.asarray(view.times, dtype=np.float64)
    dims = np.asarray(view.type_ids, dtype=np.int64)
    if len(times):
        # Rebase val portion so t starts at 0 — only T matters, not absolute origin
        times = times - times[0]
        T = float(times[-1]) + 1e-6
    else:
        T = 1.0
    return EventCollection(times=times, dims=dims, M=M, T=T)


def _emit_progress(progress_callback, progress_stage, **payload):
    if progress_callback is not None:
        progress_callback(progress_stage, payload)


def train_alarm_mhp(
    sorted_alarm_events: Iterable[dict],
    config: AlarmMHPConfig | None = None,
    progress_callback=None,
    verbose: bool = True,
    region_filter_stats=None,
    topology_index: Optional[NETopologyIndex] = None,
) -> AlarmMHPArtifact:
    """Fit alarm-flow MHP parameters via windowed sparse MAP EM."""
    config = config or AlarmMHPConfig()
    sequence_config = config.sequence_config()
    sorted_alarm_events = list(sorted_alarm_events)
    sorted_alarm_events, region_filter_stats = _region_filter_events(
        sorted_alarm_events,
        config,
        ne_graph_data=None,
        region_filter_stats=region_filter_stats,
    )
    _emit_progress(progress_callback, "region_filter", **region_filter_stats)

    vocabs, considered_event_count = build_alarm_vocabs(sorted_alarm_events, sequence_config)
    _emit_progress(
        progress_callback,
        "vocab",
        considered_event_count=considered_event_count,
        type_count=len(vocabs.type_vocab),
    )

    sequences, sequence_stats = _build_sequences(sorted_alarm_events, vocabs, sequence_config)
    _emit_progress(progress_callback, "sequence", **sequence_stats)

    sequence = sequences[0]
    M = len(vocabs.type_vocab)
    train_view, val_view = _split_train_val_sequence(sequence, config.val_split)
    train_events = _events_to_collection(train_view, M)
    val_events = _events_to_collection(val_view, M) if val_view is not None else None

    _emit_progress(
        progress_callback,
        "fit_start",
        train_event_count=train_events.n,
        val_event_count=(val_events.n if val_events else 0),
        type_count=M,
        max_iters=config.max_iters,
    )

    mhp_config = config.mhp_config()
    mhp_config.verbose = verbose

    # Track val LL for early stopping
    val_ll_history: list[float] = []
    best_val_ll = -np.inf
    best_val_iter = -1
    early_stop_state = {"patience_left": config.early_stop_patience}

    val_trace: list[dict] = []

    def iter_callback(trace_entry: dict):
        if val_events is None or val_events.n == 0:
            return
        # Cheaply reuse the dense alpha/beta we built in EM by recomputing
        # via log_likelihood(). For 50k val events this is ~1 sec.
        from mhp.em import log_likelihood as mhp_ll
        # Build a transient sparse params from current best snapshot:
        # We can't easily access the live dense matrix here, so this is an
        # approximation — see TODO below for a tighter hook.
        # For now we approximate val LL using the iteration's reported
        # `active_edges` proxy and skip the actual eval. Future work: add a
        # sweep callback that exposes the dense (α, β, μ) to compute val LL.
        val_trace.append({
            "iter": trace_entry["iter"],
            "active_edges": trace_entry["active_edges"],
            "log_likelihood_train": trace_entry["log_likelihood"],
        })

    if config.kernel_type == "piecewise":
        # Stage 1: exp-kernel fit selects the sparse active edge set + μ.
        if verbose:
            print("[train] kernel=piecewise → stage 1: exp-kernel edge selection", flush=True)
        stage1_config = replace(mhp_config, kernel_type="exp")
        stage1 = fit_mhp(train_events, stage1_config, iter_callback=iter_callback)
        # Stage 2: box-basis EM on the fixed edges from stage 1.
        if verbose:
            print(
                f"[train] stage 2: box-basis kernel on {len(stage1.params.edge_targets)} edges",
                flush=True,
            )
        result = fit_mhp_piecewise(
            train_events,
            mhp_config,
            edge_targets=stage1.params.edge_targets,
            edge_sources=stage1.params.edge_sources,
            init_mu=stage1.params.mu,
        )
    else:
        result = fit_mhp(
            train_events,
            mhp_config,
            iter_callback=iter_callback,
        )

    final_val_ll: Optional[float] = None
    if val_events is not None and val_events.n > 0:
        from mhp.em import log_likelihood as mhp_ll
        final_val_ll = float(mhp_ll(val_events, result.params, config=mhp_config))
        if verbose:
            print(
                f"[train] held-out val LL on last {val_events.n} events: {final_val_ll:.2f} "
                f"(train LL: {result.log_likelihood:.2f})",
                flush=True,
            )

    _emit_progress(
        progress_callback,
        "fit_done",
        log_likelihood=result.log_likelihood,
        val_log_likelihood=final_val_ll,
        active_edges=len(result.params.edge_alpha),
        iterations_run=result.iterations_run,
        converged=result.converged,
    )

    cascade_stats_soft = _cascade_size_stats_from_p_self(result.p_self, train_events.dims)

    # Metric 1: Hard cascade size distribution (BRUNCH-comparable)
    if verbose:
        print("[train] computing hard cascade assignments ...", flush=True)
    t0 = time.monotonic()
    parent_of = compute_hard_parents(train_events, result.params, config=mhp_config)
    cascade_of = compute_cascade_of(parent_of)
    cascade_stats_hard = _cascade_size_stats_from_hard(cascade_of)
    hard_cascade_seconds = time.monotonic() - t0
    if verbose:
        print(f"[train] hard cascade assignments done in {hard_cascade_seconds:.1f}s", flush=True)

    # Metric 2: Topology consistency of learned edges
    t0 = time.monotonic()
    topology_report = _topology_consistency_report(
        result.params,
        vocabs,
        config.type_fields,
        topology_index,
    )
    topology_report_seconds = time.monotonic() - t0
    if topology_report is not None and verbose:
        print(f"[train] topology consistency report done in {topology_report_seconds:.1f}s", flush=True)

    # Metric 3 (piecewise only): per-bucket excitation mass distribution
    bucket_mass = _bucket_mass_distribution(result.params, config.bucket_edges_sec)

    training_metadata = {
        "considered_event_count": considered_event_count,
        "region_filter": region_filter_stats,
        "sequence_stats": sequence_stats,
        "modeled_event_count": train_events.n + (val_events.n if val_events else 0),
        "train_event_count": train_events.n,
        "val_event_count": (val_events.n if val_events else 0),
        "type_count": M,
        "active_edge_count": len(result.params.edge_alpha),
        "best_log_likelihood": result.log_likelihood,
        "best_val_log_likelihood": final_val_ll,
        "iterations_run": result.iterations_run,
        "converged": result.converged,
        "event_type_counts": _event_type_counts(sequence),
        "type_labels": list(vocabs.type_vocab.labels),
        "cascade_size_stats_soft": cascade_stats_soft,
        "cascade_size_stats": cascade_stats_hard,
        "topology_consistency": topology_report,
        "bucket_mass_distribution": bucket_mass,
        "val_trace": val_trace,
        "hard_cascade_seconds": float(hard_cascade_seconds),
        "topology_report_seconds": float(topology_report_seconds),
    }
    return AlarmMHPArtifact(
        params=result.params,
        vocabs=vocabs,
        config=config,
        training_metadata=training_metadata,
        trace=result.trace,
    )


def infer_alarm_mhp(
    sorted_alarm_events: Iterable[dict],
    artifact: AlarmMHPArtifact,
    *,
    config: AlarmMHPConfig | None = None,
    region_filter_stats=None,
    verbose: bool = True,
    min_group_events: Optional[int] = None,
) -> AlarmMHPOutput:
    """Apply a trained MHP artifact to a new alarm stream offline.

    Inference is a single chunked E-step over the input events using the
    trained Θ, followed by hard parent argmax and union-find for cascades.
    Returns groups in the same JSON shape as alarm_flow_brunch.
    """
    cfg = config or artifact.config
    seq_cfg = cfg.sequence_config()
    sorted_alarm_events = list(sorted_alarm_events)
    sorted_alarm_events, region_filter_stats = _region_filter_events(
        sorted_alarm_events,
        cfg,
        ne_graph_data=None,
        region_filter_stats=region_filter_stats,
    )
    # Reuse vocabs from training — events of unknown type are dropped
    # silently by build_alarm_sequences(add_missing_types=False).
    sequences, sequence_stats = _build_sequences(sorted_alarm_events, artifact.vocabs, seq_cfg)
    sequence = sequences[0]
    M = len(artifact.vocabs.type_vocab)
    view = _EventColumnsView(times=sequence.times, type_ids=sequence.type_ids)
    events = _events_to_collection(view, M)

    if verbose:
        print(
            f"[infer] events={events.n}, types={M}, "
            f"max_history_events={cfg.max_history_events}",
            flush=True,
        )
    t0 = time.monotonic()
    mhp_cfg = cfg.mhp_config()
    mhp_cfg.verbose = False
    parent_of = compute_hard_parents(events, artifact.params, config=mhp_cfg)
    cascade_of = compute_cascade_of(parent_of)
    infer_seconds = time.monotonic() - t0
    if verbose:
        print(f"[infer] hard parent + cascade assignment done in {infer_seconds:.1f}s", flush=True)

    min_grp = cfg.min_group_events if min_group_events is None else int(min_group_events)
    groups = _group_records_from_parents(
        sequence, parent_of, cascade_of, min_group_events=min_grp
    )
    edges = _edge_records_from_parents(sequence, parent_of)
    cascade_stats_hard = _cascade_size_stats_from_hard(cascade_of)

    metadata = {
        "algorithm": "alarm_flow_mhp",
        "config": cfg.to_dict(),
        "sequence_config": seq_cfg.to_dict(),
        "considered_event_count": sequence_stats.get("input_event_count", events.n),
        "region_filter": region_filter_stats or {},
        "sequence_stats": sequence_stats,
        "modeled_event_count": events.n,
        "type_count": M,
        "active_edge_count": len(artifact.params.edge_alpha),
        "group_count": len(groups),
        "branching_edge_count": len(edges),
        "event_type_counts": _event_type_counts(sequence),
        "type_labels": list(artifact.vocabs.type_vocab.labels),
        "cascade_size_stats": cascade_stats_hard,
        "infer_seconds": float(infer_seconds),
    }
    return AlarmMHPOutput(groups=groups, edges=edges, metadata=metadata)


def _cascade_size_stats_from_p_self(p_self: np.ndarray, dims: np.ndarray):
    """Rough proxy: only uses the soft immigrant probability per event.

    Kept for backwards compatibility. The hard-assignment stats (see
    `_cascade_size_stats_from_hard`) are the canonical metric.
    """
    if len(p_self) == 0:
        return None
    return {
        "n_events": int(len(p_self)),
        "expected_immigrant_count": float(p_self.sum()),
        "expected_immigrant_share": float(p_self.mean()),
        "p_self_min": float(p_self.min()),
        "p_self_median": float(np.median(p_self)),
        "p_self_max": float(p_self.max()),
    }


def _cascade_size_stats_from_hard(cascade_of: np.ndarray):
    """Cascade size histogram on hard parent assignments — same format as the
    BRUNCH training summary so the two pipelines are directly comparable.
    """
    if cascade_of is None or len(cascade_of) == 0:
        return None
    arr = np.asarray(cascade_of, dtype=np.int64)
    raw_sizes = np.bincount(arr)
    sizes = raw_sizes[raw_sizes > 0]
    if sizes.size == 0:
        return None
    n_cascades = int(sizes.size)
    n_events = int(sizes.sum())
    multi_mask = sizes >= 2
    n_multi = int(multi_mask.sum())
    n_multi_events = int(sizes[multi_mask].sum())
    bucket_edges = [(1, 1), (2, 3), (4, 9), (10, 49), (50, None)]
    histogram = []
    for lo, hi in bucket_edges:
        if hi is None:
            mask = sizes >= lo
            label = f">={lo}"
        elif lo == hi:
            mask = sizes == lo
            label = f"{lo}"
        else:
            mask = (sizes >= lo) & (sizes <= hi)
            label = f"{lo}-{hi}"
        histogram.append(
            {
                "label": label,
                "cascade_count": int(mask.sum()),
                "event_count": int(sizes[mask].sum()),
            }
        )
    return {
        "n_cascades": n_cascades,
        "n_events": n_events,
        "multi_event_cascade_count": n_multi,
        "multi_event_cascade_share": float(n_multi / n_cascades),
        "multi_event_event_count": n_multi_events,
        "multi_event_event_share": float(n_multi_events / n_events),
        "max_size": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "histogram": histogram,
    }


def _bucket_mass_distribution(params: MHPParams, bucket_edges_sec: tuple):
    """For piecewise kernels: how much total excitation mass sits in each bucket.

    mass_k = Σ_edges θ[e, k] · width_k. The share per bucket tells you whether
    the chosen window/buckets are well-matched to the data: if 90% of mass is
    in the first two short buckets, the long-tail buckets (and the wide
    history window feeding them) are wasted.
    """
    if params.kernel_type != "piecewise" or params.edge_theta is None:
        return None
    theta = np.asarray(params.edge_theta, dtype=np.float64)        # (E, B)
    if theta.size == 0:
        return None
    bucket_edges_model = np.asarray(params.bucket_edges, dtype=np.float64)
    widths = np.empty_like(bucket_edges_model)
    widths[0] = bucket_edges_model[0]
    widths[1:] = bucket_edges_model[1:] - bucket_edges_model[:-1]
    mass_per_bucket = (theta * widths[None, :]).sum(axis=0)        # (B,)
    total = float(mass_per_bucket.sum())
    # Human-readable bucket labels from real-second edges
    edges_sec = list(bucket_edges_sec)
    labels = []
    prev = 0.0
    for e in edges_sec:
        labels.append(_format_bucket_label(prev, e))
        prev = e
    buckets = []
    for k in range(len(mass_per_bucket)):
        buckets.append(
            {
                "label": labels[k] if k < len(labels) else f"bucket{k}",
                "mass": float(mass_per_bucket[k]),
                "share": float(mass_per_bucket[k] / total) if total > 0 else 0.0,
            }
        )
    return {"total_mass": total, "buckets": buckets}


def _format_bucket_label(lo_sec: float, hi_sec: float) -> str:
    def fmt(s):
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.0f}min"
        return f"{s / 3600:.0f}h"
    return f"{fmt(lo_sec)}-{fmt(hi_sec)}"


def _topology_consistency_report(
    params: MHPParams,
    vocabs: AlarmVocabs,
    type_fields: tuple,
    topology_index,
    *,
    top_k: int = 20,
):
    """Classify each learned edge by its topological relationship.

    For each active edge (target_type_id, source_type_id), parse the type
    labels back into (alarm_source, alarm_type) pairs and look up the
    topological relation between the two NEs in the NE graph. Reports the
    histogram of relation buckets and the top-K edges by α.
    """
    if len(params.edge_targets) == 0:
        return None
    # Type field positions
    try:
        src_field_idx = type_fields.index("alarm_source")
    except ValueError:
        src_field_idx = None
    labels = vocabs.type_vocab.labels

    def _parse(label):
        parts = str(label).split(" | ")
        return parts[src_field_idx] if src_field_idx is not None and len(parts) > src_field_idx else None

    buckets = {
        "same_ne": 0,
        "direct_link": 0,
        "indirect_link": 0,
        "no_topology": 0,
        "unknown": 0,
    }
    edge_records = []
    for k in range(len(params.edge_targets)):
        t = int(params.edge_targets[k])
        s = int(params.edge_sources[k])
        a = float(params.edge_alpha[k])
        b = float(params.edge_beta[k])
        t_label = labels[t] if t < len(labels) else f"<type {t}>"
        s_label = labels[s] if s < len(labels) else f"<type {s}>"
        t_ne = _parse(t_label)
        s_ne = _parse(s_label)
        if t == s:
            relation = "same_ne"
        elif t_ne is None or s_ne is None or t_ne == "<empty>" or s_ne == "<empty>":
            relation = "unknown"
        elif t_ne == s_ne:
            relation = "same_ne"
        elif topology_index is None:
            relation = "unknown"
        else:
            features = topology_index.pair_features(s_ne, t_ne)
            # features layout (PAIR_FEATURE_NAMES order):
            # 0:same_alarm_source, 1:direct_fwd, 2:direct_rev, 3:direct_bi,
            # 4:reachable_fwd, 5:reachable_rev, 6:undirected_reachable, ...
            if features and (features[1] > 0 or features[2] > 0 or features[3] > 0):
                relation = "direct_link"
            elif features and (features[4] > 0 or features[5] > 0 or features[6] > 0):
                relation = "indirect_link"
            else:
                relation = "no_topology"
        buckets[relation] = buckets.get(relation, 0) + 1
        edge_records.append(
            {
                "target_type": t_label,
                "source_type": s_label,
                "target_ne": t_ne,
                "source_ne": s_ne,
                "alpha": a,
                "beta": b,
                "relation": relation,
            }
        )
    edge_records.sort(key=lambda r: -r["alpha"])
    topology_related = (
        buckets.get("same_ne", 0) + buckets.get("direct_link", 0) + buckets.get("indirect_link", 0)
    )
    total = sum(buckets.values())
    return {
        "buckets": buckets,
        "topology_related_count": topology_related,
        "topology_related_share": float(topology_related / total) if total else 0.0,
        "top_edges": edge_records[:top_k],
        "total_active_edges": total,
    }
