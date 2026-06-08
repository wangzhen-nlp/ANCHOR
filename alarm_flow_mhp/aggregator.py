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
import os
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
    fit_mhp_feature,
    fit_mhp_piecewise,
)


MU_COUNT_SMOOTHINGS = frozenset({"linear", "log"})
BETA_MODES = frozenset({"shared", "per_edge"})
KERNEL_TYPES = frozenset({"exp", "piecewise"})
EDGE_MODES = frozenset({"device", "feature"})
# Dynamic (stateful) α features — condition excitation on the devices' current
# uncleared-alarm state (link/power/offline), snapshotted at the source event's
# fire time (train/infer-consistent, clear-aware). feature mode only.
#   off           — static features only (current behavior)
#   source        — +3 booleans: source device's uncleared state at t_j
#                   (exact in both occurrence and compensator terms)
#   source_target — +6 booleans: source state plus target device pre-state.
#                   Training uses the B-fast approximation: E-step reads the
#                   target event's pre-state (time-slack safe); compensator
#                   buckets target state sampled at source_ts.
DYNAMIC_ALPHA_MODES = frozenset({"off", "source", "source_target"})
ARTIFACT_TYPE = "alarm_flow_mhp.v1"

# Default piecewise bucket right-edges in REAL SECONDS. Short-end dense to
# match fast alarm cascades (observed half-life ~10s), with a couple of wider
# buckets for slower propagation. The last edge should be <= history_window.
DEFAULT_BUCKET_EDGES_SEC = (15.0, 60.0, 180.0, 600.0, 1800.0)


def _fmt_secs(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m{rem:04.1f}s"


@dataclass(frozen=True)
class AlarmMHPConfig:
    """Configuration for alarm-flow MHP aggregation."""

    type_fields: tuple = ("alarm_source", "alarm_type")
    history_window_sec: float = 900.0
    # Timestamp jitter tolerance used by training and inherited by stream
    # inference. A candidate parent can be up to this many seconds later than
    # the target; negative dt is clamped to the kernel peak and discounted by
    # late_penalty_half_life_sec.
    time_slack_sec: float = 0.0
    late_penalty_half_life_sec: float = 1.0
    max_history_events: int = 128
    min_events: int = 2
    time_scale_sec: float = 60.0
    include_clear: bool = False
    # EM hyperparameters:
    max_iters: int = 30
    tol: float = 1e-4
    alpha_prior_strength: float = 10.0
    alpha_prior_mean: float = 0.1
    # Topology prior: inject extra MAP prior mass on topologically-related
    # (target, source) type pairs so rare/zero-co-occurrence but physically
    # connected device pairs still form (or strengthen) an edge. boost=0
    # disables (pure data-driven). Needs the NE graph at train time.
    topology_prior_boost: float = 0.0
    topology_prior_max_hops: int = 1         # 1 = same-NE + direct links only
    topology_prior_min_score: float = 0.6    # drop weak (far multi-hop) relations
    # Edge model: "device" = free per-(device-type) α (default, transductive);
    # "feature" = α = softplus(w·φ) learned over device-agnostic pair features
    # (inductive — generalizes to unseen pairs). feature mode needs the NE graph.
    edge_mode: str = "device"
    feature_l2: float = 1e-3                  # ridge on feature weights
    feature_topo_max_hops: int = 2            # candidate topology reach (feature mode)
    feature_topo_min_score: float = 0.0       # candidate topology score floor
    # Topology PRIOR for feature mode (device-parity): pseudo-count prior that
    # injects α≈boost·score on topology-related candidate edges, strongest where
    # data is sparse, washed out where data is rich. 0 = pure MLE (no prior).
    feature_topo_prior_boost: float = 0.0
    # Dynamic stateful α features (feature mode). See DYNAMIC_ALPHA_MODES.
    # Needs clear events in the input stream to track uncleared-alarm state.
    dynamic_alpha: str = "off"
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
        if self.time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if self.late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        if self.time_slack_sec > 0 and self.beta_mode == "per_edge":
            raise ValueError(
                "time_slack_sec > 0 currently requires beta_mode='shared'; "
                "per_edge beta needs a coupled beta/exposure M-step"
            )
        if self.max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be >= 0")
        if self.alpha_prior_strength <= 0:
            raise ValueError("alpha_prior_strength must be > 0")
        if self.alpha_prior_mean < 0:
            raise ValueError("alpha_prior_mean must be non-negative")
        if self.topology_prior_boost < 0:
            raise ValueError("topology_prior_boost must be non-negative")
        if self.topology_prior_max_hops < 1:
            raise ValueError("topology_prior_max_hops must be >= 1")
        if not (0.0 <= self.topology_prior_min_score <= 1.0):
            raise ValueError("topology_prior_min_score must be in [0, 1]")
        if self.edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {sorted(EDGE_MODES)}")
        if self.feature_l2 < 0:
            raise ValueError("feature_l2 must be non-negative")
        if self.feature_topo_max_hops < 1:
            raise ValueError("feature_topo_max_hops must be >= 1")
        if not (0.0 <= self.feature_topo_min_score <= 1.0):
            raise ValueError("feature_topo_min_score must be in [0, 1]")
        if self.feature_topo_prior_boost < 0:
            raise ValueError("feature_topo_prior_boost must be non-negative")
        if self.dynamic_alpha not in DYNAMIC_ALPHA_MODES:
            raise ValueError(f"dynamic_alpha must be one of {sorted(DYNAMIC_ALPHA_MODES)}")
        if self.dynamic_alpha != "off" and self.edge_mode != "feature":
            # Dynamic α is wired only into the feature kernel; in device mode it
            # would be silently ignored — fail loudly instead.
            raise ValueError(
                "dynamic_alpha requires edge_mode='feature' "
                f"(got edge_mode={self.edge_mode!r})"
            )
        # NOTE: dynamic state reads clears from the raw input stream directly (the
        # state machine runs before clears are dropped); modeled events still
        # exclude clears, so include_clear stays as-is (default False).
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
            time_slack=self.time_slack_sec / self.time_scale_sec,
            late_penalty_half_life=self.late_penalty_half_life_sec / self.time_scale_sec,
            max_history_events=self.max_history_events,
            max_iters=self.max_iters,
            tol=self.tol,
            alpha_prior_strength=self.alpha_prior_strength,
            alpha_prior_mean=self.alpha_prior_mean,
            topology_prior_boost=self.topology_prior_boost,
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


def _state_combo(state) -> int:
    vals = tuple(int(x) for x in state)
    vals = (vals + (0, 0, 0))[:3]
    return int(vals[0] + 2 * vals[1] + 4 * vals[2])


def _source_target_combo_bits(base_bits: np.ndarray) -> np.ndarray:
    base_bits = np.asarray(base_bits, dtype=np.float64)
    return np.concatenate(
        [
            np.repeat(base_bits, base_bits.shape[0], axis=0),
            np.tile(base_bits, (base_bits.shape[0], 1)),
        ],
        axis=1,
    )


def _combo_arrays_from_timeline(state_timeline, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return timeline change points and packed 3-bit combos for one device."""
    device = str(device or "")
    times = getattr(state_timeline, "_times", {}).get(device)
    states = getattr(state_timeline, "_states", {}).get(device)
    if not times or not states:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.uint8),
        )
    times_arr = np.asarray(times, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.uint8)
    combos = (
        states_arr[:, 0].astype(np.uint8)
        + np.uint8(2) * states_arr[:, 1].astype(np.uint8)
        + np.uint8(4) * states_arr[:, 2].astype(np.uint8)
    )
    return times_arr, combos.astype(np.uint8, copy=False)


def _timeline_combos_at_many(
    state_timeline,
    device: str,
    query_ts: np.ndarray,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Vectorized state_at(device, ts-) returning packed combo indices."""
    device = str(device or "")
    if device not in cache:
        cache[device] = _combo_arrays_from_timeline(state_timeline, device)
    times_arr, combos_arr = cache[device]
    out = np.zeros(len(query_ts), dtype=np.uint8)
    if times_arr.size == 0 or len(query_ts) == 0:
        return out
    query_before = np.nextafter(np.asarray(query_ts, dtype=np.float64), -np.inf)
    idx = np.searchsorted(times_arr, query_before, side="right") - 1
    ok = idx >= 0
    if ok.any():
        out[ok] = combos_arr[idx[ok]]
    return out


def _build_source_target_dynamic_exposure(
    train_events: EventCollection,
    cand_targets: np.ndarray,
    cand_sources: np.ndarray,
    *,
    type_ne: np.ndarray,
    train_event_ne: list[str],
    train_abs_times: list[float],
    train_src_combo: np.ndarray,
    train_pre_combo: np.ndarray,
    state_timeline,
    verbose: bool = False,
) -> np.ndarray:
    """B-fast source_target exposure buckets.

    For each source event and candidate target type, sample the target device
    state at source_ts (or the source event's pre-state for same-NE pairs to
    avoid counting the source raise itself), then bucket exposure by
    source_combo*8 + target_combo. Kernel/time-slack scale is applied in EM.
    """
    cand_targets = np.asarray(cand_targets, dtype=np.int64)
    cand_sources = np.asarray(cand_sources, dtype=np.int64)
    C = len(cand_targets)
    exposure = np.zeros((C, 64), dtype=np.float32)

    # Encode target NE strings as small integer ids, then lexsort candidate rows
    # by (source_type, target_ne). This avoids millions of Python list appends
    # when C is large.
    ne_to_id: dict[str, int] = {}
    ne_labels: list[str] = []
    type_ne_ids = np.zeros(len(type_ne), dtype=np.int32)
    for ti, ne in enumerate(type_ne):
        key = str(ne or "")
        nid = ne_to_id.get(key)
        if nid is None:
            nid = len(ne_labels)
            ne_to_id[key] = nid
            ne_labels.append(key)
        type_ne_ids[ti] = nid
    cand_ne_ids = type_ne_ids[cand_targets]
    if verbose:
        gib = C * 64 * np.dtype(exposure.dtype).itemsize / (1024 ** 3)
        print(
            f"[train] dynamic_alpha=source_target: allocating B-fast exposure "
            f"{C}x64 {exposure.dtype} (~{gib:.1f} GiB), grouping candidate rows ...",
            flush=True,
        )
    row_order = np.lexsort((cand_ne_ids, cand_sources))
    src_sorted = cand_sources[row_order]
    ne_sorted = cand_ne_ids[row_order]
    boundary = np.flatnonzero((src_sorted[1:] != src_sorted[:-1]) | (ne_sorted[1:] != ne_sorted[:-1])) + 1
    starts = np.concatenate(([0], boundary))
    ends = np.concatenate((boundary, [C]))
    group_src = src_sorted[starts]
    group_ne = ne_sorted[starts]

    dims = np.asarray(train_events.dims, dtype=np.int64)
    dim_order = np.argsort(dims, kind="stable")
    dims_sorted = dims[dim_order]
    train_event_ne_arr = np.asarray([str(ne or "") for ne in train_event_ne], dtype=object)
    train_abs_times_arr = np.asarray(train_abs_times, dtype=np.float64)
    train_src_combo_arr = np.asarray(train_src_combo, dtype=np.uint8)
    train_pre_combo_arr = np.asarray(train_pre_combo, dtype=np.uint8)
    timeline_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    n_groups = len(starts)
    last_beat = time.monotonic()
    t0 = last_beat
    for gi, (start, end, src_tid, ne_id) in enumerate(zip(starts, ends, group_src, group_ne), start=1):
        left = np.searchsorted(dims_sorted, int(src_tid), side="left")
        right = np.searchsorted(dims_sorted, int(src_tid), side="right")
        if right <= left:
            continue
        event_idx = dim_order[left:right]
        rows = row_order[start:end]
        tgt_ne = ne_labels[int(ne_id)]

        tgt_combo = _timeline_combos_at_many(
            state_timeline,
            tgt_ne,
            train_abs_times_arr[event_idx],
            timeline_cache,
        )
        # Same-NE exposure uses the source event's pre-state so the source
        # event's own raise is not included in the sampled target state.
        if tgt_ne:
            same_ne = train_event_ne_arr[event_idx] == tgt_ne
            if same_ne.any():
                tgt_combo[same_ne] = train_pre_combo_arr[event_idx[same_ne]]

        combo_idx = train_src_combo_arr[event_idx].astype(np.uint8) * np.uint8(8) + tgt_combo
        counts = np.bincount(combo_idx.astype(np.int64), minlength=64).astype(exposure.dtype, copy=False)
        nz = np.flatnonzero(counts > 0)
        for k in nz:
            exposure[rows, k] += counts[k]

        if verbose:
            now = time.monotonic()
            if now - last_beat >= 10.0:
                print(
                    f"[train] dynamic_alpha=source_target: exposure groups "
                    f"{gi}/{n_groups} ({100.0 * gi / max(n_groups, 1):.1f}%, "
                    f"{_fmt_secs(now - t0)})",
                    flush=True,
                )
                last_beat = now
    return exposure


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
    ne_graph_data=None,
    best_checkpoint_path: Optional[str] = None,
) -> AlarmMHPArtifact:
    """Fit alarm-flow MHP parameters via windowed sparse MAP EM.

    ne_graph_data: raw NE graph dict — required for edge_mode='feature' (device
    attributes for pair features). Topology_index is still used for topology
    relation features / prior.
    """
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

    # Dynamic (stateful) α: per-modeled-event marks, aligned to the TRAIN slice.
    # The state machine runs over the full stream (clears included) before
    # clears are dropped; combos pack the 3 uncleared-alarm booleans.
    train_src_combo = None
    train_tgt_combo = None
    dyn_combo_bits = None
    dyn_exposure_2d = None
    dyn_feature_names = None
    if config.edge_mode == "feature" and config.dynamic_alpha != "off":
        if config.dynamic_alpha == "source_target" and "alarm_source" not in tuple(config.type_fields):
            raise ValueError("dynamic_alpha='source_target' requires alarm_source in type_fields")
        from alarm_flow_mhp.dynamic_state import (
            ObservedStateTimeline,
            build_event_states,
            states_to_combo,
            combo_bits as _combo_bits,
        )
        from alarm_flow_mhp.feature_spec import (
            _type_field_indices,
            parse_label_ne_at,
            runtime_ne_at,
        )
        ev_state_full = build_event_states(
            sorted_alarm_events,
            sequence.events,
            is_clear=lambda e: is_clear_alarm(e.get("alarm", {})),
            device_of=lambda e: runtime_ne_at(e, config.type_fields)[0],
            alarm_type_of=lambda e: alarm_type_from_title(e.get("alarm_title", "")),
        )
        combo_full = states_to_combo(ev_state_full)
        train_src_combo = combo_full[: train_events.n]
        base_combo_bits = _combo_bits(8)
        if config.dynamic_alpha == "source_target":
            train_tgt_combo = train_src_combo.copy()
            dyn_combo_bits = _source_target_combo_bits(base_combo_bits)
            dyn_feature_names = [
                "src_uncleared_link",
                "src_uncleared_power",
                "src_uncleared_offline",
                "tgt_uncleared_link",
                "tgt_uncleared_power",
                "tgt_uncleared_offline",
            ]
        else:
            dyn_combo_bits = base_combo_bits
            dyn_feature_names = ["src_uncleared_link", "src_uncleared_power", "src_uncleared_offline"]
        if verbose:
            nz = int((train_src_combo > 0).sum())
            msg = (
                f"[train] dynamic_alpha={config.dynamic_alpha}: source marks on "
                f"{nz}/{train_events.n} train events ({100.0*nz/max(train_events.n,1):.1f}% with active state)"
            )
            if train_tgt_combo is not None:
                tnz = int((train_tgt_combo > 0).sum())
                msg += (
                    f"; target marks on {tnz}/{train_events.n} train events "
                    f"({100.0*tnz/max(train_events.n,1):.1f}% with active state)"
                )
            print(msg, flush=True)

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

    # Topology prior (optional): sparse extra prior mass on topologically
    # related (target, source) pairs so rare/zero-co-occurrence but connected
    # device pairs still get an edge. Built once from the NE graph.
    topo_prior_flat = None
    topo_prior_score = None
    # Feature-mode device attributes (NE graph → manufacturer/ne_type/site/...).
    feature_graph_context = None
    if config.edge_mode == "feature":
        if ne_graph_data is None:
            raise ValueError("edge_mode='feature' requires ne_graph_data (NE graph) for device attributes")
        from ne_link_learning.core import build_graph_context

        feature_graph_context = build_graph_context(ne_graph_data)

    if config.topology_prior_boost > 0 and topology_index is not None:
        topo_prior_flat, topo_prior_score = build_topology_pairs(
            vocabs,
            config.type_fields,
            topology_index,
            max_hops=config.topology_prior_max_hops,
            min_score=config.topology_prior_min_score,
        )
        if verbose:
            print(
                f"[train] topology prior: {len(topo_prior_flat)} related (target,source) "
                f"pairs (boost={config.topology_prior_boost}, max_hops="
                f"{config.topology_prior_max_hops}, min_score={config.topology_prior_min_score})",
                flush=True,
            )
    elif config.topology_prior_boost > 0 and topology_index is None and verbose:
        print("[train] WARN: topology_prior_boost > 0 but no NE graph loaded → prior disabled", flush=True)

    feat_at_vocab = None
    feat_mu_spec = None

    def write_best_checkpoint(best_result: MHPResult, trace_entry: dict):
        if not best_checkpoint_path:
            return
        metadata = {
            "checkpoint": True,
            "checkpoint_kind": "best_train_log_likelihood",
            "checkpoint_iter": int(trace_entry.get("iter", best_result.iterations_run - 1)),
            "checkpoint_metric": "train_log_likelihood",
            "considered_event_count": considered_event_count,
            "region_filter": region_filter_stats,
            "sequence_stats": sequence_stats,
            "modeled_event_count": train_events.n + (val_events.n if val_events else 0),
            "train_event_count": train_events.n,
            "val_event_count": (val_events.n if val_events else 0),
            "type_count": M,
            "active_edge_count": len(best_result.params.edge_alpha),
            "best_log_likelihood": float(best_result.log_likelihood),
            "best_val_log_likelihood": None,
            "iterations_run": int(best_result.iterations_run),
            "converged": False,
            "type_labels": list(vocabs.type_vocab.labels),
            "time_slack_sec": float(config.time_slack_sec),
            "late_penalty_half_life_sec": float(config.late_penalty_half_life_sec),
            "feature_kernel": (
                best_result.feature_kernel.to_dict()
                if best_result.feature_kernel is not None else None
            ),
            "feature_runtime": (
                _build_feature_runtime(best_result, vocabs, config, feat_at_vocab, feat_mu_spec)
                if config.edge_mode == "feature"
                else None
            ),
            "val_trace": list(val_trace),
        }
        artifact = AlarmMHPArtifact(
            params=best_result.params,
            vocabs=vocabs,
            config=config,
            training_metadata=metadata,
            trace=best_result.trace,
        )
        tmp_path = f"{best_checkpoint_path}.tmp"
        save_alarm_mhp_artifact(tmp_path, artifact)
        os.replace(tmp_path, best_checkpoint_path)
        if verbose:
            print(
                f"[train] best checkpoint updated: {best_checkpoint_path} "
                f"(iter={metadata['checkpoint_iter']}, ll={best_result.log_likelihood:.2f})",
                flush=True,
            )

    if config.edge_mode == "feature":
        # Feature-weighted α = softplus(w·φ) over device-agnostic pair features.
        from alarm_flow_mhp.feature_spec import build_candidate_features

        # Warn about config that feature mode does NOT honor, so users aren't
        # misled by flags that silently no-op here.
        if verbose:
            ignored = []
            if config.kernel_type == "piecewise":
                ignored.append("--kernel-type piecewise (feature mode is exp-only)")
            if config.beta_mode == "per_edge":
                ignored.append("--beta-mode per_edge (feature mode uses shared β)")
            if config.topology_prior_boost > 0:
                ignored.append(
                    "--topology-prior-boost (device-mode prior; in feature mode use "
                    "--feature-topo-prior-boost for the equivalent pseudo-count topology prior)"
                )
            for msg in ignored:
                print(f"[train] WARN: edge_mode=feature ignores {msg}", flush=True)
            print(
                "[train] NOTE: feature mode has no branching-cap/top-k/spectral cap; "
                "stationarity relies on the L2 ridge (--feature-l2) + data. ρ is checked below.",
                flush=True,
            )

        if verbose:
            print("[train] edge_mode=feature → building candidate pair features ...", flush=True)
        from alarm_flow_mhp.feature_spec import build_mu_features

        cand_t, cand_s, phi, feat_names, feat_at_vocab, feat_type_group, cand_topo_score = build_candidate_features(
            train_events,
            vocabs,
            config.type_fields,
            topology_index=topology_index,
            graph_context=feature_graph_context,
            history_window=mhp_config.history_window,
            max_history_events=mhp_config.max_history_events,
            chunk_size=mhp_config.chunk_size,
            time_slack=mhp_config.time_slack,
            topo_max_hops=config.feature_topo_max_hops,
            topo_min_score=config.feature_topo_min_score,
        )
        # Parameterized inductive μ: ψ(u) single-type features (alarm_type +
        # ne_type/vendor/domain from the NE graph), μ=softplus(w_μ·ψ).
        mu_phi, mu_spec = build_mu_features(vocabs, config.type_fields, feature_graph_context)
        feat_mu_spec = mu_spec
        if verbose:
            print(
                f"[train] feature mode: {len(cand_t)} candidate pairs, {phi.shape[1]} features "
                f"{feat_names}",
                flush=True,
            )
        if config.dynamic_alpha == "source_target":
            state_timeline = ObservedStateTimeline()
            for ev in sorted(sorted_alarm_events, key=lambda e: float(e.get("ts", 0.0))):
                ne, _ = runtime_ne_at(ev, config.type_fields)
                state_timeline.ingest(
                    float(ev.get("ts", 0.0)),
                    ne,
                    alarm_type_from_title(ev.get("alarm_title", "")),
                    is_clear_alarm(ev.get("alarm", {})),
                )
            src_idx, at_idx = _type_field_indices(config.type_fields)
            type_ne = np.asarray(
                [
                    parse_label_ne_at(label, src_idx, at_idx)[0]
                    for label in vocabs.type_vocab.labels
                ],
                dtype=object,
            )
            train_event_objs = sequence.events[: train_events.n]
            train_event_ne = [runtime_ne_at(ev, config.type_fields)[0] for ev in train_event_objs]
            train_abs_times = [float(t) for t in sequence.times[: train_events.n]]
            dyn_exposure_2d = _build_source_target_dynamic_exposure(
                train_events,
                cand_t,
                cand_s,
                type_ne=type_ne,
                train_event_ne=train_event_ne,
                train_abs_times=train_abs_times,
                train_src_combo=train_src_combo,
                train_pre_combo=train_tgt_combo,
                state_timeline=state_timeline,
                verbose=verbose,
            )
            if verbose:
                nonzero = int((dyn_exposure_2d > 0).sum())
                print(
                    f"[train] dynamic_alpha=source_target: B-fast exposure buckets "
                    f"nonzero={nonzero}/{dyn_exposure_2d.size}",
                    flush=True,
                )
        result = fit_mhp_feature(
            train_events,
            mhp_config,
            cand_targets=cand_t,
            cand_sources=cand_s,
            cand_phi=phi,
            feature_names=feat_names,
            l2=config.feature_l2,
            mu_phi=mu_phi,                       # inductive parameterized μ
            mu_feature_names=mu_spec.feature_names,
            cand_topo_score=cand_topo_score,     # topology pseudo-count prior
            topo_prior_boost=config.feature_topo_prior_boost,
            src_combo=train_src_combo,            # dynamic stateful α (source mark)
            tgt_combo=train_tgt_combo,            # source_target: target pre-state mark
            dynamic_combo_bits=dyn_combo_bits,
            dynamic_exposure_2d=dyn_exposure_2d,
            dynamic_feature_names=dyn_feature_names,
            iter_callback=iter_callback,
            best_callback=write_best_checkpoint,
        )
        # Stationarity check (feature mode has no hard cap): warn if the
        # materialized α matrix's spectral radius ≥ 1 → cluster-Poisson diverges.
        if config.stability_radius > 0 and len(result.params.edge_alpha):
            rho = result.params.spectral_radius()
            if rho >= config.stability_radius and verbose:
                print(
                    f"[train] WARN: feature-mode α spectral radius ρ={rho:.3f} >= "
                    f"{config.stability_radius}. No hard cap in feature mode — raise "
                    f"--feature-l2 to shrink weights, or expect over-excitation. ",
                    flush=True,
                )
            elif verbose:
                print(f"[train] feature-mode α spectral radius ρ={rho:.3f} (< {config.stability_radius}, OK)", flush=True)
    elif config.kernel_type == "piecewise":
        # Stage 1: exp-kernel fit selects the sparse active edge set + μ. The
        # topology prior applies HERE (edge selection); stage 2 only learns θ
        # on the resulting edges.
        if verbose:
            print("[train] kernel=piecewise → stage 1: exp-kernel edge selection", flush=True)
        stage1_config = replace(mhp_config, kernel_type="exp")
        stage1 = fit_mhp(
            train_events,
            stage1_config,
            iter_callback=iter_callback,
            topo_prior_flat=topo_prior_flat,
            topo_prior_score=topo_prior_score,
        )
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
            best_callback=write_best_checkpoint,
        )
    else:
        result = fit_mhp(
            train_events,
            mhp_config,
            iter_callback=iter_callback,
            best_callback=write_best_checkpoint,
            topo_prior_flat=topo_prior_flat,
            topo_prior_score=topo_prior_score,
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
        "time_slack_sec": float(config.time_slack_sec),
        "late_penalty_half_life_sec": float(config.late_penalty_half_life_sec),
        "cascade_size_stats_soft": cascade_stats_soft,
        "cascade_size_stats": cascade_stats_hard,
        "topology_consistency": topology_report,
        "bucket_mass_distribution": bucket_mass,
        "best_checkpoint_path": os.path.abspath(best_checkpoint_path) if best_checkpoint_path else "",
        # Feature-mode: persist the learned weights (interpretable + enables
        # live-α inference for unseen devices) and the runtime info needed to
        # rebuild the scorer (alarm-type vocab + μ aggregated per alarm-type so
        # new devices have an immigrant baseline).
        "feature_kernel": (
            result.feature_kernel.to_dict() if result.feature_kernel is not None else None
        ),
        "feature_runtime": (
            _build_feature_runtime(result, vocabs, config, feat_at_vocab, feat_mu_spec)
            if config.edge_mode == "feature"
            else None
        ),
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


def _build_feature_runtime(result, vocabs, config, at_vocab, mu_spec=None):
    """Runtime info for feature-mode inference:
      - α: feature_kernel (stored separately)
      - μ: the parameterized μ kernel (w_μ) + its MuFeatureSpec (category vocabs)
           for live μ on any device; plus a per-alarm-type μ table as a fallback
           when NE attributes are unavailable.
      - β shared scalar.
    """
    labels = vocabs.type_vocab.labels
    type_fields = tuple(config.type_fields)
    try:
        at_idx = type_fields.index("alarm_type")
    except ValueError:
        at_idx = None

    def _at_of(label):
        if at_idx is None:
            return ""
        parts = str(label).split(" | ")
        return parts[at_idx] if len(parts) > at_idx else ""

    mu = result.params.mu
    at_to_mus = defaultdict(list)
    for tid, label in enumerate(labels):
        at = _at_of(label)
        if at:
            at_to_mus[at].append(float(mu[tid]) if tid < len(mu) else 0.0)
    global_med = float(np.median(mu)) if len(mu) else 0.0
    mu_by_at = {at: (float(np.mean(v)) if v else global_med) for at, v in at_to_mus.items()}
    return {
        "at_vocab": list(at_vocab or []),
        "mu_by_alarm_type": mu_by_at,       # fallback when NE attrs missing
        "mu_default": global_med,
        "beta": float(config.beta_shared_value),
        # Parameterized μ (preferred at inference):
        "mu_kernel": (result.mu_kernel.to_dict() if result.mu_kernel is not None else None),
        "mu_spec": (mu_spec.to_dict() if mu_spec is not None else None),
    }


def _ne_pair_topo_score(source_ne: str, target_ne: str, topology_index) -> float:
    """Directed topology relation score (source → target), NE-level.

    Mirrors the cross-NE branch of _topology_relation_score (same buckets), but
    works on NE strings only (no site_id). Returns 0 if unrelated.
    """
    if source_ne == target_ne:
        return 1.0
    features = topology_index.pair_features(source_ne, target_ne)
    if not features:
        return 0.0
    # features: [same, direct_fwd, direct_rev, direct_bi, reach_fwd, reach_rev,
    #            undirected_reach, ...]
    if features[1] > 0:
        return 1.0
    if features[2] > 0 or features[3] > 0:
        return 0.85
    if features[4] > 0:
        return 0.75
    if features[5] > 0:
        return 0.6
    if features[6] > 0:
        return 0.45
    return 0.0


def build_topology_pairs(vocabs, type_fields, topology_index, *, max_hops, min_score):
    """Build the sparse topology prior: (flat_index, score) for every
    topologically-related (target_type, source_type) pair among active types.

    flat_index = target_type_id * M + source_type_id (matches the EM's α
    flattening). Score in (0, 1] graded by relation strength (same NE / direct /
    multi-hop). Edges are generated by traversing the NE graph (not M²): types
    are grouped by NE, then same-NE pairs and graph-reachable NE pairs (within
    max_hops, score >= min_score) are crossed.

    Returns (flat_idx int64 array, score float64 array), max score kept per pair.
    """
    if topology_index is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    labels = vocabs.type_vocab.labels
    M = len(labels)
    try:
        src_field_idx = tuple(type_fields).index("alarm_source")
    except ValueError:
        # No NE field in the type → topology prior is meaningless
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    def _ne_of(label):
        parts = str(label).split(" | ")
        return parts[src_field_idx] if len(parts) > src_field_idx else ""

    # Group active type_ids by their NE
    ne_to_types: dict[str, list[int]] = defaultdict(list)
    for tid, label in enumerate(labels):
        ne = _ne_of(label)
        if ne and ne != "<empty>":
            ne_to_types[ne].append(tid)

    undirected_hops = getattr(topology_index, "undirected_hops", {}) or {}
    pair_score: dict[int, float] = {}

    def _emit(target_tid, source_tid, score):
        key = target_tid * M + source_tid
        prev = pair_score.get(key)
        if prev is None or score > prev:
            pair_score[key] = score

    # 1) Same-NE pairs (includes self-loops) → score 1.0
    for ne, tids in ne_to_types.items():
        for u in tids:
            for v in tids:
                _emit(u, v, 1.0)

    # 2) Cross-NE pairs reachable within max_hops (source → target)
    for source_ne, source_tids in ne_to_types.items():
        reachable = undirected_hops.get(source_ne, {})
        for target_ne, hop in reachable.items():
            if hop > max_hops or target_ne == source_ne:
                continue
            target_tids = ne_to_types.get(target_ne)
            if not target_tids:
                continue
            score = _ne_pair_topo_score(source_ne, target_ne, topology_index)
            if score < min_score:
                continue
            for u in target_tids:        # target type
                for v in source_tids:    # source type (excites target)
                    _emit(u, v, score)

    if not pair_score:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    flat = np.fromiter(pair_score.keys(), dtype=np.int64, count=len(pair_score))
    score = np.fromiter(pair_score.values(), dtype=np.float64, count=len(pair_score))
    return flat, score


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
