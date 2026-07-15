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
import gc
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
from fault_grouping.alarm_events.identity import require_eid, require_occurrence_uuid
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
# uncleared-alarm state (link/power/offline), using read-before-write event
# snapshots (train/infer-consistent, clear-aware). feature mode only.
#   off           — static features only (current behavior)
#   source        — +3 booleans: source device's uncleared state at t_j
#                   (exact in both occurrence and compensator terms)
#   target        — +3 booleans: target device pre-state. Training uses the
#                   B-fast approximation described below for source_target.
#   source_target — +6 booleans: source state plus target device pre-state.
#                   Training uses the B-fast approximation: E-step reads the
#                   target event's pre-state (time-slack safe); compensator
#                   buckets target state sampled at source_ts.
DYNAMIC_ALPHA_MODES = frozenset({"off", "source", "target", "source_target"})
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
    # Which type field identifies the topological entity for the topology prior /
    # consistency report / feature-mode node lookups. "alarm_source" (default) =
    # per-device topology over the NE graph; "site_id" = per-site topology over a
    # site graph (same structure). Must be one of type_fields.
    # Empty means infer from type_fields: prefer alarm_source for backwards
    # compatibility, otherwise use site_id. Explicit values always win.
    topology_node_field: str = ""
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
    feature_l2_normalize: bool = False        # scale the α ridge by event/data mass
                                              # N (not raw exposure ΣE) so feature_l2
                                              # is data-size-independent and actually
                                              # bites (controls ρ); OFF = legacy raw ridge
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
    # Opt-in spectral cap for FEATURE mode (default OFF = legacy warn-only). When
    # ON: model selection prefers the val-best snapshot whose ρ ≤ stability_radius
    # (no distortion); if none qualifies, the val-best overall is rescaled (α ×
    # stability_radius/ρ, stored as FeatureKernel.alpha_scale) to guarantee ρ ≤
    # stability_radius. The parametric path always caps; this brings feature mode
    # to parity without changing existing feature-mode runs.
    feature_spectral_cap: bool = False
    chunk_size: int = 20_000
    # E-step worker threads (1 = serial; 0 = auto: min(8, cpu count)). Chunk
    # results merge in chunk order, so the trained model does not depend on it.
    estep_workers: int = 1
    # GPU offload of the E-step chunk math ("auto" = CUDA else CPU; needs
    # torch — silently stays on CPU without it). "cpu" pins the exactly
    # reproducible CPU path; "cuda" requires that device.
    estep_device: str = "cpu"
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
    # Which metric drives model SELECTION + early stop: "train" (legacy — pick
    # the train-LL-best weights, no val early stop; val LL is still printed each
    # iter when val_split>0, purely informational) or "val" (pick the val-LL
    # peak snapshot + early-stop when val LL plateaus). Default "train" keeps
    # existing runs bit-for-bit reproducible.
    selection_metric: str = "train"
    regions: tuple = ()
    min_group_events: int = 1
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "regions", parse_regions(self.regions))
        sequence_config = self.sequence_config()
        type_fields = tuple(self.type_fields)
        node_field = str(self.topology_node_field or "").strip()
        if not node_field:
            if "alarm_source" in type_fields:
                node_field = "alarm_source"
            elif "site_id" in type_fields:
                node_field = "site_id"
            else:
                raise ValueError(
                    "cannot infer topology_node_field: type_fields must contain "
                    "alarm_source or site_id, or topology_node_field must be set explicitly"
                )
            object.__setattr__(self, "topology_node_field", node_field)
        if node_field not in type_fields:
            raise ValueError(
                f"topology_node_field={node_field!r} must be one of "
                f"type_fields {type_fields} (it is parsed from the type label)"
            )
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
        if self.estep_workers < 0:
            raise ValueError("estep_workers must be >= 0 (0 = auto)")
        if self.estep_device not in ("auto", "cpu", "cuda"):
            raise ValueError("estep_device must be one of auto|cpu|cuda")
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
        if self.selection_metric not in ("train", "val"):
            raise ValueError("selection_metric must be 'train' or 'val'")
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
            estep_workers=self.estep_workers,
            estep_device=self.estep_device,
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


def _event_metadata_value(event, *keys):
    if not isinstance(event, dict):
        return ""
    candidates = [event]
    for nested_key in ("alarm", "raw_alarm", "source_alarm"):
        nested = event.get(nested_key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for source in candidates:
        for key in keys:
            value = source.get(key, "")
            text = str(value or "").strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text
    return ""


def summarize_alarm_event(event, index):
    alarm = event.get("alarm", {}) if isinstance(event, dict) else {}
    return {
        "index": int(index),
        "event_id": require_eid(event),
        "occurrence_uuid": require_occurrence_uuid(event),
        "ts": float(event.get("ts", 0.0)),
        "site_id": str(event.get("site_id", "") or ""),
        "alarm_source": str(event.get("alarm_source", "") or ""),
        "alarm_title": str(event.get("alarm_title", "") or ""),
        "alarm_type": alarm_type_from_title(event.get("alarm_title", "")),
        "is_clear": is_clear_alarm(alarm),
        "工单号": _event_metadata_value(event, "工单号", "ticket_id"),
        "故障组ID": _event_metadata_value(
            event,
            "故障组ID",
            "告警故障组ID",
            "原始故障组ID",
            "fault_group_id",
            "alarm_group_id",
            "native_group_id",
        ),
        "告警清除时间": _event_metadata_value(event, "告警清除时间", "clear_time"),
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


def _target_combo_bits(base_bits: np.ndarray) -> np.ndarray:
    """64-row combo table whose features contain only the target 3-bit state.

    ``fit_mhp_feature`` indexes target-aware combos as ``source * 8 + target``.
    Target-only mode represents the source combo as zero, but retaining all 64
    rows lets it reuse the source_target B-fast occurrence/exposure machinery.
    """
    base_bits = np.asarray(base_bits, dtype=np.float64)
    return np.tile(base_bits, (base_bits.shape[0], 1))


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


def _source_combo_prefix(combos: np.ndarray) -> np.ndarray:
    """prefix[i+1, k] = # source events < i whose source combo is k."""
    combos = np.asarray(combos, dtype=np.int64)
    prefix = np.zeros((len(combos) + 1, 8), dtype=np.float32)
    if len(combos):
        prefix[np.arange(len(combos)) + 1, combos] = 1.0
        np.cumsum(prefix, axis=0, out=prefix)
    return prefix


def _counts64_from_target_intervals(
    source_times: np.ndarray,
    source_prefix8: np.ndarray,
    target_change_times: np.ndarray,
    target_combos: np.ndarray,
) -> np.ndarray:
    """Exact B-fast exposure counts for one (source_type, target_ne) group.

    state_at(t-) means a target state written at time c is visible only for
    source event times strictly greater than c. Therefore state intervals are
    (-inf, c0] for baseline zero, then (cj, c{j+1}] for post-state j.
    """
    counts64 = np.zeros(64, dtype=np.float32)
    n = len(source_times)
    if n == 0:
        return counts64
    if target_change_times.size == 0:
        counts64[:64:8] = source_prefix8[-1]
        return counts64

    first_right = int(np.searchsorted(source_times, target_change_times[0], side="right"))
    if first_right > 0:
        counts64[:64:8] += source_prefix8[first_right] - source_prefix8[0]

    left = first_right
    for j, tgt_k in enumerate(target_combos):
        if j + 1 < len(target_change_times):
            right = int(np.searchsorted(source_times, target_change_times[j + 1], side="right"))
        else:
            right = n
        if right > left:
            start = int(tgt_k)
            counts64[start::8] += source_prefix8[right] - source_prefix8[left]
        left = right
        if left >= n:
            break
    return counts64


def _presort_feature_candidates_for_em(
    cand_t: np.ndarray,
    cand_s: np.ndarray,
    phi: np.ndarray,
    cand_topo_score: Optional[np.ndarray],
    M: int,
    *,
    verbose: bool = False,
):
    """Sort candidate feature rows by EM lookup key before large exposure build.

    Doing this before source_target exposure exists avoids EM startup holding
    both unsorted and sorted copies of the GB-scale feature and exposure tables.
    """
    keys = np.asarray(cand_t, dtype=np.int64) * int(M) + np.asarray(cand_s, dtype=np.int64)
    if len(keys) <= 1 or bool(np.all(keys[1:] >= keys[:-1])):
        return cand_t, cand_s, phi, cand_topo_score
    if verbose:
        print("[train] feature mode: sorting candidate rows for EM lookup ...", flush=True)
    order = np.argsort(keys, kind="stable")
    cand_t_sorted = np.asarray(cand_t, dtype=np.int64)[order]
    cand_s_sorted = np.asarray(cand_s, dtype=np.int64)[order]
    phi_sorted = np.asarray(phi, dtype=np.float64)[order]
    topo_sorted = None
    if cand_topo_score is not None:
        topo_sorted = np.asarray(cand_topo_score)[order]
    del order, keys
    gc.collect()
    return cand_t_sorted, cand_s_sorted, phi_sorted, topo_sorted


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
    dynamic_mode: str = "source_target",
    verbose: bool = False,
    as_coo: bool = False,
) -> np.ndarray | dict:
    """B-fast target-aware exposure buckets.

    For each source event and candidate target type, sample the target device
    state at source_ts (or the source event's pre-state for same-NE pairs to
    avoid counting the source raise itself), then bucket exposure by
    source_combo*8 + target_combo. Kernel/time-slack scale is applied in EM.
    """
    cand_targets = np.asarray(cand_targets, dtype=np.int64)
    cand_sources = np.asarray(cand_sources, dtype=np.int64)
    C = len(cand_targets)
    mode_label = str(dynamic_mode or "source_target")
    exposure = None
    row_dtype = np.int32 if C <= np.iinfo(np.int32).max else np.int64
    coo_rows = coo_cols = coo_vals = None
    coo_size = 0
    coo_cap = 0
    if as_coo:
        # Build the exact nonzero buckets directly, avoiding the dense Cx64
        # exposure allocation. Capacity grows geometrically; group rows are
        # usually tiny, so append overhead stays close to the old dense fill loop.
        coo_cap = max(1024, min(C * 2, 8_000_000))
        coo_rows = np.empty(coo_cap, dtype=row_dtype)
        coo_cols = np.empty(coo_cap, dtype=np.uint8)
        coo_vals = np.empty(coo_cap, dtype=np.float32)
    else:
        exposure = np.zeros((C, 64), dtype=np.float32)

    def append_coo(rows: np.ndarray, nz: np.ndarray, counts: np.ndarray):
        nonlocal coo_rows, coo_cols, coo_vals, coo_size, coo_cap
        if len(rows) == 0 or len(nz) == 0:
            return
        needed = int(len(rows) * len(nz))
        end = coo_size + needed
        if end > coo_cap:
            while end > coo_cap:
                coo_cap = max(end, int(coo_cap * 1.5) + 1024)
            new_rows = np.empty(coo_cap, dtype=coo_rows.dtype)
            new_cols = np.empty(coo_cap, dtype=coo_cols.dtype)
            new_vals = np.empty(coo_cap, dtype=coo_vals.dtype)
            new_rows[:coo_size] = coo_rows[:coo_size]
            new_cols[:coo_size] = coo_cols[:coo_size]
            new_vals[:coo_size] = coo_vals[:coo_size]
            coo_rows, coo_cols, coo_vals = new_rows, new_cols, new_vals
        pos = coo_size
        rows_cast = np.asarray(rows, dtype=row_dtype)
        for k in nz:
            k_int = int(k)
            nxt = pos + len(rows_cast)
            coo_rows[pos:nxt] = rows_cast
            coo_cols[pos:nxt] = k_int
            coo_vals[pos:nxt] = counts[k_int]
            pos = nxt
        coo_size = end

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
        gib = C * 64 * np.dtype(np.float32).itemsize / (1024 ** 3)
        mode = "COO" if as_coo else "dense"
        print(
            f"[train] dynamic_alpha={mode_label}: building B-fast exposure "
            f"{mode} (dense equivalent {C}x64 float32 ~{gib:.1f} GiB), "
            f"grouping candidate rows ...",
            flush=True,
        )
    row_order = np.lexsort((cand_sources, cand_ne_ids))
    src_sorted = cand_sources[row_order]
    ne_sorted = cand_ne_ids[row_order]
    boundary = np.flatnonzero((src_sorted[1:] != src_sorted[:-1]) | (ne_sorted[1:] != ne_sorted[:-1])) + 1
    starts = np.concatenate(([0], boundary))
    ends = np.concatenate((boundary, [C]))
    group_src = src_sorted[starts]
    group_ne = ne_sorted[starts]
    if verbose:
        print(
            f"[train] dynamic_alpha={mode_label}: exposure groups={len(starts)} "
            f"(target_nes={len(np.unique(group_ne))}, source_types={len(np.unique(group_src))}, "
            f"avg_candidate_rows={C / max(len(starts), 1):.1f})",
            flush=True,
        )

    dims = np.asarray(train_events.dims, dtype=np.int64)
    dim_order = np.argsort(dims, kind="stable")
    dims_sorted = dims[dim_order]
    train_abs_times_arr = np.asarray(train_abs_times, dtype=np.float64)
    train_src_combo_arr = np.asarray(train_src_combo, dtype=np.uint8)
    train_pre_combo_arr = np.asarray(train_pre_combo, dtype=np.uint8)
    timeline_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def source_stats(src_tid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        src_tid = int(src_tid)
        cached = source_cache.get(src_tid)
        if cached is not None:
            return cached
        left = np.searchsorted(dims_sorted, src_tid, side="left")
        right = np.searchsorted(dims_sorted, src_tid, side="right")
        event_idx = dim_order[left:right]
        source_times = train_abs_times_arr[event_idx]
        source_combo = train_src_combo_arr[event_idx]
        source_pre_combo = train_pre_combo_arr[event_idx]
        prefix8 = _source_combo_prefix(source_combo)
        cached = (source_times, source_combo, source_pre_combo, prefix8)
        source_cache[src_tid] = cached
        return cached

    n_groups = len(starts)
    last_beat = time.monotonic()
    t0 = last_beat
    for gi, (start, end, src_tid, ne_id) in enumerate(zip(starts, ends, group_src, group_ne), start=1):
        source_times, source_combo, source_pre_combo, prefix8 = source_stats(int(src_tid))
        if len(source_times) == 0:
            continue
        rows = row_order[start:end]
        tgt_ne = ne_labels[int(ne_id)]
        # Same-NE exposure uses each source event's pre-state so the source
        # event's own raise is not included in the sampled target state. Since
        # alarm_source is part of the type label, all events for src_tid share
        # the same source NE, making this an O(n_src) path once per source type.
        if tgt_ne and str(type_ne[int(src_tid)] or "") == tgt_ne:
            combo_idx = source_combo.astype(np.uint8) * np.uint8(8) + source_pre_combo
            counts = np.bincount(combo_idx.astype(np.int64), minlength=64).astype(np.float32, copy=False)
        else:
            if tgt_ne not in timeline_cache:
                timeline_cache[tgt_ne] = _combo_arrays_from_timeline(state_timeline, tgt_ne)
            target_times, target_combo = timeline_cache[tgt_ne]
            # Interval-prefix counting is exact and usually much cheaper than
            # querying target state at every source event. For extremely chattery
            # targets, fall back to the vectorized event-time lookup.
            if target_times.size and target_times.size > max(256, len(source_times) // 4):
                tgt_combo = _timeline_combos_at_many(
                    state_timeline,
                    tgt_ne,
                    source_times,
                    timeline_cache,
                )
                combo_idx = source_combo.astype(np.uint8) * np.uint8(8) + tgt_combo
                counts = np.bincount(combo_idx.astype(np.int64), minlength=64).astype(np.float32, copy=False)
            else:
                counts = _counts64_from_target_intervals(
                    source_times,
                    prefix8,
                    target_times,
                    target_combo,
                )
        nz = np.flatnonzero(counts > 0)
        if as_coo:
            append_coo(rows, nz, counts)
        else:
            for k in nz:
                exposure[rows, k] += counts[k]

        if verbose:
            now = time.monotonic()
            if now - last_beat >= 10.0:
                print(
                    f"[train] dynamic_alpha={mode_label}: exposure groups "
                    f"{gi}/{n_groups} ({100.0 * gi / max(n_groups, 1):.1f}%, "
                    f"{_fmt_secs(now - t0)})",
                    flush=True,
                )
                last_beat = now
    if as_coo:
        coo_rows.resize(coo_size, refcheck=False)
        coo_cols.resize(coo_size, refcheck=False)
        coo_vals.resize(coo_size, refcheck=False)
        return {
            "format": "coo",
            "shape": (int(C), 64),
            "rows": coo_rows,
            "combos": coo_cols,
            "values": coo_vals,
        }
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
    run_args: Optional[dict] = None,
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

    domain_filter_stats = {"enabled": False}
    if "device_domain" in tuple(config.type_fields):
        if ne_graph_data is None:
            raise ValueError(
                "device_domain in type_fields requires ne_graph_data so unsupported "
                "device domains can be filtered consistently"
            )
        from alarm_flow_isahp.event_domain import filter_and_annotate_device_domain

        sorted_alarm_events, domain_filter_stats = filter_and_annotate_device_domain(
            sorted_alarm_events, ne_graph_data
        )
        _emit_progress(progress_callback, "domain_filter", **domain_filter_stats)

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
    val_src_combo = None
    val_tgt_combo = None
    dyn_combo_bits = None
    dyn_exposure_2d = None
    val_dyn_exposure_2d = None
    dyn_feature_names = None
    if config.edge_mode == "feature" and config.dynamic_alpha != "off":
        from alarm_flow_mhp.dynamic_state import (
            ObservedStateTimeline,
            build_event_states,
            states_to_combo,
            combo_bits as _combo_bits,
        )
        from alarm_flow_mhp.feature_spec import (
            parse_label_entity_at,
            runtime_ne_at,
        )
        ev_state_full = build_event_states(
            sorted_alarm_events,
            sequence.events,
            is_clear=lambda e: is_clear_alarm(e.get("alarm", {})),
            device_of=lambda e: runtime_ne_at(e, config.type_fields, config.topology_node_field)[0],
            alarm_type_of=lambda e: alarm_type_from_title(e.get("alarm_title", "")),
        )
        combo_full = states_to_combo(ev_state_full)
        train_state_combo = combo_full[: train_events.n]
        val_state_combo = (
            combo_full[train_events.n:train_events.n + val_events.n]
            if val_events is not None else None
        )
        base_combo_bits = _combo_bits(8)
        if config.dynamic_alpha == "source_target":
            train_src_combo = train_state_combo
            train_tgt_combo = train_state_combo.copy()
            dyn_combo_bits = _source_target_combo_bits(base_combo_bits)
            dyn_feature_names = [
                "src_uncleared_link",
                "src_uncleared_power",
                "src_uncleared_offline",
                "tgt_uncleared_link",
                "tgt_uncleared_power",
                "tgt_uncleared_offline",
            ]
        elif config.dynamic_alpha == "target":
            # fit_mhp_feature's target-aware representation is
            # source_combo*8 + target_combo. Pinning source_combo to zero gives
            # target-only occurrence/exposure buckets while learning only the
            # three target-state weights.
            train_src_combo = np.zeros_like(train_state_combo)
            train_tgt_combo = train_state_combo.copy()
            dyn_combo_bits = _target_combo_bits(base_combo_bits)
            dyn_feature_names = [
                "tgt_uncleared_link",
                "tgt_uncleared_power",
                "tgt_uncleared_offline",
            ]
        else:
            train_src_combo = train_state_combo
            dyn_combo_bits = base_combo_bits
            dyn_feature_names = ["src_uncleared_link", "src_uncleared_power", "src_uncleared_offline"]
        if val_state_combo is not None:
            if config.dynamic_alpha == "target":
                val_src_combo = np.zeros_like(val_state_combo)
                val_tgt_combo = val_state_combo.copy()
            elif config.dynamic_alpha == "source_target":
                val_src_combo = val_state_combo
                val_tgt_combo = val_state_combo.copy()
            else:
                val_src_combo = val_state_combo
        if verbose:
            nz = int((train_state_combo > 0).sum())
            role = "target" if config.dynamic_alpha == "target" else "source"
            msg = (
                f"[train] dynamic_alpha={config.dynamic_alpha}: {role} marks on "
                f"{nz}/{train_events.n} train events ({100.0*nz/max(train_events.n,1):.1f}% with active state)"
            )
            if train_tgt_combo is not None:
                tnz = int((train_tgt_combo > 0).sum())
                if config.dynamic_alpha != "target":
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

    # Live held-out val LL → model selection + early stop. EM ascends train LL
    # monotonically, so best_callback fires ~every iter; we eval val LL on each
    # such snapshot (one forward LL pass over the val tail, no M-step) and keep
    # the snapshot at the val-LL peak. This makes the saved model the
    # generalizing one instead of the most train-overfit one. With val_split=0
    # the tracking is inert and behavior is unchanged (every train-best written).
    val_ll_history: list[float] = []
    val_trace: list[dict] = []
    val_state = {
        "best_val_ll": -np.inf,
        "best_val_iter": -1,
        "best_snapshot": None,     # val-LL peak among ELIGIBLE snapshots (ρ-gated if cap on)
        "patience_left": int(config.early_stop_patience),
        # val-LL peak ignoring the ρ gate — fallback for the spectral-cap rescale
        # when no snapshot satisfies ρ ≤ stability_radius.
        "any_val_ll": -np.inf,
        "any_snapshot": None,
    }

    def iter_callback(trace_entry: dict):
        # Per-iter diagnostics trace. The held-out val LL itself is evaluated in
        # write_best_checkpoint (the best_callback), which receives the
        # materialized (α, β, μ) snapshot; here we record the train-side
        # trajectory + ρ so the artifact carries the full per-iter history.
        val_trace.append({
            "iter": trace_entry["iter"],
            "active_edges": trace_entry["active_edges"],
            "log_likelihood_train": trace_entry["log_likelihood"],
            "spectral_radius": trace_entry.get("spectral_radius"),
            "spectral_radius_kind": trace_entry.get(
                "spectral_radius_kind", "baseline"
            ),
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
        from alarm_flow_mhp.feature_spec import build_node_context

        # device mode → NE-keyed context; site mode → site-keyed context whose
        # per-site attributes are aggregated from the NE graph.
        feature_graph_context = build_node_context(ne_graph_data, config.topology_node_field)

    if config.topology_prior_boost > 0 and topology_index is not None:
        topo_prior_flat, topo_prior_score = build_topology_pairs(
            vocabs,
            config.type_fields,
            topology_index,
            max_hops=config.topology_prior_max_hops,
            min_score=config.topology_prior_min_score,
            node_field=config.topology_node_field,
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
    feat_domain_vocab = []
    feat_node_domains = {}

    def write_best_checkpoint(best_result: MHPResult, trace_entry: dict) -> bool:
        # Called by EM whenever train LL hits a new best (≈ every iter, since EM
        # ascends monotonically). Returns True to request early stop (val LL has
        # plateaued); the feature-mode loop honors this signal.
        use_val = val_events is not None and val_events.n > 0
        # Whether the held-out metric drives selection/early-stop, vs. just being
        # printed. "train" (default) → legacy: every train-best is the new best,
        # no early stop; val LL is computed only for the log line.
        select_on_val = use_val and config.selection_metric == "val"
        # Opt-in spectral cap (feature mode): a snapshot is only ELIGIBLE as the
        # deployable best if its ρ ≤ stability_radius. Off → no ρ gate (legacy).
        cap_on = (
            bool(config.feature_spectral_cap)
            and config.edge_mode == "feature"
            and config.stability_radius > 0
        )
        val_ll = None
        improved = True               # train metric → every train-best "improves"
        eligible = True
        if use_val:
            if config.edge_mode == "feature" and config.dynamic_alpha != "off":
                from mhp.em import log_likelihood_feature_dynamic

                val_ll = float(
                    log_likelihood_feature_dynamic(
                        val_events,
                        mhp_config,
                        cand_targets=cand_t,
                        cand_sources=cand_s,
                        cand_phi=phi,
                        kernel=best_result.feature_kernel,
                        mu=best_result.params.mu,
                        src_combo=val_src_combo,
                        tgt_combo=val_tgt_combo,
                        dynamic_combo_bits=dyn_combo_bits,
                        dynamic_exposure_2d=val_dyn_exposure_2d,
                    )
                )
            else:
                from mhp.em import log_likelihood as mhp_ll
                val_ll = float(mhp_ll(val_events, best_result.params, config=mhp_config))
            val_ll_history.append(val_ll)
        any_improved = False
        if select_on_val:
            def _beats(ref):
                # strict improvement w/ relative tol; -inf ref (first eval) → any
                # finite val_ll improves (avoids -inf + 1e-6·inf = NaN → always False).
                if np.isfinite(ref):
                    return val_ll > ref + 1e-6 * abs(ref)
                return bool(np.isfinite(val_ll))
            # Overall val-best (ignores ρ). Drives early-stop (we OPTIMIZE val, so
            # patience tracks val-overall, not the ρ gate) AND is the fallback
            # target for the cap rescale when no ρ-eligible snapshot exists.
            any_improved = _beats(val_state["any_val_ll"])
            if any_improved:
                val_state["any_val_ll"] = val_ll
                val_state["any_snapshot"] = best_result
            # Deployable best: with the cap on, require a REAL stationary model,
            # 0 < ρ ≤ stability_radius. ρ == 0 is the cold-start "no active edges"
            # sentinel — it must NOT qualify, else it locks in an empty model and
            # blocks the rescale fallback.
            _rho = trace_entry.get("spectral_radius")
            eligible = (not cap_on) or (_rho is not None and 0.0 < _rho <= config.stability_radius)
            improved = eligible and _beats(val_state["best_val_ll"])
            if improved:
                val_state["best_val_ll"] = val_ll
                val_state["best_val_iter"] = int(trace_entry.get("iter", -1))
                val_state["best_snapshot"] = best_result
            # Early stop on the optimized quantity (val overall), not the ρ gate.
            if any_improved:
                val_state["patience_left"] = int(config.early_stop_patience)
            else:
                val_state["patience_left"] -= 1
        if use_val and verbose:
            _it = int(trace_entry.get("iter", -1))
            _rho = trace_entry.get("spectral_radius")
            _rho_s = f" ρ={_rho:.3f}" if _rho is not None else ""
            if not select_on_val:
                _flag = "(train-selected)"
            elif improved:
                _flag = "↑best"
            elif cap_on and not eligible:
                _flag = f"ρ>{config.stability_radius:g} excluded (patience {max(val_state['patience_left'], 0)})"
            else:
                _flag = f"no-improve (patience {max(val_state['patience_left'], 0)})"
            print(
                f"[train]   iter={_it:3d} val_ll={val_ll:.2f}{_rho_s} "
                f"(train_ll={best_result.log_likelihood:.2f}) {_flag}",
                flush=True,
            )

        # File checkpoint: when selecting on val, persist only the val-improving
        # snapshot (don't overwrite a generalizing model with a later overfit
        # one). Otherwise persist every train-best (legacy behavior).
        if best_checkpoint_path and improved:
            metadata = {
                "checkpoint": True,
                "checkpoint_kind": (
                    "best_val_log_likelihood" if select_on_val else "best_train_log_likelihood"
                ),
                "checkpoint_iter": int(trace_entry.get("iter", best_result.iterations_run - 1)),
                "checkpoint_metric": (
                    "val_log_likelihood" if select_on_val else "train_log_likelihood"
                ),
                "considered_event_count": considered_event_count,
                "region_filter": region_filter_stats,
                "domain_filter": domain_filter_stats,
                "sequence_stats": sequence_stats,
                "modeled_event_count": train_events.n + (val_events.n if val_events else 0),
                "train_event_count": train_events.n,
                "val_event_count": (val_events.n if val_events else 0),
                "type_count": M,
                "active_edge_count": len(best_result.params.edge_alpha),
                "spectral_radius": trace_entry.get("spectral_radius"),
                "spectral_radius_kind": trace_entry.get(
                    "spectral_radius_kind", "baseline"
                ),
                "stability_radius": float(config.stability_radius),
                "run_args": dict(run_args) if run_args else None,
                "best_log_likelihood": float(best_result.log_likelihood),
                "best_val_log_likelihood": val_ll,
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
                    _build_feature_runtime(best_result, vocabs, config, feat_at_vocab, feat_mu_spec,
                                           domain_vocab=feat_domain_vocab, node_domains=feat_node_domains)
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
                _vl = f", val_ll={val_ll:.2f}" if use_val else ""
                print(
                    f"[train] best checkpoint updated: {best_checkpoint_path} "
                    f"(iter={metadata['checkpoint_iter']}, ll={best_result.log_likelihood:.2f}{_vl})",
                    flush=True,
                )

        # Early stop: only when selecting on val and val LL (overall) has not
        # improved for `early_stop_patience` snapshots. Train selection never
        # early-stops.
        if select_on_val and val_state["patience_left"] <= 0:
            if verbose:
                if val_state["best_snapshot"] is not None:
                    _msg = (f"best val_ll={val_state['best_val_ll']:.2f} "
                            f"@ iter {val_state['best_val_iter']}")
                else:
                    # cap on + no ρ≤target snapshot: best_val_ll is still -inf, so
                    # report the overall val-best (it gets spectral-capped at the end).
                    _msg = (f"no ρ≤{config.stability_radius:g} snapshot; overall "
                            f"val_ll={val_state['any_val_ll']:.2f} (will be spectral-capped)")
                print(f"[train]   val LL plateaued → early stop ({_msg})", flush=True)
            return True
        return False

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

        cand_t, cand_s, phi, feat_names, feat_at_vocab, feat_type_group, cand_topo_score, feat_domain_vocab = build_candidate_features(
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
            node_field=config.topology_node_field,
        )
        feat_node_domains = dict(getattr(feature_graph_context, "node_domains", {}) or {})
        # Parameterized inductive μ: ψ(u) single-type features (alarm_type +
        # ne_type/vendor/domain from the NE graph), μ=softplus(w_μ·ψ).
        mu_phi, mu_spec = build_mu_features(
            vocabs, config.type_fields, feature_graph_context, node_field=config.topology_node_field
        )
        feat_mu_spec = mu_spec
        if verbose:
            print(
                f"[train] feature mode: {len(cand_t)} candidate pairs, {phi.shape[1]} features "
                f"{feat_names}",
                flush=True,
            )
        cand_t, cand_s, phi, cand_topo_score = _presort_feature_candidates_for_em(
            cand_t,
            cand_s,
            phi,
            cand_topo_score,
            M,
            verbose=verbose,
        )
        if config.dynamic_alpha in {"target", "source_target"}:
            state_timeline = ObservedStateTimeline()
            for ev in sorted(sorted_alarm_events, key=lambda e: float(e.get("ts", 0.0))):
                ne, _ = runtime_ne_at(ev, config.type_fields, config.topology_node_field)
                state_timeline.ingest(
                    float(ev.get("ts", 0.0)),
                    ne,
                    alarm_type_from_title(ev.get("alarm_title", "")),
                    is_clear_alarm(ev.get("alarm", {})),
                )
            type_ne = np.asarray(
                [
                    parse_label_entity_at(
                        label, config.type_fields, config.topology_node_field
                    )[0]
                    for label in vocabs.type_vocab.labels
                ],
                dtype=object,
            )
            train_event_objs = sequence.events[: train_events.n]
            train_event_ne = [runtime_ne_at(ev, config.type_fields, config.topology_node_field)[0] for ev in train_event_objs]
            # State timelines use raw event timestamps. sequence.times are
            # rebased/scaled model times and must never be used for this lookup.
            train_abs_times = [float(ev.get("ts", 0.0)) for ev in train_event_objs]
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
                dynamic_mode=config.dynamic_alpha,
                verbose=verbose,
                as_coo=True,
            )
            if verbose:
                if isinstance(dyn_exposure_2d, dict) and dyn_exposure_2d.get("format") == "coo":
                    nonzero = int(len(dyn_exposure_2d.get("values", ())))
                    total_slots = int(dyn_exposure_2d["shape"][0] * dyn_exposure_2d["shape"][1])
                else:
                    nonzero = int((dyn_exposure_2d > 0).sum())
                    total_slots = int(dyn_exposure_2d.size)
                print(
                    f"[train] dynamic_alpha={config.dynamic_alpha}: B-fast exposure buckets "
                    f"nonzero={nonzero}/{total_slots}",
                    flush=True,
                )
            if not (isinstance(dyn_exposure_2d, dict) and dyn_exposure_2d.get("format") == "coo"):
                # Dense fallback: hand ownership to fit via a holder so it can
                # release GiBs after converting to sparse COO.
                dyn_exposure_2d = [dyn_exposure_2d]
            if val_events is not None and val_events.n > 0:
                val_event_objs = sequence.events[
                    train_events.n:train_events.n + val_events.n
                ]
                val_event_ne = [
                    runtime_ne_at(ev, config.type_fields, config.topology_node_field)[0]
                    for ev in val_event_objs
                ]
                val_abs_times = [float(ev.get("ts", 0.0)) for ev in val_event_objs]
                val_dyn_exposure_2d = _build_source_target_dynamic_exposure(
                    val_events,
                    cand_t,
                    cand_s,
                    type_ne=type_ne,
                    train_event_ne=val_event_ne,
                    train_abs_times=val_abs_times,
                    train_src_combo=val_src_combo,
                    train_pre_combo=val_tgt_combo,
                    state_timeline=state_timeline,
                    dynamic_mode=config.dynamic_alpha,
                    verbose=verbose,
                    as_coo=True,
                )
        result = fit_mhp_feature(
            train_events,
            mhp_config,
            cand_targets=cand_t,
            cand_sources=cand_s,
            cand_phi=phi,
            feature_names=feat_names,
            l2=config.feature_l2,
            l2_normalize=config.feature_l2_normalize,
            mu_phi=mu_phi,                       # inductive parameterized μ
            mu_feature_names=mu_spec.feature_names,
            cand_topo_score=cand_topo_score,     # topology pseudo-count prior
            topo_prior_boost=config.feature_topo_prior_boost,
            src_combo=train_src_combo,            # dynamic stateful α (target mode pins this to zero)
            tgt_combo=train_tgt_combo,            # target-aware modes: target pre-state mark
            dynamic_combo_bits=dyn_combo_bits,
            dynamic_exposure_2d=dyn_exposure_2d,
            dynamic_feature_names=dyn_feature_names,
            iter_callback=iter_callback,
            best_callback=write_best_checkpoint,
        )
        # Stationarity is reported below, AFTER val-selection + the optional
        # spectral cap, so it reflects the actually-deployed model (not the
        # train-final, which selection may replace).
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

    from mhp.em import log_likelihood as mhp_ll
    cap_on = (
        bool(config.feature_spectral_cap)
        and config.edge_mode == "feature"
        and config.stability_radius > 0
    )

    # 1) Model selection: prefer the val-LL-optimal snapshot (captured live during
    #    EM) over the train-final weights — the point of holding out data. With the
    #    cap on and no ρ-eligible snapshot, fall back to the val-best overall (it is
    #    rescaled in step 2). Keep the run's trace/iteration fields; swap params.
    if val_events is not None and val_events.n > 0:
        snap = val_state["best_snapshot"]
        if snap is None and cap_on:
            snap = val_state["any_snapshot"]
        if snap is not None and snap is not result:
            result = replace(
                result,
                params=snap.params,
                log_likelihood=snap.log_likelihood,
                p_self=snap.p_self,
                feature_kernel=snap.feature_kernel,
                mu_kernel=snap.mu_kernel,
            )
            if verbose and val_state["best_snapshot"] is not None:
                print(
                    f"[train] selected val-best snapshot from iter "
                    f"{val_state['best_val_iter']}: val_ll={float(val_state['best_val_ll']):.2f} "
                    f"(train_ll={result.log_likelihood:.2f})",
                    flush=True,
                )

    def _feature_state_envelope_radius(kernel) -> float:
        """ρ of the entrywise max over all dynamic state combinations."""
        from mhp.em import _spectral_radius_edges
        from mhp.feature_kernel import softplus

        weights = np.asarray(kernel.weights, dtype=np.float64)
        n_static = int(phi.shape[1])
        logits = np.asarray(phi @ weights[:n_static], dtype=np.float64)
        if dyn_combo_bits is not None and weights.size > n_static:
            dynamic_logits = np.asarray(dyn_combo_bits, dtype=np.float64) @ weights[n_static:]
            logits += float(np.max(dynamic_logits))
        alpha = softplus(logits) * float(getattr(kernel, "alpha_scale", 1.0))
        keep = alpha > config.edge_threshold
        if not keep.any():
            return 0.0
        return float(
            _spectral_radius_edges(
                np.asarray(cand_t)[keep],
                np.asarray(cand_s)[keep],
                alpha[keep],
                M,
            )
        )

    # 2) Spectral cap (opt-in, feature mode). Idempotent: a no-op when the selected
    #    model already has ρ ≤ stability_radius (the common case — B picked a stable
    #    snapshot). Otherwise rescale α by stability_radius/ρ (ρ(c·A)=c·ρ(A), exact;
    #    relative edge structure preserved) and bake the scale into the kernel so
    #    inference — which recomputes α from w — applies the same cap. Works for any
    #    selection metric (incl. train) and the no-val case.
    if cap_on and result.feature_kernel is not None:
        rho = _feature_state_envelope_radius(result.feature_kernel)
        if rho > config.stability_radius:
            scale = config.stability_radius / rho
            capped_kernel = result.feature_kernel
            if capped_kernel is not None:
                capped_kernel = replace(capped_kernel, alpha_scale=capped_kernel.alpha_scale * scale)
            result = replace(
                result,
                params=MHPParams.from_edges(
                    M=len(result.params.mu),
                    mu=result.params.mu,
                    edge_targets=result.params.edge_targets,
                    edge_sources=result.params.edge_sources,
                    edge_alpha=result.params.edge_alpha * scale,
                    edge_beta=result.params.edge_beta,
                    edge_threshold=config.edge_threshold,
                    max_active_sources_per_dim=config.max_active_sources_per_dim,
                    beta_shared=True,
                ),
                feature_kernel=capped_kernel,
            )
            if verbose:
                capped_rho = _feature_state_envelope_radius(result.feature_kernel)
                print(
                    f"[train] WARN: spectral cap — no model reached ρ ≤ "
                    f"{config.stability_radius:g}; rescaled α×{scale:.3f} "
                    f"(state-envelope ρ {rho:.3f} → {capped_rho:.3f}). Relative "
                    f"structure kept; a denser-graph config (e.g. NE topology) may be the "
                    f"deeper issue.",
                    flush=True,
                )

    # 3) Stationarity report on the FINAL (selected + capped) model. Use a strict
    #    >-with-eps threshold: the cap rescales to ρ == stability_radius exactly, so
    #    a plain >= would mis-fire the WARN on a successfully-capped model.
    final_feature_rho = None
    if config.edge_mode == "feature" and result.feature_kernel is not None:
        final_feature_rho = _feature_state_envelope_radius(result.feature_kernel)
    if config.edge_mode == "feature" and config.stability_radius > 0:
        rho = float(final_feature_rho or 0.0)
        _eps = 1e-9
        if rho > config.stability_radius + _eps:
            if verbose:
                # With the cap on, ρ > target is only reachable via numerical
                # residue; don't tell the user to enable a flag that's already on.
                _hint = ("(numerical residual after spectral cap)" if cap_on
                         else "Raise --feature-l2, or enable --feature-spectral-cap to enforce it.")
                print(
                    f"[train] WARN: feature-mode α state-envelope spectral radius ρ={rho:.3f} > "
                    f"{config.stability_radius}. {_hint}",
                    flush=True,
                )
        elif verbose:
            _capped = " [spectral cap applied]" if (cap_on and result.feature_kernel is not None
                                                    and result.feature_kernel.alpha_scale != 1.0) else ""
            print(
                f"[train] feature-mode α state-envelope spectral radius "
                f"ρ={rho:.3f} (≤ {config.stability_radius}, OK){_capped}",
                flush=True,
            )

    final_val_ll: Optional[float] = None
    if val_events is not None and val_events.n > 0:
        if config.edge_mode == "feature" and config.dynamic_alpha != "off":
            from mhp.em import log_likelihood_feature_dynamic

            final_val_ll = float(
                log_likelihood_feature_dynamic(
                    val_events,
                    mhp_config,
                    cand_targets=cand_t,
                    cand_sources=cand_s,
                    cand_phi=phi,
                    kernel=result.feature_kernel,
                    mu=result.params.mu,
                    src_combo=val_src_combo,
                    tgt_combo=val_tgt_combo,
                    dynamic_combo_bits=dyn_combo_bits,
                    dynamic_exposure_2d=val_dyn_exposure_2d,
                )
            )
        else:
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
        node_field=config.topology_node_field,
    )
    topology_report_seconds = time.monotonic() - t0
    if topology_report is not None and verbose:
        print(f"[train] topology consistency report done in {topology_report_seconds:.1f}s", flush=True)

    # Metric 3 (piecewise only): per-bucket excitation mass distribution
    bucket_mass = _bucket_mass_distribution(result.params, config.bucket_edges_sec)

    training_metadata = {
        "considered_event_count": considered_event_count,
        "region_filter": region_filter_stats,
        "domain_filter": domain_filter_stats,
        "sequence_stats": sequence_stats,
        "modeled_event_count": train_events.n + (val_events.n if val_events else 0),
        "train_event_count": train_events.n,
        "val_event_count": (val_events.n if val_events else 0),
        "type_count": M,
        "active_edge_count": len(result.params.edge_alpha),
        # Stationarity: feature-mode dynamic kernels store the entrywise
        # all-state envelope ρ; other modes store the baseline α-matrix ρ.
        # Persisted so it is readable without re-running training.
        "spectral_radius": (
            final_feature_rho
            if config.edge_mode == "feature"
            else (float(result.params.spectral_radius()) if len(result.params.edge_alpha) else None)
        ),
        "spectral_radius_kind": (
            "dynamic_state_envelope"
            if config.edge_mode == "feature" and config.dynamic_alpha != "off"
            else "baseline"
        ),
        "stability_radius": float(config.stability_radius),
        "run_args": dict(run_args) if run_args else None,
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
            _build_feature_runtime(result, vocabs, config, feat_at_vocab, feat_mu_spec,
                                   domain_vocab=feat_domain_vocab, node_domains=feat_node_domains)
            if config.edge_mode == "feature"
            else None
        ),
        "val_trace": val_trace,
        "hard_cascade_seconds": float(hard_cascade_seconds),
        "topology_report_seconds": float(topology_report_seconds),
    }
    final_artifact = AlarmMHPArtifact(
        params=result.params,
        vocabs=vocabs,
        config=config,
        training_metadata=training_metadata,
        trace=result.trace,
    )
    # Finalize the best checkpoint to the FINAL selected+capped model. Mid-training
    # checkpoints wrote raw snapshots (before val-selection swap + spectral cap), so
    # the sidecar could otherwise be uncapped — or absent, when the cap fell back to
    # rescaling and no ρ-eligible snapshot was ever written. Overwriting here keeps
    # the sidecar on the same final selected/capped model as --output (metadata is
    # not bit-identical — see the log note below).
    if best_checkpoint_path:
        tmp_path = f"{best_checkpoint_path}.tmp"
        save_alarm_mhp_artifact(tmp_path, final_artifact)
        os.replace(tmp_path, best_checkpoint_path)
        if verbose:
            _capped = (result.feature_kernel is not None
                       and getattr(result.feature_kernel, "alpha_scale", 1.0) != 1.0)
            # The MODEL matches --output; metadata is not bit-identical (the CLI adds
            # input/alarm_metadata to the --output artifact afterwards, and the
            # mid-training checkpoint=True markers are gone). Those fields are
            # final-only by design / unconsumed, so only claim model parity.
            print(
                f"[train] best checkpoint finalized → {best_checkpoint_path} "
                f"(final selected{'+capped' if _capped else ''} model; "
                f"same model as --output)",
                flush=True,
            )
    return final_artifact


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


def _build_feature_runtime(result, vocabs, config, at_vocab, mu_spec=None,
                           domain_vocab=None, node_domains=None):
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
        # φ domain vocab (site×domain mode); empty in device/NE mode → legacy φ.
        "domain_vocab": list(domain_vocab or []),
        # topo node → present device domains, for missing-parent candidate
        # enumeration at inference (site×domain mode).
        "node_domains": {str(k): list(v) for k, v in (node_domains or {}).items()},
        "topology_node_field": config.topology_node_field,
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


def build_topology_pairs(vocabs, type_fields, topology_index, *, max_hops, min_score, node_field="alarm_source"):
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
        src_field_idx = tuple(type_fields).index(node_field)
    except ValueError:
        # No topology-node field in the type → topology prior is meaningless
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
    node_field: str = "alarm_source",
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
        src_field_idx = type_fields.index(node_field)
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
