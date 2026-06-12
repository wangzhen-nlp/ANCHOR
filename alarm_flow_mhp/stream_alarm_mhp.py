#!/usr/bin/env python3
"""Online (streaming) inference using a trained MHP artifact.

Consumes alarms in time order from a prepare_sorted_alarms cache (or any
sorted source) and emits fault groups as cascades close. The inference loop
maintains a sliding-window of recent real events, scores each new alarm
against candidate parents using the trained α, β, μ, and either binds the
alarm to an existing cascade (most-likely parent) or starts a new cascade
(immigrant).

Differences from alarm_flow_brunch.stream_alarm_brunch:

  - No MCMC at inference time. The trained Θ is applied directly via a
    single per-event argmax — fast and deterministic.
  - No virtual events / latent missing mode (v1 omission; can layer on
    later once the core path is validated).
  - Output schema is identical so downstream visualizers and report
    pipelines can consume either source.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from dataclasses import replace as _replace
import json
import os
import sys
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

import numpy as np

from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_brunch.visual_output import AlarmBRUNCHVisualOutputSession
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.sequences import alarm_type_label, event_type_label
from alarm_flow_mhp.aggregator import (
    AlarmMHPConfig,
    load_alarm_mhp_artifact,
    summarize_alarm_event,
)
from alarm_flow_mhp.topology_relation_prior import (
    format_topology_relation_prior,
    parse_topology_relation_prior,
    topology_relation_weights,
)
from fault_grouping.alarm_events.io import is_clear_alarm
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON, resource_display


EPS = 1e-12


def _resolve_visual_snapshot_check_interval(age_sec: float, configured_sec) -> float:
    """Stream-time cadence for visual snapshot scans.

    The snapshot scan walks live cascades/clusters, so the default is deliberately
    not per-event once age snapshots are enabled.
    """
    if age_sec <= 0:
        return 0.0
    if configured_sec is not None and float(configured_sec) > 0:
        return float(configured_sec)
    return min(max(float(age_sec) / 6.0, 1.0), 60.0)


# --------------------------------------------------------------------------
# Core data structures
# --------------------------------------------------------------------------


@dataclass
class OnlineEvent:
    """One event observed by the streaming pipeline."""

    index: int                  # global ordinal in the stream
    ts: float                   # event time (epoch seconds)
    type_id: int                # vocab type ID (-1 in feature mode if unseen)
    type_label: str             # human-readable type
    alarm: dict                 # original alarm event dict (for output)
    parent_index: int = -1      # -1 for immigrant; otherwise the OnlineEvent.index of parent
    cascade_id: int = -1        # assigned cascade ID
    parent_score: float = 0.0   # score that won (for debugging / output)
    alarm_type: str = ""        # feature mode: link/power/offline
    ne: str = ""                # feature mode: alarm_source (device id)
    src_mark: tuple = (0, 0, 0) # dynamic α: source device's uncleared (link,power,
                                # offline) booleans frozen at THIS event's fire time
                                # (excl self) — used when this event is a parent
    finalized: bool = False     # parent assignment is frozen after the slack lag


@dataclass
class Cascade:
    """A connected component of (parent, child) links."""

    cascade_id: int
    events: list = field(default_factory=list)        # list of OnlineEvent
    root_index: int = -1                              # index of root event
    last_ts: float = 0.0
    start_ts: float = 0.0
    snapshot_seq: int = 0
    last_snapshot_event_indexes: set = field(default_factory=set)
    snapshot_frontier_uuids: set = field(default_factory=set)
    snapshot_related_uuids: set = field(default_factory=set)

    def add(self, event: OnlineEvent):
        if any(e.index == event.index for e in self.events):
            return
        if not self.events:
            self.root_index = event.index
            self.start_ts = event.ts
        else:
            self.start_ts = min(self.start_ts, event.ts)
        self.events.append(event)
        self.last_ts = max(self.last_ts, event.ts)
        event.cascade_id = self.cascade_id

    def event_count(self) -> int:
        return len(self.events)


# --------------------------------------------------------------------------
# Streaming assigner: the inference engine
# --------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Inference-time knobs (mostly inherited from artifact.config)."""

    history_window_sec: float = 900.0       # sliding window for candidate parents
    time_slack_sec: float = 0.0             # fixed-lag timestamp jitter tolerance
    late_penalty_half_life_sec: float = 1.0 # late-parent discount half-life
    max_history_events: int = 256           # cap candidates per event
    time_scale_sec: float = 60.0            # convert real seconds to model time
    close_inactive_sec: float = 7200.0      # cascades quiet for this long are closed
    min_group_events: int = 1               # filter on output
    immigrant_bias: float = 1.0             # multiplier on μ at scoring time
                                            # (>1 → prefer immigrant; <1 → prefer cascade)
    feature_alpha_floor: float = 0.0        # feature mode: candidate edges with
                                            # live α below this are treated as
                                            # non-edges (score 0) — the inference
                                            # analog of device-mode edge_threshold,
                                            # guards against soft over-connection.
    topology_relation_prior: dict = field(default_factory=dict)

    @classmethod
    def from_artifact_config(cls, cfg: AlarmMHPConfig, **overrides):
        base = cls(
            history_window_sec=cfg.history_window_sec,
            time_slack_sec=getattr(cfg, "time_slack_sec", 0.0),
            late_penalty_half_life_sec=getattr(cfg, "late_penalty_half_life_sec", 1.0),
            max_history_events=cfg.max_history_events,
            time_scale_sec=cfg.time_scale_sec,
            min_group_events=cfg.min_group_events,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(base, k, v)
        base.validate()
        return base

    def validate(self):
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be > 0")
        if self.time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if self.late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        if self.max_history_events < 1:
            raise ValueError("max_history_events must be >= 1")
        if self.time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be > 0")
        if self.close_inactive_sec < 0:
            raise ValueError("close_inactive_sec must be >= 0")
        if self.min_group_events < 1:
            raise ValueError("min_group_events must be >= 1")
        if self.immigrant_bias < 0:
            raise ValueError("immigrant_bias must be >= 0")
        if self.feature_alpha_floor < 0:
            raise ValueError("feature_alpha_floor must be >= 0")
        for key, value in (self.topology_relation_prior or {}).items():
            val = float(value)
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"topology relation prior {key} must be finite and >= 0")


class StreamMHPAssigner:
    """Stateful online inference engine.

    Maintains:
      - `_buf_*`: parallel arrays (events/ts/type) + head pointer holding real
        events still within history_window, for vectorized candidate scoring
      - `cascades`: dict cascade_id → Cascade for cascades still considered active
      - vocabs for label → id resolution

    Per incoming alarm:
      1. Append the event to the recent buffer and pending queue
      2. Advance a watermark at now - time_slack_sec
      3. Finalize pending events behind the watermark by scoring candidate
         parents in [target-W, target+slack]
      4. Compare against μ[u_target] · immigrant_bias
      5. argmax → bind/merge with the parent's cascade, or mark immigrant
    """

    def __init__(self, artifact, config: StreamConfig, feature_scorer=None, mu_scorer=None):
        self.artifact = artifact
        self.params = artifact.params
        self.vocabs = artifact.vocabs
        self.config = config
        # Feature mode: live α = softplus(w·φ) per candidate, device-OPEN
        # (new devices not dropped). feature_scorer / mu_scorer are built by
        # main() from the artifact + NE graph.
        self.feature_mode = getattr(artifact.config, "edge_mode", "device") == "feature"
        self.feature_scorer = feature_scorer
        self.mu_scorer = mu_scorer            # live parameterized μ (or None → fallback table)
        self.config.validate()
        if self.feature_mode:
            if self.feature_scorer is None:
                raise ValueError("feature-mode artifact requires a RuntimeFeatureScorer")
            rt = (artifact.training_metadata or {}).get("feature_runtime") or {}
            self._mu_by_at = rt.get("mu_by_alarm_type", {}) or {}
            self._mu_default = float(rt.get("mu_default", 0.0))
            self._feat_beta = float(rt.get("beta", 1.0))
            # α floor: CLI override else fall back to the trained edge_threshold
            # (so feature inference prunes weak edges like device mode does).
            floor = getattr(config, "feature_alpha_floor", 0.0)
            if floor <= 0:
                floor = float(getattr(artifact.config, "edge_threshold", 0.0))
            self._feat_alpha_floor = float(floor)
            self._topology_relation_prior = dict(config.topology_relation_prior or {})
            self._topology_index = getattr(self.feature_scorer, "topology_index", None)
            self._node_infos = getattr(self.feature_scorer, "node_infos", {}) or {}
        # Dynamic (stateful) α: track per-device uncleared-alarm state (clear-
        # aware), snapshotting each event's source mark at fire time. Same state
        # machine as training (keyed on parsed feature NE + alarm_type_from_title),
        # so marks are train/infer-consistent.
        self.dynamic_mode = (
            self.feature_mode
            and getattr(artifact.config, "dynamic_alpha", "off") != "off"
        )
        self._state_tracker = None
        self.n_dynamic = 0
        if self.dynamic_mode:
            from alarm_flow_mhp.dynamic_state import DeviceStateTracker
            self._state_tracker = DeviceStateTracker()
            self.n_dynamic = int(getattr(self.feature_scorer, "n_dynamic", 0) or 0)
        # Sliding window of recent events kept as parallel arrays with a head
        # pointer (amortized O(1) append + eviction, O(cap) tail slice for
        # vectorized scoring). `_buf_events` holds the OnlineEvent objects;
        # `_buf_ts` / `_buf_type` are numpy-friendly parallel columns.
        self._buf_events: list[OnlineEvent] = []
        self._buf_ts: list[float] = []
        self._buf_type: list[int] = []
        self._head: int = 0                      # logical front index into the bufs
        self._pending_events: list[OnlineEvent] = []
        self._pending_head: int = 0
        self._events_by_index: dict[int, OnlineEvent] = {}
        self.cascades: dict[int, Cascade] = {}
        self._next_cascade_id: int = 0
        self._next_event_index: int = 0
        # Throttle for _close_inactive: only scan open cascades when event time
        # has advanced at least this far since the last scan (avoids an O(open
        # cascades) sweep on every single event). Scan cadence = a fraction of
        # close_inactive_sec so closures stay timely.
        self._last_close_scan_ts: float = -np.inf
        self._close_scan_interval: float = max(config.close_inactive_sec * 0.1, 1.0)
        # Precompute the sparse kernel lookup (sorted edge keys + per-edge
        # params) once — streaming params are frozen. Mirrors params.pair_score
        # but vectorized over a batch of candidates (f64, matching pair_score).
        M = self.params.M
        self._M = M
        et = self.params.edge_targets.astype(np.int64)
        es = self.params.edge_sources.astype(np.int64)
        self._edge_keys = et * M + es            # ascending (from_edges sorted)
        self._E = len(et)
        if self.params.kernel_type == "piecewise":
            self._theta = np.asarray(self.params.edge_theta, dtype=np.float64)
            self._bucket_edges = np.asarray(self.params.bucket_edges, dtype=np.float64)
        else:
            self._edge_alpha = self.params.edge_alpha.astype(np.float64)
            self._edge_beta = self.params.edge_beta.astype(np.float64)
        # Stats for reporting
        self.total_events_processed = 0
        self.total_immigrants = 0
        self.closed_cascade_count = 0
        self.dropped_unknown_type = 0
        self.dropped_clear = 0
        self.dropped_no_type = 0
        # Closed cascade output sink — only cascades passing min_group_events.
        self.closed_groups: list = []
        # Size of EVERY closed cascade (incl. those filtered from output) so the
        # diagnostic size distribution matches training (computed on all
        # cascades, not just the emitted ones). Counter keyed by size keeps
        # this O(distinct sizes) regardless of stream length.
        self.closed_size_counter: Counter = Counter()
        self.emitted_group_count = 0

    def _resolve_type_id(self, alarm_event) -> tuple[int, str] | None:
        """Translate an alarm event to (type_id, label) using artifact vocabs."""
        type_label = event_type_label(alarm_event, self.artifact.config.type_fields)
        type_id = self.vocabs.type_vocab.get(type_label)
        if type_id is None:
            return None
        return type_id, type_label

    def _evict_expired(self, now_ts: float):
        """Advance the head pointer past events older than history_window, then
        compact the buffers periodically so they don't grow unboundedly.
        """
        cutoff = now_ts - self.config.history_window_sec
        head = self._head
        ts = self._buf_ts
        n = len(ts)
        while head < n and ts[head] < cutoff:
            head += 1
        self._head = head
        # Compact when the dead prefix dominates — amortized O(1) per event.
        if head > 4096 and head * 2 > n:
            self._buf_events = self._buf_events[head:]
            self._buf_ts = self._buf_ts[head:]
            self._buf_type = self._buf_type[head:]
            self._head = 0

    def _time_slack_score(self, dts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Training-consistent signed-dt handling in model-time units."""
        dts = np.asarray(dts, dtype=np.float64)
        if dts.size == 0:
            return dts, np.ones(0, dtype=np.float64)
        slack = self.config.time_slack_sec / self.config.time_scale_sec
        dt_eff = np.maximum(dts, 0.0)
        late = np.maximum(-dts, 0.0)
        if slack <= 0:
            weight = (late <= 0).astype(np.float64)
        else:
            lam = np.log(2.0) / (self.config.late_penalty_half_life_sec / self.config.time_scale_sec)
            weight = np.where(late <= slack, np.exp(-lam * late), 0.0)
        return dt_eff, weight

    def _candidate_positions_for(self, target_event: OnlineEvent) -> np.ndarray:
        """Candidate parents in [target-W, target+slack], excluding target.

        The event buffer is time-ordered because inputs are loaded sorted by ts.
        With slack enabled, some candidates are pending/future relative to the
        target and may not have their own parent finalized yet.
        """
        n = len(self._buf_ts)
        live = n - self._head
        if live <= 0:
            return np.empty(0, dtype=np.int64)
        ts_arr = np.asarray(self._buf_ts, dtype=np.float64)
        cap = int(self.config.max_history_events)
        positions: list[int] = []

        def maybe_add(pos: int):
            ev = self._buf_events[int(pos)]
            if ev.index == target_event.index:
                return
            if self._is_descendant(ev, target_event):
                return
            if ev.cascade_id != -1 and ev.cascade_id not in self.cascades:
                return
            positions.append(int(pos))

        left = max(int(np.searchsorted(ts_arr, target_event.ts, side="left")) - 1, self._head - 1)
        right = max(int(np.searchsorted(ts_arr, target_event.ts, side="left")), self._head)
        while len(positions) < cap:
            left_ok = left >= self._head and target_event.ts - ts_arr[left] <= self.config.history_window_sec
            right_ok = right < n and ts_arr[right] - target_event.ts <= self.config.time_slack_sec
            if not left_ok and not right_ok:
                break
            if left_ok and right_ok:
                take_left = (target_event.ts - ts_arr[left]) <= (ts_arr[right] - target_event.ts)
            else:
                take_left = left_ok
            if take_left:
                maybe_add(left)
                left -= 1
            else:
                maybe_add(right)
                right += 1
        return np.asarray(positions, dtype=np.int64)

    def _is_descendant(self, maybe_descendant: OnlineEvent, ancestor: OnlineEvent) -> bool:
        cur = maybe_descendant
        guard = 0
        while cur.parent_index != -1:
            if cur.parent_index == ancestor.index:
                return True
            cur = self._events_by_index.get(cur.parent_index)
            if cur is None:
                return False
            guard += 1
            if guard > self.total_events_processed + 1:
                return True
        return False

    def _score_batch(self, target_type_id: int, src_types: np.ndarray, dts: np.ndarray) -> np.ndarray:
        """Vectorized kernel score for a batch of candidates — sparse binary
        search over sorted edge keys, dispatching on kernel_type. Bit-for-bit
        consistent with training's signed-dt slack scoring.
        """
        n = len(src_types)
        out = np.zeros(n, dtype=np.float64)
        if self._E == 0 or n == 0:
            return out
        keys = int(target_type_id) * self._M + src_types
        idx = np.minimum(np.searchsorted(self._edge_keys, keys), self._E - 1)
        dts_eff, late_weight = self._time_slack_score(dts)
        valid = (self._edge_keys[idx] == keys) & (late_weight > 0)
        if not valid.any():
            return out
        vi = idx[valid]
        if self.params.kernel_type == "piecewise":
            from mhp.params import bucket_index_vec

            pb = bucket_index_vec(dts_eff[valid], self._bucket_edges)
            out[valid] = self._theta[vi, pb] * late_weight[valid]
        else:
            a = self._edge_alpha[vi]
            b = self._edge_beta[vi]
            out[valid] = a * b * np.exp(-b * dts_eff[valid]) * late_weight[valid]
        return out

    def _candidate_scores(self, target_event: OnlineEvent):
        """Score the most recent `max_history_events` candidates in one numpy
        batch.

        Returns (positions, scores) where positions[k] is the ABSOLUTE buffer
        index of candidate k (so the caller can fetch the OnlineEvent), and
        scores[k] is its kernel score against the target.
        """
        positions = self._candidate_positions_for(target_event)
        if positions.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        ts_tail = np.asarray([self._buf_ts[int(p)] for p in positions], dtype=np.float64)
        type_tail = np.asarray([self._buf_type[int(p)] for p in positions], dtype=np.int64)
        dts = (target_event.ts - ts_tail) / self.config.time_scale_sec
        scores = self._score_batch(target_event.type_id, type_tail, dts)
        return positions, scores

    def _candidate_scores_feature(self, target_event: OnlineEvent):
        """Feature-mode candidate scoring: live α = softplus(w·φ) per source
        candidate (device-OPEN), then α·β·exp(-β·Δt). Returns (positions, scores).
        """
        positions = self._candidate_positions_for(target_event)
        if positions.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        src_events = [self._buf_events[int(p)] for p in positions]
        src_ats = [e.alarm_type for e in src_events]
        src_nes = [e.ne for e in src_events]
        ts_tail = np.asarray([e.ts for e in src_events], dtype=np.float64)
        dts = (target_event.ts - ts_tail) / self.config.time_scale_sec
        dts_eff, late_weight = self._time_slack_score(dts)
        # Dynamic α: each candidate parent's frozen source mark (its source
        # device's uncleared state at the parent's fire time).
        src_marks = (
            np.array([e.src_mark for e in src_events], dtype=np.float64)
            if self.n_dynamic > 0 else None
        )
        tgt_marks = (
            np.tile(np.asarray(target_event.src_mark, dtype=np.float64).reshape(1, -1), (len(src_events), 1))
            if self.n_dynamic > 3 else None
        )
        alpha = self.feature_scorer.alpha_for_target(
            target_event.alarm_type,
            target_event.ne,
            src_ats,
            src_nes,
            src_marks=src_marks,
            tgt_marks=tgt_marks,
        )
        # α floor: treat too-weak edges as non-edges (inference analog of
        # device-mode edge_threshold) — guards the soft model against linking
        # unrelated pairs whose baseline α is small but positive.
        if self._feat_alpha_floor > 0:
            alpha = np.where(alpha >= self._feat_alpha_floor, alpha, 0.0)
        b = self._feat_beta
        scores = alpha * b * np.exp(-b * dts_eff) * late_weight
        if self._topology_relation_prior:
            scores *= topology_relation_weights(
                src_nes,
                target_event.ne,
                self._topology_index,
                self._node_infos,
                self._topology_relation_prior,
            )
        return positions, scores

    def _new_cascade(self) -> Cascade:
        cid = self._next_cascade_id
        self._next_cascade_id += 1
        cascade = Cascade(cascade_id=cid)
        self.cascades[cid] = cascade
        return cascade

    def _ensure_cascade(self, event: OnlineEvent) -> Cascade:
        cascade = self.cascades.get(event.cascade_id)
        if cascade is not None:
            cascade.add(event)
            return cascade
        cascade = self._new_cascade()
        cascade.add(event)
        return cascade

    def _merge_cascades(self, keep_id: int, drop_id: int) -> Cascade:
        if keep_id == drop_id and keep_id in self.cascades:
            return self.cascades[keep_id]
        keep = self.cascades.get(keep_id)
        drop = self.cascades.get(drop_id)
        if keep is None and drop is None:
            return self._new_cascade()
        if keep is None:
            return drop
        if drop is None:
            return keep
        for ev in list(drop.events):
            keep.add(ev)
        keep.snapshot_seq = max(keep.snapshot_seq, drop.snapshot_seq)
        keep.last_snapshot_event_indexes |= drop.last_snapshot_event_indexes
        keep.snapshot_frontier_uuids |= drop.snapshot_frontier_uuids
        keep.snapshot_related_uuids |= drop.snapshot_related_uuids
        self.cascades.pop(drop_id, None)
        return keep

    def _assign_parent(self, event: OnlineEvent):
        """Finalize one modeled event's parent assignment."""
        if event.finalized:
            return event
        # Score candidates + immigrant baseline μ. Feature mode is INDUCTIVE:
        # μ = softplus(w_μ·ψ) from the type's own features (parameterized).
        if self.feature_mode:
            positions, scores = self._candidate_scores_feature(event)
            if self.mu_scorer is not None:
                base_mu = self.mu_scorer.mu_for(event.alarm_type, event.ne)
            else:
                base_mu = self._mu_by_at.get(event.alarm_type, self._mu_default)
            mu = base_mu * self.config.immigrant_bias
        else:
            positions, scores = self._candidate_scores(event)
            mu = float(self.params.mu[event.type_id]) * self.config.immigrant_bias

        if scores.size == 0 or scores.max() < mu:
            # Immigrant/root. If this event already has a cascade because earlier
            # children picked it as a future parent, keep that component.
            cascade = self._ensure_cascade(event)
            event.parent_index = -1
            event.parent_score = mu
            self.total_immigrants += 1
        else:
            best_local = int(scores.argmax())
            parent_event = self._buf_events[int(positions[best_local])]
            parent_cascade = self._ensure_cascade(parent_event)
            event_cascade = self._ensure_cascade(event)
            cascade = (
                self._merge_cascades(parent_cascade.cascade_id, event_cascade.cascade_id)
                if parent_cascade.cascade_id != event_cascade.cascade_id
                else parent_cascade
            )
            cascade.add(parent_event)
            cascade.add(event)
            event.parent_index = parent_event.index
            event.parent_score = float(scores[best_local])
        event.finalized = True
        return event

    def _finalize_ready(self, watermark_ts: float):
        """Finalize pending events whose slack horizon is fully observed."""
        while self._pending_head < len(self._pending_events):
            ev = self._pending_events[self._pending_head]
            if ev.ts > watermark_ts:
                break
            self._assign_parent(ev)
            self._pending_head += 1
        if self._pending_head > 4096 and self._pending_head * 2 > len(self._pending_events):
            self._pending_events = self._pending_events[self._pending_head:]
            self._pending_head = 0

    def _advance_watermark(self, now_ts: float):
        watermark = float(now_ts) - self.config.time_slack_sec
        self._finalize_ready(watermark)
        self._close_inactive(watermark)
        self._evict_expired(watermark)

    def process(self, alarm_event: dict):
        """Ingest one alarm event, delay final assignment until slack has passed."""
        alarm_dict = alarm_event.get("alarm", {}) if isinstance(alarm_event, dict) else {}
        ts = float(alarm_event.get("ts", 0.0))
        is_clear = is_clear_alarm(alarm_dict)
        # Dynamic α: feed EVERY event (raises AND clears) to the device-state
        # machine in time order, BEFORE the clear/no-type drops. The returned
        # snapshot (excl self) is this event's frozen source mark, used later if
        # it becomes a candidate parent.
        src_mark = (0, 0, 0)
        if self.dynamic_mode:
            from alarm_flow_isahp.sequences import alarm_type_from_title
            from alarm_flow_mhp.feature_spec import runtime_ne_at

            dev, _ = runtime_ne_at(alarm_event, self.artifact.config.type_fields)
            atype_state = alarm_type_from_title(alarm_event.get("alarm_title", ""))
            snap = self._state_tracker.snapshot_then_apply(dev, atype_state, is_clear)
            src_mark = (int(snap[0]), int(snap[1]), int(snap[2]))
        # Filtering: skip events we can't model
        if is_clear:
            self.dropped_clear += 1
            self._advance_watermark(ts)
            return None
        atype = alarm_type_label(alarm_event)
        if atype is None:
            self.dropped_no_type += 1
            self._advance_watermark(ts)
            return None

        if self.feature_mode:
            # Device-OPEN: accept any event with a known alarm type; the type_id
            # may be unseen (-1) — α comes from features, not a vocab edge.
            # Extract (ne, alarm_type) the SAME way training did — rebuild the
            # type label via type_fields and parse it — so the device id / alarm
            # type fed to the scorer match training's keys exactly (stripping,
            # custom type_fields), not an ad-hoc raw-field read.
            from alarm_flow_mhp.feature_spec import runtime_ne_at

            type_label = event_type_label(alarm_event, self.artifact.config.type_fields)
            type_id = self.vocabs.type_vocab.get(type_label)
            type_id = -1 if type_id is None else int(type_id)
            ne, atype_feat = runtime_ne_at(alarm_event, self.artifact.config.type_fields)
            # use the label-parsed alarm type for features/μ (consistent with
            # training's at_vocab / mu_by_alarm_type keys)
            atype = atype_feat or atype
        else:
            resolved = self._resolve_type_id(alarm_event)
            if resolved is None:
                self.dropped_unknown_type += 1
                self._advance_watermark(ts)
                return None
            type_id, type_label = resolved
            ne = ""

        event = OnlineEvent(
            index=self._next_event_index,
            ts=ts,
            type_id=type_id,
            type_label=type_label,
            alarm=alarm_event,
            parent_index=-1,
            cascade_id=-1,
            parent_score=0.0,
            alarm_type=atype if self.feature_mode else "",
            ne=ne,
            src_mark=src_mark,
        )
        self._next_event_index += 1
        self._events_by_index[event.index] = event
        self._buf_events.append(event)
        self._buf_ts.append(ts)
        self._buf_type.append(type_id)
        self._pending_events.append(event)
        self.total_events_processed += 1
        self._advance_watermark(ts)
        return event

    def _close_inactive(self, now_ts: float):
        """Move cascades whose last_ts < now - close_inactive_sec to closed_groups.

        Throttled: the O(open cascades) sweep runs only when event time has
        advanced past `_close_scan_interval` since the last scan. Cascades may
        thus linger slightly past close_inactive_sec before being emitted, but
        the assignment result is unaffected (closure only gates OUTPUT timing,
        and scoring already excludes out-of-window candidates via _evict).
        """
        if now_ts - self._last_close_scan_ts < self._close_scan_interval:
            return
        self._last_close_scan_ts = now_ts
        cutoff = now_ts - self.config.close_inactive_sec
        to_close: list[int] = []
        for cid, cascade in self.cascades.items():
            if any(not e.finalized for e in cascade.events):
                continue
            if cascade.last_ts < cutoff:
                to_close.append(cid)
        for cid in to_close:
            cascade = self.cascades.pop(cid)
            self._record_closed(cascade)

    def snapshot_ready_groups(self, age_sec: float, now_ts: float) -> list[dict]:
        """Return visual-only snapshots for old active cascades without closing.

        A cascade is emitted again only when its event membership has changed
        since the last snapshot. The real close path remains responsible for
        removing it from memory.
        """
        if age_sec <= 0:
            return []
        cutoff = float(now_ts) - float(age_sec)
        out: list[dict] = []
        for cascade in list(self.cascades.values()):
            if cascade.event_count() < self.config.min_group_events:
                continue
            if any(not e.finalized for e in cascade.events):
                continue
            if cascade.start_ts > cutoff:
                continue
            event_indexes = {e.index for e in cascade.events}
            if event_indexes == cascade.last_snapshot_event_indexes:
                continue
            group = _cascade_to_group(cascade)
            base_group_id = group["group_id"]
            cascade.snapshot_seq += 1
            snapshot_group_id = f"{base_group_id}.snapshot-{cascade.snapshot_seq:04d}"
            related = cascade.snapshot_related_uuids | cascade.snapshot_frontier_uuids
            group["base_group_id"] = base_group_id
            group["group_id"] = snapshot_group_id
            group["related_group_uuids"] = sorted(related)
            group["snapshot_seq"] = cascade.snapshot_seq
            cascade.last_snapshot_event_indexes = event_indexes
            cascade.snapshot_frontier_uuids = {snapshot_group_id}
            cascade.snapshot_related_uuids = related | {snapshot_group_id}
            out.append(group)
        return out

    def close_remaining(self):
        """Emit any still-active cascades. Call once at end of stream."""
        self._finalize_ready(np.inf)
        for cid in list(self.cascades.keys()):
            cascade = self.cascades.pop(cid)
            self._record_closed(cascade)

    def _record_closed(self, cascade: "Cascade"):
        """Record a closed cascade: count its size for the diagnostic
        distribution (ALL cascades), and emit it to output only if it passes
        the min_group_events filter.
        """
        self.closed_cascade_count += 1
        self.closed_size_counter[cascade.event_count()] += 1
        if cascade.event_count() >= self.config.min_group_events:
            group = _cascade_to_group(cascade)
            if (
                cascade.snapshot_seq
                or cascade.snapshot_related_uuids
                or cascade.snapshot_frontier_uuids
            ):
                related = cascade.snapshot_related_uuids | cascade.snapshot_frontier_uuids
                if related:
                    group["base_group_id"] = group.get("group_id", "")
                    group["related_group_uuids"] = sorted(related)
            self.closed_groups.append(group)
            self.emitted_group_count += 1

    def stats(self) -> dict:
        return {
            "total_events_processed": self.total_events_processed,
            "total_immigrants": self.total_immigrants,
            "closed_cascade_count": self.closed_cascade_count,
            "open_cascade_count": len(self.cascades),
            "recent_window_size": len(self._buf_ts) - self._head,
            "pending_event_count": len(self._pending_events) - self._pending_head,
            "dropped_clear": self.dropped_clear,
            "dropped_no_type": self.dropped_no_type,
            "dropped_unknown_type": self.dropped_unknown_type,
        }


# --------------------------------------------------------------------------
# Output formatting (group records compatible with alarm_flow_brunch)
# --------------------------------------------------------------------------


def _summary_of(e) -> dict:
    """Per-event symptom. In feature mode, override alarm_source with the
    canonical device id the model/topology keyed on (``e.ne`` from
    runtime_ne_at) — summarize_alarm_event reads the raw, unstripped field,
    which can diverge from the NE-graph keys and mis-classify topology edges.
    Mirrors the impute path's `if ne: meta["alarm_source"] = ne`."""
    s = summarize_alarm_event(e.alarm, e.index)
    if e.ne:
        s["alarm_source"] = e.ne
    return s


def _cascade_to_group(cascade: Cascade) -> dict:
    summaries = [_summary_of(e) for e in cascade.events]
    # Root is the event whose parent_index == -1 (immigrant), or the earliest by ts
    root = next((e for e in cascade.events if e.parent_index == -1), cascade.events[0])
    root_summary = _summary_of(root)
    timestamps = [s["ts"] for s in summaries]
    # Within-group parent→child edges
    idx_set = {e.index for e in cascade.events}
    by_index = {e.index: e for e in cascade.events}
    group_edges = []
    for e in cascade.events:
        if e.parent_index == -1 or e.parent_index not in idx_set:
            continue
        parent = by_index[e.parent_index]
        group_edges.append(
            {
                "source_index": parent.index,
                "target_index": e.index,
                "source_type": parent.type_label,
                "target_type": e.type_label,
                "score": float(e.parent_score),
            }
        )
    return {
        "group_id": f"mhp-stream-{cascade.cascade_id:06d}",
        "cascade_id": cascade.cascade_id,
        "event_count": len(cascade.events),
        "start_ts": min(timestamps),
        "end_ts": max(timestamps),
        "duration_sec": max(timestamps) - min(timestamps),
        "root_event": root_summary,
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


def _default_visual_metrics_output(visual_output_path: str) -> str:
    return f"{visual_output_path}.metrics.json"


def _default_baseline_visual_output(visual_output_path: str) -> str:
    if visual_output_path.endswith(".jsonl"):
        return f"{visual_output_path[:-6]}.baseline.jsonl"
    return f"{visual_output_path}.baseline.jsonl"


def _is_disabled_path(path) -> bool:
    return str(path or "").strip().lower() in {"", "0", "false", "none", "off"}


def _compact_visual_metrics(metrics, metrics_output_path: str = ""):
    if not metrics:
        return None
    compact = {
        "metrics_output": metrics_output_path or "",
        "health": metrics.get("health") or {},
        "overall": metrics.get("overall") or {},
        "topology": metrics.get("topology") or {},
        "risk_flags": metrics.get("risk_flags") or {},
    }
    distributions = metrics.get("distributions") or {}
    for key in ("duration_sec", "event_count", "real_event_count", "virtual_event_count"):
        if key in distributions:
            compact.setdefault("distributions", {})[key] = distributions[key]
    return compact


def _print_visual_health_summary(metrics, *, label: str = "visual"):
    health = (metrics or {}).get("health") or {}
    if not health:
        return
    score = health.get("grouping_health_score")
    if score is None:
        print(f"[stream] {label} health=N/A (no groups)", flush=True)
        return
    components = health.get("components") or {}
    print(
        f"[stream] {label} health={float(score):.1f}/100 "
        f"(topology={float(components.get('topology_explainability', 0.0)):.1f}, "
        f"time={float(components.get('time_compactness', 0.0)):.1f}, "
        f"size={float(components.get('size_reasonableness', 0.0)):.1f}, "
        f"singleton={float(components.get('singleton_control', 0.0)):.1f}, "
        f"virtual={float(components.get('virtual_reasonableness', 0.0)):.1f}, "
        f"risk={float(components.get('risk_cleanliness', 0.0)):.1f})",
        flush=True,
    )


def _compute_visual_metrics_if_available(args, *, visual_count: int, quiet: bool = False):
    if not args.visual_output or visual_count <= 0:
        return None, ""
    if args.visual_metrics_output is not None and _is_disabled_path(args.visual_metrics_output):
        return None, ""

    metrics_output = (
        _default_visual_metrics_output(args.visual_output)
        if args.visual_metrics_output is None
        else str(args.visual_metrics_output)
    )

    if not quiet:
        print("[stream] computing visual grouping metrics ...", flush=True)
    from fault_grouping.tools.analyze_visual_group_metrics import analyze

    metrics = analyze(
        args.visual_output,
        ne_graph_path=args.ne_graph,
        site_graph_path=args.site_graph,
        topo_max_hops=args.visual_metrics_topo_max_hops,
        max_pairwise_ne=args.visual_metrics_max_pairwise_ne,
        risk_duration_sec=args.visual_metrics_risk_duration_sec,
        risk_site_count=args.visual_metrics_risk_site_count,
        risk_unknown_pair_ratio=args.visual_metrics_risk_unknown_pair_ratio,
        health_target_duration_sec=args.health_target_duration_sec,
        health_target_virtual_ratio=args.health_target_virtual_ratio,
        health_target_size_p50=args.health_target_size_p50,
        health_target_size_p90=args.health_target_size_p90,
        health_target_size_p99=args.health_target_size_p99,
        include_details=not args.visual_metrics_no_detail,
    )
    if metrics_output:
        _write_json(metrics_output, metrics)
    if not quiet:
        _print_visual_health_summary(metrics)
        if metrics_output:
            print(f"visual metrics written to: {metrics_output}", flush=True)
    return metrics, metrics_output


def _compute_baseline_metrics_if_requested(
    args,
    alarm_events,
    *,
    min_group_events: int,
    quiet: bool = False,
):
    group_field = str(args.baseline_group_field or "").strip()
    if not group_field:
        return None, "", 0, ""

    baseline_output = (
        _default_baseline_visual_output(args.visual_output)
        if args.baseline_output is None
        else str(args.baseline_output)
    )
    if _is_disabled_path(baseline_output):
        return None, "", 0, ""

    from fault_grouping.tools.alarm_group_baseline import (
        build_baseline_records,
        load_json_if_exists,
        write_jsonl as write_baseline_jsonl,
    )

    records = build_baseline_records(
        alarm_events,
        group_field=group_field,
        ne_graph_data=load_json_if_exists(args.ne_graph),
        min_group_events=min_group_events,
    )
    count = write_baseline_jsonl(baseline_output, records)
    if not quiet:
        print(
            f"baseline visual groups written to: {baseline_output}; groups={count}",
            flush=True,
        )

    metrics_output = (
        f"{baseline_output}.metrics.json"
        if args.baseline_metrics_output is None
        else str(args.baseline_metrics_output)
    )
    if _is_disabled_path(metrics_output):
        return None, "", count, baseline_output

    from fault_grouping.tools.analyze_visual_group_metrics import analyze

    if not quiet:
        print("[stream] computing baseline alarm-group metrics ...", flush=True)
    metrics = analyze(
        baseline_output,
        ne_graph_path=args.ne_graph,
        site_graph_path=args.site_graph,
        topo_max_hops=args.visual_metrics_topo_max_hops,
        max_pairwise_ne=args.visual_metrics_max_pairwise_ne,
        risk_duration_sec=args.visual_metrics_risk_duration_sec,
        risk_site_count=args.visual_metrics_risk_site_count,
        risk_unknown_pair_ratio=args.visual_metrics_risk_unknown_pair_ratio,
        health_target_duration_sec=args.health_target_duration_sec,
        health_target_virtual_ratio=args.health_target_virtual_ratio,
        health_target_size_p50=args.health_target_size_p50,
        health_target_size_p90=args.health_target_size_p90,
        health_target_size_p99=args.health_target_size_p99,
        include_details=not args.visual_metrics_no_detail,
    )
    _write_json(metrics_output, metrics)
    if not quiet:
        _print_visual_health_summary(metrics, label="baseline alarm-group")
        print(f"baseline metrics written to: {metrics_output}", flush=True)
    return metrics, metrics_output, count, baseline_output


# --------------------------------------------------------------------------
# Cascade size diagnostic (mirrors training output)
# --------------------------------------------------------------------------


def _cascade_size_stats(size_counter: Counter) -> dict | None:
    """Cascade size distribution over ALL closed cascades (pre output-filter).

    Computed from a Counter {size: count} so it reflects every cascade the
    stream closed — including singletons that are dropped from --groups-output
    by min_group_events. This makes the distribution directly comparable to the
    training-time `cascade_size_stats`, and keeps "multi(>=2)" meaningful (it is
    the genuine non-singleton fraction, not "everything that passed the filter").
    """
    if not size_counter:
        return None
    sizes = np.array(sorted(size_counter), dtype=np.int64)
    counts = np.array([size_counter[int(s)] for s in sizes], dtype=np.int64)
    n_cascades = int(counts.sum())
    n_events = int((sizes * counts).sum())
    multi_mask = sizes >= 2
    multi_cascades = int(counts[multi_mask].sum())
    multi_events = int((sizes[multi_mask] * counts[multi_mask]).sum())
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
                "cascade_count": int(counts[mask].sum()),
                "event_count": int((sizes[mask] * counts[mask]).sum()),
            }
        )
    # Weighted median/mean over the counter
    cum = np.cumsum(counts)
    median_idx = int(np.searchsorted(cum, (n_cascades + 1) // 2))
    median_size = float(sizes[min(median_idx, len(sizes) - 1)])
    mean_size = float(n_events / n_cascades) if n_cascades else 0.0
    return {
        "n_cascades": n_cascades,
        "n_events": n_events,
        "multi_event_cascade_count": multi_cascades,
        "multi_event_cascade_share": float(multi_cascades / n_cascades) if n_cascades else 0.0,
        "multi_event_event_count": multi_events,
        "multi_event_event_share": float(multi_events / n_events) if n_events else 0.0,
        "max_size": int(sizes.max()),
        "median_size": median_size,
        "mean_size": mean_size,
        "histogram": histogram,
    }


# --------------------------------------------------------------------------
# Missing-chain imputation path (fixed-lag sampler wrapper)
# --------------------------------------------------------------------------


def _run_imputation(artifact, alarm_events, args, stream_config, quiet=False, visual_output=None):
    """Drive the fixed-lag missing-chain sampler over the alarm stream.

    Non-invasive: consumes the same alarm events, emits brunch-compatible group
    dicts (with imputed missing events tagged virtual). Returns (groups, stats).
    """
    from alarm_flow_mhp.missing_chain_sampler import (
        MissingChainSampler,
        SamplerConfig,
        device_adapter_from_artifact,
        feature_adapter_from_artifact,
    )

    feature_mode = getattr(artifact.config, "edge_mode", "device") == "feature"
    dynamic_mode = (
        feature_mode
        and getattr(artifact.config, "dynamic_alpha", "off") != "off"
    )
    state_timeline = None
    alarm_type_from_title_fn = None
    runtime_ne_at_fn = None
    if dynamic_mode:
        from alarm_flow_mhp.dynamic_state import ObservedStateTimeline
        from alarm_flow_isahp.sequences import alarm_type_from_title
        from alarm_flow_mhp.feature_spec import runtime_ne_at

        state_timeline = ObservedStateTimeline()
        alarm_type_from_title_fn = alarm_type_from_title
        runtime_ne_at_fn = runtime_ne_at
    if feature_mode:
        floor = args.feature_alpha_floor or None
        adapter = feature_adapter_from_artifact(
            artifact,
            args.ne_graph,
            alpha_floor=floor,
            time_slack_sec=stream_config.time_slack_sec,
            late_penalty_half_life_sec=stream_config.late_penalty_half_life_sec,
            source_mark_at=(
                state_timeline.source_mark_at if state_timeline is not None else None
            ),
            target_mark_at=(
                (lambda ne, ts: state_timeline.state_at(ne, np.nextafter(float(ts), -np.inf)))
                if state_timeline is not None else None
            ),
            cache_max_entries=int(getattr(args, "impute_cache_max", 200_000)),
            topology_relation_prior=stream_config.topology_relation_prior,
        )
    else:
        if getattr(artifact.params, "kernel_type", "exp") != "exp":
            raise NotImplementedError(
                "--impute device path supports exp kernel only (piecewise adapter TODO)"
            )
        adapter = device_adapter_from_artifact(
            artifact,
            time_slack_sec=stream_config.time_slack_sec,
            late_penalty_half_life_sec=stream_config.late_penalty_half_life_sec,
        )

    history = float(stream_config.history_window_sec)
    lag = float(args.impute_lag_sec) if args.impute_lag_sec is not None else history
    cfg = SamplerConfig(
        lag_sec=lag,
        history_window_sec=history,
        time_slack_sec=float(stream_config.time_slack_sec),
        late_penalty_half_life_sec=float(stream_config.late_penalty_half_life_sec),
        sweeps_per_tick=int(args.impute_sweeps),
        missing_log_prior=float(args.impute_kappa),
        max_depth=int(args.impute_max_depth),
        max_births_per_sweep=int(args.impute_max_births),
        max_history_events=int(args.impute_max_history),
        sweep_recent_events=int(args.impute_sweep_recent),
        future_candidate_reset_limit=int(args.impute_future_candidate_reset_limit),
        max_birth_attempts_per_sweep=int(args.impute_max_birth_attempts),
        # Throttle the O(live) cascade-close scan to a fraction of the lag — it
        # only governs output timing, and running it every event is wasteful.
        commit_check_interval_sec=max(lag * 0.2, 30.0),
        seed=int(getattr(artifact.config, "seed", 0) or 0),
    )
    sampler = MissingChainSampler(adapter, cfg)
    if not quiet:
        print(
            f"[stream] impute: mode={'feature' if feature_mode else 'device'} "
            f"lag={cfg.lag_sec:.0f}s history={cfg.history_window_sec:.0f}s "
            f"time_slack={cfg.time_slack_sec:.0f}s "
            f"late_half_life={cfg.late_penalty_half_life_sec:.0f}s "
            f"sweeps={cfg.sweeps_per_tick} kappa={cfg.missing_log_prior} "
            f"max_depth={cfg.max_depth}",
            flush=True,
        )

    type_fields = artifact.config.type_fields
    groups: list = []
    dropped = {"clear": 0, "no_type": 0, "unknown_type": 0}
    processed = 0
    _t0 = time.monotonic()
    _n = len(alarm_events)
    visual_snapshot_age_sec = float(getattr(args, "visual_snapshot_age_sec", 0.0) or 0.0)
    visual_snapshot_check_interval_sec = _resolve_visual_snapshot_check_interval(
        visual_snapshot_age_sec,
        getattr(args, "visual_snapshot_check_interval_sec", None),
    )
    last_visual_snapshot_check_ts = -np.inf

    def _accept_output_groups(new_groups, *, finalization_reason):
        if not new_groups:
            return []
        min_ev = int(stream_config.min_group_events)
        accepted = (
            [g for g in new_groups if g["real_event_count"] >= min_ev]
            if min_ev > 1 else list(new_groups)
        )
        if visual_output is not None and accepted:
            visual_output.emit_groups(accepted, finalization_reason=finalization_reason)
        return accepted

    def _emit_visual_snapshots(now_ts: float):
        nonlocal last_visual_snapshot_check_ts
        if visual_output is None or visual_snapshot_age_sec <= 0:
            return []
        now_ts = float(now_ts)
        if now_ts < last_visual_snapshot_check_ts + visual_snapshot_check_interval_sec:
            return []
        last_visual_snapshot_check_ts = max(last_visual_snapshot_check_ts, now_ts)
        snapshots = sampler.visual_snapshot_groups(visual_snapshot_age_sec, now_ts=now_ts)
        return _accept_output_groups(snapshots, finalization_reason="age_snapshot")

    for i, alarm in enumerate(alarm_events):
        if args.progress_every and (i + 1) % args.progress_every == 0 and not quiet:
            el = time.monotonic() - _t0
            rate = (i + 1) / el if el > 0 else 0
            st = sampler.stats()
            print(
                f"[stream] impute {i + 1}/{_n} ({rate:.0f} evt/s, "
                f"live={st['live_events']}, missing={st['live_missing']}, "
                f"births={st['births']}, groups={len(groups)}, elapsed={el:.0f}s)",
                flush=True,
            )
        is_clear = is_clear_alarm(alarm.get("alarm", {}) if isinstance(alarm, dict) else {})
        src_mark = None
        if dynamic_mode:
            dev, _ = runtime_ne_at_fn(alarm, type_fields)
            atype_state = alarm_type_from_title_fn(alarm.get("alarm_title", ""))
            src_mark = state_timeline.ingest(
                float(alarm.get("ts", 0.0)), dev, atype_state, is_clear
            )
            state_timeline.prune_before(float(alarm.get("ts", 0.0)) - cfg.window_sec())
        if is_clear:
            dropped["clear"] += 1
            _emit_visual_snapshots(float(alarm.get("ts", 0.0)))
            continue
        atype = alarm_type_label(alarm)
        if atype is None:
            dropped["no_type"] += 1
            _emit_visual_snapshots(float(alarm.get("ts", 0.0)))
            continue
        meta = summarize_alarm_event(alarm, i)
        if feature_mode:
            from alarm_flow_mhp.feature_spec import runtime_ne_at

            ne, atype_feat = runtime_ne_at(alarm, type_fields)
            at = atype_feat or atype
            type_key = (at, ne)
            meta["alarm_type"] = at
            if ne:
                meta["alarm_source"] = ne
        else:
            type_label = event_type_label(alarm, type_fields)
            tid = artifact.vocabs.type_vocab.get(type_label)
            if tid is None:
                dropped["unknown_type"] += 1
                _emit_visual_snapshots(float(alarm.get("ts", 0.0)))
                continue
            type_key = int(tid)
            meta["type_label"] = type_label
        processed += 1
        closed = sampler.ingest(
            float(alarm.get("ts", 0.0)),
            type_key,
            meta,
            src_mark=src_mark,
        )
        groups.extend(_accept_output_groups(closed, finalization_reason="closed"))
        _emit_visual_snapshots(float(alarm.get("ts", 0.0)))
    groups.extend(
        _accept_output_groups(sampler.flush(), finalization_reason="stream_end")
    )

    stats = sampler.stats()
    stats.update({"processed": processed, "dropped": dropped})
    return groups, stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Online (streaming) MHP inference: ingest alarms in time order, emit cascade groups."
    )
    parser.add_argument("model", help="Trained MHP artifact JSON (produced by train_alarm_mhp.py).")
    parser.add_argument("alarms", help="Sorted alarm cache or raw alarms — same format as train_alarm_mhp.")
    parser.add_argument(
        "--groups-output",
        default="",
        help="Path for JSON output (groups + metadata). If empty, no JSON is written.",
    )
    parser.add_argument(
        "--edges-output",
        default="",
        help="Optional JSONL output for branching edges across all cascades.",
    )
    parser.add_argument(
        "--visual-output",
        default="",
        help="Visualization JSONL compatible with the fault group browser and "
             "propagation visualizer. ALWAYS produced: if omitted, defaults to "
             "<groups-output or input-name>[.impute].visual.jsonl.",
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
    parser.add_argument(
        "--visual-snapshot-age-sec",
        type=float,
        default=0.0,
        help=(
            "Emit visual-only snapshots for active fault groups whose earliest "
            "alarm is at least this old, without closing/removing the group. "
            "Later snapshots/final output are linked through related_group_uuids. "
            "Default: 0 (disabled). Example: 360 for six minutes."
        ),
    )
    parser.add_argument(
        "--visual-snapshot-check-interval-sec",
        type=float,
        default=None,
        help=(
            "Minimum stream-time interval between active visual snapshot scans. "
            "Only used when --visual-snapshot-age-sec > 0. Default: auto "
            "(min(age/6, 60s), at least 1s)."
        ),
    )
    parser.add_argument(
        "--visual-metrics-output",
        default=None,
        help=(
            "JSON output for label-free visual grouping health metrics. Default: "
            "<visual-output>.metrics.json. Use 'none' to disable visual metrics."
        ),
    )
    parser.add_argument(
        "--visual-metrics-topo-max-hops",
        type=int,
        default=3,
        help="Max NE topology hops used by visual metrics. Default: 3.",
    )
    parser.add_argument(
        "--visual-metrics-max-pairwise-ne",
        type=int,
        default=200,
        help=(
            "Skip all-pair topology cohesion for groups with more than this many "
            "NEs. 0 disables all-pair topology cohesion. Default: 200."
        ),
    )
    parser.add_argument(
        "--visual-metrics-risk-duration-sec",
        type=float,
        default=2 * 3600.0,
        help="Risk flag threshold for long-duration groups. Default: 7200.",
    )
    parser.add_argument(
        "--visual-metrics-risk-site-count",
        type=int,
        default=10,
        help="Risk flag threshold for groups spanning many sites. Default: 10.",
    )
    parser.add_argument(
        "--visual-metrics-risk-unknown-pair-ratio",
        type=float,
        default=0.5,
        help="Risk flag threshold for unknown topology pair ratio. Default: 0.5.",
    )
    parser.add_argument(
        "--health-target-duration-sec",
        type=float,
        default=3600.0,
        help="Healthy p90 group duration target for the visual health score. Default: 3600.",
    )
    parser.add_argument(
        "--health-target-virtual-ratio",
        type=float,
        default=0.2,
        help="Healthy virtual-event ratio target for the visual health score. Default: 0.2.",
    )
    parser.add_argument(
        "--health-target-size-p50",
        type=float,
        default=2.0,
        help="Healthy p50 real-event group size target. Default: 2.",
    )
    parser.add_argument(
        "--health-target-size-p90",
        type=float,
        default=20.0,
        help="Healthy p90 real-event group size target. Default: 20.",
    )
    parser.add_argument(
        "--health-target-size-p99",
        type=float,
        default=100.0,
        help="Healthy p99 real-event group size target. Default: 100.",
    )
    parser.add_argument(
        "--visual-metrics-no-detail",
        action="store_true",
        help="Do not include per-group detail in the visual metrics JSON.",
    )
    parser.add_argument(
        "--baseline-group-field",
        default="",
        help=(
            "Evaluate a baseline formed by the alarms' own group id field, "
            "for example '故障组ID'. Empty string disables this baseline."
        ),
    )
    parser.add_argument(
        "--baseline-output",
        default=None,
        help=(
            "Visual-like JSONL for the alarm-group baseline. Default when "
            "--baseline-group-field is set: <visual-output>.baseline.jsonl. "
            "Use 'none' to disable baseline output and metrics."
        ),
    )
    parser.add_argument(
        "--baseline-metrics-output",
        default=None,
        help=(
            "Metrics JSON for the alarm-group baseline. Default: "
            "<baseline-output>.metrics.json. Use 'none' to only write the "
            "baseline visual JSONL."
        ),
    )
    parser.add_argument(
        "--baseline-min-group-events",
        type=int,
        default=None,
        help=(
            "Drop baseline alarm groups smaller than this. Default: same as "
            "the effective stream --min-group-events."
        ),
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
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--time-slack-sec",
        type=float,
        default=None,
        help=(
            "Override timestamp-jitter tolerance for online parent assignment. "
            "Default: artifact config."
        ),
    )
    parser.add_argument(
        "--late-penalty-half-life-sec",
        type=float,
        default=None,
        help=(
            "Override late-parent penalty half-life for online assignment. "
            "Default: artifact config."
        ),
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help="Override artifact regions; omit to reuse the model's regions.",
    )
    parser.add_argument(
        "--close-inactive-sec",
        type=float,
        default=7200.0,
        help="Cascades with no new events for this many seconds are closed. Default: 7200.",
    )
    parser.add_argument(
        "--min-group-events",
        type=int,
        default=None,
        help="Override min_group_events (drop groups smaller than this).",
    )
    parser.add_argument(
        "--immigrant-bias",
        type=float,
        default=1.0,
        help=(
            "Multiplier on μ at scoring time. >1 prefers immigrants (more, smaller cascades); "
            "<1 prefers binding to existing cascades (fewer, larger). Default: 1.0."
        ),
    )
    parser.add_argument(
        "--max-history-events",
        type=int,
        default=None,
        help="Override max candidate parents per event. Default: artifact value.",
    )
    parser.add_argument(
        "--feature-alpha-floor",
        type=float,
        default=0.0,
        help=(
            "Feature mode only: candidate edges with live α below this are treated "
            "as non-edges (the inference analog of device-mode edge_threshold). "
            "0 (default) falls back to the artifact's edge_threshold. Raise to "
            "curb soft over-connection of feature-similar but unrelated pairs."
        ),
    )
    parser.add_argument(
        "--topology-relation-prior",
        default="",
        help=(
            "Feature mode only: comma-separated inference-time multipliers by "
            "topology relation, e.g. "
            "'same_device=1,direct=1,same_site=0.8,indirect=0.5,cross_site=0.2,unknown=0.05'. "
            "Missing keys default to 1.0; empty string preserves current behavior."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print stream progress every N events. 0 = silent. Default: 50000.",
    )
    # ---- missing-chain imputation (fixed-lag sampler) --------------------
    parser.add_argument(
        "--impute",
        action="store_true",
        help=(
            "Reconstruct cascades through UNOBSERVED (missing) events of known "
            "types using the fixed-lag missing-chain sampler instead of the "
            "single-pass argmax assigner. Groups then include imputed missing "
            "events (tagged virtual, with confidence) that bridge otherwise-"
            "disconnected alarms. Works for device and feature edge modes (exp "
            "kernel). Adds latency: a cascade is emitted only after it leaves "
            "kernel reach."
        ),
    )
    parser.add_argument(
        "--impute-lag-sec",
        type=float,
        default=None,
        help="Fixed-lag commit delay (s). Default: history_window_sec.",
    )
    parser.add_argument(
        "--impute-sweeps",
        type=int,
        default=3,
        help="MCMC sweeps per ingested alarm. Default: 3.",
    )
    parser.add_argument(
        "--impute-kappa",
        type=float,
        default=-2.0,
        help=(
            "Log-prior penalty per imputed missing event (κ knob). More negative "
            "⇒ fewer / shallower imputed chains. Default: -2.0."
        ),
    )
    parser.add_argument(
        "--impute-max-depth",
        type=int,
        default=3,
        help="Max imputed missing-chain depth (multi-hop). Default: 3.",
    )
    parser.add_argument(
        "--impute-max-births",
        type=int,
        default=8,
        help="Max new missing events born per sweep (rate limit). Default: 8.",
    )
    parser.add_argument(
        "--impute-max-history",
        type=int,
        default=256,
        help=(
            "Max candidate parents scored per event (nearest in time). Lower = "
            "faster. The dominant perf knob in feature mode. Default: 256."
        ),
    )
    parser.add_argument(
        "--impute-sweep-recent",
        type=int,
        default=64,
        help=(
            "PARENT re-sampling only re-touches the most recent N uncommitted "
            "events (bounded local approximation). Does NOT gate birth/death. "
            "Default: 64."
        ),
    )
    parser.add_argument(
        "--impute-future-candidate-reset-limit",
        type=int,
        default=0,
        help=(
            "When time slack lets a new event become a future parent for older "
            "events, re-sample all affected older events by default. Set >0 to "
            "only repair the nearest N affected events for speed. Default: 0 "
            "(unlimited, exact old behavior)."
        ),
    )
    parser.add_argument(
        "--impute-max-birth-attempts",
        type=int,
        default=32,
        help=(
            "Birth is attempted over ALL active orphans (fair, no age bias) but "
            "bounded to this many attempts per sweep to cap cost. Default: 32."
        ),
    )
    parser.add_argument(
        "--impute-cache-max",
        type=int,
        default=200_000,
        help=(
            "Feature-adapter LRU cache size (entries) for per-pair α / per-source "
            "Σα / candidate sets. Larger = fewer recomputations on a big device "
            "space, at more memory. Default: 200000."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.visual_snapshot_age_sec < 0:
        parser.error("--visual-snapshot-age-sec must be >= 0")
    if args.visual_snapshot_check_interval_sec is not None and args.visual_snapshot_check_interval_sec < 0:
        parser.error("--visual-snapshot-check-interval-sec must be >= 0")
    if args.visual_metrics_topo_max_hops < 1:
        parser.error("--visual-metrics-topo-max-hops must be >= 1")
    if args.visual_metrics_max_pairwise_ne < 0:
        parser.error("--visual-metrics-max-pairwise-ne must be >= 0")
    if args.visual_metrics_risk_duration_sec < 0:
        parser.error("--visual-metrics-risk-duration-sec must be >= 0")
    if args.visual_metrics_risk_site_count < 0:
        parser.error("--visual-metrics-risk-site-count must be >= 0")
    if args.visual_metrics_risk_unknown_pair_ratio < 0:
        parser.error("--visual-metrics-risk-unknown-pair-ratio must be >= 0")
    if args.health_target_duration_sec <= 0:
        parser.error("--health-target-duration-sec must be > 0")
    if args.health_target_virtual_ratio <= 0:
        parser.error("--health-target-virtual-ratio must be > 0")
    if args.health_target_size_p50 <= 0 or args.health_target_size_p90 <= 0 or args.health_target_size_p99 <= 0:
        parser.error("--health-target-size-p50/p90/p99 must be > 0")
    if args.baseline_min_group_events is not None and args.baseline_min_group_events < 1:
        parser.error("--baseline-min-group-events must be >= 1")
    try:
        topology_relation_prior = parse_topology_relation_prior(args.topology_relation_prior)
    except ValueError as exc:
        parser.error(str(exc))

    # Visual output is a required artifact: if not given, derive a default path
    # from the groups-output (or the input cache) so it is always produced.
    if not args.visual_output:
        if args.groups_output:
            base = args.groups_output[:-5] if args.groups_output.endswith(".json") else args.groups_output
        else:
            base = os.path.splitext(os.path.basename(args.alarms.rstrip("/\\")))[0] or "mhp_stream"
        suffix = ".impute" if args.impute else ""
        args.visual_output = f"{base}{suffix}.visual.jsonl"

    t_total_start = time.monotonic()
    if not args.quiet:
        print(f"[stream] loading MHP artifact: {args.model}", flush=True)
    artifact = load_alarm_mhp_artifact(args.model)
    if not args.quiet:
        print(
            f"[stream] artifact: trained_events={artifact.training_metadata.get('train_event_count', 'n/a')}, "
            f"types={len(artifact.vocabs.type_vocab)}, "
            f"active_edges={len(artifact.params.edge_alpha)}",
            flush=True,
        )

    regions = parse_regions(args.regions) if args.regions is not None else artifact.config.regions
    if not args.quiet:
        print(f"[stream] region filter: {sorted(regions) if regions else '<none>'}", flush=True)
    if args.regions is not None:
        run_config = _replace(artifact.config, regions=regions)
    else:
        run_config = artifact.config

    if not args.quiet:
        print(f"[stream] loading alarms: {args.alarms}", flush=True)
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
        print(f"[stream] loaded alarm events: {len(alarm_events)}", flush=True)

    stream_config = StreamConfig.from_artifact_config(
        run_config,
        close_inactive_sec=args.close_inactive_sec,
        min_group_events=args.min_group_events,
        immigrant_bias=args.immigrant_bias,
        max_history_events=args.max_history_events,
        feature_alpha_floor=args.feature_alpha_floor,
        topology_relation_prior=topology_relation_prior,
        time_slack_sec=args.time_slack_sec,
        late_penalty_half_life_sec=args.late_penalty_half_life_sec,
    )
    baseline_min_group_events = (
        int(args.baseline_min_group_events)
        if args.baseline_min_group_events is not None
        else int(stream_config.min_group_events)
    )
    if not args.quiet:
        visual_snapshot_check_interval_sec = _resolve_visual_snapshot_check_interval(
            float(args.visual_snapshot_age_sec or 0.0),
            args.visual_snapshot_check_interval_sec,
        )
        print(
            f"[stream] config: history_window={stream_config.history_window_sec:.0f}s "
            f"time_slack={stream_config.time_slack_sec:.0f}s "
            f"late_half_life={stream_config.late_penalty_half_life_sec:.0f}s "
            f"max_history={stream_config.max_history_events} "
            f"close_inactive={stream_config.close_inactive_sec:.0f}s "
            f"min_group_events={stream_config.min_group_events} "
            f"immigrant_bias={stream_config.immigrant_bias} "
            f"topology_relation_prior={format_topology_relation_prior(stream_config.topology_relation_prior)} "
            f"visual_snapshot_age={args.visual_snapshot_age_sec:.0f}s "
            f"visual_snapshot_check_interval={visual_snapshot_check_interval_sec:.0f}s",
            flush=True,
        )

    # ---- missing-chain imputation path (fixed-lag sampler) --------------
    if args.impute:
        t_imp_start = time.monotonic()
        visual_output = None
        if args.visual_output:
            from alarm_flow_mhp.visual_output import AlarmMHPVisualOutputSession

            visual_output = AlarmMHPVisualOutputSession.from_files(
                args.visual_output,
                args.ne_graph,
                args.site_graph,
                ne_scope=args.visual_ne_scope,
            )
            visual_output.reset_output_file()
        try:
            groups, impute_stats = _run_imputation(
                artifact,
                alarm_events,
                args,
                stream_config,
                quiet=args.quiet,
                visual_output=visual_output,
            )
        finally:
            if visual_output is not None:
                visual_output.close()
        elapsed = time.monotonic() - t_imp_start
        visual_count = visual_output.emitted_count if visual_output is not None else 0
        if not args.quiet:
            print(
                f"[stream] impute done: groups={len(groups)}, "
                f"births={impute_stats['births']}, deaths={impute_stats['deaths']}, "
                f"processed={impute_stats['processed']}, elapsed={elapsed:.1f}s",
                flush=True,
            )
        visual_metrics, visual_metrics_output = _compute_visual_metrics_if_available(
            args,
            visual_count=visual_count,
            quiet=args.quiet,
        )
        (
            baseline_metrics,
            baseline_metrics_output,
            baseline_visual_count,
            baseline_visual_output,
        ) = _compute_baseline_metrics_if_requested(
            args,
            alarm_events,
            min_group_events=baseline_min_group_events,
            quiet=args.quiet,
        )
        metadata = {
            "algorithm": "alarm_flow_mhp.stream+impute",
            "model": os.path.abspath(args.model),
            "input": os.path.abspath(args.alarms),
            "alarm_metadata": alarm_metadata,
            "regions": sorted(regions) if regions else [],
            "config": {
                "history_window_sec": stream_config.history_window_sec,
                "time_slack_sec": stream_config.time_slack_sec,
                "late_penalty_half_life_sec": stream_config.late_penalty_half_life_sec,
                "min_group_events": stream_config.min_group_events,
                "time_scale_sec": stream_config.time_scale_sec,
                "impute_lag_sec": float(args.impute_lag_sec) if args.impute_lag_sec is not None else stream_config.history_window_sec,
                "impute_sweeps": args.impute_sweeps,
                "impute_kappa": args.impute_kappa,
                "impute_max_depth": args.impute_max_depth,
                "impute_max_births": args.impute_max_births,
                "impute_max_history": args.impute_max_history,
                "impute_sweep_recent": args.impute_sweep_recent,
                "impute_future_candidate_reset_limit": args.impute_future_candidate_reset_limit,
                "impute_max_birth_attempts": args.impute_max_birth_attempts,
                "topology_relation_prior": dict(stream_config.topology_relation_prior or {}),
                "visual_snapshot_age_sec": args.visual_snapshot_age_sec,
                "visual_snapshot_check_interval_sec": _resolve_visual_snapshot_check_interval(
                    float(args.visual_snapshot_age_sec or 0.0),
                    args.visual_snapshot_check_interval_sec,
                ),
                "baseline_group_field": str(args.baseline_group_field or ""),
                "baseline_min_group_events": baseline_min_group_events,
            },
            "group_count": len(groups),
            "modeled_event_count": impute_stats["processed"],
            "imputed_missing_events": sum(g.get("virtual_event_count", 0) for g in groups),
            "visual_group_count": visual_count,
            "visual_metrics": _compact_visual_metrics(visual_metrics, visual_metrics_output),
            "baseline_visual_group_count": baseline_visual_count,
            "baseline_visual_output": baseline_visual_output,
            "baseline_metrics": _compact_visual_metrics(baseline_metrics, baseline_metrics_output),
            "births": impute_stats["births"],
            "deaths": impute_stats["deaths"],
            "drop_stats": impute_stats["dropped"],
            "total_wall_clock_seconds": float(time.monotonic() - t_total_start),
        }
        if args.groups_output:
            _write_json(args.groups_output, {"metadata": metadata, "groups": groups})
            print(f"groups written to: {args.groups_output}; groups={len(groups)}")
        if args.edges_output:
            edges = []
            for group in groups:
                edges.extend(group.get("edges", []))
            n = _write_jsonl(args.edges_output, edges)
            print(f"branching edges written to: {args.edges_output}; edges={n}")
        if args.visual_output:
            print(f"visual groups written to: {args.visual_output}; groups={visual_count}")
        return
    min_close = stream_config.history_window_sec + stream_config.time_slack_sec
    if stream_config.close_inactive_sec < min_close:
        print(
            f"[stream] WARN: close_inactive_sec ({stream_config.close_inactive_sec:.0f}s) < "
            f"history_window_sec + time_slack_sec ({min_close:.0f}s). A cascade can be "
            f"closed while its events are still candidate parents, turning would-be children "
            f"into 'orphan' immigrants. Set close_inactive_sec >= history_window_sec + "
            f"time_slack_sec to avoid this.",
            flush=True,
        )

    # Feature mode: build the live-α scorer from the artifact kernel + NE graph
    # (device attributes + topology). This is what generalizes to new devices.
    feature_scorer = None
    mu_scorer = None
    if getattr(artifact.config, "edge_mode", "device") == "feature":
        from mhp.feature_kernel import FeatureKernel
        from alarm_flow_mhp.feature_spec import (
            MuFeatureSpec,
            RuntimeFeatureScorer,
            RuntimeMuScorer,
        )
        from alarm_flow_isahp.ne_topology import NETopologyIndex
        from ne_link_learning.core import build_graph_context
        from topology_tools.region_utils import load_ne_graph

        md = artifact.training_metadata or {}
        fk = md.get("feature_kernel")
        rt = md.get("feature_runtime") or {}
        if fk is None:
            raise ValueError("feature-mode artifact missing feature_kernel")
        if not args.quiet:
            print("[stream] feature mode: loading NE graph + building live-α scorer ...", flush=True)
        ne_graph_data = load_ne_graph(args.ne_graph)
        graph_ctx = build_graph_context(ne_graph_data)
        # Build the index with the SAME reach training used for feature
        # candidate generation — otherwise pairs beyond this many hops would get
        # topo_score=0 at inference, diverging from the trained φ.
        infer_hops = max(int(getattr(artifact.config, "feature_topo_max_hops", 2)), 1)
        topo_idx = NETopologyIndex.from_graph(ne_graph_data, max_hops=infer_hops)
        if not args.quiet:
            print(f"[stream] topology index max_hops={infer_hops} (from artifact.config)", flush=True)
        # Dynamic α: source = 3 bits; source_target = source 3 + target 3.
        dyn_mode = getattr(artifact.config, "dynamic_alpha", "off")
        n_dynamic = 6 if dyn_mode == "source_target" else (3 if dyn_mode != "off" else 0)
        feature_scorer = RuntimeFeatureScorer(
            kernel=FeatureKernel.from_dict(fk),
            at_vocab=rt.get("at_vocab", []),
            graph_context=graph_ctx,
            topology_index=topo_idx,
            beta=float(rt.get("beta", 1.0)),
            n_dynamic=n_dynamic,
        )
        if not args.quiet:
            print(
                f"[stream] feature scorer: {feature_scorer.layout.n_features} features"
                f"{f' + {n_dynamic} dynamic ({dyn_mode} state)' if n_dynamic else ''}, "
                f"device-OPEN (new devices accepted)",
                flush=True,
            )
        # Parameterized μ scorer (live μ for any device). Falls back to the
        # per-alarm-type table inside the assigner if absent.
        mu_fk = rt.get("mu_kernel")
        mu_sp = rt.get("mu_spec")
        if mu_fk is not None and mu_sp is not None:
            mu_scorer = RuntimeMuScorer(
                mu_kernel=FeatureKernel.from_dict(mu_fk),
                mu_spec=MuFeatureSpec.from_dict(mu_sp),
                graph_context=graph_ctx,
            )
            if not args.quiet:
                print(
                    f"[stream] μ scorer: {mu_scorer.spec.n_features} features (parameterized μ)",
                    flush=True,
                )
        elif not args.quiet:
            print("[stream] μ: per-alarm-type table (no parameterized μ in artifact)", flush=True)

    assigner = StreamMHPAssigner(
        artifact, stream_config, feature_scorer=feature_scorer, mu_scorer=mu_scorer
    )
    visual_output = None
    visual_group_cursor = 0
    visual_snapshot_age_sec = float(args.visual_snapshot_age_sec or 0.0)
    visual_snapshot_check_interval_sec = _resolve_visual_snapshot_check_interval(
        visual_snapshot_age_sec,
        args.visual_snapshot_check_interval_sec,
    )
    last_visual_snapshot_check_ts = -np.inf
    if args.visual_output:
        visual_output = AlarmBRUNCHVisualOutputSession.from_files(
            args.visual_output,
            args.ne_graph,
            args.site_graph,
            ne_scope=args.visual_ne_scope,
        )
        visual_output.reset_output_file()

    t_stream_start = time.monotonic()
    last_print = t_stream_start
    try:
        for i, alarm in enumerate(alarm_events):
            assigner.process(alarm)
            if visual_output is not None and len(assigner.closed_groups) > visual_group_cursor:
                new_groups = assigner.closed_groups[visual_group_cursor:]
                visual_output.emit_groups(new_groups, finalization_reason="closed")
                visual_group_cursor = len(assigner.closed_groups)
            if (
                visual_output is not None
                and visual_snapshot_age_sec > 0
                and float(alarm.get("ts", 0.0)) >= last_visual_snapshot_check_ts + visual_snapshot_check_interval_sec
            ):
                last_visual_snapshot_check_ts = max(
                    last_visual_snapshot_check_ts,
                    float(alarm.get("ts", 0.0)),
                )
                snapshots = assigner.snapshot_ready_groups(
                    visual_snapshot_age_sec,
                    float(alarm.get("ts", 0.0)),
                )
                if snapshots:
                    visual_output.emit_groups(snapshots, finalization_reason="age_snapshot")
            if args.progress_every and (i + 1) % args.progress_every == 0 and not args.quiet:
                now = time.monotonic()
                elapsed = now - t_stream_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                stats = assigner.stats()
                visual_count = visual_output.emitted_count if visual_output is not None else 0
                print(
                    f"[stream] processed {i + 1}/{len(alarm_events)} "
                    f"({rate:.0f} evt/s, "
                    f"open={stats['open_cascade_count']}, "
                    f"closed={stats['closed_cascade_count']}, "
                    f"visual={visual_count}, "
                    f"immigrants={stats['total_immigrants']}, "
                    f"elapsed={elapsed:.1f}s)",
                    flush=True,
                )
                last_print = now

        # Drain
        if not args.quiet:
            print("[stream] draining remaining cascades ...", flush=True)
        assigner.close_remaining()
        if visual_output is not None and len(assigner.closed_groups) > visual_group_cursor:
            new_groups = assigner.closed_groups[visual_group_cursor:]
            visual_output.emit_groups(new_groups, finalization_reason="stream_end")
            visual_group_cursor = len(assigner.closed_groups)
    finally:
        if visual_output is not None:
            visual_output.close()
    t_stream_end = time.monotonic()
    visual_count = visual_output.emitted_count if visual_output is not None else 0

    stats = assigner.stats()
    groups = assigner.closed_groups
    if not args.quiet:
        print(
            f"[stream] done: events={stats['total_events_processed']}, "
            f"closed_cascades={stats['closed_cascade_count']}, "
            f"emitted_groups(>= {stream_config.min_group_events})={len(groups)}, "
            f"immigrants={stats['total_immigrants']}, "
            f"dropped={{clear:{stats['dropped_clear']}, no_type:{stats['dropped_no_type']}, "
            f"unknown_type:{stats['dropped_unknown_type']}}}",
            flush=True,
        )
    visual_metrics, visual_metrics_output = _compute_visual_metrics_if_available(
        args,
        visual_count=visual_count,
        quiet=args.quiet,
    )
    (
        baseline_metrics,
        baseline_metrics_output,
        baseline_visual_count,
        baseline_visual_output,
        ) = _compute_baseline_metrics_if_requested(
            args,
            alarm_events,
            min_group_events=baseline_min_group_events,
            quiet=args.quiet,
        )

    # Diagnostic distribution over ALL closed cascades (pre min_group_events
    # filter) — comparable to training. `groups` (emitted) is filtered.
    cascade_stats = _cascade_size_stats(assigner.closed_size_counter)
    metadata = {
        "algorithm": "alarm_flow_mhp.stream",
        "model": os.path.abspath(args.model),
        "input": os.path.abspath(args.alarms),
        "alarm_metadata": alarm_metadata,
        "regions": sorted(regions) if regions else [],
        "config": {
            "history_window_sec": stream_config.history_window_sec,
            "time_slack_sec": stream_config.time_slack_sec,
            "late_penalty_half_life_sec": stream_config.late_penalty_half_life_sec,
            "max_history_events": stream_config.max_history_events,
            "close_inactive_sec": stream_config.close_inactive_sec,
            "min_group_events": stream_config.min_group_events,
            "immigrant_bias": stream_config.immigrant_bias,
            "time_scale_sec": stream_config.time_scale_sec,
            "topology_relation_prior": dict(stream_config.topology_relation_prior or {}),
            "visual_snapshot_age_sec": args.visual_snapshot_age_sec,
            "visual_snapshot_check_interval_sec": visual_snapshot_check_interval_sec,
            "baseline_group_field": str(args.baseline_group_field or ""),
            "baseline_min_group_events": baseline_min_group_events,
        },
        "group_count": len(groups),
        "emitted_group_count": len(groups),
        "visual_group_count": visual_count,
        "visual_metrics": _compact_visual_metrics(visual_metrics, visual_metrics_output),
        "baseline_visual_group_count": baseline_visual_count,
        "baseline_visual_output": baseline_visual_output,
        "baseline_metrics": _compact_visual_metrics(baseline_metrics, baseline_metrics_output),
        "closed_cascade_count": stats["closed_cascade_count"],
        "modeled_event_count": stats["total_events_processed"],
        "immigrant_count": stats["total_immigrants"],
        "cascade_size_stats": cascade_stats,
        "stream_seconds": float(t_stream_end - t_stream_start),
        "total_wall_clock_seconds": float(time.monotonic() - t_total_start),
        "active_edges": len(artifact.params.edge_alpha),
        "type_count": len(artifact.vocabs.type_vocab),
        "drop_stats": {
            "clear": stats["dropped_clear"],
            "no_type": stats["dropped_no_type"],
            "unknown_type": stats["dropped_unknown_type"],
        },
    }

    if args.groups_output:
        _write_json(args.groups_output, {"metadata": metadata, "groups": groups})
        print(f"groups written to: {args.groups_output}; groups={len(groups)}")

    if args.edges_output:
        edges = []
        for group in groups:
            edges.extend(group["edges"])
        n = _write_jsonl(args.edges_output, edges)
        print(f"branching edges written to: {args.edges_output}; edges={n}")

    if args.visual_output:
        print(f"visual groups written to: {args.visual_output}; groups={visual_count}")

    if cascade_stats and not args.quiet:
        print("[stream] cascade size distribution (ALL closed cascades, pre min_group_events filter):")
        for bucket in cascade_stats["histogram"]:
            print(
                f"  size={bucket['label']:>5s} : "
                f"{bucket['cascade_count']:>7d} cascades, "
                f"{bucket['event_count']:>7d} events"
            )
        print(
            f"[stream] multi(>=2) cascades: "
            f"{cascade_stats['multi_event_cascade_count']}/{cascade_stats['n_cascades']} "
            f"({cascade_stats['multi_event_cascade_share'] * 100:.1f}% of cascades, "
            f"{cascade_stats['multi_event_event_share'] * 100:.1f}% of events); "
            f"mean={cascade_stats['mean_size']:.2f}, max={cascade_stats['max_size']}"
        )

    total = time.monotonic() - t_total_start
    if not args.quiet:
        if total < 60:
            print(f"[stream] total wall-clock: {total:.1f}s")
        else:
            print(f"[stream] total wall-clock: {int(total // 60)}m{total % 60:04.1f}s")


if __name__ == "__main__":
    main()
