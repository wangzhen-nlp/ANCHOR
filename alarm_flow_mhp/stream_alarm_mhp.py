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
from alarm_flow_brunch.visual_output import write_visual_groups
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.sequences import alarm_type_label, event_type_label
from alarm_flow_mhp.aggregator import (
    AlarmMHPConfig,
    load_alarm_mhp_artifact,
    summarize_alarm_event,
)
from fault_grouping.alarm_events.io import is_clear_alarm
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON, resource_display


EPS = 1e-12


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


@dataclass
class Cascade:
    """A connected component of (parent, child) links."""

    cascade_id: int
    events: list = field(default_factory=list)        # list of OnlineEvent
    root_index: int = -1                              # index of root event
    last_ts: float = 0.0
    start_ts: float = 0.0

    def add(self, event: OnlineEvent):
        if not self.events:
            self.root_index = event.index
            self.start_ts = event.ts
        self.events.append(event)
        self.last_ts = max(self.last_ts, event.ts)

    def event_count(self) -> int:
        return len(self.events)


# --------------------------------------------------------------------------
# Streaming assigner: the inference engine
# --------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Inference-time knobs (mostly inherited from artifact.config)."""

    history_window_sec: float = 900.0       # sliding window for candidate parents
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

    @classmethod
    def from_artifact_config(cls, cfg: AlarmMHPConfig, **overrides):
        base = cls(
            history_window_sec=cfg.history_window_sec,
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


class StreamMHPAssigner:
    """Stateful online inference engine.

    Maintains:
      - `_buf_*`: parallel arrays (events/ts/type) + head pointer holding real
        events still within history_window, for vectorized candidate scoring
      - `cascades`: dict cascade_id → Cascade for cascades still considered active
      - vocabs for label → id resolution

    Per incoming alarm:
      1. Drop expired events from `recent`
      2. Look up type_id; if unseen, mark as immigrant (fall-back)
      3. Score candidates in recent: α[u_target, u_source] · β · exp(-β·Δt)
      4. Compare against μ[u_target] · immigrant_bias
      5. argmax → assign to parent's cascade, or start a new immigrant cascade
      6. Append to recent
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
        # Sliding window of recent events kept as parallel arrays with a head
        # pointer (amortized O(1) append + eviction, O(cap) tail slice for
        # vectorized scoring). `_buf_events` holds the OnlineEvent objects;
        # `_buf_ts` / `_buf_type` are numpy-friendly parallel columns.
        self._buf_events: list[OnlineEvent] = []
        self._buf_ts: list[float] = []
        self._buf_type: list[int] = []
        self._head: int = 0                      # logical front index into the bufs
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

    def _score_batch(self, target_type_id: int, src_types: np.ndarray, dts: np.ndarray) -> np.ndarray:
        """Vectorized kernel score for a batch of candidates — sparse binary
        search over sorted edge keys, dispatching on kernel_type. Bit-for-bit
        consistent with params.pair_score (f64, Δt>=0 gate; only Δt<0 excluded).
        """
        n = len(src_types)
        out = np.zeros(n, dtype=np.float64)
        if self._E == 0 or n == 0:
            return out
        keys = int(target_type_id) * self._M + src_types
        idx = np.minimum(np.searchsorted(self._edge_keys, keys), self._E - 1)
        # Gate only dt < 0 (keep dt == 0 → full peak), matching pair_score /
        # compute_hard_parents / training so run and stream agree on
        # simultaneous events.
        valid = (self._edge_keys[idx] == keys) & (dts >= 0)
        if not valid.any():
            return out
        vi = idx[valid]
        if self.params.kernel_type == "piecewise":
            from mhp.params import bucket_index_vec

            pb = bucket_index_vec(dts[valid], self._bucket_edges)
            out[valid] = self._theta[vi, pb]
        else:
            a = self._edge_alpha[vi]
            b = self._edge_beta[vi]
            out[valid] = a * b * np.exp(-b * dts[valid])
        return out

    def _candidate_scores(self, target_type_id: int, target_ts: float):
        """Score the most recent `max_history_events` candidates in one numpy
        batch.

        Returns (positions, scores) where positions[k] is the ABSOLUTE buffer
        index of candidate k (so the caller can fetch the OnlineEvent), and
        scores[k] is its kernel score against the target.
        """
        n = len(self._buf_ts)
        live = n - self._head
        if live <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        cap = min(live, self.config.max_history_events)
        lo = n - cap                                  # tail start (newest `cap`)
        positions = np.arange(lo, n, dtype=np.int64)
        ts_tail = np.asarray(self._buf_ts[lo:], dtype=np.float64)
        type_tail = np.asarray(self._buf_type[lo:], dtype=np.int64)
        dts = (target_ts - ts_tail) / self.config.time_scale_sec
        scores = self._score_batch(target_type_id, type_tail, dts)
        return positions, scores

    def _candidate_scores_feature(self, target_at: str, target_ne: str, target_ts: float):
        """Feature-mode candidate scoring: live α = softplus(w·φ) per source
        candidate (device-OPEN), then α·β·exp(-β·Δt). Returns (positions, scores).
        """
        n = len(self._buf_ts)
        live = n - self._head
        if live <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        cap = min(live, self.config.max_history_events)
        lo = n - cap
        positions = np.arange(lo, n, dtype=np.int64)
        src_events = self._buf_events[lo:]
        src_ats = [e.alarm_type for e in src_events]
        src_nes = [e.ne for e in src_events]
        ts_tail = np.asarray(self._buf_ts[lo:], dtype=np.float64)
        dts = (target_ts - ts_tail) / self.config.time_scale_sec
        alpha = self.feature_scorer.alpha_for_target(target_at, target_ne, src_ats, src_nes)
        # α floor: treat too-weak edges as non-edges (inference analog of
        # device-mode edge_threshold) — guards the soft model against linking
        # unrelated pairs whose baseline α is small but positive.
        if self._feat_alpha_floor > 0:
            alpha = np.where(alpha >= self._feat_alpha_floor, alpha, 0.0)
        b = self._feat_beta
        scores = np.where(dts >= 0, alpha * b * np.exp(-b * dts), 0.0)
        return positions, scores

    def process(self, alarm_event: dict):
        """Ingest one alarm event in time order, run inference, update state."""
        # Filtering: skip events we can't model
        if is_clear_alarm(alarm_event.get("alarm", {}) if isinstance(alarm_event, dict) else {}):
            self.dropped_clear += 1
            return None
        atype = alarm_type_label(alarm_event)
        if atype is None:
            self.dropped_no_type += 1
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
                return None
            type_id, type_label = resolved
            ne = ""

        ts = float(alarm_event.get("ts", 0.0))
        # Close cascades silent past close_inactive_sec. This only governs
        # OUTPUT timing — its order relative to scoring is irrelevant: when
        # close_inactive_sec >= history_window_sec a closable cascade has no
        # in-window candidate parents anyway, so it can never catch a new event
        # regardless of when we close it (same reason the close-scan throttle is
        # result-preserving).
        self._close_inactive(ts)
        self._evict_expired(ts)

        # Score candidates + immigrant baseline μ. Feature mode is INDUCTIVE:
        # μ = softplus(w_μ·ψ) from the type's own features (parameterized) —
        # seen and new devices treated identically, no per-device memorization.
        # Falls back to the per-alarm-type table if no μ scorer / attrs missing.
        if self.feature_mode:
            positions, scores = self._candidate_scores_feature(atype, ne, ts)
            if self.mu_scorer is not None:
                base_mu = self.mu_scorer.mu_for(atype, ne)
            else:
                base_mu = self._mu_by_at.get(atype, self._mu_default)
            mu = base_mu * self.config.immigrant_bias
        else:
            positions, scores = self._candidate_scores(type_id, ts)
            mu = float(self.params.mu[type_id]) * self.config.immigrant_bias
        # argmax over (μ, top_candidate). Use strict `< mu` so a candidate that
        # ties μ binds to its parent — matching training's compute_hard_parents
        # (immigrant iff mu > best_score, i.e. candidate wins on a tie).
        if scores.size == 0 or scores.max() < mu:
            # Immigrant: start a new cascade
            cascade_id = self._next_cascade_id
            self._next_cascade_id += 1
            cascade = Cascade(cascade_id=cascade_id)
            self.cascades[cascade_id] = cascade
            parent_index = -1
            parent_score = mu
            self.total_immigrants += 1
        else:
            # Bind to most-likely parent's cascade
            best_local = int(scores.argmax())
            parent_event = self._buf_events[int(positions[best_local])]
            cascade_id = parent_event.cascade_id
            cascade = self.cascades.get(cascade_id)
            if cascade is None:
                # Edge case: parent's cascade was already closed. Fall back
                # to immigrant rather than reviving the closed cascade —
                # closed-cascade output already emitted.
                cascade_id = self._next_cascade_id
                self._next_cascade_id += 1
                cascade = Cascade(cascade_id=cascade_id)
                self.cascades[cascade_id] = cascade
                parent_index = -1
                parent_score = mu
                self.total_immigrants += 1
            else:
                parent_index = parent_event.index
                parent_score = float(scores[best_local])

        event = OnlineEvent(
            index=self._next_event_index,
            ts=ts,
            type_id=type_id,
            type_label=type_label,
            alarm=alarm_event,
            parent_index=parent_index,
            cascade_id=cascade_id,
            parent_score=parent_score,
            alarm_type=atype if self.feature_mode else "",
            ne=ne,
        )
        self._next_event_index += 1
        cascade.add(event)
        self._buf_events.append(event)
        self._buf_ts.append(ts)
        self._buf_type.append(type_id)
        self.total_events_processed += 1
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
            if cascade.last_ts < cutoff:
                to_close.append(cid)
        for cid in to_close:
            cascade = self.cascades.pop(cid)
            self._record_closed(cascade)

    def close_remaining(self):
        """Emit any still-active cascades. Call once at end of stream."""
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
            self.closed_groups.append(_cascade_to_group(cascade))
            self.emitted_group_count += 1

    def stats(self) -> dict:
        return {
            "total_events_processed": self.total_events_processed,
            "total_immigrants": self.total_immigrants,
            "closed_cascade_count": self.closed_cascade_count,
            "open_cascade_count": len(self.cascades),
            "recent_window_size": len(self._buf_ts) - self._head,
            "dropped_clear": self.dropped_clear,
            "dropped_no_type": self.dropped_no_type,
            "dropped_unknown_type": self.dropped_unknown_type,
        }


# --------------------------------------------------------------------------
# Output formatting (group records compatible with alarm_flow_brunch)
# --------------------------------------------------------------------------


def _cascade_to_group(cascade: Cascade) -> dict:
    summaries = [summarize_alarm_event(e.alarm, e.index) for e in cascade.events]
    # Root is the event whose parent_index == -1 (immigrant), or the earliest by ts
    root = next((e for e in cascade.events if e.parent_index == -1), cascade.events[0])
    root_summary = summarize_alarm_event(root.alarm, root.index)
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


def _run_imputation(artifact, alarm_events, args, stream_config, quiet=False):
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
    if feature_mode:
        floor = args.feature_alpha_floor or None
        adapter = feature_adapter_from_artifact(artifact, args.ne_graph, alpha_floor=floor)
    else:
        if getattr(artifact.params, "kernel_type", "exp") != "exp":
            raise NotImplementedError(
                "--impute device path supports exp kernel only (piecewise adapter TODO)"
            )
        adapter = device_adapter_from_artifact(artifact)

    history = float(stream_config.history_window_sec)
    lag = float(args.impute_lag_sec) if args.impute_lag_sec is not None else history
    cfg = SamplerConfig(
        lag_sec=lag,
        history_window_sec=history,
        sweeps_per_tick=int(args.impute_sweeps),
        missing_log_prior=float(args.impute_kappa),
        max_depth=int(args.impute_max_depth),
        max_births_per_sweep=int(args.impute_max_births),
        seed=int(getattr(artifact.config, "seed", 0) or 0),
    )
    sampler = MissingChainSampler(adapter, cfg)
    if not quiet:
        print(
            f"[stream] impute: mode={'feature' if feature_mode else 'device'} "
            f"lag={cfg.lag_sec:.0f}s history={cfg.history_window_sec:.0f}s "
            f"sweeps={cfg.sweeps_per_tick} kappa={cfg.missing_log_prior} "
            f"max_depth={cfg.max_depth}",
            flush=True,
        )

    type_fields = artifact.config.type_fields
    groups: list = []
    dropped = {"clear": 0, "no_type": 0, "unknown_type": 0}
    processed = 0
    for i, alarm in enumerate(alarm_events):
        if is_clear_alarm(alarm.get("alarm", {}) if isinstance(alarm, dict) else {}):
            dropped["clear"] += 1
            continue
        atype = alarm_type_label(alarm)
        if atype is None:
            dropped["no_type"] += 1
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
                continue
            type_key = int(tid)
            meta["type_label"] = type_label
        processed += 1
        groups.extend(sampler.ingest(float(alarm.get("ts", 0.0)), type_key, meta))
    groups.extend(sampler.flush())

    # Output filter: keep cascades with at least min_group_events real alarms.
    min_ev = int(stream_config.min_group_events)
    if min_ev > 1:
        groups = [g for g in groups if g["real_event_count"] >= min_ev]
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
        help="Optional visualization JSONL compatible with the fault group browser and propagation visualizer.",
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
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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
    )
    if not args.quiet:
        print(
            f"[stream] config: history_window={stream_config.history_window_sec:.0f}s "
            f"max_history={stream_config.max_history_events} "
            f"close_inactive={stream_config.close_inactive_sec:.0f}s "
            f"min_group_events={stream_config.min_group_events} "
            f"immigrant_bias={stream_config.immigrant_bias}",
            flush=True,
        )

    # ---- missing-chain imputation path (fixed-lag sampler) --------------
    if args.impute:
        t_imp_start = time.monotonic()
        groups, impute_stats = _run_imputation(
            artifact, alarm_events, args, stream_config, quiet=args.quiet
        )
        elapsed = time.monotonic() - t_imp_start
        if not args.quiet:
            print(
                f"[stream] impute done: groups={len(groups)}, "
                f"births={impute_stats['births']}, deaths={impute_stats['deaths']}, "
                f"processed={impute_stats['processed']}, elapsed={elapsed:.1f}s",
                flush=True,
            )
        metadata = {
            "algorithm": "alarm_flow_mhp.stream+impute",
            "model": os.path.abspath(args.model),
            "input": os.path.abspath(args.alarms),
            "alarm_metadata": alarm_metadata,
            "regions": sorted(regions) if regions else [],
            "config": {
                "history_window_sec": stream_config.history_window_sec,
                "min_group_events": stream_config.min_group_events,
                "time_scale_sec": stream_config.time_scale_sec,
                "impute_lag_sec": float(args.impute_lag_sec) if args.impute_lag_sec is not None else stream_config.history_window_sec,
                "impute_sweeps": args.impute_sweeps,
                "impute_kappa": args.impute_kappa,
                "impute_max_depth": args.impute_max_depth,
            },
            "group_count": len(groups),
            "modeled_event_count": impute_stats["processed"],
            "imputed_missing_events": sum(g.get("virtual_event_count", 0) for g in groups),
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
            visual_count = write_visual_groups(
                args.visual_output,
                groups,
                ne_graph_path=args.ne_graph,
                site_graph_path=args.site_graph,
                ne_scope=args.visual_ne_scope,
            )
            print(f"visual groups written to: {args.visual_output}; groups={visual_count}")
        return
    if stream_config.close_inactive_sec < stream_config.history_window_sec:
        print(
            f"[stream] WARN: close_inactive_sec ({stream_config.close_inactive_sec:.0f}s) < "
            f"history_window_sec ({stream_config.history_window_sec:.0f}s). A cascade can be "
            f"closed while its events are still candidate parents, turning would-be children "
            f"into 'orphan' immigrants. Set close_inactive_sec >= history_window_sec to avoid this.",
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
        feature_scorer = RuntimeFeatureScorer(
            kernel=FeatureKernel.from_dict(fk),
            at_vocab=rt.get("at_vocab", []),
            graph_context=graph_ctx,
            topology_index=topo_idx,
            beta=float(rt.get("beta", 1.0)),
        )
        if not args.quiet:
            print(
                f"[stream] feature scorer: {feature_scorer.layout.n_features} features, "
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

    t_stream_start = time.monotonic()
    last_print = t_stream_start
    for i, alarm in enumerate(alarm_events):
        assigner.process(alarm)
        if args.progress_every and (i + 1) % args.progress_every == 0 and not args.quiet:
            now = time.monotonic()
            elapsed = now - t_stream_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            stats = assigner.stats()
            print(
                f"[stream] processed {i + 1}/{len(alarm_events)} "
                f"({rate:.0f} evt/s, "
                f"open={stats['open_cascade_count']}, "
                f"closed={stats['closed_cascade_count']}, "
                f"immigrants={stats['total_immigrants']}, "
                f"elapsed={elapsed:.1f}s)",
                flush=True,
            )
            last_print = now

    # Drain
    if not args.quiet:
        print("[stream] draining remaining cascades ...", flush=True)
    assigner.close_remaining()
    t_stream_end = time.monotonic()

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
            "max_history_events": stream_config.max_history_events,
            "close_inactive_sec": stream_config.close_inactive_sec,
            "min_group_events": stream_config.min_group_events,
            "immigrant_bias": stream_config.immigrant_bias,
            "time_scale_sec": stream_config.time_scale_sec,
        },
        "group_count": len(groups),
        "emitted_group_count": len(groups),
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
        visual_count = write_visual_groups(
            args.visual_output,
            groups,
            ne_graph_path=args.ne_graph,
            site_graph_path=args.site_graph,
            ne_scope=args.visual_ne_scope,
        )
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
