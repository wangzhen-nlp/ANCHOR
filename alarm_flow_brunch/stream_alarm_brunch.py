#!/usr/bin/env python3
"""Online-style alarm-flow BRUNCH inference.

This entrypoint keeps trained BRUNCH parameters fixed and assigns each incoming
alarm to a cascade using only a small reorder buffer plus recent parent
candidates. It is intended for low-latency serving, not for re-running the full
offline BRUNCH sampler.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import heapq
import json
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_brunch.aggregator import (
    load_alarm_brunch_artifact,
    summarize_alarm_event,
)
from alarm_flow_brunch.missing_intervals import (
    MissingIntervalTracker,
    load_missing_from_json,
)
from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_brunch.visual_output import AlarmBRUNCHVisualOutputSession
from alarm_flow_isahp.sequences import alarm_type_label, event_type_label
from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_tools.alarm_types import CRITICAL_ALARMS
from alarm_tools.progress_utils import ProgressBar
from brunch.kernels import exp_kernel
from fault_grouping.alarm_events.io import (
    apply_clear_delay,
    is_clear_alarm,
    parse_datetime_text,
)
from fault_grouping.alarm_events.sorted_cache import (
    is_sorted_alarm_cache_file,
    load_sorted_alarm_cache,
)
from fault_grouping.tools.prepare_sorted_alarms import _load_valid_sites_and_ne_mapping
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON, resource_display
from topology_tools.region_utils import allowed_devices_for_regions, load_ne_graph


EPS = 1e-12
_OFFLINE_REORDER_LAG_SEC = 0.0
_LIVE_REORDER_LAG_SEC = 300.0


@dataclass
class OnlineEvent:
    index: int
    event: dict
    type_id: int
    type_label: str
    event_id: str
    cascade_id: str = ""
    # Shelton-style missing-data extension: virtual events are sampled forward
    # in time during open MissingIntervals. They participate as candidate
    # parents like real events but with their score dampened by `confidence`,
    # and they are filtered out of cascade output unless they actually attached
    # to a downstream real event.
    virtual: bool = False
    confidence: float = 1.0
    virtual_source: str = ""   # MissingInterval key that produced this event, if any

    @property
    def ts(self) -> float:
        return float(self.event.get("ts", 0.0))


@dataclass
class OnlineCascade:
    cascade_id: str
    root: OnlineEvent
    events: list[OnlineEvent] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    # last_ts / real_event_count / virtual_event_count are read on the close-
    # inactive hot path (once per active cascade, per real event tick). We
    # maintain them as O(1) attributes updated in add() rather than scanning
    # self.events every read — a 10k-cascade × 1k-event/s stream would burn an
    # entire core just on those scans otherwise.
    _last_ts: float = field(default=float("-inf"))
    _real_count: int = 0
    _virtual_count: int = 0

    def add(self, event: OnlineEvent, parent: OnlineEvent | None = None):
        event.cascade_id = self.cascade_id
        self.events.append(event)
        if event.ts > self._last_ts:
            self._last_ts = event.ts
        if event.virtual:
            self._virtual_count += 1
        else:
            self._real_count += 1
        if parent is not None and parent.index != event.index:
            self.edges.append(
                {
                    "source_index": int(parent.index),
                    "target_index": int(event.index),
                    "source_event_id": parent.event_id,
                    "target_event_id": event.event_id,
                    "source_type": parent.type_label,
                    "target_type": event.type_label,
                }
            )

    @property
    def last_ts(self) -> float:
        return self._last_ts

    def real_event_count(self) -> int:
        return self._real_count

    def virtual_event_count(self) -> int:
        return self._virtual_count

    def to_group(self):
        summaries = []
        for item in self.events:
            summary = summarize_alarm_event(item.event, item.index)
            if item.virtual:
                # virtual events have no underlying alarm dict; annotate the
                # summary so downstream UIs can render them as low-confidence
                # "inferred" symptoms.
                summary["virtual"] = True
                summary["confidence"] = float(item.confidence)
                summary["virtual_source"] = item.virtual_source
            else:
                summary["virtual"] = False
                summary["confidence"] = 1.0
            summaries.append(summary)
        timestamps = [summary["ts"] for summary in summaries]
        return {
            "group_id": self.cascade_id,
            "cascade_id": self.cascade_id,
            "event_count": len(self.events),
            "real_event_count": self.real_event_count(),
            "virtual_event_count": self.virtual_event_count(),
            "start_ts": min(timestamps),
            "end_ts": max(timestamps),
            "duration_sec": max(timestamps) - min(timestamps),
            "root_event": summaries[0] if self.root is self.events[0] else summarize_alarm_event(self.root.event, self.root.index),
            "root_virtual": bool(self.root.virtual),
            "site_list": sorted({summary["site_id"] for summary in summaries if summary["site_id"]}),
            "alarm_source_list": sorted(
                {summary["alarm_source"] for summary in summaries if summary["alarm_source"]}
            ),
            "alarm_title_counts": dict(
                Counter(summary["alarm_title"] for summary in summaries if summary["alarm_title"])
            ),
            "alarm_type_counts": dict(
                Counter(summary["alarm_type"] for summary in summaries if summary["alarm_type"])
            ),
            "symptoms": summaries,
            "edges": list(self.edges),
        }


class ReorderBuffer:
    def __init__(self, lag_sec: float):
        if lag_sec < 0:
            raise ValueError("reorder_lag_sec must be non-negative")
        self.lag_sec = float(lag_sec)
        self.max_seen_ts = -math.inf
        self._seq = 0
        self._heap = []

    def push(self, event: dict):
        ts = float(event.get("ts", 0.0))
        self.max_seen_ts = max(self.max_seen_ts, ts)
        self._seq += 1
        heapq.heappush(self._heap, (ts, self._seq, event))

    def ready(self):
        watermark = self.max_seen_ts - self.lag_sec
        while self._heap and self._heap[0][0] <= watermark:
            yield heapq.heappop(self._heap)[2]

    def flush(self):
        while self._heap:
            yield heapq.heappop(self._heap)[2]

    @property
    def pending_count(self):
        return len(self._heap)


class OnlineBRUNCHAssigner:
    def __init__(
        self,
        artifact,
        *,
        config,
        active_window_sec: float | None = None,
        parent_selection: str | None = None,
        seed: int = 0,
        missing_tracker=None,
        virtual_confidence: float = 0.5,
        max_virtual_per_call: int = 50,
        max_virtual_per_dim: int = 100,
        close_check_min_interval_sec: float = 1.0,
    ):
        self.artifact = artifact
        self.config = config
        self.params = artifact.params
        self.vocabs = artifact.vocabs
        self.active_window_sec = float(
            active_window_sec if active_window_sec is not None else config.history_window_sec
        )
        if self.active_window_sec <= 0:
            raise ValueError("active_window_sec must be positive")
        self.parent_selection = parent_selection or config.parent_selection
        if self.parent_selection not in {"sample", "argmax"}:
            raise ValueError("parent_selection must be sample or argmax")
        if not 0.0 < virtual_confidence <= 1.0:
            raise ValueError("virtual_confidence must be in (0, 1]")
        if max_virtual_per_call < 0 or max_virtual_per_dim < 0:
            raise ValueError("virtual sampling caps must be non-negative")
        if close_check_min_interval_sec < 0:
            raise ValueError("close_check_min_interval_sec must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.recent: deque[OnlineEvent] = deque()
        self._active_sources_cache: dict[int, set[int]] = {}
        self.cascades: dict[str, OnlineCascade] = {}
        # Closed cascades are deleted from self.cascades and their events dropped
        # from self.recent (see close_inactive / close_remaining). For a long-
        # running stream we never want unbounded retention; "is this source
        # cascade still alive?" reduces to a membership check on self.cascades.
        self.branching_edge_count = 0
        self.skipped_count = 0
        self.processed_count = 0
        self.modeled_count = 0
        self._next_index = 0
        self._next_cascade_ordinal = 1
        # Missing-data support
        self.missing_tracker = missing_tracker
        self.virtual_confidence = float(virtual_confidence)
        self.max_virtual_per_call = int(max_virtual_per_call)
        self.max_virtual_per_dim = int(max_virtual_per_dim)
        self.virtual_event_count = 0
        self.virtual_picked_as_parent_count = 0
        # close_inactive throttle (event-time, not wall-clock). 0 disables.
        self._close_check_min_interval = float(close_check_min_interval_sec)
        self._last_close_check_ts = float("-inf")
        # Cumulative counters survive across cascade deletions so the final
        # metadata snapshot still reflects the full stream even after we've
        # garbage-collected closed cascades to bound memory.
        # - total_cascade_count: every cascade ever created (includes pure-
        #   virtual cascades that never had a real event attached).
        # - closed_cascade_count: cascades dropped from self.cascades.
        # - reportable_cascade_count: cascades that passed
        #   _include_cascade_in_output (>= min_group_events real events) at
        #   the moment they closed — i.e. what actually appears in groups
        #   snapshots / visual output. Use this if you want "user-visible"
        #   cascade count.
        self.total_cascade_count = 0
        self.closed_cascade_count = 0
        self.reportable_cascade_count = 0
        # Most recent event time observed by either assign() or close_inactive,
        # used to keep cutoffs monotonic in the presence of late events.
        self._max_seen_ts = float("-inf")

    def assign(self, event: dict):
        self.processed_count += 1
        target_ts = float(event.get("ts", 0.0))
        # Inject Shelton-style virtual events for any open MissingIntervals up
        # to this event's timestamp BEFORE deciding whether to skip this real
        # event, so the missing-data completion proceeds even when the current
        # real event is itself a clear/unknown that we wouldn't process.
        if (
            self.missing_tracker is not None
            and self.missing_tracker.has_intervals_needing_sampling(target_ts)
        ):
            self._sample_virtual_events_until(target_ts)

        if not self.config.include_clear and is_clear_alarm(event.get("alarm", {})):
            return self._skip(event, "clear_alarm_disabled")

        coarse_type = alarm_type_label(event)
        if coarse_type is None:
            return self._skip(event, "unsupported_alarm_type")

        type_label = event_type_label(event, self.config.type_fields)
        type_id = self.vocabs.type_vocab.get(type_label)
        if type_id is None:
            return self._skip(event, "unknown_event_type", details={"type_label": type_label})

        self._expire_recent(target_ts)
        event_index = self._next_index
        self._next_index += 1
        online_event = OnlineEvent(
            index=event_index,
            event=event,
            type_id=int(type_id),
            type_label=type_label,
            event_id=summarize_alarm_event(event, event_index)["event_id"],
        )

        candidates = self._candidate_scores(online_event)
        chosen_parent, chosen_score, total_score = self._choose_candidate(candidates)
        parent_is_virtual = False
        if chosen_parent is None:
            cascade = self._create_cascade(online_event)
            reason = "new_cascade"
            parent_event_id = ""
            dt_sec = None
        else:
            cascade = self.cascades[chosen_parent.cascade_id]
            cascade.add(online_event, parent=chosen_parent)
            self.branching_edge_count += 1
            reason = "assigned"
            parent_event_id = chosen_parent.event_id
            dt_sec = online_event.ts - chosen_parent.ts
            parent_is_virtual = bool(chosen_parent.virtual)
            if parent_is_virtual:
                self.virtual_picked_as_parent_count += 1

        self._push_recent_ordered(online_event)
        self.modeled_count += 1
        probability = float(chosen_score / total_score) if total_score > 0.0 else 1.0
        log_score = float(math.log(max(chosen_score, EPS)))
        return {
            "status": "clustered",
            "cascade_id": cascade.cascade_id,
            "event_id": online_event.event_id,
            "ts": online_event.ts,
            "alarm_title": str(event.get("alarm_title", "") or ""),
            "alarm_source": str(event.get("alarm_source", "") or ""),
            "site_id": str(event.get("site_id", "") or ""),
            "reason": reason,
            "probability": probability,
            "candidate_count": len(candidates),
            "log_score": log_score,
            "details": {
                "parent_event_id": parent_event_id,
                "parent_virtual": parent_is_virtual,
                "target_type": type_label,
                "parent_type": chosen_parent.type_label if chosen_parent is not None else "",
                "dt_sec": dt_sec,
                "parent_selection": self.parent_selection,
            },
        }

    def _include_cascade_in_output(self, cascade, min_group_events: int) -> bool:
        """A cascade is reported only when it covers at least one real (observed)
        event. Pure-virtual cascades (no real event ever attached) represent
        noise from the background virtual sampler and are dropped.
        """
        if len(cascade.events) < min_group_events:
            return False
        return cascade.real_event_count() > 0

    def groups(self, min_group_events: int = 1):
        # Python dict preserves insertion order, and self.cascades is only
        # extended by _create_cascade (which assigns monotonically increasing
        # cascade ordinals → root.index). So iterating self.cascades.values()
        # is already root.index-ascending without the cost of a sort.
        return [
            cascade.to_group()
            for cascade in self.cascades.values()
            if self._include_cascade_in_output(cascade, min_group_events)
        ]

    def close_inactive(self, now_ts, close_after_sec, *, min_group_events=1):
        # Event-time throttle: under high event rate this method is called per
        # tick (when --visual-output is on), and on most ticks the cutoff has
        # not advanced far enough for any cascade to become closable. Skipping
        # those calls amortises the O(K) scan to one full scan every
        # _close_check_min_interval seconds of event time.
        now_ts = float(now_ts)
        self._max_seen_ts = max(self._max_seen_ts, now_ts)
        if (
            self._close_check_min_interval > 0.0
            and self._max_seen_ts - self._last_close_check_ts < self._close_check_min_interval
        ):
            return []
        self._last_close_check_ts = self._max_seen_ts
        cutoff = self._max_seen_ts - float(close_after_sec)
        closed_groups = []
        closed_ids: list[str] = []
        # Single O(K) scan, no sort, O(1) last_ts lookup.
        for cascade in self.cascades.values():
            if cascade.last_ts > cutoff:
                continue
            closed_ids.append(cascade.cascade_id)
            if self._include_cascade_in_output(cascade, min_group_events):
                closed_groups.append(cascade.to_group())
                self.reportable_cascade_count += 1
        if closed_ids:
            closed_set = set(closed_ids)
            for cid in closed_ids:
                del self.cascades[cid]
            self.closed_cascade_count += len(closed_ids)
            # Orphan handling: when close_after_sec >= active_window_sec (the
            # default configuration) every event of a just-closed cascade is
            # already past _expire_recent's cutoff and will be popped on the
            # next assign(), so the rebuild below is pure overhead and we skip
            # it. Under tight configs (close < active) orphans would otherwise
            # linger in self.recent for up to (active - close) seconds: walk
            # self.recent once against the small closed_set to evict them
            # immediately. Correctness is preserved either way — the
            # _candidate_scores defensive skip already excludes orphans — this
            # only reclaims memory sooner and removes wasted iteration cost.
            if float(close_after_sec) < self.active_window_sec and self.recent:
                self.recent = deque(
                    event for event in self.recent if event.cascade_id not in closed_set
                )
        return closed_groups

    def close_remaining(self, *, min_group_events=1):
        # self.cascades is a dict, so iteration is insertion order, which by
        # construction matches cascade-ordinal == root.index ascending — no
        # explicit sort needed for deterministic output.
        remaining_groups = [
            cascade.to_group()
            for cascade in self.cascades.values()
            if self._include_cascade_in_output(cascade, min_group_events)
        ]
        self.closed_cascade_count += len(self.cascades)
        self.reportable_cascade_count += len(remaining_groups)
        # End-of-input: drop everything. A live stream that wants to keep state
        # across windows should be using close_inactive() with a TTL instead.
        self.cascades.clear()
        self.recent.clear()
        return remaining_groups

    def metadata(self):
        meta = {
            "algorithm": "alarm_flow_brunch_online",
            "config": self.config.to_dict(),
            "online": {
                "active_window_sec": self.active_window_sec,
                "parent_selection": self.parent_selection,
            },
            "processed_event_count": self.processed_count,
            "modeled_event_count": self.modeled_count,
            "skipped_event_count": self.skipped_count,
            # group_count = currently active in memory (live gauge); after
            # close_remaining this is 0. Use total_cascade_count for the
            # cumulative count over the full stream — main() picks the right
            # one for final reporting.
            "group_count": len(self.cascades),
            "active_cascade_count": len(self.cascades),
            "total_cascade_count": int(self.total_cascade_count),
            "closed_cascade_count": int(self.closed_cascade_count),
            # Number of cascades that actually showed up in groups output
            # (i.e. passed _include_cascade_in_output). Differs from
            # total_cascade_count because pure-virtual cascades are filtered.
            "reportable_cascade_count": int(self.reportable_cascade_count),
            "branching_edge_count": int(self.branching_edge_count),
            "type_count": len(self.vocabs.type_vocab),
            "active_edge_count": len(self.params.active_edges(include_self=True)[0]),
            "type_labels": list(self.vocabs.type_vocab.labels),
        }
        if self.missing_tracker is not None:
            meta["missing_data"] = {
                **self.missing_tracker.stats(),
                "virtual_event_count": int(self.virtual_event_count),
                "virtual_picked_as_parent_count": int(self.virtual_picked_as_parent_count),
                "virtual_confidence": float(self.virtual_confidence),
                "max_virtual_per_call": int(self.max_virtual_per_call),
                "max_virtual_per_dim": int(self.max_virtual_per_dim),
            }
        return meta

    def _skip(self, event, reason, *, details=None):
        self.skipped_count += 1
        summary = summarize_alarm_event(event, self.processed_count)
        return {
            "status": "skipped",
            "cascade_id": "",
            "event_id": summary["event_id"],
            "ts": float(event.get("ts", 0.0)),
            "alarm_title": str(event.get("alarm_title", "") or ""),
            "alarm_source": str(event.get("alarm_source", "") or ""),
            "site_id": str(event.get("site_id", "") or ""),
            "reason": reason,
            "details": details or {},
        }

    def _expire_recent(self, target_ts):
        self._max_seen_ts = max(self._max_seen_ts, float(target_ts))
        cutoff = self._max_seen_ts - self.active_window_sec
        while self.recent and self.recent[0].ts < cutoff:
            self.recent.popleft()

    def _candidate_scores(self, target: OnlineEvent):
        candidates: list[tuple[OnlineEvent | None, float]] = []
        immigrant_score = max(float(self.params.mu[target.type_id]), EPS)
        candidates.append((None, immigrant_score))
        allowed_sources = self._active_sources(target.type_id)
        max_history = int(self.config.max_history_events)
        checked = 0
        for source in reversed(self.recent):
            # Defensive: an event from a since-deleted cascade should never be
            # in self.recent (close_inactive drops them), but skip just in case.
            if source.cascade_id not in self.cascades:
                continue
            dt_sec = target.ts - source.ts
            if dt_sec <= 0.0:
                continue
            if dt_sec > self.active_window_sec:
                break
            checked += 1
            if checked > max_history:
                break
            if source.type_id not in allowed_sources:
                continue
            alpha = float(self.params.alpha_value(target.type_id, source.type_id))
            if alpha <= 0.0:
                continue
            beta = float(self.params.beta_value(target.type_id, source.type_id))
            dt_scaled = dt_sec / float(self.config.time_scale_sec)
            score = alpha * float(exp_kernel(dt_scaled, beta))
            # Virtual parents are dampened by their confidence so they can be
            # picked when no real candidate fits, but cannot dominate a real
            # candidate with comparable kernel mass.
            if source.virtual:
                score *= float(source.confidence)
            if score > EPS:
                candidates.append((source, score))
        return candidates

    def _active_sources(self, target_type_id):
        target_type_id = int(target_type_id)
        cached = self._active_sources_cache.get(target_type_id)
        if cached is not None:
            return cached
        sources = set(
            int(source)
            for source in self.params.active_sources_for_target(target_type_id, include_self=True)
        )
        self._active_sources_cache[target_type_id] = sources
        return sources

    def _push_recent_ordered(self, event: OnlineEvent):
        """Push an event to the recent queue, maintaining strict chronological order."""
        if not self.recent or event.ts >= self.recent[-1].ts:
            self.recent.append(event)
            return
        idx = len(self.recent) - 1
        while idx >= 0 and self.recent[idx].ts > event.ts:
            idx -= 1
        self.recent.insert(idx + 1, event)

    def _choose_candidate(self, candidates):
        scores = np.asarray([score for _, score in candidates], dtype=np.float64)
        total = float(scores.sum())
        if total <= 0.0:
            return None, 1.0, 1.0
        if self.parent_selection == "argmax":
            idx = int(np.argmax(scores))
        else:
            probs = scores / total
            idx = int(self.rng.choice(len(candidates), p=probs))
        parent, score = candidates[idx]
        return parent, float(score), total

    def _create_cascade(self, event: OnlineEvent):
        cascade_id = f"brunch-online-{self._next_cascade_ordinal:06d}"
        self._next_cascade_ordinal += 1
        cascade = OnlineCascade(cascade_id=cascade_id, root=event)
        cascade.add(event)
        self.cascades[cascade_id] = cascade
        self.total_cascade_count += 1
        return cascade

    # ---- missing-data (Shelton-style virtual event injection) -------------
    def declare_missing(self, key_kind: str, key_value, start_ts: float):
        """Open a new missing interval. Returns the created MissingInterval."""
        if self.missing_tracker is None:
            raise RuntimeError(
                "OnlineBRUNCHAssigner has no missing_tracker; "
                "construct it with one to use missing-data completion."
            )
        return self.missing_tracker.declare_missing(key_kind, key_value, start_ts)

    def declare_recovered(self, key_kind: str, key_value, end_ts: float) -> int:
        """Close all open intervals matching (key_kind, key_value). Returns count closed."""
        if self.missing_tracker is None:
            raise RuntimeError("OnlineBRUNCHAssigner has no missing_tracker")
        return self.missing_tracker.declare_recovered(key_kind, key_value, end_ts)

    def _sample_virtual_events_until(self, target_ts: float) -> None:
        """For every MissingInterval that still has un-sampled time before
        ``target_ts``, sample background-rate virtual events and feed them
        into the assigner in **global** chronological order.

        Two correctness properties matter here:

        1. Sampling proceeds even for already-closed intervals as long as
           ``last_sample_ts < end_ts``. This covers the case where recovery
           is declared before the assigner has caught up with the tail of the
           outage; otherwise the tail would be silently dropped.
        2. All virtual events from all (interval, type_id) groups are sorted
           by timestamp **before** any of them are injected. ``self.recent``
           must stay monotonically non-decreasing for the early-break in
           :meth:`_candidate_scores` (``dt_sec > active_window_sec``) to be a
           valid pruning condition.

        Background-only rate (λ = μ_d) is the lightweight likelihood-weighted
        variant of Shelton 2018 §"MCMC Sampler"; a history-aware rate
        (λ = μ_d + Σ kernel) is a strict superset and can replace this if
        downstream demand justifies the extra cost.
        """
        if not self.missing_tracker:
            return
        active_intervals = self.missing_tracker.intervals_needing_sampling(target_ts)
        if not active_intervals:
            # Nothing to sample, but periodically compact stale entries so the
            # tracker doesn't grow unbounded under a long outage history.
            self.missing_tracker.compact()
            return
        cutoff = float(target_ts) - self.active_window_sec
        scale = float(self.config.time_scale_sec) if self.config.time_scale_sec > 0 else 1.0
        budget = int(self.max_virtual_per_call)

        # Stage 1: enumerate all candidate virtual events into a flat list.
        # Stage 2 will sort that list by timestamp and inject in order.
        pending: list[dict] = []
        for interval in active_intervals:
            upper = min(float(target_ts), interval.effective_end())
            last = float(interval.last_sample_ts if interval.last_sample_ts is not None else interval.start_ts)
            # Don't bother sampling earlier than the assigner's recent window —
            # those events would be expired before any real event could see
            # them as a parent candidate anyway.
            start = max(last, cutoff)
            # Advance bookkeeping even when we skip; otherwise long-past tails
            # of recovered intervals would never reach is_exhausted() and the
            # tracker.compact() above would never drop them.
            interval.last_sample_ts = max(last, upper)
            if start >= upper:
                continue
            dt_scaled = (upper - start) / scale
            for type_id in self.missing_tracker.type_ids_for(interval):
                mu = float(self.params.mu[type_id])
                if mu <= 0.0:
                    continue
                expected = mu * dt_scaled
                if expected <= 0.0:
                    continue
                count = int(self.rng.poisson(expected))
                if count <= 0:
                    continue
                count = min(count, self.max_virtual_per_dim)
                times = self.rng.uniform(start, upper, size=count)
                type_label = self.vocabs.type_vocab.labels[type_id]
                source_name, _, type_part = type_label.partition(" | ")
                interval_key = f"{interval.key_kind}:{interval.key_value}"
                for vt in times:
                    pending.append(
                        {
                            "ts": float(vt),
                            "type_id": int(type_id),
                            "type_label": type_label,
                            "alarm_source": source_name,
                            "alarm_type": type_part or "",
                            "interval_key": interval_key,
                        }
                    )

        # Stage 2: enforce the per-call budget, then sort by ts and inject.
        #
        # When |pending| exceeds the budget we **uniformly subsample** rather
        # than truncate by timestamp. Truncating by ts would always drop the
        # tail — exactly the virtuals nearest to the current real event,
        # which are the most likely candidate parents under the exp kernel.
        # That biases the chain toward "no parent found" exactly when the
        # outage is busiest. Uniform thinning of a Poisson process is itself a
        # Poisson process with rate λ · (kept/total), so this is a valid
        # load-shedding policy: it lowers the effective μ uniformly across
        # the whole sampled window instead of starving the tail.
        if pending:
            if len(pending) > budget:
                kept_idx = self.rng.choice(len(pending), size=budget, replace=False)
                pending = [pending[int(i)] for i in kept_idx]
            pending.sort(key=lambda spec: spec["ts"])
            for spec in pending:
                self._inject_virtual_event(**spec)

        # Evict closed-and-fully-sampled intervals so subsequent ticks don't
        # scan them. Runs every call but is cheap: it's a single list filter.
        self.missing_tracker.compact()

    def _inject_virtual_event(
        self,
        *,
        ts: float,
        type_id: int,
        type_label: str,
        alarm_source: str,
        alarm_type: str,
        interval_key: str,
    ) -> None:
        index = self._next_index
        self._next_index += 1
        synthetic_event_id = f"virtual-{index:08d}"
        synthetic_alarm_dict = {
            # `event_id` is one of the keys aggregator._event_id() looks for,
            # so summarize_alarm_event() in to_group()/decision output will
            # surface our virtual id rather than falling back to the generic
            # alarm-NNN scheme.
            "alarm": {"__virtual__": True, "missing_interval": interval_key, "event_id": synthetic_event_id},
            "ts": ts,
            "site_id": "",
            "alarm_source": alarm_source,
            "alarm_title": "",
            "alarm_type": alarm_type,
        }
        online_event = OnlineEvent(
            index=index,
            event=synthetic_alarm_dict,
            type_id=int(type_id),
            type_label=type_label,
            event_id=synthetic_event_id,
            virtual=True,
            confidence=self.virtual_confidence,
            virtual_source=interval_key,
        )
        candidates = self._candidate_scores(online_event)
        chosen_parent, _, _ = self._choose_candidate(candidates)
        if chosen_parent is None:
            cascade = self._create_cascade(online_event)
        else:
            cascade = self.cascades[chosen_parent.cascade_id]
            cascade.add(online_event, parent=chosen_parent)
            self.branching_edge_count += 1
            if chosen_parent.virtual:
                # virtual event picking a virtual parent — purely synthetic chain.
                self.virtual_picked_as_parent_count += 1
        self._push_recent_ordered(online_event)
        self.virtual_event_count += 1


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _make_alarm_event(alarm, *, site_id, alarm_title, event_time_str, is_clear=False):
    event_alarm = dict(alarm)
    event_alarm["告警首次发生时间"] = event_time_str
    if is_clear:
        event_alarm["清除告警"] = "是"
    return {
        "alarm": event_alarm,
        "site_id": str(site_id or ""),
        "alarm_source": str(alarm.get("告警源", "") or "").strip(),
        "alarm_title": str(alarm_title or ""),
        "ts": parse_datetime_text(event_time_str, "告警时间").timestamp(),
    }


def _raw_alarm_to_events(
    alarm,
    *,
    valid_sites,
    ne_to_site,
    allowed_alarm_sources=None,
    start_ts=None,
    end_ts=None,
    include_clear=False,
    clear_delay_sec=0.0,
):
    alarm_title = str(alarm.get("告警标题", "") or "")
    if alarm_title not in CRITICAL_ALARMS:
        return []
    alarm_source = str(alarm.get("告警源", "") or "").strip()
    if allowed_alarm_sources is not None and alarm_source not in allowed_alarm_sources:
        return []

    site_id = str(alarm.get("站点ID", "") or "").strip()
    if not site_id or site_id not in valid_sites:
        site_id = ne_to_site.get(alarm_source, "")
    if not site_id or site_id not in valid_sites:
        return []

    first_occurrence_str = str(alarm.get("告警首次发生时间", "") or "").strip()
    first_ts = parse_datetime_text(first_occurrence_str, "告警首次发生时间").timestamp()
    if start_ts is not None and first_ts < start_ts:
        return []
    if end_ts is not None and first_ts > end_ts:
        return []

    events = [
        _make_alarm_event(
            alarm,
            site_id=site_id,
            alarm_title=alarm_title,
            event_time_str=first_occurrence_str,
            is_clear=False,
        )
    ]
    clear_time_str = str(alarm.get("告警清除时间", "") or "").strip()
    if include_clear and clear_time_str:
        effective_clear_time_str = apply_clear_delay(
            first_occurrence_str,
            clear_time_str,
            clear_delay_sec,
        )
        events.append(
            _make_alarm_event(
                alarm,
                site_id=site_id,
                alarm_title=alarm_title,
                event_time_str=effective_clear_time_str,
                is_clear=True,
            )
        )
    return events


def _iter_stream_events(
    alarm_input,
    *,
    topo_path,
    ne_graph_path,
    regions,
    start_time=None,
    end_time=None,
    include_clear=False,
    clear_delay_sec=0.0,
    show_progress=False,
):
    selected_regions = parse_regions(regions)
    start_ts = parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time cannot be later than end_time")

    if is_sorted_alarm_cache_file(alarm_input):
        metadata, events = load_sorted_alarm_cache(alarm_input, show_progress=show_progress)
        allowed_alarm_sources = None
        if selected_regions:
            allowed_alarm_sources = allowed_devices_for_regions(load_ne_graph(ne_graph_path), selected_regions)
        for event in events:
            if allowed_alarm_sources is not None and event.get("alarm_source", "") not in allowed_alarm_sources:
                continue
            if start_ts is not None and float(event.get("ts", 0.0)) < start_ts:
                continue
            if end_ts is not None and float(event.get("ts", 0.0)) > end_ts:
                continue
            if not include_clear and is_clear_alarm(event.get("alarm", {})):
                continue
            yield event
        return

    valid_sites, ne_to_site, ne_graph_data = _load_valid_sites_and_ne_mapping(topo_path, ne_graph_path)
    allowed_alarm_sources = (
        allowed_devices_for_regions(ne_graph_data, selected_regions) if selected_regions else None
    )
    for alarm in stream_alarm_inputs(alarm_input, show_progress=show_progress):
        for event in _raw_alarm_to_events(
            alarm,
            valid_sites=valid_sites,
            ne_to_site=ne_to_site,
            allowed_alarm_sources=allowed_alarm_sources,
            start_ts=start_ts,
            end_ts=end_ts,
            include_clear=include_clear,
            clear_delay_sec=clear_delay_sec,
        ):
            yield event


def _process_event(assigner, decision_stream, event):
    decision = assigner.assign(event)
    decision_stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    return decision


def _resolve_groups_output(args):
    """Decide whether to write a cumulative .groups.json snapshot.

    Returns the destination path or ``""`` to disable. An empty result is the
    cue for main() to skip the snapshot collector entirely — important because
    accumulating every closed group is O(stream) memory and would re-introduce
    the OOM risk we just fixed in live-stream mode.

    Resolution rules (in order):
    1. ``--no-groups-output`` always wins → disabled.
    2. ``--groups-output PATH`` → that path.
    3. ``--visual-output`` set without an explicit groups path → disabled
       (assumption: the visual file is the user's reporting sink).
    4. ``--preserve-input-order`` (live stream mode) → disabled by default;
       infinite-stream daemons should not accumulate forever, and operators
       who want the snapshot can opt in via explicit ``--groups-output``.
    5. Offline replay default → ``{output}.groups.json``.
    """
    if getattr(args, "no_groups_output", False):
        return ""
    if args.groups_output:
        return args.groups_output
    if args.visual_output:
        return ""
    if getattr(args, "preserve_input_order", False):
        return ""
    return str(Path(args.output).with_suffix(".groups.json"))


def _count_decision(counts, decision):
    status = decision.get("status", "")
    counts[status] = counts.get(status, 0) + 1


def _status_count(counts, status):
    return counts.get(status, 0)


def _emit_closed_visual_groups(visual_output, assigner, now_ts, close_after_sec, min_group_events):
    """Run close_inactive and forward any closed groups to visual_output.

    Returns ``(visual_emitted_count, closed_groups)`` so the caller can also
    accumulate the closed groups into the final snapshot collector — without
    that, ``--groups-output`` would receive an empty array because
    close_inactive deletes cascades from memory as it returns them.
    """
    closed_groups = assigner.close_inactive(
        now_ts,
        close_after_sec,
        min_group_events=min_group_events,
    )
    visual_emitted = 0
    if visual_output is not None and closed_groups:
        visual_emitted = visual_output.emit_groups(closed_groups, finalization_reason="closed")
    return visual_emitted, closed_groups


def _emit_remaining_visual_groups(visual_output, assigner, min_group_events):
    groups = assigner.close_remaining(min_group_events=min_group_events)
    if visual_output is None:
        return 0, groups
    return visual_output.emit_groups(groups, finalization_reason="stream_end"), groups


def _progress_extra_text(assigner, counts, reorder_buffer, visual_output=None):
    visual_count = visual_output.emitted_count if visual_output is not None else 0
    return (
        f"聚类 {_status_count(counts, 'clustered')}，"
        f"跳过 {_status_count(counts, 'skipped')}，"
        f"cascade {len(assigner.cascades)}，"
        f"closed {assigner.closed_cascade_count}，"
        f"visual {visual_count}，"
        f"乱序缓冲 {reorder_buffer.pending_count}"
    )


class _StreamProcessProgress:
    """Throttle processing progress for sorted files or a live stream."""

    def __init__(self, total=0, interval_sec=0.2):
        self.bar = ProgressBar(total, "处理告警流")
        self.interval_sec = interval_sec
        self.last_refresh = 0.0

    def refresh(self, processed_count, assigner, counts, reorder_buffer, visual_output=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_refresh < self.interval_sec:
            return
        self.bar.set(processed_count)
        self.bar.set_extra_text(
            _progress_extra_text(assigner, counts, reorder_buffer, visual_output),
            force=force,
        )
        self.last_refresh = now

    def close(self):
        self.bar.close()


def _print_run_configuration(args, artifact, config, groups_output):
    active_window = (
        args.active_window_sec
        if args.active_window_sec is not None
        else config.history_window_sec
    )
    print("正在初始化 alarm-flow BRUNCH 在线推理器...")
    print(
        "模型配置: "
        f"types={len(artifact.vocabs.type_vocab)}, "
        f"active_edges={len(artifact.params.active_edges(include_self=True)[0])}, "
        f"parent_selection={config.parent_selection}, "
        f"type_fields={','.join(config.type_fields)}"
    )
    print(
        "候选窗口: "
        f"active={active_window:g}s, "
        f"max_history_events={config.max_history_events}, "
        f"close={args.close_after_sec:g}s, "
        f"reorder_lag={args.reorder_lag_sec:g}s"
    )
    print(
        "输入顺序: "
        + (
            "保留源顺序，按实时流入口处理"
            if args.preserve_input_order
            else "离线文件默认先过滤并按事件时间排序后推理"
        )
    )
    print("region 过滤: " + (",".join(sorted(config.regions)) if config.regions else "未启用"))
    print(f"决策输出: {args.output}")
    if groups_output:
        print(f"cascade 快照输出: {groups_output}")
    if args.visual_output:
        print("可视化输出: cascade 关闭时追加写入，输入结束时补写仍未关闭的 cascade")
        print(f"可视化文件: {args.visual_output}")


def _apply_input_mode_defaults(args):
    if args.reorder_lag_sec is None:
        args.reorder_lag_sec = (
            _LIVE_REORDER_LAG_SEC
            if args.preserve_input_order
            else _OFFLINE_REORDER_LAG_SEC
        )
    return args


def _load_sorted_events(args, config):
    print("正在加载告警并应用时间/region/拓扑过滤...")
    start_time = time.time()
    events = list(
        _iter_stream_events(
            args.alarms,
            topo_path=args.topo,
            ne_graph_path=args.ne_graph,
            regions=config.regions,
            start_time=args.start_time or None,
            end_time=args.end_time or None,
            include_clear=config.include_clear,
            clear_delay_sec=args.clear_delay_sec,
            show_progress=True,
        )
    )
    load_elapsed = time.time() - start_time
    print(f"过滤后保留 {len(events)} 条告警事件，加载耗时 {load_elapsed:.2f} 秒")
    print("正在按事件时间排序告警...")
    sort_start = time.time()
    events.sort(key=lambda event: float(event.get("ts", 0.0)))
    sort_elapsed = time.time() - sort_start
    print(f"已准备 {len(events)} 条按事件时间排序的告警事件，排序耗时 {sort_elapsed:.2f} 秒")
    return events


def _iter_live_events(args, config):
    return _iter_stream_events(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        regions=config.regions,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        include_clear=config.include_clear,
        clear_delay_sec=args.clear_delay_sec,
        show_progress=args.show_progress,
    )


def _process_ready_event(
    assigner,
    decision_stream,
    counts,
    visual_output,
    ready_event,
    *,
    close_after_sec,
    min_group_events,
    closed_groups_sink=None,
):
    decision = _process_event(assigner, decision_stream, ready_event)
    _count_decision(counts, decision)
    _, closed_groups = _emit_closed_visual_groups(
        visual_output,
        assigner,
        decision["ts"],
        close_after_sec,
        min_group_events,
    )
    if closed_groups_sink is not None and closed_groups:
        closed_groups_sink.extend(closed_groups)
    return decision


def _process_event_iterable(
    events,
    *,
    assigner,
    decision_stream,
    counts,
    visual_output,
    reorder_buffer,
    close_after_sec,
    min_group_events,
    total=0,
    closed_groups_sink=None,
):
    """When ``closed_groups_sink`` is a list, every group closed during the
    run is appended to it so the caller can write a full-stream snapshot.
    When ``None`` (i.e. the caller wants O(1) memory and no .groups.json
    file) closed groups are discarded after being optionally forwarded to
    the visualization output.
    """
    processed = 0
    progress = _StreamProcessProgress(total=total)
    progress.refresh(processed, assigner, counts, reorder_buffer, visual_output, force=True)
    try:
        for event in events:
            reorder_buffer.push(event)
            for ready_event in reorder_buffer.ready():
                _process_ready_event(
                    assigner,
                    decision_stream,
                    counts,
                    visual_output,
                    ready_event,
                    close_after_sec=close_after_sec,
                    min_group_events=min_group_events,
                    closed_groups_sink=closed_groups_sink,
                )
            processed += 1
            progress.refresh(processed, assigner, counts, reorder_buffer, visual_output)
        print("\n数据流读取完毕，正在清空乱序缓冲并输出剩余决策...")
        for ready_event in reorder_buffer.flush():
            _process_ready_event(
                assigner,
                decision_stream,
                counts,
                visual_output,
                ready_event,
                close_after_sec=close_after_sec,
                min_group_events=min_group_events,
                closed_groups_sink=closed_groups_sink,
            )
            progress.refresh(processed, assigner, counts, reorder_buffer, visual_output)
    finally:
        progress.refresh(processed, assigner, counts, reorder_buffer, visual_output, force=True)
        progress.close()
    return processed


def main():
    parser = ArgumentParser(
        description=(
            "Assign alarms to BRUNCH cascades online using a trained artifact and a small reorder lag."
        )
    )
    parser.add_argument("model", help="Model artifact generated by train_alarm_brunch.py.")
    parser.add_argument("alarms", help="Raw alarm input, or a prepared sorted-alarm cache for replay.")
    parser.add_argument("output", help="Online decision JSONL output.")
    parser.add_argument(
        "--groups-output",
        default="",
        help=(
            "Final cascade snapshot JSON. Default behaviour: replace output suffix "
            "with .groups.json for offline replay; disabled (no file written, no "
            "in-memory accumulation) in live-stream mode (--preserve-input-order) "
            "or when --visual-output is set. Use --no-groups-output to disable "
            "explicitly regardless of mode."
        ),
    )
    parser.add_argument(
        "--no-groups-output",
        action="store_true",
        help=(
            "Disable the .groups.json snapshot entirely. Forces O(1) cumulative-"
            "group memory — required for long-running stream daemons that would "
            "otherwise accumulate every closed cascade in memory until OOM."
        ),
    )
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
    parser.add_argument("--include-clear", action="store_true", help="Also score synthetic clear events.")
    parser.add_argument(
        "--reorder-lag-sec",
        type=float,
        default=None,
        help="Event-time reorder buffer lag. Default: 0 after offline sorting, 300 with --preserve-input-order.",
    )
    parser.add_argument(
        "--active-window-sec",
        type=float,
        default=None,
        help="Recent parent-candidate window. Default: artifact history_window_sec.",
    )
    parser.add_argument(
        "--close-after-sec",
        type=float,
        default=7200.0,
        help="Close and emit quiet cascades after this many seconds. Default: 7200.",
    )
    parser.add_argument(
        "--min-group-events",
        type=int,
        default=1,
        help="Drop final groups smaller than this for --groups-output/--visual-output.",
    )
    parser.add_argument(
        "--parent-selection",
        choices=("sample", "argmax"),
        default=None,
        help="Override artifact parent selection. argmax is deterministic and usually best for serving.",
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help=(
            "Override artifact regions. Repeat this option or pass comma-separated values; "
            "omit to reuse the model's regions."
        ),
    )
    parser.add_argument(
        "--missing-intervals",
        default="",
        help=(
            "Optional path to a JSON list of missing-data intervals. Each record "
            "has one of alarm_source/alarm_type/type_label/type_id and start/end "
            "(Unix timestamp or ISO datetime; end=null for still-open). During "
            "these windows the assigner injects Shelton-style virtual events so "
            "they can serve as candidate parents for downstream real alarms."
        ),
    )
    parser.add_argument(
        "--virtual-confidence",
        type=float,
        default=0.5,
        help="Score multiplier (in (0, 1]) for virtual parent candidates. Default: 0.5.",
    )
    parser.add_argument(
        "--max-virtual-per-call",
        type=int,
        default=50,
        help="Cap on virtual events injected per real-event tick. Default: 50.",
    )
    parser.add_argument(
        "--max-virtual-per-dim",
        type=int,
        default=100,
        help="Cap on virtual events sampled per dimension per tick. Default: 100.",
    )
    parser.add_argument(
        "--close-check-interval-sec",
        type=float,
        default=1.0,
        help=(
            "Minimum event-time interval between full close_inactive scans. "
            "Under high event rates the close-check is called per tick; "
            "this throttle bounds the K-cascade scan cost. Set to 0 to "
            "scan on every tick. Default: 1.0 (event seconds)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample mode.")
    parser.add_argument("--show-progress", action="store_true", help="Show input read progress.")
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="Skip the default offline event-time sort and consume source order as a live stream.",
    )
    args = _apply_input_mode_defaults(parser.parse_args())
    groups_output = _resolve_groups_output(args)

    start_time = time.time()
    print("正在加载 BRUNCH 模型 artifact...")
    artifact = load_alarm_brunch_artifact(args.model)
    regions = parse_regions(args.regions) if args.regions is not None else artifact.config.regions
    parent_selection = args.parent_selection or artifact.config.parent_selection
    config = replace(
        artifact.config,
        regions=regions,
        include_clear=bool(args.include_clear or artifact.config.include_clear),
        parent_selection=parent_selection,
    )
    _print_run_configuration(args, artifact, config, groups_output)
    missing_tracker = None
    if args.missing_intervals:
        missing_tracker = MissingIntervalTracker(artifact.vocabs)
        loaded = load_missing_from_json(args.missing_intervals)
        for interval in loaded:
            missing_tracker.add(interval)
        print(
            f"已加载 missing intervals: 共 {len(loaded)} 条 "
            f"(open={missing_tracker.stats()['open_count']}, "
            f"closed={missing_tracker.stats()['closed_count']})，"
            f"virtual_confidence={args.virtual_confidence:g}, "
            f"max_per_call={args.max_virtual_per_call}, "
            f"max_per_dim={args.max_virtual_per_dim}"
        )
    assigner = OnlineBRUNCHAssigner(
        artifact,
        config=config,
        active_window_sec=args.active_window_sec,
        parent_selection=parent_selection,
        seed=args.seed,
        missing_tracker=missing_tracker,
        virtual_confidence=args.virtual_confidence,
        max_virtual_per_call=args.max_virtual_per_call,
        max_virtual_per_dim=args.max_virtual_per_dim,
        close_check_min_interval_sec=args.close_check_interval_sec,
    )
    reorder_buffer = ReorderBuffer(args.reorder_lag_sec)
    visual_output = None
    if args.visual_output:
        visual_output = AlarmBRUNCHVisualOutputSession.from_files(
            args.visual_output,
            args.ne_graph,
            args.site_graph,
            ne_scope=args.visual_ne_scope,
        )
        visual_output.reset_output_file()
    counts = {}

    # Closed-group accumulator. Only allocated when the user asked for a
    # .groups.json snapshot — otherwise we keep O(1) memory by discarding
    # groups as they close.
    all_closed_groups: list | None = [] if groups_output else None

    try:
        sorted_events = None
        if not args.preserve_input_order:
            sorted_events = _load_sorted_events(args, config)

        print("正在处理告警流并写出在线 cascade 决策...")
        with open(args.output, "w", encoding="utf-8") as decision_stream:
            input_events = (
                _iter_live_events(args, config)
                if sorted_events is None
                else sorted_events
            )
            processed_count = _process_event_iterable(
                input_events,
                assigner=assigner,
                decision_stream=decision_stream,
                counts=counts,
                visual_output=visual_output,
                reorder_buffer=reorder_buffer,
                close_after_sec=args.close_after_sec,
                min_group_events=args.min_group_events,
                total=len(sorted_events) if sorted_events is not None else 0,
                closed_groups_sink=all_closed_groups,
            )
            print("正在补写仍未关闭的 cascade...")
            remaining_emitted, remaining_groups = _emit_remaining_visual_groups(
                visual_output,
                assigner,
                args.min_group_events,
            )
            if all_closed_groups is not None and remaining_groups:
                all_closed_groups.extend(remaining_groups)
    finally:
        if visual_output is not None:
            visual_output.close()

    metadata = assigner.metadata()
    metadata["input"] = os.path.abspath(args.alarms)
    metadata["model"] = os.path.abspath(args.model)
    metadata["decision_output"] = os.path.abspath(args.output)
    # group_count == reportable count (matches snapshot.cascade_count). Use
    # metadata["total_cascade_count"] for the unfiltered cumulative figure
    # (includes pure-virtual cascades that never got a real event attached).
    metadata["group_count"] = int(assigner.reportable_cascade_count)
    metadata["reorder_lag_sec"] = float(args.reorder_lag_sec)
    metadata["regions"] = sorted(config.regions)

    if groups_output:
        print("正在写 cascade 快照 JSON...")
        snapshot_groups = all_closed_groups or []
        _write_json(
            groups_output,
            {
                "decision_counts": counts,
                "cascade_count": len(snapshot_groups),
                "cascades": snapshot_groups,
            },
        )
    visual_count = visual_output.emitted_count if visual_output is not None else 0
    elapsed = time.time() - start_time

    print(
        f"online BRUNCH decisions written to: {args.output}; "
        f"read={processed_count}, "
        f"events={metadata['modeled_event_count']}, "
        f"skipped={metadata['skipped_event_count']}, "
        f"groups={metadata['group_count']}, "
        f"edges={metadata['branching_edge_count']}, "
        f"elapsed={elapsed:.2f}s"
    )
    if groups_output:
        print(f"final groups written to: {groups_output}")
    if args.visual_output:
        print(f"visual groups written to: {args.visual_output}; groups={visual_count}")


if __name__ == "__main__":
    main()
