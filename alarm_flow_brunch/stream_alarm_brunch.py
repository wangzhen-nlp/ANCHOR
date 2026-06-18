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
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_brunch.aggregator import load_alarm_brunch_artifact, summarize_alarm_event
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
from fault_grouping.alarm_events.identity import input_occurrence_uuid, new_occurrence_uuid
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
BRUNCH_RULE = "alarm_flow_brunch"
BRUNCH_VIRTUAL_RULE = "alarm_flow_brunch_virtual_event"


_VIRTUAL_ALARM_TITLE_BY_TYPE = {
    "link": "Link Down",
    "power": "Power Supply",
    "offline": "NE is Disconnected",
}


def _warn(message: str) -> None:
    """Emit a structured warning to stderr.

    Goes to stderr (not stdout) so it survives stdout redirection / piping,
    and uses a ``[WARN]`` prefix so log-aggregators that scan severity tokens
    can route or filter it correctly.
    """
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def _clean_type_label_part(value) -> str:
    value = str(value or "").strip()
    return "" if value == "<empty>" else value


def _type_label_field_values(type_label, type_fields) -> dict:
    fields = tuple(type_fields or ())
    if not fields:
        return {}
    label = str(type_label or "")
    if len(fields) == 1:
        return {fields[0]: _clean_type_label_part(label)}
    parts = label.split(" | ", maxsplit=len(fields) - 1)
    if len(parts) < len(fields):
        parts.extend([""] * (len(fields) - len(parts)))
    return {
        field: _clean_type_label_part(part)
        for field, part in zip(fields, parts)
    }


def _virtual_alarm_title(alarm_type: str, alarm_title: str = "") -> str:
    alarm_title = str(alarm_title or "").strip()
    if alarm_title:
        return alarm_title
    return _VIRTUAL_ALARM_TITLE_BY_TYPE.get(str(alarm_type or "").strip(), "")


def _load_ne_to_site_map(ne_graph_path) -> dict[str, str]:
    try:
        ne_graph = load_ne_graph(ne_graph_path)
    except Exception as exc:
        _warn(f"无法为虚拟事件加载 NE 站点映射: {ne_graph_path} ({exc})")
        return {}
    ne_to_site: dict[str, str] = {}
    for ne_id, record in (ne_graph or {}).items():
        if not isinstance(record, dict):
            continue
        site_id = str(
            record.get("site_id")
            or record.get("siteId")
            or record.get("site")
            or record.get("site_name")
            or record.get("siteName")
            or ""
        ).strip()
        if site_id:
            ne_to_site[str(ne_id)] = site_id
    return ne_to_site


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
    # `latent` distinguishes "experimental latent-missing mode" virtuals from
    # known-interval virtuals. Both have virtual=True; latent additionally
    # carries this flag so audit counters and downstream UI can separate the
    # low-confidence latent path from known missing-interval imputations.
    latent: bool = False
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
                    "source_occurrence_uuid": parent.event["occurrence_uuid"],
                    "target_occurrence_uuid": event.event["occurrence_uuid"],
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
                summary["latent"] = bool(item.latent)
                summary["confidence"] = float(item.confidence)
                summary["virtual_source"] = item.virtual_source
            else:
                summary["virtual"] = False
                summary["latent"] = False
                summary["confidence"] = 1.0
            summaries.append(summary)
        timestamps = [summary["ts"] for summary in summaries]
        merged_rules = []
        if self.virtual_event_count() > 0:
            merged_rules.extend([BRUNCH_RULE, BRUNCH_VIRTUAL_RULE])
        return {
            "group_id": self.cascade_id,
            "cascade_id": self.cascade_id,
            "rule": BRUNCH_RULE,
            "merged_rules": merged_rules,
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
        ne_to_site=None,
        virtual_confidence: float = 0.5,
        max_virtual_per_call: int = 50,
        max_virtual_per_dim: int = 100,
        virtual_drop_warning_ratio: float = 0.25,
        close_check_min_interval_sec: float = 1.0,
        # ---- experimental latent-missing mode -----------------------------
        # When enabled, every dim is treated as having an implicit "data
        # could be missing at any moment" interval. Sampling rate is scaled
        # down by latent_rate_multiplier and dampened by latent_confidence so
        # the chain stays close to the observed-data interpretation; the
        # active-virtual cap and audit counters are the safety net.
        latent_missing_mode: bool = False,
        latent_rate_multiplier: float = 0.05,
        latent_confidence: float = 0.05,
        latent_max_virtual_per_call: int = 10,
        latent_max_active_virtual: int = 200,
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
        if virtual_drop_warning_ratio < 0.0 or virtual_drop_warning_ratio > 1.0:
            raise ValueError("virtual_drop_warning_ratio must be in [0, 1]")
        if close_check_min_interval_sec < 0:
            raise ValueError("close_check_min_interval_sec must be non-negative")
        if not 0.0 <= latent_rate_multiplier <= 1.0:
            raise ValueError("latent_rate_multiplier must be in [0, 1]")
        if not 0.0 < latent_confidence <= 1.0:
            raise ValueError("latent_confidence must be in (0, 1]")
        if latent_max_virtual_per_call < 0 or latent_max_active_virtual < 0:
            raise ValueError("latent_max_virtual_per_call / latent_max_active_virtual must be non-negative")
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
        self.ne_to_site = dict(ne_to_site or {})
        self.virtual_confidence = float(virtual_confidence)
        self.max_virtual_per_call = int(max_virtual_per_call)
        self.max_virtual_per_dim = int(max_virtual_per_dim)
        self.virtual_drop_warning_ratio = float(virtual_drop_warning_ratio)
        self.virtual_event_count = 0
        self.virtual_candidate_event_count = 0
        self.virtual_dropped_by_dim_cap_count = 0
        self.virtual_dropped_by_call_budget_count = 0
        self.virtual_drop_warning_count = 0
        self.virtual_unresolved_interval_tick_count = 0
        self.virtual_picked_as_parent_count = 0
        # ---- latent missing mode state + audit counters --------------------
        self.latent_missing_mode = bool(latent_missing_mode)
        self.latent_rate_multiplier = float(latent_rate_multiplier)
        self.latent_confidence = float(latent_confidence)
        self.latent_max_virtual_per_call = int(latent_max_virtual_per_call)
        self.latent_max_active_virtual = int(latent_max_active_virtual)
        # latent_virtual_generated:        total latent virtuals successfully injected
        # latent_virtual_picked_as_parent: latent virtuals picked as a parent
        # latent_virtual_dropped_by_cap:   latent virtuals pruned by per-call or active cap
        # real_events_with_latent_parent:  real events whose chosen parent was a latent virtual
        self.latent_virtual_generated = 0
        self.latent_virtual_picked_as_parent = 0
        self.latent_virtual_dropped_by_cap = 0
        self.real_events_with_latent_parent = 0
        self._latent_type_ids = np.asarray([], dtype=np.int64)
        self._latent_mu_values = np.asarray([], dtype=np.float64)
        self._latent_mu_cumsum = np.asarray([], dtype=np.float64)
        self._latent_mu_total = 0.0
        self._latent_type_metadata: dict[int, dict] = {}
        if self.latent_missing_mode and self.latent_rate_multiplier > 0.0:
            # Snapshot params.mu and the outgoing-edge structure for the
            # latent sampler. This can be O(active_edges) / O(M^2) depending
            # on parameter storage, so only pay it when latent mode is
            # actually enabled. We filter to dims that BOTH have μ > 0 AND
            # are an active source for some target (∃ target_id with
            # α[target_id, dim] > 0). A dim with no outgoing edges can never
            # be selected as a parent by _candidate_scores.
            #
            # IMPORTANT: these caches snapshot params at init time. If anyone
            # ever hot-reloads artifact.params at runtime (e.g.
            # --watch-artifact), the caches MUST be rebuilt — otherwise
            # latent sampling uses stale rates and stale edge structure.
            active_source_dims: set[int] = set()
            for target_id in range(self.params.M):
                for src in self.params.active_sources_for_target(target_id, include_self=True):
                    active_source_dims.add(int(src))
            latent_type_ids = []
            latent_mu_values = []
            for type_id in range(self.params.M):
                if type_id not in active_source_dims:
                    continue
                mu = float(self.params.mu[type_id])
                if mu > 0.0:
                    latent_type_ids.append(int(type_id))
                    latent_mu_values.append(mu)
            self._latent_type_ids = np.asarray(latent_type_ids, dtype=np.int64)
            self._latent_mu_values = np.asarray(latent_mu_values, dtype=np.float64)
            self._latent_mu_cumsum = np.cumsum(self._latent_mu_values)
            self._latent_mu_total = (
                float(self._latent_mu_cumsum[-1]) if len(self._latent_mu_cumsum) else 0.0
            )
            # Pre-compute the per-type fields a virtual event needs in
            # to_group() output. All inputs are immutable post-init, so
            # caching avoids string partitioning and topology lookups once
            # per generated virtual.
            for type_id in latent_type_ids:
                type_label = self.vocabs.type_vocab.labels[int(type_id)]
                field_values = _type_label_field_values(type_label, self.config.type_fields)
                source_name = field_values.get("alarm_source", "")
                type_part = field_values.get("alarm_type", "")
                self._latent_type_metadata[int(type_id)] = {
                    "type_label": type_label,
                    "alarm_source": source_name,
                    "alarm_type": type_part or "",
                    "alarm_title": _virtual_alarm_title(type_part, field_values.get("alarm_title", "")),
                    "site_id": field_values.get("site_id", "") or self.ne_to_site.get(source_name, ""),
                    "interval_key": f"latent:{type_label}",
                }
        # Bookkeeping for latent mode: sample-time cursor (max ts up to which
        # latent virtuals have already been considered).
        self._last_latent_sample_ts = float("-inf")
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
        # real event is itself a clear/unknown that we wouldn't process. The
        # latent-missing mode (if enabled) also gets a sampling tick here so
        # its low-rate "any-time-might-be-missing" virtuals are interleaved
        # in event-time order with the known-interval virtuals.
        if self._should_sample_virtuals(target_ts):
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
        parent_is_latent = False
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
                if chosen_parent.latent:
                    parent_is_latent = True
                    self.latent_virtual_picked_as_parent += 1
                    self.real_events_with_latent_parent += 1

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
                "parent_latent": parent_is_latent,
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
                "virtual_candidate_event_count": int(self.virtual_candidate_event_count),
                "virtual_dropped_by_dim_cap_count": int(self.virtual_dropped_by_dim_cap_count),
                "virtual_dropped_by_call_budget_count": int(self.virtual_dropped_by_call_budget_count),
                "virtual_drop_warning_count": int(self.virtual_drop_warning_count),
                "virtual_drop_warning_ratio": float(self.virtual_drop_warning_ratio),
                "virtual_unresolved_interval_tick_count": int(self.virtual_unresolved_interval_tick_count),
                "virtual_picked_as_parent_count": int(self.virtual_picked_as_parent_count),
                "virtual_confidence": float(self.virtual_confidence),
                "max_virtual_per_call": int(self.max_virtual_per_call),
                "max_virtual_per_dim": int(self.max_virtual_per_dim),
            }
        if self.latent_missing_mode:
            # latent_parent_rate is the headline safety gauge: of all real
            # modeled events, what fraction chose a latent virtual parent.
            latent_parent_rate = (
                float(self.real_events_with_latent_parent) / float(self.modeled_count)
                if self.modeled_count > 0
                else 0.0
            )
            meta["latent_missing"] = {
                "enabled": True,
                "rate_multiplier": float(self.latent_rate_multiplier),
                "confidence": float(self.latent_confidence),
                "max_virtual_per_call": int(self.latent_max_virtual_per_call),
                "max_active_virtual": int(self.latent_max_active_virtual),
                "virtual_events_generated": int(self.latent_virtual_generated),
                "virtual_events_picked_as_parent": int(self.latent_virtual_picked_as_parent),
                "virtual_events_dropped_by_cap": int(self.latent_virtual_dropped_by_cap),
                "real_events_with_latent_parent": int(self.real_events_with_latent_parent),
                "latent_parent_rate": float(latent_parent_rate),
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

    def _candidate_scores(self, target: OnlineEvent, *, exclude_latent_parents: bool = False):
        """Score candidate parents for `target` from self.recent.

        Parameters
        ----------
        exclude_latent_parents
            When True, skip any latent virtual events in self.recent. Used
            when `target` is itself a virtual event being injected: latent
            virtuals are purely speculative, so "virtual triggered by latent"
            chains amount to compounded speculation and just inflate
            audit/edge counters without affecting reportable output. By
            blocking them at scoring time we keep the latent-mode invariant
            "latent virtuals only become parents of real events".
        """
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
            if exclude_latent_parents and source.virtual and source.latent:
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

    def _should_sample_virtuals(self, target_ts: float) -> bool:
        """True iff we should run virtual-event sampling at this tick.

        Either:
        - a MissingInterval has un-sampled time before target_ts (known mode), OR
        - latent-missing mode is on (sampling is always considered, but the
          latent path will still cap itself + skip dims covered by an active
          known interval).
        """
        if self.latent_missing_mode and self.latent_rate_multiplier > 0.0:
            return True
        if self.missing_tracker is None:
            return False
        return self.missing_tracker.has_intervals_needing_sampling(target_ts)

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
        cutoff = float(target_ts) - self.active_window_sec
        scale = float(self.config.time_scale_sec) if self.config.time_scale_sec > 0 else 1.0
        budget = int(self.max_virtual_per_call)

        # Two separate pending lists: known-interval virtuals run through the
        # legacy per-call/per-dim caps; latent virtuals get their own caps and
        # the active-virtual ceiling. They merge before injection so the
        # combined timeline stays monotone.
        pending: list[dict] = []
        latent_pending: list[dict] = []
        covered_type_ids: set[int] = set()
        call_candidate_count = 0
        call_dropped_by_dim_cap = 0

        active_intervals = (
            self.missing_tracker.intervals_needing_sampling(target_ts)
            if self.missing_tracker is not None
            else []
        )
        if self.missing_tracker is not None and not active_intervals:
            # Nothing in the known path for this tick — still compact stale
            # entries so the tracker doesn't grow unbounded under a long
            # outage history. Continue down to the latent path.
            self.missing_tracker.compact()

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
            type_ids = self.missing_tracker.type_ids_for(interval)
            if not type_ids:
                self.virtual_unresolved_interval_tick_count += 1
                continue
            # Remember dims under an active known interval at this tick — the
            # latent path skips them so we never double-sample.
            covered_type_ids.update(int(t) for t in type_ids)
            for type_id in type_ids:
                mu = float(self.params.mu[type_id])
                if mu <= 0.0:
                    continue
                expected = mu * dt_scaled
                if expected <= 0.0:
                    continue
                count = int(self.rng.poisson(expected))
                call_candidate_count += count
                self.virtual_candidate_event_count += count
                if count <= 0:
                    continue
                capped_count = min(count, self.max_virtual_per_dim)
                dim_dropped = max(0, count - capped_count)
                call_dropped_by_dim_cap += dim_dropped
                self.virtual_dropped_by_dim_cap_count += dim_dropped
                count = capped_count
                if count <= 0:
                    continue
                times = self.rng.uniform(start, upper, size=count)
                type_label = self.vocabs.type_vocab.labels[type_id]
                field_values = _type_label_field_values(type_label, self.config.type_fields)
                source_name = field_values.get("alarm_source", "")
                type_part = field_values.get("alarm_type", "")
                alarm_title = _virtual_alarm_title(type_part, field_values.get("alarm_title", ""))
                site_id = field_values.get("site_id", "") or self.ne_to_site.get(source_name, "")
                interval_key = f"{interval.key_kind}:{interval.key_value}"
                for vt in times:
                    pending.append(
                        {
                            "ts": float(vt),
                            "type_id": int(type_id),
                            "type_label": type_label,
                            "alarm_source": source_name,
                            "site_id": site_id,
                            "alarm_title": alarm_title,
                            "alarm_type": type_part or "",
                            "interval_key": interval_key,
                        }
                    )

        # ---- Latent-missing path (experimental) -----------------------------
        # Implicit "always-open" interval for every dim NOT under an active
        # known interval. Rate is μ × latent_rate_multiplier, confidence is
        # latent_confidence (typically << known mode), and the global active-
        # virtual cap (`latent_max_active_virtual`) bounds how much of
        # self.recent these can occupy. See _sample_latent_pending docstring.
        if self.latent_missing_mode and self.latent_rate_multiplier > 0.0:
            self._sample_latent_pending(
                target_ts=target_ts,
                cutoff=cutoff,
                scale=scale,
                covered_type_ids=covered_type_ids,
                out=latent_pending,
            )
            self._last_latent_sample_ts = float(target_ts)

        # Stage 2: enforce per-call budgets, apply the latent active-virtual
        # cap, merge known + latent, sort by ts, inject.
        #
        # When a single pending list exceeds its budget we **uniformly
        # subsample** rather than truncate by ts. Truncating by ts would
        # always drop the tail — exactly the virtuals nearest to the current
        # real event, which are the most likely candidate parents under the
        # exp kernel. Uniform thinning of a Poisson process is itself a
        # Poisson process with rate λ · (kept/total), so this is a valid
        # load-shedding policy: it lowers the effective μ uniformly across
        # the whole sampled window instead of starving the tail.
        call_dropped_by_call_budget = 0
        if pending and len(pending) > budget:
            call_dropped_by_call_budget = len(pending) - budget
            self.virtual_dropped_by_call_budget_count += call_dropped_by_call_budget
            kept_idx = self.rng.choice(len(pending), size=budget, replace=False)
            pending = [pending[int(i)] for i in kept_idx]

        if latent_pending and len(latent_pending) > self.latent_max_virtual_per_call:
            dropped = len(latent_pending) - self.latent_max_virtual_per_call
            self.latent_virtual_dropped_by_cap += dropped
            kept_idx = self.rng.choice(
                len(latent_pending), size=self.latent_max_virtual_per_call, replace=False
            )
            latent_pending = [latent_pending[int(i)] for i in kept_idx]

        # Active-LATENT-virtual ceiling: only counts latent virtuals already in
        # self.recent. Known virtuals (from declared missing intervals) do NOT
        # consume this budget — they have their own per-dim / per-call caps and
        # are considered "the user knows what they're doing". Mixing the two
        # would let known mode silently starve latent (e.g. a few permanent
        # known intervals could fill self.recent with known virtuals and leave
        # zero room for latent), which contradicts the flag's name and intent.
        if latent_pending:
            current_active_latent = sum(1 for e in self.recent if e.virtual and e.latent)
            room = max(0, self.latent_max_active_virtual - current_active_latent)
            if room < len(latent_pending):
                dropped = len(latent_pending) - room
                self.latent_virtual_dropped_by_cap += dropped
                if room > 0:
                    kept_idx = self.rng.choice(len(latent_pending), size=room, replace=False)
                    latent_pending = [latent_pending[int(i)] for i in kept_idx]
                else:
                    latent_pending = []

        all_pending = pending + latent_pending
        if all_pending:
            all_pending.sort(key=lambda spec: spec["ts"])
            for spec in all_pending:
                self._inject_virtual_event(**spec)

        self._maybe_warn_virtual_drop(
            target_ts=target_ts,
            candidate_count=call_candidate_count,
            dropped_by_dim_cap=call_dropped_by_dim_cap,
            dropped_by_call_budget=call_dropped_by_call_budget,
            kept_count=len(pending),
            interval_count=len(active_intervals),
        )

        # Evict closed-and-fully-sampled known intervals so subsequent ticks
        # don't scan them. Runs every call but is cheap: it's a single list filter.
        if self.missing_tracker is not None:
            self.missing_tracker.compact()

    def _sample_latent_pending(
        self,
        *,
        target_ts: float,
        cutoff: float,
        scale: float,
        covered_type_ids: set,
        out: list,
    ) -> None:
        """Sample latent-missing virtuals into ``out`` for every dim NOT under
        an active known interval at this tick.

        The total rate is aggregated over eligible dims as
        ``sum(μ_d) × latent_rate_multiplier``; sampled events are then assigned
        to dims by μ-proportional categorical sampling. The window is the
        elapsed event-time since the previous latent tick clipped to the
        assigner's active window. This is the load-shedded "any moment could
        be missing" interpretation — strong damping + global caps + audit
        counters make it tolerable, but it is still a heuristic. Use known
        intervals when you actually know data is missing.
        """
        last_latent = float(self._last_latent_sample_ts)
        if not math.isfinite(last_latent):
            return
        start = max(last_latent, cutoff)
        if start >= target_ts:
            return
        dt_scaled = (target_ts - start) / scale
        if dt_scaled <= 0.0:
            return
        rate_scale = float(self.latent_rate_multiplier)
        if self._latent_mu_total <= 0.0:
            return
        if covered_type_ids:
            allowed_mask = ~np.isin(self._latent_type_ids, list(covered_type_ids))
            type_ids = self._latent_type_ids[allowed_mask]
            mu_values = self._latent_mu_values[allowed_mask]
            if len(type_ids) == 0:
                return
            mu_cumsum = np.cumsum(mu_values)
            mu_total = float(mu_cumsum[-1])
        else:
            type_ids = self._latent_type_ids
            mu_cumsum = self._latent_mu_cumsum
            mu_total = self._latent_mu_total
        expected_total = mu_total * rate_scale * dt_scaled
        if expected_total <= 0.0:
            return
        count = int(self.rng.poisson(expected_total))
        if count <= 0:
            return
        # Cap before the per-virtual allocation loop. This is an engineering
        # hard cap, not an exact Poisson thinning step: it changes the count
        # distribution when the cap is hit. That tradeoff is intentional for
        # latent mode because the alternative is allocating K - budget dicts
        # only to drop them immediately under overload.
        budget = int(self.latent_max_virtual_per_call)
        if budget >= 0 and count > budget:
            self.latent_virtual_dropped_by_cap += count - budget
            count = budget
        if count <= 0:
            return
        thresholds = self.rng.uniform(0.0, mu_total, size=count)
        chosen_positions = np.searchsorted(mu_cumsum, thresholds, side="right")
        chosen_positions = np.clip(chosen_positions, 0, len(type_ids) - 1)
        times = self.rng.uniform(start, target_ts, size=count)
        for type_id, vt in zip(type_ids[chosen_positions], times):
            meta = self._latent_type_metadata[int(type_id)]
            out.append(
                {
                    "ts": float(vt),
                    "type_id": int(type_id),
                    "type_label": meta["type_label"],
                    "alarm_source": meta["alarm_source"],
                    "site_id": meta["site_id"],
                    "alarm_title": meta["alarm_title"],
                    "alarm_type": meta["alarm_type"],
                    "interval_key": meta["interval_key"],
                    "latent": True,
                }
            )

    def _maybe_warn_virtual_drop(
        self,
        *,
        target_ts: float,
        candidate_count: int,
        dropped_by_dim_cap: int,
        dropped_by_call_budget: int,
        kept_count: int,
        interval_count: int,
    ) -> None:
        dropped = int(dropped_by_dim_cap) + int(dropped_by_call_budget)
        if candidate_count <= 0 or dropped <= 0 or self.virtual_drop_warning_ratio <= 0.0:
            return
        drop_ratio = dropped / float(candidate_count)
        if drop_ratio < self.virtual_drop_warning_ratio:
            return
        self.virtual_drop_warning_count += 1
        _warn(
            "virtual events 被 cap 丢弃比例过高: "
            f"drop_ratio={drop_ratio:.2%}, dropped={dropped}, kept={kept_count}, "
            f"candidate={candidate_count}, by_dim_cap={dropped_by_dim_cap}, "
            f"by_call_budget={dropped_by_call_budget}, intervals={interval_count}, "
            f"ts={float(target_ts):.3f}, threshold={self.virtual_drop_warning_ratio:.2%}"
        )

    def _inject_virtual_event(
        self,
        *,
        ts: float,
        type_id: int,
        type_label: str,
        alarm_source: str,
        site_id: str,
        alarm_title: str,
        alarm_type: str,
        interval_key: str,
        latent: bool = False,
    ) -> None:
        index = self._next_index
        self._next_index += 1
        synthetic_event_id = f"virtual-{index:08d}"
        synthetic_alarm_dict = {
            # `event_id` is one of the keys aggregator._event_id() looks for,
            # so summarize_alarm_event() in to_group()/decision output will
            # surface our virtual id rather than falling back to the generic
            # alarm-NNN scheme.
            "alarm": {
                "__virtual__": True,
                "missing_interval": interval_key,
                "event_id": synthetic_event_id,
                "latent": bool(latent),
            },
            "ts": ts,
            "site_id": site_id,
            "alarm_source": alarm_source,
            "alarm_title": alarm_title,
            "alarm_type": alarm_type,
            "occurrence_uuid": new_occurrence_uuid(),
        }
        # Latent virtuals get the much-lower latent_confidence so the
        # candidate score is heavily dampened relative to known virtuals;
        # known virtuals keep the user-configured virtual_confidence.
        effective_confidence = self.latent_confidence if latent else self.virtual_confidence
        online_event = OnlineEvent(
            index=index,
            event=synthetic_alarm_dict,
            type_id=int(type_id),
            type_label=type_label,
            event_id=synthetic_event_id,
            virtual=True,
            latent=bool(latent),
            confidence=effective_confidence,
            virtual_source=interval_key,
        )
        # Latent virtuals are speculative. We forbid latent virtuals from
        # being picked as parent of any virtual (whether known or latent) —
        # chains of speculation just inflate audit counters and edges without
        # affecting downstream output (pure-virtual cascades are filtered out
        # of groups anyway). Latent virtuals can still be picked as parent of
        # REAL events, which is their actual purpose.
        candidates = self._candidate_scores(online_event, exclude_latent_parents=True)
        chosen_parent, _, _ = self._choose_candidate(candidates)
        if chosen_parent is None:
            cascade = self._create_cascade(online_event)
        else:
            cascade = self.cascades[chosen_parent.cascade_id]
            cascade.add(online_event, parent=chosen_parent)
            self.branching_edge_count += 1
            if chosen_parent.virtual:
                # virtual event picking a (known-only, by the filter above)
                # virtual parent — still synthetic but at least anchored to
                # a known-missing window.
                self.virtual_picked_as_parent_count += 1
        self._push_recent_ordered(online_event)
        self.virtual_event_count += 1
        if latent:
            self.latent_virtual_generated += 1


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _make_alarm_event(
    alarm,
    *,
    site_id,
    alarm_title,
    event_time_str,
    occurrence_uuid,
    is_clear=False,
):
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
        "occurrence_uuid": occurrence_uuid,
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
    occurrence_uuid,
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
            occurrence_uuid=occurrence_uuid,
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
                occurrence_uuid=occurrence_uuid,
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
    for alarm_ordinal, alarm in enumerate(
        stream_alarm_inputs(alarm_input, show_progress=show_progress),
        start=1,
    ):
        occurrence_uuid = input_occurrence_uuid(alarm_input, alarm_ordinal)
        for event in _raw_alarm_to_events(
            alarm,
            valid_sites=valid_sites,
            ne_to_site=ne_to_site,
            allowed_alarm_sources=allowed_alarm_sources,
            start_ts=start_ts,
            end_ts=end_ts,
            include_clear=include_clear,
            clear_delay_sec=clear_delay_sec,
            occurrence_uuid=occurrence_uuid,
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
    _print_virtual_events_state(args)


def _print_virtual_events_state(args):
    """Surface the virtual-events (Shelton missing-data) state so the user
    can confirm at a glance whether it is on, off, or unconfigured. Also
    flags sampling-knob arguments that will be ignored.
    """
    # Detect sampling-knob args that were customized despite virtual events
    # being disabled / not configured — these become silent no-ops otherwise.
    knob_defaults = {
        "virtual_confidence": 0.5,
        "max_virtual_per_call": 50,
        "max_virtual_per_dim": 100,
        "virtual_drop_warning_ratio": 0.25,
    }
    customized_knobs = [
        name for name, default in knob_defaults.items()
        if getattr(args, name, default) != default
    ]
    latent_on = bool(getattr(args, "latent_missing_mode", False))

    if args.disable_virtual_events:
        print("virtual events: 已禁用（--disable-virtual-events）")
        if customized_knobs:
            _warn(
                "以下虚拟事件参数将被忽略（virtual events 已禁用）："
                f" {', '.join('--' + n.replace('_', '-') for n in customized_knobs)}"
            )
        if latent_on:
            _warn(
                "--latent-missing-mode 在 --disable-virtual-events 下也会被忽略；"
                "去掉 --disable-virtual-events 才能启用 latent 路径"
            )
        return
    # --- known mode block ---
    if args.missing_intervals:
        print(
            f"virtual events (known): 启用 "
            f"(missing_intervals={args.missing_intervals}, "
            f"confidence={args.virtual_confidence:g}, "
            f"max_per_call={args.max_virtual_per_call}, "
            f"max_per_dim={args.max_virtual_per_dim}, "
            f"drop_warning_ratio={args.virtual_drop_warning_ratio:g})"
        )
    else:
        print("virtual events (known): 未启用（未传 --missing-intervals）")
        if customized_knobs:
            _warn(
                "以下虚拟事件参数将被忽略（未传 --missing-intervals）："
                f" {', '.join('--' + n.replace('_', '-') for n in customized_knobs)}"
            )
    # --- latent mode block (always shown so users can confirm at a glance) ---
    if latent_on:
        print(
            f"virtual events (latent, EXPERIMENTAL): 启用 "
            f"(rate_multiplier={args.latent_rate_multiplier:g}, "
            f"confidence={args.latent_confidence:g}, "
            f"max_per_call={args.latent_max_virtual_per_call}, "
            f"max_active={args.latent_max_active_virtual}) — "
            f"运维上线后请观察 metadata.latent_missing.latent_parent_rate"
        )
    else:
        print("virtual events (latent): 未启用（如需启用请传 --latent-missing-mode）")


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
        "--disable-virtual-events",
        action="store_true",
        help="全局降级开关：强制关闭缺失数据插补（Virtual Events）功能，即使提供了缺失数据也会被忽略。",
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
        "--virtual-drop-warning-ratio",
        type=float,
        default=0.25,
        help=(
            "Warn when the per-tick fraction of sampled virtual events dropped "
            "by caps reaches this ratio. Set 0 to disable. Default: 0.25."
        ),
    )
    # ---- experimental latent-missing mode ----------------------------------
    parser.add_argument(
        "--latent-missing-mode",
        action="store_true",
        help=(
            "[EXPERIMENTAL] Assume any dim may have a small chance of missing data at any "
            "moment. Samples virtual events for every dim NOT covered by an active "
            "known interval, at a heavily-discounted rate. Useful when missing intervals "
            "are operationally hard to maintain, but trades cascade noise for coverage. "
            "Inspect metadata.latent_missing.latent_parent_rate to decide whether the "
            "mode is over-firing. Off by default."
        ),
    )
    parser.add_argument(
        "--latent-rate-multiplier",
        type=float,
        default=0.05,
        help=(
            "Per-dim background-rate discount for latent virtuals (in [0, 1]). The "
            "effective sampling rate is μ_d × multiplier; the default 0.05 means "
            "'at most ~5%% of events could plausibly have been missed'. Set to 0 to "
            "disable latent sampling without disabling the mode (no-op)."
        ),
    )
    parser.add_argument(
        "--latent-confidence",
        type=float,
        default=0.05,
        help=(
            "Score multiplier for latent virtual parent candidates (in (0, 1]). Much "
            "lower than --virtual-confidence so latent virtuals only win when the "
            "real candidate field is exceptionally thin. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--latent-max-virtual-per-call",
        type=int,
        default=10,
        help="Cap on latent virtuals generated per real-event tick. Default: 10.",
    )
    parser.add_argument(
        "--latent-max-active-virtual",
        type=int,
        default=200,
        help=(
            "Hard ceiling on LATENT virtuals currently living in self.recent. "
            "Known-interval virtuals are counted independently and do NOT consume "
            "this budget (they have their own per-dim / per-call caps). When "
            "reached, new latent samples this tick are dropped (audited as "
            "latent_virtual_dropped_by_cap). Bounds self.recent bloat from latent "
            "noise specifically. Default: 200."
        ),
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
    if args.disable_virtual_events:
        if args.missing_intervals:
            _warn(
                "缺失数据插补已被 --disable-virtual-events 强制关闭，"
                f"传入的 --missing-intervals {args.missing_intervals} 将被忽略。"
            )
    elif args.missing_intervals:
        missing_tracker = MissingIntervalTracker(artifact.vocabs, type_fields=config.type_fields)
        loaded = load_missing_from_json(args.missing_intervals)
        for interval in loaded:
            missing_tracker.add(interval)
        print(
            f"已加载 missing intervals: 共 {len(loaded)} 条 "
            f"(open={missing_tracker.stats()['open_count']}, "
            f"closed={missing_tracker.stats()['closed_count']})，"
            f"virtual_confidence={args.virtual_confidence:g}, "
            f"max_per_call={args.max_virtual_per_call}, "
            f"max_per_dim={args.max_virtual_per_dim}, "
            f"drop_warning_ratio={args.virtual_drop_warning_ratio:g}"
        )
    # Latent mode also wants NE→site mapping for synthetic events, even when
    # the user hasn't supplied missing-intervals.
    latent_mode_on = bool(getattr(args, "latent_missing_mode", False)) and not args.disable_virtual_events
    needs_ne_site_map = missing_tracker is not None or latent_mode_on
    virtual_ne_to_site = _load_ne_to_site_map(args.ne_graph) if needs_ne_site_map else {}
    assigner = OnlineBRUNCHAssigner(
        artifact,
        config=config,
        active_window_sec=args.active_window_sec,
        parent_selection=parent_selection,
        seed=args.seed,
        missing_tracker=missing_tracker,
        ne_to_site=virtual_ne_to_site,
        virtual_confidence=args.virtual_confidence,
        max_virtual_per_call=args.max_virtual_per_call,
        max_virtual_per_dim=args.max_virtual_per_dim,
        virtual_drop_warning_ratio=args.virtual_drop_warning_ratio,
        close_check_min_interval_sec=args.close_check_interval_sec,
        latent_missing_mode=latent_mode_on,
        latent_rate_multiplier=args.latent_rate_multiplier,
        latent_confidence=args.latent_confidence,
        latent_max_virtual_per_call=args.latent_max_virtual_per_call,
        latent_max_active_virtual=args.latent_max_active_virtual,
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
