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

from collections import Counter, deque
from dataclasses import dataclass, field
from dataclasses import replace as _replace
import json
import math
import os
import sys
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

import numpy as np

from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.sequences import alarm_type_from_title, event_type_label
from alarm_flow_mhp.aggregator import (
    AlarmMHPConfig,
    load_alarm_mhp_artifact,
    summarize_alarm_event,
)
from fault_grouping.alarm_events.io import is_clear_alarm
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, resource_display


EPS = 1e-12


# --------------------------------------------------------------------------
# Core data structures
# --------------------------------------------------------------------------


@dataclass
class OnlineEvent:
    """One event observed by the streaming pipeline."""

    index: int                  # global ordinal in the stream
    ts: float                   # event time (epoch seconds)
    type_id: int                # vocab type ID
    type_label: str             # human-readable type
    alarm: dict                 # original alarm event dict (for output)
    parent_index: int = -1      # -1 for immigrant; otherwise the OnlineEvent.index of parent
    cascade_id: int = -1        # assigned cascade ID
    parent_score: float = 0.0   # score that won (for debugging / output)


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
        return base


class StreamMHPAssigner:
    """Stateful online inference engine.

    Maintains:
      - `recent`: deque of real events still within history_window
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

    def __init__(self, artifact, config: StreamConfig):
        self.artifact = artifact
        self.params = artifact.params
        self.vocabs = artifact.vocabs
        self.config = config
        # Sliding window: events with ts >= now - window are kept
        self.recent: deque[OnlineEvent] = deque()
        self.cascades: dict[int, Cascade] = {}
        self._next_cascade_id: int = 0
        self._next_event_index: int = 0
        # Stats for reporting
        self.total_events_processed = 0
        self.total_immigrants = 0
        self.closed_cascade_count = 0
        self.dropped_unknown_type = 0
        self.dropped_clear = 0
        self.dropped_no_type = 0
        # Closed cascade output sink — populated as cascades close
        self.closed_groups: list = []

    def _resolve_type_id(self, alarm_event) -> tuple[int, str] | None:
        """Translate an alarm event to (type_id, label) using artifact vocabs."""
        type_label = event_type_label(alarm_event, self.artifact.config.type_fields)
        type_id = self.vocabs.type_vocab.get(type_label)
        if type_id is None:
            return None
        return type_id, type_label

    def _evict_expired(self, now_ts: float):
        """Drop events from the left of `recent` that fell out of the window.

        We use real-seconds × time_scale_sec for the comparison since `recent`
        stores real timestamps.
        """
        cutoff = now_ts - self.config.history_window_sec
        while self.recent and self.recent[0].ts < cutoff:
            self.recent.popleft()

    def _candidate_scores(self, target_type_id: int, target_ts: float):
        """Score each event in `recent` as a candidate parent.

        Returns a (sources, scores) pair where sources[k] is the recent-list
        index and scores[k] is the unnormalized score for that candidate.
        """
        n_recent = len(self.recent)
        if n_recent == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        cap = min(n_recent, self.config.max_history_events)
        # Iterate from newest backwards — only the last `cap` candidates matter
        # because kernel decay puts older events near zero weight.
        sources = np.empty(cap, dtype=np.int64)
        scores = np.empty(cap, dtype=np.float64)
        i = 0
        for idx in range(n_recent - 1, n_recent - 1 - cap, -1):
            cand = self.recent[idx]
            sources[i] = idx
            # Δt in model time units (scaled by time_scale_sec)
            dt = (target_ts - cand.ts) / self.config.time_scale_sec
            if dt <= 0:
                scores[i] = 0.0
            else:
                alpha = self.params.alpha_value(target_type_id, cand.type_id)
                if alpha <= 0:
                    scores[i] = 0.0
                else:
                    beta = self.params.beta_value(target_type_id, cand.type_id)
                    scores[i] = alpha * beta * math.exp(-beta * dt)
            i += 1
        return sources, scores

    def process(self, alarm_event: dict):
        """Ingest one alarm event in time order, run inference, update state."""
        # Filtering: skip events we can't model
        if is_clear_alarm(alarm_event.get("alarm", {}) if isinstance(alarm_event, dict) else {}):
            self.dropped_clear += 1
            return None
        if alarm_type_from_title(alarm_event.get("alarm_title", "")) is None:
            self.dropped_no_type += 1
            return None
        resolved = self._resolve_type_id(alarm_event)
        if resolved is None:
            self.dropped_unknown_type += 1
            return None
        type_id, type_label = resolved

        ts = float(alarm_event.get("ts", 0.0))
        # Close cascades that have been silent long enough — done before scoring
        # so a long-quiet cascade can't catch a new event by accident.
        self._close_inactive(ts)
        self._evict_expired(ts)

        # Score candidates
        sources, scores = self._candidate_scores(type_id, ts)
        mu = float(self.params.mu[type_id]) * self.config.immigrant_bias
        # argmax over (μ, top_candidate)
        if scores.size == 0 or scores.max() <= mu:
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
            parent_event = self.recent[int(sources[best_local])]
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
        )
        self._next_event_index += 1
        cascade.add(event)
        self.recent.append(event)
        # Cap recent deque so memory stays bounded — old events fall off via
        # _evict_expired in normal use, but very dense bursts could grow it
        # unboundedly between evictions.
        if len(self.recent) > self.config.max_history_events * 4:
            self.recent.popleft()
        self.total_events_processed += 1
        return event

    def _close_inactive(self, now_ts: float):
        """Move cascades whose last_ts < now - close_inactive_sec to closed_groups."""
        cutoff = now_ts - self.config.close_inactive_sec
        to_close: list[int] = []
        for cid, cascade in self.cascades.items():
            if cascade.last_ts < cutoff:
                to_close.append(cid)
        for cid in to_close:
            cascade = self.cascades.pop(cid)
            if cascade.event_count() >= self.config.min_group_events:
                self.closed_groups.append(_cascade_to_group(cascade))
                self.closed_cascade_count += 1

    def close_remaining(self):
        """Emit any still-active cascades. Call once at end of stream."""
        for cid in list(self.cascades.keys()):
            cascade = self.cascades.pop(cid)
            if cascade.event_count() >= self.config.min_group_events:
                self.closed_groups.append(_cascade_to_group(cascade))
                self.closed_cascade_count += 1

    def stats(self) -> dict:
        return {
            "total_events_processed": self.total_events_processed,
            "total_immigrants": self.total_immigrants,
            "closed_cascade_count": self.closed_cascade_count,
            "open_cascade_count": len(self.cascades),
            "recent_window_size": len(self.recent),
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


def _cascade_size_stats(groups: list) -> dict | None:
    if not groups:
        return None
    sizes = np.array([g["event_count"] for g in groups], dtype=np.int64)
    n_cascades = int(sizes.size)
    n_events = int(sizes.sum())
    multi_mask = sizes >= 2
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
        "multi_event_cascade_count": int(multi_mask.sum()),
        "multi_event_cascade_share": float(multi_mask.sum() / n_cascades),
        "multi_event_event_count": int(sizes[multi_mask].sum()),
        "multi_event_event_share": float(sizes[multi_mask].sum() / n_events),
        "max_size": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "histogram": histogram,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Online (streaming) MHP inference: ingest alarms in time order, emit cascade groups."
    )
    parser.add_argument("model", help="Trained MHP artifact JSON (produced by train_alarm_mhp.py).")
    parser.add_argument("alarms", help="Sorted alarm cache or raw alarms — same format as run_alarm_mhp.")
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
        "--progress-every",
        type=int,
        default=50_000,
        help="Print stream progress every N events. 0 = silent. Default: 50000.",
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

    assigner = StreamMHPAssigner(artifact, stream_config)

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
            f"groups={len(groups)}, "
            f"immigrants={stats['total_immigrants']}, "
            f"closed_cascades={stats['closed_cascade_count']}, "
            f"dropped={{clear:{stats['dropped_clear']}, no_type:{stats['dropped_no_type']}, "
            f"unknown_type:{stats['dropped_unknown_type']}}}",
            flush=True,
        )

    cascade_stats = _cascade_size_stats(groups)
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

    if cascade_stats and not args.quiet:
        print("[stream] cascade size distribution:")
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
