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
from alarm_flow_brunch.region_filter import parse_regions
from alarm_flow_brunch.visual_output import AlarmBRUNCHVisualOutputSession
from alarm_flow_isahp.sequences import alarm_type_label, event_type_label
from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_tools.alarm_types import CRITICAL_ALARMS
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


@dataclass
class OnlineEvent:
    index: int
    event: dict
    type_id: int
    type_label: str
    event_id: str
    cascade_id: str = ""

    @property
    def ts(self) -> float:
        return float(self.event.get("ts", 0.0))


@dataclass
class OnlineCascade:
    cascade_id: str
    root: OnlineEvent
    events: list[OnlineEvent] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def add(self, event: OnlineEvent, parent: OnlineEvent | None = None):
        event.cascade_id = self.cascade_id
        self.events.append(event)
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
    def last_ts(self):
        return max(item.ts for item in self.events)

    def to_group(self):
        summaries = [summarize_alarm_event(item.event, item.index) for item in self.events]
        timestamps = [summary["ts"] for summary in summaries]
        return {
            "group_id": self.cascade_id,
            "cascade_id": self.cascade_id,
            "event_count": len(self.events),
            "start_ts": min(timestamps),
            "end_ts": max(timestamps),
            "duration_sec": max(timestamps) - min(timestamps),
            "root_event": summarize_alarm_event(self.root.event, self.root.index),
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


class OnlineBRUNCHAssigner:
    def __init__(
        self,
        artifact,
        *,
        config,
        active_window_sec: float | None = None,
        parent_selection: str | None = None,
        seed: int = 0,
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
        self.rng = np.random.default_rng(seed)
        self.recent: deque[OnlineEvent] = deque()
        self._active_sources_cache: dict[int, set[int]] = {}
        self.cascades: dict[str, OnlineCascade] = {}
        self.closed_cascade_ids: set[str] = set()
        self.edges: list[dict] = []
        self.skipped_count = 0
        self.processed_count = 0
        self.modeled_count = 0
        self._next_index = 0
        self._next_cascade_ordinal = 1

    def assign(self, event: dict):
        self.processed_count += 1
        if not self.config.include_clear and is_clear_alarm(event.get("alarm", {})):
            return self._skip(event, "clear_alarm_disabled")

        coarse_type = alarm_type_label(event)
        if coarse_type is None:
            return self._skip(event, "unsupported_alarm_type")

        type_label = event_type_label(event, self.config.type_fields)
        type_id = self.vocabs.type_vocab.get(type_label)
        if type_id is None:
            return self._skip(event, "unknown_event_type", details={"type_label": type_label})

        self._expire_recent(float(event["ts"]))
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
        if chosen_parent is None:
            cascade = self._create_cascade(online_event)
            reason = "new_cascade"
            parent_event_id = ""
            dt_sec = None
        else:
            cascade = self.cascades[chosen_parent.cascade_id]
            cascade.add(online_event, parent=chosen_parent)
            self.edges.append(cascade.edges[-1])
            reason = "assigned"
            parent_event_id = chosen_parent.event_id
            dt_sec = online_event.ts - chosen_parent.ts

        self.recent.append(online_event)
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
                "target_type": type_label,
                "parent_type": chosen_parent.type_label if chosen_parent is not None else "",
                "dt_sec": dt_sec,
                "parent_selection": self.parent_selection,
            },
        }

    def groups(self, min_group_events: int = 1):
        return [
            cascade.to_group()
            for cascade in sorted(self.cascades.values(), key=lambda item: item.root.index)
            if len(cascade.events) >= min_group_events
        ]

    def close_inactive(self, now_ts, close_after_sec, *, min_group_events=1):
        cutoff = float(now_ts) - float(close_after_sec)
        closed_groups = []
        newly_closed = set()
        for cascade in sorted(self.cascades.values(), key=lambda item: item.root.index):
            if cascade.cascade_id in self.closed_cascade_ids:
                continue
            if cascade.last_ts > cutoff:
                continue
            self.closed_cascade_ids.add(cascade.cascade_id)
            newly_closed.add(cascade.cascade_id)
            if len(cascade.events) >= min_group_events:
                closed_groups.append(cascade.to_group())
        if newly_closed:
            self._drop_recent_for_closed()
        return closed_groups

    def close_remaining(self, *, min_group_events=1):
        remaining_groups = []
        for cascade in sorted(self.cascades.values(), key=lambda item: item.root.index):
            if cascade.cascade_id in self.closed_cascade_ids:
                continue
            self.closed_cascade_ids.add(cascade.cascade_id)
            if len(cascade.events) >= min_group_events:
                remaining_groups.append(cascade.to_group())
        if remaining_groups:
            self._drop_recent_for_closed()
        return remaining_groups

    def metadata(self):
        return {
            "algorithm": "alarm_flow_brunch_online",
            "config": self.config.to_dict(),
            "online": {
                "active_window_sec": self.active_window_sec,
                "parent_selection": self.parent_selection,
            },
            "processed_event_count": self.processed_count,
            "modeled_event_count": self.modeled_count,
            "skipped_event_count": self.skipped_count,
            "group_count": len(self.cascades),
            "branching_edge_count": len(self.edges),
            "type_count": len(self.vocabs.type_vocab),
            "active_edge_count": len(self.params.active_edges(include_self=True)[0]),
            "type_labels": list(self.vocabs.type_vocab.labels),
        }

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
        cutoff = float(target_ts) - self.active_window_sec
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
            if source.cascade_id in self.closed_cascade_ids:
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

    def _drop_recent_for_closed(self):
        self.recent = deque(
            event for event in self.recent
            if event.cascade_id not in self.closed_cascade_ids
        )

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
        return cascade


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
    if args.groups_output:
        return args.groups_output
    if args.visual_output:
        return ""
    return str(Path(args.output).with_suffix(".groups.json"))


def _count_decision(counts, decision):
    status = decision.get("status", "")
    counts[status] = counts.get(status, 0) + 1


def _emit_closed_visual_groups(visual_output, assigner, now_ts, close_after_sec, min_group_events):
    groups = assigner.close_inactive(
        now_ts,
        close_after_sec,
        min_group_events=min_group_events,
    )
    if visual_output is None:
        return 0
    return visual_output.emit_groups(groups, finalization_reason="closed")


def _emit_remaining_visual_groups(visual_output, assigner, min_group_events):
    groups = assigner.close_remaining(min_group_events=min_group_events)
    if visual_output is None:
        return 0, groups
    return visual_output.emit_groups(groups, finalization_reason="stream_end"), groups


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
            "Final cascade snapshot JSON; default: replace output suffix with "
            ".groups.json unless --visual-output is set."
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
        default=60.0,
        help="Maximum event-time wait before emitting a decision. Default: 60 seconds.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sample mode.")
    parser.add_argument("--show-progress", action="store_true", help="Show input read progress.")
    args = parser.parse_args()
    groups_output = _resolve_groups_output(args)

    artifact = load_alarm_brunch_artifact(args.model)
    regions = parse_regions(args.regions) if args.regions is not None else artifact.config.regions
    parent_selection = args.parent_selection or artifact.config.parent_selection
    config = replace(
        artifact.config,
        regions=regions,
        include_clear=bool(args.include_clear or artifact.config.include_clear),
        parent_selection=parent_selection,
    )
    assigner = OnlineBRUNCHAssigner(
        artifact,
        config=config,
        active_window_sec=args.active_window_sec,
        parent_selection=parent_selection,
        seed=args.seed,
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

    try:
        with open(args.output, "w", encoding="utf-8") as decision_stream:
            for event in _iter_stream_events(
                args.alarms,
                topo_path=args.topo,
                ne_graph_path=args.ne_graph,
                regions=config.regions,
                start_time=args.start_time or None,
                end_time=args.end_time or None,
                include_clear=config.include_clear,
                clear_delay_sec=args.clear_delay_sec,
                show_progress=args.show_progress,
            ):
                reorder_buffer.push(event)
                for ready_event in reorder_buffer.ready():
                    decision = _process_event(assigner, decision_stream, ready_event)
                    _count_decision(counts, decision)
                    _emit_closed_visual_groups(
                        visual_output,
                        assigner,
                        decision["ts"],
                        args.close_after_sec,
                        args.min_group_events,
                    )
            for ready_event in reorder_buffer.flush():
                decision = _process_event(assigner, decision_stream, ready_event)
                _count_decision(counts, decision)
                _emit_closed_visual_groups(
                    visual_output,
                    assigner,
                    decision["ts"],
                    args.close_after_sec,
                    args.min_group_events,
                )
            remaining_emitted, remaining_groups = _emit_remaining_visual_groups(
                visual_output,
                assigner,
                args.min_group_events,
            )
    finally:
        if visual_output is not None:
            visual_output.close()

    groups = assigner.groups(min_group_events=args.min_group_events)
    metadata = assigner.metadata()
    metadata["input"] = os.path.abspath(args.alarms)
    metadata["model"] = os.path.abspath(args.model)
    metadata["decision_output"] = os.path.abspath(args.output)
    metadata["group_count"] = len(groups)
    metadata["reorder_lag_sec"] = float(args.reorder_lag_sec)
    metadata["regions"] = sorted(config.regions)

    if groups_output:
        _write_json(
            groups_output,
            {
                "decision_counts": counts,
                "cascade_count": len(groups),
                "cascades": groups,
            },
        )
    visual_count = visual_output.emitted_count if visual_output is not None else 0

    print(
        f"online BRUNCH decisions written to: {args.output}; "
        f"events={metadata['modeled_event_count']}, "
        f"skipped={metadata['skipped_event_count']}, "
        f"groups={metadata['group_count']}, "
        f"edges={metadata['branching_edge_count']}"
    )
    if groups_output:
        print(f"final groups written to: {groups_output}")
    if args.visual_output:
        print(f"visual groups written to: {args.visual_output}; groups={visual_count}")


if __name__ == "__main__":
    main()
