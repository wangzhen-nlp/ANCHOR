#!/usr/bin/env python3
"""AlarmPeriod-oriented online inference for feature-mode alarm-flow MHP.

This is intentionally a separate engine from ``stream_alarm_mhp.py``. The
occurrence-oriented engine assigns one parent to every alarm occurrence; this
engine uses an AlarmPeriod as the matching and grouping unit:

* repeated ``(feature entity, alarm type)`` raises share one open period;
* a period freezes the dynamic source/target state seen before its first raise;
* matching waits for a fixed event-time aggregation lag and then harvests only
  occurrences added since the previous harvest;
* static feature amplitude, relation prior, immigrant threshold, and reachable
  past/future horizons can be loaded from an offline sparse cache; signatures
  absent from that cache are compiled once and retained in memory;
* the temporal score uses the closest valid occurrence pair between periods;
* a period has one primary fault group.  Cross-group evidence creates a merge
  proposal; it never launches an eager BFS/DFS over the historical graph.

The result is a match-rules-style execution plan driven by MHP parameters.  It
is a new grouping semantics, not a bit-for-bit replacement for the branching
parent inference engines.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
import heapq
import hashlib
from itertools import groupby
import json
import math
import os
import time
from typing import Optional

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

import numpy as np

from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.event_domain import (
    DEVICE_DOMAIN_FIELD,
    SUPPORTED_DEVICE_DOMAINS,
    build_ne_domain_bucket_map,
    filter_and_annotate_device_domain,
)
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import (
    alarm_type_from_title,
    alarm_type_label,
    event_type_label,
)
from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_mhp.candidate_policy import (
    adaptive_candidate_sources,
    candidate_policy_fingerprint,
    load_candidate_policy,
    prepare_adaptive_candidates,
    unrelated_pair_allowed,
)
from alarm_flow_mhp.dynamic_state import DeviceStateTracker
from alarm_flow_mhp.feature_spec import (
    DecomposedFeatureScorer,
    MuFeatureSpec,
    RuntimeFeatureScorer,
    RuntimeMuScorer,
    build_node_context,
    domain_of,
    make_entity,
    runtime_ne_at,
    topo_node_of,
)
from alarm_flow_mhp.stream_alarm_mhp import OnlineEvent, _summary_of
from alarm_flow_mhp.topology_relation_prior import (
    RELATION_KEYS,
    parse_topology_relation_prior,
    relation_weight,
    topology_relation_weights,
)
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.alarm_events.identity import require_alarm_identity
from fault_grouping.alarm_events.io import is_clear_alarm, parse_datetime_text
from fault_grouping.alarm_events.sorted_cache import (
    SortedAlarmCacheStream,
    is_sorted_alarm_cache_file,
    iter_sorted_alarm_cache_items,
)
from fault_grouping.matching.profiling import PhaseTimer
from mhp.feature_kernel import FeatureKernel
from ne_link_learning.core import normalize_text
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display
from topology_tools.region_utils import (
    build_ne_region_map,
    event_region,
    load_ne_graph,
    parse_regions,
)


EPS = 1e-12

# Slack subtracted from the separable-logit prescreen threshold so float
# reassociation (z_at + z_ent vs the exact left-to-right sum) can never drop a
# candidate that exact scoring would keep. Logit magnitudes are O(10²), so the
# reassociation error is bounded well below 1e-9.
PRESCREEN_LOGIT_MARGIN = 1e-6


def _profile_phase(timer, name):
    return timer.time(name) if timer is not None else nullcontext()


def _iter_profiled_events(events, timer):
    """Attribute lazy input decoding/filtering time separately from processing."""
    iterator = iter(events)
    while True:
        started = time.perf_counter()
        try:
            event = next(iterator)
        except StopIteration:
            return
        timer._record("input.read_event", time.perf_counter() - started)
        yield event


def _enable_period_profiling(timer, engine, output):
    """Instrument the AlarmPeriod hot path only when ``--profile`` is enabled.

    This follows ``fault_grouping.match_rules``: production methods stay
    untouched until profiling is requested, then bound methods are wrapped with
    aggregated ``perf_counter`` timing.  Nested phases intentionally overlap.
    """
    engine_phases = (
        ("process", "ingest.process"),
        ("_open_or_create_period", "period.open_or_create"),
        ("_handle_clear", "period.handle_clear"),
        ("_close_idle_periods", "maintenance.close_idle_periods"),
        ("_advance_watermark", "ingest.advance_watermark"),
        ("_harvest_ready", "harvest.ready"),
        ("_harvest_period", "harvest.period"),
        ("_collect_relations", "harvest.collect_relations"),
        ("_best_for_new_targets", "harvest.match_new_targets"),
        ("_best_for_new_sources", "harvest.match_new_sources"),
        ("_apply_relations", "group.apply_relations"),
        ("_choose_or_create_group", "group.choose_or_create"),
        ("_try_ready_merge_proposals", "group.scan_merge_proposals"),
        ("_merge_groups", "group.merge"),
        ("_close_inactive_groups", "maintenance.close_inactive_groups"),
        ("_finalize_group", "maintenance.finalize_group"),
        ("_evict_expired_periods", "maintenance.evict_expired_periods"),
        ("_group_record", "output.build_group_record"),
        ("flush", "flush.total"),
    )
    for method_name, phase_name in engine_phases:
        timer.wrap_method(engine, method_name, phase_name)

    plan_phases = (
        ("register_signature", "association.register_signature"),
        ("_compute_edge", "association.compute_edge"),
    )
    for method_name, phase_name in plan_phases:
        timer.wrap_method(engine.plan, method_name, phase_name)

    output_phases = (
        ("emit_group", "output.emit_group"),
        ("_write_group_record", "output.groups_jsonl"),
        ("close", "output.close"),
    )
    for method_name, phase_name in output_phases:
        timer.wrap_method(output, method_name, phase_name)
    if output.visual is not None:
        timer.wrap_method(output.visual, "emit_groups", "output.visual")

    # The sink stores a bound method, so refresh it after emit_group is wrapped.
    engine.closed_group_sink = output.emit_group


def _print_period_profile(timer):
    """Print an AlarmPeriod-specific flat summary of nested cumulative phases."""
    phases = timer.snapshot()
    wall = timer.wall_elapsed
    if not phases and wall <= 0:
        return

    blocks = (
        ("init", "准备阶段", lambda name: name.startswith("init.")),
        (
            "pipeline",
            "输入与主流程",
            lambda name: name in {
                "pipeline.total",
                "pipeline.progress",
                "input.read_event",
                "ingest.process",
                "flush.total",
            },
        ),
        (
            "period",
            "Period 生命周期",
            lambda name: name.startswith("period.") or name == "ingest.advance_watermark",
        ),
        ("harvest", "关系收集", lambda name: name.startswith("harvest.")),
        ("group", "分组与合并", lambda name: name.startswith("group.")),
        ("maintenance", "维护与淘汰", lambda name: name.startswith("maintenance.")),
        ("association", "缓存外关系编译", lambda name: name.startswith("association.")),
        ("output", "输出", lambda name: name.startswith("output.")),
    )

    def format_row(name, values):
        total = values["total_seconds"]
        count = values["count"]
        wall_pct = total / wall * 100.0 if wall > 0 else 0.0
        average_ms = total / max(count, 1) * 1000.0
        return (
            f"  {name:<38} {total:>10.3f}s {wall_pct:>7.1f}% "
            f"{count:>10}次 avg={average_ms:>10.3f}ms"
        )

    line_width = 100
    print()
    print("=" * line_width)
    print(f"AlarmPeriod MHP 性能分析（wall={wall:.3f}s）")
    print("=" * line_width)
    emitted = set()
    for _key, title, predicate in blocks:
        rows = [
            (name, values)
            for name, values in phases.items()
            if predicate(name) and name not in emitted
        ]
        if not rows:
            continue
        rows.sort(key=lambda item: -item[1]["total_seconds"])
        print()
        print(f"[{title}]")
        for name, values in rows:
            print(format_row(name, values))
            emitted.add(name)

    other_rows = [
        (name, values) for name, values in phases.items() if name not in emitted
    ]
    if other_rows:
        print()
        print("[其他]")
        for name, values in sorted(
            other_rows, key=lambda item: -item[1]["total_seconds"]
        ):
            print(format_row(name, values))
    print()
    print("说明：各阶段是累计耗时；父阶段包含子阶段，百分比不能直接相加。")
    print("      profiling 会增加少量 perf_counter/方法包装开销，仅用于定位瓶颈。")
    print("=" * line_width)


def _softplus_lower_logit(y: float) -> float:
    """Smallest logit z with softplus(z) >= y (softplus inverse, -inf if y<=0)."""
    if y <= 0.0:
        return -math.inf
    if y > 30.0:
        return float(y)
    return math.log(math.expm1(y))
ASSOCIATION_CACHE_FORMAT = "alarm_flow_mhp.period_association_cache"
ASSOCIATION_CACHE_VERSION = 4
CACHE_STATE_LAYOUT_FULL = "target_source_state"
CACHE_STATE_LAYOUT_TARGET_ONLY = "target_state_only"


def association_cache_state_layout(dynamic_mode) -> str:
    """Cache key layout required by a dynamic α mode.

    target mode is invariant to the source period's frozen state, so the cache
    stores one edge per source PeriodType. Other modes retain exact source
    signatures because source state can affect α (or for legacy off-mode
    compatibility).
    """
    return (
        CACHE_STATE_LAYOUT_TARGET_ONLY
        if str(dynamic_mode or "off") == "target"
        else CACHE_STATE_LAYOUT_FULL
    )


@dataclass
class PeriodStreamConfig:
    aggregation_wait_sec: float = 30.0
    period_idle_sec: float = 300.0
    history_window_sec: float = 900.0
    time_slack_sec: float = 0.0
    late_penalty_half_life_sec: float = 1.0
    time_scale_sec: float = 60.0
    close_inactive_sec: float = 7200.0
    min_group_events: int = 1
    immigrant_bias: float = 1.0
    feature_alpha_floor: float = 0.0
    attach_threshold_ratio: float = 1.0
    relative_attach_ratio: float = 0.8
    max_related_periods: int = 8
    max_core_periods: int = 4
    merge_strength_ratio: float = 2.0
    merge_min_evidence: int = 2
    candidate_scope: str = "related"
    topology_relation_prior: dict = field(default_factory=dict)

    def validate(self):
        if self.aggregation_wait_sec < 0:
            raise ValueError("aggregation_wait_sec must be >= 0")
        if self.period_idle_sec <= 0:
            raise ValueError("period_idle_sec must be > 0")
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be > 0")
        if self.time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if self.aggregation_wait_sec < self.time_slack_sec:
            raise ValueError("aggregation_wait_sec must be >= time_slack_sec")
        if self.late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        if self.time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be > 0")
        if self.close_inactive_sec < 0:
            raise ValueError("close_inactive_sec must be >= 0")
        if self.min_group_events < 1:
            raise ValueError("min_group_events must be >= 1")
        if self.immigrant_bias <= 0:
            raise ValueError("immigrant_bias must be > 0")
        if self.feature_alpha_floor < 0:
            raise ValueError("feature_alpha_floor must be >= 0")
        if self.attach_threshold_ratio <= 0:
            raise ValueError("attach_threshold_ratio must be > 0")
        if not 0 < self.relative_attach_ratio <= 1:
            raise ValueError("relative_attach_ratio must be in (0, 1]")
        if self.max_related_periods < 1:
            raise ValueError("max_related_periods must be >= 1")
        if self.max_core_periods < 1:
            raise ValueError("max_core_periods must be >= 1")
        if self.merge_strength_ratio <= 0:
            raise ValueError("merge_strength_ratio must be > 0")
        if self.merge_min_evidence < 1:
            raise ValueError("merge_min_evidence must be >= 1")
        if self.candidate_scope not in {"related", "global", "unrelated"}:
            raise ValueError(
                "candidate_scope must be 'related', 'global', or 'unrelated'"
            )


def _association_plan_config(config: PeriodStreamConfig) -> dict:
    """Only values that affect compiled edges or candidate coverage."""
    return {
        "history_window_sec": float(config.history_window_sec),
        "time_slack_sec": float(config.time_slack_sec),
        "late_penalty_half_life_sec": float(config.late_penalty_half_life_sec),
        "time_scale_sec": float(config.time_scale_sec),
        "immigrant_bias": float(config.immigrant_bias),
        "feature_alpha_floor": float(config.feature_alpha_floor),
        "attach_threshold_ratio": float(config.attach_threshold_ratio),
        "candidate_scope": str(config.candidate_scope),
        "topology_relation_prior": {
            str(key): float(value)
            for key, value in sorted((config.topology_relation_prior or {}).items())
        },
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def association_cache_fingerprint(
    model_path,
    ne_graph_path,
    site_graph_path,
    config,
    topology_node_field="alarm_source",
    candidate_policy_path="",
) -> dict:
    """Fingerprint every input that can change the sparse association plan."""
    node_field = str(topology_node_field or "alarm_source")
    topology_graph_path = site_graph_path if node_field == "site_id" else ne_graph_path
    fingerprint = {
        "model_sha256": _sha256_file(model_path),
        "ne_graph_sha256": _sha256_file(ne_graph_path),
        "topology_graph_sha256": _sha256_file(topology_graph_path),
        "topology_node_field": node_field,
        "plan_config": _association_plan_config(config),
    }
    if candidate_policy_path:
        fingerprint["candidate_policy_sha256"] = _sha256_file(
            candidate_policy_path
        )
    return fingerprint


_FINGERPRINT_SCOPE_AGNOSTIC_KEYS = frozenset({"candidate_policy_sha256"})


def _fingerprint_compatible_ignoring_scope(actual, expected) -> bool:
    """Whether two fingerprints agree once candidate scope/policy is ignored.

    Separately compiled caches (e.g. a ``related`` cache and an ``unrelated``
    cache built from the same model and graphs) differ only in
    ``plan_config.candidate_scope`` and the optional policy digest. They stay
    co-loadable as long as everything else — model, graphs, node field and the
    remaining plan config — is identical.
    """
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    keys = (set(actual) | set(expected)) - _FINGERPRINT_SCOPE_AGNOSTIC_KEYS
    if any(
        key != "plan_config" and actual.get(key) != expected.get(key)
        for key in keys
    ):
        return False
    actual_plan = actual.get("plan_config") or {}
    expected_plan = expected.get("plan_config") or {}
    if not isinstance(actual_plan, dict) or not isinstance(expected_plan, dict):
        return False
    plan_keys = set(actual_plan) | set(expected_plan)
    return not any(
        key != "candidate_scope" and actual_plan.get(key) != expected_plan.get(key)
        for key in plan_keys
    )


def load_association_cache(
    path,
    expected_fingerprint=None,
    allow_scope_mismatch=False,
) -> dict:
    try:
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata_json"].item()))
            array_names = (
                "target_signature_ids",
                "source_signature_ids",
                "base_scores",
                "thresholds",
                "past_windows",
                "future_windows",
                "target_offsets",
                "source_offsets",
                "source_order",
            )
            arrays = {name: archive[name] for name in array_names}
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid binary association cache: {exc}") from exc
    if header.get("format") != ASSOCIATION_CACHE_FORMAT:
        raise ValueError(f"unsupported association cache format: {header.get('format')!r}")
    if int(header.get("version", -1)) != ASSOCIATION_CACHE_VERSION:
        raise ValueError(
            f"unsupported association cache version: {header.get('version')!r}"
        )
    if expected_fingerprint is not None:
        actual = header.get("fingerprint") or {}
        compatible_scope_upgrade = (
            allow_scope_mismatch
            and _fingerprint_compatible_ignoring_scope(
                actual, expected_fingerprint
            )
        )
        if actual != expected_fingerprint and not compatible_scope_upgrade:
            changed = sorted(
                key
                for key in set(actual) | set(expected_fingerprint)
                if actual.get(key) != expected_fingerprint.get(key)
            )
            raise ValueError(
                "association cache does not match current model/graphs/config; "
                f"changed={','.join(changed) or 'unknown'}"
            )
    edge_count = int((header.get("metadata") or {}).get("edge_count", -1))
    cache_metadata = header.get("metadata") or {}
    signature_count = int(cache_metadata.get("signature_count", -1))
    source_key_count = int(cache_metadata.get("source_key_count", signature_count))
    state_layout = str(cache_metadata.get("state_layout", ""))
    if state_layout not in {
        CACHE_STATE_LAYOUT_FULL,
        CACHE_STATE_LAYOUT_TARGET_ONLY,
    }:
        raise ValueError(f"unsupported association-cache state_layout: {state_layout!r}")
    edge_arrays = (
        arrays["target_signature_ids"], arrays["source_signature_ids"],
        arrays["base_scores"], arrays["thresholds"], arrays["past_windows"],
        arrays["future_windows"], arrays["source_order"],
    )
    if edge_count < 0 or any(len(array) != edge_count for array in edge_arrays):
        raise ValueError("association-cache edge array lengths do not match metadata")
    if (
        signature_count < 0
        or source_key_count < 0
        or len(arrays["target_offsets"]) != signature_count + 1
        or len(arrays["source_offsets"]) != source_key_count + 1
    ):
        raise ValueError("association-cache CSR offsets do not match key counts")
    for name in (
        "target_signature_ids", "source_signature_ids", "target_offsets",
        "source_offsets", "source_order",
    ):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"association-cache {name} must use an integer dtype")
    for name in ("target_offsets", "source_offsets"):
        offsets = arrays[name]
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != edge_count
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(f"association-cache {name} is invalid")
    if edge_count:
        if (
            int(arrays["target_signature_ids"].max()) >= signature_count
            or int(arrays["source_signature_ids"].max()) >= source_key_count
            or int(arrays["source_order"].max()) >= edge_count
        ):
            raise ValueError("association-cache edge index is out of range")
    return {**header, "arrays": arrays}


def write_association_cache(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = f"{path}.tmp"
    try:
        arrays = dict(payload.get("arrays") or {})
        header = {key: value for key, value in payload.items() if key != "arrays"}
        with open(temp_path, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(
                    json.dumps(header, ensure_ascii=False, separators=(",", ":"))
                ),
                **arrays,
            )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@dataclass(frozen=True, slots=True)
class PeriodType:
    """Runtime event type: feature entity (normally NE id) + alarm type."""

    entity: str
    alarm_type: str


@dataclass(frozen=True, slots=True)
class PeriodSignature:
    period_type: PeriodType
    initial_state: int


def graph_period_universe(artifact, scorer):
    """Graph entities and model alarm types underlying the inductive universe."""
    rt = (artifact.training_metadata or {}).get("feature_runtime") or {}
    alarm_types = tuple(
        sorted({str(value) for value in (rt.get("at_vocab") or []) if str(value)})
    )
    if not alarm_types:
        raise ValueError("feature artifact has an empty training.feature_runtime.at_vocab")

    type_fields = tuple(artifact.config.type_fields)
    node_field = artifact.config.topology_node_field
    uses_domain = DEVICE_DOMAIN_FIELD in type_fields
    entities = []
    for node in sorted(str(value) for value in scorer.node_infos):
        if not uses_domain:
            entities.append(node)
            continue
        if node_field == "site_id":
            domains = sorted({str(value) for value in scorer.node_domains.get(node, ()) if str(value)})
        else:
            domain = str(domain_of(node, scorer.node_infos) or "")
            domains = [domain] if domain else []
        entities.extend(make_entity(node, domain) for domain in domains)

    return tuple(entities), alarm_types


def graph_period_types(artifact, scorer):
    """Full inductive universe: graph entities × model alarm-type vocabulary."""
    entities, alarm_types = graph_period_universe(artifact, scorer)
    period_types = [
        PeriodType(entity, alarm_type)
        for entity in entities
        for alarm_type in alarm_types
    ]
    return period_types, len(entities), len(alarm_types)


def build_compact_csr_arrays(
    target_signature_ids,
    source_signature_ids,
    base_scores,
    thresholds,
    past_windows,
    future_windows,
    signature_count,
    source_key_count=None,
):
    """Build compact forward/reverse CSR arrays from target-sorted edge rows."""
    source_key_count = int(
        signature_count if source_key_count is None else source_key_count
    )
    edge_count = len(target_signature_ids)
    id_dtype = np.uint32 if signature_count <= np.iinfo(np.uint32).max else np.uint64
    order_dtype = np.uint32 if edge_count <= np.iinfo(np.uint32).max else np.uint64
    target_ids = np.asarray(target_signature_ids, dtype=id_dtype)
    source_ids = np.asarray(source_signature_ids, dtype=id_dtype)
    if edge_count and np.any(target_ids[1:] < target_ids[:-1]):
        raise ValueError("compact cache edges must be sorted by target signature")
    target_counts = np.bincount(target_ids.astype(np.int64), minlength=signature_count)
    source_counts = np.bincount(source_ids.astype(np.int64), minlength=source_key_count)
    target_offsets = np.empty(signature_count + 1, dtype=np.uint64)
    source_offsets = np.empty(source_key_count + 1, dtype=np.uint64)
    target_offsets[0] = 0
    source_offsets[0] = 0
    np.cumsum(target_counts, dtype=np.uint64, out=target_offsets[1:])
    np.cumsum(source_counts, dtype=np.uint64, out=source_offsets[1:])
    source_order = np.argsort(source_ids, kind="stable").astype(order_dtype, copy=False)
    return {
        "target_signature_ids": target_ids,
        "source_signature_ids": source_ids,
        "base_scores": np.asarray(base_scores, dtype=np.float64),
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "past_windows": np.asarray(past_windows, dtype=np.float64),
        "future_windows": np.asarray(future_windows, dtype=np.float64),
        "target_offsets": target_offsets,
        "source_offsets": source_offsets,
        "source_order": source_order,
    }


@dataclass
class AlarmPeriod:
    period_id: int
    period_type: PeriodType
    initial_state: tuple
    initial_state_combo: int
    first_ts: float
    last_raise_ts: float
    events: list[OnlineEvent] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    status: str = "open"
    close_ts: Optional[float] = None
    close_reason: Optional[str] = None
    idle_generation: int = 0
    pending_generation: int = 0
    pending_ready_ts: Optional[float] = None
    harvested_version: int = 0
    primary_group_id: Optional[int] = None

    @property
    def signature(self) -> PeriodSignature:
        return PeriodSignature(self.period_type, self.initial_state_combo)

    @property
    def version(self) -> int:
        return len(self.events)

    @property
    def is_dirty(self) -> bool:
        return self.version > self.harvested_version

    def append(self, event: OnlineEvent):
        if self.status != "open":
            raise ValueError(f"cannot append to closed AlarmPeriod {self.period_id}")
        self.events.append(event)
        self.timestamps.append(float(event.ts))
        self.last_raise_ts = max(self.last_raise_ts, float(event.ts))
        self.idle_generation += 1

    def close(self, ts: float, reason: str):
        if self.status != "open":
            return False
        self.status = "closed"
        self.close_ts = float(ts)
        self.close_reason = str(reason)
        self.idle_generation += 1
        return True


@dataclass(frozen=True, slots=True)
class CompiledEdge:
    base_score: float
    threshold: float
    past_window_sec: float
    future_window_sec: float


class CompactAssociationIndex:
    """Read-only bidirectional CSR index over precompiled numeric edges."""

    def __init__(self, period_types, arrays, state_layout=CACHE_STATE_LAYOUT_FULL):
        self.period_types = tuple(period_types)
        self.type_to_id = {value: index for index, value in enumerate(self.period_types)}
        self.state_layout = str(state_layout)
        if self.state_layout not in {
            CACHE_STATE_LAYOUT_FULL,
            CACHE_STATE_LAYOUT_TARGET_ONLY,
        }:
            raise ValueError(f"unsupported compact cache state_layout={self.state_layout!r}")
        self.target_signature_ids = np.asarray(arrays["target_signature_ids"])
        self.source_signature_ids = np.asarray(arrays["source_signature_ids"])
        self.base_scores = np.asarray(arrays["base_scores"], dtype=np.float64)
        self.thresholds = np.asarray(arrays["thresholds"], dtype=np.float64)
        self.past_windows = np.asarray(arrays["past_windows"], dtype=np.float64)
        self.future_windows = np.asarray(arrays["future_windows"], dtype=np.float64)
        self.target_offsets = np.asarray(arrays["target_offsets"])
        self.source_offsets = np.asarray(arrays["source_offsets"])
        self.source_order = np.asarray(arrays["source_order"])
        self.memory_bytes = sum(
            array.nbytes
            for array in (
                self.target_signature_ids, self.source_signature_ids,
                self.base_scores, self.thresholds, self.past_windows,
                self.future_windows, self.target_offsets, self.source_offsets,
                self.source_order,
            )
        )

    def _target_signature_id(self, signature):
        type_id = self.type_to_id.get(signature.period_type)
        if type_id is None:
            return None
        return int(type_id) * 8 + int(signature.initial_state)

    def _source_key_id(self, signature):
        type_id = self.type_to_id.get(signature.period_type)
        if type_id is None:
            return None
        if self.state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY:
            return int(type_id)
        return int(type_id) * 8 + int(signature.initial_state)

    def _signature(self, signature_id):
        type_id, state = divmod(int(signature_id), 8)
        return PeriodSignature(self.period_types[type_id], state)

    def _edge(self, index):
        return CompiledEdge(
            base_score=float(self.base_scores[index]),
            threshold=float(self.thresholds[index]),
            past_window_sec=float(self.past_windows[index]),
            future_window_sec=float(self.future_windows[index]),
        )

    def iter_target(self, target):
        signature_id = self._target_signature_id(target)
        if signature_id is None:
            return
        start = int(self.target_offsets[signature_id])
        end = int(self.target_offsets[signature_id + 1])
        for index in range(start, end):
            source_id = int(self.source_signature_ids[index])
            source = (
                self.period_types[source_id]
                if self.state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
                else self._signature(source_id)
            )
            yield source, self._edge(index)

    def iter_source(self, source):
        source_key_id = self._source_key_id(source)
        if source_key_id is None:
            return
        start = int(self.source_offsets[source_key_id])
        end = int(self.source_offsets[source_key_id + 1])
        for position in range(start, end):
            index = int(self.source_order[position])
            yield self._signature(self.target_signature_ids[index]), self._edge(index)


@dataclass
class RelationEvidence:
    target_period_id: int
    source_period_id: int
    target_event: OnlineEvent
    source_event: OnlineEvent
    score: float
    strength: float
    edge: CompiledEdge

    @property
    def period_pair(self):
        return tuple(sorted((self.target_period_id, self.source_period_id)))


@dataclass
class MergeProposal:
    group_ids: tuple
    evidence_pairs: set = field(default_factory=set)
    max_strength: float = 0.0
    max_score: float = 0.0


@dataclass
class PeriodFaultGroup:
    group_id: int
    anchor_period_id: int
    period_ids: set = field(default_factory=set)
    core_period_ids: list = field(default_factory=list)
    evidence_by_pair: dict = field(default_factory=dict)
    start_ts: float = math.inf
    last_ts: float = -math.inf


def _state_combo(mark) -> int:
    mark = tuple(mark or (0, 0, 0))
    return (
        (int(mark[0]) if len(mark) > 0 else 0)
        + 2 * (int(mark[1]) if len(mark) > 1 else 0)
        + 4 * (int(mark[2]) if len(mark) > 2 else 0)
    )


def _combo_state(combo: int) -> tuple:
    combo = int(combo)
    return (combo & 1, (combo >> 1) & 1, (combo >> 2) & 1)


class CompiledAssociationPlan:
    """Lazy materialization of MHP period-signature edges.

    The first period of a new signature compiles its edges against already
    observed signatures.  Later periods reuse the same edge and horizon tables.
    ``related`` scope materializes only same-entity, same-site, or topology-hop
    pairs; ``unrelated`` uses a validated indexable candidate policy restricted
    to pairs the related predicate does *not* cover; ``global`` evaluates every
    observed signature pair.
    """

    def __init__(
        self,
        scorer,
        mu_scorer,
        artifact,
        config: PeriodStreamConfig,
        candidate_policy=None,
    ):
        self.scorer = scorer
        self.decomposed = DecomposedFeatureScorer(scorer)
        self.mu_scorer = mu_scorer
        self.artifact = artifact
        self.config = config
        self.candidate_policy = candidate_policy
        if config.candidate_scope == "unrelated" and candidate_policy is None:
            raise ValueError("unrelated candidate scope requires a candidate policy")
        self.dynamic_mode = str(getattr(artifact.config, "dynamic_alpha", "off"))
        self.cache_state_layout = association_cache_state_layout(self.dynamic_mode)
        rt = (artifact.training_metadata or {}).get("feature_runtime") or {}
        self.beta = float(rt.get("beta", scorer.beta))
        if self.beta <= 0:
            raise ValueError("feature beta must be > 0")
        self.late_lambda = math.log(2.0) / (
            config.late_penalty_half_life_sec / config.time_scale_sec
        )
        self.mu_by_at = rt.get("mu_by_alarm_type", {}) or {}
        self.mu_default = float(rt.get("mu_default", 0.0))
        # ``signatures`` contains only signatures discovered incrementally at
        # runtime.  Offline coverage is type-level because all eight states are
        # guaranteed compiled and materializing 8× coverage objects is wasteful.
        self.signatures: set[PeriodSignature] = set()
        self.covered_period_types: set[PeriodType] = set()
        self._covered_candidate_index = None
        # One index per loaded association cache; runtime unions them (e.g. a
        # ``related`` cache plus a disjoint ``unrelated`` cache).
        self.precompiled_indexes: list[CompactAssociationIndex] = []
        self.edges_by_target: dict[PeriodSignature, dict[PeriodSignature, CompiledEdge]] = defaultdict(dict)
        self.edges_by_source: dict[PeriodSignature, dict[PeriodSignature, CompiledEdge]] = defaultdict(dict)
        self._mu_cache: dict[PeriodType, float] = {}
        self.compiled_pair_count = 0
        self.pruned_pair_count = 0
        # Offline-compile telemetry: directed type pairs (not ×8 states)
        # rejected by the separable-logit prescreen before exact scoring.
        self.prescreen_dropped_pair_count = 0
        self.preloaded_signature_count = 0
        self.preloaded_edge_count = 0
        # Scope whose candidate pairs are completely covered by the preloaded
        # or in-memory compiled universe. A related cache may be reused by a
        # global runtime as an authoritative base for related pairs only.
        self.precompiled_candidate_scope = None

    @property
    def precompiled_index(self):
        """First loaded index, or ``None`` — back-compat for single-cache uses."""
        return self.precompiled_indexes[0] if self.precompiled_indexes else None

    def iter_edges_by_target(self, signature):
        for index in self.precompiled_indexes:
            yield from index.iter_target(signature)
        yield from self.edges_by_target.get(signature, {}).items()

    def iter_edges_by_source(self, signature):
        for index in self.precompiled_indexes:
            yield from index.iter_source(signature)
        yield from self.edges_by_source.get(signature, {}).items()

    def _mu(self, period_type: PeriodType) -> float:
        cached = self._mu_cache.get(period_type)
        if cached is not None:
            return cached
        if self.mu_scorer is not None:
            value = float(self.mu_scorer.mu_for(period_type.alarm_type, period_type.entity))
        else:
            value = float(self.mu_by_at.get(period_type.alarm_type, self.mu_default))
        value *= self.config.immigrant_bias * self.config.attach_threshold_ratio
        value = max(value, EPS)
        self._mu_cache[period_type] = value
        return value

    def _is_related_pair(self, a: PeriodSignature, b: PeriodSignature) -> bool:
        """Deterministic related-scope predicate, independent of runtime scope."""
        ae = a.period_type.entity
        be = b.period_type.entity
        if ae == be:
            return True
        an = topo_node_of(ae)
        bn = topo_node_of(be)
        if not an or not bn:
            return False
        if an == bn:
            return True
        infos = self.scorer.node_infos
        ai = infos.get(an)
        bi = infos.get(bn)
        a_site = str(getattr(ai, "site_id", "") or "")
        b_site = str(getattr(bi, "site_id", "") or "")
        if a_site and a_site == b_site:
            return True
        topo = self.scorer.topology_index
        hops = (getattr(topo, "undirected_hops", {}) or {}) if topo is not None else {}
        return bool(hops.get(an, {}).get(bn, 0) or hops.get(bn, {}).get(an, 0))

    def _related(self, a: PeriodSignature, b: PeriodSignature) -> bool:
        if self.config.candidate_scope == "global":
            return True
        if self.config.candidate_scope == "unrelated":
            return unrelated_pair_allowed(
                self.candidate_policy,
                a.period_type,
                b.period_type,
                self.scorer,
            )
        return self._is_related_pair(a, b)

    def _precompiled_pair_covered(self, a, b) -> bool:
        """Whether the loaded/compiled base has an authoritative result for a pair."""
        if (
            self.precompiled_candidate_scope is None
            or a.period_type not in self.covered_period_types
            or b.period_type not in self.covered_period_types
        ):
            return False
        if self.precompiled_candidate_scope == "global":
            return True
        if self.precompiled_candidate_scope == "unrelated":
            return unrelated_pair_allowed(
                self.candidate_policy,
                a.period_type,
                b.period_type,
                self.scorer,
            )
        return self._is_related_pair(a, b)

    def register_signature(self, signature: PeriodSignature):
        if signature in self.signatures:
            return
        covered_type = signature.period_type in self.covered_period_types
        partial_related_base = (
            covered_type
            and self.precompiled_candidate_scope == "related"
            and self.config.candidate_scope == "global"
        )
        if covered_type and not partial_related_base:
            return
        existing = list(self.signatures)
        self.signatures.add(signature)
        for other in existing:
            if not self._related(signature, other):
                continue
            # An uncovered signature is compiled against every candidate state
            # in the base universe below; avoid scoring an already-observed
            # covered state twice on this path.
            if (
                not covered_type
                and self._covered_candidate_index is not None
                and other.period_type in self.covered_period_types
            ):
                continue
            if self._precompiled_pair_covered(signature, other):
                continue
            self._compile(signature, other)
            if signature != other:
                self._compile(other, signature)
        # A known signature from a related base only needs delta edges against
        # other observed signatures. Unseen types retain the legacy eager
        # behavior against the full precompiled universe.
        if not covered_type and self._covered_candidate_index is not None:
            for other_type in self._candidate_sources(
                signature.period_type, self._covered_candidate_index
            ):
                for state in range(8):
                    other = PeriodSignature(other_type, state)
                    if self._precompiled_pair_covered(signature, other):
                        continue
                    self._compile(signature, other)
                    self._compile(other, signature)
        if not self._precompiled_pair_covered(signature, signature):
            self._compile(signature, signature)

    def _compile(self, target: PeriodSignature, source: PeriodSignature):
        if source in self.edges_by_target.get(target, {}):
            return
        edge = self._compute_edge(target, source)
        if edge is None:
            return
        self.edges_by_target[target][source] = edge
        self.edges_by_source[source][target] = edge

    def _compute_edge(self, target: PeriodSignature, source: PeriodSignature):
        t = target.period_type
        s = source.period_type
        src_marks = np.asarray([_combo_state(source.initial_state)], dtype=np.float64)
        tgt_marks = np.asarray([_combo_state(target.initial_state)], dtype=np.float64)
        alpha = float(
            self.decomposed.alpha_for_target(
                t.alarm_type,
                t.entity,
                [s.alarm_type],
                [s.entity],
                src_marks=src_marks if self.scorer.source_dynamic_dim else None,
                tgt_marks=tgt_marks if self.scorer.target_dynamic_dim else None,
            )[0]
        )
        if alpha < self.config.feature_alpha_floor:
            self.pruned_pair_count += 1
            return
        relation_weight = 1.0
        if self.config.topology_relation_prior:
            relation_weight = float(
                topology_relation_weights(
                    [topo_node_of(s.entity)],
                    topo_node_of(t.entity),
                    self.scorer.topology_index,
                    self.scorer.node_infos,
                    self.config.topology_relation_prior,
                )[0]
            )
        base_score = alpha * self.beta * relation_weight
        threshold = self._mu(t)
        if base_score + EPS < threshold or base_score <= 0:
            self.pruned_pair_count += 1
            return
        log_margin = max(0.0, math.log(base_score / threshold))
        past_window = min(
            self.config.history_window_sec,
            log_margin / self.beta * self.config.time_scale_sec,
        )
        future_window = 0.0
        if self.config.time_slack_sec > 0:
            future_window = min(
                self.config.time_slack_sec,
                log_margin / self.late_lambda * self.config.time_scale_sec,
            )
        edge = CompiledEdge(
            base_score=base_score,
            threshold=threshold,
            past_window_sec=past_window,
            future_window_sec=future_window,
        )
        self.compiled_pair_count += 1
        return edge

    def prepare_candidate_period_types(self, period_types, count_pairs=True):
        """Build a reusable related-type index and return its exact pair count.

        The related-scope path indexes entity/node/site/topology reach first, so
        offline compilation does not perform a blind all-signature quadratic
        scan merely to reject unrelated pairs. Candidate sets are reconstructed
        per target instead of being retained, because the graph-wide universe
        can contain millions of directed pairs.
        """
        period_types = tuple(sorted(period_types, key=lambda x: (x.entity, x.alarm_type)))
        if self.config.candidate_scope == "unrelated":
            return prepare_adaptive_candidates(
                period_types,
                self.scorer,
                self.candidate_policy,
                count_pairs=count_pairs,
                exclude_related=True,
            )
        if self.config.candidate_scope == "global":
            return {
                "period_types": period_types,
                "global": True,
                "total_pair_count": (
                    len(period_types) * len(period_types) if count_pairs else None
                ),
            }

        by_entity = defaultdict(set)
        by_node = defaultdict(set)
        by_site = defaultdict(set)
        infos = self.scorer.node_infos
        for period_type in period_types:
            by_entity[period_type.entity].add(period_type)
            node = topo_node_of(period_type.entity)
            if node:
                by_node[node].add(period_type)
                info = infos.get(node)
                site = str(getattr(info, "site_id", "") or "")
                if site:
                    by_site[site].add(period_type)

        topo = self.scorer.topology_index
        hops = (getattr(topo, "undirected_hops", {}) or {}) if topo is not None else {}
        neighbor_nodes = defaultdict(set)
        for left, row in hops.items():
            for right, distance in (row or {}).items():
                if distance:
                    neighbor_nodes[left].add(right)
                    neighbor_nodes[right].add(left)

        prepared = {
            "period_types": period_types,
            "global": False,
            "by_entity": by_entity,
            "by_node": by_node,
            "by_site": by_site,
            "neighbor_nodes": neighbor_nodes,
        }
        total_pair_count = None
        if count_pairs:
            total_pair_count = 0
            for target in period_types:
                total_pair_count += len(self._candidate_sources(target, prepared))
        prepared["total_pair_count"] = total_pair_count
        return prepared

    def _candidate_sources(self, target, prepared):
        period_types = prepared["period_types"]
        if prepared.get("adaptive"):
            return adaptive_candidate_sources(
                target,
                prepared["policy"],
                prepared,
                exclude_related=prepared.get("exclude_related", False),
            )
        if prepared["global"]:
            return period_types
        by_entity = prepared["by_entity"]
        by_node = prepared["by_node"]
        by_site = prepared["by_site"]
        neighbor_nodes = prepared["neighbor_nodes"]
        candidates = set(by_entity.get(target.entity, ()))
        node = topo_node_of(target.entity)
        if node:
            candidates.update(by_node.get(node, ()))
            info = self.scorer.node_infos.get(node)
            site = str(getattr(info, "site_id", "") or "")
            if site:
                candidates.update(by_site.get(site, ()))
            for neighbor in neighbor_nodes.get(node, ()):
                candidates.update(by_node.get(neighbor, ()))
        return tuple(sorted(candidates, key=lambda x: (x.entity, x.alarm_type)))

    def _candidate_period_type_pairs(self, prepared):
        for target in prepared["period_types"]:
            for source in self._candidate_sources(target, prepared):
                yield target, source

    def _candidate_arrays(self, source_types):
        """Candidate list indexes for the separable prescreen.

        Returns alarm-type ids, the unique-entity inverse map, the unique
        entity list, an (at_id+1, entity_index) -> candidate position grid
        (-1 where the combo is absent), and the distinct at ids present.
        """
        at_to_id = self.scorer.at_to_id
        src_at_ids = np.fromiter(
            (at_to_id.get(str(value.alarm_type), -1) for value in source_types),
            dtype=np.int64,
            count=len(source_types),
        )
        index_of = {}
        ent_inverse = np.empty(len(source_types), dtype=np.int64)
        unique_entities = []
        for i, value in enumerate(source_types):
            j = index_of.get(value.entity)
            if j is None:
                j = index_of[value.entity] = len(unique_entities)
                unique_entities.append(value.entity)
            ent_inverse[i] = j
        grid_index = np.full(
            (self.decomposed.W_at_pad.shape[0], len(unique_entities)),
            -1,
            dtype=np.int64,
        )
        grid_index[src_at_ids + 1, ent_inverse] = np.arange(
            len(source_types), dtype=np.int64
        )
        distinct_at_ids = np.unique(src_at_ids)
        return src_at_ids, ent_inverse, unique_entities, grid_index, distinct_at_ids

    def _prescreen_source_state(self, entity, source_types, shared, static_table):
        """Vectorized prescreen tables for one target entity's candidate set.

        Returns ``None`` for an empty candidate set. ``shared``/``static_table``
        let the ``global`` scope reuse its one fixed source universe; ``related``
        and ``adaptive`` rebuild the tables per candidate set (``adaptive``'s set
        depends on the target alarm type, so it is rebuilt per target row).
        """
        if not source_types:
            return None
        d = self.decomposed
        src_at_ids, ent_inverse, unique_entities, grid_index, distinct_at_ids = (
            shared if shared is not None else self._candidate_arrays(source_types)
        )
        parts = (
            d.entity_parts_from_table(entity, static_table)
            if static_table is not None
            else d.entity_parts_for_target(entity, unique_entities)
        )
        # z_entity: the exact logit with the at-pair block muted (id -1 hits the
        # zero-padded W_at row and disables same_at).
        z_entity = d.logits_from_parts(
            -1,
            np.full(len(unique_entities), -1, dtype=np.int64),
            src_mark_idx=np.zeros(len(unique_entities), dtype=np.int64),
            tgt_term=0.0,
            **parts,
        )
        # delta_same: contribution of same_alarm_type and its cross columns on
        # any v == u row.
        z_same = z_entity + d.same_at_delta_from_parts(parts)
        oov_source_indices = np.flatnonzero(src_at_ids == -1)
        return (
            src_at_ids,
            ent_inverse,
            grid_index,
            distinct_at_ids,
            parts,
            z_entity,
            z_same,
            oov_source_indices,
        )

    def _precompile_target_only_batches(self, prepared, progress, edge_batch_sink):
        """Vectorized offline compiler for target-dynamic cache rows.

        Targets are grouped by entity so the expensive entity-pair feature
        parts are computed once per (target entity, source entity) instead of
        once per candidate. Fixing the target, the logit is additively
        separable: z(v, f) = z_entity(f) + W_at[u, v], plus a same-alarm-type
        correction on the v == u row (same_alarm_type and its cross columns
        depend only on the source entity once v == u holds). A conservative
        threshold prescreen on that separable form finds survivors with direct
        vector masks over the small alarm-type vocabulary (link/power/offline),
        avoiding two entity-array sorts per target entity. Survivors are
        rescored through the exact per-candidate path, so emitted edges are
        bit-identical to the unpruned scan.
        """
        type_pair_count = 0
        d = self.decomposed
        target_terms = np.asarray(
            [d.tgt_term(_combo_state(state)) for state in range(8)],
            dtype=np.float64,
        )
        max_tgt_term = float(target_terms.max())
        rel_max = 1.0
        if self.config.topology_relation_prior:
            rel_max = max(
                relation_weight(self.config.topology_relation_prior, key)
                for key in RELATION_KEYS
            )
        shared = (
            self._candidate_arrays(prepared["period_types"])
            if prepared["global"]
            else None
        )
        # Global scope scores one fixed entity universe against every target:
        # precompute its target-independent columns once and assemble parts per
        # group vectorized. Related scope keeps the per-group loop — its
        # candidate sets are small and differ per target entity.
        static_table = (
            d.entity_static_table(shared[2]) if shared is not None else None
        )
        # ``adaptive`` candidate sets depend on the target alarm type, so the
        # prescreen tables cannot be shared across a target entity's rows the
        # way ``related``/``global`` sets can; rebuild them per target instead.
        per_target_sources = bool(prepared.get("adaptive"))
        for entity, group in groupby(
            prepared["period_types"], key=lambda value: value.entity
        ):
            targets = tuple(group)
            if not per_target_sources:
                group_source_types = self._candidate_sources(targets[0], prepared)
                group_state = self._prescreen_source_state(
                    entity, group_source_types, shared, static_table
                )
            else:
                # Adaptive rebuilds per target alarm type, but target types
                # whose policy rules resolve to the same candidate set still
                # reuse the prescreen tables — keyed by that resolved set.
                state_by_sources = {}
            for target_type in targets:
                if per_target_sources:
                    source_types = self._candidate_sources(target_type, prepared)
                    if source_types not in state_by_sources:
                        state_by_sources[source_types] = self._prescreen_source_state(
                            entity, source_types, shared, static_table
                        )
                    state = state_by_sources[source_types]
                else:
                    source_types = group_source_types
                    state = group_state
                source_count = len(source_types)
                kept_count = 0
                if source_count:
                    (
                        src_at_ids,
                        ent_inverse,
                        grid_index,
                        distinct_at_ids,
                        parts,
                        z_entity,
                        z_same,
                        oov_source_indices,
                    ) = state
                    threshold = self._mu(target_type)
                    denom = self.beta * rel_max
                    alpha_required = (
                        math.inf
                        if denom <= 0
                        else max(
                            self.config.feature_alpha_floor,
                            (threshold - EPS) / denom,
                        )
                    )
                    z_required = (
                        _softplus_lower_logit(alpha_required / d.alpha_scale)
                        - max_tgt_term
                        - PRESCREEN_LOGIT_MARGIN
                        if d.alpha_scale > 0
                        else -math.inf
                    )
                    target_at_id = self.scorer.at_to_id.get(
                        str(target_type.alarm_type), -1
                    )
                    if z_required == -math.inf:
                        survivors = np.arange(source_count, dtype=np.int64)
                    else:
                        # Keep (v, f) iff z_entity(f) >= z_required - W_at[u, v]
                        # (or the delta_same-shifted variant on the v == u
                        # row). With at most link/power/offline in the modeled
                        # vocabulary, direct vector masks are cheaper than two
                        # O(E log E) sorts per target entity. One candidate mask
                        # preserves source order without sorting the survivors.
                        w_row = d.W_at_pad[target_at_id + 1]
                        survivor_mask = np.zeros(source_count, dtype=np.bool_)
                        for v in distinct_at_ids:
                            v = int(v)
                            if v == -1:
                                # Every OOV alarm type has zero W_at and never
                                # activates same_alarm_type. Keep candidate
                                # positions directly so multiple OOV types on
                                # one entity do not collide in grid row zero.
                                keep_oov = (
                                    z_entity[ent_inverse[oov_source_indices]]
                                    >= z_required
                                )
                                if np.any(keep_oov):
                                    survivor_mask[oov_source_indices[keep_oov]] = True
                                continue
                            entity_logits = (
                                z_same if v == target_at_id else z_entity
                            )
                            rows = np.flatnonzero(
                                entity_logits >= z_required - w_row[v + 1]
                            )
                            if len(rows):
                                candidates = grid_index[v + 1, rows]
                                candidates = candidates[candidates >= 0]
                            else:
                                candidates = ()
                            if len(candidates):
                                survivor_mask[candidates] = True
                        survivors = np.flatnonzero(survivor_mask)
                    self.prescreen_dropped_pair_count += source_count - len(survivors)
                    if len(survivors):
                        surv_types = [source_types[i] for i in survivors]
                        surv_inverse = ent_inverse[survivors]
                        # Exact rescoring reuses the group's entity parts by
                        # gather; logits_from_parts is elementwise, so this is
                        # bit-identical to rebuilding features per candidate.
                        logits = d.logits_from_parts(
                            target_at_id,
                            src_at_ids[survivors],
                            src_mark_idx=np.zeros(len(survivors), dtype=np.int64),
                            tgt_term=0.0,
                            **{
                                key: value[surv_inverse]
                                if isinstance(value, np.ndarray)
                                else value
                                for key, value in parts.items()
                            },
                        )
                        alpha = d._softplus(
                            logits.reshape(1, -1) + target_terms.reshape(-1, 1)
                        )
                        if d.alpha_scale != 1.0:
                            alpha = alpha * d.alpha_scale
                        if self.config.topology_relation_prior:
                            relation_weights = topology_relation_weights(
                                [topo_node_of(value.entity) for value in surv_types],
                                topo_node_of(target_type.entity),
                                self.scorer.topology_index,
                                self.scorer.node_infos,
                                self.config.topology_relation_prior,
                            )
                        else:
                            relation_weights = np.ones(len(surv_types), dtype=np.float64)
                        base_scores = alpha * self.beta * relation_weights.reshape(1, -1)
                        keep = ~(
                            (alpha < self.config.feature_alpha_floor)
                            | (base_scores + EPS < threshold)
                            | (base_scores <= 0)
                        )
                        kept_count = int(np.count_nonzero(keep))
                        if kept_count:
                            target_states, source_indices = np.nonzero(keep)
                            kept_scores = base_scores[target_states, source_indices]
                            log_margins = np.maximum(
                                0.0, np.log(kept_scores / threshold)
                            )
                            past_windows = np.minimum(
                                self.config.history_window_sec,
                                log_margins / self.beta * self.config.time_scale_sec,
                            )
                            if self.config.time_slack_sec > 0:
                                future_windows = np.minimum(
                                    self.config.time_slack_sec,
                                    log_margins / self.late_lambda * self.config.time_scale_sec,
                                )
                            else:
                                future_windows = np.zeros(kept_count, dtype=np.float64)
                            edge_batch_sink(
                                target_type,
                                target_states,
                                surv_types,
                                source_indices,
                                kept_scores,
                                threshold,
                                past_windows,
                                future_windows,
                            )
                    self.compiled_pair_count += kept_count
                    self.pruned_pair_count += source_count * 8 - kept_count
                type_pair_count += source_count
                if progress is not None:
                    progress(
                        type_pair_count, self.compiled_pair_count, self.pruned_pair_count
                    )
        return type_pair_count

    def precompile_period_types(
        self,
        period_types,
        progress=None,
        prepared_candidates=None,
        edge_sink=None,
        edge_batch_sink=None,
    ):
        """Compile all eight frozen-state signatures for known period types."""
        prepared = prepared_candidates or self.prepare_candidate_period_types(period_types)
        period_types = prepared["period_types"]
        if (
            edge_batch_sink is not None
            and self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
        ):
            return self._precompile_target_only_batches(
                prepared, progress, edge_batch_sink
            )
        states = tuple(range(8))
        source_states = (
            (0,)
            if edge_sink is not None
            and self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else states
        )
        if edge_sink is None:
            self.covered_period_types.update(period_types)
            self._covered_candidate_index = prepared
            self.precompiled_candidate_scope = self.config.candidate_scope

        type_pair_count = 0
        for target_type in prepared["period_types"]:
            source_types = self._candidate_sources(target_type, prepared)
            for target_state in states:
                target = PeriodSignature(target_type, target_state)
                for source_type in source_types:
                    for source_state in source_states:
                        source = PeriodSignature(source_type, source_state)
                        if edge_sink is None:
                            self._compile(target, source)
                        else:
                            edge = self._compute_edge(target, source)
                            if edge is not None:
                                edge_sink(target, source, edge)
            type_pair_count += len(source_types)
            if progress is not None:
                progress(type_pair_count, self.compiled_pair_count, self.pruned_pair_count)
        return type_pair_count

    def to_cache_payload(self, fingerprint, extra_metadata=None):
        period_types = tuple(
            sorted(self.covered_period_types, key=lambda x: (x.entity, x.alarm_type))
        )
        type_to_id = {value: index for index, value in enumerate(period_types)}
        targets = sorted(
            self.edges_by_target,
            key=lambda x: (x.period_type.entity, x.period_type.alarm_type, x.initial_state),
        )
        target_ids, source_ids = [], []
        base_scores, thresholds, past_windows, future_windows = [], [], [], []
        for target in targets:
            sources = self.edges_by_target.get(target, {})
            seen_source_types = set()
            for source, edge in sorted(
                sources.items(),
                key=lambda item: (
                    item[0].period_type.entity,
                    item[0].period_type.alarm_type,
                    item[0].initial_state,
                ),
            ):
                if self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY:
                    if source.period_type in seen_source_types:
                        continue
                    seen_source_types.add(source.period_type)
                target_ids.append(type_to_id[target.period_type] * 8 + target.initial_state)
                source_ids.append(
                    type_to_id[source.period_type]
                    if self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
                    else type_to_id[source.period_type] * 8 + source.initial_state
                )
                base_scores.append(edge.base_score)
                thresholds.append(edge.threshold)
                past_windows.append(edge.past_window_sec)
                future_windows.append(edge.future_window_sec)
        signature_count = len(period_types) * 8
        source_key_count = (
            len(period_types)
            if self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else signature_count
        )
        arrays = build_compact_csr_arrays(
            target_ids,
            source_ids,
            base_scores,
            thresholds,
            past_windows,
            future_windows,
            signature_count,
            source_key_count=source_key_count,
        )
        return {
            "format": ASSOCIATION_CACHE_FORMAT,
            "version": ASSOCIATION_CACHE_VERSION,
            "fingerprint": dict(fingerprint),
            "arrays": arrays,
            "metadata": {
                "type_universe": "graph",
                "period_type_count": len(period_types),
                "signature_count": signature_count,
                "source_key_count": source_key_count,
                "state_layout": self.cache_state_layout,
                "edge_count": len(target_ids),
                "pruned_pair_count": int(self.pruned_pair_count),
                **dict(extra_metadata or {}),
            },
        }

    def load_cache_payload(self, payload):
        # Coverage is reconstructed from the fingerprinted graph and model AT
        # vocabulary; the persistent payload remains positive edges only.
        metadata = payload.get("metadata") or {}
        if metadata.get("type_universe") != "graph":
            raise ValueError("association cache is not a graph-universe cache")
        period_types, _entity_count, _alarm_type_count = graph_period_types(
            self.artifact, self.scorer
        )
        period_types = tuple(period_types)
        covered = set(period_types)
        declared_period_type_count = int(metadata.get("period_type_count", -1))
        declared_signature_count = int(metadata.get("signature_count", -1))
        declared_source_key_count = int(metadata.get("source_key_count", -1))
        declared_state_layout = str(metadata.get("state_layout", ""))
        cached_plan_config = (payload.get("fingerprint") or {}).get(
            "plan_config"
        ) or {}
        cached_candidate_scope = str(
            cached_plan_config.get("candidate_scope", self.config.candidate_scope)
        )
        # Each cache is authoritative for whatever pairs it was compiled with;
        # heterogeneous scopes (e.g. a related cache plus a disjoint unrelated
        # cache) are co-loaded and unioned at lookup, so scope is validated as a
        # known value rather than pinned to a single runtime scope.
        if cached_candidate_scope not in {"related", "global", "unrelated"}:
            raise ValueError(
                "association-cache has invalid candidate_scope: "
                f"{cached_candidate_scope!r}"
            )
        expected_source_key_count = (
            len(covered)
            if self.cache_state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else len(covered) * 8
        )
        if (
            declared_period_type_count != len(covered)
            or declared_signature_count != len(covered) * 8
            or declared_source_key_count != expected_source_key_count
            or declared_state_layout != self.cache_state_layout
        ):
            raise ValueError(
                "association-cache coverage does not match graph universe: "
                f"cache_types={declared_period_type_count}, graph_types={len(covered)}, "
                f"cache_signatures={declared_signature_count}, "
                f"cache_source_keys={declared_source_key_count}, "
                f"cache_state_layout={declared_state_layout!r}, "
                f"expected_state_layout={self.cache_state_layout!r}"
            )
        self.covered_period_types.update(covered)
        if self._covered_candidate_index is None:
            self._covered_candidate_index = self.prepare_candidate_period_types(
                covered, count_pairs=False
            )
        self.precompiled_indexes.append(
            CompactAssociationIndex(
                period_types,
                payload["arrays"],
                state_layout=declared_state_layout,
            )
        )
        # A single loaded cache can serve as an incremental base; once several
        # heterogeneous caches are unioned there is no single base scope, and
        # full coverage makes register_signature a no-op anyway.
        self.precompiled_candidate_scope = (
            cached_candidate_scope if len(self.precompiled_indexes) == 1 else None
        )
        self.preloaded_signature_count = len(covered) * 8
        self.preloaded_edge_count += int(metadata["edge_count"])


class AlarmPeriodMHPAssigner:
    """Incremental AlarmPeriod grouping engine."""

    def __init__(
        self,
        artifact,
        config: PeriodStreamConfig,
        feature_scorer,
        mu_scorer=None,
        association_cache=None,
        candidate_policy=None,
        closed_group_sink=None,
    ):
        config.validate()
        if getattr(artifact.config, "edge_mode", "device") != "feature":
            raise ValueError("AlarmPeriod engine requires a feature-mode artifact")
        if getattr(artifact.params, "kernel_type", "exp") != "exp":
            raise ValueError("AlarmPeriod engine currently supports only the exponential kernel")
        self.artifact = artifact
        self.config = config
        self.feature_scorer = feature_scorer
        self.mu_scorer = mu_scorer
        self.plan = CompiledAssociationPlan(
            feature_scorer,
            mu_scorer,
            artifact,
            config,
            candidate_policy=candidate_policy,
        )
        if association_cache is not None:
            payloads = (
                association_cache
                if isinstance(association_cache, (list, tuple))
                else [association_cache]
            )
            for payload in payloads:
                self.plan.load_cache_payload(payload)
        self.state_tracker = DeviceStateTracker()
        self.periods: dict[int, AlarmPeriod] = {}
        self.open_period_by_type: dict[PeriodType, int] = {}
        self.period_ids_by_signature: dict[PeriodSignature, set] = defaultdict(set)
        self.period_ids_by_type: dict[PeriodType, set] = defaultdict(set)
        self.period_by_occurrence: dict[tuple, int] = {}
        self._idle_heap: list = []
        self._pending_heap: list = []
        self._heap_seq = 0
        self.groups: dict[int, PeriodFaultGroup] = {}
        self._group_redirect: dict[int, int] = {}
        self.merge_proposals: dict[tuple, MergeProposal] = {}
        self.closed_group_sink = closed_group_sink
        self.closed_group_count = 0
        self._next_event_index = 0
        self._next_period_id = 0
        self._next_group_id = 0
        self.current_watermark = -math.inf
        self.total_input_events = 0
        self.total_raise_events = 0
        self.total_clear_events = 0
        self.dropped_no_type = 0
        self.created_periods = 0
        self.idle_closed_periods = 0
        self.clear_closed_periods = 0
        self.harvest_count = 0
        self.relation_count = 0
        self.period_attach_count = 0
        self.group_merge_count = 0

    # ---- ingest and period lifecycle ---------------------------------

    def process(self, alarm_event: dict):
        self.total_input_events += 1
        ts = float(alarm_event.get("ts", 0.0))
        self._close_idle_periods(ts)

        alarm_payload = alarm_event.get("alarm", {}) if isinstance(alarm_event, dict) else {}
        clear = is_clear_alarm(alarm_payload)
        entity, parsed_at = runtime_ne_at(
            alarm_event,
            self.artifact.config.type_fields,
            self.artifact.config.topology_node_field,
        )
        fallback_at = alarm_type_label(alarm_event)
        alarm_type = parsed_at or fallback_at
        state_at = alarm_type_from_title(alarm_event.get("alarm_title", ""))
        snapshot = self.state_tracker.snapshot_then_apply(entity, state_at, clear)
        frozen_mark = (int(snapshot[0]), int(snapshot[1]), int(snapshot[2]))

        if not alarm_type:
            self.dropped_no_type += 1
            self._advance_watermark(ts)
            return None

        period_type = PeriodType(str(entity), str(alarm_type))
        if clear:
            self.total_clear_events += 1
            self._handle_clear(alarm_event, period_type, ts)
            self._advance_watermark(ts)
            return None

        type_label = event_type_label(alarm_event, self.artifact.config.type_fields)
        type_id = self.artifact.vocabs.type_vocab.get(type_label)
        event = OnlineEvent(
            index=self._next_event_index,
            ts=ts,
            type_id=-1 if type_id is None else int(type_id),
            type_label=type_label,
            alarm=alarm_event,
            alarm_type=str(alarm_type),
            ne=str(entity),
            src_mark=frozen_mark,
        )
        self._next_event_index += 1
        self.total_raise_events += 1

        period = self._open_or_create_period(period_type, event, frozen_mark)
        self._remember_occurrence(alarm_event, period.period_id)
        if period.primary_group_id is not None:
            group = self._group(period.primary_group_id)
            if group is not None:
                group.last_ts = max(group.last_ts, ts)
        self._schedule_idle(period)
        self._schedule_harvest(period, ts)
        self._advance_watermark(ts)
        return period

    def _open_or_create_period(self, period_type, event, frozen_mark):
        pid = self.open_period_by_type.get(period_type)
        period = self.periods.get(pid) if pid is not None else None
        if period is None or period.status != "open":
            period = AlarmPeriod(
                period_id=self._next_period_id,
                period_type=period_type,
                initial_state=tuple(frozen_mark),
                initial_state_combo=_state_combo(frozen_mark),
                first_ts=float(event.ts),
                last_raise_ts=float(event.ts),
            )
            self._next_period_id += 1
            self.created_periods += 1
            self.periods[period.period_id] = period
            self.open_period_by_type[period_type] = period.period_id
            self.period_ids_by_signature[period.signature].add(period.period_id)
            self.period_ids_by_type[period.period_type].add(period.period_id)
            self.plan.register_signature(period.signature)
        period.append(event)
        return period

    def _identity_of(self, alarm_event):
        try:
            return require_alarm_identity(alarm_event)
        except ValueError:
            return None

    def _remember_occurrence(self, alarm_event, period_id):
        identity = self._identity_of(alarm_event)
        if identity is not None:
            self.period_by_occurrence[tuple(identity)] = int(period_id)

    def _handle_clear(self, alarm_event, period_type, ts):
        matched_period = None
        identity = self._identity_of(alarm_event)
        matched_by_identity = False
        if identity is not None:
            identity_key = tuple(identity)
            if identity_key in self.period_by_occurrence:
                matched_by_identity = True
                pid = self.period_by_occurrence[identity_key]
                matched_period = self.periods.get(pid)
        if matched_period is None and not matched_by_identity:
            pid = self.open_period_by_type.get(period_type)
            matched_period = self.periods.get(pid) if pid is not None else None
        if matched_period is None or matched_period.status != "open":
            return
        if matched_period.close(ts, "clear"):
            self.clear_closed_periods += 1
            if self.open_period_by_type.get(matched_period.period_type) == matched_period.period_id:
                self.open_period_by_type.pop(matched_period.period_type, None)

    def _schedule_idle(self, period: AlarmPeriod):
        deadline = period.last_raise_ts + self.config.period_idle_sec
        self._heap_seq += 1
        heapq.heappush(
            self._idle_heap,
            (deadline, self._heap_seq, period.period_id, period.idle_generation),
        )

    def _schedule_harvest(self, period: AlarmPeriod, occurrence_ts: float):
        if period.pending_ready_ts is not None:
            return
        period.pending_generation += 1
        period.pending_ready_ts = float(occurrence_ts) + self.config.aggregation_wait_sec
        self._heap_seq += 1
        heapq.heappush(
            self._pending_heap,
            (
                period.pending_ready_ts,
                self._heap_seq,
                period.period_id,
                period.pending_generation,
            ),
        )

    def _close_idle_periods(self, watermark: float):
        while self._idle_heap and self._idle_heap[0][0] <= watermark:
            deadline, _seq, pid, generation = heapq.heappop(self._idle_heap)
            period = self.periods.get(pid)
            if period is None or period.status != "open":
                continue
            if generation != period.idle_generation:
                continue
            if period.close(deadline, "idle"):
                self.idle_closed_periods += 1
                if self.open_period_by_type.get(period.period_type) == pid:
                    self.open_period_by_type.pop(period.period_type, None)

    def _advance_watermark(self, watermark: float):
        self.current_watermark = max(self.current_watermark, float(watermark))
        self._harvest_ready(self.current_watermark)
        self._close_inactive_groups(self.current_watermark)
        self._evict_expired_periods(self.current_watermark)

    # ---- incremental harvest -----------------------------------------

    def _harvest_ready(self, watermark: float):
        while self._pending_heap and self._pending_heap[0][0] <= watermark:
            _ready, _seq, pid, generation = heapq.heappop(self._pending_heap)
            period = self.periods.get(pid)
            if period is None or generation != period.pending_generation:
                continue
            period.pending_ready_ts = None
            if not period.is_dirty:
                continue
            self._harvest_period(period, watermark)

    def _harvest_period(self, period: AlarmPeriod, watermark: float):
        start = period.harvested_version
        # Every occurrence receives the configured fixed wait.  A pending item
        # is anchored by the first unharvested occurrence and intentionally is
        # not postponed by a storm; occurrences that arrived near its deadline
        # remain dirty and get the next coalesced pending item.
        mature_before = float(watermark) - self.config.aggregation_wait_sec + EPS
        mature_version = bisect.bisect_right(period.timestamps, mature_before)
        new_events = period.events[start:mature_version]
        if not new_events:
            if period.is_dirty:
                self._schedule_harvest(period, period.timestamps[start])
            return
        relations = self._collect_relations(period, new_events)
        self._apply_relations(period, relations)
        period.harvested_version = mature_version
        self.harvest_count += 1
        if period.is_dirty:
            self._schedule_harvest(period, period.timestamps[mature_version])

    def _collect_relations(self, period: AlarmPeriod, new_events: list[OnlineEvent]):
        best_by_directed_pair: dict[tuple, RelationEvidence] = {}
        sig = period.signature

        # Current period acts as target; only its newly mature times are probed.
        for source_key, edge in self.plan.iter_edges_by_target(sig):
            source_period_ids = (
                self.period_ids_by_type.get(source_key, ())
                if isinstance(source_key, PeriodType)
                else self.period_ids_by_signature.get(source_key, ())
            )
            for source_pid in tuple(source_period_ids):
                if source_pid == period.period_id:
                    continue
                source_period = self.periods.get(source_pid)
                if not self._candidate_period_ok(source_period):
                    continue
                ev = self._best_for_new_targets(edge, period, new_events, source_period)
                if ev is not None:
                    self._keep_best_relation(best_by_directed_pair, ev)

        # Current period acts as source; reverse index catches relationships to
        # older target periods without rescanning every historical occurrence.
        for target_sig, edge in self.plan.iter_edges_by_source(sig):
            for target_pid in tuple(self.period_ids_by_signature.get(target_sig, ())):
                if target_pid == period.period_id:
                    continue
                target_period = self.periods.get(target_pid)
                if not self._candidate_period_ok(target_period):
                    continue
                ev = self._best_for_new_sources(edge, target_period, period, new_events)
                if ev is not None:
                    self._keep_best_relation(best_by_directed_pair, ev)

        out = sorted(best_by_directed_pair.values(), key=lambda x: (-x.score, x.period_pair))
        self.relation_count += len(out)
        return out

    def _candidate_period_ok(self, period):
        if period is None or not period.events:
            return False
        if period.primary_group_id is not None and self._group(period.primary_group_id) is None:
            return False
        return True

    @staticmethod
    def _keep_best_relation(store, evidence):
        key = (evidence.target_period_id, evidence.source_period_id)
        old = store.get(key)
        if old is None or evidence.score > old.score:
            store[key] = evidence

    def _past_score(self, edge, dt_sec):
        return edge.base_score * math.exp(
            -self.plan.beta * (float(dt_sec) / self.config.time_scale_sec)
        )

    def _future_score(self, edge, late_sec):
        return edge.base_score * math.exp(
            -self.plan.late_lambda * (float(late_sec) / self.config.time_scale_sec)
        )

    def _best_for_new_targets(self, edge, target_period, new_targets, source_period):
        src_ts = source_period.timestamps
        best = None
        for target_event in new_targets:
            t = target_event.ts
            j = bisect.bisect_right(src_ts, t) - 1
            if j >= 0:
                dt = t - src_ts[j]
                if dt <= edge.past_window_sec + EPS:
                    score = self._past_score(edge, dt)
                    best = self._evidence_if_better(
                        best, edge, target_period, source_period,
                        target_event, source_period.events[j], score,
                    )
            j = bisect.bisect_right(src_ts, t)
            if j < len(src_ts):
                late = src_ts[j] - t
                if late <= edge.future_window_sec + EPS:
                    score = self._future_score(edge, late)
                    best = self._evidence_if_better(
                        best, edge, target_period, source_period,
                        target_event, source_period.events[j], score,
                    )
        return best

    def _best_for_new_sources(self, edge, target_period, source_period, new_sources):
        tgt_ts = target_period.timestamps
        best = None
        for source_event in new_sources:
            s = source_event.ts
            j = bisect.bisect_left(tgt_ts, s)
            if j < len(tgt_ts):
                dt = tgt_ts[j] - s
                if dt <= edge.past_window_sec + EPS:
                    score = self._past_score(edge, dt)
                    best = self._evidence_if_better(
                        best, edge, target_period, source_period,
                        target_period.events[j], source_event, score,
                    )
            j = bisect.bisect_left(tgt_ts, s) - 1
            if j >= 0:
                late = s - tgt_ts[j]
                if late <= edge.future_window_sec + EPS:
                    score = self._future_score(edge, late)
                    best = self._evidence_if_better(
                        best, edge, target_period, source_period,
                        target_period.events[j], source_event, score,
                    )
        return best

    @staticmethod
    def _evidence_if_better(best, edge, target_period, source_period,
                            target_event, source_event, score):
        if score + EPS < edge.threshold:
            return best
        evidence = RelationEvidence(
            target_period_id=target_period.period_id,
            source_period_id=source_period.period_id,
            target_event=target_event,
            source_event=source_event,
            score=float(score),
            strength=float(score / max(edge.threshold, EPS)),
            edge=edge,
        )
        if best is None or evidence.score > best.score:
            return evidence
        return best

    # ---- primary group assignment and controlled merging --------------

    def _apply_relations(self, period: AlarmPeriod, relations: list[RelationEvidence]):
        current_gid = self._resolve_group_id(period.primary_group_id)
        if current_gid is None:
            current_gid = self._choose_or_create_group(period, relations)
        group = self.groups[current_gid]

        usable = []
        for rel in relations:
            other_pid = rel.source_period_id if rel.target_period_id == period.period_id else rel.target_period_id
            other = self.periods.get(other_pid)
            if other is None:
                continue
            other_gid = self._resolve_group_id(other.primary_group_id)
            if other_gid is None:
                usable.append((rel, other))
            elif other_gid == current_gid:
                self._record_group_evidence(group, rel)
            else:
                self._record_merge_proposal(current_gid, other_gid, rel)

        if period.period_id in group.core_period_ids and usable:
            best_score = usable[0][0].score
            kept = 0
            for rel, other in usable:
                if kept >= self.config.max_related_periods:
                    break
                if rel.score + EPS < best_score * self.config.relative_attach_ratio:
                    break
                if other.primary_group_id is not None:
                    continue
                self._attach_period(group, other, core=False)
                self._record_group_evidence(group, rel)
                kept += 1

        self._try_ready_merge_proposals()

    def _choose_or_create_group(self, period, relations):
        by_group = defaultdict(list)
        ungrouped = []
        for rel in relations:
            other_pid = rel.source_period_id if rel.target_period_id == period.period_id else rel.target_period_id
            other = self.periods.get(other_pid)
            if other is None:
                continue
            gid = self._resolve_group_id(other.primary_group_id)
            if gid is None:
                ungrouped.append((rel, other))
            else:
                by_group[gid].append((rel, other))

        choices = []
        for gid, items in by_group.items():
            group = self.groups.get(gid)
            if group is None:
                continue
            has_core_edge = any(other.period_id in group.core_period_ids for _rel, other in items)
            distinct_members = len({other.period_id for _rel, other in items})
            if has_core_edge or distinct_members >= 2:
                choices.append((max(rel.score for rel, _other in items), gid, items))
        choices.sort(key=lambda x: (-x[0], x[1]))

        if choices:
            _score, gid, items = choices[0]
            group = self.groups[gid]
            self._attach_period(group, period, core=False)
            for rel, _other in items:
                self._record_group_evidence(group, rel)
            return gid

        if ungrouped:
            ungrouped.sort(key=lambda x: (-x[0].score, x[1].period_id))
            rel, other = ungrouped[0]
            group = self._new_group(period)
            self._attach_period(group, other, core=True)
            self._record_group_evidence(group, rel)
            return group.group_id

        return self._new_group(period).group_id

    def _new_group(self, anchor_period: AlarmPeriod):
        gid = self._next_group_id
        self._next_group_id += 1
        group = PeriodFaultGroup(group_id=gid, anchor_period_id=anchor_period.period_id)
        self.groups[gid] = group
        self._attach_period(group, anchor_period, core=True)
        return group

    def _attach_period(self, group, period, core=False):
        gid = self._resolve_group_id(group.group_id)
        if gid != group.group_id:
            group = self.groups[gid]
        existing_gid = self._resolve_group_id(period.primary_group_id)
        if existing_gid is not None and existing_gid != group.group_id:
            return False
        if period.period_id in group.period_ids:
            return False
        group.period_ids.add(period.period_id)
        period.primary_group_id = group.group_id
        group.start_ts = min(group.start_ts, period.first_ts)
        group.last_ts = max(group.last_ts, period.last_raise_ts)
        if core and len(group.core_period_ids) < self.config.max_core_periods:
            group.core_period_ids.append(period.period_id)
        self.period_attach_count += 1
        return True

    @staticmethod
    def _record_group_evidence(group, rel):
        pair = rel.period_pair
        old = group.evidence_by_pair.get(pair)
        if old is None or rel.score > old.score:
            group.evidence_by_pair[pair] = rel

    def _record_merge_proposal(self, gid1, gid2, rel):
        gid1 = self._resolve_group_id(gid1)
        gid2 = self._resolve_group_id(gid2)
        if gid1 is None or gid2 is None or gid1 == gid2:
            return
        key = tuple(sorted((gid1, gid2)))
        proposal = self.merge_proposals.get(key)
        if proposal is None:
            proposal = MergeProposal(group_ids=key)
            self.merge_proposals[key] = proposal
        proposal.evidence_pairs.add(rel.period_pair)
        proposal.max_strength = max(proposal.max_strength, rel.strength)
        proposal.max_score = max(proposal.max_score, rel.score)

    def _try_ready_merge_proposals(self):
        ready = []
        for key, proposal in list(self.merge_proposals.items()):
            if (
                len(proposal.evidence_pairs) >= self.config.merge_min_evidence
                and proposal.max_strength >= self.config.merge_strength_ratio
            ):
                ready.append((proposal.max_strength, key))
        for _strength, key in sorted(ready, key=lambda x: (-x[0], x[1])):
            proposal = self.merge_proposals.pop(key, None)
            if proposal is None:
                continue
            g1 = self._resolve_group_id(proposal.group_ids[0])
            g2 = self._resolve_group_id(proposal.group_ids[1])
            if g1 is None or g2 is None or g1 == g2:
                continue
            self._merge_groups(g1, g2)

    def _merge_groups(self, gid1, gid2):
        keep_id, drop_id = sorted((gid1, gid2))
        keep = self.groups.get(keep_id)
        drop = self.groups.get(drop_id)
        if keep is None or drop is None:
            return keep or drop
        for pid in sorted(drop.period_ids):
            period = self.periods.get(pid)
            if period is None:
                continue
            period.primary_group_id = keep_id
            keep.period_ids.add(pid)
        core_candidates = keep.core_period_ids + drop.core_period_ids
        core_candidates = sorted(
            set(core_candidates),
            key=lambda pid: (self.periods[pid].first_ts, pid),
        )
        keep.core_period_ids = core_candidates[: self.config.max_core_periods]
        keep.start_ts = min(keep.start_ts, drop.start_ts)
        keep.last_ts = max(keep.last_ts, drop.last_ts)
        for pair, rel in drop.evidence_by_pair.items():
            old = keep.evidence_by_pair.get(pair)
            if old is None or rel.score > old.score:
                keep.evidence_by_pair[pair] = rel
        self.groups.pop(drop_id, None)
        self._group_redirect[drop_id] = keep_id
        self.group_merge_count += 1
        return keep

    def _resolve_group_id(self, gid):
        if gid is None:
            return None
        path = []
        while gid in self._group_redirect:
            path.append(gid)
            gid = self._group_redirect[gid]
        for old in path:
            self._group_redirect[old] = gid
        return gid if gid in self.groups else None

    def _group(self, gid):
        gid = self._resolve_group_id(gid)
        return self.groups.get(gid) if gid is not None else None

    # ---- closure, eviction, output -----------------------------------

    def _close_inactive_groups(self, watermark):
        if self.config.close_inactive_sec <= 0:
            return
        cutoff = float(watermark) - self.config.close_inactive_sec
        ready = []
        for gid, group in self.groups.items():
            if group.last_ts >= cutoff:
                continue
            periods = [self.periods.get(pid) for pid in group.period_ids]
            if any(p is not None and (p.status == "open" or p.is_dirty) for p in periods):
                continue
            ready.append(gid)
        for gid in ready:
            self._finalize_group(gid)

    def _finalize_group(self, gid):
        group = self.groups.pop(gid, None)
        if group is None:
            return
        record = self._group_record(group)
        if record["event_count"] >= self.config.min_group_events:
            if self.closed_group_sink is None:
                raise RuntimeError("incremental closed-group sink is not configured")
            self.closed_group_sink(record)
            self.closed_group_count += 1

    def _evict_expired_periods(self, watermark):
        cutoff = float(watermark) - (
            self.config.history_window_sec
            + self.config.aggregation_wait_sec
            + self.config.time_slack_sec
        )
        dead = []
        for pid, period in self.periods.items():
            # Active groups own their periods until group finalization; output,
            # core-gating, and merge proposals all need the period metadata even
            # after it has aged out of the candidate window.
            if self._resolve_group_id(period.primary_group_id) is not None:
                continue
            if period.status == "closed" and period.last_raise_ts < cutoff:
                if period.pending_ready_ts is None and not period.is_dirty:
                    dead.append(pid)
        for pid in dead:
            period = self.periods.pop(pid, None)
            if period is None:
                continue
            ids = self.period_ids_by_signature.get(period.signature)
            if ids is not None:
                ids.discard(pid)
                if not ids:
                    self.period_ids_by_signature.pop(period.signature, None)
            type_ids = self.period_ids_by_type.get(period.period_type)
            if type_ids is not None:
                type_ids.discard(pid)
                if not type_ids:
                    self.period_ids_by_type.pop(period.period_type, None)

    def flush(self):
        for period in list(self.periods.values()):
            if period.status == "open":
                period.close(period.last_raise_ts, "stream_end")
                if self.open_period_by_type.get(period.period_type) == period.period_id:
                    self.open_period_by_type.pop(period.period_type, None)
            if period.is_dirty and period.pending_ready_ts is None:
                self._schedule_harvest(period, period.last_raise_ts)
        self._harvest_ready(math.inf)
        self._try_ready_merge_proposals()
        for gid in sorted(list(self.groups)):
            self._finalize_group(gid)

    def _group_record(self, group):
        periods = [self.periods[pid] for pid in group.period_ids if pid in self.periods]
        events = []
        seen = set()
        for period in periods:
            for event in period.events:
                if event.index not in seen:
                    seen.add(event.index)
                    events.append(event)
        events.sort(key=lambda e: (e.ts, e.index))
        summaries = [_summary_of(event) for event in events]
        summary_by_index = {event.index: summary for event, summary in zip(events, summaries)}
        anchor = self.periods.get(group.anchor_period_id)
        root_event = anchor.events[0] if anchor is not None and anchor.events else events[0]
        edges = []
        for rel in sorted(group.evidence_by_pair.values(), key=lambda x: (-x.score, x.period_pair)):
            if rel.target_period_id not in group.period_ids or rel.source_period_id not in group.period_ids:
                continue
            src = summary_by_index.get(rel.source_event.index, {})
            tgt = summary_by_index.get(rel.target_event.index, {})
            edges.append(
                {
                    "source_period_id": rel.source_period_id,
                    "target_period_id": rel.target_period_id,
                    "source_event_id": src.get("event_id", ""),
                    "target_event_id": tgt.get("event_id", ""),
                    "source_occurrence_uuid": src.get("occurrence_uuid", ""),
                    "target_occurrence_uuid": tgt.get("occurrence_uuid", ""),
                    "score": float(rel.score),
                    "strength": float(rel.strength),
                }
            )
        timestamps = [float(s["ts"]) for s in summaries]
        gid_text = f"mhp-period-{group.group_id:06d}"
        return {
            "group_id": gid_text,
            "cascade_id": group.group_id,
            "rule": "alarm_flow_mhp_period",
            "event_count": len(events),
            "alarm_period_count": len(periods),
            "start_ts": min(timestamps),
            "end_ts": max(timestamps),
            "duration_sec": max(timestamps) - min(timestamps),
            "root_event": _summary_of(root_event),
            "anchor_period_id": group.anchor_period_id,
            "core_period_ids": list(group.core_period_ids),
            "site_list": sorted({s["site_id"] for s in summaries if s.get("site_id")}),
            "alarm_source_list": sorted(
                {s["alarm_source"] for s in summaries if s.get("alarm_source")}
            ),
            "alarm_title_counts": dict(
                Counter(s["alarm_title"] for s in summaries if s.get("alarm_title"))
            ),
            "alarm_type_counts": dict(
                Counter(s["alarm_type"] for s in summaries if s.get("alarm_type"))
            ),
            "symptoms": summaries,
            "edges": edges,
        }

    def stats(self):
        open_periods = sum(1 for p in self.periods.values() if p.status == "open")
        return {
            "total_input_events": self.total_input_events,
            "total_raise_events": self.total_raise_events,
            "total_clear_events": self.total_clear_events,
            "dropped_no_type": self.dropped_no_type,
            "created_periods": self.created_periods,
            "open_periods": open_periods,
            "idle_closed_periods": self.idle_closed_periods,
            "clear_closed_periods": self.clear_closed_periods,
            "harvest_count": self.harvest_count,
            "relation_count": self.relation_count,
            "period_attach_count": self.period_attach_count,
            "group_merge_count": self.group_merge_count,
            "compiled_pair_count": self.plan.compiled_pair_count,
            "pruned_pair_count": self.plan.pruned_pair_count,
            "incremental_evaluated_pair_count": (
                self.plan.compiled_pair_count + self.plan.pruned_pair_count
            ),
            "preloaded_signature_count": self.plan.preloaded_signature_count,
            "preloaded_edge_count": self.plan.preloaded_edge_count,
            "preloaded_array_bytes": sum(
                index.memory_bytes for index in self.plan.precompiled_indexes
            ),
            "active_group_count": len(self.groups),
            "closed_group_count": self.closed_group_count,
        }


def _build_runtime_scorers(artifact, ne_graph_path, site_graph_path, quiet=False):
    if getattr(artifact.config, "edge_mode", "device") != "feature":
        raise ValueError("AlarmPeriod inference requires edge_mode=feature")
    md = artifact.training_metadata or {}
    fk = md.get("feature_kernel")
    rt = md.get("feature_runtime") or {}
    if fk is None:
        raise ValueError("feature-mode artifact missing feature_kernel")
    node_field = artifact.config.topology_node_field
    ne_graph_data = load_ne_graph(ne_graph_path)
    graph_ctx = build_node_context(ne_graph_data, node_field)
    topo_graph = load_ne_graph(site_graph_path) if node_field == "site_id" else ne_graph_data
    infer_hops = max(int(getattr(artifact.config, "feature_topo_max_hops", 2)), 1)
    topo_idx = NETopologyIndex.from_graph(
        topo_graph, max_hops=infer_hops, undirected_only=True
    )
    dyn_mode = getattr(artifact.config, "dynamic_alpha", "off")
    n_dynamic = 6 if dyn_mode == "source_target" else (3 if dyn_mode != "off" else 0)
    scorer = RuntimeFeatureScorer(
        kernel=FeatureKernel.from_dict(fk),
        at_vocab=rt.get("at_vocab", []),
        graph_context=graph_ctx,
        topology_index=topo_idx,
        beta=float(rt.get("beta", 1.0)),
        n_dynamic=n_dynamic,
        dynamic_mode=dyn_mode,
        domain_vocab=rt.get("domain_vocab", []),
        node_domains=rt.get("node_domains", {}) or getattr(graph_ctx, "node_domains", {}),
        node_field=node_field,
    )
    mu_scorer = None
    if rt.get("mu_kernel") is not None and rt.get("mu_spec") is not None:
        mu_scorer = RuntimeMuScorer(
            mu_kernel=FeatureKernel.from_dict(rt["mu_kernel"]),
            mu_spec=MuFeatureSpec.from_dict(rt["mu_spec"]),
            graph_context=graph_ctx,
        )
    if not quiet:
        print(
            f"[period] feature scorer ready: dynamic={dyn_mode}, "
            f"topology_hops={infer_hops}, node_field={node_field}",
            flush=True,
        )
    return scorer, mu_scorer, ne_graph_data


class IncrementalPeriodOutput:
    """Durable JSONL output for finalized groups and their optional views."""

    FORMAT = "alarm_flow_mhp.period_groups_jsonl"
    VERSION = 1

    def __init__(
        self,
        groups_path,
        metadata,
        edges_path="",
        visual_path="",
        ne_graph_path=NE_GRAPH_JSON,
        site_graph_path=SITE_GRAPH_JSON,
    ):
        self.groups_path = os.path.abspath(groups_path)
        self.edges_path = os.path.abspath(edges_path) if edges_path else ""
        self.group_count = 0
        self.edge_count = 0
        self._closed = False
        self.groups_stream = None
        self.edges_stream = None
        self.visual = None
        try:
            for path in (self.groups_path, self.edges_path):
                if path:
                    parent = os.path.dirname(path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
            self.groups_stream = open(
                self.groups_path, "w", encoding="utf-8", buffering=1
            )
            if self.edges_path:
                self.edges_stream = open(
                    self.edges_path, "w", encoding="utf-8", buffering=1
                )
            if visual_path:
                from alarm_flow_mhp.visual_output import AlarmMHPVisualOutputSession

                self.visual = AlarmMHPVisualOutputSession.from_files(
                    visual_path, ne_graph_path, site_graph_path
                )
                self.visual.reset_output_file()
            self._write_group_record(
                {
                    "record_type": "metadata",
                    "format": self.FORMAT,
                    "version": self.VERSION,
                    "metadata": dict(metadata),
                }
            )
        except Exception:
            self.close()
            raise

    def _write_group_record(self, record):
        if self.groups_stream is None:
            raise RuntimeError("incremental groups output is not open")
        self.groups_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.groups_stream.flush()

    def emit_group(self, group):
        if self._closed:
            raise RuntimeError("incremental period output is already closed")
        self._write_group_record({"record_type": "group", "group": group})
        self.group_count += 1
        if self.edges_stream is not None:
            for edge in group.get("edges", ()):
                self.edges_stream.write(json.dumps(edge, ensure_ascii=False) + "\n")
                self.edge_count += 1
            self.edges_stream.flush()
        if self.visual is not None:
            self.visual.emit_groups([group], finalization_reason="period_finalized")

    def close(self, summary=None):
        if self._closed:
            return
        if summary is not None and self.groups_stream is not None:
            self._write_group_record(
                {
                    "record_type": "summary",
                    "summary": {
                        **dict(summary),
                        "emitted_group_count": self.group_count,
                        "emitted_edge_count": self.edge_count,
                    },
                }
            )
        self._closed = True
        if self.groups_stream is not None:
            self.groups_stream.close()
            self.groups_stream = None
        if self.edges_stream is not None:
            self.edges_stream.close()
            self.edges_stream = None
        if self.visual is not None:
            self.visual.close()


def _default_visual_output(groups_output):
    path = str(groups_output)
    base = path[:-6] if path.lower().endswith(".jsonl") else os.path.splitext(path)[0]
    return f"{base}.visual.jsonl"


def _iter_time_windowed_cache_items(items, start_ts, end_ts, time_stats):
    """Emulate prepare-time --start-time/--end-time on an already-built cache.

    The batch path windows raw rows by 告警首次发生时间 and then trims trailing
    clear events.  A cached clear item no longer carries its raise time (the
    field is overwritten with the effective clear time), so the raise-side
    window is enforced through the shared raise/clear identity: clears whose
    raise fell before the window are dropped with it.  Clears are buffered
    until the next in-window raise, so the buffer left at the end is exactly
    the ``trim_trailing_clear_alarms`` tail; because the cache is ts-sorted,
    reading can also stop at the first item past ``end_ts``.
    """
    excluded_raise_identities = set()
    pending_clears = []
    for item in items:
        ts = float(item.get("ts", 0.0))
        if end_ts is not None and ts > end_ts:
            time_stats["stopped_early"] = True
            break
        time_stats["input_event_count"] += 1
        if is_clear_alarm(item.get("alarm", {})):
            if start_ts is not None:
                identity = require_alarm_identity(item)
                if identity in excluded_raise_identities:
                    excluded_raise_identities.discard(identity)
                    time_stats["dropped_clear_event_count"] += 1
                    continue
            pending_clears.append(item)
            continue
        if start_ts is not None and ts < start_ts:
            excluded_raise_identities.add(require_alarm_identity(item))
            time_stats["dropped_raise_event_count"] += 1
            continue
        if pending_clears:
            time_stats["kept_event_count"] += len(pending_clears)
            yield from pending_clears
            pending_clears.clear()
        time_stats["kept_event_count"] += 1
        yield item
    time_stats["trimmed_trailing_clear_count"] = len(pending_clears)


def _stream_sorted_cache_events(
    path,
    ne_graph_data,
    *,
    start_time="",
    end_time="",
    regions="",
    annotate_domain=False,
    filter_stats=None,
):
    """Lazily iterate a sorted alarm cache without materializing the events.

    Applies the same time window, region filter, and device-domain
    annotation/filter as the batch loading path, in the same order.  Counters
    accumulate into ``filter_stats`` while iterating, so they are complete
    only after the stream is exhausted.  Validation is eager; only the
    returned iterator is lazy.
    """
    filter_stats = filter_stats if filter_stats is not None else {}
    start_ts = (
        parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    )
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time 不能晚于 end_time")
    time_stats = None
    if start_ts is not None or end_ts is not None:
        time_stats = {
            "enabled": True,
            "stage": "sorted_cache_stream",
            "start_time": str(start_time or ""),
            "end_time": str(end_time or ""),
            "input_event_count": 0,
            "kept_event_count": 0,
            "dropped_raise_event_count": 0,
            "dropped_clear_event_count": 0,
            "trimmed_trailing_clear_count": 0,
            "stopped_early": False,
        }
        filter_stats["time_filter"] = time_stats
    selected_regions = frozenset(parse_regions(regions))
    region_stats = None
    ne_region_map = {}
    if selected_regions:
        ne_region_map = build_ne_region_map(ne_graph_data)
        region_stats = {
            "enabled": True,
            "stage": "sorted_cache_stream",
            "regions": sorted(selected_regions),
            "allowed_device_count": sum(
                1 for region in ne_region_map.values() if region in selected_regions
            ),
            "input_event_count": 0,
            "kept_event_count": 0,
            "dropped_event_count": 0,
            "unknown_region_event_count": 0,
            "kept_region_counts": {},
            "dropped_region_counts": {},
        }
        filter_stats["region_filter"] = region_stats
    domain_stats = None
    domain_map = {}
    if annotate_domain:
        domain_map = build_ne_domain_bucket_map(ne_graph_data)
        domain_stats = {
            "enabled": True,
            "supported_domains": sorted(SUPPORTED_DEVICE_DOMAINS),
            "input_event_count": 0,
            "kept_event_count": 0,
            "dropped_event_count": 0,
            "dropped_by_domain": {},
        }
        filter_stats["domain_filter"] = domain_stats

    def _events():
        items = iter_sorted_alarm_cache_items(path)
        if time_stats is not None:
            items = _iter_time_windowed_cache_items(items, start_ts, end_ts, time_stats)
        for event in items:
            if region_stats is not None:
                region_stats["input_event_count"] += 1
                region = event_region(event, ne_region_map)
                if region not in selected_regions:
                    if not region:
                        region_stats["unknown_region_event_count"] += 1
                    counts = region_stats["dropped_region_counts"]
                    key = region or "<unknown>"
                    counts[key] = counts.get(key, 0) + 1
                    region_stats["dropped_event_count"] += 1
                    continue
                counts = region_stats["kept_region_counts"]
                counts[region] = counts.get(region, 0) + 1
                region_stats["kept_event_count"] += 1
            if domain_stats is not None:
                domain_stats["input_event_count"] += 1
                ne_id = normalize_text(event.get("alarm_source", ""))
                domain = domain_map.get(ne_id, "")
                event[DEVICE_DOMAIN_FIELD] = domain
                if domain not in SUPPORTED_DEVICE_DOMAINS:
                    counts = domain_stats["dropped_by_domain"]
                    key = domain or "UNKNOWN_DEVICE"
                    counts[key] = counts.get(key, 0) + 1
                    domain_stats["dropped_event_count"] += 1
                    continue
                domain_stats["kept_event_count"] += 1
            yield event

    return _events()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="AlarmPeriod-oriented online MHP grouping (feature mode)."
    )
    parser.add_argument("model", help="Trained alarm-flow MHP artifact JSON.")
    parser.add_argument("alarms", help="Sorted alarm cache or raw alarm input.")
    parser.add_argument(
        "--groups-output",
        required=True,
        help="Incremental groups JSONL; truncated at startup and flushed per finalized group.",
    )
    parser.add_argument("--edges-output", default="", help="Optional period evidence JSONL.")
    parser.add_argument(
        "--visual-output",
        default="",
        help="Visual JSONL; default: <groups-output without .jsonl>.visual.jsonl.",
    )
    parser.add_argument(
        "--association-cache",
        action="append",
        default=[],
        metavar="NPZ",
        help=(
            "Compact binary association cache (.npz). Repeatable: pass once per "
            "cache (e.g. a related cache and a disjoint unrelated cache) and the "
            "runtime unions their edges. Unseen online devices are compiled "
            "incrementally in memory only."
        ),
    )
    parser.add_argument(
        "--candidate-policy",
        default="",
        help=(
            "Approved candidate policy JSON; required for "
            "--candidate-scope unrelated."
        ),
    )
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=resource_display("ne_graph.json"))
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=resource_display("site_graph.json"))
    parser.add_argument(
        "--start-time",
        default="",
        help="Only process alarms whose first-occurrence time is >= this; works for raw and sorted-cache input.",
    )
    parser.add_argument(
        "--end-time",
        default="",
        help="Only process alarms whose first-occurrence time is <= this; works for raw and sorted-cache input.",
    )
    parser.add_argument(
        "--clear-delay-sec",
        type=float,
        default=0.0,
        help="Clear-effective delay for raw input; sorted caches bake this in at prepare time.",
    )
    parser.add_argument(
        "--aggregation-wait-sec",
        type=float,
        default=None,
        help="Fixed event-time maturity lag. Default: max(30s, time_slack_sec).",
    )
    parser.add_argument("--period-idle-sec", type=float, default=300.0)
    parser.add_argument("--history-window-sec", type=float, default=None)
    parser.add_argument("--time-slack-sec", type=float, default=None)
    parser.add_argument("--late-penalty-half-life-sec", type=float, default=None)
    parser.add_argument("--close-inactive-sec", type=float, default=7200.0)
    parser.add_argument(
        "--min-group-events",
        type=int,
        default=1,
        help="Groups with fewer events are dropped at finalization; default 1 keeps all groups.",
    )
    parser.add_argument(
        "--regions",
        default="",
        help="Optional region filter for input alarms; default: no filtering (artifact regions are NOT inherited).",
    )
    parser.add_argument("--immigrant-bias", type=float, default=1.0)
    parser.add_argument("--feature-alpha-floor", type=float, default=None)
    parser.add_argument("--attach-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--relative-attach-ratio", type=float, default=0.8)
    parser.add_argument("--max-related-periods", type=int, default=8)
    parser.add_argument("--max-core-periods", type=int, default=4)
    parser.add_argument("--merge-strength-ratio", type=float, default=2.0)
    parser.add_argument("--merge-min-evidence", type=int, default=2)
    parser.add_argument(
        "--candidate-scope",
        choices=("related", "global", "unrelated"),
        default="related",
    )
    parser.add_argument(
        "--topology-relation-prior",
        default="",
        help="Comma-separated relation multipliers, same format as stream_alarm_mhp.py.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Refresh live counters every N events; default 1 (display is time-throttled). 0 disables.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print aggregated phase timings for input, ingest, harvest, grouping, "
            "maintenance, association compilation, and output."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    timer = PhaseTimer() if args.profile else None
    if timer is not None:
        timer.mark_wall_start()
    if not str(args.groups_output).lower().endswith(".jsonl"):
        parser.error("--groups-output must end with .jsonl (incremental format)")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if not args.visual_output:
        args.visual_output = _default_visual_output(args.groups_output)
    try:
        relation_prior = parse_topology_relation_prior(args.topology_relation_prior)
    except ValueError as exc:
        parser.error(str(exc))

    t0 = time.monotonic()
    with _profile_phase(timer, "init.load_model"):
        artifact = load_alarm_mhp_artifact(args.model)
    with _profile_phase(timer, "init.build_runtime_scorers"):
        scorer, mu_scorer, ne_graph_data = _build_runtime_scorers(
            artifact, args.ne_graph, args.site_graph, quiet=args.quiet
        )
    annotate_domain = DEVICE_DOMAIN_FIELD in tuple(artifact.config.type_fields)
    stream_filter_stats = {}
    with _profile_phase(timer, "init.prepare_input"):
        streaming_input = is_sorted_alarm_cache_file(args.alarms)
        if streaming_input:
            # Sorted caches are consumed straight off disk: the engine, the
            # region/domain filters, and the outputs are all incremental, so the
            # full event list never materializes in memory.  Filter statistics are
            # only complete at the end and go into the summary record.
            cache_stream = SortedAlarmCacheStream(args.alarms)
            alarm_metadata = dict(cache_stream.metadata)
            total_events = len(cache_stream)
            if args.clear_delay_sec and not args.quiet:
                print(
                    "[period] note: --clear-delay-sec is baked into the sorted cache "
                    "at prepare time and is ignored for cache input.",
                    flush=True,
                )
            try:
                events = _stream_sorted_cache_events(
                    args.alarms,
                    ne_graph_data,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    regions=args.regions,
                    annotate_domain=annotate_domain,
                    filter_stats=stream_filter_stats,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            events, alarm_metadata = load_ordered_alarm_events(
                args.alarms,
                topo_path=args.site_graph,
                ne_graph_path=args.ne_graph,
                start_time=args.start_time or None,
                end_time=args.end_time or None,
                clear_delay_sec=args.clear_delay_sec,
                regions=args.regions,
            )
            if annotate_domain:
                events, domain_stats = filter_and_annotate_device_domain(events, ne_graph_data)
                if not args.quiet:
                    print(f"[period] domain filter: {domain_stats}", flush=True)
            total_events = len(events)
    if artifact.config.regions and not args.regions:
        print(
            f"[period] note: artifact was trained with regions={list(artifact.config.regions)}, "
            "but inference no longer inherits them; pass --regions to filter.",
            flush=True,
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
    aggregation_wait = (
        float(args.aggregation_wait_sec)
        if args.aggregation_wait_sec is not None
        else max(30.0, slack)
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
    min_group_events = int(args.min_group_events)
    if int(artifact.config.min_group_events) != min_group_events:
        print(
            f"[period] note: artifact min_group_events={int(artifact.config.min_group_events)} "
            f"is no longer inherited; using {min_group_events} "
            "(override with --min-group-events).",
            flush=True,
        )
    config = PeriodStreamConfig(
        aggregation_wait_sec=aggregation_wait,
        period_idle_sec=args.period_idle_sec,
        history_window_sec=history,
        time_slack_sec=slack,
        late_penalty_half_life_sec=late_half_life,
        time_scale_sec=float(artifact.config.time_scale_sec),
        close_inactive_sec=args.close_inactive_sec,
        min_group_events=min_group_events,
        immigrant_bias=args.immigrant_bias,
        feature_alpha_floor=floor,
        attach_threshold_ratio=args.attach_threshold_ratio,
        relative_attach_ratio=args.relative_attach_ratio,
        max_related_periods=args.max_related_periods,
        max_core_periods=args.max_core_periods,
        merge_strength_ratio=args.merge_strength_ratio,
        merge_min_evidence=args.merge_min_evidence,
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
        with _profile_phase(timer, "init.load_candidate_policy"):
            try:
                policy_fingerprint = candidate_policy_fingerprint(
                    args.model,
                    args.ne_graph,
                    args.site_graph,
                    _association_plan_config(config),
                    artifact.config.topology_node_field,
                )
                candidate_policy = load_candidate_policy(
                    args.candidate_policy,
                    expected_fingerprint=policy_fingerprint,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(f"cannot load --candidate-policy: {exc}")

    association_caches = []
    if args.association_cache:
        # One expected fingerprint pins model/graphs/plan-config; per-cache scope
        # and policy digest are ignored so a related cache and a disjoint
        # unrelated cache (built from the same model/graphs) both validate.
        with _profile_phase(timer, "init.load_association_caches"):
            try:
                fingerprint = association_cache_fingerprint(
                    args.model,
                    args.ne_graph,
                    args.site_graph,
                    config,
                    artifact.config.topology_node_field,
                )
            except OSError as exc:
                parser.error(f"cannot fingerprint association cache inputs: {exc}")
            for cache_path in args.association_cache:
                try:
                    payload = load_association_cache(
                        cache_path,
                        expected_fingerprint=fingerprint,
                        allow_scope_mismatch=True,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    parser.error(f"cannot load --association-cache {cache_path}: {exc}")
                association_caches.append(payload)
                if not args.quiet:
                    cache_md = payload.get("metadata") or {}
                    array_mib = sum(
                        array.nbytes for array in payload.get("arrays", {}).values()
                    ) / (1024 * 1024)
                    print(
                        f"[period] association cache loaded ({os.path.basename(cache_path)}): "
                        f"scope={((payload.get('fingerprint') or {}).get('plan_config') or {}).get('candidate_scope', '?')}, "
                        f"edges={cache_md.get('edge_count', 0)}, arrays={array_mib:.1f}MiB",
                        flush=True,
                    )

    with _profile_phase(timer, "init.build_engine"):
        engine = AlarmPeriodMHPAssigner(
            artifact,
            config,
            feature_scorer=scorer,
            mu_scorer=mu_scorer,
            association_cache=association_caches or None,
            candidate_policy=candidate_policy,
        )
    # The plan owns decoded edge objects now; release the JSON row arrays before
    # processing a potentially large alarm stream.
    association_caches = None
    run_metadata = {
        "algorithm": "alarm_flow_mhp.alarm_period_stream",
        "model": os.path.abspath(args.model),
        "input": os.path.abspath(args.alarms),
        "association_cache": [
            os.path.abspath(path) for path in args.association_cache
        ],
        "candidate_policy": (
            os.path.abspath(args.candidate_policy) if args.candidate_policy else ""
        ),
        "groups_output": os.path.abspath(args.groups_output),
        "edges_output": os.path.abspath(args.edges_output) if args.edges_output else "",
        "visual_output": os.path.abspath(args.visual_output),
        "alarm_metadata": alarm_metadata,
        "streaming_input": bool(streaming_input),
        "profiling": bool(args.profile),
        "config": {key: value for key, value in vars(config).items()},
    }
    with _profile_phase(timer, "init.build_output"):
        output = IncrementalPeriodOutput(
            args.groups_output,
            run_metadata,
            edges_path=args.edges_output,
            visual_path=args.visual_output,
            ne_graph_path=args.ne_graph,
            site_graph_path=args.site_graph,
        )
    engine.closed_group_sink = output.emit_group
    if timer is not None:
        _enable_period_profiling(timer, engine, output)
        events = _iter_profiled_events(events, timer)
    if not args.quiet:
        print(
            f"[period] incremental outputs: groups={args.groups_output}, "
            f"edges={args.edges_output or '<disabled>'}, visual={args.visual_output}",
            flush=True,
        )
        print(
            f"[period] events={total_events}"
            f"{' (cache header, streamed)' if streaming_input else ''}, "
            f"wait={config.aggregation_wait_sec:g}s, "
            f"idle={config.period_idle_sec:g}s, history={config.history_window_sec:g}s, "
            f"scope={config.candidate_scope}, dynamic={getattr(artifact.config, 'dynamic_alpha', 'off')}",
            flush=True,
        )

    process_progress = (
        ProgressBar(total_events, "处理 AlarmPeriod 告警")
        if args.progress_every and not args.quiet
        else None
    )

    def live_progress_text():
        return (
            f"periods={engine.created_periods} "
            f"harvests={engine.harvest_count} "
            f"groups={len(engine.groups)}+{engine.closed_group_count}"
        )

    if process_progress is not None:
        process_progress.extra_text = live_progress_text()
        if timer is not None:
            timer.wrap_method(process_progress, "update", "pipeline.progress")
            timer.wrap_method(process_progress, "close", "pipeline.progress")
    completed = False
    stats = None
    elapsed = 0.0
    try:
        with _profile_phase(timer, "pipeline.total"):
            try:
                for i, event in enumerate(events):
                    engine.process(event)
                    if process_progress is not None:
                        if (i + 1) % args.progress_every == 0:
                            process_progress.extra_text = live_progress_text()
                        process_progress.update()
            finally:
                if process_progress is not None:
                    process_progress.extra_text = live_progress_text()
                    process_progress.close()
            engine.flush()
            completed = True
    finally:
        stats = engine.stats()
        elapsed = time.monotonic() - t0
        summary = {
            "status": "complete" if completed else "interrupted",
            "stats": stats,
            "elapsed_seconds": elapsed,
        }
        if stream_filter_stats:
            summary["input_filter_stats"] = stream_filter_stats
        try:
            output.close(summary)
        finally:
            if timer is not None:
                timer.mark_wall_end()
                _print_period_profile(timer)
    if not args.quiet:
        for name, filter_summary in stream_filter_stats.items():
            print(f"[period] {name} (streamed): {filter_summary}", flush=True)
        print(
            f"[period] done: groups={stats['closed_group_count']}, "
            f"periods={stats['created_periods']}, harvests={stats['harvest_count']}, "
            f"preloaded_edges={stats['preloaded_edge_count']}, "
            f"incremental_edges={stats['compiled_pair_count']}, elapsed={elapsed:.2f}s; "
            f"groups_output={args.groups_output}, visual_output={args.visual_output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
