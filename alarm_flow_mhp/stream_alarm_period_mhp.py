#!/usr/bin/env python3
"""AlarmPeriod-oriented online inference for feature-mode alarm-flow MHP.

This is intentionally a separate engine from ``stream_alarm_mhp.py``.  The
legacy/fast engines assign one parent to every alarm occurrence; this engine
uses an AlarmPeriod as the matching and grouping unit:

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
from dataclasses import dataclass, field
import heapq
import hashlib
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
    filter_and_annotate_device_domain,
)
from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_isahp.sequences import (
    alarm_type_from_title,
    alarm_type_label,
    event_type_label,
)
from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
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
    parse_topology_relation_prior,
    topology_relation_weights,
)
from fault_grouping.alarm_events.identity import require_alarm_identity
from fault_grouping.alarm_events.io import is_clear_alarm
from mhp.feature_kernel import FeatureKernel
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display
from topology_tools.region_utils import load_ne_graph


EPS = 1e-12
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
        if self.candidate_scope not in {"related", "global"}:
            raise ValueError("candidate_scope must be 'related' or 'global'")


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
) -> dict:
    """Fingerprint every input that can change the sparse association plan."""
    node_field = str(topology_node_field or "alarm_source")
    topology_graph_path = site_graph_path if node_field == "site_id" else ne_graph_path
    return {
        "model_sha256": _sha256_file(model_path),
        "ne_graph_sha256": _sha256_file(ne_graph_path),
        "topology_graph_sha256": _sha256_file(topology_graph_path),
        "topology_node_field": node_field,
        "plan_config": _association_plan_config(config),
    }


def load_association_cache(path, expected_fingerprint=None) -> dict:
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
        if actual != expected_fingerprint:
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


def graph_period_types(artifact, scorer):
    """Full inductive universe: graph entities × model alarm-type vocabulary."""
    rt = (artifact.training_metadata or {}).get("feature_runtime") or {}
    alarm_types = sorted({str(value) for value in (rt.get("at_vocab") or []) if str(value)})
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
    pairs; ``global`` evaluates every observed signature pair.
    """

    def __init__(self, scorer, mu_scorer, artifact, config: PeriodStreamConfig):
        self.scorer = scorer
        self.decomposed = DecomposedFeatureScorer(scorer)
        self.mu_scorer = mu_scorer
        self.artifact = artifact
        self.config = config
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
        self.precompiled_index: Optional[CompactAssociationIndex] = None
        self.edges_by_target: dict[PeriodSignature, dict[PeriodSignature, CompiledEdge]] = defaultdict(dict)
        self.edges_by_source: dict[PeriodSignature, dict[PeriodSignature, CompiledEdge]] = defaultdict(dict)
        self._mu_cache: dict[PeriodType, float] = {}
        self.compiled_pair_count = 0
        self.pruned_pair_count = 0
        self.preloaded_signature_count = 0
        self.preloaded_edge_count = 0

    def iter_edges_by_target(self, signature):
        if self.precompiled_index is not None:
            yield from self.precompiled_index.iter_target(signature)
        yield from self.edges_by_target.get(signature, {}).items()

    def iter_edges_by_source(self, signature):
        if self.precompiled_index is not None:
            yield from self.precompiled_index.iter_source(signature)
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

    def _related(self, a: PeriodSignature, b: PeriodSignature) -> bool:
        if self.config.candidate_scope == "global":
            return True
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

    def register_signature(self, signature: PeriodSignature):
        if (
            signature.period_type in self.covered_period_types
            or signature in self.signatures
        ):
            return
        existing = list(self.signatures)
        self.signatures.add(signature)
        for other in existing:
            if not self._related(signature, other):
                continue
            self._compile(signature, other)
            if signature != other:
                self._compile(other, signature)
        if self._covered_candidate_index is not None:
            for other_type in self._candidate_sources(
                signature.period_type, self._covered_candidate_index
            ):
                for state in range(8):
                    other = PeriodSignature(other_type, state)
                    self._compile(signature, other)
                    self._compile(other, signature)
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

    def _precompile_target_only_batches(self, prepared, progress, edge_batch_sink):
        """Vectorized offline compiler for target-dynamic cache rows.

        Static pair features are evaluated once per PeriodType pair. The eight
        target-state terms are then broadcast over that vector, avoiding eight
        repeated topology/attribute passes and one Python call per state edge.
        """
        type_pair_count = 0
        target_terms = np.asarray(
            [self.decomposed.tgt_term(_combo_state(state)) for state in range(8)],
            dtype=np.float64,
        )
        for target_type in prepared["period_types"]:
            source_types = self._candidate_sources(target_type, prepared)
            source_count = len(source_types)
            if source_count:
                source_alarm_types = [value.alarm_type for value in source_types]
                source_entities = [value.entity for value in source_types]
                logits = self.decomposed.logits_for_target(
                    target_type.alarm_type,
                    target_type.entity,
                    source_alarm_types,
                    source_entities,
                )
                alpha = self.decomposed._softplus(
                    logits.reshape(1, -1) + target_terms.reshape(-1, 1)
                )
                if self.decomposed.alpha_scale != 1.0:
                    alpha = alpha * self.decomposed.alpha_scale
                if self.config.topology_relation_prior:
                    relation_weights = topology_relation_weights(
                        [topo_node_of(value) for value in source_entities],
                        topo_node_of(target_type.entity),
                        self.scorer.topology_index,
                        self.scorer.node_infos,
                        self.config.topology_relation_prior,
                    )
                else:
                    relation_weights = np.ones(source_count, dtype=np.float64)
                base_scores = alpha * self.beta * relation_weights.reshape(1, -1)
                threshold = self._mu(target_type)
                keep = ~(
                    (alpha < self.config.feature_alpha_floor)
                    | (base_scores + EPS < threshold)
                    | (base_scores <= 0)
                )
                kept_count = int(np.count_nonzero(keep))
                self.compiled_pair_count += kept_count
                self.pruned_pair_count += int(keep.size) - kept_count
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
                        source_types,
                        source_indices,
                        kept_scores,
                        threshold,
                        past_windows,
                        future_windows,
                    )
            type_pair_count += source_count
            if progress is not None:
                progress(type_pair_count, self.compiled_pair_count, self.pruned_pair_count)
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
        self._covered_candidate_index = self.prepare_candidate_period_types(
            covered, count_pairs=False
        )
        self.precompiled_index = CompactAssociationIndex(
            period_types,
            payload["arrays"],
            state_layout=declared_state_layout,
        )
        self.preloaded_signature_count += len(covered) * 8
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
        self.plan = CompiledAssociationPlan(feature_scorer, mu_scorer, artifact, config)
        if association_cache is not None:
            self.plan.load_cache_payload(association_cache)
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
        self.closed_groups: list[dict] = []
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
            self.closed_groups.append(record)

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
            "preloaded_array_bytes": (
                self.plan.precompiled_index.memory_bytes
                if self.plan.precompiled_index is not None
                else 0
            ),
            "active_group_count": len(self.groups),
            "closed_group_count": len(self.closed_groups),
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
    topo_idx = NETopologyIndex.from_graph(topo_graph, max_hops=infer_hops)
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


def _write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_jsonl(path, records):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _build_parser():
    parser = argparse.ArgumentParser(
        description="AlarmPeriod-oriented online MHP grouping (feature mode)."
    )
    parser.add_argument("model", help="Trained alarm-flow MHP artifact JSON.")
    parser.add_argument("alarms", help="Sorted alarm cache or raw alarm input.")
    parser.add_argument("--groups-output", required=True, help="Output groups JSON.")
    parser.add_argument("--edges-output", default="", help="Optional period evidence JSONL.")
    parser.add_argument("--visual-output", default="", help="Optional visual-output JSONL.")
    parser.add_argument(
        "--association-cache",
        default="",
        help=(
            "Optional version-4 compact binary association cache (.npz). "
            "Known signatures are loaded from it; unseen online devices are "
            "compiled incrementally in memory only."
        ),
    )
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=resource_display("ne_graph.json"))
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=resource_display("site_graph.json"))
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0)
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
    parser.add_argument("--min-group-events", type=int, default=None)
    parser.add_argument("--immigrant-bias", type=float, default=1.0)
    parser.add_argument("--feature-alpha-floor", type=float, default=None)
    parser.add_argument("--attach-threshold-ratio", type=float, default=1.0)
    parser.add_argument("--relative-attach-ratio", type=float, default=0.8)
    parser.add_argument("--max-related-periods", type=int, default=8)
    parser.add_argument("--max-core-periods", type=int, default=4)
    parser.add_argument("--merge-strength-ratio", type=float, default=2.0)
    parser.add_argument("--merge-min-evidence", type=int, default=2)
    parser.add_argument("--candidate-scope", choices=("related", "global"), default="related")
    parser.add_argument(
        "--topology-relation-prior",
        default="",
        help="Comma-separated relation multipliers, same format as stream_alarm_mhp.py.",
    )
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        relation_prior = parse_topology_relation_prior(args.topology_relation_prior)
    except ValueError as exc:
        parser.error(str(exc))

    t0 = time.monotonic()
    artifact = load_alarm_mhp_artifact(args.model)
    scorer, mu_scorer, ne_graph_data = _build_runtime_scorers(
        artifact, args.ne_graph, args.site_graph, quiet=args.quiet
    )
    events, alarm_metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.site_graph,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
        regions=artifact.config.regions,
    )
    if DEVICE_DOMAIN_FIELD in tuple(artifact.config.type_fields):
        events, domain_stats = filter_and_annotate_device_domain(events, ne_graph_data)
        if not args.quiet:
            print(f"[period] domain filter: {domain_stats}", flush=True)

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
    min_events = (
        int(args.min_group_events)
        if args.min_group_events is not None
        else int(artifact.config.min_group_events)
    )
    config = PeriodStreamConfig(
        aggregation_wait_sec=aggregation_wait,
        period_idle_sec=args.period_idle_sec,
        history_window_sec=history,
        time_slack_sec=slack,
        late_penalty_half_life_sec=late_half_life,
        time_scale_sec=float(artifact.config.time_scale_sec),
        close_inactive_sec=args.close_inactive_sec,
        min_group_events=min_events,
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

    association_cache = None
    if args.association_cache:
        try:
            fingerprint = association_cache_fingerprint(
                args.model,
                args.ne_graph,
                args.site_graph,
                config,
                artifact.config.topology_node_field,
            )
            association_cache = load_association_cache(
                args.association_cache, expected_fingerprint=fingerprint
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"cannot load --association-cache: {exc}")
        if not args.quiet:
            cache_md = association_cache.get("metadata") or {}
            array_mib = sum(
                array.nbytes for array in association_cache.get("arrays", {}).values()
            ) / (1024 * 1024)
            print(
                f"[period] association cache loaded: "
                f"signatures={cache_md.get('signature_count', 0)}, "
                f"edges={cache_md.get('edge_count', 0)}, arrays={array_mib:.1f}MiB",
                flush=True,
            )

    engine = AlarmPeriodMHPAssigner(
        artifact,
        config,
        feature_scorer=scorer,
        mu_scorer=mu_scorer,
        association_cache=association_cache,
    )
    # The plan owns decoded edge objects now; release the JSON row arrays before
    # processing a potentially large alarm stream.
    association_cache = None
    if not args.quiet:
        print(
            f"[period] events={len(events)}, wait={config.aggregation_wait_sec:g}s, "
            f"idle={config.period_idle_sec:g}s, history={config.history_window_sec:g}s, "
            f"scope={config.candidate_scope}, dynamic={getattr(artifact.config, 'dynamic_alpha', 'off')}",
            flush=True,
        )

    for i, event in enumerate(events):
        engine.process(event)
        if args.progress_every and (i + 1) % args.progress_every == 0 and not args.quiet:
            stats = engine.stats()
            elapsed = time.monotonic() - t0
            print(
                f"[period] processed={i + 1}/{len(events)} "
                f"rate={(i + 1) / max(elapsed, EPS):.0f}/s "
                f"periods={stats['created_periods']} harvests={stats['harvest_count']} "
                f"groups={stats['active_group_count']}+{stats['closed_group_count']}",
                flush=True,
            )
    engine.flush()
    stats = engine.stats()
    elapsed = time.monotonic() - t0
    metadata = {
        "algorithm": "alarm_flow_mhp.alarm_period_stream",
        "model": os.path.abspath(args.model),
        "input": os.path.abspath(args.alarms),
        "association_cache": (
            os.path.abspath(args.association_cache) if args.association_cache else ""
        ),
        "alarm_metadata": alarm_metadata,
        "config": {
            key: value
            for key, value in vars(config).items()
        },
        "stats": stats,
        "elapsed_seconds": elapsed,
    }
    _write_json(args.groups_output, {"metadata": metadata, "groups": engine.closed_groups})
    if args.edges_output:
        edges = [edge for group in engine.closed_groups for edge in group.get("edges", ())]
        _write_jsonl(args.edges_output, edges)
    if args.visual_output:
        from alarm_flow_mhp.visual_output import AlarmMHPVisualOutputSession

        visual = AlarmMHPVisualOutputSession.from_files(
            args.visual_output,
            args.ne_graph,
            args.site_graph,
        )
        visual.reset_output_file()
        try:
            visual.emit_groups(engine.closed_groups, finalization_reason="stream_end")
        finally:
            visual.close()
    if not args.quiet:
        print(
            f"[period] done: groups={len(engine.closed_groups)}, "
            f"periods={stats['created_periods']}, harvests={stats['harvest_count']}, "
            f"preloaded_edges={stats['preloaded_edge_count']}, "
            f"incremental_edges={stats['compiled_pair_count']}, elapsed={elapsed:.2f}s; "
            f"output={args.groups_output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
