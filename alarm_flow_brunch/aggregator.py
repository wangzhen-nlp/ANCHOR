from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
from typing import Iterable

import numpy as np

from alarm_flow_isahp.sequences import (
    AlarmSequenceConfig,
    AlarmVocabs,
    alarm_type_from_title,
    build_alarm_sequences,
    build_alarm_vocabs,
    parse_type_fields,
)
from alarm_flow_isahp.ne_topology import NETopologyIndex
from brunch import BRUNCH, BRUNCHConfig, EventCollection, HawkesParams
from fault_grouping.alarm_events.io import is_clear_alarm
from alarm_flow_brunch.region_filter import filter_alarm_events_by_regions, parse_regions


TOPOLOGY_EDGE_POLICIES = frozenset({"off", "prefer", "require"})
PARENT_SELECTION_MODES = frozenset({"sample", "argmax"})
ARTIFACT_TYPE = "alarm_flow_brunch.v1"


@dataclass(frozen=True)
class AlarmBRUNCHConfig:
    """Configuration for BRUNCH-style alarm stream aggregation."""

    type_fields: tuple = ("alarm_source", "alarm_type")
    history_window_sec: float = 900.0
    max_history_events: int = 128
    min_events: int = 2
    time_scale_sec: float = 60.0
    include_clear: bool = False
    n_sweeps: int = 30
    burn_in: int = 5
    # Paper-faithful default: keep Θ frozen at _build_initial_params output and
    # only sample (B, C). Vanilla MLE between sweeps on a single MCMC sample is
    # numerically unstable on real alarm data and tends to collapse the chain
    # into an "all-immigrant" attractor. Opt in only after the initial-Θ prior
    # has been validated for the input.
    refit_params: bool = False
    warm_start: bool = True
    seed: int = 0
    sparse_alpha_threshold: float = 1e-4
    max_active_sources_per_dim: int = 16
    min_group_events: int = 1
    topology_edge_policy: str = "prefer"
    topology_prefer_multiplier: float = 2.0
    topology_fallback_sources_per_dim: int = 2
    regions: tuple = ()
    parent_selection: str = "sample"

    def __post_init__(self):
        object.__setattr__(self, "regions", parse_regions(self.regions))
        sequence_config = self.sequence_config()
        if self.n_sweeps < 1:
            raise ValueError("n_sweeps must be >= 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.max_active_sources_per_dim < 1:
            raise ValueError("max_active_sources_per_dim must be >= 1")
        if self.sparse_alpha_threshold < 0:
            raise ValueError("sparse_alpha_threshold must be non-negative")
        if self.min_group_events < 1:
            raise ValueError("min_group_events must be >= 1")
        if self.topology_edge_policy not in TOPOLOGY_EDGE_POLICIES:
            raise ValueError(f"topology_edge_policy must be one of {sorted(TOPOLOGY_EDGE_POLICIES)}")
        if self.topology_prefer_multiplier < 1.0:
            raise ValueError("topology_prefer_multiplier must be >= 1")
        if self.topology_fallback_sources_per_dim < 0:
            raise ValueError("topology_fallback_sources_per_dim must be >= 0")
        if self.parent_selection not in PARENT_SELECTION_MODES:
            raise ValueError(f"parent_selection must be one of {sorted(PARENT_SELECTION_MODES)}")
        del sequence_config  # consumed for its validation side effects above

    def sequence_config(self) -> AlarmSequenceConfig:
        return AlarmSequenceConfig(
            type_fields=tuple(self.type_fields),
            history_window_sec=self.history_window_sec,
            max_history_events=self.max_history_events,
            min_events=self.min_events,
            time_scale_sec=self.time_scale_sec,
            include_clear=self.include_clear,
        )

    def to_dict(self):
        payload = asdict(self)
        payload["type_fields"] = list(self.type_fields)
        payload["regions"] = list(self.regions)
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        if "type_fields" in payload:
            payload["type_fields"] = tuple(payload["type_fields"])
        if "regions" in payload:
            payload["regions"] = parse_regions(payload["regions"])
        return cls(**payload)


@dataclass
class AlarmBRUNCHOutput:
    groups: list
    edges: list
    metadata: dict
    trace: list
    params: HawkesParams

    def to_json_payload(self):
        return {
            "metadata": self.metadata,
            "groups": self.groups,
            "edges": self.edges,
            "trace": self.trace,
        }


@dataclass
class AlarmBRUNCHArtifact:
    params: HawkesParams
    vocabs: AlarmVocabs
    config: AlarmBRUNCHConfig
    training_metadata: dict
    trace: list

    def to_dict(self):
        return {
            "artifact_type": ARTIFACT_TYPE,
            "params": hawkes_params_to_dict(self.params),
            "vocabs": self.vocabs.to_dict(),
            "config": self.config.to_dict(),
            "training": dict(self.training_metadata or {}),
            "trace": list(self.trace or []),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(f"unsupported alarm BRUNCH artifact: {payload.get('artifact_type')}")
        return cls(
            params=hawkes_params_from_dict(payload["params"]),
            vocabs=AlarmVocabs.from_dict(payload["vocabs"]),
            config=AlarmBRUNCHConfig.from_dict(payload["config"]),
            training_metadata=dict(payload.get("training") or {}),
            trace=list(payload.get("trace") or []),
        )


def hawkes_params_to_dict(params: HawkesParams):
    edge_targets, edge_sources, edge_alpha, edge_beta = params.edge_values(include_self=True)
    return {
        "M": params.M,
        "mu": params.mu.tolist(),
        "edge_targets": edge_targets.astype(int).tolist(),
        "edge_sources": edge_sources.astype(int).tolist(),
        "edge_alpha": edge_alpha.astype(float).tolist(),
        "edge_beta": edge_beta.astype(float).tolist(),
        "links": list(params.links),
        "edge_threshold": params.edge_threshold,
        "max_active_sources_per_dim": params.max_active_sources_per_dim,
        "default_beta": params.default_beta,
    }


def hawkes_params_from_dict(payload) -> HawkesParams:
    payload = dict(payload or {})
    return HawkesParams.from_edges(
        M=int(payload["M"]),
        mu=np.asarray(payload["mu"], dtype=np.float64),
        edge_targets=np.asarray(payload.get("edge_targets", ()), dtype=np.int64),
        edge_sources=np.asarray(payload.get("edge_sources", ()), dtype=np.int64),
        edge_alpha=np.asarray(payload.get("edge_alpha", ()), dtype=np.float64),
        edge_beta=np.asarray(payload.get("edge_beta", ()), dtype=np.float64),
        links=list(payload.get("links") or ["linear"] * int(payload["M"])),
        edge_threshold=float(payload.get("edge_threshold", 0.0)),
        max_active_sources_per_dim=payload.get("max_active_sources_per_dim"),
        default_beta=float(payload.get("default_beta", 1.0)),
    )


def save_alarm_brunch_artifact(path, artifact: AlarmBRUNCHArtifact):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(artifact.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_alarm_brunch_artifact(path) -> AlarmBRUNCHArtifact:
    with open(path, "r", encoding="utf-8") as stream:
        return AlarmBRUNCHArtifact.from_dict(json.load(stream))


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


def _history_indices(times, target_index, window, max_history_events):
    target_time = times[target_index]
    history = []
    for source_index in range(target_index - 1, -1, -1):
        age = target_time - times[source_index]
        if age <= 0:
            continue
        if age > window:
            break
        history.append(source_index)
        if len(history) >= max_history_events:
            break
    return list(reversed(history))


def _topology_relation_score(source_event, target_event, topology_index):
    if topology_index is None:
        return 0.0
    source_ne = source_event.get("alarm_source", "")
    target_ne = target_event.get("alarm_source", "")
    if source_ne and source_ne == target_ne:
        return 1.0
    source_site = source_event.get("site_id", "")
    target_site = target_event.get("site_id", "")
    if source_site and source_site == target_site:
        return 0.9
    features = topology_index.pair_features(source_ne, target_ne)
    if not features:
        return 0.0
    # direct forward > direct reverse/bidirectional > reachable > undirected.
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


def _source_rank(counter, source_dim):
    value = counter.get(source_dim)
    if value is None:
        return (0.0, 0.0, 0.0)
    return value


def _build_initial_params(sequence, vocabs, config: AlarmBRUNCHConfig, topology_index=None):
    M = len(vocabs.type_vocab)
    times = np.asarray(sequence.times, dtype=np.float64)
    dims = np.asarray(sequence.type_ids, dtype=np.int64)
    horizon = max(float(times[-1] - times[0]) if len(times) else 0.0, 1.0)
    dim_counts = np.bincount(dims, minlength=M).astype(np.float64)
    mu = np.maximum(0.05 / horizon, 0.1 * dim_counts / horizon)

    pair_counts = Counter()
    pair_dt_sums = defaultdict(float)
    by_target_sources = [Counter() for _ in range(M)]
    topo_pair_counts = Counter()
    topo_pair_scores = defaultdict(float)
    by_target_topo_sources = [Counter() for _ in range(M)]
    window = config.history_window_sec / config.time_scale_sec
    for target_index in range(len(times)):
        target_dim = int(dims[target_index])
        target_event = sequence.events[target_index]
        for source_index in _history_indices(
            times,
            target_index,
            window,
            config.max_history_events,
        ):
            source_dim = int(dims[source_index])
            dt = float(times[target_index] - times[source_index])
            key = (target_dim, source_dim)
            pair_counts[key] += 1
            pair_dt_sums[key] += dt
            by_target_sources[target_dim][source_dim] += 1
            topo_score = _topology_relation_score(
                sequence.events[source_index],
                target_event,
                topology_index,
            )
            if topo_score > 0.0:
                topo_pair_counts[key] += 1
                topo_pair_scores[key] += topo_score
                by_target_topo_sources[target_dim][source_dim] += 1

    edge_targets = []
    edge_sources = []
    edge_alpha = []
    edge_beta = []
    for target_dim in range(M):
        source_candidates = {target_dim}
        if config.topology_edge_policy == "off" or topology_index is None:
            ranked_sources = [
                source
                for source, _count in by_target_sources[target_dim].most_common(
                    config.max_active_sources_per_dim
                )
            ]
        else:
            topo_ranked_sources = [
                source
                for source, _count in by_target_topo_sources[target_dim].most_common(
                    config.max_active_sources_per_dim
                )
            ]
            ranked_sources = list(topo_ranked_sources)
            if config.topology_edge_policy == "prefer":
                fallback_limit = max(0, config.topology_fallback_sources_per_dim)
                fallback_limit = min(fallback_limit, config.max_active_sources_per_dim)
                if fallback_limit > 0:
                    for source, _count in by_target_sources[target_dim].most_common(
                        config.max_active_sources_per_dim
                    ):
                        if source in ranked_sources:
                            continue
                        ranked_sources.append(source)
                        fallback_limit -= 1
                        if fallback_limit <= 0:
                            break
            ranked_sources = ranked_sources[: config.max_active_sources_per_dim]
        source_candidates.update(ranked_sources)
        for source_dim in sorted(source_candidates):
            key = (target_dim, source_dim)
            count = float(pair_counts.get(key, 0.0))
            if (
                source_dim != target_dim
                and config.topology_edge_policy == "require"
                and topology_index is not None
                and topo_pair_counts.get(key, 0) <= 0
            ):
                continue
            source_count = max(float(dim_counts[source_dim]), 1.0)
            alpha = (count + 0.25) / (source_count + 2.0)
            if topology_index is not None and config.topology_edge_policy != "off":
                topo_count = float(topo_pair_counts.get(key, 0.0))
                if topo_count > 0.0:
                    avg_topo_score = topo_pair_scores[key] / topo_count
                    alpha *= 1.0 + (config.topology_prefer_multiplier - 1.0) * avg_topo_score
                elif source_dim != target_dim and config.topology_edge_policy == "prefer":
                    alpha *= 0.5
            alpha = max(alpha, config.sparse_alpha_threshold * 10.0)
            beta = (count + 1.0) / (pair_dt_sums.get(key, 0.0) + 1.0)
            edge_targets.append(target_dim)
            edge_sources.append(source_dim)
            edge_alpha.append(alpha)
            edge_beta.append(beta)

    edge_targets_arr = np.asarray(edge_targets, dtype=np.int64)
    edge_sources_arr = np.asarray(edge_sources, dtype=np.int64)
    edge_alpha_arr = np.asarray(edge_alpha, dtype=np.float64)
    edge_beta_arr = np.asarray(edge_beta, dtype=np.float64)

    # Stationarity cap: Hawkes processes require ρ(α) < 1; otherwise the cluster
    # Poisson interpretation diverges and the candidate scores blow up so much
    # that real events deterministically pick the highest-α candidate as parent
    # (no real competition with μ or the kernel decay). Match brunch's MLE cap
    # of 0.95 to leave a safety margin. We rescale α uniformly — preserves the
    # relative ranking of edges (so topology-preferred edges remain dominant)
    # while restoring stationarity.
    STABILITY_RADIUS = 0.95
    tmp_params = HawkesParams.from_edges(
        M=M,
        mu=mu,
        edge_targets=edge_targets_arr,
        edge_sources=edge_sources_arr,
        edge_alpha=edge_alpha_arr,
        edge_beta=edge_beta_arr,
        links=["linear"] * M,
        edge_threshold=config.sparse_alpha_threshold,
        max_active_sources_per_dim=config.max_active_sources_per_dim,
    )
    rho = tmp_params.spectral_radius()
    if rho > STABILITY_RADIUS and rho > 0.0:
        scale = STABILITY_RADIUS / rho
        edge_alpha_arr = edge_alpha_arr * scale
        print(
            f"[_build_initial_params] α 矩阵 spectral radius ρ={rho:.2f} 超出 "
            f"stationarity 阈值 {STABILITY_RADIUS}，统一缩放 α × {scale:.4f} "
            f"使 ρ≈{STABILITY_RADIUS}（保持边之间的相对权重）"
        )
        return HawkesParams.from_edges(
            M=M,
            mu=mu,
            edge_targets=edge_targets_arr,
            edge_sources=edge_sources_arr,
            edge_alpha=edge_alpha_arr,
            edge_beta=edge_beta_arr,
            links=["linear"] * M,
            edge_threshold=config.sparse_alpha_threshold,
            max_active_sources_per_dim=config.max_active_sources_per_dim,
        )
    return tmp_params


def _build_event_collection(sequence, M):
    times = np.asarray(sequence.times, dtype=np.float64)
    dims = np.asarray(sequence.type_ids, dtype=np.int64)
    horizon = float(times[-1]) + 1e-6 if len(times) else 1.0
    return EventCollection(times=times, dims=dims, M=M, T=horizon)


def _build_sequences(sorted_alarm_events, vocabs, sequence_config, *, topology_index=None):
    sequences, sequence_stats = build_alarm_sequences(
        sorted_alarm_events,
        vocabs,
        sequence_config,
        add_missing_types=False,
        topology_index=topology_index,
    )
    if not sequences:
        raise ValueError("no global alarm flow survived preprocessing; relax min-events or inspect input alarms")
    return sequences, sequence_stats


def _fit_sequence(sequence, vocabs, config: AlarmBRUNCHConfig, *, init_params=None, topology_index=None):
    return _fit_sequence_with_progress(
        sequence,
        vocabs,
        config,
        init_params=init_params,
        topology_index=topology_index,
    )


def _fit_sequence_with_progress(
    sequence,
    vocabs,
    config: AlarmBRUNCHConfig,
    *,
    init_params=None,
    topology_index=None,
    verbose=False,
    log_every=10,
    progress_every=50000,
    sweep_callback=None,
):
    M = len(vocabs.type_vocab)
    events = _build_event_collection(sequence, M)
    if init_params is None:
        init_params = _build_initial_params(sequence, vocabs, config, topology_index=topology_index)
    brunch_config = BRUNCHConfig(
        M=M,
        window=config.history_window_sec / config.time_scale_sec,
        n_sweeps=config.n_sweeps,
        burn_in=config.burn_in,
        refit_params=config.refit_params,
        warm_start=config.warm_start,
        links=["linear"] * M,
        seed=config.seed,
        sparse_alpha_threshold=config.sparse_alpha_threshold,
        max_active_sources_per_dim=config.max_active_sources_per_dim,
        sparse_parameter_storage=True,
        materialize_branching_matrix=False,
        verbose=verbose,
        log_every=log_every,
        progress_every=progress_every,
        parent_selection=config.parent_selection,
        sweep_callback=sweep_callback,
    )
    return BRUNCH(brunch_config).fit(events, init_params=init_params)


def _emit_progress(progress_callback, progress_stage, **payload):
    if progress_callback is not None:
        progress_callback(progress_stage, payload)


def _event_type_counts(sequence):
    return {
        label: count
        for label, count in sorted(
            Counter(sequence.type_labels).items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _group_records(sequence, result, *, min_group_events=1):
    by_cascade = defaultdict(list)
    for index, cascade_id in enumerate(result.cascade_of):
        by_cascade[int(cascade_id)].append(index)

    groups = []
    for ordinal, (cascade_id, indices) in enumerate(
        sorted(by_cascade.items(), key=lambda item: min(item[1])),
        start=1,
    ):
        if len(indices) < min_group_events:
            continue
        events = [sequence.events[index] for index in indices]
        summaries = [summarize_alarm_event(event, index) for index, event in zip(indices, events)]
        parent_indices = [int(result.parent_of[index]) for index in indices]
        root_index = min(
            indices,
            key=lambda index: (
                result.parent_of[index] != index,
                float(sequence.events[index].get("ts", 0.0)),
                index,
            ),
        )
        group_edges = []
        for child_index, parent_index in zip(indices, parent_indices):
            if parent_index == child_index:
                continue
            if parent_index not in indices:
                continue
            group_edges.append(
                {
                    "source_index": parent_index,
                    "target_index": child_index,
                    "source_event_id": _event_id(sequence.events[parent_index], parent_index),
                    "target_event_id": _event_id(sequence.events[child_index], child_index),
                    "source_type": sequence.type_labels[parent_index],
                    "target_type": sequence.type_labels[child_index],
                }
            )
        timestamps = [summary["ts"] for summary in summaries]
        groups.append(
            {
                "group_id": f"brunch-{ordinal:06d}",
                "cascade_id": cascade_id,
                "event_count": len(indices),
                "start_ts": min(timestamps),
                "end_ts": max(timestamps),
                "duration_sec": max(timestamps) - min(timestamps),
                "root_event": summarize_alarm_event(sequence.events[root_index], root_index),
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
                "edges": group_edges,
            }
        )
    return groups


def _edge_records(sequence, result):
    edges = []
    for source_index, target_index in result.branching_edges:
        source_index = int(source_index)
        target_index = int(target_index)
        edges.append(
            {
                "source_index": source_index,
                "target_index": target_index,
                "source_event_id": _event_id(sequence.events[source_index], source_index),
                "target_event_id": _event_id(sequence.events[target_index], target_index),
                "source_type": sequence.type_labels[source_index],
                "target_type": sequence.type_labels[target_index],
                "source_event": summarize_alarm_event(sequence.events[source_index], source_index),
                "target_event": summarize_alarm_event(sequence.events[target_index], target_index),
            }
        )
    return edges


def _region_filter_events(
    sorted_alarm_events,
    config: AlarmBRUNCHConfig,
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


def _output_from_fit(
    sequence,
    vocabs,
    config,
    result,
    sequence_stats,
    considered_event_count,
    *,
    topology_index=None,
    region_filter_stats=None,
):
    groups = _group_records(sequence, result, min_group_events=config.min_group_events)
    edges = _edge_records(sequence, result)
    metadata = {
        "algorithm": "alarm_flow_brunch",
        "config": config.to_dict(),
        "sequence_config": config.sequence_config().to_dict(),
        "considered_event_count": considered_event_count,
        "region_filter": region_filter_stats or {},
        "sequence_stats": sequence_stats,
        "modeled_event_count": len(sequence.events),
        "type_count": len(vocabs.type_vocab),
        "active_edge_count": len(result.params.active_edges()[0]),
        "topology_edge_policy": config.topology_edge_policy,
        "topology_max_hops": getattr(topology_index, "max_hops", None),
        "group_count": len(groups),
        "branching_edge_count": len(edges),
        "event_type_counts": _event_type_counts(sequence),
        "type_labels": list(vocabs.type_vocab.labels),
        "best_log_likelihood": result.best_log_likelihood,
    }
    return AlarmBRUNCHOutput(
        groups=groups,
        edges=edges,
        metadata=metadata,
        trace=result.trace,
        params=result.params,
    )


def train_alarm_brunch(
    sorted_alarm_events: Iterable[dict],
    config: AlarmBRUNCHConfig | None = None,
    topology_index=None,
    ne_graph_data=None,
    progress_callback=None,
    verbose=False,
    log_every=10,
    region_filter_stats=None,
    progress_every=50000,
    checkpoint_callback=None,
) -> AlarmBRUNCHArtifact:
    """Fit reusable BRUNCH type-level parameters from an ordered alarm stream."""
    config = config or AlarmBRUNCHConfig()
    sequence_config = config.sequence_config()
    sorted_alarm_events = list(sorted_alarm_events)
    sorted_alarm_events, region_filter_stats = _region_filter_events(
        sorted_alarm_events,
        config,
        ne_graph_data,
        region_filter_stats=region_filter_stats,
    )
    _emit_progress(progress_callback, "region_filter", **region_filter_stats)
    vocabs, considered_event_count = build_alarm_vocabs(sorted_alarm_events, sequence_config)
    _emit_progress(
        progress_callback,
        "vocab",
        considered_event_count=considered_event_count,
        type_count=len(vocabs.type_vocab),
        alarm_source_count=len(vocabs.alarm_source_vocab),
        alarm_type_count=len(vocabs.alarm_type_vocab),
    )
    sequences, sequence_stats = _build_sequences(
        sorted_alarm_events,
        vocabs,
        sequence_config,
        topology_index=None,
    )
    _emit_progress(progress_callback, "sequence", **sequence_stats)
    sequence = sequences[0]
    _emit_progress(
        progress_callback,
        "fit_start",
        modeled_event_count=len(sequence.events),
        type_count=len(vocabs.type_vocab),
        n_sweeps=config.n_sweeps,
        burn_in=config.burn_in,
    )

    def on_sweep_checkpoint(payload):
        if checkpoint_callback is None:
            return
        params = payload["checkpoint_params"]
        training_metadata = {
            "considered_event_count": considered_event_count,
            "region_filter": region_filter_stats,
            "sequence_stats": sequence_stats,
            "modeled_event_count": len(sequence.events),
            "type_count": len(vocabs.type_vocab),
            "active_edge_count": len(params.active_edges()[0]),
            "topology_edge_policy": config.topology_edge_policy,
            "topology_max_hops": getattr(topology_index, "max_hops", None),
            "best_log_likelihood": payload["best_log_likelihood"],
            "event_type_counts": _event_type_counts(sequence),
            "type_labels": list(vocabs.type_vocab.labels),
            "checkpoint": {
                "sweep": payload["sweep"],
                "sweep1": payload["sweep"] + 1,
                "log_likelihood": payload["log_likelihood"],
                "num_clusters": payload["num_clusters"],
                "num_cascades": payload["num_cascades"],
                "is_best": payload["is_best"],
                "post_burn_in_best": payload["checkpoint_is_post_burn_in"],
            },
        }
        artifact = AlarmBRUNCHArtifact(
            params=params,
            vocabs=vocabs,
            config=config,
            training_metadata=training_metadata,
            trace=payload.get("trace") or [],
        )
        checkpoint_callback(artifact, payload)

    result = _fit_sequence_with_progress(
        sequence,
        vocabs,
        config,
        topology_index=topology_index,
        verbose=verbose,
        log_every=log_every,
        progress_every=progress_every,
        sweep_callback=on_sweep_checkpoint if checkpoint_callback is not None else None,
    )
    _emit_progress(
        progress_callback,
        "fit_done",
        active_edge_count=len(result.params.active_edges()[0]),
        best_log_likelihood=result.best_log_likelihood,
    )
    training_metadata = {
        "considered_event_count": considered_event_count,
        "region_filter": region_filter_stats,
        "sequence_stats": sequence_stats,
        "modeled_event_count": len(sequence.events),
        "type_count": len(vocabs.type_vocab),
        "active_edge_count": len(result.params.active_edges()[0]),
        "topology_edge_policy": config.topology_edge_policy,
        "topology_max_hops": getattr(topology_index, "max_hops", None),
        "best_log_likelihood": result.best_log_likelihood,
        "event_type_counts": _event_type_counts(sequence),
        "type_labels": list(vocabs.type_vocab.labels),
    }
    return AlarmBRUNCHArtifact(
        params=result.params,
        vocabs=vocabs,
        config=config,
        training_metadata=training_metadata,
        trace=result.trace,
    )


def infer_alarm_flow(
    sorted_alarm_events: Iterable[dict],
    artifact: AlarmBRUNCHArtifact,
    *,
    config: AlarmBRUNCHConfig | None = None,
    topology_index=None,
    ne_graph_data=None,
    region_filter_stats=None,
) -> AlarmBRUNCHOutput:
    """Infer fault groups with a trained BRUNCH artifact."""
    config = config or replace(artifact.config, refit_params=False)
    sequence_config = config.sequence_config()
    sorted_alarm_events = list(sorted_alarm_events)
    sorted_alarm_events, region_filter_stats = _region_filter_events(
        sorted_alarm_events,
        config,
        ne_graph_data,
        region_filter_stats=region_filter_stats,
    )
    sequences, sequence_stats = _build_sequences(
        sorted_alarm_events,
        artifact.vocabs,
        sequence_config,
        topology_index=None,
    )
    sequence = sequences[0]
    considered_event_count = sequence_stats.get("input_event_count", len(sequence.events))
    result = _fit_sequence(
        sequence,
        artifact.vocabs,
        config,
        init_params=artifact.params,
        topology_index=topology_index,
    )
    return _output_from_fit(
        sequence,
        artifact.vocabs,
        config,
        result,
        sequence_stats,
        considered_event_count,
        topology_index=topology_index,
        region_filter_stats=region_filter_stats,
    )


def aggregate_alarm_flow(
    sorted_alarm_events: Iterable[dict],
    config: AlarmBRUNCHConfig | None = None,
    topology_index=None,
    ne_graph_data=None,
    region_filter_stats=None,
) -> AlarmBRUNCHOutput:
    """Infer fault groups from an ordered alarm stream via BRUNCH branching."""
    config = config or AlarmBRUNCHConfig()
    sequence_config = config.sequence_config()
    sorted_alarm_events = list(sorted_alarm_events)
    sorted_alarm_events, region_filter_stats = _region_filter_events(
        sorted_alarm_events,
        config,
        ne_graph_data,
        region_filter_stats=region_filter_stats,
    )
    vocabs, considered_event_count = build_alarm_vocabs(sorted_alarm_events, sequence_config)
    sequences, sequence_stats = _build_sequences(
        sorted_alarm_events,
        vocabs,
        sequence_config,
        topology_index=None,
    )
    sequence = sequences[0]
    result = _fit_sequence(sequence, vocabs, config, topology_index=topology_index)
    return _output_from_fit(
        sequence,
        vocabs,
        config,
        result,
        sequence_stats,
        considered_event_count,
        topology_index=topology_index,
        region_filter_stats=region_filter_stats,
    )
