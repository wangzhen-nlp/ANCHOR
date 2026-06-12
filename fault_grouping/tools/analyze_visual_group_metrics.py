#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute unsupervised diagnostics for online fault-group visual JSONL.

The input is the visual JSONL emitted by stream inference. The metrics are not
ground truth quality scores; they are stable, comparable diagnostics for judging
whether one grouping run is more compact, more topology-explainable, or more
aggressive than another run.

Example:
    python fault_grouping/tools/analyze_visual_group_metrics.py stream.visual.jsonl -o metrics.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from alarm_flow_isahp.ne_topology import NETopologyIndex
from alarm_flow_mhp.topology_relation_prior import RELATION_KEYS, classify_topology_relation
from ne_link_learning.core import build_graph_context
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


DEFAULT_DURATION_BUCKETS_SEC = (
    0,
    60,
    5 * 60,
    15 * 60,
    30 * 60,
    60 * 60,
    2 * 60 * 60,
    6 * 60 * 60,
    12 * 60 * 60,
    24 * 60 * 60,
)

RELATION_SCORE = {
    "same_device": 1.0,
    "direct": 0.9,
    "same_site": 0.75,
    "indirect": 0.5,
    "cross_site": 0.2,
    "unknown": 0.05,
}

HEALTH_WEIGHTS = {
    "topology_explainability": 0.30,
    "time_compactness": 0.20,
    "size_reasonableness": 0.15,
    "singleton_control": 0.15,
    "virtual_reasonableness": 0.10,
    "risk_cleanliness": 0.10,
}


@dataclass
class GroupMetrics:
    line_num: int
    uuid: str
    rule: str
    event_count: int
    real_event_count: int
    virtual_event_count: int
    site_count: int
    ne_count: int
    alarm_type_count: int
    duration_sec: float
    dominant_alarm_type_ratio: float
    missing_edge_count: int
    pair_count: int
    pair_relation_counts: Counter
    edge_relation_counts: Counter
    topology_cohesion_score: float | None
    risk_flags: list[str]


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _load_json(path):
    if not Path(path).exists():
        raise SystemExit(f"资源文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path):
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"文件不存在: {path}")
    with source.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过第 {line_num} 行 JSON 解析失败: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"跳过第 {line_num} 行：JSON 顶层不是对象", file=sys.stderr)
                continue
            yield line_num, record


def _parse_time_text(value):
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(text[:26], fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _coerce_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts if ts > 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = float(text)
        return ts if ts > 0 else None
    except ValueError:
        return _parse_time_text(text)


def _symptom_ts(symptom):
    for key in ("ts", "_segment_start_ts", "first_ts", "first_occurrence_ts"):
        ts = _coerce_ts(symptom.get(key))
        if ts is not None:
            return ts
    for key in ("alarm_time", "time_str", "告警首次发生时间", "first_occurrence_time"):
        ts = _parse_time_text(symptom.get(key))
        if ts is not None:
            return ts
    return None


def group_id(record):
    match_info = _as_dict(record.get("match_info"))
    cascade_info = _as_dict(record.get("cascade_info"))
    return str(
        record.get("uuid")
        or record.get("group_id")
        or record.get("cascade_id")
        or match_info.get("uuid")
        or cascade_info.get("cascade_id")
        or ""
    )


def group_rule(record):
    match_info = _as_dict(record.get("match_info"))
    return str(record.get("rule") or match_info.get("rule") or "")


def extract_symptoms(record):
    symptoms = []
    for source in (record.get("symptoms"), _as_dict(record.get("match_info")).get("symptoms")):
        for symptom in _as_list(source):
            if isinstance(symptom, dict):
                symptoms.append(symptom)

    # Fallback for non-compact visual output where alarms live under ne_info.
    if not symptoms:
        for ne_id, ne_meta in _as_dict(record.get("ne_info")).items():
            if not isinstance(ne_meta, dict):
                continue
            for alarm in _as_list(ne_meta.get("alarm")):
                if isinstance(alarm, dict):
                    symptom = dict(alarm)
                    symptom.setdefault("alarm_source", ne_id)
                    symptom.setdefault("node", ne_meta.get("site_id", ""))
                    symptoms.append(symptom)
    return symptoms


def _symptom_ne(symptom):
    return str(
        symptom.get("alarm_source")
        or symptom.get("source_ne")
        or symptom.get("ne")
        or symptom.get("ne_id")
        or ""
    ).strip()


def _symptom_site(symptom, ne_to_site):
    site = str(symptom.get("site_id") or symptom.get("node") or "").strip()
    if site:
        return site
    return ne_to_site.get(_symptom_ne(symptom), "")


def _symptom_alarm_type(symptom):
    return str(
        symptom.get("alarm_type")
        or symptom.get("alarm")
        or symptom.get("alarm_title")
        or ""
    ).strip()


def _symptom_event_id(symptom):
    return str(symptom.get("eid") or symptom.get("event_id") or symptom.get("alarm_id") or "").strip()


def is_virtual_symptom(symptom):
    return bool(
        symptom.get("virtual")
        or symptom.get("__virtual__")
        or symptom.get("inferred_virtual")
        or symptom.get("latent")
    )


def missing_topology_edges(record):
    match_info = _as_dict(record.get("match_info"))
    edges = []
    for source in (record.get("missing_topology_edges"), match_info.get("missing_topology_edges")):
        for edge in _as_list(source):
            if isinstance(edge, dict):
                edges.append(edge)
    return edges


def _edge_ne(edge, side):
    return str(
        edge.get(f"{side}_ne")
        or edge.get(f"{side}_node")
        or edge.get(f"{side}_ne_id")
        or ""
    ).strip()


def _edge_relation(edge, topology_index, node_infos):
    relation = str(edge.get("relation") or edge.get("predicted_relation") or "").strip().lower()
    if relation.startswith("mhp_"):
        relation = relation[4:]
    if relation.endswith("_topology"):
        relation = relation[: -len("_topology")]
    if relation in RELATION_KEYS:
        return relation
    if relation in {"hawkes_unknown_context", "unknown_context", "missing"}:
        return "unknown"
    source_ne = _edge_ne(edge, "source")
    target_ne = _edge_ne(edge, "target")
    return classify_topology_relation(source_ne, target_ne, topology_index, node_infos)


def _percentiles(values, percentiles=(50, 75, 90, 95, 99)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    n = len(vals)
    out = {}
    for p in percentiles:
        idx = int((p / 100.0) * (n - 1))
        idx = max(0, min(idx, n - 1))
        out[f"p{p}"] = vals[idx]
    return out


def _numeric_summary(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "percentiles": _percentiles(vals),
    }


def _ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _counter_payload(counter, total=None):
    total = int(total if total is not None else sum(counter.values()))
    return {
        str(k): {"count": int(v), "ratio": _ratio(v, total)}
        for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))
    }


def _bucket_label(upper_sec, prev_sec=None):
    def fmt(sec):
        sec = float(sec)
        if sec < 60:
            return f"{sec:.0f}s"
        if sec < 3600:
            return f"{sec / 60:.0f}m"
        if sec < 86400:
            return f"{sec / 3600:.0f}h"
        return f"{sec / 86400:.0f}d"

    if upper_sec == 0:
        return "0s"
    if prev_sec is None or prev_sec <= 0:
        return f"(0,{fmt(upper_sec)}]"
    return f"({fmt(prev_sec)},{fmt(upper_sec)}]"


def _histogram(values, max_explicit=None, buckets=None):
    if buckets is not None:
        bucket_values = sorted(float(b) for b in buckets)
        hist = Counter()
        for value in values:
            placed = False
            prev = None
            for upper in bucket_values:
                if value <= upper:
                    hist[_bucket_label(upper, prev)] += 1
                    placed = True
                    break
                prev = upper
            if not placed and bucket_values:
                hist[f">{_bucket_label(bucket_values[-1]).split(',')[-1].rstrip(']')}"] += 1
        return dict(hist)

    hist = Counter()
    for value in values:
        ivalue = int(value)
        if max_explicit is not None and ivalue > max_explicit:
            hist[f">{max_explicit}"] += 1
        else:
            hist[ivalue] += 1
    return dict(sorted(hist.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))


def _topology_cohesion_score(relation_counts):
    total = sum(relation_counts.values())
    if total <= 0:
        return None
    return sum(RELATION_SCORE.get(rel, 0.0) * cnt for rel, cnt in relation_counts.items()) / total


def _clip_score(value):
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(100.0, float(value)))


def _exp_score(value, target):
    if target <= 0:
        return 0.0
    return 100.0 * math.exp(-max(float(value), 0.0) / float(target))


def _relation_explainability(relation_counts):
    total = sum(relation_counts.values())
    if total <= 0:
        return 100.0
    return _clip_score(100.0 * _topology_cohesion_score(relation_counts))


def _size_reasonableness_score(dist, *, target_p50, target_p90, target_p99):
    pct = dist.get("percentiles", {}) if isinstance(dist, dict) else {}
    p50 = float(pct.get("p50", dist.get("median", 0.0) or 0.0))
    p90 = float(pct.get("p90", 0.0) or 0.0)
    p99 = float(pct.get("p99", p90) or p90)

    # Reward having non-singleton groups, then penalize increasingly large tails.
    lower = min(p50 / max(float(target_p50), 1.0), 1.0)
    p90_score = math.exp(-max(0.0, p90 - target_p90) / max(float(target_p90), 1.0))
    p99_score = math.exp(-max(0.0, p99 - target_p99) / max(float(target_p99), 1.0))
    return _clip_score(100.0 * (0.35 * lower + 0.40 * p90_score + 0.25 * p99_score))


def build_health_score(
    result,
    *,
    target_duration_sec=3600.0,
    target_virtual_ratio=0.2,
    target_size_p50=2.0,
    target_size_p90=20.0,
    target_size_p99=100.0,
):
    """Build an interpretable 0-100 health score from aggregate diagnostics."""
    overall = result.get("overall", {})
    dist = result.get("distributions", {})
    topo = result.get("topology", {})

    relation_counts = Counter()
    edge_dist = topo.get("missing_edge_relation_distribution") or {}
    pair_dist = topo.get("pair_relation_distribution") or {}
    if edge_dist:
        for rel, item in edge_dist.items():
            relation_counts[rel] += int(_as_dict(item).get("count", 0))
    if not relation_counts and pair_dist:
        for rel, item in pair_dist.items():
            relation_counts[rel] += int(_as_dict(item).get("count", 0))

    duration_p90 = float(dist.get("duration_sec", {}).get("percentiles", {}).get("p90", 0.0) or 0.0)
    singleton_ratio = float(overall.get("singleton_real_group_ratio", 0.0) or 0.0)
    virtual_ratio = float(overall.get("virtual_event_ratio", 0.0) or 0.0)
    flagged_ratio = max(
        (
            float(_as_dict(item).get("ratio", 0.0) or 0.0)
            for item in (result.get("risk_flags") or {}).values()
        ),
        default=0.0,
    )

    components = {
        "topology_explainability": _relation_explainability(relation_counts),
        "time_compactness": _exp_score(duration_p90, target_duration_sec),
        "size_reasonableness": _size_reasonableness_score(
            dist.get("real_event_count", {}),
            target_p50=target_size_p50,
            target_p90=target_size_p90,
            target_p99=target_size_p99,
        ),
        "singleton_control": _clip_score(100.0 * (1.0 - singleton_ratio)),
        "virtual_reasonableness": _exp_score(virtual_ratio, target_virtual_ratio),
        "risk_cleanliness": _clip_score(100.0 * (1.0 - flagged_ratio)),
    }
    score = sum(HEALTH_WEIGHTS[name] * components[name] for name in HEALTH_WEIGHTS)
    return {
        "grouping_health_score": _clip_score(score),
        "components": components,
        "weights": dict(HEALTH_WEIGHTS),
        "targets": {
            "duration_p90_sec": float(target_duration_sec),
            "virtual_event_ratio": float(target_virtual_ratio),
            "real_event_count_p50": float(target_size_p50),
            "real_event_count_p90": float(target_size_p90),
            "real_event_count_p99": float(target_size_p99),
        },
        "interpretation": "0-100; higher is healthier. This is a label-free proxy for comparing runs, not ground truth.",
    }


def _risk_flags(metrics: GroupMetrics, *, max_duration_sec, max_site_count, max_unknown_pair_ratio):
    flags = []
    if metrics.real_event_count <= 1:
        flags.append("singleton_or_no_real_event")
    if metrics.duration_sec > max_duration_sec:
        flags.append("long_duration")
    if metrics.site_count > max_site_count:
        flags.append("many_sites")
    if metrics.pair_count:
        unknown_ratio = metrics.pair_relation_counts.get("unknown", 0) / metrics.pair_count
        if unknown_ratio > max_unknown_pair_ratio:
            flags.append("high_unknown_pair_ratio")
    if metrics.virtual_event_count > metrics.real_event_count:
        flags.append("virtual_heavy")
    return flags


def analyze_record(
    line_num,
    record,
    *,
    topology_index,
    node_infos,
    ne_to_site,
    max_pairwise_ne,
    risk_duration_sec,
    risk_site_count,
    risk_unknown_pair_ratio,
):
    symptoms = extract_symptoms(record)
    real_symptoms = [s for s in symptoms if not is_virtual_symptom(s)]
    virtual_symptoms = [s for s in symptoms if is_virtual_symptom(s)]
    timestamps = [_symptom_ts(s) for s in symptoms]
    timestamps = [t for t in timestamps if t is not None]
    duration_sec = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0

    nes = sorted({_symptom_ne(s) for s in symptoms if _symptom_ne(s)})
    sites = sorted({_symptom_site(s, ne_to_site) for s in symptoms if _symptom_site(s, ne_to_site)})
    alarm_types = [_symptom_alarm_type(s) for s in symptoms if _symptom_alarm_type(s)]
    alarm_counter = Counter(alarm_types)
    dominant_ratio = (
        max(alarm_counter.values()) / len(alarm_types)
        if alarm_types else 0.0
    )

    edge_relations = Counter()
    for edge in missing_topology_edges(record):
        edge_relations[_edge_relation(edge, topology_index, node_infos)] += 1

    pair_relations = Counter()
    pair_count = 0
    if len(nes) <= max_pairwise_ne:
        for source_ne, target_ne in combinations(nes, 2):
            pair_relations[classify_topology_relation(source_ne, target_ne, topology_index, node_infos)] += 1
            pair_count += 1

    metrics = GroupMetrics(
        line_num=line_num,
        uuid=group_id(record),
        rule=group_rule(record),
        event_count=len(symptoms),
        real_event_count=len(real_symptoms),
        virtual_event_count=len(virtual_symptoms),
        site_count=len(sites),
        ne_count=len(nes),
        alarm_type_count=len(alarm_counter),
        duration_sec=max(0.0, duration_sec),
        dominant_alarm_type_ratio=dominant_ratio,
        missing_edge_count=len(missing_topology_edges(record)),
        pair_count=pair_count,
        pair_relation_counts=pair_relations,
        edge_relation_counts=edge_relations,
        topology_cohesion_score=_topology_cohesion_score(edge_relations or pair_relations),
        risk_flags=[],
    )
    metrics.risk_flags = _risk_flags(
        metrics,
        max_duration_sec=risk_duration_sec,
        max_site_count=risk_site_count,
        max_unknown_pair_ratio=risk_unknown_pair_ratio,
    )
    return metrics


def _detail_payload(metrics):
    return {
        "line_num": metrics.line_num,
        "uuid": metrics.uuid,
        "rule": metrics.rule,
        "event_count": metrics.event_count,
        "real_event_count": metrics.real_event_count,
        "virtual_event_count": metrics.virtual_event_count,
        "site_count": metrics.site_count,
        "ne_count": metrics.ne_count,
        "alarm_type_count": metrics.alarm_type_count,
        "duration_sec": metrics.duration_sec,
        "dominant_alarm_type_ratio": metrics.dominant_alarm_type_ratio,
        "missing_edge_count": metrics.missing_edge_count,
        "pair_count": metrics.pair_count,
        "pair_relation_counts": dict(metrics.pair_relation_counts),
        "edge_relation_counts": dict(metrics.edge_relation_counts),
        "topology_cohesion_score": metrics.topology_cohesion_score,
        "risk_flags": list(metrics.risk_flags),
    }


def analyze(
    visual_jsonl,
    *,
    ne_graph_path=NE_GRAPH_JSON,
    site_graph_path=SITE_GRAPH_JSON,
    topo_max_hops=3,
    max_pairwise_ne=200,
    risk_duration_sec=2 * 3600,
    risk_site_count=10,
    risk_unknown_pair_ratio=0.5,
    health_target_duration_sec=3600.0,
    health_target_virtual_ratio=0.2,
    health_target_size_p50=2.0,
    health_target_size_p90=20.0,
    health_target_size_p99=100.0,
    include_details=True,
):
    ne_graph_data = _load_json(ne_graph_path)
    # Site graph is currently only recorded for reproducibility. Site ids used by
    # metrics come from symptoms/ne_graph, because visual output is NE-centric.
    site_graph_exists = Path(site_graph_path).exists()
    graph_context = build_graph_context(ne_graph_data)
    topology_index = NETopologyIndex.from_graph(ne_graph_data, max_hops=topo_max_hops)
    node_infos = getattr(graph_context, "node_infos", {}) or {}
    ne_to_site = {
        ne_id: str(getattr(info, "site_id", "") or "")
        for ne_id, info in node_infos.items()
    }

    groups = []
    skipped = 0
    for line_num, record in _iter_jsonl(visual_jsonl):
        try:
            groups.append(
                analyze_record(
                    line_num,
                    record,
                    topology_index=topology_index,
                    node_infos=node_infos,
                    ne_to_site=ne_to_site,
                    max_pairwise_ne=max_pairwise_ne,
                    risk_duration_sec=risk_duration_sec,
                    risk_site_count=risk_site_count,
                    risk_unknown_pair_ratio=risk_unknown_pair_ratio,
                )
            )
        except Exception as exc:
            skipped += 1
            print(f"跳过第 {line_num} 行指标计算失败: {exc}", file=sys.stderr)

    total_groups = len(groups)
    total_events = sum(g.event_count for g in groups)
    total_real = sum(g.real_event_count for g in groups)
    total_virtual = sum(g.virtual_event_count for g in groups)
    total_edges = sum(g.missing_edge_count for g in groups)
    total_pairs = sum(g.pair_count for g in groups)

    edge_relation_counts = Counter()
    pair_relation_counts = Counter()
    risk_flag_counts = Counter()
    rule_counts = Counter()
    for g in groups:
        edge_relation_counts.update(g.edge_relation_counts)
        pair_relation_counts.update(g.pair_relation_counts)
        risk_flag_counts.update(g.risk_flags)
        rule_counts[g.rule] += 1

    event_counts = [g.event_count for g in groups]
    real_counts = [g.real_event_count for g in groups]
    virtual_counts = [g.virtual_event_count for g in groups]
    site_counts = [g.site_count for g in groups]
    ne_counts = [g.ne_count for g in groups]
    alarm_type_counts = [g.alarm_type_count for g in groups]
    durations = [g.duration_sec for g in groups]
    edge_counts = [g.missing_edge_count for g in groups]
    cohesion = [g.topology_cohesion_score for g in groups if g.topology_cohesion_score is not None]
    dominant_ratios = [g.dominant_alarm_type_ratio for g in groups]

    result = {
        "meta": {
            "source_file": str(visual_jsonl),
            "ne_graph": str(ne_graph_path),
            "site_graph": str(site_graph_path),
            "site_graph_loaded": bool(site_graph_exists),
            "topo_max_hops": int(topo_max_hops),
            "max_pairwise_ne": int(max_pairwise_ne),
            "skipped_records": skipped,
            "metric_notes": [
                "No labels are used; metrics are diagnostics/proxies, not ground truth.",
                "missing_edge_relation_distribution is based on visual missing_topology_edges.",
                "pair_relation_distribution is based on all NE pairs inside each group when ne_count <= max_pairwise_ne.",
            ],
        },
        "overall": {
            "group_count": total_groups,
            "event_count": total_events,
            "real_event_count": total_real,
            "virtual_event_count": total_virtual,
            "virtual_event_ratio": _ratio(total_virtual, total_events),
            "singleton_real_group_ratio": _ratio(sum(1 for g in groups if g.real_event_count <= 1), total_groups),
            "multi_real_group_ratio": _ratio(sum(1 for g in groups if g.real_event_count >= 2), total_groups),
            "groups_with_missing_topology_edge_ratio": _ratio(sum(1 for g in groups if g.missing_edge_count > 0), total_groups),
            "groups_with_virtual_event_ratio": _ratio(sum(1 for g in groups if g.virtual_event_count > 0), total_groups),
            "missing_topology_edge_count": total_edges,
            "pair_count_evaluated": total_pairs,
        },
        "distributions": {
            "event_count": {**_numeric_summary(event_counts), "histogram": _histogram(event_counts, max_explicit=20)},
            "real_event_count": {**_numeric_summary(real_counts), "histogram": _histogram(real_counts, max_explicit=20)},
            "virtual_event_count": {**_numeric_summary(virtual_counts), "histogram": _histogram(virtual_counts, max_explicit=10)},
            "site_count": {**_numeric_summary(site_counts), "histogram": _histogram(site_counts, max_explicit=20)},
            "ne_count": {**_numeric_summary(ne_counts), "histogram": _histogram(ne_counts, max_explicit=30)},
            "alarm_type_count": {**_numeric_summary(alarm_type_counts), "histogram": _histogram(alarm_type_counts, max_explicit=10)},
            "duration_sec": {**_numeric_summary(durations), "histogram": _histogram(durations, buckets=DEFAULT_DURATION_BUCKETS_SEC)},
            "missing_topology_edge_count": {**_numeric_summary(edge_counts), "histogram": _histogram(edge_counts, max_explicit=20)},
            "dominant_alarm_type_ratio": _numeric_summary(dominant_ratios),
            "topology_cohesion_score": _numeric_summary(cohesion),
        },
        "topology": {
            "missing_edge_relation_distribution": _counter_payload(edge_relation_counts, total_edges),
            "pair_relation_distribution": _counter_payload(pair_relation_counts, total_pairs),
            "unknown_missing_edge_ratio": _ratio(edge_relation_counts.get("unknown", 0), total_edges),
            "cross_site_missing_edge_ratio": _ratio(edge_relation_counts.get("cross_site", 0), total_edges),
            "unknown_pair_ratio": _ratio(pair_relation_counts.get("unknown", 0), total_pairs),
            "cross_site_pair_ratio": _ratio(pair_relation_counts.get("cross_site", 0), total_pairs),
        },
        "risk_flags": _counter_payload(risk_flag_counts, total_groups),
        "by_rule": {},
    }

    by_rule = defaultdict(list)
    for g in groups:
        by_rule[g.rule].append(g)
    for rule, items in sorted(by_rule.items()):
        result["by_rule"][rule] = {
            "group_count": len(items),
            "event_count": sum(g.event_count for g in items),
            "real_event_count": sum(g.real_event_count for g in items),
            "virtual_event_count": sum(g.virtual_event_count for g in items),
            "missing_topology_edge_count": sum(g.missing_edge_count for g in items),
            "event_count_mean": _numeric_summary([g.event_count for g in items]).get("mean", 0.0),
            "site_count_mean": _numeric_summary([g.site_count for g in items]).get("mean", 0.0),
            "duration_sec_median": _numeric_summary([g.duration_sec for g in items]).get("median", 0.0),
        }

    result["health"] = build_health_score(
        result,
        target_duration_sec=health_target_duration_sec,
        target_virtual_ratio=health_target_virtual_ratio,
        target_size_p50=health_target_size_p50,
        target_size_p90=health_target_size_p90,
        target_size_p99=health_target_size_p99,
    )

    if include_details:
        result["detail"] = [_detail_payload(g) for g in groups]

    return result


def print_summary(result):
    overall = result["overall"]
    topo = result["topology"]
    dist = result["distributions"]

    print("\n在线故障组 visual 指标摘要")
    print("=" * 60)
    health = result.get("health") or {}
    if health:
        print(f"健康度: {health.get('grouping_health_score', 0):.1f}/100")
        comps = health.get("components") or {}
        print(
            "  components: "
            f"topology={comps.get('topology_explainability', 0):.1f}, "
            f"time={comps.get('time_compactness', 0):.1f}, "
            f"size={comps.get('size_reasonableness', 0):.1f}, "
            f"singleton={comps.get('singleton_control', 0):.1f}, "
            f"virtual={comps.get('virtual_reasonableness', 0):.1f}, "
            f"risk={comps.get('risk_cleanliness', 0):.1f}"
        )
    print(f"故障组数: {overall['group_count']}")
    print(
        f"告警数: total={overall['event_count']} real={overall['real_event_count']} "
        f"virtual={overall['virtual_event_count']} "
        f"(virtual_ratio={overall['virtual_event_ratio']:.3f})"
    )
    print(
        f"簇大小: mean={dist['real_event_count'].get('mean', 0):.2f} "
        f"median={dist['real_event_count'].get('median', 0)} "
        f"p90={dist['real_event_count'].get('percentiles', {}).get('p90', 0)}"
    )
    print(
        f"持续时间(s): median={dist['duration_sec'].get('median', 0):.1f} "
        f"p90={dist['duration_sec'].get('percentiles', {}).get('p90', 0):.1f} "
        f"max={dist['duration_sec'].get('max', 0):.1f}"
    )
    print(
        f"站点数: mean={dist['site_count'].get('mean', 0):.2f} "
        f"p90={dist['site_count'].get('percentiles', {}).get('p90', 0)} "
        f"max={dist['site_count'].get('max', 0)}"
    )
    print(
        f"missing_topology_edges: {overall['missing_topology_edge_count']} "
        f"groups_with_edge_ratio={overall['groups_with_missing_topology_edge_ratio']:.3f}"
    )
    print(
        f"missing-edge unknown_ratio={topo['unknown_missing_edge_ratio']:.3f}, "
        f"cross_site_ratio={topo['cross_site_missing_edge_ratio']:.3f}"
    )
    print(
        f"pair unknown_ratio={topo['unknown_pair_ratio']:.3f}, "
        f"cross_site_ratio={topo['cross_site_pair_ratio']:.3f}"
    )
    cohesion = dist["topology_cohesion_score"]
    if cohesion.get("count", 0):
        print(f"topology_cohesion_score: mean={cohesion['mean']:.3f}, median={cohesion['median']:.3f}")
    print("risk_flags:")
    if result["risk_flags"]:
        for flag, item in result["risk_flags"].items():
            print(f"  {flag}: {item['count']} ({item['ratio']:.3f})")
    else:
        print("  <none>")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="分析在线推理 visual JSONL 的无监督故障组指标")
    parser.add_argument("visual_jsonl", help="在线推理输出的 visual JSONL")
    parser.add_argument("-o", "--output", default="", help="输出 JSON 文件；为空则只打印摘要")
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=f"NE graph JSON，默认 {resource_display('ne_graph.json')}")
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=f"Site graph JSON，默认 {resource_display('site_graph.json')}")
    parser.add_argument("--topo-max-hops", type=int, default=3, help="拓扑关系计算的最大 hop，默认 3")
    parser.add_argument("--max-pairwise-ne", type=int, default=200, help="每组最多对多少个 NE 做全 pair 拓扑统计，默认 200")
    parser.add_argument("--risk-duration-sec", type=float, default=2 * 3600, help="long_duration 风险阈值，默认 7200")
    parser.add_argument("--risk-site-count", type=int, default=10, help="many_sites 风险阈值，默认 10")
    parser.add_argument("--risk-unknown-pair-ratio", type=float, default=0.5, help="high_unknown_pair_ratio 风险阈值，默认 0.5")
    parser.add_argument("--health-target-duration-sec", type=float, default=3600.0, help="健康度 time_compactness 的 p90 目标时长，默认 3600")
    parser.add_argument("--health-target-virtual-ratio", type=float, default=0.2, help="健康度 virtual_reasonableness 的目标虚拟告警比例，默认 0.2")
    parser.add_argument("--health-target-size-p50", type=float, default=2.0, help="健康度 size_reasonableness 的 p50 目标真实告警数，默认 2")
    parser.add_argument("--health-target-size-p90", type=float, default=20.0, help="健康度 size_reasonableness 的 p90 合理上限，默认 20")
    parser.add_argument("--health-target-size-p99", type=float, default=100.0, help="健康度 size_reasonableness 的 p99 合理上限，默认 100")
    parser.add_argument("--no-detail", action="store_true", help="输出 JSON 不包含逐组 detail，文件更小")
    args = parser.parse_args()

    if args.topo_max_hops < 1:
        parser.error("--topo-max-hops must be >= 1")
    if args.max_pairwise_ne < 2:
        parser.error("--max-pairwise-ne must be >= 2")

    result = analyze(
        args.visual_jsonl,
        ne_graph_path=args.ne_graph,
        site_graph_path=args.site_graph,
        topo_max_hops=args.topo_max_hops,
        max_pairwise_ne=args.max_pairwise_ne,
        risk_duration_sec=args.risk_duration_sec,
        risk_site_count=args.risk_site_count,
        risk_unknown_pair_ratio=args.risk_unknown_pair_ratio,
        health_target_duration_sec=args.health_target_duration_sec,
        health_target_virtual_ratio=args.health_target_virtual_ratio,
        health_target_size_p50=args.health_target_size_p50,
        health_target_size_p90=args.health_target_size_p90,
        health_target_size_p99=args.health_target_size_p99,
        include_details=not args.no_detail,
    )
    print_summary(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"指标已保存: {args.output}")


if __name__ == "__main__":
    main()
