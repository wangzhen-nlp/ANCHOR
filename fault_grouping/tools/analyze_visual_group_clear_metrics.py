#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clear-time based health diagnostics for fault-group visual JSONL.

Companion to analyze_visual_group_metrics.py, but the validation signal is the
alarm clear time (告警清除时间): alarms of the same fault are typically cleared
by the same remediation action, so their clear times cluster. No labels or
ticket data are needed — the reference is built from the stream itself:

  * clear_compactness  — within-group clear-time span, after removing members
    that fall into global bulk-clear windows (operator mass-clears would
    otherwise fake compactness).
  * clear_separation   — Mann-Whitney AUC of within-group pair |Δclear| vs a
    null of cross-group pairs matched on occurrence-time proximity. 0.5 means
    clear times carry no grouping signal; 1.0 means perfect separation. This is
    the most trustworthy component: occurrence-time matching removes the
    trivial "groups are time-local" effect.
  * clear_coverage     — fraction of real alarms that have a parseable clear
    time; compactness/separation mean little when coverage is low.
  * bulk_cleanliness   — fraction of cleared alarms sitting in bulk-clear
    windows; high values mean the clear signal is dominated by mass operations.

Example:
    python fault_grouping/tools/analyze_visual_group_clear_metrics.py stream.visual.jsonl -o clear_metrics.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import json
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from fault_grouping.tools.analyze_visual_group_metrics import (
    DEFAULT_DURATION_BUCKETS_SEC,
    _clip_score,
    _counter_payload,
    _exp_score,
    _histogram,
    _iter_jsonl,
    _coerce_ts,
    _numeric_summary,
    _ratio,
    _symptom_event_id,
    _symptom_ts,
    extract_symptoms,
    group_id,
    group_rule,
    is_virtual_symptom,
)


CLEAR_FIELDS = ("告警清除时间", "clear_time", "告警清除时间str")

HEALTH_WEIGHTS = {
    "clear_separation": 0.40,
    "clear_compactness": 0.30,
    "clear_coverage": 0.15,
    "bulk_cleanliness": 0.15,
}


def _symptom_clear_ts(symptom):
    for key in CLEAR_FIELDS:
        ts = _coerce_ts(symptom.get(key))
        if ts is not None:
            return ts
    return None


def _symptom_uid(symptom, line_num, idx):
    uid = str(symptom.get("occurrence_uuid") or "").strip()
    if uid:
        return uid
    eid = _symptom_event_id(symptom)
    if eid:
        return eid
    return f"__line{line_num}_sym{idx}"


@dataclass
class ClearedAlarm:
    uid: str
    group: str
    occur_ts: float
    clear_ts: float
    bulk: bool = False


@dataclass
class GroupClearMetrics:
    line_num: int
    uuid: str
    rule: str
    real_event_count: int
    cleared_count: int
    clear_coverage: float
    clear_span_sec: float | None          # span over ALL cleared members (>=2)
    nonbulk_cleared_count: int
    nonbulk_clear_span_sec: float | None  # span over non-bulk cleared members (>=2)
    bulk_member_ratio: float              # cleared members inside bulk windows
    risk_flags: list = field(default_factory=list)


def _load_groups(visual_jsonl, *, verbose=False, progress_every=0):
    """First pass: per-group cleared members, deduped by occurrence uid.

    Returns (groups, alarms) where `alarms` is the globally deduped cleared
    alarm list (first group wins for duplicated uids across snapshots/lines).
    """
    groups = []
    seen_uids = {}
    alarms = []
    skipped = 0
    t0 = time.monotonic()
    for line_num, record in _iter_jsonl(visual_jsonl):
        try:
            symptoms = [s for s in extract_symptoms(record) if not is_virtual_symptom(s)]
            gid = group_id(record) or f"__line{line_num}"
            dedup = {}
            for idx, s in enumerate(symptoms):
                uid = _symptom_uid(s, line_num, idx)
                if uid not in dedup:
                    dedup[uid] = s
            members = []
            for uid, s in dedup.items():
                clear_ts = _symptom_clear_ts(s)
                occur_ts = _symptom_ts(s)
                members.append((uid, occur_ts, clear_ts))
                if clear_ts is not None and occur_ts is not None and uid not in seen_uids:
                    alarm = ClearedAlarm(uid=uid, group=gid, occur_ts=occur_ts, clear_ts=clear_ts)
                    seen_uids[uid] = alarm
                    alarms.append(alarm)
            groups.append(
                {
                    "line_num": line_num,
                    "uuid": gid,
                    "rule": group_rule(record),
                    "members": members,
                }
            )
        except Exception as exc:
            skipped += 1
            print(f"跳过第 {line_num} 行清除指标解析失败: {exc}", file=sys.stderr)
        if verbose and progress_every and line_num % progress_every == 0:
            elapsed = time.monotonic() - t0
            print(
                f"[clear-metrics] scanned {line_num} records "
                f"(groups={len(groups)}, cleared_alarms={len(alarms)}, "
                f"skipped={skipped}, elapsed={elapsed:.1f}s)",
                flush=True,
            )
    return groups, alarms, skipped


def _load_grouped_symptoms(group_to_symptoms):
    """Build clear-metric inputs from an already grouped alarm mapping.

    This is the in-memory counterpart of :func:`_load_groups`.  It lets other
    evaluators score an arbitrary label assignment (for example, native alarm
    group ids) without writing a synthetic visual JSONL first.
    """
    groups = []
    seen_uids = {}
    alarms = []
    for line_num, (gid, symptoms) in enumerate(group_to_symptoms.items(), 1):
        dedup = {}
        for idx, symptom in enumerate(symptoms or []):
            if not isinstance(symptom, dict) or is_virtual_symptom(symptom):
                continue
            uid = _symptom_uid(symptom, line_num, idx)
            if uid not in dedup:
                dedup[uid] = symptom

        members = []
        group_name = str(gid)
        for uid, symptom in dedup.items():
            clear_ts = _symptom_clear_ts(symptom)
            occur_ts = _symptom_ts(symptom)
            members.append((uid, occur_ts, clear_ts))
            if clear_ts is not None and occur_ts is not None and uid not in seen_uids:
                alarm = ClearedAlarm(
                    uid=uid,
                    group=group_name,
                    occur_ts=occur_ts,
                    clear_ts=clear_ts,
                )
                seen_uids[uid] = alarm
                alarms.append(alarm)
        groups.append(
            {
                "line_num": line_num,
                "uuid": group_name,
                "rule": "",
                "members": members,
            }
        )
    return groups, alarms


def _mark_bulk_windows(alarms, *, bulk_window_sec, bulk_min_count):
    """Flag alarms whose clear falls into a global bulk-clear window.

    A bulk window is a fixed-size clear-time bucket holding >= bulk_min_count
    deduped cleared alarms across the whole file — the signature of operator
    mass-clears / timeout auto-clears, which fake clear-time coherence.
    """
    window_counts = Counter()
    for a in alarms:
        window_counts[int(a.clear_ts // bulk_window_sec)] += 1
    bulk_windows = {w for w, c in window_counts.items() if c >= bulk_min_count}
    for a in alarms:
        a.bulk = int(a.clear_ts // bulk_window_sec) in bulk_windows
    top = sorted(
        ((c, w) for w, c in window_counts.items() if w in bulk_windows),
        reverse=True,
    )[:5]
    return {
        "bulk_window_sec": float(bulk_window_sec),
        "bulk_min_count": int(bulk_min_count),
        "bulk_window_count": len(bulk_windows),
        "bulk_alarm_count": sum(1 for a in alarms if a.bulk),
        "bulk_alarm_ratio": _ratio(sum(1 for a in alarms if a.bulk), len(alarms)),
        "top_bulk_windows": [
            {"window_start_ts": w * float(bulk_window_sec), "clear_count": int(c)}
            for c, w in top
        ],
    }


def _span(values):
    return (max(values) - min(values)) if len(values) >= 2 else None


def _analyze_group(entry, cleared_index, *, risk_span_sec, risk_low_coverage, risk_bulk_ratio):
    members = entry["members"]
    cleared = [cleared_index[uid] for uid, _, clear_ts in members
               if clear_ts is not None and uid in cleared_index]
    cleared_count = sum(1 for _, _, clear_ts in members if clear_ts is not None)
    nonbulk = [a for a in cleared if not a.bulk]
    metrics = GroupClearMetrics(
        line_num=entry["line_num"],
        uuid=entry["uuid"],
        rule=entry["rule"],
        real_event_count=len(members),
        cleared_count=cleared_count,
        clear_coverage=_ratio(cleared_count, len(members)),
        clear_span_sec=_span([a.clear_ts for a in cleared]),
        nonbulk_cleared_count=len(nonbulk),
        nonbulk_clear_span_sec=_span([a.clear_ts for a in nonbulk]),
        bulk_member_ratio=_ratio(len(cleared) - len(nonbulk), len(cleared)),
    )
    flags = []
    if metrics.cleared_count == 0:
        flags.append("no_cleared_member")
    elif metrics.clear_coverage < risk_low_coverage:
        flags.append("low_clear_coverage")
    if metrics.nonbulk_clear_span_sec is not None and metrics.nonbulk_clear_span_sec > risk_span_sec:
        flags.append("wide_clear_span")
    if metrics.cleared_count >= 2 and metrics.bulk_member_ratio > risk_bulk_ratio:
        flags.append("bulk_clear_dominated")
    metrics.risk_flags = flags
    return metrics


def _sample_within_pairs(groups, cleared_index, *, max_pairs_per_group, rng):
    """Within-group pairs of non-bulk cleared members: (|Δclear|, Δoccur)."""
    pairs = []
    for entry in groups:
        cleared = [cleared_index[uid] for uid, _, clear_ts in entry["members"]
                   if clear_ts is not None and uid in cleared_index]
        cleared = [a for a in cleared if not a.bulk]
        n = len(cleared)
        if n < 2:
            continue
        if n * (n - 1) // 2 <= max_pairs_per_group:
            iterator = combinations(cleared, 2)
        else:
            def _draw(items=cleared, k=max_pairs_per_group):
                seen = set()
                m = len(items)
                for _ in range(k * 3):
                    if len(seen) >= k:
                        break
                    i = rng.randrange(m)
                    j = rng.randrange(m)
                    if i == j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield items[key[0]], items[key[1]]
            iterator = _draw()
        for a, b in iterator:
            pairs.append((abs(a.clear_ts - b.clear_ts), abs(a.occur_ts - b.occur_ts)))
    return pairs


def _sample_null_pairs(alarms, *, occ_window_sec, max_null_pairs, draws_per_alarm, rng):
    """Cross-group pairs matched on occurrence proximity: |Δclear| values.

    Alarms are non-bulk, deduped, sorted by occurrence ts. For each alarm we
    draw at most ``draws_per_alarm`` distinct random partners among the
    following alarms within occ_window_sec that belong to a DIFFERENT group.
    We allow up to three attempts per requested partner so same-group or
    duplicate draws do not immediately exhaust the budget. The null answers
    "two alarms that occurred close together but were grouped apart: how close
    are their clear times?".
    """
    pool = sorted((a for a in alarms if not a.bulk), key=lambda a: a.occur_ts)
    n = len(pool)
    deltas = []
    hi = 0
    for i in range(n):
        if len(deltas) >= max_null_pairs:
            break
        lo = i + 1
        if hi < lo:
            hi = lo
        while hi < n and pool[hi].occur_ts - pool[i].occur_ts <= occ_window_sec:
            hi += 1
        if hi <= lo:
            continue
        selected = set()
        for _ in range(max(0, draws_per_alarm) * 3):
            if (
                len(deltas) >= max_null_pairs
                or len(selected) >= draws_per_alarm
            ):
                break
            j = rng.randrange(lo, hi)
            if pool[j].group == pool[i].group or j in selected:
                continue
            selected.add(j)
            deltas.append(abs(pool[i].clear_ts - pool[j].clear_ts))
    return deltas


def _mann_whitney_auc(within, null):
    """AUC = P(|Δclear|_within < |Δclear|_null), ties counted 0.5 (midranks)."""
    n_w, n_n = len(within), len(null)
    if n_w == 0 or n_n == 0:
        return None
    combined = sorted(
        [(v, 0) for v in within] + [(v, 1) for v in null],
        key=lambda x: x[0],
    )
    rank_sum_within = 0.0
    i = 0
    total = len(combined)
    while i < total:
        j = i
        while j < total and combined[j][0] == combined[i][0]:
            j += 1
        midrank = (i + 1 + j) / 2.0  # average of ranks i+1..j
        rank_sum_within += midrank * sum(1 for k in range(i, j) if combined[k][1] == 0)
        i = j
    # U = #(within > null) + 0.5 #(ties); AUC = 1 - U / (n_w * n_n)
    u_within = rank_sum_within - n_w * (n_w + 1) / 2.0
    return 1.0 - u_within / float(n_w * n_n)


def build_health_score(
    result,
    *,
    target_coverage=0.5,
    target_span_sec=3600.0,
    target_bulk_ratio=0.2,
):
    """0-100 clear-time health score; components renormalized when N/A."""
    overall = result.get("overall", {})
    if int(overall.get("group_count", 0) or 0) <= 0:
        return {
            "clear_health_score": None,
            "components": {},
            "weights": dict(HEALTH_WEIGHTS),
            "not_applicable": True,
            "interpretation": "Health is not defined because there are no groups to evaluate.",
        }
    separation = result.get("separation", {})
    auc = separation.get("auc")
    span_p90 = (
        result.get("distributions", {})
        .get("nonbulk_clear_span_sec", {})
        .get("percentiles", {})
        .get("p90")
    )
    components = {
        "clear_separation": (
            _clip_score(100.0 * (float(auc) - 0.5) / 0.5) if auc is not None else None
        ),
        "clear_compactness": (
            _exp_score(float(span_p90), target_span_sec) if span_p90 is not None else None
        ),
        "clear_coverage": _clip_score(
            100.0 * min(_ratio(overall.get("cleared_event_count", 0), overall.get("real_event_count", 0)) / max(target_coverage, 1e-9), 1.0)
        ),
        "bulk_cleanliness": _exp_score(
            float(result.get("bulk", {}).get("bulk_alarm_ratio", 0.0) or 0.0),
            target_bulk_ratio,
        ),
    }
    usable = {k: v for k, v in components.items() if v is not None}
    weight_sum = sum(HEALTH_WEIGHTS[k] for k in usable)
    score = (
        sum(HEALTH_WEIGHTS[k] * v for k, v in usable.items()) / weight_sum
        if weight_sum > 0 else None
    )
    return {
        "clear_health_score": _clip_score(score) if score is not None else None,
        "components": components,
        "weights": dict(HEALTH_WEIGHTS),
        "targets": {
            "clear_coverage": float(target_coverage),
            "nonbulk_clear_span_p90_sec": float(target_span_sec),
            "bulk_alarm_ratio": float(target_bulk_ratio),
        },
        "interpretation": (
            "0-100; higher means grouping agrees better with clear-time evidence. "
            "clear_separation (AUC vs occurrence-matched cross-group null) is the "
            "most trustworthy component; compactness/coverage are supporting. "
            "Label-free proxy for comparing runs, not ground truth."
        ),
    }


def _detail_payload(m: GroupClearMetrics):
    return {
        "line_num": m.line_num,
        "uuid": m.uuid,
        "rule": m.rule,
        "real_event_count": m.real_event_count,
        "cleared_count": m.cleared_count,
        "clear_coverage": m.clear_coverage,
        "clear_span_sec": m.clear_span_sec,
        "nonbulk_cleared_count": m.nonbulk_cleared_count,
        "nonbulk_clear_span_sec": m.nonbulk_clear_span_sec,
        "bulk_member_ratio": m.bulk_member_ratio,
        "risk_flags": list(m.risk_flags),
    }


def _analyze_loaded_groups(
    groups_raw,
    alarms,
    *,
    source_file="<in-memory groups>",
    skipped=0,
    bulk_window_sec=60.0,
    bulk_min_count=10,
    null_occ_window_sec=0.0,
    max_pairs_per_group=500,
    max_null_pairs=200000,
    null_draws_per_alarm=3,
    seed=0,
    risk_span_sec=2 * 3600.0,
    risk_low_coverage=0.3,
    risk_bulk_ratio=0.8,
    health_target_coverage=0.5,
    health_target_span_sec=3600.0,
    health_target_bulk_ratio=0.2,
    include_details=True,
):
    rng = random.Random(seed)
    bulk_info = _mark_bulk_windows(
        alarms, bulk_window_sec=bulk_window_sec, bulk_min_count=bulk_min_count
    )
    cleared_index = {a.uid: a for a in alarms}

    groups = [
        _analyze_group(
            entry,
            cleared_index,
            risk_span_sec=risk_span_sec,
            risk_low_coverage=risk_low_coverage,
            risk_bulk_ratio=risk_bulk_ratio,
        )
        for entry in groups_raw
    ]

    within_pairs = _sample_within_pairs(
        groups_raw, cleared_index, max_pairs_per_group=max_pairs_per_group, rng=rng
    )
    within_deltas = [d for d, _ in within_pairs]
    within_occ_gaps = [g for _, g in within_pairs]
    # Auto null window: p90 of within-pair occurrence gaps, floored at 300s,
    # so the null pairs live at the same occurrence-time scale as real groups.
    if null_occ_window_sec <= 0:
        gaps_sorted = sorted(within_occ_gaps)
        p90_gap = gaps_sorted[int(0.9 * (len(gaps_sorted) - 1))] if gaps_sorted else 0.0
        occ_window = max(300.0, float(p90_gap))
    else:
        occ_window = float(null_occ_window_sec)
    null_deltas = _sample_null_pairs(
        alarms,
        occ_window_sec=occ_window,
        max_null_pairs=max_null_pairs,
        draws_per_alarm=null_draws_per_alarm,
        rng=rng,
    )
    auc = _mann_whitney_auc(within_deltas, null_deltas)

    total_groups = len(groups)
    total_real = sum(g.real_event_count for g in groups)
    total_cleared = sum(g.cleared_count for g in groups)
    risk_flag_counts = Counter()
    for g in groups:
        risk_flag_counts.update(g.risk_flags)

    spans = [g.clear_span_sec for g in groups if g.clear_span_sec is not None]
    nonbulk_spans = [g.nonbulk_clear_span_sec for g in groups if g.nonbulk_clear_span_sec is not None]
    coverages = [g.clear_coverage for g in groups]

    result = {
        "meta": {
            "source_file": str(source_file),
            "skipped_records": skipped,
            "seed": int(seed),
            "metric_notes": [
                "No labels are used; clear-time clusters are a NOISY reference, not ground truth.",
                "Members are deduped by occurrence_uuid within a group; the global cleared-alarm pool keeps first-seen group per uid (snapshot lines may repeat alarms).",
                "Bulk-clear windows (operator mass clears / timeout auto-clears) are excluded from compactness and separation.",
                "separation.auc compares within-group |Δclear| pairs against cross-group pairs matched on occurrence proximity; 0.5 = no signal.",
            ],
        },
        "overall": {
            "group_count": total_groups,
            "real_event_count": total_real,
            "cleared_event_count": total_cleared,
            "clear_coverage": _ratio(total_cleared, total_real),
            "groups_with_clear_ratio": _ratio(sum(1 for g in groups if g.cleared_count > 0), total_groups),
            "groups_with_2plus_clear_ratio": _ratio(sum(1 for g in groups if g.cleared_count >= 2), total_groups),
            "deduped_cleared_alarm_count": len(alarms),
        },
        "bulk": bulk_info,
        "distributions": {
            "clear_coverage": _numeric_summary(coverages),
            "clear_span_sec": {
                **_numeric_summary(spans),
                "histogram": _histogram(spans, buckets=DEFAULT_DURATION_BUCKETS_SEC),
            },
            "nonbulk_clear_span_sec": {
                **_numeric_summary(nonbulk_spans),
                "histogram": _histogram(nonbulk_spans, buckets=DEFAULT_DURATION_BUCKETS_SEC),
            },
        },
        "separation": {
            "auc": auc,
            "within_pair_count": len(within_deltas),
            "null_pair_count": len(null_deltas),
            "null_occurrence_window_sec": occ_window,
            "within_delta_clear_sec": _numeric_summary(within_deltas),
            "null_delta_clear_sec": _numeric_summary(null_deltas),
        },
        "risk_flags": _counter_payload(risk_flag_counts, total_groups),
        "by_rule": {},
    }

    by_rule = defaultdict(list)
    for g in groups:
        by_rule[g.rule].append(g)
    for rule, items in sorted(by_rule.items()):
        rule_spans = [g.nonbulk_clear_span_sec for g in items if g.nonbulk_clear_span_sec is not None]
        result["by_rule"][rule] = {
            "group_count": len(items),
            "real_event_count": sum(g.real_event_count for g in items),
            "cleared_event_count": sum(g.cleared_count for g in items),
            "clear_coverage": _ratio(sum(g.cleared_count for g in items), sum(g.real_event_count for g in items)),
            "nonbulk_clear_span_sec_median": _numeric_summary(rule_spans).get("median", 0.0),
        }

    result["health"] = build_health_score(
        result,
        target_coverage=health_target_coverage,
        target_span_sec=health_target_span_sec,
        target_bulk_ratio=health_target_bulk_ratio,
    )

    if include_details:
        result["detail"] = [_detail_payload(g) for g in groups]

    return result


def analyze(
    visual_jsonl,
    *,
    bulk_window_sec=60.0,
    bulk_min_count=10,
    null_occ_window_sec=0.0,
    max_pairs_per_group=500,
    max_null_pairs=200000,
    null_draws_per_alarm=3,
    seed=0,
    risk_span_sec=2 * 3600.0,
    risk_low_coverage=0.3,
    risk_bulk_ratio=0.8,
    health_target_coverage=0.5,
    health_target_span_sec=3600.0,
    health_target_bulk_ratio=0.2,
    include_details=True,
    progress_every=0,
    verbose=False,
):
    """Analyze the grouping stored in a visual JSONL file."""
    groups_raw, alarms, skipped = _load_groups(
        visual_jsonl, verbose=verbose, progress_every=progress_every
    )
    return _analyze_loaded_groups(
        groups_raw,
        alarms,
        source_file=visual_jsonl,
        skipped=skipped,
        bulk_window_sec=bulk_window_sec,
        bulk_min_count=bulk_min_count,
        null_occ_window_sec=null_occ_window_sec,
        max_pairs_per_group=max_pairs_per_group,
        max_null_pairs=max_null_pairs,
        null_draws_per_alarm=null_draws_per_alarm,
        seed=seed,
        risk_span_sec=risk_span_sec,
        risk_low_coverage=risk_low_coverage,
        risk_bulk_ratio=risk_bulk_ratio,
        health_target_coverage=health_target_coverage,
        health_target_span_sec=health_target_span_sec,
        health_target_bulk_ratio=health_target_bulk_ratio,
        include_details=include_details,
    )


def analyze_grouped_symptoms(group_to_symptoms, **kwargs):
    """Analyze clear-time consistency for an in-memory group-to-alarm map."""
    groups_raw, alarms = _load_grouped_symptoms(group_to_symptoms)
    return _analyze_loaded_groups(groups_raw, alarms, **kwargs)


def print_summary(result):
    overall = result["overall"]
    dist = result["distributions"]
    sep = result["separation"]
    bulk = result["bulk"]

    print("\n故障组清除时间健康度摘要")
    print("=" * 60)
    health = result.get("health") or {}
    score = health.get("clear_health_score")
    if score is None:
        print("清除健康度: N/A")
    else:
        print(f"清除健康度: {score:.1f}/100")
    comps = health.get("components") or {}
    if comps:
        def _fmt(v):
            return "N/A" if v is None else f"{v:.1f}"
        print(
            "  components: "
            f"separation={_fmt(comps.get('clear_separation'))}, "
            f"compactness={_fmt(comps.get('clear_compactness'))}, "
            f"coverage={_fmt(comps.get('clear_coverage'))}, "
            f"bulk={_fmt(comps.get('bulk_cleanliness'))}"
        )
    print(f"故障组数: {overall['group_count']}")
    print(
        f"清除覆盖率: {overall['clear_coverage']:.3f} "
        f"(real={overall['real_event_count']}, cleared={overall['cleared_event_count']})"
    )
    print(
        f"含清除组占比: any={overall['groups_with_clear_ratio']:.3f}, "
        f">=2条={overall['groups_with_2plus_clear_ratio']:.3f}"
    )
    print(
        f"批量清除: 窗口数={bulk['bulk_window_count']} "
        f"(window={bulk['bulk_window_sec']:.0f}s, min_count={bulk['bulk_min_count']}), "
        f"受影响告警占比={bulk['bulk_alarm_ratio']:.3f}"
    )
    nb = dist["nonbulk_clear_span_sec"]
    if nb.get("count", 0):
        print(
            f"组内清除跨度(去批量, s): median={nb['median']:.1f} "
            f"p90={nb.get('percentiles', {}).get('p90', 0):.1f} max={nb['max']:.1f}"
        )
    auc = sep.get("auc")
    if auc is not None:
        w = sep["within_delta_clear_sec"]
        nl = sep["null_delta_clear_sec"]
        print(
            f"清除分离度 AUC: {auc:.3f} "
            f"(within_pairs={sep['within_pair_count']}, null_pairs={sep['null_pair_count']}, "
            f"null_occ_window={sep['null_occurrence_window_sec']:.0f}s)"
        )
        print(
            f"  |Δclear| median: within={w.get('median', 0):.1f}s vs null={nl.get('median', 0):.1f}s"
        )
    else:
        print("清除分离度 AUC: N/A (within/null 配对不足)")
    print("risk_flags:")
    if result["risk_flags"]:
        for flag, item in result["risk_flags"].items():
            print(f"  {flag}: {item['count']} ({item['ratio']:.3f})")
    else:
        print("  <none>")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="以告警清除时间为校验标准分析 visual JSONL 的故障组健康度")
    parser.add_argument("visual_jsonl", help="在线推理输出的 visual JSONL")
    parser.add_argument("-o", "--output", default="", help="输出 JSON 文件；为空则只打印摘要")
    parser.add_argument("--bulk-window-sec", type=float, default=60.0, help="批量清除检测的时间窗宽度，默认 60s")
    parser.add_argument("--bulk-min-count", type=int, default=10, help="单窗清除数达到该值判为批量清除窗口，默认 10")
    parser.add_argument("--null-occ-window-sec", type=float, default=0.0, help="null 配对的发生时间窗；0 表示自动(组内配对发生间隔 p90，下限 300s)")
    parser.add_argument("--max-pairs-per-group", type=int, default=500, help="每组最多采样多少组内清除配对，默认 500")
    parser.add_argument("--max-null-pairs", type=int, default=200000, help="null 配对总数上限，默认 200000")
    parser.add_argument("--null-draws-per-alarm", type=int, default=3, help="每条告警最多抽多少个跨组邻居，默认 3")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子，默认 0（结果可复现）")
    parser.add_argument("--risk-span-sec", type=float, default=2 * 3600.0, help="wide_clear_span 风险阈值，默认 7200")
    parser.add_argument("--risk-low-coverage", type=float, default=0.3, help="low_clear_coverage 风险阈值，默认 0.3")
    parser.add_argument("--risk-bulk-ratio", type=float, default=0.8, help="bulk_clear_dominated 风险阈值，默认 0.8")
    parser.add_argument("--health-target-coverage", type=float, default=0.5, help="健康度 clear_coverage 的目标覆盖率，默认 0.5")
    parser.add_argument("--health-target-span-sec", type=float, default=3600.0, help="健康度 clear_compactness 的 p90 目标跨度，默认 3600")
    parser.add_argument("--health-target-bulk-ratio", type=float, default=0.2, help="健康度 bulk_cleanliness 的目标批量占比，默认 0.2")
    parser.add_argument("--no-detail", action="store_true", help="输出 JSON 不包含逐组 detail，文件更小")
    parser.add_argument("--progress-every", type=int, default=1000, help="每处理 N 行打印一次进度；0 表示关闭。默认 1000")
    parser.add_argument("--quiet", action="store_true", help="不打印阶段和进度日志，只输出最终摘要")
    args = parser.parse_args()

    if args.bulk_window_sec <= 0:
        parser.error("--bulk-window-sec must be > 0")
    if args.bulk_min_count < 2:
        parser.error("--bulk-min-count must be >= 2")
    if args.max_pairs_per_group < 1:
        parser.error("--max-pairs-per-group must be >= 1")

    result = analyze(
        args.visual_jsonl,
        bulk_window_sec=args.bulk_window_sec,
        bulk_min_count=args.bulk_min_count,
        null_occ_window_sec=args.null_occ_window_sec,
        max_pairs_per_group=args.max_pairs_per_group,
        max_null_pairs=args.max_null_pairs,
        null_draws_per_alarm=args.null_draws_per_alarm,
        seed=args.seed,
        risk_span_sec=args.risk_span_sec,
        risk_low_coverage=args.risk_low_coverage,
        risk_bulk_ratio=args.risk_bulk_ratio,
        health_target_coverage=args.health_target_coverage,
        health_target_span_sec=args.health_target_span_sec,
        health_target_bulk_ratio=args.health_target_bulk_ratio,
        include_details=not args.no_detail,
        progress_every=args.progress_every,
        verbose=not args.quiet,
    )
    print_summary(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"清除指标已保存: {args.output}")


if __name__ == "__main__":
    main()
