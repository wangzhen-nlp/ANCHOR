#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for alarm-native fault-group baselines."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json

from ticket_recall.evaluation.recall_common import _parse_group_ids


def load_json_if_exists(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _value(event, *keys):
    alarm = _as_dict(event.get("alarm"))
    for key in keys:
        value = event.get(key)
        if value is None:
            value = alarm.get(key)
        text = str(value or "").strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""


def _coerce_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts if ts > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = float(text)
        return ts if ts > 0 else None
    except ValueError:
        pass
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


def event_ts(event):
    for key in ("ts", "timestamp", "first_ts", "first_occurrence_ts"):
        ts = _coerce_ts(event.get(key))
        if ts is not None:
            return ts
    alarm = _as_dict(event.get("alarm"))
    for key in ("告警首次发生时间", "alarm_time", "first_occurrence_time", "time_str"):
        ts = _coerce_ts(event.get(key))
        if ts is not None:
            return ts
        ts = _coerce_ts(alarm.get(key))
        if ts is not None:
            return ts
    return None


def event_ne(event):
    return _value(event, "alarm_source", "告警源", "source_ne", "ne", "ne_id")


def event_site(event, ne_graph_data):
    site = _value(event, "site_id", "node", "站点ID", "site", "site_id_raw")
    if site:
        return site
    ne = event_ne(event)
    info = ne_graph_data.get(ne, {}) if isinstance(ne_graph_data, dict) else {}
    return str(info.get("site_id", "") or "").strip()


def event_group_ids(event, group_field):
    raw = event.get(group_field)
    if raw is None and isinstance(event.get("alarm"), dict):
        raw = event["alarm"].get(group_field)
    return _parse_group_ids(raw)


def alarm_to_baseline_symptom(event, *, group_field, ne_graph_data, occurrence_id=""):
    site_id = event_site(event, ne_graph_data)
    ne_id = event_ne(event)
    occurrence_id = (
        str(occurrence_id or "").strip()
        or _value(event, "occurrence_id", "_mhp_occurrence_id", "_case_alarm_seq")
    )
    return {
        "node": site_id,
        "site_id": site_id,
        "alarm_source": ne_id,
        "alarm": _value(event, "alarm_title", "alarm", "告警标题", "title"),
        "alarm_type": _value(event, "alarm_type", "alarm_title", "告警标题", "alarm", "title"),
        "ts": event_ts(event),
        "eid": _value(event, "eid", "event_id", "告警编码ID", "alarm_id"),
        "occurrence_id": occurrence_id,
        "matched_role": "alarm_group_baseline",
        "matched_rule": "alarm_group_baseline",
        "工单号": _value(event, "工单号", "ticket_id"),
        "故障组ID": _value(event, group_field, "故障组ID"),
        "告警清除时间": _value(event, "告警清除时间", "clear_time"),
        "virtual": False,
    }


def build_baseline_records(events, *, group_field, ne_graph_data=None, min_group_events=1):
    ne_graph_data = ne_graph_data or {}
    grouped = {}
    for event in events:
        for group_id in event_group_ids(event, group_field):
            grouped.setdefault(str(group_id), []).append(event)

    records = []
    min_group_events = max(1, int(min_group_events or 1))
    for group_id, group_events in grouped.items():
        if len(group_events) < min_group_events:
            continue
        symptoms = []
        for event_index, event in enumerate(group_events):
            occurrence_id = _value(event, "occurrence_id", "_mhp_occurrence_id", "_case_alarm_seq")
            if not occurrence_id:
                occurrence_id = f"alarm-baseline-{group_id}-{event_index}"
            symptoms.append(
                alarm_to_baseline_symptom(
                    event,
                    group_field=group_field,
                    ne_graph_data=ne_graph_data,
                    occurrence_id=occurrence_id,
                )
            )
        symptoms.sort(key=lambda s: (s.get("ts") is None, s.get("ts") or float("inf"), s.get("eid", "")))
        timestamps = [s["ts"] for s in symptoms if s.get("ts") is not None]
        group_uuid = f"alarm-baseline-{group_id}"
        records.append(
            {
                "uuid": group_uuid,
                "group_id": group_uuid,
                "rule": "alarm_group_baseline",
                "source_group_id": group_id,
                "event_count": len(symptoms),
                "start_ts": min(timestamps) if timestamps else None,
                "end_ts": max(timestamps) if timestamps else None,
                "duration_sec": (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0,
                "site_list": sorted({s.get("site_id", "") for s in symptoms if s.get("site_id", "")}),
                "alarm_source_list": sorted(
                    {s.get("alarm_source", "") for s in symptoms if s.get("alarm_source", "")}
                ),
                "alarm_type_counts": dict(
                    Counter(s.get("alarm_type", "") for s in symptoms if s.get("alarm_type", ""))
                ),
                "symptoms": symptoms,
                "match_info": {
                    "uuid": group_uuid,
                    "rule": "alarm_group_baseline",
                    "source_group_id": group_id,
                    "related_group_uuids": [],
                },
            }
        )
    records.sort(
        key=lambda r: (
            r.get("start_ts") is None,
            r.get("start_ts") or float("inf"),
            str(r.get("source_group_id", "")),
        )
    )
    return records


def write_jsonl(path, records):
    count = 0
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
