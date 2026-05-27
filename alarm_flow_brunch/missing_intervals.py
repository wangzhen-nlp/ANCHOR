"""Missing-data intervals for the streaming alarm-flow BRUNCH assigner.

Each ``MissingInterval`` declares a time window during which alarms of a given
business key are known to be unobserved (collector down, NE offline for
telemetry, an explicit outage log, etc.). The :class:`MissingIntervalTracker`
resolves those business keys to BRUNCH model ``type_id``s using the trained
vocab and drives Shelton-style virtual event sampling in
``stream_alarm_brunch.OnlineBRUNCHAssigner``.

This is the lightweight "likelihood-weighted" variant of Shelton 2018: we
sample virtual events forward in time during open intervals, let them
participate as candidate parents like real events, and dampen their influence
via a per-event ``confidence``. We do not run the paper's reversible-jump MCMC
because the stream is single-pass forward-only — no MCMC sweep to drive
acceptance ratios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


KEY_KINDS = frozenset({"alarm_source", "alarm_type", "type_label", "type_id"})


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


@dataclass
class MissingInterval:
    """A declared unobserved (key → time window) range.

    ``key_kind`` picks how ``key_value`` is matched against the BRUNCH vocab:

    - ``"alarm_source"`` — match any vocab label that starts with
      ``"{key_value} | "`` (the BRUNCH event-type label format
      ``"{alarm_source} | {alarm_type}"``) or equals ``key_value`` exactly when
      vocab was built without ``alarm_type``.
    - ``"alarm_type"`` — match any vocab label that ends with ``" | {key_value}"``
      or equals ``key_value`` exactly.
    - ``"type_label"`` — exact match against a single vocab label.
    - ``"type_id"`` — direct integer ``type_id`` (the vocab index).

    Times are Unix seconds (float). ``end_ts=None`` marks the interval as
    still-open. ``last_sample_ts`` is internal bookkeeping for incremental
    virtual-event sampling and should not be set by callers.
    """

    key_kind: str
    key_value: str
    start_ts: float
    end_ts: Optional[float] = None
    last_sample_ts: Optional[float] = None

    def __post_init__(self):
        if self.key_kind not in KEY_KINDS:
            raise ValueError(f"key_kind must be one of {sorted(KEY_KINDS)}; got {self.key_kind!r}")
        self.key_value = str(self.key_value)
        self.start_ts = float(self.start_ts)
        if self.end_ts is not None:
            self.end_ts = float(self.end_ts)
            if self.end_ts < self.start_ts:
                raise ValueError(f"end_ts ({self.end_ts}) must be >= start_ts ({self.start_ts})")
        if self.last_sample_ts is None:
            self.last_sample_ts = self.start_ts
        else:
            self.last_sample_ts = float(self.last_sample_ts)

    def is_open(self) -> bool:
        return self.end_ts is None

    def contains(self, t: float) -> bool:
        if t < self.start_ts:
            return False
        return self.end_ts is None or t <= self.end_ts

    def effective_end(self) -> float:
        """Upper bound on sample-able time. Infinity for still-open intervals."""
        return self.end_ts if self.end_ts is not None else float("inf")

    def needs_sampling(self, t: float) -> bool:
        """True iff there is still un-sampled time in ``(last_sample_ts, min(t, end_ts)]``.

        Importantly this returns True even after the interval has been closed
        (``end_ts`` set) as long as ``last_sample_ts < end_ts``; the streaming
        assigner may not yet have caught up to the tail of a recovered window
        when recovery is declared.
        """
        if t < self.start_ts:
            return False
        last = float(self.last_sample_ts if self.last_sample_ts is not None else self.start_ts)
        upper = min(float(t), self.effective_end())
        return last < upper

    def is_exhausted(self) -> bool:
        """True for closed intervals whose tail has been fully sampled.

        Exhausted intervals can be safely dropped from the tracker; the
        assigner will never look at them again.
        """
        if self.end_ts is None:
            return False
        last = self.last_sample_ts if self.last_sample_ts is not None else self.start_ts
        return float(last) >= float(self.end_ts)


class MissingIntervalTracker:
    """Tracks missing intervals and resolves business keys to BRUNCH type_ids.

    Resolution is lazy + memoized per (key_kind, key_value). Type_id lookup
    uses ``vocabs.type_vocab.labels`` which BRUNCH builds as
    ``"{alarm_source} | {alarm_type}"`` (or just ``"{alarm_source}"`` /
    ``"{alarm_type}"`` depending on the configured ``type_fields``). The
    matching is duck-typed so it also works when only one field is present.
    """

    def __init__(self, vocabs, type_fields=("alarm_source", "alarm_type")):
        self.vocabs = vocabs
        self.type_fields = tuple(type_fields or ("alarm_source", "alarm_type"))
        self.intervals: List[MissingInterval] = []
        self._key_to_type_ids: Dict[Tuple[str, str], List[int]] = {}

    # ---- mutation ----
    def add(self, interval: MissingInterval) -> MissingInterval:
        self.intervals.append(interval)
        return interval

    def declare_missing(self, key_kind: str, key_value: str, start_ts: float) -> MissingInterval:
        return self.add(
            MissingInterval(
                key_kind=key_kind,
                key_value=str(key_value),
                start_ts=float(start_ts),
                end_ts=None,
            )
        )

    def declare_recovered(self, key_kind: str, key_value: str, end_ts: float) -> int:
        n_closed = 0
        for interval in self.intervals:
            if interval.end_ts is not None:
                continue
            if interval.key_kind == key_kind and interval.key_value == str(key_value):
                interval.end_ts = float(end_ts)
                n_closed += 1
        return n_closed

    # ---- queries ----
    def open_intervals_at(self, t: float) -> List[MissingInterval]:
        return [iv for iv in self.intervals if iv.contains(float(t))]

    def has_open_intervals(self) -> bool:
        return any(iv.end_ts is None for iv in self.intervals)

    def intervals_needing_sampling(self, t: float) -> List[MissingInterval]:
        """Intervals (open OR closed-with-unsampled-tail) that still have time
        to sample up to ``t``. Use this rather than :meth:`open_intervals_at`
        to drive virtual-event injection so the tail of a just-recovered
        outage is not silently dropped.
        """
        t_f = float(t)
        return [iv for iv in self.intervals if iv.needs_sampling(t_f)]

    def has_intervals_needing_sampling(self, t: float) -> bool:
        t_f = float(t)
        return any(iv.needs_sampling(t_f) for iv in self.intervals)

    def compact(self) -> int:
        """Drop exhausted intervals. Returns the number evicted.

        Without compaction a long-running stream would accumulate one entry
        per past outage forever, turning every virtual-sampling tick into an
        O(N_outages) walk.
        """
        if not self.intervals:
            return 0
        before = len(self.intervals)
        self.intervals = [iv for iv in self.intervals if not iv.is_exhausted()]
        removed = before - len(self.intervals)
        if removed:
            # Cached type_id resolutions stay valid (vocab membership doesn't
            # change when an interval is dropped), so no cache invalidation
            # needed here.
            pass
        return removed

    def type_ids_for(self, interval: MissingInterval) -> List[int]:
        cache_key = (interval.key_kind, interval.key_value)
        cached = self._key_to_type_ids.get(cache_key)
        if cached is not None:
            return cached
        labels = list(self.vocabs.type_vocab.labels)
        result: List[int] = []
        if interval.key_kind == "type_id":
            try:
                tid = int(interval.key_value)
            except ValueError:
                tid = -1
            if 0 <= tid < len(labels):
                result.append(tid)
        elif interval.key_kind == "type_label":
            tid = self.vocabs.type_vocab.get(interval.key_value)
            if tid is not None:
                result.append(int(tid))
        elif interval.key_kind == "alarm_source":
            prefix = f"{interval.key_value} | "
            for idx, label in enumerate(labels):
                values = _type_label_field_values(label, self.type_fields)
                if (
                    values.get("alarm_source") == interval.key_value
                    or label == interval.key_value
                    or label.startswith(prefix)
                ):
                    result.append(idx)
        elif interval.key_kind == "alarm_type":
            suffix = f" | {interval.key_value}"
            for idx, label in enumerate(labels):
                values = _type_label_field_values(label, self.type_fields)
                if (
                    values.get("alarm_type") == interval.key_value
                    or label == interval.key_value
                    or label.endswith(suffix)
                ):
                    result.append(idx)
        self._key_to_type_ids[cache_key] = result
        return result

    def stats(self) -> dict:
        open_count = sum(1 for iv in self.intervals if iv.end_ts is None)
        return {
            "interval_count": len(self.intervals),
            "open_count": open_count,
            "closed_count": len(self.intervals) - open_count,
        }


def _parse_time_value(value):
    """Accept Unix timestamp (number / numeric string) or ISO datetime string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    try:
        return float(s)
    except ValueError:
        pass
    # Defer the heavier ISO parser until needed (and only if it's actually a
    # datetime-looking string), so this module stays importable without
    # alarm_events.io available at top level.
    from fault_grouping.alarm_events.io import parse_datetime_text

    return parse_datetime_text(s, "missing_interval time").timestamp()


def load_missing_from_json(path: str) -> List[MissingInterval]:
    """Load a list of ``MissingInterval`` from JSON.

    The file should be a JSON array of records, each carrying:
        - exactly one of ``alarm_source`` / ``alarm_type`` / ``type_label`` / ``type_id``
        - ``start`` (Unix timestamp or ISO datetime string)
        - ``end`` (same format, or null for still-open)

    Example::

        [
            {"alarm_source": "ne-12345",
             "start": "2024-05-01 09:00:00",
             "end":   "2024-05-01 09:30:00"},
            {"type_label": "ne-99 | link",
             "start": 1714564800,
             "end": null}
        ]
    """
    with open(path, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("missing intervals JSON must be a list of records")
    intervals: List[MissingInterval] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"each missing interval record must be a dict; got {type(record).__name__}")
        key_kind = None
        key_value = None
        for candidate in ("alarm_source", "alarm_type", "type_label", "type_id"):
            value = record.get(candidate)
            if value not in (None, ""):
                key_kind = candidate
                key_value = value
                break
        if key_kind is None:
            raise ValueError(
                "each missing interval record must specify exactly one of "
                f"{sorted(KEY_KINDS)}; got {record}"
            )
        start_ts = _parse_time_value(record.get("start"))
        if start_ts is None:
            raise ValueError(f"missing 'start' for record: {record}")
        intervals.append(
            MissingInterval(
                key_kind=key_kind,
                key_value=str(key_value),
                start_ts=start_ts,
                end_ts=_parse_time_value(record.get("end")),
            )
        )
    return intervals
