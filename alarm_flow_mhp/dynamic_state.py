"""Clear-aware per-device alarm-state machine for dynamic α features.

A device's "dynamic state" at a moment is which alarm TYPES it currently has
UNCLEARED (raised but not yet cleared): a 3-boolean vector over
(link, power, offline). This is a time-varying covariate that lets the feature
mode condition excitation on context — e.g. "a link alarm that fires while the
device already has a power fault propagates differently".

The state is produced by a single forward pass over the FULL time-ordered
stream (raises AND clears). Per (device, alarm_type) we keep an active count:
a raise increments, a clear decrements (floored at 0); the boolean is count>0.

read-before-write ordering: for a raise we snapshot the device state BEFORE
incrementing, so the event's own alarm is excluded (otherwise the feature
"device has an active <type>" is trivially always-on for that event's type and
leaks the label). Clears update state but emit nothing (they are not modeled
events). The child/target event is excluded automatically because, frozen at a
time at or before it, it has not occurred yet.

The exact same routine must run at training and inference (train/infer
consistency), so it is parameterized by accessor callables rather than a fixed
event schema.
"""

from __future__ import annotations

import bisect
from typing import Callable, Iterable

import numpy as np


ALARM_STATE_TYPES = ("link", "power", "offline")
STATE_DIM = len(ALARM_STATE_TYPES)
_TYPE_INDEX = {t: i for i, t in enumerate(ALARM_STATE_TYPES)}


def alarm_state_index(alarm_type) -> int:
    """Index of a link/power/offline alarm type, or -1 if outside the set."""
    return _TYPE_INDEX.get(alarm_type, -1)


def build_event_states(
    full_stream: Iterable,
    modeled_events: list,
    *,
    is_clear: Callable[[object], bool],
    device_of: Callable[[object], str],
    alarm_type_of: Callable[[object], object],
) -> np.ndarray:
    """Per-modeled-event device-state snapshot, aligned by event identity.

    The state machine runs over the FULL time-ordered stream (`full_stream`,
    which still contains clears), so uncleared-alarm state is tracked correctly;
    each NON-CLEAR event's snapshot (excl self) is keyed by id(event). The
    modeled events (`sequence.events`) are the SAME dict objects (a clear- and
    model-filtered subset), so we gather their snapshots by id — robust to
    whatever filtering produced the modeled subset (no index alignment).

    Returns (len(modeled_events), 3) uint8 in modeled-event order.
    """
    tracker = DeviceStateTracker()
    snap_by_id: dict[int, tuple] = {}
    for it in sorted(full_stream, key=lambda e: float(e.get("ts", 0.0))):
        clear = bool(is_clear(it))
        snap = tracker.snapshot_then_apply(device_of(it), alarm_type_of(it), clear)
        if not clear:
            snap_by_id[id(it)] = (int(snap[0]), int(snap[1]), int(snap[2]))
    out = np.zeros((len(modeled_events), STATE_DIM), dtype=np.uint8)
    for k, ev in enumerate(modeled_events):
        s = snap_by_id.get(id(ev))
        if s is not None:
            out[k] = s
    return out


def states_to_combo(ev_state: np.ndarray) -> np.ndarray:
    """Pack a (n, 3) 0/1 state array into a (n,) combo index in [0, 8).

    combo = link + 2*power + 4*offline (link=bit0, power=bit1, offline=bit2),
    matching ALARM_STATE_TYPES order.
    """
    s = np.asarray(ev_state, dtype=np.int64)
    if s.size == 0:
        return np.zeros(0, dtype=np.int64)
    return s[:, 0] + 2 * s[:, 1] + 4 * s[:, 2]


def combo_bits(n_combos: int = 8) -> np.ndarray:
    """(n_combos, 3) bit-decomposition table; row k = [k&1, (k>>1)&1, (k>>2)&1]."""
    k = np.arange(n_combos, dtype=np.float64)
    return np.stack([k % 2, (k // 2) % 2, (k // 4) % 2], axis=1)


def mark_to_combo(mark) -> int:
    """Pack ONE (link, power, offline) 0/1 mark into its combo index — the
    scalar counterpart of states_to_combo/combo_bits (single owner of the bit
    order; keep all three consistent)."""
    if not mark or len(mark) < STATE_DIM:
        return 0
    return int(mark[0]) + 2 * int(mark[1]) + 4 * int(mark[2])


class DeviceStateTracker:
    """Incremental device-state machine for streaming inference (mirrors the
    forward pass used by `build_event_states`).

    Maintains the same per-(device, alarm_type) active counts. At inference the
    stream feeds events in time order; call `snapshot_then_apply` for each
    incoming alarm to get its device-state (excl self) and update the counts,
    or `state_of` to read a device's current booleans (e.g. the candidate
    parent's source device, whose snapshot was taken when it arrived).
    """

    def __init__(self):
        self._counts: dict[str, list] = {}

    def state_of(self, device: str) -> np.ndarray:
        cur = self._counts.get(device)
        if cur is None:
            return np.zeros(STATE_DIM, dtype=np.uint8)
        return np.array([1 if cur[0] else 0, 1 if cur[1] else 0, 1 if cur[2] else 0], dtype=np.uint8)

    def snapshot_then_apply(self, device: str, alarm_type, is_clear: bool) -> np.ndarray:
        """Return device-state BEFORE this event (excl self); then update.

        For a clear, updates state and returns the post-nothing snapshot (the
        return value is unused for clears in practice).
        """
        ti = _TYPE_INDEX.get(alarm_type, -1)
        cur = self._counts.get(device)
        if is_clear:
            if cur is not None and ti >= 0 and cur[ti] > 0:
                cur[ti] -= 1
            return self.state_of(device)
        snap = self.state_of(device)
        if ti >= 0:
            if cur is None:
                cur = [0, 0, 0]
                self._counts[device] = cur
            cur[ti] += 1
        return snap


class ObservedStateTimeline:
    """Read-only dynamic-state history built from observed stream events.

    It records the post-event observed state per device, so a hypothesised
    missing event can read the state at its proposed timestamp without mutating
    the state machine. Observed modeled events should still use the returned
    pre-event snapshot from :meth:`ingest`, matching training's read-before-write
    convention.
    """

    def __init__(self):
        self._tracker = DeviceStateTracker()
        self._times: dict[str, list[float]] = {}
        self._states: dict[str, list[tuple]] = {}

    def ingest(self, ts: float, device: str, alarm_type, is_clear: bool) -> tuple:
        device = str(device or "")
        snap = self._tracker.snapshot_then_apply(device, alarm_type, bool(is_clear))
        post = tuple(int(x) for x in self._tracker.state_of(device))
        times = self._times.setdefault(device, [])
        states = self._states.setdefault(device, [])
        if not states or states[-1] != post:
            times.append(float(ts))
            states.append(post)
        return tuple(int(x) for x in snap)

    def state_at(self, device: str, ts: float) -> tuple:
        device = str(device or "")
        times = self._times.get(device)
        if not times:
            return tuple(0 for _ in range(STATE_DIM))
        idx = bisect.bisect_right(times, float(ts)) - 1
        if idx < 0:
            return tuple(0 for _ in range(STATE_DIM))
        return self._states[device][idx]

    def source_mark_at(self, source_type, ts: float) -> tuple:
        try:
            _at, ne = source_type
        except Exception:
            ne = ""
        # Missing events read the observed state slice just before their
        # proposed timestamp, matching read-before-write event snapshots.
        return self.state_at(ne, np.nextafter(float(ts), -np.inf))

    def prune_before(self, cutoff_ts: float):
        """Drop old per-device history while preserving state continuity.

        The latest state change before ``cutoff_ts`` is retained as the baseline
        for future ``state_at`` calls. Entries at or after the cutoff are kept.
        """
        cutoff_ts = float(cutoff_ts)
        for device in list(self._times.keys()):
            times = self._times[device]
            states = self._states[device]
            idx = bisect.bisect_left(times, cutoff_ts)
            if idx <= 1:
                continue
            keep_from = idx - 1
            self._times[device] = times[keep_from:]
            self._states[device] = states[keep_from:]
