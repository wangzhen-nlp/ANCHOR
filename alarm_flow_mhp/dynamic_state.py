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

from typing import Callable, Iterable

import numpy as np


ALARM_STATE_TYPES = ("link", "power", "offline")
STATE_DIM = len(ALARM_STATE_TYPES)
_TYPE_INDEX = {t: i for i, t in enumerate(ALARM_STATE_TYPES)}


def alarm_state_index(alarm_type) -> int:
    """Index of a link/power/offline alarm type, or -1 if outside the set."""
    return _TYPE_INDEX.get(alarm_type, -1)


def device_state_pass(
    items: Iterable,
    *,
    is_clear: Callable[[object], bool],
    device_of: Callable[[object], str],
    alarm_type_of: Callable[[object], object],
) -> np.ndarray:
    """Forward pass producing the per-(non-clear)-event device-state snapshot.

    Parameters
    ----------
    items : iterable in non-decreasing time order (raises and clears mixed)
    is_clear : item -> bool (clear vs raise)
    device_of : item -> device id (alarm_source)
    alarm_type_of : item -> "link"/"power"/"offline" (or anything else → no
        state contribution)

    Returns
    -------
    ev_state : (n_nonclear, 3) uint8
        Row k = the device-state (link, power, offline) booleans for the k-th
        NON-CLEAR item, snapshotted JUST BEFORE that item (excludes itself).
        Aligned by order with the non-clear items in `items`.
    """
    counts: dict[str, list] = {}
    rows: list = []
    for it in items:
        dev = device_of(it)
        ti = _TYPE_INDEX.get(alarm_type_of(it), -1)
        cur = counts.get(dev)
        if is_clear(it):
            # State transition only; clears are not modeled events.
            if cur is not None and ti >= 0 and cur[ti] > 0:
                cur[ti] -= 1
            continue
        # Raise: snapshot BEFORE incrementing → own alarm excluded.
        if cur is None:
            rows.append((0, 0, 0))
        else:
            rows.append((1 if cur[0] else 0, 1 if cur[1] else 0, 1 if cur[2] else 0))
        if ti >= 0:
            if cur is None:
                cur = [0, 0, 0]
                counts[dev] = cur
            cur[ti] += 1
    if not rows:
        return np.zeros((0, STATE_DIM), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8)


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
    min-events-filtered subset), so we gather their snapshots by id — robust to
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


class DeviceStateTracker:
    """Incremental version of `device_state_pass` for streaming inference.

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
