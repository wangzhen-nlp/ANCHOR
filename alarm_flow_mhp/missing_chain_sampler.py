#!/usr/bin/env python3
"""Fixed-lag missing-event chain sampler for alarm-flow MHP (online inference).

Goal
----
Given a trained MHP (fixed μ / α / β) and a forward stream of *observed* alarms,
reconstruct cascades that may have **unobserved (missing) triggering events of
KNOWN types** in between. An orphan alarm that no observed event explains well
can instead be explained by hypothesising a missing parent of some known type;
that missing parent must itself be explainable (by another event — observed or
missing — or by the background rate). Chains can be multi-hop: Y → X → e where
both Y and X are missing.

This is the lightweight, forward, *fixed-lag* analog of Shelton et al. 2018
("Hawkes Process Inference with Missing Data"). We do NOT add new latent labels
(that is the "hidden labels" variant and is statistically ill-posed at inference
without a missingness model); we only add *events of existing types*.

Why a fixed-lag window
----------------------
A purely single-pass forward assigner can't honour the coherence condition
("a missing parent must itself be explained") because that condition is a
*joint* property of the whole latent tree — see the module-level discussion in
the design notes. So we keep a trailing window of recent events whose latent
structure (parents + missing events) is continuously resampled by a persistent
MCMC chain (warm-started across ticks). When an event ages past the lag it is
*committed* (frozen) and emitted with a marginal-posterior summary aggregated
over the sweeps it experienced while in-window.

Mode transparency
-----------------
The sampler talks to the model ONLY through :class:`ModelAdapter`. Both device
and feature edge modes, and both exp and piecewise kernels, materialise at
inference into the same (μ, edge-table, kernel-eval) surface, so a single
adapter implementation per backend is enough and the sampler core is identical
across modes.

Status
------
Skeleton / v1. Parent resampling (Move A) is an exact Gibbs step. Birth/death
(Move B) uses a documented v1 acceptance ratio carrying the dominant likelihood
terms + a per-missing-event log-prior (κ knob); the exact reversible-jump
correction is marked TODO. The sampler is wired into ``stream_alarm_mhp`` via
``--impute`` and is also exercised directly by the unit tests with small
adapters.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, runtime_checkable


NEG_INF = float("-inf")
EPS = 1e-12

# A model "type" is an opaque hashable key the sampler never interprets — only
# the adapter does. Device mode uses an int vocab id; feature mode uses an
# (alarm_type, ne) tuple.
TypeKey = object

# Output rule tags, mirroring alarm_flow_brunch so downstream group/visual
# consumers can treat MHP-imputed groups the same way as BRUNCH ones.
MHP_RULE = "alarm_flow_mhp"
MHP_VIRTUAL_RULE = "alarm_flow_mhp_virtual_event"


# --------------------------------------------------------------------------
# Model interface — the only mode-specific surface
# --------------------------------------------------------------------------


@runtime_checkable
class ModelAdapter(Protocol):
    """Read-only view of a trained MHP, in terms the sampler needs.

    ``dt_sec`` values are real (un-scaled) seconds. They may be slightly
    negative when timestamp-jitter slack is enabled (parent timestamp after the
    child); the adapter applies ``time_scale`` and the late-parent penalty
    internally so the sampler never has to know about scaled time.
    """

    def mu(self, type_id: TypeKey) -> float:
        """Background (immigrant) rate of ``type_id`` — intensity density."""

    def kernel_intensity(
        self,
        source_type: int,
        target_type: int,
        dt_sec: float,
        *,
        source_mark=None,
    ) -> float:
        """Triggering intensity α·φ(dt) of one ``source_type`` event on a
        ``target_type`` event ``dt_sec`` later. Returns 0 for non-edges."""

    def compensator(
        self,
        source_type: int,
        target_type: int,
        dt_sec: float,
        *,
        source_mark=None,
    ) -> float:
        """∫_0^{dt} α·φ(u) du — expected number of ``target_type`` children a
        single ``source_type`` event triggers within ``dt_sec``. Used for the
        survival penalty when adding/removing an event."""

    def candidate_sources(self, target_type: int) -> list[tuple[int, float]]:
        """Source types that can trigger ``target_type``, as ``(source_type,
        weight)`` pairs (weight ∝ total influence α, used for birth proposals).
        Empty ⇒ no missing parent can be hypothesised for this target."""

    def type_meta(self, type_id: TypeKey) -> dict:
        """Output metadata for a synthesised missing event of ``type_id``
        (e.g. ``{"type_label", "alarm_source", "alarm_type", "site_id"}``)."""


@dataclass
class ExpKernelAdapter:
    """Concrete :class:`ModelAdapter` for an exponential kernel.

    Built from plain dicts so it is usable both in unit tests and as the basis
    for a real device-mode adapter (device params *are* just edge arrays). A
    classmethod to build one from a trained artifact is sketched at the bottom.

    Parameters
    ----------
    mu_by_type : {type_id: μ}
    edges : {(target_type, source_type): (alpha, beta)}
        Only present pairs are edges; everything else has zero influence.
    time_scale_sec : real seconds per model time unit (matches training).
    meta_by_type : optional {type_id: metadata dict}
    """

    mu_by_type: dict[int, float]
    edges: dict[tuple[int, int], tuple[float, float]]
    time_scale_sec: float = 60.0
    meta_by_type: dict[int, dict] = field(default_factory=dict)
    time_slack_sec: float = 0.0
    late_penalty_half_life_sec: float = 1.0
    _sources_by_target: dict[int, list[tuple[int, float]]] = field(default_factory=dict, init=False)
    _targets_by_source: dict[int, list[int]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be > 0")
        if self.time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if self.late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        by_target: dict[int, list[tuple[int, float]]] = {}
        by_source: dict[int, list[int]] = {}
        for (tgt, src), (alpha, _beta) in self.edges.items():
            if alpha > 0:
                by_target.setdefault(int(tgt), []).append((int(src), float(alpha)))
                by_source.setdefault(int(src), []).append(int(tgt))
        # Stable, strongest-first — birth proposals favour high-α sources.
        for tgt in by_target:
            by_target[tgt].sort(key=lambda sa: (-sa[1], sa[0]))
        self._sources_by_target = by_target
        self._targets_by_source = by_source

    def mu(self, type_id: int) -> float:
        return float(self.mu_by_type.get(int(type_id), 0.0))

    def kernel_intensity(
        self,
        source_type: int,
        target_type: int,
        dt_sec: float,
        *,
        source_mark=None,
    ) -> float:
        dt_eff_sec, late_weight = self._signed_dt(dt_sec)
        if late_weight <= 0:
            return 0.0
        edge = self.edges.get((int(target_type), int(source_type)))
        if edge is None:
            return 0.0
        alpha, beta = edge
        dt = dt_eff_sec / self.time_scale_sec
        return float(alpha * beta * math.exp(-beta * dt) * late_weight)

    def compensator(
        self,
        source_type: int,
        target_type: int,
        dt_sec: float,
        *,
        source_mark=None,
    ) -> float:
        edge = self.edges.get((int(target_type), int(source_type)))
        if edge is None:
            return 0.0
        alpha, beta = edge
        pos = 0.0
        if dt_sec > 0:
            dt = dt_sec / self.time_scale_sec
            # ∫_0^dt α·β·e^{-β u} du = α (1 - e^{-β dt})
            pos = alpha * (1.0 - math.exp(-beta * dt))
        neg = alpha * beta * self._negative_penalty_integral_model()
        return float(pos + neg)

    def _signed_dt(self, dt_sec: float) -> tuple[float, float]:
        if dt_sec >= 0:
            return float(dt_sec), 1.0
        late = -float(dt_sec)
        if self.time_slack_sec <= 0 or late > self.time_slack_sec:
            return 0.0, 0.0
        lam = math.log(2.0) / self.late_penalty_half_life_sec
        return 0.0, math.exp(-lam * late)

    def _negative_penalty_integral_model(self) -> float:
        if self.time_slack_sec <= 0:
            return 0.0
        slack = self.time_slack_sec / self.time_scale_sec
        half = self.late_penalty_half_life_sec / self.time_scale_sec
        lam = math.log(2.0) / half
        return float((1.0 - math.exp(-lam * slack)) / lam)

    def candidate_sources(self, target_type: int) -> list[tuple[int, float]]:
        return list(self._sources_by_target.get(int(target_type), ()))

    def outgoing_targets(self, source_type: int) -> list[int]:
        """Target types ``source_type`` can trigger — used by the compensator."""
        return list(self._targets_by_source.get(int(source_type), ()))

    def type_meta(self, type_id: int) -> dict:
        return dict(self.meta_by_type.get(int(type_id), {}))


# --------------------------------------------------------------------------
# Window state
# --------------------------------------------------------------------------


@dataclass
class SamplerEvent:
    """A node in the latent cascade forest within the trailing window."""

    eid: int
    ts: float
    type_id: TypeKey
    observed: bool                       # True = evidence alarm; False = imputed
    parent: int = -1                     # eid of parent; -1 = immigrant (μ root)
    children: set[int] = field(default_factory=set)
    meta: dict = field(default_factory=dict)
    # Dynamic feature mode: source device's frozen uncleared-state mark at this
    # event's fire time. Missing events may read an observed-state timeline, but
    # never write back to that state machine.
    src_mark: tuple = ()
    committed: bool = False              # frozen (aged past lag); immutable
    depth: int = 0                       # missing-chain depth (observed = 0)
    # Marginal accumulation over the sweeps this event lives through, so freezing
    # can lock the MAP (posterior) parent instead of a single noisy draw.
    parent_votes: Counter = field(default_factory=Counter)
    sweep_count: int = 0
    # Posterior confidence of the parent chosen at freeze time (votes / sweeps).
    commit_parent_prob: float = 1.0

    def is_missing(self) -> bool:
        return not self.observed


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    lag_sec: float = 300.0              # commit delay (latency vs completeness)
    history_window_sec: float = 900.0  # kernel reach for candidate parents
    time_slack_sec: float = 0.0        # small timestamp-jitter tolerance
    late_penalty_half_life_sec: float = 1.0
    sweeps_per_tick: int = 2           # local MCMC sweeps per ingest
    max_missing: int = 200             # cap on live missing events in window
    max_depth: int = 4                 # cap on missing-chain depth
    missing_log_prior: float = -2.0    # log-prior penalty per missing event (κ)
                                       # more negative ⇒ fewer / shallower chains
    max_births_per_sweep: int = 8      # cap on NEW missing events born per sweep
                                       # (rate limit; chains deepen over sweeps)
    max_history_events: int = 256      # cap on candidate parents scored per event
                                       # (nearest-in-time first) — bounds per-tick
                                       # cost; the dominant perf knob at scale
    sweep_recent_events: int = 64      # PARENT re-sampling (Move A) only re-touches
                                       # the most recent N uncommitted events. This
                                       # is a bounded local approximation: an older
                                       # event's candidate parents are fixed in the
                                       # past, but repeated Gibbs draws would still
                                       # refine its parent votes. (It does NOT gate
                                       # birth/death — those cover the whole active
                                       # window, see below — otherwise older orphans
                                       # would be starved of imputation chances.)
    future_candidate_reset_limit: int = 0  # When time_slack allows a newly inserted
                                       # event to become a future parent, clear and
                                       # re-sample all affected older events by
                                       # default. Set >0 to cap this expensive repair
                                       # to the nearest N affected events.
    max_birth_attempts_per_sweep: int = 32  # birth is attempted over ALL active
                                       # orphans (fair, shuffled) but bounded to this
                                       # many ATTEMPTS/sweep to cap cost without
                                       # introducing an age bias.
    commit_check_interval_sec: float = 0.0  # only rebuild/close cascades when event
                                       # time has advanced this far since the last
                                       # close (the O(live) cluster scan is wasteful
                                       # every tick). 0 = every tick (test default);
                                       # the stream sets a fraction of lag. Affects
                                       # only OUTPUT timing, never sampling.
    seed: int = 0

    def window_sec(self) -> float:
        # Committed events must stay available as candidate parents while any
        # still-active event can reach them, so keep them until fully out of
        # kernel reach.
        return self.lag_sec + self.history_window_sec + self.time_slack_sec

    def freeze_delay_sec(self) -> float:
        return self.lag_sec + self.time_slack_sec

    def validate(self):
        if self.lag_sec <= 0:
            raise ValueError("lag_sec must be > 0")
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be > 0")
        if self.time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if self.late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        if self.sweeps_per_tick < 0:
            raise ValueError("sweeps_per_tick must be >= 0")
        if self.max_missing < 0 or self.max_depth < 0:
            raise ValueError("caps must be >= 0")
        if self.max_births_per_sweep < 0:
            raise ValueError("max_births_per_sweep must be >= 0")
        if self.max_history_events < 1:
            raise ValueError("max_history_events must be >= 1")
        if self.sweep_recent_events < 1:
            raise ValueError("sweep_recent_events must be >= 1")
        if self.future_candidate_reset_limit < 0:
            raise ValueError("future_candidate_reset_limit must be >= 0")
        if self.max_birth_attempts_per_sweep < 0:
            raise ValueError("max_birth_attempts_per_sweep must be >= 0")


class MissingChainSampler:
    """Persistent fixed-lag chain sampler. Feed observed alarms via
    :meth:`ingest`; collect :class:`CommitRecord`s as events age out.
    """

    def __init__(self, adapter: ModelAdapter, config: Optional[SamplerConfig] = None):
        self.adapter = adapter
        self.config = config or SamplerConfig()
        self.config.validate()
        self.rng = random.Random(self.config.seed)
        self.events: dict[int, SamplerEvent] = {}
        self._order: list[int] = []          # eids kept time-ascending
        self._order_ts: list[float] = []     # parallel ts array for bisect
        self._next_eid: int = 0
        self.now: float = NEG_INF
        self._missing_count: int = 0
        # Incremental indices so birth/death don't scan the whole active window:
        #  _orphan_list/_orphan_idx : uncommitted immigrants (parent == -1) as a
        #    swap-remove list → O(1) add/remove + O(k) uniform sampling for birth.
        #  _missing_set : uncommitted missing events → death iterates only these.
        self._orphan_list: list[int] = []
        self._orphan_idx: dict[int, int] = {}
        self._missing_set: set[int] = set()
        # Incremental-freeze frontier + close throttle (both O(live)-avoidance):
        self._frozen_through_ts: float = NEG_INF
        self._last_commit_check_ts: float = NEG_INF
        # stats
        self.births = 0
        self.deaths = 0
        self.committed_count = 0
        self.closed_group_count = 0

    # ---- public API ------------------------------------------------------

    def ingest(
        self,
        ts: float,
        type_id: TypeKey,
        meta: Optional[dict] = None,
        *,
        src_mark=None,
    ) -> list[dict]:
        """Ingest one observed alarm (time-ascending); return any closed groups.

        A closed group is a brunch-style dict (see :meth:`_build_group`) covering
        a whole cascade — observed alarms AND the imputed missing events that
        bridge them — emitted once the cascade leaves kernel reach.
        """
        ts = float(ts)
        if ts < self.now:
            # Forward-only contract; out-of-order events are clamped to `now`.
            ts = self.now
        self.now = ts
        # type_id is an opaque hashable key (int in device mode, (alarm_type,
        # ne) tuple in feature mode) — never coerced here.
        ev = self._new_event(ts=ts, type_id=type_id, observed=True,
                             meta=dict(meta or {}), depth=0, src_mark=src_mark)
        self._insert_ordered(ev)
        self._reset_votes_for_future_candidate(ev)
        self._gibbs_parent(ev)            # initial assignment
        for _ in range(self.config.sweeps_per_tick):
            self._sweep()
        self._freeze_aged()               # incremental: O(newly-frozen), cheap
        # Closing rebuilds cascades over the window (O(live)); throttle it by
        # event time so it doesn't run every tick. Output-timing only.
        if (self.now - self._last_commit_check_ts) >= self.config.commit_check_interval_sec:
            self._last_commit_check_ts = self.now
            return self._close_clusters()
        return []

    def flush(self) -> list[dict]:
        """Close every remaining cascade (call at end of stream)."""
        if self._order:
            self.now = self.events[self._order[-1]].ts + self.config.window_sec() + 1.0
        self._freeze_aged(force=True)
        return self._close_clusters(force=True)

    # ---- event bookkeeping ----------------------------------------------

    def _orphan_add(self, eid: int):
        if eid in self._orphan_idx:
            return
        self._orphan_idx[eid] = len(self._orphan_list)
        self._orphan_list.append(eid)

    def _orphan_remove(self, eid: int):
        i = self._orphan_idx.pop(eid, None)
        if i is None:
            return
        last = self._orphan_list.pop()
        if last != eid:
            self._orphan_list[i] = last
            self._orphan_idx[last] = i

    def _zero_src_mark(self) -> tuple:
        n_dynamic = int(getattr(self.adapter, "n_dynamic", 0) or 0)
        return tuple(0 for _ in range(n_dynamic))

    def _source_mark_for_missing(self, source_type, ts: float) -> tuple:
        source_mark_at = getattr(self.adapter, "source_mark_at", None)
        if source_mark_at is None:
            return self._zero_src_mark()
        return self._normalise_src_mark(source_mark_at(source_type, ts))

    def _normalise_src_mark(self, src_mark) -> tuple:
        n_dynamic = int(getattr(self.adapter, "n_dynamic", 0) or 0)
        if n_dynamic <= 0:
            return ()
        if src_mark is None:
            return self._zero_src_mark()
        vals = tuple(int(v) for v in src_mark)
        if len(vals) != n_dynamic:
            return tuple((vals + self._zero_src_mark())[:n_dynamic])
        return vals

    def _new_event(self, *, ts, type_id, observed, meta, depth, src_mark=None) -> SamplerEvent:
        ev = SamplerEvent(eid=self._next_eid, ts=ts, type_id=type_id,
                          observed=observed, meta=meta, depth=depth,
                          src_mark=self._normalise_src_mark(src_mark))
        self._next_eid += 1
        self.events[ev.eid] = ev
        # New events start as immigrants (parent == -1) until Gibbs/birth.
        self._orphan_add(ev.eid)
        if not observed:
            self._missing_count += 1
            self._missing_set.add(ev.eid)
        return ev

    def _insert_ordered(self, ev: SamplerEvent):
        # Keep _order (eids) and _order_ts (parallel times) sorted by ts. Append
        # is the common case (forward stream); bisect handles past-dated missing
        # events. The parallel ts array lets _candidate_parents bisect to ev's
        # neighbourhood in O(log n) instead of scanning from the front.
        ts = ev.ts
        if not self._order_ts or self._order_ts[-1] <= ts:
            self._order.append(ev.eid)
            self._order_ts.append(ts)
            return
        pos = bisect.bisect_right(self._order_ts, ts)
        self._order.insert(pos, ev.eid)
        self._order_ts.insert(pos, ts)

    def _reset_votes_for_future_candidate(self, new_ev: SamplerEvent):
        """A newly inserted event can become a future parent for older events
        within slack. Those older events' candidate sets just changed, so old
        parent votes collected before this candidate existed should not dominate
        their eventual MAP commit.
        """
        slack = self.config.time_slack_sec
        if slack <= 0:
            return
        lo = bisect.bisect_left(self._order_ts, new_ev.ts - slack)
        hi = bisect.bisect_left(self._order_ts, new_ev.ts)
        limit = self.config.future_candidate_reset_limit
        if limit > 0:
            # Optional approximation: a dense slack window can hold hundreds of
            # events, so only repair the nearest affected events when configured.
            # The default (0) keeps the exact pre-optimization behaviour.
            lo = max(lo, hi - limit)
        for i in range(lo, hi):
            ev = self.events.get(self._order[i])
            if ev is None or ev.committed or ev.eid == new_ev.eid:
                continue
            ev.parent_votes.clear()
            ev.sweep_count = 0
            self._gibbs_parent(ev)

    def _remove_event(self, ev: SamplerEvent):
        # Detach from parent / children before dropping.
        if ev.parent != -1:
            par = self.events.get(ev.parent)
            if par is not None:
                par.children.discard(ev.eid)
        for cid in list(ev.children):
            child = self.events.get(cid)
            if child is not None:
                child.parent = -1          # orphaned; re-parented next sweep
                self._orphan_add(cid)
        if ev.is_missing():
            self._missing_count -= 1
        self._orphan_remove(ev.eid)
        self._missing_set.discard(ev.eid)
        self.events.pop(ev.eid, None)
        # Remove from the parallel order arrays. Locate by ts via bisect, then
        # scan the (tiny) equal-ts run for the matching eid.
        lo = bisect.bisect_left(self._order_ts, ev.ts)
        hi = bisect.bisect_right(self._order_ts, ev.ts)
        for i in range(lo, hi):
            if self._order[i] == ev.eid:
                del self._order[i]
                del self._order_ts[i]
                break

    def _set_parent(self, child: SamplerEvent, parent_eid: int):
        # NOTE: `depth` is the child's missing-chain layer, fixed at birth (a
        # missing event's depth = how many missing hops it sits above the
        # observed event it ultimately explains). It does NOT depend on the
        # event's own parent, so re-parenting must not touch it.
        if parent_eid == child.eid:
            parent_eid = -1
        elif parent_eid != -1:
            parent = self.events.get(parent_eid)
            if parent is None or self._is_descendant(parent_eid, child.eid):
                parent_eid = -1
        if child.parent == parent_eid:
            return
        if child.parent != -1:
            old = self.events.get(child.parent)
            if old is not None:
                old.children.discard(child.eid)
        child.parent = parent_eid
        if parent_eid != -1:
            self.events[parent_eid].children.add(child.eid)
            self._orphan_remove(child.eid)
        else:
            self._orphan_add(child.eid)

    # ---- candidate parents ----------------------------------------------

    def _candidate_parents(self, ev: SamplerEvent):
        """Up to ``max_history_events`` candidate parents nearest in signed time
        distance within [ev.ts-history, ev.ts+slack]. Returns parallel lists
        ``(eids, src_types, dts)`` so the caller can batch-score intensities.

        Capping + nearest-first matters: under an exp kernel the closest events
        carry almost all the mass, and the cap bounds per-event cost from
        O(window) to O(cap) — the difference between tractable and not at scale.
        """
        reach = self.config.history_window_sec
        slack = self.config.time_slack_sec
        cap = self.config.max_history_events
        eids: list[int] = []
        src_types: list = []
        dts: list[float] = []

        def maybe_add(pos: int) -> bool:
            oid = self._order[pos]
            if oid == ev.eid:
                return False
            other = self.events.get(oid)
            if other is None:
                return False
            dt = ev.ts - other.ts
            if dt < -slack or dt > reach:
                return False
            if self._is_descendant(oid, ev.eid):
                return False
            eids.append(oid)
            src_types.append(other.type_id)
            dts.append(dt)
            return True

        left = bisect.bisect_left(self._order_ts, ev.ts) - 1
        right = bisect.bisect_left(self._order_ts, ev.ts)
        n = len(self._order_ts)
        while len(eids) < cap:
            left_ok = left >= 0 and ev.ts - self._order_ts[left] <= reach
            right_ok = slack > 0 and right < n and self._order_ts[right] - ev.ts <= slack
            if not left_ok and not right_ok:
                break
            if left_ok and right_ok:
                left_dist = ev.ts - self._order_ts[left]
                right_dist = self._order_ts[right] - ev.ts
                take_left = left_dist <= right_dist
            else:
                take_left = left_ok
            if take_left:
                maybe_add(left)
                left -= 1
            else:
                maybe_add(right)
                right += 1
        return eids, src_types, dts

    def _is_descendant(self, maybe_descendant: int, ancestor: int) -> bool:
        cur = self.events.get(maybe_descendant)
        guard = 0
        while cur is not None and cur.parent != -1:
            if cur.parent == ancestor:
                return True
            cur = self.events.get(cur.parent)
            guard += 1
            if guard > len(self.events) + 1:
                break
        return False

    # ---- Move A: parent Gibbs (exact) -----------------------------------

    def _gibbs_parent(self, ev: SamplerEvent):
        """Resample ev's parent ∝ triggering intensity, immigrant ∝ μ.

        Exact Gibbs: the only joint-likelihood term that depends on ev's parent
        is ev's single incoming edge, so the conditional is just the normalised
        candidate intensities vs μ (Shelton Move 3 in Gibbs form).
        """
        if ev.committed:
            return
        eids, src_types, dts = self._candidate_parents(ev)
        src_marks = [self.events[pid].src_mark for pid in eids]
        intens = self._batch_intensity(ev.type_id, src_types, dts, src_marks=src_marks)
        mu = self.adapter.mu(ev.type_id)
        weights = [mu]
        choices = [-1]
        for pid, inten in zip(eids, intens):
            if inten > EPS:
                weights.append(float(inten))
                choices.append(pid)
        total = sum(weights)
        if total <= 0:
            self._set_parent(ev, -1)
        else:
            r = self.rng.random() * total
            acc = 0.0
            picked = -1
            for w, c in zip(weights, choices):
                acc += w
                if r <= acc:
                    picked = c
                    break
            self._set_parent(ev, picked)
        # Record the marginal vote (post-move) for the commit summary.
        ev.parent_votes[ev.parent] += 1
        ev.sweep_count += 1

    def _batch_intensity(self, target_type, src_types: list, dts: list, *, src_marks=None):
        """Triggering intensities for a batch of candidate parents. Uses the
        adapter's vectorised path when available (essential for feature mode,
        where each α is a feature build) and falls back to per-pair otherwise."""
        if not src_types:
            return []
        batch = getattr(self.adapter, "kernel_intensity_batch", None)
        if batch is not None:
            return batch(target_type, src_types, dts, src_marks=src_marks)
        src_marks = src_marks or [None] * len(src_types)
        return [self.adapter.kernel_intensity(s, target_type, dt, source_mark=mark)
                for s, dt, mark in zip(src_types, dts, src_marks)]

    # ---- Move B: birth / death of missing events ------------------------

    def _try_birth(self, orphan: SamplerEvent) -> bool:
        """Propose a missing parent for an immigrant ``orphan`` (observed or
        missing) and accept by a v1 likelihood ratio.

        Multi-hop falls out for free: a freshly born missing event is itself an
        immigrant, so a later sweep can birth ITS parent — up to ``max_depth``.
        """
        cfg = self.config
        if orphan.parent != -1:
            return False                   # only immigrants are birth targets
        if orphan.committed:
            return False
        if orphan.depth >= cfg.max_depth:
            return False
        if self._missing_count >= cfg.max_missing:
            return False
        cands = self.adapter.candidate_sources(orphan.type_id)
        if not cands:
            return False
        # Birth must land in the still-mutable region, so committed structure
        # stays final. With timestamp slack, a missing parent can be slightly
        # after the child, but never beyond the current stream time.
        lower = max(self.now - cfg.freeze_delay_sec(), orphan.ts - cfg.history_window_sec)
        upper = min(self.now, orphan.ts + cfg.time_slack_sec)
        if upper - lower <= EPS:
            return False

        # --- propose source type s ∝ edge weight ---
        s_type = self._weighted_choice(cands)
        # --- propose time t' from the signed kernel-implied parent-time density ---
        #     dt = child - parent may be slightly negative under timestamp slack.
        #     We sample from the actual intensity profile by inverse-CDF on a
        #     fine grid (kernel-agnostic).
        t_prime = self._propose_parent_time(s_type, orphan, lower, upper)
        if t_prime is None:
            return False
        dt = orphan.ts - t_prime

        source_mark = self._source_mark_for_missing(s_type, t_prime)
        inten = self.adapter.kernel_intensity(
            s_type, orphan.type_id, dt, source_mark=source_mark
        )
        if inten <= EPS:
            return False
        mu_orphan = self.adapter.mu(orphan.type_id)

        # --- likelihood ratio (v1, dominant terms) ---
        # orphan switches immigrant→child of X:   inten / μ_orphan
        # X's own incoming term (immigrant for now): μ_s
        # X's survival penalty over its window:    exp(-compensator_total)
        # per-missing-event prior:                 exp(missing_log_prior)
        #
        # NOTE (asymmetry with _try_death, intentional — not a bug): here X's
        # incoming term is μ_s because at PROPOSAL time X has no parent yet — the
        # _gibbs_parent(x) below (a separate, reversible Gibbs move) assigns it
        # only AFTER acceptance. _try_death instead evaluates the CURRENT state,
        # where X may already have a real parent, so it uses the actual incoming
        # intensity (see _incoming_intensity). Birth=proposal-state (immigrant),
        # death=current-state (actual parent); the intervening Gibbs re-parent
        # reconciles the two.
        mu_s = self.adapter.mu(s_type)
        comp = self._total_compensator(s_type, t_prime, source_mark=source_mark)
        log_ratio = (
            math.log(max(inten, EPS)) - math.log(max(mu_orphan, EPS))
            + math.log(max(mu_s, EPS))
            - comp
            + cfg.missing_log_prior
        )
        # TODO(rj): exact reversible-jump correction — divide by the birth
        # proposal density q(s,t') and multiply by the reverse death proposal
        # density. Because t' is drawn from the kernel profile the dominant
        # kernel factors largely cancel; this v1 omits the residual proposal
        # ratio. See Shelton 2018 Move 1/2 acceptance ratios.
        if not self._accept(log_ratio):
            return False

        x = self._new_event(ts=t_prime, type_id=s_type, observed=False,
                            meta=self.adapter.type_meta(s_type),
                            depth=orphan.depth + 1,
                            src_mark=source_mark)
        self._insert_ordered(x)
        self._reset_votes_for_future_candidate(x)
        self._set_parent(orphan, x.eid)
        self._gibbs_parent(x)              # give X its own parent immediately
        self.births += 1
        return True

    def _incoming_intensity(self, ev: SamplerEvent) -> float:
        """ev's ACTUAL incoming-edge term in the joint: μ if it's an immigrant,
        else the triggering intensity from its current (real) parent."""
        if ev.parent == -1:
            return self.adapter.mu(ev.type_id)
        par = self.events.get(ev.parent)
        if par is None:
            return self.adapter.mu(ev.type_id)
        return self.adapter.kernel_intensity(
            par.type_id, ev.type_id, ev.ts - par.ts, source_mark=par.src_mark
        )

    def _try_death(self, ev: SamplerEvent) -> bool:
        """Remove a childless missing event by the joint likelihood ratio of the
        current state with vs without it.

        The removed terms are ev's incoming edge, its survival penalty exp(-comp)
        and the per-missing-event prior. Crucially the incoming edge is ev's
        ACTUAL support — μ only if ev is still an immigrant, otherwise the
        triggering intensity from its current parent. (After birth, a later
        Gibbs step may have given ev a real parent; charging it μ here would
        over-delete missing events a parent genuinely supports.)
        """
        if not ev.is_missing() or ev.committed or ev.children:
            return False
        comp = self._total_compensator(ev.type_id, ev.ts, source_mark=ev.src_mark)
        incoming = self._incoming_intensity(ev)
        # log p(without ev) / p(with ev) = -(log incoming - comp + prior)
        log_ratio = -(math.log(max(incoming, EPS)) - comp + self.config.missing_log_prior)
        if not self._accept(log_ratio):
            return False
        self._remove_event(ev)
        self.deaths += 1
        return True

    def _total_compensator(self, source_type, ts: float, *, source_mark=None) -> float:
        """Sum of expected children a ``source_type`` event at ``ts`` would
        trigger across all target types within the remaining window. Acts as the
        survival penalty exp(-Φ) for introducing the event.

        Adapter contract (checked in priority order):
          1. ``adapter.total_compensator(source, horizon_sec)`` — a batched /
             cached implementation (feature mode uses this: the per-source Σα is
             horizon-independent and cached, so only the (1-e^{-βH}) factor is
             recomputed per call).
          2. ``adapter.outgoing_targets(source)`` + ``adapter.compensator(...)``
             — the simple per-edge loop (device mode).
          3. Neither ⇒ 0; the κ prior alone regularises.
        """
        horizon = min(max(self.now - ts, 0.0), self.config.history_window_sec)
        batched = getattr(self.adapter, "total_compensator", None)
        if batched is not None:
            return float(batched(source_type, horizon, source_mark=source_mark))
        outgoing = getattr(self.adapter, "outgoing_targets", None)
        if outgoing is None:
            return 0.0
        total = 0.0
        for tgt in outgoing(source_type):
            total += self.adapter.compensator(
                source_type, tgt, horizon, source_mark=source_mark
            )
        return total

    # ---- proposal helpers -----------------------------------------------

    def _weighted_choice(self, items: list[tuple[int, float]]) -> int:
        total = sum(w for _, w in items)
        if total <= 0:
            return items[self.rng.randrange(len(items))][0]
        r = self.rng.random() * total
        acc = 0.0
        for k, w in items:
            acc += w
            if r <= acc:
                return k
        return items[-1][0]

    def _propose_parent_time(self, s_type: int, orphan: SamplerEvent,
                             lower: float, upper: float) -> Optional[float]:
        """Sample a parent time in (lower, upper) ∝ intensity profile.

        Inverse-CDF on a coarse grid — kernel-agnostic so it works for exp and
        piecewise alike (the adapter only needs kernel_intensity).
        """
        span = upper - lower
        if span <= 0:
            return None
        n = 32
        step = span / n
        grid = [lower + (i + 0.5) * step for i in range(n)]
        marks = [self._source_mark_for_missing(s_type, t) for t in grid]
        batch = getattr(self.adapter, "kernel_intensity_batch", None)
        if batch is not None:
            # One batched α call over the whole grid (per-point source marks)
            # instead of 32 scalar kernel_intensity calls.
            dts = [orphan.ts - t for t in grid]
            ws = [max(float(w), 0.0)
                  for w in batch(orphan.type_id, [s_type] * n, dts, src_marks=marks)]
        else:
            ws = [max(self.adapter.kernel_intensity(s_type, orphan.type_id, orphan.ts - t,
                                                    source_mark=m), 0.0)
                  for t, m in zip(grid, marks)]
        total = float(sum(ws))
        if total <= 0:
            return None
        r = self.rng.random() * total
        cum = 0.0
        for t, w in zip(grid, ws):
            cum += w
            if r <= cum:
                # jitter within the cell so times aren't quantised to the grid
                return min(upper - EPS, max(lower + EPS, t + (self.rng.random() - 0.5) * step))
        return grid[-1]

    def _accept(self, log_ratio: float) -> bool:
        if log_ratio >= 0:
            return True
        return self.rng.random() < math.exp(log_ratio)

    # ---- one sweep over the active set ----------------------------------

    def _active_eids(self) -> list[int]:
        """The most recent ``sweep_recent_events`` uncommitted events (LOCAL
        sweep scope), time-ascending. Walking from the newest end and stopping
        early bounds per-tick work independent of how dense the lag window is."""
        cutoff = self.now - self.config.freeze_delay_sec()
        limit = self.config.sweep_recent_events
        out: list[int] = []
        for i in range(len(self._order) - 1, -1, -1):
            if self._order_ts[i] <= cutoff:
                break
            ev = self.events.get(self._order[i])
            if ev is None or ev.committed:
                continue
            out.append(self._order[i])
            if len(out) >= limit:
                break
        out.reverse()
        return out

    def _sweep(self):
        # ---- Move A (re-parent): LOCAL to the most recent events. This is a
        # bounded approximation: an older event's candidate parents are fixed in
        # the past, but extra Gibbs draws would still refine its parent votes.
        for eid in self._active_eids():            # recent-N
            ev = self.events.get(eid)
            if ev is not None and not ev.committed:
                self._gibbs_parent(ev)

        # ---- Move B (death + birth): cover ALL uncommitted orphans/missing,
        # NOT just recent-N — restricting to recent-N would bias toward UNDER-
        # imputation (an orphan ageing out of scope before a birth succeeds would
        # never be retried). Both run off incremental indices (no window scan):
        #   death  → iterate the (small) uncommitted-missing set
        #   birth  → uniform sample of up to max_birth_attempts orphans
        for eid in list(self._missing_set):
            ev = self.events.get(eid)
            if ev is not None:
                self._try_death(ev)                 # childless-missing only; cheap

        n = len(self._orphan_list)
        if n:
            k = min(self.config.max_birth_attempts_per_sweep, n)
            if k > 0:
                # Sample eids up front (O(k)); the orphan index mutates as births
                # succeed, so we re-check each candidate before using it.
                picks = [self._orphan_list[j] for j in self.rng.sample(range(n), k)]
                births = 0
                for eid in picks:
                    if births >= self.config.max_births_per_sweep:
                        break
                    ev = self.events.get(eid)
                    if ev is None or ev.committed or ev.parent != -1:
                        continue
                    if self._birth(ev):
                        births += 1

    def _birth(self, ev: SamplerEvent) -> bool:
        return self._try_birth(ev)

    # ---- freeze / close / group output -----------------------------------

    def _freeze_aged(self, force: bool = False):
        """Freeze events older than the lag: lock in the MAP (marginal) parent
        and mark them immutable. They stay in the window as candidate parents
        until their whole cascade leaves kernel reach (see _close_clusters).

        Incremental: only the events whose ts entered ``(_frozen_through_ts,
        cutoff]`` since the last call are visited (each event is frozen exactly
        once over its lifetime → O(N) amortised, not O(live) per tick). Missing
        events are always created at ts > cutoff, so they never fall inside an
        already-passed freeze range.
        """
        cutoff = self.now - self.config.freeze_delay_sec()
        if force:
            lo, hi = 0, len(self._order)
        else:
            if cutoff <= self._frozen_through_ts:
                return
            lo = bisect.bisect_right(self._order_ts, self._frozen_through_ts)
            hi = bisect.bisect_right(self._order_ts, cutoff)
        for idx in range(lo, hi):
            eid = self._order[idx]
            ev = self.events.get(eid)
            if ev is None or ev.committed:
                continue
            # MAP parent from accumulated votes, FILTERED to parents that still
            # exist (a voted missing parent may have been culled meanwhile) — so
            # we never lock a dangling pointer. Fall back to the current parent.
            map_parent, prob = ev.parent, 1.0
            if ev.parent_votes:
                valid = [(p, c) for p, c in ev.parent_votes.items()
                         if p == -1 or p in self.events]
                if valid:
                    map_parent, votes = max(valid, key=lambda pc: pc[1])
                    prob = votes / max(1, ev.sweep_count)
            self._set_parent(ev, map_parent if (map_parent == -1 or map_parent in self.events) else -1)
            ev.commit_parent_prob = float(prob)
            ev.committed = True
            # Committed events are no longer birth/death targets.
            self._orphan_remove(ev.eid)
            self._missing_set.discard(ev.eid)
            self.committed_count += 1
        if not force:
            self._frozen_through_ts = cutoff

    def _root_of(self, ev: SamplerEvent) -> int:
        """Immigrant root eid of ev's cascade (follow parent pointers)."""
        cur = ev
        guard = 0
        while cur.parent != -1 and cur.parent in self.events:
            cur = self.events[cur.parent]
            guard += 1
            if guard > self.config.max_depth + len(self.events):
                break
        return cur.eid

    def _close_clusters(self, force: bool = False) -> list[dict]:
        """Emit + remove cascades that have fully left kernel reach.

        A cascade closes when all its members are frozen and its newest member is
        older than the evict boundary (now - lag - history - slack) — past that
        point no future alarm can attach to any member, so the structure is
        final.
        """
        evict_cut = self.now - self.config.window_sec()
        # Cheap guard: if nothing has crossed the evict boundary there is nothing
        # to close, so skip the O(live) cluster rebuild entirely.
        if not force and (not self._order_ts or self._order_ts[0] >= evict_cut):
            return []
        # Group live events by cascade root. Memoize roots with path compression
        # so grouping is O(live), not O(live·depth) — deep chains (the
        # over-merging case) would otherwise make this O(live²) per close.
        root_memo: dict[int, int] = {}

        def _memo_root(start_eid: int) -> int:
            path = []
            seen = set()
            cur = start_eid
            while True:
                r = root_memo.get(cur)
                if r is not None:
                    break
                if cur in seen:
                    r = cur
                    break
                seen.add(cur)
                ev = self.events.get(cur)
                if ev is None or ev.parent == -1 or ev.parent not in self.events:
                    r = cur
                    break
                path.append(cur)
                cur = ev.parent
            for p in path:
                root_memo[p] = r
            root_memo[cur] = r
            return r

        clusters: dict[int, list[SamplerEvent]] = {}
        for eid in self._order:
            ev = self.events.get(eid)
            if ev is not None:
                clusters.setdefault(_memo_root(eid), []).append(ev)
        out: list[dict] = []
        for members in clusters.values():
            all_frozen = all(m.committed for m in members)
            newest = max(m.ts for m in members)
            if not (force or (all_frozen and newest < evict_cut)):
                continue
            group = self._build_group(members)
            if group is not None:          # pure-missing cascades are dropped
                out.append(group)
                self.closed_group_count += 1
            for m in members:
                self._remove_event(m)
        return out

    # ---- group / event serialization (brunch-compatible) -----------------

    def _event_id(self, ev: SamplerEvent) -> str:
        if ev.is_missing():
            return f"missing-{ev.eid}"
        return str(ev.meta.get("event_id") or f"obs-{ev.eid}")

    def _event_summary(self, ev: SamplerEvent, eid_to_id: dict, child_probs: dict) -> dict:
        m = ev.meta or {}
        title = str(m.get("alarm_title", "") or m.get("type_label", "") or "")
        summary = {
            "event_id": self._event_id(ev),
            "ts": float(ev.ts),
            "site_id": str(m.get("site_id", "") or ""),
            "alarm_source": str(m.get("alarm_source", "") or ""),
            "alarm_title": title,
            "alarm_type": str(m.get("alarm_type", "") or ""),
            "is_clear": bool(m.get("is_clear", False)),
            "parent_event_id": eid_to_id.get(ev.parent, ""),
        }
        if ev.is_missing():
            # Confidence of an imputed node = how confidently its children leaned
            # on it (mean child attach prob); 0 if somehow childless.
            cps = child_probs.get(ev.eid, [])
            conf = sum(cps) / len(cps) if cps else 0.0
            summary.update({
                "virtual": True,
                "latent": False,                       # no latent-everywhere mode
                "inferred_virtual": True,
                "confidence": float(conf),
                # readable imputed-type label (falls back to the raw key)
                "virtual_source": self._type_label(ev),
                "parent_virtual": self._is_parent_missing(ev),
            })
        else:
            summary.update({
                "virtual": False,
                "latent": False,
                "inferred_virtual": False,
                "confidence": 1.0,
                "virtual_source": "",
                "parent_virtual": self._is_parent_missing(ev),
            })
        return summary

    def _type_label(self, ev: SamplerEvent) -> str:
        m = ev.meta or {}
        return str(m.get("type_label") or m.get("alarm_type") or ev.type_id)

    def _is_parent_missing(self, ev: SamplerEvent) -> bool:
        par = self.events.get(ev.parent) if ev.parent != -1 else None
        return bool(par is not None and par.is_missing())

    def _build_group(self, members: list[SamplerEvent]) -> Optional[dict]:
        members = sorted(members, key=lambda e: e.ts)
        real = [m for m in members if m.observed]
        if not real:
            return None                    # pure-missing cascade → drop (noise)
        virtual = [m for m in members if m.is_missing()]
        eid_to_id = {m.eid: self._event_id(m) for m in members}
        # child attach-prob aggregation for missing-event confidence
        child_probs: dict[int, list] = {}
        for m in members:
            if m.parent != -1 and m.parent in eid_to_id:
                child_probs.setdefault(m.parent, []).append(m.commit_parent_prob)
        summaries = [self._event_summary(m, eid_to_id, child_probs) for m in members]
        timestamps = [s["ts"] for s in summaries]
        root = next((m for m in members if m.parent == -1), members[0])
        root_summary = next(s for s, m in zip(summaries, members) if m.eid == root.eid)
        merged_rules = [MHP_RULE, MHP_VIRTUAL_RULE] if virtual else []
        # Parent→child edges (brunch/visual schema): event-id keyed so the
        # propagation visualizer can draw links, including through missing nodes.
        edges = []
        for m in members:
            if m.parent != -1 and m.parent in eid_to_id:
                par = self.events[m.parent]
                edges.append({
                    "source_event_id": eid_to_id[m.parent],
                    "target_event_id": eid_to_id[m.eid],
                    "source_type": self._type_label(par),
                    "target_type": self._type_label(m),
                    "score": float(m.commit_parent_prob),
                    "source_virtual": par.is_missing(),
                    "target_virtual": m.is_missing(),
                })
        return {
            "group_id": f"mhp-online-{root.eid:06d}",
            "cascade_id": f"mhp-online-{root.eid:06d}",
            "rule": MHP_RULE,
            "merged_rules": merged_rules,
            "event_count": len(members),
            "real_event_count": len(real),
            "virtual_event_count": len(virtual),
            "start_ts": min(timestamps),
            "end_ts": max(timestamps),
            "duration_sec": max(timestamps) - min(timestamps),
            "root_event": root_summary,
            "root_virtual": bool(root.is_missing()),
            "site_list": sorted({s["site_id"] for s in summaries if s["site_id"]}),
            "alarm_source_list": sorted({s["alarm_source"] for s in summaries if s["alarm_source"]}),
            "alarm_title_counts": dict(Counter(s["alarm_title"] for s in summaries if s["alarm_title"])),
            "alarm_type_counts": dict(Counter(s["alarm_type"] for s in summaries if s["alarm_type"])),
            # `symptoms` is the brunch/visual_output key (NOT `events`).
            "symptoms": summaries,
            "edges": edges,
        }

    # ---- diagnostics -----------------------------------------------------

    def stats(self) -> dict:
        return {
            "live_events": len(self.events),
            "live_missing": self._missing_count,
            "births": self.births,
            "deaths": self.deaths,
            "committed": self.committed_count,
            "closed_groups": self.closed_group_count,
            "now": self.now,
        }


# --------------------------------------------------------------------------
# Feature-mode adapter (device-OPEN, inductive α)
# --------------------------------------------------------------------------


class FeatureKernelAdapter:
    """:class:`ModelAdapter` for an ``edge_mode='feature'`` artifact.

    The model "type" here is **not** a vocab id but an ``(alarm_type, ne)`` pair
    — feature mode is inductive / device-OPEN, so a missing parent is identified
    by its alarm type and the device it would live on. The sampler treats the
    type as an opaque hashable key, so this works unchanged with the core.

    α is the live ``softplus(w·φ)`` from a :class:`RuntimeFeatureScorer`; μ from
    a :class:`RuntimeMuScorer` (or a per-alarm-type table fallback). Candidate
    missing-parent types are enumerated over ``at_vocab × ({target_ne} ∪
    topology-neighbours(target_ne))`` — bounded by the topology reach, which is
    exactly the inductive-richness advantage over device mode, kept finite.

    Cost notes
    ----------
    - ``candidate_sources`` is vectorised over source candidates; in dynamic
      mode it takes a small upper envelope over all source-state bit patterns
      so state-activated missing-parent types are not filtered out too early.
    - ``total_compensator`` needs the transpose (one source vs many targets),
      which the scorer can't vectorise; but the per-source Σα is
      horizon-independent, so we compute it once per source key and cache it.
    """

    def __init__(
        self,
        feature_scorer,
        *,
        mu_scorer=None,
        mu_by_alarm_type: Optional[dict] = None,
        mu_default: float = 0.0,
        time_scale_sec: float = 60.0,
        time_slack_sec: float = 0.0,
        late_penalty_half_life_sec: float = 1.0,
        alpha_floor: float = 0.0,
        candidate_max_hops: Optional[int] = None,
        max_candidates: int = 256,
        cache_max_entries: int = 200_000,
        source_mark_at=None,
    ):
        if time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be > 0")
        if time_slack_sec < 0:
            raise ValueError("time_slack_sec must be >= 0")
        if late_penalty_half_life_sec <= 0:
            raise ValueError("late_penalty_half_life_sec must be > 0")
        self.fs = feature_scorer
        self.mu_scorer = mu_scorer
        self.mu_by_alarm_type = dict(mu_by_alarm_type or {})
        self.mu_default = float(mu_default)
        self.time_scale_sec = float(time_scale_sec)
        self.time_slack_sec = float(time_slack_sec)
        self.late_penalty_half_life_sec = float(late_penalty_half_life_sec)
        self.alpha_floor = float(alpha_floor)
        self.beta = float(getattr(feature_scorer, "beta", 1.0))
        self.max_candidates = int(max_candidates)
        self._source_mark_at = source_mark_at
        # alarm-type vocabulary, id-ordered for deterministic enumeration
        at_to_id = getattr(feature_scorer, "at_to_id", {}) or {}
        self._at_vocab = [a for a, _ in sorted(at_to_id.items(), key=lambda kv: kv[1])]
        self._topo = getattr(feature_scorer, "topology_index", None)
        self._node_infos = getattr(feature_scorer, "node_infos", {}) or {}
        self.n_dynamic = int(getattr(feature_scorer, "n_dynamic", 0) or 0)
        if candidate_max_hops is not None:
            self._max_hops = int(candidate_max_hops)
        else:
            self._max_hops = int(getattr(self._topo, "max_hops", 1) or 1)
        # caches — bounded LRU (OrderedDict) so a long stream can't grow them
        # without limit. Over 442k feature events the (src,tgt) pair space is
        # (ATs×NEs)², which would otherwise balloon to GB and slow the run down
        # as it progresses. _neighbor_cache is naturally bounded by #NEs.
        self._cache_max = max(1, int(cache_max_entries))
        self._pair_alpha: "OrderedDict[tuple, float]" = OrderedDict()
        self._alpha_out_sum: "OrderedDict[tuple, float]" = OrderedDict()
        self._candidate_source_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._neighbor_cache: dict[str, list] = {}
        self._envelope_mark = None        # cached α-maximizing source mark

    # ---- helpers ----
    def _mark_tuple(self, source_mark=None) -> tuple:
        if self.n_dynamic <= 0:
            return ()
        if source_mark is None:
            return tuple(0.0 for _ in range(self.n_dynamic))
        vals = tuple(float(v) for v in source_mark)
        if len(vals) != self.n_dynamic:
            return tuple((vals + tuple(0.0 for _ in range(self.n_dynamic)))[: self.n_dynamic])
        return vals

    def _mark_matrix(self, n: int, src_marks=None):
        if self.n_dynamic <= 0:
            return None
        import numpy as np

        if src_marks is None:
            return np.zeros((n, self.n_dynamic), dtype=np.float64)
        return np.asarray([self._mark_tuple(mark) for mark in src_marks], dtype=np.float64)

    def source_mark_at(self, source_type, ts: float) -> tuple:
        if self.n_dynamic <= 0:
            return ()
        if self._source_mark_at is None:
            return self._mark_tuple()
        return self._mark_tuple(self._source_mark_at(source_type, ts))

    def _envelope_mark_matrix(self, n: int):
        """Upper-envelope source mark for candidate enumeration: the mark that
        MAXIMIZES α over all 2^n_dynamic states. Since α = softplus(z_static +
        Σ_i mark_i·w_dyn_i) is monotonic in z, the max is at mark_i = 1{w_dyn_i>0}
        — so one call instead of 2^n. Returns None when no dynamic features."""
        if self.n_dynamic <= 0:
            return None
        import numpy as np

        if self._envelope_mark is None:
            ev = getattr(self.fs, "envelope_mark", None)
            if callable(ev):
                self._envelope_mark = np.asarray(ev(), dtype=np.float64).reshape(self.n_dynamic)
            else:
                # Scorer without weight introspection → over-inclusive all-ones.
                self._envelope_mark = np.ones(self.n_dynamic, dtype=np.float64)
        return np.tile(self._envelope_mark, (n, 1))

    def _neighbors(self, ne: str) -> list:
        cached = self._neighbor_cache.get(ne)
        if cached is not None:
            return cached
        out: list = []
        if self._topo is not None:
            hops = getattr(self._topo, "undirected_hops", None)
            if hops:
                out = [n for n in hops.get(ne, {}).keys() if n and n != ne]
        self._neighbor_cache[ne] = out
        return out

    def _candidate_nes(self, ne: str) -> list:
        return [ne] + self._neighbors(ne)

    def _alpha(self, src_key, tgt_key, source_mark=None) -> float:
        mark = self._mark_tuple(source_mark)
        ck = (src_key, tgt_key, mark)
        cache = self._pair_alpha
        a = cache.get(ck)
        if a is not None:
            cache.move_to_end(ck)              # LRU touch
            return a
        (s_at, s_ne) = src_key
        (t_at, t_ne) = tgt_key
        arr = self.fs.alpha_for_target(
            t_at, t_ne, [s_at], [s_ne], src_marks=self._mark_matrix(1, [mark])
        )
        val = float(arr[0]) if len(arr) else 0.0
        if val < self.alpha_floor:
            val = 0.0
        cache[ck] = val
        if len(cache) > self._cache_max:
            cache.popitem(last=False)          # evict least-recently-used
        return val

    # ---- ModelAdapter surface ----
    def mu(self, type_key) -> float:
        at, ne = type_key
        if self.mu_scorer is not None:
            return float(self.mu_scorer.mu_for(at, ne))
        return float(self.mu_by_alarm_type.get(at, self.mu_default))

    def kernel_intensity(self, source_type, target_type, dt_sec: float, *, source_mark=None) -> float:
        dt_eff_sec, late_weight = self._signed_dt(dt_sec)
        if late_weight <= 0:
            return 0.0
        a = self._alpha(source_type, target_type, source_mark)
        if a <= 0:
            return 0.0
        dt = dt_eff_sec / self.time_scale_sec
        return float(a * self.beta * math.exp(-self.beta * dt) * late_weight)

    def _alpha_batch_cached(self, target_type, source_types, marks_norm):
        """Floored α for (each source, fixed target, its mark), served from the
        shared _pair_alpha LRU. Only cache MISSES go through alpha_for_target
        (one batched feature build) — the win for Gibbs, which re-scores the same
        (event, candidate-parent, mark) triples every sweep until the event
        freezes. α is deterministic in (target, source, mark) so this is exact."""
        import numpy as np

        n = len(source_types)
        alphas = np.empty(n, dtype=np.float64)
        cache = self._pair_alpha
        miss_i: list = []
        miss_at: list = []
        miss_ne: list = []
        miss_mk: list = []
        for i in range(n):
            s = source_types[i]
            ck = (s, target_type, marks_norm[i])
            v = cache.get(ck)
            if v is None:
                miss_i.append(i)
                miss_at.append(s[0])
                miss_ne.append(s[1])
                miss_mk.append(marks_norm[i])
            else:
                cache.move_to_end(ck)
                alphas[i] = v
        if miss_i:
            (t_at, t_ne) = target_type
            ma = np.asarray(
                self.fs.alpha_for_target(
                    t_at, t_ne, miss_at, miss_ne,
                    src_marks=self._mark_matrix(len(miss_i), miss_mk),
                ),
                dtype=np.float64,
            )
            if self.alpha_floor > 0:
                ma = np.where(ma >= self.alpha_floor, ma, 0.0)
            for j, i in enumerate(miss_i):
                val = float(ma[j])
                alphas[i] = val
                cache[(source_types[i], target_type, marks_norm[i])] = val
            while len(cache) > self._cache_max:
                cache.popitem(last=False)
        return alphas

    def kernel_intensity_batch(self, target_type, source_types: list, dts: list, *, src_marks=None):
        """Vectorised intensities for many candidate sources vs one target — the
        feature-mode hot path. α is served from the shared LRU cache (only misses
        rebuild features), then the kernel/decay factor is applied vectorised."""
        import numpy as np

        n = len(source_types)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        marks_in = src_marks if src_marks is not None else [None] * n
        marks_norm = [self._mark_tuple(m) for m in marks_in]
        alphas = self._alpha_batch_cached(target_type, source_types, marks_norm)
        dt_sec = np.asarray(dts, dtype=np.float64)
        dt_eff_sec, late_weight = self._signed_dt_vec(dt_sec)
        dt = dt_eff_sec / self.time_scale_sec
        return alphas * self.beta * np.exp(-self.beta * dt) * late_weight

    def compensator(self, source_type, target_type, dt_sec: float, *, source_mark=None) -> float:
        a = self._alpha(source_type, target_type, source_mark)
        if a <= 0:
            return 0.0
        pos = 0.0
        if dt_sec > 0:
            dt = dt_sec / self.time_scale_sec
            pos = a * (1.0 - math.exp(-self.beta * dt))
        neg = a * self.beta * self._negative_penalty_integral_model()
        return float(pos + neg)

    def _signed_dt(self, dt_sec: float) -> tuple[float, float]:
        if dt_sec >= 0:
            return float(dt_sec), 1.0
        late = -float(dt_sec)
        if self.time_slack_sec <= 0 or late > self.time_slack_sec:
            return 0.0, 0.0
        lam = math.log(2.0) / self.late_penalty_half_life_sec
        return 0.0, math.exp(-lam * late)

    def _signed_dt_vec(self, dt_sec):
        import numpy as np

        dt_sec = np.asarray(dt_sec, dtype=np.float64)
        dt_eff = np.maximum(dt_sec, 0.0)
        late = np.maximum(-dt_sec, 0.0)
        if self.time_slack_sec <= 0:
            weight = (late <= 0).astype(np.float64)
        else:
            lam = math.log(2.0) / self.late_penalty_half_life_sec
            weight = np.where(late <= self.time_slack_sec, np.exp(-lam * late), 0.0)
        return dt_eff, weight

    def _negative_penalty_integral_model(self) -> float:
        if self.time_slack_sec <= 0:
            return 0.0
        slack = self.time_slack_sec / self.time_scale_sec
        half = self.late_penalty_half_life_sec / self.time_scale_sec
        lam = math.log(2.0) / half
        return float((1.0 - math.exp(-lam * slack)) / lam)

    def candidate_sources(self, target_type) -> list[tuple]:
        import numpy as np

        cached = self._candidate_source_cache.get(target_type)
        if cached is not None:
            self._candidate_source_cache.move_to_end(target_type)
            return list(cached)
        (t_at, t_ne) = target_type
        nes = self._candidate_nes(t_ne)
        src_ats: list = []
        src_nes: list = []
        for ne in nes:
            for at in self._at_vocab:
                src_ats.append(at)
                src_nes.append(ne)
        if not src_ats:
            return []
        # Birth first needs a finite candidate source set before it has sampled
        # t'. In dynamic mode, take a possible-state upper envelope so edges that
        # only activate under a nonzero observed state slice are not filtered out.
        # The envelope is a SINGLE call with the α-maximizing mark (no 2^n loop).
        alphas = self.fs.alpha_for_target(
            t_at, t_ne, src_ats, src_nes,
            src_marks=self._envelope_mark_matrix(len(src_ats)),
        )
        out: list[tuple] = []
        for at, ne, a in zip(src_ats, src_nes, alphas):
            av = float(a)
            if av >= self.alpha_floor and av > 0:
                out.append(((at, ne), av))
        if len(out) > self.max_candidates:
            out.sort(key=lambda ka: -ka[1])
            out = out[: self.max_candidates]
        self._candidate_source_cache[target_type] = tuple(out)
        if len(self._candidate_source_cache) > self._cache_max:
            self._candidate_source_cache.popitem(last=False)
        return out

    def total_compensator(self, source_type, horizon_sec: float, *, source_mark=None) -> float:
        horizon_sec = max(float(horizon_sec), 0.0)
        cache = self._alpha_out_sum
        mark = self._mark_tuple(source_mark)
        cache_key = (source_type, mark)
        s_sum = cache.get(cache_key)
        if s_sum is not None:
            cache.move_to_end(cache_key)          # LRU touch
        else:
            import numpy as np

            (s_at, s_ne) = source_type
            # ONE batched transpose call (one source vs all candidate targets)
            # instead of a per-target scalar loop.
            tgt_ats: list = []
            tgt_nes: list = []
            for ne in self._candidate_nes(s_ne):
                for at in self._at_vocab:
                    tgt_ats.append(at)
                    tgt_nes.append(ne)
            if tgt_ats:
                a = np.asarray(
                    self.fs.alpha_for_source(s_at, s_ne, tgt_ats, tgt_nes, src_mark=mark),
                    dtype=np.float64,
                )
                a = a[a >= self.alpha_floor]
                s_sum = float(a[a > 0].sum())
            else:
                s_sum = 0.0
            cache[cache_key] = s_sum
            if len(cache) > self._cache_max:
                cache.popitem(last=False)          # evict least-recently-used
        dt = horizon_sec / self.time_scale_sec
        pos = s_sum * (1.0 - math.exp(-self.beta * dt)) if horizon_sec > 0 else 0.0
        neg = s_sum * self.beta * self._negative_penalty_integral_model()
        return float(pos + neg)

    def type_meta(self, type_key) -> dict:
        at, ne = type_key
        info = self._node_infos.get(ne)
        site = getattr(info, "site_id", "") if info is not None else ""
        return {
            "alarm_type": at,
            "alarm_source": ne,
            "type_label": f"{ne} | {at}",
            "site_id": site or "",
        }


def feature_adapter_from_artifact(artifact, ne_graph_path, *, alpha_floor=None,
                                  candidate_max_hops=None, time_slack_sec=None,
                                  late_penalty_half_life_sec=None,
                                  source_mark_at=None,
                                  cache_max_entries=200_000) -> FeatureKernelAdapter:
    """Build a :class:`FeatureKernelAdapter` from a feature-mode artifact.

    Mirrors the scorer construction in ``stream_alarm_mhp.main`` so the imputed
    α/μ match streaming inference exactly. ``ne_graph_path`` is the NE graph used
    for device attributes + topology.
    """
    if getattr(artifact.config, "edge_mode", "device") != "feature":
        raise ValueError("feature_adapter_from_artifact requires edge_mode='feature'")
    from mhp.feature_kernel import FeatureKernel
    from alarm_flow_mhp.feature_spec import MuFeatureSpec, RuntimeFeatureScorer, RuntimeMuScorer
    from alarm_flow_mhp.dynamic_state import STATE_DIM
    from alarm_flow_isahp.ne_topology import NETopologyIndex
    from ne_link_learning.core import build_graph_context
    from topology_tools.region_utils import load_ne_graph

    md = artifact.training_metadata or {}
    fk = md.get("feature_kernel")
    rt = md.get("feature_runtime") or {}
    if fk is None:
        raise ValueError("feature-mode artifact missing feature_kernel")
    ne_graph_data = load_ne_graph(ne_graph_path)
    graph_ctx = build_graph_context(ne_graph_data)
    infer_hops = max(int(getattr(artifact.config, "feature_topo_max_hops", 2)), 1)
    topo_idx = NETopologyIndex.from_graph(ne_graph_data, max_hops=infer_hops)
    feature_scorer = RuntimeFeatureScorer(
        kernel=FeatureKernel.from_dict(fk),
        at_vocab=rt.get("at_vocab", []),
        graph_context=graph_ctx,
        topology_index=topo_idx,
        beta=float(rt.get("beta", 1.0)),
        n_dynamic=STATE_DIM if getattr(artifact.config, "dynamic_alpha", "off") != "off" else 0,
    )
    mu_scorer = None
    mu_fk = rt.get("mu_kernel")
    mu_sp = rt.get("mu_spec")
    if mu_fk is not None and mu_sp is not None:
        mu_scorer = RuntimeMuScorer(
            mu_kernel=FeatureKernel.from_dict(mu_fk),
            mu_spec=MuFeatureSpec.from_dict(mu_sp),
            graph_context=graph_ctx,
        )
    floor = alpha_floor
    if floor is None:
        floor = float(getattr(artifact.config, "edge_threshold", 0.0))
    return FeatureKernelAdapter(
        feature_scorer,
        mu_scorer=mu_scorer,
        mu_by_alarm_type=rt.get("mu_by_alarm_type", {}) or {},
        mu_default=float(rt.get("mu_default", 0.0)),
        time_scale_sec=float(getattr(artifact.config, "time_scale_sec", 60.0)),
        time_slack_sec=float(
            getattr(artifact.config, "time_slack_sec", 0.0)
            if time_slack_sec is None else time_slack_sec
        ),
        late_penalty_half_life_sec=float(
            getattr(artifact.config, "late_penalty_half_life_sec", 1.0)
            if late_penalty_half_life_sec is None else late_penalty_half_life_sec
        ),
        alpha_floor=float(floor),
        candidate_max_hops=candidate_max_hops,
        source_mark_at=source_mark_at,
        cache_max_entries=int(cache_max_entries),
    )


# --------------------------------------------------------------------------
# Building an adapter from a trained artifact (sketch)
# --------------------------------------------------------------------------


def device_adapter_from_artifact(
    artifact, *, time_slack_sec=None, late_penalty_half_life_sec=None
) -> ExpKernelAdapter:
    """Build an ExpKernelAdapter from a device-mode exp-kernel artifact.

    NOTE: only valid for ``edge_mode='device'`` + ``kernel_type='exp'``. Feature
    mode and piecewise kernels need their own adapter (live α via the feature
    scorer / per-bucket θ via the kernel-eval). This helper is a starting point;
    it mirrors the edge-table materialisation in ``stream_alarm_mhp``.
    """
    params = artifact.params
    if getattr(params, "kernel_type", "exp") != "exp":
        raise NotImplementedError("device_adapter_from_artifact: exp kernel only (v1)")
    import numpy as np

    et = np.asarray(params.edge_targets).astype(int)
    es = np.asarray(params.edge_sources).astype(int)
    ea = np.asarray(params.edge_alpha).astype(float)
    eb = np.asarray(params.edge_beta).astype(float)
    edges = {(int(t), int(s)): (float(a), float(b))
             for t, s, a, b in zip(et, es, ea, eb)}
    mu_by_type = {i: float(m) for i, m in enumerate(np.asarray(params.mu).astype(float))}
    # ExpKernelAdapter builds the outgoing-target index (for the compensator)
    # itself in __post_init__, so no extra wiring is needed here.
    return ExpKernelAdapter(
        mu_by_type=mu_by_type,
        edges=edges,
        time_scale_sec=float(getattr(artifact.config, "time_scale_sec", 60.0)),
        time_slack_sec=float(
            getattr(artifact.config, "time_slack_sec", 0.0)
            if time_slack_sec is None else time_slack_sec
        ),
        late_penalty_half_life_sec=float(
            getattr(artifact.config, "late_penalty_half_life_sec", 1.0)
            if late_penalty_half_life_sec is None else late_penalty_half_life_sec
        ),
    )
