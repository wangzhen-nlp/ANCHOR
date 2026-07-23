#!/usr/bin/env python3
"""One-hop virtual source-period imputation for the AlarmPeriod engine.

This is the "native approach B (one hop)" companion to
``stream_alarm_period_mhp.py``.  Where ``stream_alarm_mhp.py`` imputes missing
*occurrences* through ``MissingChainSampler`` (occurrence granularity), the
AlarmPeriod engine's grouping unit is a whole *period*, so the natural analogue
of "impute an event" is: when a harvested period cannot find any credible
*observed* source (it anchors its own solo group), hypothesise a single virtual
**source period** of a known period-type that explains it.

Design (kept deliberately minimal for v1):

* **One hop only.**  The virtual source is a terminal background immigrant — it
  is explained by the target-type immigrant rate ``μ`` and is NOT itself given a
  parent.  Multi-hop (a virtual source that is re-imputed a parent of its own)
  is a future extension: raise ``max_depth`` and mark the virtual period as a
  fresh birth target.  Nothing here writes the virtual period into the matcher's
  candidate buckets, so it stays inert to normal period matching and cannot
  accrete beyond the group it was born to explain.
* **κ-penalised acceptance.**  A birth is accepted only when the time-decayed
  edge score, penalised by a per-virtual-period log-prior ``κ`` (≤ 0), still
  beats the background immigrant explanation ``μ`` of the target.  Because the
  observed-relation bar is already ``score ≥ μ`` (``edge.threshold``), κ raises
  the bar for a *virtual* source above that of a real one — exactly the intent.
* **Modular.**  All *policy* (orphan detection, candidate scoring, acceptance,
  the one-hop rule) lives here.  The engine only exposes a one-line post-harvest
  hook plus a few thin plumbing helpers; the classes it constructs are imported
  lazily to avoid an import cycle with the engine module.

The acceptance mirrors the occurrence sampler's birth ratio
``intensity / μ · exp(κ)`` so the two engines share one knob vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

EPS = 1e-12


@dataclass
class PeriodImputeConfig:
    """Knobs for one-hop virtual source-period imputation.

    ``kappa`` is a non-positive log-prior penalty per imputed virtual period
    (more negative ⇒ fewer / stricter births), aligned with
    ``--impute-kappa`` in the
    occurrence engine.  ``lag_sec`` places the virtual source that many seconds
    before the target's first occurrence; ``0`` (the default) means the exp
    kernel's MAP placement, dt -> 0 (as close to the target as possible, the
    highest-scoring position). It does not fall back to ``time_slack_sec``.
    """

    enabled: bool = False
    kappa: float = -2.0
    max_candidates: int = 16
    lag_sec: float = 0.0
    min_score_ratio: float = 1.0

    def validate(self):
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if not math.isfinite(self.kappa) or self.kappa > 0:
            raise ValueError("kappa must be finite and <= 0")
        if not math.isfinite(self.lag_sec) or self.lag_sec < 0:
            raise ValueError("lag_sec must be finite and >= 0")
        if not math.isfinite(self.min_score_ratio) or self.min_score_ratio <= 0:
            raise ValueError("min_score_ratio must be finite and > 0")

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "kappa": float(self.kappa),
            "max_candidates": int(self.max_candidates),
            "lag_sec": float(self.lag_sec),
            "min_score_ratio": float(self.min_score_ratio),
        }


class PeriodSourceImputer:
    """Post-harvest one-hop birth of virtual source periods for orphans.

    Holds only decision logic; the engine (``AlarmPeriodMHPAssigner``) supplies
    state access and a couple of construction helpers.  Instantiate with the
    engine and a :class:`PeriodImputeConfig`; call :meth:`maybe_impute` once per
    period right after ``_apply_relations`` has run in ``_harvest_period``.
    """

    def __init__(self, engine, config: PeriodImputeConfig):
        config.validate()
        self.engine = engine
        self.config = config
        # Counters surfaced through the engine's stats().
        self.births = 0
        self.rejected = 0
        self.orphans_seen = 0
        # Diagnostics for tuning kappa when births stay at 0. ``candidates_seen``
        # counts window-eligible incoming edges scored across all orphans;
        # ``max_score_ratio`` is the largest decayed score / mu (== edge.threshold)
        # observed. A birth needs score/mu > exp(-kappa), so max_score_ratio tells
        # you the loosest kappa that could ever fire: kappa > -ln(max_score_ratio).
        # orphans_with_candidates separates "no candidate edge at all" (a coverage
        # problem) from "candidates exist but all too weak" (a kappa problem).
        self.candidates_seen = 0
        self.orphans_with_candidates = 0
        self.max_score_ratio = 0.0
        # Actual dt (placement/window offset) used; set on first _best_candidate.
        # Proves whether the dt=0 fix is the running code path.
        self.source_offset_sec = None
        # Coverage breakdown for orphans that scored no candidate (see
        # _diagnose_no_candidate). Exactly one is incremented per such orphan.
        self.orphan_type_absent = 0
        self.orphan_no_incoming = 0
        self.orphan_state_gap = 0
        self.orphan_selfloop_only = 0
        self.orphan_crosstype_bug = 0
        # Orthogonal to the above: 0-candidate orphans that DO have a cross-type
        # outgoing edge (orphan as source). A high count means the relationships
        # exist but in the reverse direction imputation never reads.
        self.orphan_has_crosstype_outgoing = 0

    # ---- orchestration ---------------------------------------------------

    def maybe_impute(self, period) -> None:
        cfg = self.config
        if not cfg.enabled:
            return
        engine = self.engine
        # One hop: never re-impute a period that is itself virtual.
        if getattr(period, "is_virtual", False):
            return
        group = self._orphan_group(period)
        if group is None:
            return
        self.orphans_seen += 1

        best = self._best_candidate(period)
        if best is None:
            self.rejected += 1
            return
        score, src_sig, edge = best
        self._birth(period, group, src_sig, edge, score)

    # ---- orphan gate -----------------------------------------------------

    def _orphan_group(self, period):
        """Return the period's group iff it is a fresh solo group.

        A period that failed to attach any observed source anchors a group whose
        only member is itself.  That is the one-hop birth target.  Once a virtual
        source is attached the group is no longer solo, so repeated harvests of a
        still-orphan period are naturally idempotent.
        """
        engine = self.engine
        gid = engine._resolve_group_id(period.primary_group_id)
        if gid is None:
            return None
        group = engine.groups.get(gid)
        if group is None:
            return None
        if len(group.period_ids) != 1 or period.period_id not in group.period_ids:
            return None
        return group

    # ---- candidate scoring + acceptance ----------------------------------

    def _best_candidate(self, period):
        """Pick the κ-accepted, highest-scoring virtual source, or ``None``.

        Iterates the compiled association plan's incoming edges for the target
        signature, scores each at the MAP placement, and keeps the best whose
        κ-penalised score clears the target's immigrant threshold μ
        (``edge.threshold``).
        """
        engine = self.engine
        cfg = self.config
        sig = period.signature
        offset = self._source_offset_sec()
        dt = float(offset)
        # Surface the actual placement/window offset so a run can prove whether
        # dt==0 (MAP placement) is in effect. If this reports ~time_slack_sec, the
        # dt=0 fix is not the code path running (stale .pyc / partial sync).
        self.source_offset_sec = dt
        accept_factor = math.exp(cfg.kappa)

        top_edges = getattr(engine.plan, "top_edges_by_target", None)
        if top_edges is None:
            # Lightweight compatibility path for policy-only test doubles.
            incoming = engine.plan.iter_edges_by_target(sig)
        else:
            incoming = top_edges(sig, cfg.max_candidates, dt)

        best = None
        examined = 0
        scored_any = False
        for source_key, edge in incoming:
            src_sig = self._source_signature(source_key, sig)
            if src_sig is None:
                continue
            # Skip a same-type self-source: a period explaining "itself" is a
            # repeat, not an imputed upstream cause.
            if src_sig.period_type == sig.period_type:
                continue
            if dt > edge.past_window_sec + EPS:
                continue
            if top_edges is None and examined >= cfg.max_candidates:
                break
            examined += 1
            score = engine._past_score(edge, dt)
            # Tuning telemetry: record the raw score/mu ratio of every scored
            # candidate before any acceptance guard, so a run whose births stay
            # at 0 still reveals how far kappa is from firing.
            self.candidates_seen += 1
            scored_any = True
            ratio = score / max(edge.threshold, EPS)
            if ratio > self.max_score_ratio:
                self.max_score_ratio = ratio
            # Apply the optional raw-score guard before the independent
            # κ-penalised birth-vs-background test.  Keeping these separate
            # matches the CLI contract and avoids multiplying the ratio by
            # exp(-κ).
            if score + EPS < edge.threshold * cfg.min_score_ratio:
                continue
            if score * accept_factor <= edge.threshold:
                continue
            if best is None or score > best[0]:
                best = (score, src_sig, edge)
        if scored_any:
            self.orphans_with_candidates += 1
        else:
            self._diagnose_no_candidate(sig)
        return best

    def _diagnose_no_candidate(self, sig) -> None:
        """Classify why an orphan scored no candidate (one-run coverage probe).

        The candidate path excludes same-period-type (self-loop) sources, so raw
        incoming-edge presence is not enough — this separates the *imputable*
        cross-type edges from self-loops. Exactly one counter is bumped:

        * ``type_absent``     — target period-type is in no loaded cache.
        * ``no_incoming``     — type present but zero incoming edges in any state
                                (a root/immigrant-only type; nothing to impute).
        * ``state_gap``       — the type has incoming edges under some frozen
                                state, but none under this orphan's exact state.
        * ``selfloop_only``   — the exact signature has incoming edges but every
                                one is a self-loop (same entity+alarm type), which
                                imputation excludes by design (a repeat, not an
                                upstream cause). Expected, not a bug.
        * ``crosstype_bug``   — the exact signature has a *cross-type* incoming
                                edge that should have scored yet didn't: a genuine
                                lookup/filter bug.
        """
        from alarm_flow_mhp.stream_alarm_period_mhp import PeriodSignature

        plan = self.engine.plan
        indexes = list(getattr(plan, "precompiled_indexes", None) or [])
        dynamic = getattr(plan, "edges_by_target", {}) or {}

        type_present = False
        exact_incoming = 0
        exact_crosstype = 0
        type_incoming = 0
        for index in indexes:
            type_id = index.type_to_id.get(sig.period_type)
            if type_id is None:
                continue
            type_present = True
            exact_incoming += index.target_edge_count(sig)
            exact_crosstype += self._index_crosstype_incoming(index, sig, type_id)
            for state in range(8):
                type_incoming += index.target_edge_count(
                    PeriodSignature(sig.period_type, state)
                )
        for target_sig, row in dynamic.items():
            if target_sig.period_type != sig.period_type:
                continue
            type_present = True
            count = len(row)
            type_incoming += count
            if target_sig == sig:
                exact_incoming += count
                exact_crosstype += sum(
                    1
                    for source_key in row
                    if getattr(source_key, "period_type", source_key)
                    != sig.period_type
                )

        if not type_present:
            self.orphan_type_absent += 1
        elif exact_crosstype > 0:
            self.orphan_crosstype_bug += 1
        elif exact_incoming > 0:
            self.orphan_selfloop_only += 1
        elif type_incoming > 0:
            self.orphan_state_gap += 1
        else:
            self.orphan_no_incoming += 1

        # Orthogonal probe (independent of the incoming classification above):
        # does this orphan participate in a cross-type relationship as a SOURCE,
        # i.e. an outgoing edge orphan->target? Imputation only reads incoming
        # edges, so if training/time-slack stored the relation in the reverse
        # direction, it lives here and imputation never sees it.
        outgoing_crosstype = 0
        for index in indexes:
            source_type_id = index.type_to_id.get(sig.period_type)
            if source_type_id is None:
                continue
            outgoing_crosstype += self._index_crosstype_outgoing(
                index, sig, source_type_id
            )
        outgoing = getattr(plan, "edges_by_source", {}) or {}
        for source_sig, row in outgoing.items():
            if getattr(source_sig, "period_type", source_sig) != sig.period_type:
                continue
            outgoing_crosstype += sum(
                1
                for target_key in row
                if getattr(target_key, "period_type", target_key)
                != sig.period_type
            )
        if outgoing_crosstype > 0:
            self.orphan_has_crosstype_outgoing += 1

    @staticmethod
    def _index_crosstype_incoming(index, sig, target_type_id) -> int:
        """Count incoming edges for ``sig`` whose source period-type differs."""
        from alarm_flow_mhp.stream_alarm_period_mhp import (
            CACHE_STATE_LAYOUT_TARGET_ONLY,
        )

        signature_id = index._target_signature_id(sig)
        if signature_id is None:
            return 0
        start = int(index.target_offsets[signature_id])
        end = int(index.target_offsets[signature_id + 1])
        if start >= end:
            return 0
        source_ids = index.source_signature_ids[start:end]
        source_type_ids = (
            source_ids
            if index.state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY
            else source_ids // 8
        )
        return int(np.count_nonzero(source_type_ids != int(target_type_id)))

    @staticmethod
    def _index_crosstype_outgoing(index, sig, source_type_id) -> int:
        """Count outgoing edges (``sig`` as source) whose target type differs.

        Targets in the cache always carry full state (type_id*8 + state), so the
        target type id is the row value // 8 regardless of state layout.
        """
        source_key_id = index._source_key_id(sig)
        if source_key_id is None:
            return 0
        start = int(index.source_offsets[source_key_id])
        end = int(index.source_offsets[source_key_id + 1])
        if start >= end:
            return 0
        order = index.source_order[start:end]
        target_type_ids = index.target_signature_ids[order] // 8
        return int(np.count_nonzero(target_type_ids != int(source_type_id)))

    def _source_offset_sec(self) -> float:
        # MAP placement for the backward exp-excitation kernel is dt -> 0: the
        # closer the hypothesised source sits to the target, the higher its
        # score. So the default offset is 0. An explicit --impute-lag-sec pushes
        # the source further back if desired.
        #
        # This deliberately does NOT fall back to the engine's time_slack_sec.
        # time_slack_sec is a *future*-window slack (a source may appear that
        # many seconds *after* the target and still match); using it as a
        # *backward* offset both needlessly decays the score by
        # exp(-beta*slack/scale) and, because the same offset is passed as the
        # top_edges min_past_window floor, filters out every candidate edge whose
        # reachable past window is shorter than the slack. With a large slack
        # (e.g. 300s) that silently zeroes out all imputation candidates.
        return float(self.config.lag_sec)

    @staticmethod
    def _source_signature(source_key, target_signature):
        """Coerce a plan candidate key into a concrete source PeriodSignature.

        The compact index yields a bare ``PeriodType`` under the target-only
        state layout; the full layout already yields a ``PeriodSignature``.  A
        virtual source has no observed frozen state, so it defaults to the
        all-clear combo (0), the neutral choice.

        Derive the classes from the live target instead of importing them back
        from ``stream_alarm_period_mhp``.  When that file is executed by path,
        its classes belong to ``__main__``; importing the canonical module again
        would create lookalike classes for which ``isinstance`` is false.
        """
        signature_type = type(target_signature)
        period_type = type(target_signature.period_type)
        if isinstance(source_key, signature_type):
            return source_key
        if isinstance(source_key, period_type):
            return signature_type(source_key, 0)

        # Be tolerant of a key produced by another import identity.  This can
        # happen in embedding/runpy environments even when the CLI alias above
        # is not installed.
        source_period_type = getattr(source_key, "period_type", None)
        initial_state = getattr(source_key, "initial_state", 0)
        if source_period_type is None:
            source_period_type = source_key
        entity = getattr(source_period_type, "entity", None)
        alarm_type = getattr(source_period_type, "alarm_type", None)
        if entity is None or alarm_type is None:
            return None
        normalized_type = period_type(str(entity), str(alarm_type))
        return signature_type(normalized_type, int(initial_state))

    # ---- birth -----------------------------------------------------------

    def _birth(self, period, group, src_sig, edge, score) -> None:
        from alarm_flow_mhp.stream_alarm_period_mhp import RelationEvidence

        engine = self.engine
        offset = self._source_offset_sec()
        ts = float(period.first_ts) - float(offset)

        virtual = engine.create_virtual_source_period(src_sig, ts)
        # Attach as a core member so the imputed cause participates in output and
        # core-gating just like an observed anchor's peer.
        engine._attach_period(group, virtual, core=True)
        evidence = RelationEvidence(
            target_period_id=period.period_id,
            source_period_id=virtual.period_id,
            target_event=period.events[0],
            source_event=virtual.events[0],
            score=float(score),
            strength=float(score / max(edge.threshold, EPS)),
            edge=edge,
        )
        engine._record_group_evidence(group, evidence)
        self.births += 1

    # ---- reporting -------------------------------------------------------

    def stats(self) -> dict:
        # Loosest kappa that could produce any birth given what was observed:
        # a candidate fires when score/mu > exp(-kappa), so any birth needs
        # kappa > -ln(max_score_ratio). None when no candidate was ever scored.
        kappa_for_any_birth = (
            -math.log(self.max_score_ratio)
            if self.max_score_ratio > 0.0
            else None
        )
        return {
            "impute_births": self.births,
            "impute_rejected": self.rejected,
            "impute_orphans_seen": self.orphans_seen,
            "impute_orphans_with_candidates": self.orphans_with_candidates,
            "impute_candidates_seen": self.candidates_seen,
            "impute_max_score_ratio": float(self.max_score_ratio),
            "impute_kappa_for_any_birth": kappa_for_any_birth,
            "impute_orphan_type_absent": self.orphan_type_absent,
            "impute_orphan_no_incoming": self.orphan_no_incoming,
            "impute_orphan_state_gap": self.orphan_state_gap,
            "impute_orphan_selfloop_only": self.orphan_selfloop_only,
            "impute_orphan_crosstype_bug": self.orphan_crosstype_bug,
            "impute_orphan_has_crosstype_outgoing": self.orphan_has_crosstype_outgoing,
            "impute_source_offset_sec": self.source_offset_sec,
        }
