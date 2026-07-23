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

EPS = 1e-12


@dataclass
class PeriodImputeConfig:
    """Knobs for one-hop virtual source-period imputation.

    ``kappa`` is a non-positive log-prior penalty per imputed virtual period
    (more negative ⇒ fewer / stricter births), aligned with
    ``--impute-kappa`` in the
    occurrence engine.  ``lag_sec`` places the virtual source that many seconds
    before the target's first occurrence; ``0`` falls back to the engine's
    ``time_slack_sec`` (the exp kernel's MAP placement is "as close as allowed").
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
            src_sig = self._source_signature(source_key)
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
        return best

    def _source_offset_sec(self) -> float:
        cfg = self.config
        if cfg.lag_sec > 0:
            return float(cfg.lag_sec)
        return float(self.engine.config.time_slack_sec)

    def _source_signature(self, source_key):
        """Coerce a plan candidate key into a concrete source PeriodSignature.

        The compact index yields a bare ``PeriodType`` under the target-only
        state layout; the full layout already yields a ``PeriodSignature``.  A
        virtual source has no observed frozen state, so it defaults to the
        all-clear combo (0), the neutral choice.
        """
        from alarm_flow_mhp.stream_alarm_period_mhp import PeriodSignature, PeriodType

        if isinstance(source_key, PeriodSignature):
            return source_key
        if isinstance(source_key, PeriodType):
            return PeriodSignature(source_key, 0)
        return None

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
        }
