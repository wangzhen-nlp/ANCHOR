"""Indexed fast streaming engine for FEATURE-mode MHP inference.

Same grouping semantics as :class:`StreamMHPAssigner` (it subclasses it and
reuses ingest / cascade / merge / close / snapshot / output verbatim), but the
per-event parent search is restructured from "scan the nearest
max_history_events buffer entries and build a (C, F) φ matrix" into indexed
lookups, borrowing the aggregation architecture of
fault_grouping/match_rules.py:

  1. GLOBAL KEY TABLES — live events are bucketed by
     ``key = (alarm_type_id, feature_entity, src_mark_combo)``. α depends only
     on the key (target fixed), and the exp kernel is monotone in Δt, so per
     key only the boundary events matter: the ts==t run start, the latest
     strictly-past event, and (with time slack) the earliest future event.
     A storm of same-key repeats collapses to O(1) probes per key.
  2. INVERTED INDEXES — entity → key buckets, topo node → live entities,
     site → live nodes, alarm-type → live entities: the candidate gathering
     analog of match_rules' per-node event_cache + trigger index.
  3. ADMISSIBILITY TIERS — softplus is monotone, so a per-(target_at,
     source_at) logit UPPER BOUND decides, per target event, whether a source
     alarm type can beat the α floor at all ("never"), only when topologically
     / site related ("related-only"), or anywhere ("global"). Never-pairs are
     skipped outright; related-only pairs are gathered exclusively from the
     precomputed topology neighbors + live same-site nodes.
  4. DECOMPOSED α — w·φ is computed by table lookups + a few scalar terms
     (see feature_spec.DecomposedFeatureScorer); no (C, F) matrix, no Python
     per-candidate attribute loop (attributes are cached per entity).

Equivalence to the legacy scan:
  - candidate pruned by a tier ⇒ its α < floor ⇒ legacy zeroed its score, and
    a zero score never beats μ = softplus(·)·bias > 0;
  - within a key the boundary events dominate all others (kernel monotone);
  - ties are broken to match the legacy nearest-first argmax enumeration
    (see _better_candidate).
  Two DOCUMENTED deviations: (a) the degenerate μ == 0 fallback-table case,
  where legacy binds to a zero-score parent and this engine declares an
  immigrant; (b) the legacy max_history_events truncation, which caps
  candidates by recency regardless of relevance — this engine scores all
  relevant in-window events by default (usually strictly better in storms).
  Pass candidate_cap > 0 to approximate the legacy cap via a |Δt| radius.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from alarm_flow_mhp.dynamic_state import mark_to_combo

from alarm_flow_mhp.feature_spec import (
    DecomposedFeatureScorer,
    _topo_score,
    domain_of,
    topo_node_of,
)
from alarm_flow_mhp.stream_alarm_mhp import (
    OnlineEvent,
    StreamConfig,
    StreamMHPAssigner,
)
from alarm_flow_mhp.topology_relation_prior import topology_relation_weights
from mhp.feature_kernel import softplus


class _KeyBucket:
    """Append-only (ts-ordered) events of one (at, entity, mark) key, with a
    parallel ts list for bisect and front pruning."""

    __slots__ = ("ts", "events")

    def __init__(self):
        self.ts: list[float] = []
        self.events: list[OnlineEvent] = []

    def append(self, event: OnlineEvent):
        self.ts.append(event.ts)
        self.events.append(event)

    def prune_before(self, cutoff: float):
        """Drop events with ts < cutoff. Safe once no future target can reach
        them (targets finalize in ts order, so cutoff = t - window only grows)."""
        k = bisect.bisect_left(self.ts, cutoff)
        if k:
            del self.ts[:k]
            del self.events[:k]


class FastStreamMHPAssigner(StreamMHPAssigner):
    """Feature-mode streaming assigner with indexed candidate gathering."""

    def __init__(
        self,
        artifact,
        config: StreamConfig,
        feature_scorer=None,
        mu_scorer=None,
        candidate_cap: int = 0,
        mu_cache_max: int = 200_000,
    ):
        super().__init__(artifact, config, feature_scorer=feature_scorer, mu_scorer=mu_scorer)
        if not self.feature_mode:
            raise ValueError(
                "FastStreamMHPAssigner requires a feature-mode artifact; "
                "use StreamMHPAssigner for device mode"
            )
        self.decomposed = DecomposedFeatureScorer(self.feature_scorer)
        self._at_to_id = self.feature_scorer.at_to_id
        self._n_at = self.feature_scorer.layout.n_at
        self._layout_dom_to_id = self.feature_scorer.layout._dom_to_id
        self._n_dom = self.feature_scorer.layout.n_dom
        self._scale = float(config.time_scale_sec)
        self._window = float(config.history_window_sec)
        self._slack = float(config.time_slack_sec)
        self._lam = np.log(2.0) / (config.late_penalty_half_life_sec / config.time_scale_sec)
        self._candidate_cap = int(candidate_cap or 0)
        # --- live indexes (all keyed by live activity, never by graph size) ---
        self._buckets: dict[tuple, _KeyBucket] = {}
        self._by_entity: dict[str, dict] = {}      # entity -> {(at_id, mark): bucket}
        self._node_entities: dict[str, set] = {}   # topo node -> live entities
        self._site_nodes: dict[str, set] = {}      # site -> live topo nodes
        # at_id -> live bucket count. Drives the global-bound live mask; a
        # counter (vs an entity set) makes the sweep cleanup trivially exact.
        self._at_live_counts: dict[int, int] = {}
        self._entity_attrs: dict[str, tuple] = {}  # entity -> (node, site, vendor, netype, dom_id)
        # Signature buckets for the GLOBAL (unrelated) tier: an unrelated
        # candidate's α does not depend on entity identity (topo = same_ne =
        # same_site = 0), only on (at, vendor, netype, dom, has_site, mark) —
        # so all unrelated entities collapse into per-signature buckets and a
        # global sweep probes O(live signatures), not O(live keys). Related
        # entities inside a signature bucket are skipped at probe time (their
        # α differs and stage 1 already scored them exactly).
        self._sig_buckets: dict[tuple, _KeyBucket] = {}
        self._children: dict[int, list] = {}       # parent index -> child indexes (slack descendants)
        self._mu_cache: dict[tuple, float] = {}
        self._mu_cache_max = int(mu_cache_max)
        self._rel_base, self._glob_base = self._build_tier_bases()
        # Per-event memoization: tier vectors and the global bound depend only
        # on (target at, target mark combo) — ≤ (n_at+1)·8 distinct inputs —
        # plus, for the bound, the set of live source ats (version-bumped by
        # _bump_live_at, which clears the bound memo).
        self._tier_memo: dict[tuple, tuple] = {}
        self._bound_memo: dict[tuple, float] = {}
        self._tier_all_ok = (
            np.ones(self._n_at + 1, dtype=bool),
            np.ones(self._n_at + 1, dtype=bool),
        )
        # index sweep runs on its own cadence (window-scaled), decoupled from
        # the cascade-close throttle which can be configured much tighter.
        self._sweep_interval = max(self._window * 0.5, self._close_scan_interval)
        self._last_sweep_ts = -np.inf
        # watermark defer plumbing: index the freshly ingested event BEFORE the
        # watermark advance it triggers, matching the legacy buffer-append order.
        self._defer_watermark = False
        self._deferred_ts = None
        # diagnostics
        self.scored_targets = 0
        self.gathered_keys_total = 0
        self.fast_immigrant_shortcuts = 0
        self.global_sweeps = 0

    # ------------------------------------------------------------------
    # Admissibility: per-(target_at+1, source_at+1) logit upper bounds.
    # ------------------------------------------------------------------

    def _build_tier_bases(self):
        """(related_base, global_base) matrices of sound logit upper bounds.

        global  = best achievable with NO topology/same-ne/same-site relation
                  (vendor/netype/domain equality can hold for unrelated devices);
        related = global + the best relation terms.
        Each term is over-approximated independently (relu), so the bound is
        sound; the per-target dynamic term is added at query time (exact).
        """
        d = self.decomposed
        n_at = self._n_at

        def relu(x):
            return max(0.0, float(x))

        dom_ub = 0.0
        if d.n_dom:
            # active domain pair (a, b) contributes W_dom[a,b] (+ w_same_dom on
            # the diagonal — already folded into W_dom_pad); OOV contributes 0.
            dom_ub = max(0.0, float(np.max(d.W_dom_pad)))
        dyn_src_ub = max(0.0, float(np.max(d.src_mark_table))) if d.n_dynamic else 0.0
        common = d.w_bias + relu(d.w_same_vendor) + relu(d.w_same_netype) + dom_ub + dyn_src_ub

        glob = common + d.W_at_pad.copy()
        diag_bonus = relu(d.w_same_at)
        for a in range(1, n_at + 1):
            glob[a, a] += diag_bonus

        topo_ub_same_at = relu(d.w_topo + relu(d.w_topo_x_same_at) + relu(d.w_topo_x_same_site))
        topo_ub_diff_at = relu(d.w_topo + relu(d.w_topo_x_same_site))
        rel = glob + relu(d.w_same_ne) + relu(d.w_same_site)
        rel += topo_ub_diff_at
        for a in range(1, n_at + 1):
            rel[a, a] += topo_ub_same_at - topo_ub_diff_at
        return rel, glob

    def _tier_vectors(self, tgt_at_id: int, mark_combo: int, tgt_term: float):
        """Boolean (n_at+1,) vectors over source-at ids (+1 shifted): can this
        pair's α clear the floor when related / when unrelated? Memoized on
        (target at, target mark combo) — the only inputs."""
        floor = self._feat_alpha_floor
        if floor <= 0:
            return self._tier_all_ok
        key = (tgt_at_id, mark_combo)
        hit = self._tier_memo.get(key)
        if hit is not None:
            return hit
        scale = self.decomposed.alpha_scale
        rel = softplus(self._rel_base[tgt_at_id + 1] + tgt_term) * scale >= floor
        glob = softplus(self._glob_base[tgt_at_id + 1] + tgt_term) * scale >= floor
        out = (rel, glob)
        self._tier_memo[key] = out
        return out

    def _bump_live_at(self, at_id: int, delta: int):
        """Adjust the live-bucket count of a source at; a 0↔live transition
        changes the global-bound live mask, so the bound memo is invalidated."""
        counts = self._at_live_counts
        new = counts.get(at_id, 0) + delta
        if new > 0:
            if at_id not in counts:
                self._bound_memo.clear()
            counts[at_id] = new
        else:
            if counts.pop(at_id, None) is not None:
                self._bound_memo.clear()

    # ------------------------------------------------------------------
    # Ingest hooks: index each modeled event before its watermark advance.
    # ------------------------------------------------------------------

    def _advance_watermark(self, now_ts: float):
        if self._defer_watermark:
            self._deferred_ts = now_ts
            return
        super()._advance_watermark(now_ts)

    def process(self, alarm_event: dict):
        self._defer_watermark = True
        self._deferred_ts = None
        try:
            event = super().process(alarm_event)
        finally:
            self._defer_watermark = False
        if event is not None:
            self._index_event(event)
        if self._deferred_ts is not None:
            ts = self._deferred_ts
            self._deferred_ts = None
            super()._advance_watermark(ts)
        return event

    def _resolve_entity_attrs(self, entity: str) -> tuple:
        node = topo_node_of(entity)
        info = self.feature_scorer.node_infos.get(node)
        site = (info.site_id or "") if info is not None else ""
        vendor = (info.manufacturer or "") if info is not None else ""
        netype = (info.ne_type or "") if info is not None else ""
        dom_id = -1
        if self._n_dom:
            dom_id = self._layout_dom_to_id.get(
                domain_of(entity, self.feature_scorer.node_infos), -1
            )
        return (node, site, vendor, netype, dom_id)

    def _index_event(self, event: OnlineEvent):
        at_id = self._at_to_id.get(str(event.alarm_type), -1)
        mark_idx = mark_to_combo(event.src_mark)
        ent = event.ne
        attrs = self._entity_attrs.get(ent)
        if attrs is None:
            attrs = self._resolve_entity_attrs(ent)
            self._entity_attrs[ent] = attrs
        key = (at_id, ent, mark_idx)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _KeyBucket()
            self._buckets[key] = bucket
            ent_map = self._by_entity.get(ent)
            if ent_map is None:
                ent_map = {}
                self._by_entity[ent] = ent_map
                node, site = attrs[0], attrs[1]
                self._node_entities.setdefault(node, set()).add(ent)
                if site:
                    self._site_nodes.setdefault(site, set()).add(node)
            ent_map[(at_id, mark_idx)] = bucket
            self._bump_live_at(at_id, 1)
        bucket.append(event)
        _node, site, vendor, netype, dom_id = attrs
        sig = (at_id, vendor, netype, dom_id, bool(site), mark_idx)
        sig_bucket = self._sig_buckets.get(sig)
        if sig_bucket is None:
            sig_bucket = _KeyBucket()
            self._sig_buckets[sig] = sig_bucket
        sig_bucket.append(event)

    # ------------------------------------------------------------------
    # μ cache
    # ------------------------------------------------------------------

    def _immigrant_mu(self, event: OnlineEvent) -> float:
        if self.mu_scorer is None:
            return super()._immigrant_mu(event)
        key = (event.alarm_type, event.ne)
        hit = self._mu_cache.get(key)
        if hit is None:
            hit = float(self.mu_scorer.mu_for(event.alarm_type, event.ne))
            if len(self._mu_cache) >= self._mu_cache_max:
                # FIFO-evict one entry (dicts are insertion-ordered); a full
                # clear() would force a thundering re-fill of the hot set.
                self._mu_cache.pop(next(iter(self._mu_cache)))
            self._mu_cache[key] = hit
        return hit * self.config.immigrant_bias

    # ------------------------------------------------------------------
    # Parent search
    # ------------------------------------------------------------------

    def _assign_parent(self, event: OnlineEvent):
        if event.finalized:
            return event
        mu = self._immigrant_mu(event)
        best_event, best_score = self._best_parent(event, mu)
        if best_event is None or best_score < mu:
            self._bind_immigrant(event, mu)
        else:
            self._bind_to_parent(event, best_event, best_score)
            self._children.setdefault(best_event.index, []).append(event.index)
        event.finalized = True
        return event

    def _descendant_exclusions(self, event: OnlineEvent) -> set:
        """Indexes excluded as parents: the target itself + its descendants
        (events that already picked the target as a future parent under slack)."""
        out = {event.index}
        first = self._children.get(event.index)
        if not first:
            return out
        stack = list(first)
        while stack:
            idx = stack.pop()
            if idx in out:
                continue
            out.add(idx)
            more = self._children.get(idx)
            if more:
                stack.extend(more)
        return out

    def _candidate_ok(self, ev: OnlineEvent, excluded: set) -> bool:
        if ev.index in excluded:
            return False
        cid = ev.cascade_id
        if cid != -1 and cid not in self.cascades:
            return False
        return True

    @staticmethod
    def _better_candidate(cand, best) -> bool:
        """Match the legacy nearest-first argmax enumeration on ties.

        cand/best: (score, abs_dt, side, event) — side 0 = past/equal-ts,
        1 = future. Legacy order: strictly larger score wins; then smaller
        |Δt| (nearest scanned first, argmax keeps the first max); then past
        before future; then the within-side index order legacy encountered:
        equal-ts run ascending (smaller index first), strictly-past descending
        (larger index first), future ascending (smaller index first).
        """
        if best is None:
            return True
        s1, d1, side1, e1 = cand
        s2, d2, side2, e2 = best
        if s1 != s2:
            return s1 > s2
        if d1 != d2:
            return d1 < d2
        if side1 != side2:
            return side1 < side2
        if side1 == 0 and d1 == 0.0:
            return e1.index < e2.index
        if side1 == 0:
            return e1.index > e2.index
        return e1.index < e2.index

    def _cap_radius(self, t: float):
        """Optional legacy-cap approximation: |Δt| radius containing ~cap live
        events, computed by bisect over the (ts-sorted) event buffer."""
        cap = self._candidate_cap
        if cap <= 0:
            return None
        buf = self._buf_ts
        head = self._head
        lo = bisect.bisect_left(buf, t - self._window, head)
        hi = bisect.bisect_right(buf, t + self._slack, head)
        if hi - lo - 1 <= cap:  # -1: the target itself is in the buffer
            return None
        lo_r, hi_r = 0.0, self._window
        for _ in range(40):
            mid = (lo_r + hi_r) / 2.0
            c = (
                bisect.bisect_right(buf, t + min(mid, self._slack), head)
                - bisect.bisect_left(buf, t - mid, head)
                - 1
            )
            if c > cap:
                hi_r = mid
            else:
                lo_r = mid
        return lo_r

    def _alpha_for_keys(self, keys, tgt_at_id, t_entity, t_node, t_site, t_vendor,
                        t_netype, t_dom, tgt_term):
        """Vectorized decomposed α (+ floor + relation prior) for gathered keys."""
        d = self.decomposed
        K = len(keys)
        at_v = np.empty(K, dtype=np.int64)
        mark_idx_arr = np.empty(K, dtype=np.int64)
        topo = np.empty(K, dtype=np.float64)
        is_same_ne = np.empty(K, dtype=np.float64)
        same_site = np.empty(K, dtype=np.float64)
        same_vendor = np.empty(K, dtype=np.float64)
        same_netype = np.empty(K, dtype=np.float64)
        dom_v = np.full(K, -1, dtype=np.int64)
        entity_attrs = self._entity_attrs
        topo_idx = self.feature_scorer.topology_index
        tcache = self.feature_scorer._topo_cache
        src_nodes = []
        for i, (a, ent, m, _bucket) in enumerate(keys):
            s_node, s_site, s_vendor, s_netype, s_dom = entity_attrs[ent]
            src_nodes.append(s_node)
            at_v[i] = a
            mark_idx_arr[i] = m
            topo[i] = _topo_score(s_node, t_node, topo_idx, tcache)
            is_same_ne[i] = 1.0 if ent == t_entity else 0.0
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
            dom_v[i] = s_dom
        alpha = d.alpha_from_parts(
            tgt_at_id, at_v, topo, is_same_ne, same_site, same_vendor, same_netype,
            t_dom, dom_v, mark_idx_arr, tgt_term,
        )
        if self._feat_alpha_floor > 0:
            alpha = np.where(alpha >= self._feat_alpha_floor, alpha, 0.0)
        if self._topology_relation_prior:
            alpha = alpha * topology_relation_weights(
                src_nodes,
                t_node,
                self._topology_index,
                self._node_infos,
                self._topology_relation_prior,
            )
        return alpha

    def _probe_keys(self, buckets, alpha, t, cutoff, slack_edge, excluded, best,
                    skip_entities=None):
        """Boundary-event probes over buckets in DESCENDING α order with exact
        early termination: score ≤ α·β (exp and late weights ≤ 1), so once
        α_i·β < the running best score no remaining bucket can win or tie.

        skip_entities: entities whose events must be ignored — used by the
        signature sweep to skip RELATED entities (already scored in stage 1
        with their own, different α)."""
        beta = self._feat_beta
        inv_scale = 1.0 / self._scale
        window_cut = t - self._window
        order = np.argsort(-alpha, kind="stable")

        def ok(ev):
            if skip_entities is not None and ev.ne in skip_entities:
                return False
            return self._candidate_ok(ev, excluded)

        for i in order:
            a_i = float(alpha[i])
            if a_i <= 0.0:
                break  # α sorted: the rest are all zero
            if best is not None and a_i * beta < best[0]:
                break  # no remaining bucket can beat OR tie the running best
            bucket = buckets[i]
            bucket.prune_before(window_cut)
            ts_list = bucket.ts
            n = len(ts_list)
            if not n:
                continue
            j_right = bisect.bisect_right(ts_list, t)
            j_left = bisect.bisect_left(ts_list, t)
            # past side: the ts == t run start dominates (dt = 0); else the
            # latest strictly-past event. Walk over rare exclusions only.
            cand_ev = None
            cand_dt = 0.0
            jj = j_left
            while jj < j_right:
                ev = bucket.events[jj]
                if ok(ev):
                    cand_ev = ev
                    break
                jj += 1
            if cand_ev is None:
                jj = j_left - 1
                while jj >= 0 and ts_list[jj] >= cutoff:
                    ev = bucket.events[jj]
                    if ok(ev):
                        cand_ev = ev
                        cand_dt = (t - ts_list[jj]) * inv_scale
                        break
                    jj -= 1
            if cand_ev is not None:
                score = a_i * beta * math.exp(-beta * cand_dt)
                cand = (score, abs(t - cand_ev.ts), 0, cand_ev)
                if self._better_candidate(cand, best):
                    best = cand
            # future side (time slack): earliest event after t.
            if self._slack > 0 and j_right < n:
                jj = j_right
                while jj < n and ts_list[jj] <= slack_edge:
                    ev = bucket.events[jj]
                    if ok(ev):
                        late = (ts_list[jj] - t) * inv_scale
                        score = a_i * beta * math.exp(-self._lam * late)
                        cand = (score, ts_list[jj] - t, 1, ev)
                        if self._better_candidate(cand, best):
                            best = cand
                        break
                    jj += 1
        return best

    def _global_bound_score(self, tgt_at_id, mark_combo, tgt_term, glob_ok) -> float:
        """Sound upper bound on any UNRELATED (global-tier) candidate's score:
        max over live glob-ok source ats of softplus(glob_base)·scale·β, times
        the best relation weight an unrelated pair can classify to. Memoized on
        (target at, target mark combo); _bump_live_at invalidates the memo
        whenever the set of live source ats changes."""
        key = (tgt_at_id, mark_combo)
        hit = self._bound_memo.get(key)
        if hit is not None:
            return hit
        live_mask = np.zeros(self._n_at + 1, dtype=bool)
        for at_id in self._at_live_counts:
            live_mask[at_id + 1] = True
        mask = glob_ok & live_mask
        if not mask.any():
            bound = -1.0
        else:
            z = self._glob_base[tgt_at_id + 1][mask] + tgt_term
            bound = float(np.max(softplus(z))) * self.decomposed.alpha_scale * self._feat_beta
            prior = self._topology_relation_prior
            if prior:
                # unrelated pairs classify to cross_site or unknown only
                bound *= max(float(prior.get("cross_site", 1.0)), float(prior.get("unknown", 1.0)))
        self._bound_memo[key] = bound
        return bound

    def _best_parent(self, event: OnlineEvent, mu: float):
        """(best_parent_event, score) over all admissible live keys, or
        (None, 0.0). Exact argmax over the same candidate population the
        legacy scan scores (modulo the documented cap semantics).

        Two-stage, match_rules-style: RELATED keys (target's node + topology
        neighbors + live same-site nodes) are scored first; the GLOBAL tier is
        swept only when its sound score upper bound could still beat both the
        related best and μ — in storms that bound almost never clears the
        same-device score, so the sweep is skipped wholesale.
        """
        self.scored_targets += 1
        d = self.decomposed
        tgt_at_id = self._at_to_id.get(str(event.alarm_type), -1)
        mark_combo = mark_to_combo(event.src_mark)
        tgt_term = d.tgt_term(event.src_mark) if len(d.w_dyn_tgt) else 0.0
        rel_ok, glob_ok = self._tier_vectors(tgt_at_id, mark_combo, tgt_term)
        if not rel_ok.any():
            self.fast_immigrant_shortcuts += 1
            return None, 0.0

        t_entity = event.ne
        attrs = self._entity_attrs.get(t_entity)
        if attrs is None:
            attrs = self._resolve_entity_attrs(t_entity)
            self._entity_attrs[t_entity] = attrs
        t_node, t_site, t_vendor, t_netype, t_dom = attrs

        t = event.ts
        cutoff = t - self._window
        slack_edge = t + self._slack
        cap_r = self._cap_radius(t)
        if cap_r is not None:
            cutoff = max(cutoff, t - cap_r)
            slack_edge = min(slack_edge, t + cap_r)
        excluded = self._descendant_exclusions(event)

        # ---- stage 1: related tier --------------------------------------
        keys = []
        seen_entities = set()
        topo_idx = self.feature_scorer.topology_index
        # t_node unconditionally: entities on the target's own node (incl. the
        # empty/placeholder node) can still match via is_same_ne.
        cand_nodes = {t_node}
        if t_node and topo_idx is not None:
            cand_nodes.update(
                (getattr(topo_idx, "undirected_hops", {}) or {}).get(t_node, ())
            )
        if t_site:
            live_site_nodes = self._site_nodes.get(t_site)
            if live_site_nodes:
                cand_nodes.update(live_site_nodes)
        for node in cand_nodes:
            ents = self._node_entities.get(node)
            if not ents:
                continue
            for ent in ents:
                seen_entities.add(ent)
                for (at_id, mark_idx), bucket in self._by_entity[ent].items():
                    if rel_ok[at_id + 1] and bucket.ts:
                        keys.append((at_id, ent, mark_idx, bucket))
        best = None
        if keys:
            self.gathered_keys_total += len(keys)
            alpha = self._alpha_for_keys(
                keys, tgt_at_id, t_entity, t_node, t_site, t_vendor, t_netype, t_dom, tgt_term
            )
            best = self._probe_keys(
                [k[3] for k in keys], alpha, t, cutoff, slack_edge, excluded, best
            )

        # ---- stage 2: global (unrelated) tier over SIGNATURE buckets -----
        # With a related best of score s, a global candidate only changes the
        # outcome if it can reach s (ties included — the tie-break could pick
        # it), so sweep iff bound >= s. With no related candidate, a global
        # candidate only matters if it can reach μ (score == μ binds), so
        # sweep iff bound >= μ. Everything below is unreachable → skip.
        # The sweep itself is O(live signatures): unrelated α is entity-free,
        # so per signature only the boundary events matter; related entities
        # inside a signature bucket are skipped (stage 1 scored them exactly).
        # The final decision only changes if a global candidate can reach μ
        # (binding requires score >= μ) AND beat/tie the related best, so the
        # sweep threshold is max(best, μ): a global argmax below μ still ends
        # as the same immigrant, and one below the best never wins the argmax.
        if glob_ok.any():
            bound = self._global_bound_score(tgt_at_id, mark_combo, tgt_term, glob_ok)
            threshold = mu if best is None else max(best[0], mu)
            if bound >= threshold:
                best = self._probe_global_signatures(
                    tgt_at_id, tgt_term, glob_ok, t_site, t_vendor, t_netype, t_dom,
                    t, cutoff, slack_edge, excluded, seen_entities, best,
                )

        if best is None:
            self.fast_immigrant_shortcuts += 1
            return None, 0.0
        return best[3], float(best[0])

    def _probe_global_signatures(self, tgt_at_id, tgt_term, glob_ok, t_site,
                                 t_vendor, t_netype, t_dom, t, cutoff, slack_edge,
                                 excluded, related_entities, best):
        d = self.decomposed
        sigs = []
        buckets = []
        for sig, bucket in self._sig_buckets.items():
            if bucket.ts and glob_ok[sig[0] + 1]:
                sigs.append(sig)
                buckets.append(bucket)
        if not sigs:
            return best
        self.global_sweeps += 1
        self.gathered_keys_total += len(sigs)
        K = len(sigs)
        at_v = np.empty(K, dtype=np.int64)
        mark_idx_arr = np.empty(K, dtype=np.int64)
        same_vendor = np.empty(K, dtype=np.float64)
        same_netype = np.empty(K, dtype=np.float64)
        dom_v = np.full(K, -1, dtype=np.int64)
        zeros = np.zeros(K, dtype=np.float64)
        has_site = np.empty(K, dtype=bool)
        for i, (a, vendor, netype, dom_id, sited, m) in enumerate(sigs):
            at_v[i] = a
            mark_idx_arr[i] = m
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == netype) else 0.0
            dom_v[i] = dom_id
            has_site[i] = sited
        alpha = d.alpha_from_parts(
            tgt_at_id, at_v, zeros, zeros, zeros, same_vendor, same_netype,
            t_dom, dom_v, mark_idx_arr, tgt_term,
        )
        if self._feat_alpha_floor > 0:
            alpha = np.where(alpha >= self._feat_alpha_floor, alpha, 0.0)
        if self._topology_relation_prior:
            # unrelated pairs classify to cross_site (both sides sited — the
            # same-site case is always in the related tier) or unknown.
            prior = self._topology_relation_prior
            w_cross = float(prior.get("cross_site", 1.0))
            w_unknown = float(prior.get("unknown", 1.0))
            relw = np.where(has_site & bool(t_site), w_cross, w_unknown)
            alpha = alpha * relw
        return self._probe_keys(
            buckets, alpha, t, cutoff, slack_edge, excluded, best,
            skip_entities=related_entities,
        )

    # ------------------------------------------------------------------
    # Housekeeping: piggyback the throttled close scan for index sweeps.
    # ------------------------------------------------------------------

    def _close_inactive(self, now_ts: float):
        prev = self._last_close_scan_ts
        super()._close_inactive(now_ts)
        # The full O(live buckets) sweep gets its own window-scaled cadence:
        # per-probe prune_before keeps hot buckets tidy, and the close throttle
        # can be configured far tighter than index hygiene warrants.
        if (
            self._last_close_scan_ts != prev
            and now_ts - self._last_sweep_ts >= self._sweep_interval
        ):
            self._last_sweep_ts = now_ts
            self._sweep_indexes(now_ts)

    def _sweep_indexes(self, now_ts: float):
        """Prune dead bucket prefixes and drop empty keys/entities from every
        index. now_ts is the watermark; future targets have ts > watermark,
        so ts < watermark - window can never be a candidate again."""
        cutoff = now_ts - self._window
        dead_keys = []
        for key, bucket in self._buckets.items():
            bucket.prune_before(cutoff)
            if not bucket.ts:
                dead_keys.append(key)
        for key in dead_keys:
            self._buckets.pop(key, None)
            at_id, ent, mark_idx = key
            self._bump_live_at(at_id, -1)
            ent_map = self._by_entity.get(ent)
            if ent_map is not None:
                ent_map.pop((at_id, mark_idx), None)
                if not ent_map:
                    self._by_entity.pop(ent, None)
                    attrs = self._entity_attrs.get(ent)
                    node = attrs[0] if attrs else topo_node_of(ent)
                    node_ents = self._node_entities.get(node)
                    if node_ents is not None:
                        node_ents.discard(ent)
                        if not node_ents:
                            self._node_entities.pop(node, None)
                            site = attrs[1] if attrs else ""
                            if site:
                                site_nodes = self._site_nodes.get(site)
                                if site_nodes is not None:
                                    site_nodes.discard(node)
                                    if not site_nodes:
                                        self._site_nodes.pop(site, None)
        dead_sigs = []
        for sig, bucket in self._sig_buckets.items():
            bucket.prune_before(cutoff)
            if not bucket.ts:
                dead_sigs.append(sig)
        for sig in dead_sigs:
            self._sig_buckets.pop(sig, None)
        # Attr cache: keep live entities only; a returning entity recomputes its
        # attrs from node_infos in O(1). Without this the cache grows with every
        # distinct entity ever seen over the stream's lifetime.
        if len(self._entity_attrs) > 2 * len(self._by_entity) + 1024:
            self._entity_attrs = {
                ent: attrs
                for ent, attrs in self._entity_attrs.items()
                if ent in self._by_entity
            }
        if self._children:
            child_cutoff = cutoff - self._slack
            events_by_index = self._events_by_index
            dead = [
                k
                for k in self._children
                if (e := events_by_index.get(k)) is None or e.ts < child_cutoff
            ]
            for k in dead:
                self._children.pop(k, None)

    def stats(self) -> dict:
        out = super().stats()
        out.update(
            {
                "engine": "fast",
                "live_key_count": len(self._buckets),
                "live_entity_count": len(self._by_entity),
                "scored_targets": self.scored_targets,
                "gathered_keys_total": self.gathered_keys_total,
                "fast_immigrant_shortcuts": self.fast_immigrant_shortcuts,
                "global_sweeps": self.global_sweeps,
                "mu_cache_size": len(self._mu_cache),
            }
        )
        return out
