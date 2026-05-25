"""intCRP weight functions (paper §3.1 inner, §3.2 outer)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .events import EventCollection
from .kernels import exp_kernel
from .params import HawkesParams
from .state import BranchingState


def inner_intcrp_weights(
    child_g: int,
    candidate_parents: Sequence[int],
    events: EventCollection,
    params: HawkesParams,
    window: float,
) -> np.ndarray:
    """Eq. 3.1: prior weight for inner intCRP event-link assignment.

    For each candidate parent in `candidate_parents` (must all be in the same
    dimension as `child_g`, strictly preceding child_g in time, OR equal to
    child_g for the self-loop), returns the unnormalized prior weight:
        - self-loop (immigrant): ρ_i = μ_i
        - real parent t_il: 1[t_ik - t_il < W] · α_ii · φ_ii(t_ik - t_il)
    """
    i = int(events.dims[child_g])
    t_child = float(events.times[child_g])
    weights = np.zeros(len(candidate_parents), dtype=np.float64)
    rho_i = max(float(params.mu[i]), 0.0)
    alpha_ii = max(params.alpha_value(i, i), 0.0)
    if params.edge_threshold > 0.0 and alpha_ii <= params.edge_threshold:
        alpha_ii = 0.0
    beta_ii = params.beta_value(i, i)
    for k, g in enumerate(candidate_parents):
        if g == child_g:
            weights[k] = rho_i
            continue
        tau = t_child - float(events.times[g])
        if tau <= 0.0 or tau >= window:
            weights[k] = 0.0
            continue
        weights[k] = alpha_ii * exp_kernel(tau, beta_ii)
    return weights


def inner_candidate_parents(
    child_g: int,
    events: EventCollection,
    window: float,
) -> List[int]:
    """All within-dim events preceding child_g in time within `window`, plus child_g itself."""
    i = int(events.dims[child_g])
    t_child = float(events.times[child_g])
    dim_idx = events.dim_indices(i)
    # dim_idx is sorted by time (since events are time-sorted globally)
    candidates: List[int] = []
    for g in dim_idx:
        if int(g) == child_g:
            break  # all later events in this dim are excluded
        tau = t_child - float(events.times[g])
        if tau <= 0.0:
            continue
        if tau >= window:
            continue
        candidates.append(int(g))
    candidates.append(int(child_g))  # self-loop / immigrant option last
    return candidates


def outer_intcrp_weight(
    source_cluster: int,
    target_cluster: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
) -> float:
    """Eq. 3.2: prior weight for outer intCRP cluster-link s → g (s ≠ g).

    weight = max over t_ik ∈ s with t_ik < t_je of |α_{j,i}| · φ_{j,i}(t_je - t_ik)
    where i = dim(s), j = dim(g), t_je = earliest event time in g.
    """
    if source_cluster == target_cluster:
        return 0.0
    src_events = state.cluster_events(source_cluster)
    tgt_events = state.cluster_events(target_cluster)
    if not src_events or not tgt_events:
        return 0.0
    i = int(events.dims[src_events[0]])
    j = int(events.dims[tgt_events[0]])
    if i == j:
        return 0.0  # within-dim relationships belong to event links, not cluster links
    t_je = float(events.times[tgt_events].min())
    src_times = events.times[src_events]
    valid = src_times < t_je
    if not np.any(valid):
        return 0.0
    alpha_ji = abs(params.alpha_value(j, i))
    if params.edge_threshold > 0.0 and alpha_ji <= params.edge_threshold:
        return 0.0
    beta_ji = params.beta_value(j, i)
    taus = t_je - src_times[valid]
    return float(alpha_ji * np.max(exp_kernel(taus, beta_ji)))


def outer_candidate_parents(
    target_cluster: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams | None = None,
    window: float | None = None,
) -> List[int]:
    """All clusters that could be the cross-dim parent of `target_cluster`:
    different dim, at least one event preceding target's earliest event, and
    (optionally) at least one event no older than `t_je - window`.

    Uses per-cluster earliest/latest caches in `state` for O(1) admissibility
    checks, so the total cost is O(M + Σ_d |clusters_in_dim(d)|) rather than
    O(M · K · avg_cluster_size). Self-loop ('no parent') handled separately.
    """
    target_dim = state.cluster_dim(target_cluster)
    t_je = state.cluster_earliest_time(target_cluster)
    earliest_arr = state.cluster_earliest_array
    latest_arr = state.cluster_latest_array
    earliest_cutoff = -np.inf if window is None else (t_je - float(window))
    parents: List[int] = []
    if params is None:
        source_dims = [d for d in range(events.M) if d != target_dim]
    else:
        source_dims = [
            int(d)
            for d in params.active_sources_for_target(target_dim, include_self=False)
            if int(d) != target_dim
        ]
    for source_dim in source_dims:
        for c in state.clusters_in_dim(source_dim):
            if c == target_cluster:
                continue
            if earliest_arr[c] >= t_je:
                continue  # cluster has no event strictly preceding target's earliest
            if latest_arr[c] <= earliest_cutoff:
                continue  # cluster's most recent event is beyond the look-back window
            parents.append(c)
    return parents
