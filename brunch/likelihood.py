"""Hawkes log-likelihood under a given branching structure.

We use the cluster Poisson view (paper §2.2, Eq. 4.3–4.4): conditional on the cascade
assignment, each event's instantaneous rate is determined by its parent.

Under the linear link function, for cascade I:
- Each immigrant contributes log(μ_i) to the rate term (rate at its instant ≈ μ_i)
- Each offspring with parent t_p contributes log(α_ij β_ij exp(-β_ij Δt)) to the rate
- The full intensity integral is approximated by ∑_m (t_m - t_{m-1}) λ_i(t_m) over the
  global time axis (Eq. 4.4 rectangular rule).

For non-linear link F_i, the conditional rate at the event time is
F_i(μ_i + contribution from parent), and the integral approximation uses
F_i(μ_i + Σ all-history kernel sum). We treat the parent-conditional rate term as
F_i(μ_i + α_ij β_ij e^{-β_ij Δt}) for non-immigrants; immigrants use F_i(μ_i).
"""

from __future__ import annotations

import numpy as np

from .events import EventCollection
from .kernels import apply_link, exp_kernel, exp_kernel_integral
from .params import HawkesParams
from .state import BranchingState


_EPS = 1e-12


def _apply_links_vector(x: np.ndarray, links: list[str]) -> np.ndarray:
    """Apply per-dimension link functions to a vector."""
    if not links:
        return apply_link(x, "linear")
    if all(name == links[0] for name in links):
        return apply_link(x, links[0])
    out = np.empty_like(x, dtype=np.float64)
    links_arr = np.asarray(links, dtype=object)
    for name in set(links):
        mask = links_arr == name
        out[mask] = apply_link(x[mask], name)
    return out


def conditional_rate_at_event(event_g: int, parent_g: int, events: EventCollection, params: HawkesParams) -> float:
    """Rate λ_i(t_ik | parent, Θ) under the cluster Poisson factorization
    (paper §2.2). For immigrants the rate is the background μ_i; for offspring
    it is solely the kernel contribution α_ij φ_ij(Δt) from the parent — adding
    μ_i would double-count the immigrant arrival rate.

    The link function F_i is applied per-dim. For non-linear F_i the cluster
    Poisson view is heuristic, but we keep the same per-parent factorization
    (paper does the same; see §4 commentary on Eq. 4.4).
    """
    i = int(events.dims[event_g])
    if parent_g == event_g:
        return float(apply_link(params.mu[i], params.links[i])) + _EPS
    j = int(events.dims[parent_g])
    dt = float(events.times[event_g] - events.times[parent_g])
    alpha_ij = params.alpha_value(i, j)
    if params.edge_threshold > 0.0 and abs(alpha_ij) <= params.edge_threshold:
        contrib = 0.0
    else:
        contrib = alpha_ij * exp_kernel(dt, params.beta_value(i, j))
    return float(apply_link(contrib, params.links[i])) + _EPS


def survival_integral(events: EventCollection, params: HawkesParams) -> np.ndarray:
    """Rectangular approximation of ∫_0^T λ_i(s) ds for each dim i (Eq. 4.4).

    Uses unconditional intensity (every event contributes), evaluated at observed
    event times. Active α edges are tracked as a sparse edge list, which avoids
    touching the full M×M parameter grid when α has been thresholded or pruned.
    """
    M = params.M
    n = events.n
    out = np.zeros(M, dtype=np.float64)
    if n == 0:
        out = _apply_links_vector(params.mu, params.links) * events.T
        return out

    sorted_idx = np.argsort(events.times, kind="stable")  # times are already sorted but be explicit
    times = events.times[sorted_idx]
    dims = events.dims[sorted_idx]

    targets, sources, edge_alpha, edge_beta = params.edge_values(include_self=True)
    edge_count = len(targets)
    if edge_count:
        order = np.argsort(sources, kind="stable")
        targets = targets[order]
        sources = sources[order]
        edge_alpha = edge_alpha[order]
        edge_beta = edge_beta[order]
        edge_contrib = np.zeros(edge_count, dtype=np.float64)
        source_offsets = np.searchsorted(sources, np.arange(M + 1), side="left")
    else:
        edge_alpha = edge_beta = edge_contrib = np.zeros(0, dtype=np.float64)
        source_offsets = np.zeros(M + 1, dtype=np.int64)

    # accumulator[i] = μ_i + Σ_j active α_ij s_ij(t)
    accumulator = params.mu.astype(np.float64, copy=True)
    prev_t = 0.0
    # First rectangle: from 0 to times[0] uses baseline-only rate.
    out += _apply_links_vector(params.mu, params.links) * times[0]
    for m in range(n):
        t = times[m]
        if m > 0 and edge_count:
            dt = t - prev_t
            old = edge_contrib.copy()
            edge_contrib *= np.exp(-edge_beta * dt)
            np.add.at(accumulator, targets, edge_contrib - old)
        # λ at t_m: F_i(μ_i + Σ_j active α_ij · s_ij)
        lam = _apply_links_vector(accumulator, params.links)
        # Width of rectangle ending at t_m
        if m + 1 < n:
            width = times[m + 1] - t
        else:
            width = max(events.T - t, 0.0)
        out += lam * width
        # Inject the event at t into the kernel state: add β_{i, j} for all i, where j = dim of event m.
        j = int(dims[m])
        start = int(source_offsets[j])
        end = int(source_offsets[j + 1])
        if end > start:
            delta = edge_alpha[start:end] * edge_beta[start:end]
            edge_contrib[start:end] += delta
            np.add.at(accumulator, targets[start:end], delta)
        prev_t = t
    return out


def log_likelihood(events: EventCollection, state: BranchingState, params: HawkesParams) -> float:
    """log p(X | B, C, Θ) using cluster Poisson factorization with rectangular integral."""
    n = events.n
    parent = state.parent_of()
    log_rate_term = 0.0
    for g in range(n):
        rate = conditional_rate_at_event(g, int(parent[g]), events, params)
        log_rate_term += np.log(rate + _EPS)
    integrals = survival_integral(events, params)
    return float(log_rate_term - np.sum(integrals))


def parent_rate_change(
    child_g: int,
    new_parent_g: int,
    events: EventCollection,
    params: HawkesParams,
) -> float:
    """log F_i(μ_i + α_ij β_ij e^{-β_ij Δt}) for a candidate parent.

    Used by Gibbs over event_parent: when only event_g's parent changes, the only
    log-rate term in log p(X | B, C, Θ) that changes is the one for event_g (since
    in the cluster Poisson view each event's rate depends solely on its own parent
    under our approximation). The survival integral is independent of (B, C).
    """
    return float(np.log(conditional_rate_at_event(child_g, new_parent_g, events, params) + _EPS))
