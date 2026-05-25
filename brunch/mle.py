"""MLE update of Hawkes parameters given a branching structure.

Closed-form MLE under the linear cluster Poisson view with exponential kernels.
For non-linear dimensions, we still use these formulas as a pragmatic update —
they correspond to the linearized model and are a reasonable initializer that
gets refined as MCMC explores (B, C). The paper itself sweeps Θ from a prior
and does not prescribe an update; alternating MLE in practice converges much
faster than holding Θ fixed.
"""

from __future__ import annotations

import numpy as np

from .events import EventCollection
from .params import HawkesParams
from .state import BranchingState


_EPS = 1e-9


def mle_update(
    events: EventCollection,
    state: BranchingState,
    params: HawkesParams,
    *,
    min_alpha: float = 1e-3,
    min_beta: float = 1e-3,
    max_beta: float = 50.0,
    stability_radius: float = 0.95,
    alpha_prior_count: float = 1.0,
    alpha_prior_weight: float = 2.0,
    beta_prior_count: float = 1.0,
    beta_prior_sum_dt: float = 1.0,
    mu_smoothing: float = 0.5,
    edge_threshold: float | None = None,
    max_active_sources_per_dim: int | None = None,
) -> HawkesParams:
    """Bayesian (smoothed) parameter update given the current (B, C).

    Without smoothing, vanilla MLE on a single MCMC state is degenerate:
    if no cross-dim parents are currently assigned, α_ij collapses to 0,
    which kills the prior weight for cluster_link and locks the chain into
    an "all-immigrant" attractor. We apply weak Dirichlet/Gamma priors so
    every (i, j) pair retains nonzero mass even with zero observed
    transitions.

    Hyperparameters
    ---------------
    alpha_prior_count, alpha_prior_weight : pseudo-count for α_ij ratio
        (Dirichlet-style); higher → stronger pull toward 1/(M·weight).
    beta_prior_count, beta_prior_sum_dt : Gamma prior on β_ij.
    mu_smoothing : convex blend with the previous μ_i; 1.0 = pure MLE,
        0.0 = freeze μ. Keeps μ from oscillating between "all events
        immigrant" and "no events immigrant" extremes.
    """
    M = events.M
    if edge_threshold is None:
        edge_threshold = params.edge_threshold
    if max_active_sources_per_dim is None:
        max_active_sources_per_dim = params.max_active_sources_per_dim
    parent = state.parent_of()
    is_immigrant = parent == np.arange(events.n)

    # μ_i = immigrants per dim / horizon, blended with previous value.
    raw_mu = np.zeros(M)
    for i in range(M):
        raw_mu[i] = float(np.sum(is_immigrant & (events.dims == i))) / max(events.T, _EPS)
    raw_mu = np.maximum(raw_mu, _EPS)
    new_mu = mu_smoothing * raw_mu + (1.0 - mu_smoothing) * params.mu

    if params.is_sparse:
        return _mle_update_sparse(
            events,
            parent,
            is_immigrant,
            params,
            new_mu,
            min_alpha=min_alpha,
            min_beta=min_beta,
            max_beta=max_beta,
            stability_radius=stability_radius,
            alpha_prior_count=alpha_prior_count,
            alpha_prior_weight=alpha_prior_weight,
            beta_prior_count=beta_prior_count,
            beta_prior_sum_dt=beta_prior_sum_dt,
            edge_threshold=edge_threshold,
            max_active_sources_per_dim=max_active_sources_per_dim,
        )

    # Count parents by (target_dim, source_dim) and sum Δt
    n_ij = np.zeros((M, M), dtype=np.float64)
    sum_dt = np.zeros((M, M), dtype=np.float64)
    for g in range(events.n):
        if is_immigrant[g]:
            continue
        i = int(events.dims[g])
        j = int(events.dims[parent[g]])
        n_ij[i, j] += 1.0
        sum_dt[i, j] += float(events.times[g] - events.times[parent[g]])

    # α_ij with Dirichlet-like shrinkage toward 1 / (M · alpha_prior_weight)
    n_j = np.bincount(events.dims, minlength=M).astype(np.float64)
    new_alpha = (n_ij + alpha_prior_count) / (n_j[None, :] + alpha_prior_count * M * alpha_prior_weight)
    new_alpha = np.maximum(new_alpha, min_alpha)

    # β_ij with Gamma prior (n_ij + a) / (sum_dt + b)
    new_beta = (n_ij + beta_prior_count) / (sum_dt + beta_prior_sum_dt)
    new_beta = np.clip(new_beta, min_beta, max_beta)

    if max_active_sources_per_dim is not None and max_active_sources_per_dim < M:
        keep = np.zeros((M, M), dtype=bool)
        k = int(max_active_sources_per_dim)
        # Scores retain observed parent pairs first, then previous high-mass edges.
        scores = n_ij + np.maximum(params.alpha_matrix(), 0.0)
        for i in range(M):
            row = scores[i]
            if k >= M:
                keep[i] = True
            else:
                top = np.argpartition(row, -k)[-k:]
                keep[i, top] = True
            keep[i, i] = True
            observed = n_ij[i] > 0.0
            keep[i, observed] = True
        new_alpha = np.where(keep, new_alpha, 0.0)

    if edge_threshold > 0.0:
        new_alpha = np.where(np.abs(new_alpha) > edge_threshold, new_alpha, 0.0)

    # Cap spectral radius for stationarity (linear MHP only).
    tmp_params = HawkesParams(
        M=M,
        mu=new_mu,
        alpha=new_alpha,
        beta=new_beta,
        links=list(params.links),
        edge_threshold=edge_threshold,
        max_active_sources_per_dim=max_active_sources_per_dim,
    )
    rho = tmp_params.spectral_radius()
    if rho > stability_radius and rho > 0:
        new_alpha *= stability_radius / rho
        if edge_threshold > 0.0:
            new_alpha = np.where(np.abs(new_alpha) > edge_threshold, new_alpha, 0.0)

    return HawkesParams(
        M=M,
        mu=new_mu,
        alpha=new_alpha,
        beta=new_beta,
        links=list(params.links),
        edge_threshold=edge_threshold,
        max_active_sources_per_dim=max_active_sources_per_dim,
    )


def _mle_update_sparse(
    events: EventCollection,
    parent: np.ndarray,
    is_immigrant: np.ndarray,
    params: HawkesParams,
    new_mu: np.ndarray,
    *,
    min_alpha: float,
    min_beta: float,
    max_beta: float,
    stability_radius: float,
    alpha_prior_count: float,
    alpha_prior_weight: float,
    beta_prior_count: float,
    beta_prior_sum_dt: float,
    edge_threshold: float,
    max_active_sources_per_dim: int | None,
) -> HawkesParams:
    """Sparse MLE update without allocating M×M parameter/count matrices."""
    M = events.M
    n_j = np.bincount(events.dims, minlength=M).astype(np.float64)

    counts: dict[tuple[int, int], float] = {}
    sum_dt: dict[tuple[int, int], float] = {}
    by_target = [set() for _ in range(M)]
    observed_by_target = [set() for _ in range(M)]

    prev_targets, prev_sources = params.active_edges(include_self=True)
    for i, j in zip(prev_targets, prev_sources):
        by_target[int(i)].add(int(j))
    for i in range(M):
        by_target[i].add(i)

    for g in range(events.n):
        if is_immigrant[g]:
            continue
        i = int(events.dims[g])
        j = int(events.dims[parent[g]])
        key = (i, j)
        counts[key] = counts.get(key, 0.0) + 1.0
        sum_dt[key] = sum_dt.get(key, 0.0) + float(events.times[g] - events.times[parent[g]])
        by_target[i].add(j)
        observed_by_target[i].add(j)

    edge_targets = []
    edge_sources = []
    edge_alpha = []
    edge_beta = []

    for i in range(M):
        candidates = list(by_target[i])
        if max_active_sources_per_dim is not None and max_active_sources_per_dim < len(candidates):
            must_keep = set(observed_by_target[i])
            must_keep.add(i)
            rest = [j for j in candidates if j not in must_keep]
            rest.sort(key=lambda j: (counts.get((i, j), 0.0) + max(params.alpha_value(i, j), 0.0)), reverse=True)
            room = max(int(max_active_sources_per_dim) - len(must_keep), 0)
            candidates = sorted(must_keep | set(rest[:room]))

        for j in candidates:
            n = counts.get((i, j), 0.0)
            dt_sum = sum_dt.get((i, j), 0.0)
            a = (n + alpha_prior_count) / (
                n_j[j] + alpha_prior_count * M * alpha_prior_weight
            )
            a = max(float(a), min_alpha)
            if edge_threshold > 0.0 and abs(a) <= edge_threshold:
                continue
            b = (n + beta_prior_count) / (dt_sum + beta_prior_sum_dt)
            b = float(np.clip(b, min_beta, max_beta))
            edge_targets.append(i)
            edge_sources.append(j)
            edge_alpha.append(a)
            edge_beta.append(b)

    edge_targets_arr = np.asarray(edge_targets, dtype=np.int64)
    edge_sources_arr = np.asarray(edge_sources, dtype=np.int64)
    edge_alpha_arr = np.asarray(edge_alpha, dtype=np.float64)
    edge_beta_arr = np.asarray(edge_beta, dtype=np.float64)

    tmp_params = HawkesParams.from_edges(
        M=M,
        mu=new_mu,
        edge_targets=edge_targets_arr,
        edge_sources=edge_sources_arr,
        edge_alpha=edge_alpha_arr,
        edge_beta=edge_beta_arr,
        links=list(params.links),
        edge_threshold=edge_threshold,
        max_active_sources_per_dim=max_active_sources_per_dim,
        default_beta=params.default_beta,
    )
    rho = tmp_params.spectral_radius()
    if rho > stability_radius and rho > 0:
        edge_alpha_arr = edge_alpha_arr * (stability_radius / rho)
        if edge_threshold > 0.0:
            keep = np.abs(edge_alpha_arr) > edge_threshold
            edge_targets_arr = edge_targets_arr[keep]
            edge_sources_arr = edge_sources_arr[keep]
            edge_alpha_arr = edge_alpha_arr[keep]
            edge_beta_arr = edge_beta_arr[keep]

    return HawkesParams.from_edges(
        M=M,
        mu=new_mu,
        edge_targets=edge_targets_arr,
        edge_sources=edge_sources_arr,
        edge_alpha=edge_alpha_arr,
        edge_beta=edge_beta_arr,
        links=list(params.links),
        edge_threshold=edge_threshold,
        max_active_sources_per_dim=max_active_sources_per_dim,
        default_beta=params.default_beta,
    )
