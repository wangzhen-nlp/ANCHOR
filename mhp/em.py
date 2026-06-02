"""MAP EM for multivariate Hawkes processes (chunked windowed sparse).

Implements an MAP EM iteration over the multivariate Hawkes likelihood, with
three scalability changes vs. Morse's reference MHP:

  1. Windowed candidate parents. Each event i only considers parent candidates
     j with t_j ∈ [t_i - W, t_i) and at most `max_history_events` of them. The
     (event, candidate) pair list grows linearly in N, not quadratically.

  2. Chunked E-step. Pair arrays are never fully materialized; events are
     processed in fixed-size chunks (`chunk_size`, default 20k events) and
     sufficient statistics are accumulated across chunks. Memory peak is
     bounded by O(chunk_size · K + M²) — independent of N. Enables training
     on 2M+ events at M ~ 10k without OOM.

  3. Sparse-aware Bayesian shrinkage M-step. α and (optionally) β are updated
     with Gamma-Poisson conjugate priors driven by `alpha_prior_strength`,
     `alpha_prior_mean`, `beta_prior_strength`, `beta_prior_mean`. Same prior
     as `_build_initial_params`, so init and refit speak the same prior.

The internal α and β representations during EM are dense float32 matrices
(M × M × 4 bytes — about 320 MB at M=8898). The returned MHPParams is sparse:
edges with α below `edge_threshold` after the final iteration are dropped,
and at most `max_active_sources_per_dim` sources are kept per target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Optional

import numpy as np

from .events import EventCollection
from .params import MHPParams, bucket_index_vec, bucket_widths


_EPS = 1e-12


def _fmt_secs(seconds: float) -> str:
    """Compact human-readable elapsed time."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m{rem:04.1f}s"


@dataclass
class MHPConfig:
    """EM hyperparameters."""

    history_window: float = 10.0
    max_history_events: int = 128
    max_iters: int = 50
    tol: float = 1e-4
    log_every: int = 1
    # MAP shrinkage (mirrors alarm_flow_brunch.aggregator):
    alpha_prior_strength: float = 10.0
    alpha_prior_mean: float = 0.1
    mu_count_smoothing: str = "log"
    # β handling:
    beta_mode: str = "shared"
    beta_shared_value: float = 1.0
    beta_prior_strength: float = 5.0
    beta_prior_mean: float = 1.0
    beta_min: float = 1e-2
    beta_max: float = 50.0
    # Sparsity and stability:
    edge_threshold: float = 1e-3
    max_active_sources_per_dim: Optional[int] = 16
    branching_cap: float = 0.9
    stability_radius: float = 0.95
    # Chunked-processing knob. Each chunk holds chunk_size events; per-chunk
    # peak pair memory is chunk_size · max_history_events · ~16 bytes. With
    # defaults: 20k · 128 · 16 = 40 MB.
    chunk_size: int = 20_000
    # Piecewise (box-basis) kernel. When kernel_type == "piecewise", training
    # runs two stages: (1) exp-kernel fit selects the sparse active edge set,
    # (2) a box-basis EM learns per-edge per-bucket weights θ on those edges.
    # bucket_edges are right edges in MODEL TIME (ascending, last == window).
    kernel_type: str = "exp"                 # "exp" | "piecewise"
    bucket_edges: tuple = ()                  # set by aggregator from real-sec config
    # Topology prior: extra MAP prior mass on topologically-related (target,
    # source) type pairs so they get (or strengthen) an edge even with little/no
    # co-occurrence. The actual (flat_idx, score) pairs are passed to fit_mhp
    # by the alarm layer (it knows the NE graph); this scalar is the strength.
    # α[u,v] gains K · boost · topo_score[u,v] of prior mass in the numerator.
    topology_prior_boost: float = 0.0        # 0 = disabled
    seed: int = 0
    verbose: bool = True


@dataclass
class MHPResult:
    params: MHPParams
    log_likelihood: float
    iterations_run: int
    converged: bool
    trace: list = field(default_factory=list)
    # Per-event posterior of "this event is an immigrant" — useful for cascade
    # output downstream. Length N, in [0, 1].
    p_self: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


def _segment_sum(values: np.ndarray, segment_ids: np.ndarray, n_segments: int) -> np.ndarray:
    out = np.zeros(n_segments, dtype=np.float64)
    np.add.at(out, segment_ids, values)
    return out


def _build_chunk_pair_arrays(
    times: np.ndarray,
    dims: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    history_window: float,
    max_history_events: int,
):
    """Build pair arrays for a chunk of events.

    Returns
    -------
    pair_target : (P,) int64        # global event id of child
    pair_source : (P,) int64        # global event id of candidate parent
    pair_dt     : (P,) float32      # t[child] - t[parent]
    pair_target_dim, pair_source_dim : (P,) int64
    pair_target_local : (P,) int64  # pair_target - chunk_start
    counts_per_event : (chunk_size,) int64

    All arrays are fully vectorized — no Python per-event loop.
    """
    chunk_size = chunk_end - chunk_start
    target_event_ids = np.arange(chunk_start, chunk_end, dtype=np.int64)
    # Window lower bound for each event in chunk
    window_starts = np.searchsorted(times, times[target_event_ids] - history_window, side="left")
    lower = np.maximum(window_starts, target_event_ids - max_history_events)
    counts = np.maximum(target_event_ids - lower, 0).astype(np.int64)
    P = int(counts.sum())
    if P == 0:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f32 = np.empty(0, dtype=np.float32)
        return empty_i64, empty_i64, empty_f32, empty_i64, empty_i64, empty_i64, counts
    # Vectorized pair construction (no Python loop):
    pair_offsets = np.zeros(chunk_size + 1, dtype=np.int64)
    np.cumsum(counts, out=pair_offsets[1:])
    pair_target = np.repeat(target_event_ids, counts)
    event_idx_in_chunk = np.repeat(np.arange(chunk_size, dtype=np.int64), counts)
    within_event = np.arange(P, dtype=np.int64) - pair_offsets[event_idx_in_chunk]
    pair_source = lower[event_idx_in_chunk] + within_event
    pair_dt = (times[pair_target] - times[pair_source]).astype(np.float32)
    pair_target_dim = dims[pair_target]
    pair_source_dim = dims[pair_source]
    pair_target_local = pair_target - chunk_start
    return (
        pair_target,
        pair_source,
        pair_dt,
        pair_target_dim,
        pair_source_dim,
        pair_target_local,
        counts,
    )


def _accumulate_initial_pair_stats(
    events: EventCollection,
    config: MHPConfig,
):
    """First pass to accumulate (n_pair, sum_dt) for data-driven α/β init.

    Returns
    -------
    n_pair : (M, M) float64   # count of (target, source) candidate pairs
    sum_dt : (M, M) float64   # sum of Δt over those pairs (for per_edge β init)
    """
    M = events.M
    N = events.n
    times = events.times
    dims = events.dims
    n_pair = np.zeros((M, M), dtype=np.float64)
    sum_dt = np.zeros((M, M), dtype=np.float64) if config.beta_mode == "per_edge" else None
    chunk_size = max(int(config.chunk_size), 1)
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        (
            _,
            _,
            pair_dt,
            pair_target_dim,
            pair_source_dim,
            _,
            _,
        ) = _build_chunk_pair_arrays(
            times,
            dims,
            chunk_start,
            chunk_end,
            config.history_window,
            max(int(config.max_history_events), 1),
        )
        if pair_dt.size == 0:
            continue
        flat_uv = pair_target_dim.astype(np.int64) * M + pair_source_dim.astype(np.int64)
        np.add.at(n_pair.ravel(), flat_uv, 1.0)
        if sum_dt is not None:
            np.add.at(sum_dt.ravel(), flat_uv, pair_dt.astype(np.float64))
    return n_pair, sum_dt


def _add_topology_prior(
    alpha_num: np.ndarray,
    config: MHPConfig,
    topo_prior_flat: Optional[np.ndarray],
    topo_prior_score: Optional[np.ndarray],
) -> None:
    """Scatter-add topology prior mass into the α numerator (in place).

    For each topology-related (target, source) pair, adds K · boost · score to
    the numerator, so its MAP estimate becomes
        α = (count + K·m_base + K·boost·score) / (n_v + K).
    Pairs with no co-occurrence (count=0) get a nonzero α purely from this term
    — i.e. a zero-shot topology edge. The (n_v + K) denominator (applied by the
    caller) naturally gives this term MORE weight for rare sources (small n_v),
    which is exactly where data-driven edges are missing.
    """
    if (
        config.topology_prior_boost <= 0.0
        or topo_prior_flat is None
        or len(topo_prior_flat) == 0
    ):
        return
    mass = config.alpha_prior_strength * config.topology_prior_boost * topo_prior_score
    np.add.at(alpha_num.reshape(-1), topo_prior_flat, mass)


def _compute_initial_alpha_beta(
    events: EventCollection,
    n_pair: np.ndarray,
    sum_dt: Optional[np.ndarray],
    config: MHPConfig,
    topo_prior_flat: Optional[np.ndarray] = None,
    topo_prior_score: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP point estimate from accumulated pair statistics.

    Memory note: for shared β, β is returned as a 0-d scalar (np.float32)
    rather than a dense (M, M) matrix — at M=24356 that saves ~2.4 GB and is
    mathematically identical (every entry was the same constant). The α
    computation reuses the n_pair buffer in place to avoid a transient
    (M, M) float64 copy (~4.4 GB), which is what triggers OOM at large M.
    `n_pair` is consumed (mutated) by this call.
    """
    M = events.M
    K = config.alpha_prior_strength
    m = config.alpha_prior_mean
    n_source = np.bincount(events.dims, minlength=M).astype(np.float64)
    denom = n_source[np.newaxis, :] + K
    if config.beta_mode == "per_edge":
        # β must be computed from n_pair BEFORE n_pair is mutated below.
        K_b = config.beta_prior_strength
        m_b = max(config.beta_prior_mean, _EPS)
        beta = ((n_pair + K_b) / ((sum_dt if sum_dt is not None else 0.0) + K_b / m_b)).astype(np.float32)
        np.clip(beta, config.beta_min, config.beta_max, out=beta)
    else:
        beta = np.float32(config.beta_shared_value)  # 0-d scalar, not (M, M)
    # In-place: alpha = (n_pair + K*m [+ topology prior]) / denom.
    n_pair += K * m
    _add_topology_prior(n_pair, config, topo_prior_flat, topo_prior_score)
    n_pair /= denom
    alpha = n_pair.astype(np.float32)
    return alpha, beta


def _compute_mu_initial(
    events: EventCollection,
    horizon: float,
    config: MHPConfig,
) -> np.ndarray:
    M = events.M
    counts = np.bincount(events.dims, minlength=M).astype(np.float64)
    if config.mu_count_smoothing == "log":
        signal = 0.1 * np.log1p(counts) / horizon
    else:
        signal = 0.1 * counts / horizon
    return np.maximum(0.05 / horizon, signal)


def _apply_branching_cap(alpha: np.ndarray, branching_cap: float) -> int:
    if branching_cap <= 0:
        return 0
    col_sums = alpha.sum(axis=0)
    over = col_sums > branching_cap
    if not over.any():
        return 0
    scale = np.ones_like(col_sums)
    scale[over] = branching_cap / col_sums[over]
    alpha *= scale[np.newaxis, :]
    return int(over.sum())


def _apply_top_k_per_target(
    alpha: np.ndarray,
    k: Optional[int],
    edge_threshold: float,
) -> None:
    if k is not None and k > 0 and k < alpha.shape[1]:
        for u in range(alpha.shape[0]):
            row = alpha[u]
            nonzero = (row > 0).sum()
            if nonzero <= k:
                continue
            kth = np.argpartition(-row, k)[k:]
            row[kth] = 0.0
    if edge_threshold > 0:
        alpha[np.abs(alpha) <= edge_threshold] = 0.0


def _run_estep_iteration(
    events: EventCollection,
    alpha: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    config: MHPConfig,
):
    """One full E-step pass, chunked, accumulating M-step sufficient stats.

    Returns
    -------
    p_self : (N,) float64           # per-event immigrant posterior
    alpha_num : (M, M) float64      # Σ p_ij per (target, source) type pair
    beta_num_dt : (M, M) float64    # Σ p_ij · dt (None if shared β)
    mu_num : (M,) float64           # Σ p_self per target type
    log_likelihood : float          # Σ log rate_i (term1 of LL)
    """
    M = events.M
    N = events.n
    times = events.times
    dims = events.dims
    history_window = config.history_window
    max_history_events = max(int(config.max_history_events), 1)
    chunk_size = max(int(config.chunk_size), 1)

    p_self = np.zeros(N, dtype=np.float64)
    alpha_num = np.zeros((M, M), dtype=np.float64)
    beta_num_dt = (
        np.zeros((M, M), dtype=np.float64) if config.beta_mode == "per_edge" else None
    )
    mu_num = np.zeros(M, dtype=np.float64)
    log_likelihood = 0.0

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_size_local = chunk_end - chunk_start
        target_dims_chunk = dims[chunk_start:chunk_end]
        mu_chunk = mu[target_dims_chunk]

        (
            _,
            _,
            pair_dt,
            pair_target_dim,
            pair_source_dim,
            pair_target_local,
            _,
        ) = _build_chunk_pair_arrays(
            times,
            dims,
            chunk_start,
            chunk_end,
            history_window,
            max_history_events,
        )

        if pair_dt.size == 0:
            # All chunk events are immigrants (no candidate parents in window)
            rate = np.maximum(mu_chunk, _EPS)
            p_self_chunk = np.ones(chunk_size_local, dtype=np.float64)
            p_self[chunk_start:chunk_end] = p_self_chunk
            mu_num_chunk = _segment_sum(p_self_chunk, target_dims_chunk, M)
            mu_num += mu_num_chunk
            log_likelihood += float(np.log(rate).sum())
            continue

        # E-step on this chunk
        alpha_pair = alpha[pair_target_dim, pair_source_dim]
        # β is a 0-d scalar in shared mode (saves a dense (M,M) array) or a
        # dense matrix in per_edge mode. Both broadcast in the score formula.
        beta_pair = beta if np.ndim(beta) == 0 else beta[pair_target_dim, pair_source_dim]
        # Score: α · β · exp(-β · Δt)
        score_pair = alpha_pair * beta_pair * np.exp(-beta_pair * pair_dt)
        # rate_per_event = μ_{u_i} + Σ_j score(i, j)
        sum_score = _segment_sum(score_pair.astype(np.float64), pair_target_local, chunk_size_local)
        rate = np.maximum(mu_chunk + sum_score, _EPS)
        p_self_chunk = mu_chunk / rate
        p_self[chunk_start:chunk_end] = p_self_chunk
        # p_ij[p] = score[p] / rate[target_local]
        p_ij = score_pair / rate[pair_target_local].astype(np.float32)

        # Accumulate per-type sufficient stats
        flat_uv = pair_target_dim.astype(np.int64) * M + pair_source_dim.astype(np.int64)
        np.add.at(alpha_num.ravel(), flat_uv, p_ij.astype(np.float64))
        if beta_num_dt is not None:
            np.add.at(
                beta_num_dt.ravel(),
                flat_uv,
                p_ij.astype(np.float64) * pair_dt.astype(np.float64),
            )
        mu_num_chunk = _segment_sum(p_self_chunk, target_dims_chunk, M)
        mu_num += mu_num_chunk
        log_likelihood += float(np.log(rate).sum())

    return p_self, alpha_num, beta_num_dt, mu_num, log_likelihood


def _log_likelihood_global(
    rate_term: float,
    mu: np.ndarray,
    alpha: np.ndarray,
    horizon: float,
) -> float:
    """LL ≈ Σ_i log rate_i − T · Σ_d μ_d − Σ_{u,v} α[u,v]·G_int.

    The third term approximates the kernel integral assuming each parent
    contributes its full unit mass (β·exp integrates to 1 over [0, ∞)). Since
    Σ α[u,v]·n_v under windowing is hard to bound tightly, we use the proxy
    Σ α — sufficient for relative LL comparison across iterations / models.
    """
    term2 = horizon * float(mu.sum())
    term3 = float(alpha.sum())
    return rate_term - term2 - term3


def fit_mhp(
    events: EventCollection,
    config: MHPConfig,
    *,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    init_mu: Optional[np.ndarray] = None,
    iter_callback: Optional[Callable[[dict], None]] = None,
    topo_prior_flat: Optional[np.ndarray] = None,
    topo_prior_score: Optional[np.ndarray] = None,
) -> MHPResult:
    """Run MAP EM on the event sequence.

    topo_prior_flat / topo_prior_score: optional sparse topology prior. When
    config.topology_prior_boost > 0, these (flat index = target*M + source,
    score in [0,1]) inject extra MAP prior mass on topology-related pairs in
    both the init and every M-step — see _add_topology_prior.
    """
    M = events.M
    N = events.n
    horizon = events.T

    t_total_start = time.monotonic()
    if config.verbose:
        print(
            f"[mhp] events={N}, types={M}, chunk_size={config.chunk_size}, "
            f"max_history_events={config.max_history_events}",
            flush=True,
        )

    # Initial α, β, μ from data heuristic if not caller-provided.
    if init_alpha is None or init_beta is None:
        if config.verbose:
            print("[mhp] pass 1: accumulating initial pair statistics ...", flush=True)
        t_init_start = time.monotonic()
        n_pair, sum_dt = _accumulate_initial_pair_stats(events, config)
        alpha_data, beta_data = _compute_initial_alpha_beta(
            events, n_pair, sum_dt, config, topo_prior_flat, topo_prior_score
        )
        alpha = alpha_data if init_alpha is None else init_alpha.astype(np.float32)
        beta = beta_data if init_beta is None else init_beta.astype(np.float32)
        del n_pair, sum_dt
        init_pass_seconds = time.monotonic() - t_init_start
    else:
        alpha = init_alpha.astype(np.float32)
        beta = init_beta.astype(np.float32)
        init_pass_seconds = 0.0
    if init_mu is None:
        mu = _compute_mu_initial(events, horizon, config)
    else:
        mu = np.asarray(init_mu, dtype=np.float64).reshape(-1)

    # Apply initial stability constraints
    n_rescaled_init = _apply_branching_cap(alpha, config.branching_cap)
    _apply_top_k_per_target(alpha, config.max_active_sources_per_dim, config.edge_threshold)
    if config.verbose:
        active = int((alpha > 0).sum())
        print(
            f"[mhp] init: active_edges={active} rescaled_cols={n_rescaled_init} "
            f"α.max={alpha.max():.4f} μ.max={mu.max():.4f} "
            f"init_pass={_fmt_secs(init_pass_seconds)}",
            flush=True,
        )

    trace: list[dict] = []
    best_ll = -np.inf
    # Best-iteration snapshot stored SPARSELY (edge list), not as a dense
    # (M, M) copy — after top-k pruning α has <= max_active_sources_per_dim
    # nonzeros per row, so this is a few hundred KB instead of ~2.4 GB held
    # across all iterations.
    best_edge_targets = np.zeros(0, dtype=np.int64)
    best_edge_sources = np.zeros(0, dtype=np.int64)
    best_edge_alpha = np.zeros(0, dtype=np.float64)
    best_edge_beta = None                                   # set in per_edge mode
    best_beta_scalar = float(beta) if np.ndim(beta) == 0 else None
    best_mu = mu.copy()
    best_p_self = np.zeros(N, dtype=np.float64)
    converged = False
    prev_ll = -np.inf

    for it in range(config.max_iters):
        t_iter_start = time.monotonic()
        # E-step (chunked) returns sufficient statistics for M-step
        p_self, alpha_num, beta_num_dt, mu_num, ll_term1 = _run_estep_iteration(
            events, alpha, beta, mu, config
        )
        t_estep_end = time.monotonic()

        # M-step
        mu_new = np.maximum(mu_num / max(horizon, _EPS), 0.05 / horizon)

        n_source = np.bincount(events.dims, minlength=M).astype(np.float64)
        K = config.alpha_prior_strength
        m = config.alpha_prior_mean
        denom = n_source[np.newaxis, :] + K

        if config.beta_mode == "per_edge":
            # β must be read from alpha_num BEFORE the in-place α mutation below.
            K_b = config.beta_prior_strength
            m_b = max(config.beta_prior_mean, _EPS)
            beta_new = (
                (alpha_num + K_b) / ((beta_num_dt if beta_num_dt is not None else 0.0) + K_b / m_b)
            ).astype(np.float32)
            np.clip(beta_new, config.beta_min, config.beta_max, out=beta_new)
        else:
            beta_new = beta  # scalar

        # In-place: alpha_new = (alpha_num + K*m [+ topology prior]) / denom,
        # reusing the f64 accumulator buffer to avoid a transient (M,M) copy
        # (~4.4 GB at large M).
        alpha_num += K * m
        _add_topology_prior(alpha_num, config, topo_prior_flat, topo_prior_score)
        alpha_num /= denom
        alpha_new = alpha_num.astype(np.float32)
        del alpha_num  # release the 4.4 GB f64 buffer immediately

        # Sparsity and stability
        n_rescaled = _apply_branching_cap(alpha_new, config.branching_cap)
        _apply_top_k_per_target(alpha_new, config.max_active_sources_per_dim, config.edge_threshold)

        ll = _log_likelihood_global(ll_term1, mu_new, alpha_new, horizon)
        delta_rel = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf

        beta_is_scalar = np.ndim(beta_new) == 0
        if ll > best_ll:
            best_ll = ll
            nz_t, nz_s = np.nonzero(alpha_new)
            best_edge_targets = nz_t.copy()
            best_edge_sources = nz_s.copy()
            best_edge_alpha = alpha_new[nz_t, nz_s].astype(np.float64)
            if beta_is_scalar:
                best_beta_scalar = float(beta_new)
                best_edge_beta = None
            else:
                best_beta_scalar = None
                best_edge_beta = beta_new[nz_t, nz_s].astype(np.float64)
            best_mu = mu_new.copy()
            best_p_self = p_self

        active_edges = int((alpha_new > 0).sum())
        t_iter_end = time.monotonic()
        iter_total = t_iter_end - t_iter_start
        iter_estep = t_estep_end - t_iter_start
        iter_mstep = t_iter_end - t_estep_end
        # β stats — only meaningful on active edges (β=0 where α=0)
        if beta_is_scalar:
            bval = float(beta_new)
            beta_median_active = bval if active_edges else 0.0
            beta_max_active = bval if active_edges else 0.0
            beta_min_active = bval if active_edges else 0.0
        else:
            active_mask = alpha_new > 0
            if active_mask.any():
                beta_active = beta_new[active_mask]
                beta_median_active = float(np.median(beta_active))
                beta_max_active = float(beta_active.max())
                beta_min_active = float(beta_active.min())
            else:
                beta_median_active = 0.0
                beta_max_active = 0.0
                beta_min_active = 0.0
        trace_entry = {
            "iter": it,
            "log_likelihood": float(ll),
            "delta_rel": float(delta_rel),
            "branching_rescaled": n_rescaled,
            "active_edges": active_edges,
            "mu_max": float(mu_new.max()),
            "mu_median": float(np.median(mu_new)),
            "alpha_max": float(alpha_new.max()),
            "alpha_median_active": float(np.median(alpha_new[alpha_new > 0])) if active_edges else 0.0,
            "beta_min_active": beta_min_active,
            "beta_median_active": beta_median_active,
            "beta_max_active": beta_max_active,
            "p_self_mean": float(p_self.mean()),
            "iter_seconds": float(iter_total),
            "estep_seconds": float(iter_estep),
            "mstep_seconds": float(iter_mstep),
        }
        trace.append(trace_entry)
        if iter_callback is not None:
            iter_callback(trace_entry)
        if config.verbose and (it % max(config.log_every, 1) == 0 or it == config.max_iters - 1):
            print(
                f"[mhp] iter={it:3d} ll={ll:.2f} Δ={delta_rel:.2e} "
                f"active_edges={active_edges} "
                f"μ.max={trace_entry['mu_max']:.4f} "
                f"α.max={trace_entry['alpha_max']:.4f} "
                f"β.median={beta_median_active:.3f} β.max={beta_max_active:.3f} "
                f"p_self.mean={trace_entry['p_self_mean']:.3f} "
                f"rescaled_cols={n_rescaled} "
                f"t={_fmt_secs(iter_total)} (E={_fmt_secs(iter_estep)} M={_fmt_secs(iter_mstep)})",
                flush=True,
            )

        alpha = alpha_new
        beta = beta_new
        mu = mu_new

        if it > 0 and delta_rel < config.tol:
            if config.verbose:
                print(
                    f"[mhp] converged at iter {it} (Δrel={delta_rel:.2e} < tol={config.tol:.2e})",
                    flush=True,
                )
            converged = True
            break
        prev_ll = ll

    # The best snapshot is already a sparse edge list (best_edge_*). Build the
    # per-edge β array: replicate the scalar for shared mode.
    if best_beta_scalar is not None:
        best_edge_beta_arr = np.full(len(best_edge_targets), best_beta_scalar, dtype=np.float64)
    else:
        best_edge_beta_arr = (
            best_edge_beta if best_edge_beta is not None else np.zeros(len(best_edge_targets))
        )

    # Final spectral-radius safety net (the per-source cap above already
    # implies ρ ≤ branching_cap, so this rarely fires).
    if config.stability_radius > 0 and len(best_edge_targets):
        tmp_params = MHPParams.from_edges(
            M=M,
            mu=best_mu,
            edge_targets=best_edge_targets,
            edge_sources=best_edge_sources,
            edge_alpha=best_edge_alpha,
            edge_beta=best_edge_beta_arr,
            edge_threshold=config.edge_threshold,
            max_active_sources_per_dim=config.max_active_sources_per_dim,
            beta_shared=(config.beta_mode == "shared"),
        )
        rho = tmp_params.spectral_radius()
        if rho > config.stability_radius and rho > 0:
            scale = config.stability_radius / rho
            best_edge_alpha = best_edge_alpha * scale
            if config.verbose:
                print(
                    f"[mhp] spectral safety net: ρ={rho:.4f} > {config.stability_radius} "
                    f"→ α × {scale:.4f}",
                    flush=True,
                )

    # Build final sparse params (edge_threshold filter applied inside from_edges)
    final_params = MHPParams.from_edges(
        M=M,
        mu=best_mu,
        edge_targets=best_edge_targets,
        edge_sources=best_edge_sources,
        edge_alpha=best_edge_alpha,
        edge_beta=best_edge_beta_arr,
        edge_threshold=config.edge_threshold,
        max_active_sources_per_dim=config.max_active_sources_per_dim,
        beta_shared=(config.beta_mode == "shared"),
    )

    if config.verbose:
        total_fit = time.monotonic() - t_total_start
        avg_iter = sum(e["iter_seconds"] for e in trace) / max(len(trace), 1)
        print(
            f"[mhp] fit complete: iters={len(trace)} converged={converged} "
            f"total={_fmt_secs(total_fit)} avg_iter={_fmt_secs(avg_iter)}",
            flush=True,
        )
    return MHPResult(
        params=final_params,
        log_likelihood=best_ll,
        iterations_run=len(trace),
        converged=converged,
        trace=trace,
        p_self=best_p_self,
    )


def _make_pair_scorer(params: MHPParams):
    """Return a vectorized score_fn(pair_target_dim, pair_source_dim, pair_dt)
    that dispatches on params.kernel_type.

    Uses a SPARSE edge lookup (binary search over sorted edge keys) instead of
    a dense (M, M) table. At M=24356 the dense tables were ~2.4 GB each; the
    sparse keys array is only O(active_edges). MHPParams.from_edges stores
    edges sorted by (target, source), so key = target*M + source is ascending
    and np.searchsorted gives an O(log E) vectorized lookup.

    Used by compute_hard_parents and log_likelihood so both kernels share one
    inference path.
    """
    M = params.M
    E = len(params.edge_targets)
    keys = (
        params.edge_targets.astype(np.int64) * M + params.edge_sources.astype(np.int64)
        if E
        else np.zeros(0, dtype=np.int64)
    )

    def _lookup(pair_tdim, pair_sdim):
        """Return (edge_idx, valid_mask) for each pair via binary search."""
        if E == 0:
            n = len(pair_tdim)
            return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
        pair_key = pair_tdim.astype(np.int64) * M + pair_sdim.astype(np.int64)
        idx = np.searchsorted(keys, pair_key)
        idx_clip = np.minimum(idx, E - 1)
        valid = keys[idx_clip] == pair_key
        return idx_clip, valid

    if params.kernel_type == "piecewise":
        bucket_edges = np.asarray(params.bucket_edges, dtype=np.float64)
        theta = np.asarray(params.edge_theta, dtype=np.float64)

        def score_fn(pair_tdim, pair_sdim, pair_dt):
            out = np.zeros(pair_dt.shape, dtype=np.float64)
            ei, valid = _lookup(pair_tdim, pair_sdim)
            if valid.any():
                pb = bucket_index_vec(pair_dt[valid].astype(np.float64), bucket_edges)
                out[valid] = theta[ei[valid], pb]
            return out

        return score_fn

    # exp kernel. Keep α/β/Δt in float32 so the score matches the old dense
    # scorer (and the training E-step) bit-for-bit — both computed in f32 and
    # only widened to f64 on assignment into `out`.
    edge_alpha = params.edge_alpha.astype(np.float32)
    edge_beta = params.edge_beta.astype(np.float32)

    def score_fn(pair_tdim, pair_sdim, pair_dt):
        out = np.zeros(pair_dt.shape, dtype=np.float64)
        ei, valid = _lookup(pair_tdim, pair_sdim)
        if valid.any():
            a = edge_alpha[ei[valid]]                       # f32
            b = edge_beta[ei[valid]]                        # f32
            dt = pair_dt[valid]                             # f32 (as passed)
            out[valid] = a * b * np.exp(-b * dt)            # f32 compute, widened on assign
        return out

    return score_fn


def fit_mhp_piecewise(
    events: EventCollection,
    config: MHPConfig,
    *,
    edge_targets: np.ndarray,
    edge_sources: np.ndarray,
    init_mu: np.ndarray,
    iter_callback: Optional[Callable[[dict], None]] = None,
) -> MHPResult:
    """Stage-2 box-basis kernel learning on a FIXED active edge set.

    Given the sparse edges selected by stage 1, learn per-edge per-bucket
    weights θ[e, k] ≥ 0 such that the kernel is the step function
        g_e(Δt) = θ[e, bucket(Δt)].
    μ is also refined. The branching ratio of edge e is Σ_k θ[e,k]·width_k,
    which is what the stationarity cap constrains.

    M-step (closed form, MAP with Gamma prior):
        θ[e,k] = (Σ responsibility_{e,k} + K·m) / (n_v[e]·width_k + K)
    where the width_k in the denominator is the exposure of bucket k — this
    is what makes wide and narrow buckets comparable.
    """
    M = events.M
    N = events.n
    horizon = events.T
    bucket_edges = np.asarray(config.bucket_edges, dtype=np.float64)
    B = len(bucket_edges)
    if B == 0:
        raise ValueError("piecewise kernel requires non-empty bucket_edges")
    widths = bucket_widths(bucket_edges)                      # (B,)
    E = len(edge_targets)
    edge_targets = np.asarray(edge_targets, dtype=np.int64)
    edge_sources = np.asarray(edge_sources, dtype=np.int64)
    # Sparse edge lookup via sorted keys (binary search) instead of a dense
    # (M, M) int32 index map (~2.4 GB at M=24356). Sort defensively so
    # searchsorted is valid regardless of caller ordering; θ/resp are built
    # AFTER this so they align with the sorted edge order.
    if E:
        _order = np.lexsort((edge_sources, edge_targets))
        edge_targets = edge_targets[_order]
        edge_sources = edge_sources[_order]
        edge_keys = edge_targets * M + edge_sources
    else:
        edge_keys = np.zeros(0, dtype=np.int64)

    def _edge_lookup(pair_tdim, pair_sdim):
        """(edge_idx_clipped, valid_mask) for each pair via binary search."""
        if E == 0:
            n = len(pair_tdim)
            return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
        pk = pair_tdim.astype(np.int64) * M + pair_sdim.astype(np.int64)
        idx = np.minimum(np.searchsorted(edge_keys, pk), E - 1)
        return idx, edge_keys[idx] == pk

    n_source = np.bincount(events.dims, minlength=M).astype(np.float64)
    # n_v per edge (source-type event count) — the exposure base
    n_v_per_edge = n_source[edge_sources]                     # (E,)
    K = config.alpha_prior_strength
    m = config.alpha_prior_mean

    t_total_start = time.monotonic()
    if config.verbose:
        print(
            f"[mhp-pw] stage 2 box-basis: edges={E}, buckets={B}, "
            f"bucket_edges(model-time)={list(np.round(bucket_edges, 3))}",
            flush=True,
        )

    # Initialize θ from windowed co-occurrence per bucket (shape-agnostic).
    resp_init = np.zeros((E, B), dtype=np.float64)
    chunk_size = max(int(config.chunk_size), 1)
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        (_, _, pair_dt, pair_tdim, pair_sdim, _, _) = _build_chunk_pair_arrays(
            events.times, events.dims, chunk_start, chunk_end,
            config.history_window, max(int(config.max_history_events), 1),
        )
        if pair_dt.size == 0:
            continue
        pe, valid = _edge_lookup(pair_tdim, pair_sdim)
        if not valid.any():
            continue
        pb = bucket_index_vec(pair_dt[valid].astype(np.float64), bucket_edges)
        flat = pe[valid].astype(np.int64) * B + pb
        np.add.at(resp_init.ravel(), flat, 1.0)
    theta = (resp_init + K * m) / (n_v_per_edge[:, None] * widths[None, :] + K)
    mu = np.asarray(init_mu, dtype=np.float64).reshape(-1).copy()

    trace: list[dict] = []
    best_ll = -np.inf
    best_theta = theta.copy()
    best_mu = mu.copy()
    best_p_self = np.zeros(N, dtype=np.float64)
    converged = False
    prev_ll = -np.inf

    for it in range(config.max_iters):
        t_iter = time.monotonic()
        p_self = np.zeros(N, dtype=np.float64)
        resp = np.zeros((E, B), dtype=np.float64)
        mu_num = np.zeros(M, dtype=np.float64)
        ll_term1 = 0.0

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            csize = chunk_end - chunk_start
            tdims_chunk = events.dims[chunk_start:chunk_end]
            mu_chunk = mu[tdims_chunk]
            (_, _, pair_dt, pair_tdim, pair_sdim, pair_tlocal, _) = _build_chunk_pair_arrays(
                events.times, events.dims, chunk_start, chunk_end,
                config.history_window, max(int(config.max_history_events), 1),
            )
            if pair_dt.size == 0:
                rate = np.maximum(mu_chunk, _EPS)
                p_self[chunk_start:chunk_end] = 1.0
                mu_num += _segment_sum(np.ones(csize), tdims_chunk, M)
                ll_term1 += float(np.log(rate).sum())
                continue
            pe, valid = _edge_lookup(pair_tdim, pair_sdim)
            pb = np.zeros(pair_dt.shape, dtype=np.int64)
            score_pair = np.zeros(pair_dt.shape, dtype=np.float64)
            if valid.any():
                pb_valid = bucket_index_vec(pair_dt[valid].astype(np.float64), bucket_edges)
                pb[valid] = pb_valid
                score_pair[valid] = theta[pe[valid], pb_valid]
            sum_score = _segment_sum(score_pair, pair_tlocal, csize)
            rate = np.maximum(mu_chunk + sum_score, _EPS)
            p_self_chunk = mu_chunk / rate
            p_self[chunk_start:chunk_end] = p_self_chunk
            p_ij = score_pair / rate[pair_tlocal]
            if valid.any():
                flat = pe[valid].astype(np.int64) * B + pb[valid]
                np.add.at(resp.ravel(), flat, p_ij[valid])
            mu_num += _segment_sum(p_self_chunk, tdims_chunk, M)
            ll_term1 += float(np.log(rate).sum())

        # M-step
        mu_new = np.maximum(mu_num / max(horizon, _EPS), 0.05 / horizon)
        theta_new = (resp + K * m) / (n_v_per_edge[:, None] * widths[None, :] + K)

        # Branching cap on Σ_k θ·w, grouped per source type.
        if config.branching_cap > 0:
            branching_per_edge = (theta_new * widths[None, :]).sum(axis=1)   # (E,)
            col_sums = np.zeros(M, dtype=np.float64)
            np.add.at(col_sums, edge_sources, branching_per_edge)
            over = col_sums > config.branching_cap
            n_rescaled = int(over.sum())
            if n_rescaled:
                scale = np.ones(M, dtype=np.float64)
                scale[over] = config.branching_cap / col_sums[over]
                theta_new *= scale[edge_sources][:, None]
        else:
            n_rescaled = 0

        branching_per_edge = (theta_new * widths[None, :]).sum(axis=1)
        ll = ll_term1 - horizon * float(mu_new.sum()) - float(branching_per_edge.sum())
        delta_rel = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf

        if ll > best_ll:
            best_ll = ll
            best_theta = theta_new.copy()
            best_mu = mu_new.copy()
            best_p_self = p_self

        trace_entry = {
            "iter": it,
            "log_likelihood": float(ll),
            "delta_rel": float(delta_rel),
            "branching_rescaled": n_rescaled,
            "active_edges": E,
            "mu_max": float(mu_new.max()),
            "branching_max": float(branching_per_edge.max()),
            "branching_median": float(np.median(branching_per_edge)),
            "p_self_mean": float(p_self.mean()),
            "iter_seconds": float(time.monotonic() - t_iter),
        }
        trace.append(trace_entry)
        if iter_callback is not None:
            iter_callback(trace_entry)
        if config.verbose and (it % max(config.log_every, 1) == 0 or it == config.max_iters - 1):
            print(
                f"[mhp-pw] iter={it:3d} ll={ll:.2f} Δ={delta_rel:.2e} "
                f"branch.median={trace_entry['branching_median']:.4f} "
                f"branch.max={trace_entry['branching_max']:.4f} "
                f"μ.max={trace_entry['mu_max']:.4f} "
                f"p_self.mean={trace_entry['p_self_mean']:.3f} "
                f"rescaled_cols={n_rescaled} "
                f"t={_fmt_secs(trace_entry['iter_seconds'])}",
                flush=True,
            )

        theta = theta_new
        mu = mu_new
        if it > 0 and delta_rel < config.tol:
            converged = True
            if config.verbose:
                print(f"[mhp-pw] converged at iter {it} (Δrel={delta_rel:.2e})", flush=True)
            break
        prev_ll = ll

    if config.verbose:
        print(
            f"[mhp-pw] fit complete: iters={len(trace)} converged={converged} "
            f"total={_fmt_secs(time.monotonic() - t_total_start)}",
            flush=True,
        )

    # edge_alpha summary = branching ratio per edge; edge_beta = 0 placeholder
    branching_per_edge = (best_theta * widths[None, :]).sum(axis=1)
    final_params = MHPParams.from_edges(
        M=M,
        mu=best_mu,
        edge_targets=edge_targets,
        edge_sources=edge_sources,
        edge_alpha=branching_per_edge,
        edge_beta=np.zeros(E, dtype=np.float64),
        edge_threshold=0.0,                       # edges already selected in stage 1
        max_active_sources_per_dim=config.max_active_sources_per_dim,
        kernel_type="piecewise",
        edge_theta=best_theta,
        bucket_edges=tuple(config.bucket_edges),
    )
    return MHPResult(
        params=final_params,
        log_likelihood=best_ll,
        iterations_run=len(trace),
        converged=converged,
        trace=trace,
        p_self=best_p_self,
    )


def compute_hard_parents(
    events: EventCollection,
    params: MHPParams,
    *,
    config: Optional[MHPConfig] = None,
) -> np.ndarray:
    """Decode soft p_ij assignments into a hard per-event parent decision.

    For each event i, we choose the parent that maximizes the unnormalized
    score among (immigrant μ_{u_i}, score(i, j) over candidates j). If the
    immigrant score wins, parent[i] = i (event is its own root); otherwise
    parent[i] = j*, the chosen candidate.

    Returns
    -------
    parent : (N,) int64
        parent[i] is the global event index of the chosen parent (or i if
        i is an immigrant).
    """
    cfg = config or MHPConfig()
    M = events.M
    N = events.n
    times = events.times
    dims = events.dims
    history_window = cfg.history_window
    max_history_events = max(int(cfg.max_history_events), 1)
    chunk_size = max(int(cfg.chunk_size), 1)
    score_fn = _make_pair_scorer(params)
    parent = np.arange(N, dtype=np.int64)  # default: each event is its own parent

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_size_local = chunk_end - chunk_start
        target_dims_chunk = dims[chunk_start:chunk_end]
        mu_chunk = params.mu[target_dims_chunk]
        (
            pair_target,
            pair_source,
            pair_dt,
            pair_target_dim,
            pair_source_dim,
            pair_target_local,
            _,
        ) = _build_chunk_pair_arrays(
            times,
            dims,
            chunk_start,
            chunk_end,
            history_window,
            max_history_events,
        )
        if pair_dt.size == 0:
            # All events in chunk are immigrants (no candidates in window)
            continue
        score_pair = score_fn(pair_target_dim, pair_source_dim, pair_dt)
        # For each event in chunk, find the max candidate score and which parent
        # wins it. Use np.maximum.reduceat by sorted (pair_target_local, score).
        # Faster: sort pairs by (target_local, -score), take first per group.
        # Even simpler: scan with np.add.at on -inf accumulators.
        best_score = np.full(chunk_size_local, -np.inf, dtype=np.float64)
        best_parent = np.full(chunk_size_local, -1, dtype=np.int64)
        # We need argmax per segment; np.maximum.at handles the max, but to get
        # the parent index we need a second pass. Cheaper: sort within groups.
        order = np.lexsort((-score_pair.astype(np.float64), pair_target_local))
        sorted_target_local = pair_target_local[order]
        sorted_parent = pair_source[order]
        sorted_score = score_pair[order]
        # First occurrence of each target_local in sorted order = its best score
        _, first_idx = np.unique(sorted_target_local, return_index=True)
        idxs = sorted_target_local[first_idx]
        best_score[idxs] = sorted_score[first_idx].astype(np.float64)
        best_parent[idxs] = sorted_parent[first_idx]
        # Decide: immigrant if mu > best_score, else best_parent
        immigrant_wins = mu_chunk > best_score
        chunk_event_ids = np.arange(chunk_start, chunk_end, dtype=np.int64)
        parent[chunk_event_ids] = np.where(immigrant_wins, chunk_event_ids, best_parent)
    return parent


def compute_cascade_of(parent: np.ndarray) -> np.ndarray:
    """Union-find on the parent pointers → cascade id per event.

    Returns cascade_of[i] ∈ [0, C) for some C ≤ N.
    """
    N = len(parent)
    cascade = -np.ones(N, dtype=np.int64)
    next_id = 0
    # Iterative path compression
    for i in range(N):
        if cascade[i] != -1:
            continue
        # Walk up parents until we find a known cascade or a root
        path = []
        cur = i
        while cascade[cur] == -1:
            path.append(cur)
            if parent[cur] == cur:
                break
            cur = int(parent[cur])
        if cascade[cur] != -1:
            cid = int(cascade[cur])
        else:
            cid = next_id
            next_id += 1
            cascade[cur] = cid
        for node in path:
            cascade[node] = cid
    return cascade


def log_likelihood(
    events: EventCollection,
    params: MHPParams,
    *,
    config: Optional[MHPConfig] = None,
) -> float:
    """Stand-alone LL evaluation for held-out data with frozen params.

    Uses the same chunked windowed approximation as `fit_mhp`. If `config`
    is omitted, falls back to defaults — supply one matching the training
    config for apples-to-apples val LL comparison.
    """
    cfg = config or MHPConfig()
    N = events.n
    times = events.times
    dims = events.dims
    history_window = cfg.history_window
    max_history_events = max(int(cfg.max_history_events), 1)
    chunk_size = max(int(cfg.chunk_size), 1)
    score_fn = _make_pair_scorer(params)

    rate_term = 0.0
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_size_local = chunk_end - chunk_start
        target_dims_chunk = dims[chunk_start:chunk_end]
        mu_chunk = params.mu[target_dims_chunk]
        (
            _,
            _,
            pair_dt,
            pair_target_dim,
            pair_source_dim,
            pair_target_local,
            _,
        ) = _build_chunk_pair_arrays(
            times,
            dims,
            chunk_start,
            chunk_end,
            history_window,
            max_history_events,
        )
        if pair_dt.size == 0:
            rate = np.maximum(mu_chunk, _EPS)
            rate_term += float(np.log(rate).sum())
            continue
        score_pair = score_fn(pair_target_dim, pair_source_dim, pair_dt)
        sum_score = _segment_sum(score_pair.astype(np.float64), pair_target_local, chunk_size_local)
        rate = np.maximum(mu_chunk + sum_score, _EPS)
        rate_term += float(np.log(rate).sum())

    # term3: integral proxy = total branching mass (Σ edge_alpha, which for
    # piecewise is Σ branching ratio per edge).
    term2 = events.T * float(params.mu.sum())
    term3 = float(params.edge_alpha.sum()) if len(params.edge_alpha) else 0.0
    return rate_term - term2 - term3
