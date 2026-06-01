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
from .params import MHPParams


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


def _compute_initial_alpha_beta(
    events: EventCollection,
    n_pair: np.ndarray,
    sum_dt: Optional[np.ndarray],
    config: MHPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """MAP point estimate from accumulated pair statistics."""
    M = events.M
    K = config.alpha_prior_strength
    m = config.alpha_prior_mean
    n_source = np.bincount(events.dims, minlength=M).astype(np.float64)
    denom = n_source[np.newaxis, :] + K
    alpha = ((n_pair + K * m) / denom).astype(np.float32)
    if config.beta_mode == "per_edge":
        K_b = config.beta_prior_strength
        m_b = max(config.beta_prior_mean, _EPS)
        beta = ((n_pair + K_b) / ((sum_dt if sum_dt is not None else 0.0) + K_b / m_b)).astype(np.float32)
        np.clip(beta, config.beta_min, config.beta_max, out=beta)
    else:
        beta = np.full((M, M), config.beta_shared_value, dtype=np.float32)
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
        beta_pair = beta[pair_target_dim, pair_source_dim]
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
) -> MHPResult:
    """Run MAP EM on the event sequence."""
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
        alpha_data, beta_data = _compute_initial_alpha_beta(events, n_pair, sum_dt, config)
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
    best_alpha = alpha.copy()
    best_beta = beta.copy()
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
        alpha_new = ((alpha_num + K * m) / denom).astype(np.float32)

        if config.beta_mode == "per_edge":
            K_b = config.beta_prior_strength
            m_b = max(config.beta_prior_mean, _EPS)
            beta_new = (
                (alpha_num + K_b) / ((beta_num_dt if beta_num_dt is not None else 0.0) + K_b / m_b)
            ).astype(np.float32)
            np.clip(beta_new, config.beta_min, config.beta_max, out=beta_new)
        else:
            beta_new = beta

        # Sparsity and stability
        n_rescaled = _apply_branching_cap(alpha_new, config.branching_cap)
        _apply_top_k_per_target(alpha_new, config.max_active_sources_per_dim, config.edge_threshold)

        ll = _log_likelihood_global(ll_term1, mu_new, alpha_new, horizon)
        delta_rel = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf

        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha_new.copy()
            best_beta = beta_new.copy()
            best_mu = mu_new.copy()
            best_p_self = p_self

        active_edges = int((alpha_new > 0).sum())
        t_iter_end = time.monotonic()
        iter_total = t_iter_end - t_iter_start
        iter_estep = t_estep_end - t_iter_start
        iter_mstep = t_iter_end - t_estep_end
        # β stats — only meaningful on active edges (β=0 where α=0)
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

    # Final spectral-radius safety net (the per-source cap above already
    # implies ρ ≤ branching_cap, so this rarely fires).
    if config.stability_radius > 0:
        tmp_targets, tmp_sources = np.nonzero(best_alpha)
        if len(tmp_targets):
            tmp_params = MHPParams.from_edges(
                M=M,
                mu=best_mu,
                edge_targets=tmp_targets,
                edge_sources=tmp_sources,
                edge_alpha=best_alpha[tmp_targets, tmp_sources],
                edge_beta=best_beta[tmp_targets, tmp_sources],
                edge_threshold=config.edge_threshold,
                max_active_sources_per_dim=config.max_active_sources_per_dim,
                beta_shared=(config.beta_mode == "shared"),
            )
            rho = tmp_params.spectral_radius()
            if rho > config.stability_radius and rho > 0:
                scale = config.stability_radius / rho
                best_alpha = best_alpha * scale
                if config.verbose:
                    print(
                        f"[mhp] spectral safety net: ρ={rho:.4f} > {config.stability_radius} "
                        f"→ α × {scale:.4f}",
                        flush=True,
                    )

    # Build final sparse params
    tgts, srcs = np.nonzero(best_alpha)
    if config.edge_threshold > 0 and len(tgts):
        keep = best_alpha[tgts, srcs] > config.edge_threshold
        tgts = tgts[keep]
        srcs = srcs[keep]
    final_params = MHPParams.from_edges(
        M=M,
        mu=best_mu,
        edge_targets=tgts,
        edge_sources=srcs,
        edge_alpha=best_alpha[tgts, srcs] if len(tgts) else np.zeros(0),
        edge_beta=best_beta[tgts, srcs] if len(tgts) else np.zeros(0),
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
    # Build dense α/β for fast lookup
    alpha_dense = np.zeros((M, M), dtype=np.float32)
    beta_dense = np.zeros((M, M), dtype=np.float32)
    if len(params.edge_targets):
        alpha_dense[params.edge_targets, params.edge_sources] = params.edge_alpha.astype(np.float32)
        beta_dense[params.edge_targets, params.edge_sources] = params.edge_beta.astype(np.float32)
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
        alpha_pair = alpha_dense[pair_target_dim, pair_source_dim]
        beta_pair = beta_dense[pair_target_dim, pair_source_dim]
        score_pair = alpha_pair * beta_pair * np.exp(-beta_pair * pair_dt)
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
    M = events.M
    N = events.n
    times = events.times
    dims = events.dims
    history_window = cfg.history_window
    max_history_events = max(int(cfg.max_history_events), 1)
    chunk_size = max(int(cfg.chunk_size), 1)

    # Build dense α/β for fast lookup (lazy — keep it local to this call).
    alpha_dense = np.zeros((M, M), dtype=np.float32)
    beta_dense = np.zeros((M, M), dtype=np.float32)
    if len(params.edge_targets):
        alpha_dense[params.edge_targets, params.edge_sources] = params.edge_alpha.astype(np.float32)
        beta_dense[params.edge_targets, params.edge_sources] = params.edge_beta.astype(np.float32)

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
        alpha_pair = alpha_dense[pair_target_dim, pair_source_dim]
        beta_pair = beta_dense[pair_target_dim, pair_source_dim]
        safe_beta = np.where(beta_pair > 0, beta_pair, 1.0)
        score_pair = alpha_pair * safe_beta * np.exp(-safe_beta * pair_dt)
        score_pair = np.where(alpha_pair > 0, score_pair, 0.0)
        sum_score = _segment_sum(score_pair.astype(np.float64), pair_target_local, chunk_size_local)
        rate = np.maximum(mu_chunk + sum_score, _EPS)
        rate_term += float(np.log(rate).sum())

    return _log_likelihood_global(rate_term, params.mu, alpha_dense.astype(np.float64), events.T)
