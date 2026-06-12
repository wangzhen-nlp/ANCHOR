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
    # Optional timestamp-jitter tolerance in model-time units. When > 0, a
    # candidate parent may occur up to this far after the target event. The
    # signed dt is still treated as timestamp disorder, not reverse causality:
    # scoring uses max(dt, 0) and discounts late parents by an exponential
    # penalty.
    time_slack: float = 0.0
    late_penalty_half_life: float = 1.0
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


def _late_penalty_lambda(config: MHPConfig) -> float:
    half_life = float(getattr(config, "late_penalty_half_life", 1.0))
    if half_life <= 0:
        raise ValueError("late_penalty_half_life must be > 0")
    return float(np.log(2.0) / half_life)


def _negative_penalty_integral(config: MHPConfig) -> float:
    """∫_0^slack exp(-λs) ds in model-time units."""
    slack = float(getattr(config, "time_slack", 0.0))
    if slack <= 0:
        return 0.0
    lam = _late_penalty_lambda(config)
    return float((1.0 - np.exp(-lam * slack)) / lam)


def _apply_time_slack(pair_dt: np.ndarray, config: MHPConfig):
    """Return (effective_dt, late_weight) for signed candidate dt values.

    Positive dt is unchanged. Negative dt (parent timestamp slightly after child)
    is clamped to 0 and exponentially discounted, modeling timestamp jitter
    rather than reverse-time triggering.
    """
    dt = np.asarray(pair_dt)
    slack = float(getattr(config, "time_slack", 0.0))
    if dt.size == 0:
        return dt, np.ones(dt.shape, dtype=np.float32)
    dt_eff = np.maximum(dt, 0.0).astype(dt.dtype, copy=False)
    late = np.maximum(-dt.astype(np.float64), 0.0)
    if slack <= 0:
        weight = (late <= 0).astype(np.float32)
    else:
        weight = np.where(
            late <= slack,
            np.exp(-_late_penalty_lambda(config) * late),
            0.0,
        ).astype(np.float32)
    return dt_eff, weight


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
    # Feature-mode only: the learned log-linear amplitude model. None for
    # device-mode / piecewise. Lets inference compute α for unseen pairs.
    feature_kernel: object = None
    # Feature-mode only: the learned log-linear immigrant model μ=softplus(w·ψ).
    mu_kernel: object = None


def _segment_sum(values: np.ndarray, segment_ids: np.ndarray, n_segments: int) -> np.ndarray:
    out = np.zeros(n_segments, dtype=np.float64)
    np.add.at(out, segment_ids, values)
    return out


def _spectral_radius_edges(edge_targets, edge_sources, edge_alpha, M, power_iter: int = 80) -> float:
    """Spectral radius ρ of the branching matrix A[target, source]=|α| via power
    iteration — identical to MHPParams.spectral_radius, but callable on raw edge
    arrays so the EM loop can report ρ each iteration without materializing the
    full MHPParams. ρ<1 ⇒ stationary/subcritical; ρ≥1 ⇒ over-excitation."""
    et = np.asarray(edge_targets, dtype=np.int64)
    if et.size == 0 or M <= 0:
        return 0.0
    es = np.asarray(edge_sources, dtype=np.int64)
    a = np.abs(np.asarray(edge_alpha, dtype=np.float64))
    v = np.ones(M, dtype=np.float64) / np.sqrt(M)
    prev = 0.0
    for _ in range(power_iter):
        y = np.zeros(M, dtype=np.float64)
        np.add.at(y, et, a * v[es])               # y[target] = Σ α · v[source]
        nrm = float(np.linalg.norm(y))
        if nrm <= 1e-20:
            return 0.0
        v = y / nrm
        if abs(nrm - prev) / max(nrm, 1e-12) < 1e-6:
            break
        prev = nrm
    return float(prev)


def _append_sparse_dynamic_resp(parts, cand_idx: np.ndarray, combo_idx: np.ndarray, values: np.ndarray, K: int):
    """Append chunk-level sparse responsibility sums by dynamic combo."""
    if len(cand_idx) == 0:
        return
    flat = cand_idx.astype(np.int64, copy=False) * int(K) + combo_idx.astype(np.int64, copy=False)
    uniq, inv = np.unique(flat, return_inverse=True)
    sums = np.bincount(inv, weights=values).astype(np.float32, copy=False)
    combos = (uniq % int(K)).astype(np.int64, copy=False)
    rows = (uniq // int(K)).astype(np.int64, copy=False)
    for k in np.unique(combos):
        mask = combos == k
        parts[int(k)].append((rows[mask].astype(np.int32, copy=False), sums[mask]))


def _finalize_sparse_dynamic_resp(parts, C: int):
    """Consolidate chunk sparse responsibility parts into one sparse column per combo."""
    out = []
    for col_parts in parts:
        if not col_parts:
            out.append((np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)))
            continue
        rows = np.concatenate([p[0] for p in col_parts])
        vals = np.concatenate([p[1] for p in col_parts])
        if len(rows) == 0:
            out.append((np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)))
            continue
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        vals = vals[order]
        boundary = np.flatnonzero(rows[1:] != rows[:-1]) + 1
        starts = np.concatenate(([0], boundary))
        rows_u = rows[starts]
        vals_u = np.add.reduceat(vals, starts).astype(np.float32, copy=False)
        out.append((rows_u.astype(np.int64 if C > np.iinfo(np.int32).max else np.int32, copy=False), vals_u))
    return out


def _is_dynamic_exposure_coo(obj) -> bool:
    return isinstance(obj, dict) and obj.get("format") == "coo"


def _normalize_dynamic_exposure_coo(obj, C: int, K: int):
    shape = tuple(obj.get("shape", ()))
    if shape != (int(C), int(K)):
        raise ValueError(f"dynamic_exposure_2d shape {shape} != {(C, K)}")
    rows = np.asarray(obj.get("rows"))
    combos = np.asarray(obj.get("combos", obj.get("cols")))
    values = np.asarray(obj.get("values"))
    if rows.shape != combos.shape or rows.shape != values.shape:
        raise ValueError("dynamic_exposure_2d COO rows/combos/values must have the same shape")
    if rows.size:
        if int(rows.min()) < 0 or int(rows.max()) >= C:
            raise ValueError("dynamic_exposure_2d COO row index out of range")
        if int(combos.min()) < 0 or int(combos.max()) >= K:
            raise ValueError("dynamic_exposure_2d COO combo index out of range")
    if not np.issubdtype(rows.dtype, np.integer):
        rows = rows.astype(np.int64)
    if not np.issubdtype(combos.dtype, np.integer):
        combos = combos.astype(np.int64)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    return rows, combos, values


def _build_chunk_pair_arrays(
    times: np.ndarray,
    dims: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    history_window: float,
    max_history_events: int,
    time_slack: float = 0.0,
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

    When time_slack == 0, all arrays are fully vectorized and only earlier
    events are candidates. When time_slack > 0, signed dt candidates are drawn
    from [target_time - history_window, target_time + time_slack], excluding the
    target event itself and keeping the nearest max_history_events by timestamp
    distance.
    """
    time_slack = max(float(time_slack), 0.0)
    chunk_size = chunk_end - chunk_start
    target_event_ids = np.arange(chunk_start, chunk_end, dtype=np.int64)
    if time_slack > 0:
        # Vectorized signed-window build (no Python per-target loop). For each
        # target, candidates are the `max_hist` NEAREST by |dt| within
        # [tt - history_window, tt + time_slack], excluding self — matching the
        # stream's two-pointer nearest expansion. Bounded-memory trick: since
        # `times` is sorted, |dt| is V-shaped, so the nearest set is a subset of
        # [max(lo, target - max_hist), hi) — any past event outside the max_hist
        # most-recent is farther than all of them. So we only materialize that
        # bounded range, then keep the nearest max_hist per target.
        max_hist = max(int(max_history_events), 1)
        tt = times[target_event_ids]
        lo = np.searchsorted(times, tt - history_window, side="left")
        hi = np.searchsorted(times, tt + time_slack, side="right")
        start = np.maximum(lo, target_event_ids - max_hist)
        raw_counts = np.maximum(hi - start, 0).astype(np.int64)
        Praw = int(raw_counts.sum())
        if Praw == 0:
            empty_i64 = np.empty(0, dtype=np.int64)
            empty_f32 = np.empty(0, dtype=np.float32)
            return empty_i64, empty_i64, empty_f32, empty_i64, empty_i64, empty_i64, np.zeros(chunk_size, dtype=np.int64)
        offs = np.zeros(chunk_size + 1, dtype=np.int64)
        np.cumsum(raw_counts, out=offs[1:])
        ev_idx = np.repeat(np.arange(chunk_size, dtype=np.int32), raw_counts)
        within = np.arange(Praw, dtype=np.int32) - offs[ev_idx].astype(np.int32, copy=False)
        pair_source = start[ev_idx] + within
        pair_target = target_event_ids[ev_idx]
        # Exclude the target event itself.
        keep_self = pair_source != pair_target
        pair_source = pair_source[keep_self]
        pair_target = pair_target[keep_self]
        ev_idx = ev_idx[keep_self]
        abs_dt = np.abs(times[pair_target] - times[pair_source]).astype(np.float32)
        # Keep the nearest `max_hist` per target: sort by (target, |dt|), then
        # take within-group rank < max_hist.
        counts_excl = np.bincount(ev_idx, minlength=chunk_size)
        g_start = np.zeros(chunk_size, dtype=np.int64)
        np.cumsum(counts_excl[:-1], out=g_start[1:])
        order = np.lexsort((abs_dt, ev_idx))         # primary ev_idx, secondary |dt|
        rank = np.arange(order.size, dtype=np.int32) - g_start[ev_idx[order]].astype(np.int32, copy=False)
        sel = order[rank < max_hist]
        pair_source = pair_source[sel]
        pair_target = pair_target[sel]
        counts = np.bincount(ev_idx[sel], minlength=chunk_size).astype(np.int64)
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
            getattr(config, "time_slack", 0.0),
        )
        if pair_dt.size == 0:
            continue
        flat_uv = pair_target_dim.astype(np.int64) * M + pair_source_dim.astype(np.int64)
        pair_dt_eff, late_weight = _apply_time_slack(pair_dt, config)
        np.add.at(n_pair.ravel(), flat_uv, late_weight.astype(np.float64))
        if sum_dt is not None:
            np.add.at(sum_dt.ravel(), flat_uv, pair_dt_eff.astype(np.float64) * late_weight.astype(np.float64))
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
    if config.beta_mode == "per_edge":
        # β must be computed from n_pair BEFORE n_pair is mutated below.
        K_b = config.beta_prior_strength
        m_b = max(config.beta_prior_mean, _EPS)
        beta = ((n_pair + K_b) / ((sum_dt if sum_dt is not None else 0.0) + K_b / m_b)).astype(np.float32)
        np.clip(beta, config.beta_min, config.beta_max, out=beta)
    else:
        beta = np.float32(config.beta_shared_value)  # 0-d scalar, not (M, M)
    neg_int = _negative_penalty_integral(config)
    exposure_factor = 1.0 + (beta.astype(np.float64) if np.ndim(beta) else float(beta)) * neg_int
    denom = n_source[np.newaxis, :] * exposure_factor + K
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
            getattr(config, "time_slack", 0.0),
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
        pair_dt_eff, late_weight = _apply_time_slack(pair_dt, config)
        # Score: α · β · exp(-β · max(Δt,0)) · late_penalty(max(-Δt,0))
        score_pair = alpha_pair * beta_pair * np.exp(-beta_pair * pair_dt_eff) * late_weight
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
                p_ij.astype(np.float64) * pair_dt_eff.astype(np.float64),
            )
        mu_num_chunk = _segment_sum(p_self_chunk, target_dims_chunk, M)
        mu_num += mu_num_chunk
        log_likelihood += float(np.log(rate).sum())

    return p_self, alpha_num, beta_num_dt, mu_num, log_likelihood


def _log_likelihood_global(
    rate_term: float,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    horizon: float,
    config: MHPConfig,
    n_source: Optional[np.ndarray] = None,
) -> float:
    """LL ≈ Σ_i log rate_i − T · Σ_d μ_d − Σ_{u,v} α[u,v]·G_int.

    The third term uses the same source exposure as the α M-step:
    each source-type event contributes one positive-time kernel mass plus the
    optional negative-jitter mass β·∫late_penalty.
    """
    term2 = horizon * float(mu.sum())
    neg_int = _negative_penalty_integral(config)
    if neg_int <= 0:
        return rate_term - term2 - float(alpha.sum())
    if n_source is None:
        # Backwards-compatible fallback; callers in this module pass n_source.
        n_source = np.ones(alpha.shape[1], dtype=np.float64)
    else:
        n_source = np.asarray(n_source, dtype=np.float64)
    if np.ndim(beta) == 0:
        exposure = n_source * (1.0 + float(beta) * neg_int)
        term3 = float(np.dot(alpha.sum(axis=0, dtype=np.float64), exposure))
    else:
        # Avoid materializing another full MxM float64 exposure matrix.
        beta_arr = np.asarray(beta)
        term3 = 0.0
        row_chunk = 1024
        for start in range(0, alpha.shape[0], row_chunk):
            end = min(start + row_chunk, alpha.shape[0])
            factor = 1.0 + beta_arr[start:end].astype(np.float64) * neg_int
            term3 += float((alpha[start:end].astype(np.float64) * factor * n_source[np.newaxis, :]).sum())
    return rate_term - term2 - term3


def fit_mhp(
    events: EventCollection,
    config: MHPConfig,
    *,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    init_mu: Optional[np.ndarray] = None,
    iter_callback: Optional[Callable[[dict], None]] = None,
    best_callback: Optional[Callable[[MHPResult, dict], None]] = None,
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
    if getattr(config, "time_slack", 0.0) > 0 and config.beta_mode == "per_edge":
        raise NotImplementedError(
            "time_slack > 0 currently supports beta_mode='shared' only; "
            "per_edge beta needs a coupled beta/exposure M-step"
        )

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

        n_source = np.bincount(events.dims, minlength=M).astype(np.float64)

        # TRUE observed LL for the parameters used in this E-step. The M-step
        # below produces the NEXT parameters; their LL is evaluated by the next
        # iteration's E-step.
        ll = _log_likelihood_global(ll_term1, mu, alpha, beta, horizon, config, n_source)
        delta_rel = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf

        beta_is_scalar = np.ndim(beta) == 0
        is_best = ll > best_ll
        if is_best:
            best_ll = ll
            nz_t, nz_s = np.nonzero(alpha)
            best_edge_targets = nz_t.copy()
            best_edge_sources = nz_s.copy()
            best_edge_alpha = alpha[nz_t, nz_s].astype(np.float64)
            if beta_is_scalar:
                best_beta_scalar = float(beta)
                best_edge_beta = None
            else:
                best_beta_scalar = None
                best_edge_beta = beta[nz_t, nz_s].astype(np.float64)
            best_mu = mu.copy()
            best_p_self = p_self

        active_edges = int((alpha > 0).sum())
        # β stats — only meaningful on active edges (β=0 where α=0)
        if beta_is_scalar:
            bval = float(beta)
            beta_median_active = bval if active_edges else 0.0
            beta_max_active = bval if active_edges else 0.0
            beta_min_active = bval if active_edges else 0.0
        else:
            active_mask = alpha > 0
            if active_mask.any():
                beta_active = beta[active_mask]
                beta_median_active = float(np.median(beta_active))
                beta_max_active = float(beta_active.max())
                beta_min_active = float(beta_active.min())
            else:
                beta_median_active = 0.0
                beta_max_active = 0.0
                beta_min_active = 0.0

        # M-step
        mu_new = np.maximum(mu_num / max(horizon, _EPS), 0.05 / horizon)

        K = config.alpha_prior_strength
        m = config.alpha_prior_mean
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
        neg_int = _negative_penalty_integral(config)
        exposure_factor = 1.0 + (
            beta_new.astype(np.float64) if np.ndim(beta_new) else float(beta_new)
        ) * neg_int
        denom = n_source[np.newaxis, :] * exposure_factor + K

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

        t_iter_end = time.monotonic()
        iter_total = t_iter_end - t_iter_start
        iter_estep = t_estep_end - t_iter_start
        iter_mstep = t_iter_end - t_estep_end
        trace_entry = {
            "iter": it,
            "log_likelihood": float(ll),
            "delta_rel": float(delta_rel),
            "branching_rescaled": n_rescaled,
            "active_edges": active_edges,
            "mu_max": float(mu.max()),
            "mu_median": float(np.median(mu)),
            "alpha_max": float(alpha.max()),
            "alpha_median_active": float(np.median(alpha[alpha > 0])) if active_edges else 0.0,
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
        if is_best and best_callback is not None:
            if best_beta_scalar is not None:
                edge_beta_arr = np.full(len(best_edge_targets), best_beta_scalar, dtype=np.float64)
            else:
                edge_beta_arr = (
                    best_edge_beta if best_edge_beta is not None else np.zeros(len(best_edge_targets))
                )
            checkpoint_params = MHPParams.from_edges(
                M=M,
                mu=best_mu,
                edge_targets=best_edge_targets,
                edge_sources=best_edge_sources,
                edge_alpha=best_edge_alpha,
                edge_beta=edge_beta_arr,
                edge_threshold=config.edge_threshold,
                max_active_sources_per_dim=config.max_active_sources_per_dim,
                beta_shared=(config.beta_mode == "shared"),
            )
            best_callback(
                MHPResult(
                    params=checkpoint_params,
                    log_likelihood=best_ll,
                    iterations_run=len(trace),
                    converged=False,
                    trace=list(trace),
                    p_self=best_p_self,
                ),
                trace_entry,
            )
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


def _make_pair_scorer(params: MHPParams, config: Optional[MHPConfig] = None):
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
                if config is not None:
                    dt_eff, late_weight = _apply_time_slack(pair_dt[valid], config)
                else:
                    dt_eff = pair_dt[valid]
                    late_weight = 1.0
                pb = bucket_index_vec(dt_eff.astype(np.float64), bucket_edges)
                out[valid] = theta[ei[valid], pb] * late_weight
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
            if config is not None:
                dt, late_weight = _apply_time_slack(pair_dt[valid], config)
            else:
                dt = pair_dt[valid]
                late_weight = 1.0
            out[valid] = a * b * np.exp(-b * dt) * late_weight
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
    best_callback: Optional[Callable[[MHPResult, dict], None]] = None,
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
    fit_widths = widths.copy()
    if config.time_slack > 0:
        # Negative jitter maps to the dt=0 bucket, discounted by the late-parent
        # penalty; its integrated exposure is added to bucket 0's width.
        fit_widths[0] += _negative_penalty_integral(config)
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
            getattr(config, "time_slack", 0.0),
        )
        if pair_dt.size == 0:
            continue
        pe, valid = _edge_lookup(pair_tdim, pair_sdim)
        if not valid.any():
            continue
        pair_dt_eff, late_weight = _apply_time_slack(pair_dt[valid], config)
        pb = bucket_index_vec(pair_dt_eff.astype(np.float64), bucket_edges)
        flat = pe[valid].astype(np.int64) * B + pb
        np.add.at(resp_init.ravel(), flat, late_weight.astype(np.float64))
    theta = (resp_init + K * m) / (n_v_per_edge[:, None] * fit_widths[None, :] + K)
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
                getattr(config, "time_slack", 0.0),
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
                pair_dt_eff, late_weight = _apply_time_slack(pair_dt[valid], config)
                pb_valid = bucket_index_vec(pair_dt_eff.astype(np.float64), bucket_edges)
                pb[valid] = pb_valid
                score_pair[valid] = theta[pe[valid], pb_valid] * late_weight
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

        branching_eval = (theta * widths[None, :]).sum(axis=1)
        compensator_eval = (theta * fit_widths[None, :]).sum(axis=1)
        if config.time_slack > 0:
            term3_eval = float((n_v_per_edge * compensator_eval).sum())
        else:
            term3_eval = float(branching_eval.sum())
        ll = ll_term1 - horizon * float(mu.sum()) - term3_eval
        delta_rel = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf

        is_best = ll > best_ll
        if is_best:
            best_ll = ll
            best_theta = theta.copy()
            best_mu = mu.copy()
            best_p_self = p_self

        # M-step
        mu_new = np.maximum(mu_num / max(horizon, _EPS), 0.05 / horizon)
        theta_new = (resp + K * m) / (n_v_per_edge[:, None] * fit_widths[None, :] + K)

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

        trace_entry = {
            "iter": it,
            "log_likelihood": float(ll),
            "delta_rel": float(delta_rel),
            "branching_rescaled": n_rescaled,
            "active_edges": E,
            "mu_max": float(mu.max()),
            "branching_max": float(branching_eval.max()),
            "branching_median": float(np.median(branching_eval)),
            "p_self_mean": float(p_self.mean()),
            "iter_seconds": float(time.monotonic() - t_iter),
        }
        trace.append(trace_entry)
        if iter_callback is not None:
            iter_callback(trace_entry)
        if is_best and best_callback is not None:
            branching = (best_theta * widths[None, :]).sum(axis=1)
            checkpoint_params = MHPParams.from_edges(
                M=M,
                mu=best_mu,
                edge_targets=edge_targets,
                edge_sources=edge_sources,
                edge_alpha=branching,
                edge_beta=np.zeros(E, dtype=np.float64),
                edge_threshold=0.0,
                max_active_sources_per_dim=config.max_active_sources_per_dim,
                kernel_type="piecewise",
                edge_theta=best_theta,
                bucket_edges=tuple(config.bucket_edges),
            )
            best_callback(
                MHPResult(
                    params=checkpoint_params,
                    log_likelihood=best_ll,
                    iterations_run=len(trace),
                    converged=False,
                    trace=list(trace),
                    p_self=best_p_self,
                ),
                trace_entry,
            )
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


def fit_mhp_feature(
    events: EventCollection,
    config: MHPConfig,
    *,
    cand_targets: np.ndarray,
    cand_sources: np.ndarray,
    cand_phi: np.ndarray,
    feature_names: list,
    init_mu: Optional[np.ndarray] = None,
    w_prior_mean: Optional[np.ndarray] = None,
    l2: float = 1e-3,
    l2_normalize: bool = False,
    mu_phi: Optional[np.ndarray] = None,
    mu_feature_names: Optional[list] = None,
    cand_topo_score: Optional[np.ndarray] = None,
    topo_prior_boost: float = 0.0,
    src_combo: Optional[np.ndarray] = None,
    tgt_combo: Optional[np.ndarray] = None,
    dynamic_combo_bits: Optional[np.ndarray] = None,
    dynamic_exposure_2d: Optional[np.ndarray] = None,
    dynamic_feature_names: Optional[list] = None,
    iter_callback: Optional[Callable[[dict], None]] = None,
    best_callback: Optional[Callable[[MHPResult, dict], None]] = None,
) -> MHPResult:
    """Feature-weighted MAP EM: α on a fixed CANDIDATE pair set is a log-linear
    function of pair features, α_c = softplus(w · φ_c). EM alternates:

      E-step  (same windowed/chunked machinery as the other fits) — uses the
              current α_c on candidate pairs to compute parent responsibilities.
      M-step  — aggregate responsibility N_c and exposure E_c=n_{source} per
              candidate, then fit w by gradient ascent (fit_weights_mstep);
              μ closed-form as usual.

    The learned FeatureKernel is returned so inference can compute α for pairs
    unseen in training (new devices) from their features — the inductive part.

    cand_targets/cand_sources : (C,) int64   the modeled (target,source) type pairs
    cand_phi                  : (C, F) float64 their feature matrix (row-aligned)
    """
    from .feature_kernel import FeatureKernel, softplus

    M = events.M
    N = events.n
    horizon = events.T
    if getattr(config, "time_slack", 0.0) > 0 and config.beta_mode != "shared":
        raise NotImplementedError(
            "feature time_slack > 0 currently supports beta_mode='shared' only"
        )
    beta_scalar = np.float32(config.beta_shared_value)

    cand_targets = np.asarray(cand_targets, dtype=np.int64)
    cand_sources = np.asarray(cand_sources, dtype=np.int64)
    cand_phi = np.asarray(cand_phi, dtype=np.float64)
    C = len(cand_targets)
    if C == 0:
        raise ValueError("feature mode requires a non-empty candidate pair set")
    # The caller may hand the GiB-sized source_target exposure in a 1-element
    # holder list, transferring ownership so this function can RELEASE the dense
    # array (after building the sparse COO) without the caller's frame pinning
    # it for the whole EM loop. None / a bare array pass through unchanged.
    if isinstance(dynamic_exposure_2d, list):
        dynamic_exposure_2d = dynamic_exposure_2d.pop()
    # Sort candidates by key so the E-step can binary-search. Large training
    # callers may pre-sort to avoid holding both unsorted and sorted GB-sized
    # feature/exposure tables during EM startup.
    keys = cand_targets * M + cand_sources
    order = None
    if C > 1 and not bool(np.all(keys[1:] >= keys[:-1])):
        order = np.argsort(keys, kind="stable")
        cand_targets = cand_targets[order]
        cand_sources = cand_sources[order]
        cand_phi = cand_phi[order]
        if dynamic_exposure_2d is not None:
            if _is_dynamic_exposure_coo(dynamic_exposure_2d):
                inv_order = np.empty(C, dtype=np.int64)
                inv_order[order] = np.arange(C, dtype=np.int64)
                row_dtype = np.int64 if C > np.iinfo(np.int32).max else np.int32
                dynamic_exposure_2d = dict(dynamic_exposure_2d)
                dynamic_exposure_2d["rows"] = inv_order[
                    np.asarray(dynamic_exposure_2d["rows"], dtype=np.int64)
                ].astype(row_dtype, copy=False)
            else:
                dynamic_exposure_2d = np.asarray(dynamic_exposure_2d)[order]
        keys = keys[order]
    cand_keys = keys

    n_source = np.bincount(events.dims, minlength=M).astype(np.float64)
    exposure = n_source[cand_sources]                      # E_c
    if config.time_slack > 0:
        exposure = exposure * (1.0 + float(beta_scalar) * _negative_penalty_integral(config))

    # Topology PRIOR as pseudo-observations (mirrors device-mode
    # topology_prior_boost). For a topology-related candidate c we inject a
    # Gamma-style prior on α_c by augmenting its sufficient statistics:
    #   N_c += K·boost·score_c   (pseudo excitation count)
    #   E_c += K                 (pseudo exposure)
    # so the per-row MAP target becomes (N_c + K·boost·score)/(E_c + K) — at
    # zero co-occurrence α_c → boost·score_c (a zero-shot topology edge), and
    # for rare sources (small E_c) the prior dominates, exactly where data is
    # missing; for data-rich pairs the likelihood washes it out. The prior
    # steers the weight-fit only; the reported LL below uses the RAW exposure.
    prior_num = np.zeros(C, dtype=np.float64)
    prior_exp = np.zeros(C, dtype=np.float64)
    if topo_prior_boost > 0.0 and cand_topo_score is not None:
        topo_score_sorted = np.asarray(cand_topo_score, dtype=np.float64)
        if order is not None:
            topo_score_sorted = topo_score_sorted[order]
        K_topo = float(config.alpha_prior_strength)
        prior_num = K_topo * float(topo_prior_boost) * np.maximum(topo_score_sorted, 0.0)
        prior_exp = np.where(topo_score_sorted > 0.0, K_topo, 0.0)

    F = cand_phi.shape[1]
    w = np.zeros(F) if w_prior_mean is None else np.asarray(w_prior_mean, dtype=np.float64).copy()

    # Seed the α BIAS (feature index 0) so the initial model is SUBCRITICAL.
    # With all non-bias weights at 0, α_c = softplus(w[0]) uniformly. Left at 0,
    # softplus(0)=0.693 on every candidate → with a large candidate set the
    # initial branching ratio Σ_c E_c·α/N is enormous, and the first M-step sees
    # a ~ΣE_c-sized integral-term gradient that overshoots α (and then μ) to the
    # degenerate all-zero fixed point. Choose w[0] so the initial global
    # branching ratio Σ_c E_c·α / N ≈ 0.5 (symmetric to the μ-bias seed below).
    if w_prior_mean is None:
        total_exposure = float(exposure.sum())
        if total_exposure > 0.0:
            alpha_init = min(max(0.5 * N / total_exposure, 1e-6), 0.5)
            w[0] = float(np.log(np.expm1(alpha_init)))   # inverse softplus

    # Dynamic (stateful) α: α on (candidate c, combo k) =
    # softplus(cand_phi[c]·w_static + combo_bits[k]·w_dyn). For source-only,
    # combo is the source event's 3-bit mark. For source_target B-fast, combo is
    # source_combo*8 + target_pre_combo: the E-step uses the target event's
    # read-before-write state (time-slack safe), while the compensator/exposure
    # is precomputed with the target state sampled at source_ts.
    use_dynamic = src_combo is not None and dynamic_combo_bits is not None
    if use_dynamic:
        dynamic_combo_bits = np.asarray(dynamic_combo_bits, dtype=np.float64)  # (K, D)
        K_combo, D_dyn = dynamic_combo_bits.shape
        src_combo = np.asarray(src_combo, dtype=np.int64).reshape(-1)
        if len(src_combo) != N:
            raise ValueError("src_combo must be aligned to events")
        use_target_dynamic = tgt_combo is not None
        if use_target_dynamic:
            tgt_combo = np.asarray(tgt_combo, dtype=np.int64).reshape(-1)
            if len(tgt_combo) != N:
                raise ValueError("tgt_combo must be aligned to events")
            if K_combo != 64:
                raise ValueError("source_target dynamic mode requires 64 combo rows")
            if dynamic_exposure_2d is None:
                raise ValueError("source_target dynamic mode requires dynamic_exposure_2d")
        exposure_2d = None
        exposure_combo_idx = None
        e_coo_real = None       # real exposure COO (no prior) — for the LL compensator
        e_coo_fixed = None      # real exposure + combo-0 topology prior — for the M-step
        slack_scale = 1.0 + float(beta_scalar) * _negative_penalty_integral(config)
        if dynamic_exposure_2d is not None and _is_dynamic_exposure_coo(dynamic_exposure_2d):
            if not use_target_dynamic:
                raise ValueError("COO dynamic_exposure_2d is only supported for source_target dynamic mode")
            _er, _ec, _ev = _normalize_dynamic_exposure_coo(dynamic_exposure_2d, C, K_combo)
            if config.time_slack > 0:
                _ev = _ev.astype(np.float32, copy=False)
                _ev *= slack_scale
            exposure_combo_idx = np.unique(_ec) if _ec.size else np.zeros(0, dtype=np.int64)
            e_coo_real = (_er, _ec, _ev)
            dynamic_exposure_2d = None
        elif dynamic_exposure_2d is not None:
            exposure_2d = np.asarray(dynamic_exposure_2d)
            if exposure_2d.shape != (C, K_combo):
                raise ValueError(
                    f"dynamic_exposure_2d shape {exposure_2d.shape} != {(C, K_combo)}"
                )
        else:
            # n_source_by_combo[v, k] = #events of type v with source-mark combo k.
            n_src_by_combo = np.zeros((M, K_combo), dtype=np.float64)
            np.add.at(n_src_by_combo, (events.dims, src_combo), 1.0)
            exposure_2d = n_src_by_combo[cand_sources]            # (C, K) E_{c,k}
        if exposure_2d is not None and config.time_slack > 0:
            if np.issubdtype(exposure_2d.dtype, np.floating):
                exposure_2d *= slack_scale
            else:
                exposure_2d = exposure_2d.astype(np.float32, copy=False)
                exposure_2d *= slack_scale
        if exposure_combo_idx is None:
            exposure_combo_idx = np.flatnonzero(exposure_2d.sum(axis=0) > 0.0)
        dynamic_n0_extra = None
        dynamic_e0_extra = None
        # Topology pseudo-count prior: attach to the baseline combo (k=0, no
        # active alarms) — it is a prior on edge existence, state-independent.
        if topo_prior_boost > 0.0 and cand_topo_score is not None:
            dynamic_n0_extra = prior_num
            dynamic_e0_extra = prior_exp
        if use_target_dynamic:
            if e_coo_real is None:
                # Dense fallback: convert once and release the dense (C, K)
                # exposure before the EM loop.
                _er, _ec = np.nonzero(exposure_2d)
                _ev = exposure_2d[_er, _ec].astype(np.float32, copy=False)
                _er = _er.astype(np.int64)
                _ec = _ec.astype(np.int64)
                exposure_2d = None
                dynamic_exposure_2d = None
                e_coo_real = (_er, _ec, _ev)
            else:
                _er, _ec, _ev = e_coo_real
            e_coo_real = (_er, _ec, _ev)            # for the LL compensator
            if dynamic_e0_extra is not None:        # fold combo-0 topology prior for the M-step
                _xr = np.flatnonzero(dynamic_e0_extra).astype(_er.dtype, copy=False)
                e_coo_fixed = (
                    np.concatenate([_er, _xr]),
                    np.concatenate([_ec, np.zeros(len(_xr), dtype=_ec.dtype)]),
                    np.concatenate([_ev, np.asarray(dynamic_e0_extra, dtype=np.float64)[_xr]]),
                )
            else:
                e_coo_fixed = e_coo_real
        w = np.concatenate([w, np.zeros(D_dyn)])                  # static ⊕ dynamic

    # μ is INDUCTIVE in feature mode: μ(u) = softplus(w_μ · ψ(u)), a log-linear
    # function of the type's OWN features ψ (alarm_type, ne_type, vendor, domain)
    # — symmetric to α. Learned by the same gradient M-step. A brand-new device
    # gets μ from its ψ; seen and new devices are treated identically (no
    # per-device frequency memorization). Falls back to per-type closed-form μ
    # if no mu_phi is supplied.
    use_mu_features = mu_phi is not None
    if use_mu_features:
        mu_phi = np.asarray(mu_phi, dtype=np.float64)
        F_mu = mu_phi.shape[1]
        w_mu = np.zeros(F_mu)
        mu_exposure = np.full(M, max(horizon, _EPS), dtype=np.float64)   # ∫ over T per type

    if init_mu is None:
        mu = _compute_mu_initial(events, horizon, config)
    else:
        mu = np.asarray(init_mu, dtype=np.float64).reshape(-1).copy()

    if use_mu_features:
        # Seed the μ bias so the initial parameterized μ matches the heuristic
        # median rate (inverse-softplus), avoiding a high-μ first iteration.
        m0 = float(np.median(mu)) if len(mu) else 0.01
        m0 = max(m0, 1e-6)
        w_mu[0] = float(np.log(np.expm1(m0)))   # inverse softplus of m0
        mu = softplus(mu_phi @ w_mu)

    requested_chunk_size = max(int(config.chunk_size), 1)
    chunk_size = requested_chunk_size
    history_window = config.history_window
    max_history_events = max(int(config.max_history_events), 1)
    if getattr(config, "time_slack", 0.0) > 0.0:
        # With time slack, one target can consider up to max_history_events
        # recent past events plus nearby future events. Large chunks therefore
        # create tens of millions of temporary pair rows before candidate-edge
        # filtering. Keep the raw pair builder in a bounded memory regime.
        pair_row_cap = 3_000_000
        safe_chunk = max(1_000, int(pair_row_cap // max(max_history_events + 1, 1)))
        chunk_size = min(chunk_size, safe_chunk)

    t_total_start = time.monotonic()
    if config.verbose:
        chunk_msg = f"chunk_size={chunk_size}"
        if chunk_size != requested_chunk_size:
            chunk_msg += f" (auto-reduced from {requested_chunk_size} for time_slack)"
        print(
            f"[mhp-feat] events={N}, candidate pairs={C}, features={F}, "
            f"mu_features={(mu_phi.shape[1] if use_mu_features else 0)}, "
            f"{chunk_msg}",
            flush=True,
        )

    trace: list[dict] = []
    best_ll = -np.inf
    best_w = w.copy()
    best_w_mu = w_mu.copy() if use_mu_features else None
    best_mu = mu.copy()
    best_p_self = np.zeros(N, dtype=np.float64)
    converged = False
    prev_ll = -np.inf

    # OPT-IN α-ridge normalization (l2_normalize). An unscaled ridge λ·‖w‖² (a
    # handful of weights) is negligible against the α data term, so λ never bites.
    # The data term Σ_c [N_c·logα − E_c·α] has magnitude ≈ ΣN_c = Σ E_c·α_c ≈ N
    # (the event count) at the optimum — NOT the exposure mass ΣE, which is huge
    # because most of it sits on candidates whose α≈0. So we scale the ridge by N:
    # --feature-l2 becomes a data-size-independent shrinkage (≈ a fraction of the
    # data term) where λ≈0.01–0.1 actually bites and controls ρ. (Scaling by ΣE
    # instead over-shrinks by ~ΣE/N, often 1e4–1e5×, collapsing every α below
    # edge_threshold → empty active set → ρ reads 0.) OFF by default → unchanged.
    l2_alpha = float(l2)
    if l2_normalize:
        _data_mass = float(N)
        l2_alpha = float(l2) * max(_data_mass, 1.0)
        if config.verbose:
            print(
                f"[mhp-feat] l2_normalize=ON: α-ridge λ·N={l2_alpha:.4g} "
                f"(λ={l2:g}, N={_data_mass:.4g}); μ-ridge stays λ={l2:g}",
                flush=True,
            )

    for it in range(config.max_iters):
        t_iter = time.monotonic()
        if use_dynamic:
            z_static_c = cand_phi @ w[:F]                  # (C,)
            z_dyn_k = dynamic_combo_bits @ w[F:]           # (K,)
            # Baseline α (combo 0 = no active alarms) for diagnostics / materialization.
            alpha_cand = softplus(z_static_c + z_dyn_k[0])
            use_sparse_resp = bool(use_target_dynamic)
            if use_sparse_resp:
                n_resp2d_parts = [[] for _ in range(K_combo)]
                n_resp2d = None
            else:
                n_resp2d = np.zeros((C, K_combo), dtype=np.float64)
        else:
            alpha_cand = softplus(cand_phi @ w)            # (C,) current amplitudes
            n_resp = np.zeros(C, dtype=np.float64)         # N_c

        p_self = np.zeros(N, dtype=np.float64)
        mu_num = np.zeros(M, dtype=np.float64)
        ll_term1 = 0.0

        # Intra-iteration progress: the E-step scans ~N/chunk_size chunks, which
        # can take minutes on large data — print a throttled heartbeat so a long
        # iteration doesn't look hung.
        n_chunks = (N + chunk_size - 1) // chunk_size
        _estep_t0 = time.monotonic()
        _last_beat = _estep_t0
        for ci, chunk_start in enumerate(range(0, N, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, N)
            csize = chunk_end - chunk_start
            if config.verbose and n_chunks > 1:
                _now = time.monotonic()
                if _now - _last_beat >= 10.0:
                    rate = chunk_end / max(_now - _estep_t0, 1e-9)
                    print(
                        f"[mhp-feat]   iter={it:3d} E-step chunk {ci + 1}/{n_chunks} "
                        f"({chunk_end}/{N} events, {rate:.0f} evt/s, "
                        f"{_fmt_secs(_now - _estep_t0)})",
                        flush=True,
                    )
                    _last_beat = _now
            tdims_chunk = events.dims[chunk_start:chunk_end]
            mu_chunk = mu[tdims_chunk]
            (_, pair_source, pair_dt, pair_tdim, pair_sdim, pair_tlocal, _) = _build_chunk_pair_arrays(
                events.times, events.dims, chunk_start, chunk_end,
                history_window, max_history_events,
                getattr(config, "time_slack", 0.0),
            )
            if pair_dt.size == 0:
                rate = np.maximum(mu_chunk, _EPS)
                p_self[chunk_start:chunk_end] = 1.0
                mu_num += _segment_sum(np.ones(csize), tdims_chunk, M)
                ll_term1 += float(np.log(rate).sum())
                continue
            # Map each pair to its candidate index (binary search). Negative dt
            # candidates are possible only within time_slack and are discounted.
            pk = pair_tdim.astype(np.int64) * M + pair_sdim.astype(np.int64)
            idx = np.minimum(np.searchsorted(cand_keys, pk), C - 1)
            valid = cand_keys[idx] == pk
            score_pair = np.zeros(pair_dt.shape, dtype=np.float64)
            combo_v = None
            if valid.any():
                vi = idx[valid]
                b = float(beta_scalar)
                pair_dt_eff, late_weight = _apply_time_slack(pair_dt[valid], config)
                if use_dynamic:
                    if use_target_dynamic:
                        # B-fast + time-slack-safe: target mark is the target
                        # event's pre-state, so a late parent never sees the
                        # target event's own raise.
                        target_global = chunk_start + pair_tlocal[valid]
                        combo_v = src_combo[pair_source[valid]] * 8 + tgt_combo[target_global]
                    else:
                        # Source-only: per-pair α uses the source event's mark
                        # combo at fire time.
                        combo_v = src_combo[pair_source[valid]]
                    a_pair = softplus(z_static_c[vi] + z_dyn_k[combo_v])
                    score_pair[valid] = a_pair * b * np.exp(-b * pair_dt_eff) * late_weight
                else:
                    score_pair[valid] = alpha_cand[vi] * b * np.exp(-b * pair_dt_eff) * late_weight
            sum_score = _segment_sum(score_pair, pair_tlocal, csize)
            rate = np.maximum(mu_chunk + sum_score, _EPS)
            p_self_chunk = mu_chunk / rate
            p_self[chunk_start:chunk_end] = p_self_chunk
            p_ij = score_pair / rate[pair_tlocal]
            if valid.any():
                if use_dynamic:
                    if use_sparse_resp:
                        _append_sparse_dynamic_resp(
                            n_resp2d_parts,
                            vi,
                            combo_v,
                            p_ij[valid],
                            K_combo,
                        )
                    else:
                        np.add.at(n_resp2d.reshape(-1), vi * K_combo + combo_v, p_ij[valid])
                else:
                    np.add.at(n_resp, idx[valid], p_ij[valid])
            mu_num += _segment_sum(p_self_chunk, tdims_chunk, M)
            ll_term1 += float(np.log(rate).sum())

        estep_seconds = time.monotonic() - _estep_t0
        if config.verbose and n_chunks > 1:
            sparse_msg = ""
            if use_dynamic and use_sparse_resp:
                sparse_parts = sum(len(col) for col in n_resp2d_parts)
                sparse_msg = f", sparse_resp_parts={sparse_parts}"
            print(
                f"[mhp-feat]   iter={it:3d} E-step done in {_fmt_secs(estep_seconds)}, "
                f"fitting weights (M-step){sparse_msg} ...",
                flush=True,
            )
        # TRUE observed LL for the parameters used in this E-step. The M-step
        # below produces the NEXT parameters; their LL is evaluated by the next
        # iteration's E-step, avoiding a second full pass over the event stream.
        if use_dynamic:
            if use_target_dynamic:
                _er, _ec, _ev = e_coo_real
                compensator_eval = float(np.sum(_ev * softplus(z_static_c[_er] + z_dyn_k[_ec])))
            else:
                compensator_eval = sum(
                    float((exposure_2d[:, k] * softplus(z_static_c + z_dyn_k[k])).sum())
                    for k in exposure_combo_idx
                )
        else:
            compensator_eval = float((alpha_cand * exposure).sum())
        ll_eval = ll_term1 - horizon * float(mu.sum()) - compensator_eval
        delta_rel = abs(ll_eval - prev_ll) / max(abs(prev_ll), 1.0) if it > 0 else np.inf
        is_best = ll_eval > best_ll
        if is_best:
            best_ll = ll_eval
            best_w = w.copy()
            best_mu = mu.copy()
            best_w_mu = w_mu.copy() if use_mu_features else None
            best_p_self = p_self

        active_mask = alpha_cand > config.edge_threshold
        active_eval = int(active_mask.sum())
        # Spectral radius ρ of the active α matrix (combo-0 baseline) — the
        # stationarity indicator. Reported each iter so over-excitation (ρ≥1)
        # is visible during training, not only at the end.
        _ai = np.flatnonzero(active_mask)
        rho_eval = (
            _spectral_radius_edges(cand_targets[_ai], cand_sources[_ai], alpha_cand[_ai], M)
            if _ai.size else 0.0
        )
        entry = {
            "iter": it,
            "log_likelihood": float(ll_eval),
            "delta_rel": float(delta_rel),
            "active_edges": active_eval,
            "candidate_pairs": C,
            "mu_max": float(mu.max()),
            "alpha_max": float(alpha_cand.max()),
            "alpha_median": float(np.median(alpha_cand)),
            "spectral_radius": float(rho_eval),
            "p_self_mean": float(p_self.mean()),
            "estep_seconds": float(estep_seconds),
            "mstep_seconds": 0.0,
            "iter_seconds": 0.0,
        }

        # M-step. α: gradient ascent on Σ[N_c log α_c − E_c α_c]. μ: same
        # gradient optimizer on the symmetric Σ_u[S_u log μ_u − T·μ_u] when
        # parameterized (mu_phi), else per-type closed form.
        from .feature_kernel import (
            fit_weights_mstep, fit_dynamic_weights_mstep, fit_dynamic_weights_mstep_coo,
        )

        _MSTEP_MAX = 50          # max inner gradient-ascent iters (usually fewer)
        _ms_t0 = time.monotonic()
        _ms_last = [0]           # records the last inner iter actually run
        if use_dynamic:
            # Bucketed M-step over (candidate, source-mark combo). Topology prior
            # is passed as combo-0 extra vectors to avoid copying the whole
            # (candidate, combo) table.
            if use_sparse_resp:
                _final_t0 = time.monotonic()
                n2d = _finalize_sparse_dynamic_resp(n_resp2d_parts, C)
                if config.verbose:
                    nnz_resp = sum(len(idx_col) for idx_col, _ in n2d)
                    print(
                        f"[mhp-feat]   iter={it:3d} sparse responsibility "
                        f"nnz={nnz_resp} finalized in {_fmt_secs(time.monotonic() - _final_t0)}",
                        flush=True,
                    )
            else:
                n2d = n_resp2d
            # Heartbeat for the dynamic M-step (it can take tens of seconds at
            # large C — print throttled so it never looks hung). It converges
            # early (||g||→0), so the inner count is usually well below the cap.
            _ms_beat = [_ms_t0]

            def _mstep_progress(mi, q, gn, _it=it, _t0=_ms_t0, _beat=_ms_beat,
                                _last=_ms_last, _mx=_MSTEP_MAX):
                _last[0] = mi
                if not config.verbose or C < 200_000:
                    return
                now = time.monotonic()
                if now - _beat[0] >= 8.0:
                    print(
                        f"[mhp-feat]   iter={_it:3d} M-step inner {mi}/{_mx} "
                        f"(Q={q:.1f}, |g|={gn:.2e}, {_fmt_secs(now - _t0)})",
                        flush=True,
                    )
                    _beat[0] = now

            if use_target_dynamic:
                # Flatten the sparse per-combo responsibility into one flat COO
                # and fold the combo-0 prior; M-step then touches only nonzero
                # (candidate, combo) buckets instead of the dense (C, K) table.
                _nr_parts, _nc_parts, _nv_parts = [], [], []
                _row_dtype = np.int64 if C > np.iinfo(np.int32).max else np.int32
                for k, (idx, val) in enumerate(n2d):
                    if len(idx):
                        _nr_parts.append(np.asarray(idx, dtype=_row_dtype))
                        _nc_parts.append(np.full(len(idx), k, dtype=np.uint8))
                        _nv_parts.append(np.asarray(val, dtype=np.float32))
                if dynamic_n0_extra is not None:
                    _xr = np.flatnonzero(dynamic_n0_extra)
                    _nr_parts.append(_xr.astype(_row_dtype, copy=False))
                    _nc_parts.append(np.zeros(len(_xr), dtype=np.uint8))
                    _nv_parts.append(np.asarray(dynamic_n0_extra, dtype=np.float32)[_xr])
                n_coo = (
                    np.concatenate(_nr_parts) if _nr_parts else np.zeros(0, _row_dtype),
                    np.concatenate(_nc_parts) if _nc_parts else np.zeros(0, np.uint8),
                    np.concatenate(_nv_parts) if _nv_parts else np.zeros(0, np.float32),
                )
                w_new = fit_dynamic_weights_mstep_coo(
                    cand_phi, dynamic_combo_bits, n_coo, e_coo_fixed, w,
                    l2=l2_alpha, w_prior_mean=w_prior_mean, max_iter=_MSTEP_MAX,
                    progress=_mstep_progress,
                )
            else:
                w_new = fit_dynamic_weights_mstep(
                    cand_phi, dynamic_combo_bits, n2d, exposure_2d, w,
                    l2=l2_alpha, w_prior_mean=w_prior_mean, max_iter=_MSTEP_MAX,
                    progress=_mstep_progress,
                    n0_extra=dynamic_n0_extra,
                    e0_extra=dynamic_e0_extra,
                )
            # Baseline α (combo 0) for diagnostics / materialization.
            alpha_new = softplus(cand_phi @ w_new[:F])
        else:
            # Topology prior enters here as pseudo-observations on the candidate
            # statistics (no-op when prior_num/prior_exp are all zero).
            w_new = fit_weights_mstep(
                cand_phi, n_resp + prior_num, exposure + prior_exp, w,
                l2=l2_alpha, w_prior_mean=w_prior_mean, max_iter=_MSTEP_MAX
            )
            alpha_new = softplus(cand_phi @ w_new)
        if use_mu_features:
            w_mu_new = fit_weights_mstep(
                mu_phi, mu_num, mu_exposure, w_mu, l2=l2, max_iter=50
            )
            mu_new = softplus(mu_phi @ w_mu_new)
        else:
            w_mu_new = None
            mu_new = np.maximum(mu_num / max(horizon, _EPS), 0.05 / horizon)

        w = w_new
        mu = mu_new
        if use_mu_features:
            w_mu = w_mu_new
        mstep_seconds = time.monotonic() - _ms_t0
        entry["mstep_seconds"] = float(mstep_seconds)
        entry["iter_seconds"] = float(time.monotonic() - t_iter)
        trace.append(entry)
        if iter_callback is not None:
            iter_callback(entry)
        if is_best and best_callback is not None:
            kernel_names = list(feature_names)
            if use_dynamic:
                kernel_names = kernel_names + list(dynamic_feature_names or [])
            checkpoint_kernel = FeatureKernel(weights=best_w, feature_names=kernel_names, l2=l2_alpha)
            checkpoint_mu_kernel = (
                FeatureKernel(weights=best_w_mu, feature_names=list(mu_feature_names or []), l2=l2)
                if use_mu_features and best_w_mu is not None
                else None
            )
            if use_dynamic:
                alpha_checkpoint = softplus(cand_phi @ best_w[:F])
            else:
                alpha_checkpoint = checkpoint_kernel.alpha(cand_phi)
            keep = alpha_checkpoint > config.edge_threshold
            checkpoint_params = MHPParams.from_edges(
                M=M,
                mu=best_mu,
                edge_targets=cand_targets[keep],
                edge_sources=cand_sources[keep],
                edge_alpha=alpha_checkpoint[keep],
                edge_beta=np.full(int(keep.sum()), float(beta_scalar), dtype=np.float64),
                edge_threshold=config.edge_threshold,
                max_active_sources_per_dim=config.max_active_sources_per_dim,
                beta_shared=True,
            )
            _stop_signal = best_callback(
                MHPResult(
                    params=checkpoint_params,
                    log_likelihood=best_ll,
                    iterations_run=len(trace),
                    converged=False,
                    trace=list(trace),
                    p_self=best_p_self,
                    feature_kernel=checkpoint_kernel,
                    mu_kernel=checkpoint_mu_kernel,
                ),
                entry,
            )
            if _stop_signal:
                # best_callback (held-out LL tracker) asked to stop: val LL has
                # plateaued. The val-optimal snapshot is already captured caller-
                # side, so break here rather than burn iters overfitting train.
                if config.verbose:
                    print(
                        f"[mhp-feat] early stop at iter {it} (held-out LL plateaued)",
                        flush=True,
                    )
                break
        if config.verbose and n_chunks > 1:
            _iters_msg = f" ({_ms_last[0] + 1} inner iters)" if use_dynamic else ""
            print(
                f"[mhp-feat]   iter={it:3d} M-step done in "
                f"{_fmt_secs(mstep_seconds)}{_iters_msg}",
                flush=True,
            )
        if config.verbose and (it % max(config.log_every, 1) == 0 or it == config.max_iters - 1):
            print(
                f"[mhp-feat] iter={it:3d} ll={ll_eval:.2f} Δ={delta_rel:.2e} "
                f"active(>thr)={active_eval}/{C} "
                f"α.median={entry['alpha_median']:.4f} α.max={entry['alpha_max']:.4f} "
                f"ρ={entry['spectral_radius']:.3f} "
                f"μ.max={entry['mu_max']:.4f} p_self.mean={entry['p_self_mean']:.3f} "
                f"t={_fmt_secs(entry['iter_seconds'])} "
                f"(E={_fmt_secs(entry['estep_seconds'])} M={_fmt_secs(entry['mstep_seconds'])})",
                flush=True,
            )
        if it > 0 and delta_rel < config.tol:
            converged = True
            if config.verbose:
                print(f"[mhp-feat] converged at iter {it} (Δrel={delta_rel:.2e})", flush=True)
            break
        prev_ll = ll_eval

    # In dynamic mode the kernel carries F static ⊕ D dynamic weights; its
    # feature_names are extended so inference can append the live mark bits to φ.
    kernel_names = list(feature_names)
    if use_dynamic:
        kernel_names = kernel_names + list(dynamic_feature_names or [])
    kernel = FeatureKernel(weights=best_w, feature_names=kernel_names, l2=l2_alpha)
    mu_kernel = (
        FeatureKernel(weights=best_w_mu, feature_names=list(mu_feature_names or []), l2=l2)
        if use_mu_features and best_w_mu is not None
        else None
    )
    # Materialize the final sparse α on candidate pairs (for artifact / device-
    # style consumers); inference can instead recompute α from the kernel. In
    # dynamic mode the materialized α is the BASELINE (combo 0, no active alarms);
    # dynamic boosts are applied live at inference.
    if use_dynamic:
        alpha_final = softplus(cand_phi @ best_w[:F])
    else:
        alpha_final = kernel.alpha(cand_phi)
    keep = alpha_final > config.edge_threshold
    final_params = MHPParams.from_edges(
        M=M,
        mu=best_mu,
        edge_targets=cand_targets[keep],
        edge_sources=cand_sources[keep],
        edge_alpha=alpha_final[keep],
        edge_beta=np.full(int(keep.sum()), float(beta_scalar), dtype=np.float64),
        edge_threshold=config.edge_threshold,
        max_active_sources_per_dim=config.max_active_sources_per_dim,
        beta_shared=True,
    )
    if config.verbose:
        print(
            f"[mhp-feat] fit complete: iters={len(trace)} converged={converged} "
            f"active_edges={int(keep.sum())}/{C} total={_fmt_secs(time.monotonic()-t_total_start)}",
            flush=True,
        )
    return MHPResult(
        params=final_params,
        log_likelihood=best_ll,
        iterations_run=len(trace),
        converged=converged,
        trace=trace,
        p_self=best_p_self,
        feature_kernel=kernel,
        mu_kernel=mu_kernel,
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
    score_fn = _make_pair_scorer(params, cfg)
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
            getattr(cfg, "time_slack", 0.0),
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
        seen = {}
        while cascade[cur] == -1:
            if cur in seen:
                # A small time-slack window can produce parent cycles (e.g. two
                # near-simultaneous events choose each other). Treat the whole
                # cycle as one root component instead of assuming every chain
                # reaches a self-parent.
                break
            seen[cur] = len(path)
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
    score_fn = _make_pair_scorer(params, cfg)

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
            getattr(cfg, "time_slack", 0.0),
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
    # piecewise is Σ positive-time branching ratio per edge). With time_slack,
    # add the matching negative-jitter exposure so held-out LL uses the same
    # scoring surface as training.
    term2 = events.T * float(params.mu.sum())
    n_source = np.bincount(dims, minlength=params.M).astype(np.float64)
    if not len(params.edge_alpha):
        term3 = 0.0
    elif getattr(cfg, "time_slack", 0.0) > 0 and params.kernel_type == "piecewise":
        neg_int = _negative_penalty_integral(cfg)
        per_edge = params.edge_alpha.astype(np.float64)
        if params.edge_theta is not None and params.edge_theta.shape[1] > 0:
            per_edge = per_edge + params.edge_theta[:, 0].astype(np.float64) * neg_int
        term3 = float((n_source[params.edge_sources] * per_edge).sum())
    elif getattr(cfg, "time_slack", 0.0) > 0:
        neg_int = _negative_penalty_integral(cfg)
        factor = 1.0 + np.asarray(params.edge_beta, dtype=np.float64) * neg_int
        term3 = float((n_source[params.edge_sources] * params.edge_alpha * factor).sum())
    else:
        term3 = float(params.edge_alpha.sum())
    return rate_term - term2 - term3
