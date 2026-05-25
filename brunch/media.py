"""MEDIA inference (paper §4): block Gibbs over event links + cluster links.

Simplification w.r.t. the paper
-------------------------------
The paper presents Metropolis-Hastings with four cases (split / split+merge /
merge / no-change) and worked Hastings ratios (Eq. 4.5–4.10). We implement the
equivalent Metropolis-within-Gibbs in block form:

- For each event g, the candidate set of parents is enumerated under the inner
  intCRP window. The conditional posterior is sampled exactly:
      P(parent | rest) ∝ inner_intcrp_weight(parent) · F_i(rate at g | parent)
  This corresponds to block-resampling event_parent[g] in one shot, which is a
  valid Gibbs move (acceptance probability 1) and respects detailed balance.

- For each cluster c, the conditional posterior of cluster_parent[c] (plus the
  "cascade root" option) is sampled likewise from outer intCRP × F_j(...).

After every sweep we optionally re-fit Θ via closed-form MLE (paper sets Θ from
priors; we found alternating MLE converges much faster on real data).
"""

from __future__ import annotations

import time

import numpy as np

from .events import EventCollection
from .intcrp import (
    inner_candidate_parents,
    inner_intcrp_weights,
    outer_candidate_parents,
)
from .kernels import apply_link
from .likelihood import conditional_rate_at_event, log_likelihood
from .mle import mle_update
from .params import HawkesParams
from .state import BranchingState


_EPS = 1e-12
PARENT_SELECTION_MODES = frozenset({"sample", "argmax"})


def _elapsed_text(start_time: float) -> str:
    elapsed = max(0.0, time.time() - start_time)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def _print_phase_progress(sweep, phase, done, total, start_time, *, force=False, progress_every=50000):
    if total <= 0:
        return
    if not force and progress_every == 0:
        return
    if not force and progress_every > 0 and done % progress_every != 0:
        return
    percent = 100.0 * done / total
    print(
        f"sweep={sweep:4d} {phase}: {done}/{total} "
        f"({percent:5.1f}%) elapsed={_elapsed_text(start_time)}",
        flush=True,
    )


def _sample_categorical(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        # Defensive: fall back to uniform if every option got zero weight.
        return int(rng.integers(len(weights)))
    probs = weights / total
    return int(rng.choice(len(weights), p=probs))


def _select_parent(weights: np.ndarray, rng: np.random.Generator, mode: str) -> int:
    if mode == "argmax":
        return int(np.argmax(weights))
    if mode == "sample":
        return _sample_categorical(weights, rng)
    raise ValueError(f"unknown parent_selection: {mode}")


def resample_event_link(
    child_g: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    window: float,
    rng: np.random.Generator,
    parent_selection: str = "sample",
) -> None:
    candidates = inner_candidate_parents(child_g, events, window)
    prior = inner_intcrp_weights(child_g, candidates, events, params, window)
    # Likelihood contribution from this single event under each candidate parent.
    like = np.empty(len(candidates))
    for k, g in enumerate(candidates):
        like[k] = conditional_rate_at_event(child_g, int(g), events, params)
    weights = prior * like
    if not np.any(weights > 0):
        # All-zero: keep current parent (no-op rather than picking nonsense).
        return
    pick = _select_parent(weights, rng, parent_selection)
    state.set_event_parent(child_g, int(candidates[pick]))


_EPS_CRA = 1e-12  # mirror likelihood._EPS so the rate matches conditional_rate_at_event


def _outer_weight_source_rate(
    source_cluster: int,
    t_je: float,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    target_dim: int,
) -> tuple:
    """Single pass for the outer-intCRP cluster-link sampler. Returns
    `(prior_weight, src_event, target_rate)`:

    - prior_weight = max over t_ik ∈ source cluster (t_ik < t_je) of
      |α_ji| · φ_ji(t_je − t_ik)  (paper Eq. 3.2)
    - src_event = the event achieving that max (paper §3.2 convention)
    - target_rate = F_j applied to that same kernel max (i.e. the
      cluster-Poisson conditional rate at the target's earliest event
      attributed to src_event). Folding the rate into this pass saves a
      duplicate `conditional_rate_at_event` call + exp_kernel evaluation
      per candidate.

    Returns (0.0, -1, 0.0) when no source event qualifies or the (i, j)
    edge is below the sparsity threshold.
    """
    src_events = state.cluster_events(source_cluster)
    src_times = events.times[src_events]
    mask = src_times < t_je
    if not np.any(mask):
        return 0.0, -1, 0.0
    src_dim = int(events.dims[src_events[0]])
    signed_alpha = params.alpha_value(target_dim, src_dim)
    alpha_abs = abs(signed_alpha)
    if params.edge_threshold > 0.0 and alpha_abs <= params.edge_threshold:
        return 0.0, -1, 0.0
    beta_ji = params.beta_value(target_dim, src_dim)
    if mask.all():
        taus = t_je - src_times
        idx = np.arange(len(src_times))
    else:
        idx = np.flatnonzero(mask)
        taus = t_je - src_times[idx]
    # taus are strictly positive — skip the np.where/clip in exp_kernel.
    # |α| factors out of argmax, so the best source event is just the closest in time.
    decay = beta_ji * np.exp(-beta_ji * taus)
    best = int(np.argmax(decay))
    weight = float(alpha_abs * decay[best])
    # Rate uses signed α so non-linear inhibition (α<0 with exp/softplus link) matches
    # the original conditional_rate_at_event semantics. Under linear F and non-negative
    # α the two paths are identical.
    rate = float(apply_link(signed_alpha * decay[best], params.links[target_dim])) + _EPS_CRA
    return weight, int(src_events[idx[best]]), rate


def resample_cluster_link(
    target_cluster: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    rng: np.random.Generator,
    *,
    window: float | None = None,
    parent_selection: str = "sample",
) -> None:
    candidates = outer_candidate_parents(target_cluster, state, events, params, window=window)
    # Build weight vector: [each candidate cluster, then 'cascade root' (no parent)].
    tgt_events = state.cluster_events(target_cluster)
    tgt_times = events.times[tgt_events]
    earliest_g = int(tgt_events[int(np.argmin(tgt_times))])
    t_je = float(events.times[earliest_g])
    target_dim = int(events.dims[earliest_g])

    n_opt = len(candidates) + 1  # +1 for 'no parent' option
    prior = np.zeros(n_opt)
    like = np.zeros(n_opt)
    for k, c in enumerate(candidates):
        weight, src_event, rate = _outer_weight_source_rate(c, t_je, state, events, params, target_dim)
        if src_event < 0:
            continue
        prior[k] = weight
        like[k] = rate
    # 'No parent' option: prior weight = ρ = μ_target_dim (paper convention for self-affinity);
    # likelihood = F_j(μ_j) at earliest event.
    prior[-1] = max(float(params.mu[target_dim]), _EPS)
    like[-1] = float(apply_link(params.mu[target_dim], params.links[target_dim])) + _EPS

    weights = prior * like
    if not np.any(weights > 0):
        return
    pick = _select_parent(weights, rng, parent_selection)
    if pick == n_opt - 1:
        state.set_cluster_parent(target_cluster, -1)
    else:
        state.set_cluster_parent(target_cluster, int(candidates[pick]))


def media_sweep(
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    window: float,
    rng: np.random.Generator,
    *,
    refit_params: bool = True,
    event_order: str = "time",
    sweep: int | None = None,
    verbose: bool = False,
    progress_every: int = 50000,
    parent_selection: str = "sample",
) -> HawkesParams:
    """One full MCMC sweep: event links → cluster links → (optional) param refit."""
    if parent_selection not in PARENT_SELECTION_MODES:
        raise ValueError(f"parent_selection must be one of {sorted(PARENT_SELECTION_MODES)}")
    sweep_label = int(sweep or 0)
    n = events.n
    if event_order == "time":
        order = np.argsort(events.times, kind="stable")
    elif event_order == "random":
        order = rng.permutation(n)
    else:
        raise ValueError(f"unknown event_order: {event_order}")

    phase_start = time.time()
    if verbose:
        _print_phase_progress(
            sweep_label,
            "event_links",
            0,
            n,
            phase_start,
            force=True,
            progress_every=progress_every,
        )
    for offset, g in enumerate(order, start=1):
        resample_event_link(
            int(g),
            state,
            events,
            params,
            window,
            rng,
            parent_selection=parent_selection,
        )
        if verbose:
            _print_phase_progress(
                sweep_label,
                "event_links",
                offset,
                n,
                phase_start,
                progress_every=progress_every,
                force=offset == n,
            )

    # cluster_parent indexing depends on cluster ids, which can shift after event
    # link changes. Materialize the current cluster list and resample each.
    num_clusters = state.num_clusters
    cluster_order = rng.permutation(num_clusters)
    phase_start = time.time()
    if verbose:
        _print_phase_progress(
            sweep_label,
            "cluster_links",
            0,
            num_clusters,
            phase_start,
            force=True,
            progress_every=progress_every,
        )
    for offset, c in enumerate(cluster_order, start=1):
        # During the sweep the cluster id space stays fixed (no event link changes
        # until the next sweep), so this is safe.
        resample_cluster_link(
            int(c),
            state,
            events,
            params,
            rng,
            window=window,
            parent_selection=parent_selection,
        )
        if verbose:
            _print_phase_progress(
                sweep_label,
                "cluster_links",
                offset,
                num_clusters,
                phase_start,
                progress_every=progress_every,
                force=offset == num_clusters,
            )

    if refit_params:
        phase_start = time.time()
        if verbose:
            print(f"sweep={sweep_label:4d} refit_params: start", flush=True)
        params = mle_update(
            events,
            state,
            params,
            edge_threshold=params.edge_threshold,
            max_active_sources_per_dim=params.max_active_sources_per_dim,
        )
        if verbose:
            print(
                f"sweep={sweep_label:4d} refit_params: done "
                f"elapsed={_elapsed_text(phase_start)}",
                flush=True,
            )
    return params


def run_media(
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    *,
    window: float,
    n_sweeps: int = 50,
    burn_in: int = 10,
    seed: int = 0,
    refit_params: bool = True,
    verbose: bool = False,
    log_every: int = 10,
    progress_every: int = 50000,
    parent_selection: str = "sample",
    sweep_callback=None,
) -> dict:
    """Drive MEDIA for n_sweeps. Returns a dict with best params, final state,
    and a likelihood trace.
    """
    rng = np.random.default_rng(seed)
    trace = []
    best_ll = -np.inf
    best_params = params.copy()
    best_event_parent = state.event_parent.copy()
    best_cluster_parent = None

    for sweep in range(n_sweeps):
        params = media_sweep(
            state,
            events,
            params,
            window,
            rng,
            refit_params=refit_params,
            sweep=sweep,
            verbose=verbose,
            progress_every=progress_every,
            parent_selection=parent_selection,
        )
        ll_start = time.time()
        if verbose:
            print(f"sweep={sweep:4d} log_likelihood: start", flush=True)
        ll = log_likelihood(events, state, params)
        if verbose:
            print(
                f"sweep={sweep:4d} log_likelihood: done "
                f"elapsed={_elapsed_text(ll_start)}",
                flush=True,
            )
        trace_record = {"sweep": sweep, "log_likelihood": ll, "num_cascades": state.num_cascades}
        trace.append(trace_record)
        if verbose and (sweep % log_every == 0 or sweep == n_sweeps - 1):
            print(
                f"sweep={sweep:4d} log_lik={ll:.4f} "
                f"clusters={state.num_clusters} cascades={state.num_cascades} "
                f"rho={params.spectral_radius():.3f}"
            )
        is_best = False
        if sweep >= burn_in and ll > best_ll:
            best_ll = ll
            best_params = params.copy()
            best_event_parent = state.event_parent.copy()
            state._ensure_clusters()
            best_cluster_parent = state._cluster_parent.copy()
            is_best = True
        if sweep_callback is not None:
            has_post_burn_in_best = np.isfinite(best_ll)
            checkpoint_params = best_params.copy() if has_post_burn_in_best else params.copy()
            sweep_callback(
                {
                    "sweep": sweep,
                    "log_likelihood": ll,
                    "num_clusters": state.num_clusters,
                    "num_cascades": state.num_cascades,
                    "is_best": is_best,
                    "best_log_likelihood": best_ll if has_post_burn_in_best else ll,
                    "checkpoint_is_post_burn_in": bool(has_post_burn_in_best),
                    "checkpoint_params": checkpoint_params,
                    "trace": list(trace),
                }
            )

    return {
        "params": best_params,
        "event_parent": best_event_parent,
        "cluster_parent": best_cluster_parent,
        "trace": trace,
        "best_log_likelihood": best_ll,
    }
