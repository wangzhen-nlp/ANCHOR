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

import numpy as np

from .events import EventCollection
from .intcrp import (
    inner_candidate_parents,
    inner_intcrp_weights,
    outer_candidate_parents,
    outer_intcrp_weight,
)
from .kernels import apply_link, exp_kernel
from .likelihood import conditional_rate_at_event, log_likelihood
from .mle import mle_update
from .params import HawkesParams
from .state import BranchingState


_EPS = 1e-12


def _sample_categorical(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        # Defensive: fall back to uniform if every option got zero weight.
        return int(rng.integers(len(weights)))
    probs = weights / total
    return int(rng.choice(len(weights), p=probs))


def resample_event_link(
    child_g: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    window: float,
    rng: np.random.Generator,
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
    pick = _sample_categorical(weights, rng)
    state.set_event_parent(child_g, int(candidates[pick]))


def _pick_source_event_in_cluster(
    source_cluster: int,
    t_je: float,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    target_dim: int,
) -> int:
    """Argmax-kernel pick: the event in source cluster that most likely triggered
    the earliest event of target cluster (paper §3.2 convention).
    """
    src_events = np.asarray(state.cluster_events(source_cluster))
    src_times = events.times[src_events]
    valid = src_times < t_je
    idx = np.where(valid)[0]
    if not len(idx):
        return -1
    src_dim = int(events.dims[src_events[0]])
    a = abs(params.alpha_value(target_dim, src_dim))
    b = params.beta_value(target_dim, src_dim)
    contribs = a * exp_kernel(t_je - src_times[idx], b)
    return int(src_events[idx[int(np.argmax(contribs))]])


def resample_cluster_link(
    target_cluster: int,
    state: BranchingState,
    events: EventCollection,
    params: HawkesParams,
    rng: np.random.Generator,
) -> None:
    candidates = outer_candidate_parents(target_cluster, state, events, params)
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
        prior[k] = outer_intcrp_weight(c, target_cluster, state, events, params)
        src_event = _pick_source_event_in_cluster(c, t_je, state, events, params, target_dim)
        if src_event < 0:
            prior[k] = 0.0
            like[k] = 0.0
            continue
        like[k] = conditional_rate_at_event(earliest_g, src_event, events, params)
    # 'No parent' option: prior weight = ρ = μ_target_dim (paper convention for self-affinity);
    # likelihood = F_j(μ_j) at earliest event.
    prior[-1] = max(float(params.mu[target_dim]), _EPS)
    like[-1] = float(apply_link(params.mu[target_dim], params.links[target_dim])) + _EPS

    weights = prior * like
    if not np.any(weights > 0):
        return
    pick = _sample_categorical(weights, rng)
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
) -> HawkesParams:
    """One full MCMC sweep: event links → cluster links → (optional) param refit."""
    n = events.n
    if event_order == "time":
        order = np.argsort(events.times, kind="stable")
    elif event_order == "random":
        order = rng.permutation(n)
    else:
        raise ValueError(f"unknown event_order: {event_order}")

    for g in order:
        resample_event_link(int(g), state, events, params, window, rng)

    # cluster_parent indexing depends on cluster ids, which can shift after event
    # link changes. Materialize the current cluster list and resample each.
    num_clusters = state.num_clusters
    cluster_order = rng.permutation(num_clusters)
    for c in cluster_order:
        # During the sweep the cluster id space stays fixed (no event link changes
        # until the next sweep), so this is safe.
        resample_cluster_link(int(c), state, events, params, rng)

    if refit_params:
        params = mle_update(
            events,
            state,
            params,
            edge_threshold=params.edge_threshold,
            max_active_sources_per_dim=params.max_active_sources_per_dim,
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
        params = media_sweep(state, events, params, window, rng, refit_params=refit_params)
        ll = log_likelihood(events, state, params)
        trace.append({"sweep": sweep, "log_likelihood": ll, "num_cascades": state.num_cascades})
        if verbose and (sweep % log_every == 0 or sweep == n_sweeps - 1):
            print(
                f"sweep={sweep:4d} log_lik={ll:.4f} "
                f"clusters={state.num_clusters} cascades={state.num_cascades} "
                f"rho={params.spectral_radius():.3f}"
            )
        if sweep >= burn_in and ll > best_ll:
            best_ll = ll
            best_params = params.copy()
            best_event_parent = state.event_parent.copy()
            state._ensure_clusters()
            best_cluster_parent = state._cluster_parent.copy()

    return {
        "params": best_params,
        "event_parent": best_event_parent,
        "cluster_parent": best_cluster_parent,
        "trace": trace,
        "best_log_likelihood": best_ll,
    }
