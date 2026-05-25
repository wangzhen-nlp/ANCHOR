"""Synthetic multivariate Hawkes data generator (Ogata's thinning with parent attribution).

Generates linear MHP events together with the ground-truth branching tree, which
lets us evaluate BRUNCH's recovery quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .events import EventCollection
from .params import HawkesParams


@dataclass
class SyntheticData:
    events: EventCollection
    parent_of: np.ndarray   # ground-truth parent index per event (self = immigrant)


def generate(
    params: HawkesParams,
    T: float,
    seed: int = 0,
    max_events: int = 100_000,
) -> SyntheticData:
    """Cluster Poisson generator for a linear MHP with exponential kernels.

    Each event in dim j spawns a Poisson(α_ij) cluster of dim-i offspring whose
    waiting times are drawn from φ_ij(τ) = β_ij exp(-β_ij τ). Background
    immigrants in dim i arrive as a Poisson process of rate μ_i.

    Requires the spectral radius of α to be < 1; otherwise the process is
    non-stationary and the recursion may not terminate.
    """
    M = params.M
    rng = np.random.default_rng(seed)
    if params.spectral_radius() >= 1.0:
        raise ValueError(f"spectral radius {params.spectral_radius():.3f} ≥ 1; generator would diverge")

    times: List[float] = []
    dims: List[int] = []
    parents: List[int] = []  # parent global index, or -1 placeholder during construction

    # Immigrants per dim
    queue: List[Tuple[float, int, int]] = []  # (time, dim, parent_global_idx or -1)
    for i in range(M):
        n_imm = rng.poisson(params.mu[i] * T)
        for t in rng.uniform(0.0, T, size=n_imm):
            queue.append((float(t), int(i), -1))

    # Process events chronologically, generating offspring as we go.
    queue.sort(key=lambda e: e[0])
    while queue:
        t, j, parent_g = queue.pop(0)
        if t > T or len(times) >= max_events:
            continue
        my_g = len(times)
        times.append(t)
        dims.append(j)
        parents.append(parent_g if parent_g >= 0 else my_g)  # self for immigrants
        # Spawn offspring for every active target dim i.
        for i in params.active_targets_for_source(j):
            i = int(i)
            a = params.alpha_value(i, j)
            if a <= 0:
                continue
            n_off = rng.poisson(a)
            if n_off == 0:
                continue
            # Waiting times from φ_ij = β exp(-β τ) → Exponential(β)
            taus = rng.exponential(1.0 / params.beta_value(i, j), size=n_off)
            for tau in taus:
                t_child = t + float(tau)
                if t_child >= T:
                    continue
                queue.append((t_child, int(i), my_g))
        # Keep queue sorted by time (insertion sort would be faster but n is small here)
        queue.sort(key=lambda e: e[0])

    times_arr = np.asarray(times, dtype=np.float64)
    dims_arr = np.asarray(dims, dtype=np.int64)
    parents_arr = np.asarray(parents, dtype=np.int64)
    # Sort events by time and remap parent ids consistently
    order = np.argsort(times_arr, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    times_arr = times_arr[order]
    dims_arr = dims_arr[order]
    new_parents = np.empty_like(parents_arr)
    for new_i, old_i in enumerate(order):
        old_parent = parents[old_i]
        if old_parent == old_i:
            new_parents[new_i] = new_i
        else:
            new_parents[new_i] = int(inv[old_parent])
    return SyntheticData(
        events=EventCollection(times=times_arr, dims=dims_arr, M=M, T=T),
        parent_of=new_parents,
    )


def f1_parent_recovery(true_parent: np.ndarray, pred_parent: np.ndarray) -> dict:
    """F1 score over directed parent assignments (S matrix entries).

    Treats each event's parent as an edge (parent → child); immigrants contribute
    a self-loop (counted as "no incoming edge" rather than a true positive).
    """
    n = len(true_parent)
    true_edges = {(int(true_parent[c]), c) for c in range(n) if int(true_parent[c]) != c}
    pred_edges = {(int(pred_parent[c]), c) for c in range(n) if int(pred_parent[c]) != c}
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    immigrants_true = int(np.sum(true_parent == np.arange(n)))
    immigrants_pred = int(np.sum(pred_parent == np.arange(n)))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_edges": len(true_edges),
        "pred_edges": len(pred_edges),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "immigrants_true": immigrants_true,
        "immigrants_pred": immigrants_pred,
    }
