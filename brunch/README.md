# BRUNCH

End-to-end Python implementation of

> Li, Li, Bhowmick. *BRUNCH: Branching Structure Inference of Hybrid Multivariate
> Hawkes Processes with Application to Social Media.* PAKDD 2020.

Output of `BRUNCH.fit(events)` is the **branching tree** for a set of timestamped,
multi-typed events: for each event, the event that most likely triggered it
(or a flag marking it as a spontaneous immigrant).

## Quick start

```python
import numpy as np
from brunch import BRUNCH, BRUNCHConfig, HawkesParams, generate, f1_parent_recovery

# Ground-truth synthetic data (3 dims, linear MHP, exponential kernels)
true_params = HawkesParams(
    M=3,
    mu=np.array([0.2, 0.15, 0.1]),
    alpha=np.array([[0.3, 0.4, 0.0], [0.0, 0.2, 0.5], [0.3, 0.0, 0.1]]),
    beta=np.array([[1.0, 0.8, 1.2], [0.6, 1.0, 0.9], [1.1, 0.7, 1.0]]),
)
data = generate(true_params, T=200.0, seed=42)

# Fit BRUNCH
result = BRUNCH(BRUNCHConfig(M=3, window=10.0, n_sweeps=40, verbose=True)).fit(data.events)

# Inspect
print("parent of each event:", result.parent_of)             # (n,) global event id
print("branching edges:", result.branching_edges)            # sparse rows (parent, child)
print("branching matrix S[parent, child]:", result.branching_matrix)
print("cascade id per event:", result.cascade_of)
print(f1_parent_recovery(data.parent_of, result.parent_of))  # parent-recovery F1
```

The full demo script lives at [`examples/synthetic_demo.py`](examples/synthetic_demo.py).

## What this implements

| Paper section | Module | Notes |
|---|---|---|
| §2.1 hybrid MHP intensity | `kernels.py` `params.py` | Per-dimension link function (`linear`, `exp`, `softplus`) + exponential triggering kernel |
| §2.2, §3 branching state | `state.py` | Event links (within-dim) + cluster links (cross-dim), derived clusters and cascades |
| §3.1 inner intCRP | `intcrp.py::inner_*` | Eq. 3.1 with time window `W` and self-triggering kernel `α_ii · φ_ii(τ)` |
| §3.2 outer intCRP | `intcrp.py::outer_*` | Eq. 3.2 cluster-link weight `max |α_ji| · φ_ji(t_je − t_ik)` |
| §4 MEDIA inference | `media.py` | Block Gibbs over event links and cluster links (see *Simplification* below) |
| §4 likelihood | `likelihood.py` | Eq. 4.4 with the paper's rectangular integral approximation `Σ_m (t_m − t_{m−1}) λ_i(t_m)` |
| Parameter inference | `mle.py` | Closed-form MLE for `μ, α, β` between sweeps (alternates with MCMC) |
| Synthetic data + eval | `synthetic.py` | Cluster Poisson generator + F1 recovery |

## Sparse / large-M settings

For many event types, set:

```python
BRUNCHConfig(
    M=K,
    sparse_alpha_threshold=1e-3,
    max_active_sources_per_dim=32,
    sparse_parameter_storage=True,
    materialize_branching_matrix=False,
)
```

- `sparse_alpha_threshold` makes likelihood and intCRP ignore tiny `α_ij` edges.
- `max_active_sources_per_dim` prunes MLE updates to top-k source dimensions per
  target dimension while keeping observed parent pairs and self edges.
- `materialize_branching_matrix=False` avoids allocating the dense `n × n`
  adjacency; use `result.branching_edges` instead.
- `sparse_parameter_storage=True` stores `α/β` as edge arrays instead of dense
  `M × M` matrices. Prefer `HawkesParams.from_edges(...)` for very large `M`.

With sparse active edges `E`, the rectangular likelihood update is `O(n · E + n · M)`
instead of touching the full `M²` grid at every event. In sparse storage mode,
parameters are held as:

```python
params = HawkesParams.from_edges(
    M=K,
    mu=mu,
    edge_targets=target_dims,
    edge_sources=source_dims,
    edge_alpha=edge_alpha,
    edge_beta=edge_beta,
    links=links,
)
```

For diagnostics on small models use `params.alpha_matrix()` and
`params.beta_matrix()`. Avoid `params.alpha` / `params.beta` on very large sparse
models because those compatibility properties materialize dense matrices.

## Simplifications vs. the paper

1. **Block Gibbs instead of the 4-case MH (Eq. 4.5–4.10).** Because the
   paper's MH proposal is the inner intCRP prior, the Metropolis-Hastings
   acceptance reduces to a likelihood ratio. We enumerate every candidate
   parent (within the time window `W`) plus the self-loop, weight each by
   `prior · F_i(rate-at-event)`, and sample exactly. This is a valid Gibbs
   move (acceptance probability 1) with the same stationary distribution.
   It is `O(L)` likelihood evaluations per event instead of a single MH
   step, but it removes all the split/merge bookkeeping.
2. **Alternating parameter MLE.** The paper draws `Θ` once from priors and
   leaves it fixed. We re-fit `μ, α, β` by closed-form MLE between sweeps,
   which converges much faster in practice. Disable with
   `BRUNCHConfig(refit_params=False)` if you want pure paper behavior.
3. **Linear MLE for non-linear dims.** When a dim uses `exp`/`softplus`, the
   parameter update still uses the linear closed-form (treating the
   linearized intensity). For pure non-linear inference, plug in your own
   `Θ` via the `init_params` argument and disable `refit_params`.
4. **Spectral radius cap.** `mle.py` clips `α`'s spectral radius to `0.95` to
   keep the linear MHP stationary. Synthetic generators with `ρ ≥ 1` are
   rejected up front.

## Scope and complexity

| Aspect | Cost |
|---|---|
| Memory | `O(n + M²)` for state, `O(n²)` for the dense branching matrix `S` |
| Per-sweep compute | `O(n · L)` for inner Gibbs (L = avg candidates per event), plus `O(K · K')` for outer Gibbs (K clusters, K' cross-dim candidates), plus sparse likelihood `O(n · E + n · M)` when pruning is enabled |
| Cluster recomputation | `O(n)` per event-link mutation in the naive impl — fine for `n ≲ a few thousand`, becomes the bottleneck above that |

For BRUNCH-scale alarm streams (10⁵–10⁶ events), incremental cluster/cascade
updates and a sparse branching matrix are required. Both are localized changes
to `state.py` and would not affect the algorithm itself.

## Hooks for plugging in real data

`EventCollection.from_pairs([(time, dim), ...], M=K, T=horizon)` is the only
adapter you need. Map your event categories to integer dimension ids and feed
them in chronological order. The result `BRUNCHResult` exposes:

- `parent_of[i]` — global id of the event that triggered event `i` (self = immigrant);
- `cascade_of[i]` — cascade membership (mutually independent under the cluster Poisson view);
- `branching_edges` — sparse edge list `(parent, child)`;
- `branching_matrix` — dense binary tree (`S[parent, child] = 1`) when materialized;
- `params` — fit `μ, α, β` for downstream interpretation.
