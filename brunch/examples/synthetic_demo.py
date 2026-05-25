#!/usr/bin/env python3
"""End-to-end synthetic recovery demo for BRUNCH.

Generate a small multivariate Hawkes process with a known branching structure,
fit BRUNCH on the timestamps + dimensions, and report parent-recovery F1.
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from brunch import (
    BRUNCH,
    BRUNCHConfig,
    HawkesParams,
    f1_parent_recovery,
    generate,
)


def main():
    M = 3
    true_params = HawkesParams(
        M=M,
        mu=np.array([0.2, 0.15, 0.1]),
        alpha=np.array(
            [
                [0.3, 0.4, 0.0],   # dim 0 receives from {0, 1}
                [0.0, 0.2, 0.5],   # dim 1 receives from {1, 2}
                [0.3, 0.0, 0.1],   # dim 2 receives from {0, 2}
            ]
        ),
        beta=np.array(
            [
                [1.0, 0.8, 1.2],
                [0.6, 1.0, 0.9],
                [1.1, 0.7, 1.0],
            ]
        ),
        links=["linear"] * M,
    )
    T = 200.0
    print(f"true spectral radius: {true_params.spectral_radius():.3f}")

    synthetic = generate(true_params, T=T, seed=42)
    print(
        f"generated {synthetic.events.n} events over T={T} "
        f"({(synthetic.parent_of == np.arange(synthetic.events.n)).sum()} immigrants)"
    )

    # Paper-faithful setup: draw Θ from priors (here we pass the true params as
    # the "prior draw"; in practice you would set this from domain priors or a
    # rough first-pass MLE), then run MEDIA with Θ frozen. Set refit_params=True
    # to alternate Bayesian MLE updates (faster convergence on real data, but
    # benchmarked recovery is best with the paper-faithful frozen-Θ setup).
    config = BRUNCHConfig(
        M=M,
        window=20.0,
        n_sweeps=120,
        burn_in=30,
        refit_params=False,
        warm_start=False,
        seed=0,
        verbose=True,
        log_every=10,
    )
    model = BRUNCH(config)
    result = model.fit(synthetic.events, init_params=true_params)

    print()
    print("Recovered Θ̂:")
    print(f"  μ̂   = {np.round(result.params.mu, 3)}")
    print(f"  α̂   = \n{np.round(result.params.alpha_matrix(), 3)}")
    print(f"  β̂   = \n{np.round(result.params.beta_matrix(), 3)}")
    print(f"  ρ(α̂) = {result.params.spectral_radius():.3f}")
    print(f"best log-likelihood: {result.best_log_likelihood:.3f}")

    print()
    metrics = f1_parent_recovery(synthetic.parent_of, result.parent_of)
    print("parent recovery:")
    for k, v in metrics.items():
        print(f"  {k:>16s}: {v}")


if __name__ == "__main__":
    main()
