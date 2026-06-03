"""Multivariate Hawkes Process with MAP EM learning.

Continuous-time multivariate Hawkes process whose triggering matrix α and
background rate μ are learned jointly via Maximum a Posteriori (MAP) EM —
adapted from Steve Morse's MHP (https://github.com/stmorse/hawkes) and
Veen & Schoenberg 2008's branching-process EM, with two structural changes:

  1. Windowed sparse E-step. Morse's dense O(N²) p_ij matrix is replaced by a
     per-event history window. Each event only enumerates candidate parents
     within `history_window`; the (event, candidate) pair list grows linearly,
     not quadratically.
  2. Sparse α update with Gamma prior. M-step uses Bayesian shrinkage
     (count + K·m) / (n + K), preventing the all-immigrant attractor that
     vanilla MLE EM collapses into on heavy-tail multivariate data.

The package contains:

  - EventCollection : tuple of (times, dims, M, T) for an input sequence
  - MHPParams       : sparse-storage triggering matrix + background rate
  - MHPConfig       : EM hyperparameters
  - fit_mhp         : MAP EM driver, returns MHPResult
"""

from .events import EventCollection
from .params import MHPParams
from .em import (
    MHPConfig,
    MHPResult,
    compute_cascade_of,
    compute_hard_parents,
    fit_mhp,
    fit_mhp_feature,
    fit_mhp_piecewise,
    log_likelihood,
)
from .feature_kernel import FeatureKernel

__all__ = [
    "EventCollection",
    "MHPParams",
    "MHPConfig",
    "MHPResult",
    "fit_mhp",
    "fit_mhp_feature",
    "fit_mhp_piecewise",
    "FeatureKernel",
    "log_likelihood",
    "compute_hard_parents",
    "compute_cascade_of",
]
