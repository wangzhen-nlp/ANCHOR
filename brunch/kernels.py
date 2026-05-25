"""Triggering kernels and link functions for hybrid MHPs (paper §2.1)."""

from __future__ import annotations

import numpy as np


def exp_kernel(tau, beta):
    """φ(τ) = β · exp(-β τ) for τ ≥ 0, else 0."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.where(tau >= 0.0, beta * np.exp(-beta * np.clip(tau, 0.0, None)), 0.0)
    return out


def exp_kernel_integral(t, beta):
    """∫_0^t β · exp(-β s) ds = 1 - exp(-β t)."""
    t = np.asarray(t, dtype=np.float64)
    return 1.0 - np.exp(-beta * np.clip(t, 0.0, None))


_LINK_FUNCTIONS = {
    "linear": lambda x: np.maximum(x, 0.0),
    "exp": lambda x: np.exp(np.clip(x, -50.0, 50.0)),
    "softplus": lambda x: np.log1p(np.exp(np.clip(x, -50.0, 50.0))),
}


def apply_link(x, name):
    if name not in _LINK_FUNCTIONS:
        raise ValueError(f"unknown link function: {name}")
    return _LINK_FUNCTIONS[name](x)


def link_names():
    return tuple(_LINK_FUNCTIONS.keys())
