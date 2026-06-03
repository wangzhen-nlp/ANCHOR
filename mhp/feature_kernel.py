"""Feature-weighted excitation amplitude for the MHP.

Instead of a free per-(target, source) edge weight α[u, v], the feature kernel
models the amplitude as a log-linear function of pair features:

    α(u, v) = softplus( w · φ(u, v) )

where φ(u, v) is a device-/topology-agnostic feature vector (alarm-type pair,
topology relation, same-site, same-vendor, ...) and w is a small learned weight
vector. This is what gives the model *inductive* generalization: a (target,
source) pair that never appeared in training — including pairs involving brand
new devices — still gets a sensible α as long as its features φ are computable
(from the NE graph / alarm type), because w was learned at the feature level.

The kernel is fit by maximizing the EM M-step objective

    Q(w) = Σ_c [ N_c · log α_c − E_c · α_c ] − λ·||w||²

over candidate pairs c, where N_c is the aggregated parent responsibility for
pair c and E_c its exposure (source-type event count). softplus keeps α > 0 and
its gradient is the logistic sigmoid, so Q is smooth and L-BFGS-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_EPS = 1e-12


def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(z))."""
    # log1p(exp(-|z|)) + max(z, 0)
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable logistic sigmoid = d softplus / dz."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class FeatureKernel:
    """Log-linear amplitude model α = softplus(w · φ).

    Attributes
    ----------
    weights : (F,) float64
        Learned feature weights w.
    feature_names : list[str]
        Human-readable names, len F (for interpretability / serialization).
    l2 : float
        Ridge penalty λ on ||w||² (excluding the bias term, index 0).
    """

    weights: np.ndarray
    feature_names: list = field(default_factory=list)
    l2: float = 1e-3

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.feature_names and len(self.feature_names) != len(self.weights):
            raise ValueError("feature_names length must match weights")

    @property
    def n_features(self) -> int:
        return int(len(self.weights))

    def alpha(self, phi: np.ndarray) -> np.ndarray:
        """α = softplus(φ · w) for a (P, F) feature matrix → (P,) amplitudes."""
        if len(phi) == 0:
            return np.zeros(0, dtype=np.float64)
        return softplus(phi @ self.weights)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.astype(float).tolist(),
            "feature_names": list(self.feature_names),
            "l2": float(self.l2),
        }

    @classmethod
    def from_dict(cls, payload) -> "FeatureKernel":
        payload = dict(payload or {})
        return cls(
            weights=np.asarray(payload.get("weights", ()), dtype=np.float64),
            feature_names=list(payload.get("feature_names", [])),
            l2=float(payload.get("l2", 1e-3)),
        )


def _q_and_grad(w, phi, n_resp, exposure, l2, w0, reg_mask):
    """Objective Q(w) = Σ_c [N_c·log α_c − E_c·α_c] − λ·Σ reg·(w−w0)² and its
    gradient (to be MAXIMIZED).
    """
    z = phi @ w
    a = softplus(z)
    a_safe = np.maximum(a, _EPS)
    q = float(np.sum(n_resp * np.log(a_safe) - exposure * a) - l2 * np.sum(reg_mask * (w - w0) ** 2))
    coef = (n_resp / a_safe - exposure) * sigmoid(z)
    grad = phi.T @ coef - 2.0 * l2 * reg_mask * (w - w0)
    return q, grad


def fit_weights_mstep(
    phi: np.ndarray,
    n_resp: np.ndarray,
    exposure: np.ndarray,
    w_init: np.ndarray,
    *,
    l2: float = 1e-3,
    w_prior_mean: np.ndarray | None = None,
    max_iter: int = 50,
) -> np.ndarray:
    """One M-step: maximize Q(w) = Σ_c [N_c·log α_c − E_c·α_c] − λ·||w−w0||²
    over candidate pairs, via gradient ascent with Armijo backtracking line
    search (pure numpy — no scipy dependency). F is small (dozens) and the
    step is warm-started each EM iteration, so a few iterations converge.

    Parameters
    ----------
    phi : (C, F) float64        per-candidate feature matrix
    n_resp : (C,) float64       aggregated parent responsibility N_c
    exposure : (C,) float64     exposure E_c (source-type event count)
    w_init : (F,) float64       warm-start weights
    l2 : float                  ridge strength (bias term, index 0, is exempt)
    w_prior_mean : (F,) or None informative prior mean for w (e.g. high on the
                                topology feature); ridge pulls toward it.
    """
    F = phi.shape[1]
    w0 = np.zeros(F) if w_prior_mean is None else np.asarray(w_prior_mean, dtype=np.float64).reshape(-1)
    reg_mask = np.ones(F)
    reg_mask[0] = 0.0  # do not regularize the bias term
    w = np.asarray(w_init, dtype=np.float64).reshape(-1).copy()

    q, grad = _q_and_grad(w, phi, n_resp, exposure, l2, w0, reg_mask)
    for _ in range(max_iter):
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-8:
            break
        # Armijo backtracking ascent: find t so Q(w + t·g) >= Q(w) + c·t·||g||²
        t = 1.0
        c = 1e-4
        improved = False
        for _bt in range(30):
            w_new = w + t * grad
            q_new, grad_new = _q_and_grad(w_new, phi, n_resp, exposure, l2, w0, reg_mask)
            if q_new >= q + c * t * gnorm * gnorm:
                w, q, grad = w_new, q_new, grad_new
                improved = True
                break
            t *= 0.5
        if not improved:
            break  # line search failed → at (local) optimum for this M-step
    return w
