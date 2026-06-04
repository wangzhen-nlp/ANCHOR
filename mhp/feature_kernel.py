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


def _q_and_grad_dynamic(w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask):
    """Bucketed objective Σ_{c,k}[N_{c,k}·log α_{c,k} − E_{c,k}·α_{c,k}] − ridge,
    with α_{c,k} = softplus(z_s[c] + z_d[k]), z_s = cand_phi·w_s, z_d = combo_bits·w_d
    and w = [w_s (F), w_d (D)]. Gradient decomposes over the static and dynamic
    blocks WITHOUT materializing the (C, K) row matrix:
        ∂Q/∂w_s = cand_phiᵀ · Σ_k coef[:,k]
        ∂Q/∂w_d = combo_bitsᵀ · (Σ_c coef[:,k])_k
    """
    F = cand_phi.shape[1]
    K, D = combo_bits.shape
    ws, wd = w[:F], w[F:]
    z_s = cand_phi @ ws                     # (C,)
    z_d = combo_bits @ wd                   # (K,)
    q = 0.0
    grad_s = np.zeros(F)
    coef_sum_per_combo = np.zeros(K)        # Σ_c coef[:,k]
    for k in range(K):
        z = z_s + z_d[k]
        a = softplus(z)
        a_safe = np.maximum(a, _EPS)
        nk = n2d[:, k]
        ek = e2d[:, k]
        q += float(np.sum(nk * np.log(a_safe) - ek * a))
        coef = (nk / a_safe - ek) * sigmoid(z)
        grad_s += cand_phi.T @ coef
        coef_sum_per_combo[k] = float(coef.sum())
    grad_d = combo_bits.T @ coef_sum_per_combo   # (D,)
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    grad = np.concatenate([grad_s, grad_d]) - 2.0 * l2 * reg_mask * (w - w0)
    return q, grad


def fit_dynamic_weights_mstep(
    cand_phi: np.ndarray,
    combo_bits: np.ndarray,
    n2d: np.ndarray,
    e2d: np.ndarray,
    w_init: np.ndarray,
    *,
    l2: float = 1e-3,
    w_prior_mean: np.ndarray | None = None,
    max_iter: int = 50,
) -> np.ndarray:
    """M-step for dynamic (bucketed) α. w = [static (F), dynamic (D)]; α on row
    (candidate c, combo k) = softplus(cand_phi[c]·w_s + combo_bits[k]·w_d).

    cand_phi : (C, F)   static per-candidate features
    combo_bits : (K, D) one row per mark combo, its D dynamic-feature bits
    n2d, e2d : (C, K)   responsibility N and exposure E per (candidate, combo)
    """
    F = cand_phi.shape[1]
    D = combo_bits.shape[1]
    n_w = F + D
    w0 = np.zeros(n_w) if w_prior_mean is None else np.asarray(w_prior_mean, dtype=np.float64).reshape(-1)
    reg_mask = np.ones(n_w)
    reg_mask[0] = 0.0  # bias exempt
    w = np.asarray(w_init, dtype=np.float64).reshape(-1).copy()

    q, grad = _q_and_grad_dynamic(w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask)
    for _ in range(max_iter):
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-8:
            break
        t = 1.0
        c = 1e-4
        improved = False
        for _bt in range(30):
            w_new = w + t * grad
            q_new, grad_new = _q_and_grad_dynamic(w_new, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask)
            if q_new >= q + c * t * gnorm * gnorm:
                w, q, grad = w_new, q_new, grad_new
                improved = True
                break
            t *= 0.5
        if not improved:
            break
    return w


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
