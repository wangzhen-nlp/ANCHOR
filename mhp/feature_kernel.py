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

# Keep mixed-precision BLAS work cache-sized.  Candidate features are normally
# float32 while weights / sufficient statistics are float64.  On NumPy builds
# backed by Accelerate (and several OpenBLAS builds), one giant mixed-dtype GEMV
# falls back to a much slower path.  Fixed-order row blocks both bound temporary
# memory and turn Hessian construction into efficient GEMM calls.
_MATMUL_BLOCK_ROWS = 4096
_MATMUL_WORK_BYTES = 8 * 1024 * 1024
# A dense (candidate, dynamic-feature) cross-curvature buffer is much faster
# than D separate feature reductions, but must not grow without bound on very
# large candidate sets.  Above this cap the code takes an algebraically
# equivalent low-memory path.
_DYNAMIC_CROSS_WORK_BYTES = 128 * 1024 * 1024


def _matmul_block_rows(*matrices, extra_bytes_per_row=0) -> int:
    """Choose a deterministic cache-sized row block for aligned matrices."""
    if not matrices:
        return 1
    n_rows = int(matrices[0].shape[0])
    bytes_per_row = int(extra_bytes_per_row)
    for matrix in matrices:
        if int(matrix.shape[0]) != n_rows:
            raise ValueError("matmul inputs must have the same row count")
        width = 1 if matrix.ndim == 1 else int(matrix.shape[1])
        bytes_per_row += width * int(matrix.dtype.itemsize)
    memory_rows = max(1, _MATMUL_WORK_BYTES // max(bytes_per_row, 1))
    return max(1, min(n_rows, _MATMUL_BLOCK_ROWS, memory_rows))


def _row_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute matrix @ vector, blocking the slow mixed-dtype case."""
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    if matrix.dtype == vector.dtype or matrix.shape[0] <= _MATMUL_BLOCK_ROWS:
        return matrix @ vector
    out = np.empty(matrix.shape[0], dtype=np.result_type(matrix, vector))
    block_rows = _matmul_block_rows(matrix)
    for start in range(0, matrix.shape[0], block_rows):
        stop = min(start + block_rows, matrix.shape[0])
        out[start:stop] = matrix[start:stop] @ vector
    return out


def _transpose_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute matrix.T @ vector with cache-sized mixed-dtype reductions."""
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    if matrix.dtype == vector.dtype or matrix.shape[0] <= _MATMUL_BLOCK_ROWS:
        return matrix.T @ vector
    out = np.zeros(matrix.shape[1], dtype=np.result_type(matrix, vector))
    block_rows = _matmul_block_rows(matrix, vector)
    for start in range(0, matrix.shape[0], block_rows):
        stop = min(start + block_rows, matrix.shape[0])
        out += matrix[start:stop].T @ vector[start:stop]
    return out


def _transpose_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute left.T @ right without a slow giant mixed-dtype GEMM."""
    left = np.asarray(left)
    right = np.asarray(right)
    if right.ndim != 2:
        raise ValueError("right must be a matrix")
    if left.dtype == right.dtype or left.shape[0] <= _MATMUL_BLOCK_ROWS:
        return left.T @ right
    out = np.zeros(
        (left.shape[1], right.shape[1]), dtype=np.result_type(left, right)
    )
    block_rows = _matmul_block_rows(left, right)
    for start in range(0, left.shape[0], block_rows):
        stop = min(start + block_rows, left.shape[0])
        out += left[start:stop].T @ right[start:stop]
    return out


def _weighted_gram(matrix: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    """Return matrix.T @ diag(row_weights) @ matrix via bounded GEMMs.

    The former implementation issued one full candidate-length GEMV per
    feature.  This reads the feature matrix F times.  Row blocking reads each
    block once and uses a level-3 BLAS operation, while keeping the temporary
    weighted block under ``_MATMUL_WORK_BYTES`` where possible.
    """
    matrix = np.asarray(matrix)
    row_weights = np.asarray(row_weights)
    n_rows, n_features = matrix.shape
    out_dtype = np.result_type(matrix, row_weights, np.float64)
    out = np.zeros((n_features, n_features), dtype=out_dtype)
    if n_rows == 0:
        return out
    weighted_itemsize = int(np.dtype(np.result_type(matrix, row_weights)).itemsize)
    block_rows = _matmul_block_rows(
        matrix,
        row_weights,
        extra_bytes_per_row=n_features * weighted_itemsize,
    )
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        block = matrix[start:stop]
        weighted = block * row_weights[start:stop, None]
        out += block.T @ weighted
    return out


def _zeroed_workspace_array(workspace, key, shape) -> np.ndarray:
    """Get a reusable float64 work array, zeroing it before each evaluation."""
    if workspace is None:
        return np.zeros(shape, dtype=np.float64)
    array = workspace.get(key)
    if array is None or array.shape != shape or array.dtype != np.float64:
        array = np.empty(shape, dtype=np.float64)
        workspace[key] = array
    array.fill(0.0)
    return array


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


def softplus_sigmoid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return softplus(z) and its sigmoid derivative with one activation pass.

    ``sigmoid(z) = 1 - exp(-softplus(z))``.  ``-expm1(-a)`` preserves accuracy
    when ``a`` is tiny, so this identity is stable at both tails and avoids the
    second exponential plus sign-mask temporaries in :func:`sigmoid`.
    """
    a = softplus(z)
    return a, -np.expm1(-a)


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
    alpha_scale : float
        Global multiplier applied to every materialized α (default 1.0 = no-op).
        Used by the opt-in spectral cap to enforce ρ ≤ target: since softplus is
        NOT scale-equivariant the cap cannot be folded into w, so it is stored as
        an explicit scalar applied here — the single point through which all
        inference α (target-side, source-side, dynamic boosts) flows.
    """

    weights: np.ndarray
    feature_names: list = field(default_factory=list)
    l2: float = 1e-3
    alpha_scale: float = 1.0

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.feature_names and len(self.feature_names) != len(self.weights):
            raise ValueError("feature_names length must match weights")

    @property
    def n_features(self) -> int:
        return int(len(self.weights))

    def alpha(self, phi: np.ndarray) -> np.ndarray:
        """α = alpha_scale · softplus(φ · w) for a (P, F) matrix → (P,) amplitudes."""
        if len(phi) == 0:
            return np.zeros(0, dtype=np.float64)
        a = softplus(phi @ self.weights)
        if self.alpha_scale != 1.0:
            a = a * self.alpha_scale
        return a

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.astype(float).tolist(),
            "feature_names": list(self.feature_names),
            "l2": float(self.l2),
            "alpha_scale": float(self.alpha_scale),
        }

    @classmethod
    def from_dict(cls, payload) -> "FeatureKernel":
        payload = dict(payload or {})
        return cls(
            weights=np.asarray(payload.get("weights", ()), dtype=np.float64),
            feature_names=list(payload.get("feature_names", [])),
            l2=float(payload.get("l2", 1e-3)),
            alpha_scale=float(payload.get("alpha_scale", 1.0)),
        )


def _newton_direction(H, grad, eye):
    """Ascent direction Δ solving (−H + μI)·Δ = grad for MAXIMIZATION, with
    Levenberg damping μ raised until −H+μI is positive-definite (Cholesky
    succeeds). Falls back to the diagonal-scaled gradient if that fails.
    """
    negH = -H
    diag = np.diag(negH).copy()
    base = max(1e-12, float(np.max(np.abs(diag))))
    mu = 0.0
    for _ in range(40):
        try:
            L = np.linalg.cholesky(negH + mu * eye)
        except np.linalg.LinAlgError:
            mu = base * 1e-8 if mu == 0.0 else mu * 10.0
            continue
        y = np.linalg.solve(L, grad)
        return np.linalg.solve(L.T, y)
    return grad / np.maximum(diag, _EPS)


def _q_only(w, phi, n_resp, exposure, l2, w0, reg_mask):
    """Objective Q(w) only — for the Armijo line search (skips the phiᵀ·coef matmul)."""
    z = _row_matvec(phi, w)
    a = softplus(z)
    a_safe = np.maximum(a, _EPS)
    return float(np.sum(n_resp * np.log(a_safe) - exposure * a) - l2 * np.sum(reg_mask * (w - w0) ** 2))


def _q_and_grad(w, phi, n_resp, exposure, l2, w0, reg_mask):
    """Objective Q(w) = Σ_c [N_c·log α_c − E_c·α_c] − λ·Σ reg·(w−w0)² and its
    gradient (to be MAXIMIZED).
    """
    z = _row_matvec(phi, w)
    a, s = softplus_sigmoid(z)
    a_safe = np.maximum(a, _EPS)
    q = float(np.sum(n_resp * np.log(a_safe) - exposure * a) - l2 * np.sum(reg_mask * (w - w0) ** 2))
    coef = (n_resp / a_safe - exposure) * s
    grad = _transpose_matvec(phi, coef) - 2.0 * l2 * reg_mask * (w - w0)
    return q, grad


def _q_grad_hess(w, phi, n_resp, exposure, l2, w0, reg_mask):
    """_q_and_grad plus the full F×F Hessian (for Newton): H = phiᵀ·diag(h)·phi
    − 2λ·reg, h = (N/α−E)·σ(1−σ) − N·σ²/α²."""
    F = phi.shape[1]
    z = _row_matvec(phi, w)
    a, s = softplus_sigmoid(z)
    a_safe = np.maximum(a, _EPS)
    q = float(np.sum(n_resp * np.log(a_safe) - exposure * a) - l2 * np.sum(reg_mask * (w - w0) ** 2))
    coef = (n_resp / a_safe - exposure) * s
    grad = _transpose_matvec(phi, coef) - 2.0 * l2 * reg_mask * (w - w0)
    h = (n_resp / a_safe - exposure) * s * (1.0 - s) - n_resp * (s * s) / (a_safe * a_safe)
    H = _weighted_gram(phi, h)
    H[np.arange(F), np.arange(F)] -= 2.0 * l2 * reg_mask
    return q, grad, H


def _with_extra(x, extra):
    return x if extra is None else x + extra


def _sparse_column(n2d, k):
    if hasattr(n2d, "shape"):
        return None
    idx, val = n2d[k]
    return np.asarray(idx, dtype=np.int64), np.asarray(val, dtype=np.float64)


def _q_dynamic(w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
               n0_extra=None, e0_extra=None):
    """Objective Q only (no gradient) — for the Armijo line search, which only
    needs to test the Q increase. Skips the expensive cand_phiᵀ·coef matmul.
    """
    F = cand_phi.shape[1]
    K = combo_bits.shape[0]
    ws, wd = w[:F], w[F:]
    z_s = _row_matvec(cand_phi, ws)         # (C,) one blocked matmul
    z_d = combo_bits @ wd                   # (K,)
    q = 0.0
    for k in range(K):
        z = z_s + z_d[k]
        a = softplus(z)
        a_safe = np.maximum(a, _EPS)
        ek = e2d[:, k]
        sp = _sparse_column(n2d, k)
        if sp is None:
            nk = n2d[:, k]
            q += float(np.sum(nk * np.log(a_safe)))
        else:
            idx, val = sp
            if len(idx):
                q += float(np.sum(val * np.log(a_safe[idx])))
        q -= float(np.sum(ek * a))
        if k == 0:
            if n0_extra is not None:
                q += float(np.sum(n0_extra * np.log(a_safe)))
            if e0_extra is not None:
                q -= float(np.sum(e0_extra * a))
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    return q


def _q_and_grad_dynamic(w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
                        n0_extra=None, e0_extra=None):
    """Bucketed objective Σ_{c,k}[N_{c,k}·log α_{c,k} − E_{c,k}·α_{c,k}] − ridge,
    with α_{c,k} = softplus(z_s[c] + z_d[k]), z_s = cand_phi·w_s, z_d = combo_bits·w_d
    and w = [w_s (F), w_d (D)]. Gradient decomposes over the static and dynamic
    blocks WITHOUT materializing the (C, K) row matrix. The static-block gradient
    collapses the K matmuls into ONE via linearity:
        ∂Q/∂w_s = cand_phiᵀ · Σ_k coef[:,k]   (one cand_phiᵀ·coef_total matmul)
        ∂Q/∂w_d = combo_bitsᵀ · (Σ_c coef[:,k])_k
    """
    F = cand_phi.shape[1]
    K, D = combo_bits.shape
    ws, wd = w[:F], w[F:]
    z_s = _row_matvec(cand_phi, ws)         # (C,) one blocked matmul
    z_d = combo_bits @ wd                   # (K,)
    q = 0.0
    coef_total = np.zeros(cand_phi.shape[0])   # Σ_k coef[:,k]  (C,)
    coef_sum_per_combo = np.zeros(K)           # Σ_c coef[:,k]
    for k in range(K):
        z = z_s + z_d[k]
        a, s = softplus_sigmoid(z)
        a_safe = np.maximum(a, _EPS)
        ek = e2d[:, k]
        coef = -ek * s
        q -= float(np.sum(ek * a))
        sp = _sparse_column(n2d, k)
        if sp is None:
            nk = n2d[:, k]
            q += float(np.sum(nk * np.log(a_safe)))
            coef += (nk / a_safe) * s
        else:
            idx, val = sp
            if len(idx):
                q += float(np.sum(val * np.log(a_safe[idx])))
                np.add.at(coef, idx, (val / a_safe[idx]) * s[idx])
        if k == 0:
            if n0_extra is not None:
                q += float(np.sum(n0_extra * np.log(a_safe)))
                coef += (n0_extra / a_safe) * s
            if e0_extra is not None:
                q -= float(np.sum(e0_extra * a))
                coef -= e0_extra * s
        coef_total += coef
        coef_sum_per_combo[k] = float(coef.sum())
    grad_s = _transpose_matvec(cand_phi, coef_total)  # ONE reduction instead of K
    grad_d = combo_bits.T @ coef_sum_per_combo   # (D,)
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    grad = np.concatenate([grad_s, grad_d]) - 2.0 * l2 * reg_mask * (w - w0)
    return q, grad


def _q_grad_hess_dynamic(w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
                         n0_extra=None, e0_extra=None, workspace=None):
    """_q_and_grad_dynamic plus the FULL (F+D)×(F+D) Hessian, for Newton steps.
    Per-(c,k) 2nd derivative w.r.t. z:  h = (N/α−E)·σ(1−σ) − N·σ²/α².
    Hessian blocks (no (C,K) materialization):
        H_ss = cand_phiᵀ·diag(Σ_k h)·cand_phi      (via blocked GEMM)
        H_dd = combo_bitsᵀ·diag(Σ_c h)·combo_bits
        H_sd = Σ_k (cand_phiᵀ·h[:,k]) ⊗ combo_bits[k]
    minus the ridge curvature 2λ·reg on the diagonal.
    """
    C, F = cand_phi.shape
    K, D = combo_bits.shape
    nw = F + D
    ws, wd = w[:F], w[F:]
    z_s = _row_matvec(cand_phi, ws)
    z_d = combo_bits @ wd
    q = 0.0
    coef_total = _zeroed_workspace_array(workspace, "dense_coef_total", (C,))
    coef_sum_per_combo = _zeroed_workspace_array(
        workspace, "dense_coef_per_combo", (K,)
    )
    h_total = _zeroed_workspace_array(workspace, "dense_h_total", (C,))
    h_per_combo = _zeroed_workspace_array(workspace, "dense_h_per_combo", (K,))
    # Accumulate the static×dynamic curvature by candidate, then issue one
    # blocked GEMM.  The old path performed one candidate-length GEMV per combo.
    use_cross_buffer = C * D * np.dtype(np.float64).itemsize <= _DYNAMIC_CROSS_WORK_BYTES
    if use_cross_buffer:
        hsd_by_candidate = _zeroed_workspace_array(
            workspace, "dense_hsd_by_candidate", (C, D)
        )
        Hsd = None
    else:
        hsd_by_candidate = None
        Hsd = np.zeros((F, D), dtype=np.float64)
    for k in range(K):
        z = z_s + z_d[k]
        a, s = softplus_sigmoid(z)
        a_safe = np.maximum(a, _EPS)
        ek = e2d[:, k]
        q -= float(np.sum(ek * a))
        coef = -ek * s
        h = -ek * s * (1.0 - s)
        sp = _sparse_column(n2d, k)
        if sp is None:
            nk = n2d[:, k]
            q += float(np.sum(nk * np.log(a_safe)))
            coef += (nk / a_safe) * s
            h += (nk / a_safe) * s * (1.0 - s) - nk * (s * s) / (a_safe * a_safe)
        else:
            idx, val = sp
            if len(idx):
                q += float(np.sum(val * np.log(a_safe[idx])))
                s_idx = s[idx]
                a_idx = a_safe[idx]
                np.add.at(coef, idx, (val / a_idx) * s_idx)
                np.add.at(
                    h,
                    idx,
                    (val / a_idx) * s_idx * (1.0 - s_idx)
                    - val * (s_idx * s_idx) / (a_idx * a_idx),
                )
        if k == 0:
            if n0_extra is not None:
                q += float(np.sum(n0_extra * np.log(a_safe)))
                coef += (n0_extra / a_safe) * s
                h += (n0_extra / a_safe) * s * (1.0 - s) - n0_extra * (s * s) / (a_safe * a_safe)
            if e0_extra is not None:
                q -= float(np.sum(e0_extra * a))
                coef -= e0_extra * s
                h -= e0_extra * s * (1.0 - s)
        coef_total += coef
        coef_sum_per_combo[k] = float(coef.sum())
        h_total += h
        h_per_combo[k] = float(h.sum())
        if use_cross_buffer:
            for d in np.flatnonzero(combo_bits[k]):
                hsd_by_candidate[:, d] += h * combo_bits[k, d]
        elif np.any(combo_bits[k]):
            # Low-memory fallback: reduce this combo immediately instead of
            # retaining C×D curvature.  K is only 8 on this dense path.
            Hsd += np.outer(_transpose_matvec(cand_phi, h), combo_bits[k])
    grad_s = _transpose_matvec(cand_phi, coef_total)
    grad_d = combo_bits.T @ coef_sum_per_combo
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    grad = np.concatenate([grad_s, grad_d]) - 2.0 * l2 * reg_mask * (w - w0)
    Hss = _weighted_gram(cand_phi, h_total)
    if use_cross_buffer:
        Hsd = _transpose_matmul(cand_phi, hsd_by_candidate)
    Hdd = combo_bits.T @ (combo_bits * h_per_combo[:, None])
    H = np.empty((nw, nw))
    H[:F, :F] = Hss
    H[:F, F:] = Hsd
    H[F:, :F] = Hsd.T
    H[F:, F:] = Hdd
    H[np.arange(nw), np.arange(nw)] -= 2.0 * l2 * reg_mask
    return q, grad, H


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
    progress=None,
    n0_extra: np.ndarray | None = None,
    e0_extra: np.ndarray | None = None,
) -> np.ndarray:
    """M-step for dynamic (bucketed) α. w = [static (F), dynamic (D)]; α on row
    (candidate c, combo k) = softplus(cand_phi[c]·w_s + combo_bits[k]·w_d).

    cand_phi : (C, F)   static per-candidate features
    combo_bits : (K, D) one row per mark combo, its D dynamic-feature bits
    n2d, e2d : (C, K)   responsibility N and exposure E per (candidate, combo)
    n0_extra, e0_extra : optional (C,) pseudo-counts added to combo 0 only,
        without materializing another (C, K) matrix.
    progress : optional callable(outer_iter, q, gnorm) for a heartbeat on large C.
    """
    F = cand_phi.shape[1]
    D = combo_bits.shape[1]
    n_w = F + D
    w0 = np.zeros(n_w) if w_prior_mean is None else np.asarray(w_prior_mean, dtype=np.float64).reshape(-1)
    reg_mask = np.ones(n_w)
    reg_mask[0] = 0.0  # bias exempt
    w = np.asarray(w_init, dtype=np.float64).reshape(-1).copy()
    n0_extra = None if n0_extra is None else np.asarray(n0_extra, dtype=np.float64).reshape(-1)
    e0_extra = None if e0_extra is None else np.asarray(e0_extra, dtype=np.float64).reshape(-1)
    if n0_extra is not None and len(n0_extra) != cand_phi.shape[0]:
        raise ValueError("n0_extra must be aligned to candidate rows")
    if e0_extra is not None and len(e0_extra) != cand_phi.shape[0]:
        raise ValueError("e0_extra must be aligned to candidate rows")

    # DAMPED NEWTON. The objective is badly conditioned (the unregularized bias
    # has a huge curvature from Σ E, the others tiny), so plain gradient ascent
    # crawls (50 iters). With only ~21 parameters the full Hessian is cheap;
    # Newton converges in a handful of iters to the SAME optimum. Levenberg
    # damping (−H + μI) keeps the step an ascent direction when the (non-canonical
    # softplus link) objective is locally non-concave.
    workspace = {}
    q, grad, H = _q_grad_hess_dynamic(
        w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
        n0_extra=n0_extra, e0_extra=e0_extra, workspace=workspace,
    )
    eye = np.eye(F + D)
    for _it in range(max_iter):
        if progress is not None:
            progress(_it, q, float(np.linalg.norm(grad)))
        direction = _newton_direction(H, grad, eye)
        dderiv = float(grad @ direction)          # Newton decrement ≈ 2·(Q* − Q)
        if dderiv <= 1e-9 * max(abs(q), 1.0):
            break  # near-optimal (scale-invariant test)
        t = 1.0
        c = 1e-4
        improved = False
        for _bt in range(40):
            w_new = w + t * direction
            q_new = _q_dynamic(
                w_new, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
                n0_extra=n0_extra, e0_extra=e0_extra,
            )
            if q_new >= q + c * t * dderiv:
                w, q = w_new, q_new
                q, grad, H = _q_grad_hess_dynamic(
                    w, cand_phi, combo_bits, n2d, e2d, l2, w0, reg_mask,
                    n0_extra=n0_extra, e0_extra=e0_extra, workspace=workspace,
                )
                improved = True
                break
            t *= 0.5
        if not improved:
            break
    return w


# --------------------------------------------------------------------------
# Flat-COO dynamic M-step (source_target / large K). When BOTH responsibility N
# and exposure E are sparse, the dense (C, K) K-loop wastes ~K× work on empty
# (candidate, combo) buckets. Here N and E are flat COO lists (rows, combos,
# vals); the objective Σ N·logα − E·α and the Hessian h = h_N + h_E DECOMPOSE
# additively, so the two lists are processed independently and combined via
# bincount into per-candidate / per-combo aggregates — no union merge needed.
# --------------------------------------------------------------------------


def _q_dynamic_coo(w, cand_phi, combo_bits, n_coo, e_coo, l2, w0, reg_mask):
    """Objective Q only (line search) over flat-COO N and E entries."""
    F = cand_phi.shape[1]
    z_s = _row_matvec(cand_phi, w[:F])
    z_d = combo_bits @ w[F:]
    nr, nc, nv = n_coo
    er, ec, ev = e_coo
    q = 0.0
    if nr.size:
        aN = np.maximum(softplus(z_s[nr] + z_d[nc]), _EPS)
        q += float(np.sum(nv * np.log(aN)))
    if er.size:
        aE = softplus(z_s[er] + z_d[ec])
        q -= float(np.sum(ev * aE))
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    return q


def _q_grad_hess_dynamic_coo(
    w, cand_phi, combo_bits, n_coo, e_coo, l2, w0, reg_mask, workspace=None
):
    """Q, gradient, and full Hessian over flat-COO N and E. Per-entry 2nd
    derivative decomposes: h_N = (N/α)σ(1−σ) − Nσ²/α² (N-entries),
    h_E = −E·σ(1−σ) (E-entries). Aggregated to per-candidate / per-combo via
    bincount; matmuls stay over C (not over the nnz)."""
    C, F = cand_phi.shape
    K, D = combo_bits.shape
    nw = F + D
    z_s = _row_matvec(cand_phi, w[:F])
    z_d = combo_bits @ w[F:]
    nr, nc, nv = n_coo
    er, ec, ev = e_coo
    coef_c = _zeroed_workspace_array(workspace, "coo_coef_c", (C,))
    coef_k = _zeroed_workspace_array(workspace, "coo_coef_k", (K,))
    h_c = _zeroed_workspace_array(workspace, "coo_h_c", (C,))
    h_k = _zeroed_workspace_array(workspace, "coo_h_k", (K,))
    use_cross_buffer = C * D * np.dtype(np.float64).itemsize <= _DYNAMIC_CROSS_WORK_BYTES
    if use_cross_buffer:
        hsd_by_candidate = _zeroed_workspace_array(
            workspace, "coo_hsd_by_candidate", (C, D)
        )
        Hsd = None
    else:
        hsd_by_candidate = None
        Hsd = np.zeros((F, D), dtype=np.float64)
    q = 0.0
    if nr.size:
        zN = z_s[nr] + z_d[nc]
        aN, sN = softplus_sigmoid(zN)
        aN = np.maximum(aN, _EPS)
        q += float(np.sum(nv * np.log(aN)))
        cN = (nv / aN) * sN
        hN = (nv / aN) * sN * (1.0 - sN) - nv * (sN * sN) / (aN * aN)
        coef_c += np.bincount(nr, cN, C)
        coef_k += np.bincount(nc, cN, K)
        h_c += np.bincount(nr, hN, C)
        h_k += np.bincount(nc, hN, K)
        for d in range(D):
            # Avoid materializing combo_bits[nc] as (nnz, D), which is huge for
            # source_target exposure. Build one weighted column at a time.
            hsd_col = np.bincount(nr, hN * combo_bits[nc, d], C)
            if use_cross_buffer:
                hsd_by_candidate[:, d] += hsd_col
            else:
                Hsd[:, d] += _transpose_matvec(cand_phi, hsd_col)
    if er.size:
        zE = z_s[er] + z_d[ec]
        aE, sE = softplus_sigmoid(zE)
        q -= float(np.sum(ev * aE))
        cE = -ev * sE
        hE = -ev * sE * (1.0 - sE)
        coef_c += np.bincount(er, cE, C)
        coef_k += np.bincount(ec, cE, K)
        h_c += np.bincount(er, hE, C)
        h_k += np.bincount(ec, hE, K)
        for d in range(D):
            hsd_col = np.bincount(er, hE * combo_bits[ec, d], C)
            if use_cross_buffer:
                hsd_by_candidate[:, d] += hsd_col
            else:
                Hsd[:, d] += _transpose_matvec(cand_phi, hsd_col)
    q -= l2 * float(np.sum(reg_mask * (w - w0) ** 2))
    grad = np.concatenate(
        [_transpose_matvec(cand_phi, coef_c), combo_bits.T @ coef_k]
    ) - 2.0 * l2 * reg_mask * (w - w0)
    Hss = _weighted_gram(cand_phi, h_c)
    if use_cross_buffer:
        Hsd = _transpose_matmul(cand_phi, hsd_by_candidate)
    Hdd = combo_bits.T @ (combo_bits * h_k[:, None])
    H = np.empty((nw, nw))
    H[:F, :F] = Hss
    H[:F, F:] = Hsd
    H[F:, :F] = Hsd.T
    H[F:, F:] = Hdd
    H[np.arange(nw), np.arange(nw)] -= 2.0 * l2 * reg_mask
    return q, grad, H


def fit_dynamic_weights_mstep_coo(
    cand_phi: np.ndarray,
    combo_bits: np.ndarray,
    n_coo: tuple,
    e_coo: tuple,
    w_init: np.ndarray,
    *,
    l2: float = 1e-3,
    w_prior_mean: np.ndarray | None = None,
    max_iter: int = 50,
    progress=None,
) -> np.ndarray:
    """Damped-Newton dynamic M-step over flat-COO N/E (see fit_dynamic_weights_mstep).
    n_coo / e_coo : (rows int, combos int, vals float) — any combo-0 pseudo-counts
    (topology prior) must already be appended by the caller.
    """
    F = cand_phi.shape[1]
    D = combo_bits.shape[1]
    n_w = F + D
    w0 = np.zeros(n_w) if w_prior_mean is None else np.asarray(w_prior_mean, dtype=np.float64).reshape(-1)
    reg_mask = np.ones(n_w)
    reg_mask[0] = 0.0
    w = np.asarray(w_init, dtype=np.float64).reshape(-1).copy()

    def _coo(coo):
        r, c, v = coo
        r = np.asarray(r)
        c = np.asarray(c)
        v = np.asarray(v)
        if not np.issubdtype(r.dtype, np.integer):
            r = r.astype(np.int64)
        if not np.issubdtype(c.dtype, np.integer):
            c = c.astype(np.int64)
        if not np.issubdtype(v.dtype, np.floating):
            v = v.astype(np.float64)
        return r, c, v

    n_coo = _coo(n_coo)
    e_coo = _coo(e_coo)
    workspace = {}
    q, grad, H = _q_grad_hess_dynamic_coo(
        w, cand_phi, combo_bits, n_coo, e_coo, l2, w0, reg_mask,
        workspace=workspace,
    )
    eye = np.eye(n_w)
    for _it in range(max_iter):
        if progress is not None:
            progress(_it, q, float(np.linalg.norm(grad)))
        direction = _newton_direction(H, grad, eye)
        dderiv = float(grad @ direction)
        if dderiv <= 1e-9 * max(abs(q), 1.0):
            break
        t = 1.0
        c = 1e-4
        improved = False
        for _bt in range(40):
            w_new = w + t * direction
            q_new = _q_dynamic_coo(w_new, cand_phi, combo_bits, n_coo, e_coo, l2, w0, reg_mask)
            if q_new >= q + c * t * dderiv:
                w, q = w_new, q_new
                q, grad, H = _q_grad_hess_dynamic_coo(
                    w, cand_phi, combo_bits, n_coo, e_coo, l2, w0, reg_mask,
                    workspace=workspace,
                )
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

    # Damped Newton — see fit_dynamic_weights_mstep.
    q, grad, H = _q_grad_hess(w, phi, n_resp, exposure, l2, w0, reg_mask)
    eye = np.eye(F)
    for _ in range(max_iter):
        direction = _newton_direction(H, grad, eye)
        dderiv = float(grad @ direction)
        if dderiv <= 1e-9 * max(abs(q), 1.0):
            break  # Newton decrement negligible → converged
        t = 1.0
        c = 1e-4
        improved = False
        for _bt in range(40):
            w_new = w + t * direction
            q_new = _q_only(w_new, phi, n_resp, exposure, l2, w0, reg_mask)
            if q_new >= q + c * t * dderiv:
                w, q = w_new, q_new
                q, grad, H = _q_grad_hess(w, phi, n_resp, exposure, l2, w0, reg_mask)
                improved = True
                break
            t *= 0.5
        if not improved:
            break  # line search failed → at (local) optimum for this M-step
    return w
