"""Sparse-storage Hawkes parameters for MHP.

Stores μ (M,), and α as sparse COO indexed by (target, source). β can be either
a single shared scalar (paper-standard for Morse MHP) or one value per edge.

The class is intentionally minimal: it exposes only what the EM loop and the
stream/run consumers need — lookup by (target, source), iteration over active
edges, and spectral radius for the stability cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MHPParams:
    """Parameters of a multivariate Hawkes process.

    α is stored sparsely (only edges with |α| > edge_threshold are kept).
    β can be a single scalar or a per-edge array — the EM driver chooses
    based on MHPConfig.beta_mode.
    """

    M: int
    mu: np.ndarray                                  # (M,)
    edge_targets: np.ndarray                        # (E,) int64
    edge_sources: np.ndarray                        # (E,) int64
    edge_alpha: np.ndarray                          # (E,) float64
    edge_beta: np.ndarray                           # (E,) float64
    edge_threshold: float = 1e-4
    max_active_sources_per_dim: Optional[int] = None
    beta_shared: bool = False                       # True → all edge_beta equal
    _lookup: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float64).reshape(-1)
        self.edge_targets = np.asarray(self.edge_targets, dtype=np.int64).reshape(-1)
        self.edge_sources = np.asarray(self.edge_sources, dtype=np.int64).reshape(-1)
        self.edge_alpha = np.asarray(self.edge_alpha, dtype=np.float64).reshape(-1)
        self.edge_beta = np.asarray(self.edge_beta, dtype=np.float64).reshape(-1)
        if not (
            len(self.edge_targets)
            == len(self.edge_sources)
            == len(self.edge_alpha)
            == len(self.edge_beta)
        ):
            raise ValueError("edge arrays must have matching length")
        if self.mu.shape[0] != self.M:
            raise ValueError(f"mu length {self.mu.shape[0]} != M {self.M}")
        if len(self.edge_targets):
            if self.edge_targets.min() < 0 or self.edge_targets.max() >= self.M:
                raise ValueError("edge_targets must be in [0, M)")
            if self.edge_sources.min() < 0 or self.edge_sources.max() >= self.M:
                raise ValueError("edge_sources must be in [0, M)")
            if np.any(self.edge_beta <= 0):
                raise ValueError("edge_beta must be positive")
        self._rebuild_lookup()

    def _rebuild_lookup(self):
        # (target, source) → edge index. Used by alpha_value/beta_value lookups
        # and by the stream candidate scoring loop.
        keys = self.edge_targets.astype(np.int64) * self.M + self.edge_sources.astype(np.int64)
        self._lookup = {int(k): int(i) for i, k in enumerate(keys)}

    @classmethod
    def from_edges(
        cls,
        M: int,
        mu: np.ndarray,
        edge_targets: np.ndarray,
        edge_sources: np.ndarray,
        edge_alpha: np.ndarray,
        edge_beta: np.ndarray,
        *,
        edge_threshold: float = 1e-4,
        max_active_sources_per_dim: Optional[int] = None,
        beta_shared: bool = False,
    ) -> "MHPParams":
        edge_targets = np.asarray(edge_targets, dtype=np.int64).reshape(-1)
        edge_sources = np.asarray(edge_sources, dtype=np.int64).reshape(-1)
        edge_alpha = np.asarray(edge_alpha, dtype=np.float64).reshape(-1)
        edge_beta = np.asarray(edge_beta, dtype=np.float64).reshape(-1)
        if edge_threshold > 0.0 and len(edge_alpha):
            keep = np.abs(edge_alpha) > edge_threshold
            edge_targets = edge_targets[keep]
            edge_sources = edge_sources[keep]
            edge_alpha = edge_alpha[keep]
            edge_beta = edge_beta[keep]
        # Stable order: sort by (target, source) so iteration by target is fast.
        if len(edge_targets):
            order = np.lexsort((edge_sources, edge_targets))
            edge_targets = edge_targets[order]
            edge_sources = edge_sources[order]
            edge_alpha = edge_alpha[order]
            edge_beta = edge_beta[order]
        return cls(
            M=M,
            mu=mu,
            edge_targets=edge_targets,
            edge_sources=edge_sources,
            edge_alpha=edge_alpha,
            edge_beta=edge_beta,
            edge_threshold=edge_threshold,
            max_active_sources_per_dim=max_active_sources_per_dim,
            beta_shared=beta_shared,
        )

    def alpha_value(self, target: int, source: int) -> float:
        idx = self._lookup.get(int(target) * self.M + int(source))
        return 0.0 if idx is None else float(self.edge_alpha[idx])

    def beta_value(self, target: int, source: int) -> float:
        idx = self._lookup.get(int(target) * self.M + int(source))
        if idx is None:
            # When edge is absent there's no kernel — return any positive
            # placeholder. The caller should gate on alpha_value first.
            return float(self.edge_beta[0]) if len(self.edge_beta) else 1.0
        return float(self.edge_beta[idx])

    def active_sources_for_target(self, target: int) -> np.ndarray:
        mask = self.edge_targets == int(target)
        return self.edge_sources[mask]

    def active_edges(self):
        """Return (targets, sources, alphas, betas) for all stored edges."""
        return self.edge_targets, self.edge_sources, self.edge_alpha, self.edge_beta

    def spectral_radius(self, *, power_iter: int = 80) -> float:
        """Power iteration over the sparse non-negative α matrix.

        Cheap O(E·power_iter) approximation; exact eigendecomposition would
        require dense M×M which is ~640 MB at M=8898.
        """
        if not len(self.edge_targets):
            return 0.0
        alpha_abs = np.abs(self.edge_alpha)
        v = np.ones(self.M, dtype=np.float64) / np.sqrt(self.M)
        prev = 0.0
        for _ in range(power_iter):
            # y[target] = sum over edges (target, source): α · v[source]
            y = np.zeros(self.M, dtype=np.float64)
            np.add.at(y, self.edge_targets, alpha_abs * v[self.edge_sources])
            nrm = float(np.linalg.norm(y))
            if nrm <= 1e-20:
                return 0.0
            v = y / nrm
            if abs(nrm - prev) / max(nrm, 1e-12) < 1e-6:
                break
            prev = nrm
        return float(prev)

    def copy(self) -> "MHPParams":
        return MHPParams(
            M=self.M,
            mu=self.mu.copy(),
            edge_targets=self.edge_targets.copy(),
            edge_sources=self.edge_sources.copy(),
            edge_alpha=self.edge_alpha.copy(),
            edge_beta=self.edge_beta.copy(),
            edge_threshold=self.edge_threshold,
            max_active_sources_per_dim=self.max_active_sources_per_dim,
            beta_shared=self.beta_shared,
        )
