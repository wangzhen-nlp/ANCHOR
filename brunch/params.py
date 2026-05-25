"""Hybrid MHP parameter container.

The class accepts the original dense ``alpha``/``beta`` matrices, but can also
store interactions as a sparse edge table:

    edge_targets[k], edge_sources[k], edge_alpha[k], edge_beta[k]

where each edge represents source dimension ``edge_sources[k]`` influencing
target dimension ``edge_targets[k]``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class HawkesParams:
    """Hybrid multivariate Hawkes parameters.

    Conventions
    -----------
    α, β are indexed [target_dim, source_dim] consistent with Eq. 2.1:
        λ_i(t) = F_i(μ_i + Σ_j Σ_{t_jl<t} α_ij φ_ij(t - t_jl))
    α may be negative when the i-th dim uses a non-linear link function.
    """

    def __init__(
        self,
        M: int,
        mu: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        links: Optional[List[str]] = None,
        edge_threshold: float = 0.0,
        max_active_sources_per_dim: Optional[int] = None,
        *,
        edge_targets: Optional[np.ndarray] = None,
        edge_sources: Optional[np.ndarray] = None,
        edge_alpha: Optional[np.ndarray] = None,
        edge_beta: Optional[np.ndarray] = None,
        sparse_storage: bool = False,
        default_beta: float = 1.0,
    ):
        self.M = int(M)
        self.mu = np.asarray(mu, dtype=np.float64).reshape(self.M)
        self.links = list(links) if links else ["linear"] * self.M
        self.edge_threshold = float(edge_threshold)
        self.max_active_sources_per_dim = max_active_sources_per_dim
        self.default_beta = float(default_beta)
        self._source_index_cache = None

        if self.M < 1:
            raise ValueError("M must be positive")
        if len(self.links) != self.M:
            raise ValueError("links must have length M")
        if np.any(self.mu < 0):
            raise ValueError("mu must be non-negative")
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold must be non-negative")
        if self.max_active_sources_per_dim is not None and self.max_active_sources_per_dim < 1:
            raise ValueError("max_active_sources_per_dim must be positive when set")
        if self.default_beta <= 0:
            raise ValueError("default_beta must be positive")

        has_edges = edge_targets is not None or edge_sources is not None or edge_alpha is not None or edge_beta is not None
        if has_edges:
            if any(v is None for v in (edge_targets, edge_sources, edge_alpha, edge_beta)):
                raise ValueError("edge_targets, edge_sources, edge_alpha, and edge_beta must be provided together")
            self._init_sparse(edge_targets, edge_sources, edge_alpha, edge_beta)
            self._alpha_dense = None
            self._beta_dense = None
            return

        if alpha is None or beta is None:
            raise ValueError("dense alpha/beta or sparse edge arrays must be provided")

        alpha_arr = np.asarray(alpha, dtype=np.float64).reshape(self.M, self.M)
        beta_arr = np.asarray(beta, dtype=np.float64).reshape(self.M, self.M)
        if np.any(beta_arr <= 0):
            raise ValueError("beta must be positive")

        if sparse_storage:
            targets, sources = self._dense_active_edges(alpha_arr)
            self._init_sparse(targets, sources, alpha_arr[targets, sources], beta_arr[targets, sources])
            self._alpha_dense = None
            self._beta_dense = None
        else:
            self._alpha_dense = alpha_arr
            self._beta_dense = beta_arr
            self._edge_targets = None
            self._edge_sources = None
            self._edge_alpha = None
            self._edge_beta = None
            self._target_offsets = None
            self._edge_lookup = None

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
        links: Optional[List[str]] = None,
        edge_threshold: float = 0.0,
        max_active_sources_per_dim: Optional[int] = None,
        default_beta: float = 1.0,
    ) -> "HawkesParams":
        return cls(
            M=M,
            mu=mu,
            links=links,
            edge_threshold=edge_threshold,
            max_active_sources_per_dim=max_active_sources_per_dim,
            edge_targets=edge_targets,
            edge_sources=edge_sources,
            edge_alpha=edge_alpha,
            edge_beta=edge_beta,
            default_beta=default_beta,
        )

    @classmethod
    def initial(
        cls,
        M: int,
        links: Optional[List[str]] = None,
        rng: Optional[np.random.Generator] = None,
        edge_threshold: float = 0.0,
        max_active_sources_per_dim: Optional[int] = None,
        sparse_storage: bool = False,
    ):
        rng = rng or np.random.default_rng()
        mu = rng.uniform(0.05, 0.2, size=M)
        if sparse_storage:
            # Sparse initialization never allocates M×M. If no top-k is supplied,
            # start from self edges only and let MCMC/MLE add observed edges.
            k = int(max_active_sources_per_dim) if max_active_sources_per_dim is not None else 0
            targets = []
            sources = []
            for i in range(M):
                src = set()
                if k > 0:
                    src.update(rng.choice(M, size=min(k, M), replace=False).astype(int).tolist())
                src.add(i)
                for j in sorted(src):
                    targets.append(i)
                    sources.append(j)
            edge_targets = np.asarray(targets, dtype=np.int64)
            edge_sources = np.asarray(sources, dtype=np.int64)
            edge_alpha = rng.uniform(0.05, 0.2, size=len(edge_targets))
            edge_beta = rng.uniform(0.5, 2.0, size=len(edge_targets))
            return cls.from_edges(
                M=M,
                mu=mu,
                edge_targets=edge_targets,
                edge_sources=edge_sources,
                edge_alpha=edge_alpha,
                edge_beta=edge_beta,
                links=links or ["linear"] * M,
                edge_threshold=edge_threshold,
                max_active_sources_per_dim=max_active_sources_per_dim,
            )

        alpha = rng.uniform(0.05, 0.2, size=(M, M))
        beta = rng.uniform(0.5, 2.0, size=(M, M))
        return cls(
            M=M,
            mu=mu,
            alpha=alpha,
            beta=beta,
            links=links or ["linear"] * M,
            edge_threshold=edge_threshold,
            max_active_sources_per_dim=max_active_sources_per_dim,
            sparse_storage=sparse_storage,
        )

    @property
    def is_sparse(self) -> bool:
        return self._alpha_dense is None

    def _dense_active_edges(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.edge_threshold > 0.0:
            mask = np.abs(alpha) > self.edge_threshold
        else:
            mask = alpha != 0.0
        return np.nonzero(mask)

    def _init_sparse(self, targets, sources, alpha, beta) -> None:
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        if not (len(targets) == len(sources) == len(alpha) == len(beta)):
            raise ValueError("sparse edge arrays must have the same length")
        if len(targets) and (targets.min() < 0 or targets.max() >= self.M or sources.min() < 0 or sources.max() >= self.M):
            raise ValueError("sparse edge dimensions must be in [0, M)")
        if np.any(beta <= 0):
            raise ValueError("edge_beta must be positive")

        if self.edge_threshold > 0.0:
            keep = np.abs(alpha) > self.edge_threshold
            targets, sources, alpha, beta = targets[keep], sources[keep], alpha[keep], beta[keep]

        if len(targets):
            order = np.lexsort((sources, targets))
            targets, sources, alpha, beta = targets[order], sources[order], alpha[order], beta[order]
            keys = targets * self.M + sources
            unique_keys, first, counts = np.unique(keys, return_index=True, return_counts=True)
            if np.any(counts > 1):
                # Keep the last occurrence in target/source sort order for duplicate edges.
                last = first + counts - 1
                targets, sources, alpha, beta = targets[last], sources[last], alpha[last], beta[last]
                keys = unique_keys
            self._edge_lookup = {int(k): idx for idx, k in enumerate(keys)}
        else:
            self._edge_lookup = {}

        self._edge_targets = targets
        self._edge_sources = sources
        self._edge_alpha = alpha
        self._edge_beta = beta
        self._target_offsets = np.searchsorted(targets, np.arange(self.M + 1), side="left")

    def copy(self) -> "HawkesParams":
        if self.is_sparse:
            return HawkesParams.from_edges(
                M=self.M,
                mu=self.mu.copy(),
                edge_targets=self._edge_targets.copy(),
                edge_sources=self._edge_sources.copy(),
                edge_alpha=self._edge_alpha.copy(),
                edge_beta=self._edge_beta.copy(),
                links=list(self.links),
                edge_threshold=self.edge_threshold,
                max_active_sources_per_dim=self.max_active_sources_per_dim,
                default_beta=self.default_beta,
            )
        return HawkesParams(
            M=self.M,
            mu=self.mu.copy(),
            alpha=self._alpha_dense.copy(),
            beta=self._beta_dense.copy(),
            links=list(self.links),
            edge_threshold=self.edge_threshold,
            max_active_sources_per_dim=self.max_active_sources_per_dim,
            default_beta=self.default_beta,
        )

    def as_sparse(self) -> "HawkesParams":
        if self.is_sparse:
            return self.copy()
        targets, sources = self._dense_active_edges(self._alpha_dense)
        return HawkesParams.from_edges(
            M=self.M,
            mu=self.mu.copy(),
            edge_targets=targets,
            edge_sources=sources,
            edge_alpha=self._alpha_dense[targets, sources],
            edge_beta=self._beta_dense[targets, sources],
            links=list(self.links),
            edge_threshold=self.edge_threshold,
            max_active_sources_per_dim=self.max_active_sources_per_dim,
            default_beta=self.default_beta,
        )

    def alpha_matrix(self) -> np.ndarray:
        if not self.is_sparse:
            return self._alpha_dense.copy()
        out = np.zeros((self.M, self.M), dtype=np.float64)
        out[self._edge_targets, self._edge_sources] = self._edge_alpha
        return out

    def beta_matrix(self) -> np.ndarray:
        if not self.is_sparse:
            return self._beta_dense.copy()
        out = np.full((self.M, self.M), self.default_beta, dtype=np.float64)
        out[self._edge_targets, self._edge_sources] = self._edge_beta
        return out

    @property
    def alpha(self) -> np.ndarray:
        """Dense α matrix view.

        This materializes an M×M array for sparse params; prefer edge methods in
        large-M code.
        """
        return self.alpha_matrix()

    @property
    def beta(self) -> np.ndarray:
        """Dense β matrix view. Prefer ``beta_value``/``edge_values`` for large M."""
        return self.beta_matrix()

    def alpha_value(self, target_dim: int, source_dim: int) -> float:
        i = int(target_dim)
        j = int(source_dim)
        if not self.is_sparse:
            return float(self._alpha_dense[i, j])
        idx = self._edge_lookup.get(i * self.M + j)
        return 0.0 if idx is None else float(self._edge_alpha[idx])

    def beta_value(self, target_dim: int, source_dim: int) -> float:
        i = int(target_dim)
        j = int(source_dim)
        if not self.is_sparse:
            return float(self._beta_dense[i, j])
        idx = self._edge_lookup.get(i * self.M + j)
        return self.default_beta if idx is None else float(self._edge_beta[idx])

    def active_edge_mask(self, *, include_self: bool = True) -> np.ndarray:
        """Dense boolean mask for compatibility and diagnostics."""
        mask = np.zeros((self.M, self.M), dtype=bool)
        targets, sources = self.active_edges(include_self=include_self)
        mask[targets, sources] = True
        return mask

    def active_edges(self, *, include_self: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return active edge arrays as (target_dims, source_dims)."""
        if self.is_sparse:
            targets = self._edge_targets
            sources = self._edge_sources
        else:
            targets, sources = self._dense_active_edges(self._alpha_dense)
        if not include_self and len(targets):
            keep = targets != sources
            return targets[keep].astype(np.int64, copy=False), sources[keep].astype(np.int64, copy=False)
        return targets.astype(np.int64, copy=False), sources.astype(np.int64, copy=False)

    def edge_values(self, *, include_self: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return active edges and their α/β values."""
        targets, sources = self.active_edges(include_self=include_self)
        if self.is_sparse:
            if include_self:
                alpha = self._edge_alpha
                beta = self._edge_beta
            else:
                keep = self._edge_targets != self._edge_sources
                alpha = self._edge_alpha[keep]
                beta = self._edge_beta[keep]
            return targets, sources, alpha.astype(np.float64, copy=False), beta.astype(np.float64, copy=False)
        return targets, sources, self._alpha_dense[targets, sources], self._beta_dense[targets, sources]

    def active_sources_for_target(self, target_dim: int, *, include_self: bool = True) -> np.ndarray:
        """Source dimensions with active influence into `target_dim`."""
        target_dim = int(target_dim)
        if self.is_sparse:
            start = int(self._target_offsets[target_dim])
            end = int(self._target_offsets[target_dim + 1])
            sources = self._edge_sources[start:end]
            if not include_self:
                sources = sources[sources != target_dim]
            return sources.astype(np.int64, copy=False)

        if self.edge_threshold > 0.0:
            mask = np.abs(self._alpha_dense[target_dim]) > self.edge_threshold
        else:
            mask = self._alpha_dense[target_dim] != 0.0
        if not include_self:
            mask = mask.copy()
            mask[target_dim] = False
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    def _source_index(self):
        if self._source_index_cache is not None:
            return self._source_index_cache
        targets, sources = self.active_edges(include_self=True)
        if not len(targets):
            order = np.asarray([], dtype=np.int64)
            offsets = np.zeros(self.M + 1, dtype=np.int64)
        else:
            order = np.argsort(sources, kind="stable")
            offsets = np.searchsorted(sources[order], np.arange(self.M + 1), side="left")
        self._source_index_cache = (targets, sources, order, offsets)
        return self._source_index_cache

    def active_targets_for_source(self, source_dim: int, *, include_self: bool = True) -> np.ndarray:
        """Target dimensions that an event in `source_dim` may trigger."""
        source_dim = int(source_dim)
        if self.is_sparse:
            targets, sources, order, offsets = self._source_index()
            start = int(offsets[source_dim])
            end = int(offsets[source_dim + 1])
            idx = order[start:end]
            out = targets[idx]
        else:
            col = self._alpha_dense[:, source_dim]
            if self.edge_threshold > 0.0:
                out = np.flatnonzero(np.abs(col) > self.edge_threshold)
            else:
                out = np.flatnonzero(col != 0.0)
        if not include_self:
            out = out[out != source_dim]
        return out.astype(np.int64, copy=False)

    def spectral_radius(self, *, max_exact_dim: int = 256, power_iter: int = 80) -> float:
        """Branching matrix spectral radius.

        Dense small matrices use an exact eigendecomposition. Sparse or large
        matrices use non-negative power iteration over active edges.
        """
        if not self.is_sparse and self.M <= max_exact_dim:
            alpha = np.abs(self._alpha_dense)
            if self.edge_threshold > 0.0:
                alpha = np.where(alpha > self.edge_threshold, alpha, 0.0)
            return float(np.max(np.abs(np.linalg.eigvals(alpha))))

        targets, sources, alpha, _ = self.edge_values(include_self=True)
        if len(targets) == 0:
            return 0.0
        values = np.abs(alpha)
        x = np.full(self.M, 1.0 / self.M, dtype=np.float64)
        y = np.zeros_like(x)
        radius = 0.0
        for _ in range(power_iter):
            y.fill(0.0)
            np.add.at(y, targets, values * x[sources])
            norm = float(np.linalg.norm(y, ord=np.inf))
            if norm <= 0.0 or not np.isfinite(norm):
                return 0.0
            x = y / norm
            radius = norm
        y.fill(0.0)
        np.add.at(y, targets, values * x[sources])
        denom = float(np.dot(x, x))
        if denom > 0.0:
            radius = float(np.dot(x, y) / denom)
        return max(radius, 0.0)
