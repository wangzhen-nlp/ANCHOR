"""Event collection across M dimensions, with per-dim indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class EventCollection:
    """Global time-stamped events with dimension labels.

    Parameters
    ----------
    times : (n,) float
        Global timestamps, will be sorted ascending.
    dims  : (n,) int
        Dimension id in [0, M) for each event.
    M     : int
        Number of dimensions.
    T     : float, optional
        Observation horizon. If None, set to max(times) + small slack.
    """

    times: np.ndarray
    dims: np.ndarray
    M: int
    T: float = 0.0
    _dim_idx: List[np.ndarray] = field(default_factory=list, repr=False)

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple], M: int, T: float = 0.0):
        """Build from iterable of (time, dim) pairs."""
        times = np.asarray([p[0] for p in pairs], dtype=np.float64)
        dims = np.asarray([p[1] for p in pairs], dtype=np.int64)
        return cls(times=times, dims=dims, M=M, T=T)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.dims = np.asarray(self.dims, dtype=np.int64)
        if len(self.times) != len(self.dims):
            raise ValueError("times and dims must have the same length")
        if len(self.times) and (self.dims.min() < 0 or self.dims.max() >= self.M):
            raise ValueError("dims must be in [0, M)")
        # Sort by time (stable, so ties keep insertion order)
        if len(self.times) > 1 and np.any(np.diff(self.times) < 0):
            order = np.argsort(self.times, kind="stable")
            self.times = self.times[order]
            self.dims = self.dims[order]
        if not self.T:
            self.T = float(self.times[-1]) + 1e-6 if len(self.times) else 1.0
        # Per-dim indexing
        self._dim_idx = [np.where(self.dims == d)[0] for d in range(self.M)]

    @property
    def n(self) -> int:
        return int(len(self.times))

    def dim_indices(self, d: int) -> np.ndarray:
        return self._dim_idx[d]

    def dim_size(self, d: int) -> int:
        return int(len(self._dim_idx[d]))
