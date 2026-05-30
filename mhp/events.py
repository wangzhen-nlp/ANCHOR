"""Event sequence container for MHP fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EventCollection:
    """A finite multivariate point process observation.

    Attributes
    ----------
    times : np.ndarray, shape (N,)
        Event timestamps in monotonically non-decreasing order. Units must be
        consistent with MHPConfig.history_window.
    dims : np.ndarray, shape (N,)
        Event type IDs (int64), each in [0, M).
    M : int
        Number of types in the underlying multivariate process.
    T : float
        Observation horizon (typically times[-1] + epsilon). Used in M-step
        for μ normalization and in log-likelihood integral term.
    """

    times: np.ndarray
    dims: np.ndarray
    M: int
    T: float

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.dims = np.asarray(self.dims, dtype=np.int64)
        if self.times.shape != self.dims.shape:
            raise ValueError("times and dims must have the same shape")
        if len(self.times) and not np.all(np.diff(self.times) >= 0):
            raise ValueError("times must be sorted non-decreasing")
        if len(self.dims) and (self.dims.min() < 0 or self.dims.max() >= self.M):
            raise ValueError("dims must be in [0, M)")
        if self.T <= 0:
            raise ValueError("T must be positive")

    @property
    def n(self) -> int:
        return int(len(self.times))
