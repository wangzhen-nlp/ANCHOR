"""High-level BRUNCH model: orchestrate intCRP priors + MEDIA inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .events import EventCollection
from .media import PARENT_SELECTION_MODES, run_media
from .params import HawkesParams
from .state import BranchingState


@dataclass
class BRUNCHConfig:
    M: int                                   # number of dimensions (event types)
    window: float = 10.0                     # intCRP time window W (paper §3.1)
    n_sweeps: int = 50                       # MEDIA MCMC sweeps
    burn_in: int = 10                        # burn-in before tracking best sample
    refit_params: bool = True                # MLE update of Θ between sweeps
    warm_start: bool = True                  # init each event linked to nearest in-window within-dim predecessor
    links: Optional[List[str]] = None        # per-dim link name ("linear"/"exp"/"softplus")
    sparse_alpha_threshold: float = 0.0      # ignore α entries at or below this magnitude
    max_active_sources_per_dim: Optional[int] = None  # optional top-k source dims per target after MLE
    sparse_parameter_storage: bool = False   # store α/β as sparse edge arrays
    materialize_branching_matrix: bool = True # set False to avoid dense n×n result allocation
    seed: int = 0
    verbose: bool = False
    log_every: int = 10
    progress_every: int = 50000
    parent_selection: str = "sample"

    def __post_init__(self):
        if self.M < 1:
            raise ValueError("M must be ≥ 1")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.sparse_alpha_threshold < 0:
            raise ValueError("sparse_alpha_threshold must be non-negative")
        if self.max_active_sources_per_dim is not None and self.max_active_sources_per_dim < 1:
            raise ValueError("max_active_sources_per_dim must be positive when set")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
        if self.parent_selection not in PARENT_SELECTION_MODES:
            raise ValueError(f"parent_selection must be one of {sorted(PARENT_SELECTION_MODES)}")
        if self.links is None:
            self.links = ["linear"] * self.M
        elif len(self.links) != self.M:
            raise ValueError("links must have length M")


@dataclass
class BRUNCHResult:
    events: EventCollection
    params: HawkesParams
    event_parent: np.ndarray            # within-dim parent for each event (self if immigrant)
    cluster_parent: Optional[np.ndarray]  # cluster_id → parent cluster_id (-1 = root)
    cluster_of: np.ndarray              # event → cluster_id
    cascade_of: np.ndarray              # event → cascade_id
    parent_of: np.ndarray               # event → trigger event id (combines event + cluster links)
    branching_edges: np.ndarray         # sparse rows (parent, child)
    branching_matrix: Optional[np.ndarray]  # binary (n × n) S, optional for large n
    trace: list = field(default_factory=list)
    best_log_likelihood: float = -np.inf


class BRUNCH:
    """Branching structure inference for hybrid MHPs (paper BRUNCH model)."""

    def __init__(self, config: BRUNCHConfig):
        self.config = config

    def fit(self, events: EventCollection, init_params: Optional[HawkesParams] = None) -> BRUNCHResult:
        cfg = self.config
        if events.M != cfg.M:
            raise ValueError(f"events.M={events.M} disagrees with config.M={cfg.M}")
        rng = np.random.default_rng(cfg.seed)
        if init_params is None:
            init_params = HawkesParams.initial(
                cfg.M,
                links=list(cfg.links),
                rng=rng,
                edge_threshold=cfg.sparse_alpha_threshold,
                max_active_sources_per_dim=cfg.max_active_sources_per_dim,
                sparse_storage=cfg.sparse_parameter_storage,
            )
        else:
            if init_params.M != cfg.M:
                raise ValueError("init_params.M disagrees with config.M")
            init_params = init_params.copy()
            init_params.edge_threshold = max(init_params.edge_threshold, cfg.sparse_alpha_threshold)
            if cfg.max_active_sources_per_dim is not None:
                init_params.max_active_sources_per_dim = cfg.max_active_sources_per_dim
            if cfg.sparse_parameter_storage:
                init_params = init_params.as_sparse()

        state = BranchingState(events)
        if cfg.warm_start:
            state.init_nearest_in_window(cfg.window)
        out = run_media(
            state,
            events,
            init_params,
            window=cfg.window,
            n_sweeps=cfg.n_sweeps,
            burn_in=cfg.burn_in,
            seed=cfg.seed,
            refit_params=cfg.refit_params,
            verbose=cfg.verbose,
            log_every=cfg.log_every,
            progress_every=cfg.progress_every,
            parent_selection=cfg.parent_selection,
        )
        # Reconstruct the BEST state to materialize derived structures.
        best_state = BranchingState(events)
        best_state.event_parent = out["event_parent"].copy()
        best_state._dirty_clusters = True
        best_state._dirty_cascades = True
        best_state._ensure_clusters()
        if out["cluster_parent"] is not None and len(out["cluster_parent"]) == best_state.num_clusters:
            best_state._cluster_parent = out["cluster_parent"].copy()
            best_state._dirty_cascades = True
        best_state._ensure_cascades()
        branching_edges = best_state.branching_edges()
        branching_matrix = best_state.branching_matrix() if cfg.materialize_branching_matrix else None
        return BRUNCHResult(
            events=events,
            params=out["params"],
            event_parent=best_state.event_parent.copy(),
            cluster_parent=best_state._cluster_parent.copy(),
            cluster_of=best_state.cluster_of.copy(),
            cascade_of=best_state.cascade_of.copy(),
            parent_of=best_state.parent_of(),
            branching_edges=branching_edges,
            branching_matrix=branching_matrix,
            trace=out["trace"],
            best_log_likelihood=out["best_log_likelihood"],
        )

    def fit_from_pairs(self, pairs: Sequence[tuple], T: float = 0.0, init_params: Optional[HawkesParams] = None) -> BRUNCHResult:
        events = EventCollection.from_pairs(pairs, M=self.config.M, T=T)
        return self.fit(events, init_params=init_params)
