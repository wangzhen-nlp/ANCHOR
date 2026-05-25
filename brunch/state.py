"""Branching state: event links B and cluster links C, with derived clusters/cascades.

Paper §2.2, §3.1, §3.2.

Convention
----------
event_parent[g] is the global event index of g's parent (or g itself = immigrant within-dim).
Event links are strictly within-dimension: events.dims[event_parent[g]] == events.dims[g].

cluster_parent[c] is the cluster id whose representative event triggered c's earliest event,
or -1 if c is a cascade root. Cluster links are cross-dimension by construction (within-dim
relationships are already captured by event_parent).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .events import EventCollection


class BranchingState:
    def __init__(self, events: EventCollection):
        self.events = events
        n = events.n
        self.event_parent = np.arange(n, dtype=np.int64)   # all immigrants initially
        self._dirty_clusters = True
        self._dirty_cascades = True
        self._cluster_of: Optional[np.ndarray] = None
        self._cluster_events: Optional[List[List[int]]] = None
        self._clusters_by_dim: Optional[List[List[int]]] = None
        self._cluster_parent: Optional[np.ndarray] = None
        self._cascade_of: Optional[np.ndarray] = None

    def init_nearest_in_window(self, window: float) -> None:
        """Warm-start event links: each event takes its nearest within-dim predecessor
        inside `window`. Builds non-trivial clusters from the start so cluster_link
        sampling has somewhere to begin.
        """
        for d in range(self.events.M):
            dim_idx = self.events.dim_indices(d)
            if len(dim_idx) < 2:
                continue
            t = self.events.times[dim_idx]
            for j in range(1, len(dim_idx)):
                dt = t[j] - t[j - 1]
                if 0 < dt < window:
                    self.event_parent[int(dim_idx[j])] = int(dim_idx[j - 1])
        self._dirty_clusters = True
        self._dirty_cascades = True

    # ---- event link mutation ----
    def set_event_parent(self, child: int, parent: int) -> None:
        if child == parent:
            self.event_parent[child] = child
        else:
            if self.events.dims[parent] != self.events.dims[child]:
                raise ValueError("event links must stay within a dimension")
            if self.events.times[parent] >= self.events.times[child]:
                raise ValueError("parent must strictly precede child in time")
            self.event_parent[child] = parent
        self._dirty_clusters = True
        self._dirty_cascades = True

    # ---- cluster derivation ----
    def _recompute_clusters(self) -> None:
        """Within-dim connected components of the event_parent forest.

        Uses path memoization so each event is visited O(1) amortized, regardless
        of how long the event_parent chain it sits on becomes. Total cost is
        O(n) instead of O(n · depth).
        """
        n = self.events.n
        event_parent = self.event_parent
        cluster_of = -np.ones(n, dtype=np.int64)
        cluster_events: List[List[int]] = []
        for g in range(n):
            if cluster_of[g] != -1:
                continue
            # Walk up event_parent until we hit a root (self-loop) or an
            # already-classified ancestor.
            path: List[int] = []
            cur = g
            while cluster_of[cur] == -1:
                parent = int(event_parent[cur])
                if parent == cur:
                    break  # within-dim immigrant: new cluster root
                path.append(cur)
                cur = parent
            if cluster_of[cur] == -1:
                cid = len(cluster_events)
                cluster_of[cur] = cid
                cluster_events.append([cur])
            cid = int(cluster_of[cur])
            for node in path:
                cluster_of[node] = cid
                cluster_events[cid].append(node)
        self._cluster_of = cluster_of
        self._cluster_events = cluster_events
        clusters_by_dim: List[List[int]] = [[] for _ in range(self.events.M)]
        # Per-cluster earliest/latest event time cached so outer_candidate_parents
        # can do O(1) admissibility checks instead of an O(size) np-min per cluster.
        K = len(cluster_events)
        cluster_earliest = np.empty(K, dtype=np.float64)
        cluster_latest = np.empty(K, dtype=np.float64)
        for cid, cluster in enumerate(cluster_events):
            times = self.events.times[cluster]
            cluster_earliest[cid] = float(times.min())
            cluster_latest[cid] = float(times.max())
            dim = int(self.events.dims[cluster[0]])
            clusters_by_dim[dim].append(cid)
        self._clusters_by_dim = clusters_by_dim
        self._cluster_earliest_arr = cluster_earliest
        self._cluster_latest_arr = cluster_latest
        # Reset cluster_parent (length and ids change when clusters restructure).
        self._cluster_parent = -np.ones(K, dtype=np.int64)
        self._dirty_clusters = False
        self._dirty_cascades = True

    def _ensure_clusters(self):
        if self._dirty_clusters:
            self._recompute_clusters()

    # ---- cluster link mutation ----
    def set_cluster_parent(self, child_cluster: int, parent_cluster: int) -> None:
        """parent_cluster == -1 means 'no parent' (cascade root)."""
        self._ensure_clusters()
        if parent_cluster == child_cluster:
            raise ValueError("self-loop cluster link is not allowed")
        if parent_cluster >= 0:
            # Cross-dim only — within-dim relationships are encoded in event links.
            p_dim = self.events.dims[self._cluster_events[parent_cluster][0]]
            c_dim = self.events.dims[self._cluster_events[child_cluster][0]]
            if p_dim == c_dim:
                raise ValueError("cluster links must be cross-dimension")
            # Source cluster must have at least one event preceding target's earliest.
            t_je = self.events.times[self._cluster_events[child_cluster]].min()
            src_times = self.events.times[self._cluster_events[parent_cluster]]
            if not np.any(src_times < t_je):
                raise ValueError("source cluster has no event preceding target's earliest")
        self._cluster_parent[child_cluster] = parent_cluster
        self._dirty_cascades = True

    # ---- cascade derivation ----
    def _recompute_cascades(self) -> None:
        """Connected components of the cluster_parent forest.

        Each cluster is visited O(1) amortized via path memoization. A small
        `seen` set guards against malformed cycles (should not occur in a
        valid BRUNCH state, but we keep the safety net).
        """
        self._ensure_clusters()
        K = len(self._cluster_events)
        cluster_parent = self._cluster_parent
        cascade_of_cluster = -np.ones(K, dtype=np.int64)
        n_cascades = 0
        for c in range(K):
            if cascade_of_cluster[c] != -1:
                continue
            path: List[int] = []
            cur = c
            seen = {cur}
            while cascade_of_cluster[cur] == -1:
                parent = int(cluster_parent[cur])
                if parent == -1:
                    break  # cascade root
                if parent in seen:
                    break  # cycle (defensive)
                path.append(cur)
                seen.add(parent)
                cur = parent
            if cascade_of_cluster[cur] == -1:
                cascade_of_cluster[cur] = n_cascades
                n_cascades += 1
            cid = int(cascade_of_cluster[cur])
            for node in path:
                cascade_of_cluster[node] = cid
        cascade_of = cascade_of_cluster[self._cluster_of]
        self._cascade_of = cascade_of
        self._cascade_of_cluster = cascade_of_cluster
        self._dirty_cascades = False

    def _ensure_cascades(self):
        if self._dirty_clusters or self._dirty_cascades:
            self._recompute_cascades()

    # ---- queries ----
    @property
    def cluster_of(self) -> np.ndarray:
        self._ensure_clusters()
        return self._cluster_of

    @property
    def num_clusters(self) -> int:
        self._ensure_clusters()
        return len(self._cluster_events)

    def cluster_events(self, c: int) -> List[int]:
        self._ensure_clusters()
        return self._cluster_events[c]

    def cluster_parent(self, c: int) -> int:
        self._ensure_clusters()
        return int(self._cluster_parent[c])

    def cluster_dim(self, c: int) -> int:
        self._ensure_clusters()
        return int(self.events.dims[self._cluster_events[c][0]])

    def clusters_in_dim(self, d: int) -> List[int]:
        self._ensure_clusters()
        return self._clusters_by_dim[int(d)]

    def cluster_earliest_time(self, c: int) -> float:
        self._ensure_clusters()
        return float(self._cluster_earliest_arr[c])

    def cluster_latest_time(self, c: int) -> float:
        self._ensure_clusters()
        return float(self._cluster_latest_arr[c])

    @property
    def cluster_earliest_array(self) -> np.ndarray:
        self._ensure_clusters()
        return self._cluster_earliest_arr

    @property
    def cluster_latest_array(self) -> np.ndarray:
        self._ensure_clusters()
        return self._cluster_latest_arr

    @property
    def cascade_of(self) -> np.ndarray:
        self._ensure_cascades()
        return self._cascade_of

    @property
    def num_cascades(self) -> int:
        self._ensure_cascades()
        return int(self._cascade_of.max() + 1) if len(self._cascade_of) else 0

    # ---- branching tree materialization ----
    def branching_edges(self) -> np.ndarray:
        """Edge list with rows (parent, child), combining event and cluster links.

        This is the sparse representation of the branching tree and should be
        preferred for large event sets.
        """
        self._ensure_cascades()
        n = self.events.n
        edges = []
        # within-dim event links
        for child in range(n):
            parent = int(self.event_parent[child])
            if parent != child:
                edges.append((parent, child))
        # cross-dim cluster links
        for c in range(self.num_clusters):
            pc = int(self._cluster_parent[c])
            if pc == -1:
                continue
            tgt_events = self._cluster_events[c]
            tgt_times = self.events.times[tgt_events]
            earliest = tgt_events[int(np.argmin(tgt_times))]
            t_je = float(self.events.times[earliest])
            src_events = np.asarray(self._cluster_events[pc])
            src_times = self.events.times[src_events]
            valid = src_times < t_je
            if not np.any(valid):
                continue
            idx = np.where(valid)[0]
            picked = int(src_events[idx[int(np.argmax(src_times[idx]))]])
            edges.append((picked, int(earliest)))
        return np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    def branching_matrix(self, params=None) -> np.ndarray:
        """Binary adjacency S ∈ {0,1}^{n×n} with S[parent, child] = 1.

        Combines event links (within-dim) and cluster links (cross-dim). For each
        cluster link s → g, the triggering event in s is chosen as the latest
        event in s whose time precedes g's earliest event (paper convention).
        """
        n = self.events.n
        S = np.zeros((n, n), dtype=np.uint8)
        edges = self.branching_edges()
        if len(edges):
            S[edges[:, 0], edges[:, 1]] = 1
        return S

    def parent_of(self) -> np.ndarray:
        """For each event, the global id of its trigger (self if immigrant).

        Combines event link and cluster link information.
        """
        self._ensure_cascades()
        n = self.events.n
        parent = self.event_parent.copy()
        for c in range(self.num_clusters):
            pc = int(self._cluster_parent[c])
            if pc == -1:
                continue
            tgt_events = self._cluster_events[c]
            tgt_times = self.events.times[tgt_events]
            earliest = int(tgt_events[int(np.argmin(tgt_times))])
            t_je = float(self.events.times[earliest])
            src_events = np.asarray(self._cluster_events[pc])
            src_times = self.events.times[src_events]
            valid = src_times < t_je
            if not np.any(valid):
                continue
            idx = np.where(valid)[0]
            picked = int(src_events[idx[int(np.argmax(src_times[idx]))]])
            parent[earliest] = picked
        return parent
