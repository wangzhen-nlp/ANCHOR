"""Pair-feature construction for the feature-weighted MHP kernel.

Builds, for every modeled (target_type, source_type) candidate pair, a feature
vector φ(u, v) that depends only on device-/alarm-level attributes — NOT on
device identity. This is what lets α = softplus(w·φ) generalize to pairs (and
devices) never seen in training: as long as a new device's attributes are in
the NE graph, its φ is computable and the learned w applies.

Device attributes (manufacturer, ne_type, site, ...) are pulled from the NE
graph via ne_link_learning.core.build_graph_context — the established extractor
in this repo, with multi-field-name tolerance.

Candidate pairs = co-occurring pairs (windowed) ∪ topology-related pairs. Pairs
outside this set have no edge (α=0); the candidate set is what the kernel scores.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mhp.events import EventCollection
from mhp.em import _build_chunk_pair_arrays


class FeatureLayout:
    """Canonical φ(target, source) construction shared by training and
    inference, so the feature vector is byte-identical on both sides.

    Given per-pair attribute arrays (alarm-type ids, topology score, and the
    same-* booleans), produces the (C, F) feature matrix and the feature names.
    The layout is fully determined by the alarm-type vocabulary size n_at.
    """

    def __init__(self, at_vocab):
        self.at_vocab = list(at_vocab)
        self.n_at = max(len(self.at_vocab), 1)
        self.feature_names = self._names()

    def _names(self):
        names = ["bias"]
        for a in range(self.n_at):
            for b in range(self.n_at):
                names.append(f"at[{a}->{b}]")
        names += [
            "same_alarm_type",
            "topo_score",
            "is_same_ne",
            "same_site",
            "same_vendor",
            "same_ne_type",
            "topo_x_same_at",
            "topo_x_same_site",
        ]
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def build_matrix(
        self,
        at_u: np.ndarray,
        at_v: np.ndarray,
        topo: np.ndarray,
        is_same_ne: np.ndarray,
        same_site: np.ndarray,
        same_vendor: np.ndarray,
        same_netype: np.ndarray,
    ) -> np.ndarray:
        """All inputs are length-C arrays (at_* int, rest float/bool)."""
        C = len(at_u)
        cols = [np.ones(C)]
        for a in range(self.n_at):
            for b in range(self.n_at):
                cols.append(((at_u == a) & (at_v == b)).astype(np.float64))
        same_at = ((at_u == at_v) & (at_u >= 0)).astype(np.float64)
        topo = np.asarray(topo, dtype=np.float64)
        same_site = np.asarray(same_site, dtype=np.float64)
        cols += [
            same_at,
            topo,
            np.asarray(is_same_ne, dtype=np.float64),
            same_site,
            np.asarray(same_vendor, dtype=np.float64),
            np.asarray(same_netype, dtype=np.float64),
            topo * same_at,
            topo * same_site,
        ]
        return np.column_stack(cols)


class MuFeatureSpec:
    """Single-type features ψ(u) for the inductive immigrant baseline
    μ(u) = softplus(w_μ · ψ(u)).

    Uses only INDUCTIVE attributes of the type's device (alarm_type, ne_type,
    vendor, network domain) — deliberately NOT the type's own historical event
    count, which wouldn't generalize to new devices and would reintroduce
    per-device memorization. Categorical blocks are one-hot with capped vocabs;
    an unseen category at inference falls back to the bias + remaining blocks.
    """

    def __init__(self, at_vocab, ne_type_vocab, vendor_vocab, domain_vocab):
        self.at_vocab = list(at_vocab)
        self.ne_type_vocab = list(ne_type_vocab)
        self.vendor_vocab = list(vendor_vocab)
        self.domain_vocab = list(domain_vocab)
        self._at = {v: i for i, v in enumerate(self.at_vocab)}
        self._ne = {v: i for i, v in enumerate(self.ne_type_vocab)}
        self._ve = {v: i for i, v in enumerate(self.vendor_vocab)}
        self._dm = {v: i for i, v in enumerate(self.domain_vocab)}
        self.feature_names = (
            ["bias"]
            + [f"at={v}" for v in self.at_vocab]
            + [f"ne_type={v}" for v in self.ne_type_vocab]
            + [f"vendor={v}" for v in self.vendor_vocab]
            + [f"domain={v}" for v in self.domain_vocab]
        )

    @property
    def n_features(self):
        return len(self.feature_names)

    def build_matrix(self, ats, ne_types, vendors, domains):
        """Per-type attribute arrays (object/str) → (n, F) one-hot matrix."""
        n = len(ats)
        blocks = [np.ones((n, 1))]
        for vocab_map, vals in (
            (self._at, ats),
            (self._ne, ne_types),
            (self._ve, vendors),
            (self._dm, domains),
        ):
            blk = np.zeros((n, len(vocab_map)))
            for i, v in enumerate(vals):
                j = vocab_map.get(v)
                if j is not None:
                    blk[i, j] = 1.0
            blocks.append(blk)
        return np.column_stack(blocks)

    def build_row(self, at, ne_type, vendor, domain):
        return self.build_matrix([at], [ne_type], [vendor], [domain])[0]

    def to_dict(self):
        return {
            "at_vocab": self.at_vocab,
            "ne_type_vocab": self.ne_type_vocab,
            "vendor_vocab": self.vendor_vocab,
            "domain_vocab": self.domain_vocab,
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            payload.get("at_vocab", []),
            payload.get("ne_type_vocab", []),
            payload.get("vendor_vocab", []),
            payload.get("domain_vocab", []),
        )


def _capped_vocab(values, cap=50):
    """Top-`cap` distinct non-empty values by frequency (rest → fall to no-hot)."""
    from collections import Counter

    c = Counter(v for v in values if v)
    return [v for v, _ in c.most_common(cap)]


def build_mu_features(vocabs, type_fields, graph_context, *, cap=50):
    """Per-type μ feature matrix ψ (M, Fμ) + the MuFeatureSpec (for inference).

    Attributes per type: alarm_type (from label) + ne_type/vendor/domain_bucket
    (from the NE graph). Returns (psi, spec).
    """
    labels = vocabs.type_vocab.labels
    M = len(labels)
    src_idx, at_idx = _type_field_indices(type_fields)
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}

    ats, ne_types, vendors, domains = [], [], [], []
    for label in labels:
        ne_id, at = parse_label_ne_at(label, src_idx, at_idx)
        ats.append(at)
        info = node_infos.get(ne_id)
        ne_types.append((info.ne_type or "") if info is not None else "")
        vendors.append((info.manufacturer or "") if info is not None else "")
        domains.append((getattr(info, "domain_bucket", "") or "") if info is not None else "")

    spec = MuFeatureSpec(
        at_vocab=sorted({a for a in ats if a}),
        ne_type_vocab=_capped_vocab(ne_types, cap),
        vendor_vocab=_capped_vocab(vendors, cap),
        domain_vocab=_capped_vocab(domains, cap),
    )
    psi = spec.build_matrix(ats, ne_types, vendors, domains)
    return psi, spec


class RuntimeMuScorer:
    """Inference-time live μ(u) = softplus(w_μ · ψ(u)) for ANY type, including
    new devices — ψ built from the event's alarm_type + NE-graph attributes.
    """

    def __init__(self, mu_kernel, mu_spec: MuFeatureSpec, graph_context):
        from mhp.feature_kernel import softplus

        self.kernel = mu_kernel
        self.spec = mu_spec
        self._softplus = softplus
        self.node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
        if mu_spec.n_features != mu_kernel.n_features:
            raise ValueError(
                f"μ feature layout ({mu_spec.n_features}) != μ kernel ({mu_kernel.n_features})"
            )

    def mu_for(self, alarm_type, ne):
        info = self.node_infos.get(ne)
        ne_type = (info.ne_type or "") if info is not None else ""
        vendor = (info.manufacturer or "") if info is not None else ""
        domain = (getattr(info, "domain_bucket", "") or "") if info is not None else ""
        row = self.spec.build_row(alarm_type, ne_type, vendor, domain)
        return float(self.kernel.alpha(row[None, :])[0])


def _type_field_indices(type_fields):
    tf = tuple(type_fields)
    src_idx = tf.index("alarm_source") if "alarm_source" in tf else None
    at_idx = tf.index("alarm_type") if "alarm_type" in tf else None
    return src_idx, at_idx


def parse_label_ne_at(label, src_idx, at_idx):
    """Parse (ne, alarm_type) from a type label string, exactly as training does.

    The label is " | ".join of the type_fields values; ne/at are taken by the
    field positions. Used by BOTH training (vocab labels) and inference (label
    rebuilt from the raw event) so extraction is byte-identical on both sides.
    """
    parts = str(label).split(" | ")
    ne = parts[src_idx] if (src_idx is not None and len(parts) > src_idx) else ""
    at = parts[at_idx] if (at_idx is not None and len(parts) > at_idx) else ""
    return ne, at


def runtime_ne_at(alarm_event, type_fields):
    """Inference-time (ne, alarm_type) for a raw alarm event — reconstructs the
    SAME type label training built (via event_type_label) then parses it with
    the SAME field indices. Guarantees the device id / alarm type fed to the
    feature scorer match what training keyed on (stripping, custom type_fields,
    etc.), not an ad-hoc raw-field read.
    """
    from alarm_flow_isahp.sequences import event_type_label

    label = event_type_label(alarm_event, type_fields)
    src_idx, at_idx = _type_field_indices(type_fields)
    return parse_label_ne_at(label, src_idx, at_idx)


def _topo_score(source_ne, target_ne, topology_index, cache):
    """Directed topology relation score (source → target), cached per NE pair."""
    if topology_index is None or not source_ne or not target_ne:
        return 0.0
    key = (source_ne, target_ne)
    hit = cache.get(key)
    if hit is not None:
        return hit
    if source_ne == target_ne:
        score = 1.0
    else:
        feats = topology_index.pair_features(source_ne, target_ne)
        if not feats:
            score = 0.0
        elif feats[1] > 0:
            score = 1.0
        elif feats[2] > 0 or feats[3] > 0:
            score = 0.85
        elif feats[4] > 0:
            score = 0.75
        elif feats[5] > 0:
            score = 0.6
        elif feats[6] > 0:
            score = 0.45
        else:
            score = 0.0
    cache[key] = score
    return score


def _build_type_attributes(vocabs, type_fields, graph_context):
    """Per-type-id attribute arrays parsed from the vocab labels + NE graph.

    Returns a dict of arrays indexed by type_id:
      at_id (int, alarm-type index, -1 unknown), ne (object), site/vendor/netype
      (object), plus the alarm-type vocabulary list.
    """
    labels = vocabs.type_vocab.labels
    M = len(labels)
    type_fields = tuple(type_fields)
    src_idx, at_idx = _type_field_indices(type_fields)

    ne = np.empty(M, dtype=object)
    site = np.empty(M, dtype=object)
    vendor = np.empty(M, dtype=object)
    netype = np.empty(M, dtype=object)
    at_raw = np.empty(M, dtype=object)
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
    for tid, label in enumerate(labels):
        ne_id, at_val = parse_label_ne_at(label, src_idx, at_idx)
        ne[tid] = ne_id
        at_raw[tid] = at_val
        info = node_infos.get(ne_id)
        if info is not None:
            site[tid] = info.site_id or ""
            vendor[tid] = info.manufacturer or ""
            netype[tid] = info.ne_type or ""
        else:
            site[tid] = ""
            vendor[tid] = ""
            netype[tid] = ""

    # alarm-type vocabulary → small integer index
    at_vocab = sorted({str(a) for a in at_raw if a})
    at_to_id = {a: i for i, a in enumerate(at_vocab)}
    at_id = np.array([at_to_id.get(str(at_raw[t]), -1) for t in range(M)], dtype=np.int64)
    return {
        "ne": ne,
        "site": site,
        "vendor": vendor,
        "netype": netype,
        "at_id": at_id,
        "at_vocab": at_vocab,
    }


def _collect_cooccurred_pairs(events: EventCollection, window, max_hist, chunk_size, time_slack=0.0):
    """Distinct (target_type, source_type) flat keys that co-occur in a window."""
    M = events.M
    keys = set()
    N = events.n
    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        _, _, pair_dt, ptd, psd, _, _ = _build_chunk_pair_arrays(
            events.times, events.dims, cs, ce, window, max_hist, time_slack
        )
        if pair_dt.size == 0:
            continue
        flat = (ptd.astype(np.int64) * M + psd.astype(np.int64))
        keys.update(np.unique(flat).tolist())
    return keys


def build_candidate_features(
    events: EventCollection,
    vocabs,
    type_fields,
    *,
    topology_index=None,
    graph_context=None,
    history_window,
    max_history_events,
    chunk_size,
    time_slack=0.0,
    topo_max_hops=2,
    topo_min_score=0.0,
):
    """Build candidate (target, source) type pairs and their feature matrix.

    Candidates = co-occurring pairs ∪ topology-related pairs (among active
    types). Returns (cand_targets, cand_sources, phi (C,F), feature_names).
    """
    M = events.M
    attrs = _build_type_attributes(vocabs, type_fields, graph_context)
    at_id = attrs["at_id"]
    ne = attrs["ne"]
    site = attrs["site"]
    vendor = attrs["vendor"]
    netype = attrs["netype"]
    n_at = max(len(attrs["at_vocab"]), 1)

    # --- candidate pair set ---
    cooccur = _collect_cooccurred_pairs(
        events, history_window, max_history_events, chunk_size, time_slack
    )
    cand_keys = set(cooccur)

    # topology pairs: group active types by NE, cross same-NE + reachable NEs
    if topology_index is not None:
        ne_to_types = defaultdict(list)
        for tid in range(M):
            if ne[tid]:
                ne_to_types[ne[tid]].append(tid)
        undirected_hops = getattr(topology_index, "undirected_hops", {}) or {}
        topo_cache = {}
        for ne_id, tids in ne_to_types.items():
            # same-NE pairs
            for u in tids:
                for v in tids:
                    cand_keys.add(u * M + v)
            # cross-NE within hops
            for tgt_ne, hop in undirected_hops.get(ne_id, {}).items():
                if hop > topo_max_hops or tgt_ne == ne_id:
                    continue
                tgt_tids = ne_to_types.get(tgt_ne)
                if not tgt_tids:
                    continue
                # ne_id is SOURCE, tgt_ne is TARGET (source excites target)
                score = _topo_score(ne_id, tgt_ne, topology_index, topo_cache)
                if score < topo_min_score:
                    continue
                for u in tgt_tids:        # target
                    for v in tids:        # source
                        cand_keys.add(u * M + v)

    if not cand_keys:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 0)), [],
                list(attrs["at_vocab"]), at_id.copy(), np.zeros(0, np.float64))

    cand_flat = np.fromiter(cand_keys, dtype=np.int64, count=len(cand_keys))
    cand_flat.sort()
    cand_t = (cand_flat // M).astype(np.int64)
    cand_s = (cand_flat % M).astype(np.int64)
    C = len(cand_t)

    # --- feature matrix φ via the shared FeatureLayout ---
    layout = FeatureLayout(attrs["at_vocab"])
    topo_cache = {}
    topo_vec = np.array(
        [_topo_score(ne[cand_s[i]], ne[cand_t[i]], topology_index, topo_cache) for i in range(C)],
        dtype=np.float64,
    )

    def _same(arr):
        a = arr[cand_t]
        b = arr[cand_s]
        return np.array([1.0 if (a[i] and a[i] == b[i]) else 0.0 for i in range(C)], dtype=np.float64)

    phi = layout.build_matrix(
        at_u=at_id[cand_t],
        at_v=at_id[cand_s],
        topo=topo_vec,
        is_same_ne=(ne[cand_t] == ne[cand_s]),
        same_site=_same(site),
        same_vendor=_same(vendor),
        same_netype=_same(netype),
    )
    # topo_vec (C,) = per-candidate topology score, returned so the feature-mode
    # fit can apply it as a pseudo-count topology prior (device-parity).
    return cand_t, cand_s, phi, layout.feature_names, list(attrs["at_vocab"]), at_id.copy(), topo_vec


class RuntimeFeatureScorer:
    """Inference-time live α = softplus(w·φ) for ANY (target, source) pair,
    including pairs whose devices were never seen in training.

    φ is computed from the events' (alarm_type, NE) plus NE-graph attributes —
    so as long as a new device is in the NE graph (or even if not, degrading to
    alarm-type-only features), the kernel produces a sensible amplitude. This is
    the inductive generalization route (b): nothing is keyed by a training-time
    device vocabulary.
    """

    def __init__(self, kernel, at_vocab, graph_context, topology_index, beta: float, n_dynamic: int = 0):
        from mhp.feature_kernel import softplus

        self.kernel = kernel
        self.layout = FeatureLayout(at_vocab)
        self._softplus = softplus
        self.at_to_id = {str(a): i for i, a in enumerate(at_vocab)}
        self.node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
        self.topology_index = topology_index
        self.beta = float(beta)
        self._topo_cache = {}
        # Dynamic (stateful) α: the kernel carries n_dynamic extra weights after
        # the static features; the caller appends per-candidate mark bits to φ.
        self.n_dynamic = int(n_dynamic)
        if self.layout.n_features + self.n_dynamic != kernel.n_features:
            raise ValueError(
                f"feature layout ({self.layout.n_features}) + dynamic ({self.n_dynamic}) "
                f"!= kernel weights ({kernel.n_features}); artifact/feature mismatch"
            )

    def _attr(self, ne_id):
        info = self.node_infos.get(ne_id)
        if info is None:
            return ("", "", "")
        return (info.site_id or "", info.manufacturer or "", info.ne_type or "")

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None):
        """Vectorized α for one target vs a batch of source candidates.

        target_at/target_ne : scalars (alarm_type str, ne str)
        src_ats / src_nes   : lists of source alarm_type / ne
        src_marks : (n, n_dynamic) per-candidate source-mark bits (dynamic mode),
                    each row the source device's frozen uncleared-state booleans
                    at the candidate parent's fire time. Required iff n_dynamic>0.
        Returns (n,) α array.
        """
        n = len(src_nes)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        at_u = np.full(n, self.at_to_id.get(str(target_at), -1), dtype=np.int64)
        at_v = np.array([self.at_to_id.get(str(a), -1) for a in src_ats], dtype=np.int64)
        t_site, t_vendor, t_netype = self._attr(target_ne)
        topo = np.empty(n, dtype=np.float64)
        is_same_ne = np.empty(n, dtype=np.float64)
        same_site = np.empty(n, dtype=np.float64)
        same_vendor = np.empty(n, dtype=np.float64)
        same_netype = np.empty(n, dtype=np.float64)
        for i, sne in enumerate(src_nes):
            topo[i] = _topo_score(sne, target_ne, self.topology_index, self._topo_cache)
            is_same_ne[i] = 1.0 if sne == target_ne else 0.0
            s_site, s_vendor, s_netype = self._attr(sne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
        phi = self.layout.build_matrix(at_u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype)
        if self.n_dynamic > 0:
            if src_marks is None:
                raise ValueError("src_marks is required when RuntimeFeatureScorer.n_dynamic > 0")
            marks = np.asarray(src_marks, dtype=np.float64).reshape(n, self.n_dynamic)
            phi = np.concatenate([phi, marks], axis=1)
        return self.kernel.alpha(phi)
