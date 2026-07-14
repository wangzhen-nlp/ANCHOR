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


# An MHP "feature entity" identifies the device-agnostic node whose attributes
# define φ/ψ. In device mode it is just the topology node (the NE). In the
# site×domain mode the node (site) is NOT enough — two types at the same site
# but different device_domain must be distinct — so the entity folds the domain
# in: ``"{site}<SEP>{domain}"``. This keeps the sampler's ``(alarm_type, entity)``
# type-key a 2-tuple while still separating domains. The separator is the ASCII
# unit-separator, which cannot occur in site/NE ids.
ENTITY_SEP = "\x1f"


def make_entity(node, domain=""):
    """Compose a feature entity from a topology node id and an optional domain.

    ``domain`` empty → entity is the bare node (device-mode behavior, unchanged).
    """
    node = str(node or "")
    domain = str(domain or "")
    return f"{node}{ENTITY_SEP}{domain}" if domain else node


def split_entity(entity):
    """Inverse of :func:`make_entity`: ``entity -> (topo_node, domain)``."""
    s = str(entity or "")
    if ENTITY_SEP in s:
        node, domain = s.split(ENTITY_SEP, 1)
        return node, domain
    return s, ""


def topo_node_of(entity):
    """The topology-graph node (NE or site) an entity maps to — used for topo
    scoring / neighbor lookups, which are keyed by node, not by entity."""
    return split_entity(entity)[0]


def domain_of(entity, node_infos=None):
    """The entity's device domain. For composite (site×domain) entities it is the
    embedded domain; for bare nodes it falls back to the node's NE-graph
    ``domain_bucket`` (so device-mode μ keeps using the device's domain)."""
    node, dom = split_entity(entity)
    if dom:
        return dom
    info = (node_infos or {}).get(node)
    return (getattr(info, "domain_bucket", "") or "") if info is not None else ""


class FeatureLayout:
    """Canonical φ(target, source) construction shared by training and
    inference, so the feature vector is byte-identical on both sides.

    Given per-pair attribute arrays (alarm-type ids, topology score, and the
    same-* booleans), produces the (C, F) feature matrix and the feature names.
    The layout is fully determined by the alarm-type vocabulary size n_at.
    """

    def __init__(self, at_vocab, domain_vocab=()):
        self.at_vocab = list(at_vocab)
        self.n_at = max(len(self.at_vocab), 1)
        # Domain-pair features are OFF (empty vocab) unless device_domain is part
        # of the type — so device/NE-mode φ is byte-identical to the legacy layout.
        self.domain_vocab = list(domain_vocab)
        self.n_dom = len(self.domain_vocab)
        self._dom_to_id = {d: i for i, d in enumerate(self.domain_vocab)}
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
        if self.n_dom:
            names.append("same_domain")
            for a in range(self.n_dom):
                for b in range(self.n_dom):
                    names.append(f"dom[{a}->{b}]")
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def domain_ids(self, domains) -> np.ndarray:
        """Map a sequence of domain strings to layout ids (-1 = OOV / no domain)."""
        return np.array([self._dom_to_id.get(str(d), -1) for d in domains], dtype=np.int64)

    def build_matrix(
        self,
        at_u: np.ndarray,
        at_v: np.ndarray,
        topo: np.ndarray,
        is_same_ne: np.ndarray,
        same_site: np.ndarray,
        same_vendor: np.ndarray,
        same_netype: np.ndarray,
        dom_u: np.ndarray = None,
        dom_v: np.ndarray = None,
    ) -> np.ndarray:
        """All inputs are length-C arrays (at_*/dom_* int, rest float/bool).

        ``dom_u``/``dom_v`` are domain layout ids (see :meth:`domain_ids`); they
        are required iff this layout has a non-empty domain vocab.
        """
        # φ is built in float32 (halves the (C, F) matrix, the dominant memory block
        # at large candidate counts; 0/1 indicators + a topo score in [0,1] are
        # exactly/near-exactly representable, and φ·w promotes to float64 so the dot
        # product keeps full precision). Columns are written into a PREALLOCATED
        # matrix in place rather than column_stack'd, so the 18 column arrays and the
        # stacked output never coexist (which would ~double the peak). dom_u/dom_v
        # stay int64 (they are ids, not features).
        _F = np.float32
        C = len(at_u)
        phi = np.empty((C, self.n_features), dtype=_F)
        j = 0
        phi[:, j] = 1.0; j += 1
        for a in range(self.n_at):
            for b in range(self.n_at):
                phi[:, j] = (at_u == a) & (at_v == b); j += 1
        same_at = ((at_u == at_v) & (at_u >= 0)).astype(_F)
        topo = np.asarray(topo, dtype=_F)
        same_site = np.asarray(same_site, dtype=_F)
        phi[:, j] = same_at; j += 1
        phi[:, j] = topo; j += 1
        phi[:, j] = np.asarray(is_same_ne, dtype=_F); j += 1
        phi[:, j] = same_site; j += 1
        phi[:, j] = np.asarray(same_vendor, dtype=_F); j += 1
        phi[:, j] = np.asarray(same_netype, dtype=_F); j += 1
        phi[:, j] = topo * same_at; j += 1
        phi[:, j] = topo * same_site; j += 1
        if self.n_dom:
            if dom_u is None or dom_v is None:
                raise ValueError("domain ids required: this FeatureLayout has a domain vocab")
            dom_u = np.asarray(dom_u, dtype=np.int64)
            dom_v = np.asarray(dom_v, dtype=np.int64)
            phi[:, j] = (dom_u == dom_v) & (dom_u >= 0); j += 1
            for a in range(self.n_dom):
                for b in range(self.n_dom):
                    phi[:, j] = (dom_u == a) & (dom_v == b); j += 1
        return phi


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


class _NodeContext:
    """Minimal graph-context surface the feature pipeline consumes: ``node_infos``
    keyed by topology node, plus ``node_domains`` (node → domains present, used
    for site×domain missing-parent candidate enumeration)."""

    def __init__(self, node_infos, node_domains=None):
        self.node_infos = node_infos
        self.node_domains = dict(node_domains or {})


def _dominant(counter):
    """Most-frequent non-empty/non-MISSING key of a Counter, else ''."""
    best, best_n = "", -1
    for k, n in (counter or {}).items():
        if not k or k == "MISSING":
            continue
        if n > best_n:
            best, best_n = k, n
    return best


def build_node_context(ne_graph_data, node_field="alarm_source"):
    """Node-attribute context keyed by the topology node.

    device mode (``node_field='alarm_source'``): the NE-keyed GraphContext, as
    before. site mode (``node_field='site_id'``): a SITE-keyed context whose
    per-site attributes (ne_type, vendor, domain_bucket) are the site's dominant
    NE attributes, aggregated from the NE graph — no extra data needed. The
    site's set of present device domains is exposed via ``node_domains``.
    """
    from alarm_flow_isahp.event_domain import MODELED_DOMAINS
    from ne_link_learning.core import NodeInfo, build_graph_context

    gc = build_graph_context(ne_graph_data)
    if node_field != "site_id":
        return gc

    node_infos = {}
    node_domains = {}
    for site, ne_ids in (gc.site_to_nodes or {}).items():
        dom_counts = (gc.site_domain_bucket_counts or {}).get(site, {})
        # Missing-chain candidates must obey the same domain whitelist as
        # observed events; otherwise the sampler can synthesize site×OTHER
        # event types that training and streaming deliberately filtered out.
        node_domains[site] = sorted(d for d in dom_counts if d in MODELED_DOMAINS)
        lat_lon = (gc.site_coords or {}).get(site)
        node_infos[site] = NodeInfo(
            ne_id=site,
            site_id=site,
            site_name="",
            domain="",
            domain_bucket=_dominant(dom_counts),
            ne_type=_dominant((gc.site_type_counts or {}).get(site, {})),
            network_type=_dominant((gc.site_network_type_counts or {}).get(site, {})),
            manufacturer=_dominant((gc.site_manufacturer_counts or {}).get(site, {})),
            region_id="",
            latitude=(lat_lon[0] if lat_lon else None),
            longitude=(lat_lon[1] if lat_lon else None),
        )
    return _NodeContext(node_infos, node_domains)


def build_mu_features(vocabs, type_fields, graph_context, *, cap=50, node_field="alarm_source"):
    """Per-type μ feature matrix ψ (M, Fμ) + the MuFeatureSpec (for inference).

    Attributes per type: alarm_type (from label) + ne_type/vendor (from the node
    graph) + domain. The domain is the type's own device_domain when that is a
    type field (site×domain mode), else the node's NE-graph domain_bucket (so
    device-mode μ is unchanged). Returns (psi, spec).
    """
    labels = vocabs.type_vocab.labels
    M = len(labels)
    type_fields = tuple(type_fields)
    src_idx, at_idx = _type_field_indices(type_fields, node_field)
    dom_idx = type_fields.index("device_domain") if "device_domain" in type_fields else None
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}

    ats, ne_types, vendors, domains = [], [], [], []
    for label in labels:
        node_id, at = parse_label_ne_at(label, src_idx, at_idx)
        ats.append(at)
        info = node_infos.get(node_id)
        ne_types.append((info.ne_type or "") if info is not None else "")
        vendors.append((info.manufacturer or "") if info is not None else "")
        if dom_idx is not None:
            parts = str(label).split(" | ")
            dom_val = parts[dom_idx] if len(parts) > dom_idx else ""
            domains.append("" if dom_val == "<empty>" else dom_val)
        else:
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
        # ne is the feature ENTITY (topo node, optionally + domain). Node
        # attributes come from the topo node; domain from the entity (embedded
        # domain in site×domain mode, else the node's NE-graph domain_bucket).
        info = self.node_infos.get(topo_node_of(ne))
        ne_type = (info.ne_type or "") if info is not None else ""
        vendor = (info.manufacturer or "") if info is not None else ""
        domain = domain_of(ne, self.node_infos)
        row = self.spec.build_row(alarm_type, ne_type, vendor, domain)
        return float(self.kernel.alpha(row[None, :])[0])


def _type_field_indices(type_fields, node_field="alarm_source"):
    """Label positions of (topology-node field, alarm_type).

    ``node_field`` is the type field that identifies the topological entity —
    ``alarm_source`` (device) in the default mode, ``site_id`` for the
    site-level mode. It must be one of ``type_fields`` to be parseable from the
    label; otherwise the node index is None (→ empty node id at parse time).
    """
    tf = tuple(type_fields)
    src_idx = tf.index(node_field) if node_field in tf else None
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


def parse_label_entity_at(label, type_fields, node_field="alarm_source"):
    """Parse the feature entity and alarm type from a persisted type label.

    In site×domain mode the entity is ``site + domain``; in device mode it is
    the bare topology node. This is the label-side counterpart of
    :func:`runtime_ne_at` and is useful where only vocab labels are available.
    """
    type_fields = tuple(type_fields)
    src_idx, at_idx = _type_field_indices(type_fields, node_field)
    node, at = parse_label_ne_at(label, src_idx, at_idx)
    domain = ""
    if "device_domain" in type_fields:
        dom_idx = type_fields.index("device_domain")
        parts = str(label).split(" | ")
        dom_val = parts[dom_idx] if len(parts) > dom_idx else ""
        domain = "" if dom_val == "<empty>" else dom_val
    return make_entity(node, domain), at


def runtime_ne_at(alarm_event, type_fields, node_field="alarm_source"):
    """Inference-time (entity, alarm_type) for a raw alarm event — reconstructs
    the SAME type label training built (via event_type_label) then parses it with
    the SAME field indices. Returns the feature ENTITY (topology node, with the
    device_domain folded in when that is a type field — site×domain mode) so the
    id matches what training keyed on, byte-for-byte (stripping, custom
    type_fields, domain), not an ad-hoc raw-field read.

    ``node_field`` selects which type field is the topology node (``alarm_source``
    for device mode, ``site_id`` for site mode).
    """
    from alarm_flow_isahp.sequences import event_type_label

    type_fields = tuple(type_fields)
    label = event_type_label(alarm_event, type_fields)
    return parse_label_entity_at(label, type_fields, node_field)


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


def _factorize_attr(arr):
    """Map an (M,) object attribute array to int64 codes (−1 for empty/None) plus
    the code→value list. Lets per-candidate same-* comparisons run as vectorized
    integer ops on gathered codes instead of O(C) Python loops over object cells."""
    n = len(arr)
    codes = np.full(n, -1, dtype=np.int64)
    mapping = {}
    values = []
    for i in range(n):
        v = arr[i]
        s = "" if v is None else str(v)
        if not s:
            continue
        c = mapping.get(s)
        if c is None:
            c = len(values)
            mapping[s] = c
            values.append(s)
        codes[i] = c
    return codes, values


def _build_type_attributes(vocabs, type_fields, graph_context, node_field="alarm_source"):
    """Per-type-id attribute arrays parsed from the vocab labels + node graph.

    Returns a dict of arrays indexed by type_id:
      at_id (int, alarm-type index, -1 unknown), ne (object — the feature
      *entity*, i.e. topology node optionally folded with device_domain),
      site/vendor/netype (object), domain (object — label-sourced device_domain,
      empty unless device_domain ∈ type_fields), plus the alarm-type vocabulary
      and the φ domain vocabulary (empty in device mode → no domain φ columns).
    """
    labels = vocabs.type_vocab.labels
    M = len(labels)
    type_fields = tuple(type_fields)
    src_idx, at_idx = _type_field_indices(type_fields, node_field)
    dom_idx = type_fields.index("device_domain") if "device_domain" in type_fields else None

    ne = np.empty(M, dtype=object)        # feature entity (node[+domain])
    site = np.empty(M, dtype=object)
    vendor = np.empty(M, dtype=object)
    netype = np.empty(M, dtype=object)
    domain = np.empty(M, dtype=object)    # label-sourced domain (φ); "" in device mode
    at_raw = np.empty(M, dtype=object)
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
    for tid, label in enumerate(labels):
        node_id, at_val = parse_label_ne_at(label, src_idx, at_idx)
        dom_val = ""
        if dom_idx is not None:
            parts = str(label).split(" | ")
            dom_val = parts[dom_idx] if len(parts) > dom_idx else ""
            if dom_val == "<empty>":
                dom_val = ""
        ne[tid] = make_entity(node_id, dom_val)   # entity == node when domain empty
        at_raw[tid] = at_val
        domain[tid] = dom_val
        info = node_infos.get(node_id)            # node graph keyed by topo node
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
    # φ domain vocab: only the label-sourced domains → empty when device_domain is
    # not a type field, so device/NE-mode φ keeps the legacy (no-domain) layout.
    domain_vocab = sorted({str(d) for d in domain if d})
    return {
        "ne": ne,
        "site": site,
        "vendor": vendor,
        "netype": netype,
        "domain": domain,
        "at_id": at_id,
        "at_vocab": at_vocab,
        "domain_vocab": domain_vocab,
    }


def _collect_cooccurred_pairs(events: EventCollection, window, max_hist, chunk_size, time_slack=0.0):
    """Distinct (target_type, source_type) flat keys that co-occur in a window, as
    a sorted int64 array.

    Per-chunk uniques are folded into a running global-unique array, collapsed
    whenever the pending batch grows to ~its current size. Type-pairs repeat
    heavily across chunks, so this keeps peak memory bounded by the RESULT size
    (8 bytes/key) rather than the sum over all chunks — and avoids the ~50 bytes/
    key overhead of a Python set."""
    M = events.M
    N = events.n
    acc = np.zeros(0, dtype=np.int64)        # global-unique so far (sorted)
    pending = []                             # chunk uniques not yet folded in
    pending_n = 0
    floor = 2_000_000                        # don't collapse for trivially small batches
    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        _, _, pair_dt, ptd, psd, _, _ = _build_chunk_pair_arrays(
            events.times, events.dims, cs, ce, window, max_hist, time_slack
        )
        if pair_dt.size == 0:
            continue
        pending.append(np.unique(ptd.astype(np.int64) * M + psd.astype(np.int64)))
        pending_n += pending[-1].size
        if pending_n >= max(acc.size, floor):
            acc = np.unique(np.concatenate([acc] + pending))
            pending = []
            pending_n = 0
    if pending:
        acc = np.unique(np.concatenate([acc] + pending))
    return acc


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
    node_field="alarm_source",
):
    """Build candidate (target, source) type pairs and their feature matrix.

    Candidates = co-occurring pairs ∪ topology-related pairs (among active
    types). Returns (cand_targets, cand_sources, phi (C,F), feature_names,
    at_vocab, at_id, topo_vec, domain_vocab).
    """
    M = events.M
    attrs = _build_type_attributes(vocabs, type_fields, graph_context, node_field)
    at_id = attrs["at_id"]
    ne = attrs["ne"]
    site = attrs["site"]
    vendor = attrs["vendor"]
    netype = attrs["netype"]
    domain_vocab = attrs["domain_vocab"]
    layout = FeatureLayout(attrs["at_vocab"], domain_vocab)
    dom_layout_id = layout.domain_ids(attrs["domain"])    # per-type domain φ id

    # --- candidate pair set ---
    # Flat keys (target*M + source) accumulated as numpy arrays, deduped once at
    # the end via np.unique — avoids a tens-of-millions-element Python set.
    # _collect_cooccurred_pairs already returns int64; it goes straight into the
    # list with no separate named reference (a `cooccur = ...` binding would keep
    # the co-occurrence array alive through the φ build even after `del flat_parts`).
    flat_parts = [_collect_cooccurred_pairs(
        events, history_window, max_history_events, chunk_size, time_slack
    )]

    # topology pairs: group active types by TOPOLOGY NODE (NE, or site in
    # site×domain mode — entities at the same node, even across domains, are
    # co-located), cross same-node + reachable nodes within hops. Each group's
    # pairs are built as a vectorized cartesian product rather than per-pair adds.
    if topology_index is not None:
        node_to_types = defaultdict(list)
        for tid in range(M):
            node = topo_node_of(ne[tid])
            if node:
                node_to_types[node].append(tid)
        undirected_hops = getattr(topology_index, "undirected_hops", {}) or {}
        topo_cache = {}
        for node_id, tids in node_to_types.items():
            tids_arr = np.asarray(tids, dtype=np.int64)
            # same-node pairs (includes cross-domain at the same site)
            flat_parts.append((tids_arr[:, None] * M + tids_arr[None, :]).ravel())
            # cross-node within hops
            for tgt_node, hop in undirected_hops.get(node_id, {}).items():
                if hop > topo_max_hops or tgt_node == node_id:
                    continue
                tgt_tids = node_to_types.get(tgt_node)
                if not tgt_tids:
                    continue
                # node_id is SOURCE, tgt_node is TARGET (source excites target)
                score = _topo_score(node_id, tgt_node, topology_index, topo_cache)
                if score < topo_min_score:
                    continue
                tgt_arr = np.asarray(tgt_tids, dtype=np.int64)      # target
                flat_parts.append((tgt_arr[:, None] * M + tids_arr[None, :]).ravel())

    cand_flat = np.unique(np.concatenate(flat_parts)) if flat_parts else np.zeros(0, np.int64)
    if cand_flat.size == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 0), np.float32), [],
                list(attrs["at_vocab"]), at_id.copy(), np.zeros(0, np.float64), list(domain_vocab))

    cand_t = (cand_flat // M).astype(np.int64)
    cand_s = (cand_flat % M).astype(np.int64)
    C = len(cand_t)
    del cand_flat, flat_parts                        # candidate keys no longer needed

    # --- feature matrix φ via the shared FeatureLayout ---
    # Vectorized topo score: evaluate _topo_score once per UNIQUE (src_node,
    # tgt_node) pair, then gather to candidates (was an O(C) Python loop). The
    # C-length scratch (pair key, reverse index) is freed before the φ build so it
    # does not coexist with the (C, F) matrix.
    node_str = np.empty(M, dtype=object)
    for t in range(M):
        node_str[t] = topo_node_of(ne[t])
    node_codes, node_vals = _factorize_attr(node_str)   # (M,) — node arrays are small
    base = len(node_vals) + 2
    pair_key = (node_codes[cand_s] + 1) * base + (node_codes[cand_t] + 1)   # shifted codes (−1→0)
    uniq, inv = np.unique(pair_key, return_inverse=True)
    del pair_key
    topo_cache = {}
    uniq_scores = np.empty(len(uniq), dtype=np.float64)
    for j in range(len(uniq)):
        pk = int(uniq[j])
        s_code = pk // base - 1
        t_code = pk % base - 1
        s_node = node_vals[s_code] if s_code >= 0 else ""
        t_node = node_vals[t_code] if t_code >= 0 else ""
        uniq_scores[j] = _topo_score(s_node, t_node, topology_index, topo_cache)
    topo_vec = uniq_scores[inv]                      # float64 (also returned as the topo prior)
    del uniq, inv, uniq_scores

    # Vectorized same-* : compare gathered integer codes (−1 = empty), reproducing
    # the old "a is truthy AND a == b" semantics as (code_t >= 0) & (code_t == code_s).
    # The per-type code arrays are length M (small); the gathered (C,) compares are
    # transient inside build_matrix.
    site_codes, _ = _factorize_attr(site)
    vendor_codes, _ = _factorize_attr(vendor)
    netype_codes, _ = _factorize_attr(netype)

    def _same(codes):
        a = codes[cand_t]
        b = codes[cand_s]
        return ((a >= 0) & (a == b)).astype(np.float32)

    phi = layout.build_matrix(
        at_u=at_id[cand_t],
        at_v=at_id[cand_s],
        topo=topo_vec,
        is_same_ne=(ne[cand_t] == ne[cand_s]),
        same_site=_same(site_codes),
        same_vendor=_same(vendor_codes),
        same_netype=_same(netype_codes),
        dom_u=dom_layout_id[cand_t] if layout.n_dom else None,
        dom_v=dom_layout_id[cand_s] if layout.n_dom else None,
    )
    # topo_vec (C,) = per-candidate topology score, returned so the feature-mode
    # fit can apply it as a pseudo-count topology prior (device-parity).
    return (cand_t, cand_s, phi, layout.feature_names, list(attrs["at_vocab"]),
            at_id.copy(), topo_vec, list(domain_vocab))


class RuntimeFeatureScorer:
    """Inference-time live α = softplus(w·φ) for ANY (target, source) pair,
    including pairs whose devices were never seen in training.

    φ is computed from the events' (alarm_type, NE) plus NE-graph attributes —
    so as long as a new device is in the NE graph (or even if not, degrading to
    alarm-type-only features), the kernel produces a sensible amplitude. This is
    the inductive generalization route (b): nothing is keyed by a training-time
    device vocabulary.
    """

    def __init__(self, kernel, at_vocab, graph_context, topology_index, beta: float,
                 n_dynamic: int = 0, domain_vocab=(), node_domains=None,
                 dynamic_mode: str | None = None):
        from mhp.feature_kernel import softplus

        self.kernel = kernel
        self.layout = FeatureLayout(at_vocab, domain_vocab)
        self._softplus = softplus
        self.at_to_id = {str(a): i for i, a in enumerate(at_vocab)}
        self.node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
        self.topology_index = topology_index
        self.beta = float(beta)
        self._topo_cache = {}
        # topo node -> domains present (for site×domain missing-parent candidate
        # enumeration). Empty → device mode (entity == node, single implicit domain).
        self.node_domains = dict(node_domains or {})
        # Dynamic (stateful) α: the kernel carries n_dynamic extra weights after
        # the static features; the caller appends per-candidate mark bits to φ.
        self.n_dynamic = int(n_dynamic)
        if dynamic_mode is None:
            # Backward-compatible inference for direct callers and old tests.
            dynamic_mode = "source_target" if self.n_dynamic == 6 else (
                "source" if self.n_dynamic else "off"
            )
        self.dynamic_mode = str(dynamic_mode)
        expected_dynamic = {
            "off": 0,
            "source": 3,
            "target": 3,
            "source_target": 6,
        }
        if self.dynamic_mode not in expected_dynamic:
            raise ValueError(f"unknown dynamic_mode={self.dynamic_mode!r}")
        if self.n_dynamic != expected_dynamic[self.dynamic_mode]:
            raise ValueError(
                f"dynamic_mode={self.dynamic_mode!r} requires "
                f"n_dynamic={expected_dynamic[self.dynamic_mode]}, got {self.n_dynamic}"
            )
        self.source_dynamic_dim = 3 if self.dynamic_mode in {"source", "source_target"} else 0
        self.target_dynamic_dim = 3 if self.dynamic_mode in {"target", "source_target"} else 0
        if self.layout.n_features + self.n_dynamic != kernel.n_features:
            raise ValueError(
                f"feature layout ({self.layout.n_features}) + dynamic ({self.n_dynamic}) "
                f"!= kernel weights ({kernel.n_features}); artifact/feature mismatch"
            )

    def _attr(self, entity):
        """(site, vendor, ne_type) for a feature entity, via its topology node."""
        info = self.node_infos.get(topo_node_of(entity))
        if info is None:
            return ("", "", "")
        return (info.site_id or "", info.manufacturer or "", info.ne_type or "")

    def _dynamic_mark_matrix(self, n: int, src_marks=None, tgt_marks=None) -> np.ndarray:
        if self.n_dynamic <= 0:
            return np.zeros((n, 0), dtype=np.float64)

        def _rows(marks, width):
            if width <= 0:
                return np.zeros((n, 0), dtype=np.float64)
            if marks is None:
                return np.zeros((n, width), dtype=np.float64)
            arr = np.asarray(marks, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != n:
                arr = np.tile(arr.reshape(1, -1), (n, 1))
            out = np.zeros((n, width), dtype=np.float64)
            out[:, : min(arr.shape[1], width)] = arr[:, :width]
            return out

        src = _rows(src_marks, self.source_dynamic_dim)
        tgt = _rows(tgt_marks, self.target_dynamic_dim)
        return np.concatenate([src, tgt], axis=1)

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None, tgt_marks=None):
        """Vectorized α for one target vs a batch of source candidates.

        target_at/target_ne : scalars (alarm_type str, ne str)
        src_ats / src_nes   : lists of source alarm_type / ne
        src_marks : source-device state bits, required by source modes.
        tgt_marks : target pre-state bits, required by target modes.
        Returns (n,) α array.
        """
        n = len(src_nes)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        t_node = topo_node_of(target_ne)
        at_u = np.full(n, self.at_to_id.get(str(target_at), -1), dtype=np.int64)
        at_v = np.array([self.at_to_id.get(str(a), -1) for a in src_ats], dtype=np.int64)
        t_site, t_vendor, t_netype = self._attr(target_ne)
        topo = np.empty(n, dtype=np.float64)
        is_same_ne = np.empty(n, dtype=np.float64)
        same_site = np.empty(n, dtype=np.float64)
        same_vendor = np.empty(n, dtype=np.float64)
        same_netype = np.empty(n, dtype=np.float64)
        for i, sne in enumerate(src_nes):
            topo[i] = _topo_score(topo_node_of(sne), t_node, self.topology_index, self._topo_cache)
            is_same_ne[i] = 1.0 if sne == target_ne else 0.0
            s_site, s_vendor, s_netype = self._attr(sne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
        dom_u = dom_v = None
        if self.layout.n_dom:
            dom_u = np.full(n, self.layout._dom_to_id.get(domain_of(target_ne, self.node_infos), -1), dtype=np.int64)
            dom_v = self.layout.domain_ids([domain_of(s, self.node_infos) for s in src_nes])
        phi = self.layout.build_matrix(at_u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype, dom_u, dom_v)
        if self.n_dynamic > 0:
            if self.source_dynamic_dim and src_marks is None:
                raise ValueError("src_marks is required for source dynamic mode")
            if self.target_dynamic_dim and tgt_marks is None:
                raise ValueError("tgt_marks is required for target dynamic mode")
            marks = self._dynamic_mark_matrix(n, src_marks=src_marks, tgt_marks=tgt_marks)
            phi = np.concatenate([phi, marks], axis=1)
        return self.kernel.alpha(phi)

    def envelope_mark(self):
        """The dynamic mark that MAXIMIZES α over all 2^n_dynamic states: bit i = 1
        iff the dynamic weight i is positive (softplus is monotonic in w·φ). Lets
        candidate enumeration use one call instead of a 2^n sweep."""
        if self.n_dynamic <= 0:
            return np.zeros(0, dtype=np.float64)
        w_dyn = np.asarray(self.kernel.weights, dtype=np.float64)[-self.n_dynamic:]
        return (w_dyn > 0).astype(np.float64)

    def alpha_for_source(self, source_at, source_ne, tgt_ats, tgt_nes, src_mark=None, tgt_marks=None):
        """Vectorized α for ONE source vs a batch of targets — the transpose of
        alpha_for_target (the compensator hot path: one source excites many
        targets). The source mark is fixed (one row, broadcast to all targets).
        Returns (n,) α array.
        """
        n = len(tgt_nes)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        s_node = topo_node_of(source_ne)
        at_v = np.full(n, self.at_to_id.get(str(source_at), -1), dtype=np.int64)
        at_u = np.array([self.at_to_id.get(str(a), -1) for a in tgt_ats], dtype=np.int64)
        s_site, s_vendor, s_netype = self._attr(source_ne)
        topo = np.empty(n, dtype=np.float64)
        is_same_ne = np.empty(n, dtype=np.float64)
        same_site = np.empty(n, dtype=np.float64)
        same_vendor = np.empty(n, dtype=np.float64)
        same_netype = np.empty(n, dtype=np.float64)
        for i, tne in enumerate(tgt_nes):
            topo[i] = _topo_score(s_node, topo_node_of(tne), self.topology_index, self._topo_cache)
            is_same_ne[i] = 1.0 if source_ne == tne else 0.0
            t_site, t_vendor, t_netype = self._attr(tne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
        dom_u = dom_v = None
        if self.layout.n_dom:
            dom_v = np.full(n, self.layout._dom_to_id.get(domain_of(source_ne, self.node_infos), -1), dtype=np.int64)
            dom_u = self.layout.domain_ids([domain_of(t, self.node_infos) for t in tgt_nes])
        phi = self.layout.build_matrix(at_u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype, dom_u, dom_v)
        if self.n_dynamic > 0:
            if self.source_dynamic_dim and src_mark is None:
                raise ValueError("src_mark is required for source dynamic mode")
            if self.target_dynamic_dim and tgt_marks is None:
                raise ValueError("tgt_marks is required for target dynamic mode")
            marks = self._dynamic_mark_matrix(n, src_marks=src_mark, tgt_marks=tgt_marks)
            phi = np.concatenate([phi, marks], axis=1)
        return self.kernel.alpha(phi)


class DecomposedFeatureScorer:
    """φ-decomposed live α — numerically equal to RuntimeFeatureScorer but with
    NO (C, F) feature-matrix construction.

    The FeatureLayout is a fixed linear layout (bias + at-pair one-hot + 8
    scalars + optional domain one-hot + dynamic mark bits), so w·φ collapses to
    table lookups + a handful of scalar terms:

        z = w_bias + W_at[at_u, at_v] + w_same_at·[at_u==at_v]
          + (w_topo + w_txa·same_at + w_txs·same_site)·topo
          + w_same_ne·b + w_same_site·b + w_same_vendor·b + w_same_netype·b
          + W_dom'[dom_u, dom_v]                       (same_domain folded in)
          + SRC_TABLE[mark_combo] + tgt_term           (dynamic bits)
        α = alpha_scale · softplus(z)

    The one-hot blocks become the precomputed W_at / W_dom' tables; the 8
    possible source-mark combos become an 8-entry table; the target-side
    dynamic term is a per-target constant. To match the legacy path bit-for-bit
    on the topo interaction columns, the topo score is rounded through float32
    exactly like FeatureLayout.build_matrix stores it.
    """

    def __init__(self, scorer: RuntimeFeatureScorer):
        from alarm_flow_mhp.dynamic_state import combo_bits as _combo_bits
        from mhp.feature_kernel import softplus as _softplus

        self.scorer = scorer
        self._softplus = _softplus
        layout = scorer.layout
        self.layout = layout
        w = np.asarray(scorer.kernel.weights, dtype=np.float64)
        n_at = layout.n_at
        j = 0
        self.w_bias = float(w[j]); j += 1
        self.W_at = w[j:j + n_at * n_at].reshape(n_at, n_at).copy(); j += n_at * n_at
        (self.w_same_at, self.w_topo, self.w_same_ne, self.w_same_site,
         self.w_same_vendor, self.w_same_netype,
         self.w_topo_x_same_at, self.w_topo_x_same_site) = (float(x) for x in w[j:j + 8])
        j += 8
        self.n_dom = layout.n_dom
        if layout.n_dom:
            self.w_same_dom = float(w[j]); j += 1
            self.W_dom = w[j:j + layout.n_dom ** 2].reshape(layout.n_dom, layout.n_dom).copy()
            j += layout.n_dom ** 2
        else:
            self.w_same_dom = 0.0
            self.W_dom = np.zeros((0, 0), dtype=np.float64)
        if j != layout.n_features:
            raise ValueError(
                f"feature layout mismatch: consumed {j} weights, layout has {layout.n_features}"
            )
        self.n_dynamic = int(scorer.n_dynamic)
        w_dyn = w[layout.n_features:layout.n_features + self.n_dynamic]
        bits = _combo_bits(8)
        n_src = int(scorer.source_dynamic_dim)
        self.src_mark_table = (
            bits[:, :n_src] @ w_dyn[:n_src] if n_src else np.zeros(8, dtype=np.float64)
        )
        self.w_dyn_tgt = (
            w_dyn[n_src:n_src + scorer.target_dynamic_dim].copy()
            if scorer.target_dynamic_dim else np.zeros(0, dtype=np.float64)
        )
        self.alpha_scale = float(scorer.kernel.alpha_scale)
        # OOV-padded lookup tables: index [id+1] so id=-1 hits the zero row/col
        # (an OOV at/domain contributes no one-hot column, exactly like build_matrix).
        self.W_at_pad = np.zeros((n_at + 1, n_at + 1), dtype=np.float64)
        self.W_at_pad[1:, 1:] = self.W_at
        self.W_dom_pad = np.zeros((layout.n_dom + 1, layout.n_dom + 1), dtype=np.float64)
        if layout.n_dom:
            self.W_dom_pad[1:, 1:] = self.W_dom + np.eye(layout.n_dom) * self.w_same_dom

    def tgt_term(self, tgt_mark) -> float:
        """Per-target dynamic constant: w_dyn[3:]·(target pre-state mark)."""
        wt = self.w_dyn_tgt
        if not len(wt):
            return 0.0
        m = tgt_mark or (0, 0, 0)
        out = 0.0
        for i in range(min(len(wt), len(m))):
            out += float(wt[i]) * float(m[i])
        return out

    def logits_from_parts(
        self,
        tgt_at_id: int,
        src_at_ids: np.ndarray,
        topo: np.ndarray,
        is_same_ne: np.ndarray,
        same_site: np.ndarray,
        same_vendor: np.ndarray,
        same_netype: np.ndarray,
        tgt_dom_id: int,
        src_dom_ids: np.ndarray,
        src_mark_idx: np.ndarray,
        tgt_term: float,
    ) -> np.ndarray:
        """w·φ for a batch of source candidates against one target, from
        precomputed per-candidate parts. All boolean arrays are 0/1 float."""
        u = int(tgt_at_id)
        at_v = np.asarray(src_at_ids, dtype=np.int64)
        # Match build_matrix: φ stores topo (and its interaction columns) as
        # float32; round through float32 so w·φ is numerically identical.
        topo_r = np.asarray(topo, dtype=np.float32).astype(np.float64)
        same_at = ((at_v == u) & (u >= 0) & (at_v >= 0)).astype(np.float64)
        z = (
            self.w_bias
            + self.W_at_pad[u + 1, at_v + 1]
            + self.w_same_at * same_at
            + self.w_topo * topo_r
            + self.w_same_ne * np.asarray(is_same_ne, dtype=np.float64)
            + self.w_same_site * np.asarray(same_site, dtype=np.float64)
            + self.w_same_vendor * np.asarray(same_vendor, dtype=np.float64)
            + self.w_same_netype * np.asarray(same_netype, dtype=np.float64)
            + self.w_topo_x_same_at * (topo_r * same_at)
            + self.w_topo_x_same_site * (topo_r * np.asarray(same_site, dtype=np.float64))
        )
        if self.n_dom:
            dv = np.asarray(src_dom_ids, dtype=np.int64)
            z = z + self.W_dom_pad[int(tgt_dom_id) + 1, dv + 1]
        if self.n_dynamic:
            z = z + self.src_mark_table[np.asarray(src_mark_idx, dtype=np.int64)]
            if tgt_term:
                z = z + tgt_term
        return z

    def alpha_from_parts(self, *args, **kwargs) -> np.ndarray:
        """softplus(logits)·alpha_scale — same postprocessing as FeatureKernel.alpha."""
        z = self.logits_from_parts(*args, **kwargs)
        a = self._softplus(z)
        if self.alpha_scale != 1.0:
            a = a * self.alpha_scale
        return a

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None, tgt_marks=None):
        """Drop-in equivalent of RuntimeFeatureScorer.alpha_for_target (string
        inputs). Attribute/topology lookups reuse the wrapped scorer's caches."""
        s = self.scorer
        n = len(src_nes)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        t_node = topo_node_of(target_ne)
        u = s.at_to_id.get(str(target_at), -1)
        at_v = np.array([s.at_to_id.get(str(a), -1) for a in src_ats], dtype=np.int64)
        t_site, t_vendor, t_netype = s._attr(target_ne)
        topo = np.empty(n, dtype=np.float64)
        is_same_ne = np.empty(n, dtype=np.float64)
        same_site = np.empty(n, dtype=np.float64)
        same_vendor = np.empty(n, dtype=np.float64)
        same_netype = np.empty(n, dtype=np.float64)
        for i, sne in enumerate(src_nes):
            topo[i] = _topo_score(topo_node_of(sne), t_node, s.topology_index, s._topo_cache)
            is_same_ne[i] = 1.0 if sne == target_ne else 0.0
            s_site, s_vendor, s_netype = s._attr(sne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
        tgt_dom_id = -1
        src_dom_ids = np.full(n, -1, dtype=np.int64)
        if self.n_dom:
            tgt_dom_id = self.layout._dom_to_id.get(domain_of(target_ne, s.node_infos), -1)
            src_dom_ids = self.layout.domain_ids([domain_of(x, s.node_infos) for x in src_nes])
        # dynamic marks: mirror _dynamic_mark_matrix semantics for the common
        # streaming shapes — per-candidate (n,3) src marks + one target mark row.
        src_mark_idx = np.zeros(n, dtype=np.int64)
        tgt_term = 0.0
        if self.n_dynamic:
            if src_marks is not None:
                sm = np.asarray(src_marks, dtype=np.float64)
                if sm.ndim == 1:
                    sm = np.tile(sm.reshape(1, -1), (n, 1))
                sm = sm[:, :3]
                src_mark_idx = (
                    sm[:, 0].astype(np.int64)
                    + 2 * (sm[:, 1].astype(np.int64) if sm.shape[1] > 1 else 0)
                    + 4 * (sm[:, 2].astype(np.int64) if sm.shape[1] > 2 else 0)
                )
            if len(self.w_dyn_tgt) and tgt_marks is not None:
                tm = np.asarray(tgt_marks, dtype=np.float64).reshape(-1)
                tgt_term = self.tgt_term(tuple(tm[:3]))
        return self.alpha_from_parts(
            u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype,
            tgt_dom_id, src_dom_ids, src_mark_idx, tgt_term,
        )
