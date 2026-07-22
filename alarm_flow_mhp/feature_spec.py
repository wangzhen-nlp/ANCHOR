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

from collections import Counter, defaultdict
from itertools import combinations

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


# Domain buckets kept as-is by phi_node_domain; OTHER/MISSING/unknown-device
# all collapse into OTHER (they carry no distinct propagation semantics, and
# merging keeps the dom-pair one-hot block at 4×4). Both φ and ψ (μ features)
# use this merged 4-bucket view of the NE graph's domain_bucket.
_PHI_NODE_DOMAINS = frozenset({"DATA", "RAN", "TRANSMISSION"})


def phi_node_domain(info):
    """Merged domain bucket of a graph node: RAN/TRANSMISSION/DATA pass through,
    everything else (OTHER, MISSING, node absent from the graph) → OTHER."""
    bucket = (getattr(info, "domain_bucket", "") or "") if info is not None else ""
    return bucket if bucket in _PHI_NODE_DOMAINS else "OTHER"


def phi_domain_of(entity, node_infos=None):
    """Merged feature domain of an entity. Composite (site×domain) entities pass
    their embedded label domain through unchanged; bare nodes merge their
    NE-graph ``domain_bucket`` via :func:`phi_node_domain`."""
    node, dom = split_entity(entity)
    if dom:
        return dom
    return phi_node_domain((node_infos or {}).get(node))


class SiteStats:
    """Site/device structural summaries for the φ graph columns.

    Derived deterministically from ``node_infos`` (NE → site) and the topology
    index's undirected 1-hop adjacency, so training and inference agree given
    the same graph snapshot — the same consistency contract topo_score already
    relies on. In site mode the NE-level summary is retained for pair-level
    site features, while single-device degree features remain zero.

    Feature scaling is saturating x/(x+k) — bounded in [0,1), monotone, and
    dataset-independent (no normalizer to persist).
    """

    SIZE_K = 8.0    # site with 8 NEs → 0.5; hubs saturate toward 1
    LINK_K = 4.0    # 4 inter-site NE links → 0.5
    DEGREE_K = 4.0  # device with 4 undirected neighbors → 0.5
    SITE_LOAD_K = 8.0  # site with 8 total NE-link endpoints → 0.5
    EXTERNAL_NEIGHBOR_K = 4.0  # site connected to 4 distinct other sites → 0.5

    def __init__(self, node_infos, topology_index, undirected_neighbors=None):
        node_site = {}
        sizes = Counter()
        site_domain_counts = defaultdict(Counter)
        for node, info in (node_infos or {}).items():
            s = getattr(info, "site_id", "") or ""
            node_site[node] = s
            if s:
                sizes[s] += 1
                site_domain_counts[s][phi_node_domain(info)] += 1
        pair_links = Counter()
        site_link_ends = Counter()
        site_all_link_ends = Counter()
        site_external_neighbors = defaultdict(set)
        node_degrees = Counter()
        undirected = (
            undirected_neighbors
            if undirected_neighbors is not None
            else (getattr(topology_index, "undirected_hops", {}) or {})
        )
        self._undirected = undirected
        self._direct_neighbor_sets = undirected_neighbors is not None
        for node, neighbors in undirected.items():
            hop_items = (
                ((nbr, 1) for nbr in neighbors)
                if undirected_neighbors is not None
                else neighbors.items()
            )
            for nbr, hop in hop_items:
                if hop != 1 or not (node < nbr):   # each undirected link once
                    continue
                node_degrees[node] += 1
                node_degrees[nbr] += 1
                site_a = node_site.get(node, "")
                site_b = node_site.get(nbr, "")
                if site_a:
                    site_all_link_ends[site_a] += 1
                if site_b:
                    site_all_link_ends[site_b] += 1
                if not site_a or not site_b:
                    continue
                # Link endpoints are the denominator for normalized shares.
                # An intra-site link contributes two endpoints to that site.
                site_link_ends[site_a] += 1
                site_link_ends[site_b] += 1
                if site_b == site_a:
                    continue
                key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
                pair_links[key] += 1
                # Count distinct EXTERNAL sites, not NE edges: parallel NE links
                # between the same two sites still contribute one neighbour.
                site_external_neighbors[site_a].add(site_b)
                site_external_neighbors[site_b].add(site_a)
        self.sizes = sizes
        self.pair_links = pair_links
        self.site_link_ends = site_link_ends
        self.site_all_link_ends = site_all_link_ends
        self.site_external_neighbor_counts = Counter({
            site: len(neighbors) for site, neighbors in site_external_neighbors.items()
        })
        self.node_degrees = node_degrees
        self.site_domain_counts = site_domain_counts
        self.site_domain_norms = {
            site: sum(float(n) ** 2 for n in counts.values()) ** 0.5
            for site, counts in site_domain_counts.items()
        }

    def size_feat(self, site) -> float:
        n = self.sizes.get(site, 0) if site else 0
        return n / (n + self.SIZE_K)

    def link_feat(self, site_a, site_b) -> float:
        """Inter-site connectivity; 0 for same/unknown sites (same_site covers
        the former, and an unknown site has no graph links by construction)."""
        if not site_a or not site_b or site_a == site_b:
            return 0.0
        key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
        n = self.pair_links.get(key, 0)
        return n / (n + self.LINK_K)

    def site_link_ratio(self, site_a, site_b) -> float:
        """Share of both sites' NE-link endpoints devoted to this site pair."""
        if not site_a or not site_b or site_a == site_b:
            return 0.0
        key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
        between = self.pair_links.get(key, 0)
        total_ends = self.site_link_ends.get(site_a, 0) + self.site_link_ends.get(site_b, 0)
        return (2.0 * between / total_ends) if total_ends else 0.0

    def site_link_density(self, site_a, site_b) -> float:
        """Inter-site NE links divided by all possible cross-site NE pairs."""
        if not site_a or not site_b or site_a == site_b:
            return 0.0
        key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
        possible = self.sizes.get(site_a, 0) * self.sizes.get(site_b, 0)
        return (self.pair_links.get(key, 0) / possible) if possible else 0.0

    def site_size_balance(self, site_a, site_b) -> float:
        """Similarity of site sizes: min(size_a,size_b) / max(size_a,size_b)."""
        if not site_a or not site_b:
            return 0.0
        a = self.sizes.get(site_a, 0)
        b = self.sizes.get(site_b, 0)
        return (min(a, b) / max(a, b)) if a and b else 0.0

    def site_domain_cosine(self, site_a, site_b) -> float:
        """Cosine similarity of the sites' merged four-bucket domain counts."""
        if not site_a or not site_b:
            return 0.0
        a = self.site_domain_counts.get(site_a)
        b = self.site_domain_counts.get(site_b)
        if site_a == site_b and a:
            return 1.0
        norm = self.site_domain_norms.get(site_a, 0.0) * self.site_domain_norms.get(site_b, 0.0)
        if not a or not b or not norm:
            return 0.0
        # Four buckets at most; iterate the smaller mapping.
        if len(a) > len(b):
            a, b = b, a
        dot = sum(float(n) * b.get(domain, 0) for domain, n in a.items())
        return dot / norm

    def degree_feat(self, node) -> float:
        """Saturating undirected 1-hop degree of one device."""
        n = self.node_degrees.get(node, 0) if node else 0
        return n / (n + self.DEGREE_K)

    def site_link_load(self, site) -> float:
        """Saturating number of all undirected NE-link endpoints at a site."""
        n = self.site_all_link_ends.get(site, 0) if site else 0
        return n / (n + self.SITE_LOAD_K)

    def site_external_neighbor_count(self, site) -> float:
        """Saturating count of distinct other sites joined by a direct NE edge.

        A site with no cross-site edge maps to 0 even when it has intra-site NE
        links. Multiple NE edges to the same external site count only once.
        """
        n = self.site_external_neighbor_counts.get(site, 0) if site else 0
        return n / (n + self.EXTERNAL_NEIGHBOR_K)

    def domain_share(self, site, domain) -> float:
        """Share of a site's devices in one merged domain bucket."""
        total = self.sizes.get(site, 0) if site else 0
        if not total or not domain:
            return 0.0
        return self.site_domain_counts.get(site, {}).get(str(domain), 0) / total

    def device_link_ratio(self, node_a, node_b) -> float:
        """Share of both devices' link endpoints occupied by their direct edge."""
        if not node_a or not node_b or node_a == node_b:
            return 0.0
        neighbors = self._undirected.get(
            node_a, set() if self._direct_neighbor_sets else {}
        )
        linked = (
            node_b in neighbors
            if self._direct_neighbor_sets
            else neighbors.get(node_b) == 1
        )
        if not linked:
            return 0.0
        total_degree = self.node_degrees.get(node_a, 0) + self.node_degrees.get(node_b, 0)
        return (2.0 / total_degree) if total_degree else 0.0

    def undirected_neighbor_map(self) -> dict:
        """Symmetric node adjacency covering every stored neighbor relation.

        A superset of the pairs where ``device_link_ratio`` can be nonzero
        (dict-mode entries beyond hop 1 are included but score 0), so sparse
        consumers can enumerate candidates instead of probing every pair.
        """
        symmetric = defaultdict(set)
        for node, neighbors in self._undirected.items():
            for neighbor in neighbors:
                symmetric[node].add(neighbor)
                symmetric[neighbor].add(node)
        return dict(symmetric)


class GeoStats:
    """Cached-input geographic features for a pair of sites.

    Coordinates come from ``GraphContext.site_coords`` when available, falling
    back to the first valid NE coordinate per site. Distance is transformed to
    a bounded proximity so it stays on the same numerical scale as the other φ
    columns and requires no fitted/persisted standardizer.
    """

    PROXIMITY_KM = 10.0  # 10 km -> 0.5

    def __init__(self, node_infos, site_coords=None):
        self.coords = dict(site_coords or {})
        for info in (node_infos or {}).values():
            site = getattr(info, "site_id", "") or ""
            lat = getattr(info, "latitude", None)
            lon = getattr(info, "longitude", None)
            if site and site not in self.coords and lat is not None and lon is not None:
                self.coords[site] = (float(lat), float(lon))
        from ne_link_learning.core import haversine_km

        self._haversine_km = haversine_km
        self._pair_cache = {}

    def pair_features(self, site_a, site_b) -> tuple[float, float]:
        """Return ``(geo_proximity, geo_distance_missing)`` for one site pair."""
        key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
        hit = self._pair_cache.get(key)
        if hit is not None:
            return hit
        # A shared, known site id proves zero distance even when that site's
        # coordinates are absent. Empty ids remain genuinely unknown.
        if site_a and site_a == site_b:
            out = (1.0, 0.0)
            self._pair_cache[key] = out
            return out
        a = self.coords.get(site_a) if site_a else None
        b = self.coords.get(site_b) if site_b else None
        if a is None or b is None:
            out = (0.0, 1.0)
            self._pair_cache[key] = out
            return out
        distance_km = self._haversine_km(a[0], a[1], b[0], b[1])
        if distance_km is None:
            out = (0.0, 1.0)
            self._pair_cache[key] = out
            return out
        k = self.PROXIMITY_KM
        out = (k / (float(distance_km) + k), 0.0)
        self._pair_cache[key] = out
        return out


class FeatureLayout:
    """Canonical φ(target, source) construction shared by training and
    inference, so the feature vector is byte-identical on both sides.

    Given per-pair attribute arrays (alarm-type ids, topology score, and the
    same-* booleans), produces the (C, F) feature matrix and the feature names.
    The layout is fully determined by the alarm-type vocabulary size n_at.
    """

    # Systematic second-order cross block: EVERY pairwise product of the scalar
    # columns below becomes a φ column (x[a*b]) — the linear-in-parameters way
    # to learn feature interactions (e.g. geo×same_alarm_type, isolation×geo)
    # without hand-picking them. Excluded by rule, not by hypothesis:
    #   - alternate normalizations of an included column (site_link_ratio/
    #     density share site_link_score's interaction role; size_balance and
    #     degrees share the size/ext-count role; device_link_ratio ≈ topo),
    #   - complements (geo_distance_missing = 1 - known(geo_proximity)).
    # Degenerate products (identically 0, or duplicating a base column, e.g.
    # same_site×site_link_score ≡ 0) are tolerated: the ridge absorbs them.
    # The old hand-crafted topo_x_same_at / topo_x_same_site columns are
    # subsumed by this block (topo_score × same_alarm_type / same_site).
    CROSS_FEATURES = (
        "same_alarm_type",
        "topo_score",
        "is_same_ne",
        "same_site",
        "same_vendor",
        "same_ne_type",
        "site_link_score",
        "site_domain_cosine",
        "geo_proximity",
        "tgt_site_size",
        "src_site_size",
        "tgt_site_external_neighbor_count",
        "src_site_external_neighbor_count",
    )
    CROSS_PAIRS = tuple(combinations(CROSS_FEATURES, 2))
    _CROSS_SET = frozenset(CROSS_FEATURES)

    def __init__(self, at_vocab, domain_vocab=()):
        self.at_vocab = list(at_vocab)
        self.n_at = max(len(self.at_vocab), 1)
        # Domain block: label-sourced device_domain vocab in site×domain mode,
        # merged 4-bucket node domains (phi_node_domain) in device mode.
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
            "tgt_site_size",       # saturating site NE-count of the target's site
            "src_site_size",       # ... of the source's site
            "site_link_score",     # saturating inter-site NE-link count (0 = same/unknown site)
            "site_link_ratio",     # share of the two sites' NE-link endpoints
            "site_link_density",   # inter-site NE links / possible cross-site NE pairs
            "site_size_balance",   # min(site sizes) / max(site sizes)
            "site_domain_cosine",  # cosine similarity of merged domain-count vectors
            "tgt_undirected_degree",  # saturating target-device undirected degree
            "src_undirected_degree",  # saturating source-device undirected degree
            "device_link_ratio",   # share of the two devices' link endpoints
            "geo_proximity",       # 10 / (distance_km + 10); 0 when coordinates are missing
            "geo_distance_missing",
            "tgt_site_external_neighbor_count",  # distinct external sites, saturating count
            "src_site_external_neighbor_count",
        ]
        names += [f"x[{a}*{b}]" for a, b in self.CROSS_PAIRS]
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
        """Map a sequence of domain strings to layout ids (-1 = OOV / no domain).
        Callers resolve domains via phi_domain_of, so values are already merged
        buckets; an OOV here is a bucket absent from training → no dom column."""
        return np.array([self._dom_to_id.get(str(d), -1) for d in domains], dtype=np.int64)

    def dom_id(self, domain) -> int:
        """Scalar counterpart of :meth:`domain_ids`."""
        return self._dom_to_id.get(str(domain), -1)

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
        tgt_site_size: np.ndarray = None,
        src_site_size: np.ndarray = None,
        site_link: np.ndarray = None,
        site_link_ratio: np.ndarray = None,
        site_link_density: np.ndarray = None,
        site_size_balance: np.ndarray = None,
        site_domain_cosine: np.ndarray = None,
        tgt_undirected_degree: np.ndarray = None,
        src_undirected_degree: np.ndarray = None,
        device_link_ratio: np.ndarray = None,
        geo_proximity: np.ndarray = None,
        geo_missing: np.ndarray = None,
        tgt_site_external_neighbor_count: np.ndarray = None,
        src_site_external_neighbor_count: np.ndarray = None,
    ) -> np.ndarray:
        """All inputs are length-C arrays (at_*/dom_* int, rest float/bool).

        ``dom_u``/``dom_v`` are domain layout ids (see :meth:`domain_ids`); they
        are required iff this layout has a non-empty domain vocab.
        Optional structural/geographic inputs become zero columns when omitted.
        """
        # φ is built in float32 (halves the (C, F) matrix, the dominant memory block
        # at large candidate counts; 0/1 indicators + a topo score in [0,1] are
        # exactly/near-exactly representable, and φ·w promotes to float64 so the dot
        # product keeps full precision). Columns are written into a PREALLOCATED
        # matrix in place rather than column_stack'd, so feature-column arrays and
        # the stacked output never coexist (which would ~double the peak). dom_u/dom_v
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

        def _col(x):
            return 0.0 if x is None else np.asarray(x, dtype=_F)

        # Scalar columns in _names order; the cross block below multiplies the
        # SAME float32 columns, so training φ and the decomposed scorer agree
        # bit-for-bit (f32×f32 = round(exact product)).
        scalar_cols = (
            ("same_alarm_type", same_at),
            ("topo_score", np.asarray(topo, dtype=_F)),
            ("is_same_ne", np.asarray(is_same_ne, dtype=_F)),
            ("same_site", np.asarray(same_site, dtype=_F)),
            ("same_vendor", np.asarray(same_vendor, dtype=_F)),
            ("same_ne_type", np.asarray(same_netype, dtype=_F)),
            ("tgt_site_size", _col(tgt_site_size)),
            ("src_site_size", _col(src_site_size)),
            ("site_link_score", _col(site_link)),
            ("site_link_ratio", _col(site_link_ratio)),
            ("site_link_density", _col(site_link_density)),
            ("site_size_balance", _col(site_size_balance)),
            ("site_domain_cosine", _col(site_domain_cosine)),
            ("tgt_undirected_degree", _col(tgt_undirected_degree)),
            ("src_undirected_degree", _col(src_undirected_degree)),
            ("device_link_ratio", _col(device_link_ratio)),
            ("geo_proximity", _col(geo_proximity)),
            ("geo_distance_missing", _col(geo_missing)),
            ("tgt_site_external_neighbor_count", _col(tgt_site_external_neighbor_count)),
            ("src_site_external_neighbor_count", _col(src_site_external_neighbor_count)),
        )
        cross_base = {}
        for name, col in scalar_cols:
            phi[:, j] = col; j += 1
            if name in self._CROSS_SET:
                cross_base[name] = col
        for a, b in self.CROSS_PAIRS:
            phi[:, j] = cross_base[a] * cross_base[b]; j += 1
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

    Uses only INDUCTIVE attributes of the type's graph entity (categorical
    alarm/device attributes plus a small persisted set of node/site structural
    scalars) — deliberately NOT the type's own historical event count, which
    wouldn't generalize and would reintroduce per-entity memorization.
    """

    def __init__(self, at_vocab, ne_type_vocab, vendor_vocab, domain_vocab,
                 numeric_feature_names=()):
        self.at_vocab = list(at_vocab)
        self.ne_type_vocab = list(ne_type_vocab)
        self.vendor_vocab = list(vendor_vocab)
        self.domain_vocab = list(domain_vocab)
        self.numeric_feature_names = list(numeric_feature_names)
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
            + self.numeric_feature_names
        )

    @property
    def n_features(self):
        return len(self.feature_names)

    def build_matrix(self, ats, ne_types, vendors, domains, numeric_features=None):
        """Per-type categorical + structural attributes → (n, F) matrix."""
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
        numeric_features = numeric_features or {}
        for name in self.numeric_feature_names:
            arr = np.asarray(numeric_features.get(name, 0.0), dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(n, float(arr), dtype=np.float64)
            if len(arr) != n:
                raise ValueError(
                    f"μ numeric feature {name!r} has length {len(arr)}, expected {n}"
                )
            blocks.append(arr.reshape(n, 1))
        return np.column_stack(blocks)

    def build_row(self, at, ne_type, vendor, domain, numeric_features=None):
        return self.build_matrix(
            [at], [ne_type], [vendor], [domain], numeric_features=numeric_features
        )[0]

    def to_dict(self):
        return {
            "at_vocab": self.at_vocab,
            "ne_type_vocab": self.ne_type_vocab,
            "vendor_vocab": self.vendor_vocab,
            "domain_vocab": self.domain_vocab,
            "numeric_feature_names": self.numeric_feature_names,
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            payload.get("at_vocab", []),
            payload.get("ne_type_vocab", []),
            payload.get("vendor_vocab", []),
            payload.get("domain_vocab", []),
            payload.get("numeric_feature_names", []),
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

    def __init__(self, node_infos, node_domains=None, site_coords=None,
                 device_node_infos=None, device_undirected_neighbors=None):
        self.node_infos = node_infos
        self.node_domains = dict(node_domains or {})
        self.site_coords = dict(site_coords or {})
        self.device_node_infos = device_node_infos
        self.device_undirected_neighbors = device_undirected_neighbors


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
    from alarm_flow_isahp.ne_topology import build_undirected_neighbors
    from ne_link_learning.core import NodeInfo, build_graph_context

    gc = build_graph_context(ne_graph_data)
    device_neighbors = build_undirected_neighbors(ne_graph_data)
    if node_field != "site_id":
        return _NodeContext(
            gc.node_infos,
            site_coords=gc.site_coords,
            device_node_infos=gc.node_infos,
            device_undirected_neighbors=device_neighbors,
        )

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
    return _NodeContext(
        node_infos,
        node_domains,
        gc.site_coords,
        device_node_infos=gc.node_infos,
        # Use the same non-empty-direction rule as NETopologyIndex instead of
        # ne_link_learning's arrow-only adjacency, so device/site feature modes
        # have identical NE-link counts and ratios.
        device_undirected_neighbors=device_neighbors,
    )


def _graph_site_stats(graph_context):
    """Build the shared NE/site summary once for μ train or runtime setup."""
    if graph_context is None:
        return None
    cached = getattr(graph_context, "_mhp_site_stats", None)
    if cached is not None:
        return cached
    node_infos = getattr(graph_context, "device_node_infos", None)
    neighbors = getattr(graph_context, "device_undirected_neighbors", None)
    if node_infos is None or neighbors is None:
        return None
    stats = SiteStats(node_infos, None, undirected_neighbors=neighbors)
    graph_context._mhp_site_stats = stats
    return stats


def _mu_numeric_values(feature_names, stats, node, site, domain):
    """One entity's persisted-order μ structural feature mapping."""
    if stats is None:
        return {name: 0.0 for name in feature_names}
    values = {
        "site_size": stats.size_feat(site),
        "undirected_degree": stats.degree_feat(node),
        "site_link_load": stats.site_link_load(site),
        "site_external_neighbor_count": stats.site_external_neighbor_count(site),
        "domain_share_in_site": stats.domain_share(site, domain),
    }
    return {name: values[name] for name in feature_names}


def build_mu_features(vocabs, type_fields, graph_context, *, cap=50, node_field="alarm_source"):
    """Per-type μ feature matrix ψ (M, Fμ) + the MuFeatureSpec (for inference).

    Categorical attributes are alarm_type + ne_type/vendor/domain. Structural
    scalars are site_size + undirected_degree + external-site-neighbour count in
    device mode, or site_size + site_link_load + external-site-neighbour count
    (+ domain_share_in_site for site×domain) in site mode.
    Returns (psi, spec).
    """
    labels = vocabs.type_vocab.labels
    M = len(labels)
    type_fields = tuple(type_fields)
    src_idx, at_idx = _type_field_indices(type_fields, node_field)
    dom_idx = type_fields.index("device_domain") if "device_domain" in type_fields else None
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
    stats = _graph_site_stats(graph_context)
    if node_field == "site_id":
        numeric_names = ["site_size", "site_link_load", "site_external_neighbor_count"]
        if dom_idx is not None:
            numeric_names.append("domain_share_in_site")
    else:
        numeric_names = ["site_size", "undirected_degree", "site_external_neighbor_count"]

    ats, ne_types, vendors, domains = [], [], [], []
    numeric = {name: [] for name in numeric_names}
    for label in labels:
        node_id, at = parse_label_ne_at(label, src_idx, at_idx)
        ats.append(at)
        info = node_infos.get(node_id)
        ne_types.append((info.ne_type or "") if info is not None else "")
        vendors.append((info.manufacturer or "") if info is not None else "")
        if dom_idx is not None:
            parts = str(label).split(" | ")
            dom_val = parts[dom_idx] if len(parts) > dom_idx else ""
            domain = "" if dom_val == "<empty>" else dom_val
        else:
            # device mode: merged 4-bucket node domain (OTHER+MISSING+unknown → OTHER),
            # same view as φ so train/inference agree on one domain vocabulary.
            domain = phi_node_domain(info)
        domains.append(domain)
        site = (info.site_id or "") if info is not None else ""
        values = _mu_numeric_values(numeric_names, stats, node_id, site, domain)
        for name in numeric_names:
            numeric[name].append(values[name])

    spec = MuFeatureSpec(
        at_vocab=sorted({a for a in ats if a}),
        ne_type_vocab=_capped_vocab(ne_types, cap),
        vendor_vocab=_capped_vocab(vendors, cap),
        domain_vocab=_capped_vocab(domains, cap),
        numeric_feature_names=numeric_names,
    )
    psi = spec.build_matrix(ats, ne_types, vendors, domains, numeric_features=numeric)
    return psi, spec


class RuntimeMuScorer:
    """Inference-time live μ(u) = softplus(w_μ · ψ(u)) for ANY type, including
    new devices — ψ built from alarm/device attributes + cached graph summaries.
    """

    def __init__(self, mu_kernel, mu_spec: MuFeatureSpec, graph_context):
        from mhp.feature_kernel import softplus

        self.kernel = mu_kernel
        self.spec = mu_spec
        self._softplus = softplus
        self.node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
        self.stats = _graph_site_stats(graph_context)
        if mu_spec.n_features != mu_kernel.n_features:
            raise ValueError(
                f"μ feature layout ({mu_spec.n_features}) != μ kernel ({mu_kernel.n_features})"
            )

    def mu_for(self, alarm_type, ne):
        # ne is the feature ENTITY (topo node, optionally + domain). Node
        # attributes come from the topo node; domain from the entity (embedded
        # domain in site×domain mode, else the node's merged 4-bucket domain —
        # the same phi_domain_of view build_mu_features trained on).
        info = self.node_infos.get(topo_node_of(ne))
        ne_type = (info.ne_type or "") if info is not None else ""
        vendor = (info.manufacturer or "") if info is not None else ""
        domain = phi_domain_of(ne, self.node_infos)
        site = (info.site_id or "") if info is not None else ""
        numeric = _mu_numeric_values(
            self.spec.numeric_feature_names,
            self.stats,
            topo_node_of(ne),
            site,
            domain,
        )
        row = self.spec.build_row(
            alarm_type, ne_type, vendor, domain, numeric_features=numeric
        )
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
    """Symmetric undirected topology proximity, cached per node pair."""
    if topology_index is None or not source_ne or not target_ne:
        return 0.0
    key = (source_ne, target_ne) if source_ne < target_ne else (target_ne, source_ne)
    hit = cache.get(key)
    if hit is not None:
        return hit
    from alarm_flow_isahp.ne_topology import undirected_topology_score

    score = undirected_topology_score(topology_index, source_ne, target_ne)
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
      site/vendor/netype (object), domain (object — the φ domain), plus the
      alarm-type vocabulary and the φ domain vocabulary.

    φ domain: label-sourced device_domain when it is a type field (site×domain
    mode); otherwise the node's merged 4-bucket domain via
    :func:`phi_node_domain` — activating the same_domain + dom-pair φ columns
    in device mode too. The feature ENTITY stays the bare node in device mode
    (identity is unchanged; only φ gains columns).
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
        info = node_infos.get(node_id)            # node graph keyed by topo node
        if dom_idx is None:
            domain[tid] = phi_node_domain(info)   # φ-only; entity stays bare
        else:
            domain[tid] = dom_val
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
    # φ domain vocab: label-sourced domains in site×domain mode, merged 4-bucket
    # node domains (phi_node_domain) in device mode — device φ carries the
    # same_domain + dom-pair block too.
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
    node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
    graph_stats = _graph_site_stats(graph_context)
    site_stats = graph_stats if node_field != "site_id" else None
    link_stats = graph_stats

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
    uniq_device_link_ratio = np.empty(len(uniq), dtype=np.float32)
    for j in range(len(uniq)):
        pk = int(uniq[j])
        s_code = pk // base - 1
        t_code = pk % base - 1
        s_node = node_vals[s_code] if s_code >= 0 else ""
        t_node = node_vals[t_code] if t_code >= 0 else ""
        uniq_scores[j] = _topo_score(s_node, t_node, topology_index, topo_cache)
        uniq_device_link_ratio[j] = (
            site_stats.device_link_ratio(s_node, t_node) if site_stats is not None else 0.0
        )
    topo_vec = uniq_scores[inv]                      # float64 (also returned as the topo prior)
    device_link_ratio_vec = uniq_device_link_ratio[inv]
    del uniq, inv, uniq_scores, uniq_device_link_ratio

    tgt_degree_vec = src_degree_vec = None
    if site_stats is not None and node_vals:
        degree_by_code = np.array(
            [site_stats.degree_feat(v) for v in node_vals], dtype=np.float32
        )
        degree_by_type = np.where(
            node_codes >= 0, degree_by_code[np.maximum(node_codes, 0)], 0.0
        )
        tgt_degree_vec = degree_by_type[cand_t]
        src_degree_vec = degree_by_type[cand_s]

    # Vectorized same-* : compare gathered integer codes (−1 = empty), reproducing
    # the old "a is truthy AND a == b" semantics as (code_t >= 0) & (code_t == code_s).
    # The per-type code arrays are length M (small); the gathered (C,) compares are
    # transient inside build_matrix.
    site_codes, site_vals = _factorize_attr(site)
    vendor_codes, _ = _factorize_attr(vendor)
    netype_codes, _ = _factorize_attr(netype)

    def _same(codes):
        a = codes[cand_t]
        b = codes[cand_s]
        return ((a >= 0) & (a == b)).astype(np.float32)

    # Site/geo columns: evaluate pair features once per UNIQUE site-code pair,
    # then gather to candidates. Haversine is therefore O(unique site pairs),
    # not O(candidate pairs). Individual site-size/device-degree columns remain
    # device-mode only; pair-level site and geographic columns work in both modes.
    tgt_size_vec = src_size_vec = site_link_vec = None
    tgt_external_neighbor_vec = src_external_neighbor_vec = None
    if site_stats is not None and site_vals:
        size_by_code = np.array(
            [site_stats.size_feat(v) for v in site_vals], dtype=np.float64
        )
        size_by_type = np.where(site_codes >= 0, size_by_code[np.maximum(site_codes, 0)], 0.0)
        tgt_size_vec = size_by_type[cand_t]
        src_size_vec = size_by_type[cand_s]
    if link_stats is not None and site_vals:
        external_neighbor_by_code = np.array(
            [link_stats.site_external_neighbor_count(v) for v in site_vals],
            dtype=np.float32,
        )
        external_neighbor_by_type = np.where(
            site_codes >= 0,
            external_neighbor_by_code[np.maximum(site_codes, 0)],
            0.0,
        )
        tgt_external_neighbor_vec = external_neighbor_by_type[cand_t]
        src_external_neighbor_vec = external_neighbor_by_type[cand_s]
    geo_stats = GeoStats(
        node_infos,
        getattr(graph_context, "site_coords", {}) if graph_context is not None else {},
    )
    base_site = len(site_vals) + 2
    spair_key = (site_codes[cand_s] + 1) * base_site + (site_codes[cand_t] + 1)
    uniq_sp, inv_sp = np.unique(spair_key, return_inverse=True)
    del spair_key
    uniq_geo = np.empty(len(uniq_sp), dtype=np.float32)
    uniq_geo_missing = np.empty(len(uniq_sp), dtype=np.float32)
    uniq_link = np.empty(len(uniq_sp), dtype=np.float64) if site_stats is not None else None
    uniq_site_link_ratio = np.empty(len(uniq_sp), dtype=np.float32)
    uniq_site_link_density = np.empty(len(uniq_sp), dtype=np.float32)
    uniq_site_size_balance = np.empty(len(uniq_sp), dtype=np.float32)
    uniq_site_domain_cosine = np.empty(len(uniq_sp), dtype=np.float32)
    for j in range(len(uniq_sp)):
        pk = int(uniq_sp[j])
        s_code = pk // base_site - 1
        t_code = pk % base_site - 1
        s_site = site_vals[s_code] if s_code >= 0 else ""
        t_site = site_vals[t_code] if t_code >= 0 else ""
        if uniq_link is not None:
            uniq_link[j] = site_stats.link_feat(s_site, t_site)
        uniq_site_link_ratio[j] = (
            link_stats.site_link_ratio(s_site, t_site) if link_stats is not None else 0.0
        )
        uniq_site_link_density[j] = (
            link_stats.site_link_density(s_site, t_site) if link_stats is not None else 0.0
        )
        uniq_site_size_balance[j] = (
            link_stats.site_size_balance(s_site, t_site) if link_stats is not None else 0.0
        )
        uniq_site_domain_cosine[j] = (
            link_stats.site_domain_cosine(s_site, t_site) if link_stats is not None else 0.0
        )
        uniq_geo[j], uniq_geo_missing[j] = geo_stats.pair_features(s_site, t_site)
    if uniq_link is not None:
        site_link_vec = uniq_link[inv_sp]
    geo_proximity_vec = uniq_geo[inv_sp]
    geo_missing_vec = uniq_geo_missing[inv_sp]
    site_link_ratio_vec = uniq_site_link_ratio[inv_sp]
    site_link_density_vec = uniq_site_link_density[inv_sp]
    site_size_balance_vec = uniq_site_size_balance[inv_sp]
    site_domain_cosine_vec = uniq_site_domain_cosine[inv_sp]
    del (uniq_sp, inv_sp, uniq_geo, uniq_geo_missing, uniq_link,
         uniq_site_link_ratio, uniq_site_link_density, uniq_site_size_balance,
         uniq_site_domain_cosine)

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
        tgt_site_size=tgt_size_vec,
        src_site_size=src_size_vec,
        site_link=site_link_vec,
        site_link_ratio=site_link_ratio_vec,
        site_link_density=site_link_density_vec,
        site_size_balance=site_size_balance_vec,
        site_domain_cosine=site_domain_cosine_vec,
        tgt_undirected_degree=tgt_degree_vec,
        src_undirected_degree=src_degree_vec,
        device_link_ratio=device_link_ratio_vec,
        geo_proximity=geo_proximity_vec,
        geo_missing=geo_missing_vec,
        tgt_site_external_neighbor_count=tgt_external_neighbor_vec,
        src_site_external_neighbor_count=src_external_neighbor_vec,
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
                 dynamic_mode: str | None = None, node_field: str = "alarm_source"):
        from mhp.feature_kernel import softplus

        self.kernel = kernel
        self.layout = FeatureLayout(at_vocab, domain_vocab)
        self._softplus = softplus
        self.at_to_id = {str(a): i for i, a in enumerate(at_vocab)}
        self.node_infos = getattr(graph_context, "node_infos", {}) if graph_context is not None else {}
        self.topology_index = topology_index
        self.beta = float(beta)
        self._topo_cache = {}
        # Device mode derives all SiteStats from the graph. Site mode keeps a
        # separate NE-level link_stats summary for pair-level site features;
        # individual device size/degree columns remain zero there.
        graph_stats = _graph_site_stats(graph_context)
        self.site_stats = graph_stats if str(node_field) != "site_id" else None
        self.link_stats = graph_stats
        self.geo_stats = GeoStats(
            self.node_infos,
            getattr(graph_context, "site_coords", {}) if graph_context is not None else {},
        )
        # One cache lookup returns every pairwise site feature; Haversine and
        # link-count lookup are each performed at most once per unique site pair.
        self._site_pair_cache = {}
        self._device_link_ratio_cache = {}
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

    def _site_pair_features(self, site_a, site_b) -> tuple[float, ...]:
        """Cached structural/domain/geographic features for one site pair."""
        key = (site_a, site_b) if site_a < site_b else (site_b, site_a)
        hit = self._site_pair_cache.get(key)
        if hit is None:
            hit = self._site_pair_features_uncached(site_a, site_b)
            self._site_pair_cache[key] = hit
        return hit

    def _site_pair_features_uncached(self, site_a, site_b) -> tuple[float, ...]:
        """Structural/domain/geographic site features without cache growth."""
        link = (
            self.site_stats.link_feat(site_a, site_b)
            if self.site_stats is not None
            else 0.0
        )
        link_ratio = (
            self.link_stats.site_link_ratio(site_a, site_b)
            if self.link_stats is not None else 0.0
        )
        link_density = (
            self.link_stats.site_link_density(site_a, site_b)
            if self.link_stats is not None else 0.0
        )
        size_balance = (
            self.link_stats.site_size_balance(site_a, site_b)
            if self.link_stats is not None else 0.0
        )
        domain_cosine = (
            self.link_stats.site_domain_cosine(site_a, site_b)
            if self.link_stats is not None else 0.0
        )
        proximity, missing = self.geo_stats.pair_features(site_a, site_b)
        return (
            link,
            link_ratio,
            link_density,
            size_balance,
            domain_cosine,
            proximity,
            missing,
        )

    def _device_link_ratio(self, node_a, node_b) -> float:
        if self.site_stats is None:
            return 0.0
        key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
        hit = self._device_link_ratio_cache.get(key)
        if hit is None:
            hit = self.site_stats.device_link_ratio(node_a, node_b)
            self._device_link_ratio_cache[key] = hit
        return hit

    def _site_size(self, site) -> float:
        return self.site_stats.size_feat(site) if self.site_stats is not None else 0.0

    def _site_external_neighbor_count(self, site) -> float:
        return (
            self.link_stats.site_external_neighbor_count(site)
            if self.link_stats is not None else 0.0
        )

    def _device_degree(self, node) -> float:
        return self.site_stats.degree_feat(node) if self.site_stats is not None else 0.0

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
        src_size = np.empty(n, dtype=np.float64)
        site_link = np.empty(n, dtype=np.float64)
        site_link_ratio = np.empty(n, dtype=np.float64)
        site_link_density = np.empty(n, dtype=np.float64)
        site_size_balance = np.empty(n, dtype=np.float64)
        site_domain_cosine = np.empty(n, dtype=np.float64)
        src_degree = np.empty(n, dtype=np.float64)
        src_external_neighbors = np.empty(n, dtype=np.float64)
        device_link_ratio = np.empty(n, dtype=np.float64)
        geo_proximity = np.empty(n, dtype=np.float64)
        geo_missing = np.empty(n, dtype=np.float64)
        for i, sne in enumerate(src_nes):
            s_node = topo_node_of(sne)
            topo[i] = _topo_score(s_node, t_node, self.topology_index, self._topo_cache)
            is_same_ne[i] = 1.0 if sne == target_ne else 0.0
            s_site, s_vendor, s_netype = self._attr(sne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
            src_size[i] = self._site_size(s_site)
            (site_link[i], site_link_ratio[i], site_link_density[i],
             site_size_balance[i], site_domain_cosine[i], geo_proximity[i],
             geo_missing[i]) = self._site_pair_features(s_site, t_site)
            src_degree[i] = self._device_degree(s_node)
            src_external_neighbors[i] = self._site_external_neighbor_count(s_site)
            device_link_ratio[i] = self._device_link_ratio(s_node, t_node)
        tgt_size = np.full(n, self._site_size(t_site), dtype=np.float64)
        tgt_degree = np.full(n, self._device_degree(t_node), dtype=np.float64)
        tgt_external_neighbors = np.full(
            n, self._site_external_neighbor_count(t_site), dtype=np.float64
        )
        dom_u = dom_v = None
        if self.layout.n_dom:
            dom_u = np.full(n, self.layout.dom_id(phi_domain_of(target_ne, self.node_infos)), dtype=np.int64)
            dom_v = self.layout.domain_ids([phi_domain_of(s, self.node_infos) for s in src_nes])
        phi = self.layout.build_matrix(
            at_u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype,
            dom_u, dom_v, tgt_size, src_size, site_link,
            site_link_ratio=site_link_ratio, site_link_density=site_link_density,
            site_size_balance=site_size_balance, site_domain_cosine=site_domain_cosine,
            tgt_undirected_degree=tgt_degree, src_undirected_degree=src_degree,
            device_link_ratio=device_link_ratio,
            geo_proximity=geo_proximity, geo_missing=geo_missing,
            tgt_site_external_neighbor_count=tgt_external_neighbors,
            src_site_external_neighbor_count=src_external_neighbors,
        )
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
        tgt_size = np.empty(n, dtype=np.float64)
        site_link = np.empty(n, dtype=np.float64)
        site_link_ratio = np.empty(n, dtype=np.float64)
        site_link_density = np.empty(n, dtype=np.float64)
        site_size_balance = np.empty(n, dtype=np.float64)
        site_domain_cosine = np.empty(n, dtype=np.float64)
        tgt_degree = np.empty(n, dtype=np.float64)
        tgt_external_neighbors = np.empty(n, dtype=np.float64)
        device_link_ratio = np.empty(n, dtype=np.float64)
        geo_proximity = np.empty(n, dtype=np.float64)
        geo_missing = np.empty(n, dtype=np.float64)
        for i, tne in enumerate(tgt_nes):
            t_node = topo_node_of(tne)
            topo[i] = _topo_score(s_node, t_node, self.topology_index, self._topo_cache)
            is_same_ne[i] = 1.0 if source_ne == tne else 0.0
            t_site, t_vendor, t_netype = self._attr(tne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
            tgt_size[i] = self._site_size(t_site)
            (site_link[i], site_link_ratio[i], site_link_density[i],
             site_size_balance[i], site_domain_cosine[i], geo_proximity[i],
             geo_missing[i]) = self._site_pair_features(s_site, t_site)
            tgt_degree[i] = self._device_degree(t_node)
            tgt_external_neighbors[i] = self._site_external_neighbor_count(t_site)
            device_link_ratio[i] = self._device_link_ratio(s_node, t_node)
        src_size = np.full(n, self._site_size(s_site), dtype=np.float64)
        src_degree = np.full(n, self._device_degree(s_node), dtype=np.float64)
        src_external_neighbors = np.full(
            n, self._site_external_neighbor_count(s_site), dtype=np.float64
        )
        dom_u = dom_v = None
        if self.layout.n_dom:
            dom_v = np.full(n, self.layout.dom_id(phi_domain_of(source_ne, self.node_infos)), dtype=np.int64)
            dom_u = self.layout.domain_ids([phi_domain_of(t, self.node_infos) for t in tgt_nes])
        phi = self.layout.build_matrix(
            at_u, at_v, topo, is_same_ne, same_site, same_vendor, same_netype,
            dom_u, dom_v, tgt_size, src_size, site_link,
            site_link_ratio=site_link_ratio, site_link_density=site_link_density,
            site_size_balance=site_size_balance, site_domain_cosine=site_domain_cosine,
            tgt_undirected_degree=tgt_degree, src_undirected_degree=src_degree,
            device_link_ratio=device_link_ratio,
            geo_proximity=geo_proximity, geo_missing=geo_missing,
            tgt_site_external_neighbor_count=tgt_external_neighbors,
            src_site_external_neighbor_count=src_external_neighbors,
        )
        if self.n_dynamic > 0:
            if self.source_dynamic_dim and src_mark is None:
                raise ValueError("src_mark is required for source dynamic mode")
            if self.target_dynamic_dim and tgt_marks is None:
                raise ValueError("tgt_marks is required for target dynamic mode")
            marks = self._dynamic_mark_matrix(n, src_marks=src_mark, tgt_marks=tgt_marks)
            phi = np.concatenate([phi, marks], axis=1)
        return self.kernel.alpha(phi)


class EntityStaticTable:
    """Target-independent per-entity columns plus sparse adjacency indexes.

    Built by ``DecomposedFeatureScorer.entity_static_table`` and consumed by
    ``entity_parts_from_table``; also caches per-target-site site-pair feature
    rows across targets. Attributes are assigned by the builder.
    """

    @staticmethod
    def rows(mapping, key):
        """Return one compact row-map value through a uniform iterable view."""
        value = mapping.get(key)
        if value is None:
            return ()
        if isinstance(value, (int, np.integer)):
            return (int(value),)
        return value


class DecomposedFeatureScorer:
    """φ-decomposed live α — numerically equal to RuntimeFeatureScorer but with
    NO (C, F) feature-matrix construction.

    The FeatureLayout is a fixed linear layout (bias + at-pair one-hot + 20
    scalars + the second-order cross block + optional domain one-hot + dynamic
    mark bits), so w·φ collapses to table lookups + scalar terms:

        z = w_bias + W_at[at_u, at_v]
          + Σ_s w_s · scalar_s                          (the 20 scalar columns)
          + Σ_{(a,b) ∈ CROSS_PAIRS} w_ab · scalar_a·scalar_b
          + W_dom'[dom_u, dom_v]                        (same_domain folded in)
          + SRC_TABLE[mark_combo] + tgt_term            (dynamic bits)
        α = alpha_scale · softplus(z)

    The one-hot blocks become the precomputed W_at / W_dom' tables; the 8
    possible source-mark combos become an 8-entry table; the target-side
    dynamic term is a per-target constant. To match φ bit-for-bit, every scalar
    (and every cross product) is rounded through float32 exactly like
    FeatureLayout.build_matrix stores it.
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
         self.w_tgt_site_size, self.w_src_site_size,
         self.w_site_link, self.w_site_link_ratio,
         self.w_site_link_density, self.w_site_size_balance,
         self.w_site_domain_cosine, self.w_tgt_undirected_degree,
         self.w_src_undirected_degree,
         self.w_device_link_ratio, self.w_geo_proximity,
         self.w_geo_missing, self.w_tgt_site_external_neighbor_count,
         self.w_src_site_external_neighbor_count) = (float(x) for x in w[j:j + 20])
        j += 20
        n_cross = len(layout.CROSS_PAIRS)
        self.w_cross = w[j:j + n_cross].copy()
        j += n_cross
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
        tgt_site_size: float = 0.0,
        src_site_size: np.ndarray = None,
        site_link: np.ndarray = None,
        site_link_ratio: np.ndarray = None,
        site_link_density: np.ndarray = None,
        site_size_balance: np.ndarray = None,
        site_domain_cosine: np.ndarray = None,
        tgt_undirected_degree: float = 0.0,
        src_undirected_degree: np.ndarray = None,
        device_link_ratio: np.ndarray = None,
        geo_proximity: np.ndarray = None,
        geo_missing: np.ndarray = None,
        tgt_site_external_neighbor_count: float = 0.0,
        src_site_external_neighbor_count: np.ndarray = None,
    ) -> np.ndarray:
        """w·φ for a batch of source candidates against one target, from
        precomputed per-candidate parts. All boolean arrays are 0/1 float."""
        u = int(tgt_at_id)
        at_v = np.asarray(src_at_ids, dtype=np.int64)

        # Match build_matrix: φ stores every scalar column as float32; round
        # through float32 so w·φ is numerically identical. ``None`` inputs are
        # zero columns (0.0 scalars) exactly like build_matrix's _col.
        def _f32a(x):
            return 0.0 if x is None else np.asarray(x, dtype=np.float32).astype(np.float64)

        def _f32s(x):
            return float(np.float32(x))

        same_at = ((at_v == u) & (u >= 0) & (at_v >= 0)).astype(np.float64)
        vals = {
            "same_alarm_type": same_at,
            "topo_score": _f32a(topo),
            "is_same_ne": _f32a(is_same_ne),
            "same_site": _f32a(same_site),
            "same_vendor": _f32a(same_vendor),
            "same_ne_type": _f32a(same_netype),
            "tgt_site_size": _f32s(tgt_site_size),
            "src_site_size": _f32a(src_site_size),
            "site_link_score": _f32a(site_link),
            "site_link_ratio": _f32a(site_link_ratio),
            "site_link_density": _f32a(site_link_density),
            "site_size_balance": _f32a(site_size_balance),
            "site_domain_cosine": _f32a(site_domain_cosine),
            "tgt_undirected_degree": _f32s(tgt_undirected_degree),
            "src_undirected_degree": _f32a(src_undirected_degree),
            "device_link_ratio": _f32a(device_link_ratio),
            "geo_proximity": _f32a(geo_proximity),
            "geo_distance_missing": _f32a(geo_missing),
            "tgt_site_external_neighbor_count": _f32s(tgt_site_external_neighbor_count),
            "src_site_external_neighbor_count": _f32a(src_site_external_neighbor_count),
        }
        z = (
            self.w_bias
            + self.W_at_pad[u + 1, at_v + 1]
            + self.w_same_at * same_at
            + self.w_topo * vals["topo_score"]
            + self.w_same_ne * vals["is_same_ne"]
            + self.w_same_site * vals["same_site"]
            + self.w_same_vendor * vals["same_vendor"]
            + self.w_same_netype * vals["same_ne_type"]
            + self.w_tgt_site_size * vals["tgt_site_size"]
            + self.w_src_site_size * vals["src_site_size"]
            + self.w_site_link * vals["site_link_score"]
            + self.w_site_link_ratio * vals["site_link_ratio"]
            + self.w_site_link_density * vals["site_link_density"]
            + self.w_site_size_balance * vals["site_size_balance"]
            + self.w_site_domain_cosine * vals["site_domain_cosine"]
            + self.w_tgt_undirected_degree * vals["tgt_undirected_degree"]
            + self.w_src_undirected_degree * vals["src_undirected_degree"]
            + self.w_device_link_ratio * vals["device_link_ratio"]
            + self.w_geo_proximity * vals["geo_proximity"]
            + self.w_geo_missing * vals["geo_distance_missing"]
            + self.w_tgt_site_external_neighbor_count * vals["tgt_site_external_neighbor_count"]
            + self.w_src_site_external_neighbor_count * vals["src_site_external_neighbor_count"]
        )
        # Cross block: products of the SAME float32-rounded bases, re-rounded
        # like φ stores them (f32×f32 = round(exact product)). Zero-weight
        # crosses are skipped — after ridge most stay tiny but nonzero, so this
        # mainly saves work for hand-zeroed kernels.
        for w_k, (na, nb) in zip(self.w_cross, self.layout.CROSS_PAIRS):
            if w_k == 0.0:
                continue
            prod = vals[na] * vals[nb]
            if isinstance(prod, np.ndarray):
                prod = prod.astype(np.float32).astype(np.float64)
            else:
                prod = float(np.float32(prod))
            z = z + w_k * prod
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

    def entity_parts_for_target(self, target_ne, src_nes):
        """Entity-pair feature parts for one target against many source entities.

        Returns keyword arguments for ``logits_from_parts`` covering every
        input that depends only on the (source entity, target entity) pair —
        topology, attribute, site, geo, and domain columns. Alarm-type ids and
        dynamic mark terms stay with the caller, so one parts dict can be
        shared across every alarm type of the same entity pair.
        """
        s = self.scorer
        n = len(src_nes)
        t_node = topo_node_of(target_ne)
        t_site, t_vendor, t_netype = s._attr(target_ne)
        topo = np.empty(n, dtype=np.float64)
        is_same_ne = np.empty(n, dtype=np.float64)
        same_site = np.empty(n, dtype=np.float64)
        same_vendor = np.empty(n, dtype=np.float64)
        same_netype = np.empty(n, dtype=np.float64)
        src_size = np.empty(n, dtype=np.float64)
        site_link = np.empty(n, dtype=np.float64)
        site_link_ratio = np.empty(n, dtype=np.float64)
        site_link_density = np.empty(n, dtype=np.float64)
        site_size_balance = np.empty(n, dtype=np.float64)
        site_domain_cosine = np.empty(n, dtype=np.float64)
        src_degree = np.empty(n, dtype=np.float64)
        src_external_neighbors = np.empty(n, dtype=np.float64)
        device_link_ratio = np.empty(n, dtype=np.float64)
        geo_proximity = np.empty(n, dtype=np.float64)
        geo_missing = np.empty(n, dtype=np.float64)
        for i, sne in enumerate(src_nes):
            s_node = topo_node_of(sne)
            topo[i] = _topo_score(s_node, t_node, s.topology_index, s._topo_cache)
            is_same_ne[i] = 1.0 if sne == target_ne else 0.0
            s_site, s_vendor, s_netype = s._attr(sne)
            same_site[i] = 1.0 if (t_site and t_site == s_site) else 0.0
            same_vendor[i] = 1.0 if (t_vendor and t_vendor == s_vendor) else 0.0
            same_netype[i] = 1.0 if (t_netype and t_netype == s_netype) else 0.0
            src_size[i] = s._site_size(s_site)
            (site_link[i], site_link_ratio[i], site_link_density[i],
             site_size_balance[i], site_domain_cosine[i], geo_proximity[i],
             geo_missing[i]) = s._site_pair_features(s_site, t_site)
            src_degree[i] = s._device_degree(s_node)
            src_external_neighbors[i] = s._site_external_neighbor_count(s_site)
            device_link_ratio[i] = s._device_link_ratio(s_node, t_node)
        tgt_dom_id = -1
        src_dom_ids = np.full(n, -1, dtype=np.int64)
        if self.n_dom:
            tgt_dom_id = self.layout.dom_id(phi_domain_of(target_ne, s.node_infos))
            src_dom_ids = self.layout.domain_ids([phi_domain_of(x, s.node_infos) for x in src_nes])
        return {
            "topo": topo, "is_same_ne": is_same_ne, "same_site": same_site,
            "same_vendor": same_vendor, "same_netype": same_netype,
            "tgt_dom_id": tgt_dom_id, "src_dom_ids": src_dom_ids,
            "tgt_site_size": s._site_size(t_site), "src_site_size": src_size,
            "site_link": site_link, "site_link_ratio": site_link_ratio,
            "site_link_density": site_link_density,
            "site_size_balance": site_size_balance,
            "site_domain_cosine": site_domain_cosine,
            "tgt_undirected_degree": s._device_degree(t_node),
            "src_undirected_degree": src_degree,
            "device_link_ratio": device_link_ratio, "geo_proximity": geo_proximity,
            "geo_missing": geo_missing,
            "tgt_site_external_neighbor_count": s._site_external_neighbor_count(t_site),
            "src_site_external_neighbor_count": src_external_neighbors,
        }

    # CROSS_FEATURES column name -> entity-parts dict key (identity omitted).
    _CROSS_NAME_TO_PART = {
        "topo_score": "topo",
        "same_ne_type": "same_netype",
        "site_link_score": "site_link",
    }

    def same_at_delta_from_parts(self, parts):
        """Per-entity logit shift of the v == u (same alarm type) row.

        Equals w_same_at plus every cross column involving same_alarm_type:
        with same_alarm_type exactly 1.0 the stored f32 product collapses to
        the f32-rounded partner scalar, so twelve fused ops replace a full
        ``logits_from_parts`` pass. Real-arithmetic equal to
        lfp(0, 0) - lfp(-1, -1) - W_at[0, 0]; consumed only by the prescreen,
        whose margin absorbs the float reassociation difference.
        """
        delta = np.full(len(parts["topo"]), self.w_same_at, dtype=np.float64)
        for w_k, (a, b) in zip(self.w_cross, self.layout.CROSS_PAIRS):
            if w_k == 0.0 or "same_alarm_type" not in (a, b):
                continue
            partner = b if a == "same_alarm_type" else a
            value = parts[self._CROSS_NAME_TO_PART.get(partner, partner)]
            if isinstance(value, np.ndarray):
                delta += w_k * np.asarray(value, dtype=np.float32).astype(np.float64)
            else:
                delta += w_k * float(np.float32(value))
        return delta

    def entity_static_table(self, entities, row_of=None):
        """Precompute target-independent entity columns for the offline compiler.

        One pass over the entity universe captures attribute codes, structural
        scalars, domain ids, and sparse adjacency indexes. Together with
        ``entity_parts_from_table`` this replaces the per-target Python loop of
        ``entity_parts_for_target`` when the same entity universe is scored
        against many targets (global candidate scope).
        """
        from alarm_flow_isahp.ne_topology import _normalize_ne_id

        s = self.scorer
        n = len(entities)
        table = EntityStaticTable()
        table.n = n
        table.entities = (
            entities if isinstance(entities, (list, tuple)) else tuple(entities)
        )
        table.row_of = (
            row_of
            if row_of is not None
            else {entity: i for i, entity in enumerate(table.entities)}
        )
        table.nodes = []
        table.site_code_of = {}
        table.vendor_code_of = {}
        table.netype_code_of = {}
        table.site_list = []
        table.site_codes = np.empty(n, dtype=np.int64)
        table.vendor_codes = np.empty(n, dtype=np.int64)
        table.netype_codes = np.empty(n, dtype=np.int64)
        table.src_site_size = np.empty(n, dtype=np.float64)
        table.src_degree = np.empty(n, dtype=np.float64)
        table.src_external = np.empty(n, dtype=np.float64)
        node_rows = {}
        nodes_are_normalized = True

        def _add_row(mapping, key, row):
            previous = mapping.get(key)
            if previous is None:
                mapping[key] = row
            elif isinstance(previous, list):
                previous.append(row)
            else:
                mapping[key] = [previous, row]

        def _finalize_rows(mapping):
            return {
                key: (
                    np.asarray(value, dtype=np.int64)
                    if isinstance(value, list)
                    else value
                )
                for key, value in mapping.items()
            }

        def _code(mapping, value, values=None):
            code = mapping.get(value)
            if code is None:
                code = mapping[value] = len(mapping)
                if values is not None:
                    values.append(value)
            return code

        for i, entity in enumerate(table.entities):
            node = topo_node_of(entity)
            table.nodes.append(node)
            _add_row(node_rows, node, i)
            nodes_are_normalized = (
                nodes_are_normalized and _normalize_ne_id(node) == node
            )
            site, vendor, netype = s._attr(entity)
            table.site_codes[i] = _code(table.site_code_of, site, table.site_list)
            table.vendor_codes[i] = _code(table.vendor_code_of, vendor)
            table.netype_codes[i] = _code(table.netype_code_of, netype)
            table.src_site_size[i] = s._site_size(site)
            table.src_degree[i] = s._device_degree(node)
            table.src_external[i] = s._site_external_neighbor_count(site)
        table.node_rows = _finalize_rows(node_rows)
        if nodes_are_normalized:
            table.norm_rows = table.node_rows
        else:
            norm_rows = {}
            for row, node in enumerate(table.nodes):
                _add_row(norm_rows, _normalize_ne_id(node), row)
            table.norm_rows = _finalize_rows(norm_rows)
        table.src_dom_ids = (
            self.layout.domain_ids([phi_domain_of(x, s.node_infos) for x in table.entities])
            if self.n_dom
            else np.full(n, -1, dtype=np.int64)
        )
        # topo_score(src, tgt) is nonzero only when the normalized nodes match
        # or tgt appears in src's undirected-hop row; invert that row lookup so
        # each target enumerates its possible sources directly.
        hops = (
            (getattr(s.topology_index, "undirected_hops", {}) or {})
            if s.topology_index is not None
            else {}
        )
        topo_sources = defaultdict(set)
        for source_norm, row in hops.items():
            for target_norm, hop in (row or {}).items():
                if hop:
                    topo_sources[target_norm].add(source_norm)
        table.topo_sources = dict(topo_sources)
        table.link_neighbors = (
            s.site_stats.undirected_neighbor_map() if s.site_stats is not None else {}
        )
        table.site_pair_rows = {}
        return table

    def entity_parts_from_table(self, target_ne, table):
        """``entity_parts_for_target`` over a static table, elementwise equal.

        Dense columns come from vectorized code comparisons and per-site
        gathers; the two node-pair columns (topo score, device link ratio) are
        filled sparsely from the target node's adjacency and are exactly zero
        elsewhere. All values are produced by the same underlying feature
        functions as the loop path, so downstream logits are bit-identical.
        """
        from alarm_flow_isahp.ne_topology import _normalize_ne_id

        s = self.scorer
        n = table.n
        t_node = topo_node_of(target_ne)
        t_site, t_vendor, t_netype = s._attr(target_ne)

        topo = np.zeros(n, dtype=np.float64)
        if s.topology_index is not None and t_node:
            t_norm = _normalize_ne_id(t_node)
            candidate_norms = {t_norm}
            candidate_norms.update(table.topo_sources.get(t_norm, ()))
            for source_norm in candidate_norms:
                for row in table.rows(table.norm_rows, source_norm):
                    topo[row] = _topo_score(
                        table.nodes[row], t_node, s.topology_index, s._topo_cache
                    )

        is_same_ne = np.zeros(n, dtype=np.float64)
        own_row = table.row_of.get(target_ne)
        if own_row is not None:
            is_same_ne[own_row] = 1.0

        def _same(codes, code_of, value):
            if not value:
                return np.zeros(n, dtype=np.float64)
            code = code_of.get(value)
            if code is None:
                return np.zeros(n, dtype=np.float64)
            return (codes == code).astype(np.float64)

        same_site = _same(table.site_codes, table.site_code_of, t_site)
        same_vendor = _same(table.vendor_codes, table.vendor_code_of, t_vendor)
        same_netype = _same(table.netype_codes, table.netype_code_of, t_netype)

        pair_rows = table.site_pair_rows.get(t_site)
        if pair_rows is None:
            pair_rows = np.asarray(
                [s._site_pair_features(site, t_site) for site in table.site_list],
                dtype=np.float64,
            )
            table.site_pair_rows[t_site] = pair_rows
        pair_cols = pair_rows[table.site_codes]

        device_link_ratio = np.zeros(n, dtype=np.float64)
        if s.site_stats is not None and t_node:
            for neighbor in table.link_neighbors.get(t_node, ()):
                rows = table.rows(table.node_rows, neighbor)
                if len(rows):
                    device_link_ratio[rows] = s._device_link_ratio(neighbor, t_node)

        tgt_dom_id = (
            self.layout.dom_id(phi_domain_of(target_ne, s.node_infos))
            if self.n_dom
            else -1
        )
        return {
            "topo": topo, "is_same_ne": is_same_ne, "same_site": same_site,
            "same_vendor": same_vendor, "same_netype": same_netype,
            "tgt_dom_id": tgt_dom_id, "src_dom_ids": table.src_dom_ids,
            "tgt_site_size": s._site_size(t_site), "src_site_size": table.src_site_size,
            "site_link": pair_cols[:, 0], "site_link_ratio": pair_cols[:, 1],
            "site_link_density": pair_cols[:, 2],
            "site_size_balance": pair_cols[:, 3],
            "site_domain_cosine": pair_cols[:, 4],
            "tgt_undirected_degree": s._device_degree(t_node),
            "src_undirected_degree": table.src_degree,
            "device_link_ratio": device_link_ratio,
            "geo_proximity": pair_cols[:, 5],
            "geo_missing": pair_cols[:, 6],
            "tgt_site_external_neighbor_count": s._site_external_neighbor_count(t_site),
            "src_site_external_neighbor_count": table.src_external,
        }

    def entity_parts_from_table_rows(self, target_ne, table, rows):
        """Build entity parts only for sorted candidate rows of a static table.

        Unlike ``entity_parts_from_table``, every dense temporary has exactly
        ``len(rows)`` entries. Sparse topology/device-link columns intersect
        their full-table adjacency rows with the candidate rows, and site-pair
        features are computed once for the unique sites present in this target's
        candidates without populating an unbounded cross-target cache.
        """
        from alarm_flow_isahp.ne_topology import _normalize_ne_id

        s = self.scorer
        rows = np.asarray(rows, dtype=np.int64)
        n = len(rows)
        invalid_rows = n and (
            np.any(rows[1:] <= rows[:-1])
            or rows[0] < 0
            or rows[-1] >= table.n
        )
        if invalid_rows:
            raise ValueError("candidate table rows must be sorted, unique, and in range")

        t_node = topo_node_of(target_ne)
        t_site, t_vendor, t_netype = s._attr(target_ne)

        def _candidate_positions(full_rows):
            full_rows = np.atleast_1d(np.asarray(full_rows, dtype=np.int64))
            if not n or not len(full_rows):
                return np.zeros(0, dtype=np.int64)
            positions = np.searchsorted(rows, full_rows)
            valid = positions < n
            if not np.any(valid):
                return np.zeros(0, dtype=np.int64)
            positions = positions[valid]
            full_rows = full_rows[valid]
            return positions[rows[positions] == full_rows]

        topo = np.zeros(n, dtype=np.float64)
        if s.topology_index is not None and t_node:
            t_norm = _normalize_ne_id(t_node)
            candidate_norms = {t_norm}
            candidate_norms.update(table.topo_sources.get(t_norm, ()))
            for source_norm in candidate_norms:
                full_rows = table.rows(table.norm_rows, source_norm)
                local_rows = _candidate_positions(full_rows)
                for local_row in local_rows:
                    table_row = rows[local_row]
                    topo[local_row] = _topo_score(
                        table.nodes[table_row],
                        t_node,
                        s.topology_index,
                        s._topo_cache,
                    )

        is_same_ne = np.zeros(n, dtype=np.float64)
        own_row = table.row_of.get(target_ne)
        if own_row is not None and n:
            own_position = int(np.searchsorted(rows, own_row))
            if own_position < n and rows[own_position] == own_row:
                is_same_ne[own_position] = 1.0

        site_codes = table.site_codes[rows]
        vendor_codes = table.vendor_codes[rows]
        netype_codes = table.netype_codes[rows]

        def _same(codes, code_of, value):
            if not value:
                return np.zeros(n, dtype=np.float64)
            code = code_of.get(value)
            if code is None:
                return np.zeros(n, dtype=np.float64)
            return (codes == code).astype(np.float64)

        same_site = _same(site_codes, table.site_code_of, t_site)
        same_vendor = _same(vendor_codes, table.vendor_code_of, t_vendor)
        same_netype = _same(netype_codes, table.netype_code_of, t_netype)

        unique_site_codes, site_inverse = np.unique(
            site_codes, return_inverse=True
        )
        pair_rows = np.asarray(
            [
                s._site_pair_features_uncached(
                    table.site_list[int(site_code)], t_site
                )
                for site_code in unique_site_codes
            ],
            dtype=np.float64,
        ).reshape((-1, 7))
        pair_cols = pair_rows[site_inverse]

        device_link_ratio = np.zeros(n, dtype=np.float64)
        if s.site_stats is not None and t_node:
            for neighbor in table.link_neighbors.get(t_node, ()):
                local_rows = _candidate_positions(
                    table.rows(table.node_rows, neighbor)
                )
                if len(local_rows):
                    device_link_ratio[local_rows] = s._device_link_ratio(
                        neighbor, t_node
                    )

        tgt_dom_id = (
            self.layout.dom_id(phi_domain_of(target_ne, s.node_infos))
            if self.n_dom
            else -1
        )
        return {
            "topo": topo,
            "is_same_ne": is_same_ne,
            "same_site": same_site,
            "same_vendor": same_vendor,
            "same_netype": same_netype,
            "tgt_dom_id": tgt_dom_id,
            "src_dom_ids": table.src_dom_ids[rows],
            "tgt_site_size": s._site_size(t_site),
            "src_site_size": table.src_site_size[rows],
            "site_link": pair_cols[:, 0],
            "site_link_ratio": pair_cols[:, 1],
            "site_link_density": pair_cols[:, 2],
            "site_size_balance": pair_cols[:, 3],
            "site_domain_cosine": pair_cols[:, 4],
            "tgt_undirected_degree": s._device_degree(t_node),
            "src_undirected_degree": table.src_degree[rows],
            "device_link_ratio": device_link_ratio,
            "geo_proximity": pair_cols[:, 5],
            "geo_missing": pair_cols[:, 6],
            "tgt_site_external_neighbor_count": s._site_external_neighbor_count(t_site),
            "src_site_external_neighbor_count": table.src_external[rows],
        }

    def logits_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None):
        """Static/source-state logits for one target against many sources.

        The target-state contribution is deliberately omitted so callers that
        need all eight target states can compute the relatively expensive
        topology/attribute parts once, then add eight scalar target terms.
        Entity parts are evaluated once per unique source entity and expanded
        by gather, which keeps candidate lists that repeat each entity once per
        alarm type (the offline compiler universe) off the slow Python path.
        """
        s = self.scorer
        n = len(src_nes)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        u = s.at_to_id.get(str(target_at), -1)
        at_v = np.array([s.at_to_id.get(str(a), -1) for a in src_ats], dtype=np.int64)
        index_of = {}
        inverse = np.empty(n, dtype=np.int64)
        unique_nes = []
        for i, sne in enumerate(src_nes):
            j = index_of.get(sne)
            if j is None:
                j = index_of[sne] = len(unique_nes)
                unique_nes.append(sne)
            inverse[i] = j
        parts = self.entity_parts_for_target(target_ne, unique_nes)
        if len(unique_nes) != n:
            parts = {
                key: value[inverse] if isinstance(value, np.ndarray) else value
                for key, value in parts.items()
            }
        # dynamic marks: mirror _dynamic_mark_matrix semantics for the common
        # streaming shapes — per-candidate (n,3) src marks + one target mark row.
        src_mark_idx = np.zeros(n, dtype=np.int64)
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
        return self.logits_from_parts(
            u, at_v, src_mark_idx=src_mark_idx, tgt_term=0.0, **parts
        )

    def alpha_for_target(self, target_at, target_ne, src_ats, src_nes, src_marks=None, tgt_marks=None):
        """Drop-in equivalent of RuntimeFeatureScorer.alpha_for_target (string
        inputs). Attribute/topology lookups reuse the wrapped scorer's caches."""
        z = self.logits_for_target(
            target_at, target_ne, src_ats, src_nes, src_marks=src_marks
        )
        if len(self.w_dyn_tgt) and tgt_marks is not None:
            tm = np.asarray(tgt_marks, dtype=np.float64).reshape(-1)
            target_term = self.tgt_term(tuple(tm[:3]))
            if target_term:
                z = z + target_term
        a = self._softplus(z)
        if self.alpha_scale != 1.0:
            a = a * self.alpha_scale
        return a
