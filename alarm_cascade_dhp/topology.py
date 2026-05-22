import json

from collections import defaultdict, deque


def _text(value):
    return str(value or "").strip()


def _iter_neighbors(raw_neighbors):
    if isinstance(raw_neighbors, dict):
        yield from raw_neighbors.keys()
    elif isinstance(raw_neighbors, (list, tuple, set)):
        yield from raw_neighbors


class TopologyIndex:
    """Topology facade for explicit NE links plus weaker site context.

    The gateable relations come from same-NE or ne_graph link hops. Site graph
    relations remain undirected soft affinity signals for cascade clustering.
    """

    def __init__(self, site_graph=None, ne_graph=None):
        self.site_graph = site_graph or {}
        self.ne_graph = ne_graph or {}
        self.adjacency = defaultdict(set)
        self.ne_adjacency = defaultdict(set)
        self.ne_to_site = {}
        self.ne_to_domain = {}
        self.ne_to_type = {}
        self._hop_cache = {}
        self._ne_hop_cache = {}
        self._build_site_adjacency(self.site_graph)
        self._build_ne_indexes(self.ne_graph)

    @classmethod
    def from_files(cls, site_graph_path="", ne_graph_path=""):
        return cls(
            site_graph=_load_json_if_present(site_graph_path),
            ne_graph=_load_json_if_present(ne_graph_path),
        )

    def _build_site_adjacency(self, site_graph):
        for raw_site, raw_neighbors in (site_graph or {}).items():
            site_id = _text(raw_site)
            if not site_id:
                continue
            self.adjacency.setdefault(site_id, set())
            for raw_neighbor in _iter_neighbors(raw_neighbors):
                neighbor = _text(raw_neighbor)
                if not neighbor or neighbor == site_id:
                    continue
                self.adjacency[site_id].add(neighbor)
                self.adjacency[neighbor].add(site_id)

    def _build_ne_indexes(self, ne_graph):
        for raw_ne_id, raw_info in (ne_graph or {}).items():
            if not isinstance(raw_info, dict):
                continue
            ne_id = _text(raw_ne_id)
            if not ne_id:
                continue
            site_id = _text(raw_info.get("site_id"))
            domain = _text(
                raw_info.get("domain")
                or raw_info.get("网络专业")
                or raw_info.get("network_domain")
            )
            device_type = _text(
                raw_info.get("type")
                or raw_info.get("设备类型")
                or raw_info.get("device_type")
            )
            if site_id:
                self.ne_to_site[ne_id] = site_id
                self.adjacency.setdefault(site_id, set())
            if domain:
                self.ne_to_domain[ne_id] = domain
            if device_type:
                self.ne_to_type[ne_id] = device_type
            self.ne_adjacency.setdefault(ne_id, set())
            for raw_neighbor in _iter_neighbors(raw_info.get("link", {})):
                neighbor = _text(raw_neighbor)
                if neighbor and neighbor != ne_id:
                    self.ne_adjacency[ne_id].add(neighbor)
                    self.ne_adjacency[neighbor].add(ne_id)

    def resolve_site(self, site_id="", alarm_source=""):
        return _text(site_id) or self.ne_to_site.get(_text(alarm_source), "")

    def resolve_domain(self, alarm_source="", fallback=""):
        return self.ne_to_domain.get(_text(alarm_source), _text(fallback))

    def resolve_device_type(self, alarm_source="", fallback=""):
        return self.ne_to_type.get(_text(alarm_source), _text(fallback))

    def hop_distance(self, left_site, right_site, max_hops=2):
        return _hop_distance(
            left_site,
            right_site,
            self.adjacency,
            self._hop_cache,
            max_hops=max_hops,
        )

    def ne_hop_distance(self, left_ne, right_ne, max_hops=2):
        return _hop_distance(
            left_ne,
            right_ne,
            self.ne_adjacency,
            self._ne_hop_cache,
            max_hops=max_hops,
        )

    def explicit_ne_relation(self, left_event, right_event, max_hops=2):
        """Return only same-NE or explicit ne_graph link relations."""
        left_source = _text(left_event.alarm_source)
        right_source = _text(right_event.alarm_source)
        if left_source and left_source == right_source:
            return "same_device"

        hop = self.ne_hop_distance(left_source, right_source, max_hops=max_hops)
        if hop == 1:
            return "ne_hop_1"
        if hop == 2:
            return "ne_hop_2"
        if hop is not None:
            return "ne_hop_far"
        return ""

    def relation(self, left_event, right_event, max_hops=2):
        explicit_relation = self.explicit_ne_relation(
            left_event,
            right_event,
            max_hops=max_hops,
        )
        if explicit_relation:
            return explicit_relation

        left_site = _text(left_event.site_id)
        right_site = _text(right_event.site_id)
        if left_site and left_site == right_site:
            return "same_site"

        hop = self.hop_distance(left_site, right_site, max_hops=max_hops)
        if hop == 1:
            return "hop_1"
        if hop == 2:
            return "hop_2"
        if hop is not None:
            return "hop_far"

        if left_site and right_site:
            return "disconnected"

        left_domain = _text(left_event.device_domain)
        right_domain = _text(right_event.device_domain)
        if left_domain and left_domain == right_domain:
            return "same_domain"
        return "disconnected"

    def content_context_tokens(self, site_id, alarm_source, max_hops=1, limit=16):
        """Encode baseline topology hints as document tokens."""
        site_id = self.resolve_site(site_id, alarm_source)
        alarm_source = _text(alarm_source)
        tokens = []
        if site_id:
            tokens.append(f"site:{site_id}")
        if alarm_source:
            tokens.append(f"device:{alarm_source}")
            domain = self.resolve_domain(alarm_source)
            if domain:
                tokens.append(f"device_domain:{domain}")
            device_type = self.resolve_device_type(alarm_source)
            if device_type:
                tokens.append(f"device_type:{device_type}")

        if not site_id or max_hops <= 0 or limit <= 0:
            return tokens

        queue = deque([(site_id, 0)])
        seen = {site_id}
        emitted = 0
        while queue and emitted < limit:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor in sorted(self.adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_hop = hop + 1
                tokens.append(f"topo_site_hop_{next_hop}:{neighbor}")
                emitted += 1
                if emitted >= limit:
                    break
                queue.append((neighbor, next_hop))
        return tokens


def _hop_distance(left, right, adjacency, cache, max_hops=2):
    left = _text(left)
    right = _text(right)
    if not left or not right:
        return None
    if left == right:
        return 0
    max_hops = max(int(max_hops), 0)
    cache_key = (left, right, max_hops)
    if cache_key in cache:
        return cache[cache_key]

    queue = deque([(left, 0)])
    seen = {left}
    found = None
    while queue:
        item, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for neighbor in adjacency.get(item, ()):
            if neighbor in seen:
                continue
            next_hop = hop + 1
            if neighbor == right:
                found = next_hop
                queue.clear()
                break
            seen.add(neighbor)
            queue.append((neighbor, next_hop))
    cache[cache_key] = found
    cache[(right, left, max_hops)] = found
    return found


def _load_json_if_present(path):
    path = _text(path)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}
