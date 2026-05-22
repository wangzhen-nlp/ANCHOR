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
    """Small topology facade for site-hop relations and NE metadata.

    The current rule pipeline already maintains a site graph and an NE graph.
    Cascade clustering uses the site graph as an undirected affinity graph; it
    does not require root-cause direction to cluster alarms.
    """

    def __init__(self, site_graph=None, ne_graph=None):
        self.site_graph = site_graph or {}
        self.ne_graph = ne_graph or {}
        self.adjacency = defaultdict(set)
        self.ne_to_site = {}
        self.ne_to_domain = {}
        self.ne_to_type = {}
        self._hop_cache = {}
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

    def resolve_site(self, site_id="", alarm_source=""):
        return _text(site_id) or self.ne_to_site.get(_text(alarm_source), "")

    def resolve_domain(self, alarm_source="", fallback=""):
        return self.ne_to_domain.get(_text(alarm_source), _text(fallback))

    def resolve_device_type(self, alarm_source="", fallback=""):
        return self.ne_to_type.get(_text(alarm_source), _text(fallback))

    def hop_distance(self, left_site, right_site, max_hops=2):
        left_site = _text(left_site)
        right_site = _text(right_site)
        if not left_site or not right_site:
            return None
        if left_site == right_site:
            return 0
        max_hops = max(int(max_hops), 0)
        cache_key = (left_site, right_site, max_hops)
        if cache_key in self._hop_cache:
            return self._hop_cache[cache_key]

        queue = deque([(left_site, 0)])
        seen = {left_site}
        found = None
        while queue:
            site_id, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor in self.adjacency.get(site_id, ()):
                if neighbor in seen:
                    continue
                next_hop = hop + 1
                if neighbor == right_site:
                    found = next_hop
                    queue.clear()
                    break
                seen.add(neighbor)
                queue.append((neighbor, next_hop))
        self._hop_cache[cache_key] = found
        self._hop_cache[(right_site, left_site, max_hops)] = found
        return found

    def relation(self, left_event, right_event, max_hops=2):
        left_source = _text(left_event.alarm_source)
        right_source = _text(right_event.alarm_source)
        if left_source and left_source == right_source:
            return "same_device"

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

        left_domain = _text(left_event.device_domain)
        right_domain = _text(right_event.device_domain)
        if left_domain and left_domain == right_domain:
            return "same_domain"
        if not left_site or not right_site:
            return "unknown"
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
