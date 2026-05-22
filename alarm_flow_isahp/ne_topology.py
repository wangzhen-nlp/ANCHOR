import json

from collections import defaultdict, deque
from dataclasses import dataclass

from fault_grouping.site_topology import extract_link_direction_values


PAIR_FEATURE_NAMES = (
    "same_alarm_source",
    "direct_source_to_target",
    "direct_target_to_source",
    "direct_bidirection",
    "reachable_source_to_target",
    "reachable_target_to_source",
    "undirected_reachable",
    "directed_hop_inverse",
    "reverse_directed_hop_inverse",
    "undirected_hop_inverse",
)


def _normalize_ne_id(value):
    return str(value or "").strip()


def _update_shortest_hop(hops, target, hop):
    previous = hops.get(target)
    if previous is None or hop < previous:
        hops[target] = hop


def _walk_hops(graph, source, max_hops):
    if source not in graph or max_hops <= 0:
        return {}
    hops = {}
    queue = deque([(source, 0)])
    while queue:
        node, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for neighbor in graph.get(node, ()):
            next_hop = hop + 1
            previous = hops.get(neighbor)
            if neighbor == source or (previous is not None and previous <= next_hop):
                continue
            hops[neighbor] = next_hop
            queue.append((neighbor, next_hop))
    return hops


@dataclass
class NETopologyIndex:
    directed_hops: dict
    undirected_hops: dict
    direct_edges: set
    max_hops: int

    @classmethod
    def from_file(cls, path, *, max_hops=2):
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_graph(json.load(stream), max_hops=max_hops)

    @classmethod
    def from_graph(cls, ne_graph_data, *, max_hops=2):
        directed = defaultdict(set)
        undirected = defaultdict(set)
        direct_edges = set()
        nodes = {_normalize_ne_id(ne_id) for ne_id in (ne_graph_data or {})}
        nodes.discard("")
        for node in nodes:
            directed.setdefault(node, set())
            undirected.setdefault(node, set())

        for source_ne, source_info in (ne_graph_data or {}).items():
            source_ne = _normalize_ne_id(source_ne)
            if not source_ne or not isinstance(source_info, dict):
                continue
            raw_links = source_info.get("link", {})
            if not isinstance(raw_links, dict):
                continue
            for target_ne, link_meta in raw_links.items():
                target_ne = _normalize_ne_id(target_ne)
                if not target_ne or target_ne == source_ne:
                    continue
                direction_values = extract_link_direction_values(link_meta)
                if not direction_values:
                    continue
                undirected[source_ne].add(target_ne)
                undirected[target_ne].add(source_ne)
                if any("<-" in direction for direction in direction_values):
                    directed[source_ne].add(target_ne)
                    direct_edges.add((source_ne, target_ne))
                if any("->" in direction for direction in direction_values):
                    directed[target_ne].add(source_ne)
                    direct_edges.add((target_ne, source_ne))

        max_hops = max(1, int(max_hops or 1))
        return cls(
            directed_hops={node: _walk_hops(directed, node, max_hops) for node in undirected},
            undirected_hops={node: _walk_hops(undirected, node, max_hops) for node in undirected},
            direct_edges=direct_edges,
            max_hops=max_hops,
        )

    @property
    def feature_dim(self):
        return len(PAIR_FEATURE_NAMES)

    def _hop(self, hop_index, source, target):
        if not source or not target or source == target:
            return 0
        return int(hop_index.get(source, {}).get(target, 0) or 0)

    def pair_features(self, source_ne, target_ne):
        source_ne = _normalize_ne_id(source_ne)
        target_ne = _normalize_ne_id(target_ne)
        same_source = bool(source_ne and source_ne == target_ne)
        forward_hop = self._hop(self.directed_hops, source_ne, target_ne)
        reverse_hop = self._hop(self.directed_hops, target_ne, source_ne)
        undirected_hop = self._hop(self.undirected_hops, source_ne, target_ne)
        direct_forward = (source_ne, target_ne) in self.direct_edges
        direct_reverse = (target_ne, source_ne) in self.direct_edges
        return [
            float(same_source),
            float(direct_forward),
            float(direct_reverse),
            float(direct_forward and direct_reverse),
            float(forward_hop > 0),
            float(reverse_hop > 0),
            float(undirected_hop > 0),
            1.0 / forward_hop if forward_hop else 0.0,
            1.0 / reverse_hop if reverse_hop else 0.0,
            1.0 / undirected_hop if undirected_hop else 0.0,
        ]
