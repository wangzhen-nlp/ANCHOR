import collections

from fault_grouping_official.temporal_engine.utils import _normalize_edge_directions


class TemporalGraphEngineTraversalMixin:
    def _role_filtered_candidate_hops(
        self,
        rule_name,
        target_role,
        candidate_hops,
        target_node_config=None,
    ):
        if not candidate_hops:
            return candidate_hops

        if target_node_config is not None:
            role_candidates = self.role_site_index.config_candidates(target_node_config)
        else:
            role_candidates = self.role_site_index.role_candidates(rule_name, target_role)
        if not role_candidates:
            return {}
        return {
            site_id: hop
            for site_id, hop in candidate_hops.items()
            if site_id in role_candidates
        }

    def _traverse_graph_role_filtered(
        self,
        rule_name,
        start_node,
        target_role,
        direction,
        max_hops=None,
        traversal_cache=None,
        target_node_config=None,
    ):
        directions = _normalize_edge_directions(direction)
        structure_key = (
            ("config", id(target_node_config))
            if target_node_config is not None
            else ("role", rule_name, target_role)
        )
        cache_key = (structure_key, start_node, directions, max_hops)
        cached = self.global_role_filtered_neighbor_cache.get(cache_key)
        if cached is not None:
            self.global_role_filtered_neighbor_cache.move_to_end(cache_key)
            return cached

        candidate_hops = self._traverse_graph(
            start_node=start_node,
            direction=direction,
            max_hops=max_hops,
            traversal_cache=traversal_cache,
        )
        filtered = self._role_filtered_candidate_hops(
            rule_name,
            target_role,
            candidate_hops,
            target_node_config=target_node_config,
        )
        self.global_role_filtered_neighbor_cache[cache_key] = filtered
        self.global_role_filtered_neighbor_cache.move_to_end(cache_key)
        if len(self.global_role_filtered_neighbor_cache) > self.max_role_filtered_neighbor_cache_size:
            self.global_role_filtered_neighbor_cache.popitem(last=False)
        return filtered

    def _get_precomputed_site_chain_candidates(self, start_node, direction, max_hops=None):
        directions = _normalize_edge_directions(direction)
        if len(directions) > 1:
            candidate_maps = []
            for single_direction in directions:
                candidates = self._get_precomputed_site_chain_candidates(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                )
                if candidates is None:
                    return None
                candidate_maps.append(candidates)
            return self._merge_candidate_hops(*candidate_maps)

        start_node = str(start_node or "").strip()
        chain_info = self.site_chain_index.get(start_node)
        if chain_info is None:
            return None

        candidates = {}

        def add_candidate(site_id, hop):
            site_id = str(site_id or "").strip()
            if not site_id or site_id == start_node:
                return
            if max_hops is not None and hop > max_hops:
                return
            previous_hop = candidates.get(site_id)
            if previous_hop is None or hop < previous_hop:
                candidates[site_id] = hop

        direction = directions[0]
        if direction == "downstream":
            for site_id, hop in chain_info["downstream_site_hops"].items():
                add_candidate(site_id, hop)
            return candidates
        if direction == "upstream":
            for site_id, hop in chain_info["upstream_site_hops"].items():
                add_candidate(site_id, hop)
            return candidates
        if direction == "bidirectional":
            for site_id in chain_info["bidirectional_sites"]:
                add_candidate(site_id, 1)
            return candidates
        return None

    def _traverse_graph(self, start_node, direction, max_hops=None, traversal_cache=None):
        directions = _normalize_edge_directions(direction)
        local_cache_key = (start_node, directions, max_hops)
        if traversal_cache is not None and local_cache_key in traversal_cache:
            return traversal_cache[local_cache_key]

        if len(directions) > 1:
            result = self._merge_candidate_hops(*[
                self._traverse_graph(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                    traversal_cache=traversal_cache,
                )
                for single_direction in directions
            ])
        else:
            result = self._traverse_graph_single_direction(
                start_node,
                directions[0],
                max_hops,
            )

        if traversal_cache is not None:
            traversal_cache[local_cache_key] = result
        return result

    def _traverse_graph_single_direction(self, start_node, direction, max_hops):
        cache_key = (start_node, direction, max_hops)
        cached = self.global_topo_cache.get(cache_key)
        if cached is not None:
            self.global_topo_cache.move_to_end(cache_key)
            return cached

        result = self._get_precomputed_site_chain_candidates(
            start_node,
            direction,
            max_hops=max_hops,
        )
        if result is None:
            result = self._traverse_site_topology_bfs(
                start_node,
                direction,
                max_hops,
            )

        self.global_topo_cache[cache_key] = result
        self.global_topo_cache.move_to_end(cache_key)
        if len(self.global_topo_cache) > self.max_topo_cache_size:
            self.global_topo_cache.popitem(last=False)
        return result

    def _get_site_topology_neighbors(self, node, direction):
        if direction == "downstream":
            return self.topo_down.get(node, ())
        if direction == "upstream":
            return self.topo_up.get(node, ())
        if direction == "bidirectional":
            return tuple(
                set(self.topo_down.get(node, ()))
                & set(self.topo_up.get(node, ()))
            )
        raise ValueError(f"不支持遍历方向 {direction!r}")

    def _traverse_site_topology_bfs(self, start_node, direction, max_hops):
        start_node = str(start_node or "").strip()
        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        result = {}

        while queue:
            current_node, hops = queue.popleft()
            if hops > 0:
                result[current_node] = hops
            if max_hops is not None and hops >= max_hops:
                continue
            for neighbor in self._get_site_topology_neighbors(current_node, direction):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, hops + 1))

        return result
