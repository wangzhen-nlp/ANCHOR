import collections


class TemporalGraphEngineTraversalMixin:
    def _matches_node_structure_cached(self, node, node_config, helper, structure_match_cache=None):
        role_site_index = getattr(self, "role_site_index", None)
        if role_site_index is not None:
            indexed_result = role_site_index.matches_config(node, node_config)
            if indexed_result is not None:
                return indexed_result

        if structure_match_cache is None:
            return helper.matches_node_structure(self.sites_domain_map.get(node, {}), node_config)

        cache_key = (node, id(node_config))
        if cache_key not in structure_match_cache:
            structure_match_cache[cache_key] = helper.matches_node_structure(
                self.sites_domain_map.get(node, {}),
                node_config
            )
        return structure_match_cache[cache_key]

    def _role_filtered_candidate_hops(
        self,
        rule_name,
        target_role,
        candidate_hops,
        target_node_config=None,
    ):
        role_site_index = getattr(self, "role_site_index", None)
        if role_site_index is None or not candidate_hops:
            return candidate_hops

        if target_node_config is not None:
            role_candidates = role_site_index.config_candidates(target_node_config)
        else:
            role_candidates = role_site_index.role_candidates(rule_name, target_role)
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
        reference_ts=None,
        edge_window=0,
        path_requirements=None,
        node_rule_helper=None,
        traversal_cache=None,
        path_validation_cache=None,
        filtered_neighbor_cache=None,
        target_node_config=None,
    ):
        """Traverse topology and filter candidates by static target role structure.

        The cross-batch cache is used only when path requirements are absent,
        because then the result depends solely on static topology and static
        role/site compatibility.
        """
        directions = self._normalize_traverse_directions(direction)
        static_cache_key = None
        if path_requirements is None:
            structure_key = ("config", id(target_node_config)) if target_node_config is not None else ("role", rule_name, target_role)
            static_cache_key = (
                structure_key,
                start_node,
                directions,
                max_hops,
            )
            with self._topo_cache_lock:
                cached = self.global_role_filtered_neighbor_cache.get(static_cache_key)
                if cached is not None:
                    self.global_role_filtered_neighbor_cache.move_to_end(static_cache_key)
                    return cached

        candidate_hops = self._traverse_graph(
            start_node=start_node,
            direction=direction,
            max_hops=max_hops,
            reference_ts=reference_ts,
            edge_window=edge_window,
            path_requirements=path_requirements,
            node_rule_helper=node_rule_helper,
            traversal_cache=traversal_cache,
            path_validation_cache=path_validation_cache,
            filtered_neighbor_cache=filtered_neighbor_cache,
        )
        filtered = self._role_filtered_candidate_hops(
            rule_name,
            target_role,
            candidate_hops,
            target_node_config=target_node_config,
        )
        if static_cache_key is not None:
            with self._topo_cache_lock:
                self.global_role_filtered_neighbor_cache[static_cache_key] = filtered
                self.global_role_filtered_neighbor_cache.move_to_end(static_cache_key)
                if len(self.global_role_filtered_neighbor_cache) > self.max_role_filtered_neighbor_cache_size:
                    self.global_role_filtered_neighbor_cache.popitem(last=False)
        return filtered

    def _get_precomputed_site_chain_candidates(self, start_node, direction, max_hops=None):
        """从 site_chains.json 预计算结果中取候选 hop；不支持混合多跳 either。"""
        if not self.site_chain_index:
            return None

        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            candidate_maps = []
            for single_direction in directions:
                single_candidates = self._get_precomputed_site_chain_candidates(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                )
                if single_candidates is None:
                    return None
                candidate_maps.append(single_candidates)
            return self._merge_candidate_hops(*candidate_maps)

        start_node = str(start_node or "").strip()
        chain_info = self.site_chain_index.get(start_node)
        if chain_info is None:
            return None

        direction = directions[0]

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

        if direction == "downstream":
            for site_id, hop in chain_info.get("downstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            return candidates

        if direction == "upstream":
            for site_id, hop in chain_info.get("upstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            return candidates

        if direction == "either":
            # site_chains 只保存纯上游/纯下游的可达关系；混合方向多跳仍回退到原 BFS。
            if max_hops != 1:
                return None
            for site_id, hop in chain_info.get("downstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            for site_id, hop in chain_info.get("upstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            for site_id in chain_info.get("bidirectional_sites", set()):
                add_candidate(site_id, 1)
            return candidates

        if direction in {"bidirection", "bidirectional"}:
            for site_id in chain_info.get("bidirectional_sites", set()):
                add_candidate(site_id, 1)
            return candidates

        return None

    def _validate_path_node_for_traversal(
        self,
        node,
        path_requirements,
        reference_ts,
        edge_window,
        helper,
        path_validation_cache=None,
    ):
        if path_validation_cache is None:
            node_domain = self.sites_domain_map.get(node, {})
            is_valid_path_node, _ = helper.validate_node(
                node, node_domain, path_requirements, reference_ts, edge_window
            )
            return is_valid_path_node

        cache_key = (
            node,
            id(path_requirements),
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
        )
        if cache_key not in path_validation_cache:
            node_domain = self.sites_domain_map.get(node, {})
            is_valid_path_node, _ = helper.validate_node(
                node, node_domain, path_requirements, reference_ts, edge_window
            )
            path_validation_cache[cache_key] = is_valid_path_node
        return path_validation_cache[cache_key]

    def _traverse_graph_nearest_matching(
        self,
        start_node,
        direction,
        target_node_config,
        max_hops=None,
        reference_ts=None,
        edge_window=0,
        path_requirements=None,
        node_rule_helper=None,
        traversal_cache=None,
        path_validation_cache=None,
        structure_match_cache=None,
        filtered_neighbor_cache=None,
    ):
        """nearest_matching 专用 BFS。

        一旦在某个 hop 首次命中结构匹配节点，就在该 hop 层结束后停止继续向外扩张，
        以避免在稠密图上无意义地遍历整张图。
        """
        helper = node_rule_helper or self.node_rule_helper
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            cache_key = (
                "nearest_matching_multi",
                start_node,
                directions,
                max_hops,
                reference_ts,
                self._make_edge_window_cache_key(edge_window),
                id(path_requirements),
                id(target_node_config),
            )
            if traversal_cache is not None and cache_key in traversal_cache:
                return traversal_cache[cache_key]

            candidate_maps = []
            had_topology_candidate = False
            for single_direction in directions:
                single_candidates, single_had_topology = self._traverse_graph_nearest_matching(
                    start_node=start_node,
                    direction=single_direction,
                    target_node_config=target_node_config,
                    max_hops=max_hops,
                    reference_ts=reference_ts,
                    edge_window=edge_window,
                    path_requirements=path_requirements,
                    node_rule_helper=helper,
                    traversal_cache=traversal_cache,
                    path_validation_cache=path_validation_cache,
                    structure_match_cache=structure_match_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                )
                candidate_maps.append(single_candidates)
                had_topology_candidate = had_topology_candidate or single_had_topology

            result = self._merge_candidate_hops(*candidate_maps)
            if result:
                nearest_hop = min(result.values())
                result = {
                    node: hop
                    for node, hop in result.items()
                    if hop == nearest_hop
                }
            cached_result = (result, had_topology_candidate)
            if traversal_cache is not None:
                traversal_cache[cache_key] = cached_result
            return cached_result

        direction = directions[0]

        static_cache_key = None
        if path_requirements is None:
            static_cache_key = (
                start_node,
                direction,
                max_hops,
                id(target_node_config),
            )
            with self._topo_cache_lock:
                if static_cache_key in self.global_nearest_match_cache:
                    self.global_nearest_match_cache.move_to_end(static_cache_key)
                    result = self.global_nearest_match_cache[static_cache_key]
                    if traversal_cache is not None:
                        traversal_cache[(
                            "nearest_matching",
                            start_node,
                            direction,
                            max_hops,
                            reference_ts,
                            self._make_edge_window_cache_key(edge_window),
                            id(path_requirements),
                            id(target_node_config),
                        )] = result
                    return result

        cache_key = (
            "nearest_matching",
            start_node,
            direction,
            max_hops,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
            id(target_node_config),
        )
        if traversal_cache is not None and cache_key in traversal_cache:
            return traversal_cache[cache_key]

        if path_requirements is None:
            precomputed_candidates = self._get_precomputed_site_chain_candidates(
                start_node,
                direction,
                max_hops=max_hops,
            )
            if precomputed_candidates is not None:
                had_topology_candidate = bool(precomputed_candidates)
                result = {}
                nearest_hop = None
                for curr, hops in sorted(precomputed_candidates.items(), key=lambda item: (item[1], str(item[0]))):
                    if nearest_hop is not None and hops > nearest_hop:
                        break
                    if self._matches_node_structure_cached(
                        curr,
                        target_node_config,
                        helper,
                        structure_match_cache=structure_match_cache,
                    ):
                        if nearest_hop is None:
                            nearest_hop = hops
                        result[curr] = hops

                cached_result = (result, had_topology_candidate)
                if traversal_cache is not None:
                    traversal_cache[cache_key] = cached_result
                if static_cache_key is not None:
                    with self._topo_cache_lock:
                        self.global_nearest_match_cache[static_cache_key] = cached_result
                        self.global_nearest_match_cache.move_to_end(static_cache_key)
                        if len(self.global_nearest_match_cache) > self.max_nearest_match_cache_size:
                            self.global_nearest_match_cache.popitem(last=False)
                return cached_result

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])

        result = {}
        nearest_hop = None
        had_topology_candidate = False

        while queue:
            curr, hops = queue.popleft()
            if nearest_hop is not None and hops > nearest_hop:
                break

            if hops > 0:
                had_topology_candidate = True
                if self._matches_node_structure_cached(
                    curr,
                    target_node_config,
                    helper,
                    structure_match_cache=structure_match_cache,
                ):
                    if nearest_hop is None:
                        nearest_hop = hops
                    if hops == nearest_hop:
                        result[curr] = hops

            if nearest_hop is not None and hops >= nearest_hop:
                continue

            if max_hops is None or hops < max_hops:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    curr,
                    direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    queue.append((nxt, hops + 1))

        if traversal_cache is not None:
            traversal_cache[cache_key] = (result, had_topology_candidate)
        if static_cache_key is not None:
            with self._topo_cache_lock:
                self.global_nearest_match_cache[static_cache_key] = (result, had_topology_candidate)
                self.global_nearest_match_cache.move_to_end(static_cache_key)
                if len(self.global_nearest_match_cache) > self.max_nearest_match_cache_size:
                    self.global_nearest_match_cache.popitem(last=False)

        return result, had_topology_candidate

    def _traverse_graph(self, start_node, direction, max_hops=None,
                        reference_ts=None, edge_window=0,
                        path_requirements=None, node_rule_helper=None,
                        traversal_cache=None, path_validation_cache=None,
                        filtered_neighbor_cache=None):
        """通用的广度优先搜索，支持路径节点约束"""
        helper = node_rule_helper or self.node_rule_helper
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            local_cache_key = (
                "full_multi",
                start_node,
                directions,
                max_hops,
                reference_ts,
                self._make_edge_window_cache_key(edge_window),
                id(path_requirements),
            )
            if traversal_cache is not None and local_cache_key in traversal_cache:
                return traversal_cache[local_cache_key]

            result = self._merge_candidate_hops(*[
                self._traverse_graph(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                    reference_ts=reference_ts,
                    edge_window=edge_window,
                    path_requirements=path_requirements,
                    node_rule_helper=helper,
                    traversal_cache=traversal_cache,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                )
                for single_direction in directions
            ])
            if traversal_cache is not None:
                traversal_cache[local_cache_key] = result
            return result

        direction = directions[0]

        if direction == "self":
            return {start_node: 0}

        local_cache_key = (
            "full",
            start_node,
            direction,
            max_hops,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
        )
        if traversal_cache is not None and local_cache_key in traversal_cache:
            return traversal_cache[local_cache_key]

        cache_key = (start_node, direction, max_hops)

        if path_requirements is None:
            with self._topo_cache_lock:
                if cache_key in self.global_topo_cache:
                    self.global_topo_cache.move_to_end(cache_key)
                    result = self.global_topo_cache[cache_key]
                    if traversal_cache is not None:
                        traversal_cache[local_cache_key] = result
                    return result

            precomputed_candidates = self._get_precomputed_site_chain_candidates(
                start_node,
                direction,
                max_hops=max_hops,
            )
            if precomputed_candidates is not None:
                with self._topo_cache_lock:
                    self.global_topo_cache[cache_key] = precomputed_candidates
                    self.global_topo_cache.move_to_end(cache_key)
                    if len(self.global_topo_cache) > self.max_topo_cache_size:
                        self.global_topo_cache.popitem(last=False)
                if traversal_cache is not None:
                    traversal_cache[local_cache_key] = precomputed_candidates
                return precomputed_candidates

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        result = {}

        while queue:
            curr, hops = queue.popleft()
            if hops > 0:
                result[curr] = hops
            if max_hops is None or hops < max_hops:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    curr,
                    direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, hops + 1))

        # 写缓存：仅缓存不带路径约束的纯拓扑结果
        if path_requirements is None:
            with self._topo_cache_lock:
                self.global_topo_cache[cache_key] = result
                self.global_topo_cache.move_to_end(cache_key)
                if len(self.global_topo_cache) > self.max_topo_cache_size:
                    self.global_topo_cache.popitem(last=False)
        if traversal_cache is not None:
            traversal_cache[local_cache_key] = result
        return result

    def _get_filtered_neighbors_for_traversal(
        self,
        node,
        direction,
        reference_ts,
        edge_window,
        path_requirements,
        helper,
        path_validation_cache=None,
        filtered_neighbor_cache=None,
    ):
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            neighbors = []
            seen = set()
            for single_direction in directions:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    node,
                    single_direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    neighbors.append(nxt)
            return tuple(neighbors)

        direction = directions[0]
        if direction == "upstream":
            topo_neighbors = tuple(self.topo_up.get(node, []))
        elif direction == "either":
            seen = set()
            topo_neighbors = []
            for nxt in self.topo_up.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    topo_neighbors.append(nxt)
            for nxt in self.topo_down.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    topo_neighbors.append(nxt)
            topo_neighbors = tuple(topo_neighbors)
        elif direction in {"bidirection", "bidirectional"}:
            topo_neighbors = tuple(
                sorted(set(self.topo_down.get(node, ())) & set(self.topo_up.get(node, ())))
            )
        else:
            topo_neighbors = tuple(self.topo_down.get(node, []))

        if path_requirements is None:
            return topo_neighbors

        cache_key = (
            node,
            direction,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
        )
        if filtered_neighbor_cache is not None and cache_key in filtered_neighbor_cache:
            return filtered_neighbor_cache[cache_key]

        valid_neighbors = tuple(
            nxt
            for nxt in topo_neighbors
            if self._validate_path_node_for_traversal(
                nxt,
                path_requirements,
                reference_ts,
                edge_window,
                helper,
                path_validation_cache=path_validation_cache,
            )
        )
        if filtered_neighbor_cache is not None:
            filtered_neighbor_cache[cache_key] = valid_neighbors
        return valid_neighbors
