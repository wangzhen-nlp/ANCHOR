class TemporalGraphEngineCommonMixin:
    @staticmethod
    def _normalize_traverse_directions(direction):
        if isinstance(direction, str):
            text = direction.strip()
            return (text,) if text else ("downstream",)
        if isinstance(direction, (list, tuple, set)):
            directions = []
            seen = set()
            for item in direction:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                directions.append(text)
            return tuple(directions) if directions else ("downstream",)
        return (str(direction).strip() or "downstream",)

    @staticmethod
    def _merge_candidate_hops(*candidate_maps):
        merged = {}
        for candidate_map in candidate_maps:
            for site_id, hop in candidate_map.items():
                previous_hop = merged.get(site_id)
                if previous_hop is None or hop < previous_hop:
                    merged[site_id] = hop
        return merged

    @staticmethod
    def _make_edge_window_cache_key(edge_window):
        if isinstance(edge_window, dict):
            return tuple(sorted(edge_window.items()))
        return edge_window

    @staticmethod
    def _create_eval_caches():
        return {
            "validation_cache": {},
            "traversal_cache": {},
            "path_validation_cache": {},
            "structure_match_cache": {},
            "filtered_neighbor_cache": {},
            "support_cache": {},
            "support_count_cache": {},
        }

    @staticmethod
    def _normalize_site_chain_hops(raw_hops):
        if not isinstance(raw_hops, dict):
            return {}

        normalized = {}
        for raw_site_id, raw_hop in raw_hops.items():
            site_id = str(raw_site_id or "").strip()
            if not site_id:
                continue
            try:
                hop = int(raw_hop)
            except (TypeError, ValueError):
                continue
            if hop <= 0:
                continue
            normalized[site_id] = hop
        return normalized

    @classmethod
    def _normalize_site_chain_index(cls, site_chain_index):
        if not isinstance(site_chain_index, dict):
            return {}

        normalized = {}
        for raw_site_id, raw_info in site_chain_index.items():
            site_id = str(raw_site_id or "").strip()
            if not site_id or not isinstance(raw_info, dict):
                continue

            bidirectional_sites = {
                str(site_id).strip()
                for site_id in raw_info.get("bidirectional_sites", [])
                if str(site_id).strip()
            }
            normalized[site_id] = {
                "downstream_site_hops": cls._normalize_site_chain_hops(
                    raw_info.get("downstream_site_hops")
                ),
                "upstream_site_hops": cls._normalize_site_chain_hops(
                    raw_info.get("upstream_site_hops")
                ),
                "bidirectional_sites": bidirectional_sites,
            }

        return normalized
