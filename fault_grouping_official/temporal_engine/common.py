class TemporalGraphEngineCommonMixin:
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
    def _create_eval_caches():
        return {
            "validation_cache": {},
            "traversal_cache": {},
        }
