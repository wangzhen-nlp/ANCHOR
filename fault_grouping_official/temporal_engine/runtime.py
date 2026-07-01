from fault_grouping_official.temporal_engine.utils import add_merge_stats, merge_match_batch


class TemporalGraphEngineRuntimeMixin:
    def _prepare_mature_pending_batch(self, force=False):
        mature_items = self._collect_mature_pending(force=force)
        if not mature_items:
            return [], None
        seed_nodes = {trigger_key[0] for trigger_key, _ in mature_items}
        event_cache_snapshot = self._snapshot_event_cache_subset(seed_nodes)
        return mature_items, self._build_snapshot_helper(event_cache_snapshot)

    def _evaluate_mature_pending_items(self, mature_items, helper, batch_eval_caches):
        raw_matches = []
        for trigger_key, trigger_anchor in mature_items:
            trig_node, trig_rule_name = trigger_key
            rule = self.rules[trig_rule_name]
            trigger_ts, _trigger_seq = trigger_anchor
            results = self._evaluate_rule(
                trig_rule_name,
                rule,
                trig_node,
                trigger_ts,
                node_rule_helper=helper,
                eval_caches=batch_eval_caches,
            )
            if results:
                raw_matches.extend(results)
        return raw_matches

    def _merge_and_expand_raw_matches(self, raw_matches, helper, batch_eval_caches):
        merged_matches, batch_merge_stats = merge_match_batch(
            raw_matches,
            return_stats=True,
        )
        expanded_matches, expanded_merge_stats = self._expand_matches_with_pending_context(
            merged_matches,
            helper,
            eval_caches=batch_eval_caches,
        )
        return expanded_matches, add_merge_stats(batch_merge_stats, expanded_merge_stats)

    def _finalize_expanded_matches_for_output(self, expanded_matches, collection_merge_stats):
        self._record_batch_merge_stats(collection_merge_stats)
        self._prune_expired_state(self.latest_arrived_event_ts)
        finalized_matches = self._finalize_matches_with_history(expanded_matches)

        return self._apply_output_visibility_filters_to_matches(finalized_matches)

    def _collect_pending_matches(self, force=False):
        """收割已成熟的 pending trigger，并执行对应规则评估。"""
        mature_items, helper = self._prepare_mature_pending_batch(force=force)
        if not mature_items:
            return []

        batch_eval_caches = self._create_eval_caches()
        raw_matches = self._evaluate_mature_pending_items(
            mature_items,
            helper,
            batch_eval_caches,
        )
        expanded_matches, collection_merge_stats = self._merge_and_expand_raw_matches(
            raw_matches,
            helper,
            batch_eval_caches,
        )
        return self._finalize_expanded_matches_for_output(
            expanded_matches,
            collection_merge_stats,
        )
