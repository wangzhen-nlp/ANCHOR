from fault_grouping.temporal_engine.utils import add_merge_stats, merge_match_batch


class TemporalGraphEngineRuntimeMixin:
    def _prepare_mature_pending_batch(self, force=False):
        with self._lock:
            mature_items = self._collect_mature_pending_locked(force=force)
            if not mature_items:
                return [], None
            seed_nodes = {trigger_key[0] for trigger_key, _ in mature_items}
            event_cache_snapshot = self._snapshot_event_cache_subset_locked(seed_nodes)
        return mature_items, self._build_snapshot_helper(event_cache_snapshot)

    def _evaluate_mature_pending_items(self, mature_items, helper, batch_eval_caches):
        raw_matches = []
        pending_eval_profiles = []
        for trigger_key, trigger_anchor in mature_items:
            trig_node, trig_rule_name = trigger_key
            rule = self.rules[trig_rule_name]
            trigger_ts, _trigger_seq = trigger_anchor
            debug_trace = None
            if self.debug_observer:
                results, debug_trace = self._evaluate_rule(
                    trig_rule_name,
                    rule,
                    trig_node,
                    trigger_ts,
                    node_rule_helper=helper,
                    eval_caches=batch_eval_caches,
                    return_debug_trace=True,
                )
            else:
                results = self._evaluate_rule(
                    trig_rule_name,
                    rule,
                    trig_node,
                    trigger_ts,
                    node_rule_helper=helper,
                    eval_caches=batch_eval_caches,
                )
            pending_eval_profiles.append({
                "node": trig_node,
                "rule": trig_rule_name,
                "trigger_ts": trigger_ts,
                "trigger_seq": trigger_anchor[1],
                "raw_match_count": len(results),
                "raw_matches": results,
                "debug_trace": debug_trace,
            })
            if results:
                raw_matches.extend(results)
        return raw_matches, pending_eval_profiles

    def _merge_and_expand_raw_matches(self, raw_matches, helper, batch_eval_caches):
        merged_matches, batch_merge_stats = merge_match_batch(
            raw_matches,
            site_merge_helper=self.site_merge_helper,
            return_stats=True,
            use_alarm_period_cache=self.use_alarm_period_cache,
        )
        expanded_matches, expanded_merge_stats = self._expand_matches_with_pending_context(
            merged_matches,
            helper,
            eval_caches=batch_eval_caches,
        )
        return (
            merged_matches,
            expanded_matches,
            batch_merge_stats,
            expanded_merge_stats,
            add_merge_stats(batch_merge_stats, expanded_merge_stats),
        )

    def _finalize_expanded_matches_for_output(self, expanded_matches, collection_merge_stats):
        with self._lock:
            self._record_batch_merge_stats_locked(collection_merge_stats)
            self._prune_expired_state_locked(self.latest_arrived_event_ts)
            current_watermark = self.current_watermark
            effective_harvest_ts = (
                self.latest_arrived_event_ts
                if self.latest_arrived_event_ts > 0
                else self.current_watermark
            )
            if self.debug_observer:
                finalized_matches, finalize_profiles = self._finalize_matches_with_history(
                    expanded_matches,
                    return_debug_trace=True,
                )
            else:
                finalized_matches = self._finalize_matches_with_history(expanded_matches)
                finalize_profiles = []

        owned_matches = self._apply_default_output_site_role_ownership_to_matches(finalized_matches)
        output_matches = self._apply_output_visibility_filters_to_matches(owned_matches)
        return output_matches, finalize_profiles, current_watermark, effective_harvest_ts

    def _emit_pending_collection_debug(
        self,
        force,
        mature_items,
        pending_eval_profiles,
        raw_matches,
        collection_merge_stats,
        batch_merge_stats,
        expanded_merge_stats,
        merged_matches,
        expanded_matches,
        output_matches,
        finalize_profiles,
        current_watermark,
        effective_harvest_ts,
    ):
        if not self.debug_observer:
            return

        self.debug_observer({
            "use_alarm_period_cache": self.use_alarm_period_cache,
            "force": force,
            "watermark": current_watermark,
            "effective_harvest_ts": effective_harvest_ts,
            "mature_items": [
                {
                    "node": trigger_key[0],
                    "rule": trigger_key[1],
                    "trigger_ts": trigger_anchor[0],
                    "trigger_seq": trigger_anchor[1],
                }
                for trigger_key, trigger_anchor in mature_items
            ],
            "pending_eval_profiles": pending_eval_profiles,
            "raw_matches": raw_matches,
            "merge_stats": collection_merge_stats,
            "batch_merge_stats": batch_merge_stats,
            "expanded_merge_stats": expanded_merge_stats,
            "batch_merged_matches": merged_matches,
            "expanded_matches": expanded_matches,
            "finalized_matches": output_matches,
            "finalize_profiles": finalize_profiles,
        })

    def _collect_pending_matches(self, force=False):
        """收割已成熟的 pending trigger，并执行对应规则评估。"""
        mature_items, helper = self._prepare_mature_pending_batch(force=force)
        if not mature_items:
            return []

        batch_eval_caches = self._create_eval_caches()
        raw_matches, pending_eval_profiles = self._evaluate_mature_pending_items(
            mature_items,
            helper,
            batch_eval_caches,
        )
        (
            merged_matches,
            expanded_matches,
            batch_merge_stats,
            expanded_merge_stats,
            collection_merge_stats,
        ) = self._merge_and_expand_raw_matches(raw_matches, helper, batch_eval_caches)
        output_matches, finalize_profiles, current_watermark, effective_harvest_ts = (
            self._finalize_expanded_matches_for_output(expanded_matches, collection_merge_stats)
        )
        self._emit_pending_collection_debug(
            force,
            mature_items,
            pending_eval_profiles,
            raw_matches,
            collection_merge_stats,
            batch_merge_stats,
            expanded_merge_stats,
            merged_matches,
            expanded_matches,
            output_matches,
            finalize_profiles,
            current_watermark,
            effective_harvest_ts,
        )
        return output_matches
