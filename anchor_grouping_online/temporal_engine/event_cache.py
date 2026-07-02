import collections


class TemporalGraphEngineEventCacheMixin:
    """活跃告警缓存（event_cache）的维护：按 TTL 过期清理与清除事件删除。

    event_cache 结构为 站点 -> deque[事件 dict]，保留原始告警 payload 供后续
    端口/对端解析。
    """

    def _prune_expired_raw_events_in_place(self, node, current_ts):
        q = self.event_cache.get(node)
        if not q:
            return

        while q and (current_ts - q[0]["ts"]) > self._get_event_ttl(q[0]["alarm"]):
            q.popleft()

        if not q:
            self.event_cache.pop(node, None)

    def _remove_cleared_raw_event(
        self,
        node,
        alarm_id,
        alarm_type,
        alarm_source,
    ):
        q = self.event_cache[node]
        kept = collections.deque()
        target_alarm_source = str(alarm_source or "")

        for cached_event in q:
            matches_clear = (
                cached_event["eid"] == alarm_id
                and cached_event["alarm"] == alarm_type
                and str(cached_event.get("alarm_source", "") or "") == target_alarm_source
            )
            if matches_clear:
                continue
            kept.append(cached_event)

        self.event_cache[node] = kept

    def _remove_trigger_events_by_rule_event_keys(self, node, removed_event_keys_by_rule):
        if not removed_event_keys_by_rule:
            return

        affected_rule_names = set()
        for rule_name, removed_event_keys in removed_event_keys_by_rule.items():
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                if trigger_key in self.pending_triggers:
                    affected_rule_names.add(rule_name)
                continue

            pending_anchor = self.pending_triggers.get(trigger_key)
            pending_anchor_kept = pending_anchor is None
            kept = collections.deque()
            for trigger_event in trigger_events:
                event_ts, alarm_id, event_seq, alarm_type, alarm_source = (
                    self._unpack_trigger_event(trigger_event)
                )
                event_key = (
                    event_ts,
                    alarm_id,
                    alarm_type,
                    alarm_source,
                )
                if event_key not in removed_event_keys:
                    kept.append(trigger_event)
                    if pending_anchor == (event_ts, event_seq):
                        pending_anchor_kept = True
            if kept:
                self.trigger_event_index[trigger_key] = kept
            else:
                self.trigger_event_index.pop(trigger_key, None)
            if not pending_anchor_kept:
                affected_rule_names.add(rule_name)

        # 已消费的 trigger event 可能正是某个尚未成熟 pending 的锚点。
        # trigger index 删除后必须同步推进或移除 pending，否则旧锚点成熟时还会
        # 触发一次无效规则评估，并在此期间阻止后续 trigger 建立新 pending。
        if affected_rule_names:
            self._refresh_pending_triggers_for_node(
                node,
                affected_rule_names=affected_rule_names,
            )

    def _prune_node_alarm_history_before(
        self,
        node,
        alarm_type,
        alarm_source,
        cutoff_by_rule,
    ):
        events = self.event_cache.get(node)
        if not events:
            return

        target_alarm_source = str(alarm_source or "")
        removed_event_keys_by_rule = collections.defaultdict(set)
        updated_events = collections.deque()
        for event in events:
            matched_rules = set()
            if (
                event.get("alarm") == alarm_type
                and str(event.get("alarm_source", "") or "") == target_alarm_source
            ):
                matched_rules = {
                    rule_name
                    for rule_name, cutoff_ts in cutoff_by_rule.items()
                    if event.get("ts") <= cutoff_ts
                }
            if not matched_rules:
                updated_events.append(event)
                continue

            updated_event = dict(event)
            updated_event["consumed_trigger_rules"] = frozenset(
                set(event.get("consumed_trigger_rules", ())) | matched_rules
            )
            updated_events.append(updated_event)
            event_key = (
                event.get("ts"),
                event.get("eid"),
                event.get("alarm"),
                str(event.get("alarm_source", "") or ""),
            )
            for rule_name in matched_rules:
                removed_event_keys_by_rule[rule_name].add(event_key)

        self.event_cache[node] = updated_events
        self._remove_trigger_events_by_rule_event_keys(node, removed_event_keys_by_rule)
