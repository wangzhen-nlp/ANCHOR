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

        while q and (current_ts - q[0]["ts"]) > self.global_ttl:
            expired_event = q.popleft()
            self._forget_batch_cached_event(node, expired_event)

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
                self._forget_batch_cached_event(node, cached_event)
                continue
            kept.append(cached_event)

        self.event_cache[node] = kept

    def _remove_trigger_events_by_rule_event_keys(self, node, removed_event_keys_by_rule):
        if not removed_event_keys_by_rule:
            return set()

        removed_trigger_seqs = set()
        for rule_name, removed_event_keys in removed_event_keys_by_rule.items():
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

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
                if event_key in removed_event_keys:
                    removed_trigger_seqs.add(event_seq)
                    continue
                kept.append(trigger_event)
            if kept:
                self.trigger_event_index[trigger_key] = kept
            else:
                self.trigger_event_index.pop(trigger_key, None)

        return removed_trigger_seqs

    def _prune_node_alarm_history_before(
        self,
        node,
        alarm_type,
        alarm_source,
        cutoff_by_rule,
    ):
        events = self.event_cache.get(node)
        if not events:
            return set()

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

            updated_event = event
            if self._batch_event_by_alarm_id is None:
                # 隔离模式使用复制替换语义；持久批处理必须原地
                # 更新，确保 eid 索引继续指向缓存中的同一个事件对象。
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
        return self._remove_trigger_events_by_rule_event_keys(
            node,
            removed_event_keys_by_rule,
        )
