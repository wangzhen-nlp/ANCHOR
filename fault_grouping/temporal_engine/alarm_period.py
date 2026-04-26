import collections


class TemporalGraphEngineAlarmPeriodMixin:
    @staticmethod
    def _make_alarm_period_key(alarm_type, alarm_source):
        return str(alarm_type or ""), str(alarm_source or "")

    @staticmethod
    def _build_output_symptom_interval_key(symptom):
        node = symptom.get("node")
        alarm = symptom.get("alarm")
        alarm_source = symptom.get("alarm_source", "")
        start_ts = symptom.get("_segment_start_ts")
        end_ts = symptom.get("_segment_end_ts")
        if start_ts is None:
            start_ts = symptom.get("ts")
        if end_ts is None:
            end_ts = start_ts

        if (
            node not in (None, "")
            and alarm not in (None, "")
            and start_ts is not None
            and end_ts is not None
        ):
            return f"{node}|{alarm_source}|{alarm}|{start_ts:.6f}|{end_ts:.6f}"

        return (
            symptom.get("_segment_key")
            or symptom.get("eid")
            or (symptom.get("node"), symptom.get("ts"), symptom.get("alarm"), symptom.get("alarm_source"))
        )

    @staticmethod
    def _period_state_to_cached_event(node, period_state):
        raw_event_items = tuple(period_state.get("active_event_ids", {}).items())
        return {
            "node": node,
            "ts": period_state["ts"],
            "end_ts": period_state.get("end_ts"),
            "eid": period_state["eid"],
            "alarm": period_state["alarm_type"],
            "alarm_source": period_state["alarm_source"],
            "consumed_trigger_rules": period_state["consumed_trigger_rules"],
            "_segment_key": period_state["segment_key"],
            "_segment_start_ts": period_state["ts"],
            "_segment_end_ts": period_state.get("end_ts"),
            "_raw_event_items": raw_event_items,
            "_raw_event_ts_list": tuple(raw_ts for _raw_event_id, raw_ts in raw_event_items),
            "_consumed_cutoff_by_rule": dict(period_state.get("consumed_trigger_cutoff_by_rule", {})),
        }

    @staticmethod
    def _ensure_ordered_active_event_ids(period_state):
        active_event_ids = period_state.get("active_event_ids")
        if isinstance(active_event_ids, collections.OrderedDict):
            return active_event_ids
        ordered_active_event_ids = collections.OrderedDict()
        if active_event_ids:
            for raw_event_id, raw_ts in active_event_ids.items():
                ordered_active_event_ids[raw_event_id] = raw_ts
        period_state["active_event_ids"] = ordered_active_event_ids
        return ordered_active_event_ids

    @classmethod
    def _refresh_alarm_period_state(cls, node, period_state):
        active_event_ids = cls._ensure_ordered_active_event_ids(period_state)
        if not active_event_ids:
            return False

        leader_event_id = next(iter(active_event_ids))
        leader_ts = active_event_ids[leader_event_id]
        tail_event_id = next(reversed(active_event_ids))
        tail_ts = active_event_ids[tail_event_id]
        period_state["ts"] = leader_ts
        period_state["eid"] = leader_event_id
        period_state["latest_active_ts"] = tail_ts
        period_state["end_ts"] = tail_ts
        period_state["consumed_trigger_rules"] = frozenset(
            period_state.get("consumed_trigger_cutoff_by_rule", {}).keys()
        )
        period_state["segment_key"] = (
            f"{node}|{period_state['alarm_source']}|{period_state['alarm_type']}|"
            f"{leader_ts:.6f}|{tail_ts:.6f}|{leader_event_id}"
        )
        return True

    def _rebuild_node_event_cache(self, node):
        periods = self.active_alarm_periods.get(node)
        if not periods:
            self.event_cache.pop(node, None)
            self.active_alarm_periods.pop(node, None)
            self.active_event_to_period.pop(node, None)
            return

        ordered_periods = sorted(
            periods.values(),
            key=lambda period: (
                period["ts"],
                str(period["eid"]),
                period["alarm_type"],
                period["alarm_source"],
            )
        )
        self.event_cache[node] = collections.deque(
            self._period_state_to_cached_event(node, period)
            for period in ordered_periods
        )

    def _register_alarm_period_occurrence(self, node, alarm_type, ts, event_id, alarm_source=""):
        period_key = self._make_alarm_period_key(alarm_type, alarm_source)
        periods = self.active_alarm_periods[node]
        period = periods.get(period_key)
        created = False
        if period is None:
            created = True
            period = {
                "ts": ts,
                "end_ts": None,
                "eid": event_id,
                "alarm_type": alarm_type,
                "alarm_source": alarm_source,
                "consumed_trigger_rules": frozenset(),
                "consumed_trigger_cutoff_by_rule": {},
                "active_event_ids": collections.OrderedDict(),
                "latest_active_ts": ts,
                "segment_key": "",
            }
            periods[period_key] = period

        if event_id not in (None, ""):
            active_event_ids = self._ensure_ordered_active_event_ids(period)
            active_event_ids[event_id] = ts
            self.active_event_to_period[node][event_id] = period_key

        if self._refresh_alarm_period_state(node, period):
            self._rebuild_node_event_cache(node)
        return created

    def _prune_expired_raw_events_in_place(self, node, current_ts):
        q = self.event_cache.get(node)
        if not q:
            return

        while q and (current_ts - q[0][0]) > self._get_event_ttl(q[0][2]):
            expired_event = q.popleft()
            self._log_debug_event_removal(node, expired_event, "ttl", current_ts=current_ts)

        if not q:
            self.event_cache.pop(node, None)

    def _prune_expired_alarm_periods(self, node, current_ts):
        periods = self.active_alarm_periods.get(node)
        if not periods:
            self.event_cache.pop(node, None)
            return

        changed = False
        for period_key, period in list(periods.items()):
            ttl = self._get_event_ttl(period["alarm_type"])
            active_event_ids = self._ensure_ordered_active_event_ids(period)
            expired_event_ids = []
            while active_event_ids:
                leader_event_id = next(iter(active_event_ids))
                leader_ts = active_event_ids[leader_event_id]
                if (current_ts - leader_ts) <= ttl:
                    break
                expired_event_ids.append(leader_event_id)
                active_event_ids.popitem(last=False)
            if not expired_event_ids:
                continue

            changed = True
            for raw_event_id in expired_event_ids:
                if raw_event_id not in (None, ""):
                    self.active_event_to_period.get(node, {}).pop(raw_event_id, None)

            if self._refresh_alarm_period_state(node, period):
                continue

            removed_event = self._period_state_to_cached_event(node, period)
            self._log_debug_event_removal(node, removed_event, "ttl", current_ts=current_ts)
            periods.pop(period_key, None)

        if changed:
            if not self.active_event_to_period.get(node):
                self.active_event_to_period.pop(node, None)
            self._rebuild_node_event_cache(node)

    def _remove_cleared_raw_event(self, node, event_id):
        q = self.event_cache[node]
        kept = collections.deque()

        for cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules in q:
            if event_id and cached_eid == event_id:
                self._log_debug_event_removal(
                    node,
                    (cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules),
                    "clear",
                    cleared_event_id=event_id,
                )
                continue
            kept.append((cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules))

        self.event_cache[node] = kept

    def _remove_cleared_alarm_period_event(self, node, event_id):
        if event_id in (None, ""):
            return

        period_key = self.active_event_to_period.get(node, {}).pop(event_id, None)
        if period_key is None:
            return

        periods = self.active_alarm_periods.get(node)
        if not periods:
            self.active_event_to_period.pop(node, None)
            self.event_cache.pop(node, None)
            return

        period = periods.get(period_key)
        if period is None:
            if not self.active_event_to_period.get(node):
                self.active_event_to_period.pop(node, None)
            return

        active_event_ids = self._ensure_ordered_active_event_ids(period)
        active_event_ids.pop(event_id, None)
        if active_event_ids:
            self._refresh_alarm_period_state(node, period)
            self._rebuild_node_event_cache(node)
            if not self.active_event_to_period.get(node):
                self.active_event_to_period.pop(node, None)
            return

        removed_event = self._period_state_to_cached_event(node, period)
        periods.pop(period_key, None)
        self._log_debug_event_removal(
            node,
            removed_event,
            "clear",
            cleared_event_id=event_id,
        )

        if not self.active_event_to_period.get(node):
            self.active_event_to_period.pop(node, None)
        self._rebuild_node_event_cache(node)

    def _remove_cleared_events(self, node, event_id):
        if self.use_alarm_period_cache:
            self._remove_cleared_alarm_period_event(node, event_id)
            return
        self._remove_cleared_raw_event(node, event_id)

    def _remove_trigger_events_by_rule_event_ids(self, node, removed_event_ids_by_rule):
        if not removed_event_ids_by_rule:
            return

        for rule_name, removed_event_ids in removed_event_ids_by_rule.items():
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            kept_trigger_events = collections.deque()
            for event_ts, indexed_event_id, indexed_seq, indexed_alarm_type in trigger_events:
                if indexed_event_id in removed_event_ids:
                    continue
                kept_trigger_events.append((event_ts, indexed_event_id, indexed_seq, indexed_alarm_type))

            if kept_trigger_events:
                self.trigger_event_index[trigger_key] = kept_trigger_events
            else:
                self.trigger_event_index.pop(trigger_key, None)

        self._refresh_pending_triggers_for_node(
            node,
            affected_rule_names=removed_event_ids_by_rule.keys(),
        )

    def _prune_raw_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        q = self.event_cache.get(node)
        if not q:
            return

        removed_event_ids_by_rule = collections.defaultdict(set)
        kept = collections.deque()
        target_alarm_source = str(alarm_source or "")
        for cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules in q:
            if (
                cached_alarm_type == alarm_type
                and str(cached_alarm_source or "") == target_alarm_source
            ):
                matched_rules = {
                    rule_name
                    for rule_name, cutoff_ts in cutoff_by_rule.items()
                    if cached_ts <= cutoff_ts
                }
            else:
                matched_rules = set()

            if matched_rules:
                if cached_eid not in (None, ""):
                    for rule_name in matched_rules:
                        removed_event_ids_by_rule[rule_name].add(cached_eid)
                updated_consumed_rules = frozenset(set(consumed_trigger_rules) | matched_rules)
                kept.append((cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, updated_consumed_rules))
                continue
            kept.append((cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules))

        self.event_cache[node] = kept
        self._remove_trigger_events_by_rule_event_ids(node, removed_event_ids_by_rule)

    def _prune_period_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        periods = self.active_alarm_periods.get(node)
        if not periods:
            return

        removed_event_ids_by_rule = collections.defaultdict(set)
        changed = False
        for period in periods.values():
            if period["alarm_type"] != alarm_type:
                continue
            if period.get("alarm_source", "") != str(alarm_source or ""):
                continue

            active_event_ids = self._ensure_ordered_active_event_ids(period)
            if not active_event_ids:
                continue

            consumed_cutoff_by_rule = period.setdefault("consumed_trigger_cutoff_by_rule", {})
            for rule_name, cutoff_ts in cutoff_by_rule.items():
                removable_event_ids = set()
                for raw_event_id, raw_ts in active_event_ids.items():
                    if raw_ts > cutoff_ts:
                        break
                    if raw_event_id not in (None, ""):
                        removable_event_ids.add(raw_event_id)
                if not removable_event_ids:
                    continue

                removed_event_ids_by_rule[rule_name].update(removable_event_ids)
                previous_cutoff_ts = consumed_cutoff_by_rule.get(rule_name)
                if previous_cutoff_ts is None or cutoff_ts > previous_cutoff_ts:
                    consumed_cutoff_by_rule[rule_name] = cutoff_ts
                    changed = True

            updated_consumed_rules = frozenset(consumed_cutoff_by_rule.keys())
            if updated_consumed_rules != period["consumed_trigger_rules"]:
                period["consumed_trigger_rules"] = updated_consumed_rules
                changed = True

        if changed:
            self._rebuild_node_event_cache(node)
        self._remove_trigger_events_by_rule_event_ids(node, removed_event_ids_by_rule)

    def _prune_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        if self.use_alarm_period_cache:
            self._prune_period_node_alarm_history_before(node, alarm_type, alarm_source, cutoff_by_rule)
            return
        self._prune_raw_node_alarm_history_before(node, alarm_type, alarm_source, cutoff_by_rule)
