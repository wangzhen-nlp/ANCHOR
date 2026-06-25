import collections

class TemporalGraphEngineAlarmPeriodMixin:
    @staticmethod
    def _cached_event_field(cached_event, field_name, tuple_index=None, default=None):
        if isinstance(cached_event, dict):
            return cached_event.get(field_name, default)
        if tuple_index is None:
            return default
        try:
            return cached_event[tuple_index]
        except IndexError:
            return default

    @classmethod
    def _cached_event_parts(cls, cached_event):
        if isinstance(cached_event, dict):
            return (
                cached_event.get("ts"),
                cached_event.get("eid"),
                cached_event.get("alarm"),
                cached_event.get("alarm_source", ""),
                cached_event.get("consumed_trigger_rules", ()),
                cached_event.get("occurrence_uuid"),
            )
        return cached_event

    @staticmethod
    def _active_event_item(raw_occurrence_key, raw_value):
        event_id, ts = raw_value
        return raw_occurrence_key, event_id, ts

    @classmethod
    def _iter_active_event_items(cls, active_event_ids):
        for raw_occurrence_key, raw_value in active_event_ids.items():
            yield cls._active_event_item(raw_occurrence_key, raw_value)

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
        raw_event_payloads = dict(period_state.get("active_event_payloads", {}))
        raw_event_items = tuple(
            (raw_event_id, raw_ts, raw_occurrence_key)
            for raw_occurrence_key, raw_event_id, raw_ts in (
                TemporalGraphEngineAlarmPeriodMixin._iter_active_event_items(
                    period_state.get("active_event_ids", {})
                )
            )
        )
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
            "_raw_event_ts_list": tuple(raw_item[1] for raw_item in raw_event_items),
            "_raw_event_payloads": raw_event_payloads,
            "_consumed_cutoff_by_rule": dict(period_state.get("consumed_trigger_cutoff_by_rule", {})),
        }

    @staticmethod
    def _ensure_ordered_active_event_ids(period_state):
        active_event_ids = period_state.get("active_event_ids")
        if not isinstance(active_event_ids, collections.OrderedDict):
            active_event_ids = collections.OrderedDict(active_event_ids or {})
            period_state["active_event_ids"] = active_event_ids
        return active_event_ids

    @classmethod
    def _refresh_alarm_period_state(cls, node, period_state):
        active_event_ids = cls._ensure_ordered_active_event_ids(period_state)
        if not active_event_ids:
            return False

        leader_occurrence_key = next(iter(active_event_ids))
        _leader_occurrence_key, leader_event_id, leader_ts = cls._active_event_item(
            leader_occurrence_key,
            active_event_ids[leader_occurrence_key],
        )
        tail_occurrence_key = next(reversed(active_event_ids))
        _tail_occurrence_key, tail_event_id, tail_ts = cls._active_event_item(
            tail_occurrence_key,
            active_event_ids[tail_occurrence_key],
        )
        period_state["ts"] = leader_ts
        period_state["eid"] = leader_event_id
        period_state["latest_active_ts"] = tail_ts
        period_state["end_ts"] = tail_ts
        period_state["consumed_trigger_rules"] = frozenset(
            period_state.get("consumed_trigger_cutoff_by_rule", {}).keys()
        )
        period_state["segment_key"] = (
            f"{node}|{period_state['alarm_source']}|{period_state['alarm_type']}|"
            f"{leader_ts:.6f}|{tail_ts:.6f}|{leader_occurrence_key}"
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

    def _register_alarm_period_occurrence(
        self,
        node,
        alarm_type,
        ts,
        event_id,
        occurrence_uuid,
        alarm_source="",
        alarm_payload=None,
    ):
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
                "active_event_payloads": {},
                "_occurrence_seq": 0,
                "latest_active_ts": ts,
                "segment_key": "",
            }
            periods[period_key] = period

        if event_id not in (None, ""):
            active_event_ids = self._ensure_ordered_active_event_ids(period)
            active_event_ids[occurrence_uuid] = (event_id, ts)
            period.setdefault("active_event_payloads", {})[occurrence_uuid] = (
                alarm_payload if isinstance(alarm_payload, dict) else {}
            )
            self.active_event_to_period[node][(str(event_id), occurrence_uuid)] = period_key

        if self._refresh_alarm_period_state(node, period):
            self._rebuild_node_event_cache(node)
        return created

    def _prune_expired_raw_events_in_place(self, node, current_ts):
        q = self.event_cache.get(node)
        if not q:
            return

        while q and (
            current_ts - self._cached_event_field(q[0], "ts", 0, 0)
        ) > self._get_event_ttl(self._cached_event_field(q[0], "alarm", 2)):
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
            expired_items = []
            while active_event_ids:
                leader_occurrence_key = next(iter(active_event_ids))
                _leader_occurrence_key, leader_event_id, leader_ts = self._active_event_item(
                    leader_occurrence_key,
                    active_event_ids[leader_occurrence_key],
                )
                if (current_ts - leader_ts) <= ttl:
                    break
                expired_items.append((leader_occurrence_key, leader_event_id))
                active_event_ids.popitem(last=False)
            if not expired_items:
                continue

            changed = True
            for raw_occurrence_key, raw_event_id in expired_items:
                if raw_event_id not in (None, ""):
                    self.active_event_to_period.get(node, {}).pop(
                        (str(raw_event_id), raw_occurrence_key),
                        None,
                    )
                period.get("active_event_payloads", {}).pop(raw_occurrence_key, None)

            if self._refresh_alarm_period_state(node, period):
                continue

            removed_event = self._period_state_to_cached_event(node, period)
            self._log_debug_event_removal(node, removed_event, "ttl", current_ts=current_ts)
            periods.pop(period_key, None)

        if changed:
            if not self.active_event_to_period.get(node):
                self.active_event_to_period.pop(node, None)
            self._rebuild_node_event_cache(node)

    def _remove_cleared_raw_event(
        self,
        node,
        event_id,
        occurrence_uuid,
        alarm_type=None,
        alarm_source=None,
    ):
        q = self.event_cache[node]
        kept = collections.deque()
        target_alarm_source = None if alarm_source is None else str(alarm_source or "")

        for cached_event in q:
            (
                cached_ts,
                cached_eid,
                cached_alarm_type,
                cached_alarm_source,
                _consumed_trigger_rules,
                cached_occurrence_uuid,
            ) = self._cached_event_parts(cached_event)
            matches_clear = (
                event_id
                and cached_eid == event_id
                and cached_occurrence_uuid == occurrence_uuid
                and (alarm_type is None or cached_alarm_type == alarm_type)
                and (
                    target_alarm_source is None
                    or str(cached_alarm_source or "") == target_alarm_source
                )
            )
            if matches_clear:
                self._log_debug_event_removal(
                    node,
                    cached_event,
                    "clear",
                    cleared_event_id=event_id,
                )
                continue
            kept.append(cached_event)

        self.event_cache[node] = kept

    def _remove_cleared_alarm_period_event(
        self,
        node,
        event_id,
        occurrence_uuid,
        alarm_type=None,
        alarm_source=None,
    ):
        if event_id in (None, ""):
            return

        period_key = None
        if alarm_type is not None:
            candidate_key = self._make_alarm_period_key(alarm_type, alarm_source)
            if candidate_key in self.active_alarm_periods.get(node, {}):
                period_key = candidate_key
            else:
                return
        else:
            period_key = self.active_event_to_period.get(node, {}).get((str(event_id), occurrence_uuid))
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
        for occurrence_key, raw_event_id, _raw_ts in list(self._iter_active_event_items(active_event_ids)):
            if raw_event_id == event_id and occurrence_key == occurrence_uuid:
                active_event_ids.pop(occurrence_key, None)
                period.get("active_event_payloads", {}).pop(occurrence_key, None)
        self.active_event_to_period.get(node, {}).pop((str(event_id), occurrence_uuid), None)
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

    def _remove_cleared_events(
        self,
        node,
        event_id,
        occurrence_uuid,
        alarm_type=None,
        alarm_source=None,
    ):
        if self.use_alarm_period_cache:
            self._remove_cleared_alarm_period_event(
                node,
                event_id,
                occurrence_uuid,
                alarm_type=alarm_type,
                alarm_source=alarm_source,
            )
            return
        self._remove_cleared_raw_event(
            node,
            event_id,
            occurrence_uuid,
            alarm_type=alarm_type,
            alarm_source=alarm_source,
        )

    def _remove_trigger_events_by_rule_event_keys(self, node, removed_event_keys_by_rule):
        if not removed_event_keys_by_rule:
            return

        for rule_name, removed_event_keys in removed_event_keys_by_rule.items():
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            kept_trigger_events = collections.deque()
            for trigger_event in trigger_events:
                (
                    event_ts,
                    indexed_event_id,
                    indexed_seq,
                    indexed_alarm_type,
                    indexed_alarm_source,
                    indexed_occurrence_uuid,
                ) = self._unpack_trigger_event(trigger_event)
                event_key = (
                    event_ts,
                    indexed_event_id,
                    indexed_alarm_type,
                    indexed_alarm_source,
                    indexed_occurrence_uuid,
                )
                if event_key in removed_event_keys:
                    continue
                kept_trigger_events.append(trigger_event)

            if kept_trigger_events:
                self.trigger_event_index[trigger_key] = kept_trigger_events
            else:
                self.trigger_event_index.pop(trigger_key, None)

        self._refresh_pending_triggers_for_node(
            node,
            affected_rule_names=removed_event_keys_by_rule.keys(),
        )

    def _prune_raw_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        q = self.event_cache.get(node)
        if not q:
            return

        removed_event_keys_by_rule = collections.defaultdict(set)
        kept = collections.deque()
        target_alarm_source = str(alarm_source or "")
        for cached_event in q:
            (
                cached_ts,
                cached_eid,
                cached_alarm_type,
                cached_alarm_source,
                consumed_trigger_rules,
                cached_occurrence_uuid,
            ) = self._cached_event_parts(cached_event)
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
                        removed_event_keys_by_rule[rule_name].add((
                            cached_ts,
                            cached_eid,
                            cached_alarm_type,
                            str(cached_alarm_source or ""),
                            cached_occurrence_uuid,
                        ))
                updated_consumed_rules = frozenset(set(consumed_trigger_rules) | matched_rules)
                if isinstance(cached_event, dict):
                    updated_event = dict(cached_event)
                    updated_event["consumed_trigger_rules"] = updated_consumed_rules
                    kept.append(updated_event)
                else:
                    kept.append((
                        cached_ts,
                        cached_eid,
                        cached_alarm_type,
                        cached_alarm_source,
                        updated_consumed_rules,
                        cached_occurrence_uuid,
                    ))
                continue
            kept.append(cached_event)

        self.event_cache[node] = kept
        self._remove_trigger_events_by_rule_event_keys(node, removed_event_keys_by_rule)

    def _prune_period_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        periods = self.active_alarm_periods.get(node)
        if not periods:
            return

        removed_event_keys_by_rule = collections.defaultdict(set)
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
                removable_event_keys = set()
                for raw_occurrence_key, raw_event_id, raw_ts in self._iter_active_event_items(active_event_ids):
                    if raw_ts > cutoff_ts:
                        break
                    if raw_event_id not in (None, ""):
                        removable_event_keys.add((
                            raw_ts,
                            raw_event_id,
                            period["alarm_type"],
                            str(period.get("alarm_source", "") or ""),
                            raw_occurrence_key,
                        ))
                if not removable_event_keys:
                    continue

                removed_event_keys_by_rule[rule_name].update(removable_event_keys)
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
        self._remove_trigger_events_by_rule_event_keys(node, removed_event_keys_by_rule)

    def _prune_node_alarm_history_before(self, node, alarm_type, alarm_source, cutoff_by_rule):
        if self.use_alarm_period_cache:
            self._prune_period_node_alarm_history_before(node, alarm_type, alarm_source, cutoff_by_rule)
            return
        self._prune_raw_node_alarm_history_before(node, alarm_type, alarm_source, cutoff_by_rule)
