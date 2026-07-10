from anchor_grouping_online.temporal_engine.utils import _has_domain, _matches_source_domain


class NodeRuleHelper:
    """封装节点结构匹配、告警窗口校验和失败诊断逻辑。"""

    def __init__(self, event_getter, alarm_source_domain_map=None):
        self.event_getter = event_getter
        self.alarm_source_domain_map = alarm_source_domain_map or {}

    def events_in_window(self, physical_node, reference_ts, edge_window, exclude_consumed_trigger_rule=None):
        """获取某个节点在指定时间窗口内命中的告警发生事件。

        只看“告警发生时间是否落在窗口里”。event_cache 每个节点保存一个
        按到达顺序排列的事件 dict deque。
        """
        window_sec = float(edge_window)
        window_start = reference_ts - window_sec
        window_end = reference_ts + window_sec
        matched_events = []

        for cached_event in self.event_getter(physical_node):
            ts = cached_event.get("ts")
            if ts is None:
                continue
            consumed_trigger_rules = cached_event.get("consumed_trigger_rules", ())
            if exclude_consumed_trigger_rule and exclude_consumed_trigger_rule in consumed_trigger_rules:
                continue
            if ts < window_start or ts > window_end:
                continue

            alarm_source = cached_event.get("alarm_source", "")
            matched_events.append({
                "node": physical_node,
                "ts": ts,
                "eid": cached_event.get("eid"),
                "alarm": cached_event.get("alarm"),
                "alarm_source": alarm_source,
                "alarm_source_domain": self.alarm_source_domain_map.get(alarm_source, ""),
                "alarm_payload": cached_event.get("alarm_payload") or {},
            })

        return matched_events

    @staticmethod
    def filter_events_by_alarm_and_source(events, alarms, source_domains=None):
        return [
            event for event in events
            if event["alarm"] in alarms
            and _matches_source_domain(event.get("alarm_source_domain", ""), source_domains)
        ]

    @staticmethod
    def match_site_rule(physical_node_domain, site_rule):
        """判断站点画像是否命中单条 site_rule。"""
        include = site_rule.get("include", [])
        exclude = site_rule.get("exclude", [])

        include_ok = all(_has_domain(physical_node_domain, d) for d in include)
        exclude_ok = all(not _has_domain(physical_node_domain, d) for d in exclude)
        return include_ok and exclude_ok

    def matches_node_structure(self, physical_node_domain, node_config):
        """判断站点画像是否满足节点结构约束，支持 compound 递归。"""
        site_rules = node_config.get("site_rules")
        if site_rules:
            if not any(self.match_site_rule(physical_node_domain, rule) for rule in site_rules):
                return False

        node_type = node_config.get("type", "primitive")
        if node_type == "compound":
            patterns = node_config.get("patterns", [])
            if not patterns:
                return False
            return any(self.matches_node_structure(physical_node_domain, pattern) for pattern in patterns)

        return True

    def resolve_expected_alarms(self, physical_node_domain, node_config):
        """根据命中的 site_rule 解析该节点当前应满足的告警集合。"""
        site_rules = node_config.get("site_rules")
        if site_rules:
            for rule in site_rules:
                if self.match_site_rule(physical_node_domain, rule):
                    return rule.get("expected_alarms")
            return None
        return None

    def validate_node(
        self, physical_node, physical_node_domain, node_config, reference_ts,
        edge_window, exclude_consumed_trigger_rule=None,
        allowed_alarm_source_nes=None,
    ):
        """按结构与时间窗口告警共同校验一个节点是否满足规则定义。

        allowed_alarm_source_nes: 可选 frozenset，若提供则仅保留 alarm_source 在该集合内
        的 events 参与谓词判定（用于实现 alarm_source_ne_anchor 的 NE 级过滤）。
        """
        if not self.matches_node_structure(physical_node_domain, node_config):
            return False, []
        node_type = node_config.get("type", "primitive")
        if node_type == "primitive":
            return self._validate_primitive_node(
                physical_node, physical_node_domain, node_config, reference_ts,
                edge_window, exclude_consumed_trigger_rule,
                allowed_alarm_source_nes,
            )
        if node_type == "compound":
            return self._validate_compound_node(
                physical_node, physical_node_domain, node_config, reference_ts,
                edge_window, exclude_consumed_trigger_rule,
                allowed_alarm_source_nes,
            )
        return False, []

    def _validate_primitive_node(
        self, physical_node, physical_node_domain, node_config, reference_ts,
        edge_window, exclude_consumed_trigger_rule, allowed_alarm_source_nes,
    ):
        """primitive 节点：按 expected_alarms 的必含/禁止/可选告警校验。"""
        expected = self.resolve_expected_alarms(physical_node_domain, node_config)
        if expected is None:
            return False, []
        events_in_win = self.events_in_window(
            physical_node, reference_ts, edge_window, exclude_consumed_trigger_rule
        )
        if allowed_alarm_source_nes is not None:
            events_in_win = [
                e for e in events_in_win
                if e.get("alarm_source") in allowed_alarm_source_nes
            ]
        required_alarms = expected.get("required_alarms")
        forbidden_alarms = expected.get("forbidden_alarms")
        optional_alarms = expected.get("optional_alarms")
        required_source_domains = expected.get("required_alarm_source_domains")
        if forbidden_alarms is not None:
            if any(e["alarm"] in forbidden_alarms for e in events_in_win):
                return False, []
        collected_events = []
        if required_alarms is not None:
            valid = self.filter_events_by_alarm_and_source(
                events_in_win,
                required_alarms,
                required_source_domains,
            )
            if not valid:
                return False, []
            collected_events.extend(valid)
        if optional_alarms is not None:
            collected_events.extend(
                self.filter_events_by_alarm_and_source(
                    events_in_win,
                    optional_alarms,
                )
            )
        if collected_events:
            return True, self._dedupe_events(collected_events)
        if optional_alarms is not None or forbidden_alarms is not None:
            return True, []
        return False, []

    @staticmethod
    def _dedupe_events(events):
        """按告警发生 eid 去重，保留首次出现顺序。"""
        deduped_events = []
        seen_event_ids = set()
        for event in events:
            event_id = str(event["eid"])
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            deduped_events.append(event)
        return deduped_events

    def _validate_compound_node(
        self, physical_node, physical_node_domain, node_config, reference_ts,
        edge_window, exclude_consumed_trigger_rule, allowed_alarm_source_nes,
    ):
        """compound 节点：任一 pattern 命中即通过，事件取命中 pattern 的并集。"""
        matched_patterns = 0
        collected_events = []
        for pattern in node_config.get("patterns", []):
            is_valid, events = self.validate_node(
                physical_node,
                physical_node_domain,
                pattern,
                reference_ts,
                edge_window,
                exclude_consumed_trigger_rule,
                allowed_alarm_source_nes=allowed_alarm_source_nes,
            )
            if is_valid:
                matched_patterns += 1
                collected_events.extend(events)
        return matched_patterns > 0, collected_events
