from datetime import datetime
from collections.abc import Iterable


class NodeRuleHelper:
    """封装节点结构匹配、告警窗口校验和失败诊断逻辑。"""

    def __init__(self, sites_domain_map, critical_alarms, event_getter):
        self.sites_domain_map = sites_domain_map
        self.critical_alarms = critical_alarms
        self.event_getter = event_getter

    @staticmethod
    def normalize_edge_window(edge_window):
        """把边时间窗规范化成 before/after 形式，支持对称和非对称窗口。"""
        if isinstance(edge_window, dict):
            before_sec = float(edge_window.get("before_sec", edge_window.get("backward_sec", 0)))
            after_sec = float(edge_window.get("after_sec", edge_window.get("forward_sec", 0)))
            return before_sec, after_sec

        win = float(edge_window)
        return win, win

    def events_in_window(self, physical_node, reference_ts, edge_window, exclude_consumed_trigger_rule=None):
        """获取某个节点在指定时间窗口内的缓存事件。"""
        before_sec, after_sec = self.normalize_edge_window(edge_window)
        return [
            {"node": physical_node, "ts": ts, "eid": eid, "alarm": alarm, "alarm_source": alarm_source}
            for ts, eid, alarm, alarm_source, consumed_trigger_rules in self.event_getter(physical_node)
            if (reference_ts - before_sec) <= ts <= (reference_ts + after_sec)
            and not (
                exclude_consumed_trigger_rule
                and exclude_consumed_trigger_rule in consumed_trigger_rules
            )
        ]

    @staticmethod
    def format_events_for_reason(events):
        """把事件列表格式化成可读的诊断时间线。"""
        return [
            f"{datetime.fromtimestamp(e['ts']).strftime('%Y-%m-%d %H:%M:%S')}|{e['alarm']}"
            for e in sorted(events, key=lambda item: item["ts"])
        ]

    @staticmethod
    def format_window_for_reason(reference_ts, edge_window):
        """把参考时间和窗口宽度格式化成可读区间。"""
        before_sec, after_sec = NodeRuleHelper.normalize_edge_window(edge_window)
        start_ts = reference_ts - before_sec
        end_ts = reference_ts + after_sec
        start_str = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')
        end_str = datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S')
        return f"[{start_str}, {end_str}]"

    @staticmethod
    def has_domain(physical_node_domain, domain):
        """判断站点画像里是否具备某个域能力/设备类型。"""
        if isinstance(physical_node_domain, dict):
            if domain not in physical_node_domain:
                return False
            value = physical_node_domain.get(domain)
            if isinstance(value, (int, float)):
                return value > 0
            if isinstance(value, str):
                return value not in ("", "0")
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) > 0
            return bool(value)

        if isinstance(physical_node_domain, (list, tuple, set)):
            return domain in physical_node_domain

        if isinstance(physical_node_domain, str):
            return domain == physical_node_domain

        return False

    @staticmethod
    def match_site_rule(physical_node_domain, site_rule):
        """判断站点画像是否命中单条 site_rule。"""
        include = site_rule.get("include", [])
        exclude = site_rule.get("exclude", [])

        include_ok = all(NodeRuleHelper.has_domain(physical_node_domain, d) for d in include)
        exclude_ok = all(not NodeRuleHelper.has_domain(physical_node_domain, d) for d in exclude)
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

    def select_candidates_by_rule(self, candidates, candidate_hops, target_node_config, candidate_selector):
        """根据 candidate_selector 对拓扑候选做二次筛选。"""
        if not candidate_selector:
            return candidates

        mode = candidate_selector.get("mode")
        if mode == "nearest_matching":
            matching_candidates = [
                node for node in candidates
                if self.matches_node_structure(self.sites_domain_map.get(node, {}), target_node_config)
            ]
            if not matching_candidates:
                return []
            nearest_hop = min(candidate_hops[node] for node in matching_candidates)
            return [node for node in matching_candidates if candidate_hops[node] == nearest_hop]

        return candidates

    def resolve_expected_alarms(self, physical_node_domain, node_config):
        """根据命中的 site_rule 解析该节点当前应满足的告警集合。"""
        site_rules = node_config.get("site_rules")
        if site_rules:
            for rule in site_rules:
                if self.match_site_rule(physical_node_domain, rule):
                    return rule.get("expected_alarms")
            return None
        return None

    @staticmethod
    def format_expected_alarms_for_reason(expected):
        if expected is None:
            return "未命中任何 site_rule 的 expected_alarms"
        if expected == "ANY":
            return "ANY"
        if expected == "NONE":
            return "NONE"
        if isinstance(expected, dict):
            required_alarms = expected.get("required_alarms")
            forbidden_alarms = expected.get("forbidden_alarms")
            parts = []
            if isinstance(required_alarms, Iterable) and not isinstance(required_alarms, str):
                parts.append(
                    f"required={sorted(str(alarm) for alarm in required_alarms)}"
                )
            if isinstance(forbidden_alarms, Iterable) and not isinstance(forbidden_alarms, str):
                parts.append(
                    f"forbidden={sorted(str(alarm) for alarm in forbidden_alarms)}"
                )
            if parts:
                return ", ".join(parts)
            return str(expected)
        if isinstance(expected, Iterable) and not isinstance(expected, str):
            return str(sorted(str(alarm) for alarm in expected))
        return str(expected)

    def explain_node_validation(self, physical_node, physical_node_domain, node_config, reference_ts, edge_window, exclude_consumed_trigger_rule=None):
        """返回节点校验的可读诊断信息，仅用于 debug 解释。"""
        if not self.matches_node_structure(physical_node_domain, node_config):
            return {
                "valid": False,
                "reason": f"节点 {physical_node} 的站点画像不满足 role 结构约束",
            }

        node_type = node_config.get("type", "primitive")
        if node_type == "primitive":
            expected = self.resolve_expected_alarms(physical_node_domain, node_config)
            if expected is None:
                return {
                    "valid": False,
                    "reason": (
                        f"节点 {physical_node} 未命中任何 site_rule，无法解析 expected_alarms"
                    ),
                }

            events_in_win = self.events_in_window(
                physical_node, reference_ts, edge_window, exclude_consumed_trigger_rule
            )
            window_text = self.format_window_for_reason(reference_ts, edge_window)
            event_timeline = self.format_events_for_reason(events_in_win)

            if expected == "NONE":
                critical_events = [e for e in events_in_win if e["alarm"] in self.critical_alarms]
                if critical_events:
                    return {
                        "valid": False,
                        "reason": (
                            f"窗口 {window_text} 内要求 NONE，但出现 critical 告警: "
                            f"{self.format_events_for_reason(critical_events)}"
                        ),
                    }
                return {
                    "valid": True,
                    "reason": f"窗口 {window_text} 内未出现 critical 告警，满足 NONE",
                }

            if isinstance(expected, dict):
                required_alarms = expected.get("required_alarms")
                forbidden_alarms = expected.get("forbidden_alarms")
                required_events = []
                if isinstance(forbidden_alarms, Iterable) and not isinstance(forbidden_alarms, str):
                    forbidden_events = [e for e in events_in_win if e["alarm"] in forbidden_alarms]
                    if forbidden_events:
                        return {
                            "valid": False,
                            "reason": (
                                f"窗口 {window_text} 内命中 forbidden alarms: "
                                f"{self.format_events_for_reason(forbidden_events)}"
                            ),
                        }
                elif forbidden_alarms is not None:
                    return {
                        "valid": False,
                        "reason": f"forbidden_alarms 配置无法识别: {forbidden_alarms}",
                    }

                if isinstance(required_alarms, Iterable) and not isinstance(required_alarms, str):
                    required_events = [e for e in events_in_win if e["alarm"] in required_alarms]
                    if not required_events:
                        if event_timeline:
                            return {
                                "valid": False,
                                "reason": (
                                    f"窗口 {window_text} 内未命中 required alarms "
                                    f"{self.format_expected_alarms_for_reason(expected)}；"
                                    f" 实际事件: {event_timeline}"
                                ),
                            }
                        return {
                            "valid": False,
                            "reason": (
                                f"窗口 {window_text} 内没有任何事件，未命中 required alarms "
                                f"{self.format_expected_alarms_for_reason(expected)}"
                            ),
                        }
                    return {
                        "valid": True,
                        "reason": (
                            f"窗口 {window_text} 内命中 required alarms: "
                            f"{self.format_events_for_reason(required_events)}"
                        ),
                    }
                if required_alarms is not None:
                    return {
                        "valid": False,
                        "reason": f"required_alarms 配置无法识别: {required_alarms}",
                    }

                if forbidden_alarms is not None:
                    return {
                        "valid": True,
                        "reason": f"窗口 {window_text} 内未命中 forbidden alarms，满足约束",
                    }
                return {
                    "valid": False,
                    "reason": f"expected_alarms 配置无法识别: {expected}",
                }

            if expected == "ANY":
                return {
                    "valid": True,
                    "reason": (
                        f"ANY 不限制告警类型，窗口 {window_text} 内事件数={len(events_in_win)}"
                    ),
                }

            if isinstance(expected, Iterable) and not isinstance(expected, str):
                valid = [e for e in events_in_win if e["alarm"] in expected]
                if valid:
                    return {
                        "valid": True,
                        "reason": (
                            f"窗口 {window_text} 内命中期望告警: {self.format_events_for_reason(valid)}"
                        ),
                    }
                if event_timeline:
                    return {
                        "valid": False,
                        "reason": (
                            f"窗口 {window_text} 内未命中期望告警 {self.format_expected_alarms_for_reason(expected)}；"
                            f" 实际事件: {event_timeline}"
                        ),
                    }
                return {
                    "valid": False,
                    "reason": (
                        f"窗口 {window_text} 内没有任何事件，未命中期望告警 "
                        f"{self.format_expected_alarms_for_reason(expected)}"
                    ),
                }

            return {
                "valid": False,
                "reason": f"expected_alarms 配置无法识别: {expected}",
            }

        if node_type == "compound":
            pattern_reasons = []
            patterns = node_config.get("patterns", [])
            for idx, pattern in enumerate(patterns, start=1):
                pattern_result = self.explain_node_validation(
                    physical_node,
                    physical_node_domain,
                    pattern,
                    reference_ts,
                    edge_window,
                    exclude_consumed_trigger_rule
                )
                if pattern_result.get("valid"):
                    return {
                        "valid": True,
                        "reason": f"compound pattern[{idx}] 满足: {pattern_result.get('reason', '')}",
                    }
                pattern_reasons.append(f"pattern[{idx}]: {pattern_result.get('reason', '')}")
            return {
                "valid": False,
                "reason": "compound 所有 pattern 都不满足: " + "; ".join(pattern_reasons),
            }

        return {
            "valid": False,
            "reason": f"未知节点类型: {node_type}",
        }

    def validate_node(self, physical_node, physical_node_domain, node_config, reference_ts, edge_window, exclude_consumed_trigger_rule=None):
        """按结构与时间窗口告警共同校验一个节点是否满足规则定义。"""
        if not self.matches_node_structure(physical_node_domain, node_config):
            return False, []

        node_type = node_config.get("type", "primitive")

        if node_type == "primitive":
            expected = self.resolve_expected_alarms(physical_node_domain, node_config)
            if expected is None:
                return False, []
            events_in_win = self.events_in_window(
                physical_node, reference_ts, edge_window, exclude_consumed_trigger_rule
            )

            if expected == "NONE":
                has_crit = any(e["alarm"] in self.critical_alarms for e in events_in_win)
                return not has_crit, []
            if isinstance(expected, dict):
                required_alarms = expected.get("required_alarms")
                forbidden_alarms = expected.get("forbidden_alarms")
                if isinstance(forbidden_alarms, Iterable) and not isinstance(forbidden_alarms, str):
                    has_forbidden = any(e["alarm"] in forbidden_alarms for e in events_in_win)
                    if has_forbidden:
                        return False, []
                elif forbidden_alarms is not None:
                    return False, []

                if isinstance(required_alarms, Iterable) and not isinstance(required_alarms, str):
                    valid = [e for e in events_in_win if e["alarm"] in required_alarms]
                    return len(valid) > 0, valid
                if required_alarms is not None:
                    return False, []

                if forbidden_alarms is not None:
                    return True, []
                return False, []
            if expected == "ANY":
                return True, events_in_win
            if isinstance(expected, Iterable):
                valid = [e for e in events_in_win if e["alarm"] in expected]
                return len(valid) > 0, valid
            return False, []

        if node_type == "compound":
            patterns = node_config.get("patterns", [])
            matched_patterns = 0
            collected_events = []

            for pattern in patterns:
                is_valid, events = self.validate_node(
                    physical_node,
                    physical_node_domain,
                    pattern,
                    reference_ts,
                    edge_window,
                    exclude_consumed_trigger_rule
                )
                if is_valid:
                    matched_patterns += 1
                    collected_events.extend(events)

            return matched_patterns > 0, collected_events

        return False, []
