from datetime import datetime
from collections.abc import Iterable


class NodeRuleHelper:
    """封装节点结构匹配、告警窗口校验和失败诊断逻辑。"""

    def __init__(self, sites_domain_map, critical_alarms, event_getter):
        self.sites_domain_map = sites_domain_map
        self.critical_alarms = critical_alarms
        self.event_getter = event_getter

    def events_in_window(self, physical_node, reference_ts, edge_window):
        """获取某个节点在指定时间窗口内的缓存事件。"""
        return [
            {"node": physical_node, "ts": ts, "eid": eid, "alarm": alarm, "alarm_source": alarm_source}
            for ts, eid, alarm, alarm_source in self.event_getter(physical_node)
            if abs(reference_ts - ts) <= edge_window
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
        start_ts = reference_ts - edge_window
        end_ts = reference_ts + edge_window
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

    def validate_node(self, physical_node, physical_node_domain, node_config, reference_ts, edge_window):
        """按结构与时间窗口告警共同校验一个节点是否满足规则定义。"""
        if not self.matches_node_structure(physical_node_domain, node_config):
            return False, []

        node_type = node_config.get("type", "primitive")

        if node_type == "primitive":
            expected = self.resolve_expected_alarms(physical_node_domain, node_config)
            if expected is None:
                return False, []
            events_in_win = self.events_in_window(physical_node, reference_ts, edge_window)

            if expected == "NONE":
                has_crit = any(e["alarm"] in self.critical_alarms for e in events_in_win)
                return not has_crit, []
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
                    physical_node, physical_node_domain, pattern, reference_ts, edge_window
                )
                if is_valid:
                    matched_patterns += 1
                    collected_events.extend(events)

            return matched_patterns > 0, collected_events

        return False, []
