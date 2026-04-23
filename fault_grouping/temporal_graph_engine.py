import collections
import heapq
import logging
import time
import threading
import uuid

from datetime import datetime

from fault_grouping.emitted_group_store import EmittedGroupStore
from fault_grouping.node_rule_helper import NodeRuleHelper
from alarm_tools.alarm_types import CRITICAL_ALARMS, POWER_ALARMS
from fault_grouping.temporal_graph_engine_utils import (
    add_merge_stats,
    build_pattern_adj,
    build_empty_merge_stats,
    clone_instance_with_updates,
    matches_expected_alarm,
    merge_match_batch,
)

logger = logging.getLogger(__name__)


class TemporalGraphEngine:
    @staticmethod
    def _normalize_traverse_directions(direction):
        if isinstance(direction, str):
            text = direction.strip()
            return (text,) if text else ("downstream",)
        if isinstance(direction, (list, tuple, set)):
            directions = []
            seen = set()
            for item in direction:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                directions.append(text)
            return tuple(directions) if directions else ("downstream",)
        return (str(direction).strip() or "downstream",)

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
    def _make_edge_window_cache_key(edge_window):
        if isinstance(edge_window, dict):
            return tuple(sorted(edge_window.items()))
        return edge_window

    @staticmethod
    def _create_eval_caches():
        return {
            "validation_cache": {},
            "traversal_cache": {},
            "path_validation_cache": {},
            "structure_match_cache": {},
            "filtered_neighbor_cache": {},
        }

    @staticmethod
    def _normalize_site_chain_hops(raw_hops):
        if not isinstance(raw_hops, dict):
            return {}

        normalized = {}
        for raw_site_id, raw_hop in raw_hops.items():
            site_id = str(raw_site_id or "").strip()
            if not site_id:
                continue
            try:
                hop = int(raw_hop)
            except (TypeError, ValueError):
                continue
            if hop <= 0:
                continue
            normalized[site_id] = hop
        return normalized

    @classmethod
    def _normalize_site_chain_index(cls, site_chain_index):
        if not isinstance(site_chain_index, dict):
            return {}

        normalized = {}
        for raw_site_id, raw_info in site_chain_index.items():
            site_id = str(raw_site_id or "").strip()
            if not site_id or not isinstance(raw_info, dict):
                continue

            bidirectional_sites = {
                str(site_id).strip()
                for site_id in raw_info.get("bidirectional_sites", [])
                if str(site_id).strip()
            }
            normalized[site_id] = {
                "downstream_site_hops": cls._normalize_site_chain_hops(
                    raw_info.get("downstream_site_hops")
                ),
                "upstream_site_hops": cls._normalize_site_chain_hops(
                    raw_info.get("upstream_site_hops")
                ),
                "bidirectional_sites": bidirectional_sites,
            }

        return normalized

    def _compile_rule_execution_plan(self, rule):
        """为单条规则预编译静态执行计划，避免每次评估重复构图和排边。"""
        nodes_cfg = rule.get("nodes", {})
        edges_cfg = rule.get("edges", [])
        trigger_role = rule["trigger_role"]

        pattern_adj = build_pattern_adj(edges_cfg)

        edges_to_explore = []
        visited_edges = set()
        queue = collections.deque([trigger_role])
        while queue:
            curr = queue.popleft()
            for edge in pattern_adj[curr]:
                tgt = edge["role"]
                edge_id = (curr, tgt)
                if edge_id in visited_edges:
                    continue
                visited_edges.add(edge_id)
                edges_to_explore.append((curr, tgt, edge))
                queue.append(tgt)

        targets = {edge["target"] for edge in edges_cfg}
        root_roles = tuple(role for role in nodes_cfg.keys() if role not in targets)

        return {
            "trigger_role": trigger_role,
            "edges_to_explore": tuple(edges_to_explore),
            "root_roles": root_roles,
        }

    def _compile_rule_execution_plans(self):
        """预编译所有规则的静态执行计划。"""
        self.rule_execution_plans = {
            rule_name: self._compile_rule_execution_plan(rule)
            for rule_name, rule in self.rules.items()
        }

    def _record_instance_dependency(self, inst, curr_role, tgt_role, curr_support_targets):
        """记录一条已处理边上的节点依赖关系，供后续收敛裁剪使用。"""
        if not curr_support_targets:
            return

        src_to_dst = {
            curr_node: set(target_nodes)
            for curr_node, target_nodes in curr_support_targets.items()
            if target_nodes
        }
        if not src_to_dst:
            return

        dst_to_src = collections.defaultdict(set)
        for curr_node, target_nodes in src_to_dst.items():
            for target_node in target_nodes:
                dst_to_src[target_node].add(curr_node)

        dependencies = inst.setdefault("_dependencies", {})
        dependencies[(curr_role, tgt_role)] = {
            "src_to_dst": src_to_dst,
            "dst_to_src": {target_node: set(src_nodes) for target_node, src_nodes in dst_to_src.items()},
        }
        dependencies[(tgt_role, curr_role)] = {
            "src_to_dst": {
                target_node: set(src_nodes)
                for target_node, src_nodes in dst_to_src.items()
            },
            "dst_to_src": {
                curr_node: set(target_nodes)
                for curr_node, target_nodes in src_to_dst.items()
            },
        }

    def _stabilize_instance_dependencies(self, inst, nodes_cfg):
        """基于已记录的边依赖做收敛裁剪，把深层失效回传到上下游角色。"""
        dependencies = inst.get("_dependencies")
        if not dependencies:
            return inst

        stabilized_inst = dict(inst)
        stabilized_inst["roles"] = {
            role: {
                "nodes": dict(role_state["nodes"]),
                "checked": role_state["checked"]
            }
            for role, role_state in inst.get("roles", {}).items()
        }
        stabilized_roles = stabilized_inst["roles"]

        while True:
            changed = False

            for (src_role, dst_role), dep in dependencies.items():
                src_state = stabilized_roles.get(src_role)
                dst_state = stabilized_roles.get(dst_role)
                if src_state is None or dst_state is None:
                    continue

                live_src_nodes = src_state["nodes"]
                live_dst_nodes = dst_state["nodes"]
                live_dst_set = set(live_dst_nodes)
                live_src_set = set(live_src_nodes)

                kept_src_nodes = {
                    src_node: live_src_nodes[src_node]
                    for src_node in live_src_nodes
                    if dep.get("src_to_dst", {}).get(src_node, set()) & live_dst_set
                }
                if len(kept_src_nodes) != len(live_src_nodes):
                    src_state["nodes"] = kept_src_nodes
                    live_src_nodes = kept_src_nodes
                    live_src_set = set(kept_src_nodes)
                    changed = True

                kept_dst_nodes = {
                    dst_node: live_dst_nodes[dst_node]
                    for dst_node in live_dst_nodes
                    if dep.get("dst_to_src", {}).get(dst_node, set()) & live_src_set
                }
                if len(kept_dst_nodes) != len(live_dst_nodes):
                    dst_state["nodes"] = kept_dst_nodes
                    changed = True

            for role, role_state in stabilized_roles.items():
                if role_state["checked"] and len(role_state["nodes"]) < nodes_cfg.get(role, {}).get("min_count", 1):
                    return None

            if not changed:
                return stabilized_inst

    @staticmethod
    def _hide_role_node_without_alarm(node_config):
        """是否在最终输出中隐藏该 role 下没有贡献任何告警的站点。"""
        return bool(
            node_config.get("hide_if_no_alarms")
            or node_config.get("hide_if_no_events")
        )

    def _get_role_node_config_for_output(self, role, match_result):
        merged_rules = match_result.get("merged_rules", [match_result.get("rule")])
        for rule_name in merged_rules:
            rule = self.rules.get(rule_name)
            if not rule:
                continue
            node_config = rule.get("nodes", {}).get(role)
            if node_config is not None:
                return node_config
        return {}

    def _apply_output_visibility_filters(self, match_result):
        """仅过滤最终输出视图，不影响规则匹配和 result_constraints 判断。"""
        alarm_nodes_by_role = collections.defaultdict(set)
        for symptom in match_result.get("symptoms", []):
            role = symptom.get("matched_role")
            node = symptom.get("node")
            if role and node not in (None, ""):
                alarm_nodes_by_role[role].add(node)

        filtered_role_mapping = {}
        for role, nodes in match_result.get("role_mapping", {}).items():
            node_config = self._get_role_node_config_for_output(role, match_result)
            if self._hide_role_node_without_alarm(node_config):
                nodes = [
                    node for node in nodes
                    if node in alarm_nodes_by_role.get(role, set())
                ]
            if nodes:
                filtered_role_mapping[role] = nodes

        filtered_inferred_roots = {}
        for role, nodes in match_result.get("inferred_roots", {}).items():
            node_config = self._get_role_node_config_for_output(role, match_result)
            if self._hide_role_node_without_alarm(node_config):
                nodes = [
                    node for node in nodes
                    if node in alarm_nodes_by_role.get(role, set())
                ]
            if nodes:
                filtered_inferred_roots[role] = nodes

        if (
            filtered_role_mapping == match_result.get("role_mapping", {})
            and filtered_inferred_roots == match_result.get("inferred_roots", {})
        ):
            return match_result

        return {
            **match_result,
            "role_mapping": filtered_role_mapping,
            "inferred_roots": filtered_inferred_roots,
        }

    def _apply_output_visibility_filters_to_matches(self, matches):
        if not matches:
            return matches
        return [
            self._apply_output_visibility_filters(match_result)
            for match_result in matches
        ]

    def _get_parent_roles_for_site_ownership(self, role, match_result):
        parent_roles = []
        seen = set()
        merged_rules = match_result.get("merged_rules", [match_result.get("rule")])
        for rule_name in merged_rules:
            rule = self.rules.get(rule_name)
            if not rule:
                continue
            for edge in rule.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                directions = self._normalize_traverse_directions(edge.get("direction", "downstream"))
                if source == role and "upstream" in directions:
                    parent_role = target
                elif target == role and "downstream" in directions:
                    parent_role = source
                else:
                    continue
                if parent_role and parent_role not in seen:
                    seen.add(parent_role)
                    parent_roles.append(parent_role)
        return parent_roles

    def _get_match_role_site_owner_distance(self, match_result, role, site):
        if not self.site_chain_index:
            return None

        role_mapping = match_result.get("role_mapping", {})
        best_hop = None
        for parent_role in self._get_parent_roles_for_site_ownership(role, match_result):
            for parent_site in role_mapping.get(parent_role, []):
                hop = self._get_site_chain_downstream_hop(parent_site, site)
                if hop is None:
                    continue
                if best_hop is None or hop < best_hop:
                    best_hop = hop
        return best_hop

    def _choose_match_site_owner_role(self, match_result, site, roles, role_order):
        distance_candidates = []
        for role in roles:
            distance = self._get_match_role_site_owner_distance(match_result, role, site)
            if distance is not None:
                distance_candidates.append((distance, role_order[role], role))

        if distance_candidates:
            return min(distance_candidates)[2]
        return min(roles, key=lambda role: role_order[role])

    @staticmethod
    def _normalize_exclusive_site_role_groups(raw_config, available_roles):
        if not raw_config:
            return []

        available_role_set = set(available_roles)

        def normalize_role_group(raw_group):
            if raw_group is True:
                return list(available_roles)
            if isinstance(raw_group, str):
                if raw_group.strip().lower() in {"all", "*"}:
                    return list(available_roles)
                raw_group = [part.strip() for part in raw_group.split(",")]
            if isinstance(raw_group, dict):
                raw_group = raw_group.get("roles", raw_group.get("role", []))
            if not isinstance(raw_group, (list, tuple, set)):
                return []

            roles = []
            seen = set()
            for role in raw_group:
                if not isinstance(role, str):
                    continue
                role = role.strip()
                if not role or role not in available_role_set or role in seen:
                    continue
                seen.add(role)
                roles.append(role)
            return roles

        if raw_config is True or isinstance(raw_config, str):
            raw_groups = [raw_config]
        elif isinstance(raw_config, dict):
            if "groups" in raw_config:
                raw_groups = raw_config.get("groups") or []
            else:
                raw_groups = [raw_config]
        elif isinstance(raw_config, (list, tuple, set)):
            if all(isinstance(item, str) for item in raw_config):
                raw_groups = [raw_config]
            else:
                raw_groups = raw_config
        else:
            raw_groups = []

        groups = []
        seen_groups = set()
        for raw_group in raw_groups:
            group = normalize_role_group(raw_group)
            if len(group) <= 1:
                continue
            group_key = tuple(group)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            groups.append(group)
        return groups

    def _get_exclusive_site_role_groups_for_output(self, match_result, available_roles):
        groups = []
        merged_rules = match_result.get("merged_rules", [match_result.get("rule")])
        for rule_name in merged_rules:
            rule = self.rules.get(rule_name)
            if not rule:
                continue
            groups.extend(
                self._normalize_exclusive_site_role_groups(
                    rule.get("exclusive_site_roles"),
                    available_roles,
                )
            )
        return groups

    def _apply_default_output_site_role_ownership(self, match_result):
        role_mapping = match_result.get("role_mapping", {})
        if len(role_mapping) <= 1:
            return match_result

        exclusive_role_groups = self._get_exclusive_site_role_groups_for_output(
            match_result,
            list(role_mapping.keys()),
        )
        if not exclusive_role_groups:
            return match_result

        role_order = {role: idx for idx, role in enumerate(role_mapping.keys())}
        remove_by_role = collections.defaultdict(set)
        owner_by_removed_role_site = {}
        for exclusive_roles in exclusive_role_groups:
            site_to_roles = collections.defaultdict(list)
            for role in exclusive_roles:
                for site in role_mapping.get(role, []):
                    site_to_roles[site].append(role)

            for site, site_roles in site_to_roles.items():
                if len(site_roles) <= 1:
                    continue
                owner_role = self._choose_match_site_owner_role(match_result, site, site_roles, role_order)
                for role in site_roles:
                    if role != owner_role:
                        remove_by_role[role].add(site)
                        owner_by_removed_role_site[(role, site)] = owner_role

        if not remove_by_role:
            return match_result

        filtered_role_mapping = {}
        for role, nodes in role_mapping.items():
            filtered_nodes = [
                node for node in nodes
                if node not in remove_by_role.get(role, set())
            ]
            if filtered_nodes:
                filtered_role_mapping[role] = filtered_nodes

        filtered_inferred_roots = {}
        for role, nodes in match_result.get("inferred_roots", {}).items():
            filtered_nodes = [
                node for node in nodes
                if node not in remove_by_role.get(role, set())
            ]
            if filtered_nodes:
                filtered_inferred_roots[role] = filtered_nodes

        filtered_symptoms = []
        for symptom in match_result.get("symptoms", []):
            role = symptom.get("matched_role")
            node = symptom.get("node")
            if role in remove_by_role and node in remove_by_role.get(role, set()):
                owner_role = owner_by_removed_role_site.get((role, node))
                if owner_role and node in filtered_role_mapping.get(owner_role, []):
                    symptom = {**symptom, "matched_role": owner_role}
                else:
                    continue
            filtered_symptoms.append(symptom)

        return {
            **match_result,
            "role_mapping": filtered_role_mapping,
            "inferred_roots": filtered_inferred_roots,
            "symptoms": filtered_symptoms,
        }

    def _apply_default_output_site_role_ownership_to_matches(self, matches):
        if not matches:
            return matches
        return [
            self._apply_default_output_site_role_ownership(match_result)
            for match_result in matches
        ]

    @staticmethod
    def _get_optional_only_roles(rule):
        """识别仅通过 optional 边引入的角色；这些 role 被裁空时可视作未命中。"""
        optional_incident = set()
        required_incident = set()
        for edge in rule.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue
            if edge.get("optional"):
                optional_incident.update([source, target])
            else:
                required_incident.update([source, target])
        return optional_incident - required_incident

    def _get_site_chain_downstream_hop(self, parent_site, child_site):
        if not self.site_chain_index:
            return None

        parent_site = str(parent_site or "").strip()
        child_site = str(child_site or "").strip()
        if not parent_site or not child_site or parent_site == child_site:
            return None

        chain_info = self.site_chain_index.get(parent_site)
        if not chain_info:
            return None

        return chain_info.get("downstream_site_hops", {}).get(child_site)

    def _get_role_site_owner_distance(self, inst, role, site):
        """用 site_chains 判断某 role 的上游支撑节点到该站点的最短 downstream hop。"""
        if not self.site_chain_index:
            return None

        best_hop = None
        dependencies = inst.get("_dependencies", {})
        for (src_role, _dst_role), dep in dependencies.items():
            if src_role != role:
                continue
            support_nodes = dep.get("src_to_dst", {}).get(site, set())
            for support_node in support_nodes:
                hop = self._get_site_chain_downstream_hop(support_node, site)
                if hop is None:
                    continue
                if best_hop is None or hop < best_hop:
                    best_hop = hop

        return best_hop

    def _choose_site_owner_role(self, inst, site, roles, role_order):
        """同一站点命中多个互斥 role 时，优先归属到 parent->child hop 最近的一边。"""
        distance_candidates = []
        for role in roles:
            distance = self._get_role_site_owner_distance(inst, role, site)
            if distance is not None:
                distance_candidates.append((distance, role_order[role], role))

        if distance_candidates:
            return min(distance_candidates)[2]

        # 无 site_chains 或无法用 downstream_hops 判定时，保持原有遍历顺序 first-win。
        return min(roles, key=lambda role: role_order[role])

    def _apply_default_site_role_ownership(self, inst, rule, nodes_cfg):
        """按 rule.exclusive_site_roles 裁剪重复站点归属，避免环/双向边导致角色串位。"""
        roles = inst.get("roles", {})
        if len(roles) <= 1:
            return inst

        exclusive_role_groups = self._normalize_exclusive_site_role_groups(
            rule.get("exclusive_site_roles"),
            list(roles.keys()),
        )
        if not exclusive_role_groups:
            return inst

        role_order = {role: idx for idx, role in enumerate(roles.keys())}
        remove_by_role = collections.defaultdict(set)
        for exclusive_roles in exclusive_role_groups:
            site_to_roles = collections.defaultdict(list)
            for role in exclusive_roles:
                for site in roles.get(role, {}).get("nodes", {}):
                    site_to_roles[site].append(role)

            for site, site_roles in site_to_roles.items():
                if len(site_roles) <= 1:
                    continue
                owner_role = self._choose_site_owner_role(inst, site, site_roles, role_order)
                for role in site_roles:
                    if role != owner_role:
                        remove_by_role[role].add(site)

        if not remove_by_role:
            return inst

        optional_only_roles = self._get_optional_only_roles(rule)
        new_inst = dict(inst)
        new_roles = {}
        for role, role_state in roles.items():
            new_nodes = dict(role_state.get("nodes", {}))
            for site in remove_by_role.get(role, set()):
                new_nodes.pop(site, None)

            min_count = nodes_cfg.get(role, {}).get("min_count", 1)
            if len(new_nodes) < min_count:
                if not new_nodes and role in optional_only_roles:
                    continue
                return None

            new_roles[role] = {
                "nodes": new_nodes,
                "checked": role_state.get("checked", False),
            }

        new_inst["roles"] = new_roles
        return new_inst

    @staticmethod
    def _keep_symmetric_pair_candidate(curr_role, tgt_role, edge, curr_phys, cand_phys):
        source_role = edge.get("source_role")
        target_role = edge.get("target_role")

        if curr_role == source_role and tgt_role == target_role:
            source_site = curr_phys
            target_site = cand_phys
        elif curr_role == target_role and tgt_role == source_role:
            source_site = cand_phys
            target_site = curr_phys
        else:
            source_site = curr_phys
            target_site = cand_phys

        return str(source_site) < str(target_site)

    def _filter_symmetric_pair_candidates(self, candidate_hops, curr_role, tgt_role, edge, curr_phys):
        if not edge.get("dedupe_symmetric_pair") or not candidate_hops:
            return candidate_hops, 0

        filtered = {}
        removed_count = 0
        for cand_phys, hop in candidate_hops.items():
            if self._keep_symmetric_pair_candidate(curr_role, tgt_role, edge, curr_phys, cand_phys):
                filtered[cand_phys] = hop
            else:
                removed_count += 1

        return filtered, removed_count

    def __init__(
        self,
        topo_downstream_map,
        rules_config,
        site_domain_map,
        alarm_source_domain_map=None,
        aggregation_wait_sec=420,
        site_merge_helper=None,
        site_chain_index=None,
    ):
        """初始化拓扑、缓存、触发索引以及历史故障组状态。"""
        # 规则配置总表：按规则名保存匹配图、触发角色和节点约束。
        self.rules = rules_config

        # 建立正反向物理拓扑索引，供多向 BFS 搜索使用
        self.topo_down = topo_downstream_map
        self.topo_up = collections.defaultdict(list)
        for up, downs in self.topo_down.items():
            for down in downs:
                self.topo_up[down].append(up)
        self.site_chain_index = self._normalize_site_chain_index(site_chain_index)

        # 状态缓存: { node: deque([(ts, event_id, alarm_type, alarm_source, consumed_trigger_rules)]) }
        self.event_cache = collections.defaultdict(collections.deque)
        # 默认告警缓存保留时长，单位秒
        self.global_ttl = 3600
        # 电源类告警缓存单独保留 3 小时，避免长时间窗根因回看失效
        self.power_alarm_ttl = 10800

        # 站点画像信息：供节点匹配领域使用
        self.sites_domain_map = site_domain_map
        self.alarm_source_domain_map = alarm_source_domain_map or {}

        # 全局拓扑穿透缓存
        self.global_topo_cache = collections.OrderedDict()
        self.max_topo_cache_size = 10000
        # 站点级批内弱合并辅助器：统一承接 hop 合并和空间密度合并
        self.site_merge_helper = site_merge_helper
        # 最近一次收割的批内合并统计，以及累计统计
        self.last_batch_merge_stats = build_empty_merge_stats()
        self.total_batch_merge_stats = build_empty_merge_stats()
        # nearest_matching 在不带 path_requirements 时只依赖静态拓扑和站点画像，可跨批次复用
        self.global_nearest_match_cache = collections.OrderedDict()
        self.max_nearest_match_cache_size = 10000

        # 故障传播等待时间
        self.aggregation_wait_sec = aggregation_wait_sec

        # 流式时间水印
        self.current_watermark = 0.0
        # 已到达事件时间上界：TTL 清理只跟真实已进入引擎的事件时间走，不跟 live 模式下的模拟水印走
        self.latest_arrived_event_ts = 0.0

        # 延迟触发队列：记录“当前仍在等待聚合”的 trigger 起点锚点，结构为 (ts, seq)
        self.pending_triggers = {}
        # 延迟触发最小堆：按 ready_ts 排序，快速摘取已成熟的 pending trigger
        self.pending_trigger_heap = []
        # 保存某个 (node, rule) 下所有还能作为 trigger 候选的事件，结构为 (ts, event_id, seq, alarm_type)
        self.trigger_event_index = collections.defaultdict(collections.deque)
        # trigger 候选事件的全局递增序号，用于精确定位“下一条”事件。
        self._trigger_seq = 0

        # 负责历史组保留、按 eid 合并和替换落库
        self.emitted_group_store = EmittedGroupStore(self.rules, self.global_ttl)

        # 负责站点结构匹配、告警窗口校验和失败原因解释
        self.node_rule_helper = NodeRuleHelper(
            self.sites_domain_map,
            CRITICAL_ALARMS,
            lambda node: self.event_cache.get(node, []),
            self.alarm_source_domain_map,
        )

        # 每条规则的静态执行计划：提前把模式图邻接、遍历顺序和 root roles 预编译出来。
        self.rule_execution_plans = {}
        self._compile_rule_execution_plans()

        # 站点可以作为 trigger 的规则+告警组合，{node: ((rule, (alarm_type, ...)), ...)}
        self.trigger_specs_by_node = {}
        self._build_trigger_indexes()
        
        # 后台收割线程对象：live 模式下按固定周期推进 watermark
        self._harvest_thread = None
        # 后台收割线程停止信号
        self._harvest_stop_event = None
        # 后台收割线程的真实运行周期，单位秒
        self._harvest_interval_sec = None
        # 后台收割产出结果时的回调函数
        self._harvest_callback = None
        # 后台收割线程使用的当前时间函数；默认为真实时间，可由调用方注入模拟时钟
        self._harvest_now_ts_getter = None
        # Debug 观察回调：仅在调试模式下接收一次收割过程中的中间阶段结果
        self.debug_observer = None
        # Debug 事件回调：仅在调试模式下记录 event_cache 中事件被移除的原因
        self.debug_event_logger = None
        
        # 引擎主锁：保护 event_cache、pending、watermark 和历史组状态的一致性
        self._lock = threading.RLock()
        # 拓扑缓存专用锁，避免全局主锁被纯缓存读写长期占用
        self._topo_cache_lock = threading.Lock()
        # 事件快照缓存锁，保证锁外评估阶段的按需快照填充安全
        self._event_snapshot_lock = threading.Lock()

        # 分批清理过期节点状态时的游标
        self._prune_cursor = 0
        # 每轮清理最多处理的节点数，避免单次 prune 开销过大
        self._prune_batch_size = 256
        # 当 heap 脏条目过多时触发重建的倍率阈值
        self._pending_heap_rebuild_factor = 3

    def process_event(self, node, alarm_type, ts, event_id, alarm_source="", is_clear=False, collect_matches=False, register_trigger=True):
        """接收单条事件并更新内部状态。默认只更新内部状态；当 collect_matches=True 时，会在事件时间点立即收割已成熟的故障组。
        """
        with self._lock:
            # 1. 当前仍保留事件时间水印，便于离线按事件时间回放。
            self.current_watermark = max(self.current_watermark, ts)
            self.latest_arrived_event_ts = max(self.latest_arrived_event_ts, ts)

            # 2. 先清理过期缓存，再按上报/清除事件更新状态。
            q = self.event_cache[node]
            while q and (ts - q[0][0]) > self._get_event_ttl(q[0][2]):
                expired_event = q.popleft()
                self._log_debug_event_removal(node, expired_event, "ttl", current_ts=ts)
            self._prune_expired_trigger_index(node, ts)

            if is_clear:
                # 清除事件只按 event_id 删除对应实例，并联动修正 trigger / pending 状态。
                self._remove_cleared_events(node, event_id)
                affected_rule_names = self._remove_cleared_trigger_events(node, event_id)
                if affected_rule_names:
                    self._refresh_pending_triggers_for_node(
                        node,
                        affected_rule_names=affected_rule_names
                    )
            else:
                q.append((ts, event_id, alarm_type, alarm_source, frozenset()))

            # 3. 命中 trigger 的事件只负责入 pending，不在这里直接做匹配评估。
            if not is_clear and register_trigger:
                for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
                    if any(matches_expected_alarm(alarm_type, expected) for expected in expected_list):
                        trigger_key = (node, rule_name)
                        self._trigger_seq += 1
                        trigger_seq = self._trigger_seq
                        self.trigger_event_index[trigger_key].append((ts, event_id, trigger_seq, alarm_type))
                        # 如果这段时间内已经触发过，就不更新时间，以“第一声警报”为准
                        if trigger_key not in self.pending_triggers:
                            self._set_pending_trigger(trigger_key, ts, trigger_seq)

        # 离线模式通过事件触发收割
        if collect_matches:
            return self._collect_pending_matches(force=False)

        return []

    def _get_event_ttl(self, alarm_type):
        return self.power_alarm_ttl if alarm_type in POWER_ALARMS else self.global_ttl

    def _log_debug_event_removal(self, node, event, reason, **extra):
        if not self.debug_event_logger:
            return

        ts, event_id, alarm_type, alarm_source, consumed_trigger_rules = event
        payload = {
            "node": node,
            "ts": ts,
            "event_id": event_id,
            "alarm_type": alarm_type,
            "alarm_source": alarm_source,
            "consumed_trigger_rules": sorted(consumed_trigger_rules),
            "reason": reason,
        }
        payload.update(extra)
        self.debug_event_logger(payload)

    def _set_pending_trigger(self, trigger_key, first_trigger_ts, trigger_seq):
        trigger_anchor = (first_trigger_ts, trigger_seq)
        self.pending_triggers[trigger_key] = trigger_anchor
        ready_ts = first_trigger_ts + self.aggregation_wait_sec
        heapq.heappush(self.pending_trigger_heap, (ready_ts, first_trigger_ts, trigger_seq, trigger_key))
        self._maybe_rebuild_pending_heap_locked()

    def _maybe_rebuild_pending_heap_locked(self):
        if len(self.pending_trigger_heap) <= max(64, len(self.pending_triggers) * self._pending_heap_rebuild_factor):
            return

        self.pending_trigger_heap = [
            (trigger_ts + self.aggregation_wait_sec, trigger_ts, trigger_seq, trigger_key)
            for trigger_key, (trigger_ts, trigger_seq) in self.pending_triggers.items()
        ]
        heapq.heapify(self.pending_trigger_heap)

    def _collect_trigger_expected_list(self, trigger_node_domain, trigger_config):
        expected_list = []
        node_type = trigger_config.get("type", "primitive")
        if node_type == "primitive":
            if not self.node_rule_helper.matches_node_structure(trigger_node_domain, trigger_config):
                return expected_list
            expected = self.node_rule_helper.resolve_expected_alarms(trigger_node_domain, trigger_config)
            if expected not in (None, "NONE"):
                expected_list.append(expected)
            return expected_list

        if node_type == "compound":
            for pattern in trigger_config.get("patterns", []):
                if not self.node_rule_helper.matches_node_structure(trigger_node_domain, pattern):
                    continue
                expected = self.node_rule_helper.resolve_expected_alarms(trigger_node_domain, pattern)
                if expected not in (None, "NONE"):
                    expected_list.append(expected)
        return expected_list

    def _build_trigger_indexes(self):
        """预编译 node -> trigger 规则索引，减少运行时全规则扫描。"""
        trigger_specs_by_node = {}

        for node, node_domain in self.sites_domain_map.items():
            specs = []
            for rule_name, rule in self.rules.items():
                trigger_role = rule["trigger_role"]
                trigger_config = rule["nodes"][trigger_role]
                expected_list = self._collect_trigger_expected_list(node_domain, trigger_config)
                if expected_list:
                    specs.append((rule_name, tuple(expected_list)))

            if specs:
                trigger_specs_by_node[node] = tuple(specs)

        self.trigger_specs_by_node = trigger_specs_by_node

    def _snapshot_event_cache_subset_locked(self, seed_nodes):
        event_cache_snapshot = {}
        for node in seed_nodes:
            events = self.event_cache.get(node)
            if events:
                event_cache_snapshot[node] = tuple(events)
        return event_cache_snapshot

    def _build_snapshot_helper(self, event_cache_snapshot):
        def get_events(node, cache=event_cache_snapshot):
            if node in cache:
                return cache[node]
            with self._event_snapshot_lock:
                if node in cache:
                    return cache[node]
                with self._lock:
                    events = tuple(self.event_cache.get(node, ()))
                cache[node] = events
                return events

        return NodeRuleHelper(
            self.sites_domain_map,
            CRITICAL_ALARMS,
            get_events,
            self.alarm_source_domain_map,
        )

    def _collect_mature_pending_locked(self, force=False):
        """在锁内摘取当前已成熟的 pending trigger。"""
        mature_items = []
        effective_harvest_ts = self.latest_arrived_event_ts if self.latest_arrived_event_ts > 0 else self.current_watermark

        if force:
            for trigger_key, trigger_anchor in list(self.pending_triggers.items()):
                self.pending_triggers.pop(trigger_key, None)
                mature_items.append((trigger_key, trigger_anchor))
            return mature_items

        while self.pending_trigger_heap:
            ready_ts, first_trigger_ts, trigger_seq, trigger_key = self.pending_trigger_heap[0]
            if ready_ts > effective_harvest_ts:
                break

            heapq.heappop(self.pending_trigger_heap)
            current_pending_anchor = self.pending_triggers.get(trigger_key)
            if current_pending_anchor != (first_trigger_ts, trigger_seq):
                continue

            self.pending_triggers.pop(trigger_key, None)
            self._prune_trigger_index_before(trigger_key, trigger_seq)
            mature_items.append((trigger_key, (first_trigger_ts, trigger_seq)))

        return mature_items

    def _collect_pending_matches(self, force=False):
        """收割已成熟的 pending trigger，并执行对应规则评估。"""
        with self._lock:
            mature_items = self._collect_mature_pending_locked(force=force)
            if not mature_items:
                return []
            seed_nodes = {trigger_key[0] for trigger_key, _ in mature_items}
            event_cache_snapshot = self._snapshot_event_cache_subset_locked(seed_nodes)

        helper = self._build_snapshot_helper(event_cache_snapshot)
        batch_eval_caches = self._create_eval_caches()
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
                    return_debug_trace=True
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

        merged_matches, batch_merge_stats = merge_match_batch(
            raw_matches,
            site_merge_helper=self.site_merge_helper,
            return_stats=True,
        )
        expanded_matches, expanded_merge_stats = self._expand_matches_with_pending_context(
            merged_matches,
            helper,
            eval_caches=batch_eval_caches,
        )
        collection_merge_stats = add_merge_stats(batch_merge_stats, expanded_merge_stats)
        with self._lock:
            self._record_batch_merge_stats_locked(collection_merge_stats)
            self._prune_expired_state_locked(self.latest_arrived_event_ts)
            current_watermark = self.current_watermark
            effective_harvest_ts = self.latest_arrived_event_ts if self.latest_arrived_event_ts > 0 else self.current_watermark
            if self.debug_observer:
                finalized_matches, finalize_profiles = self._finalize_matches_with_history(
                    expanded_matches,
                    return_debug_trace=True
                )
            else:
                finalized_matches = self._finalize_matches_with_history(expanded_matches)
                finalize_profiles = []

        owned_matches = self._apply_default_output_site_role_ownership_to_matches(finalized_matches)
        output_matches = self._apply_output_visibility_filters_to_matches(owned_matches)

        if self.debug_observer:
            self.debug_observer({
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

        return output_matches

    def advance_watermark(self, now_ts=None):
        """通过定时任务推进水印，并收割已成熟的故障组。"""
        with self._lock:
            if now_ts is None:
                now_ts = time.time()

            self.current_watermark = max(self.current_watermark, now_ts)
        return self._collect_pending_matches(force=False)

    def start_periodic_harvest(self, interval_sec=10, on_matches=None, now_ts_getter=None):
        """启动后台定时收割线程。

        on_matches 是一个可选 callback，后台线程每次收割出故障组后会把整批结果交给它。
        这里先不做线程安全承诺，后续如果线上启用双线程，需要再配合加锁方案一起看。
        """
        with self._lock:
            if interval_sec <= 0:
                raise ValueError("interval_sec must be > 0")
            if self._harvest_thread and self._harvest_thread.is_alive():
                raise RuntimeError("periodic harvest thread is already running")

            self._harvest_interval_sec = interval_sec
            self._harvest_callback = on_matches
            self._harvest_now_ts_getter = now_ts_getter
            self._harvest_stop_event = threading.Event()
            self._harvest_thread = threading.Thread(
                target=self._periodic_harvest_loop,
                name="TemporalGraphEngineHarvest",
                daemon=True
            )
            self._harvest_thread.start()

    def stop_periodic_harvest(self, timeout=None):
        """停止后台定时收割线程。"""
        with self._lock:
            thread = self._harvest_thread
            stop_event = self._harvest_stop_event

        if not thread:
            return

        if stop_event:
            stop_event.set()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError("periodic harvest thread did not stop within the timeout")

        with self._lock:
            if self._harvest_thread is thread and not self._harvest_thread.is_alive():
                self._harvest_thread = None
                self._harvest_stop_event = None
                self._harvest_interval_sec = None
                self._harvest_callback = None
                self._harvest_now_ts_getter = None

    def _periodic_harvest_loop(self):
        while self._harvest_stop_event and not self._harvest_stop_event.is_set():
            try:
                # 定时线程按调用方给定的时间轴推进 watermark；默认退化到真实时间。
                now_ts_getter = self._harvest_now_ts_getter or time.time
                matches = self.advance_watermark(now_ts_getter())
                if matches and self._harvest_callback:
                    self._harvest_callback(matches)
            except Exception:
                logger.exception("Periodic harvest loop failed")
            self._harvest_stop_event.wait(self._harvest_interval_sec)

    def _remove_cleared_events(self, node, event_id):
        """按 event_id 从节点事件缓存中移除已清除的告警实例。"""
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

    def _prune_node_alarm_history_before(self, node, alarm_type, cutoff_by_rule):
        """把某节点同告警名下不晚于各 rule cutoff 的事件标记为已被对应 rule 消费并移出对应 trigger 候选。"""
        q = self.event_cache.get(node)
        if not q:
            return

        removed_event_ids_by_rule = collections.defaultdict(set)
        kept = collections.deque()
        for cached_ts, cached_eid, cached_alarm_type, cached_alarm_source, consumed_trigger_rules in q:
            if cached_alarm_type == alarm_type:
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
            affected_rule_names=removed_event_ids_by_rule.keys()
        )

    def _prune_consumed_alarm_history(self, matches):
        """在本轮定时收割结束时，只回收命中 trigger_role 的节点告警历史。"""
        prune_points = {}
        for match in matches:
            merged_rules = match.get("merged_rules", [match.get("rule")])
            rule_to_trigger_role = {
                rule_name: self.rules[rule_name]["trigger_role"]
                for rule_name in merged_rules
                if rule_name in self.rules and self.rules[rule_name].get("trigger_role")
            }
            for symptom in match.get("symptoms", []):
                matched_role = symptom.get("matched_role")
                matched_rule_names = {
                    rule_name
                    for rule_name, trigger_role in rule_to_trigger_role.items()
                    if matched_role == trigger_role
                }
                if not matched_rule_names:
                    continue
                node = symptom.get("node")
                alarm_type = symptom.get("alarm")
                ts = symptom.get("ts")
                if node in (None, "") or alarm_type in (None, "") or ts is None:
                    continue
                key = (node, alarm_type)
                entry = prune_points.setdefault(key, {})
                for rule_name in matched_rule_names:
                    entry[rule_name] = max(entry.get(rule_name, float("-inf")), ts)

        for (node, alarm_type), cutoff_by_rule in prune_points.items():
            self._prune_node_alarm_history_before(
                node,
                alarm_type,
                cutoff_by_rule,
            )

    def _prune_expired_trigger_index(self, node, current_ts):
        """清理某个节点 trigger 索引中超出 TTL 的旧事件。"""
        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            while trigger_events and (current_ts - trigger_events[0][0]) > self._get_event_ttl(trigger_events[0][3]):
                trigger_events.popleft()

            if not trigger_events:
                self.trigger_event_index.pop(trigger_key, None)
                self.pending_triggers.pop(trigger_key, None)
        self._maybe_rebuild_pending_heap_locked()

    def _prune_trigger_index_before(self, trigger_key, cutoff_seq):
        """删除某个 trigger_key 下序号不大于 cutoff_seq 的已消费 trigger 事件。"""
        trigger_events = self.trigger_event_index.get(trigger_key)
        if not trigger_events:
            return

        while trigger_events and trigger_events[0][2] <= cutoff_seq:
            trigger_events.popleft()

        if not trigger_events:
            self.trigger_event_index.pop(trigger_key, None)

    def _remove_cleared_trigger_events(self, node, event_id):
        """按 event_id 从 trigger 索引中移除已清除的触发事件。

        返回值是“当前 pending anchor 也被清掉”的 rule 名集合。只有这些 rule
        需要把 pending 起点推进到下一条 trigger；如果删掉的只是后续候选，
        pending 应保持不变。
        """
        if not event_id:
            return set()

        affected_rule_names = set()
        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            current_pending_anchor = self.pending_triggers.get(trigger_key)
            kept = collections.deque()
            for event_ts, indexed_event_id, indexed_seq, indexed_alarm_type in trigger_events:
                if indexed_event_id == event_id:
                    if current_pending_anchor == (event_ts, indexed_seq):
                        affected_rule_names.add(rule_name)
                    continue
                kept.append((event_ts, indexed_event_id, indexed_seq, indexed_alarm_type))

            if kept:
                self.trigger_event_index[trigger_key] = kept
            else:
                self.trigger_event_index.pop(trigger_key, None)

        return affected_rule_names

    def _refresh_pending_triggers_for_node(self, node, affected_rule_names=None):
        """在 trigger 候选被删除后，重新校正该节点对应 rule 的 pending 起点。"""
        if affected_rule_names is None:
            rule_names = [rule_name for rule_name, _ in self.trigger_specs_by_node.get(node, ())]
        else:
            rule_names = [rule_name for rule_name in affected_rule_names if rule_name]

        for rule_name in rule_names:
            trigger_key = (node, rule_name)
            if trigger_key not in self.pending_triggers:
                continue

            original_pending_anchor = self.pending_triggers[trigger_key]
            # 清除后只允许把 pending 起点推进到“原 trigger 之后”的下一条，避免回退到同一时间更早的故障上下文。
            next_trigger_anchor = self._find_next_trigger_anchor(node, rule_name, original_pending_anchor)
            if next_trigger_anchor is None:
                del self.pending_triggers[trigger_key]
            else:
                next_trigger_ts, next_trigger_seq = next_trigger_anchor
                self._set_pending_trigger(trigger_key, next_trigger_ts, next_trigger_seq)
        self._maybe_rebuild_pending_heap_locked()

    def _find_next_trigger_anchor(self, node, rule_name, lower_bound_anchor):
        """找到严格晚于 lower_bound_anchor 的下一条可用 trigger。"""
        trigger_events = self.trigger_event_index.get((node, rule_name))
        if not trigger_events:
            return None

        _lower_bound_ts, lower_bound_seq = lower_bound_anchor
        for event_ts, _event_id, event_seq, _alarm_type in trigger_events:
            if event_seq > lower_bound_seq:
                return event_ts, event_seq
        return None

    def flush_pending(self):
        """流处理结束时，强制执行所有还在等待的触发器"""
        return self._collect_pending_matches(force=True)

    def _prune_expired_state_locked(self, current_ts):
        """分批清理长期未再触达节点的过期缓存，避免状态无限滞留。"""
        nodes = list(self.event_cache.keys())
        if not nodes:
            return

        total_nodes = len(nodes)
        batch_size = min(self._prune_batch_size, total_nodes)
        start_idx = self._prune_cursor % total_nodes

        for offset in range(batch_size):
            node = nodes[(start_idx + offset) % total_nodes]
            q = self.event_cache.get(node)
            if not q:
                continue

            while q and (current_ts - q[0][0]) > self._get_event_ttl(q[0][2]):
                expired_event = q.popleft()
                self._log_debug_event_removal(node, expired_event, "ttl", current_ts=current_ts)

            if not q:
                self.event_cache.pop(node, None)

            self._prune_expired_trigger_index(node, current_ts)

        self._prune_cursor = (start_idx + batch_size) % max(total_nodes, 1)

    def _finalize_matches_with_history(self, matches, return_debug_trace=False):
        """把当前批次结果与历史组做最终合并并落库。"""
        finalized = []
        finalize_profiles = []
        current_time = self.latest_arrived_event_ts if self.latest_arrived_event_ts > 0 else (
            self.current_watermark if hasattr(self, 'current_watermark') else time.time()
        )
        self.emitted_group_store.prune_expired(current_time)

        for match_result in matches:
            group_anchor_ts = self.emitted_group_store.get_group_anchor_ts(match_result, current_time)
            original_uuid = match_result.get("uuid", "")
            original_rule = match_result.get("rule", "")
            match_result, merged_group_indexes, related_group_uuids, should_emit, emit_reason = self.emitted_group_store.merge_with_related(match_result)
            match_result = self._apply_default_output_site_role_ownership(match_result)
            if not should_emit:
                if return_debug_trace:
                    finalize_profiles.append({
                        "uuid": original_uuid,
                        "rule": original_rule,
                        "action": "suppressed",
                        "reason": emit_reason,
                        "related_group_uuids": sorted(related_group_uuids),
                        "merged_group_count": len(merged_group_indexes),
                    })
                self.emitted_group_store.extend_related_expire_ts(
                    merged_group_indexes,
                    match_result,
                    group_anchor_ts
                )
                continue
            if related_group_uuids:
                existing_uuids = set(match_result.get("related_group_uuids", []))
                match_result["related_group_uuids"] = sorted(existing_uuids | set(related_group_uuids))
            self.emitted_group_store.replace_and_store(
                merged_group_indexes,
                group_anchor_ts,
                match_result
            )
            finalized.append(match_result)
            if return_debug_trace:
                finalize_profiles.append({
                    "uuid": original_uuid,
                    "rule": original_rule,
                    "action": "emitted",
                    "reason": emit_reason,
                    "related_group_uuids": sorted(related_group_uuids),
                    "merged_group_count": len(merged_group_indexes),
                })

        self._prune_consumed_alarm_history(finalized)
        if return_debug_trace:
            return finalized, finalize_profiles
        return finalized

    def _get_trigger_roles_for_match(self, match_result):
        merged_rules = match_result.get("merged_rules", [match_result.get("rule")])
        return {
            self.rules[rule_name]["trigger_role"]
            for rule_name in merged_rules
            if rule_name in self.rules and self.rules[rule_name].get("trigger_role")
        }

    def _get_non_trigger_nodes_for_match(self, match_result):
        trigger_roles = self._get_trigger_roles_for_match(match_result)
        non_trigger_nodes = set()

        for role, nodes in match_result.get("role_mapping", {}).items():
            if role in trigger_roles:
                continue
            for node in nodes:
                if node not in (None, ""):
                    non_trigger_nodes.add(node)

        return non_trigger_nodes

    def _expand_matches_with_pending_context(self, matches, node_rule_helper, eval_caches=None):
        """只读扩充当前批次：汇总整批非 trigger 节点上的 pending，再和当前批统一做一次批内合并。"""
        if not matches:
            return matches, build_empty_merge_stats()

        non_trigger_nodes = set()
        for match_result in matches:
            non_trigger_nodes.update(self._get_non_trigger_nodes_for_match(match_result))

        if not non_trigger_nodes:
            return matches, build_empty_merge_stats()

        with self._lock:
            pending_candidates = [
                ((node, rule_name), trigger_anchor)
                for (node, rule_name), trigger_anchor in self.pending_triggers.items()
                if node in non_trigger_nodes
            ]

        if not pending_candidates:
            return matches, build_empty_merge_stats()

        extra_matches = []
        for (node, rule_name), trigger_anchor in pending_candidates:
            rule = self.rules.get(rule_name)
            if not rule:
                continue
            trigger_ts, _trigger_seq = trigger_anchor
            results = self._evaluate_rule(
                rule_name,
                rule,
                node,
                trigger_ts,
                node_rule_helper=node_rule_helper,
                eval_caches=eval_caches,
            )
            if results:
                extra_matches.extend(results)

        if not extra_matches:
            return matches, build_empty_merge_stats()

        return merge_match_batch(
            list(matches) + extra_matches,
            site_merge_helper=self.site_merge_helper,
            return_stats=True,
        )

    def _record_batch_merge_stats_locked(self, merge_stats):
        self.last_batch_merge_stats = build_empty_merge_stats()
        self.last_batch_merge_stats.update(merge_stats or {})
        self.total_batch_merge_stats = add_merge_stats(
            self.total_batch_merge_stats,
            self.last_batch_merge_stats,
        )

    def get_batch_merge_stats_snapshot(self):
        with self._lock:
            return {
                "last_batch": dict(self.last_batch_merge_stats),
                "total": dict(self.total_batch_merge_stats),
            }

    def _evaluate_rule(
        self,
        rule_name,
        rule,
        trigger_node,
        trigger_ts,
        node_rule_helper=None,
        eval_caches=None,
        return_debug_trace=False,
    ):
        """
        全向动态图调度器 (State-Forking Matcher)：
        支持平行宇宙分叉、严格结构匹配、局部性能缓存。
        """
        helper = node_rule_helper or self.node_rule_helper
        nodes_cfg = rule.get("nodes", {})
        debug_trace = None
        if return_debug_trace:
            debug_trace = {
                "rule": rule_name,
                "trigger_node": trigger_node,
                "trigger_ts": trigger_ts,
                "trigger_role": None,
                "trigger_validation": None,
                "edges": [],
                "raw_match_count": 0,
            }

        # 1. 读取规则的静态执行计划
        plan = self.rule_execution_plans.get(rule_name)
        if plan is None:
            plan = self._compile_rule_execution_plan(rule)
            self.rule_execution_plans[rule_name] = plan

        # 2. 取出本次匹配需要的静态遍历信息
        trigger_role = plan["trigger_role"]
        edges_to_explore = plan["edges_to_explore"]
        root_roles = plan["root_roles"]
        if debug_trace is not None:
            debug_trace["trigger_role"] = trigger_role

        caches = eval_caches or self._create_eval_caches()
        validation_cache = caches["validation_cache"]
        traversal_cache = caches["traversal_cache"]
        path_validation_cache = caches["path_validation_cache"]
        structure_match_cache = caches["structure_match_cache"]
        filtered_neighbor_cache = caches["filtered_neighbor_cache"]

        # 3. 校验触发节点自身
        trigger_node_domain = self.sites_domain_map.get(trigger_node, {})
        trigger_validation_cache_key = (
            trigger_node,
            trigger_role,
            trigger_ts,
            0,
            rule_name,
        )
        if trigger_validation_cache_key in validation_cache:
            is_trig_valid, trig_evts = validation_cache[trigger_validation_cache_key]
        else:
            is_trig_valid, trig_evts = helper.validate_node(
                trigger_node,
                trigger_node_domain,
                nodes_cfg[trigger_role],
                trigger_ts,
                edge_window=0,
                exclude_consumed_trigger_rule=rule_name
            )
            validation_cache[trigger_validation_cache_key] = (is_trig_valid, trig_evts)
        if debug_trace is not None:
            debug_trace["trigger_validation"] = helper.explain_node_validation(
                trigger_node,
                trigger_node_domain,
                nodes_cfg[trigger_role],
                trigger_ts,
                edge_window=0,
                exclude_consumed_trigger_rule=rule_name
            )
        if not is_trig_valid:
            if debug_trace is not None:
                debug_trace["final_reason"] = (
                    debug_trace["trigger_validation"].get("reason", "trigger 节点未通过校验")
                )
                return [], debug_trace
            return []

        # 4. 初始化独立子图实例池与备忘录缓存
        initial_inst = {
            "roles": {
                trigger_role: {'nodes': {trigger_node: trig_evts}, 'checked': False}
            },
            "_dependencies": {}
        }
        instances = [initial_inst]

        # 5. 核心引擎：逐边推演，触发 分叉 (Fork) 或 聚合 (Aggregate)
        for curr_role, tgt_role, edge in edges_to_explore:
            next_instances = []
            edge_trace = None
            if debug_trace is not None:
                edge_trace = {
                    "from_role": curr_role,
                    "to_role": tgt_role,
                    "instances_in": len(instances),
                    "instances_out": 0,
                    "failures": [],
                }

            for inst in instances:
                inst_roles = inst["roles"]

                if curr_role not in inst_roles:
                    if edge_trace is not None:
                        edge_trace["failures"].append(
                            f"实例缺少源 role {curr_role}，无法继续扩展到 {tgt_role}"
                        )
                    next_instances.append(inst)
                    continue

                if tgt_role in inst_roles and inst_roles[tgt_role]["checked"]:
                    next_instances.append(inst)
                    continue

                curr_phys_dict = inst_roles[curr_role]
                tgt_cfg = nodes_cfg[tgt_role]
                min_c = tgt_cfg.get("min_count", 1)
                node_type = tgt_cfg.get("type", "primitive")
                match_mode = tgt_cfg.get("match", "ANY")

                valid_targets = {}
                surviving_curr_phys = {}
                curr_support_targets = {}
                branch_survived = False
                branch_failure_reasons = []

                for curr_phys, curr_evts in curr_phys_dict['nodes'].items():
                    ref_ts = curr_evts[0]["ts"] if curr_evts else trigger_ts
                    selector = edge.get("candidate_selector") or {}
                    selector_mode = selector.get("mode", "default")
                    if selector_mode == "nearest_matching":
                        candidate_hops, had_topology_candidate = self._traverse_graph_nearest_matching(
                            start_node=curr_phys,
                            direction=edge["traverse_dir"],
                            target_node_config=tgt_cfg,
                            max_hops=edge["hops"],
                            reference_ts=ref_ts,
                            edge_window=edge["win"],
                            path_requirements=edge.get("path_requirements"),
                            node_rule_helper=helper,
                            traversal_cache=traversal_cache,
                            path_validation_cache=path_validation_cache,
                            structure_match_cache=structure_match_cache,
                            filtered_neighbor_cache=filtered_neighbor_cache,
                        )
                        candidate_hops, symmetric_deduped_count = self._filter_symmetric_pair_candidates(
                            candidate_hops,
                            curr_role,
                            tgt_role,
                            edge,
                            curr_phys,
                        )
                        raw_candidates = sorted(candidate_hops.keys(), key=lambda n: (candidate_hops[n], str(n)))
                        candidates = list(raw_candidates)
                        if edge_trace is not None and symmetric_deduped_count:
                            edge_trace["failures"].append(
                                f"{curr_role}:{curr_phys} 的 {tgt_role} 候选因 symmetric pair 去重过滤 {symmetric_deduped_count} 个"
                            )
                        if edge_trace is not None and not raw_candidates:
                            if had_topology_candidate:
                                branch_failure_reasons.append(
                                    f"{curr_role}:{curr_phys} 的 {tgt_role} 候选在 selector(mode={selector_mode}) 后为空，"
                                    "原始拓扑候选存在，但没有结构命中"
                                )
                            else:
                                branch_failure_reasons.append(
                                    f"{curr_role}:{curr_phys} 在拓扑方向 {edge['traverse_dir']} 上找不到任何 {tgt_role} 候选节点"
                                )
                            continue
                    else:
                        candidate_hops = self._traverse_graph(
                            start_node=curr_phys,
                            direction=edge["traverse_dir"],
                            max_hops=edge["hops"],
                            reference_ts=ref_ts,
                            edge_window=edge["win"],
                            path_requirements=edge.get("path_requirements"),
                            node_rule_helper=helper,
                            traversal_cache=traversal_cache,
                            path_validation_cache=path_validation_cache,
                            filtered_neighbor_cache=filtered_neighbor_cache,
                        )
                        candidate_hops, symmetric_deduped_count = self._filter_symmetric_pair_candidates(
                            candidate_hops,
                            curr_role,
                            tgt_role,
                            edge,
                            curr_phys,
                        )
                        raw_candidates = sorted(candidate_hops.keys(), key=lambda n: (candidate_hops[n], str(n)))
                        candidates = list(raw_candidates)
                        if edge_trace is not None and symmetric_deduped_count:
                            edge_trace["failures"].append(
                                f"{curr_role}:{curr_phys} 的 {tgt_role} 候选因 symmetric pair 去重过滤 {symmetric_deduped_count} 个"
                            )
                        # 先跑拓扑可达，再按 rule 里的 selector 收窄候选集
                        candidates = helper.select_candidates_by_rule(
                            candidates, candidate_hops, tgt_cfg, edge.get("candidate_selector")
                        )
                        if edge_trace is not None and not raw_candidates:
                            branch_failure_reasons.append(
                                f"{curr_role}:{curr_phys} 在拓扑方向 {edge['traverse_dir']} 上找不到任何 {tgt_role} 候选节点"
                            )
                            continue
                        if edge_trace is not None and raw_candidates and not candidates:
                            branch_failure_reasons.append(
                                f"{curr_role}:{curr_phys} 的 {tgt_role} 候选在 selector(mode={selector_mode}) 后为空，"
                                f"原始候选={raw_candidates[:8]}"
                            )
                            continue

                    curr_valid_targets = {}
                    all_passed = True
                    window_cache_key = self._make_edge_window_cache_key(edge["win"])
                    candidate_failure_details = []

                    # 如果 candidates 为空，天然跳过内层循环，该源节点被淘汰
                    for cand_phys in candidates:
                        # 校验结果依赖候选节点、目标角色以及参考时间窗口
                        cache_key = (cand_phys, tgt_role, ref_ts, window_cache_key)
                        if cache_key in validation_cache:
                            is_valid, evts = validation_cache[cache_key]
                        else:
                            cand_phys_domain = self.sites_domain_map.get(cand_phys, {})
                            is_valid, evts = helper.validate_node(
                                cand_phys,
                                cand_phys_domain,
                                tgt_cfg,
                                ref_ts,
                                edge["win"],
                                exclude_consumed_trigger_rule=(rule_name if tgt_role == trigger_role else None)
                            )
                            validation_cache[cache_key] = (is_valid, evts)

                        if is_valid:
                            curr_valid_targets[cand_phys] = evts
                        else:
                            if edge_trace is not None:
                                explain = helper.explain_node_validation(
                                    cand_phys,
                                    self.sites_domain_map.get(cand_phys, {}),
                                    tgt_cfg,
                                    ref_ts,
                                    edge["win"],
                                    exclude_consumed_trigger_rule=(rule_name if tgt_role == trigger_role else None)
                                )
                                candidate_failure_details.append(
                                    f"{cand_phys}: {explain.get('reason', '节点校验失败')}"
                                )
                            if match_mode == "ALL":
                                all_passed = False
                                break

                    if match_mode == "ALL" and not all_passed:
                        if edge_trace is not None:
                            detail = candidate_failure_details[0] if candidate_failure_details else "存在候选节点未通过 ALL 校验"
                            branch_failure_reasons.append(
                                f"{curr_role}:{curr_phys} 在 ALL 模式下失败，{detail}"
                            )
                        continue

                    if curr_valid_targets:
                        branch_survived = True
                        surviving_curr_phys[curr_phys] = curr_evts
                        curr_support_targets[curr_phys] = set(curr_valid_targets)
                        for key, value in curr_valid_targets.items():
                            valid_targets[key] = value
                    elif edge_trace is not None:
                        if candidate_failure_details:
                            branch_failure_reasons.append(
                                f"{curr_role}:{curr_phys} 没有满足 {tgt_role} 的节点，"
                                f"失败原因: {candidate_failure_details[:3]}"
                            )
                        else:
                            branch_failure_reasons.append(
                                f"{curr_role}:{curr_phys} 没有满足 {tgt_role} 的节点"
                            )

                if not branch_survived:
                    if edge_trace is not None and branch_failure_reasons:
                        edge_trace["failures"].extend(branch_failure_reasons[:6])
                    if edge.get("optional"):
                        if edge_trace is not None:
                            edge_trace["failures"].append(
                                f"可选边 {curr_role}->{tgt_role} 未命中，保留当前实例"
                            )
                        next_instances.append(inst)
                        continue
                    continue

                # 回溯检查数量
                curr_cfg = nodes_cfg[curr_role]
                if inst_roles[curr_role]["checked"] and len(surviving_curr_phys) < curr_cfg.get("min_count", 1):
                    if edge_trace is not None:
                        edge_trace["failures"].append(
                            f"{curr_role} 回溯后仅剩 {len(surviving_curr_phys)} 个节点，"
                            f"低于 min_count={curr_cfg.get('min_count', 1)}"
                        )
                    continue

                existing_targets = inst_roles.get(tgt_role, {}).get('nodes', {})
                merged_targets = {**existing_targets, **valid_targets}

                # 状态分叉 vs 聚合
                if node_type == 'primitive' and not existing_targets:
                    for t_node, t_evts in valid_targets.items():
                        new_inst = clone_instance_with_updates(
                            inst, curr_role, surviving_curr_phys, tgt_role, {t_node: t_evts}
                        )
                        self._record_instance_dependency(
                            new_inst,
                            curr_role,
                            tgt_role,
                            {
                                curr_node: ({t_node} if t_node in target_nodes else set())
                                for curr_node, target_nodes in curr_support_targets.items()
                            }
                        )
                        stabilized_inst = self._stabilize_instance_dependencies(new_inst, nodes_cfg)
                        if stabilized_inst is not None:
                            next_instances.append(stabilized_inst)
                else:
                    if len(merged_targets) < min_c:
                        if edge_trace is not None:
                            edge_trace["failures"].append(
                                f"{tgt_role} 合并后仅有 {len(merged_targets)} 个节点，低于 min_count={min_c}"
                            )
                        continue

                    new_inst = clone_instance_with_updates(
                        inst, curr_role, surviving_curr_phys, tgt_role, merged_targets
                    )
                    self._record_instance_dependency(new_inst, curr_role, tgt_role, curr_support_targets)
                    stabilized_inst = self._stabilize_instance_dependencies(new_inst, nodes_cfg)
                    if stabilized_inst is not None:
                        next_instances.append(stabilized_inst)

            instances = next_instances
            if edge_trace is not None:
                edge_trace["instances_out"] = len(instances)
                edge_trace["failures"] = edge_trace["failures"][:8]
                debug_trace["edges"].append(edge_trace)

            if not instances:
                if debug_trace is not None:
                    if edge_trace and edge_trace["failures"]:
                        debug_trace["final_reason"] = (
                            f"在边 {curr_role} -> {tgt_role} 上全部分支失效；"
                            f"主要原因: {edge_trace['failures'][:3]}"
                        )
                        return [], debug_trace
                    debug_trace["final_reason"] = f"在边 {curr_role} -> {tgt_role} 上全部分支失效"
                    return [], debug_trace
                return []

        # 6. 提取结果：把一次结构匹配结果整理成候选故障组。
        results = []

        for inst in instances:
            stabilized_inst = self._stabilize_instance_dependencies(inst, nodes_cfg)
            if stabilized_inst is None:
                continue
            ownership_inst = self._apply_default_site_role_ownership(stabilized_inst, rule, nodes_cfg)
            if ownership_inst is None:
                continue
            stabilized_inst = self._stabilize_instance_dependencies(ownership_inst, nodes_cfg)
            if stabilized_inst is None:
                continue
            inst = stabilized_inst
            inst_roles = inst["roles"]
            # 提取物理根因节点
            inferred_roots = {}
            for r_role in root_roles:
                nodes = list(inst_roles.get(r_role, {}).get('nodes', {}).keys())
                inferred_roots[r_role] = nodes

            symp_dict = {}
            role_mapping = {}

            for role, role_state in inst_roles.items():
                nodes_dict = role_state['nodes']
                valid_phys_nodes = []

                for phys_node, evts in nodes_dict.items():
                    valid_phys_nodes.append(phys_node)

                    for ev in evts:
                        # 给每个告警事件注入它所匹配的逻辑角色！
                        ev_enriched = dict(ev)  # 浅拷贝，防止污染原始缓存
                        ev_enriched["matched_role"] = role
                        ev_enriched["time_str"] = datetime.fromtimestamp(ev["ts"]).strftime('%Y-%m-%d %H:%M:%S')
                        symp_dict[ev["eid"]] = ev_enriched

                if valid_phys_nodes:
                    role_mapping[role] = valid_phys_nodes

            match_result = {
                "uuid": str(uuid.uuid4()),
                "rule": rule_name,
                "merged_rules": [rule_name],
                "inferred_roots": inferred_roots,
                "role_mapping": role_mapping,
                "symptoms": list(symp_dict.values()),
                "_expire_ts_hint": (
                    min((symptom["ts"] for symptom in symp_dict.values() if "ts" in symptom), default=trigger_ts)
                    + rule.get("max_stay_time_sec", self.global_ttl)
                )
            }
            is_valid_result, result_failure_reason = self._validate_result_constraints(rule, match_result)
            if not is_valid_result:
                if debug_trace is not None and result_failure_reason:
                    debug_trace.setdefault("result_constraint_failures", []).append(result_failure_reason)
                continue
            results.append(match_result)
        if debug_trace is not None:
            debug_trace["raw_match_count"] = len(results)
            if not results and "final_reason" not in debug_trace:
                result_constraint_failures = debug_trace.get("result_constraint_failures", [])
                if result_constraint_failures:
                    debug_trace["final_reason"] = (
                        "规则评估完成，但候选组被后置约束过滤；"
                        f"主要原因: {result_constraint_failures[:3]}"
                    )
                else:
                    debug_trace["final_reason"] = "规则评估完成，但未产出原始候选组"
            return results, debug_trace
        return results

    def _validate_result_constraints(self, rule, match_result):
        """对已成型的候选故障组做规则级后置约束校验。"""
        result_constraints = rule.get("result_constraints") or {}
        if not result_constraints:
            return True, None

        role_alarm_requirements_any = result_constraints.get("role_alarm_requirements_any", [])
        for requirement in role_alarm_requirements_any:
            roles = {
                str(role).strip()
                for role in requirement.get("roles", [])
                if str(role).strip()
            }
            alarms = {
                str(alarm).strip()
                for alarm in requirement.get("alarms", [])
                if str(alarm).strip()
            }
            min_roles = max(1, int(requirement.get("min_roles", 1) or 1))
            if not roles or not alarms:
                continue

            matched_roles = {
                symptom.get("matched_role")
                for symptom in match_result.get("symptoms", [])
                if symptom.get("matched_role") in roles
                and str(symptom.get("alarm", "")).strip() in alarms
            }
            if len(matched_roles) < min_roles:
                return (
                    False,
                    (
                        f"后置约束失败：角色 {sorted(roles)} 中至少 {min_roles} 个需要命中告警 "
                        f"{sorted(alarms)}，实际命中角色={sorted(role for role in matched_roles if role)}"
                    ),
                )

        role_alarm_or_presence_any = result_constraints.get("role_alarm_or_presence_any", [])
        for requirement in role_alarm_or_presence_any:
            alarm_roles = {
                str(role).strip()
                for role in requirement.get("alarm_roles", [])
                if str(role).strip()
            }
            alarms = {
                str(alarm).strip()
                for alarm in requirement.get("alarms", [])
                if str(alarm).strip()
            }
            presence_roles = {
                str(role).strip()
                for role in requirement.get("presence_roles", [])
                if str(role).strip()
            }
            min_matches = max(1, int(requirement.get("min_matches", 1) or 1))

            matched_alarm_roles = {
                symptom.get("matched_role")
                for symptom in match_result.get("symptoms", [])
                if symptom.get("matched_role") in alarm_roles
                and str(symptom.get("alarm", "")).strip() in alarms
            }
            role_mapping = match_result.get("role_mapping", {})
            matched_presence_roles = {
                role
                for role in presence_roles
                if role_mapping.get(role)
            }
            matched_items = matched_alarm_roles | matched_presence_roles
            if len(matched_items) < min_matches:
                return (
                    False,
                    (
                        "后置约束失败：需要满足至少 "
                        f"{min_matches} 个条件，告警角色={sorted(alarm_roles)} 命中告警={sorted(alarms)} "
                        f"或存在角色={sorted(presence_roles)}；"
                        f"实际告警命中={sorted(role for role in matched_alarm_roles if role)}，"
                        f"实际存在角色={sorted(role for role in matched_presence_roles if role)}"
                    ),
                )

        return True, None

    def _matches_node_structure_cached(self, node, node_config, helper, structure_match_cache=None):
        if structure_match_cache is None:
            return helper.matches_node_structure(self.sites_domain_map.get(node, {}), node_config)

        cache_key = (node, id(node_config))
        if cache_key not in structure_match_cache:
            structure_match_cache[cache_key] = helper.matches_node_structure(
                self.sites_domain_map.get(node, {}),
                node_config
            )
        return structure_match_cache[cache_key]

    def _get_precomputed_site_chain_candidates(self, start_node, direction, max_hops=None):
        """从 site_chains.json 预计算结果中取候选 hop；不支持混合多跳 either。"""
        if not self.site_chain_index:
            return None

        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            candidate_maps = []
            for single_direction in directions:
                single_candidates = self._get_precomputed_site_chain_candidates(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                )
                if single_candidates is None:
                    return None
                candidate_maps.append(single_candidates)
            return self._merge_candidate_hops(*candidate_maps)

        start_node = str(start_node or "").strip()
        chain_info = self.site_chain_index.get(start_node)
        if chain_info is None:
            return None

        direction = directions[0]

        candidates = {}

        def add_candidate(site_id, hop):
            site_id = str(site_id or "").strip()
            if not site_id or site_id == start_node:
                return
            if max_hops is not None and hop > max_hops:
                return
            previous_hop = candidates.get(site_id)
            if previous_hop is None or hop < previous_hop:
                candidates[site_id] = hop

        if direction == "downstream":
            for site_id, hop in chain_info.get("downstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            return candidates

        if direction == "upstream":
            for site_id, hop in chain_info.get("upstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            return candidates

        if direction == "either":
            # site_chains 只保存纯上游/纯下游的可达关系；混合方向多跳仍回退到原 BFS。
            if max_hops != 1:
                return None
            for site_id, hop in chain_info.get("downstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            for site_id, hop in chain_info.get("upstream_site_hops", {}).items():
                add_candidate(site_id, hop)
            for site_id in chain_info.get("bidirectional_sites", set()):
                add_candidate(site_id, 1)
            return candidates

        if direction in {"bidirection", "bidirectional"}:
            for site_id in chain_info.get("bidirectional_sites", set()):
                add_candidate(site_id, 1)
            return candidates

        return None

    def _validate_path_node_for_traversal(
        self,
        node,
        path_requirements,
        reference_ts,
        edge_window,
        helper,
        path_validation_cache=None,
    ):
        if path_validation_cache is None:
            node_domain = self.sites_domain_map.get(node, {})
            is_valid_path_node, _ = helper.validate_node(
                node, node_domain, path_requirements, reference_ts, edge_window
            )
            return is_valid_path_node

        cache_key = (
            node,
            id(path_requirements),
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
        )
        if cache_key not in path_validation_cache:
            node_domain = self.sites_domain_map.get(node, {})
            is_valid_path_node, _ = helper.validate_node(
                node, node_domain, path_requirements, reference_ts, edge_window
            )
            path_validation_cache[cache_key] = is_valid_path_node
        return path_validation_cache[cache_key]

    def _traverse_graph_nearest_matching(
        self,
        start_node,
        direction,
        target_node_config,
        max_hops=None,
        reference_ts=None,
        edge_window=0,
        path_requirements=None,
        node_rule_helper=None,
        traversal_cache=None,
        path_validation_cache=None,
        structure_match_cache=None,
        filtered_neighbor_cache=None,
    ):
        """nearest_matching 专用 BFS。

        一旦在某个 hop 首次命中结构匹配节点，就在该 hop 层结束后停止继续向外扩张，
        以避免在稠密图上无意义地遍历整张图。
        """
        helper = node_rule_helper or self.node_rule_helper
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            cache_key = (
                "nearest_matching_multi",
                start_node,
                directions,
                max_hops,
                reference_ts,
                self._make_edge_window_cache_key(edge_window),
                id(path_requirements),
                id(target_node_config),
            )
            if traversal_cache is not None and cache_key in traversal_cache:
                return traversal_cache[cache_key]

            candidate_maps = []
            had_topology_candidate = False
            for single_direction in directions:
                single_candidates, single_had_topology = self._traverse_graph_nearest_matching(
                    start_node=start_node,
                    direction=single_direction,
                    target_node_config=target_node_config,
                    max_hops=max_hops,
                    reference_ts=reference_ts,
                    edge_window=edge_window,
                    path_requirements=path_requirements,
                    node_rule_helper=helper,
                    traversal_cache=traversal_cache,
                    path_validation_cache=path_validation_cache,
                    structure_match_cache=structure_match_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                )
                candidate_maps.append(single_candidates)
                had_topology_candidate = had_topology_candidate or single_had_topology

            result = self._merge_candidate_hops(*candidate_maps)
            if result:
                nearest_hop = min(result.values())
                result = {
                    node: hop
                    for node, hop in result.items()
                    if hop == nearest_hop
                }
            cached_result = (result, had_topology_candidate)
            if traversal_cache is not None:
                traversal_cache[cache_key] = cached_result
            return cached_result

        direction = directions[0]

        static_cache_key = None
        if path_requirements is None:
            static_cache_key = (
                start_node,
                direction,
                max_hops,
                id(target_node_config),
            )
            with self._topo_cache_lock:
                if static_cache_key in self.global_nearest_match_cache:
                    self.global_nearest_match_cache.move_to_end(static_cache_key)
                    result = self.global_nearest_match_cache[static_cache_key]
                    if traversal_cache is not None:
                        traversal_cache[(
                            "nearest_matching",
                            start_node,
                            direction,
                            max_hops,
                            reference_ts,
                            self._make_edge_window_cache_key(edge_window),
                            id(path_requirements),
                            id(target_node_config),
                        )] = result
                    return result

        cache_key = (
            "nearest_matching",
            start_node,
            direction,
            max_hops,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
            id(target_node_config),
        )
        if traversal_cache is not None and cache_key in traversal_cache:
            return traversal_cache[cache_key]

        if path_requirements is None:
            precomputed_candidates = self._get_precomputed_site_chain_candidates(
                start_node,
                direction,
                max_hops=max_hops,
            )
            if precomputed_candidates is not None:
                had_topology_candidate = bool(precomputed_candidates)
                result = {}
                nearest_hop = None
                for curr, hops in sorted(precomputed_candidates.items(), key=lambda item: (item[1], str(item[0]))):
                    if nearest_hop is not None and hops > nearest_hop:
                        break
                    if self._matches_node_structure_cached(
                        curr,
                        target_node_config,
                        helper,
                        structure_match_cache=structure_match_cache,
                    ):
                        if nearest_hop is None:
                            nearest_hop = hops
                        result[curr] = hops

                cached_result = (result, had_topology_candidate)
                if traversal_cache is not None:
                    traversal_cache[cache_key] = cached_result
                if static_cache_key is not None:
                    with self._topo_cache_lock:
                        self.global_nearest_match_cache[static_cache_key] = cached_result
                        self.global_nearest_match_cache.move_to_end(static_cache_key)
                        if len(self.global_nearest_match_cache) > self.max_nearest_match_cache_size:
                            self.global_nearest_match_cache.popitem(last=False)
                return cached_result

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        topo = self.topo_up if direction == "upstream" else self.topo_down

        result = {}
        nearest_hop = None
        had_topology_candidate = False

        while queue:
            curr, hops = queue.popleft()
            if nearest_hop is not None and hops > nearest_hop:
                break

            if hops > 0:
                had_topology_candidate = True
                if self._matches_node_structure_cached(
                    curr,
                    target_node_config,
                    helper,
                    structure_match_cache=structure_match_cache,
                ):
                    if nearest_hop is None:
                        nearest_hop = hops
                    if hops == nearest_hop:
                        result[curr] = hops

            if nearest_hop is not None and hops >= nearest_hop:
                continue

            if max_hops is None or hops < max_hops:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    curr,
                    direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    queue.append((nxt, hops + 1))

        if traversal_cache is not None:
            traversal_cache[cache_key] = (result, had_topology_candidate)
        if static_cache_key is not None:
            with self._topo_cache_lock:
                self.global_nearest_match_cache[static_cache_key] = (result, had_topology_candidate)
                self.global_nearest_match_cache.move_to_end(static_cache_key)
                if len(self.global_nearest_match_cache) > self.max_nearest_match_cache_size:
                    self.global_nearest_match_cache.popitem(last=False)

        return result, had_topology_candidate

    def _traverse_graph(self, start_node, direction, max_hops=None,
                        reference_ts=None, edge_window=0, 
                        path_requirements=None, node_rule_helper=None,
                        traversal_cache=None, path_validation_cache=None,
                        filtered_neighbor_cache=None):
        """通用的广度优先搜索，支持路径节点约束"""
        helper = node_rule_helper or self.node_rule_helper
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            local_cache_key = (
                "full_multi",
                start_node,
                directions,
                max_hops,
                reference_ts,
                self._make_edge_window_cache_key(edge_window),
                id(path_requirements),
            )
            if traversal_cache is not None and local_cache_key in traversal_cache:
                return traversal_cache[local_cache_key]

            result = self._merge_candidate_hops(*[
                self._traverse_graph(
                    start_node,
                    single_direction,
                    max_hops=max_hops,
                    reference_ts=reference_ts,
                    edge_window=edge_window,
                    path_requirements=path_requirements,
                    node_rule_helper=helper,
                    traversal_cache=traversal_cache,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                )
                for single_direction in directions
            ])
            if traversal_cache is not None:
                traversal_cache[local_cache_key] = result
            return result

        direction = directions[0]

        if direction == "self":
            return {start_node: 0}

        local_cache_key = (
            "full",
            start_node,
            direction,
            max_hops,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
        )
        if traversal_cache is not None and local_cache_key in traversal_cache:
            return traversal_cache[local_cache_key]

        cache_key = (start_node, direction, max_hops)

        if path_requirements is None:
            with self._topo_cache_lock:
                if cache_key in self.global_topo_cache:
                    self.global_topo_cache.move_to_end(cache_key)
                    result = self.global_topo_cache[cache_key]
                    if traversal_cache is not None:
                        traversal_cache[local_cache_key] = result
                    return result

            precomputed_candidates = self._get_precomputed_site_chain_candidates(
                start_node,
                direction,
                max_hops=max_hops,
            )
            if precomputed_candidates is not None:
                with self._topo_cache_lock:
                    self.global_topo_cache[cache_key] = precomputed_candidates
                    self.global_topo_cache.move_to_end(cache_key)
                    if len(self.global_topo_cache) > self.max_topo_cache_size:
                        self.global_topo_cache.popitem(last=False)
                if traversal_cache is not None:
                    traversal_cache[local_cache_key] = precomputed_candidates
                return precomputed_candidates

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        result = {}

        topo = self.topo_up if direction == "upstream" else self.topo_down

        while queue:
            curr, hops = queue.popleft()
            if hops > 0:
                result[curr] = hops
            if max_hops is None or hops < max_hops:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    curr,
                    direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, hops + 1))

        # 写缓存：仅缓存不带路径约束的纯拓扑结果
        if path_requirements is None:
            with self._topo_cache_lock:
                self.global_topo_cache[cache_key] = result
                self.global_topo_cache.move_to_end(cache_key)
                if len(self.global_topo_cache) > self.max_topo_cache_size:
                    self.global_topo_cache.popitem(last=False)
        if traversal_cache is not None:
            traversal_cache[local_cache_key] = result
        return result

    def _get_filtered_neighbors_for_traversal(
        self,
        node,
        direction,
        reference_ts,
        edge_window,
        path_requirements,
        helper,
        path_validation_cache=None,
        filtered_neighbor_cache=None,
    ):
        directions = self._normalize_traverse_directions(direction)
        if len(directions) > 1:
            neighbors = []
            seen = set()
            for single_direction in directions:
                for nxt in self._get_filtered_neighbors_for_traversal(
                    node,
                    single_direction,
                    reference_ts,
                    edge_window,
                    path_requirements,
                    helper,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                ):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    neighbors.append(nxt)
            return tuple(neighbors)

        direction = directions[0]
        if direction == "upstream":
            topo_neighbors = tuple(self.topo_up.get(node, []))
        elif direction == "either":
            seen = set()
            topo_neighbors = []
            for nxt in self.topo_up.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    topo_neighbors.append(nxt)
            for nxt in self.topo_down.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    topo_neighbors.append(nxt)
            topo_neighbors = tuple(topo_neighbors)
        elif direction in {"bidirection", "bidirectional"}:
            topo_neighbors = tuple(
                sorted(set(self.topo_down.get(node, ())) & set(self.topo_up.get(node, ())))
            )
        else:
            topo_neighbors = tuple(self.topo_down.get(node, []))

        if path_requirements is None:
            return topo_neighbors

        cache_key = (
            node,
            direction,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            id(path_requirements),
        )
        if filtered_neighbor_cache is not None and cache_key in filtered_neighbor_cache:
            return filtered_neighbor_cache[cache_key]

        valid_neighbors = tuple(
            nxt
            for nxt in topo_neighbors
            if self._validate_path_node_for_traversal(
                nxt,
                path_requirements,
                reference_ts,
                edge_window,
                helper,
                path_validation_cache=path_validation_cache,
            )
        )
        if filtered_neighbor_cache is not None:
            filtered_neighbor_cache[cache_key] = valid_neighbors
        return valid_neighbors
