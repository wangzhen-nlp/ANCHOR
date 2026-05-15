import collections

from fault_grouping.temporal_engine.utils import qualify_role_key


class TemporalGraphEngineOutputMixin:
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
            raw_role = role
            prefix = f"{rule_name}."
            if isinstance(role, str) and role.startswith(prefix):
                raw_role = role[len(prefix):]
            node_config = rule.get("nodes", {}).get(raw_role)
            if node_config is not None:
                return node_config
        return {}

    def _apply_output_visibility_filters(self, match_result):
        """仅过滤最终输出视图，不影响规则匹配和 result_constraints 判断。"""
        alarm_nodes_by_role = collections.defaultdict(set)
        for symptom in match_result.get("symptoms", []):
            node = symptom.get("node")
            if node in (None, ""):
                continue
            for role in (symptom.get("matched_role"), symptom.get("matched_role_key")):
                if role:
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
            role_key_prefix = f"{rule_name}."
            raw_role = role
            qualified = False
            if isinstance(role, str) and role.startswith(role_key_prefix):
                raw_role = role[len(role_key_prefix):]
                qualified = True
            for edge in rule.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                directions = self._normalize_traverse_directions(edge.get("direction", "downstream"))
                if source == raw_role and "upstream" in directions:
                    parent_role = target
                elif target == raw_role and "downstream" in directions:
                    parent_role = source
                else:
                    continue
                if qualified:
                    parent_role = qualify_role_key(rule_name, parent_role)
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
        available_role_set = set(available_roles)
        merged_rules = match_result.get("merged_rules", [match_result.get("rule")])
        for rule_name in merged_rules:
            rule = self.rules.get(rule_name)
            if not rule:
                continue
            raw_groups = self._normalize_exclusive_site_role_groups(
                rule.get("exclusive_site_roles"),
                list(rule.get("nodes", {}).keys()),
            )
            for raw_group in raw_groups:
                group = []
                seen = set()
                for raw_role in raw_group:
                    for role in (raw_role, qualify_role_key(rule_name, raw_role)):
                        if role in available_role_set and role not in seen:
                            seen.add(role)
                            group.append(role)
                if len(group) > 1:
                    groups.append(group)
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
                    matched_rule = symptom.get("matched_rule")
                    matched_role = owner_role
                    if isinstance(owner_role, str) and "." in owner_role:
                        matched_rule, matched_role = owner_role.split(".", 1)
                    symptom = {
                        **symptom,
                        "matched_role": matched_role,
                        "matched_rule": matched_rule,
                        "matched_role_key": owner_role,
                    }
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
