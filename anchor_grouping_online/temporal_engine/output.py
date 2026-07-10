import collections

from anchor_grouping_online.temporal_engine.utils import _normalize_edge_directions, qualify_role_key


class TemporalGraphEngineOutputMixin:
    @staticmethod
    def _hide_role_node_without_alarm(node_config):
        """是否在最终输出中隐藏该 role 下没有贡献任何告警的站点。"""
        return bool(node_config.get("hide_if_no_alarms"))

    def _get_role_node_config_for_output(self, role, match_result):
        merged_rules = match_result["merged_rules"]
        for rule_name in merged_rules:
            rule = self.rules[rule_name]
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
        for symptom in match_result["symptoms"]:
            node = symptom.get("node")
            if node in (None, ""):
                continue
            for role in (symptom.get("matched_role"), symptom.get("matched_role_key")):
                if role:
                    alarm_nodes_by_role[role].add(node)

        filtered_role_mapping = {}
        for role, nodes in match_result["role_mapping"].items():
            node_config = self._get_role_node_config_for_output(role, match_result)
            if self._hide_role_node_without_alarm(node_config):
                nodes = [
                    node for node in nodes
                    if node in alarm_nodes_by_role.get(role, set())
                ]
            if nodes:
                filtered_role_mapping[role] = nodes

        filtered_inferred_roots = {}
        for role, nodes in match_result["inferred_roots"].items():
            node_config = self._get_role_node_config_for_output(role, match_result)
            if self._hide_role_node_without_alarm(node_config):
                nodes = [
                    node for node in nodes
                    if node in alarm_nodes_by_role.get(role, set())
                ]
            if nodes:
                filtered_inferred_roots[role] = nodes

        if (
            filtered_role_mapping == match_result["role_mapping"]
            and filtered_inferred_roots == match_result["inferred_roots"]
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
        merged_rules = match_result["merged_rules"]
        for rule_name in merged_rules:
            rule = self.rules[rule_name]
            role_key_prefix = f"{rule_name}."
            raw_role = role
            qualified = False
            if isinstance(role, str) and role.startswith(role_key_prefix):
                raw_role = role[len(role_key_prefix):]
                qualified = True
            for edge in rule["edges"]:
                source = edge.get("source")
                target = edge.get("target")
                directions = _normalize_edge_directions(edge["direction"])
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
        role_mapping = match_result["role_mapping"]
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

    def _get_exclusive_site_role_groups_for_output(self, match_result, available_roles):
        groups = []
        available_role_set = set(available_roles)
        merged_rules = match_result["merged_rules"]
        for rule_name in merged_rules:
            rule = self.rules[rule_name]
            group = []
            seen = set()
            for raw_role in rule["exclusive_site_roles"]:
                for role in (raw_role, qualify_role_key(rule_name, raw_role)):
                    if role in available_role_set and role not in seen:
                        seen.add(role)
                        group.append(role)
            if len(group) > 1:
                groups.append(group)
        return groups

    def _apply_default_output_site_role_ownership(self, match_result):
        role_mapping = match_result["role_mapping"]
        if len(role_mapping) <= 1:
            return match_result

        remove_by_role, owner_by_removed_role_site = (
            self._collect_output_site_role_removals(match_result, role_mapping)
        )
        if not remove_by_role:
            return match_result

        filtered_role_mapping = self._filter_role_nodes(role_mapping, remove_by_role)
        filtered_inferred_roots = self._filter_role_nodes(
            match_result["inferred_roots"], remove_by_role
        )
        filtered_symptoms = self._reassign_removed_symptoms(
            match_result["symptoms"], remove_by_role,
            owner_by_removed_role_site, filtered_role_mapping,
        )
        return {
            **match_result,
            "role_mapping": filtered_role_mapping,
            "inferred_roots": filtered_inferred_roots,
            "symptoms": filtered_symptoms,
        }

    def _collect_output_site_role_removals(self, match_result, role_mapping):
        """按互斥 role 组挑选站点归属，返回 (待移除站点, 被移除者的归属 role)。"""
        remove_by_role = collections.defaultdict(set)
        owner_by_removed_role_site = {}
        exclusive_role_groups = self._get_exclusive_site_role_groups_for_output(
            match_result,
            list(role_mapping.keys()),
        )
        if not exclusive_role_groups:
            return remove_by_role, owner_by_removed_role_site

        role_order = {role: idx for idx, role in enumerate(role_mapping.keys())}
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
        return remove_by_role, owner_by_removed_role_site

    @staticmethod
    def _filter_role_nodes(mapping, remove_by_role):
        """移除各 role 中被裁掉的站点，裁空的 role 整个删除。"""
        filtered = {}
        for role, nodes in mapping.items():
            filtered_nodes = [
                node for node in nodes
                if node not in remove_by_role.get(role, set())
            ]
            if filtered_nodes:
                filtered[role] = filtered_nodes
        return filtered

    @staticmethod
    def _reassign_removed_symptoms(
        symptoms, remove_by_role, owner_by_removed_role_site, filtered_role_mapping
    ):
        """被裁站点的症状改挂归属 role；归属 role 不含该站点时丢弃症状。"""
        filtered_symptoms = []
        for symptom in symptoms:
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
        return filtered_symptoms

    @staticmethod
    def _get_optional_only_roles(rule):
        """识别仅通过 optional 边引入的角色；这些 role 被裁空时可视作未命中。"""
        optional_incident = set()
        required_incident = set()
        for edge in rule["edges"]:
            source = edge["source"]
            target = edge["target"]
            if edge.get("optional"):
                optional_incident.update([source, target])
            else:
                required_incident.update([source, target])
        return optional_incident - required_incident

    def _get_site_chain_downstream_hop(self, parent_site, child_site):
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

        # 无 site_chains 或无法用 downstream_hops 判定时，按遍历顺序 first-win。
        return min(roles, key=lambda role: role_order[role])

    def _apply_default_site_role_ownership(self, inst, rule, nodes_cfg):
        """按 rule.exclusive_site_roles 裁剪重复站点归属，避免环/双向边导致角色串位。"""
        roles = inst["roles"]
        if len(roles) <= 1:
            return inst

        exclusive_roles = [
            role
            for role in rule["exclusive_site_roles"]
            if role in roles
        ]
        if len(exclusive_roles) <= 1:
            return inst

        remove_by_role = self._collect_instance_site_role_removals(
            inst, roles, exclusive_roles
        )
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

    def _collect_instance_site_role_removals(self, inst, roles, exclusive_roles):
        """同一站点命中多个互斥 role 时选归属 role，其余 role 移除该站点。"""
        role_order = {role: idx for idx, role in enumerate(roles.keys())}
        remove_by_role = collections.defaultdict(set)
        site_to_roles = collections.defaultdict(list)
        for role in exclusive_roles:
            for site in roles[role]["nodes"]:
                site_to_roles[site].append(role)

        for site, site_roles in site_to_roles.items():
            if len(site_roles) <= 1:
                continue
            owner_role = self._choose_site_owner_role(inst, site, site_roles, role_order)
            for role in site_roles:
                if role != owner_role:
                    remove_by_role[role].add(site)
        return remove_by_role

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
            return candidate_hops

        filtered = {}
        for cand_phys, hop in candidate_hops.items():
            if self._keep_symmetric_pair_candidate(curr_role, tgt_role, edge, curr_phys, cand_phys):
                filtered[cand_phys] = hop

        return filtered
