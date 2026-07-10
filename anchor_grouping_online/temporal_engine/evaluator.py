from datetime import datetime

from anchor_grouping_online.alarm_types import LINK_ALARMS
from anchor_grouping_online.temporal_engine.utils import (
    clone_instance_with_updates,
    get_symptom_alarm_identity,
    qualify_role_key,
)
from anchor_grouping_online.link_peer_index import resolve_link_alarm_endpoints_from_peer_index


def link_alarm_points_to_site(alarm_info, target_site, ne_to_site, peer_index, alarm_source=""):
    endpoints = resolve_link_alarm_endpoints_from_peer_index(
        alarm_info,
        peer_index=peer_index,
        alarm_source=alarm_source,
    )
    if not endpoints.remote_ne:
        return False
    return str(ne_to_site.get(endpoints.remote_ne, "") or "").strip() == str(target_site or "").strip()


class TemporalGraphEngineEvaluatorMixin:
    def _validate_trigger_node_for_rule(
        self,
        rule_name,
        nodes_cfg,
        trigger_role,
        trigger_node,
        trigger_ts,
        helper,
        validation_cache,
    ):
        trigger_node_domain = self.sites_domain_map.get(trigger_node, {})
        cache_key = (
            trigger_node,
            trigger_role,
            trigger_ts,
            0,
            rule_name,
        )
        if cache_key in validation_cache:
            is_valid, trigger_events = validation_cache[cache_key]
        else:
            is_valid, trigger_events = helper.validate_node(
                trigger_node,
                trigger_node_domain,
                nodes_cfg[trigger_role],
                trigger_ts,
                edge_window=0,
                exclude_consumed_trigger_rule=rule_name,
            )
            validation_cache[cache_key] = (is_valid, trigger_events)

        return is_valid, trigger_events

    def _collect_edge_candidates(
        self,
        curr_phys,
        curr_role,
        tgt_role,
        edge,
        tgt_cfg,
        traversal_cache,
        rule_name=None,
    ):
        candidate_hops = self._traverse_graph_role_filtered(
            rule_name=rule_name,
            start_node=curr_phys,
            target_role=tgt_role,
            direction=edge["traverse_dir"],
            max_hops=edge["hops"],
            traversal_cache=traversal_cache,
            target_node_config=tgt_cfg,
        )
        candidate_hops = self._filter_symmetric_pair_candidates(
            candidate_hops,
            curr_role,
            tgt_role,
            edge,
            curr_phys,
        )
        # _traverse_graph_role_filtered() 的缓存已按 (hop, site_id) 排序；对称边
        # 过滤通过 dict comprehension 保持该顺序，因此这里只需按原接口返回 list。
        return list(candidate_hops)

    def _validate_candidate_nodes_for_edge(
        self, candidates, tgt_role, tgt_cfg, ref_ts, edge, rule_name,
        trigger_role, helper, validation_cache, allowed_alarm_source_nes=None,
    ):
        curr_valid_targets = {}
        window_cache_key = edge["win"]

        # 当 NE 锚点约束生效时，allowed_alarm_source_nes 是 frozenset；为 None 时不过滤。
        # cache_key 必须区分这一维度，否则不同 anchor 绑定的实例会复用错误结果。
        ne_filter_key = allowed_alarm_source_nes if allowed_alarm_source_nes is not None else None

        for cand_phys in candidates:
            exclude_consumed_trigger_rule = rule_name if tgt_role == trigger_role else None
            cache_key = self._candidate_validation_cache_key(
                rule_name, tgt_role, tgt_cfg, cand_phys, ref_ts,
                window_cache_key, ne_filter_key, exclude_consumed_trigger_rule,
            )
            if cache_key in validation_cache:
                is_valid, events = validation_cache[cache_key]
            else:
                cand_phys_domain = self.sites_domain_map.get(cand_phys, {})
                is_valid, events = helper.validate_node(
                    cand_phys,
                    cand_phys_domain,
                    tgt_cfg,
                    ref_ts,
                    edge["win"],
                    exclude_consumed_trigger_rule=exclude_consumed_trigger_rule,
                    allowed_alarm_source_nes=allowed_alarm_source_nes,
                )
                validation_cache[cache_key] = (is_valid, events)

            if is_valid:
                curr_valid_targets[cand_phys] = events

        return curr_valid_targets

    @staticmethod
    def _candidate_validation_cache_key(
        rule_name, tgt_role, tgt_cfg, cand_phys, ref_ts,
        window_cache_key, ne_filter_key, exclude_consumed_trigger_rule,
    ):
        """候选校验缓存键；无 trigger 排除维度时跨规则共享。"""
        if exclude_consumed_trigger_rule is None:
            return (
                "candidate_node_shared", id(tgt_cfg), cand_phys, ref_ts,
                window_cache_key, ne_filter_key,
            )
        return (
            "candidate_node", rule_name, tgt_role, id(tgt_cfg), cand_phys,
            ref_ts, window_cache_key, ne_filter_key,
        )

    def _get_required_link_alarm_set_for_role(self, phys_node, role_cfg, helper):
        expected = helper.resolve_expected_alarms(
            self.sites_domain_map.get(phys_node, {}),
            role_cfg,
        )
        # Compound roles do not define ``site_rules`` themselves; their child
        # patterns do.  In that case resolve_expected_alarms() legitimately
        # returns None, so there is no role-level link-alarm requirement to
        # apply here.
        if not isinstance(expected, dict):
            return frozenset()

        required_alarms = expected.get("required_alarms")
        if required_alarms is None:
            return frozenset()
        return frozenset(alarm for alarm in required_alarms if alarm in LINK_ALARMS)

    def _filter_link_role_events_for_related_site(
        self,
        phys_node,
        role_cfg,
        events,
        related_site,
        helper,
    ):
        required_link_alarms = self._get_required_link_alarm_set_for_role(
            phys_node,
            role_cfg,
            helper,
        )
        if not required_link_alarms:
            return list(events), True

        required_link_events = [
            event for event in events
            if event.get("alarm") in required_link_alarms
        ]
        if not required_link_events:
            return [], False

        matched_link_event_ids = {
            id(event)
            for event in required_link_events
            if link_alarm_points_to_site(
                event.get("alarm_payload") or {},
                related_site,
                self._ne_to_site,
                peer_index=self._link_peer_index,
                alarm_source=event.get("alarm_source", ""),
            ) is True
        }
        if not matched_link_event_ids:
            return [], False

        filtered_events = [
            event for event in events
            if event.get("alarm") not in required_link_alarms
            or id(event) in matched_link_event_ids
        ]
        return filtered_events, True

    def _apply_link_peer_site_filter_for_edge(
        self,
        curr_phys,
        curr_events,
        curr_cfg,
        tgt_cfg,
        curr_valid_targets,
        curr_events_by_target,
        helper,
    ):
        if not curr_valid_targets:
            return curr_valid_targets, curr_events_by_target

        filtered_targets = {}
        filtered_curr_events_by_target = {}

        for target_phys, target_events in curr_valid_targets.items():
            filtered_target_events, target_ok = self._filter_link_role_events_for_related_site(
                target_phys,
                tgt_cfg,
                target_events,
                curr_phys,
                helper,
            )
            if not target_ok:
                continue

            source_events = curr_events_by_target.get(target_phys, curr_events)
            filtered_source_events, source_ok = self._filter_link_role_events_for_related_site(
                curr_phys,
                curr_cfg,
                source_events,
                target_phys,
                helper,
            )
            if not source_ok:
                continue

            filtered_targets[target_phys] = filtered_target_events
            filtered_curr_events_by_target[target_phys] = filtered_source_events

        return filtered_targets, filtered_curr_events_by_target


    def _evaluate_edge_source_node(
        self, curr_phys, curr_events, curr_role, tgt_role, tgt_cfg, curr_cfg,
        edge, rule_name, trigger_role, trigger_ts, helper,
        validation_cache, traversal_cache, allowed_alarm_source_nes=None,
    ):
        ref_ts = curr_events[0]["ts"] if curr_events else trigger_ts
        candidates = self._collect_edge_candidates(
            curr_phys,
            curr_role,
            tgt_role,
            edge,
            tgt_cfg,
            traversal_cache,
            rule_name=rule_name,
        )
        curr_valid_targets = self._validate_candidate_nodes_for_edge(
            candidates, tgt_role, tgt_cfg, ref_ts, edge, rule_name,
            trigger_role, helper, validation_cache,
            allowed_alarm_source_nes=allowed_alarm_source_nes,
        )

        curr_events_by_target = {
            target_phys: curr_events
            for target_phys in curr_valid_targets
        }
        curr_valid_targets, curr_events_by_target = self._apply_link_peer_site_filter_for_edge(
            curr_phys,
            curr_events,
            curr_cfg,
            tgt_cfg,
            curr_valid_targets,
            curr_events_by_target,
            helper,
        )

        if curr_valid_targets:
            return curr_valid_targets, curr_events_by_target
        return {}, {}

    def _compute_allowed_alarm_source_nes_for_role(self, rule_name, tgt_role, inst):
        """返回 tgt_role 的 NE 锚点允许告警源；无锚点约束时返回 None。"""
        plan = self.rule_execution_plans[rule_name]
        anchors = plan["alarm_source_ne_anchors"]
        anchor_cfg = anchors.get(tgt_role)
        if not anchor_cfg:
            return None
        anchor_role = anchor_cfg["anchor_role"]
        anchor_nodes = inst["roles"][anchor_role]["nodes"]
        anchor_site = next(iter(anchor_nodes))
        return self._compute_anchor_ne_reachable_set(
            anchor_site,
            anchor_cfg["max_ne_hops"],
        )

    def _collect_instance_edge_targets(
        self, inst, curr_role, tgt_role, edge, tgt_cfg, rule_name,
        trigger_role, trigger_ts, helper, nodes_cfg,
        validation_cache, traversal_cache,
    ):
        valid_targets = {}
        surviving_curr_phys = {}
        curr_support_targets = {}
        curr_events_by_target = {}
        curr_cfg = nodes_cfg[curr_role]

        # 计算 NE 锚点允许 alarm_source NE 集合（如果 tgt_role 配置了 alarm_source_ne_anchor）
        allowed_alarm_source_nes = self._compute_allowed_alarm_source_nes_for_role(
            rule_name, tgt_role, inst,
        )

        for curr_phys, curr_events in inst["roles"][curr_role]["nodes"].items():
            curr_valid_targets, source_events_by_target = self._evaluate_edge_source_node(
                curr_phys, curr_events, curr_role, tgt_role, tgt_cfg, curr_cfg,
                edge, rule_name, trigger_role, trigger_ts, helper,
                validation_cache, traversal_cache,
                allowed_alarm_source_nes=allowed_alarm_source_nes,
            )
            if not curr_valid_targets:
                continue

            combined_curr_events = self._combine_source_events(
                curr_phys, curr_events, curr_valid_targets,
                source_events_by_target, curr_events_by_target,
            )
            surviving_curr_phys[curr_phys] = combined_curr_events or curr_events
            curr_support_targets[curr_phys] = set(curr_valid_targets)
            for key, value in curr_valid_targets.items():
                valid_targets[key] = value

        return valid_targets, surviving_curr_phys, curr_support_targets, curr_events_by_target

    @staticmethod
    def _combine_source_events(
        curr_phys, curr_events, curr_valid_targets,
        source_events_by_target, curr_events_by_target,
    ):
        """按事件身份合并各 target 的 source 事件，并登记 (curr, tgt) 明细。"""
        combined_curr_events = []
        seen_curr_event_ids = set()
        for target_phys in curr_valid_targets:
            events = source_events_by_target.get(target_phys, curr_events)
            for event in events:
                event_id = get_symptom_alarm_identity(event) or (
                    event.get("node"),
                    event.get("ts"),
                    event.get("alarm"),
                    event.get("alarm_source"),
                )
                if event_id in seen_curr_event_ids:
                    continue
                seen_curr_event_ids.add(event_id)
                combined_curr_events.append(event)
            curr_events_by_target[(curr_phys, target_phys)] = events
        return combined_curr_events

    def _materialize_edge_advanced_instances(
        self,
        inst,
        curr_role,
        tgt_role,
        tgt_cfg,
        nodes_cfg,
        surviving_curr_phys,
        valid_targets,
        curr_support_targets,
        curr_events_by_target=None,
    ):
        existing_targets = inst["roles"].get(tgt_role, {}).get("nodes", {})
        merged_targets = {**existing_targets, **valid_targets}
        min_count = tgt_cfg.get("min_count", 1)
        node_type = tgt_cfg.get("type", "primitive")

        if node_type == "primitive" and not existing_targets:
            return self._fork_primitive_target_instances(
                inst,
                curr_role,
                tgt_role,
                surviving_curr_phys,
                valid_targets,
                curr_support_targets,
                nodes_cfg,
                curr_events_by_target=curr_events_by_target,
            )

        if len(merged_targets) < min_count:
            return []

        new_inst = clone_instance_with_updates(
            inst,
            curr_role,
            surviving_curr_phys,
            tgt_role,
            merged_targets,
        )
        self._record_instance_dependency(new_inst, curr_role, tgt_role, curr_support_targets)
        stabilized_inst = self._stabilize_instance_dependencies(new_inst, nodes_cfg)
        return [stabilized_inst] if stabilized_inst is not None else []

    def _advance_instance_across_edge(
        self, inst, curr_role, tgt_role, edge, nodes_cfg, rule_name,
        trigger_role, trigger_ts, helper, validation_cache, traversal_cache,
    ):
        inst_roles = inst["roles"]

        if curr_role not in inst_roles:
            return [inst]

        if tgt_role in inst_roles and inst_roles[tgt_role]["checked"]:
            return [inst]

        tgt_cfg = nodes_cfg[tgt_role]
        valid_targets, surviving_curr_phys, curr_support_targets, curr_events_by_target = (
            self._collect_instance_edge_targets(
                inst, curr_role, tgt_role, edge, tgt_cfg, rule_name,
                trigger_role, trigger_ts, helper, nodes_cfg,
                validation_cache, traversal_cache,
            )
        )

        if not valid_targets:
            if edge.get("optional"):
                return [inst]
            return []

        curr_cfg = nodes_cfg[curr_role]
        if inst_roles[curr_role]["checked"] and len(surviving_curr_phys) < curr_cfg.get("min_count", 1):
            return []

        return self._materialize_edge_advanced_instances(
            inst,
            curr_role,
            tgt_role,
            tgt_cfg,
            nodes_cfg,
            surviving_curr_phys,
            valid_targets,
            curr_support_targets,
            curr_events_by_target=curr_events_by_target,
        )

    def _fork_primitive_target_instances(
        self, inst, curr_role, tgt_role, surviving_curr_phys, valid_targets,
        curr_support_targets, nodes_cfg, curr_events_by_target=None,
    ):
        next_instances = []
        curr_events_by_target = curr_events_by_target or {}
        supporting_nodes_for = self._build_target_support_index(
            surviving_curr_phys, curr_support_targets, valid_targets
        )
        for target_node, target_events in valid_targets.items():
            supporting_curr_nodes = supporting_nodes_for(target_node)
            target_surviving_curr_phys = {
                curr_node: curr_events_by_target.get(
                    (curr_node, target_node),
                    surviving_curr_phys[curr_node],
                )
                for curr_node in supporting_curr_nodes
            }
            new_inst = clone_instance_with_updates(
                inst,
                curr_role,
                target_surviving_curr_phys,
                tgt_role,
                {target_node: target_events},
            )
            self._record_instance_dependency(
                new_inst,
                curr_role,
                tgt_role,
                {
                    curr_node: {target_node}
                    for curr_node in supporting_curr_nodes
                },
            )
            stabilized_inst = self._stabilize_instance_dependencies(new_inst, nodes_cfg)
            if stabilized_inst is not None:
                next_instances.append(stabilized_inst)
        return next_instances

    @staticmethod
    def _build_target_support_index(
        surviving_curr_phys, curr_support_targets, valid_targets
    ):
        """反转支撑关系，返回 target_node -> 支撑它的 current nodes 的查询函数。

        旧实现对每个 target 都完整扫描一次 surviving_curr_phys，并再次扫描
        curr_support_targets 来构造依赖，复杂度接近 O(targets * current_nodes)。
        这里先反转一次支撑关系，后续只访问真正支撑该 target 的 current nodes；
        按 surviving_curr_phys 的原顺序构建 list，保持实例内容及迭代顺序不变。
        单 current node（trigger 首次向外分叉的最常见形态）时直接做集合成员
        判断，避免为单个 current node 构建反向字典。
        """
        if len(surviving_curr_phys) == 1:
            single_curr_node = next(iter(surviving_curr_phys))
            supported_targets = curr_support_targets.get(single_curr_node, ())
            return lambda target_node: (
                (single_curr_node,) if target_node in supported_targets else ()
            )
        supporting_curr_nodes_by_target = {
            target_node: []
            for target_node in valid_targets
        }
        for curr_node in surviving_curr_phys:
            for target_node in curr_support_targets.get(curr_node, ()):
                supporting_curr_nodes = supporting_curr_nodes_by_target.get(
                    target_node
                )
                if supporting_curr_nodes is not None:
                    supporting_curr_nodes.append(curr_node)
        return lambda target_node: supporting_curr_nodes_by_target.get(target_node, ())

    def _advance_instances_across_edge(
        self,
        instances,
        curr_role,
        tgt_role,
        edge,
        nodes_cfg,
        rule_name,
        trigger_role,
        trigger_ts,
        helper,
        caches,
    ):
        next_instances = []
        for inst in instances:
            next_instances.extend(
                self._advance_instance_across_edge(
                    inst,
                    curr_role,
                    tgt_role,
                    edge,
                    nodes_cfg,
                    rule_name,
                    trigger_role,
                    trigger_ts,
                    helper,
                    caches["validation_cache"],
                    caches["traversal_cache"],
                )
            )
        return next_instances

    def _build_symptoms_and_role_mapping_from_instance(self, inst_roles, rule_name):
        symptoms_by_key = {}
        role_mapping = {}

        for role, role_state in inst_roles.items():
            valid_phys_nodes = []
            for phys_node, events in role_state["nodes"].items():
                valid_phys_nodes.append(phys_node)
                for event in events:
                    self._add_event_to_symptom_dict(symptoms_by_key, event, role, rule_name)
            if valid_phys_nodes:
                role_mapping[role] = valid_phys_nodes

        return symptoms_by_key, role_mapping

    def _add_event_to_symptom_dict(self, symptoms_by_key, event, role, rule_name):
        event_enriched = dict(event)
        event_enriched["matched_role"] = role
        event_enriched["matched_rule"] = rule_name
        event_enriched["matched_role_key"] = qualify_role_key(rule_name, role)
        event_enriched["time_str"] = datetime.fromtimestamp(event["ts"]).strftime("%Y-%m-%d %H:%M:%S")

        alarm_key = get_symptom_alarm_identity(event_enriched)
        symptoms_by_key[alarm_key] = event_enriched


    def _build_match_result_from_instance(self, inst, rule_name, rule, root_roles, trigger_ts):
        inst_roles = inst["roles"]
        inferred_roots = {
            root_role: list(inst_roles.get(root_role, {}).get("nodes", {}).keys())
            for root_role in root_roles
        }
        symptoms_by_key, role_mapping = self._build_symptoms_and_role_mapping_from_instance(inst_roles, rule_name)
        match_result = {
            "rule": rule_name,
            "merged_rules": [rule_name],
            "inferred_roots": inferred_roots,
            "role_mapping": role_mapping,
            "symptoms": list(symptoms_by_key.values()),
            "_expire_ts_hint": (
                min((symptom["ts"] for symptom in symptoms_by_key.values() if "ts" in symptom), default=trigger_ts)
                + rule["max_stay_time_sec"]
            ),
        }
        return match_result

    def _build_match_results_from_instances(
        self,
        instances,
        rule_name,
        rule,
        nodes_cfg,
        root_roles,
        trigger_ts,
    ):
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

            match_result = self._build_match_result_from_instance(
                stabilized_inst,
                rule_name,
                rule,
                root_roles,
                trigger_ts,
            )
            if not self._validate_result_constraints(rule, match_result):
                continue
            results.append(match_result)
        return results

    def _evaluate_rule(
        self,
        rule_name,
        rule,
        trigger_node,
        trigger_ts,
        eval_caches=None,
    ):
        """
        全向动态图调度器 (State-Forking Matcher)：
        支持平行宇宙分叉、严格结构匹配、局部性能缓存。
        """
        helper = self.node_rule_helper
        nodes_cfg = rule["nodes"]

        plan = self.rule_execution_plans[rule_name]
        trigger_role = plan["trigger_role"]
        edges_to_explore = plan["edges_to_explore"]
        root_roles = plan["root_roles"]
        caches = eval_caches or self._create_eval_caches()
        is_trigger_valid, trigger_events = self._validate_trigger_node_for_rule(
            rule_name, nodes_cfg, trigger_role, trigger_node, trigger_ts,
            helper, caches["validation_cache"],
        )
        if not is_trigger_valid:
            return []

        instances = [{
            "roles": {
                trigger_role: {"nodes": {trigger_node: trigger_events}, "checked": False}
            },
            "_dependencies": {},
        }]

        for curr_role, tgt_role, edge in edges_to_explore:
            instances = self._advance_instances_across_edge(
                instances, curr_role, tgt_role, edge, nodes_cfg,
                rule_name, trigger_role, trigger_ts, helper, caches,
            )
            if instances:
                continue
            return []

        return self._build_match_results_from_instances(
            instances, rule_name, rule, nodes_cfg, root_roles, trigger_ts,
        )
