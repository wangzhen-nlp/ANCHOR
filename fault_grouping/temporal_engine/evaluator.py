import uuid

from datetime import datetime

from fault_grouping.temporal_engine.utils import clone_instance_with_updates, qualify_role_key


class TemporalGraphEngineEvaluatorMixin:
    MISSING_TOPOLOGY_RULE = "missing_topology_rule"

    def _make_rule_debug_trace(self, rule_name, trigger_node, trigger_ts):
        return {
            "rule": rule_name,
            "trigger_node": trigger_node,
            "trigger_ts": trigger_ts,
            "trigger_role": None,
            "trigger_validation": None,
            "edges": [],
            "raw_match_count": 0,
        }

    def _get_eval_plan(self, rule_name, rule):
        plan = self.rule_execution_plans.get(rule_name)
        if plan is None:
            plan = self._compile_rule_execution_plan(rule)
            self.rule_execution_plans[rule_name] = plan
        return plan

    def _validate_trigger_node_for_rule(
        self,
        rule_name,
        nodes_cfg,
        trigger_role,
        trigger_node,
        trigger_ts,
        helper,
        validation_cache,
        debug_trace=None,
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

        if debug_trace is not None:
            debug_trace["trigger_validation"] = helper.explain_node_validation(
                trigger_node,
                trigger_node_domain,
                nodes_cfg[trigger_role],
                trigger_ts,
                edge_window=0,
                exclude_consumed_trigger_rule=rule_name,
            )

        return is_valid, trigger_events

    def _collect_edge_candidates(
        self,
        curr_phys,
        curr_role,
        tgt_role,
        edge,
        tgt_cfg,
        ref_ts,
        helper,
        traversal_cache,
        path_validation_cache,
        structure_match_cache,
        filtered_neighbor_cache,
        rule_name=None,
        match_mode="ANY",
    ):
        selector = edge.get("candidate_selector") or {}
        selector_mode = selector.get("mode", "default")
        had_topology_candidate = None

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
        else:
            if match_mode == "ALL":
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
            else:
                candidate_hops = self._traverse_graph_role_filtered(
                    rule_name=rule_name,
                    start_node=curr_phys,
                    target_role=tgt_role,
                    direction=edge["traverse_dir"],
                    max_hops=edge["hops"],
                    reference_ts=ref_ts,
                    edge_window=edge["win"],
                    path_requirements=edge.get("path_requirements"),
                    node_rule_helper=helper,
                    traversal_cache=traversal_cache,
                    path_validation_cache=path_validation_cache,
                    filtered_neighbor_cache=filtered_neighbor_cache,
                    target_node_config=tgt_cfg,
                )
            candidate_hops, symmetric_deduped_count = self._filter_symmetric_pair_candidates(
                candidate_hops,
                curr_role,
                tgt_role,
                edge,
                curr_phys,
            )
            raw_candidates = sorted(candidate_hops.keys(), key=lambda n: (candidate_hops[n], str(n)))
            candidates = helper.select_candidates_by_rule(
                list(raw_candidates),
                candidate_hops,
                tgt_cfg,
                edge.get("candidate_selector"),
            )

        return {
            "candidate_hops": candidate_hops,
            "raw_candidates": raw_candidates,
            "candidates": candidates,
            "had_topology_candidate": had_topology_candidate,
            "selector_mode": selector_mode,
            "symmetric_deduped_count": symmetric_deduped_count,
        }

    def _candidate_support_count(
        self,
        rule_name,
        role,
        site_id,
        nodes_cfg,
        trigger_role,
        reference_ts,
        helper,
        caches,
    ):
        cache_key = (rule_name, role, site_id, reference_ts)
        support_count_cache = caches.get("support_count_cache", {})
        if cache_key in support_count_cache:
            return support_count_cache[cache_key]

        total = 0
        for edge in self._get_rule_pattern_adj(rule_name).get(role, ()):
            neighbor_role = edge["role"]
            neighbor_cfg = nodes_cfg[neighbor_role]
            candidate_hops = self._traverse_graph_role_filtered(
                rule_name=rule_name,
                start_node=site_id,
                target_role=neighbor_role,
                direction=edge["traverse_dir"],
                max_hops=edge["hops"],
                reference_ts=reference_ts,
                edge_window=edge["win"],
                path_requirements=edge.get("path_requirements"),
                node_rule_helper=helper,
                traversal_cache=caches["traversal_cache"],
                path_validation_cache=caches["path_validation_cache"],
                filtered_neighbor_cache=caches["filtered_neighbor_cache"],
                target_node_config=neighbor_cfg,
            )
            for neighbor_site in candidate_hops:
                valid, _events = self._validate_node_cached_for_support(
                    neighbor_site,
                    neighbor_role,
                    neighbor_cfg,
                    edge,
                    rule_name,
                    trigger_role,
                    reference_ts,
                    helper,
                    caches["validation_cache"],
                )
                if valid:
                    total += 1
        support_count_cache[cache_key] = total
        return total

    def _validate_node_cached_for_support(
        self,
        site_id,
        role,
        node_config,
        edge,
        rule_name,
        trigger_role,
        reference_ts,
        helper,
        validation_cache,
    ):
        window_cache_key = self._make_edge_window_cache_key(edge["win"])
        exclude_consumed_trigger_rule = rule_name if role == trigger_role else None
        if exclude_consumed_trigger_rule is None:
            cache_key = (
                "support_node_shared",
                id(node_config),
                site_id,
                reference_ts,
                window_cache_key,
            )
        else:
            cache_key = (
                "support_node",
                rule_name,
                role,
                id(node_config),
                site_id,
                reference_ts,
                window_cache_key,
            )
        if cache_key in validation_cache:
            return validation_cache[cache_key]
        site_domain = self.sites_domain_map.get(site_id, {})
        result = helper.validate_node(
            site_id,
            site_domain,
            node_config,
            reference_ts,
            edge["win"],
            exclude_consumed_trigger_rule=exclude_consumed_trigger_rule,
        )
        validation_cache[cache_key] = result
        return result

    def _get_rule_pattern_adj(self, rule_name):
        plan = self.rule_execution_plans.get(rule_name) or self._get_eval_plan(rule_name, self.rules[rule_name])
        pattern_adj = plan.get("pattern_adj")
        if pattern_adj is None:
            pattern_adj = {}
        return pattern_adj

    def _candidate_has_required_support(
        self,
        rule_name,
        role,
        site_id,
        nodes_cfg,
        trigger_role,
        reference_ts,
        helper,
        caches,
        bound_roles=None,
    ):
        bound_roles = bound_roles or set()
        cache_key = (
            rule_name,
            role,
            site_id,
            reference_ts,
            tuple(sorted(bound_roles)),
        )
        support_cache = caches.get("support_cache", {})
        if cache_key in support_cache:
            return support_cache[cache_key]

        for edge in self._get_rule_pattern_adj(rule_name).get(role, ()):
            if edge.get("optional"):
                continue
            neighbor_role = edge["role"]
            if neighbor_role in bound_roles:
                continue
            neighbor_cfg = nodes_cfg[neighbor_role]
            candidate_hops = self._traverse_graph_role_filtered(
                rule_name=rule_name,
                start_node=site_id,
                target_role=neighbor_role,
                direction=edge["traverse_dir"],
                max_hops=edge["hops"],
                reference_ts=reference_ts,
                edge_window=edge["win"],
                path_requirements=edge.get("path_requirements"),
                node_rule_helper=helper,
                traversal_cache=caches["traversal_cache"],
                path_validation_cache=caches["path_validation_cache"],
                filtered_neighbor_cache=caches["filtered_neighbor_cache"],
                target_node_config=neighbor_cfg,
            )
            has_support = False
            for neighbor_site in candidate_hops:
                valid, _events = self._validate_node_cached_for_support(
                    neighbor_site,
                    neighbor_role,
                    neighbor_cfg,
                    edge,
                    rule_name,
                    trigger_role,
                    reference_ts,
                    helper,
                    caches["validation_cache"],
                )
                if valid:
                    has_support = True
                    break
            if not has_support:
                support_cache[cache_key] = False
                return False

        support_cache[cache_key] = True
        return True

    def _sort_candidates_by_support_count(
        self,
        candidates,
        rule_name,
        role,
        nodes_cfg,
        trigger_role,
        reference_ts,
        helper,
        caches,
    ):
        if len(candidates) <= 1:
            return candidates
        return sorted(
            candidates,
            key=lambda site_id: (
                self._candidate_support_count(
                    rule_name,
                    role,
                    site_id,
                    nodes_cfg,
                    trigger_role,
                    reference_ts,
                    helper,
                    caches,
                ),
                str(site_id),
            ),
        )

    def _append_candidate_collection_failures(
        self,
        curr_role,
        tgt_role,
        curr_phys,
        edge,
        candidate_info,
        edge_trace,
        branch_failure_reasons,
    ):
        if edge_trace is None:
            return False

        symmetric_deduped_count = candidate_info["symmetric_deduped_count"]
        if symmetric_deduped_count:
            edge_trace["failures"].append(
                f"{curr_role}:{curr_phys} 的 {tgt_role} 候选因 symmetric pair 去重过滤 {symmetric_deduped_count} 个"
            )

        raw_candidates = candidate_info["raw_candidates"]
        candidates = candidate_info["candidates"]
        selector_mode = candidate_info["selector_mode"]
        if selector_mode == "nearest_matching":
            if not raw_candidates:
                if candidate_info["had_topology_candidate"]:
                    branch_failure_reasons.append(
                        f"{curr_role}:{curr_phys} 的 {tgt_role} 候选在 selector(mode={selector_mode}) 后为空，"
                        "原始拓扑候选存在，但没有结构命中"
                    )
                else:
                    branch_failure_reasons.append(
                        f"{curr_role}:{curr_phys} 在拓扑方向 {edge['traverse_dir']} 上找不到任何 {tgt_role} 候选节点"
                    )
                return True
            return False

        if not raw_candidates:
            branch_failure_reasons.append(
                f"{curr_role}:{curr_phys} 在拓扑方向 {edge['traverse_dir']} 上找不到任何 {tgt_role} 候选节点"
            )
            return True
        if raw_candidates and not candidates:
            branch_failure_reasons.append(
                f"{curr_role}:{curr_phys} 的 {tgt_role} 候选在 selector(mode={selector_mode}) 后为空，"
                f"原始候选={raw_candidates[:8]}"
            )
            return True
        return False

    def _validate_candidate_nodes_for_edge(
        self,
        candidates,
        tgt_role,
        tgt_cfg,
        nodes_cfg,
        ref_ts,
        edge,
        rule_name,
        trigger_role,
        match_mode,
        helper,
        validation_cache,
        caches=None,
        bound_roles=None,
        edge_trace=None,
        allowed_alarm_source_nes=None,
    ):
        curr_valid_targets = {}
        all_passed = True
        window_cache_key = self._make_edge_window_cache_key(edge["win"])
        candidate_failure_details = []

        # 当 NE 锚点约束生效时，allowed_alarm_source_nes 是 frozenset；为 None 时不过滤。
        # cache_key 必须区分这一维度，否则不同 anchor 绑定的实例会复用错误结果。
        ne_filter_key = allowed_alarm_source_nes if allowed_alarm_source_nes is not None else None

        for cand_phys in candidates:
            exclude_consumed_trigger_rule = rule_name if tgt_role == trigger_role else None
            if exclude_consumed_trigger_rule is None:
                cache_key = (
                    "candidate_node_shared",
                    id(tgt_cfg),
                    cand_phys,
                    ref_ts,
                    window_cache_key,
                    ne_filter_key,
                )
            else:
                cache_key = (
                    "candidate_node",
                    rule_name,
                    tgt_role,
                    id(tgt_cfg),
                    cand_phys,
                    ref_ts,
                    window_cache_key,
                    ne_filter_key,
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
                support_ref_ts = events[0]["ts"] if events else ref_ts
                if (
                    self.enable_support_pruning
                    and match_mode != "ALL"
                    and caches is not None
                    and not self._candidate_has_required_support(
                        rule_name,
                        tgt_role,
                        cand_phys,
                        nodes_cfg,
                        trigger_role,
                        support_ref_ts,
                        helper,
                        caches,
                        bound_roles=bound_roles,
                    )
                ):
                    if edge_trace is not None:
                        candidate_failure_details.append(
                            f"{cand_phys}: 缺少必选邻接 role 的活跃支撑，support check 剪枝"
                        )
                    continue
                curr_valid_targets[cand_phys] = events
                continue

            if edge_trace is not None:
                explain = helper.explain_node_validation(
                    cand_phys,
                    self.sites_domain_map.get(cand_phys, {}),
                    tgt_cfg,
                    ref_ts,
                    edge["win"],
                    exclude_consumed_trigger_rule=exclude_consumed_trigger_rule,
                )
                candidate_failure_details.append(
                    f"{cand_phys}: {explain.get('reason', '节点校验失败')}"
                )
            if match_mode == "ALL":
                all_passed = False
                break

        return curr_valid_targets, all_passed, candidate_failure_details

    @staticmethod
    def _normalize_mutual_alarm_source_ne_anchor_config(edge):
        cfg = edge.get("mutual_alarm_source_ne_anchor")
        if not cfg:
            return None
        if cfg is True:
            cfg = {}
        if not isinstance(cfg, dict):
            return None
        return {
            "max_ne_hops": int(cfg.get("max_ne_hops", 1)),
        }

    def _validate_role_node_with_ne_anchor(
        self,
        phys_node,
        role_cfg,
        reference_ts,
        edge_window,
        helper,
        allowed_alarm_source_nes,
        rule_name=None,
        role_name=None,
        trigger_role=None,
        validation_cache=None,
    ):
        exclude_consumed_trigger_rule = rule_name if role_name == trigger_role else None
        if validation_cache is None:
            return helper.validate_node(
                phys_node,
                self.sites_domain_map.get(phys_node, {}),
                role_cfg,
                reference_ts,
                edge_window,
                exclude_consumed_trigger_rule=exclude_consumed_trigger_rule,
                allowed_alarm_source_nes=allowed_alarm_source_nes,
            )

        cache_key = (
            "mutual_ne_anchor_node",
            rule_name if exclude_consumed_trigger_rule else None,
            role_name if exclude_consumed_trigger_rule else None,
            id(role_cfg),
            phys_node,
            reference_ts,
            self._make_edge_window_cache_key(edge_window),
            allowed_alarm_source_nes,
        )
        cached = validation_cache.get(cache_key)
        if cached is not None:
            return cached

        result = helper.validate_node(
            phys_node,
            self.sites_domain_map.get(phys_node, {}),
            role_cfg,
            reference_ts,
            edge_window,
            exclude_consumed_trigger_rule=exclude_consumed_trigger_rule,
            allowed_alarm_source_nes=allowed_alarm_source_nes,
        )
        validation_cache[cache_key] = result
        return result

    def _alarm_source_nes_within_hops(self, source_ne, target_ne, max_hops):
        if source_ne and target_ne:
            left_ne, right_ne = sorted((source_ne, target_ne), key=str)
            cache_key = (left_ne, right_ne, max_hops)
        else:
            cache_key = (source_ne, target_ne, max_hops)
        cached = self._alarm_source_ne_reachability_cache.get(cache_key)
        if cached is not None:
            return cached
        if not source_ne or not target_ne:
            self._alarm_source_ne_reachability_cache[cache_key] = False
            return False
        if source_ne == target_ne:
            self._alarm_source_ne_reachability_cache[cache_key] = True
            return True
        if max_hops <= 0 or not self._ne_adjacency:
            self._alarm_source_ne_reachability_cache[cache_key] = False
            return False
        visited = {source_ne}
        frontier = {source_ne}
        for _ in range(max_hops):
            next_frontier = set()
            for ne_id in frontier:
                for neighbor in self._ne_adjacency.get(ne_id, ()):
                    if neighbor == target_ne:
                        self._alarm_source_ne_reachability_cache[cache_key] = True
                        return True
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        self._alarm_source_ne_reachability_cache[cache_key] = False
        return False

    def _get_required_alarm_set_for_role(self, phys_node, role_cfg, helper):
        cache = getattr(self, "_required_alarm_set_cache", None)
        cache_key = (id(role_cfg), phys_node)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        expected = helper.resolve_expected_alarms(
            self.sites_domain_map.get(phys_node, {}),
            role_cfg,
        )
        required_alarms = expected.get("required_alarms") if isinstance(expected, dict) else None
        result = frozenset(required_alarms) if required_alarms else None
        if cache is not None:
            cache[cache_key] = result
        return result

    def _get_required_alarm_events_for_role(self, phys_node, role_cfg, events, helper):
        required_alarms = self._get_required_alarm_set_for_role(phys_node, role_cfg, helper)
        if not required_alarms:
            return list(events)
        return [
            event for event in events
            if event.get("alarm") in required_alarms
        ]

    @staticmethod
    def _alarm_sources_from_events(events):
        return {
            event.get("alarm_source")
            for event in events
            if event.get("alarm_source")
        }

    def _filter_events_by_supported_alarm_sources(self, events, supported_alarm_sources):
        filtered = [
            event for event in events
            if event.get("alarm_source") in supported_alarm_sources
        ]
        return filtered

    def _event_alarm_sources_known_and_disjoint(self, events, allowed_alarm_source_nes):
        if allowed_alarm_source_nes is None:
            return False
        event_sources = self._alarm_sources_from_events(events)
        if not event_sources:
            return False
        return event_sources.isdisjoint(allowed_alarm_source_nes)

    def _filter_mutual_ne_supported_events(
        self,
        curr_phys,
        curr_cfg,
        curr_events,
        target_phys,
        tgt_cfg,
        target_events,
        helper,
        max_ne_hops,
    ):
        # 没有 NE 数据时与 role-level anchor 一致：降级为不做 NE 级精确裁剪。
        if not self._site_to_ne_ids:
            return curr_events, target_events

        curr_required_events = self._get_required_alarm_events_for_role(
            curr_phys, curr_cfg, curr_events, helper
        )
        target_required_events = self._get_required_alarm_events_for_role(
            target_phys, tgt_cfg, target_events, helper
        )
        if not curr_required_events or not target_required_events:
            return [], []

        curr_required_sources = self._alarm_sources_from_events(curr_required_events)
        target_required_sources = self._alarm_sources_from_events(target_required_events)
        if not curr_required_sources or not target_required_sources:
            return [], []

        curr_supported_sources = set()
        target_supported_sources = set()
        for curr_alarm_source in curr_required_sources:
            for target_alarm_source in target_required_sources:
                if self._alarm_source_nes_within_hops(curr_alarm_source, target_alarm_source, max_ne_hops):
                    curr_supported_sources.add(curr_alarm_source)
                    target_supported_sources.add(target_alarm_source)

        if not curr_supported_sources or not target_supported_sources:
            return [], []

        return (
            self._filter_events_by_supported_alarm_sources(curr_events, curr_supported_sources),
            self._filter_events_by_supported_alarm_sources(target_events, target_supported_sources),
        )

    def _apply_mutual_alarm_source_ne_anchor_for_edge(
        self,
        curr_phys,
        curr_events,
        curr_role,
        curr_cfg,
        tgt_role,
        tgt_cfg,
        curr_valid_targets,
        edge,
        rule_name,
        trigger_role,
        ref_ts,
        helper,
        validation_cache=None,
        edge_trace=None,
    ):
        anchor_cfg = self._normalize_mutual_alarm_source_ne_anchor_config(edge)
        if not anchor_cfg or not curr_valid_targets:
            return curr_valid_targets, {
                target_phys: curr_events
                for target_phys in curr_valid_targets
            }

        max_ne_hops = anchor_cfg["max_ne_hops"]
        filtered_targets = {}
        curr_events_by_target = {}
        failure_details = []
        allowed_for_target = self._compute_anchor_ne_reachable_set(curr_phys, max_ne_hops)

        for target_phys, target_events in curr_valid_targets.items():
            allowed_for_curr = self._compute_anchor_ne_reachable_set(target_phys, max_ne_hops)

            # target 侧不再重 validate：进入这里前 _validate_candidate_nodes_for_edge
            # 已经用 role-level allowed_alarm_source_nes 过滤过 target_events，并完成
            # forbidden/required 等谓词校验。mutual 阶段只需把 events 收敛到当前 curr_phys
            # 单点对应的更紧 allowed_for_target（role-level 在多 anchor 站点场景下会取
            # 并集，比 mutual 略宽；单 anchor 时两者完全一致，filter 为 identity）。
            if allowed_for_target is None:
                filtered_target_events = target_events
            else:
                filtered_target_events = [
                    e for e in target_events
                    if e.get("alarm_source") in allowed_for_target
                ]
            if not filtered_target_events:
                if edge_trace is not None:
                    failure_details.append(
                        f"{target_phys}: {tgt_role} 没有 alarm_source 落在 {curr_phys} 的 NE 拓扑邻域内"
                    )
                continue

            # curr 侧仍需 disjoint 早退 + 重 validate：curr_events 来自 trigger
            # validation（edge_window=0），mutual 需要 edge_window=edge["win"] 的更宽
            # 视角并在新窗口内复查 forbidden。disjoint 检查能在 curr_events 完全不命中
            # 时省掉一次 validate_node 调用。
            if self._event_alarm_sources_known_and_disjoint(curr_events, allowed_for_curr):
                if edge_trace is not None:
                    failure_details.append(
                        f"{curr_phys}: 已命中告警源不在 {target_phys} 的 NE 拓扑邻域内，跳过 mutual anchor 校验"
                    )
                continue

            if allowed_for_curr is None:
                curr_valid, filtered_curr_events = True, curr_events
            else:
                curr_ref_ts = filtered_target_events[0]["ts"] if filtered_target_events else ref_ts
                curr_valid, filtered_curr_events = self._validate_role_node_with_ne_anchor(
                    curr_phys,
                    curr_cfg,
                    curr_ref_ts,
                    edge["win"],
                    helper,
                    allowed_for_curr,
                    rule_name=rule_name,
                    role_name=curr_role,
                    trigger_role=trigger_role,
                    validation_cache=validation_cache,
                )

            if not curr_valid:
                if edge_trace is not None:
                    failure_details.append(
                        f"{curr_phys}: {curr_role} 没有 alarm_source 落在 {target_phys} 的 NE 拓扑邻域内"
                    )
                continue

            filtered_curr_events, filtered_target_events = self._filter_mutual_ne_supported_events(
                curr_phys,
                curr_cfg,
                filtered_curr_events,
                target_phys,
                tgt_cfg,
                filtered_target_events,
                helper,
                max_ne_hops,
            )
            if not filtered_curr_events or not filtered_target_events:
                if edge_trace is not None:
                    failure_details.append(
                        f"{curr_phys}<->{target_phys}: 两侧 required 告警的 alarm_source NE 之间没有拓扑支撑边"
                    )
                continue

            filtered_targets[target_phys] = filtered_target_events
            curr_events_by_target[target_phys] = filtered_curr_events

        if edge_trace is not None and failure_details:
            edge_trace["failures"].extend(failure_details[:4])

        return filtered_targets, curr_events_by_target

    def _evaluate_edge_source_node(
        self,
        curr_phys,
        curr_events,
        curr_role,
        tgt_role,
        tgt_cfg,
        curr_cfg,
        edge,
        rule_name,
        trigger_role,
        trigger_ts,
        match_mode,
        helper,
        nodes_cfg,
        caches,
        bound_roles,
        validation_cache,
        traversal_cache,
        path_validation_cache,
        structure_match_cache,
        filtered_neighbor_cache,
        edge_trace=None,
        allowed_alarm_source_nes=None,
    ):
        ref_ts = curr_events[0]["ts"] if curr_events else trigger_ts
        bound_roles = set(bound_roles or ())
        branch_failure_reasons = []
        candidate_info = self._collect_edge_candidates(
            curr_phys,
            curr_role,
            tgt_role,
            edge,
            tgt_cfg,
            ref_ts,
            helper,
            traversal_cache,
            path_validation_cache,
            structure_match_cache,
            filtered_neighbor_cache,
            rule_name=rule_name,
            match_mode=match_mode,
        )
        if self._append_candidate_collection_failures(
            curr_role,
            tgt_role,
            curr_phys,
            edge,
            candidate_info,
            edge_trace,
            branch_failure_reasons,
        ):
            return {}, branch_failure_reasons, {}

        ordered_candidates = candidate_info["candidates"]
        if self.enable_support_count_sort and match_mode != "ALL":
            ordered_candidates = self._sort_candidates_by_support_count(
                ordered_candidates,
                rule_name,
                tgt_role,
                nodes_cfg,
                trigger_role,
                ref_ts,
                helper,
                caches,
            )

        curr_valid_targets, all_passed, candidate_failure_details = self._validate_candidate_nodes_for_edge(
            ordered_candidates,
            tgt_role,
            tgt_cfg,
            nodes_cfg,
            ref_ts,
            edge,
            rule_name,
            trigger_role,
            match_mode,
            helper,
            validation_cache,
            caches=caches,
            bound_roles=bound_roles | {curr_role},
            edge_trace=edge_trace,
            allowed_alarm_source_nes=allowed_alarm_source_nes,
        )

        curr_valid_targets, curr_events_by_target = self._apply_mutual_alarm_source_ne_anchor_for_edge(
            curr_phys,
            curr_events,
            curr_role,
            curr_cfg,
            tgt_role,
            tgt_cfg,
            curr_valid_targets,
            edge,
            rule_name,
            trigger_role,
            ref_ts,
            helper,
            validation_cache=validation_cache,
            edge_trace=edge_trace,
        )

        if match_mode == "ALL" and not all_passed:
            if edge_trace is not None:
                detail = candidate_failure_details[0] if candidate_failure_details else "存在候选节点未通过 ALL 校验"
                branch_failure_reasons.append(
                    f"{curr_role}:{curr_phys} 在 ALL 模式下失败，{detail}"
                )
            return {}, branch_failure_reasons, {}

        if curr_valid_targets:
            return curr_valid_targets, branch_failure_reasons, curr_events_by_target

        if edge_trace is not None:
            if candidate_failure_details:
                branch_failure_reasons.append(
                    f"{curr_role}:{curr_phys} 没有满足 {tgt_role} 的节点，"
                    f"失败原因: {candidate_failure_details[:3]}"
                )
            else:
                branch_failure_reasons.append(
                    f"{curr_role}:{curr_phys} 没有满足 {tgt_role} 的节点"
                )
        return {}, branch_failure_reasons, {}

    def _compute_allowed_alarm_source_nes_for_role(self, rule_name, tgt_role, inst):
        """若 tgt_role 配置了 alarm_source_ne_anchor，查询 anchor_role 已绑定的 site，
        返回允许的 alarm_source NE 集合（多个 anchor site 合并）。

        返回 None 表示无 NE 锚点约束（或引擎未配置 NE 数据，降级为不过滤）。
        """
        plan = self.rule_execution_plans.get(rule_name)
        if not plan:
            return None
        anchors = plan.get("alarm_source_ne_anchors") or {}
        anchor_cfg = anchors.get(tgt_role)
        if not anchor_cfg:
            return None
        anchor_role = anchor_cfg["anchor_role"]
        max_hops = anchor_cfg["max_ne_hops"]
        anchor_nodes = inst.get("roles", {}).get(anchor_role, {}).get("nodes", {})
        if not anchor_nodes:
            # anchor 尚未绑定（编译期 bind_order 校验排除该路径，这里保留防御性返回 None）
            return None
        anchor_sites = list(anchor_nodes.keys())
        # 单 anchor 时直接复用 _compute_anchor_ne_reachable_set 的 frozenset，
        # 避免一次 set->frozenset 复制；多 anchor 时合并后再 frozenset。
        if len(anchor_sites) == 1:
            return self._compute_anchor_ne_reachable_set(anchor_sites[0], max_hops)
        combined = None
        for anchor_site in anchor_sites:
            reachable = self._compute_anchor_ne_reachable_set(anchor_site, max_hops)
            if reachable is None:
                # 引擎未配置 NE 数据，直接降级为不过滤
                return None
            if combined is None:
                combined = set(reachable)
            else:
                combined.update(reachable)
        return frozenset(combined) if combined is not None else frozenset()

    def _collect_instance_edge_targets(
        self,
        inst,
        curr_role,
        tgt_role,
        edge,
        tgt_cfg,
        rule_name,
        trigger_role,
        trigger_ts,
        match_mode,
        helper,
        nodes_cfg,
        caches,
        validation_cache,
        traversal_cache,
        path_validation_cache,
        structure_match_cache,
        filtered_neighbor_cache,
        edge_trace=None,
    ):
        valid_targets = {}
        surviving_curr_phys = {}
        curr_support_targets = {}
        curr_events_by_target = {}
        branch_failure_reasons = []
        bound_roles = set(inst.get("roles", {}).keys())
        curr_cfg = nodes_cfg[curr_role]

        # 计算 NE 锚点允许 alarm_source NE 集合（如果 tgt_role 配置了 alarm_source_ne_anchor）
        allowed_alarm_source_nes = self._compute_allowed_alarm_source_nes_for_role(
            rule_name, tgt_role, inst,
        )

        for curr_phys, curr_events in inst["roles"][curr_role]["nodes"].items():
            curr_valid_targets, source_failure_reasons, source_events_by_target = self._evaluate_edge_source_node(
                curr_phys,
                curr_events,
                curr_role,
                tgt_role,
                tgt_cfg,
                curr_cfg,
                edge,
                rule_name,
                trigger_role,
                trigger_ts,
                match_mode,
                helper,
                nodes_cfg,
                caches,
                bound_roles,
                validation_cache,
                traversal_cache,
                path_validation_cache,
                structure_match_cache,
                filtered_neighbor_cache,
                edge_trace=edge_trace,
                allowed_alarm_source_nes=allowed_alarm_source_nes,
            )
            branch_failure_reasons.extend(source_failure_reasons)
            if not curr_valid_targets:
                continue

            combined_curr_events = []
            seen_curr_event_ids = set()
            for target_phys in curr_valid_targets:
                for event in source_events_by_target.get(target_phys, curr_events):
                    event_id = event.get("eid") or (
                        event.get("node"),
                        event.get("ts"),
                        event.get("alarm"),
                        event.get("alarm_source"),
                    )
                    if event_id in seen_curr_event_ids:
                        continue
                    seen_curr_event_ids.add(event_id)
                    combined_curr_events.append(event)
                curr_events_by_target[(curr_phys, target_phys)] = source_events_by_target.get(target_phys, curr_events)

            surviving_curr_phys[curr_phys] = combined_curr_events or curr_events
            curr_support_targets[curr_phys] = set(curr_valid_targets)
            for key, value in curr_valid_targets.items():
                valid_targets[key] = value

        return valid_targets, surviving_curr_phys, curr_support_targets, curr_events_by_target, branch_failure_reasons

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
        edge_trace=None,
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
            if edge_trace is not None:
                edge_trace["failures"].append(
                    f"{tgt_role} 合并后仅有 {len(merged_targets)} 个节点，低于 min_count={min_count}"
                )
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
        self,
        inst,
        curr_role,
        tgt_role,
        edge,
        nodes_cfg,
        rule_name,
        trigger_role,
        trigger_ts,
        helper,
        validation_cache,
        traversal_cache,
        path_validation_cache,
        structure_match_cache,
        filtered_neighbor_cache,
        caches,
        edge_trace=None,
    ):
        inst_roles = inst["roles"]

        if curr_role not in inst_roles:
            if edge_trace is not None:
                edge_trace["failures"].append(
                    f"实例缺少源 role {curr_role}，无法继续扩展到 {tgt_role}"
                )
            return [inst]

        if tgt_role in inst_roles and inst_roles[tgt_role]["checked"]:
            return [inst]

        tgt_cfg = nodes_cfg[tgt_role]
        valid_targets, surviving_curr_phys, curr_support_targets, curr_events_by_target, branch_failure_reasons = (
            self._collect_instance_edge_targets(
                inst,
                curr_role,
                tgt_role,
                edge,
                tgt_cfg,
                rule_name,
                trigger_role,
                trigger_ts,
                tgt_cfg.get("match", "ANY"),
                helper,
                nodes_cfg,
                caches,
                validation_cache,
                traversal_cache,
                path_validation_cache,
                structure_match_cache,
                filtered_neighbor_cache,
                edge_trace=edge_trace,
            )
        )

        if not valid_targets:
            if edge_trace is not None and branch_failure_reasons:
                edge_trace["failures"].extend(branch_failure_reasons[:6])
            if edge.get("optional"):
                if edge_trace is not None:
                    edge_trace["failures"].append(
                        f"可选边 {curr_role}->{tgt_role} 未命中，保留当前实例"
                    )
                return [inst]
            return []

        curr_cfg = nodes_cfg[curr_role]
        if inst_roles[curr_role]["checked"] and len(surviving_curr_phys) < curr_cfg.get("min_count", 1):
            if edge_trace is not None:
                edge_trace["failures"].append(
                    f"{curr_role} 回溯后仅剩 {len(surviving_curr_phys)} 个节点，"
                    f"低于 min_count={curr_cfg.get('min_count', 1)}"
                )
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
            edge_trace=edge_trace,
        )

    def _fork_primitive_target_instances(
        self,
        inst,
        curr_role,
        tgt_role,
        surviving_curr_phys,
        valid_targets,
        curr_support_targets,
        nodes_cfg,
        curr_events_by_target=None,
    ):
        next_instances = []
        curr_events_by_target = curr_events_by_target or {}
        for target_node, target_events in valid_targets.items():
            target_surviving_curr_phys = {
                curr_node: curr_events_by_target.get((curr_node, target_node), curr_events)
                for curr_node, curr_events in surviving_curr_phys.items()
                if target_node in curr_support_targets.get(curr_node, set())
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
                    curr_node: ({target_node} if target_node in target_nodes else set())
                    for curr_node, target_nodes in curr_support_targets.items()
                },
            )
            stabilized_inst = self._stabilize_instance_dependencies(new_inst, nodes_cfg)
            if stabilized_inst is not None:
                next_instances.append(stabilized_inst)
        return next_instances

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
        debug_trace=None,
    ):
        edge_trace = None
        if debug_trace is not None:
            edge_trace = {
                "from_role": curr_role,
                "to_role": tgt_role,
                "instances_in": len(instances),
                "instances_out": 0,
                "failures": [],
            }

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
                    caches["path_validation_cache"],
                    caches["structure_match_cache"],
                    caches["filtered_neighbor_cache"],
                    caches,
                    edge_trace=edge_trace,
                )
            )

        if edge_trace is not None:
            edge_trace["instances_out"] = len(next_instances)
            edge_trace["failures"] = edge_trace["failures"][:8]
            debug_trace["edges"].append(edge_trace)
        return next_instances, edge_trace

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
        hit_event_id = event.get("eid")
        event_enriched["eid_list"] = [hit_event_id] if hit_event_id not in (None, "") else []

        if not self.use_alarm_period_cache:
            alarm_key = hit_event_id
            if alarm_key in (None, ""):
                return
            symptoms_by_key[alarm_key] = event_enriched
            return

        source_segment_key = (
            event.get("_segment_key")
            or event.get("eid")
            or (event.get("node"), event.get("ts"), event.get("alarm"), event.get("alarm_source"))
        )
        event_enriched["_segment_start_ts"] = event["ts"]
        event_enriched["_segment_end_ts"] = event["ts"]
        event_enriched["_segment_key"] = self._build_output_symptom_interval_key(event_enriched)

        existing_symptom = symptoms_by_key.get(source_segment_key)
        if existing_symptom is None:
            symptoms_by_key[source_segment_key] = event_enriched
            return

        current_start_ts = existing_symptom.get("_segment_start_ts", existing_symptom.get("ts"))
        current_end_ts = existing_symptom.get("_segment_end_ts", current_start_ts)
        hit_ts = event["ts"]
        if hit_event_id not in (None, ""):
            existing_eid_list = existing_symptom.setdefault("eid_list", [])
            if hit_event_id not in existing_eid_list:
                existing_eid_list.append(hit_event_id)

        if current_start_ts is None or hit_ts < current_start_ts:
            existing_symptom["ts"] = hit_ts
            existing_symptom["eid"] = hit_event_id
            existing_symptom["time_str"] = event_enriched["time_str"]
            existing_symptom["_segment_start_ts"] = hit_ts
        if current_end_ts is None or hit_ts > current_end_ts:
            existing_symptom["_segment_end_ts"] = hit_ts

        existing_symptom["_segment_key"] = self._build_output_symptom_interval_key(existing_symptom)

    def _get_missing_topology_edge_meta_for_direction(self, source_site, target_site, direction):
        if not self.missing_topology_edges:
            return None
        source_site = str(source_site or "").strip()
        target_site = str(target_site or "").strip()
        if not source_site or not target_site or source_site == target_site:
            return None

        for single_direction in self._normalize_traverse_directions(direction):
            if single_direction == "upstream":
                meta = self.missing_topology_edges.get((target_site, source_site))
            elif single_direction in {"bidirection", "bidirectional"}:
                meta = self.missing_topology_edges.get((source_site, target_site))
                if meta and meta.get("relation") != "bidirection":
                    meta = None
            elif single_direction == "either":
                meta = (
                    self.missing_topology_edges.get((source_site, target_site))
                    or self.missing_topology_edges.get((target_site, source_site))
                )
            else:
                meta = self.missing_topology_edges.get((source_site, target_site))
            if meta:
                return dict(meta)
        return None

    def _collect_missing_topology_edges_for_match(self, role_mapping, rule):
        if not self.missing_topology_edges:
            return []

        used_edges = {}
        for edge in rule.get("edges", []):
            source_role = edge.get("source")
            target_role = edge.get("target")
            if not source_role or not target_role:
                continue
            for source_site in role_mapping.get(source_role, []):
                for target_site in role_mapping.get(target_role, []):
                    meta = self._get_missing_topology_edge_meta_for_direction(
                        source_site,
                        target_site,
                        edge.get("direction", "downstream"),
                    )
                    if not meta:
                        continue
                    edge_key = (
                        meta.get("source_site", ""),
                        meta.get("target_site", ""),
                        meta.get("relation", ""),
                    )
                    used_edges[edge_key] = {
                        **meta,
                        "source_role": source_role,
                        "target_role": target_role,
                        "rule_direction": edge.get("direction", "downstream"),
                    }

        return sorted(
            used_edges.values(),
            key=lambda item: (
                str(item.get("source_site", "")),
                str(item.get("target_site", "")),
                str(item.get("relation", "")),
            ),
        )

    def _build_match_result_from_instance(self, inst, rule_name, rule, root_roles, trigger_ts):
        inst_roles = inst["roles"]
        inferred_roots = {
            root_role: list(inst_roles.get(root_role, {}).get("nodes", {}).keys())
            for root_role in root_roles
        }
        symptoms_by_key, role_mapping = self._build_symptoms_and_role_mapping_from_instance(inst_roles, rule_name)
        match_result = {
            "uuid": str(uuid.uuid4()),
            "rule": rule_name,
            "merged_rules": [rule_name],
            "inferred_roots": inferred_roots,
            "role_mapping": role_mapping,
            "symptoms": list(symptoms_by_key.values()),
            "_expire_ts_hint": (
                min((symptom["ts"] for symptom in symptoms_by_key.values() if "ts" in symptom), default=trigger_ts)
                + rule.get("max_stay_time_sec", self.global_ttl)
            ),
        }
        missing_topology_edges = self._collect_missing_topology_edges_for_match(role_mapping, rule)
        if missing_topology_edges:
            match_result["uses_missing_topology"] = True
            match_result["missing_topology_edges"] = missing_topology_edges
            match_result["merged_rules"] = [rule_name, self.MISSING_TOPOLOGY_RULE]
        return match_result

    def _build_match_results_from_instances(
        self,
        instances,
        rule_name,
        rule,
        nodes_cfg,
        root_roles,
        trigger_ts,
        debug_trace=None,
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
            is_valid_result, result_failure_reason = self._validate_result_constraints(rule, match_result)
            if not is_valid_result:
                if debug_trace is not None and result_failure_reason:
                    debug_trace.setdefault("result_constraint_failures", []).append(result_failure_reason)
                continue
            results.append(match_result)
        return results

    def _finalize_rule_debug_trace(self, debug_trace, results):
        if debug_trace is None:
            return
        debug_trace["raw_match_count"] = len(results)
        if results or "final_reason" in debug_trace:
            return
        result_constraint_failures = debug_trace.get("result_constraint_failures", [])
        if result_constraint_failures:
            debug_trace["final_reason"] = (
                "规则评估完成，但候选组被后置约束过滤；"
                f"主要原因: {result_constraint_failures[:3]}"
            )
        else:
            debug_trace["final_reason"] = "规则评估完成，但未产出原始候选组"

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
        debug_trace = (
            self._make_rule_debug_trace(rule_name, trigger_node, trigger_ts)
            if return_debug_trace else None
        )

        plan = self._get_eval_plan(rule_name, rule)
        trigger_role = plan["trigger_role"]
        edges_to_explore = plan["edges_to_explore"]
        root_roles = plan["root_roles"]
        if debug_trace is not None:
            debug_trace["trigger_role"] = trigger_role

        caches = eval_caches or self._create_eval_caches()
        is_trigger_valid, trigger_events = self._validate_trigger_node_for_rule(
            rule_name,
            nodes_cfg,
            trigger_role,
            trigger_node,
            trigger_ts,
            helper,
            caches["validation_cache"],
            debug_trace=debug_trace,
        )
        if not is_trigger_valid:
            if debug_trace is not None:
                debug_trace["final_reason"] = (
                    debug_trace["trigger_validation"].get("reason", "trigger 节点未通过校验")
                )
                return [], debug_trace
            return []

        if self.enable_support_pruning and not self._candidate_has_required_support(
            rule_name,
            trigger_role,
            trigger_node,
            nodes_cfg,
            trigger_role,
            trigger_ts,
            helper,
            caches,
            bound_roles={trigger_role},
        ):
            if debug_trace is not None:
                debug_trace["final_reason"] = "trigger 节点缺少必选邻接 role 的活跃支撑，support check 剪枝"
                return [], debug_trace
            return []

        instances = [{
            "roles": {
                trigger_role: {"nodes": {trigger_node: trigger_events}, "checked": False}
            },
            "_dependencies": {},
        }]

        for curr_role, tgt_role, edge in edges_to_explore:
            instances, edge_trace = self._advance_instances_across_edge(
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
                debug_trace=debug_trace,
            )
            if instances:
                continue

            if debug_trace is not None:
                if edge_trace and edge_trace["failures"]:
                    debug_trace["final_reason"] = (
                        f"在边 {curr_role} -> {tgt_role} 上全部分支失效；"
                        f"主要原因: {edge_trace['failures'][:3]}"
                    )
                else:
                    debug_trace["final_reason"] = f"在边 {curr_role} -> {tgt_role} 上全部分支失效"
                return [], debug_trace
            return []

        results = self._build_match_results_from_instances(
            instances,
            rule_name,
            rule,
            nodes_cfg,
            root_roles,
            trigger_ts,
            debug_trace=debug_trace,
        )
        self._finalize_rule_debug_trace(debug_trace, results)
        if debug_trace is not None:
            return results, debug_trace
        return results
