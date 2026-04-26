import uuid

from datetime import datetime

from fault_grouping.temporal_engine.utils import clone_instance_with_updates


class TemporalGraphEngineEvaluatorMixin:
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
        ref_ts,
        edge,
        rule_name,
        trigger_role,
        match_mode,
        helper,
        validation_cache,
        edge_trace=None,
    ):
        curr_valid_targets = {}
        all_passed = True
        window_cache_key = self._make_edge_window_cache_key(edge["win"])
        candidate_failure_details = []

        for cand_phys in candidates:
            cache_key = (cand_phys, tgt_role, ref_ts, window_cache_key)
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
                    exclude_consumed_trigger_rule=(rule_name if tgt_role == trigger_role else None),
                )
                validation_cache[cache_key] = (is_valid, events)

            if is_valid:
                curr_valid_targets[cand_phys] = events
                continue

            if edge_trace is not None:
                explain = helper.explain_node_validation(
                    cand_phys,
                    self.sites_domain_map.get(cand_phys, {}),
                    tgt_cfg,
                    ref_ts,
                    edge["win"],
                    exclude_consumed_trigger_rule=(rule_name if tgt_role == trigger_role else None),
                )
                candidate_failure_details.append(
                    f"{cand_phys}: {explain.get('reason', '节点校验失败')}"
                )
            if match_mode == "ALL":
                all_passed = False
                break

        return curr_valid_targets, all_passed, candidate_failure_details

    def _evaluate_edge_source_node(
        self,
        curr_phys,
        curr_events,
        curr_role,
        tgt_role,
        tgt_cfg,
        edge,
        rule_name,
        trigger_role,
        trigger_ts,
        match_mode,
        helper,
        validation_cache,
        traversal_cache,
        path_validation_cache,
        structure_match_cache,
        filtered_neighbor_cache,
        edge_trace=None,
    ):
        ref_ts = curr_events[0]["ts"] if curr_events else trigger_ts
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
            return {}, branch_failure_reasons

        curr_valid_targets, all_passed, candidate_failure_details = self._validate_candidate_nodes_for_edge(
            candidate_info["candidates"],
            tgt_role,
            tgt_cfg,
            ref_ts,
            edge,
            rule_name,
            trigger_role,
            match_mode,
            helper,
            validation_cache,
            edge_trace=edge_trace,
        )

        if match_mode == "ALL" and not all_passed:
            if edge_trace is not None:
                detail = candidate_failure_details[0] if candidate_failure_details else "存在候选节点未通过 ALL 校验"
                branch_failure_reasons.append(
                    f"{curr_role}:{curr_phys} 在 ALL 模式下失败，{detail}"
                )
            return {}, branch_failure_reasons

        if curr_valid_targets:
            return curr_valid_targets, branch_failure_reasons

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
        return {}, branch_failure_reasons

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
        branch_failure_reasons = []

        for curr_phys, curr_events in inst["roles"][curr_role]["nodes"].items():
            curr_valid_targets, source_failure_reasons = self._evaluate_edge_source_node(
                curr_phys,
                curr_events,
                curr_role,
                tgt_role,
                tgt_cfg,
                edge,
                rule_name,
                trigger_role,
                trigger_ts,
                match_mode,
                helper,
                validation_cache,
                traversal_cache,
                path_validation_cache,
                structure_match_cache,
                filtered_neighbor_cache,
                edge_trace=edge_trace,
            )
            branch_failure_reasons.extend(source_failure_reasons)
            if not curr_valid_targets:
                continue

            surviving_curr_phys[curr_phys] = curr_events
            curr_support_targets[curr_phys] = set(curr_valid_targets)
            for key, value in curr_valid_targets.items():
                valid_targets[key] = value

        return valid_targets, surviving_curr_phys, curr_support_targets, branch_failure_reasons

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
        valid_targets, surviving_curr_phys, curr_support_targets, branch_failure_reasons = (
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
    ):
        next_instances = []
        for target_node, target_events in valid_targets.items():
            new_inst = clone_instance_with_updates(
                inst,
                curr_role,
                surviving_curr_phys,
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
                    edge_trace=edge_trace,
                )
            )

        if edge_trace is not None:
            edge_trace["instances_out"] = len(next_instances)
            edge_trace["failures"] = edge_trace["failures"][:8]
            debug_trace["edges"].append(edge_trace)
        return next_instances, edge_trace

    def _build_symptoms_and_role_mapping_from_instance(self, inst_roles):
        symptoms_by_key = {}
        role_mapping = {}

        for role, role_state in inst_roles.items():
            valid_phys_nodes = []
            for phys_node, events in role_state["nodes"].items():
                valid_phys_nodes.append(phys_node)
                for event in events:
                    self._add_event_to_symptom_dict(symptoms_by_key, event, role)
            if valid_phys_nodes:
                role_mapping[role] = valid_phys_nodes

        return symptoms_by_key, role_mapping

    def _add_event_to_symptom_dict(self, symptoms_by_key, event, role):
        event_enriched = dict(event)
        event_enriched["matched_role"] = role
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

    def _build_match_result_from_instance(self, inst, rule_name, rule, root_roles, trigger_ts):
        inst_roles = inst["roles"]
        inferred_roots = {
            root_role: list(inst_roles.get(root_role, {}).get("nodes", {}).keys())
            for root_role in root_roles
        }
        symptoms_by_key, role_mapping = self._build_symptoms_and_role_mapping_from_instance(inst_roles)
        return {
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
