import collections
import heapq
import logging
import time
import threading
import uuid

from datetime import datetime

from emitted_group_store import EmittedGroupStore
from node_rule_helper import NodeRuleHelper
from alarm_types import CRITICAL_ALARMS, POWER_ALARMS
from temporal_graph_engine_utils import (
    build_pattern_adj,
    clone_instance_with_updates,
    matches_expected_alarm,
    merge_match_batch,
)

logger = logging.getLogger(__name__)


class TemporalGraphEngine:
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

    def __init__(self, topo_downstream_map, rules_config, site_domain_map, aggregation_wait_sec=420):
        """初始化拓扑、缓存、触发索引以及历史故障组状态。"""
        # 规则配置总表：按规则名保存匹配图、触发角色和节点约束。
        self.rules = rules_config

        # 建立正反向物理拓扑索引，供多向 BFS 搜索使用
        self.topo_down = topo_downstream_map
        self.topo_up = collections.defaultdict(list)
        for up, downs in self.topo_down.items():
            for down in downs:
                self.topo_up[down].append(up)

        # 状态缓存: { node: deque([(ts, event_id, alarm_type, alarm_source, consumed_trigger_rules)]) }
        self.event_cache = collections.defaultdict(collections.deque)
        # 默认告警缓存保留时长，单位秒
        self.global_ttl = 3600
        # 电源类告警缓存单独保留 3 小时，避免长时间窗根因回看失效
        self.power_alarm_ttl = 10800

        # 站点画像信息：供节点匹配领域使用
        self.sites_domain_map = site_domain_map

        # 全局拓扑穿透缓存
        self.global_topo_cache = collections.OrderedDict()
        self.max_topo_cache_size = 10000

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
            lambda node: self.event_cache.get(node, [])
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
            get_events
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
                    return_debug_trace=True
                )
            else:
                results = self._evaluate_rule(
                    trig_rule_name,
                    rule,
                    trig_node,
                    trigger_ts,
                    node_rule_helper=helper
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

        merged_matches = merge_match_batch(raw_matches)
        expanded_matches = self._expand_matches_with_pending_context(merged_matches, helper)
        with self._lock:
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
                "batch_merged_matches": merged_matches,
                "expanded_matches": expanded_matches,
                "finalized_matches": finalized_matches,
                "finalize_profiles": finalize_profiles,
            })

        return finalized_matches

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

    def _expand_matches_with_pending_context(self, matches, node_rule_helper):
        """只读扩充当前批次：汇总整批非 trigger 节点上的 pending，再和当前批统一做一次批内合并。"""
        if not matches:
            return matches

        non_trigger_nodes = set()
        for match_result in matches:
            non_trigger_nodes.update(self._get_non_trigger_nodes_for_match(match_result))

        if not non_trigger_nodes:
            return matches

        with self._lock:
            pending_candidates = [
                ((node, rule_name), trigger_anchor)
                for (node, rule_name), trigger_anchor in self.pending_triggers.items()
                if node in non_trigger_nodes
            ]

        if not pending_candidates:
            return matches

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
                node_rule_helper=node_rule_helper
            )
            if results:
                extra_matches.extend(results)

        if not extra_matches:
            return matches

        return merge_match_batch(list(matches) + extra_matches)

    def _evaluate_rule(self, rule_name, rule, trigger_node, trigger_ts, node_rule_helper=None, return_debug_trace=False):
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

        # 3. 校验触发节点自身
        trigger_node_domain = self.sites_domain_map.get(trigger_node, {})
        is_trig_valid, trig_evts = helper.validate_node(
            trigger_node,
            trigger_node_domain,
            nodes_cfg[trigger_role],
            trigger_ts,
            edge_window=0,
            exclude_consumed_trigger_rule=rule_name
        )
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
        validation_cache = {}

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
                    candidate_hops = self._traverse_graph(
                        start_node=curr_phys,
                        direction=edge["traverse_dir"],
                        max_hops=edge["hops"],
                        reference_ts=ref_ts,
                        edge_window=edge["win"],
                        path_requirements=edge.get("path_requirements"),
                        node_rule_helper=helper
                    )
                    raw_candidates = sorted(candidate_hops.keys(), key=lambda n: (candidate_hops[n], str(n)))
                    candidates = list(raw_candidates)
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
                        selector = edge.get("candidate_selector") or {}
                        selector_mode = selector.get("mode", "default")
                        branch_failure_reasons.append(
                            f"{curr_role}:{curr_phys} 的 {tgt_role} 候选在 selector(mode={selector_mode}) 后为空，"
                            f"原始候选={raw_candidates[:8]}"
                        )
                        continue

                    curr_valid_targets = {}
                    all_passed = True
                    window_cache_key = (
                        tuple(sorted(edge["win"].items()))
                        if isinstance(edge["win"], dict)
                        else edge["win"]
                    )
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
            results.append(match_result)
        if debug_trace is not None:
            debug_trace["raw_match_count"] = len(results)
            if not results and "final_reason" not in debug_trace:
                debug_trace["final_reason"] = "规则评估完成，但未产出原始候选组"
            return results, debug_trace
        return results

    def _traverse_graph(self, start_node, direction, max_hops=None,
                        reference_ts=None, edge_window=0, 
                        path_requirements=None, node_rule_helper=None):
        """通用的广度优先搜索，支持路径节点约束"""
        helper = node_rule_helper or self.node_rule_helper

        if direction == "self":
            return {start_node: 0}

        cache_key = (start_node, direction, max_hops)

        if path_requirements is None:
            with self._topo_cache_lock:
                if cache_key in self.global_topo_cache:
                    self.global_topo_cache.move_to_end(cache_key)
                    return self.global_topo_cache[cache_key]

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        result = {}

        topo = self.topo_up if direction == "upstream" else self.topo_down

        while queue:
            curr, hops = queue.popleft()
            if hops > 0:
                result[curr] = hops
            if max_hops is None or hops < max_hops:
                for nxt in topo.get(curr, []):
                    if nxt not in visited:
                        if path_requirements is not None:
                            nxt_domain = self.sites_domain_map.get(nxt, {})
                            # 只有路径上的每一跳都过约束，传播才允许继续向外走
                            is_valid_path_node, _ = helper.validate_node(
                                nxt, nxt_domain, path_requirements, reference_ts, edge_window
                            )
                            if not is_valid_path_node:
                                continue

                        visited.add(nxt)
                        queue.append((nxt, hops + 1))

        # 写缓存：仅缓存不带路径约束的纯拓扑结果
        if path_requirements is None:
            with self._topo_cache_lock:
                self.global_topo_cache[cache_key] = result
                self.global_topo_cache.move_to_end(cache_key)
                if len(self.global_topo_cache) > self.max_topo_cache_size:
                    self.global_topo_cache.popitem(last=False)
        return result
