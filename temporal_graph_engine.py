import collections
import heapq
import logging
import time
import threading
import uuid

from datetime import datetime

from emitted_group_store import EmittedGroupStore
from node_rule_helper import NodeRuleHelper
from alarm_types import CRITICAL_ALARMS
from temporal_graph_engine_utils import (
    build_pattern_adj,
    clone_instance_with_updates,
    matches_expected_alarm,
    merge_match_batch,
)

logger = logging.getLogger(__name__)


class TemporalGraphEngine:
    def __init__(self, topo_downstream_map, rules_config, site_domain_map, aggregation_wait_sec=300):
        """初始化拓扑、缓存、触发索引以及历史故障组状态。"""
        # 规则配置总表：按规则名保存匹配图、触发角色和节点约束。
        self.rules = rules_config

        # 建立正反向物理拓扑索引，供多向 BFS 搜索使用
        self.topo_down = topo_downstream_map
        self.topo_up = collections.defaultdict(list)
        for up, downs in self.topo_down.items():
            for down in downs:
                self.topo_up[down].append(up)

        # 状态缓存: { node: deque([(ts, event_id, alarm_type)]) }
        self.event_cache = collections.defaultdict(collections.deque)
        self.global_ttl = 3600

        # 站点画像信息：供节点匹配领域使用
        self.sites_domain_map = site_domain_map

        # 全局拓扑穿透缓存
        self.global_topo_cache = collections.OrderedDict()
        self.max_topo_cache_size = 10000

        # 故障传播等待时间
        self.aggregation_wait_sec = aggregation_wait_sec

        # 流式时间水印
        self.current_watermark = 0.0

        # 延迟触发队列：记录“当前仍在等待聚合”的 trigger 起点锚点，结构为 (ts, seq)
        self.pending_triggers = {}
        # 延迟触发最小堆：按 ready_ts 排序，快速摘取已成熟的 pending trigger
        self.pending_trigger_heap = []
        # 保存某个 (node, rule) 下所有还能作为 trigger 候选的事件，结构为 (ts, event_id, seq)
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

    def process_event(self, node, alarm_type, ts, event_id, is_clear=False, collect_matches=False):
        """接收单条事件并更新内部状态。默认只更新内部状态；当 collect_matches=True 时，会在事件时间点立即收割已成熟的故障组。
        """
        with self._lock:
            # 1. 当前仍保留事件时间水印，便于离线按事件时间回放。
            self.current_watermark = max(self.current_watermark, ts)

            # 2. 先清理过期缓存，再按上报/清除事件更新状态。
            q = self.event_cache[node]
            while q and (ts - q[0][0]) > self.global_ttl:
                q.popleft()
            self._prune_expired_trigger_index(node, ts)

            if is_clear:
                # 清除事件只按 event_id 删除对应实例，并联动修正 trigger / pending 状态。
                self._remove_cleared_events(node, event_id)
                self._remove_cleared_trigger_events(node, event_id)
                self._refresh_pending_triggers_for_node(node)
            else:
                q.append((ts, event_id, alarm_type))

            # 3. 命中 trigger 的事件只负责入 pending，不在这里直接做匹配评估。
            if not is_clear:
                for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
                    if any(matches_expected_alarm(alarm_type, expected) for expected in expected_list):
                        trigger_key = (node, rule_name)
                        self._trigger_seq += 1
                        trigger_seq = self._trigger_seq
                        self.trigger_event_index[trigger_key].append((ts, event_id, trigger_seq))
                        # 如果这段时间内已经触发过，就不更新时间，以“第一声警报”为准
                        if trigger_key not in self.pending_triggers:
                            self._set_pending_trigger(trigger_key, ts, trigger_seq)

        # 离线模式通过事件触发收割
        if collect_matches:
            return self._collect_pending_matches(force=False)

        return []

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

        if force:
            for trigger_key, trigger_anchor in list(self.pending_triggers.items()):
                self.pending_triggers.pop(trigger_key, None)
                mature_items.append((trigger_key, trigger_anchor))
            return mature_items

        while self.pending_trigger_heap:
            ready_ts, first_trigger_ts, trigger_seq, trigger_key = self.pending_trigger_heap[0]
            if ready_ts > self.current_watermark:
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
        for trigger_key, trigger_anchor in mature_items:
            trig_node, trig_rule_name = trigger_key
            rule = self.rules[trig_rule_name]
            trigger_ts, _trigger_seq = trigger_anchor
            results = self._evaluate_rule(
                trig_rule_name,
                rule,
                trig_node,
                trigger_ts,
                node_rule_helper=helper
            )
            if results:
                raw_matches.extend(results)

        merged_matches = merge_match_batch(raw_matches)
        with self._lock:
            self._prune_expired_state_locked(self.current_watermark)
            return self._finalize_matches_with_history(merged_matches)

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

        for cached_ts, cached_eid, cached_alarm_type in q:
            if event_id and cached_eid == event_id:
                continue
            kept.append((cached_ts, cached_eid, cached_alarm_type))

        self.event_cache[node] = kept

    def _prune_node_alarm_history_before(self, node, alarm_type, cutoff_ts):
        """删除某节点同告警名下不晚于 cutoff_ts 的缓存与 trigger 候选。"""
        q = self.event_cache.get(node)
        if not q:
            return

        removed_event_ids = set()
        kept = collections.deque()
        for cached_ts, cached_eid, cached_alarm_type in q:
            if cached_alarm_type == alarm_type and cached_ts <= cutoff_ts:
                if cached_eid not in (None, ""):
                    removed_event_ids.add(cached_eid)
                continue
            kept.append((cached_ts, cached_eid, cached_alarm_type))
        self.event_cache[node] = kept

        if not removed_event_ids:
            return

        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            kept_trigger_events = collections.deque()
            for event_ts, indexed_event_id, indexed_seq in trigger_events:
                if indexed_event_id in removed_event_ids:
                    continue
                kept_trigger_events.append((event_ts, indexed_event_id, indexed_seq))

            if kept_trigger_events:
                self.trigger_event_index[trigger_key] = kept_trigger_events
            else:
                self.trigger_event_index.pop(trigger_key, None)

        self._refresh_pending_triggers_for_node(node)

    def _prune_consumed_alarm_history(self, matches):
        """在本轮定时收割结束时，只回收命中 trigger_role 的节点告警历史。"""
        prune_points = {}
        for match in matches:
            merged_rules = match.get("merged_rules", [match.get("rule")])
            trigger_roles = {
                self.rules[rule_name]["trigger_role"]
                for rule_name in merged_rules
                if rule_name in self.rules and self.rules[rule_name].get("trigger_role")
            }
            for symptom in match.get("symptoms", []):
                matched_role = symptom.get("matched_role")
                if matched_role not in trigger_roles:
                    continue
                node = symptom.get("node")
                alarm_type = symptom.get("alarm")
                ts = symptom.get("ts")
                if node in (None, "") or alarm_type in (None, "") or ts is None:
                    continue
                key = (node, alarm_type)
                prune_points[key] = max(prune_points.get(key, float("-inf")), ts)

        for (node, alarm_type), cutoff_ts in prune_points.items():
            self._prune_node_alarm_history_before(node, alarm_type, cutoff_ts)

    def _prune_expired_trigger_index(self, node, current_ts):
        """清理某个节点 trigger 索引中超出 TTL 的旧事件。"""
        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            while trigger_events and (current_ts - trigger_events[0][0]) > self.global_ttl:
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
        """按 event_id 从 trigger 索引中移除已清除的触发事件。"""
        if not event_id:
            return

        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            kept = collections.deque()
            for event_ts, indexed_event_id, indexed_seq in trigger_events:
                if indexed_event_id == event_id:
                    continue
                kept.append((event_ts, indexed_event_id, indexed_seq))

            if kept:
                self.trigger_event_index[trigger_key] = kept
            else:
                self.trigger_event_index.pop(trigger_key, None)

    def _refresh_pending_triggers_for_node(self, node):
        """在清除事件后，重新校正该节点仍然有效的 pending 起点。"""
        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
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
        for event_ts, _event_id, event_seq in trigger_events:
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

            while q and (current_ts - q[0][0]) > self.global_ttl:
                q.popleft()

            if not q:
                self.event_cache.pop(node, None)

            self._prune_expired_trigger_index(node, current_ts)

        self._prune_cursor = (start_idx + batch_size) % max(total_nodes, 1)

    def _finalize_matches_with_history(self, matches):
        """把当前批次结果与历史组做最终合并并落库。"""
        finalized = []
        current_time = self.current_watermark if hasattr(self, 'current_watermark') else time.time()
        self.emitted_group_store.prune_expired(current_time)

        for match_result in matches:
            group_anchor_ts = self.emitted_group_store.get_group_anchor_ts(match_result, current_time)
            match_result, merged_group_indexes, related_group_uuids, should_emit = self.emitted_group_store.merge_with_related(match_result)
            if not should_emit:
                continue
            if related_group_uuids:
                existing_uuids = set(match_result.get("related_group_uuids", []))
                match_result["related_group_uuids"] = sorted(existing_uuids | set(related_group_uuids))
            self.emitted_group_store.replace_and_store(
                merged_group_indexes,
                match_result.get("rule"),
                group_anchor_ts,
                match_result
            )
            finalized.append(match_result)

        self._prune_consumed_alarm_history(finalized)
        return finalized

    def _evaluate_rule(self, rule_name, rule, trigger_node, trigger_ts, node_rule_helper=None):
        """
        全向动态图调度器 (State-Forking Matcher)：
        支持平行宇宙分叉、严格结构匹配、局部性能缓存。
        """
        helper = node_rule_helper or self.node_rule_helper
        nodes_cfg = rule.get("nodes", {})
        edges_cfg = rule.get("edges", [])
        trigger_role = rule["trigger_role"]

        # 1. 建立模式图的双向邻接表
        pattern_adj = build_pattern_adj(edges_cfg)

        # 2. 生成图遍历计划
        edges_to_explore = []
        visited_edges = set()
        queue = collections.deque([trigger_role])
        while queue:
            curr = queue.popleft()
            for edge in pattern_adj[curr]:
                tgt = edge["role"]
                edge_id = (curr, tgt)
                if edge_id not in visited_edges:
                    visited_edges.add(edge_id)
                    edges_to_explore.append((curr, tgt, edge))
                    queue.append(tgt)

        # 3. 校验触发节点自身
        trigger_node_domain = self.sites_domain_map.get(trigger_node, {})
        is_trig_valid, trig_evts = helper.validate_node(
            trigger_node, trigger_node_domain, nodes_cfg[trigger_role], trigger_ts, edge_window=0
        )
        if not is_trig_valid:
            return []

        # 4. 初始化独立子图实例池与备忘录缓存
        initial_inst = {trigger_role: {'nodes': {trigger_node: trig_evts}, 'checked': False}}
        instances = [initial_inst]
        validation_cache = {}

        # 5. 核心引擎：逐边推演，触发 分叉 (Fork) 或 聚合 (Aggregate)
        for curr_role, tgt_role, edge in edges_to_explore:
            next_instances = []

            for inst in instances:
                if curr_role not in inst:
                    next_instances.append(inst)
                    continue

                if tgt_role in inst and inst[tgt_role]["checked"]:
                    next_instances.append(inst)
                    continue

                curr_phys_dict = inst[curr_role]
                tgt_cfg = nodes_cfg[tgt_role]
                min_c = tgt_cfg.get("min_count", 1)
                node_type = tgt_cfg.get("type", "primitive")
                match_mode = tgt_cfg.get("match", "ANY")

                valid_targets = {}
                surviving_curr_phys = {}
                branch_survived = False

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
                    candidates = sorted(candidate_hops.keys(), key=lambda n: (candidate_hops[n], str(n)))
                    # 先跑拓扑可达，再按 rule 里的 selector 收窄候选集
                    candidates = helper.select_candidates_by_rule(
                        candidates, candidate_hops, tgt_cfg, edge.get("candidate_selector")
                    )

                    curr_valid_targets = {}
                    all_passed = True

                    # 如果 candidates 为空，天然跳过内层循环，该源节点被淘汰
                    for cand_phys in candidates:
                        # 校验结果依赖候选节点、目标角色以及参考时间窗口
                        cache_key = (cand_phys, tgt_role, ref_ts, edge["win"])
                        if cache_key in validation_cache:
                            is_valid, evts = validation_cache[cache_key]
                        else:
                            cand_phys_domain = self.sites_domain_map.get(cand_phys, {})
                            is_valid, evts = helper.validate_node(
                                cand_phys, cand_phys_domain, tgt_cfg, ref_ts, edge["win"]
                            )
                            validation_cache[cache_key] = (is_valid, evts)

                        if is_valid:
                            curr_valid_targets[cand_phys] = evts
                        else:
                            if match_mode == "ALL":
                                all_passed = False
                                break

                    if match_mode == "ALL" and not all_passed:
                        continue

                    if curr_valid_targets:
                        branch_survived = True
                        surviving_curr_phys[curr_phys] = curr_evts
                        for key, value in curr_valid_targets.items():
                            valid_targets[key] = value

                if not branch_survived:
                    continue

                # 回溯检查数量
                curr_cfg = nodes_cfg[curr_role]
                if inst[curr_role]["checked"] and len(surviving_curr_phys) < curr_cfg.get("min_count", 1):
                    continue

                existing_targets = inst.get(tgt_role, {}).get('nodes', {})
                merged_targets = {**existing_targets, **valid_targets}

                # 状态分叉 vs 聚合
                if node_type == 'primitive' and not existing_targets:
                    for t_node, t_evts in valid_targets.items():
                        new_inst = clone_instance_with_updates(
                            inst, curr_role, surviving_curr_phys, tgt_role, {t_node: t_evts}
                        )
                        next_instances.append(new_inst)
                else:
                    if len(merged_targets) < min_c:
                        continue

                    new_inst = clone_instance_with_updates(
                        inst, curr_role, surviving_curr_phys, tgt_role, merged_targets
                    )
                    next_instances.append(new_inst)

            instances = next_instances

            if not instances:
                return []

        # 6. 提取结果：把一次结构匹配结果整理成候选故障组。
        results = []
        targets = {e["target"] for e in edges_cfg}
        root_roles = [r for r in nodes_cfg.keys() if r not in targets]

        for inst in instances:
            # 提取物理根因节点
            inferred_roots = {}
            for r_role in root_roles:
                nodes = list(inst.get(r_role, {}).get('nodes', {}).keys())
                inferred_roots[r_role] = nodes

            symp_dict = {}
            role_mapping = {}

            for role in inst:
                nodes_dict = inst[role]['nodes']
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
                "symptoms": list(symp_dict.values())
            }
            results.append(match_result)

        return results

    def _traverse_graph(self, start_node, direction, max_hops=None,
                        reference_ts=None, edge_window=0, 
                        path_requirements=None, node_rule_helper=None):
        """通用的广度优先搜索，支持路径节点约束"""
        helper = node_rule_helper or self.node_rule_helper

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
