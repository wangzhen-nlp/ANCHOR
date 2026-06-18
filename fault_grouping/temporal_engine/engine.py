import collections
import heapq
import logging
import time
import threading

from fault_grouping.emitted_group_store import EmittedGroupStore
from fault_grouping.alarm_events.identity import require_alarm_identity
from fault_grouping.node_rule_helper import NodeRuleHelper
from alarm_tools.alarm_types import CRITICAL_ALARMS, POWER_ALARMS
from fault_grouping.temporal_engine.alarm_period import TemporalGraphEngineAlarmPeriodMixin
from fault_grouping.temporal_engine.common import TemporalGraphEngineCommonMixin
from fault_grouping.temporal_engine.constraints import TemporalGraphEngineConstraintMixin
from fault_grouping.temporal_engine.dependencies import TemporalGraphEngineDependencyMixin
from fault_grouping.temporal_engine.evaluator import TemporalGraphEngineEvaluatorMixin
from fault_grouping.temporal_engine.indexes import RoleSiteIndex
from fault_grouping.temporal_engine.output import TemporalGraphEngineOutputMixin
from fault_grouping.temporal_engine.runtime import TemporalGraphEngineRuntimeMixin
from fault_grouping.temporal_engine.traversal import TemporalGraphEngineTraversalMixin
from fault_grouping.temporal_engine.utils import (
    add_merge_stats,
    build_pattern_adj,
    build_empty_merge_stats,
    matches_expected_alarm,
    node_has_required_alarm_anchor,
    merge_match_batch,
)

logger = logging.getLogger(__name__)


class TemporalGraphEngine(
    TemporalGraphEngineCommonMixin,
    TemporalGraphEngineDependencyMixin,
    TemporalGraphEngineConstraintMixin,
    TemporalGraphEngineEvaluatorMixin,
    TemporalGraphEngineOutputMixin,
    TemporalGraphEngineRuntimeMixin,
    TemporalGraphEngineAlarmPeriodMixin,
    TemporalGraphEngineTraversalMixin,
):
    def _compile_rule_execution_plan(self, rule):
        """为单条规则预编译静态执行计划，避免每次评估重复构图和排边。"""
        nodes_cfg = rule.get("nodes", {})
        edges_cfg = rule.get("edges", [])
        trigger_role = rule["trigger_role"]
        enriched_edges_cfg = []
        for edge_cfg in edges_cfg:
            source_role = edge_cfg.get("source")
            target_role = edge_cfg.get("target")
            enriched_edge = dict(edge_cfg)
            enriched_edge["source_node"] = nodes_cfg.get(source_role)
            enriched_edge["target_node"] = nodes_cfg.get(target_role)
            enriched_edges_cfg.append(enriched_edge)

        pattern_adj = build_pattern_adj(enriched_edges_cfg)

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

        # 解析每个 role 的 alarm_source_ne_anchor 配置 —— 把隐式 anchor_role 标记
        # "<edge_source>" 替换成具体 role 名，运行时直接查表使用。
        # BFS 绑定顺序由 edges_to_explore 决定：每条 (curr, tgt, edge) 触发 tgt 绑定。
        # 这里建立 role → 首次被绑定的 BFS step，用于校验 anchor 早于 target。
        bind_order = {trigger_role: 0}
        for step_idx, (_, tgt_role_, _) in enumerate(edges_to_explore, start=1):
            bind_order.setdefault(tgt_role_, step_idx)

        def role_has_mutual_ne_anchor_edge(role_name):
            for edge_cfg in enriched_edges_cfg:
                constraints = edge_cfg.get("constraints", {})
                source_node = edge_cfg.get("source_node")
                target_node = edge_cfg.get("target_node")
                source_has_required_anchor = node_has_required_alarm_anchor(source_node)
                target_has_required_anchor = node_has_required_alarm_anchor(target_node)
                if not (
                    constraints.get("mutual_alarm_source_ne_anchor")
                    or (source_has_required_anchor and target_has_required_anchor)
                ):
                    continue
                if role_name in {edge_cfg.get("source"), edge_cfg.get("target")}:
                    return True
            return False

        alarm_source_ne_anchors = {}
        for role, node_cfg in nodes_cfg.items():
            if not isinstance(node_cfg, dict):
                continue
            anchor_cfg = node_cfg.get("alarm_source_ne_anchor")
            if not anchor_cfg:
                continue
            anchor_role = anchor_cfg.get("anchor_role", "<edge_source>")
            if anchor_role == "<edge_source>":
                # 去重：相同 source 的多条 edge 视为单个入边
                incoming_sources = list(dict.fromkeys(
                    e["source"] for e in edges_cfg if e["target"] == role
                ))
                if len(incoming_sources) == 1:
                    anchor_role = incoming_sources[0]
                elif not incoming_sources:
                    if role == trigger_role and role_has_mutual_ne_anchor_edge(role):
                        # trigger role 没有入边，无法用 role-level "<edge_source>" 解析。
                        # 这类场景由“边两端 role 都配置 alarm_source_ne_anchor”隐式启用的
                        # pair 级双向 NE 告警源约束处理。
                        continue
                    raise ValueError(
                        f"role {role!r} 有 alarm_source_ne_anchor 但没有入边；"
                        f"<edge_source> 隐式 anchor_role 无法解析"
                    )
                else:
                    raise ValueError(
                        f"role {role!r} 有多个不同入边 source={incoming_sources!r}，"
                        f"alarm_source_ne_anchor.anchor_role 必须显式给出"
                    )
            if anchor_role not in nodes_cfg:
                raise ValueError(
                    f"role {role!r} 的 alarm_source_ne_anchor.anchor_role={anchor_role!r} "
                    f"不在规则 nodes 中"
                )
            if anchor_role == role:
                raise ValueError(
                    f"role {role!r} 的 alarm_source_ne_anchor.anchor_role 不能指向自身"
                )
            # 强校验：role 本身必须被 BFS 绑定，否则该 role 永远不参与匹配，
            # 配 alarm_source_ne_anchor 无意义。
            if role not in bind_order:
                raise ValueError(
                    f"role {role!r} 从 trigger {trigger_role!r} 不可达（BFS 不会绑定），"
                    f"alarm_source_ne_anchor 配置无效"
                )
            # 强校验：anchor_role 必须在 BFS 顺序上先于 target_role 绑定，
            # 否则运行时 anchor 还未绑定，allowed_nes 会被悄悄置 None（降级不过滤）
            if anchor_role not in bind_order:
                raise ValueError(
                    f"role {role!r} 的 alarm_source_ne_anchor.anchor_role={anchor_role!r} "
                    f"从 trigger {trigger_role!r} 不可达，BFS 不会绑定该 anchor"
                )
            if bind_order[anchor_role] >= bind_order[role]:
                raise ValueError(
                    f"role {role!r}（BFS 顺序 {bind_order[role]}）的 anchor_role="
                    f"{anchor_role!r}（BFS 顺序 {bind_order[anchor_role]}）必须更早绑定"
                )
            max_hops = int(anchor_cfg.get("max_ne_hops", 1))
            if max_hops < 0:
                raise ValueError(f"role {role!r} alarm_source_ne_anchor.max_ne_hops 必须 ≥ 0")
            alarm_source_ne_anchors[role] = {
                "anchor_role": anchor_role,
                "max_ne_hops": max_hops,
            }

        return {
            "trigger_role": trigger_role,
            "pattern_adj": pattern_adj,
            "edges_to_explore": tuple(edges_to_explore),
            "root_roles": root_roles,
            "alarm_source_ne_anchors": alarm_source_ne_anchors,
        }

    def _compile_rule_execution_plans(self):
        """预编译所有规则的静态执行计划。"""
        self.rule_execution_plans = {
            rule_name: self._compile_rule_execution_plan(rule)
            for rule_name, rule in self.rules.items()
        }

    def _build_ne_adjacency(self):
        """从 ne_graph_data 构造 NE 级双向邻接表（任一方向有 link 即视为相邻）。"""
        adj = collections.defaultdict(set)
        for src_ne, info in self._ne_graph_data.items():
            if not src_ne or not isinstance(info, dict):
                continue
            links = info.get("link")
            if not isinstance(links, dict):
                continue
            for tgt_ne in links.keys():
                if not tgt_ne or tgt_ne == src_ne:
                    continue
                adj[src_ne].add(tgt_ne)
                adj[tgt_ne].add(src_ne)
        return adj

    def _compute_anchor_ne_reachable_set(self, anchor_site, max_ne_hops):
        """以 anchor_site 内的所有 NE 为起点，BFS 在 NE 图上扩展 max_ne_hops 跳。

        - 引擎未配置 NE 数据（site_to_ne_ids 为空）时，返回 None，表示"NE 锚点过滤不可用，
          降级为不过滤"——这是为了向后兼容（旧用法不传 NE 数据时，含 alarm_source_ne_anchor
          的规则也能跑，仅退化为站点级语义）。
        - 配置了 NE 数据但 anchor_site 自身无 NE：返回空 frozenset，所有 alarm 都被过滤掉。
        - 正常情况：返回 frozenset(reachable_ne_ids)（含起点 NE 自身）。

        结果按 (anchor_site, max_ne_hops) 缓存，跨规则、跨 trigger 自动复用。
        """
        if not self._site_to_ne_ids:
            return None
        cache_key = (anchor_site, max_ne_hops)
        cached = self._anchor_ne_reachable_cache.get(cache_key)
        if cached is not None:
            return cached
        anchor_nes = self._site_to_ne_ids.get(anchor_site, ())
        if not anchor_nes:
            result = frozenset()
        else:
            visited = set(anchor_nes)
            if max_ne_hops > 0 and self._ne_adjacency:
                frontier = set(anchor_nes)
                for _ in range(max_ne_hops):
                    next_frontier = set()
                    for ne in frontier:
                        for nb in self._ne_adjacency.get(ne, ()):
                            if nb not in visited:
                                next_frontier.add(nb)
                    if not next_frontier:
                        break
                    visited.update(next_frontier)
                    frontier = next_frontier
            result = frozenset(visited)
        self._anchor_ne_reachable_cache[cache_key] = result
        return result

    def __init__(
        self,
        topo_downstream_map,
        rules_config,
        site_domain_map,
        alarm_source_domain_map=None,
        aggregation_wait_sec=420,
        site_merge_helper=None,
        site_chain_index=None,
        use_alarm_period_cache=False,
        enable_support_pruning=False,
        enable_support_count_sort=False,
        missing_topology_edges=None,
        ne_graph_data=None,
        site_to_ne_ids=None,
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

        # event_cache 的两种运行模式:
        # - 默认(raw): 站点 -> deque[(ts, event_id, alarm_type, alarm_source, consumed_trigger_rules)]
        # - 可选(period): 站点 -> 活跃告警时段摘要 deque
        self.use_alarm_period_cache = bool(use_alarm_period_cache)
        self.event_cache = collections.defaultdict(collections.deque)
        # 仅在 period 模式下使用；raw 模式保持为空。
        self.active_alarm_periods = collections.defaultdict(dict)
        self.active_event_to_period = collections.defaultdict(dict)
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
        # role-filtered topology candidate cache: topology and role structure are static.
        self.global_role_filtered_neighbor_cache = collections.OrderedDict()
        self.max_role_filtered_neighbor_cache_size = 20000

        # 故障传播等待时间
        self.aggregation_wait_sec = aggregation_wait_sec

        # 流式时间水印
        self.current_watermark = 0.0
        # 已到达事件时间上界：TTL 清理只跟真实已进入引擎的事件时间走，不跟 live 模式下的模拟水印走
        self.latest_arrived_event_ts = 0.0

        # 延迟触发队列：记录“当前仍在等待聚合”的 trigger 起点锚点，结构为 (ts, seq)
        self.pending_triggers = {}
        # node -> {(node, rule_name)} 反向索引，供 _expand_matches_with_pending_context
        # 按本批 non-trigger 节点直接定位相关 pending，避免扫全量。
        # 与 pending_triggers 在同一把 self._lock 下同步维护。
        self._pending_triggers_by_node = collections.defaultdict(set)
        # 延迟触发最小堆：按 ready_ts 排序，快速摘取已成熟的 pending trigger
        self.pending_trigger_heap = []
        # 保存某个 (node, rule) 下所有还能作为 trigger 候选的事件，结构为 (ts, event_id, seq, alarm_type)
        self.trigger_event_index = collections.defaultdict(collections.deque)
        # trigger 候选事件的全局递增序号，用于精确定位“下一条”事件。
        self._trigger_seq = 0

        # 负责历史组保留、按 eid 合并和替换落库
        self.emitted_group_store = EmittedGroupStore(
            self.rules,
            self.global_ttl,
            use_alarm_period_cache=self.use_alarm_period_cache,
        )

        # 负责站点结构匹配、告警窗口校验和失败原因解释
        self.node_rule_helper = NodeRuleHelper(
            self.sites_domain_map,
            CRITICAL_ALARMS,
            lambda node: self.event_cache.get(node, []),
            self.alarm_source_domain_map,
        )
        self.role_site_index = RoleSiteIndex(
            self.rules,
            self.sites_domain_map,
            self.node_rule_helper,
        )
        self.enable_support_pruning = bool(enable_support_pruning)
        self.enable_support_count_sort = bool(enable_support_count_sort)
        self.missing_topology_edges = dict(missing_topology_edges or {})
        self.optimization_stats = collections.Counter()

        # NE 级拓扑数据（用于 alarm_source_ne_anchor 约束）。
        # ne_graph_data: {ne_id: {"site_id": ..., "link": {neighbor_ne_id: {...}}}}
        # site_to_ne_ids: {site_id: (ne_id, ...)}
        self._ne_graph_data = ne_graph_data or {}
        self._site_to_ne_ids = {
            site_id: tuple(ne_ids)
            for site_id, ne_ids in (site_to_ne_ids or {}).items()
        }
        self._ne_adjacency = self._build_ne_adjacency()
        # 缓存键: (anchor_site, max_ne_hops) -> frozenset(reachable_ne_ids)
        # 不含规则名，跨规则、跨 trigger 自动复用。
        self._anchor_ne_reachable_cache = {}
        # 缓存键: (source_ne, target_ne, max_ne_hops) -> bool。
        # 用于边级 mutual_alarm_source_ne_anchor，避免两侧 link 告警较多时重复 BFS。
        self._alarm_source_ne_reachability_cache = {}
        # 缓存每个站点/role 配置解析出的 required alarm 集合，供 mutual NE anchor
        # 做事件源精确过滤时复用，避免同一候选对反复解析 site_rules。
        self._required_alarm_set_cache = {}

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

    def process_event(
        self,
        node,
        alarm_type,
        ts,
        event_id,
        occurrence_uuid,
        alarm_source="",
        is_clear=False,
        collect_matches=False,
        register_trigger=True,
    ):
        """接收单条事件并更新内部状态。默认只更新内部状态；当 collect_matches=True 时，会在事件时间点立即收割已成熟的故障组。
        """
        with self._lock:
            event_id, occurrence_uuid = require_alarm_identity({
                "eid": event_id,
                "occurrence_uuid": occurrence_uuid,
            })
            # 1. 当前仍保留事件时间水印，便于离线按事件时间回放。
            self.current_watermark = max(self.current_watermark, ts)
            self.latest_arrived_event_ts = max(self.latest_arrived_event_ts, ts)

            # 2. 先清理过期缓存，再按上报/清除事件更新状态。
            if self.use_alarm_period_cache:
                self._prune_expired_alarm_periods(node, ts)
            else:
                self._prune_expired_raw_events_in_place(node, ts)
            self._prune_expired_trigger_index(node, ts)

            if self.use_alarm_period_cache:
                if is_clear:
                    # 清除事件只按 event_id 删除对应实例，并联动修正 trigger / pending 状态。
                    self._remove_cleared_events(
                        node,
                        event_id,
                        occurrence_uuid,
                        alarm_type=alarm_type,
                        alarm_source=alarm_source,
                    )
                    affected_rule_names = self._remove_cleared_trigger_events(
                        node,
                        event_id,
                        occurrence_uuid,
                        alarm_type=alarm_type,
                        alarm_source=alarm_source,
                    )
                    if affected_rule_names:
                        self._refresh_pending_triggers_for_node(
                            node,
                            affected_rule_names=affected_rule_names
                        )
                else:
                    self._register_alarm_period_occurrence(
                        node,
                        alarm_type,
                        ts,
                        event_id,
                        occurrence_uuid,
                        alarm_source=alarm_source,
                    )
            else:
                if is_clear:
                    self._remove_cleared_events(
                        node,
                        event_id,
                        occurrence_uuid,
                        alarm_type=alarm_type,
                        alarm_source=alarm_source,
                    )
                    affected_rule_names = self._remove_cleared_trigger_events(
                        node,
                        event_id,
                        occurrence_uuid,
                        alarm_type=alarm_type,
                        alarm_source=alarm_source,
                    )
                    if affected_rule_names:
                        self._refresh_pending_triggers_for_node(
                            node,
                            affected_rule_names=affected_rule_names
                        )
                else:
                    self.event_cache[node].append(
                        (ts, event_id, alarm_type, alarm_source, frozenset(), occurrence_uuid)
                    )

            # 3. 命中 trigger 的事件只负责入 pending，不在这里直接做匹配评估。
            if not is_clear and register_trigger:
                alarm_source_domain = self.alarm_source_domain_map.get(alarm_source, "")
                for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
                    if any(
                        matches_expected_alarm(alarm_type, expected, alarm_source_domain)
                        for expected in expected_list
                    ):
                        trigger_key = (node, rule_name)
                        self._trigger_seq += 1
                        trigger_seq = self._trigger_seq
                        self.trigger_event_index[trigger_key].append(
                            (ts, event_id, trigger_seq, alarm_type, str(alarm_source or ""), occurrence_uuid)
                        )
                        # 如果这段时间内已经触发过，就不更新时间，以“第一声警报”为准
                        if trigger_key not in self.pending_triggers:
                            self._set_pending_trigger(trigger_key, ts, trigger_seq)

        # 离线模式通过事件触发收割
        if collect_matches:
            # 快路径：没有 mature pending trigger 时跳过 _collect_pending_matches，
            # 省掉一次 lock acquire + helper 准备的开销。
            # race 安全（仅在 live 模式的后台 harvest 线程下才有 race）：
            #   - 假阴性（实际有 mature）→ 下一个事件会重试，最多延迟一个 ingest 周期
            #   - 假阳性（heap 顶是 stale anchor 但满足 ts 条件）→ 落回慢路径，
            #     _collect_mature_pending_locked 会正确丢弃 stale 项
            heap = self.pending_trigger_heap
            if heap:
                effective_ts = self.latest_arrived_event_ts if self.latest_arrived_event_ts > 0 else self.current_watermark
                if heap[0][0] <= effective_ts:
                    return self._collect_pending_matches(force=False)

        return []

    def _get_event_ttl(self, alarm_type):
        return self.power_alarm_ttl if alarm_type in POWER_ALARMS else self.global_ttl

    def _log_debug_event_removal(self, node, event, reason, **extra):
        if not self.debug_event_logger:
            return

        if isinstance(event, dict):
            ts = event.get("ts")
            event_id = event.get("eid")
            alarm_type = event.get("alarm")
            alarm_source = event.get("alarm_source", "")
            consumed_trigger_rules = event.get("consumed_trigger_rules", ())
            occurrence_uuid = event.get("occurrence_uuid")
        else:
            ts, event_id, alarm_type, alarm_source, consumed_trigger_rules, occurrence_uuid = event
        payload = {
            "node": node,
            "ts": ts,
            "event_id": event_id,
            "occurrence_uuid": occurrence_uuid,
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
        # 反向索引同步维护：set.add 幂等，重复 set 同一 trigger_key 也安全。
        self._pending_triggers_by_node[trigger_key[0]].add(trigger_key)
        ready_ts = first_trigger_ts + self.aggregation_wait_sec
        heapq.heappush(self.pending_trigger_heap, (ready_ts, first_trigger_ts, trigger_seq, trigger_key))
        self._maybe_rebuild_pending_heap_locked()

    def _remove_pending_trigger(self, trigger_key):
        """从 pending_triggers 和反向索引同时移除。返回原 anchor 或 None。"""
        anchor = self.pending_triggers.pop(trigger_key, None)
        if anchor is None:
            return None
        node = trigger_key[0]
        bucket = self._pending_triggers_by_node.get(node)
        if bucket is not None:
            bucket.discard(trigger_key)
            if not bucket:
                del self._pending_triggers_by_node[node]
        return anchor

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
                self._remove_pending_trigger(trigger_key)
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

            self._remove_pending_trigger(trigger_key)
            self._prune_trigger_index_before(trigger_key, trigger_seq)
            mature_items.append((trigger_key, (first_trigger_ts, trigger_seq)))

        return mature_items

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
                matched_rule_names = set()

                raw_matched_rules = symptom.get("matched_rule_list")
                if not isinstance(raw_matched_rules, list):
                    raw_matched_rules = [symptom.get("matched_rule")]
                raw_matched_roles = symptom.get("matched_role_list")
                if not isinstance(raw_matched_roles, list):
                    raw_matched_roles = [matched_role]

                for raw_rule in raw_matched_rules:
                    matched_rule = str(raw_rule or "").strip()
                    if not matched_rule or matched_rule not in rule_to_trigger_role:
                        continue
                    trigger_role = rule_to_trigger_role[matched_rule]
                    if trigger_role in raw_matched_roles or matched_role == trigger_role:
                        matched_rule_names.add(matched_rule)

                if not matched_rule_names:
                    matched_rule_names = {
                        rule_name
                        for rule_name, trigger_role in rule_to_trigger_role.items()
                        if matched_role == trigger_role
                    }
                if not matched_rule_names:
                    continue
                node = symptom.get("node")
                alarm_type = symptom.get("alarm")
                alarm_source = symptom.get("alarm_source", "")
                ts = symptom.get("ts")
                if node in (None, "") or alarm_type in (None, "") or ts is None:
                    continue
                key = (node, alarm_type, str(alarm_source or ""))
                entry = prune_points.setdefault(key, {})
                for rule_name in matched_rule_names:
                    entry[rule_name] = max(entry.get(rule_name, float("-inf")), ts)

        for (node, alarm_type, alarm_source), cutoff_by_rule in prune_points.items():
            self._prune_node_alarm_history_before(
                node,
                alarm_type,
                alarm_source,
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
                self._remove_pending_trigger(trigger_key)
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

    @staticmethod
    def _unpack_trigger_event(trigger_event):
        event_ts, event_id, event_seq, alarm_type, alarm_source, occurrence_uuid = trigger_event
        return event_ts, event_id, event_seq, alarm_type, str(alarm_source or ""), occurrence_uuid

    def _remove_cleared_trigger_events(
        self,
        node,
        event_id,
        occurrence_uuid,
        alarm_type=None,
        alarm_source=None,
    ):
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
            target_alarm_source = None if alarm_source is None else str(alarm_source or "")
            for trigger_event in trigger_events:
                (
                    event_ts,
                    indexed_event_id,
                    indexed_seq,
                    indexed_alarm_type,
                    indexed_alarm_source,
                    indexed_occurrence_uuid,
                ) = self._unpack_trigger_event(trigger_event)
                matches_clear = (
                    indexed_event_id == event_id
                    and indexed_occurrence_uuid == occurrence_uuid
                    and (alarm_type is None or indexed_alarm_type == alarm_type)
                    and (
                        target_alarm_source is None
                        or indexed_alarm_source == target_alarm_source
                    )
                )
                if matches_clear:
                    if current_pending_anchor == (event_ts, indexed_seq):
                        affected_rule_names.add(rule_name)
                    continue
                kept.append(trigger_event)

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
                self._remove_pending_trigger(trigger_key)
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
        for trigger_event in trigger_events:
            event_ts, _event_id, event_seq, _alarm_type, _alarm_source, _occurrence_uuid = (
                self._unpack_trigger_event(trigger_event)
            )
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
            if self.use_alarm_period_cache:
                self._prune_expired_alarm_periods(node, current_ts)
            else:
                self._prune_expired_raw_events_in_place(node, current_ts)
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
            # 通过反向索引按 non_trigger_nodes 直接取 pending，避免扫全量 pending_triggers。
            pending_candidates = [
                (trigger_key, self.pending_triggers[trigger_key])
                for node in non_trigger_nodes
                for trigger_key in self._pending_triggers_by_node.get(node, ())
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
