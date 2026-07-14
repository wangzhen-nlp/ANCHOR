import collections

from anchor_grouping_online.emitted_group_store import EmittedGroupStore
from anchor_grouping_online.alarm_events.identity import require_eid
from anchor_grouping_online.node_rule_helper import NodeRuleHelper
from anchor_grouping_online.alarm_types import LINK_ALARMS
from anchor_grouping_online.temporal_engine.event_cache import TemporalGraphEngineEventCacheMixin
from anchor_grouping_online.temporal_engine.common import TemporalGraphEngineCommonMixin
from anchor_grouping_online.temporal_engine.constraints import TemporalGraphEngineConstraintMixin
from anchor_grouping_online.temporal_engine.dependencies import TemporalGraphEngineDependencyMixin
from anchor_grouping_online.temporal_engine.evaluator import TemporalGraphEngineEvaluatorMixin
from anchor_grouping_online.temporal_engine.indexes import RoleSiteIndex
from anchor_grouping_online.temporal_engine.output import TemporalGraphEngineOutputMixin
from anchor_grouping_online.temporal_engine.traversal import TemporalGraphEngineTraversalMixin
from anchor_grouping_online.temporal_engine.utils import (
    build_pattern_adj,
    matches_expected_alarm,
)

class TemporalGraphEngine(
    TemporalGraphEngineCommonMixin,
    TemporalGraphEngineDependencyMixin,
    TemporalGraphEngineConstraintMixin,
    TemporalGraphEngineEvaluatorMixin,
    TemporalGraphEngineOutputMixin,
    TemporalGraphEngineEventCacheMixin,
    TemporalGraphEngineTraversalMixin,
):
    @staticmethod
    def _expected_requires_link_alarm(expected):
        required_alarms = expected.get("required_alarms")
        return required_alarms is not None and any(
            alarm in LINK_ALARMS for alarm in required_alarms
        )

    @classmethod
    def _rules_require_link_peer_index(cls, rules_config):
        for rule in rules_config.values():
            for node_cfg in rule["nodes"].values():
                for site_rule in node_cfg.get("site_rules", ()):
                    if cls._expected_requires_link_alarm(site_rule["expected_alarms"]):
                        return True
        return False

    def _compile_rule_execution_plan(self, rule):
        """为单条规则预编译静态执行计划，避免每次评估重复构图和排边。"""
        nodes_cfg = rule["nodes"]
        edges_cfg = rule["edges"]
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
            "alarm_source_ne_anchors": self._resolve_alarm_source_ne_anchors(
                nodes_cfg, edges_cfg, trigger_role, edges_to_explore
            ),
        }

    @classmethod
    def _resolve_alarm_source_ne_anchors(
        cls, nodes_cfg, edges_cfg, trigger_role, edges_to_explore
    ):
        """解析每个 role 的 alarm_source_ne_anchor 配置。

        把隐式 anchor_role 标记 "<edge_source>" 替换成具体 role 名，运行时
        直接查表使用。BFS 绑定顺序由 edges_to_explore 决定：每条
        (curr, tgt, edge) 触发 tgt 绑定。这里建立 role → 首次被绑定的 BFS
        step，用于校验 anchor 早于 target。
        """
        bind_order = {trigger_role: 0}
        for step_idx, (_, tgt_role_, _) in enumerate(edges_to_explore, start=1):
            bind_order.setdefault(tgt_role_, step_idx)

        alarm_source_ne_anchors = {}
        for role, node_cfg in nodes_cfg.items():
            if not isinstance(node_cfg, dict):
                continue
            anchor_cfg = node_cfg.get("alarm_source_ne_anchor")
            if not anchor_cfg:
                continue
            incoming_sources = list(dict.fromkeys(
                edge["source"] for edge in edges_cfg if edge["target"] == role
            ))
            if anchor_cfg.get("anchor_role") != "<edge_source>" or len(incoming_sources) != 1:
                raise ValueError(
                    f"role {role!r} 的 alarm_source_ne_anchor 必须使用唯一入边 source"
                )
            anchor_role = incoming_sources[0]
            cls._validate_anchor_bind_order(role, anchor_role, bind_order, trigger_role)
            max_hops = int(anchor_cfg.get("max_ne_hops", 1))
            if max_hops < 0:
                raise ValueError(f"role {role!r} alarm_source_ne_anchor.max_ne_hops 必须 ≥ 0")
            alarm_source_ne_anchors[role] = {
                "anchor_role": anchor_role,
                "max_ne_hops": max_hops,
            }
        return alarm_source_ne_anchors

    @staticmethod
    def _validate_anchor_bind_order(role, anchor_role, bind_order, trigger_role):
        """强校验 anchor/target 都被 BFS 绑定，且 anchor 更早绑定。

        role 本身必须被 BFS 绑定，否则该 role 永远不参与匹配，配
        alarm_source_ne_anchor 无意义；anchor_role 必须在 BFS 顺序上先于
        target_role 绑定。
        """
        if role not in bind_order:
            raise ValueError(
                f"role {role!r} 从 trigger {trigger_role!r} 不可达（BFS 不会绑定），"
                f"alarm_source_ne_anchor 配置无效"
            )
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

    def _compile_rule_execution_plans(self):
        """预编译所有规则的静态执行计划。"""
        self.rule_execution_plans = {
            rule_name: self._compile_rule_execution_plan(rule)
            for rule_name, rule in self.rules.items()
        }

    @staticmethod
    def _build_ne_adjacency(ne_graph_data):
        """从 ne_graph_data 构造 NE 级双向邻接表（任一方向有 link 即视为相邻）。"""
        adj = collections.defaultdict(set)
        for src_ne, info in ne_graph_data.items():
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

        - anchor_site 自身无 NE 时返回空 frozenset，所有 alarm 都被过滤掉。
        - 正常情况：返回 frozenset(reachable_ne_ids)（含起点 NE 自身）。

        结果按 (anchor_site, max_ne_hops) 缓存，跨规则、跨 trigger 自动复用。
        """
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
        self, rules_config, site_domain_map, site_chain_index, ne_graph_data,
        site_to_ne_ids, link_peer_index, event_ttl, alarm_source_domain_map=None,
        enable_batch_upsert_indexes=False, shared_static_context=None,
    ):
        """初始化拓扑、缓存、触发索引以及历史故障组状态。

        shared_static_context 用于跨引擎复用与「规则边时间窗 / max_stay」
        无关的静态结构（NE 映射、trigger 索引、role_site_index）。
        提供时直接引用、跳过重建；这些结构构造后只读，跨引擎共享安全。
        仅执行计划（依赖各引擎自己的边时间窗）始终按本引擎的规则重编。
        要求各引擎的规则共享同一批 node_config 对象（batch 侧浅拷贝规则时
        保持 node 身份不变），以保证 role_site_index 的 id(node_config) 命中。
        """
        # 规则配置总表：按规则名保存匹配图、触发角色和节点约束。
        self.rules = rules_config
        if self._rules_require_link_peer_index(self.rules) and not link_peer_index:
            raise ValueError("规则包含必需 link 告警 role，必须提供 link_peer_index")
        self.site_chain_index = site_chain_index
        if not self.site_chain_index:
            raise ValueError("必须提供非空 site_chain_index")
        # 站点画像信息：供节点匹配领域使用
        self.sites_domain_map = site_domain_map
        self.alarm_source_domain_map = alarm_source_domain_map or {}
        self._init_runtime_state(event_ttl, enable_batch_upsert_indexes)
        # NE 级拓扑数据（用于 alarm_source_ne_anchor 约束）。
        # ne_graph_data: {ne_id: {"site_id": ..., "link": {neighbor_ne_id: {...}}}}
        # site_to_ne_ids: {site_id: (ne_id, ...)}
        self._link_peer_index = link_peer_index
        self._init_topology_indexes(
            ne_graph_data, site_to_ne_ids, shared_static_context
        )
        # 缓存键: (anchor_site, max_ne_hops) -> frozenset(reachable_ne_ids)
        # 不含规则名，跨规则、跨 trigger 自动复用。
        self._anchor_ne_reachable_cache = {}
        # 每条规则的静态执行计划：依赖本引擎自身的边时间窗，始终按本引擎规则
        # 编译（不参与跨引擎共享）。
        self.rule_execution_plans = {}
        self._compile_rule_execution_plans()
        # 站点可以作为 trigger 的规则+告警组合，{node: ((rule, (alarm_type, ...)), ...)}
        if shared_static_context is not None:
            self.trigger_specs_by_node = shared_static_context["trigger_specs_by_node"]
        else:
            self.trigger_specs_by_node = {}
            self._build_trigger_indexes()

    def _init_runtime_state(self, event_ttl, enable_batch_upsert_indexes):
        """初始化事件缓存、拓扑穿透缓存、trigger 索引与历史组存储。"""
        # event_cache: 站点 -> deque[事件 dict]，保留原始告警 payload 供后续端口/对端解析
        self.event_cache = collections.defaultdict(collections.deque)
        # 持久批处理会话可选的幂等写入索引。隔离临时会话不建，
        # 避免无必要的常驻内存；启用后当前批告警可按 eid O(1) 覆盖缓存。
        self._batch_event_by_alarm_id = (
            {} if enable_batch_upsert_indexes else None
        )
        # 告警缓存保留时长，单位秒。
        self.global_ttl = float(event_ttl)
        # 全局拓扑穿透缓存
        self.global_topo_cache = collections.OrderedDict()
        self.max_topo_cache_size = 10000
        # role-filtered topology candidate cache: topology and role structure are static.
        self.global_role_filtered_neighbor_cache = collections.OrderedDict()
        self.max_role_filtered_neighbor_cache_size = 20000
        # 已到达事件的时间上界。
        self.latest_arrived_event_ts = 0.0
        # 保存某个 (node, rule) 下所有还能作为 trigger 候选的事件，
        # 结构为 (ts, alarm_id, seq, alarm_type, alarm_source)
        self.trigger_event_index = collections.defaultdict(collections.deque)
        # trigger 候选事件的全局递增序号，用于精确定位“下一条”事件。
        self._trigger_seq = 0
        # 负责历史组保留、按 eid 合并和替换落库
        self.emitted_group_store = EmittedGroupStore()
        # 负责站点结构匹配、告警窗口校验和失败原因解释
        event_cache = self.event_cache
        self.node_rule_helper = NodeRuleHelper(
            lambda node, cache=event_cache: cache.get(node, []),
            self.alarm_source_domain_map,
        )

    def _init_topology_indexes(
        self, ne_graph_data, site_to_ne_ids, shared_static_context
    ):
        """构建（或直接复用共享的）NE 级静态结构与 role_site_index。"""
        if shared_static_context is not None:
            self.role_site_index = shared_static_context["role_site_index"]
            self._ne_to_site = shared_static_context["ne_to_site"]
            self._site_to_ne_ids = shared_static_context["site_to_ne_ids"]
            self._ne_adjacency = shared_static_context["ne_adjacency"]
            return
        self.role_site_index = RoleSiteIndex(
            self.rules,
            self.sites_domain_map,
            self.node_rule_helper,
        )
        self._ne_to_site = {}
        for ne_id, info in ne_graph_data.items():
            if not isinstance(info, dict):
                continue
            site_id = str(info.get("site_id", "") or "").strip()
            if not site_id:
                continue
            self._ne_to_site[ne_id] = site_id
        self._site_to_ne_ids = {
            site_id: tuple(ne_ids)
            for site_id, ne_ids in site_to_ne_ids.items()
        }
        if not self._site_to_ne_ids:
            raise ValueError("必须提供非空 site_to_ne_ids")
        self._ne_adjacency = self._build_ne_adjacency(ne_graph_data)

    def export_static_context(self):
        """导出可跨引擎共享的静态结构。

        这些结构只依赖拓扑与规则的 node 结构，与规则边时间窗 / max_stay
        无关，构造后只读；可注入其他引擎（shared_static_context）避免重建。
        执行计划因依赖各引擎自身的边时间窗，不在此列。
        """
        return {
            "role_site_index": self.role_site_index,
            "ne_to_site": self._ne_to_site,
            "site_to_ne_ids": self._site_to_ne_ids,
            "ne_adjacency": self._ne_adjacency,
            "trigger_specs_by_node": self.trigger_specs_by_node,
        }

    def process_event(
        self, node, alarm_type, ts, alarm_id, alarm_source="",
        is_clear=False, alarm_payload=None, index_trigger=True,
        cache_event=True, trigger_candidates=None,
    ):
        """接收单条事件，更新事件缓存与 trigger 索引。

        本方法只维护 event_cache 与 trigger 索引，
        命中 trigger 的事件写入 trigger_candidates 供调用方按批收割，不在此
        触发任何规则评估。
        index_trigger=False 用于加载只作为症状候选的历史告警：事件仍进入
        event_cache，但不建立 trigger 索引。
        cache_event=False 用于当前批重发历史告警：复用已有缓存事件，
        只按当前告警语义处理 trigger（清除告警仍会删除已有缓存）。
        """
        alarm_id = require_eid({"eid": alarm_id})
        # 1. 按已到达事件推进时间上界。
        self.latest_arrived_event_ts = max(self.latest_arrived_event_ts, ts)

        # 2. 按上报/清除事件更新状态。批开始时由 matcher 统一做视界清理。
        if is_clear:
            self._remove_cleared_event_state(node, alarm_id, alarm_type, alarm_source)
        elif cache_event:
            cached_event = {
                "ts": ts,
                "eid": alarm_id,
                "alarm": alarm_type,
                "alarm_source": alarm_source,
                "alarm_payload": alarm_payload if isinstance(alarm_payload, dict) else {},
                "consumed_trigger_rules": frozenset(),
            }
            self.event_cache[node].append(cached_event)
            batch_event_index = self._batch_event_by_alarm_id
            if batch_event_index is not None:
                batch_event_index[alarm_id] = cached_event

        # 3. 命中 trigger 的事件写入 trigger 索引，并汇报到本批候选。
        if index_trigger and not is_clear:
            self._index_trigger_events(
                node, alarm_type, alarm_source, ts, alarm_id, trigger_candidates
            )

        return []

    def _index_trigger_events(
        self, node, alarm_type, alarm_source, ts, alarm_id, trigger_candidates
    ):
        """把命中 trigger 的事件写入 trigger 索引，并汇报到本批候选。"""
        for rule_name in self._iter_trigger_rules(node, alarm_type, alarm_source):
            trigger_key = (node, rule_name)
            self._trigger_seq += 1
            trigger_seq = self._trigger_seq
            self.trigger_event_index[trigger_key].append((
                ts,
                alarm_id,
                trigger_seq,
                alarm_type,
                str(alarm_source or ""),
            ))
            if trigger_candidates is not None:
                trigger_candidates.append((ts, trigger_seq, trigger_key))

    def _remove_cleared_event_state(self, node, alarm_id, alarm_type, alarm_source):
        """清除告警：同时移除原始缓存事件与 trigger 候选。"""
        self._remove_cleared_raw_event(
            node,
            alarm_id,
            alarm_type=alarm_type,
            alarm_source=alarm_source,
        )
        self._remove_cleared_trigger_events(
            node,
            alarm_id,
            alarm_type=alarm_type,
            alarm_source=alarm_source,
        )

    def _iter_trigger_rules(self, node, alarm_type, alarm_source):
        """产出该事件在 node 上命中 trigger 的规则名。"""
        alarm_source_domain = self.alarm_source_domain_map.get(alarm_source, "")
        for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
            if any(
                matches_expected_alarm(alarm_type, expected, alarm_source_domain)
                for expected in expected_list
            ):
                yield rule_name

    def process_batch_event(
        self, node, alarm_type, ts, alarm_id, alarm_source="",
        is_clear=False, alarm_payload=None, index_trigger=True,
    ):
        """持久批处理幂等写入当前告警，返回本批新建 trigger 候选。

        当前批再次提供同一 eid 时原地刷新事件并清空历史消费标记。trigger
        只属于当前调用，每次均重新建立，不跨批复用；index_trigger=False
        用于外置调用结束时仅把当前事件同步进持久缓存。
        该接口要求构造引擎时启用 batch event upsert 索引。
        """
        event_index = self._batch_event_by_alarm_id
        if event_index is None:
            raise RuntimeError("process_batch_event 需要启用 batch upsert 索引")

        alarm_id = require_eid({"eid": alarm_id})
        self.latest_arrived_event_ts = max(self.latest_arrived_event_ts, ts)

        if is_clear:
            self._remove_cleared_event_state(node, alarm_id, alarm_type, alarm_source)
            return []

        payload = alarm_payload if isinstance(alarm_payload, dict) else {}
        self._upsert_batch_cached_event(
            node, alarm_id, alarm_type, alarm_source, ts, payload
        )

        if not index_trigger:
            return []

        trigger_candidates = []
        for rule_name in self._iter_trigger_rules(node, alarm_type, alarm_source):
            trigger_key = (node, rule_name)
            self._trigger_seq += 1
            trigger_seq = self._trigger_seq
            self._insert_batch_trigger_event(
                trigger_key,
                (
                    ts,
                    alarm_id,
                    trigger_seq,
                    alarm_type,
                    str(alarm_source or ""),
                ),
            )
            trigger_candidates.append((ts, trigger_seq, trigger_key))
        return trigger_candidates

    def _upsert_batch_cached_event(
        self, node, alarm_id, alarm_type, alarm_source, ts, payload
    ):
        """新建或按 eid 幂等覆盖持久缓存事件。

        同一 eid 的发生时间、类型或告警源发生变化时报错；覆盖时清空历史
        消费标记（出现在当前批即重新获得 trigger 资格，本批前序匹配仍会
        再次写入消费标记并使后续 trigger 失效）。
        """
        event_index = self._batch_event_by_alarm_id
        cached_event = event_index.get(alarm_id)
        if cached_event is None:
            cached_event = {
                "ts": ts,
                "eid": alarm_id,
                "alarm": alarm_type,
                "alarm_source": alarm_source,
                "alarm_payload": payload,
                "consumed_trigger_rules": frozenset(),
            }
            self._insert_batch_cached_event(node, cached_event)
            event_index[alarm_id] = cached_event
            return
        previous_ts = cached_event.get("ts")
        if previous_ts != ts:
            raise ValueError(
                f"同一告警 {alarm_id!r} 的发生时间从 {previous_ts!r}"
                f" 变为 {ts!r}，无法幂等覆盖"
            )
        if (
            cached_event.get("alarm") != alarm_type
            or str(cached_event.get("alarm_source", "") or "")
            != str(alarm_source or "")
        ):
            raise ValueError(
                f"同一告警 {alarm_id!r} 的类型或告警源发生变化，"
                "无法幂等覆盖"
            )
        cached_event.update({
            "ts": ts,
            "alarm": alarm_type,
            "alarm_source": alarm_source,
            "alarm_payload": payload,
            "consumed_trigger_rules": frozenset(),
        })

    def _insert_batch_cached_event(self, node, cached_event):
        """按 ts 插入持久批处理缓存；顺序到达保持 O(1) 追加快路径。"""
        events = self.event_cache[node]
        ts = cached_event["ts"]
        if not events or events[-1]["ts"] <= ts:
            events.append(cached_event)
            return
        for index, event in enumerate(events):
            if event["ts"] > ts:
                events.insert(index, cached_event)
                return
        events.append(cached_event)

    def _insert_batch_trigger_event(self, trigger_key, trigger_event):
        """O(1) 追加到本次调用的临时 trigger 工作集。

        收割候选会独立按 (ts, seq) 排序；消费删除按事件键过滤整条 deque，
        因而临时索引无需维护时间顺序，也不参与跨批 TTL。
        """
        self.trigger_event_index[trigger_key].append(trigger_event)

    def _forget_batch_cached_event(self, node, cached_event):
        event_index = self._batch_event_by_alarm_id
        if event_index is None:
            return
        alarm_id = cached_event.get("eid")
        indexed = event_index.get(alarm_id)
        if indexed is cached_event:
            event_index.pop(alarm_id, None)

    def _collect_trigger_expected_list(self, trigger_node_domain, trigger_config):
        expected_list = []
        node_type = trigger_config.get("type", "primitive")
        if node_type == "primitive":
            if not self.node_rule_helper.matches_node_structure(trigger_node_domain, trigger_config):
                return expected_list
            expected = self.node_rule_helper.resolve_expected_alarms(trigger_node_domain, trigger_config)
            if expected is not None:
                expected_list.append(expected)
            return expected_list

        if node_type == "compound":
            for pattern in trigger_config.get("patterns", []):
                if not self.node_rule_helper.matches_node_structure(trigger_node_domain, pattern):
                    continue
                expected = self.node_rule_helper.resolve_expected_alarms(trigger_node_domain, pattern)
                if expected is not None:
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

    def _prune_consumed_alarm_history(self, matches):
        """回收命中 trigger_role 的告警历史，返回被删除的 trigger 序号集合。"""
        prune_points = {}
        for match in matches:
            rule_to_trigger_role = {
                rule_name: self.rules[rule_name]["trigger_role"]
                for rule_name in match["merged_rules"]
                if rule_name in self.rules and self.rules[rule_name].get("trigger_role")
            }
            for symptom in match["symptoms"]:
                matched_rule_names = self._symptom_trigger_rule_names(
                    symptom, rule_to_trigger_role
                )
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

        removed_trigger_seqs = set()
        for (node, alarm_type, alarm_source), cutoff_by_rule in prune_points.items():
            removed_trigger_seqs.update(
                self._prune_node_alarm_history_before(
                    node,
                    alarm_type,
                    alarm_source,
                    cutoff_by_rule,
                )
            )
        return removed_trigger_seqs

    @staticmethod
    def _symptom_trigger_rule_names(symptom, rule_to_trigger_role):
        """按 matched_rule/matched_role 把症状归属到命中的 trigger 规则集合。"""
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
        return matched_rule_names

    @staticmethod
    def _unpack_trigger_event(trigger_event):
        event_ts, alarm_id, event_seq, alarm_type, alarm_source = trigger_event
        return event_ts, alarm_id, event_seq, alarm_type, str(alarm_source or "")

    def _remove_cleared_trigger_events(
        self,
        node,
        alarm_id,
        alarm_type,
        alarm_source,
    ):
        """按 alarm_id 从 trigger 索引中移除已清除的触发事件。"""
        target_alarm_source = str(alarm_source or "")
        for rule_name, _ in self.trigger_specs_by_node.get(node, ()):
            trigger_key = (node, rule_name)
            trigger_events = self.trigger_event_index.get(trigger_key)
            if not trigger_events:
                continue

            kept = collections.deque()
            for trigger_event in trigger_events:
                (
                    _event_ts,
                    indexed_event_id,
                    _indexed_seq,
                    indexed_alarm_type,
                    indexed_alarm_source,
                ) = self._unpack_trigger_event(trigger_event)
                matches_clear = (
                    indexed_event_id == alarm_id
                    and indexed_alarm_type == alarm_type
                    and indexed_alarm_source == target_alarm_source
                )
                if matches_clear:
                    continue
                kept.append(trigger_event)

            if kept:
                self.trigger_event_index[trigger_key] = kept
            else:
                self.trigger_event_index.pop(trigger_key, None)

    def _finalize_matches_with_history(self, matches):
        """把当前批次结果与历史组做最终合并并落库。"""
        finalized = []
        self.emitted_group_store.prune_expired(self.latest_arrived_event_ts)

        for match_result in matches:
            match_result, merged_group_indexes, should_emit = (
                self.emitted_group_store.merge_with_related(match_result)
            )
            if not should_emit:
                self.emitted_group_store.extend_related_expire_ts(
                    merged_group_indexes,
                    match_result,
                )
                continue
            match_result = self._apply_default_output_site_role_ownership(match_result)
            self.emitted_group_store.replace_and_store(
                merged_group_indexes,
                match_result
            )
            finalized.append(match_result)

        self._prune_consumed_alarm_history(finalized)
        return finalized
