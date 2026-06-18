"""
IncrementalFaultEngine: TurboFlux-inspired continuous subgraph matching engine.

核心思路
--------
fault_grouping 的 TemporalGraphEngine 采用"聚合等待"模型：
  告警到达 → event_cache + pending_triggers → 等待 aggregation_wait_sec →
  批量收割 → _evaluate_rule → 输出故障组

本引擎改为"增量匹配"模型（借鉴 TurboFlux 的多个核心数据结构）：
  告警到达 → event_cache → 立即执行受影响规则的评估 → 输出故障组
  告警失效 → event_cache 删除 → 失效匹配（按 eid 置空历史故障组）

TurboFlux 借鉴点
-----------------
1. RoleSiteIndex（显式候选索引，对应 TurboFlux BuildDCS 的结构预筛选）
   在引擎启动时一次性预建 (rule_name, role) → frozenset[site_id] 映射。
   每次增量匹配时直接 O(1) 查询站点是否结构匹配某 role，省去逐次扫描。
   类比 TurboFlux 的 DCS（Dynamic Candidate Set）中对查询顶点 candidate
   集的静态结构预筛选（BuildDCS top-down pass）。

2. ActiveTriggerTracker（d1 活跃标志，对应 TurboFlux d1/d2 有效性位）
   运行时动态维护 (site, rule) → {eid: ts} 映射：
   - 告警插入且满足 trigger_role 谓词时置为活跃（d1 = True）
   - 告警清除/过期后移除对应 eid；若 eid 集合为空则 d1 = False
   用于 Strategy 2（间接触发）：非 trigger 站点收到新告警时，查询该
   tracker 找到拓扑相邻且仍活跃的 trigger 站点，从这些站点重新评估规则。
   类比 TurboFlux 的 d1 计数器：新增边时沿 DAG 向下传播 d1 有效性。

3. RoleFilteredNeighborCache（对应 StreamingIntermediateCache.candidate_neighbors）
   将 _traverse_graph + role_candidates 的交集结果永久缓存。
   拓扑静态不变，因此该缓存从不失效，彻底消除重复 BFS 和锁竞争开销。
   每次 Strategy 2 连通性检查由完整 BFS 降为单次字典查找 + frozenset 成员测试。

4. NonTriggerAlarmSpecsIndex（类比 _event_seed_roles_for_plan 谓词预过滤）
   与 trigger_specs_by_node 对称，为非 trigger role 预编译告警谓词索引：
   (site, rule, role) → tuple[expected]。
   Strategy 2 在连通性检查之前先做告警类型匹配，若当前告警不满足该 role
   的期望告警集则直接跳过，避免无效的拓扑遍历和评估开销。

5. DCS-style 邻域告警支持检查（对应 _dcs_candidate_supported + support_cache）
   在调用 _evaluate_rule 之前，快速检查 trigger_site 拓扑邻域内
   各必选 non-trigger role 是否至少有一个站点持有活跃的匹配告警。
   结果按 alarm_generation 分代缓存（告警插入/删除时 generation +1，自动失效）。
   等效于 TurboFlux DCS 的 d2 可行性验证：若任意必选 role 无支撑则提前剪枝，
   避免进入复杂度更高的 BFS 回溯评估。

增量匹配策略（alarm INSERT at site S）
---------------------------------------
Strategy 1 – 直接触发
  若 S 对某规则 R 满足 trigger_role 结构+告警谓词 → 立即以 S 为 trigger
  评估 R，无需等待聚合窗口。

Strategy 2 – 间接触发（非 trigger 角色激活）
  若 S 结构上匹配规则 R 的某个非 trigger role，查询 ActiveTriggerTracker
  找出当前对 R 活跃的 trigger 站点集合，过滤出拓扑上与 S 相连的，
  从这些 trigger 站点重新评估 R。
  捕获场景：原有 trigger 已触发但当时缺少某 non-trigger site 的告警，
  新告警的到来使该 pattern 可以完整命中。
  预过滤链（按代价从低到高）：
    ① 告警谓词匹配（NonTriggerAlarmSpecsIndex）—— 纯字典查找
    ② 拓扑连通性检查（RoleFilteredNeighborCache） —— frozenset 成员测试
    ③ 邻域告警支持检查（DCS-style support cache） —— event_cache 遍历+分代缓存
    ④ 全量 BFS 回溯评估（_evaluate_rule）

失效匹配（alarm CLEAR for eid X）
----------------------------------
  - 从 event_cache 删除对应告警
  - 从 ActiveTriggerTracker 中移除 eid X（d1 可能变 False）
  - 在 EmittedGroupStore 中把含有 eid X 的历史组标为 tombstone，
    防止后续被当做有效历史重新合并输出
"""

import collections
import logging
import time

from fault_grouping.alarm_events.identity import require_alarm_identity
from fault_grouping.temporal_engine.engine import TemporalGraphEngine
from fault_grouping.temporal_engine.utils import (
    matches_expected_alarm,
    merge_match_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TurboFlux-inspired static structural index
# ---------------------------------------------------------------------------

class RoleSiteIndex:
    """
    显式静态候选索引（对应 TurboFlux BuildDCS 的结构预筛选）。

    在引擎启动时遍历所有站点和规则，预建：
      _by_rule_role: (rule_name, role) → frozenset[site_id]
      _by_site:      site_id → {(rule_name, role)}

    增量匹配时通过 O(1) 的 frozenset 成员检查替代重复的
    node_rule_helper.matches_node_structure() 调用，减少运行时开销。
    """

    def __init__(self, rules, sites_domain_map, node_rule_helper):
        # (rule_name, role) → frozenset[site_id]
        self._by_rule_role = {}
        # site_id → {(rule_name, role)}
        self._by_site = collections.defaultdict(set)
        self._build(rules, sites_domain_map, node_rule_helper)

    def _build(self, rules, sites_domain_map, node_rule_helper):
        for rule_name, rule in rules.items():
            for role, node_config in rule.get("nodes", {}).items():
                candidates = frozenset(
                    site_id
                    for site_id, domain in sites_domain_map.items()
                    if node_rule_helper.matches_node_structure(domain, node_config)
                )
                self._by_rule_role[(rule_name, role)] = candidates
                for site_id in candidates:
                    self._by_site[site_id].add((rule_name, role))

    def matches(self, rule_name, role, site_id):
        """O(1) 结构匹配检查，利用预建索引。"""
        cands = self._by_rule_role.get((rule_name, role))
        return cands is None or site_id in cands

    def role_candidates(self, rule_name, role):
        """返回结构上匹配该 (rule, role) 的所有站点。"""
        return self._by_rule_role.get((rule_name, role), frozenset())

    def site_rule_roles(self, site_id):
        """返回该站点结构上匹配的所有 (rule_name, role) 对。"""
        return self._by_site.get(site_id, set())

    @property
    def entry_count(self):
        return len(self._by_rule_role)


# ---------------------------------------------------------------------------
# TurboFlux-inspired dynamic d1 validity tracker
# ---------------------------------------------------------------------------

class ActiveTriggerTracker:
    """
    动态维护"活跃 trigger 告警"的 d1 有效性状态。

    类比 TurboFlux d1 标志：
    - d1[(site, rule)] = True  ↔  该站点当前至少有一条满足 trigger_role
      告警谓词的活跃告警
    - 告警插入 → add(site, rule, eid, ts) → d1 = True
    - 告警清除 → remove_eid_from_all(eid) → 若 eid 集合变空则 d1 = False

    用于 Strategy 2 增量匹配：新告警到达非 trigger 站点 N 时，
    查询 active_sites_for_rule(rule) 得到所有 d1=True 的 trigger 站点，
    过滤拓扑相连的，再从这些站点重新评估规则。
    """

    def __init__(self):
        # (site, rule_name) → {eid: ts}
        self._active = collections.defaultdict(dict)
        # rule_name → {site: max_ts}   用于 O(1) 按规则迭代
        self._by_rule = collections.defaultdict(dict)

    def add(self, site, rule_name, identity, ts):
        """注册 eid 为 (site, rule) 的活跃 trigger 告警（d1 置为 True）。"""
        key = (site, rule_name)
        self._active[key][identity] = ts
        prev = self._by_rule[rule_name].get(site, 0.0)
        if ts > prev:
            self._by_rule[rule_name][site] = ts

    def remove_identity_from_all(self, identity):
        """
        从所有 (site, rule) 记录中删除 eid。
        返回被影响的 (site, rule_name) 列表。
        """
        affected = []
        for key in list(self._active):
            if identity not in self._active[key]:
                continue
            site, rule_name = key
            del self._active[key][identity]
            affected.append(key)
            if not self._active[key]:
                # eid 集合为空 → d1 = False，移除记录
                del self._active[key]
                self._by_rule[rule_name].pop(site, None)
            else:
                self._by_rule[rule_name][site] = max(self._active[key].values())
        return affected

    def get_latest_ts(self, site, rule_name):
        """返回 (site, rule) 最新 trigger 告警时间，无活跃告警则返回 None。"""
        eids = self._active.get((site, rule_name))
        return max(eids.values()) if eids else None

    def is_active(self, site, rule_name):
        """d1 检查：该 (site, rule) 是否有活跃 trigger 告警。"""
        return bool(self._active.get((site, rule_name)))

    def active_sites_for_rule(self, rule_name):
        """返回当前对规则 rule_name d1=True 的所有站点及其最新 ts。"""
        return dict(self._by_rule.get(rule_name, {}))

    def prune_expired(self, current_ts, global_ttl):
        """
        清理超过 TTL 的过期 trigger 记录。
        在每次事件处理后按需调用，防止内存无限增长。
        """
        cutoff = current_ts - global_ttl
        for key in list(self._active):
            eids = self._active.get(key)
            if not eids:
                continue
            expired = [eid for eid, ts in eids.items() if ts < cutoff]
            for eid in expired:
                del eids[eid]
            if not eids:
                site, rule_name = key
                del self._active[key]
                self._by_rule[rule_name].pop(site, None)
            elif expired:
                site, rule_name = key
                self._by_rule[rule_name][site] = max(eids.values())


# ---------------------------------------------------------------------------
# Incremental fault matching engine
# ---------------------------------------------------------------------------

class IncrementalFaultEngine(TemporalGraphEngine):
    """
    基于 TurboFlux 索引思想的增量故障匹配引擎。

    继承 TemporalGraphEngine 以复用：
    - 拓扑索引（topo_down / topo_up / site_chain_index）
    - 规则评估逻辑（_evaluate_rule / _traverse_graph 等所有 mixin 方法）
    - 事件缓存管理（event_cache / _prune_expired_raw_events_in_place）
    - 历史故障组去重（emitted_group_store / _finalize_matches_with_history）
    - trigger_specs_by_node（预编译 site → 可触发的 rule+alarm 规格）

    覆盖或禁用：
    - process_event：移除聚合等待，改为立即增量匹配
    - flush_pending：无 pending_triggers，直接返回空列表
    - advance_watermark：只做过期清理，不触发收割
    """

    def __init__(
        self,
        topo_downstream_map,
        rules_config,
        site_domain_map,
        alarm_source_domain_map=None,
        site_merge_helper=None,
        site_chain_index=None,
        use_alarm_period_cache=False,
    ):
        # 调用父类初始化，聚合等待时间设为 0（字段仍会存在，但从不使用）
        super().__init__(
            topo_downstream_map=topo_downstream_map,
            rules_config=rules_config,
            site_domain_map=site_domain_map,
            alarm_source_domain_map=alarm_source_domain_map,
            aggregation_wait_sec=0,
            site_merge_helper=site_merge_helper,
            site_chain_index=site_chain_index,
            use_alarm_period_cache=use_alarm_period_cache,
        )

        # ── TurboFlux 显式静态候选索引 ────────────────────────────────────
        logger.info("IncrementalFaultEngine: 正在构建 RoleSiteIndex（TurboFlux 显式候选索引）...")
        self._role_site_index = RoleSiteIndex(
            self.rules,
            self.sites_domain_map,
            self.node_rule_helper,
        )
        logger.info(
            "IncrementalFaultEngine: RoleSiteIndex 就绪，共 %d 条 (rule,role) 索引项",
            self._role_site_index.entry_count,
        )

        # ── TurboFlux d1 动态活跃 trigger 跟踪器 ──────────────────────────
        self._trigger_tracker = ActiveTriggerTracker()

        # ── 非 trigger 角色参与索引 ────────────────────────────────────────
        # site_id → [(rule_name, role)]  仅包含非 trigger role
        # 用于 Strategy 2：快速定位新告警站点可能参与的非 trigger 规则
        self._non_trigger_index = self._build_non_trigger_index()

        # ── 加速结构 1：角色过滤邻居永久缓存 ─────────────────────────────
        # 对应 fault_csm_codex StreamingIntermediateCache.candidate_neighbors
        # key: (source_site, direction, max_hops_str, rule_name, target_role)
        #   → frozenset[site_id]  (topology × role_candidates 交集，永不失效)
        # 拓扑静态，因此无需分代失效；彻底消除重复 BFS 和 _topo_cache_lock 竞争。
        self._role_neighbor_cache = {}

        # ── 加速结构 2：非 trigger 告警谓词预过滤索引 ────────────────────
        # 对应 fault_csm_codex _event_seed_roles_for_plan 的非 trigger 侧谓词
        # key: (site_id, rule_name, role) → tuple[expected]
        # Strategy 2 连通性检查前先做告警类型快速匹配，不满足则跳过后续全部开销
        self._non_trigger_alarm_specs = self._build_non_trigger_alarm_specs()

        # ── 加速结构 3：DCS-style 邻域告警支持分代缓存 ───────────────────
        # 对应 fault_csm_codex _dcs_candidate_supported + StreamingIntermediateCache.support_cache
        # key: (alarm_generation, trigger_site, rule_name) → bool
        # 每次告警插入/删除 _alarm_generation +1，旧缓存条目自动失效
        # 值为 True 表示该 trigger_site 当前邻域内各必选 non-trigger role 均有活跃告警支撑
        self._support_cache: dict = {}
        self._alarm_generation: int = 0

    # ------------------------------------------------------------------
    # 预编译：非 trigger 角色参与索引
    # ------------------------------------------------------------------

    def _build_non_trigger_index(self):
        """
        预建 site → [(rule_name, non_trigger_role)] 索引。

        Strategy 2 依赖此索引：当新告警到达非 trigger 站点 S 时，
        直接查表得到 S 可能参与的规则+角色组合，避免全规则扫描。
        """
        index = collections.defaultdict(list)
        for rule_name, rule in self.rules.items():
            trigger_role = rule["trigger_role"]
            for role in rule.get("nodes", {}):
                if role == trigger_role:
                    continue
                for site_id in self._role_site_index.role_candidates(rule_name, role):
                    index[site_id].append((rule_name, role))
        return dict(index)

    # ------------------------------------------------------------------
    # 预编译：非 trigger 告警谓词索引（加速结构 2）
    # ------------------------------------------------------------------

    def _build_non_trigger_alarm_specs(self):
        """
        预编译 (site, rule_name, role) → tuple[expected] 索引。

        与 trigger_specs_by_node 对称：为每个非 trigger role 在每个候选站点上
        预算出期望告警集合，供 Strategy 2 做到达告警的快速类型匹配。
        若返回值为空 tuple（role 无告警约束，如纯结构 context 节点），
        则任何告警类型都通过，不做过滤。
        """
        specs = {}
        for rule_name, rule in self.rules.items():
            trigger_role = rule["trigger_role"]
            for role, node_config in rule.get("nodes", {}).items():
                if role == trigger_role:
                    continue
                for site_id in self._role_site_index.role_candidates(rule_name, role):
                    node_domain = self.sites_domain_map.get(site_id, {})
                    expected_list = self._collect_trigger_expected_list(node_domain, node_config)
                    # 存 tuple（可为空，空表示无告警约束，不过滤）
                    specs[(site_id, rule_name, role)] = tuple(expected_list)
        return specs

    # ------------------------------------------------------------------
    # 覆盖父类核心接口
    # ------------------------------------------------------------------

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
        """
        处理单条告警事件，立即执行增量匹配。

        与父类 TemporalGraphEngine.process_event 的区别：
        - 不向 pending_triggers 队列入队
        - 不等待 aggregation_wait_sec
        - 告警插入后立即评估受影响规则（Strategy 1 + Strategy 2）
        - 告警清除后立即执行失效匹配
        """
        with self._lock:
            identity = require_alarm_identity({
                "eid": event_id,
                "occurrence_uuid": occurrence_uuid,
            })
            self.current_watermark = max(self.current_watermark, ts)
            self.latest_arrived_event_ts = max(self.latest_arrived_event_ts, ts)
            # 先清理该站点过期缓存，再写入新事件
            self._prune_expired_raw_events_in_place(node, ts)

            if is_clear:
                return self._handle_alarm_clear(
                    node,
                    event_id,
                    occurrence_uuid,
                    alarm_type,
                    alarm_source,
                    ts,
                )

            # 将告警写入 event_cache（与父类格式兼容）
            self.event_cache[node].append(
                (ts, event_id, alarm_type, alarm_source, frozenset(), occurrence_uuid)
            )

            # 更新 TurboFlux d1 状态：若该告警满足某规则的 trigger 谓词，则记为活跃
            alarm_source_domain = self.alarm_source_domain_map.get(alarm_source, "")
            for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
                if any(
                    matches_expected_alarm(alarm_type, exp, alarm_source_domain)
                    for exp in expected_list
                ):
                    self._trigger_tracker.add(node, rule_name, identity, ts)

            # 告警插入使邻域支持状态可能改变：推进分代，失效支持缓存
            self._alarm_generation += 1

        # 锁外执行增量匹配（避免长时间持锁）
        return self._incremental_match(node, alarm_type, ts, alarm_source)

    def flush_pending(self):
        """增量模式下无 pending_triggers，直接返回空列表。"""
        return []

    def advance_watermark(self, now_ts=None):
        """仅推进水印并清理过期状态，不触发收割。"""
        with self._lock:
            if now_ts is None:
                now_ts = time.time()
            self.current_watermark = max(self.current_watermark, now_ts)
            self._prune_expired_state_locked(self.latest_arrived_event_ts)
        return []

    # ------------------------------------------------------------------
    # 失效匹配：告警清除
    # ------------------------------------------------------------------

    def _handle_alarm_clear(
        self,
        node,
        event_id,
        occurrence_uuid,
        alarm_type,
        alarm_source,
        ts,
    ):
        """
        处理告警清除事件（在 self._lock 内调用）：

        1. 从 event_cache 删除对应告警
        2. 从 ActiveTriggerTracker 移除 eid（d1 可能变 False）
        3. 在 EmittedGroupStore 中将含有该 eid 的历史组标为 tombstone
           （失效匹配：该告警已不活跃，关联的故障组不再有效）
        """
        identity = (str(event_id), str(occurrence_uuid))
        self._remove_cleared_events(
            node,
            event_id,
            occurrence_uuid,
            alarm_type=alarm_type,
            alarm_source=alarm_source,
        )
        affected = self._trigger_tracker.remove_identity_from_all(identity)
        if affected:
            logger.debug(
                "告警清除 eid=%s，从 %d 条 (site,rule) trigger 记录中移除",
                event_id,
                len(affected),
            )
        # 告警删除同样推进分代，失效支持缓存
        self._alarm_generation += 1
        # 失效匹配：标记含该 eid 的历史组为无效
        self._invalidate_history_by_identity(identity)
        return []

    def _invalidate_history_by_identity(self, identity):
        """
        失效匹配核心：将 EmittedGroupStore 中含有 eid 的历史组标为 tombstone。

        直接操作 EmittedGroupStore 内部的 groups 列表和 eid_to_group_indexes
        索引（与 CSM RemoveEdge 后的 DCS 失效传播类似）：
        - 找到所有引用该 eid 的组索引
        - 将对应 groups 槽位置为 None（tombstone）
        - 当 tombstone 占比超过阈值时触发索引重建，防止内存泄漏
        """
        store = self.emitted_group_store
        affected_indexes = set(store.eid_to_group_indexes.get(identity, ()))
        if not affected_indexes:
            return

        invalidated = 0
        for idx in affected_indexes:
            if 0 <= idx < len(store.groups) and store.groups[idx] is not None:
                store.groups[idx] = None
                store.deleted_group_count += 1
                invalidated += 1

        if invalidated:
            logger.debug(
                "失效匹配：eid=%s，置空 %d 条历史故障组",
                identity,
                invalidated,
            )

        # tombstone 占比过高时触发重建，避免内存无限增长
        total = len(store.groups)
        if total > 0 and store.deleted_group_count / total > 0.3:
            store._rebuild_alarm_index()
            store.deleted_group_count = 0

    # ------------------------------------------------------------------
    # 增量匹配：告警插入
    # ------------------------------------------------------------------

    def _incremental_match(self, node, alarm_type, ts, alarm_source):
        """
        新告警到达时的两路增量匹配。

        Strategy 1 – 直接触发
          若 (node, alarm_type) 满足某规则 R 的 trigger_role 谓词，
          立即以 node 为 trigger 评估 R。

        Strategy 2 – 间接触发（非 trigger 角色激活）
          若 node 结构上匹配规则 R 的某个非 trigger role，
          查询 ActiveTriggerTracker 中对 R 当前活跃的 trigger 站点集合，
          经预过滤链（①告警谓词 → ②连通性 → ③邻域支持）剪枝后，
          从剩余 trigger 站点重新评估 R。
        """
        alarm_source_domain = self.alarm_source_domain_map.get(alarm_source, "")
        eval_caches = self._create_eval_caches()
        raw_matches = []
        # 记录本次事件已评估的 (trigger_site, rule_name)，避免重复评估
        evaluated = set()

        # ── Strategy 1: 直接触发 ─────────────────────────────────────────
        for rule_name, expected_list in self.trigger_specs_by_node.get(node, ()):
            if not any(
                matches_expected_alarm(alarm_type, exp, alarm_source_domain)
                for exp in expected_list
            ):
                continue
            key = (node, rule_name)
            if key in evaluated:
                continue
            evaluated.add(key)
            results = self._evaluate_rule(
                rule_name,
                self.rules[rule_name],
                node,
                ts,
                node_rule_helper=self.node_rule_helper,
                eval_caches=eval_caches,
            )
            raw_matches.extend(results)

        # ── Strategy 2: 间接触发（非 trigger 角色激活）────────────────────
        for rule_name, non_trigger_role in self._non_trigger_index.get(node, ()):
            # ① 加速：告警谓词预过滤（NonTriggerAlarmSpecsIndex）
            #    确认当前告警类型满足该 non_trigger_role 的期望告警集
            #    空 expected_list 表示该 role 无告警约束（context-only），放行
            expected_specs = self._non_trigger_alarm_specs.get((node, rule_name, non_trigger_role))
            if expected_specs:  # non-empty → role has alarm constraint
                if not any(
                    matches_expected_alarm(alarm_type, exp, alarm_source_domain)
                    for exp in expected_specs
                ):
                    continue  # 告警类型不匹配，跳过拓扑检查和评估

            # 获取该规则当前所有 d1=True 的 trigger 站点
            active_triggers = self._trigger_tracker.active_sites_for_rule(rule_name)
            if not active_triggers:
                continue
            rule = self.rules[rule_name]
            trigger_role = rule["trigger_role"]

            for trigger_site, trigger_ts in active_triggers.items():
                if trigger_site == node:
                    continue
                key = (trigger_site, rule_name)
                if key in evaluated:
                    continue

                # ② 加速：连通性检查（RoleFilteredNeighborCache）
                #    利用永久缓存的 BFS×role_candidates 交集，O(1) frozenset 成员测试
                if not self._are_rule_connected_cached(
                    trigger_site, node, trigger_role, non_trigger_role, rule, rule_name
                ):
                    continue

                # ③ 加速：DCS-style 邻域告警支持检查（分代缓存）
                #    验证 trigger_site 拓扑邻域中各必选 non-trigger role 均有活跃告警
                if not self._has_neighborhood_support(trigger_site, rule_name, rule, trigger_ts):
                    continue

                evaluated.add(key)
                results = self._evaluate_rule(
                    rule_name,
                    rule,
                    trigger_site,
                    trigger_ts,
                    node_rule_helper=self.node_rule_helper,
                    eval_caches=eval_caches,
                )
                raw_matches.extend(results)

        if not raw_matches:
            return []

        # 批内合并（同一轮事件产出的多个候选组去重）
        merged, merge_stats = merge_match_batch(
            raw_matches,
            site_merge_helper=self.site_merge_helper,
            return_stats=True,
        )

        # 与历史故障组合并并落库
        with self._lock:
            self._record_batch_merge_stats_locked(merge_stats)
            self._prune_expired_state_locked(self.latest_arrived_event_ts)
            finalized = self._finalize_matches_with_history(merged)

        owned = self._apply_default_output_site_role_ownership_to_matches(finalized)
        return self._apply_output_visibility_filters_to_matches(owned)

    # ------------------------------------------------------------------
    # 加速结构 1：角色过滤邻居永久缓存 + 连通性检查
    # 对应 StreamingIntermediateCache.candidate_neighbors
    # ------------------------------------------------------------------

    def _role_edge_neighbors(self, rule_name, source_site, target_role, direction, max_hops):
        """
        返回从 source_site 沿 direction/max_hops 可达的、结构上匹配
        (rule_name, target_role) 的所有站点（frozenset）。

        结果永久缓存：拓扑静态不变，无需失效。
        与 fault_csm_codex StreamingIntermediateCache.candidate_neighbors 等价。
        """
        key = (source_site, direction, max_hops, rule_name, target_role)
        if key not in self._role_neighbor_cache:
            reachable = self._traverse_graph(
                start_node=source_site,
                direction=direction,
                max_hops=max_hops,
            )
            role_candidates = self._role_site_index.role_candidates(rule_name, target_role)
            self._role_neighbor_cache[key] = frozenset(reachable) & role_candidates
        return self._role_neighbor_cache[key]

    def _are_rule_connected_cached(
        self, trigger_site, candidate_site, trigger_role, non_trigger_role, rule, rule_name
    ):
        """
        利用永久缓存的角色过滤邻居集合检查 trigger_site 与 candidate_site 是否
        在规则拓扑约束内相连。

        相比原 _traverse_graph 调用，每次检查降为一次字典查找 + frozenset 成员测试，
        完全消除重复 BFS 和 _topo_cache_lock 竞争开销。

        对规则每条边：
        - 若 trigger 是边的 source → 从 trigger_site 出发，看 candidate_site 是否可达
        - 若 trigger 是边的 target → 从 candidate_site 出发，看 trigger_site 是否可达
        - 多跳中间节点规则（trigger/candidate 不直接相连的边）也作双向检查兜底
        """
        for edge in rule.get("edges", []):
            src_role = edge.get("source", "")
            tgt_role = edge.get("target", "")
            direction = edge.get("direction", "downstream")
            max_hops = edge.get("max_hops")

            if src_role == trigger_role and tgt_role == non_trigger_role:
                # trigger 是 source：从 trigger_site 出发检查 candidate_site
                if candidate_site in self._role_edge_neighbors(
                    rule_name, trigger_site, non_trigger_role, direction, max_hops
                ):
                    return True
            elif src_role == non_trigger_role and tgt_role == trigger_role:
                # candidate 是 source：从 candidate_site 出发检查 trigger_site
                if trigger_site in self._role_edge_neighbors(
                    rule_name, candidate_site, trigger_role, direction, max_hops
                ):
                    return True
            else:
                # 多角色规则：trigger/candidate 不直接相连，双向兜底
                if candidate_site in self._role_edge_neighbors(
                    rule_name, trigger_site, tgt_role, direction, max_hops
                ):
                    return True
                if trigger_site in self._role_edge_neighbors(
                    rule_name, candidate_site, tgt_role, direction, max_hops
                ):
                    return True
        return False

    # ------------------------------------------------------------------
    # 加速结构 3：DCS-style 邻域告警支持检查（分代缓存）
    # 对应 fault_csm_codex _dcs_candidate_supported + support_cache
    # ------------------------------------------------------------------

    def _has_neighborhood_support(self, trigger_site, rule_name, rule, trigger_ts):
        """
        DCS-style 邻域告警支持检查：验证 trigger_site 在规则拓扑内的邻域中，
        每个必选 non-trigger role 至少有一个站点持有活跃的匹配告警。

        等效于 TurboFlux DCS 的 d2 可行性验证：若任意必选 role 无支撑，
        该 trigger_site 对当前规则的评估必然失败，可提前剪枝。

        结果按 alarm_generation 分代缓存：
        - 每次告警插入/删除 _alarm_generation +1
        - 旧 generation 的缓存条目不再被查到，等效于自动失效
        - 分代避免了主动清除缓存的开销（空间由 dict 自然增长，可按需截断）
        对应 fault_csm_codex StreamingIntermediateCache.support_get/support_set 机制。
        """
        cache_key = (self._alarm_generation, trigger_site, rule_name)
        cached = self._support_cache.get(cache_key)
        if cached is not None:
            return cached

        trigger_role = rule["trigger_role"]
        result = True

        for edge in rule.get("edges", []):
            src_role = edge.get("source", "")
            tgt_role = edge.get("target", "")
            optional = edge.get("optional", False)
            if optional:
                continue

            # 确定从 trigger_site 出发需要覆盖的 non-trigger role 及遍历方向
            if src_role == trigger_role:
                check_role = tgt_role
                direction = edge.get("direction", "downstream")
                max_hops = edge.get("max_hops")
                neighbor_sites = self._role_edge_neighbors(
                    rule_name, trigger_site, check_role, direction, max_hops
                )
            elif tgt_role == trigger_role:
                check_role = src_role
                # 从 non-trigger 出发到 trigger：反向检查，从邻域找 src_role 站点
                # 用 role_candidates 筛出结构匹配的站点再检查拓扑
                direction = edge.get("direction", "downstream")
                max_hops = edge.get("max_hops")
                neighbor_sites = self._role_edge_neighbors(
                    rule_name, trigger_site, check_role, direction, max_hops
                )
                if not neighbor_sites:
                    # 尝试反向：从结构候选中找能到达 trigger_site 的站点
                    candidates = self._role_site_index.role_candidates(rule_name, check_role)
                    neighbor_sites = frozenset(
                        s for s in candidates
                        if trigger_site in self._role_edge_neighbors(
                            rule_name, s, trigger_role, direction, max_hops
                        )
                    )
            else:
                # 该边不直接涉及 trigger_role，跳过（多跳中间节点由评估器处理）
                continue

            # 检查 neighbor_sites 中是否至少有一个有活跃告警
            has_active = False
            for site in neighbor_sites:
                if self.event_cache.get(site):
                    has_active = True
                    break
            if not has_active:
                result = False
                break

        # 支持缓存过大时清理旧分代（防止内存持续增长）
        if len(self._support_cache) > 50000:
            self._support_cache.clear()

        self._support_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 旧接口保留（内部调用路径重定向到带缓存版本）
    # ------------------------------------------------------------------

    def _are_rule_connected(self, trigger_site, candidate_site, rule):
        """向后兼容接口，内部已被 _are_rule_connected_cached 替代。"""
        rule_name = None
        for rn, r in self.rules.items():
            if r is rule:
                rule_name = rn
                break
        if rule_name is None:
            # 降级：直接遍历
            for edge in rule.get("edges", []):
                direction = edge.get("direction", "downstream")
                max_hops = edge.get("max_hops")
                reachable = self._traverse_graph(
                    start_node=trigger_site, direction=direction, max_hops=max_hops
                )
                if candidate_site in reachable:
                    return True
                reachable2 = self._traverse_graph(
                    start_node=candidate_site, direction=direction, max_hops=max_hops
                )
                if trigger_site in reachable2:
                    return True
            return False
        # non_trigger_role 未知时作双向全边检查
        for edge in rule.get("edges", []):
            tgt_role = edge.get("target", "")
            direction = edge.get("direction", "downstream")
            max_hops = edge.get("max_hops")
            if candidate_site in self._role_edge_neighbors(
                rule_name, trigger_site, tgt_role, direction, max_hops
            ):
                return True
            if trigger_site in self._role_edge_neighbors(
                rule_name, candidate_site, tgt_role, direction, max_hops
            ):
                return True
        return False
