"""ANCHOR 二次汇聚（离线批处理入口，不依赖 match_rules.py）。

对外接口只有 BatchFaultGroupMatcher.aggregate_alarm_groups：输入已成组的
故障组集合 {故障组id: [告警, ...]}，输出 {汇聚故障组id: [故障组id, ...]}。
告警为告警生成器（alarm_events.generator.AlarmGenerator）yield 出的字典，
字段形如：

    {
        "站点ID": "site-1",
        "告警标题": "Link Down",
        "告警首次发生时间": 123.5,          # 数值时间戳或时间文本
        "告警编码ID": "eid-1::<uuid>",
        "告警源": "ne-1",
        "物理端口名称": "port-1",
        "是否清除": True,                   # 可选，缺省视为上报告警
    }

与在线 match_rules.py 的流式收割不同，批处理不考虑聚合成熟时间：

1. 整批告警按发生时间正序喂给引擎，只更新状态（事件缓存 / trigger 索引），
   过程中不做任何收割——引擎的聚合等待时间被设为无穷大，pending 永不成熟；
2. 喂流结束后，对每个 trigger 告警逐一触发故障组汇聚，按告警 ts 从早到晚
   进行（与在线成熟队列的收割顺序一致）；每次汇聚成功立即执行既有的消费
   回收逻辑（_prune_consumed_alarm_history），把已被故障组消费的 trigger
   告警从 trigger 索引中删除，后续同类 trigger 因此不再重复汇聚；
3. 全部 trigger 处理完后统一做批内合并（merge_match_batch），再走与在线
   一致的历史合并与输出可见性过滤；
4. 匹配出的故障组经可落盘规则过滤、故障模式过滤后，作为原始故障组之间的
   关联证据：同一匹配组覆盖到的原始组用并查集汇聚到一起，未被覆盖的原始
   组独立成组，最后按 max_group_member 做后处理拆分。
"""

import copy

from types import SimpleNamespace

from anchor_grouping_online.alarm_events.generator import to_matching_alarm
from anchor_grouping_online.matching.runtime import (
    build_fault_pattern_filter,
    build_rules_config,
    collect_output_eligible_rules,
    load_static_context,
)
from anchor_grouping_online.temporal_engine.engine import TemporalGraphEngine
from anchor_grouping_online.temporal_engine.utils import merge_match_batch
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


class BatchFaultGroupMatcher:
    """按批次做 ANCHOR 二次汇聚。

    静态拓扑上下文与规则配置在构造时加载一次，可跨批次复用；
    时序引擎是有状态的，每次 aggregate_alarm_groups() 都新建一个，
    批与批之间互不影响。
    """

    def __init__(self, resource_buffer=RESOURCE_BUFFER_JSONL, static_context=None):
        if static_context is None:
            static_context = load_static_context(
                SimpleNamespace(resource_buffer=resource_buffer)
            )
        self.static_context = static_context
        self.rules_config = build_rules_config()
        self.output_eligible_rules = collect_output_eligible_rules(self.rules_config)
        self.fault_pattern_filter = build_fault_pattern_filter(static_context)

    def aggregate_alarm_groups(
        self,
        alarm_groups,
        associate_time,
        max_group_time,
        max_group_member,
    ):
        """ANCHOR 二次汇聚：按拓扑匹配关系把已成组的故障组合并为汇聚故障组。

        整体逻辑：拿全量告警按 ANCHOR 规则聚合故障组，再通过并查集把原始组
        汇聚到一起——alarm1 属于 group1、alarm2 属于 group2，若 alarm1 与
        alarm2 被聚合进同一个故障组，则 group1 与 group2 汇聚为一组；
        未被任何匹配组覆盖到的原始 group 独立成组。

        输入参数
            alarm_groups     Dict[str, List]  必选
                待二次汇聚的故障组集合。key 为故障组 ID，value 为该故障组包含
                的告警列表（告警为告警生成器输出的字典，见模块 docstring）。
                格式：{故障组id1: [告警1, ...], ...}
            associate_time   int              必选
                关联告警的时间窗，单位分钟。即匹配阶段的规则边时间窗
                （替换 RULE_DEFAULT_EDGE_TIME_WINDOW_SEC）。
            max_group_time   int              必选
                汇聚后故障组允许的最大时间窗，单位分钟。即匹配阶段的组保留
                时长（替换 RULE_DEFAULT_MAX_STAY_TIME_SEC）。
            max_group_member int              必选
                汇聚后故障组允许的最大告警数量（按去重后的告警编码 ID 计）。
                作为后处理生效：并查集汇聚完成后，超限的汇聚组按成员时间顺序
                贪心拆分为多个汇聚组；单个原始组自身超限时不拆分、独立成组。

        输出参数
            agg_alarm_groups Dict[str, List]
                二次汇聚后的故障组集合。key 为汇聚故障组 ID（agg_group_1 起
                顺序编号，按组内最早告警时间排序），value 为被汇聚到该组内的
                原始故障组 ID 列表。每个输入故障组恰好出现在一个汇聚组中。
                格式：{汇聚故障组id1: [故障组id1, ...], ...}
        """
        associate_window_sec = float(associate_time) * 60.0
        max_stay_sec = float(max_group_time) * 60.0
        max_member_count = int(max_group_member)

        # 1. 展平输入：告警按编码 ID 去重，记录归属组与各组时间范围/成员集。
        unique_alarms = {}
        alarm_to_group_ids = {}
        group_alarm_ids = {}
        group_min_ts = {}
        group_max_ts = {}
        for group_id, group_alarms in alarm_groups.items():
            alarm_ids = set()
            min_ts = None
            max_ts = None
            for generated_alarm in group_alarms or ():
                matching_alarm = to_matching_alarm(generated_alarm)
                alarm_id = matching_alarm["alarm_id"]
                ts = matching_alarm["ts"]
                unique_alarms.setdefault(alarm_id, generated_alarm)
                owner_group_ids = alarm_to_group_ids.setdefault(alarm_id, [])
                if group_id not in owner_group_ids:
                    owner_group_ids.append(group_id)
                alarm_ids.add(alarm_id)
                min_ts = ts if min_ts is None else min(min_ts, ts)
                max_ts = ts if max_ts is None else max(max_ts, ts)
            group_alarm_ids[group_id] = alarm_ids
            group_min_ts[group_id] = min_ts
            group_max_ts[group_id] = max_ts

        # 2. 用 associate_time / max_group_time 替换匹配阶段的两个时间参数，
        #    对全量去重告警跑一遍批量匹配。
        engine = self._new_engine(
            _override_rule_time_config(
                self.rules_config, associate_window_sec, max_stay_sec
            )
        )
        # 输入故障组可能横跨很长时间；事件 TTL 只是在线流的内存回收机制，
        # 批内不应让它把早期 trigger 清掉，否则早期故障组之间无法建立关联。
        # 放宽到覆盖整批时间跨度 + 关联窗。
        valid_min_ts = [ts for ts in group_min_ts.values() if ts is not None]
        valid_max_ts = [ts for ts in group_max_ts.values() if ts is not None]
        batch_span_sec = (max(valid_max_ts) - min(valid_min_ts)) if valid_min_ts else 0.0
        ttl_floor_sec = batch_span_sec + associate_window_sec
        engine.global_ttl = max(engine.global_ttl, ttl_floor_sec)
        engine.power_alarm_ttl = max(engine.power_alarm_ttl, ttl_floor_sec)
        output_matches = self._run_matching(unique_alarms.values(), engine=engine)

        # 3. 并查集汇聚：同一个匹配组覆盖到的原始组无条件汇聚到一起。
        parent = {group_id: group_id for group_id in alarm_groups}

        def find_root(group_id):
            root = group_id
            while parent[root] != root:
                root = parent[root]
            while parent[group_id] != root:
                parent[group_id], group_id = root, parent[group_id]
            return root

        for match in self._iter_output_matches(output_matches):
            related_group_ids = []
            for alarm_id in _collect_symptom_alarm_ids(match):
                for group_id in alarm_to_group_ids.get(alarm_id, ()):
                    if group_id not in related_group_ids:
                        related_group_ids.append(group_id)
            if len(related_group_ids) < 2:
                continue
            base_root = find_root(related_group_ids[0])
            for other_group_id in related_group_ids[1:]:
                other_root = find_root(other_group_id)
                if other_root != base_root:
                    parent[other_root] = base_root

        # 4. 收集连通分量；未与任何组关联上的原始组即单元素分量，独立成组。
        components = {}
        for group_id in alarm_groups:
            components.setdefault(find_root(group_id), []).append(group_id)

        def member_order_key(group_id):
            min_ts = group_min_ts.get(group_id)
            return (min_ts is None, min_ts if min_ts is not None else 0.0, str(group_id))

        # 5. 后处理：按 max_group_member 拆分超限的汇聚组。成员按最早告警时间
        #    排序后贪心装包（去重告警数不超过上限即并入当前包，否则另起新包）；
        #    单个原始组自身超限时无法再拆，独立成组。
        packed_groups = []
        for member_group_ids in components.values():
            ordered_members = sorted(member_group_ids, key=member_order_key)
            current_members = []
            current_alarm_ids = set()
            for group_id in ordered_members:
                candidate_alarm_ids = current_alarm_ids | group_alarm_ids[group_id]
                if current_members and len(candidate_alarm_ids) > max_member_count:
                    packed_groups.append(current_members)
                    current_members = [group_id]
                    current_alarm_ids = set(group_alarm_ids[group_id])
                else:
                    current_members.append(group_id)
                    current_alarm_ids = candidate_alarm_ids
            if current_members:
                packed_groups.append(current_members)

        # 6. 输出：按组内最早告警时间排序编号。
        def packed_order_key(member_group_ids):
            head = member_group_ids[0]
            min_ts = group_min_ts.get(head)
            return (min_ts is None, min_ts if min_ts is not None else 0.0, str(head))

        packed_groups.sort(key=packed_order_key)
        return {
            f"agg_group_{index}": member_group_ids
            for index, member_group_ids in enumerate(packed_groups, start=1)
        }

    def _run_matching(self, generated_alarms, engine):
        """喂流 + 逐 trigger 汇聚 + 批内合并，返回待输出的 match 列表。"""
        # 1. 缓存整批告警并转成引擎内部字段，按发生时间正序喂流，
        #    保证 latest_arrived_event_ts 单调推进与 TTL 清理正确。
        matching_alarms = sorted(
            (to_matching_alarm(alarm) for alarm in generated_alarms),
            key=lambda alarm: alarm["ts"],
        )

        for alarm in matching_alarms:
            alarm_payload = None
            if alarm["physical_port_name"]:
                alarm_payload = {"物理端口名称": alarm["physical_port_name"]}
            # 聚合等待为无穷大，pending 永不成熟，这里必然返回空列表，
            # 喂流阶段只累积事件缓存与 trigger 索引。
            engine.process_event(
                node=alarm["site_id"],
                alarm_source=alarm["alarm_source"],
                alarm_type=alarm["alarm_title"],
                ts=alarm["ts"],
                alarm_id=alarm["alarm_id"],
                alarm_payload=alarm_payload,
                is_clear=alarm["is_clear"],
            )

        raw_matches = self._aggregate_per_trigger(engine)
        return self._merge_batch_and_finalize(engine, raw_matches)

    def _new_engine(self, rules_config):
        """新建批处理引擎：聚合等待时间为无穷大，喂流阶段不触发收割。"""
        static_context = self.static_context
        return TemporalGraphEngine(
            rules_config,
            static_context.site_domain_map,
            alarm_source_domain_map=static_context.alarm_source_domain_map,
            aggregation_wait_sec=float("inf"),
            topo_downstream_map=static_context.topo_downstream_map,
            site_chain_index=static_context.site_chain_index,
            ne_graph_data=static_context.ne_graph_data,
            site_to_ne_ids=static_context.site_to_ne_ids,
            link_peer_index=static_context.link_peer_index,
        )

    @staticmethod
    def _aggregate_per_trigger(engine):
        """按 trigger 告警 ts 从早到晚逐个触发故障组汇聚，返回原始 match 列表。

        顺序与在线成熟队列的收割顺序一致（首触发时间升序），因此消费回收的
        cutoff 只会清理不晚于本组症状时间的历史，不影响更晚的独立故障。
        每次汇聚成功后立即调用引擎既有的 _prune_consumed_alarm_history：
        被故障组消费的 trigger 告警会从 trigger_event_index 中删除
        （与在线收割后的 trigger 删除逻辑一致），后续 trigger 若已被消费
        会被直接跳过，不再重复汇聚。
        """
        trigger_candidates = []
        for trigger_key, trigger_events in engine.trigger_event_index.items():
            for trigger_event in trigger_events:
                event_ts, _alarm_id, event_seq, _alarm_type, _alarm_source = (
                    engine._unpack_trigger_event(trigger_event)
                )
                trigger_candidates.append((event_ts, event_seq, trigger_key))
        # ts 从早到晚；同一时刻按到达序号从先到后，保证顺序确定。
        trigger_candidates.sort()

        raw_matches = []
        for event_ts, event_seq, trigger_key in trigger_candidates:
            if not _trigger_event_alive(engine, trigger_key, event_seq):
                continue
            trigger_node, rule_name = trigger_key
            rule = engine.rules.get(rule_name)
            if not rule:
                continue
            # 每次汇聚用全新 eval cache：上一次汇聚的消费回收会改变告警的
            # consumed 状态，复用缓存会读到过期的节点验证结果。
            results = engine._evaluate_rule(
                rule_name,
                rule,
                trigger_node,
                event_ts,
            )
            if not results:
                continue
            raw_matches.extend(results)
            engine._prune_consumed_alarm_history(results)
        return raw_matches

    @staticmethod
    def _merge_batch_and_finalize(engine, raw_matches):
        """批内合并 + 历史合并落库 + 输出可见性过滤（与在线收割尾部一致）。"""
        if not raw_matches:
            return []
        merged_matches = merge_match_batch(raw_matches)
        finalized_matches = engine._finalize_matches_with_history(merged_matches)
        return engine._apply_output_visibility_filters_to_matches(finalized_matches)

    def _iter_output_matches(self, matches):
        """按输出口径过滤 match：可落盘规则过滤 + 故障模式过滤。

        与在线 match_rules.py 的落盘口径一致，通过过滤的匹配组才能作为
        原始故障组之间的汇聚证据。
        """
        for match in matches:
            if not self._match_is_output_eligible(match):
                continue
            if (
                self.fault_pattern_filter is not None
                and self.fault_pattern_filter.analyze_match(match) is None
            ):
                continue
            yield match

    def _match_is_output_eligible(self, match):
        """故障组是否满足可落盘规则要求。

        output_eligible_rules 为 None 时全部放行；否则要求 merged_rules（单个规则名
        列表，权威来源）与可落盘规则集合有交集。merged_rules 缺失时退回到 rule 字段。
        """
        eligible = self.output_eligible_rules
        if eligible is None:
            return True
        merged_rules = match.get("merged_rules")
        if isinstance(merged_rules, list):
            for rule_name in merged_rules:
                if str(rule_name).strip() in eligible:
                    return True
            return False
        rule = match.get("rule")
        return isinstance(rule, str) and rule.strip() in eligible


def _override_rule_time_config(rules_config, edge_window_sec, max_stay_sec):
    """深拷贝规则配置，替换两个时间参数：

    - 所有边的 time_window_sec -> edge_window_sec（对应 associate_time，
      即在线常量 RULE_DEFAULT_EDGE_TIME_WINDOW_SEC 的角色）；
    - max_stay_time_sec -> max_stay_sec（对应 max_group_time，
      即在线常量 RULE_DEFAULT_MAX_STAY_TIME_SEC 的角色）。
    """
    adjusted_rules = copy.deepcopy(rules_config)
    for rule in adjusted_rules.values():
        for edge in rule.get("edges", ()):
            edge["time_window_sec"] = edge_window_sec
        rule["max_stay_time_sec"] = max_stay_sec
    return adjusted_rules


def _collect_symptom_alarm_ids(match):
    """从 match 症状中按出现顺序提取去重后的告警编码 ID 列表。"""
    alarm_ids = []
    seen = set()
    for symptom in match.get("symptoms", ()):
        alarm_id = str(symptom.get("eid", "") or "").strip()
        if alarm_id and alarm_id not in seen:
            seen.add(alarm_id)
            alarm_ids.append(alarm_id)
    return alarm_ids


def _trigger_event_alive(engine, trigger_key, event_seq):
    """trigger 告警是否仍在索引中（未被此前汇聚消费删除、未被清除/过期）。"""
    trigger_events = engine.trigger_event_index.get(trigger_key)
    if not trigger_events:
        return False
    for trigger_event in trigger_events:
        if trigger_event[2] == event_seq:
            return True
    return False
