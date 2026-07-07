"""ANCHOR 二次汇聚批处理入口。

对外接口只有 BatchFaultGroupMatcher.aggregate_alarm_groups：输入已成组的
故障组集合 {故障组id: [告警, ...]}，输出
{汇聚故障组id: [{故障组id: [告警, ...]}, ...], ...}。
告警为 alarm_events.generator.generate_alarm() 生成的字典，
字段形如：

    {
        "alarmName": "Link Down",
        "firstOccurrence": 123.5,           # 数值时间戳或时间文本
        "vid": "eid-1::<uuid>",             # 告警唯一 ID
        "neVid": "ne-1",                    # 网元 vid，可空
        "ownerVid": "site-1",               # 归属资源 vid，neVid 为空时兜底
        "extendedattr": "portVid:port-1",   # 自由键值文本，取 portVid 条目
        "是否清除": True,                   # 可选，缺省视为上报告警
    }

告警源取 neVid，为空时退回 ownerVid；站点用告警源在静态拓扑的
ne_to_site（网元 ID -> 站点 ID）中反查得到，查不到时按无站点处理
（匹配不上拓扑规则，仅能靠组 ID 连续性等与拓扑无关的逻辑参与汇聚）。
物理端口名称从 extendedattr 的 portVid 条目解析。
生成器不再输出 是否清除（清除告警在流式入口按原始载荷过滤），
调用方自带该字段时仍按清除告警处理。

批处理不考虑实时聚合成熟时间：

1. 整批告警喂给引擎，只更新事件缓存并建立本次调用的临时 trigger 工作集，
   不建立延迟等待队列，过程中不做任何收割；持久会话按发生时间正序喂入，
   隔离会话无后续 TTL 状态，省略这次预排序；
2. 喂流结束后，只对本批输入产生的 trigger 逐一触发故障组汇聚（告警可以
   是历史重发），按告警 ts 从早到晚进行；
   不在本批输入中的历史 trigger 不重新评估，但历史事件缓存仍可作为症状证据
   参与本批匹配；每次汇聚成功立即执行既有的
   消费回收逻辑（_prune_consumed_alarm_history），把已被故障组消费的 trigger
   告警从本批工作集中删除，后续同类 trigger 因此不再重复汇聚；本次调用
   结束后丢弃剩余 trigger，不进入下一批；
3. 本批 trigger 处理完后统一做批内合并（merge_match_batch），再执行历史
   合并与输出可见性过滤；
4. 匹配出的故障组经可参与二次汇聚规则过滤、故障模式过滤后，作为原始故障组之间的
   关联证据：同一匹配组覆盖到的本批原始组连到一起，并通过症状告警的既有
   归属挂到历史汇聚组上；只输出本批发生变化（新汇聚组／新成员组／组内
   新告警）的汇聚组——key 是稳定的汇聚组 ID，value 为本批输入的原始组
   条目（组 ID 及其告警）；纯重发、无新增的汇聚组不输出。
"""

import heapq
import uuid

from types import SimpleNamespace

from anchor_grouping_online.alarm_events.generator import to_matching_alarm
from anchor_grouping_online.batch_context import (
    build_fault_pattern_filter,
    build_rules_config,
    collect_output_eligible_rules,
    load_static_context,
)
from anchor_grouping_online.temporal_engine.engine import TemporalGraphEngine
from anchor_grouping_online.temporal_engine.utils import merge_match_batch
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


def _matching_alarm(generated_alarm, cache, ne_to_site):
    """按对象身份缓存 to_matching_alarm 结果，避免同一告警在批内重复转换。

    to_matching_alarm 内含多次 str.strip 及可能的时间文本解析，一条告警在
    登记、喂流、输出与外置合并等阶段会被转换 2~4 次。缓存仅在单次
    aggregate_alarm_groups 调用内使用：入参 generated_alarm 全部由 alarm_groups
    / old_agg 在整个调用期间持有，id() 稳定不复用；按对象身份而非 alarm_id
    做键，保证同 eid 的上报/清除两条告警（is_clear 不同）各自独立转换。
    cache 为 None 时退回逐次转换（等价旧行为，供私有方法被单独调用时使用）。
    ne_to_site 为静态拓扑的网元->站点映射，站点由 告警源 反查得到。
    """
    if cache is None:
        return to_matching_alarm(generated_alarm, ne_to_site)
    key = id(generated_alarm)
    cached = cache.get(key)
    if cached is None:
        cached = to_matching_alarm(generated_alarm, ne_to_site)
        cache[key] = cached
    return cached


class BatchFaultGroupMatcher:
    """按批次做 ANCHOR 二次汇聚。

    静态拓扑上下文与规则配置在构造时加载一次；时序引擎与汇聚状态持久
    保留，后一批可以关联到前面批次喂入的告警。历史保留视界 = 本批最早
    告警时间 − max_group_time，每批
    开始时统一清理。需要重新开始时调用 reset()。
    """

    def __init__(
        self,
        resource_buffer=RESOURCE_BUFFER_JSONL,
        static_context=None,
        batch_isolated=False,
        full_output=False,
    ):
        if static_context is None:
            static_context = load_static_context(
                SimpleNamespace(resource_buffer=resource_buffer)
            )
        self.static_context = static_context
        self.rules_config = build_rules_config()
        self.output_eligible_rules = collect_output_eligible_rules(self.rules_config)
        self.fault_pattern_filter = build_fault_pattern_filter(static_context)
        # 引擎与汇聚状态持久保留，后一批可以关联到之前喂入的告警
        # 供后续批次继续关联。
        # batch_isolated=True 时每批处理前丢弃会话：只做本窗口（本批）
        # 内的告警汇聚。
        self.batch_isolated = bool(batch_isolated)
        # 两种模式都只返回本批发生变化的汇聚组，仅成员完整度不同：
        # full_output=False 输出本批增量成员；full_output=True 且传入
        # old_agg 时合并既有全量得到完整成员（无 old_agg 时无全量可回显、
        # 退回增量成员，不跨批持久保存历史全量）。
        self.full_output = bool(full_output)
        self._session = None
        # 跨引擎复用的静态结构（NE 映射、trigger 索引、
        # role_site_index），首个引擎构造时算出后共享给后续每批临时引擎，
        # 避免隔离/外置模式每批重建。只依赖 rules_config 的 node 结构与静态
        # 拓扑，构造后只读，故可安全跨引擎共享；由于所有引擎的规则都由
        # _override_rule_time_config 浅拷贝自同一 rules_config、共享同一批
        # node 对象，role_site_index 的 id(node_config) 命中一致。
        self._shared_static_context = None

    def reset(self):
        """清空会话状态（引擎、告警归属映射、汇聚组画像）。"""
        self._session = None

    def aggregate_alarm_groups(
        self,
        alarm_groups,
        associate_time,
        max_group_time,
        max_group_member,
        old_agg_alarm_groups=None,
    ):
        """ANCHOR 二次汇聚：按拓扑匹配关系把已成组的故障组合并为汇聚故障组。

        整体逻辑：拿全量告警按 ANCHOR 规则聚合故障组，再通过并查集把原始组
        汇聚到一起——alarm1 属于 group1、alarm2 属于 group2，若 alarm1 与
        alarm2 被聚合进同一个故障组，则 group1 与 group2 汇聚为一组；
        未形成关联且无法附着到既有汇聚组的单个原始 group 不输出。

        输入参数
            alarm_groups     Dict[str, List]  必选
                待二次汇聚的故障组集合。key 为故障组 ID，value 为该故障组包含
                的告警列表（告警为告警生成器输出的字典，见模块 docstring）。
                格式：{故障组id1: [告警1, ...], ...}
            associate_time   int              必选
                关联告警的时间窗，单位分钟。即匹配阶段的规则边时间窗。
            max_group_time   int              必选
                汇聚后故障组允许的最大时间窗，单位分钟。即匹配阶段的组保留
                时长。
            max_group_member int              必选
                汇聚故障组允许包含的最大**原始故障组个数**。附着会
                使组数超限时放弃附着；本批内部分量超限时按成员时间顺序
                切分。单个原始组恒为 1，永远不会被拆开。
            old_agg_alarm_groups Dict[str, List[Dict[str, List]]] 可选
                既有的二次汇聚故障组（格式与本方法输出一致，可为多次输出
                的累积合并）。提供时以它为跨窗口信息：据其重建上下文后再
                汇聚本批，汇聚逻辑与内部持久会话一致；本批可关联其中的
                告警/原始组/汇聚组 ID，新汇聚组 ID 为全新 UUID、不会与
                既有 ID 冲突。它引入的原始组/告警/汇聚组归属与计数完整保留，
                不因超过 max_group_time 而改变已落定归属；但 old_agg 告警
                只有位于「本批最早告警时间 - max_group_time」之后时才写入
                临时 event_cache，作为规则匹配的直接证据（边匹配时仍受
                associate_time 时间窗约束）。调用结束时会把完整外置历史
                的归属/计数与本批结果
                同步到内部会话（既有组/告警自动去重），但 old_agg 历史
                event_cache 仅在本次临时会话有效；持久引擎只喂本批告警。
                因此后续不传 old_agg 时计数和归属完整，但规则匹配不再引用
                old_agg 的历史事件。适合调用方自行保管状态的无状态服务部署。
                累积输出到 old_agg 时（输出只含变化的汇聚组，未出现的汇聚组
                应原样保留）：full_output=True 按 agg_id 整体替换该汇聚组
                （输出即完整成员）；full_output=False 须按原始组 ID 合并、
                告警编码 ID 去重。两种模式直接 extend 都会让同组条目重复
                累积，内部虽仍去重、结果正确，但 old_agg 计算与内存膨胀。
        输出参数
            agg_alarm_groups Dict[str, List[Dict[str, List]]]
                只返回本批发生变化（新汇聚组／新成员组／组内新告警）的
                汇聚组；纯重发、无新增的汇聚组不返回，整批无变化时返回
                空字典。key 为汇聚故障组 ID（UUID，稳定不变，同一场故障
                后续沿用同一个 ID）；value 是原始组条目列表，每个条目为
                {原始组id: [告警, ...]}，告警为本批输入的原始字典（组内
                按告警编码 ID 去重）。只输出累计至少包含 2 个原始故障组
                的汇聚组。未形成汇聚且未附着到既有汇聚组的单成员组不分配
                二次汇聚 ID，也不建立候选状态。
                例：窗口 1 输出 {"<uuid-1>": [{"g1": [告警1]},
                {"g2": [告警2]}]}；窗口 2 只输入 g3 且 g3 关联上同一场
                故障，则输出 {"<uuid-1>": [{"g3": [告警3]}]}。

        有状态语义
            引擎与汇聚状态持久保留：本批的故障组可以与之前喂入的告警
            建立关联。
            要点：
            - 同一告警编码 ID 在 event_cache 中只保留一份；当前批重发时
              幂等覆盖并重新获得 trigger 资格，不追加重复缓存；
            - 组 ID 连续性：同一原始组 ID 分批送来的告警稳定归属同一
              汇聚组（视界内），无需依赖告警重叠或拓扑匹配；
            - 历史保留视界统一为「本批最早告警时间 − max_group_time」，
              每批开始时把早于视界的历史状态一次性清理；
            - 已输出汇聚组不可变：原始组一旦归属某个汇聚组，
              不再改挂，既有汇聚组之间不合并；未汇聚的单成员组
              不建立二次汇聚候选状态；
            - max_group_member 按原始组个数计：附着会使汇聚组组数
              超限时放弃附着；已有归属的组仍沿用原汇聚组输出，未归属的
              直接关联组按上限切分，切分后的单成员不分配 ID；
            - associate_time / max_group_time 逐次可变：变化时原地更新
              会话（规则时间窗重编、TTL 更新），从本批起按新参数判定与
              过期，已积累的历史不重算；
            - 对输入顺序不做假设：批内、批间乱序均不影响聚合正确性——
              判定只依据告警自身的发生时间（持久缓存有序、编码 ID 去重、
              跨批时间回退单调安全）；过老的告警不会出错，只是超出
              关联窗/视界时自然关联不上。

        批隔离模式（构造时 batch_isolated=True）
            不保留内部会话：未传 old_agg 时只做本批（本窗口）内的告警
            汇聚；显式传入 old_agg 时仅在本次调用内把它作为历史上下文，
            调用结束即释放。新汇聚组 ID 每批全新生成，跨批比较无意义。

        全量输出模式（构造时 full_output=True）
            返回哪些汇聚组与默认口径一致（只返回本批发生变化的组），只把
            成员完整度从「本批增量」提升为「完整成员」：传入 old_agg 时，
            把该汇聚组在 old_agg 中的既有成员与本批增量合并输出完整成员
            （组内按告警编码 ID 去重）；未传 old_agg 时不跨批持久保存历史
            全量，退回本批增量成员（历史成员不回显，此时与默认口径等价）。
        """
        max_member_count = int(max_group_member)
        # 本次调用内共享的告警转换缓存：登记、外置重建、喂流、输出、外置
        # 合并各阶段复用同一份 to_matching_alarm 结果，避免重复转换。
        matching_cache = {}
        ne_to_site = self.static_context.ne_to_site
        # 外置历史在临时引擎里只保留 max_group_time 视界内的
        # 告警匹配证据。会话必须先于本批登记重建，因此这里先做一次
        # O(batch) 的时间下界预计算；转换结果写入 matching_cache，后续
        # 登记/喂流复用，不重复解析告警。归属注册表仍完整导入。
        external_history_horizon_ts = None
        if old_agg_alarm_groups is not None:
            external_batch_min_ts = None
            for group_alarms in alarm_groups.values():
                for generated_alarm in group_alarms or ():
                    ts = _matching_alarm(
                        generated_alarm, matching_cache, ne_to_site
                    )["ts"]
                    if external_batch_min_ts is None or ts < external_batch_min_ts:
                        external_batch_min_ts = ts
            if external_batch_min_ts is not None:
                external_history_horizon_ts = (
                    external_batch_min_ts - float(max_group_time) * 60.0
                )
        if self.batch_isolated:
            # 隔离模式所有状态仅在本次调用内存活，不在 matcher 上保留
            # 上一批会话，避免与本批 old_agg 临时会话叠加占用内存。
            self._session = None
        if old_agg_alarm_groups is not None:
            # 外置状态模式：以调用方提供的既有二次汇聚结果为跨窗口上下文
            # 构建临时会话。非隔离模式下先触达内部会话，参数有变化时原地
            # 更新（保证结束时增量落库与视界清理按本次参数执行）；隔离
            # 模式不维护内部会话，无需触达。
            if not self.batch_isolated:
                self._ensure_session(associate_time, max_group_time)
            session = self._rebuild_session_from_agg(
                old_agg_alarm_groups,
                associate_time,
                max_group_time,
                # 隔离模式不会在后续同步阶段再次处理
                # old_agg；历史转换结果写入 event_cache 后即可
                # 释放，避免将 O(|old_agg|) 的转换缓存保留到
                # 调用结束。非隔离模式后续还要同步 old_agg，
                # 继续复用调用级缓存。
                matching_cache=None if self.batch_isolated else matching_cache,
                history_horizon_ts=external_history_horizon_ts,
            )
        else:
            if self.batch_isolated:
                # 批隔离模式：本批自成一个不落到 self 的全新会话。
                session = self._new_session(associate_time, max_group_time)
            else:
                session = self._ensure_session(associate_time, max_group_time)
        engine = session["engine"]
        # trigger 只描述“本批哪些告警主动发起匹配”。持久会话只保留事件
        # 缓存和汇聚归属；上次调用的 trigger 工作集已在 finally 中清空。

        # 1. 登记本次输入。原始组的告警集/时间只作为本批的局部档案；
        #    会话里只记「告警 -> 汇聚组 ID」的归属。
        local_group_alarm_ids = {}
        local_group_alarms = {}
        local_group_min_ts = {}
        local_alarm_owners = {}
        current_generated_alarms = []
        new_generated_alarms = []
        overlapping_generated_alarms = []
        batch_alarm_ids = set()
        batch_min_ts = None
        use_batch_upsert = engine._batch_event_by_alarm_id is not None
        # 持久会话通过 eid upsert 索引判断实际缓存；隔离/外置临时会话
        # 要么全新为空，要么由 old_agg 完整重建，登记状态即等于缓存状态。
        # 因此所有模式都无需再全量扫描 event_cache 构造 ID set。
        for group_id, group_alarms in alarm_groups.items():
            alarm_ids = set()
            kept_alarms = []
            min_ts = None
            group_last_ts = None
            for generated_alarm in group_alarms or ():
                matching_alarm = _matching_alarm(
                    generated_alarm, matching_cache, ne_to_site
                )
                alarm_id = matching_alarm["alarm_id"]
                ts = matching_alarm["ts"]
                first_in_batch = alarm_id not in batch_alarm_ids
                batch_alarm_ids.add(alarm_id)
                if first_in_batch:
                    current_generated_alarms.append(generated_alarm)
                batch_min_ts = ts if batch_min_ts is None else min(batch_min_ts, ts)
                min_ts = ts if min_ts is None else min(min_ts, ts)
                if alarm_id not in alarm_ids:
                    kept_alarms.append(generated_alarm)
                alarm_ids.add(alarm_id)
                owner_group_ids = local_alarm_owners.setdefault(alarm_id, [])
                if group_id not in owner_group_ids:
                    owner_group_ids.append(group_id)
                alarm_entry = session["alarm_registry"].get(alarm_id)
                alarm_was_registered = alarm_entry is not None
                if alarm_entry is None:
                    alarm_entry = _register_alarm(session, alarm_id, ts, None)
                registered_ts = alarm_entry[0]
                if group_last_ts is None or registered_ts > group_last_ts:
                    group_last_ts = registered_ts
                if not use_batch_upsert:
                    if first_in_batch and (
                        not alarm_was_registered
                        or matching_alarm["is_clear"]
                    ):
                        new_generated_alarms.append(generated_alarm)
                    elif first_in_batch:
                        overlapping_generated_alarms.append(generated_alarm)
            local_group_alarm_ids[group_id] = alarm_ids
            local_group_alarms[group_id] = kept_alarms
            local_group_min_ts[group_id] = min_ts
            # 组注册表：刷新最近告警时间（归属在第 4 步分配时写入）。
            group_entry = session["group_registry"].get(group_id)
            if group_entry is None:
                _register_group(session, group_id, group_last_ts, None)
            else:
                _touch_group(session, group_id, group_last_ts)

        # 2. 批开始的历史回收：统一视界 = 本批最早告警时间 − max_group_time。
        #    这是唯一的 TTL 清理点：TTL 固定为 max_group_time，prune 原语以 batch_min_ts
        #    为基准算出的清理线恰为视界。
        #    外置状态模式跳过会话级回收：old_agg 归属永久保留，
        #    其告警匹配证据已在重建时按同一视界过滤。
        if (
            batch_min_ts is not None
            and old_agg_alarm_groups is None
            and not self.batch_isolated
        ):
            for node in list(engine.event_cache.keys()):
                engine._prune_expired_raw_events_in_place(node, batch_min_ts)
            _prune_session_history(session, batch_min_ts - session["max_stay_sec"])

        # 3. 增量喂流 + 窗口收割。匹配组做两件事：给本批原始组之间连边
        #    （局部并查集）；通过症状告警的既有归属，记录本批原始组与
        #    历史汇聚组的关联。
        local_union = _GroupUnionFind()
        for group_id in alarm_groups:
            local_union.add(group_id)
        linked_agg_ids_by_group = {group_id: [] for group_id in alarm_groups}
        # 本批组之间的直接关联边（每个匹配组一条超边）。附着失败拆分时
        # 用它重算子图：只有直接共存于同一匹配组的成员才保持连通。
        current_group_edges = []

        def link_group_to_agg(group_id, agg_id):
            # 急切改写保证指针直指存活汇聚组，这里只需做存在性防御。
            linked = linked_agg_ids_by_group[group_id]
            if (
                agg_id is not None
                and agg_id in session["agg_registry"]
                and agg_id not in linked
            ):
                linked.append(agg_id)

        try:
            if use_batch_upsert and current_generated_alarms:
                batch_trigger_candidates = self._feed_engine(
                    engine,
                    current_generated_alarms,
                    sort_events=True,
                    upsert_events=True,
                    matching_cache=matching_cache,
                    ne_to_site=ne_to_site,
                )
            elif new_generated_alarms or overlapping_generated_alarms:
                batch_trigger_candidates = self._feed_engine(
                    engine,
                    new_generated_alarms,
                    # 已在历史缓存中的本批告警不重复缓存，但与新告警完全
                    # 一样重新获得本批 trigger 资格。
                    trigger_only_alarms=overlapping_generated_alarms,
                    sort_events=not self.batch_isolated,
                    # 本批告警在登记阶段已全部转换并写入
                    # matching_cache。隔离模式无需排序，可直接
                    # 流式喂入，不再构造 O(batch) 中间列表。
                    preconverted_events=self.batch_isolated,
                    matching_cache=matching_cache,
                    ne_to_site=ne_to_site,
                )
            else:
                batch_trigger_candidates = []

            if current_generated_alarms:
                raw_matches = self._aggregate_per_trigger(
                    engine,
                    trigger_candidates=batch_trigger_candidates,
                    owns_trigger_candidates=True,
                    trigger_candidates_sorted=not self.batch_isolated,
                )
        finally:
            # trigger_event_index 仅服务本次收割期间的消费删除；剩余
            # 候选没有跨批意义，异常退出时也立即释放内存。
            engine.trigger_event_index.clear()

        if current_generated_alarms:
            output_matches = self._merge_batch_and_finalize(engine, raw_matches)
            for match in self._iter_output_matches(output_matches):
                related_group_ids = []
                related_agg_ids = []
                for alarm_id in _collect_symptom_alarm_ids(match):
                    for group_id in local_alarm_owners.get(alarm_id, ()):
                        if group_id not in related_group_ids:
                            related_group_ids.append(group_id)
                    alarm_entry = session["alarm_registry"].get(alarm_id)
                    if alarm_entry is not None and alarm_entry[1] is not None:
                        related_agg_ids.append(alarm_entry[1])
                local_union.union_all(related_group_ids)
                if len(related_group_ids) >= 2:
                    current_group_edges.append(list(related_group_ids))
                for group_id in related_group_ids:
                    for agg_id in related_agg_ids:
                        link_group_to_agg(group_id, agg_id)

        # 3b. 原始组自身的连续性：组 ID 出现过就沿用其既有汇聚组归属
        #     （同一原始组分批送来的新告警落在同一汇聚组下）；组内告警
        #     若已归属某汇聚组也沿用（覆盖重叠窗口重发、跨组共享告警）。
        for group_id, alarm_ids in local_group_alarm_ids.items():
            link_group_to_agg(group_id, session["group_registry"][group_id][1])
            for alarm_id in alarm_ids:
                alarm_entry = session["alarm_registry"].get(alarm_id)
                if alarm_entry is not None:
                    link_group_to_agg(group_id, alarm_entry[1])

        # 4. 按本批连通分量分配稳定的汇聚组 ID（上限按原始组个数计）：
        #    - 已有归属的组沿用原汇聚组，既有汇聚组之间不合并；
        #    - 未归属组关联到多个历史组时，选一个容量足够的目标；
        #    - 附着会使组数超限：放弃附着，已有归属的组沿用原
        #      汇聚组，未归属的直接关联组按上限切分；
        #    - 无关联：按上限切分，至少两个成员的包才分配新 ID。
        def member_order_key(group_id):
            min_ts = local_group_min_ts.get(group_id)
            return (min_ts is None, min_ts if min_ts is not None else 0.0, str(group_id))

        batch_member_entries_by_agg = {}
        # full_output 口径下记录本批真正发生变化的汇聚组：新增了原始组
        # 成员（new_group_count>0），或已有成员组内新增了告警
        # （new_alarm_ids 非空）。新建汇聚组必然带来新成员，天然纳入。
        changed_agg_ids = set()

        def assign_members_to_agg(agg_id, member_group_ids):
            """成员组划入汇聚组：登记归属/计数/告警映射与本批条目。"""
            new_alarm_ids = []
            new_group_count = 0
            for group_id in member_group_ids:
                group_entry = session["group_registry"][group_id]
                if group_entry[1] is not None and group_entry[1] != agg_id:
                    raise ValueError(
                        f"原始故障组 {group_id!r} 已归属汇聚组 "
                        f"{group_entry[1]!r}，不允许改挂到 {agg_id!r}"
                    )
                if group_entry[1] is None:
                    new_group_count += 1
                _set_group_owner(session, group_id, agg_id)
                new_alarm_ids.extend(
                    alarm_id for alarm_id in local_group_alarm_ids[group_id]
                    if session["alarm_registry"][alarm_id][1] is None
                )
            session["agg_registry"][agg_id][0] += new_group_count
            _attach_alarms_to_agg(session, agg_id, new_alarm_ids)
            if new_group_count or new_alarm_ids:
                changed_agg_ids.add(agg_id)
            batch_member_entries_by_agg.setdefault(agg_id, []).extend(
                {group_id: local_group_alarms[group_id]}
                for group_id in member_group_ids
            )

        components = local_union.components(alarm_groups)
        for members in components:
            members.sort(key=member_order_key)
        components.sort(key=lambda members: member_order_key(members[0]))

        for members in components:
            linked_agg_ids = []
            for group_id in members:
                for agg_id in linked_agg_ids_by_group[group_id]:
                    if (
                        agg_id in session["agg_registry"]
                        and agg_id not in linked_agg_ids
                    ):
                        linked_agg_ids.append(agg_id)

            # 已有归属一旦落定就不改挂；其余成员仅在能附着到
            # 既有汇聚组或与至少一个其他成员形成新汇聚时才分配 ID。
            unassigned_members = []
            for group_id in members:
                existing_agg_id = session["group_registry"][group_id][1]
                if existing_agg_id in session["agg_registry"]:
                    assign_members_to_agg(existing_agg_id, [group_id])
                else:
                    _set_group_owner(session, group_id, None)
                    unassigned_members.append(group_id)

            if not unassigned_members:
                continue

            # 新成员同时关联多个历史汇聚组时，只选一个能整体
            # 容纳本子分量的目标，既有汇聚组之间不合并。
            eligible_targets = [
                agg_id for agg_id in linked_agg_ids
                if session["agg_registry"][agg_id][0] + len(unassigned_members)
                <= max_member_count
            ]
            if eligible_targets:
                eligible_targets.sort(
                    key=lambda a: (-session["agg_registry"][a][2], str(a))
                )
                assign_members_to_agg(eligible_targets[0], unassigned_members)
                continue

            # 无历史目标可容纳：只保留未归属成员彼此间的直接
            # 匹配边重算连通子图。仅通过已落定成员传递连通的新组
            # 在移除该中介后会自然拆开。
            unassigned_set = set(unassigned_members)
            sub_union = _GroupUnionFind()
            for group_id in unassigned_members:
                sub_union.add(group_id)
            for edge_group_ids in current_group_edges:
                sub_union.union_all(
                    [g for g in edge_group_ids if g in unassigned_set]
                )
            sub_components = sub_union.components(unassigned_members)
            for sub_members in sub_components:
                sub_members.sort(key=member_order_key)
            sub_components.sort(
                key=lambda sub_members: member_order_key(sub_members[0])
            )
            for sub_members in sub_components:
                for pack_members in _split_members_by_cap(
                    sub_members, max_member_count
                ):
                    if len(pack_members) < 2:
                        continue
                    assign_members_to_agg(
                        _new_agg_group(session), pack_members
                    )

        # 本批各汇聚组的增量条目（count>=2）。仅两处消费：外置同步（需全部
        # 增量，含纯重发条目）与对外输出（仅变化汇聚组）。因此不做外置同步
        # 时只构建变化组，跳过纯重发、未变化汇聚组的成员合并。
        need_full_increment = (
            old_agg_alarm_groups is not None and not self.batch_isolated
        )
        batch_increment_by_agg = {}
        for agg_id, member_entries in batch_member_entries_by_agg.items():
            if not need_full_increment and agg_id not in changed_agg_ids:
                continue
            if session["agg_registry"][agg_id][0] < 2:
                continue
            groups_by_id = {}
            _merge_member_entries(
                groups_by_id, member_entries, matching_cache, ne_to_site
            )
            batch_increment_by_agg[agg_id] = [
                {group_id: list(alarms_by_id.values())}
                for group_id, alarms_by_id in groups_by_id.items()
            ]

        if old_agg_alarm_groups is not None and not self.batch_isolated:
            # 把完整外置历史的归属/计数、本批增量和本批原始告警并入
            # 内部会话。old_agg 的历史事件缓存只属于本次临时会话，持久
            # 引擎仅接收本批告警；未汇聚组不创建二次汇聚 ID。
            # 隔离模式不维护内部会话，临时会话用完即弃。
            state_to_import = {
                agg_id: list(member_entries or ())
                for agg_id, member_entries in old_agg_alarm_groups.items()
            }
            for agg_id, member_entries in batch_increment_by_agg.items():
                state_to_import.setdefault(agg_id, []).extend(member_entries)
            self._merge_increment_into_session(
                state_to_import,
                associate_time,
                max_group_time,
                batch_min_ts,
                raw_alarm_groups=alarm_groups,
                matching_cache=matching_cache,
                imported_agg_counts={
                    agg_id: (
                        session["agg_registry"][agg_id][0],
                        session["agg_registry"][agg_id][2],
                    )
                    for agg_id in state_to_import
                },
            )

        if self.batch_isolated:
            # 隔离模式不跨批复用会话：本批算完立即释放，避免连同引擎
            # （事件缓存/trigger 索引等）在下次调用前一直驻留内存。
            self._session = None

        # 对外只暴露本批发生变化的汇聚组（新汇聚组／新成员组／组内新告警）；
        # 纯重发、无新增的汇聚组不输出，整批无变化时返回 {}。成员完整度由
        # full_output 决定：默认输出本批增量成员；full_output=True 且传入
        # old_agg 时合并既有全量得到完整成员，无 old_agg 时无全量可回显、
        # 退回增量成员。
        agg_alarm_groups = {}
        for agg_id in changed_agg_ids:
            batch_entries = batch_increment_by_agg.get(agg_id)
            if batch_entries is None:
                # 变化组必然收到过本批条目且 count>=2，此处仅为边界防御。
                continue
            if not self.full_output or old_agg_alarm_groups is None:
                agg_alarm_groups[agg_id] = batch_entries
                continue
            groups_by_id = {}
            _merge_member_entries(
                groups_by_id,
                old_agg_alarm_groups.get(agg_id) or (),
                matching_cache,
                ne_to_site,
            )
            _merge_member_entries(
                groups_by_id, batch_entries, matching_cache, ne_to_site
            )
            agg_alarm_groups[agg_id] = [
                {group_id: list(alarms_by_id.values())}
                for group_id, alarms_by_id in groups_by_id.items()
            ]

        return agg_alarm_groups

    def _merge_increment_into_session(
        self,
        agg_output,
        associate_time,
        max_group_time,
        batch_min_ts,
        raw_alarm_groups=None,
        imported_agg_counts=None,
        matching_cache=None,
    ):
        """把外置归属快照、本批增量及本批原始告警并入内部会话。

        按增量合并：汇聚组条目缺失才创建；新原始组才计数；
        本批告警统一幂等写入内部引擎。既有原始组的归属发生变化
        视为外置状态冲突，直接报错。
        raw_alarm_groups 只补齐原始组/告警缓存；未汇聚成员的归属保持
        None，不创建单成员二次汇聚组。
        外置历史告警只登记归属，不写入持久引擎；持久 event_cache 和
        trigger 只接收本批原始告警。
        合并后对内部会话执行与内部调用相同的视界清理——纯外置使用形态
        （每批都传 old_agg）下这是内部状态唯一的过期出口，否则只进不出、
        无界增长。
        """
        session = self._ensure_session(associate_time, max_group_time)
        ne_to_site = self.static_context.ne_to_site
        current_alarm_records_by_group = {}
        current_matching_alarms_by_id = {}
        current_generated_alarms_by_id = {}
        for group_id, group_alarms in (raw_alarm_groups or {}).items():
            records = []
            for generated_alarm in group_alarms or ():
                matching_alarm = _matching_alarm(
                    generated_alarm, matching_cache, ne_to_site
                )
                records.append((generated_alarm, matching_alarm))
                current_matching_alarms_by_id.setdefault(
                    matching_alarm["alarm_id"], matching_alarm
                )
                current_generated_alarms_by_id.setdefault(
                    matching_alarm["alarm_id"], generated_alarm
                )
            current_alarm_records_by_group[group_id] = records

        def resolve_matching_alarm(generated_alarm):
            if isinstance(generated_alarm, dict):
                raw_alarm_id = generated_alarm.get("vid")
                if isinstance(raw_alarm_id, str):
                    current_alarm = current_matching_alarms_by_id.get(
                        raw_alarm_id.strip()
                    )
                    if current_alarm is not None:
                        return current_alarm
            return _matching_alarm(generated_alarm, matching_cache, ne_to_site)

        # 先做全量归属校验，再修改会话，避免中途发现冲突时
        # 留下一部分已写入、一部分未写入的状态。agg_output 是完整外置
        # 快照。正常调用直接复用临时会话重建时已经得到的唯一成员计数；
        # 私有方法被单独调用时才在本次必经遍历中构造去重集合兜底。
        imported_group_ids = (
            {agg_id: set() for agg_id in agg_output}
            if imported_agg_counts is None else None
        )
        imported_alarm_ids = (
            {agg_id: set() for agg_id in agg_output}
            if imported_agg_counts is None else None
        )
        for agg_id, member_entries in agg_output.items():
            for member_entry in member_entries:
                for group_id, group_alarms in member_entry.items():
                    if imported_group_ids is not None:
                        imported_group_ids[agg_id].add(group_id)
                    group_entry = session["group_registry"].get(group_id)
                    if (
                        group_entry is not None
                        and group_entry[1] is not None
                        and group_entry[1] != agg_id
                    ):
                        raise ValueError(
                            f"外置增量冲突：原始故障组 {group_id!r} 在内部"
                            f"会话中已归属 {group_entry[1]!r}，新增量却归属"
                            f" {agg_id!r}"
                        )
                    for generated_alarm in group_alarms:
                        matching_alarm = resolve_matching_alarm(generated_alarm)
                        if imported_alarm_ids is not None:
                            imported_alarm_ids[agg_id].add(
                                matching_alarm["alarm_id"]
                            )
                        alarm_entry = session["alarm_registry"].get(
                            matching_alarm["alarm_id"]
                        )
                        if (
                            alarm_entry is not None
                            and alarm_entry[1] is not None
                            and alarm_entry[1] != agg_id
                        ):
                            raise ValueError(
                                f"外置增量冲突：告警 "
                                f"{matching_alarm['alarm_id']!r} 在内部会话中"
                                f"已归属 {alarm_entry[1]!r}，新增量却归属"
                                f" {agg_id!r}"
                            )

        for agg_id, member_entries in agg_output.items():
            agg_entry = _ensure_agg(session, agg_id)
            imported_agg_last_ts = agg_entry[1]
            for member_entry in member_entries:
                for group_id, group_alarms in member_entry.items():
                    group_entry = session["group_registry"].get(group_id)
                    if group_entry is None:
                        group_entry = _register_group(
                            session, group_id, None, None
                        )
                    if group_entry[1] != agg_id:
                        agg_entry[0] += 1
                        _set_group_owner(session, group_id, agg_id)
                    imported_group_last_ts = group_entry[0]
                    for generated_alarm in group_alarms:
                        matching_alarm = resolve_matching_alarm(generated_alarm)
                        alarm_id = matching_alarm["alarm_id"]
                        ts = matching_alarm["ts"]
                        alarm_entry = session["alarm_registry"].get(alarm_id)
                        if alarm_entry is None:
                            _register_alarm(session, alarm_id, ts, agg_id)
                            agg_entry[2] += 1
                        elif alarm_entry[1] is None:
                            # 告警可能在此前外置调用中作为未汇聚原始告警
                            # 写入；本次真正汇聚时补齐归属，但不重复喂引擎。
                            alarm_entry[1] = agg_id
                            agg_entry[2] += 1
                        if (
                            imported_agg_last_ts is None
                            or ts > imported_agg_last_ts
                        ):
                            imported_agg_last_ts = ts
                        if (
                            imported_group_last_ts is None
                            or ts > imported_group_last_ts
                        ):
                            imported_group_last_ts = ts
                    _touch_group(session, group_id, imported_group_last_ts)
            if imported_agg_last_ts is not None:
                _touch_agg(session, agg_id, imported_agg_last_ts)

        # 本批未进入可见汇聚输出的原始告警也要保留，确保外置调用后
        # 切换到内部模式时，时序引擎拥有与连续内部模式一致的事件历史。
        for group_id, alarm_records in current_alarm_records_by_group.items():
            group_entry = session["group_registry"].get(group_id)
            if group_entry is None:
                group_entry = _register_group(session, group_id, None, None)
            current_group_last_ts = group_entry[0]
            for generated_alarm, matching_alarm in alarm_records:
                alarm_id = matching_alarm["alarm_id"]
                ts = matching_alarm["ts"]
                if alarm_id not in session["alarm_registry"]:
                    _register_alarm(session, alarm_id, ts, None)
                if current_group_last_ts is None or ts > current_group_last_ts:
                    current_group_last_ts = ts
            _touch_group(session, group_id, current_group_last_ts)

        if imported_agg_counts is None:
            imported_agg_counts = {
                agg_id: (
                    len(imported_group_ids[agg_id]),
                    len(imported_alarm_ids[agg_id]),
                )
                for agg_id in agg_output
            }

        # 直接覆盖临时会话已经按唯一 ID 计算好的计数，保证重复导入幂等，
        # 也避免 O(|group_registry| + |alarm_registry|) 的内部全表扫描。
        for agg_id in agg_output:
            agg_entry = session["agg_registry"].get(agg_id)
            if agg_entry is not None:
                agg_entry[0], agg_entry[2] = imported_agg_counts[agg_id]

        if current_generated_alarms_by_id:
            self._feed_engine(
                session["engine"],
                list(current_generated_alarms_by_id.values()),
                upsert_events=True,
                index_trigger=False,
                matching_cache=matching_cache,
                ne_to_site=ne_to_site,
            )
        # 视界清理：本批告警均不早于 batch_min_ts，只会清掉更早的历史。
        if batch_min_ts is not None:
            engine = session["engine"]
            for node in list(engine.event_cache.keys()):
                engine._prune_expired_raw_events_in_place(node, batch_min_ts)
            _prune_session_history(
                session, batch_min_ts - session["max_stay_sec"]
            )

    def _rebuild_session_from_agg(
        self, old_agg_alarm_groups, associate_time, max_group_time,
        matching_cache=None, history_horizon_ts=None,
    ):
        """用外部提供的既有二次汇聚结果构建临时会话（外置状态模式）。

        注册表按条目重建（告警/组的归属、汇聚组的组数/告警数/最晚时间），
        历史有效发生告警只有在 history_horizon_ts 之后（边界包含）
        才写入全新引擎的 event_cache，作为本批的症状关联证据，
        不建立历史 trigger。告警/原始组/汇聚组归属及完整计数不过期；
        新汇聚组 ID 为全新 UUID，不会与既有 ID 冲突。

        既有汇聚组不可变：同一原始组或同一告警若出现在多个
        汇聚组下，说明外置历史已冲突，直接报错，不静默合并。
        临时会话不写回 self._session：外置调用不以整份会话落库。
        """
        session = self._new_session(associate_time, max_group_time)
        ne_to_site = self.static_context.ne_to_site
        for raw_agg_id, member_entries in old_agg_alarm_groups.items():
            agg_id = raw_agg_id
            agg_entry = _ensure_agg(session, agg_id)
            agg_last_ts = None
            for member_entry in member_entries or ():
                for group_id, group_alarms in member_entry.items():
                    group_entry = session["group_registry"].get(group_id)
                    if group_entry is None:
                        group_entry = _register_group(
                            session, group_id, None, agg_id
                        )
                        agg_entry[0] += 1
                    elif group_entry[1] != agg_id:
                        raise ValueError(
                            f"外置历史冲突：原始故障组 {group_id!r} 同时归属"
                            f" {group_entry[1]!r} 和 {agg_id!r}"
                        )
                    group_last_ts = None
                    for generated_alarm in group_alarms or ():
                        matching_alarm = _matching_alarm(
                            generated_alarm, matching_cache, ne_to_site
                        )
                        if matching_alarm["is_clear"]:
                            raise ValueError(
                                "old_agg 只允许包含有效发生告警，不允许清除告警"
                            )
                        alarm_id = matching_alarm["alarm_id"]
                        ts = matching_alarm["ts"]
                        alarm_entry = session["alarm_registry"].get(alarm_id)
                        if alarm_entry is None:
                            _register_alarm(session, alarm_id, ts, agg_id)
                            if (
                                history_horizon_ts is None
                                or ts >= history_horizon_ts
                            ):
                                self._process_matching_alarm(
                                    session["engine"],
                                    matching_alarm,
                                    index_trigger=False,
                                )
                            agg_entry[2] += 1
                        elif alarm_entry[1] != agg_id:
                            raise ValueError(
                                f"外置历史冲突：告警 {alarm_id!r} 同时归属"
                                f" {alarm_entry[1]!r} 和 {agg_id!r}"
                            )
                        if agg_last_ts is None or ts > agg_last_ts:
                            agg_last_ts = ts
                        if group_last_ts is None or ts > group_last_ts:
                            group_last_ts = ts
                    if group_last_ts is not None:
                        _touch_group(session, group_id, group_last_ts)
            if agg_last_ts is not None:
                _touch_agg(session, agg_id, agg_last_ts)
        return session

    def _ensure_session(self, associate_time, max_group_time):
        """惰性创建并持有内部会话；时间参数变化时原地更新会话。"""
        associate_window_sec = float(associate_time) * 60.0
        max_stay_sec = float(max_group_time) * 60.0
        session = self._session
        if session is not None:
            if (
                session["associate_window_sec"] != associate_window_sec
                or session["max_stay_sec"] != max_stay_sec
            ):
                _update_session_time_params(
                    session, associate_window_sec, max_stay_sec
                )
            return session
        session = self._new_session(
            associate_time,
            max_group_time,
            enable_batch_upsert_indexes=True,
        )
        self._session = session
        return session

    def _new_session(
        self,
        associate_time,
        max_group_time,
        enable_batch_upsert_indexes=False,
    ):
        """构建全新会话（引擎 + 空注册表），不改动 self._session。"""
        associate_window_sec = float(associate_time) * 60.0
        max_stay_sec = float(max_group_time) * 60.0
        engine = self._new_engine(
            _override_rule_time_config(
                self.rules_config, associate_window_sec, max_stay_sec
            ),
            event_ttl=max_stay_sec,
            enable_batch_upsert_indexes=enable_batch_upsert_indexes,
        )
        session = {
            "engine": engine,
            "associate_window_sec": associate_window_sec,
            "max_stay_sec": max_stay_sec,
            # 会话状态按视界回收。已输出的汇聚组在视界内不改挂；
            # 未汇聚的单成员组不建立二次汇聚候选状态。
            "alarm_registry": {},  # 告警编码ID -> [ts, 汇聚组ID或None]
                                   #   归属校验 + 历史关联的载体
            "group_registry": {},  # 原始组ID -> [最近告警ts或None, 汇聚组ID]
                                   #   组 ID 连续性（稳定归属）
            "agg_registry": {},    # 汇聚组ID -> [原始组个数, 最晚告警ts,
                                   #   告警数]：上限检查 / 视界回收 / 附着选主
            # 仅持久会话维护。临时/隔离会话不做 TTL，避免额外堆内存。
            "history_expiry_indexes": (
                _new_history_expiry_indexes()
                if enable_batch_upsert_indexes else None
            ),
        }
        return session

    @staticmethod
    def _feed_engine(
        engine,
        generated_alarms,
        trigger_only_alarms=None,
        sort_events=True,
        upsert_events=False,
        index_trigger=True,
        preconverted_events=False,
        matching_cache=None,
        ne_to_site=None,
    ):
        """整批当前告警喂流；持久会话可按 eid 幂等覆盖。

        持久会话排序以保证后续 TTL 能从队头正确清理；隔离会话用完即弃，
        可跳过预排序，最终 trigger 收割仍会独立按时间排序。
        喂流阶段只累积事件缓存与 trigger 索引，
        不触发收割。
        """
        event_batches = (
            (generated_alarms, True),
            (trigger_only_alarms or (), False),
        )
        if not sort_events and preconverted_events:
            if matching_cache is None:
                raise ValueError("preconverted_events 需要 matching_cache")
            # 调用方已在修改引擎前完成全批转换；此处只按
            # 原有顺序惰性读取，不改变异常原子性。
            matching_events = (
                (matching_cache[id(alarm)], cache_event)
                for alarms, cache_event in event_batches
                for alarm in alarms
            )
        else:
            matching_events = [
                (_matching_alarm(alarm, matching_cache, ne_to_site), cache_event)
                for alarms, cache_event in event_batches
                for alarm in alarms
            ]
            if sort_events:
                matching_events.sort(key=lambda item: item[0]["ts"])
        trigger_candidates = []
        for alarm, cache_event in matching_events:
            if upsert_events:
                alarm_payload = None
                if alarm["physical_port_name"]:
                    alarm_payload = {
                        "物理端口名称": alarm["physical_port_name"]
                    }
                trigger_candidates.extend(engine.process_batch_event(
                    node=alarm["site_id"],
                    alarm_source=alarm["alarm_source"],
                    alarm_type=alarm["alarm_title"],
                    ts=alarm["ts"],
                    alarm_id=alarm["alarm_id"],
                    alarm_payload=alarm_payload,
                    is_clear=alarm["is_clear"],
                    index_trigger=index_trigger,
                ))
            else:
                BatchFaultGroupMatcher._process_matching_alarm(
                    engine,
                    alarm,
                    index_trigger=index_trigger,
                    cache_event=cache_event,
                    trigger_candidates=trigger_candidates,
                )
        return trigger_candidates

    @staticmethod
    def _process_matching_alarm(
        engine,
        alarm,
        index_trigger=True,
        cache_event=True,
        trigger_candidates=None,
    ):
        """把已转换的告警写入批处理引擎（只维护缓存与 trigger 索引）。"""
        alarm_payload = None
        if alarm["physical_port_name"]:
            alarm_payload = {"物理端口名称": alarm["physical_port_name"]}
        engine.process_event(
            node=alarm["site_id"],
            alarm_source=alarm["alarm_source"],
            alarm_type=alarm["alarm_title"],
            ts=alarm["ts"],
            alarm_id=alarm["alarm_id"],
            alarm_payload=alarm_payload,
            is_clear=alarm["is_clear"],
            index_trigger=index_trigger,
            cache_event=cache_event,
            trigger_candidates=trigger_candidates,
        )

    def _new_engine(
        self,
        rules_config,
        event_ttl,
        enable_batch_upsert_indexes=False,
    ):
        """新建批处理引擎：喂流只维护事件缓存与 trigger 索引，不触发规则
        评估、也不做 TTL 清理（历史回收统一在每批开始的视界清扫完成）。"""
        static_context = self.static_context
        shared_static_context = self._shared_static_context
        engine = TemporalGraphEngine(
            rules_config,
            static_context.site_domain_map,
            alarm_source_domain_map=static_context.alarm_source_domain_map,
            site_chain_index=static_context.site_chain_index,
            ne_graph_data=static_context.ne_graph_data,
            site_to_ne_ids=static_context.site_to_ne_ids,
            link_peer_index=static_context.link_peer_index,
            event_ttl=event_ttl,
            enable_batch_upsert_indexes=enable_batch_upsert_indexes,
            shared_static_context=shared_static_context,
        )
        # 首个引擎完成全量静态构建后，把可共享结构缓存到 matcher，供后续
        # 每批临时引擎复用（跳过 NE 邻接 / trigger 索引 / role_site_index 重建）。
        if shared_static_context is None:
            self._shared_static_context = engine.export_static_context()
        return engine

    @staticmethod
    def _aggregate_per_trigger(
        engine,
        trigger_candidates,
        owns_trigger_candidates=False,
        trigger_candidates_sorted=False,
    ):
        """按 trigger 告警 ts 从早到晚逐个触发故障组汇聚，返回原始 match 列表。

        按首触发时间升序处理，因此消费回收的 cutoff 只会清理不晚于本组
        症状时间的历史，不影响更晚的独立故障。
        每次汇聚成功后立即调用引擎既有的 _prune_consumed_alarm_history：
        被故障组消费的 trigger 告警会从 trigger_event_index 中删除；
        后续 trigger 若已被消费，会被直接跳过，不再重复汇聚。

        trigger_candidates 是喂流阶段直接返回的本批候选三元组
        (ts, seq, trigger_key)，无需再扫描 trigger_event_index 筛本批序号。
        不在本批输入中的历史事件不会发起 trigger；本批 trigger 评估时仍可
        引用 event_cache 中的历史告警作为症状证据。
        """
        # aggregate_alarm_groups 传入的是 _feed_engine 刚创建的
        # 独占列表，可原地排序；其他直接调用默认仍复制，
        # 不修改调用方容器。
        if not owns_trigger_candidates or not isinstance(trigger_candidates, list):
            trigger_candidates = list(trigger_candidates)
        # 非隔离模式喂流前已按 ts 排序，同时 trigger seq
        # 单调递增，候选已按 (ts, seq) 有序；隔离模式仍
        # 在此排序，保证无序输入的收割语义。
        if not trigger_candidates_sorted:
            trigger_candidates.sort()
        consumed_trigger_seqs = set()

        raw_matches = []
        for event_ts, event_seq, trigger_key in trigger_candidates:
            if event_seq in consumed_trigger_seqs:
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
            removed_trigger_seqs = engine._prune_consumed_alarm_history(results)
            consumed_trigger_seqs.update(removed_trigger_seqs)
        return raw_matches

    @staticmethod
    def _merge_batch_and_finalize(engine, raw_matches):
        """批内合并 + 历史合并 + 输出可见性过滤。"""
        if not raw_matches:
            return []
        merged_matches = merge_match_batch(raw_matches)
        finalized_matches = engine._finalize_matches_with_history(merged_matches)
        return engine._apply_output_visibility_filters_to_matches(finalized_matches)

    def _iter_output_matches(self, matches):
        """按二次汇聚口径过滤 match：规则过滤 + 故障模式过滤。

        通过过滤的匹配组才能作为原始故障组之间的汇聚证据。
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
        """故障组是否满足可参与二次汇聚的规则要求。

        output_eligible_rules 为 None 时全部放行；否则要求 merged_rules（单个规则名
        列表，权威来源）与可参与二次汇聚规则集合有交集。
        """
        eligible = self.output_eligible_rules
        if eligible is None:
            return True
        return any(
            str(rule_name).strip() in eligible
            for rule_name in match["merged_rules"]
        )


def _override_rule_time_config(rules_config, edge_window_sec, max_stay_sec):
    """浅拷贝规则配置，替换两个时间参数：

    - 所有边的 time_window_sec -> edge_window_sec（对应 associate_time）；
    - max_stay_time_sec -> max_stay_sec（对应 max_group_time）。

    只有被改写的部分（rule 顶层字典、edges 列表及各 edge 字典）新建副本；
    nodes / patterns / expected_alarms 等庞大且不被改写的子结构按引用共享。
    因此各批引擎的 node_config 保持同一批对象身份，使 role_site_index 等
    静态结构可跨引擎共享（见 _shared_static_context）；同时避免每批对整份
    规则做深拷贝。engine 侧只从 edge 字典读 time_window_sec、从 rule 字典读
    max_stay_time_sec，不改写 node，故按引用共享安全。
    """
    adjusted_rules = {}
    for rule_name, rule in rules_config.items():
        new_rule = dict(rule)
        new_rule["edges"] = [
            {**edge, "time_window_sec": edge_window_sec}
            for edge in rule.get("edges", ())
        ]
        new_rule["max_stay_time_sec"] = max_stay_sec
        adjusted_rules[rule_name] = new_rule
    return adjusted_rules


class _GroupUnionFind:
    """原始故障组的并查集（带路径压缩）。"""

    def __init__(self):
        self.parent = {}

    def add(self, group_id):
        self.parent.setdefault(group_id, group_id)

    def find(self, group_id):
        parent = self.parent
        root = group_id
        while parent[root] != root:
            root = parent[root]
        while parent[group_id] != root:
            parent[group_id], group_id = root, parent[group_id]
        return root

    def union_all(self, group_ids):
        """把一批组无条件并到一起（不足两个时不动作）。"""
        if len(group_ids) < 2:
            return
        for group_id in group_ids:
            self.add(group_id)
        base_root = self.find(group_ids[0])
        for other_group_id in group_ids[1:]:
            other_root = self.find(other_group_id)
            if other_root != base_root:
                self.parent[other_root] = base_root

    def components(self, seed_group_ids):
        """返回触及 seed_group_ids 的连通分量成员列表（每分量一个列表）。

        分量成员可能包含 seed 之外的组（有状态模式下被关联进来的历史组）。
        """
        seed_roots = {self.find(group_id) for group_id in seed_group_ids}
        members_by_root = {}
        for group_id in self.parent:
            root = self.find(group_id)
            if root in seed_roots:
                members_by_root.setdefault(root, []).append(group_id)
        return list(members_by_root.values())


def _update_session_time_params(session, associate_window_sec, max_stay_sec):
    """原地变更会话的时间参数，从当前批次起生效。

    关联窗写进规则边并重编执行计划（编译时会把 time_window_sec 拷入
    计划，仅改规则字典不生效）；组保留时长更新规则与引擎 TTL。已积累
    的历史（已成组、已消费 trigger、已清理的状态）不重算。
    """
    engine = session["engine"]
    for rule in engine.rules.values():
        for edge in rule.get("edges", ()):
            edge["time_window_sec"] = associate_window_sec
        rule["max_stay_time_sec"] = max_stay_sec
    engine._compile_rule_execution_plans()
    engine.global_ttl = max_stay_sec
    session["associate_window_sec"] = associate_window_sec
    session["max_stay_sec"] = max_stay_sec


def _merge_member_entries(
    groups_by_id, member_entries, matching_cache=None, ne_to_site=None
):
    """把增量原始组条目并入待输出映射，组内告警按 ID 去重。"""
    for member_entry in member_entries:
        for group_id, group_alarms in member_entry.items():
            alarms_by_id = groups_by_id.setdefault(group_id, {})
            for generated_alarm in group_alarms:
                alarm_id = _matching_alarm(
                    generated_alarm, matching_cache, ne_to_site
                )["alarm_id"]
                alarms_by_id.setdefault(alarm_id, generated_alarm)


def _prune_session_history(session, horizon_ts):
    """按三个最小堆增量回收视界之前的会话状态。"""
    indexes = session["history_expiry_indexes"]

    alarm_registry = session["alarm_registry"]
    alarm_heap = indexes["alarm_heap"]
    while alarm_heap and alarm_heap[0][0] < horizon_ts:
        ts, _seq, alarm_id = heapq.heappop(alarm_heap)
        entry = alarm_registry.get(alarm_id)
        if entry is not None and entry[0] == ts:
            del alarm_registry[alarm_id]

    # 汇聚组：最晚告警早于视界（或从未有告警）即整体回收。
    agg_registry = session["agg_registry"]
    for agg_id in tuple(indexes["empty_agg_ids"]):
        entry = agg_registry.get(agg_id)
        if entry is not None and entry[1] is None:
            _expire_agg(session, agg_id)
        else:
            indexes["empty_agg_ids"].discard(agg_id)
    agg_heap = indexes["agg_heap"]
    while agg_heap and agg_heap[0][0] < horizon_ts:
        ts, _seq, agg_id = heapq.heappop(agg_heap)
        entry = agg_registry.get(agg_id)
        if entry is not None and entry[1] == ts:
            _expire_agg(session, agg_id)

    # 原始组自身过期时删除；汇聚组过期导致的归属清空由 groups_by_agg
    # 反向索引精确处理，不再扫描全部 group_registry。
    group_registry = session["group_registry"]
    group_heap = indexes["group_heap"]
    while group_heap and group_heap[0][0] < horizon_ts:
        ts, _seq, group_id = heapq.heappop(group_heap)
        entry = group_registry.get(group_id)
        if entry is not None and entry[0] == ts:
            _delete_group(session, group_id)

    _maybe_compact_history_expiry_heaps(session)


def _new_history_expiry_indexes():
    return {
        "seq": 0,
        "alarm_heap": [],
        "group_heap": [],
        "agg_heap": [],
        "empty_agg_ids": set(),
        "groups_by_agg": {},
    }


def _push_history_expiry(session, registry_name, item_id, ts):
    indexes = session.get("history_expiry_indexes")
    if indexes is None or ts is None:
        return
    indexes["seq"] += 1
    heapq.heappush(
        indexes[f"{registry_name}_heap"],
        (ts, indexes["seq"], item_id),
    )


def _register_alarm(session, alarm_id, ts, owner_agg_id):
    entry = [ts, owner_agg_id]
    session["alarm_registry"][alarm_id] = entry
    _push_history_expiry(session, "alarm", alarm_id, ts)
    return entry


def _register_group(session, group_id, last_ts, owner_agg_id):
    entry = [last_ts, owner_agg_id]
    session["group_registry"][group_id] = entry
    _push_history_expiry(session, "group", group_id, last_ts)
    indexes = session.get("history_expiry_indexes")
    if indexes is not None and owner_agg_id is not None:
        indexes["groups_by_agg"].setdefault(owner_agg_id, set()).add(group_id)
    return entry


def _touch_group(session, group_id, ts):
    if ts is None:
        return
    entry = session["group_registry"][group_id]
    if entry[0] is None or ts > entry[0]:
        entry[0] = ts
        _push_history_expiry(session, "group", group_id, ts)


def _set_group_owner(session, group_id, owner_agg_id):
    entry = session["group_registry"][group_id]
    old_owner_agg_id = entry[1]
    if old_owner_agg_id == owner_agg_id:
        return
    indexes = session.get("history_expiry_indexes")
    if indexes is not None and old_owner_agg_id is not None:
        old_groups = indexes["groups_by_agg"].get(old_owner_agg_id)
        if old_groups is not None:
            old_groups.discard(group_id)
            if not old_groups:
                indexes["groups_by_agg"].pop(old_owner_agg_id, None)
    entry[1] = owner_agg_id
    if indexes is not None and owner_agg_id is not None:
        indexes["groups_by_agg"].setdefault(owner_agg_id, set()).add(group_id)


def _delete_group(session, group_id):
    entry = session["group_registry"].pop(group_id, None)
    if entry is None:
        return
    owner_agg_id = entry[1]
    indexes = session.get("history_expiry_indexes")
    if indexes is None or owner_agg_id is None:
        return
    groups = indexes["groups_by_agg"].get(owner_agg_id)
    if groups is not None:
        groups.discard(group_id)
        if not groups:
            indexes["groups_by_agg"].pop(owner_agg_id, None)


def _ensure_agg(session, agg_id):
    entry = session["agg_registry"].get(agg_id)
    if entry is not None:
        return entry
    entry = [0, None, 0]
    session["agg_registry"][agg_id] = entry
    indexes = session.get("history_expiry_indexes")
    if indexes is not None:
        indexes["empty_agg_ids"].add(agg_id)
    return entry


def _touch_agg(session, agg_id, ts):
    entry = session["agg_registry"][agg_id]
    if entry[1] is None or ts > entry[1]:
        entry[1] = ts
        indexes = session.get("history_expiry_indexes")
        if indexes is not None:
            indexes["empty_agg_ids"].discard(agg_id)
        _push_history_expiry(session, "agg", agg_id, ts)


def _expire_agg(session, agg_id):
    if session["agg_registry"].pop(agg_id, None) is None:
        return
    indexes = session["history_expiry_indexes"]
    indexes["empty_agg_ids"].discard(agg_id)
    for group_id in tuple(indexes["groups_by_agg"].pop(agg_id, ())):
        entry = session["group_registry"].get(group_id)
        if entry is None or entry[1] != agg_id:
            continue
        if entry[0] is None:
            _delete_group(session, group_id)
        else:
            entry[1] = None


def _maybe_compact_history_expiry_heaps(session):
    """失效版本过多时偶尔重建，约束懒删除堆的常驻内存。"""
    indexes = session["history_expiry_indexes"]
    specs = (
        ("alarm", "alarm_registry", 0),
        ("group", "group_registry", 0),
        ("agg", "agg_registry", 1),
    )
    for name, registry_key, ts_index in specs:
        heap = indexes[f"{name}_heap"]
        registry = session[registry_key]
        if len(heap) <= max(64, len(registry) * 3):
            continue
        rebuilt = []
        for item_id, entry in registry.items():
            ts = entry[ts_index]
            if ts is None:
                continue
            indexes["seq"] += 1
            rebuilt.append((ts, indexes["seq"], item_id))
        heapq.heapify(rebuilt)
        indexes[f"{name}_heap"] = rebuilt


def _new_agg_group(session):
    """分配一个新的汇聚组 ID（UUID）并初始化注册表条目。"""
    agg_id = str(uuid.uuid4())
    _ensure_agg(session, agg_id)
    return agg_id


def _attach_alarms_to_agg(session, agg_id, alarm_ids):
    """把（尚未归属的）告警划入汇聚组：更新归属与最晚时间画像。

    组个数计数由调用方按新成员组数维护，这里只做告警级登记。
    """
    if not alarm_ids:
        return
    alarm_registry = session["alarm_registry"]
    agg_entry = session["agg_registry"][agg_id]
    agg_entry[2] += len(alarm_ids)
    latest_ts = agg_entry[1]
    for alarm_id in alarm_ids:
        alarm_entry = alarm_registry[alarm_id]
        alarm_entry[1] = agg_id
        if latest_ts is None or alarm_entry[0] > latest_ts:
            latest_ts = alarm_entry[0]
    if latest_ts is not None:
        _touch_agg(session, agg_id, latest_ts)


def _split_members_by_cap(members, max_member_count):
    """把分量成员按原始组个数上限切分，返回成员列表的列表。

    成员需已按时间排序；上限按组个数计，单个原始组恒为 1，不会被拆开。
    """
    cap = max(1, max_member_count)
    return [members[i:i + cap] for i in range(0, len(members), cap)]


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
