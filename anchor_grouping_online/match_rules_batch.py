"""ANCHOR 二次汇聚批处理入口。

对外接口只有 BatchFaultGroupMatcher.aggregate_alarm_groups：输入已成组的
故障组集合 {故障组id: [告警, ...]}，输出
{汇聚故障组id: {"is_alive": bool, "group_members": [...]}, ...}。
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
全局端口 VID 从 extendedattr 的 portVid 条目解析。
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
   新告警）的存活汇聚组；给定 old_agg_alarm_groups 时，还输出所有历史告警
   都早于历史视界的失活汇聚组。key 是稳定的汇聚组 ID，value 包含
   is_alive 与 group_members；纯重发、无新增且仍存活的汇聚组不输出。
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


def _merge_output_members(member_entry_sets, matching_cache, ne_to_site):
    """合并多组成员条目，并按原始组 ID / 告警 ID 去重。"""
    groups_by_id = {}
    for member_entries in member_entry_sets:
        _merge_member_entries(
            groups_by_id, member_entries or (), matching_cache, ne_to_site
        )
    return [
        {group_id: list(alarms_by_id.values())}
        for group_id, alarms_by_id in groups_by_id.items()
    ]


class BatchFaultGroupMatcher:
    """按批次做 ANCHOR 二次汇聚。

    静态拓扑上下文与规则配置在构造时加载一次；时序引擎与汇聚状态持久
    保留，后一批可以关联到前面批次喂入的告警。历史保留视界 = 本批最早
    告警时间 − max_group_time，每批开始时统一清理。需要重新开始时调用
    reset()。

    有状态语义要点：
    - 同一告警编码 ID 在 event_cache 中只保留一份；当前批重发时幂等覆盖
      并重新获得 trigger 资格，不追加重复缓存；
    - 组 ID 连续性：同一原始组 ID 分批送来的告警稳定归属同一汇聚组
      （视界内），无需依赖告警重叠或拓扑匹配；
    - 已输出汇聚组不可变：原始组一旦归属某个汇聚组不再改挂，既有汇聚组
      之间不合并；未汇聚的单成员组不建立二次汇聚候选状态；
    - max_group_member 按原始组个数计：附着会超限时放弃附着，已有归属的
      组仍沿用原汇聚组输出，未归属的直接关联组按上限切分，切分后的单
      成员不分配 ID；
    - associate_time / max_group_time 逐次可变：变化时原地更新会话（规则
      时间窗重编、TTL 更新），从本批起按新参数判定与过期，已积累的历史
      不重算；
    - 对输入顺序不做假设：批内、批间乱序均不影响聚合正确性——判定只依据
      告警自身的发生时间（持久缓存有序、编码 ID 去重、跨批时间回退单调
      安全）；过老的告警只是超出关联窗/视界时自然关联不上。

    批隔离模式（batch_isolated=True）：不保留内部会话，未传 old_agg 时只
    做本批（本窗口）内的告警汇聚；显式传入 old_agg 时仅在本次调用内把它
    作为历史上下文，调用结束即释放。新汇聚组 ID 每批全新生成，跨批比较
    无意义。

    全量输出模式（full_output=True）：存活组仍只返回本批发生变化的组，
    成员完整度从「本批增量」提升为「完整成员」；失活组返回 old_agg 中的
    全部成员。full_output=False 时，失活组仍返回但 group_members 为空。

    外置状态模式（old_agg_alarm_groups）：传入旧格式的既有二次汇聚故障组
    {汇聚组ID: [{原始组ID: [告警, ...]}, ...]} 时，以它为跨窗口上下文
    重建临时会话后再汇聚本批，可关联
    其中的告警/原始组/汇聚组 ID；其归属与计数完整保留、不过期，但告警
    只有位于「本批最早告警时间 - max_group_time」之后才写入临时
    event_cache 作为匹配证据（边匹配仍受 associate_time 约束）。调用结束
    把外置归属/计数与本批结果同步进内部会话（自动去重），但 old_agg
    历史事件仅本次有效，持久引擎只喂本批告警。适合调用方自行保管状态的
    无状态服务部署。old_agg 中同一告警出现在多个汇聚组下时记为冲突
    告警（不报错）：注册表保留先遇到的归属、但不计入任何汇聚组的告警
    数，连边阶段挂起其连续性拉力——既不把所在组拉向任何历史汇聚组、
    也不作为附着目标证据。同一原始组出现在多个汇聚组下时同样降级为
    组级冲突（不报错）：各汇聚组仍把它挂名为成员（组计数照加）、归属
    先到先得，但连续性效力全面挂起（不参与本批归属分配），其所有告警
    连坐冲突（谁都不计）。把新结构输出累积到旧格式 old_agg 时，先读取每个输出
    value 的 group_members：is_alive=False 时删除对应历史组；存活组在
    full_output=True 时按 agg_id 用 group_members 整体替换，False 时按原始
    组 ID 合并并按告警编码 ID 去重（直接 extend 会重复累积、内存膨胀）。

    输出口径：存活组只返回本批发生变化（新汇聚组／新成员组／组内新告警）
    的组，纯重发不返回；外置状态中所有告警都早于 history_horizon_ts 的组
    额外返回为 is_alive=False。key 为稳定的汇聚故障组 ID；value 为
    {"is_alive": bool, "group_members": 原始组条目列表}。成员条目格式为
    {原始组id: [告警, ...]}（按告警编码 ID 去重）。只输出累计至少含 2 个
    原始故障组的存活汇聚组；未汇聚且未附着的单成员组不分配 ID。
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
        """按 ANCHOR 拓扑匹配关系把已成组的故障组合并为汇聚故障组。

        alarm_groups 为 {故障组id: [告警, ...]}（字段见模块 docstring）；
        associate_time / max_group_time 为关联时间窗与组保留时长（分钟）；
        max_group_member 为汇聚组最大原始组个数（超限时放弃附着或按成员
        时间顺序切分，单个原始组不拆开）；old_agg_alarm_groups 为既有二次
        汇聚故障组（外置状态模式）。输出口径与各模式语义详见类 docstring。
        """
        max_member_count = int(max_group_member)
        # 调用内共享的告警转换缓存：各阶段复用同一份 to_matching_alarm 结果。
        matching_cache = {}
        ne_to_site = self.static_context.ne_to_site
        session = self._open_batch_session(
            alarm_groups, associate_time, max_group_time,
            old_agg_alarm_groups, matching_cache, ne_to_site,
        )
        engine = session["engine"]
        # 1. 登记本次输入；trigger 只描述“本批哪些告警主动发起匹配”。
        batch = self._register_batch_input(
            session, engine, alarm_groups, matching_cache, ne_to_site
        )
        # 2. 批开始的历史回收（外置状态模式跳过）。
        self._prune_batch_horizon(session, batch, old_agg_alarm_groups)
        # 3. 喂流 + 收割，再把匹配组转成组间连边与历史汇聚组关联。
        output_matches = self._collect_batch_matches(
            engine, batch, matching_cache, ne_to_site
        )
        links = self._link_batch_groups(session, alarm_groups, batch, output_matches)
        # 4. 按本批连通分量分配稳定的汇聚组 ID。
        batch_member_entries_by_agg, changed_agg_ids = self._assign_components(
            session, alarm_groups, batch, links, max_member_count
        )
        # 5~6. 构建增量、同步外置状态并生成对外输出。
        return self._finalize_batch(
            session, batch, alarm_groups, associate_time, max_group_time,
            old_agg_alarm_groups, batch_member_entries_by_agg, changed_agg_ids,
            matching_cache, ne_to_site,
        )

    def _prune_batch_horizon(self, session, batch, old_agg_alarm_groups):
        """批开始的历史回收：统一视界 = 本批最早告警时间 − max_group_time。

        外置状态模式跳过：old_agg 归属永久保留，其匹配证据已按视界过滤。
        """
        if (
            batch.batch_min_ts is None
            or old_agg_alarm_groups is not None
            or self.batch_isolated
        ):
            return
        _prune_session_horizon(session, batch.batch_min_ts)

    def _finalize_batch(
        self, session, batch, alarm_groups, associate_time, max_group_time,
        old_agg_alarm_groups, batch_member_entries_by_agg, changed_agg_ids,
        matching_cache, ne_to_site,
    ):
        """构建本批增量、同步外置状态并生成对外输出。"""
        need_full_increment = (
            old_agg_alarm_groups is not None and not self.batch_isolated
        )
        batch_increment_by_agg = self._build_batch_increments(
            session, batch_member_entries_by_agg, changed_agg_ids,
            need_full_increment, matching_cache, ne_to_site,
        )
        if need_full_increment:
            self._sync_external_increment(
                session, old_agg_alarm_groups, batch_increment_by_agg,
                alarm_groups, associate_time, max_group_time,
                batch.batch_min_ts, matching_cache,
            )
        if self.batch_isolated:
            # 隔离模式不跨批复用会话：本批算完立即释放引擎内存。
            self._session = None
        return self._build_batch_output(
            session, batch_increment_by_agg, changed_agg_ids,
            old_agg_alarm_groups, matching_cache, ne_to_site,
        )

    def _open_batch_session(
        self,
        alarm_groups,
        associate_time,
        max_group_time,
        old_agg_alarm_groups,
        matching_cache,
        ne_to_site,
    ):
        """为本次调用准备会话：内部持久会话 / 隔离新会话 / 外置重建会话。

        外置状态模式先做一次 O(batch) 的时间下界预计算：old_agg 历史只在
        「本批最早告警时间 - max_group_time」之后才作为临时引擎的告警匹配
        证据；转换结果写入 matching_cache 供后续登记/喂流复用。归属注册表
        仍完整导入。
        """
        external_history_horizon_ts = self._external_history_horizon(
            alarm_groups, max_group_time, old_agg_alarm_groups,
            matching_cache, ne_to_site,
        )
        if self.batch_isolated:
            # 隔离模式所有状态仅在本次调用内存活，不保留上一批会话，
            # 避免与本批 old_agg 临时会话叠加占用内存。
            self._session = None
        if old_agg_alarm_groups is not None:
            # 外置状态模式：以既有二次汇聚结果为跨窗口上下文构建临时会话。
            # 非隔离模式先触达内部会话，参数有变化时原地更新（保证结束时
            # 增量落库与视界清理按本次参数执行）。
            if not self.batch_isolated:
                self._ensure_session(associate_time, max_group_time)
            return self._rebuild_session_from_agg(
                old_agg_alarm_groups,
                associate_time,
                max_group_time,
                # 隔离模式后续不再同步 old_agg：历史转换结果写入
                # event_cache 后即可释放，不把 O(|old_agg|) 的缓存
                # 保留到调用结束；非隔离模式继续复用调用级缓存。
                matching_cache=None if self.batch_isolated else matching_cache,
                history_horizon_ts=external_history_horizon_ts,
            )
        if self.batch_isolated:
            # 批隔离模式：本批自成一个不落到 self 的全新会话。
            return self._new_session(associate_time, max_group_time)
        return self._ensure_session(associate_time, max_group_time)

    @staticmethod
    def _external_history_horizon(
        alarm_groups, max_group_time, old_agg_alarm_groups,
        matching_cache, ne_to_site,
    ):
        """外置状态模式的历史证据时间下界：本批最早告警时间 - max_group_time。

        非外置模式返回 None。O(batch) 的预计算；转换结果写入
        matching_cache 供后续登记/喂流复用。
        """
        if old_agg_alarm_groups is None:
            return None
        batch_min_ts = None
        for group_alarms in alarm_groups.values():
            for generated_alarm in group_alarms or ():
                ts = _matching_alarm(generated_alarm, matching_cache, ne_to_site)["ts"]
                if batch_min_ts is None or ts < batch_min_ts:
                    batch_min_ts = ts
        if batch_min_ts is None:
            return None
        return batch_min_ts - float(max_group_time) * 60.0

    def _register_batch_input(
        self, session, engine, alarm_groups, matching_cache, ne_to_site
    ):
        """登记本次输入，返回本批工作档案（SimpleNamespace）。

        原始组的告警集/时间只作为本批的局部档案；会话里只记「告警 -> 汇聚
        组 ID」的归属。持久会话通过 eid upsert 索引判断实际缓存；隔离/外置
        临时会话要么全新为空，要么由 old_agg 完整重建，登记状态即等于缓存
        状态，因此都无需全量扫描 event_cache 构造 ID set。
        """
        batch = SimpleNamespace(
            local_group_alarm_ids={},
            local_group_alarms={},
            local_group_min_ts={},
            local_alarm_owners={},
            current_generated_alarms=[],
            new_generated_alarms=[],
            overlapping_generated_alarms=[],
            batch_min_ts=None,
            use_batch_upsert=engine._batch_event_by_alarm_id is not None,
        )
        batch_alarm_ids = set()
        for group_id, group_alarms in alarm_groups.items():
            self._register_batch_group(
                session, batch, batch_alarm_ids, group_id, group_alarms,
                matching_cache, ne_to_site,
            )
        return batch

    def _register_batch_group(
        self, session, batch, batch_alarm_ids, group_id, group_alarms,
        matching_cache, ne_to_site,
    ):
        """登记单个原始组的本批告警，并刷新组注册表的最近告警时间。"""
        alarm_ids = set()
        kept_alarms = []
        min_ts = None
        group_last_ts = None
        for generated_alarm in group_alarms or ():
            matching_alarm = _matching_alarm(generated_alarm, matching_cache, ne_to_site)
            alarm_id = matching_alarm["alarm_id"]
            ts = matching_alarm["ts"]
            first_in_batch = alarm_id not in batch_alarm_ids
            batch_alarm_ids.add(alarm_id)
            if first_in_batch:
                batch.current_generated_alarms.append(generated_alarm)
            if batch.batch_min_ts is None or ts < batch.batch_min_ts:
                batch.batch_min_ts = ts
            min_ts = ts if min_ts is None else min(min_ts, ts)
            if alarm_id not in alarm_ids:
                kept_alarms.append(generated_alarm)
            alarm_ids.add(alarm_id)
            owner_group_ids = batch.local_alarm_owners.setdefault(alarm_id, [])
            if group_id not in owner_group_ids:
                owner_group_ids.append(group_id)
            alarm_entry = session["alarm_registry"].get(alarm_id)
            alarm_was_registered = alarm_entry is not None
            if alarm_entry is None:
                alarm_entry = _register_alarm(session, alarm_id, ts, None)
            registered_ts = alarm_entry[0]
            if group_last_ts is None or registered_ts > group_last_ts:
                group_last_ts = registered_ts
            self._classify_trigger_eligibility(
                batch, generated_alarm, first_in_batch, alarm_was_registered,
                matching_alarm["is_clear"],
            )
        batch.local_group_alarm_ids[group_id] = alarm_ids
        batch.local_group_alarms[group_id] = kept_alarms
        batch.local_group_min_ts[group_id] = min_ts
        # 组注册表：刷新最近告警时间（归属在分量分配阶段写入）。
        group_entry = session["group_registry"].get(group_id)
        if group_entry is None:
            _register_group(session, group_id, group_last_ts, None)
        else:
            _touch_group(session, group_id, group_last_ts)

    @staticmethod
    def _classify_trigger_eligibility(
        batch, generated_alarm, first_in_batch, alarm_was_registered, is_clear
    ):
        """非 upsert 引擎按「新告警/重叠告警」划分本批 trigger 资格。"""
        if batch.use_batch_upsert or not first_in_batch:
            return
        if not alarm_was_registered or is_clear:
            batch.new_generated_alarms.append(generated_alarm)
        else:
            batch.overlapping_generated_alarms.append(generated_alarm)

    def _collect_batch_matches(self, engine, batch, matching_cache, ne_to_site):
        """增量喂流 + 逐 trigger 收割 + 批内/历史合并与输出可见性过滤。"""
        raw_matches = []
        try:
            if batch.use_batch_upsert and batch.current_generated_alarms:
                batch_trigger_candidates = self._feed_engine(
                    engine,
                    batch.current_generated_alarms,
                    sort_events=True,
                    upsert_events=True,
                    matching_cache=matching_cache,
                    ne_to_site=ne_to_site,
                )
            elif batch.new_generated_alarms or batch.overlapping_generated_alarms:
                batch_trigger_candidates = self._feed_engine(
                    engine,
                    batch.new_generated_alarms,
                    # 已在历史缓存中的本批告警不重复缓存，但与新告警一样
                    # 重新获得本批 trigger 资格。
                    trigger_only_alarms=batch.overlapping_generated_alarms,
                    sort_events=not self.batch_isolated,
                    # 本批告警在登记阶段已全部转换并写入 matching_cache。
                    # 隔离模式无需排序，可直接流式喂入。
                    preconverted_events=self.batch_isolated,
                    matching_cache=matching_cache,
                    ne_to_site=ne_to_site,
                )
            else:
                batch_trigger_candidates = []
            if batch.current_generated_alarms:
                raw_matches = self._aggregate_per_trigger(
                    engine,
                    trigger_candidates=batch_trigger_candidates,
                    owns_trigger_candidates=True,
                    trigger_candidates_sorted=not self.batch_isolated,
                )
        finally:
            # trigger_event_index 仅服务本次收割期间的消费删除；剩余候选
            # 没有跨批意义，异常退出时也立即释放内存。
            engine.trigger_event_index.clear()
        if not batch.current_generated_alarms:
            return []
        return self._merge_batch_and_finalize(engine, raw_matches)

    def _link_batch_groups(self, session, alarm_groups, batch, output_matches):
        """匹配组转关联证据：本批原始组之间连边（局部并查集）；跨组共享
        告警的原始组直接连边；通过症状告警的既有归属记录本批原始组与
        历史汇聚组的关联；再按组 ID/告警既有归属做连续性沿用。
        """
        local_union = _GroupUnionFind()
        for group_id in alarm_groups:
            local_union.add(group_id)
        linked_agg_ids_by_group = {group_id: [] for group_id in alarm_groups}
        # 本批组之间的直接关联边（每个匹配组/每条共享告警一条超边）。附着
        # 失败拆分时用它重算子图：只有直接关联的成员才保持连通。
        current_group_edges = []
        # 共享告警刚性边（current_group_edges 的子集，单独维护）：同一告警
        # 编码 ID 属于多个本批原始组时，这些组被视为不可分割的刚性簇，必须
        # 落进同一汇聚组，可突破 max_group_member。区别于症状匹配边——后者
        # 只提供“可同组”资格、允许按上限切分，共享告警边则是硬约束。若某组
        # 在外置历史里已跨多个既有汇聚组，则被判为组级冲突（conflicted_group_ids，
        # 降级容忍、不报错：各 agg 挂名保留、连续性挂起、告警连坐）。
        shared_alarm_edges = []
        for owner_group_ids in batch.local_alarm_owners.values():
            if len(owner_group_ids) >= 2:
                local_union.union_all(owner_group_ids)
                edge = list(owner_group_ids)
                current_group_edges.append(edge)
                shared_alarm_edges.append(edge)

        def link_group_to_agg(group_id, agg_id):
            # 急切改写保证指针直指存活汇聚组，这里只需做存在性防御。
            linked = linked_agg_ids_by_group[group_id]
            if (
                agg_id is not None
                and agg_id in session["agg_registry"]
                and agg_id not in linked
            ):
                linked.append(agg_id)

        # 冲突告警（外置历史中同一告警挂多个汇聚组）的归属字段是遍历序
        # 先到先得的产物，连续性效力挂起：不作为附着目标证据、不把所在
        # 组拉向任何历史汇聚组。作为本批组间连边/匹配证据不受影响。
        conflicted_alarm_ids = session.get("conflicted_alarm_ids") or frozenset()
        # 冲突组（同一原始组挂多个汇聚组）连续性效力同样挂起：不把它拉向
        # 任何历史汇聚组、不作附着目标证据（仍可作本批组间连边/匹配证据）。
        conflicted_group_ids = session.get("conflicted_group_ids") or frozenset()

        for match in self._iter_output_matches(output_matches):
            related_group_ids = []
            related_agg_ids = []
            for alarm_id in _collect_symptom_alarm_ids(match):
                for group_id in batch.local_alarm_owners.get(alarm_id, ()):
                    if group_id not in related_group_ids:
                        related_group_ids.append(group_id)
                alarm_entry = session["alarm_registry"].get(alarm_id)
                if (
                    alarm_entry is not None
                    and alarm_entry[1] is not None
                    and alarm_id not in conflicted_alarm_ids
                ):
                    related_agg_ids.append(alarm_entry[1])
            local_union.union_all(related_group_ids)
            if len(related_group_ids) >= 2:
                current_group_edges.append(list(related_group_ids))
            for group_id in related_group_ids:
                for agg_id in related_agg_ids:
                    link_group_to_agg(group_id, agg_id)

        # 原始组自身的连续性：组 ID 出现过就沿用其既有汇聚组归属；组内
        # 告警若已归属某汇聚组也沿用（覆盖重叠窗口重发、跨组共享告警）。
        for group_id, alarm_ids in batch.local_group_alarm_ids.items():
            if group_id in conflicted_group_ids:
                continue
            link_group_to_agg(group_id, session["group_registry"][group_id][1])
            for alarm_id in alarm_ids:
                alarm_entry = session["alarm_registry"].get(alarm_id)
                if alarm_entry is not None and alarm_id not in conflicted_alarm_ids:
                    link_group_to_agg(group_id, alarm_entry[1])
        return (
            local_union,
            linked_agg_ids_by_group,
            current_group_edges,
            shared_alarm_edges,
        )

    def _assign_components(self, session, alarm_groups, batch, links, max_member_count):
        """按本批连通分量分配稳定的汇聚组 ID（上限按原始组个数计）：
        已有归属的组沿用原汇聚组、既有汇聚组之间不合并；未归属组关联到
        多个历史组时选一个容量足够的目标；附着会超限则放弃附着、未归属的
        直接关联组按上限切分；无关联时按上限切分，至少两个成员的包才分配
        新 ID。返回 (本批各汇聚组成员条目, 本批发生变化的汇聚组 ID 集合)。
        """
        (
            local_union,
            linked_agg_ids_by_group,
            current_group_edges,
            shared_alarm_edges,
        ) = links
        # 共享告警刚性并查集：只并合“共享同一条告警”的原始组，得到必须
        # 整体同组的最小不可分单元（症状匹配边不参与），供归属分配/切分
        # 以刚性簇为单位处理。
        rigid_union = _GroupUnionFind()
        for group_id in alarm_groups:
            rigid_union.add(group_id)
        for edge_group_ids in shared_alarm_edges:
            rigid_union.union_all(edge_group_ids)
        member_order_key = _batch_member_order_key(batch)
        # changed_agg_ids 记录本批真正变化的汇聚组：新增原始组成员，或已有
        # 成员组内新增告警；新建汇聚组必然带来新成员，天然纳入。
        batch_member_entries_by_agg = {}
        changed_agg_ids = set()
        components = local_union.components(alarm_groups)
        for members in components:
            members.sort(key=member_order_key)
        components.sort(key=lambda members: member_order_key(members[0]))
        for members in components:
            self._assign_component(
                session, batch, members, linked_agg_ids_by_group,
                current_group_edges, rigid_union, max_member_count,
                batch_member_entries_by_agg, changed_agg_ids,
            )
        return batch_member_entries_by_agg, changed_agg_ids

    def _assign_component(
        self, session, batch, members, linked_agg_ids_by_group,
        current_group_edges, rigid_union, max_member_count,
        batch_member_entries_by_agg, changed_agg_ids,
    ):
        """单个连通分量的归属分配：沿用既有归属 → 附着历史组 → 新建。

        以共享告警刚性簇为单位处理：同簇原始组必须进同一汇聚组（见
        _assign_rigid_cluster），簇内若有既有归属则整簇跟随、可突破
        max_group_member。
        """
        # 冲突组效力全面挂起：不作为附着目标候选来源（也在
        # _assign_rigid_cluster 里被整体剔除、不参与归属分配）。
        conflicted_group_ids = session.get("conflicted_group_ids") or frozenset()
        linked_agg_ids = []
        for group_id in members:
            if group_id in conflicted_group_ids:
                continue
            for agg_id in linked_agg_ids_by_group[group_id]:
                if agg_id in session["agg_registry"] and agg_id not in linked_agg_ids:
                    linked_agg_ids.append(agg_id)
        # 已有归属一旦落定就不改挂；其余成员仅在能附着到既有汇聚组或
        # 与至少一个其他成员形成新汇聚时才分配 ID。刚性簇内若已有归属，
        # 整簇跟随（可突破上限）；全未归属的簇整体进入下面的附着/新建。
        unassigned_members = []
        for cluster_members in _group_by_rigid_cluster(members, rigid_union):
            self._assign_rigid_cluster(
                session, batch, cluster_members, unassigned_members,
                batch_member_entries_by_agg, changed_agg_ids,
            )
        if not unassigned_members:
            return
        # 新成员同时关联多个历史汇聚组时，只选一个能整体容纳本子分量
        # 的目标，既有汇聚组之间不合并。
        eligible_targets = [
            agg_id for agg_id in linked_agg_ids
            if session["agg_registry"][agg_id][0] + len(unassigned_members)
            <= max_member_count
        ]
        if eligible_targets:
            eligible_targets.sort(
                key=lambda a: (-session["agg_registry"][a][2], str(a))
            )
            self._assign_members_to_agg(
                session, eligible_targets[0], unassigned_members, batch,
                batch_member_entries_by_agg, changed_agg_ids,
            )
            return
        self._assign_split_components(
            session, batch, unassigned_members, current_group_edges,
            rigid_union, max_member_count, batch_member_entries_by_agg,
            changed_agg_ids,
        )

    def _assign_rigid_cluster(
        self, session, batch, cluster_members, unassigned_members,
        batch_member_entries_by_agg, changed_agg_ids,
    ):
        """处理单个共享告警刚性簇：同簇原始组必须进同一汇聚组。

        既有归属同时来自两个层面（都视为刚性）：
        - 组级：簇内原始组自身已归属的汇聚组；
        - 告警级：簇内告警已归属的历史汇聚组——共享告警若已属某历史
          汇聚组，认领它的新组必须并入该组，否则会出现“组归新汇聚组、
          共享告警仍归旧汇聚组”的分裂（成员非空但告警计数为 0，回灌
          还会形成同一告警跨汇聚组冲突）。冲突告警（continuity 已挂起）
          不计入，与连边阶段口径一致。

        - 簇内全部未归属：整簇作为未归属成员进入后续附着/新建流程，
          切分阶段按簇不可分处理，不会被上限切开。
        - 簇内恰有一个既有汇聚组：整簇并入该组，未归属成员跟随
          （共享告警是硬约束，可突破 max_group_member；已归属该组的
          成员/告警不重复计数）。
        - 簇内跨多个既有汇聚组：满足“共享必同组”需合并既有汇聚组，
          该决策未落地——此处退回逐组沿用既有归属（等价旧行为，不
          强制同簇），既有汇聚组之间仍不合并。
        """
        conflicted_alarm_ids = session.get("conflicted_alarm_ids") or frozenset()
        conflicted_group_ids = session.get("conflicted_group_ids") or frozenset()
        # 冲突组效力全面挂起：整体从簇里剔除，不参与本批任何归属分配——
        # 既不把簇拉向其历史归属，也不被簇拉去改挂（否则会撞
        # _assign_members_to_agg 的“不可改挂”校验而报错），冻结在挂名状态。
        # 只对非冲突成员做归属；剔空则本簇不分配。
        cluster_members = [
            group_id for group_id in cluster_members
            if group_id not in conflicted_group_ids
        ]
        if not cluster_members:
            return
        existing_aggs = []

        def _note_existing_agg(agg_id):
            if agg_id in session["agg_registry"] and agg_id not in existing_aggs:
                existing_aggs.append(agg_id)

        for group_id in cluster_members:
            _note_existing_agg(session["group_registry"][group_id][1])
            for alarm_id in batch.local_group_alarm_ids[group_id]:
                if alarm_id in conflicted_alarm_ids:
                    continue
                alarm_entry = session["alarm_registry"].get(alarm_id)
                if alarm_entry is not None:
                    _note_existing_agg(alarm_entry[1])
        if len(existing_aggs) == 1:
            self._assign_members_to_agg(
                session, existing_aggs[0], cluster_members, batch,
                batch_member_entries_by_agg, changed_agg_ids,
            )
            return
        if len(existing_aggs) >= 2:
            # ③ 未覆盖：逐组沿用既有归属，未归属组按普通未归属成员处理。
            for group_id in cluster_members:
                existing_agg_id = session["group_registry"][group_id][1]
                if existing_agg_id in session["agg_registry"]:
                    self._assign_members_to_agg(
                        session, existing_agg_id, [group_id], batch,
                        batch_member_entries_by_agg, changed_agg_ids,
                    )
                else:
                    _set_group_owner(session, group_id, None)
                    unassigned_members.append(group_id)
            return
        for group_id in cluster_members:
            _set_group_owner(session, group_id, None)
            unassigned_members.append(group_id)

    def _assign_split_components(
        self, session, batch, unassigned_members, current_group_edges,
        rigid_union, max_member_count, batch_member_entries_by_agg,
        changed_agg_ids,
    ):
        """无历史目标可容纳时的新建分配。

        只保留未归属成员彼此间的直接匹配边重算连通子图（仅通过已落定
        成员传递连通的新组自然拆开）；子图内以共享告警刚性簇为不可分
        物品，按最早告警时间做 next-fit 装箱（上限对能装下的物品严格
        生效，单个刚性簇超过上限时独占一个超限包、不被拆开），为 >=2
        成员的包分配全新汇聚组 ID。
        """
        member_order_key = _batch_member_order_key(batch)
        unassigned_set = set(unassigned_members)
        sub_union = _GroupUnionFind()
        for group_id in unassigned_members:
            sub_union.add(group_id)
        for edge_group_ids in current_group_edges:
            sub_union.union_all([g for g in edge_group_ids if g in unassigned_set])
        sub_components = sub_union.components(unassigned_members)
        for sub_members in sub_components:
            sub_members.sort(key=member_order_key)
        sub_components.sort(key=lambda sub: member_order_key(sub[0]))
        for sub_members in sub_components:
            clusters = _group_by_rigid_cluster(sub_members, rigid_union)
            for cluster in clusters:
                cluster.sort(key=member_order_key)
            clusters.sort(key=lambda cluster: member_order_key(cluster[0]))
            for pack_members in _pack_clusters_by_cap(clusters, max_member_count):
                if len(pack_members) < 2:
                    continue
                self._assign_members_to_agg(
                    session, _new_agg_group(session), pack_members, batch,
                    batch_member_entries_by_agg, changed_agg_ids,
                )

    def _assign_members_to_agg(
        self, session, agg_id, member_group_ids, batch,
        batch_member_entries_by_agg, changed_agg_ids,
    ):
        """成员组划入汇聚组：登记归属/计数/告警映射与本批条目。"""
        new_alarm_ids = []
        seen_new_alarm_ids = set()
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
            # 跨组去重：同一条新告警被簇内多个成员组共同包含时只登记一次。
            # 告警归属在本方法末尾（_attach_alarms_to_agg）才统一写入，遍历
            # 期间共享告警始终显示未归属，不去重会被每个组各收一次，导致
            # 汇聚组告警计数按重复次数虚增。
            for alarm_id in batch.local_group_alarm_ids[group_id]:
                if alarm_id in seen_new_alarm_ids:
                    continue
                if session["alarm_registry"][alarm_id][1] is not None:
                    continue
                seen_new_alarm_ids.add(alarm_id)
                new_alarm_ids.append(alarm_id)
        session["agg_registry"][agg_id][0] += new_group_count
        _attach_alarms_to_agg(session, agg_id, new_alarm_ids)
        if new_group_count or new_alarm_ids:
            changed_agg_ids.add(agg_id)
        batch_member_entries_by_agg.setdefault(agg_id, []).extend(
            {group_id: batch.local_group_alarms[group_id]}
            for group_id in member_group_ids
        )

    @staticmethod
    def _build_batch_increments(
        session, batch_member_entries_by_agg, changed_agg_ids,
        need_full_increment, matching_cache, ne_to_site,
    ):
        """构建本批各汇聚组的增量条目。

        仅两处消费：外置同步（需全部增量，含纯重发和暂未达到输出门槛
        的条目）与对外输出（仅变化且累计 count>=2 的汇聚组）。不做
        外置同步时只构建变化组，跳过纯重发、未变化汇聚组的成员合并。
        """
        batch_increment_by_agg = {}
        for agg_id, member_entries in batch_member_entries_by_agg.items():
            if not need_full_increment and agg_id not in changed_agg_ids:
                continue
            if (
                session["agg_registry"][agg_id][0] < 2
                and not need_full_increment
            ):
                continue
            groups_by_id = {}
            _merge_member_entries(
                groups_by_id, member_entries, matching_cache, ne_to_site
            )
            batch_increment_by_agg[agg_id] = [
                {group_id: list(alarms_by_id.values())}
                for group_id, alarms_by_id in groups_by_id.items()
            ]
        return batch_increment_by_agg

    def _sync_external_increment(
        self, session, old_agg_alarm_groups, batch_increment_by_agg,
        alarm_groups, associate_time, max_group_time, batch_min_ts,
        matching_cache,
    ):
        """把完整外置历史的归属/计数、本批增量和本批原始告警并入内部会话。

        old_agg 的历史事件缓存只属于本次临时会话，持久引擎仅接收本批告警；
        未汇聚组不创建二次汇聚 ID。
        """
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
            conflicted_alarm_ids=session.get("conflicted_alarm_ids"),
            conflicted_group_ids=session.get("conflicted_group_ids"),
            imported_agg_counts={
                agg_id: (
                    session["agg_registry"][agg_id][0],
                    session["agg_registry"][agg_id][2],
                )
                for agg_id in state_to_import
            },
        )

    def _build_batch_output(
        self, session, batch_increment_by_agg, changed_agg_ids,
        old_agg_alarm_groups, matching_cache, ne_to_site,
    ):
        """构造带 is_alive 状态的对外输出。

        存活组只暴露本批发生变化的组。成员完整度由 full_output 决定：
        默认输出本批增量成员；
        full_output=True 且传入 old_agg 时合并既有全量得到完整成员，
        无 old_agg 时无全量可回显、退回增量成员。失活组始终返回；
        full_output=True 时带 old_agg 全量成员，否则 group_members=[]。
        """
        agg_alarm_groups = self._build_changed_agg_outputs(
            session,
            batch_increment_by_agg,
            changed_agg_ids,
            old_agg_alarm_groups,
            matching_cache,
            ne_to_site,
        )
        self._append_inactive_agg_outputs(
            agg_alarm_groups,
            session,
            changed_agg_ids,
            old_agg_alarm_groups,
            matching_cache,
            ne_to_site,
        )
        return agg_alarm_groups

    def _build_changed_agg_outputs(
        self, session, batch_increment_by_agg, changed_agg_ids,
        old_agg_alarm_groups, matching_cache, ne_to_site,
    ):
        """构造本批发生变化且累计 count>=2 的存活汇聚组输出。"""
        outputs = {}
        for agg_id in changed_agg_ids:
            if session["agg_registry"][agg_id][0] < 2:
                continue
            batch_entries = batch_increment_by_agg.get(agg_id)
            if batch_entries is None:
                continue
            member_entry_sets = [batch_entries]
            if self.full_output and old_agg_alarm_groups is not None:
                member_entry_sets.insert(
                    0, old_agg_alarm_groups.get(agg_id) or ()
                )
            group_members = _merge_output_members(
                member_entry_sets,
                matching_cache,
                ne_to_site,
            )
            outputs[agg_id] = {
                "is_alive": True,
                "group_members": group_members,
            }
        return outputs

    def _append_inactive_agg_outputs(
        self, outputs, session, changed_agg_ids, old_agg_alarm_groups,
        matching_cache, ne_to_site,
    ):
        """追加 old_agg 中所有历史告警均已越过视界的汇聚组。"""
        if old_agg_alarm_groups is None:
            return
        inactive_agg_ids = session.get("inactive_external_agg_ids", set())
        for agg_id, member_entries in old_agg_alarm_groups.items():
            if agg_id not in inactive_agg_ids or agg_id in changed_agg_ids:
                continue
            group_members = []
            if self.full_output:
                group_members = _merge_output_members(
                    [member_entries], matching_cache, ne_to_site
                )
            outputs[agg_id] = {
                "is_alive": False,
                "group_members": group_members,
            }

    def _merge_increment_into_session(
        self, agg_output, associate_time, max_group_time, batch_min_ts,
        raw_alarm_groups=None, imported_agg_counts=None, matching_cache=None,
        conflicted_alarm_ids=None, conflicted_group_ids=None,
    ):
        """把外置归属快照、本批增量及本批原始告警并入内部会话。

        按增量合并：汇聚组条目缺失才创建；新原始组才计数；本批告警统一
        幂等写入内部引擎；原始组既有归属发生变化视为外置状态冲突直接
        报错，告警级归属冲突降级为标记（conflicted_alarm_ids，含临时
        会话已发现的冲突），连边阶段挂起其连续性效力。
        raw_alarm_groups 只补齐原始组/告警缓存；未汇聚成员的归属保持
        None，不创建单成员二次汇聚组。外置历史告警只登记归属，不写入
        持久引擎；持久 event_cache 和 trigger 只接收本批原始告警。合并后
        执行与内部调用相同的视界清理——纯外置使用形态（每批都传 old_agg）
        下这是内部状态唯一的过期出口，否则只进不出、无界增长。
        """
        session = self._ensure_session(associate_time, max_group_time)
        # 临时会话已识别的冲突不能先直接 update 到持久集合：
        # _mark_conflicted_alarm 会把“已在集合中”视为已回滚，从而遗漏
        # 持久会话中旧归属方的告警计数。逐条标记才能保证新冲突
        # 恰好回滚一次，旧冲突仍幂等。
        incoming_conflicted_alarm_ids = frozenset(conflicted_alarm_ids or ())
        for alarm_id in incoming_conflicted_alarm_ids:
            alarm_entry = session["alarm_registry"].get(alarm_id)
            _mark_conflicted_alarm(
                session,
                alarm_id,
                alarm_entry[1] if alarm_entry is not None else None,
            )
        if conflicted_group_ids:
            session.setdefault("conflicted_group_ids", set()).update(
                conflicted_group_ids
            )
        ne_to_site = self.static_context.ne_to_site
        (
            current_alarm_records_by_group,
            current_matching_alarms_by_id,
            current_generated_alarms_by_id,
        ) = _index_current_alarms(raw_alarm_groups, matching_cache, ne_to_site)
        resolve_matching_alarm = _make_matching_alarm_resolver(
            current_matching_alarms_by_id, matching_cache, ne_to_site
        )
        imported_group_ids, imported_alarm_ids = _validate_external_increment(
            session, agg_output, resolve_matching_alarm,
            need_id_sets=imported_agg_counts is None,
        )
        # imported_agg_counts 来自临时会话。若某些冲突只在与持久
        # 会话比对时才发现，临时计数仍把这些告警算在目标汇聚组
        # 里。覆盖画像前按最终冲突集合扣除“临时会话未知”的部分；
        # 临时会话已知的冲突本来就未计数，不能重复扣减。
        if imported_agg_counts is not None:
            final_conflicted_alarm_ids = (
                session.get("conflicted_alarm_ids") or frozenset()
            )
            newly_excluded_alarm_ids = (
                set(final_conflicted_alarm_ids)
                - incoming_conflicted_alarm_ids
            )
            if newly_excluded_alarm_ids:
                imported_agg_counts = _exclude_conflicts_from_imported_counts(
                    agg_output,
                    imported_agg_counts,
                    newly_excluded_alarm_ids,
                    resolve_matching_alarm,
                )
        _import_external_entries(session, agg_output, resolve_matching_alarm)
        # 本批未进入可见汇聚输出的原始告警也要保留，确保外置调用后切换到
        # 内部模式时，时序引擎拥有与连续内部模式一致的事件历史。
        _register_current_batch_groups(session, current_alarm_records_by_group)
        _overwrite_agg_counts(
            session, agg_output, imported_agg_counts,
            imported_group_ids, imported_alarm_ids,
        )
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
            _prune_session_horizon(session, batch_min_ts)

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

        同一原始组出现在多个汇聚组下（组级冲突）与同一告警出现在多个
        汇聚组下（告警级冲突）都降级容忍、不报错：
        - 组级：先两遍扫描定位冲突组（出现在 ≥2 个 agg），各 agg 仍把它
          挂名为成员（组计数照加），归属先到先得、不改挂，连边阶段挂起
          其连续性拉力，其所有告警连坐 conflicted（谁都不计）；
        - 告警级：记入 conflicted_alarm_ids，注册表保留先遇到的归属但
          回滚其告警计数，连边阶段挂起其连续性拉力。
        临时会话不写回 self._session：外置调用不以整份会话落库。
        """
        session = self._new_session(associate_time, max_group_time)
        session.setdefault("inactive_external_agg_ids", set())
        session.setdefault("conflicted_alarm_ids", set())
        conflicted_group_ids = session.setdefault("conflicted_group_ids", set())
        # 第一遍：统计每个原始组落在哪些汇聚组，跨 >=2 个即组级冲突。
        group_to_aggs = {}
        for agg_id, member_entries in old_agg_alarm_groups.items():
            for member_entry in member_entries or ():
                for group_id in member_entry:
                    group_to_aggs.setdefault(group_id, set()).add(agg_id)
        for group_id, aggs in group_to_aggs.items():
            if len(aggs) >= 2:
                conflicted_group_ids.add(group_id)
        ne_to_site = self.static_context.ne_to_site
        for agg_id, member_entries in old_agg_alarm_groups.items():
            agg_entry = _ensure_agg(session, agg_id)
            agg_last_ts = None
            agg_has_alive_history = False
            for member_entry in member_entries or ():
                for group_id, group_alarms in member_entry.items():
                    group_last_ts, group_has_alive_history = self._rebuild_group_entry(
                        session, agg_id, agg_entry, group_id, group_alarms,
                        matching_cache, ne_to_site, history_horizon_ts,
                    )
                    agg_has_alive_history = (
                        agg_has_alive_history or group_has_alive_history
                    )
                    if group_last_ts is not None and (
                        agg_last_ts is None or group_last_ts > agg_last_ts
                    ):
                        agg_last_ts = group_last_ts
            if agg_last_ts is not None:
                _touch_agg(session, agg_id, agg_last_ts)
            if history_horizon_ts is not None and not agg_has_alive_history:
                session["inactive_external_agg_ids"].add(agg_id)
        return session

    def _rebuild_group_entry(
        self, session, agg_id, agg_entry, group_id, group_alarms,
        matching_cache, ne_to_site, history_horizon_ts,
    ):
        """导入单个成员条目，返回 (最晚告警时间, 是否有视界内告警)。"""
        is_conflicted_group = group_id in session["conflicted_group_ids"]
        group_entry = session["group_registry"].get(group_id)
        if group_entry is None:
            _register_group(session, group_id, None, agg_id)
            agg_entry[0] += 1
        elif group_entry[1] != agg_id:
            # 组级冲突降级：g1 已在先到 agg 挂名，本 agg 也把它算作成员
            # （不撤回、双挂名，组计数照加），保留先到归属、不改挂、不报错。
            # 两遍扫描已把这类组标为 conflicted_group_ids，其告警全部连坐。
            agg_entry[0] += 1
        group_last_ts = None
        group_has_alive_history = False
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
            alarm_is_alive = history_horizon_ts is None or ts >= history_horizon_ts
            group_has_alive_history = group_has_alive_history or alarm_is_alive
            if is_conflicted_group:
                # 冲突组的所有告警连坐 conflicted：不计入任何 agg 告警数、
                # 不喂引擎、不产生连续性拉力；已被别的组计入的回滚一次。
                self._register_conflicted_group_alarm(session, alarm_id, ts)
            else:
                alarm_entry = session["alarm_registry"].get(alarm_id)
                if alarm_entry is None:
                    _register_alarm(session, alarm_id, ts, agg_id)
                    if alarm_is_alive:
                        self._process_matching_alarm(
                            session["engine"], matching_alarm, index_trigger=False
                        )
                    agg_entry[2] += 1
                elif alarm_entry[1] != agg_id:
                    # 告警级冲突：同一告警出现在多个汇聚组下。不整批报错，
                    # 记为冲突告警——注册表保留先遇到的归属，但回滚先到方
                    # 的告警计数（冲突告警谁都不计，计数与遍历序解耦），
                    # 连边阶段挂起其连续性拉力，防止归属随遍历序漂移或扩散。
                    _mark_conflicted_alarm(session, alarm_id, alarm_entry[1])
            if group_last_ts is None or ts > group_last_ts:
                group_last_ts = ts
        if group_last_ts is not None:
            _touch_group(session, group_id, group_last_ts)
        return group_last_ts, group_has_alive_history

    @staticmethod
    def _register_conflicted_group_alarm(session, alarm_id, ts):
        """冲突组的告警连坐 conflicted：无有效归属、不计数、不产生连续性。

        首次出现登记为无主并记入冲突集合；已被别的（正常）组登记并计数
        的，回滚其归属方计数一次（幂等，与告警级冲突同一套）。
        """
        alarm_entry = session["alarm_registry"].get(alarm_id)
        if alarm_entry is None:
            _register_alarm(session, alarm_id, ts, None)
            session.setdefault("conflicted_alarm_ids", set()).add(alarm_id)
        else:
            _mark_conflicted_alarm(session, alarm_id, alarm_entry[1])

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
            # 仅外置重建会话使用：old_agg 中没有任何视界内告警的汇聚组。
            "inactive_external_agg_ids": set(),
            # 组级冲突：同一原始组落在多个汇聚组下（old_agg 自身矛盾或与
            # 内部会话既有归属不一致）。降级容忍——各 agg 仍把它挂名为
            # 成员、归属先到先得，但连续性挂起、其所有告警连坐 conflicted。
            "conflicted_group_ids": set(),
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
        matching_events = BatchFaultGroupMatcher._collect_feed_events(
            event_batches, sort_events, preconverted_events,
            matching_cache, ne_to_site,
        )
        trigger_candidates = []
        for alarm, cache_event in matching_events:
            if upsert_events:
                BatchFaultGroupMatcher._upsert_matching_alarm(
                    engine, alarm, index_trigger, trigger_candidates
                )
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
    def _collect_feed_events(
        event_batches, sort_events, preconverted_events, matching_cache, ne_to_site
    ):
        """构造 (转换后告警, 是否缓存) 事件序列，需要时按 ts 预排序。"""
        if not sort_events and preconverted_events:
            if matching_cache is None:
                raise ValueError("preconverted_events 需要 matching_cache")
            # 调用方已在修改引擎前完成全批转换；此处只按
            # 原有顺序惰性读取，不改变异常原子性。
            return (
                (matching_cache[id(alarm)], cache_event)
                for alarms, cache_event in event_batches
                for alarm in alarms
            )
        matching_events = [
            (_matching_alarm(alarm, matching_cache, ne_to_site), cache_event)
            for alarms, cache_event in event_batches
            for alarm in alarms
        ]
        if sort_events:
            matching_events.sort(key=lambda item: item[0]["ts"])
        return matching_events

    @staticmethod
    def _upsert_matching_alarm(engine, alarm, index_trigger, trigger_candidates):
        """按 eid 幂等覆盖写入持久引擎，累积本批 trigger 候选。"""
        alarm_payload = None
        if alarm.get("extendedattr"):
            alarm_payload = {"extendedattr": alarm["extendedattr"]}
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
        if alarm.get("extendedattr"):
            alarm_payload = {"extendedattr": alarm["extendedattr"]}
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

        升序处理保证消费回收的 cutoff 只清理不晚于本组症状时间的历史，
        不影响更晚的独立故障；每次汇聚成功后立即调用
        _prune_consumed_alarm_history，被消费的 trigger 直接跳过、不再
        重复汇聚。trigger_candidates 是喂流阶段直接返回的本批候选三元组
        (ts, seq, trigger_key)；不在本批输入中的历史事件不发起 trigger，
        但历史告警仍可作为症状证据参与评估。
        """
        trigger_candidates = BatchFaultGroupMatcher._prepare_trigger_candidates(
            trigger_candidates,
            owns_trigger_candidates,
            trigger_candidates_sorted,
        )
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
    def _prepare_trigger_candidates(
        trigger_candidates,
        owns_trigger_candidates,
        trigger_candidates_sorted,
    ):
        """必要时复制并排序 trigger 候选，避免修改调用方容器。"""
        if not owns_trigger_candidates or not isinstance(trigger_candidates, list):
            trigger_candidates = list(trigger_candidates)
        if not trigger_candidates_sorted:
            trigger_candidates.sort()
        return trigger_candidates

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


def _prune_session_horizon(session, batch_min_ts):
    """视界清理：清除 batch_min_ts 之前的引擎事件与过期会话状态。"""
    engine = session["engine"]
    for node in list(engine.event_cache.keys()):
        engine._prune_expired_raw_events_in_place(node, batch_min_ts)
    _prune_session_history(session, batch_min_ts - session["max_stay_sec"])


def _make_matching_alarm_resolver(
    current_matching_alarms_by_id, matching_cache, ne_to_site
):
    """优先复用本批已转换告警的解析器：外置条目与本批重叠时避免重复转换。"""
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
    return resolve_matching_alarm


def _register_current_batch_groups(session, current_alarm_records_by_group):
    """登记本批原始组/告警（归属保持 None），并刷新组的最近告警时间。"""
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


def _exclude_conflicts_from_imported_counts(
    agg_output, imported_agg_counts, excluded_alarm_ids, resolve_matching_alarm
):
    """从临时会话计数中扣除写回持久会话时新发现的冲突告警。

    imported_agg_counts 已按告警 ID 去重，因此同一冲突告警在一个
    汇聚组的多个成员条目中重复出现时也只扣一次。原始组计数不受
    告警冲突影响，原样保留。
    """
    adjusted_counts = dict(imported_agg_counts)
    for agg_id, member_entries in agg_output.items():
        excluded_in_agg = set()
        for member_entry in member_entries:
            for group_alarms in member_entry.values():
                for generated_alarm in group_alarms:
                    alarm_id = resolve_matching_alarm(generated_alarm)["alarm_id"]
                    if alarm_id in excluded_alarm_ids:
                        excluded_in_agg.add(alarm_id)
        if not excluded_in_agg:
            continue
        group_count, alarm_count = adjusted_counts[agg_id]
        adjusted_counts[agg_id] = (
            group_count,
            alarm_count - len(excluded_in_agg),
        )
    return adjusted_counts


def _overwrite_agg_counts(
    session, agg_output, imported_agg_counts, imported_group_ids, imported_alarm_ids
):
    """用唯一 ID 计数覆盖汇聚组画像，保证重复导入幂等。

    直接覆盖临时会话已按唯一 ID 算好的计数，避免
    O(|group_registry| + |alarm_registry|) 的内部全表扫描。
    """
    if imported_agg_counts is None:
        conflicted_alarm_ids = session.get("conflicted_alarm_ids") or frozenset()
        imported_agg_counts = {
            agg_id: (
                len(imported_group_ids[agg_id]),
                len(imported_alarm_ids[agg_id] - conflicted_alarm_ids),
            )
            for agg_id in agg_output
        }
    for agg_id in agg_output:
        agg_entry = session["agg_registry"].get(agg_id)
        if agg_entry is not None:
            agg_entry[0], agg_entry[2] = imported_agg_counts[agg_id]


def _prune_session_history(session, horizon_ts):
    """按三个最小堆增量回收视界之前的会话状态。"""
    indexes = session["history_expiry_indexes"]

    alarm_registry = session["alarm_registry"]
    conflicted_alarm_ids = session.get("conflicted_alarm_ids")
    alarm_heap = indexes["alarm_heap"]
    while alarm_heap and alarm_heap[0][0] < horizon_ts:
        ts, _seq, alarm_id = heapq.heappop(alarm_heap)
        entry = alarm_registry.get(alarm_id)
        if entry is not None and entry[0] == ts:
            del alarm_registry[alarm_id]
            if conflicted_alarm_ids:
                conflicted_alarm_ids.discard(alarm_id)

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


def _mark_conflicted_alarm(session, alarm_id, owner_agg_id):
    """把告警记为归属冲突；仅首次标记时回滚既有归属方的告警计数。

    冲突告警不计入任何汇聚组的告警数，附着目标按告警数排序时不再受
    old_agg 遍历序影响。重复标记（同一告警挂三个以上汇聚组、或跨调用
    再次发现）幂等，不重复回滚。
    """
    conflicted_alarm_ids = session.setdefault("conflicted_alarm_ids", set())
    if alarm_id in conflicted_alarm_ids:
        return
    conflicted_alarm_ids.add(alarm_id)
    owner_entry = session["agg_registry"].get(owner_agg_id)
    if owner_entry is not None:
        owner_entry[2] -= 1


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
    conflicted_group_ids = session.get("conflicted_group_ids")
    if conflicted_group_ids:
        conflicted_group_ids.discard(group_id)
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


def _batch_member_order_key(batch):
    """成员组排序键：按本批最早告警时间升序，缺失时间的组排最后。"""
    def member_order_key(group_id):
        min_ts = batch.local_group_min_ts.get(group_id)
        return (min_ts is None, min_ts if min_ts is not None else 0.0, str(group_id))
    return member_order_key


def _group_by_rigid_cluster(members, rigid_union):
    """按共享告警刚性簇给成员分组：同簇的原始组必须进同一汇聚组。

    rigid_union 只并合“共享同一条告警”的原始组（不含症状匹配边），因此
    返回的每个簇是必须整体同组的最小不可分单元。按成员首次出现顺序聚簇，
    簇内保持传入顺序（调用方需要时另行排序）。
    """
    clusters_by_root = {}
    for group_id in members:
        clusters_by_root.setdefault(
            rigid_union.find(group_id), []
        ).append(group_id)
    return list(clusters_by_root.values())


def _pack_clusters_by_cap(clusters, max_member_count):
    """把刚性簇按原始组个数上限做 next-fit 装箱，返回成员列表的列表。

    clusters 需已按时间排序，每个簇不可分。上限对能装下的簇严格生效
    （多个小簇可拼进同一个包）；单个簇的组数超过上限时独占一个超限包、
    不被拆开——这是“共享告警必同组”对 max_group_member 的唯一破例。
    next-fit（按时间顺序、不回头填空）保证同包内的组发生时间连续，且
    历史前缀不因后来的簇而重排。
    """
    cap = max(1, max_member_count)
    packs = []
    current = []
    for cluster in clusters:
        if current and len(current) + len(cluster) > cap:
            packs.append(current)
            current = []
        current.extend(cluster)
    if current:
        packs.append(current)
    return packs


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


def _index_current_alarms(raw_alarm_groups, matching_cache, ne_to_site):
    """索引本批原始告警：按组的 (原始, 转换) 记录 + 按告警 ID 的两份映射。"""
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
    return (
        current_alarm_records_by_group,
        current_matching_alarms_by_id,
        current_generated_alarms_by_id,
    )


def _validate_external_increment(
    session, agg_output, resolve_matching_alarm, need_id_sets
):
    """先做全量归属校验（组级/告警级冲突都降级为标记、不报错）再修改
    会话，避免中途冲突留下半写状态。

    组级冲突（同一原始组落在多个汇聚组，或与内部会话既有归属不一致）记入
    conflicted_group_ids：各 agg 仍把它挂名为成员、归属先到先得，其所有
    告警连坐 conflicted。告警级冲突记入 conflicted_alarm_ids。

    agg_output 是完整外置快照。正常调用直接复用临时会话重建时已经得到的
    唯一成员计数（need_id_sets=False，返回 (None, None)）；私有方法被单独
    调用时才在本次必经遍历中构造去重集合兜底。
    """
    imported_group_ids = (
        {agg_id: set() for agg_id in agg_output} if need_id_sets else None
    )
    imported_alarm_ids = (
        {agg_id: set() for agg_id in agg_output} if need_id_sets else None
    )
    conflicted_group_ids = session.setdefault("conflicted_group_ids", set())
    # 两遍：先定位组级冲突（增量自身跨多个 agg，或与内部会话既有归属不一致）。
    group_to_aggs = {}
    for agg_id, member_entries in agg_output.items():
        for member_entry in member_entries:
            for group_id in member_entry:
                aggs = group_to_aggs.setdefault(group_id, set())
                aggs.add(agg_id)
                internal_entry = session["group_registry"].get(group_id)
                if internal_entry is not None and internal_entry[1] is not None:
                    aggs.add(internal_entry[1])
    for group_id, aggs in group_to_aggs.items():
        if len(aggs) >= 2:
            conflicted_group_ids.add(group_id)
    for agg_id, member_entries in agg_output.items():
        for member_entry in member_entries:
            for group_id, group_alarms in member_entry.items():
                if imported_group_ids is not None:
                    imported_group_ids[agg_id].add(group_id)
                _validate_external_alarm_owners(
                    session, agg_id, group_alarms, resolve_matching_alarm,
                    imported_alarm_ids,
                    group_conflicted=group_id in conflicted_group_ids,
                )
    return imported_group_ids, imported_alarm_ids


def _validate_external_alarm_owners(
    session, agg_id, group_alarms, resolve_matching_alarm, imported_alarm_ids,
    group_conflicted=False,
):
    """校验条目内各告警在内部会话中的既有归属，冲突则标记不报错。

    group_conflicted=True（所在组是冲突组）时，该条目所有告警一律连坐
    conflicted，与它当前是否已有归属无关。否则仅在归属与 agg_id 不一致时
    记入 conflicted_alarm_ids。两种标记都会回滚既有归属方的告警计数（冲突
    告警谁都不计），连边阶段挂起其连续性效力。标记与回滚幂等。
    """
    for generated_alarm in group_alarms:
        matching_alarm = resolve_matching_alarm(generated_alarm)
        alarm_id = matching_alarm["alarm_id"]
        if imported_alarm_ids is not None:
            imported_alarm_ids[agg_id].add(alarm_id)
        alarm_entry = session["alarm_registry"].get(alarm_id)
        if group_conflicted:
            _mark_conflicted_alarm(
                session, alarm_id,
                alarm_entry[1] if alarm_entry is not None else None,
            )
        elif (
            alarm_entry is not None
            and alarm_entry[1] is not None
            and alarm_entry[1] != agg_id
        ):
            _mark_conflicted_alarm(session, alarm_id, alarm_entry[1])


def _import_external_entries(session, agg_output, resolve_matching_alarm):
    """把外置快照/增量条目写入会话注册表（归属、计数、最晚时间）。

    冲突组（conflicted_group_ids）各 agg 挂名 +组计数，但保留先到归属、
    不改挂；其所有告警连坐 conflicted、不计入告警数。计数最终由
    _overwrite_agg_counts 按去重集合覆盖，这里的增量只服务最晚时间维护。
    """
    conflicted_group_ids = session.get("conflicted_group_ids") or frozenset()
    for agg_id, member_entries in agg_output.items():
        agg_entry = _ensure_agg(session, agg_id)
        imported_agg_last_ts = agg_entry[1]
        for member_entry in member_entries:
            for group_id, group_alarms in member_entry.items():
                group_conflicted = group_id in conflicted_group_ids
                group_entry = session["group_registry"].get(group_id)
                if group_entry is None:
                    group_entry = _register_group(session, group_id, None, None)
                if group_conflicted:
                    # 挂名：本 agg 也把 g1 算作成员，但归属保持先到、不改挂。
                    agg_entry[0] += 1
                    if group_entry[1] is None:
                        _set_group_owner(session, group_id, agg_id)
                elif group_entry[1] != agg_id:
                    agg_entry[0] += 1
                    _set_group_owner(session, group_id, agg_id)
                imported_group_last_ts = group_entry[0]
                for generated_alarm in group_alarms:
                    matching_alarm = resolve_matching_alarm(generated_alarm)
                    alarm_id = matching_alarm["alarm_id"]
                    ts = matching_alarm["ts"]
                    alarm_entry = session["alarm_registry"].get(alarm_id)
                    if group_conflicted:
                        # 冲突组告警连坐：登记无主、记入冲突集合、不计数。
                        if alarm_entry is None:
                            _register_alarm(session, alarm_id, ts, None)
                        session.setdefault("conflicted_alarm_ids", set()).add(
                            alarm_id
                        )
                    elif alarm_entry is None:
                        _register_alarm(session, alarm_id, ts, agg_id)
                        agg_entry[2] += 1
                    elif alarm_entry[1] is None:
                        # 告警可能在此前外置调用中作为未汇聚原始告警写入；
                        # 本次真正汇聚时补齐归属，但不重复喂引擎。
                        alarm_entry[1] = agg_id
                        agg_entry[2] += 1
                    if imported_agg_last_ts is None or ts > imported_agg_last_ts:
                        imported_agg_last_ts = ts
                    if imported_group_last_ts is None or ts > imported_group_last_ts:
                        imported_group_last_ts = ts
                _touch_group(session, group_id, imported_group_last_ts)
        if imported_agg_last_ts is not None:
            _touch_agg(session, agg_id, imported_agg_last_ts)
