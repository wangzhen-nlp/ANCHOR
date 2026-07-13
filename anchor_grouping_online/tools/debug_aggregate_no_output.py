#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 BatchFaultGroupMatcher.aggregate_alarm_groups 为什么没有输出。

把一次真实调用的入参放进一个参数文件（JSON 或 Python 模块均可），
本脚本用带插桩的 matcher 重放这次调用，输出各阶段的漏斗统计，
指出结果是在哪个阶段变空的，并针对该阶段给出排查提示。

用法：
    python -m anchor_grouping_online.tools.debug_aggregate_no_output params.py
    python -m anchor_grouping_online.tools.debug_aggregate_no_output params.json \
        [--show-alarms] [--show-matches N]

参数文件字段（JSON 顶层 key / Python 模块同名变量）：
    alarm_groups          必填，{故障组id: [告警, ...]}（告警字段见
                          match_rules_batch 模块 docstring）
    associate_time        必填，关联时间窗（分钟）
    max_group_time        必填，组保留时长（分钟）
    max_group_member      必填，汇聚组最大原始组个数
    old_agg_alarm_groups  可选，外置状态模式的既有二次汇聚故障组
    batch_isolated        可选，bool，默认 False
    full_output           可选，bool，默认 False
    resource_buffer       可选，资源缓冲 jsonl 路径；缺省用包内默认路径

漏斗阶段（任一阶段归零，后面必然全空）：
    [1]  输入规模          原始组数 / 告警条数
    [2]  告警转换          清除/无站点告警（alarm_source 不在 ne_to_site）、
                           组内重复与跨组共享告警
    [3]  trigger 候选      告警名/站点是否命中任何规则的 trigger
    [4]  规则评估          按规则统计尝试/命中次数，raw match 总数
    [5]  批内合并          merge_match_batch 去重合并
    [6]  历史合并          与既有已输出组的历史合并
    [7]  可见性过滤        输出可见性（无新增症状的组被吃掉）
    [8]  二次汇聚规则过滤  merged_rules 需与 output_eligible 集合有交集
    [9]  故障模式过滤      fault_pattern_filter.analyze_match
    [10] 组间关联          匹配超边 / 历史汇聚组关联 / 组ID·告警归属连续性
    [11] 汇聚组分配        每个组的分配路径：沿用 / 附着 / 新建 / 未分配
    [12] 增量与变化过滤    汇聚组成员数 <2 被滤掉；纯重发不输出
    [13] 最终输出          存活变化组 + （外置模式）失活组
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

if __package__ in (None, ""):
    from _script_env import ensure_package_parent  # type: ignore

    ensure_package_parent()

from anchor_grouping_online.alarm_events.generator import to_matching_alarm
from anchor_grouping_online.match_rules_batch import (
    BatchFaultGroupMatcher,
    _collect_symptom_alarm_ids,
)
from anchor_grouping_online.matching.fault_pattern_analysis import (
    MAX_ANALYSIS_SITES,
    classify_component,
    extract_offline_sites,
    prepare_case_record,
)
from anchor_grouping_online.temporal_engine.utils import (
    matches_expected_alarm,
    merge_match_batch,
)
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


# ---------------------------------------------------------------------------
# 参数文件加载
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = (
    "alarm_groups", "associate_time", "max_group_time", "max_group_member",
)
_OPTIONAL_FIELDS = (
    "old_agg_alarm_groups", "batch_isolated", "full_output", "resource_buffer",
)


def load_params(path):
    """从 JSON 文件或 Python 模块读取 aggregate_alarm_groups 的入参。"""
    if path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_debug_params", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = {name: getattr(module, name) for name in dir(module)}
    else:
        with open(path, "r", encoding="utf-8") as file_obj:
            source = json.load(file_obj)
    missing = [name for name in _REQUIRED_FIELDS if name not in source]
    if missing:
        raise SystemExit(f"参数文件缺少必填字段: {', '.join(missing)}")
    params = {name: source[name] for name in _REQUIRED_FIELDS}
    for name in _OPTIONAL_FIELDS:
        if name in source:
            params[name] = source[name]
    return params


# ---------------------------------------------------------------------------
# 插桩 matcher：只记录中间结果，不改变任何行为
# ---------------------------------------------------------------------------

class DebugBatchFaultGroupMatcher(BatchFaultGroupMatcher):
    """在关键阶段记录中间结果的 matcher，行为与父类完全一致。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = SimpleNamespace(
            fault_pattern_filter=self.fault_pattern_filter,
            session=None,
            engine=None,
            batch=None,
            trigger_candidates=[],
            evaluations=[],           # (站点, 规则名, ts, 结果数)
            raw_matches=[],
            merged_matches=[],        # 批内合并后
            finalized_matches=[],     # 历史合并后
            visible_matches=[],       # 可见性过滤后
            dropped_by_rule_filter=[],
            dropped_by_pattern_filter=[],
            eligible_matches=[],
            preexisting_group_owners={},   # 组ID连续性沿用来源
            preexisting_alarm_owners={},   # 告警归属连续性沿用来源
            links=None,
            assignments=[],           # (组ID, 汇聚组ID, 路径: 沿用/附着/新建)
            member_entries_by_agg={},
            changed_agg_ids=set(),
            increment_by_agg={},
        )

    def _register_batch_input(self, session, engine, alarm_groups,
                              matching_cache, ne_to_site):
        batch = super()._register_batch_input(
            session, engine, alarm_groups, matching_cache, ne_to_site
        )
        self.debug.session = session
        self.debug.engine = engine
        self.debug.batch = batch
        self._record_rule_evaluations(engine)
        return batch

    def _record_rule_evaluations(self, engine):
        """包一层引擎的规则评估入口，记录每次 (站点, 规则) 的评估结果数。"""
        original_evaluate = engine._evaluate_rule

        def recording_evaluate(rule_name, rule, trigger_node, event_ts,
                               *args, **kwargs):
            results = original_evaluate(
                rule_name, rule, trigger_node, event_ts, *args, **kwargs
            )
            self.debug.evaluations.append(
                (trigger_node, rule_name, event_ts, len(results or ()))
            )
            return results

        engine._evaluate_rule = recording_evaluate

    def _feed_engine(self, engine, generated_alarms, **kwargs):
        candidates = BatchFaultGroupMatcher._feed_engine(
            engine, generated_alarms, **kwargs
        )
        # 外置同步阶段 index_trigger=False 返回空列表，不覆盖主喂流结果。
        if kwargs.get("index_trigger", True):
            self.debug.trigger_candidates = list(candidates)
        return candidates

    def _aggregate_per_trigger(self, engine, trigger_candidates, **kwargs):
        raw_matches = BatchFaultGroupMatcher._aggregate_per_trigger(
            engine, trigger_candidates, **kwargs
        )
        self.debug.raw_matches = list(raw_matches)
        return raw_matches

    def _merge_batch_and_finalize(self, engine, raw_matches):
        # 与父类逻辑一致，拆开三步分别记录（父类实现见
        # BatchFaultGroupMatcher._merge_batch_and_finalize）。
        if not raw_matches:
            return []
        merged = merge_match_batch(raw_matches)
        self.debug.merged_matches = list(merged)
        finalized = engine._finalize_matches_with_history(merged)
        self.debug.finalized_matches = list(finalized)
        visible = engine._apply_output_visibility_filters_to_matches(finalized)
        self.debug.visible_matches = list(visible)
        return visible

    def _iter_output_matches(self, matches):
        # 与父类逻辑一致，只是分别记录两道过滤各自砍掉的 match。
        for match in matches:
            if not self._match_is_output_eligible(match):
                self.debug.dropped_by_rule_filter.append(match)
                continue
            if (
                self.fault_pattern_filter is not None
                and self.fault_pattern_filter.analyze_match(match) is None
            ):
                self.debug.dropped_by_pattern_filter.append(match)
                continue
            self.debug.eligible_matches.append(match)
            yield match

    def _link_batch_groups(self, session, alarm_groups, batch, output_matches):
        # 快照两类"连续性沿用"的来源，供 [10] 区分关联证据从哪来。
        self.debug.preexisting_group_owners = {
            group_id: entry[1]
            for group_id, entry in session["group_registry"].items()
            if group_id in alarm_groups and entry[1] is not None
        }
        self.debug.preexisting_alarm_owners = {
            alarm_id: entry[1]
            for group_id, alarm_ids in batch.local_group_alarm_ids.items()
            for alarm_id in alarm_ids
            for entry in (session["alarm_registry"].get(alarm_id),)
            if entry is not None and entry[1] is not None
        }
        links = super()._link_batch_groups(
            session, alarm_groups, batch, output_matches
        )
        self.debug.links = links
        return links

    def _assign_members_to_agg(self, session, agg_id, member_group_ids, batch,
                               batch_member_entries_by_agg, changed_agg_ids):
        agg_entry = session["agg_registry"].get(agg_id)
        agg_existed = agg_entry is not None and agg_entry[0] > 0
        paths = []
        for group_id in member_group_ids:
            previous_owner = session["group_registry"][group_id][1]
            if previous_owner == agg_id:
                kind = "沿用"
            elif agg_existed:
                kind = "附着"
            else:
                kind = "新建"
            paths.append((group_id, agg_id, kind))
        super()._assign_members_to_agg(
            session, agg_id, member_group_ids, batch,
            batch_member_entries_by_agg, changed_agg_ids,
        )
        self.debug.assignments.extend(paths)

    def _assign_components(self, session, alarm_groups, batch, links,
                           max_member_count):
        member_entries, changed = super()._assign_components(
            session, alarm_groups, batch, links, max_member_count
        )
        self.debug.member_entries_by_agg = member_entries
        self.debug.changed_agg_ids = set(changed)
        return member_entries, changed

    def _build_batch_increments(self, session, batch_member_entries_by_agg,
                                changed_agg_ids, need_full_increment,
                                matching_cache, ne_to_site):
        increments = BatchFaultGroupMatcher._build_batch_increments(
            session, batch_member_entries_by_agg, changed_agg_ids,
            need_full_increment, matching_cache, ne_to_site,
        )
        self.debug.increment_by_agg = increments
        return increments


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def _fmt_ts(ts):
    if not isinstance(ts, (int, float)):
        return str(ts)
    try:
        return f"{ts:.3f} ({datetime.fromtimestamp(ts):%Y-%m-%d %H:%M:%S})"
    except (OverflowError, OSError, ValueError):
        return f"{ts:.3f}"


def _convert_all_alarms(alarm_groups, ne_to_site):
    """逐条转换输入告警，返回 [(组id, 原始告警, 转换结果或异常文本)]。"""
    rows = []
    for group_id, group_alarms in (alarm_groups or {}).items():
        for generated_alarm in group_alarms or ():
            try:
                converted = to_matching_alarm(generated_alarm, ne_to_site)
            except Exception as exc:  # noqa: BLE001 诊断场景要兜住一切
                rows.append((group_id, generated_alarm, f"转换失败: {exc}"))
            else:
                rows.append((group_id, generated_alarm, converted))
    return rows


def report_input(alarm_groups, params):
    print("=" * 72)
    print("[1] 输入规模")
    total = sum(len(alarms or ()) for alarms in alarm_groups.values())
    print(f"  原始故障组: {len(alarm_groups)} 个, 告警共 {total} 条")
    print(
        "  associate_time={}min  max_group_time={}min  max_group_member={}".format(
            params["associate_time"], params["max_group_time"],
            params["max_group_member"],
        )
    )
    old_agg = params.get("old_agg_alarm_groups")
    print(
        "  old_agg_alarm_groups: "
        + (f"{len(old_agg)} 个汇聚组" if old_agg else str(old_agg))
        + f"  batch_isolated={params.get('batch_isolated', False)}"
        + f"  full_output={params.get('full_output', False)}"
    )
    if len(alarm_groups) < 2 and not old_agg:
        print("  !! 只有 1 个原始组且无 old_agg：汇聚组至少要 2 个原始组，"
              "本次调用不可能有输出")


def report_conversion(rows, show_alarms):
    print("=" * 72)
    print("[2] 告警转换 (to_matching_alarm)")
    failed = [r for r in rows if isinstance(r[2], str)]
    converted = [r for r in rows if not isinstance(r[2], str)]
    cleared = [r for r in converted if r[2]["is_clear"]]
    no_site = [r for r in converted if not r[2]["site_id"] and not r[2]["is_clear"]]
    ts_values = [r[2]["ts"] for r in converted]
    groups_by_alarm = {}
    duplicate_in_group = 0
    seen_in_group = set()
    for group_id, _alarm, conv in converted:
        alarm_id = conv["alarm_id"]
        if (group_id, alarm_id) in seen_in_group:
            duplicate_in_group += 1
        seen_in_group.add((group_id, alarm_id))
        groups_by_alarm.setdefault(alarm_id, set()).add(group_id)
    shared_alarms = {
        alarm_id: owners for alarm_id, owners in groups_by_alarm.items()
        if len(owners) > 1
    }
    print(f"  转换成功 {len(converted)} / 失败 {len(failed)}；"
          f"清除告警 {len(cleared)} 条；无站点告警 {len(no_site)} 条")
    print(f"  组内重复 vid {duplicate_in_group} 条（登记时去重）；"
          f"跨组共享告警 {len(shared_alarms)} 个"
          "（仅当共享告警已归属某汇聚组时，各组才沿用该归属）")
    for alarm_id, owners in list(shared_alarms.items())[:5]:
        print(f"  - 共享告警 {alarm_id!r}: 属于组 {sorted(map(str, owners))}")
    if ts_values:
        print(f"  批内时间范围: {_fmt_ts(min(ts_values))} ~ {_fmt_ts(max(ts_values))}"
              f"  (跨度 {max(ts_values) - min(ts_values):.1f}s)")
    for group_id, alarm, error in failed:
        print(f"  !! 组 {group_id!r} 告警 {alarm.get('vid')!r} {error}")
    if no_site:
        print("  !! 以下告警的 alarm_source(neVid/ownerVid) 在 ne_to_site 查不到"
              "站点，匹配不上任何拓扑规则:")
        for group_id, alarm, conv in no_site[:20]:
            src = conv["alarm_source"]
            print(f"     组 {group_id!r}  vid={conv['alarm_id']!r}  "
                  f"alarm_source={src!r}")
        if len(no_site) > 20:
            print(f"     ... 共 {len(no_site)} 条")
    if show_alarms:
        for group_id, alarm, conv in converted:
            if isinstance(conv, str):
                continue
            print(f"  组 {group_id!r}  {conv['alarm_title']!r}  "
                  f"vid={conv['alarm_id']!r}  site={conv['site_id']!r}  "
                  f"source={conv['alarm_source']!r}  "
                  f"is_clear={conv['is_clear']}  ts={_fmt_ts(conv['ts'])}")
    return converted, no_site


def _flatten_expected_alarm_names(specs):
    """把 trigger expected 条目（字符串或 required/optional 集合）摊平成告警名集合。"""
    names = set()
    for _rule_name, expected_list in specs:
        for expected in expected_list:
            if isinstance(expected, dict):
                for key in ("required_alarms", "optional_alarms"):
                    names.update(str(name) for name in expected.get(key) or ())
            else:
                names.add(str(expected))
    return names


def _diagnose_zero_triggers(engine, converted_rows):
    """trigger 候选为 0 时，逐告警解释没命中 trigger 的原因。"""
    print("  逐告警 trigger 诊断:")
    ne_to_site_hint_shown = False
    for group_id, _alarm, conv in converted_rows:
        if conv["is_clear"]:
            reason = "清除告警不建立 trigger"
        elif not conv["site_id"]:
            reason = "无站点（alarm_source 不在 ne_to_site）"
            upper = str(conv["alarm_source"]).strip().upper()
            if not ne_to_site_hint_shown and upper != conv["alarm_source"]:
                reason += "；注意 generate_alarm 会把 neVid 大写化，" \
                          "自行构造的告警需保持与拓扑一致的大小写"
                ne_to_site_hint_shown = True
        else:
            specs = engine.trigger_specs_by_node.get(conv["site_id"], ())
            if not specs:
                reason = (f"站点 {conv['site_id']!r} 没有任何规则的 trigger "
                          "（站点域与所有规则的 trigger 节点域都不匹配）")
            else:
                domain = engine.alarm_source_domain_map.get(
                    conv["alarm_source"], ""
                )
                hit = [
                    rule_name for rule_name, expected_list in specs
                    if any(
                        matches_expected_alarm(conv["alarm_title"], e, domain)
                        for e in expected_list
                    )
                ]
                if hit:
                    reason = f"命中规则 {hit}（不应出现在 0-trigger 分支）"
                else:
                    expected_names = _flatten_expected_alarm_names(specs)
                    reason = (f"告警名 {conv['alarm_title']!r} 不在站点可触发"
                              f"告警集合中（共 {len(expected_names)} 种），"
                              f"示例: {sorted(expected_names)[:6]}...")
        print(f"    组 {group_id!r}  {conv['alarm_title']!r}"
              f"@{conv['site_id'] or '-'}: {reason}")


def _diagnose_pattern_filter_drop(fault_pattern_filter, match, indent="    "):
    """复算 FaultPatternFilter.analyze_match 的各步骤，指出在哪一步被拒绝。

    与 analyze_match 的差异仅在 component_limit=None（枚举全部投影分量，
    便于展示），不影响判定结论。依赖过滤器内部字段，仅用于诊断展示。
    """
    if fault_pattern_filter is None:
        print(f"{indent}(未启用故障模式过滤器，无法复算)")
        return
    try:
        site_ids = fault_pattern_filter._extract_match_sites(match)
        ne_to_site = fault_pattern_filter._ne_to_site
        relation_index = fault_pattern_filter._relation_index
        offline_sites = sorted(
            extract_offline_sites(match, ne_to_site) & set(site_ids)
        )
        print(f"{indent}症状站点 {len(site_ids)} 个: {site_ids[:10]}"
              + ("..." if len(site_ids) > 10 else "")
              + f"；其中 offline 站点: {offline_sites[:10]}")
        if len(site_ids) > MAX_ANALYSIS_SITES:
            print(f"{indent}!! 拒绝原因: 站点数超过分析上限 "
                  f"MAX_ANALYSIS_SITES={MAX_ANALYSIS_SITES}")
            return
        prepared = prepare_case_record(
            match,
            relation_index,
            ne_to_site,
            fault_pattern_filter._site_has_router_device,
            site_ids=site_ids,
            component_limit=None,
        )
        if prepared.router_device_sites:
            print(f"{indent}路由设备站点: "
                  f"{sorted(prepared.router_device_sites)[:10]}")
        if prepared.absorbed_by:
            shown = dict(list(prepared.absorbed_by.items())[:5])
            print(f"{indent}被吸收的无管理下游站点(站点->吸收者): {shown}")
        components = prepared.projected_components
        if len(components) != 1:
            print(f"{indent}!! 拒绝原因: one-component-only 不通过——"
                  f"站点在拓扑投影上形成 {len(components)} 个连通分量"
                  "（各分量之间无站点链邻接关系）:")
            for component_sites in list(components)[:5]:
                print(f"{indent}  - 分量: {sorted(component_sites)}")
            return
        patterns = [
            classify_component(
                component_sites,
                prepared.active_unmanaged_sites,
                relation_index,
                router_device_sites=prepared.router_device_sites,
                absorbed_by=prepared.absorbed_by,
            )
            for component_sites in components
        ]
        print(f"{indent}!! 拒绝原因: filter-others 不通过——分量分类结果为 "
              f"{patterns}，只有 unknown/ip_ring_others 之外的模式才可参与"
              "二次汇聚")
    except Exception as exc:  # noqa: BLE001 诊断复算失败不影响主流程
        print(f"{indent}(复算故障模式过滤细节失败: {exc})")


def _print_match_symptoms(match, indent="      ", limit=12):
    """打印单个 match 的症状明细：告警名 / alarm_source / 站点 / 角色 / 时间。"""
    symptoms = list(match.get("symptoms", ()) or ())
    for symptom in symptoms[:limit]:
        print(f"{indent}告警 {symptom.get('alarm')!r}  "
              f"源 {symptom.get('alarm_source')!r}  "
              f"站点 {symptom.get('node')!r}  "
              f"角色 {symptom.get('matched_role')}  "
              f"ts={_fmt_ts(symptom.get('ts'))}  "
              f"eid={symptom.get('eid')!r}")
    if len(symptoms) > limit:
        print(f"{indent}... 症状共 {len(symptoms)} 条，仅显示前 {limit} 条")


def report_matches(debug, converted_rows, show_matches):
    if debug.session is None:
        print("=" * 72)
        print("!! 调用在会话/引擎构建阶段就失败了，未进入喂流；"
              "请先解决上面的异常（常见于资源缓冲不完整）")
        return
    print("=" * 72)
    print(f"[3] trigger 候选: {len(debug.trigger_candidates)} 个")
    if not debug.trigger_candidates:
        _diagnose_zero_triggers(debug.engine, converted_rows)
        return
    print("=" * 72)
    evaluated = len(debug.evaluations)
    skipped = len(debug.trigger_candidates) - evaluated
    print(f"[4] 规则评估: 实际评估 {evaluated} 次"
          f"（{skipped} 个候选因已被更早的汇聚消费而跳过），"
          f"raw_matches 共 {len(debug.raw_matches)} 个")
    per_rule = {}
    for _node, rule_name, _ts, n_results in debug.evaluations:
        attempts, hits = per_rule.get(rule_name, (0, 0))
        per_rule[rule_name] = (attempts + 1, hits + (1 if n_results else 0))
    for rule_name, (attempts, hits) in sorted(per_rule.items()):
        print(f"  - {rule_name}: 评估 {attempts} 次, 命中 {hits} 次")
    if debug.raw_matches:
        print("  命中明细:")
        for i, match in enumerate(debug.raw_matches[:10]):
            print(f"  - raw match#{i} 规则 {match.get('merged_rules')}:")
            _print_match_symptoms(match)
        if len(debug.raw_matches) > 10:
            print(f"  ... raw match 共 {len(debug.raw_matches)} 个，"
                  "仅显示前 10 个")
    if not debug.raw_matches:
        for node, rule_name, ts, _n in debug.evaluations[:10]:
            print(f"  - 评估无结果: 站点 {node!r} 规则 {rule_name} "
                  f"ts={_fmt_ts(ts)}")
        print("  有 trigger 但规则评估为空，常见原因:")
        print("  - 症状告警不齐：规则要求的相邻/下挂节点告警缺失，"
              "或对应网元不在拓扑邻接关系里")
        print("  - associate_time 太小：症状之间 ts 差超过关联窗"
              f"（当前 {debug.session['associate_window_sec']:.0f}s）")
        print("  - 症状告警是清除告警或已被本批更早的汇聚消费")
        return
    print("=" * 72)
    print(f"[5] 批内合并 merge_match_batch: "
          f"{len(debug.raw_matches)} -> {len(debug.merged_matches)} 个")
    print("=" * 72)
    print(f"[6] 历史合并: "
          f"{len(debug.merged_matches)} -> {len(debug.finalized_matches)} 个")
    if not debug.finalized_matches:
        print("  !! 全部被历史合并吃掉（与既有已输出组重叠）")
        return
    print("=" * 72)
    print(f"[7] 输出可见性过滤: "
          f"{len(debug.finalized_matches)} -> {len(debug.visible_matches)} 个")
    if not debug.visible_matches:
        print("  !! 全部被可见性过滤吃掉（相对既有输出无新增症状，纯重发）")
        return
    print("=" * 72)
    after_rule_filter = (
        len(debug.visible_matches) - len(debug.dropped_by_rule_filter)
    )
    print(f"[8] 二次汇聚规则过滤: "
          f"{len(debug.visible_matches)} -> {after_rule_filter} 个")
    for match in debug.dropped_by_rule_filter[:5]:
        print(f"  - 丢弃: merged_rules={match.get('merged_rules')}"
              "（与 output_eligible 规则集合无交集）")
    if not after_rule_filter:
        print("  !! 全部被规则过滤吃掉：匹配组不含任何可参与二次汇聚的规则")
        return
    print("=" * 72)
    print(f"[9] 故障模式过滤: "
          f"{after_rule_filter} -> {len(debug.eligible_matches)} 个")
    for match in debug.dropped_by_pattern_filter[:5]:
        print(f"  - 丢弃: merged_rules={match.get('merged_rules')} "
              f"症状 {len(match.get('symptoms', ()))} 条:")
        _print_match_symptoms(match, indent="    ", limit=8)
        _diagnose_pattern_filter_drop(debug.fault_pattern_filter, match)
    if len(debug.dropped_by_pattern_filter) > 5:
        print(f"  ... 被故障模式过滤丢弃的 match 共 "
              f"{len(debug.dropped_by_pattern_filter)} 个，仅展开前 5 个")
    if not debug.eligible_matches:
        print("  !! 全部被故障模式过滤吃掉")
        return
    if show_matches:
        print("  通过全部过滤的匹配组明细:")
        for i, match in enumerate(debug.eligible_matches[:show_matches]):
            print(f"  - match#{i} 规则 {match.get('merged_rules')}:")
            _print_match_symptoms(match)


def report_linking(debug, alarm_groups):
    if debug.links is None:
        return
    print("=" * 72)
    local_union, linked_agg_ids_by_group, current_group_edges = debug.links
    print(f"[10] 组间关联: 匹配产生的组间超边 {len(current_group_edges)} 条")
    for edge in current_group_edges[:10]:
        print(f"  - 边: {edge}")
    linked_any = {g: aggs for g, aggs in linked_agg_ids_by_group.items() if aggs}
    if linked_any:
        print(f"  关联到历史汇聚组的原始组: { {k: v for k, v in list(linked_any.items())[:10]} }")
    if debug.preexisting_group_owners:
        shown = dict(list(debug.preexisting_group_owners.items())[:10])
        print(f"  组ID连续性沿用（组曾归属过汇聚组）: {shown}")
    if debug.preexisting_alarm_owners:
        print(f"  告警归属连续性沿用: {len(debug.preexisting_alarm_owners)} 条"
              "告警已有汇聚组归属")
    if debug.eligible_matches and not current_group_edges and not linked_any:
        print("  !! 有通过过滤的匹配组，但没有形成任何组间关联——"
              "每个 match 的症状告警都只落在同一个原始组里，或症状告警的 "
              "eid 不在本批 alarm_groups 的告警 vid 中")
        for i, match in enumerate(debug.eligible_matches[:5]):
            symptom_ids = _collect_symptom_alarm_ids(match)
            owners = sorted({
                g for aid in symptom_ids
                for g in debug.batch.local_alarm_owners.get(aid, ())
            })
            missing = [aid for aid in symptom_ids
                       if aid not in debug.batch.local_alarm_owners]
            print(f"     match#{i}: 覆盖原始组 {owners}"
                  + (f"，症状不属于本批的告警 {missing}" if missing else ""))
    components = local_union.components(alarm_groups)
    multi = [c for c in components if len(c) >= 2]
    print(f"  连通分量: 共 {len(components)} 个，其中 >=2 成员的 {len(multi)} 个")
    for members in multi[:10]:
        print(f"  - 分量: {members}")


def report_assignment(debug, alarm_groups, output):
    print("=" * 72)
    session = debug.session
    print(f"[11] 汇聚组分配: 本批涉及汇聚组 {len(debug.member_entries_by_agg)} 个")
    path_counts = {}
    for _group_id, _agg_id, kind in debug.assignments:
        path_counts[kind] = path_counts.get(kind, 0) + 1
    assigned_group_ids = {group_id for group_id, _a, _k in debug.assignments}
    unassigned = [g for g in alarm_groups if g not in assigned_group_ids]
    print(f"  分配路径: 沿用既有归属 {path_counts.get('沿用', 0)}, "
          f"附着历史汇聚组 {path_counts.get('附着', 0)}, "
          f"新建汇聚组 {path_counts.get('新建', 0)}; "
          f"未分配 {len(unassigned)} 个组")
    for group_id, agg_id, kind in debug.assignments[:15]:
        print(f"  - 组 {group_id!r} --{kind}--> {agg_id}")
    if unassigned:
        print(f"  未分配的组（无关联、或与其它成员的直接连边不足、"
              f"或附着超限后成单成员包）: {unassigned[:15]}")
    print("=" * 72)
    dropped_small = []
    dropped_unchanged = []
    for agg_id in debug.member_entries_by_agg:
        if agg_id in debug.increment_by_agg:
            continue
        entry = session["agg_registry"].get(agg_id)
        if entry is not None and entry[0] < 2:
            dropped_small.append(agg_id)
        elif agg_id not in debug.changed_agg_ids:
            dropped_unchanged.append(agg_id)
    print(f"[12] 增量与变化过滤: 变化汇聚组 {len(debug.changed_agg_ids)} 个, "
          f"增量保留 {len(debug.increment_by_agg)} 个 "
          f"(成员数<2 丢弃 {len(dropped_small)}, "
          f"纯重发未变化跳过 {len(dropped_unchanged)})")
    for agg_id in dropped_small[:10]:
        print(f"  - {agg_id}: 累计原始组 <2，单成员组不输出")
    if debug.member_entries_by_agg and not debug.changed_agg_ids:
        print("  !! 所有汇聚组本批均无新成员组、无新增告警（纯重发），"
              "按输出口径不返回；如需回显全量成员请传 old_agg 并 full_output=True")
    print("=" * 72)
    n_alive = sum(1 for v in output.values() if v.get("is_alive"))
    print(f"[13] 最终输出: {len(output)} 个汇聚组 "
          f"(存活 {n_alive}, 失活 {len(output) - n_alive})")
    for agg_id, value in list(output.items())[:10]:
        members = value.get("group_members") or []
        print(f"  - {agg_id}: is_alive={value.get('is_alive')}, "
              f"成员组 {[list(m)[0] for m in members]}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="诊断 aggregate_alarm_groups 无输出的原因"
    )
    parser.add_argument("params", help="参数文件路径 (.json 或 .py)")
    parser.add_argument("--show-alarms", action="store_true",
                        help="打印全部告警的转换结果")
    parser.add_argument("--show-matches", type=int, default=0, metavar="N",
                        help="打印前 N 个通过过滤的匹配组明细")
    args = parser.parse_args()

    params = load_params(args.params)
    resource_buffer = params.get("resource_buffer") or RESOURCE_BUFFER_JSONL
    if not os.path.exists(resource_buffer):
        raise SystemExit(
            f"资源缓冲文件不存在: {resource_buffer}\n"
            "请在参数文件中用 resource_buffer 指定，或先运行 "
            "tools/build_resource_buffer.py 生成"
        )

    matcher = DebugBatchFaultGroupMatcher(
        resource_buffer=resource_buffer,
        batch_isolated=params.get("batch_isolated", False),
        full_output=params.get("full_output", False),
    )
    alarm_groups = params["alarm_groups"]
    report_input(alarm_groups, params)
    rows = _convert_all_alarms(alarm_groups, matcher.static_context.ne_to_site)
    converted_rows, _no_site = report_conversion(rows, args.show_alarms)

    try:
        output = matcher.aggregate_alarm_groups(
            alarm_groups,
            params["associate_time"],
            params["max_group_time"],
            params["max_group_member"],
            params.get("old_agg_alarm_groups"),
        )
    except Exception:
        print("=" * 72)
        print("!! aggregate_alarm_groups 抛出异常，以下为已收集到的阶段信息")
        report_matches(matcher.debug, converted_rows, args.show_matches)
        report_linking(matcher.debug, alarm_groups)
        raise

    report_matches(matcher.debug, converted_rows, args.show_matches)
    report_linking(matcher.debug, alarm_groups)
    report_assignment(matcher.debug, alarm_groups, output)

    print("=" * 72)
    if output:
        print(f"结论: 本次调用有 {len(output)} 个汇聚组输出（见 [13]）")
    else:
        print("结论: 输出为空。请从上面第一个数量归零/带 !! 标记的阶段入手排查。")
    return output


if __name__ == "__main__":
    main()
