#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解释 complete_group_topology.py --per-file 生成的单个故障组结果。"""

import argparse
import json
import re
import sys
from pathlib import Path


ROLE_LABELS = {
    "common_upstream_site": "最低公共 upstream 站点",
    "farthest_upstream_site": "最远 upstream 站点",
    "ran_data_upstream_site": "多个源站共同连接的 Ran-Data 相邻 Data 站点",
}

FILTER_REASONS = (
    (
        "hub_filtered_ancestor_site_ids",
        "不是 hub 站点且站内没有 Data 设备",
    ),
    (
        "ran_without_data_link_filtered_ancestor_site_ids",
        "站内有 Ran、没有 Data，且没有连接到任何 Data 站点",
    ),
    (
        "data_link_pruned_ancestor_site_ids",
        "与另一个高亮 Data 站点直接相连，优先保留 Data 站点",
    ),
    (
        "shared_data_link_pruned_ancestor_site_ids",
        "与其他非 Data 候选共享 Data 邻站，链路类型比较后落选",
    ),
    (
        "single_data_ancestor_pruned_site_ids",
        "最终只剩一个 Data 祖先候选，因此移除其他非 Data 候选",
    ),
)

ROOT_CAUSE_REASONS = {
    "non_offline_alarm": "站内最早的非 Offline 告警优先",
    "offline_alarm": "站内没有非 Offline 告警，选择最早的 Offline 告警",
    "transmission_device": "站内没有告警，选择下游连接最多的微波设备",
}

ALARM_TITLE_FIELDS = ("告警标题", "告警标准名", "alarm_type", "title", "alarm")


def _text(value):
    return str(value or "").strip()


def _group_id(group):
    match_info = group.get("match_info") if isinstance(group.get("match_info"), dict) else {}
    return _text(group.get("uuid")) or _text(match_info.get("uuid")) or _text(group.get("故障组ID"))


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _site_names(ne_info):
    result = {}
    for entry in (ne_info or {}).values():
        if not isinstance(entry, dict):
            continue
        site_id = _text(entry.get("site_id"))
        site_name = _text(entry.get("site_name"))
        if site_id and site_name:
            result.setdefault(site_id, site_name)
    return result


def _site_label(site_id, site_names):
    site_id = _text(site_id)
    site_name = site_names.get(site_id, "")
    return f"{site_id} ({site_name})" if site_name and site_name != site_id else site_id


def _site_list(site_ids, site_names):
    values = [_site_label(site_id, site_names) for site_id in _as_list(site_ids) if _text(site_id)]
    return ", ".join(values) if values else "无"


def _alarm_title(alarm):
    if not isinstance(alarm, dict):
        return ""
    for field in ALARM_TITLE_FIELDS:
        value = _text(alarm.get(field))
        if value:
            return value
    return "未命名告警"


def _text_has_token(text, token):
    if len(token) <= 2:
        return re.search(rf"(?<![A-Z0-9]){token}(?![A-Z0-9])", text) is not None
    return token in text


def _device_role(entry):
    domain = _text(entry.get("domain") if isinstance(entry, dict) else "").upper()
    if any(_text_has_token(domain, token) for token in ("DATA", "IP", "ROUTER", "METRO")):
        return "Data"
    if any(
        _text_has_token(domain, token)
        for token in ("MICROWAVE", "MW", "RTN", "TRANSMISSION", "DWDM", "OTN", "OPTICAL", "WDM")
    ):
        return "Microwave"
    if any(_text_has_token(domain, token) for token in ("RAN", "WIRELESS", "NODEB", "BTS", "LTE")):
        return "Ran"
    return "Other"


def _site_role_index(ne_info):
    roles_by_site = {}
    for entry in (ne_info or {}).values():
        if not isinstance(entry, dict):
            continue
        site_id = _text(entry.get("site_id"))
        if site_id:
            roles_by_site.setdefault(site_id, set()).add(_device_role(entry))
    return roles_by_site


def _has_ran_data_link(source_site, data_site, ne_info):
    source_site = _text(source_site)
    data_site = _text(data_site)
    for entry in (ne_info or {}).values():
        if not isinstance(entry, dict) or _text(entry.get("site_id")) != source_site:
            continue
        if _device_role(entry) != "Ran":
            continue
        links = entry.get("link") if isinstance(entry.get("link"), dict) else {}
        for peer_ne_id in links:
            peer_entry = ne_info.get(peer_ne_id) if isinstance(ne_info.get(peer_ne_id), dict) else {}
            if _text(peer_entry.get("site_id")) == data_site and _device_role(peer_entry) == "Data":
                return True
    return False


def _load_single_group(input_path):
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    groups = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            groups.append(record)
    if not groups:
        raise ValueError(f"输入文件没有故障组记录: {path}")
    if len(groups) != 1:
        raise ValueError(f"输入必须是单个 --per-file 文件，实际包含 {len(groups)} 条记录")
    return groups[0]


def _append_section(lines, title):
    if lines:
        lines.append("")
    lines.append(f"[{title}]")


def _format_hops(hops, source_site, site_names):
    if not isinstance(hops, dict):
        return "无"
    values = []
    for site_id, hop in sorted(
        hops.items(),
        key=lambda item: (item[1] if isinstance(item[1], (int, float)) else float("inf"), _text(item[0])),
    ):
        if _text(site_id) == _text(source_site):
            continue
        values.append(f"{_site_label(site_id, site_names)}({hop} 跳)")
    return " -> ".join(values) if values else "无 upstream"


def _append_summary(lines, group, completion, site_names):
    original_ne_ids = _as_list(completion.get("original_alarm_ne_ids"))
    original_site_ids = _as_list(completion.get("original_alarm_site_ids"))
    source_site_ids = _as_list(completion.get("ancestor_source_site_ids"))
    non_offline_site_ids = _as_list(completion.get("non_offline_alarm_site_ids"))

    lines.append(f"故障组: {_group_id(group) or '未知'}")
    lines.append(
        "模式: "
        f"{_text(completion.get('mode')) or '未知'}，"
        f"restrict_relation={'是' if completion.get('restrict_relation') else '否'}"
    )
    lines.append(f"原始告警设备 ({len(original_ne_ids)}): {', '.join(map(str, original_ne_ids)) or '无'}")
    lines.append(f"原始告警站点 ({len(original_site_ids)}): {_site_list(original_site_ids, site_names)}")
    lines.append(f"用于 upstream 推断的 Offline 站点 ({len(source_site_ids)}): {_site_list(source_site_ids, site_names)}")
    lines.append(f"不参与祖先推断的非 Offline 告警站点 ({len(non_offline_site_ids)}): {_site_list(non_offline_site_ids, site_names)}")
    lines.append(f"最终选中站点: {_site_list(completion.get('selected_site_ids'), site_names)}")
    lines.append(f"拓扑新增站点: {_site_list(completion.get('added_site_ids'), site_names)}")
    added_ne_ids = _as_list(completion.get("added_ne_ids"))
    lines.append(f"拓扑新增设备 ({len(added_ne_ids)}): {', '.join(map(str, added_ne_ids)) or '无'}")


def _append_upstream_reasoning(lines, completion, site_names):
    _append_section(lines, "upstream 计算")
    source_site_ids = _as_list(completion.get("ancestor_source_site_ids"))
    upstream_hops = completion.get("upstream_site_hops") or {}
    if not source_site_ids:
        lines.append("没有 Offline 告警站点，因此没有执行祖先站点推断。")
        return

    for source_site in source_site_ids:
        source_hops = upstream_hops.get(source_site, {}) if isinstance(upstream_hops, dict) else {}
        lines.append(
            f"- {_site_label(source_site, site_names)}: "
            f"{_format_hops(source_hops, source_site, site_names)}"
        )

    common_sites = _as_list(completion.get("common_upstream_sites"))
    common_site = _text(completion.get("common_upstream_site"))
    if common_site and common_site not in common_sites:
        common_sites.insert(0, common_site)
    if common_sites:
        lines.append(f"结论: 找到最低公共 upstream，候选为 {_site_list(common_sites, site_names)}。")
        hops_by_target = completion.get("common_upstream_hops_by_site") or {}
        for target_site in common_sites:
            hops_by_source = hops_by_target.get(target_site, {}) if isinstance(hops_by_target, dict) else {}
            details = ", ".join(
                f"{_site_label(source, site_names)}={hop} 跳"
                for source, hop in sorted(hops_by_source.items())
            )
            if details:
                lines.append(f"  {_site_label(target_site, site_names)}: {details}")
    else:
        lines.append("结论: 各 Offline 告警站点没有共同 upstream，进入逐站回退逻辑。")
        farthest = completion.get("farthest_upstream_sites") or {}
        for source_site in source_site_ids:
            selected = farthest.get(source_site) if isinstance(farthest, dict) else None
            if isinstance(selected, dict) and _text(selected.get("site_id")):
                detail = (
                    f"选择 {_site_label(selected.get('site_id'), site_names)}"
                    f"（{selected.get('hop', '?')} 跳）作为最远 upstream"
                )
                router_site = _text(selected.get("router_ancestor_site_id"))
                if router_site:
                    detail += f"，再提升到 Data 祖先 {_site_label(router_site, site_names)}"
                lines.append(f"- {_site_label(source_site, site_names)}: {detail}")
            elif source_site in _as_list(completion.get("no_upstream_sites")):
                lines.append(f"- {_site_label(source_site, site_names)}: 没有可用 upstream")

    chains = completion.get("intermediate_site_chains") or {}
    if isinstance(chains, dict) and chains:
        lines.append("restrict_relation 补入的中间链路:")
        for source_site, chain in sorted(chains.items()):
            lines.append(
                f"- {_site_label(source_site, site_names)}: "
                + " -> ".join(_site_label(site_id, site_names) for site_id in _as_list(chain))
            )


def _append_data_reasoning(lines, completion, site_names):
    _append_section(lines, "Data 祖先与最终高亮")
    promotions = _as_list(completion.get("data_ancestor_promotions"))
    if promotions:
        lines.append("非 Data 祖先向上提升到 Data 站点:")
        for promotion in promotions:
            if not isinstance(promotion, dict):
                continue
            lines.append(
                f"- {_site_label(promotion.get('from_site_id'), site_names)} -> "
                f"{_site_label(promotion.get('to_site_id'), site_names)}"
                f"（{promotion.get('upstream_hop', '?')} 跳）"
            )
    missing = _as_list(completion.get("data_ancestor_missing_site_ids"))
    if missing:
        lines.append(f"未找到更上游 Data 站点: {_site_list(missing, site_names)}")

    highlights = _as_list(completion.get("highlight_sites"))
    if not highlights:
        lines.append("最终没有高亮祖先站点。")
    ran_data_highlights = [
        item
        for item in highlights
        if isinstance(item, dict) and item.get("role") == "ran_data_upstream_site"
    ]
    for item in highlights:
        if not isinstance(item, dict):
            continue
        site_id = _text(item.get("site_id"))
        role = _text(item.get("role"))
        role_label = ROLE_LABELS.get(role, _text(item.get("label")) or role or "未知角色")
        lines.append(f"- {_site_label(site_id, site_names)}: {role_label}")
        source_sites = _as_list(item.get("source_sites"))
        if source_sites:
            lines.append(f"  来源站点: {_site_list(source_sites, site_names)}")
        if role == "ran_data_upstream_site":
            if len(set(map(_text, source_sites))) >= 2:
                lines.append("  原因: 上述多个候选源站位于同一站点链路组件，并共同连接到这个 Data 站点。")
            else:
                lines.append("  警告: source_sites 少于两个，与当前 Ran-Data 补标规则不一致。")
            lines.append("  可复核范围: per-file 未保存组件编号和原始设备链路，无法仅凭此文件重放连通性判断。")
        hops_by_source = item.get("hops_by_source_site")
        if isinstance(hops_by_source, dict) and hops_by_source:
            details = ", ".join(
                f"{_site_label(source, site_names)}={hop} 跳"
                for source, hop in sorted(hops_by_source.items())
            )
            lines.append(f"  upstream 跳数: {details}")

    if not ran_data_highlights:
        if _as_list(completion.get("common_upstream_sites")) or _text(completion.get("common_upstream_site")):
            lines.append("Ran-Data 补标未执行: 已找到最低公共 upstream，算法直接采用公共 upstream 结果。")
        else:
            lines.append("Ran-Data 补标没有产生最终结果。")
            lines.append(
                "可复核范围: per-file 未保存落选候选、站点组件和原始设备链路，"
                "无法仅凭此文件区分是候选不足、没有共享同一 Data 邻站、组件不连通，还是未满足 Ran-Data 链路。"
            )


def _append_shared_data_neighbors(lines, group, completion, site_names):
    highlights = [
        item
        for item in _as_list(completion.get("highlight_sites"))
        if isinstance(item, dict) and item.get("role") == "ran_data_upstream_site"
    ]
    if not highlights:
        return

    _append_section(lines, "无 Data upstream -> 公共 Data 邻站")
    ne_info = group.get("ne_info") if isinstance(group.get("ne_info"), dict) else {}
    roles_by_site = _site_role_index(ne_info)
    data_site_ids = {
        site_id for site_id, roles in roles_by_site.items() if "Data" in roles
    }
    upstream_hops = completion.get("upstream_site_hops") or {}

    for item in highlights:
        data_site = _text(item.get("site_id"))
        source_sites = sorted({_text(site_id) for site_id in _as_list(item.get("source_sites")) if _text(site_id)})
        data_status = "已从 ne_info 确认为 Data 站点" if data_site in data_site_ids else "ne_info 中未识别到 Data 设备"
        lines.append(f"公共 Data 站点: {_site_label(data_site, site_names)}（{data_status}）")
        lines.append(f"共同连接源站 ({len(source_sites)}): {_site_list(source_sites, site_names)}")
        for source_site in source_sites:
            source_hops = upstream_hops.get(source_site, {}) if isinstance(upstream_hops, dict) else {}
            upstream_site_ids = {
                _text(site_id)
                for site_id in source_hops
                if _text(site_id) and _text(site_id) != source_site
            }
            data_upstream_site_ids = sorted(upstream_site_ids & data_site_ids)
            unknown_upstream_site_ids = sorted(upstream_site_ids - set(roles_by_site))
            lines.append(f"- 源站 {_site_label(source_site, site_names)}")
            lines.append(f"  upstream: {_format_hops(source_hops, source_site, site_names)}")
            if data_upstream_site_ids:
                lines.append(
                    "  Data upstream 检查: 不符合，发现 "
                    + _site_list(data_upstream_site_ids, site_names)
                )
            elif unknown_upstream_site_ids:
                lines.append(
                    "  Data upstream 检查: 文件内可见 upstream 未发现 Data；"
                    "以下站点缺少设备信息，无法完整确认: "
                    + _site_list(unknown_upstream_site_ids, site_names)
                )
            else:
                lines.append("  Data upstream 检查: 通过，所有可见 upstream 站点均为非 Data")
            lines.append(
                "  Ran->Data 直连检查: "
                + ("通过" if _has_ran_data_link(source_site, data_site, ne_info) else "未能从 per-file 的设备链路中复核")
            )
        lines.append("站点链路组件检查: per-file 未保存组件编号，只能看到算法记录的最终通过结果。")


def _append_filters(lines, completion, site_names):
    _append_section(lines, "候选过滤")
    found = False
    for field, reason in FILTER_REASONS:
        site_ids = _as_list(completion.get(field))
        if not site_ids:
            continue
        found = True
        lines.append(f"- {_site_list(site_ids, site_names)}")
        lines.append(f"  原因: {reason}")
    if not found:
        lines.append("没有记录到被后处理规则移除的祖先候选。")


def _append_offline_duration(lines, completion, site_names):
    durations = completion.get("offline_site_max_duration_seconds")
    summary = completion.get("offline_duration_filter")
    if not isinstance(durations, dict) and not isinstance(summary, dict):
        return

    _append_section(lines, "Offline 持续时间")
    if isinstance(durations, dict) and durations:
        for site_id, seconds in sorted(durations.items()):
            duration = "持续中/未清除" if seconds is None else f"{seconds} 秒"
            lines.append(f"- {_site_label(site_id, site_names)}: {duration}")
    else:
        lines.append("没有可计算持续时间的 Offline 告警。")

    if isinstance(summary, dict):
        for rule in _as_list(summary.get("rules")):
            if not isinstance(rule, dict):
                continue
            lines.append(
                f"- 阈值 >= {rule.get('min_minutes', '?')} 分钟且至少 "
                f"{rule.get('min_site_count', '?')} 站: 实际 {rule.get('qualifying_site_count', '?')} 站，"
                f"{'通过' if rule.get('passes') else '不通过'}"
            )
        lines.append(f"持续时间筛选总结果: {'通过' if summary.get('passes') else '不通过'}")


def _append_root_causes(lines, group, completion, site_names):
    _append_section(lines, "自动根因")
    annotations = _as_list(completion.get("auto_root_cause_annotations"))
    if not annotations:
        if isinstance(group.get("root_cause_annotations"), dict) and group.get("root_cause_annotations"):
            lines.append("没有生成自动根因摘要；文件中已有根因标注，自动逻辑不会覆盖它。")
        else:
            lines.append("没有生成自动根因标注。")
        return
    for item in annotations:
        if not isinstance(item, dict):
            continue
        kind = _text(item.get("kind"))
        lines.append(
            f"- {_site_label(item.get('site_id'), site_names)}: 设备 {_text(item.get('ne_id'))}；"
            f"{ROOT_CAUSE_REASONS.get(kind, kind or '原因未知')}"
        )
        if _text(item.get("alarm_key")):
            lines.append(f"  告警键: {_text(item.get('alarm_key'))}")


def _append_ne_details(lines, group, completion, all_ne):
    ne_info = group.get("ne_info") if isinstance(group.get("ne_info"), dict) else {}
    if not ne_info:
        return
    original_ne_ids = set(map(_text, _as_list(completion.get("original_alarm_ne_ids"))))
    highlight_sites = {
        _text(item.get("site_id"))
        for item in _as_list(completion.get("highlight_sites"))
        if isinstance(item, dict) and _text(item.get("site_id"))
    }
    root_cause_ne_ids = {
        _text(item.get("ne_id"))
        for item in _as_list(completion.get("auto_root_cause_annotations"))
        if isinstance(item, dict) and _text(item.get("ne_id"))
    }
    selected_ne_ids = sorted(
        ne_id
        for ne_id, entry in ne_info.items()
        if all_ne
        or _text(ne_id) in original_ne_ids
        or _text(ne_id) in root_cause_ne_ids
        or (isinstance(entry, dict) and _text(entry.get("site_id")) in highlight_sites)
    )
    _append_section(lines, "设备与告警")
    for ne_id in selected_ne_ids:
        entry = ne_info.get(ne_id) if isinstance(ne_info.get(ne_id), dict) else {}
        flags = []
        if _text(ne_id) in original_ne_ids:
            flags.append("原始告警设备")
        if entry.get("topology_added"):
            flags.append("拓扑补入")
        if _text(ne_id) in root_cause_ne_ids:
            flags.append("自动根因")
        suffix = f" [{' / '.join(flags)}]" if flags else ""
        lines.append(
            f"- {ne_id}: site={_text(entry.get('site_id')) or '未知'}，"
            f"domain={_text(entry.get('domain')) or '未知'}{suffix}"
        )
        alarms = [alarm for alarm in _as_list(entry.get("alarm")) if isinstance(alarm, dict)]
        if alarms:
            lines.append("  告警: " + "；".join(_alarm_title(alarm) for alarm in alarms))


def _append_consistency_checks(lines, group, completion, site_names):
    warnings = []
    selected_sites = set(map(_text, _as_list(completion.get("selected_site_ids"))))
    original_sites = set(map(_text, _as_list(completion.get("original_alarm_site_ids"))))
    missing_original = sorted((original_sites - selected_sites) - {""})
    if missing_original:
        warnings.append(f"原始告警站点未出现在最终站点中: {_site_list(missing_original, site_names)}")

    highlight_ids = {
        _text(item.get("site_id"))
        for item in _as_list(completion.get("highlight_sites"))
        if isinstance(item, dict) and _text(item.get("site_id"))
    }
    recorded_highlight_ids = set(map(_text, _as_list(completion.get("highlight_site_ids"))))
    if highlight_ids != recorded_highlight_ids:
        warnings.append("highlight_site_ids 与 highlight_sites 中的站点不一致")

    source_sites = [_text(site_id) for site_id in _as_list(completion.get("ancestor_source_site_ids"))]
    excluded_sites = set(map(_text, _as_list(completion.get("non_offline_alarm_site_ids")))) - {""}
    upstream_hops = completion.get("upstream_site_hops") or {}
    common_candidates = None
    for source_site in source_sites:
        source_hops = upstream_hops.get(source_site, {}) if isinstance(upstream_hops, dict) else {}
        candidates = {
            _text(site_id)
            for site_id in source_hops
            if _text(site_id) and _text(site_id) not in excluded_sites
        }
        common_candidates = candidates if common_candidates is None else common_candidates & candidates
    common_candidates = common_candidates or set()
    recorded_common_sites = set(map(_text, _as_list(completion.get("common_upstream_sites")))) - {""}
    recorded_common_site = _text(completion.get("common_upstream_site"))
    if recorded_common_site:
        recorded_common_sites.add(recorded_common_site)
    if source_sites and common_candidates and not recorded_common_sites:
        warnings.append(
            "文件记录为“无公共 upstream”，但 upstream_site_hops 的交集非空: "
            + _site_list(sorted(common_candidates), site_names)
        )
    invalid_common_sites = recorded_common_sites - common_candidates
    if invalid_common_sites:
        warnings.append(
            "记录的公共 upstream 不在所有源站的 upstream 交集中: "
            + _site_list(sorted(invalid_common_sites), site_names)
        )

    role_mapping = group.get("role_mapping") if isinstance(group.get("role_mapping"), dict) else {}
    for role in ROLE_LABELS:
        role_sites = set(map(_text, _as_list(role_mapping.get(role)))) - {""}
        item_sites = {
            _text(item.get("site_id"))
            for item in _as_list(completion.get("highlight_sites"))
            if isinstance(item, dict) and item.get("role") == role and _text(item.get("site_id"))
        }
        if role_sites != item_sites:
            warnings.append(f"role_mapping.{role} 与最终 highlight_sites 不一致")

    _append_section(lines, "一致性检查")
    if warnings:
        lines.extend(f"- 警告: {warning}" for warning in warnings)
    else:
        lines.append("未发现输出字段之间的明显矛盾。")


def explain_group(group, input_path=None, all_ne=False):
    completion = group.get("topology_completion")
    if not isinstance(completion, dict):
        raise ValueError("输入记录没有 topology_completion，可能不是 --per-file 的拓扑补全结果")

    ne_info = group.get("ne_info") if isinstance(group.get("ne_info"), dict) else {}
    site_names = _site_names(ne_info)
    lines = ["故障组拓扑诊断"]
    if input_path:
        lines.append(f"输入: {input_path}")
    _append_summary(lines, group, completion, site_names)
    _append_upstream_reasoning(lines, completion, site_names)
    _append_data_reasoning(lines, completion, site_names)
    _append_shared_data_neighbors(lines, group, completion, site_names)
    _append_filters(lines, completion, site_names)
    _append_offline_duration(lines, completion, site_names)
    _append_root_causes(lines, group, completion, site_names)
    _append_ne_details(lines, group, completion, all_ne)
    _append_consistency_checks(lines, group, completion, site_names)
    return "\n".join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="解释单个 complete_group_topology.py --per-file 输出为什么得到当前结果"
    )
    parser.add_argument("input", help="单个故障组的单行 JSONL 文件")
    parser.add_argument(
        "--all-ne",
        action="store_true",
        help="展开打印所有输出设备；默认只打印告警、高亮站点和自动根因相关设备",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    try:
        group = _load_single_group(args.input)
        print(explain_group(group, input_path=args.input, all_ne=args.all_ne))
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
