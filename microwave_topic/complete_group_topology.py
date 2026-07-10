#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按站点 upstream_site_hops 信息补齐故障组拓扑。"""

import argparse
import copy
import heapq
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_types import OFFLINE_ALARMS
from fault_grouping.site_topology import build_site_to_ne_ids, normalize_site_chain_hops
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_CHAINS_JSON,
    SITE_GRAPH_JSON,
    resource_display,
)


BLOCKED_ANCESTOR_SITE_IDS = {"13PWK0024"}
DEBUG_SITE_ID = "13SRN0089"
OFFLINE_ALARM_KEYS = {str(alarm or "").strip().upper() for alarm in OFFLINE_ALARMS} | {"OFFLINE"}

# --filter 使用：每条规则为 (offline 持续时间阈值秒数, 需要满足的站点数)。
# 只要任一规则满足（用该阈值卡后仍剩至少这么多个有 offline 告警的站点），故障组即保留。
OFFLINE_DURATION_FILTER_RULES = (
    (30 * 60, 4),
    (15 * 60, 10),
    (7 * 60, 30),
)
# 计算每个站最长 offline 告警持续时间时读取的时间字段（起始取最早发生时间，结束取清除时间）。
OFFLINE_START_TIME_FIELDS = (
    "ts",
    "告警首次发生时间",
    "告警发生时间",
    "发生时间",
    "首次发生时间",
    "alarm_time",
    "time",
)
OFFLINE_CLEAR_TIME_FIELDS = ("告警清除时间", "alarm_clear_time", "clear_time", "清除时间")


def _parse_timestamp(value):
    """把告警时间字段解析为 epoch 秒；支持数值（已是 epoch）与常见日期字符串。"""
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("T", " ")).timestamp()
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _first_record_timestamp(record, fields):
    for field_name in fields:
        ts = _parse_timestamp(record.get(field_name))
        if ts is not None:
            return ts
    return None


def _normalize_text(value):
    return str(value or "").strip()


def _load_json_object(path, label, warn_if_missing=False):
    if not path:
        return {}
    if not Path(path).exists():
        if warn_if_missing:
            print(f"⚠️ {label} 文件不存在，跳过对应补充信息: {path}", file=sys.stderr)
        return {}
    with open(path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    if not isinstance(data, dict):
        raise ValueError(f"{label} 顶层必须是对象: {path}")
    return data


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fr:
        for line_num, raw_line in enumerate(fr, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_num} 行 JSON 解析失败: {exc}") from exc
            if isinstance(record, dict):
                yield record


def _count_jsonl_records(path):
    count = 0
    with open(path, "r", encoding="utf-8") as fr:
        for raw_line in fr:
            if raw_line.strip():
                count += 1
    return count


def _format_link_meta(link_meta, distance=""):
    if isinstance(link_meta, dict):
        connection_types = sorted(str(key) for key in link_meta.keys())
        topologies = sorted({str(value) for value in link_meta.values() if value})
    else:
        connection_types = [str(link_meta)] if link_meta not in (None, "") else []
        topologies = []
    return {
        "connection_type": ",".join(connection_types),
        "distance": distance,
        "topology": ",".join(topologies),
        "time_window": "",
        "left_alarm": {},
        "right_alarm": {},
    }


def _site_link_distance_km(source_site, target_site, site_graph_data):
    """读取两个站点间已记录的物理距离（公里）；同站设备距离为 0。"""
    source_site = _normalize_text(source_site)
    target_site = _normalize_text(target_site)
    if not source_site or not target_site:
        return ""
    if source_site == target_site:
        return 0.0
    if not isinstance(site_graph_data, dict):
        return ""

    # site_graph 的邻接关系是双向的；兼容仅一侧记录了距离的文件。
    for current_site, neighbor_site in (
        (source_site, target_site),
        (target_site, source_site),
    ):
        site_info = site_graph_data.get(current_site, {})
        if not isinstance(site_info, dict):
            continue
        distances = site_info.get("link_distance_km", {})
        if not isinstance(distances, dict) or neighbor_site not in distances:
            continue
        try:
            distance = float(distances[neighbor_site])
        except (TypeError, ValueError):
            continue
        if math.isfinite(distance) and distance >= 0:
            return round(distance, 2)
    return ""


def _detect_restrict_relation(meta):
    """从 site_chains 的 meta 判断是否由 --restrict-relation 生成。"""
    if not isinstance(meta, dict):
        return False
    relation_options = meta.get("relation_options")
    if isinstance(relation_options, dict) and "restrict_relation_effective" in relation_options:
        return bool(relation_options.get("restrict_relation_effective"))
    input_config = meta.get("input_config")
    if isinstance(input_config, dict):
        return bool(input_config.get("restrict_relation"))
    return False


def _load_site_chain_index(site_chains_path):
    if not site_chains_path or not Path(site_chains_path).exists():
        if site_chains_path:
            print(f"⚠️ site_chains 文件不存在，将只保留原始告警站点: {site_chains_path}", file=sys.stderr)
        return {}, False
    with open(site_chains_path, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    if not isinstance(data, dict):
        raise ValueError(f"site_chains 顶层必须是对象: {site_chains_path}")
    restrict_relation = _detect_restrict_relation(data.get("meta", {}))
    raw_sites = data.get("sites", {})
    site_chain_index = {}
    if isinstance(raw_sites, dict):
        for raw_site_id, raw_info in raw_sites.items():
            site_id = _normalize_text(raw_site_id)
            if not site_id or not isinstance(raw_info, dict):
                continue
            site_chain_index[site_id] = {
                "upstream_site_hops": normalize_site_chain_hops(raw_info.get("upstream_site_hops")),
                "downstream_site_hops": normalize_site_chain_hops(raw_info.get("downstream_site_hops")),
                "bidirectional_sites": _normalize_site_id_set(raw_info.get("bidirectional_sites")),
            }
    return site_chain_index, restrict_relation


def _normalize_site_id_set(site_ids):
    return {
        _normalize_text(site_id)
        for site_id in (site_ids or [])
        if _normalize_text(site_id)
    }


def _build_site_chain_component_index(site_chain_index):
    """site_chains 无向连通分量索引：upstream/downstream/bidirectional 关系都视为无向边。

    与故障组无关，入口处预构建一次复用；返回 站点ID -> 分量代表站点ID。
    """
    parent = {}

    def _find(site_id):
        root = site_id
        while parent[root] != root:
            root = parent[root]
        while parent[site_id] != root:
            parent[site_id], site_id = root, parent[site_id]
        return root

    def _union(site_a, site_b):
        parent.setdefault(site_a, site_a)
        parent.setdefault(site_b, site_b)
        root_a, root_b = _find(site_a), _find(site_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for site_id, info in (site_chain_index or {}).items():
        if not site_id or not isinstance(info, dict):
            continue
        parent.setdefault(site_id, site_id)
        neighbor_sites = (
            set(info.get("upstream_site_hops") or ())
            | set(info.get("downstream_site_hops") or ())
            | set(info.get("bidirectional_sites") or ())
        )
        for neighbor_site in neighbor_sites:
            if neighbor_site and neighbor_site != site_id:
                _union(site_id, neighbor_site)

    return {site_id: _find(site_id) for site_id in parent}


def _site_context(site_id, site_graph_data, ne_info):
    site_info = site_graph_data.get(site_id, {}) if isinstance(site_graph_data, dict) else {}
    if not isinstance(site_info, dict):
        site_info = {}
    return {
        "site_name": (
            _normalize_text(ne_info.get("site_name", ""))
            or _normalize_text(site_info.get("site_name", ""))
            or _normalize_text(site_info.get("name", ""))
        ),
        "site_type": _normalize_text(ne_info.get("site_type", "")) or _normalize_text(site_info.get("site_type", "")),
        "region_id": _normalize_text(ne_info.get("region_id", "")) or _normalize_text(site_info.get("region_id", "")),
        "longitude": ne_info.get("longitude", site_info.get("longitude", site_info.get("lon", site_info.get("lng", "")))),
        "latitude": ne_info.get("latitude", site_info.get("latitude", site_info.get("lat", ""))),
    }


def _site_of_ne(ne_id, ne_graph_data, group_site_by_ne=None):
    info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    if isinstance(info, dict):
        site_id = _normalize_text(info.get("site_id", ""))
        if site_id:
            return site_id
    return _normalize_text((group_site_by_ne or {}).get(ne_id, ""))


def _site_is_hub(site_id, site_graph_data):
    site_id = _normalize_text(site_id)
    site_info = site_graph_data.get(site_id, {}) if isinstance(site_graph_data, dict) else {}
    if not isinstance(site_info, dict) or "is_hub" not in site_info:
        return True
    value = site_info.get("is_hub")
    if isinstance(value, bool):
        return value
    return _normalize_text(value).lower() in {"1", "true", "t", "yes", "y", "是"}


def _text_has_token(text, token):
    # 短 token（IP/MW）必须独立成词，否则 EQU"IP"MENT 这类子串会误判设备角色。
    if len(token) <= 2:
        return re.search(rf"(?<![A-Z0-9]){token}(?![A-Z0-9])", text) is not None
    return token in text


def _ne_domain_text(ne_info):
    # 设备角色判定只看 domain 字段，不引入 network_type/type 等其他字段。
    for field_name in ("domain", "Domain", "DOMAIN"):
        value = _normalize_text(ne_info.get(field_name, ""))
        if value:
            return value.upper()
    return ""


def _is_data_ne(ne_info):
    if not isinstance(ne_info, dict):
        return False
    text = _ne_domain_text(ne_info)
    return any(_text_has_token(text, token) for token in ("DATA", "IP", "ROUTER", "METRO"))


def _device_role(ne_info):
    if not isinstance(ne_info, dict):
        return "Other"
    text = _ne_domain_text(ne_info)
    if any(_text_has_token(text, token) for token in ("DATA", "IP", "ROUTER", "METRO")):
        return "Data"
    if any(_text_has_token(text, token) for token in ("MICROWAVE", "MW", "RTN", "TRANSMISSION", "DWDM", "OTN", "OPTICAL", "WDM")):
        return "Microwave"
    if any(_text_has_token(text, token) for token in ("RAN", "WIRELESS", "NODEB", "BTS", "LTE")):
        return "Ran"
    return "Other"


def _build_site_data_and_link_index(ne_graph_data):
    site_has_data = set()
    site_has_ran = set()
    site_links = defaultdict(set)
    directed_edge_types = defaultdict(set)
    ne_to_site = {}
    ne_roles = {}

    if not isinstance(ne_graph_data, dict):
        return site_has_data, site_has_ran, site_links, directed_edge_types

    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        site_id = _normalize_text(ne_info.get("site_id", ""))
        if not site_id:
            continue
        ne_to_site[ne_id] = site_id
        role = _device_role(ne_info)
        ne_roles[ne_id] = role
        if role == "Data":
            site_has_data.add(site_id)
        elif role == "Ran":
            site_has_ran.add(site_id)

    for source_ne, source_info in ne_graph_data.items():
        if not isinstance(source_info, dict):
            continue
        source_site = ne_to_site.get(source_ne, "")
        links = source_info.get("link", {})
        if not source_site or not isinstance(links, dict):
            continue
        for target_ne in links:
            target_site = ne_to_site.get(target_ne, "")
            if not target_site or target_site == source_site:
                continue
            site_links[source_site].add(target_site)
            site_links[target_site].add(source_site)
            source_role = ne_roles.get(source_ne, "Other")
            target_role = ne_roles.get(target_ne, "Other")
            directed_edge_types[(source_site, target_site)].add((source_role, target_role))
            directed_edge_types[(target_site, source_site)].add((target_role, source_role))

    return site_has_data, site_has_ran, site_links, directed_edge_types


def _group_site_by_ne(group):
    mapping = {}
    ne_info = group.get("ne_info", {})
    if isinstance(ne_info, dict):
        for ne_id, info in ne_info.items():
            if isinstance(info, dict):
                site_id = _normalize_text(info.get("site_id", ""))
                if site_id:
                    mapping[ne_id] = site_id
    for symptom in group.get("symptoms") or []:
        if not isinstance(symptom, dict):
            continue
        ne_id = _normalize_text(symptom.get("alarm_source") or symptom.get("ne_id") or symptom.get("source") or "")
        site_id = _normalize_text(symptom.get("node") or symptom.get("site_id") or "")
        if ne_id and site_id and ne_id not in mapping:
            mapping[ne_id] = site_id
    return mapping


def _extract_alarm_ne_ids(group):
    ne_ids = []
    for ne_id in group.get("alarm_sources") or []:
        ne_id = _normalize_text(ne_id)
        if ne_id and ne_id not in ne_ids:
            ne_ids.append(ne_id)
    for alarm in group.get("alarms") or []:
        if isinstance(alarm, dict):
            ne_id = _normalize_text(alarm.get("告警源", ""))
            if ne_id and ne_id not in ne_ids:
                ne_ids.append(ne_id)
    for symptom in group.get("symptoms") or []:
        if not isinstance(symptom, dict):
            continue
        ne_id = _normalize_text(symptom.get("alarm_source") or symptom.get("ne_id") or symptom.get("source") or "")
        if ne_id and ne_id not in ne_ids:
            ne_ids.append(ne_id)
    ne_info = group.get("ne_info", {})
    if isinstance(ne_info, dict):
        for ne_id, info in ne_info.items():
            alarms = info.get("alarm") if isinstance(info, dict) else None
            if isinstance(alarms, list) and alarms and ne_id not in ne_ids:
                ne_ids.append(ne_id)
    return ne_ids


def _record_alarm_source(record):
    if not isinstance(record, dict):
        return ""
    return _normalize_text(
        record.get("告警源")
        or record.get("alarm_source")
        or record.get("ne_id")
        or record.get("source")
        or ""
    )


def _iter_group_alarm_records(group):
    records = [alarm for alarm in group.get("alarms") or [] if isinstance(alarm, dict)]
    if records:
        yield from records
        return

    records = [symptom for symptom in group.get("symptoms") or [] if isinstance(symptom, dict)]
    if records:
        yield from records
        return

    ne_info = group.get("ne_info", {})
    if not isinstance(ne_info, dict):
        return
    for ne_id, info in ne_info.items():
        if not isinstance(info, dict):
            continue
        for alarm in info.get("alarm") or []:
            if isinstance(alarm, dict):
                alarm = copy.deepcopy(alarm)
                alarm.setdefault("alarm_source", ne_id)
                yield alarm


def _append_unique(values, value):
    if value and value not in values:
        values.append(value)


def _missing_coordinate_ne_ids(group):
    """返回经度或纬度缺失的输出设备 ID；数值 0 是有效坐标。"""
    ne_info = group.get("ne_info") if isinstance(group, dict) else None
    if not isinstance(ne_info, dict):
        return []

    def _is_blank(value):
        return value is None or (isinstance(value, str) and not value.strip())

    return sorted(
        ne_id
        for ne_id, entry in ne_info.items()
        if not isinstance(entry, dict)
        or _is_blank(entry.get("longitude"))
        or _is_blank(entry.get("latitude"))
    )


def _check_group_alarm_topology(group, ne_graph_data, site_graph_data):
    result = {
        "checked_alarm_count": 0,
        "missing_alarm_source_count": 0,
        "missing_ne_ids": [],
        "missing_site_ne_ids": [],
        "missing_site_graph_ids": [],
    }

    alarm_records = list(_iter_group_alarm_records(group))
    if not alarm_records:
        alarm_records = [{"alarm_source": ne_id} for ne_id in _extract_alarm_ne_ids(group)]

    for record in alarm_records:
        result["checked_alarm_count"] += 1
        ne_id = _record_alarm_source(record)
        if not ne_id:
            result["missing_alarm_source_count"] += 1
            continue

        if not isinstance(ne_graph_data, dict) or ne_id not in ne_graph_data:
            _append_unique(result["missing_ne_ids"], ne_id)
            continue
        ne_info = ne_graph_data.get(ne_id, {})
        if not isinstance(ne_info, dict):
            _append_unique(result["missing_ne_ids"], ne_id)
            continue

        site_id = _normalize_text(ne_info.get("site_id", ""))
        if not site_id:
            _append_unique(result["missing_site_ne_ids"], ne_id)
            continue

        if not isinstance(site_graph_data, dict) or site_id not in site_graph_data:
            _append_unique(result["missing_site_graph_ids"], site_id)
            continue
        site_info = site_graph_data.get(site_id, {})
        if not isinstance(site_info, dict):
            _append_unique(result["missing_site_graph_ids"], site_id)

    result["ok"] = (
        result["checked_alarm_count"] > 0
        and result["missing_alarm_source_count"] == 0
        and not result["missing_ne_ids"]
        and not result["missing_site_ne_ids"]
        and not result["missing_site_graph_ids"]
    )
    return result


def _is_offline_alarm_type(value):
    text = _normalize_text(value)
    if not text:
        return False
    upper_text = text.upper()
    return upper_text in OFFLINE_ALARM_KEYS or "OFFLINE" in upper_text or "断站" in text


def _record_has_offline_alarm(record):
    if not isinstance(record, dict):
        return False
    for field_name in ("告警标题", "告警标准名", "alarm", "alarm_type", "title"):
        if _is_offline_alarm_type(record.get(field_name, "")):
            return True
    return False


def _append_unique_site(site_ids, site_id):
    site_id = _normalize_text(site_id)
    if site_id and site_id not in site_ids:
        site_ids.append(site_id)


def _iter_offline_alarm_site_records(group, ne_graph_data, group_site_by_ne):
    """遍历故障组内所有 Offline/断站 告警记录，产出 (站点ID, 告警记录)。"""

    def _site_for(site_id, ne_id):
        return _normalize_text(site_id) or _site_of_ne(
            _normalize_text(ne_id), ne_graph_data, group_site_by_ne
        )

    for symptom in group.get("symptoms") or []:
        if not isinstance(symptom, dict) or not _record_has_offline_alarm(symptom):
            continue
        site_id = _site_for(
            symptom.get("node") or symptom.get("site_id") or "",
            symptom.get("alarm_source") or symptom.get("ne_id") or symptom.get("source") or "",
        )
        if site_id:
            yield site_id, symptom

    match_info = group.get("match_info") if isinstance(group.get("match_info"), dict) else {}
    for symptom in match_info.get("symptoms") or []:
        if not isinstance(symptom, dict) or not _record_has_offline_alarm(symptom):
            continue
        site_id = _site_for(
            symptom.get("node") or symptom.get("site_id") or "",
            symptom.get("alarm_source") or symptom.get("ne_id") or symptom.get("source") or "",
        )
        if site_id:
            yield site_id, symptom

    for alarm in group.get("alarms") or []:
        if not isinstance(alarm, dict) or not _record_has_offline_alarm(alarm):
            continue
        site_id = _site_for(
            alarm.get("站点ID") or alarm.get("site_id") or alarm.get("node") or "",
            alarm.get("告警源") or alarm.get("alarm_source") or alarm.get("ne_id") or alarm.get("source") or "",
        )
        if site_id:
            yield site_id, alarm

    ne_info = group.get("ne_info", {})
    if isinstance(ne_info, dict):
        for ne_id, info in ne_info.items():
            if not isinstance(info, dict):
                continue
            for alarm in info.get("alarm") or []:
                if not isinstance(alarm, dict) or not _record_has_offline_alarm(alarm):
                    continue
                site_id = _site_for(
                    alarm.get("site_id") or alarm.get("node") or info.get("site_id") or "",
                    ne_id,
                )
                if site_id:
                    yield site_id, alarm


def _extract_offline_alarm_site_ids(group, ne_graph_data, group_site_by_ne):
    site_ids = []
    for site_id, _record in _iter_offline_alarm_site_records(group, ne_graph_data, group_site_by_ne):
        _append_unique_site(site_ids, site_id)
    return sorted(site_ids)


def _record_offline_duration_seconds(record):
    """单条 offline 告警的持续时间（秒）：清除时间 - 起始时间；未清除记为 inf（持续中）。"""
    start_ts = _first_record_timestamp(record, OFFLINE_START_TIME_FIELDS)
    if start_ts is None:
        return None
    clear_ts = _first_record_timestamp(record, OFFLINE_CLEAR_TIME_FIELDS)
    if clear_ts is None:
        return float("inf")
    return max(0.0, clear_ts - start_ts)


def _offline_site_max_durations(group, ne_graph_data, group_site_by_ne):
    """统计每个站点最长的一条 offline 告警持续时间（秒），inf 表示仍在持续/未清除。"""
    durations = {}
    for site_id, record in _iter_offline_alarm_site_records(group, ne_graph_data, group_site_by_ne):
        duration = _record_offline_duration_seconds(record)
        if duration is None:
            continue
        if site_id not in durations or duration > durations[site_id]:
            durations[site_id] = duration
    return durations


def _serialize_offline_durations(durations):
    """把每站最长 offline 时长转成 JSON 安全的映射：inf（未清除）序列化为 null。"""
    return {
        site_id: (None if duration == float("inf") else int(round(duration)))
        for site_id, duration in sorted(durations.items())
    }


def _offline_duration_filter_summary(durations):
    """按 OFFLINE_DURATION_FILTER_RULES 逐条统计满足阈值的站点数，并给出总体是否保留。"""
    rules = []
    passes = False
    for min_seconds, min_site_count in OFFLINE_DURATION_FILTER_RULES:
        qualifying_site_count = sum(
            1 for duration in durations.values() if duration >= min_seconds
        )
        rule_passes = qualifying_site_count >= min_site_count
        passes = passes or rule_passes
        rules.append({
            "min_minutes": min_seconds // 60,
            "min_site_count": min_site_count,
            "qualifying_site_count": qualifying_site_count,
            "passes": rule_passes,
        })
    return {"rules": rules, "passes": passes}


def _build_weighted_upstream_adjacency(site_chain_index):
    """构建 站点 -> {上游站点: 权重} 的带权邻接，权重取 upstream_site_hops 里存的跳数。

    restrict-relation 生成的 site_chains 把 upstream_site_hops 裁成了“与本站点有 ne_graph 直连边
    的站点”，但每条仍保留其原始链路跳数。沿这些带权边累加（取最小）即可还原完整上游闭包与正确
    跳数，例如 a:{b:1,c:3} + b:{d:2} + c:{d:2} 合并得 a:{b:1,c:3,d:3}（d 经 b 是 1+2=3）。
    """
    adjacency = defaultdict(dict)
    for site_id, info in site_chain_index.items():
        upstream_hops = info.get("upstream_site_hops", {}) if isinstance(info, dict) else {}
        for upstream_site, hop in upstream_hops.items():
            upstream_site = _normalize_text(upstream_site)
            if not upstream_site or upstream_site == site_id:
                continue
            hop = int(hop)
            if hop <= 0:
                continue
            existing = adjacency[site_id].get(upstream_site)
            if existing is None or hop < existing:
                adjacency[site_id][upstream_site] = hop
    return adjacency


def _reachable_upstream_sites(start_site, upstream_adjacency):
    """沿带权(=stored hop) 上游边做 Dijkstra 累加，得到到各祖先的最小跳数。

    跳数相同时偏向“边数更多”的路径（best 元组的第二维 -edges），以尽量保留物理链路上的中间站点，
    避免被跨站长跳捷径吞掉。返回 {祖先站点: 累计跳数} 与 {站点: 前驱站点}。
    """
    start_site = _normalize_text(start_site)
    best = {start_site: (0, 0)}  # site -> (累计跳数, -经过的边数)，按字典序取最小
    parents = {start_site: None}
    heap = [(0, 0, start_site)]
    while heap:
        dist, neg_edges, current = heapq.heappop(heap)
        if (dist, neg_edges) != best.get(current):
            continue
        for upstream_site, weight in sorted(upstream_adjacency.get(current, {}).items()):
            candidate = (dist + weight, neg_edges - 1)
            if upstream_site not in best or candidate < best[upstream_site]:
                best[upstream_site] = candidate
                parents[upstream_site] = current
                heapq.heappush(heap, (candidate[0], candidate[1], upstream_site))
    hops = {site_id: value[0] for site_id, value in best.items()}
    return hops, parents


def _chain_sites(start_site, target_site, parents):
    """根据 Dijkstra 前驱还原 start_site -> target_site 链路上经过的全部站点。"""
    chain = []
    current = target_site
    while current is not None:
        chain.append(current)
        if current == start_site:
            break
        current = parents.get(current)
    return list(reversed(chain))


def _closure_upstream_hops(site_id, site_chain_index):
    """非 restrict 模式下 upstream_site_hops 已是传递闭包，直接读取（含自身 hop 0）。"""
    site_id = _normalize_text(site_id)
    hops = {site_id: 0}
    info = site_chain_index.get(site_id, {}) if isinstance(site_chain_index, dict) else {}
    for upstream_site, hop in (info.get("upstream_site_hops") or {}).items():
        upstream_site = _normalize_text(upstream_site)
        if upstream_site:
            hops[upstream_site] = min(hop, hops.get(upstream_site, hop))
    return hops


def _reachable_hops_from_site(site_id, site_chain_index, restrict_relation, upstream_adjacency):
    if restrict_relation:
        hops, _parents = _reachable_upstream_sites(site_id, upstream_adjacency or {})
        return hops
    return _closure_upstream_hops(site_id, site_chain_index)


def _promote_to_data_ancestor(
    ancestor_site,
    allowed_ancestor_sites,
    data_site_ids,
    site_chain_index,
    restrict_relation,
    upstream_adjacency,
):
    ancestor_site = _normalize_text(ancestor_site)
    data_site_ids = {_normalize_text(site_id) for site_id in (data_site_ids or ()) if _normalize_text(site_id)}
    if not ancestor_site or not data_site_ids:
        return ancestor_site, None
    if ancestor_site in data_site_ids:
        return ancestor_site, None

    allowed_ancestor_sites = {
        _normalize_text(site_id) for site_id in (allowed_ancestor_sites or ()) if _normalize_text(site_id)
    }
    upstream_hops = _reachable_hops_from_site(
        ancestor_site,
        site_chain_index,
        restrict_relation,
        upstream_adjacency,
    )
    candidates = sorted(
        site_id
        for site_id in (set(upstream_hops) & allowed_ancestor_sites & data_site_ids)
        if site_id != ancestor_site
    )
    if not candidates:
        return ancestor_site, None

    promoted_site = min(candidates, key=lambda site_id: (upstream_hops[site_id], site_id))
    return promoted_site, {
        "from_site_id": ancestor_site,
        "to_site_id": promoted_site,
        "upstream_hop": upstream_hops[promoted_site],
    }


def _build_site_completion(
    alarm_sites,
    site_chain_index,
    restrict_relation,
    upstream_adjacency=None,
    data_site_ids=None,
):
    alarm_sites = sorted({_normalize_text(site) for site in alarm_sites if _normalize_text(site)})
    data_site_ids = {_normalize_text(site) for site in (data_site_ids or ()) if _normalize_text(site)}
    selected_sites = set(alarm_sites)

    # restrict-relation 把 upstream_site_hops 裁成了直接邻居，需要沿带权边累加还原完整上游链
    # （并补全中间站点）；非 restrict 模式下闭包已完整，保持原有“只取最低公共祖先”的行为不变。
    reach_by_site = {}
    parents_by_site = {}
    if restrict_relation:
        if upstream_adjacency is None:
            upstream_adjacency = _build_weighted_upstream_adjacency(site_chain_index)
        for site_id in alarm_sites:
            reach_by_site[site_id], parents_by_site[site_id] = _reachable_upstream_sites(
                site_id, upstream_adjacency
            )
    else:
        for site_id in alarm_sites:
            reach_by_site[site_id] = _closure_upstream_hops(site_id, site_chain_index)
            parents_by_site[site_id] = None

    # 完全没有 upstream 的源站不参与公共 upstream 求交，避免孤立站点把其余
    # 有 upstream 源站本可得到的公共候选交集拉成空集。
    common_upstream_source_sites = []
    no_upstream_sites = []
    for site_id in alarm_sites:
        actual_upstream_sites = set(reach_by_site[site_id]) - {site_id}
        if actual_upstream_sites:
            common_upstream_source_sites.append(site_id)
        else:
            no_upstream_sites.append(site_id)

    common_candidates = None
    for site_id in common_upstream_source_sites:
        candidates = set(reach_by_site[site_id])
        common_candidates = candidates if common_candidates is None else common_candidates & candidates
    common_candidates = common_candidates or set()

    common_upstream_site = None
    common_upstream_sites = []
    common_upstream_hops = {}
    common_upstream_hops_by_site = {}
    farthest_upstream_sites = {}
    no_upstream_data_self_fallback_site_ids = []
    no_upstream_non_data_excluded_site_ids = []
    intermediate_site_chains = {}
    intermediate_site_chains_by_target = {}
    data_ancestor_promotions = []
    data_ancestor_missing_site_ids = []

    def _remember_data_ancestor(site_id):
        site_id = _normalize_text(site_id)
        if site_id and site_id not in data_site_ids:
            _append_unique(data_ancestor_missing_site_ids, site_id)

    def _promote_selected_ancestor(ancestor_site, allowed_ancestor_sites):
        promoted_site, promotion = _promote_to_data_ancestor(
            ancestor_site,
            allowed_ancestor_sites,
            data_site_ids,
            site_chain_index,
            restrict_relation,
            upstream_adjacency,
        )
        if promotion:
            _append_unique(data_ancestor_promotions, promotion)
        _remember_data_ancestor(promoted_site)
        return promoted_site

    def _select_path(site_id, target_site):
        # restrict 模式沿直接边补全中间站点；非 restrict 模式维持原行为，只纳入目标祖先站点。
        if restrict_relation:
            chain = _chain_sites(site_id, target_site, parents_by_site[site_id])
            intermediate_site_chains.setdefault(site_id, chain)
            intermediate_site_chains_by_target.setdefault(target_site, {})[site_id] = chain
            selected_sites.update(chain)
        else:
            selected_sites.add(target_site)

    if common_candidates:
        def _common_rank(candidate):
            return (
                sum(reach_by_site[site_id][candidate] for site_id in common_upstream_source_sites),
                max(reach_by_site[site_id][candidate] for site_id in common_upstream_source_sites),
            )

        best_common_rank = min(_common_rank(candidate) for candidate in common_candidates)
        lowest_common_sites = sorted(
            candidate for candidate in common_candidates if _common_rank(candidate) == best_common_rank
        )
        for lowest_common_site in lowest_common_sites:
            if lowest_common_site not in common_upstream_sites:
                common_upstream_sites.append(lowest_common_site)
                common_upstream_hops_by_site[lowest_common_site] = {
                    site_id: reach_by_site[site_id][lowest_common_site]
                    for site_id in common_upstream_source_sites
                }
            router_site = _promote_selected_ancestor(lowest_common_site, common_candidates)
            for site_id in common_upstream_source_sites:
                _select_path(site_id, lowest_common_site)
                if router_site != lowest_common_site and router_site in reach_by_site[site_id]:
                    _select_path(site_id, router_site)
        common_upstream_site = common_upstream_sites[0] if common_upstream_sites else None
        common_upstream_hops = (
            common_upstream_hops_by_site.get(common_upstream_site, {})
            if common_upstream_site
            else {}
        )
    else:
        for site_id in alarm_sites:
            # 有 upstream 时选择最远 upstream；没有 upstream 时，只有包含 Data 设备的
            # 源站才允许回退到自身（hop=0），非 Data 源站不加入回退结果。
            hops = {
                upstream_site: hop
                for upstream_site, hop in reach_by_site[site_id].items()
                if upstream_site != site_id
            }
            if not hops and site_id not in data_site_ids:
                _append_unique(no_upstream_sites, site_id)
                _append_unique(no_upstream_non_data_excluded_site_ids, site_id)
                continue
            if not hops:
                hops = {site_id: 0}
                _append_unique(no_upstream_data_self_fallback_site_ids, site_id)
            max_hop = max(hops.values())
            farthest_site = min(candidate for candidate, hop in hops.items() if hop == max_hop)
            farthest_upstream_sites[site_id] = {
                "site_id": farthest_site,
                "hop": max_hop,
            }
            if farthest_site == site_id:
                farthest_upstream_sites[site_id]["self_fallback"] = True
            router_site = _promote_selected_ancestor(farthest_site, set(hops))
            if router_site != farthest_site:
                farthest_upstream_sites[site_id]["router_ancestor_site_id"] = router_site
                farthest_upstream_sites[site_id]["router_ancestor_hop"] = reach_by_site[site_id].get(
                    router_site,
                    max_hop,
                )
                farthest_upstream_sites[site_id]["router_promoted"] = True
            _select_path(site_id, farthest_site)
            if router_site != farthest_site and router_site in reach_by_site[site_id]:
                _select_path(site_id, router_site)

    return {
        "selected_sites": selected_sites,
        "common_upstream_site": common_upstream_site,
        "common_upstream_sites": common_upstream_sites,
        "common_upstream_hops": common_upstream_hops,
        "common_upstream_hops_by_site": common_upstream_hops_by_site,
        "common_upstream_source_site_ids": common_upstream_source_sites,
        "common_upstream_excluded_no_upstream_site_ids": sorted(
            set(alarm_sites) - set(common_upstream_source_sites)
        ),
        "farthest_upstream_sites": farthest_upstream_sites,
        "no_upstream_sites": sorted(no_upstream_sites),
        "no_upstream_data_self_fallback_site_ids": sorted(
            no_upstream_data_self_fallback_site_ids
        ),
        "no_upstream_non_data_excluded_site_ids": sorted(
            no_upstream_non_data_excluded_site_ids
        ),
        "upstream_site_hops": reach_by_site,
        "intermediate_site_chains": intermediate_site_chains,
        "intermediate_site_chains_by_target": intermediate_site_chains_by_target,
        "data_ancestor_promotions": data_ancestor_promotions,
        "data_ancestor_missing_site_ids": sorted(data_ancestor_missing_site_ids),
        "restrict_relation": restrict_relation,
    }


def _build_ran_data_upstream_highlight_sites(
    source_site_ids,
    site_has_data,
    site_links,
    directed_edge_types,
    site_chain_components,
    diagnostics=None,
):
    site_has_data = set(site_has_data or ())
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    source_evaluations = {
        item.get("site_id"): item
        for item in diagnostics.get("source_evaluations") or []
        if isinstance(item, dict) and item.get("site_id")
    }
    data_neighbors_by_site = {}
    for site_id in source_site_ids:
        site_id = _normalize_text(site_id)
        if not site_id or site_id in site_has_data:
            continue
        data_neighbor_sites = [
            peer_site
            for peer_site in sorted((site_links or {}).get(site_id, ()))
            if peer_site in site_has_data
            and ("Ran", "Data") in (directed_edge_types or {}).get((site_id, peer_site), set())
        ]
        evaluation = source_evaluations.get(site_id)
        if evaluation is not None:
            evaluation["ran_data_neighbor_site_ids"] = data_neighbor_sites
            evaluation["site_chain_component_id"] = (site_chain_components or {}).get(site_id)
        if data_neighbor_sites:
            data_neighbors_by_site[site_id] = data_neighbor_sites

    # 至少两个经 site_chains 连通的候选源站必须共同连接到同一个 Data 站点；
    # 各自连接不同 Data 站点或只有一个源站连接到该 Data 站点时不补标。
    site_chain_components = site_chain_components or {}
    sources_by_component_and_data_site = defaultdict(set)
    sources_by_data_site = defaultdict(set)
    for site_id, data_neighbor_sites in data_neighbors_by_site.items():
        component = site_chain_components.get(site_id)
        for peer_site in data_neighbor_sites:
            sources_by_data_site[peer_site].add(site_id)
            if component is None:
                continue
            sources_by_component_and_data_site[(component, peer_site)].add(site_id)

    marks_by_site = {}
    qualified_source_site_ids = set()
    for (_component, peer_site), source_sites in sorted(sources_by_component_and_data_site.items()):
        if len(source_sites) < 2:
            continue
        qualified_source_site_ids.update(source_sites)
        mark = marks_by_site.setdefault(peer_site, {
            "site_id": peer_site,
            "role": "ran_data_upstream_site",
            "label": "Ran-Data 相邻 Data 站点",
            "source_sites": [],
        })
        mark["source_sites"].extend(source_sites)
    result = []
    for peer_site in sorted(marks_by_site):
        mark = marks_by_site[peer_site]
        mark["source_sites"] = sorted(set(mark["source_sites"]))
        result.append(mark)

    shared_data_site_evaluations = []
    for data_site_id in sorted(sources_by_data_site):
        source_sites = sorted(sources_by_data_site[data_site_id])
        component_groups = []
        for (component, peer_site), component_source_sites in sorted(
            sources_by_component_and_data_site.items()
        ):
            if peer_site != data_site_id:
                continue
            component_groups.append({
                "component_id": component,
                "source_site_ids": sorted(component_source_sites),
                "passes": len(component_source_sites) >= 2,
            })
        shared_data_site_evaluations.append({
            "data_site_id": data_site_id,
            "source_site_ids": source_sites,
            "component_groups": component_groups,
            "passes": any(group["passes"] for group in component_groups),
        })

    for site_id in source_site_ids:
        evaluation = source_evaluations.get(site_id)
        if evaluation is None:
            continue
        neighbor_sites = data_neighbors_by_site.get(site_id, [])
        component = site_chain_components.get(site_id)
        if not neighbor_sites:
            evaluation["result"] = "no_ran_data_neighbor"
        elif component is None:
            evaluation["result"] = "missing_site_chain_component"
        elif site_id in qualified_source_site_ids:
            evaluation["result"] = "qualified"
        elif any(len(sources_by_data_site[peer_site]) >= 2 for peer_site in neighbor_sites):
            evaluation["result"] = "shared_data_neighbor_different_components"
        else:
            evaluation["result"] = "data_neighbor_not_shared"

    diagnostics["eligible_source_site_ids"] = sorted(set(source_site_ids))
    diagnostics["sources_with_ran_data_neighbor_site_ids"] = sorted(data_neighbors_by_site)
    diagnostics["shared_data_site_evaluations"] = shared_data_site_evaluations
    diagnostics["generated_highlight_site_ids"] = [item["site_id"] for item in result]
    if result:
        diagnostics["status"] = "produced"
    elif len(set(source_site_ids)) < 2:
        diagnostics["status"] = "insufficient_eligible_source_sites"
    elif len(data_neighbors_by_site) < 2:
        diagnostics["status"] = "insufficient_sources_with_ran_data_neighbor"
    else:
        diagnostics["status"] = "no_shared_data_site_in_same_component"
    return result


def _build_topology_highlight_sites(
    completion,
    site_has_data,
    site_links,
    directed_edge_types,
    site_chain_components,
    ran_data_diagnostics=None,
):
    common_upstream_sites = list(completion.get("common_upstream_sites") or [])
    common_upstream_site = completion.get("common_upstream_site")
    if common_upstream_site and common_upstream_site not in common_upstream_sites:
        common_upstream_sites.insert(0, common_upstream_site)
    result = []
    if common_upstream_sites:
        promotions_by_source = defaultdict(list)
        for promotion in completion.get("data_ancestor_promotions") or []:
            if isinstance(promotion, dict) and promotion.get("from_site_id"):
                promotions_by_source[promotion.get("from_site_id")].append(promotion)
        hops_by_target = completion.get("common_upstream_hops_by_site") or {}
        missing_data_ancestor_site_ids = set(completion.get("data_ancestor_missing_site_ids") or [])
        for site_id in common_upstream_sites:
            item = {
                "site_id": site_id,
                "role": "common_upstream_site",
                "label": (
                    "最低公共祖先站点（未找到上游路由站点）"
                    if site_id in missing_data_ancestor_site_ids
                    else "最低公共祖先站点"
                ),
                "hops_by_source_site": hops_by_target.get(site_id, completion.get("common_upstream_hops", {})),
            }
            promotions = promotions_by_source.get(site_id, [])
            if promotions:
                item["router_ancestor_site_ids"] = sorted({
                    _normalize_text(promotion.get("to_site_id", ""))
                    for promotion in promotions
                    if _normalize_text(promotion.get("to_site_id", ""))
                })
                item["router_promoted"] = True
                item["router_promotion_hop"] = min(
                    promotion.get("upstream_hop")
                    for promotion in promotions
                    if promotion.get("upstream_hop") is not None
                )
            result.append(item)

    else:
        farthest_by_target = {}
        for source_site, selected in (completion.get("farthest_upstream_sites") or {}).items():
            if not isinstance(selected, dict):
                continue
            target_site = _normalize_text(selected.get("site_id", ""))
            if not target_site:
                continue
            item = farthest_by_target.setdefault(target_site, {
                "site_id": target_site,
                "role": "farthest_upstream_site",
                "label": "最远 upstream 站点",
                "source_sites": [],
                "hops_by_source_site": {},
            })
            item["source_sites"].append(source_site)
            item["hops_by_source_site"][source_site] = selected.get("hop")
            if selected.get("self_fallback"):
                item.setdefault("self_fallback_source_sites", []).append(source_site)
            if selected.get("router_promoted"):
                item["router_promoted"] = True
                item.setdefault("router_ancestor_site_ids", [])
                item["router_ancestor_site_ids"].append(
                    _normalize_text(selected.get("router_ancestor_site_id", ""))
                )

        for site_id in sorted(farthest_by_target):
            item = farthest_by_target[site_id]
            item["source_sites"] = sorted(set(item["source_sites"]))
            if item.get("self_fallback_source_sites"):
                item["self_fallback_source_sites"] = sorted(
                    set(item["self_fallback_source_sites"])
                )
            if item.get("router_ancestor_site_ids"):
                item["router_ancestor_site_ids"] = sorted({
                    site_id for site_id in item["router_ancestor_site_ids"] if site_id
                })
            result.append(item)
    # 非 Data 告警源站只要完整 upstream 闭包中不存在 Data 站点，就可用于触发
    # Ran-Data 相邻 Data 站点补标；允许其 upstream 中存在其他非 Data 站点。
    normalized_data_site_ids = {
        _normalize_text(site_id)
        for site_id in site_has_data or ()
        if _normalize_text(site_id)
    }
    source_site_ids = []
    source_evaluations = []
    for site_id, upstream_hops in (completion.get("upstream_site_hops") or {}).items():
        site_id = _normalize_text(site_id)
        if not site_id:
            continue
        upstream_site_ids = {
            _normalize_text(upstream_site_id)
            for upstream_site_id in (upstream_hops or {})
            if _normalize_text(upstream_site_id) and _normalize_text(upstream_site_id) != site_id
        }
        data_upstream_site_ids = sorted(upstream_site_ids & normalized_data_site_ids)
        evaluation = {
            "site_id": site_id,
            "is_data_site": site_id in normalized_data_site_ids,
            "upstream_site_ids": sorted(upstream_site_ids),
            "data_upstream_site_ids": data_upstream_site_ids,
            "eligible": False,
            "ran_data_neighbor_site_ids": [],
            "site_chain_component_id": (site_chain_components or {}).get(site_id),
        }
        if site_id in normalized_data_site_ids:
            evaluation["result"] = "source_is_data_site"
        elif data_upstream_site_ids:
            evaluation["result"] = "has_data_upstream"
        else:
            evaluation["eligible"] = True
            source_site_ids.append(site_id)
        source_evaluations.append(evaluation)
    if isinstance(ran_data_diagnostics, dict):
        ran_data_diagnostics.update({
            "status": "evaluating",
            "source_evaluations": source_evaluations,
        })
    result.extend(_build_ran_data_upstream_highlight_sites(
        source_site_ids,
        site_has_data,
        site_links,
        directed_edge_types,
        site_chain_components,
        ran_data_diagnostics,
    ))
    return result


def _filter_hub_highlight_sites(highlight_sites, site_graph_data, site_has_data):
    # 只保留 hub 站点或含 Data 设备的站点。
    site_has_data = set(site_has_data or ())
    kept = []
    removed_site_ids = []
    for item in highlight_sites or []:
        if not isinstance(item, dict):
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if (
            site_id
            and site_id not in site_has_data
            and not _site_is_hub(site_id, site_graph_data)
        ):
            removed_site_ids.append(site_id)
            continue
        kept.append(item)
    return kept, sorted(set(removed_site_ids))


def _filter_ran_without_data_link_highlight_sites(highlight_sites, site_has_data, site_has_ran, site_links):
    kept = []
    removed_site_ids = []
    site_has_data = set(site_has_data or ())
    site_has_ran = set(site_has_ran or ())
    for item in highlight_sites or []:
        if not isinstance(item, dict):
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if (
            site_id
            and site_id not in site_has_data
            and site_id in site_has_ran
            and not (set(site_links.get(site_id, ())) & site_has_data)
        ):
            removed_site_ids.append(site_id)
            continue
        kept.append(item)
    return kept, sorted(set(removed_site_ids))


def _filter_to_single_data_ancestor_highlight_site(highlight_sites, site_has_data):
    highlight_sites = list(highlight_sites or [])
    site_has_data = set(site_has_data or ())
    data_ancestor_site_ids = {
        _normalize_text(item.get("site_id", ""))
        for item in highlight_sites
        if isinstance(item, dict) and _normalize_text(item.get("site_id", "")) in site_has_data
    }
    if len(data_ancestor_site_ids) != 1:
        return highlight_sites, []

    keep_site_id = next(iter(data_ancestor_site_ids))
    removed_site_ids = []
    kept = []
    for item in highlight_sites:
        if not isinstance(item, dict):
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if site_id and site_id != keep_site_id:
            removed_site_ids.append(site_id)
            continue
        kept.append(item)
    return kept, sorted(set(removed_site_ids))


def _edge_type_rank(edge_type):
    role_rank = {"Data": 3, "Microwave": 2, "Ran": 1, "Other": 0}
    return tuple(sorted((role_rank.get(role, 0) for role in edge_type)))


def _data_to_ancestor_edge_score(data_site, ancestor_site, directed_edge_types):
    edge_types = {
        edge_type for edge_type in directed_edge_types.get((data_site, ancestor_site), set())
        if "Other" not in edge_type
    }
    ranked_types = tuple(sorted((_edge_type_rank(edge_type) for edge_type in edge_types), reverse=True))
    # 连边种类数优先（与共享 Data 邻站的连接种类越多越紧密），强度排序作平手破除。
    return (len(edge_types), ranked_types)


def _shared_data_neighbor_winner(left_site, right_site, common_data_sites, directed_edge_types):
    winners = set()
    for data_site in common_data_sites:
        left_score = _data_to_ancestor_edge_score(data_site, left_site, directed_edge_types)
        right_score = _data_to_ancestor_edge_score(data_site, right_site, directed_edge_types)
        if left_score == right_score:
            continue
        winners.add(left_site if left_score > right_score else right_site)
    if len(winners) == 1:
        return next(iter(winners))
    return None


def _postprocess_data_linked_ancestor_sites(highlight_sites, site_has_data, site_links, directed_edge_types):
    if len(highlight_sites or []) <= 1:
        return list(highlight_sites or []), [], []

    highlight_site_ids = {
        _normalize_text(item.get("site_id", ""))
        for item in highlight_sites
        if isinstance(item, dict) and _normalize_text(item.get("site_id", ""))
    }
    if len(highlight_site_ids) <= 1:
        return list(highlight_sites or []), [], []

    removed_site_ids = set()
    for site_id in sorted(highlight_site_ids):
        if site_id in removed_site_ids:
            continue
        site_is_data = site_id in site_has_data
        for peer_site in sorted(site_links.get(site_id, ())):
            if peer_site not in highlight_site_ids or peer_site in removed_site_ids:
                continue
            peer_is_data = peer_site in site_has_data
            if site_is_data == peer_is_data:
                continue
            removed_site_ids.add(peer_site if site_is_data else site_id)
            break

    shared_data_removed_site_ids = set()
    # 含 Data 设备的 highlight 站点视为高一级，不参与共享 Data 邻站比较、也不会被剪。
    remaining_site_ids = highlight_site_ids - removed_site_ids
    remaining_non_data_sites = sorted(site_id for site_id in remaining_site_ids if site_id not in site_has_data)
    for index, left_site in enumerate(remaining_non_data_sites):
        if left_site in shared_data_removed_site_ids:
            continue
        for right_site in remaining_non_data_sites[index + 1:]:
            if right_site in shared_data_removed_site_ids:
                continue
            common_data_sites = sorted(
                site for site in (site_links.get(left_site, set()) & site_links.get(right_site, set()))
                if site in site_has_data
            )
            if not common_data_sites:
                continue
            winner_site = _shared_data_neighbor_winner(
                left_site,
                right_site,
                common_data_sites,
                directed_edge_types,
            )
            if not winner_site:
                continue
            if winner_site == left_site:
                # left 胜出时继续与其余站点比较，否则后续较弱站点会因提前 break 漏剪。
                shared_data_removed_site_ids.add(right_site)
            else:
                shared_data_removed_site_ids.add(left_site)
                break

    all_removed_site_ids = removed_site_ids | shared_data_removed_site_ids
    if not all_removed_site_ids:
        return list(highlight_sites or []), [], []

    return [
        item for item in highlight_sites
        if _normalize_text(item.get("site_id", "")) not in all_removed_site_ids
    ], sorted(removed_site_ids), sorted(shared_data_removed_site_ids)


def _stringify_like_js(value):
    # 对齐标注工具 JS 的字符串化：整数值的 float 不带小数点。
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _alarm_root_cause_key(alarm, index):
    """与 ne_propagation_labeling.html 的 getAlarmRootCauseKey 保持一致。"""
    alarm_id = alarm.get("alarm_id") or alarm.get("eid")
    if alarm_id:
        return "id:" + _stringify_like_js(alarm_id)
    alarm_type = alarm.get("alarm_type") or alarm.get("title") or ""
    ticket = alarm.get("工单号") or ""
    ts = alarm.get("ts") or alarm.get("alarm_time") or ""
    return (
        "k:" + _stringify_like_js(alarm_type)
        + "|" + _stringify_like_js(ticket)
        + "|" + _stringify_like_js(ts)
        + "|#" + str(index)
    )


def _pick_site_root_cause(site_id, ne_ids, ne_info, ne_graph_data, site_chain_index):
    """按规则为标记站点挑根因：最早非断站告警 > 最早断站告警 > 下游连接最多的 Transmission 设备。"""
    non_offline_records = []
    offline_records = []
    for ne_id in ne_ids:
        entry = ne_info.get(ne_id) or {}
        for index, alarm in enumerate(entry.get("alarm") or []):
            if not isinstance(alarm, dict):
                continue
            ts = _first_record_timestamp(alarm, OFFLINE_START_TIME_FIELDS)
            # 无时间戳的告警排在有时间戳的之后；ne_id/下标兜底保证结果稳定。
            sort_key = (ts is None, ts if ts is not None else 0.0, ne_id, index)
            record = (sort_key, ne_id, index, alarm)
            if _record_has_offline_alarm(alarm):
                offline_records.append(record)
            else:
                non_offline_records.append(record)
    for records, kind in (
        (non_offline_records, "non_offline_alarm"),
        (offline_records, "offline_alarm"),
    ):
        if records:
            _sort_key, ne_id, index, alarm = min(records)
            return ne_id, kind, _alarm_root_cause_key(alarm, index)

    # 无告警：标和下游站点（site_chains）连接最多的 Transmission 设备，平局取 ne_id 最小的。
    downstream_sites = set(
        (site_chain_index.get(site_id) or {}).get("downstream_site_hops") or ()
    )
    best = None
    for ne_id in sorted(ne_ids):
        if _device_role(ne_info.get(ne_id) or {}) != "Microwave":
            continue
        raw_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
        links = raw_info.get("link", {}) if isinstance(raw_info, dict) else {}
        connected_downstream_sites = set()
        for peer_ne in links or ():
            peer_site = _site_of_ne(peer_ne, ne_graph_data)
            if peer_site and peer_site in downstream_sites:
                connected_downstream_sites.add(peer_site)
        if best is None or len(connected_downstream_sites) > best[0]:
            best = (len(connected_downstream_sites), ne_id)
    if best is not None:
        return best[1], "transmission_device", None
    return None


def _annotate_root_cause_for_highlight_sites(
    group,
    highlight_sites,
    ne_info,
    ne_graph_data,
    site_chain_index,
):
    """为最终标记候选站点预填根因标注；输入已有人工标注时不覆盖。"""
    existing = group.get("root_cause_annotations")
    if isinstance(existing, dict) and existing:
        return []

    ne_ids_by_site = defaultdict(list)
    for ne_id, entry in ne_info.items():
        if isinstance(entry, dict):
            entry_site_id = _normalize_text(entry.get("site_id", ""))
            if entry_site_id:
                ne_ids_by_site[entry_site_id].append(ne_id)

    annotations = {}
    summary = []
    for item in highlight_sites or []:
        if not isinstance(item, dict):
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if not site_id:
            continue
        picked = _pick_site_root_cause(
            site_id,
            ne_ids_by_site.get(site_id) or [],
            ne_info,
            ne_graph_data,
            site_chain_index,
        )
        if not picked:
            continue
        ne_id, kind, alarm_key = picked
        annotation = annotations.setdefault(ne_id, {"device": False, "alarms": {}})
        if alarm_key:
            annotation["alarms"][alarm_key] = True
        else:
            annotation["device"] = True
        summary.append({
            "site_id": site_id,
            "ne_id": ne_id,
            "kind": kind,
            "alarm_key": alarm_key,
        })
    if annotations:
        group["root_cause_annotations"] = annotations
    return summary


def _ancestor_highlight_count(completion):
    highlight_sites = completion.get("highlight_sites") or []
    ancestor_roles = {
        "common_upstream_site",
        "farthest_upstream_site",
        "ran_data_upstream_site",
    }
    ancestor_site_ids = {
        _normalize_text(item.get("site_id", ""))
        for item in highlight_sites
        if isinstance(item, dict) and item.get("role") in ancestor_roles and _normalize_text(item.get("site_id", ""))
    }
    return len(ancestor_site_ids)


def _blocked_ancestor_site_ids(completion):
    blocked_site_ids = {_normalize_text(site_id) for site_id in BLOCKED_ANCESTOR_SITE_IDS}
    ancestor_site_ids = set()
    ancestor_roles = {
        "common_upstream_site",
        "farthest_upstream_site",
        "ran_data_upstream_site",
    }
    for item in completion.get("highlight_sites") or []:
        if not isinstance(item, dict) or item.get("role") not in ancestor_roles:
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if site_id:
            ancestor_site_ids.add(site_id)

    return sorted(ancestor_site_ids & blocked_site_ids)


def _build_filtered_link_info(
    ne_id,
    included_ne_ids,
    ne_graph_data,
    site_graph_data,
    group_site_by_ne,
):
    info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    links = info.get("link", {}) if isinstance(info, dict) else {}
    if not isinstance(links, dict):
        return {}
    included_ne_ids = set(included_ne_ids)
    source_site = _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    result = {}
    for target_ne, link_meta in sorted(links.items()):
        if target_ne not in included_ne_ids or target_ne == ne_id:
            continue
        target_site = _site_of_ne(target_ne, ne_graph_data, group_site_by_ne)
        distance = _site_link_distance_km(source_site, target_site, site_graph_data)
        result[target_ne] = _format_link_meta(link_meta, distance)
    return result


def _build_ne_info_entry(
    ne_id,
    group,
    included_ne_ids,
    alarm_ne_ids,
    ne_graph_data,
    site_graph_data,
    group_site_by_ne,
):
    existing = {}
    if isinstance(group.get("ne_info"), dict) and isinstance(group["ne_info"].get(ne_id), dict):
        existing = copy.deepcopy(group["ne_info"][ne_id])
    raw_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    if not isinstance(raw_info, dict):
        raw_info = {}
    site_id = _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    site_ctx = _site_context(site_id, site_graph_data, raw_info)
    is_alarm_ne = ne_id in set(alarm_ne_ids)
    entry = {
        "link": _build_filtered_link_info(
            ne_id,
            included_ne_ids,
            ne_graph_data,
            site_graph_data,
            group_site_by_ne,
        ),
        "group": group.get("uuid") or group.get("故障组ID") or group.get("match_info", {}).get("uuid", ""),
        "name": raw_info.get("name", existing.get("name", ne_id)),
        "site_id": site_id or existing.get("site_id", ""),
        "site_name": site_ctx["site_name"] or existing.get("site_name", ""),
        "site_type": site_ctx["site_type"] or existing.get("site_type", ""),
        "type": str(raw_info.get("type", existing.get("type", ""))).upper(),
        "network_type": str(raw_info.get("network_type", existing.get("network_type", ""))).upper(),
        "manufacturer": str(raw_info.get("manufacturer", existing.get("manufacturer", ""))).upper(),
        "running_status": raw_info.get("running_status", raw_info.get("status", existing.get("running_status", ""))),
        "domain": str(raw_info.get("domain", existing.get("domain", ""))).upper(),
        "region_id": site_ctx["region_id"] or existing.get("region_id", ""),
        "longitude": site_ctx["longitude"] if site_ctx["longitude"] != "" else existing.get("longitude", ""),
        "latitude": site_ctx["latitude"] if site_ctx["latitude"] != "" else existing.get("latitude", ""),
        "alarm": existing.get("alarm", []) if is_alarm_ne else [],
    }
    if not is_alarm_ne:
        entry["topology_added"] = True
    return entry


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _NullProgress:
    def update(self, _stats):
        pass

    def close(self):
        pass


class _TqdmGroupProgress:
    def __init__(self, total):
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc="补齐拓扑", unit="组", dynamic_ncols=True, file=sys.stderr)

    def update(self, stats):
        self._bar.update(1)
        self._bar.set_postfix({"新增设备": stats["added_ne_count"], "公共祖先": stats["common_upstream_group_count"]})

    def close(self):
        self._bar.close()


class _StderrGroupProgress:
    def __init__(self, total):
        self.total = max(int(total), 0)
        self.current = 0
        self.start_time = time.time()
        self._render({"added_ne_count": 0, "common_upstream_group_count": 0}, force=True)

    def update(self, stats):
        self.current += 1
        self._render(stats)

    def close(self):
        self._render({"added_ne_count": "", "common_upstream_group_count": ""}, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, stats, force=False):
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = self.current / elapsed
        if self.total > 0:
            percent = min(self.current / self.total, 1.0) * 100
            remaining = max(self.total - self.current, 0)
            eta = _format_duration(remaining / rate) if rate > 0 else "00:00"
            message = f"\r补齐拓扑: {self.current}/{self.total} {percent:6.2f}% ({rate:.1f}组/s, ETA {eta})"
        else:
            message = f"\r补齐拓扑: {self.current} ({rate:.1f}组/s)"
        if stats.get("added_ne_count", "") != "":
            message += f" | 新增设备 {stats['added_ne_count']}，公共祖先 {stats['common_upstream_group_count']}"
        sys.stderr.write(message)
        sys.stderr.flush()


def _build_group_progress(input_path, enabled):
    if not enabled:
        return _NullProgress()
    total = _count_jsonl_records(input_path)
    try:
        return _TqdmGroupProgress(total)
    except ImportError:
        return _StderrGroupProgress(total)


def _should_output_by_ancestor_count(completion, ancestor_output):
    ancestor_output = _normalize_text(ancestor_output).lower() or "all"
    if ancestor_output == "all":
        return True

    ancestor_count = _ancestor_highlight_count(completion)
    if ancestor_output == "one":
        return ancestor_count == 1
    if ancestor_output == "multiple":
        return ancestor_count > 1
    raise ValueError(f"未知 ancestor_output: {ancestor_output}")


def complete_group_topology(
    group,
    ne_graph_data,
    site_graph_data,
    site_to_ne_ids,
    site_chain_index,
    restrict_relation=False,
    upstream_adjacency=None,
    site_has_data=None,
    site_has_ran=None,
    site_links=None,
    directed_edge_types=None,
    site_chain_components=None,
):
    group = copy.deepcopy(group)
    group_id = group.get("uuid") or group.get("故障组ID") or group.get("match_info", {}).get("uuid", "")
    group["uuid"] = group_id

    group_site_by_ne = _group_site_by_ne(group)
    alarm_ne_ids = _extract_alarm_ne_ids(group)
    alarm_sites = sorted({
        _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
        for ne_id in alarm_ne_ids
        if _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    })
    offline_alarm_sites = _extract_offline_alarm_site_ids(group, ne_graph_data, group_site_by_ne)
    offline_site_max_durations = _offline_site_max_durations(group, ne_graph_data, group_site_by_ne)
    non_offline_alarm_sites = sorted(set(alarm_sites) - set(offline_alarm_sites))
    if site_has_data is None or site_has_ran is None or site_links is None or directed_edge_types is None:
        site_has_data, site_has_ran, site_links, directed_edge_types = _build_site_data_and_link_index(ne_graph_data)
    if DEBUG_SITE_ID in alarm_sites:
        site_device_domains = {
            ne_id: _ne_domain_text(ne_info)
            for ne_id, ne_info in (
                ne_graph_data.items() if isinstance(ne_graph_data, dict) else ()
            )
            if isinstance(ne_info, dict)
            and _normalize_text(ne_info.get("site_id", "")) == DEBUG_SITE_ID
        }
        print(
            "[topology-debug:data-site] "
            + json.dumps({
                "site_id": DEBUG_SITE_ID,
                "in_data_site_ids": DEBUG_SITE_ID in set(site_has_data or ()),
                "needs_upstream": DEBUG_SITE_ID in set(offline_alarm_sites),
                "device_domains": site_device_domains,
            }, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    if site_chain_components is None:
        site_chain_components = _build_site_chain_component_index(site_chain_index)
    completion = _build_site_completion(
        offline_alarm_sites,
        site_chain_index,
        restrict_relation,
        upstream_adjacency,
        data_site_ids=site_has_data,
    )
    if DEBUG_SITE_ID in offline_alarm_sites:
        raw_site_chain = (
            site_chain_index.get(DEBUG_SITE_ID, {})
            if isinstance(site_chain_index, dict)
            else {}
        )
        resolved_hops = (completion.get("upstream_site_hops") or {}).get(
            DEBUG_SITE_ID, {}
        )
        found_upstream_hops = {
            site_id: hop
            for site_id, hop in resolved_hops.items()
            if site_id != DEBUG_SITE_ID
        }
        print(
            "[topology-debug:upstream] "
            + json.dumps({
                "site_id": DEBUG_SITE_ID,
                "restrict_relation": restrict_relation,
                "raw_upstream_site_hops": raw_site_chain.get("upstream_site_hops", {}),
                "found_upstream_site_hops": found_upstream_hops,
                "no_upstream": not bool(found_upstream_hops),
            }, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    # 上游祖先仍只由断站/Offline 告警站点推断，但只要站点上存在任意告警，
    # 就应把该站点纳入设备展开范围，补齐站内没有告警的设备。
    selected_sites = set(completion["selected_sites"]) | set(alarm_sites)
    ran_data_upstream_diagnostics = {}
    topology_highlight_sites = _build_topology_highlight_sites(
        completion,
        site_has_data,
        site_links,
        directed_edge_types,
        site_chain_components,
        ran_data_upstream_diagnostics,
    )
    topology_highlight_sites, hub_filtered_ancestor_site_ids = _filter_hub_highlight_sites(
        topology_highlight_sites,
        site_graph_data,
        site_has_data,
    )
    (
        topology_highlight_sites,
        ran_without_data_link_filtered_ancestor_site_ids,
    ) = _filter_ran_without_data_link_highlight_sites(
        topology_highlight_sites,
        site_has_data,
        site_has_ran,
        site_links,
    )
    (
        topology_highlight_sites,
        data_link_pruned_ancestor_site_ids,
        shared_data_link_pruned_ancestor_site_ids,
    ) = _postprocess_data_linked_ancestor_sites(
        topology_highlight_sites,
        site_has_data,
        site_links,
        directed_edge_types,
    )
    (
        topology_highlight_sites,
        single_data_ancestor_pruned_site_ids,
    ) = _filter_to_single_data_ancestor_highlight_site(
        topology_highlight_sites,
        site_has_data,
    )
    # Ran-Data 补出的 Data 站点不在 completion 的 selected_sites 里，
    # 只把过滤后仍保留的补进输出拓扑。
    selected_sites.update(
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "ran_data_upstream_site" and item.get("site_id")
    )
    topology_highlight_site_ids = sorted(
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("site_id")
    )
    ran_data_upstream_diagnostics["final_highlight_site_ids"] = sorted(
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "ran_data_upstream_site" and item.get("site_id")
    )
    ran_data_upstream_diagnostics["filtered_out_highlight_site_ids"] = sorted(
        set(ran_data_upstream_diagnostics.get("generated_highlight_site_ids") or ())
        - set(ran_data_upstream_diagnostics["final_highlight_site_ids"])
    )

    pruned_output_site_ids = (
        set(data_link_pruned_ancestor_site_ids)
        | set(shared_data_link_pruned_ancestor_site_ids)
        | set(single_data_ancestor_pruned_site_ids)
    ) - set(alarm_sites)
    selected_sites = set(selected_sites) - pruned_output_site_ids

    included_ne_ids = set()
    for site_id in selected_sites:
        included_ne_ids.update(site_to_ne_ids.get(site_id, ()))
    included_ne_ids.update(alarm_ne_ids)

    all_site_ids = sorted({
        _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
        for ne_id in included_ne_ids
        if _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    })
    ne_info = {
        ne_id: _build_ne_info_entry(
            ne_id,
            group,
            included_ne_ids,
            alarm_ne_ids,
            ne_graph_data,
            site_graph_data,
            group_site_by_ne,
        )
        for ne_id in sorted(included_ne_ids)
    }

    group["ne_info"] = ne_info
    auto_root_cause_annotations = _annotate_root_cause_for_highlight_sites(
        group,
        topology_highlight_sites,
        ne_info,
        ne_graph_data,
        site_chain_index,
    )
    group["group_info"] = {
        group_id: {
            "ne_list": sorted(included_ne_ids),
            "site_list": all_site_ids,
        }
    }

    existing_role_mapping = {}
    if isinstance(group.get("role_mapping"), dict):
        existing_role_mapping.update(copy.deepcopy(group["role_mapping"]))
    match_info = group.get("match_info") if isinstance(group.get("match_info"), dict) else {}
    if isinstance(match_info.get("role_mapping"), dict):
        existing_role_mapping.update(copy.deepcopy(match_info["role_mapping"]))
    for derived_role in (
        "context_site",
        "common_upstream_site",
        "farthest_upstream_site",
        "no_upstream_site",
        "ran_data_upstream_site",
    ):
        existing_role_mapping.pop(derived_role, None)
    alarm_site_set = set(alarm_sites)
    existing_role_mapping["associated_site"] = sorted(alarm_site_set)
    context_sites = sorted(set(all_site_ids) - alarm_site_set)
    if context_sites:
        existing_role_mapping["context_site"] = context_sites
    common_upstream_sites = [
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "common_upstream_site"
    ]
    farthest_upstream_sites = [
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "farthest_upstream_site"
    ]
    ran_data_upstream_sites = [
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "ran_data_upstream_site"
    ]
    if common_upstream_sites:
        existing_role_mapping["common_upstream_site"] = sorted(common_upstream_sites)
    if farthest_upstream_sites:
        existing_role_mapping["farthest_upstream_site"] = sorted(farthest_upstream_sites)
    if ran_data_upstream_sites:
        existing_role_mapping["ran_data_upstream_site"] = sorted(ran_data_upstream_sites)
    group["role_mapping"] = existing_role_mapping

    match_info = copy.deepcopy(match_info)
    match_info.setdefault("uuid", group_id)
    match_info.setdefault("rule", group.get("rule", "alarm_group_id_rule"))
    match_info.setdefault("merged_rules", group.get("merged_rules", ["alarm_group_id_rule"]))
    match_info["role_mapping"] = existing_role_mapping
    group["match_info"] = match_info

    group["topology_completion"] = {
        "mode": "site_upstream_hops",
        "restrict_relation": restrict_relation,
        "original_alarm_ne_ids": sorted(alarm_ne_ids),
        "original_alarm_site_ids": alarm_sites,
        "ancestor_source_site_ids": offline_alarm_sites,
        "non_offline_alarm_site_ids": non_offline_alarm_sites,
        "selected_site_ids": all_site_ids,
        "added_site_ids": context_sites,
        "added_ne_ids": sorted(ne_id for ne_id in included_ne_ids if ne_id not in set(alarm_ne_ids)),
        "common_upstream_site": completion["common_upstream_site"],
        "common_upstream_sites": completion["common_upstream_sites"],
        "common_upstream_hops": completion["common_upstream_hops"],
        "common_upstream_hops_by_site": completion["common_upstream_hops_by_site"],
        "common_upstream_source_site_ids": completion["common_upstream_source_site_ids"],
        "common_upstream_excluded_no_upstream_site_ids": completion[
            "common_upstream_excluded_no_upstream_site_ids"
        ],
        "farthest_upstream_sites": completion["farthest_upstream_sites"],
        "no_upstream_sites": completion["no_upstream_sites"],
        "no_upstream_data_self_fallback_site_ids": completion[
            "no_upstream_data_self_fallback_site_ids"
        ],
        "no_upstream_non_data_excluded_site_ids": completion[
            "no_upstream_non_data_excluded_site_ids"
        ],
        "upstream_site_hops": completion["upstream_site_hops"],
        "intermediate_site_chains": completion["intermediate_site_chains"],
        "intermediate_site_chains_by_target": completion["intermediate_site_chains_by_target"],
        "data_ancestor_promotions": completion["data_ancestor_promotions"],
        "data_ancestor_missing_site_ids": completion["data_ancestor_missing_site_ids"],
        "hub_filtered_ancestor_site_ids": hub_filtered_ancestor_site_ids,
        "ran_without_data_link_filtered_ancestor_site_ids": ran_without_data_link_filtered_ancestor_site_ids,
        "data_link_pruned_ancestor_site_ids": data_link_pruned_ancestor_site_ids,
        "shared_data_link_pruned_ancestor_site_ids": shared_data_link_pruned_ancestor_site_ids,
        "single_data_ancestor_pruned_site_ids": single_data_ancestor_pruned_site_ids,
        "highlight_site_ids": topology_highlight_site_ids,
        "highlight_sites": topology_highlight_sites,
        "ran_data_upstream_diagnostics": ran_data_upstream_diagnostics,
        "auto_root_cause_annotations": auto_root_cause_annotations,
        "site_level_connected": bool(completion["common_upstream_site"]) or len(offline_alarm_sites) <= 1,
        "offline_site_max_duration_seconds": _serialize_offline_durations(offline_site_max_durations),
        "offline_duration_filter": _offline_duration_filter_summary(offline_site_max_durations),
    }
    return group


def _group_uuid(group):
    return (
        _normalize_text(group.get("uuid"))
        or _normalize_text((group.get("match_info") or {}).get("uuid"))
        or _normalize_text(group.get("故障组ID"))
    )


def _safe_filename(name, fallback):
    """把故障组ID转成文件系统安全的文件名（标注工具按文件名实时回写，需稳定且合法）。"""
    text = _normalize_text(name) or fallback
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    text = text.strip().strip(".").strip()  # Windows 不允许以空格或点结尾
    if not text:
        text = fallback
    return text[:120]


def complete_groups(
    input_path,
    output_path,
    ne_graph_path,
    site_graph_path,
    site_chains_path,
    show_progress=True,
    ancestor_output="all",
    per_file=False,
    offline_duration_filter=False,
):
    ne_graph_data = _load_json_object(ne_graph_path, "ne_graph", warn_if_missing=True)
    site_graph_data = _load_json_object(site_graph_path, "site_graph", warn_if_missing=True)
    site_chain_index, restrict_relation = _load_site_chain_index(site_chains_path)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_has_data, site_has_ran, site_links, directed_edge_types = _build_site_data_and_link_index(ne_graph_data)
    # 带权上游邻接只依赖 site_chain_index，与故障组无关，预先构建一次复用，避免逐组重建。
    upstream_adjacency = (
        _build_weighted_upstream_adjacency(site_chain_index) if restrict_relation else None
    )
    site_chain_components = _build_site_chain_component_index(site_chain_index)

    stats = {
        "input_group_count": 0,
        "output_group_count": 0,
        "common_upstream_group_count": 0,
        "fallback_upstream_group_count": 0,
        "one_ancestor_group_count": 0,
        "multiple_ancestor_group_count": 0,
        "skipped_by_ancestor_output_group_count": 0,
        "skipped_by_offline_duration_filter_group_count": 0,
        "skipped_by_blocked_ancestor_site_group_count": 0,
        "skipped_by_missing_device_coordinates_group_count": 0,
        "missing_device_coordinates_count": 0,
        "skipped_by_missing_alarm_topology_group_count": 0,
        "missing_alarm_source_group_count": 0,
        "missing_ne_graph_group_count": 0,
        "missing_ne_site_group_count": 0,
        "missing_site_graph_group_count": 0,
        "added_site_count": 0,
        "added_ne_count": 0,
    }
    progress = _build_group_progress(input_path, show_progress)
    # per_file=True：每个故障组写一个单行 jsonl 文件到 output_path 目录（标注工具 data/ 所需格式，
    # 支持按文件实时回写）；否则保持原行为：所有组流式写入单个多行 jsonl 文件。
    out_dir = None
    out_fh = None
    per_file_used = {}
    per_file_count = 0
    if per_file:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_fh = open(output_path, "w", encoding="utf-8")
    try:
        try:
            for group in _iter_jsonl(input_path):
                stats["input_group_count"] += 1
                completed = complete_group_topology(
                    group,
                    ne_graph_data,
                    site_graph_data,
                    site_to_ne_ids,
                    site_chain_index,
                    restrict_relation,
                    upstream_adjacency,
                    site_has_data,
                    site_has_ran,
                    site_links,
                    directed_edge_types,
                    site_chain_components,
                )
                completion = completed.get("topology_completion", {})
                if completion.get("common_upstream_site"):
                    stats["common_upstream_group_count"] += 1
                elif len(completion.get("ancestor_source_site_ids") or []) > 1:
                    stats["fallback_upstream_group_count"] += 1
                ancestor_count = _ancestor_highlight_count(completion)
                if ancestor_count == 1:
                    stats["one_ancestor_group_count"] += 1
                elif ancestor_count > 1:
                    stats["multiple_ancestor_group_count"] += 1
                if not _should_output_by_ancestor_count(completion, ancestor_output):
                    stats["skipped_by_ancestor_output_group_count"] += 1
                    progress.update(stats)
                    continue
                if offline_duration_filter and not completion.get("offline_duration_filter", {}).get("passes", False):
                    stats["skipped_by_offline_duration_filter_group_count"] += 1
                    progress.update(stats)
                    continue
                if _blocked_ancestor_site_ids(completion):
                    stats["skipped_by_blocked_ancestor_site_group_count"] += 1
                    progress.update(stats)
                    continue
                alarm_topology_check = _check_group_alarm_topology(
                    completed,
                    ne_graph_data,
                    site_graph_data,
                )
                if not alarm_topology_check["ok"]:
                    stats["skipped_by_missing_alarm_topology_group_count"] += 1
                    if alarm_topology_check["missing_alarm_source_count"]:
                        stats["missing_alarm_source_group_count"] += 1
                    if alarm_topology_check["missing_ne_ids"]:
                        stats["missing_ne_graph_group_count"] += 1
                    if alarm_topology_check["missing_site_ne_ids"]:
                        stats["missing_ne_site_group_count"] += 1
                    if alarm_topology_check["missing_site_graph_ids"]:
                        stats["missing_site_graph_group_count"] += 1
                    progress.update(stats)
                    continue
                # 仅在最终写出前检查坐标；拓扑计算和其他筛选逻辑不依赖经纬度。
                missing_coordinate_ne_ids = _missing_coordinate_ne_ids(completed)
                if missing_coordinate_ne_ids:
                    stats["skipped_by_missing_device_coordinates_group_count"] += 1
                    stats["missing_device_coordinates_count"] += len(missing_coordinate_ne_ids)
                    progress.update(stats)
                    continue
                stats["added_site_count"] += len(completion.get("added_site_ids", []))
                stats["added_ne_count"] += len(completion.get("added_ne_ids", []))
                line = json.dumps(completed, ensure_ascii=False, separators=(",", ":"))
                if per_file:
                    base = _safe_filename(_group_uuid(completed), f"group_{per_file_count}")
                    if base in per_file_used:
                        per_file_used[base] += 1
                        name = f"{base}_{per_file_used[base]}"
                    else:
                        per_file_used[base] = 0
                        name = base
                    (out_dir / f"{name}.jsonl").write_text(line + "\n", encoding="utf-8")
                    per_file_count += 1
                else:
                    out_fh.write(line)
                    out_fh.write("\n")
                stats["output_group_count"] += 1
                progress.update(stats)
        finally:
            progress.close()
    finally:
        if out_fh is not None:
            out_fh.close()

    stats["input"] = input_path
    if per_file:
        stats["output_dir"] = output_path
        stats["output_file_count"] = per_file_count
    else:
        stats["output"] = output_path
    stats["per_file"] = per_file
    stats["ne_graph"] = ne_graph_path
    stats["site_graph"] = site_graph_path
    stats["site_chains"] = site_chains_path
    stats["restrict_relation"] = restrict_relation
    stats["ancestor_output"] = ancestor_output
    stats["offline_duration_filter"] = offline_duration_filter
    return stats


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="按站点 upstream_site_hops 信息为故障组补齐站点级拓扑"
    )
    parser.add_argument("input", help="输入故障组 JSONL")
    parser.add_argument(
        "output",
        help="输出位置：默认为单个多行 JSONL 文件；加 --per-file 时为输出目录（每组一个单行 jsonl）",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="每个故障组输出为单独的单行 jsonl 文件到 output 目录（标注工具 data/ 所需格式，支持按文件实时回写）",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"site_graph.json 文件，默认: {resource_display('site_graph.json')}",
    )
    parser.add_argument(
        "--site-chains",
        default=SITE_CHAINS_JSON,
        help=f"site_chains.json 文件，默认: {resource_display('site_chains.json')}",
    )
    parser.add_argument(
        "--ancestor-output",
        choices=("all", "one", "multiple"),
        default="all",
        help=(
            "按补出的祖先站点数量筛选输出："
            "all 输出全部；one 只输出 1 个祖先站点的故障组；"
            "multiple 只输出多个祖先站点的故障组。默认 all"
        ),
    )
    parser.add_argument(
        "--filter",
        dest="offline_duration_filter",
        action="store_true",
        help=(
            "按每站最长 offline 告警持续时间筛选故障组，满足任一情况即保留："
            "用 ≥30 分钟卡后仍剩至少 4 个有 offline 的站点；"
            "或用 ≥15 分钟卡后仍剩至少 10 个；"
            "或用 ≥7 分钟卡后仍剩至少 30 个。未清除的 offline 视为持续中（满足任一阈值）"
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭处理进度输出")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    stats = complete_groups(
        args.input,
        args.output,
        args.ne_graph,
        args.site_graph,
        args.site_chains,
        show_progress=not args.no_progress,
        ancestor_output=args.ancestor_output,
        per_file=args.per_file,
        offline_duration_filter=args.offline_duration_filter,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
