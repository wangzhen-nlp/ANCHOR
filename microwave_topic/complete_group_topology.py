#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按站点 upstream_site_hops 信息补齐故障组拓扑。"""

import argparse
import copy
import heapq
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fault_grouping.site_topology import build_site_to_ne_ids, normalize_site_chain_hops
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_CHAINS_JSON,
    SITE_GRAPH_JSON,
    resource_display,
)


BLOCKED_ANCESTOR_SITE_IDS = {"13PWK0024"}


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


def _format_link_meta(link_meta):
    if isinstance(link_meta, dict):
        connection_types = sorted(str(key) for key in link_meta.keys())
        topologies = sorted({str(value) for value in link_meta.values() if value})
    else:
        connection_types = [str(link_meta)] if link_meta not in (None, "") else []
        topologies = []
    return {
        "connection_type": ",".join(connection_types),
        "distance": "",
        "topology": ",".join(topologies),
        "time_window": "",
        "left_alarm": {},
        "right_alarm": {},
    }


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
            }
    return site_chain_index, restrict_relation


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
    site_links = defaultdict(set)
    directed_edge_types = defaultdict(set)
    ne_to_site = {}
    ne_roles = {}

    if not isinstance(ne_graph_data, dict):
        return site_has_data, site_links, directed_edge_types

    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        site_id = _normalize_text(ne_info.get("site_id", ""))
        if not site_id:
            continue
        ne_to_site[ne_id] = site_id
        ne_roles[ne_id] = _device_role(ne_info)
        if _is_data_ne(ne_info):
            site_has_data.add(site_id)

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

    return site_has_data, site_links, directed_edge_types


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


def _build_site_completion(alarm_sites, site_chain_index, restrict_relation, upstream_adjacency=None):
    alarm_sites = sorted({_normalize_text(site) for site in alarm_sites if _normalize_text(site)})
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

    common_candidates = None
    for site_id in alarm_sites:
        candidates = set(reach_by_site[site_id])
        common_candidates = candidates if common_candidates is None else common_candidates & candidates
    common_candidates = common_candidates or set()

    common_upstream_site = None
    common_upstream_hops = {}
    farthest_upstream_sites = {}
    no_upstream_sites = []
    intermediate_site_chains = {}

    def _select_path(site_id, target_site):
        # restrict 模式沿直接边补全中间站点；非 restrict 模式维持原行为，只纳入目标祖先站点。
        if restrict_relation:
            chain = _chain_sites(site_id, target_site, parents_by_site[site_id])
            intermediate_site_chains[site_id] = chain
            selected_sites.update(chain)
        else:
            selected_sites.add(target_site)

    if common_candidates:
        common_upstream_site = min(
            common_candidates,
            key=lambda candidate: (
                sum(reach_by_site[site_id][candidate] for site_id in alarm_sites),
                max(reach_by_site[site_id][candidate] for site_id in alarm_sites),
                candidate,
            ),
        )
        common_upstream_hops = {
            site_id: reach_by_site[site_id][common_upstream_site]
            for site_id in alarm_sites
        }
        for site_id in alarm_sites:
            _select_path(site_id, common_upstream_site)
    else:
        for site_id in alarm_sites:
            hops = {
                upstream_site: hop
                for upstream_site, hop in reach_by_site[site_id].items()
                if upstream_site != site_id
            }
            if not hops:
                no_upstream_sites.append(site_id)
                continue
            max_hop = max(hops.values())
            farthest_site = min(candidate for candidate, hop in hops.items() if hop == max_hop)
            farthest_upstream_sites[site_id] = {"site_id": farthest_site, "hop": max_hop}
            _select_path(site_id, farthest_site)

    return {
        "selected_sites": selected_sites,
        "common_upstream_site": common_upstream_site,
        "common_upstream_hops": common_upstream_hops,
        "farthest_upstream_sites": farthest_upstream_sites,
        "no_upstream_sites": sorted(no_upstream_sites),
        "upstream_site_hops": reach_by_site,
        "intermediate_site_chains": intermediate_site_chains,
        "restrict_relation": restrict_relation,
    }


def _build_topology_highlight_sites(completion):
    common_upstream_site = completion.get("common_upstream_site")
    if common_upstream_site:
        return [{
            "site_id": common_upstream_site,
            "role": "common_upstream_site",
            "label": "最低公共祖先站点",
            "hops_by_source_site": completion.get("common_upstream_hops", {}),
        }]

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

    result = []
    for site_id in sorted(farthest_by_target):
        item = farthest_by_target[site_id]
        item["source_sites"] = sorted(set(item["source_sites"]))
        result.append(item)
    for site_id in completion.get("no_upstream_sites") or []:
        site_id = _normalize_text(site_id)
        if not site_id:
            continue
        result.append({
            "site_id": site_id,
            "role": "no_upstream_site",
            "label": "无可用 upstream 站点",
        })
    return result


def _filter_hub_highlight_sites(highlight_sites, site_graph_data):
    kept = []
    removed_site_ids = []
    for item in highlight_sites or []:
        if not isinstance(item, dict):
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if site_id and not _site_is_hub(site_id, site_graph_data):
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


def _ancestor_highlight_count(completion):
    highlight_sites = completion.get("highlight_sites") or []
    ancestor_roles = {"common_upstream_site", "farthest_upstream_site", "no_upstream_site"}
    ancestor_site_ids = {
        _normalize_text(item.get("site_id", ""))
        for item in highlight_sites
        if isinstance(item, dict) and item.get("role") in ancestor_roles and _normalize_text(item.get("site_id", ""))
    }
    return len(ancestor_site_ids)


def _blocked_ancestor_site_ids(completion):
    blocked_site_ids = {_normalize_text(site_id) for site_id in BLOCKED_ANCESTOR_SITE_IDS}
    ancestor_site_ids = set()
    ancestor_roles = {"common_upstream_site", "farthest_upstream_site", "no_upstream_site"}
    for item in completion.get("highlight_sites") or []:
        if not isinstance(item, dict) or item.get("role") not in ancestor_roles:
            continue
        site_id = _normalize_text(item.get("site_id", ""))
        if site_id:
            ancestor_site_ids.add(site_id)

    return sorted(ancestor_site_ids & blocked_site_ids)


def _build_filtered_link_info(ne_id, included_ne_ids, ne_graph_data):
    info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    links = info.get("link", {}) if isinstance(info, dict) else {}
    if not isinstance(links, dict):
        return {}
    included_ne_ids = set(included_ne_ids)
    return {
        target_ne: _format_link_meta(link_meta)
        for target_ne, link_meta in sorted(links.items())
        if target_ne in included_ne_ids and target_ne != ne_id
    }


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
        "link": _build_filtered_link_info(ne_id, included_ne_ids, ne_graph_data),
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
    site_links=None,
    directed_edge_types=None,
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
    completion = _build_site_completion(
        alarm_sites, site_chain_index, restrict_relation, upstream_adjacency
    )
    selected_sites = completion["selected_sites"]
    topology_highlight_sites = _build_topology_highlight_sites(completion)
    topology_highlight_sites, hub_filtered_ancestor_site_ids = _filter_hub_highlight_sites(
        topology_highlight_sites,
        site_graph_data,
    )
    if site_has_data is None or site_links is None or directed_edge_types is None:
        site_has_data, site_links, directed_edge_types = _build_site_data_and_link_index(ne_graph_data)
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
    topology_highlight_site_ids = sorted(
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("site_id")
    )

    pruned_output_site_ids = (
        set(data_link_pruned_ancestor_site_ids)
        | set(shared_data_link_pruned_ancestor_site_ids)
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
    for derived_role in ("context_site", "common_upstream_site", "farthest_upstream_site", "no_upstream_site"):
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
    no_upstream_sites = [
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("role") == "no_upstream_site"
    ]
    if common_upstream_sites:
        existing_role_mapping["common_upstream_site"] = sorted(common_upstream_sites)
    if farthest_upstream_sites:
        existing_role_mapping["farthest_upstream_site"] = sorted(farthest_upstream_sites)
    if no_upstream_sites:
        existing_role_mapping["no_upstream_site"] = sorted(no_upstream_sites)
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
        "selected_site_ids": all_site_ids,
        "added_site_ids": context_sites,
        "added_ne_ids": sorted(ne_id for ne_id in included_ne_ids if ne_id not in set(alarm_ne_ids)),
        "common_upstream_site": completion["common_upstream_site"],
        "common_upstream_hops": completion["common_upstream_hops"],
        "farthest_upstream_sites": completion["farthest_upstream_sites"],
        "no_upstream_sites": completion["no_upstream_sites"],
        "upstream_site_hops": completion["upstream_site_hops"],
        "intermediate_site_chains": completion["intermediate_site_chains"],
        "hub_filtered_ancestor_site_ids": hub_filtered_ancestor_site_ids,
        "data_link_pruned_ancestor_site_ids": data_link_pruned_ancestor_site_ids,
        "shared_data_link_pruned_ancestor_site_ids": shared_data_link_pruned_ancestor_site_ids,
        "highlight_site_ids": topology_highlight_site_ids,
        "highlight_sites": topology_highlight_sites,
        "site_level_connected": bool(completion["common_upstream_site"]) or len(alarm_sites) <= 1,
    }
    return group


def complete_groups(
    input_path,
    output_path,
    ne_graph_path,
    site_graph_path,
    site_chains_path,
    show_progress=True,
    ancestor_output="all",
):
    ne_graph_data = _load_json_object(ne_graph_path, "ne_graph", warn_if_missing=True)
    site_graph_data = _load_json_object(site_graph_path, "site_graph", warn_if_missing=True)
    site_chain_index, restrict_relation = _load_site_chain_index(site_chains_path)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_has_data, site_links, directed_edge_types = _build_site_data_and_link_index(ne_graph_data)
    # 带权上游邻接只依赖 site_chain_index，与故障组无关，预先构建一次复用，避免逐组重建。
    upstream_adjacency = (
        _build_weighted_upstream_adjacency(site_chain_index) if restrict_relation else None
    )

    stats = {
        "input_group_count": 0,
        "output_group_count": 0,
        "common_upstream_group_count": 0,
        "fallback_upstream_group_count": 0,
        "one_ancestor_group_count": 0,
        "multiple_ancestor_group_count": 0,
        "skipped_by_ancestor_output_group_count": 0,
        "skipped_by_blocked_ancestor_site_group_count": 0,
        "added_site_count": 0,
        "added_ne_count": 0,
    }
    progress = _build_group_progress(input_path, show_progress)
    with open(output_path, "w", encoding="utf-8") as fw:
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
                    site_links,
                    directed_edge_types,
                )
                completion = completed.get("topology_completion", {})
                if completion.get("common_upstream_site"):
                    stats["common_upstream_group_count"] += 1
                elif len(completion.get("original_alarm_site_ids") or []) > 1:
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
                if _blocked_ancestor_site_ids(completion):
                    stats["skipped_by_blocked_ancestor_site_group_count"] += 1
                    progress.update(stats)
                    continue
                stats["added_site_count"] += len(completion.get("added_site_ids", []))
                stats["added_ne_count"] += len(completion.get("added_ne_ids", []))
                fw.write(json.dumps(completed, ensure_ascii=False, separators=(",", ":")))
                fw.write("\n")
                stats["output_group_count"] += 1
                progress.update(stats)
        finally:
            progress.close()

    stats["input"] = input_path
    stats["output"] = output_path
    stats["ne_graph"] = ne_graph_path
    stats["site_graph"] = site_graph_path
    stats["site_chains"] = site_chains_path
    stats["restrict_relation"] = restrict_relation
    stats["ancestor_output"] = ancestor_output
    return stats


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="按站点 upstream_site_hops 信息为故障组补齐站点级拓扑"
    )
    parser.add_argument("input", help="输入故障组 JSONL")
    parser.add_argument("output", help="输出补齐拓扑后的故障组 JSONL")
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
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
