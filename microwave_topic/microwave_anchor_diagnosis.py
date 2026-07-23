#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微波拓扑根因诊断：diagnose_root_cause_devices(input_json) 返回 (root_cause_resources,
ne_to_alarm_objs)，root_cause_resources 每项形如 {"resourceName": neName, "confidence": 0.9}。

input_json 字段：
  - alarms:            告警（neName、alarmVertexVid、alarmName、发生时间等）
  - resources:         网元资源（resourceVid、neName、domain、siteId 等）
  - resourceRelations: 网元-网元拓扑连边（srcVid/dstVid、linkLayer）
  - happenRelations:   告警顶点 -> owner 资源（srcVid=alarmVertexVid、dstVid=resourceVid），
                       用于把告警定位到 neName；缺失时退回 alarm.neName

推断口径：只用断站告警站点推断；每个非 Data 源站 BFS 找最近 Data 站点，只保留通往它们
的最短路径作为“上游”，不穿越 Data。多源优先最低公共上游，无公共上游时逐站取最远上游。
再在候选站点里挑根因网元（非断站告警 > 传输设备断站告警 > 无告警取下游连接最多的传输
设备 > 任意设备断站告警），以 neName 返回（resourceName）。
"""

import re
from collections import OrderedDict, defaultdict, deque

# 断站告警名称集合：判定一条告警是否“断站/掉线”，_pick_site_root_cause 据此区分断站/非断站。
OFFLINE_ALARMS = {
    "BASE STATION FAULTY",
    "BCF FAULTY",
    "BN EMS Alarm NE Communication Failure",
    "BTS Down",
    "BTS O&M LINK FAILURE",
    "Communication FAIL",
    "Ericsson 2G NE Down",
    "Ericsson 4G NE Down",
    "Ericsson 4G S1 NE Down",
    "Heartbeat Failure",
    "Huawei 2G NE Down",
    "Huawei 4G NE Down",
    "Huawei 4G S1 NE Down",
    "Loss of communications with NE",
    "NE Is Disconnected",
    "NE O&M CONNECTION FAILURE",
    "NE OM CONNECTION FAILURE",
    "NE is Disconnected",
    "NE3SWS AGENT NOT RESPONDING TO REQUESTS",
    "NE_COMMU_BREAK",
    "NE_NOT_LOGIN",
    "NodeB Down",
    "Nokia 2G NE Down",
    "Nokia 4G NE Down",
    "Nokia 4G S1 NE Down",
    "ReachabilityProblem",
    "SWT_SWITCH_DOWN",
    "The Device is offline",
    "The link between the server and the NE is broken",
    "WCDMA BASE STATION OUT OF USE",
    "eNodeB Out of Service",
    "gNodeB Out of Service",
}

OFFLINE_ALARM_KEYS = {str(alarm or "").strip().upper() for alarm in OFFLINE_ALARMS} | {"OFFLINE"}


def _text(value):
    return str(value if value is not None else "").strip()


def _as_list(value):
    """把输入字段归一成 list：非 list（标量/None/dict 等畸形输入）一律当空，避免迭代报错。"""
    return value if isinstance(value, list) else []


def _ms_to_ts(value):
    """毫秒时间戳 -> 秒（float）。空/非法返回 None。"""
    if value in (None, "", 0, "0"):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num / 1000.0


# --------------------------------------------------------------------------- #
# 资源 / 拓扑索引
# --------------------------------------------------------------------------- #
def _dedup_resources_by_vid(resources):
    """以 resourceVid 为准去重（resources 可能有重复行）。"""
    vid_to_resource = OrderedDict()
    for res in _as_list(resources):
        if not isinstance(res, dict):
            continue
        vid = _text(res.get("resourceVid"))
        if vid and vid not in vid_to_resource:
            vid_to_resource[vid] = res
    return vid_to_resource


def _resource_name_maps(vid_to_resource):
    """resourceVid -> neName、neName -> 代表资源。仅认 neName（无 name 兜底）。"""
    vid_to_nename = {}
    resource_by_nename = {}
    for vid, res in vid_to_resource.items():
        nename = _text(res.get("neName"))
        if not nename:
            continue
        vid_to_nename[vid] = nename
        resource_by_nename.setdefault(nename, res)
    return vid_to_nename, resource_by_nename


def _build_adjacency(res_rels, vid_to_nename):
    """网元-网元无向邻接（以 neName 为节点）；resourceRelations 的 src/dst 不作为方向。"""
    adjacency = defaultdict(dict)  # ne -> {peer_ne: {layers:set}}
    for rel in _as_list(res_rels):
        if not isinstance(rel, dict):
            continue
        src = vid_to_nename.get(_text(rel.get("srcVid")))
        dst = vid_to_nename.get(_text(rel.get("dstVid")))
        if not src or not dst or src == dst:
            continue
        layer = _text(rel.get("linkLayer")) or "link"
        adjacency[src].setdefault(dst, {"layers": set()})["layers"].add(layer)
        adjacency[dst].setdefault(src, {"layers": set()})["layers"].add(layer)
    return adjacency


def build_indexes(topic):
    vid_to_resource = _dedup_resources_by_vid(topic.get("resources") or [])
    vid_to_nename, resource_by_nename = _resource_name_maps(vid_to_resource)
    adjacency = _build_adjacency(topic.get("resourceRelations"), vid_to_nename)
    return vid_to_resource, vid_to_nename, resource_by_nename, adjacency


def _site_of(nename, resource_by_nename, alarm_fallback=None):
    """返回 (site_id, site_name)。优先用资源，退回告警字段。"""
    res = resource_by_nename.get(nename) or {}
    site_id = _text(res.get("siteId"))
    site_name = _text(res.get("siteName"))
    if alarm_fallback is not None:
        site_name = site_name or _text(alarm_fallback.get("siteName"))
    # 没有 siteId 时用 siteName 兜底作为聚类键
    site_id = site_id or site_name
    return site_id, site_name


# --------------------------------------------------------------------------- #
# 找上游 / 根因：resourceRelations 只作为无向物理拓扑，不解释 src/dst 方向。
# 从每个 Offline/断站非 Data 源站 BFS 到最近 Data 站点，只保留最短路径且不穿越 Data。
# 多源优先最低公共上游，无公共上游时逐站取最远上游。
# --------------------------------------------------------------------------- #
def _text_has_token(text, token):
    # 按词边界匹配（token 前后不紧挨字母/数字），避免 RAN 误命中 TRANSMISSION/BRANCH 等子串
    return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", text) is not None


# 设备角色识别：按顺序匹配 domain 里的关键词（只看 domain）。
_ROLE_TOKENS = (
    ("Data", ("DATA", "IP", "ROUTER", "METRO")),
    ("Microwave", ("MICROWAVE", "MW", "RTN", "TRANSMISSION", "DWDM", "OTN", "OPTICAL", "WDM")),
    ("Ran", ("RAN", "WIRELESS", "NODEB", "BTS", "LTE")),
)


def _device_role(domain_text):
    text = (domain_text or "").strip().upper()
    for role, tokens in _ROLE_TOKENS:
        if any(_text_has_token(text, t) for t in tokens):
            return role
    return "Other"


def _is_offline_alarm(alarm_type):
    text = _text(alarm_type)
    if not text:
        return False
    upper_text = text.upper()
    return upper_text in OFFLINE_ALARM_KEYS or "OFFLINE" in upper_text or "断站" in text


def _build_site_topology(display_nes, ne_site, ne_role, adjacency):
    """由 resourceRelations 派生站点无向邻接及“含 Data 设备的站点”。"""
    site_links = defaultdict(set)
    site_has_data = set()
    for ne in display_nes:
        s = ne_site.get(ne)
        if not s:
            continue
        if ne_role.get(ne) == "Data":
            site_has_data.add(s)
        for peer in adjacency.get(ne, {}):
            ps = ne_site.get(peer)
            if not ps or ps == s:
                continue
            site_links[s].add(ps)
            site_links[ps].add(s)
    return site_links, site_has_data


def _backtrack_kept_hops(start_site, nearest_data_sites, predecessors, hops):
    """从最近 Data 终点反向回溯，只保留至少一条最短上游路径上的站点及其跳数。"""
    kept_sites = set(nearest_data_sites)
    stack = list(nearest_data_sites)
    while stack:
        current = stack.pop()
        for previous in predecessors.get(current, ()):
            if previous not in kept_sites:
                kept_sites.add(previous)
                stack.append(previous)
    kept_sites.add(start_site)
    return {site_id: hops[site_id] for site_id in kept_sites}


def _expand_neighbors(cur, cur_hop, site_links, hops, predecessors, nearest_data_hop, site_has_data, q):
    """处理 cur 的邻居：更新 hops/predecessors/队列，返回新的 nearest_data_hop 与命中的 Data 站点。"""
    hit_data = set()
    for nb in sorted(site_links.get(cur, ())):
        candidate_hop = cur_hop + 1
        if nearest_data_hop is not None and candidate_hop > nearest_data_hop:
            continue
        if nb not in hops:
            hops[nb] = candidate_hop
            predecessors[nb].add(cur)
            q.append(nb)
        elif hops[nb] == candidate_hop:
            predecessors[nb].add(cur)
        else:
            continue
        if nb in site_has_data and nearest_data_hop in (None, candidate_hop):
            nearest_data_hop = candidate_hop
            hit_data.add(nb)
    return nearest_data_hop, hit_data


def _source_upstream_sites(start_site, site_links, site_has_data):
    """从源站 BFS 到最近 Data 边界，返回上游路径内的跳数；到不了 Data 则只返回自身。"""
    if start_site in site_has_data:
        return {start_site: 0}

    predecessors = defaultdict(set)
    hops = {start_site: 0}
    q = deque([start_site])
    nearest_data_hop = None
    nearest_data_sites = set()
    while q:
        cur = q.popleft()
        if nearest_data_hop is not None and hops[cur] >= nearest_data_hop:
            continue
        nearest_data_hop, hit = _expand_neighbors(
            cur, hops[cur], site_links, hops, predecessors, nearest_data_hop, site_has_data, q)
        nearest_data_sites |= hit

    if not nearest_data_sites:
        return {start_site: 0}  # 分量内无路由(Data)，无方向锚点 -> 无上游
    return _backtrack_kept_hops(start_site, nearest_data_sites, predecessors, hops)


def _bucket_site_alarms(site_nes, ne_alarms, ne_role):
    """把站内告警按 (sort_key, ne_id) 分桶：非断站 / 传输设备断站 / 任意设备断站。"""
    non_offline, offline, offline_any = [], [], []
    for ne_id in site_nes:
        for index, alarm in enumerate(ne_alarms.get(ne_id) or []):
            ts = alarm.get("ts")
            record = ((ts is None, ts if ts is not None else 0.0, ne_id, index), ne_id)
            if not _is_offline_alarm(alarm.get("alarm_type")):
                non_offline.append(record)
                continue
            offline_any.append(record)
            if ne_role.get(ne_id) == "Microwave":
                offline.append(record)
    return non_offline, offline, offline_any


def _most_connected_microwave(site_nes, ne_role, ne_site, adjacency, downstream_sites):
    """挑与下游站点连接最多的 Microwave(传输) 设备，平局取 ne_id 最小；无则返回 ""。"""
    best = None
    for ne_id in sorted(site_nes):
        if ne_role.get(ne_id) != "Microwave":
            continue
        connected = {
            ne_site.get(peer) for peer in adjacency.get(ne_id, {})
            if ne_site.get(peer) in downstream_sites
        }
        if best is None or len(connected) > best[0]:
            best = (len(connected), ne_id)
    return best[1] if best is not None else ""


def _pick_site_root_cause(site_nes, ne_alarms, ne_role, ne_site, adjacency, downstream_sites):
    """在候选站点内挑根因网元：
    最早非断站告警 > 最早“传输(Microwave)设备”断站告警 > 有传输设备但无其断站告警时挑
    “下游连接最多的 Microwave 设备” > 无任何传输设备时回退到最早断站告警(任意设备)。
    返回 (root_ne, kind)。"""
    non_offline, offline, offline_any = _bucket_site_alarms(site_nes, ne_alarms, ne_role)
    for records, kind in ((non_offline, "non_offline_alarm"), (offline, "offline_alarm")):
        if records:
            return min(records)[1], kind

    microwave = _most_connected_microwave(site_nes, ne_role, ne_site, adjacency, downstream_sites)
    if microwave:
        return microwave, "transmission_device"

    if offline_any:  # 站内无传输设备：回退到最早的断站告警(任意设备)
        return min(offline_any)[1], "offline_alarm"
    return "", ""


def _offline_alarm_sites(core_nes, ne_alarms, ne_site):
    """有断站告警的网元所在站点集合（排序）。"""
    offline_nes = {
        ne for ne in core_nes
        if any(_is_offline_alarm(a.get("alarm_type")) for a in ne_alarms.get(ne) or [])
    }
    return sorted({ne_site.get(ne) for ne in offline_nes if ne_site.get(ne)})


def _make_candidate(base_site, source_sites):
    # 根因站点保持在最低公共祖先(base_site)，即使它不是 Data。
    return {"site_id": base_site, "source_sites": sorted(set(source_sites))}


def _common_reachable(common_source_sites, reach_by_site):
    """所有公共源站都能到达的站点交集。"""
    common = None
    for site_id in common_source_sites:
        cand = set(reach_by_site[site_id])
        common = cand if common is None else common & cand
    return common or set()


def _lowest_common_sites(common_candidates, common_source_sites, reach_by_site):
    """在公共可达站点里取“到各源站跳数总和/最大值”最小者（可并列）。"""
    def rank(c):
        hops = [reach_by_site[s][c] for s in common_source_sites]
        return sum(hops), max(hops)

    best = min(rank(c) for c in common_candidates)
    return sorted(c for c in common_candidates if rank(c) == best)


def _farthest_upstream(site_id, reach_by_site):
    """单个源站取最远的上游站点（平局取名字最小）。"""
    upstream = {c: h for c, h in reach_by_site[site_id].items() if c != site_id}
    max_hop = max(upstream.values())
    return min(c for c, h in upstream.items() if h == max_hop)


def _candidate_root_sites(common_source_sites, self_sites, reach_by_site):
    """产出候选根因站点：最低公共上游 / 无公共上游时逐站取最远上游 / 自身即根因。"""
    common_candidates = _common_reachable(common_source_sites, reach_by_site)
    candidate_roots = []
    if common_candidates:
        for base_site in _lowest_common_sites(common_candidates, common_source_sites, reach_by_site):
            candidate_roots.append(_make_candidate(base_site, common_source_sites))
    else:
        for site_id in common_source_sites:
            candidate_roots.append(_make_candidate(_farthest_upstream(site_id, reach_by_site), [site_id]))
    for site_id in self_sites:  # 无上游 -> 自身为根因
        candidate_roots.append(_make_candidate(site_id, [site_id]))
    return candidate_roots


def _downstream_by_root(candidate_roots):
    """每个根因站点的下游源站集合（供 _pick_site_root_cause 选“下游连接最多”的传输设备）。"""
    downstream = defaultdict(set)
    for c in candidate_roots:
        root_site = c["site_id"]
        for source_site in c["source_sites"]:
            if root_site != source_site:
                downstream[root_site].add(source_site)
    return downstream


def _sites_to_nes(display_nes, ne_site):
    """站点 -> 站内网元列表。"""
    site_to_nes = defaultdict(list)
    for ne in display_nes:
        s = ne_site.get(ne)
        if s:
            site_to_nes[s].append(ne)
    return site_to_nes


def find_upstream_roots(core_nes, ne_alarms, display_nes, ne_site, ne_role, adjacency):
    """据断站站点找公共/最远上游站点，再在每个上游站点挑根因网元。
    返回 {"root_sites": [...], "site_root_cause": {site_id: {"root_ne", "root_cause_kind"}}}。"""
    site_links, site_has_data = _build_site_topology(display_nes, ne_site, ne_role, adjacency)
    alarm_sites = _offline_alarm_sites(core_nes, ne_alarms, ne_site)

    reach_by_site = {}
    common_source_sites, self_sites = [], []
    for site_id in alarm_sites:
        hops = _source_upstream_sites(site_id, site_links, site_has_data)
        reach_by_site[site_id] = hops
        # 能到达路由且路径上有上游 -> 公共源站；否则自身即根因
        (common_source_sites if set(hops) - {site_id} else self_sites).append(site_id)

    candidate_roots = _candidate_root_sites(common_source_sites, self_sites, reach_by_site)
    root_sites = sorted({c["site_id"] for c in candidate_roots if c.get("site_id")})
    downstream_by_root = _downstream_by_root(candidate_roots)
    site_to_nes = _sites_to_nes(display_nes, ne_site)

    site_root_cause = {}
    for root_site in root_sites:
        ne_id, kind = _pick_site_root_cause(
            site_to_nes.get(root_site, []), ne_alarms,
            ne_role, ne_site, adjacency, downstream_by_root.get(root_site, set()),
        )
        site_root_cause[root_site] = {"root_ne": ne_id, "root_cause_kind": kind}

    return {"root_sites": root_sites, "site_root_cause": site_root_cause}


# --------------------------------------------------------------------------- #
# 告警定位（neName）与诊断入口
# --------------------------------------------------------------------------- #
def _alarm_vid_to_ne(happen_rels, vid_to_nename):
    """告警顶点 vid -> neName：经 happenRelations(srcVid->dstVid) 再查 resourceVid->neName。"""
    mapping = {}
    for h in _as_list(happen_rels):
        if not isinstance(h, dict):
            continue
        src = _text(h.get("srcVid"))
        ne = vid_to_nename.get(_text(h.get("dstVid")))
        if src and ne:
            mapping[src] = ne
    return mapping


def _resolve_nename(alarm, alarm_vid_to_ne):
    """告警定位到 neName：优先经 happenRelations(alarmVertexVid->neName)，再退回自带 neName
    （不用 neVid 兜底）。"""
    return alarm_vid_to_ne.get(_text(alarm.get("alarmVertexVid"))) or _text(alarm.get("neName"))


def _collect_ne_alarms(alarms, alarm_vid_to_ne):
    """收集有告警网元及其精简告警（只留驱动根因选择的 alarm_type / ts）。
    返回 (ne_alarms: neName->[{alarm_type,ts}], ne_alarm_fallback: neName->一条原始告警)。"""
    ne_alarms = OrderedDict()
    ne_alarm_fallback = {}
    for alarm in _as_list(alarms):
        if not isinstance(alarm, dict):
            continue
        nename = _resolve_nename(alarm, alarm_vid_to_ne)
        if not nename:
            continue
        ne_alarm_fallback.setdefault(nename, alarm)
        ts = _ms_to_ts(alarm.get("firstOccurrence")) or _ms_to_ts(alarm.get("lastOccurrence"))
        ne_alarms.setdefault(nename, []).append({
            "alarm_type": _text(alarm.get("alarmName")),
            "ts": ts,
        })
    return ne_alarms, ne_alarm_fallback


def _collect_ne_to_alarm_objs(alarms, alarm_vid_to_ne):
    """按定位到的 neName 归拢原始告警对象（丢弃非 dict / 无法定位 neName 的告警）。"""
    ne_to_alarm_objs = OrderedDict()
    for alarm in _as_list(alarms):
        if not isinstance(alarm, dict):
            continue
        nename = _resolve_nename(alarm, alarm_vid_to_ne)
        if nename:
            ne_to_alarm_objs.setdefault(nename, []).append(alarm)
    return ne_to_alarm_objs


def _ne_site_and_role(display_nes, resource_by_nename, ne_alarm_fallback):
    """每个网元的站点 id 与设备角色。"""
    ne_site, ne_role = {}, {}
    for ne in display_nes:
        s, _ = _site_of(ne, resource_by_nename, ne_alarm_fallback.get(ne))
        ne_site[ne] = s
        res = resource_by_nename.get(ne) or {}
        fallback = ne_alarm_fallback.get(ne)
        domain = _text(res.get("domain")) or (_text(fallback.get("domain")) if fallback else "")
        ne_role[ne] = _device_role(domain)
    return ne_site, ne_role


def _dedup_extend(head, rest):
    """在 head 后追加 rest 中尚未出现的元素，保序。"""
    out = list(head)
    seen = set(head)
    for item in rest:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _root_cause_nes(root):
    """site_root_cause 里的根因网元 neName（去重、保序，过滤空）。"""
    nes, seen = [], set()
    for info in root["site_root_cause"].values():
        ne = info.get("root_ne", "")
        if ne and ne not in seen:
            seen.add(ne)
            nes.append(ne)
    return nes


def _diagnose_root_nes(alarms, resource_by_nename, adjacency, all_ne_names, alarm_vid_to_ne):
    """对整批告警做一次诊断（不按 faultGroupId 切），返回根因网元 neName 列表。
    0 个或多个根因站点视为无法唯一定位，返回空。"""
    ne_alarms, ne_alarm_fallback = _collect_ne_alarms(alarms, alarm_vid_to_ne)
    core_nes = list(ne_alarms.keys())
    if not core_nes:
        return []

    # 全量网元进图（上游探索需要完整拓扑，避免路径被裁断）。
    display_nes = _dedup_extend(core_nes, all_ne_names)
    ne_site, ne_role = _ne_site_and_role(display_nes, resource_by_nename, ne_alarm_fallback)
    root = find_upstream_roots(core_nes, ne_alarms, display_nes, ne_site, ne_role, adjacency)
    if len(root["root_sites"]) != 1:
        return []
    return _root_cause_nes(root)


def diagnose_root_cause_devices(input_json):
    """根因诊断入口。整个 input_json 当作一张图诊断（不按 faultGroupId 切分）。
    输入 input_json: 含 alarms / resources / resourceRelations / happenRelations 的 dict。
    输出: (root_cause_resources, ne_to_alarm_objs)
        - root_cause_resources: [{"resourceName": neName, "confidence": 0.9}, ...]，
          去重、保序；0 个或多个根因站点视为无法定位，返回空列表。
        - ne_to_alarm_objs: {neName: [原始告警对象, ...]}，按定位到的 neName 归拢全部告警。
    输入格式不正确（非 dict、字段非 list 等）不抛异常，返回 ([], {})。
    """
    if not isinstance(input_json, dict):
        return [], {}

    _, vid_to_nename, resource_by_nename, adjacency = build_indexes(input_json)
    all_ne_names = sorted(resource_by_nename.keys())
    # 告警顶点 vid -> neName（经 happenRelations），供告警定位网元。
    alarm_vid_to_ne = _alarm_vid_to_ne(input_json.get("happenRelations"), vid_to_nename)
    ne_to_alarm_objs = _collect_ne_to_alarm_objs(input_json.get("alarms"), alarm_vid_to_ne)

    root_nes = _diagnose_root_nes(input_json.get("alarms") or [], resource_by_nename,
                                  adjacency, all_ne_names, alarm_vid_to_ne)
    root_cause_resources = [{"resourceName": ne, "confidence": 0.9} for ne in root_nes]
    return root_cause_resources, dict(ne_to_alarm_objs)
