#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 microwave_topic 拓扑导出（test.json）转成 ne_propagation_visualizer.html 可直接展示的故障组。

输入 test.json 已经自包含了展示所需的全部信息：
  - alarms:            每条告警（含 faultGroupId、neName、neVid、ownerVid、发生/清除时间等）
  - resources:         网元资源（resourceVid、neName、domain、networkType、vendor、siteId/siteName ...）
  - resourceRelations: 网元-网元拓扑连边（srcVid/dstVid、linkLayer）
  - happenRelations:   告警顶点 -> owner 网元/站点

因此本脚本只依赖 test.json，无需任何外部文件。test.json 里没有经纬度，页面又用真实
地图布局，所以脚本按 site_id 确定性哈希给每个站点合成一个聚拢的经纬度（同站点的所有
设备坐标完全一致），可用 --base-lat/--base-lon/--spread 调整中心与散布范围。

另外按 complete_group_topology.py 的候选选择思路（只把它依赖的外部 ne_graph/site_chains
换成 test.json 自带的 resourceRelations 无向拓扑）：只用 Offline/断站告警站点推断；每个
非 Data 源站通过 BFS 寻找最近 Data 站点，并只保留通往这些 Data 站点的最短路径作为“上游”，
搜索不会穿过 Data 站点。多个源站优先选择离大家最近的公共上游，没有公共上游时逐站取最远
上游；非 Data 候选优先提升到公共上游中的最近 Data 站点，找不到则保留非 Data 候选。再用
_pick_site_root_cause（非断站告警 > 断站告警 > 无告警取下游连接最多的传输设备）在候选站点里
挑根因网元。结果写成 visualizer 可直接高亮的字段：
上游根因站点标红（fault_pattern_managed_sites / role_mapping.common_upstream_site），根因设备
is_root_cause_device、根因告警 is_root_cause。

输出：每个 faultGroupId 一个「原始格式」故障组对象（含 ne_info / group_info / symptoms /
match_info），可被 ne_propagation_visualizer.html 的 loadOriginalFormat 直接加载。
  - <output>.jsonl        每行一个故障组，供故障组总览页浏览
  - 当只有一个故障组时，额外写出同名 <output>.json 单对象，方便直接拖进 NE 传播图页面
"""

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict, defaultdict, deque
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 断站告警集合（complete_group_topology._pick_site_root_cause 用它区分“非断站/断站”）
from alarm_tools.alarm_types import OFFLINE_ALARMS

OFFLINE_ALARM_KEYS = {str(alarm or "").strip().upper() for alarm in OFFLINE_ALARMS} | {"OFFLINE"}

# 合成经纬度的默认中心与散布（度）。spread≈0.06° 约 6~7km，保证站点聚拢不散。
DEFAULT_BASE_LAT = 39.90
DEFAULT_BASE_LON = 116.40
DEFAULT_SPREAD = 0.06


def _text(value):
    return str(value if value is not None else "").strip()


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
    # test.json 中的时间是毫秒
    return num / 1000.0


def _fmt_ts(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _append_unique(values, value):
    if value and value not in values:
        values.append(value)


# --------------------------------------------------------------------------- #
# 站点经纬度合成（test.json 无经纬度；页面用真实地图布局，需要合法 WGS84 坐标）
# --------------------------------------------------------------------------- #
class SiteCoordGenerator:
    """按 site_id 确定性哈希生成聚拢的经纬度。

    - 同一个 site_id 永远返回相同坐标（同站点所有设备坐标一致）。
    - 所有站点落在以 (base_lat, base_lon) 为中心、±spread 度的方框内，保证不散。
    """

    def __init__(self, base_lat=DEFAULT_BASE_LAT, base_lon=DEFAULT_BASE_LON, spread=DEFAULT_SPREAD):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.spread = spread
        self._cache = {}

    @staticmethod
    def _unit(seed):
        """由字符串确定性得到 [0,1) 的浮点。"""
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(1 << 48)

    def coords(self, site_id):
        key = _text(site_id)
        if not key:
            return "", ""
        if key not in self._cache:
            # 纬度随 |纬度| 增大会拉伸经度间距，这里散布很小可忽略，直接线性铺开即可。
            lat_off = (self._unit(key + "#lat") * 2 - 1) * self.spread
            lon_off = (self._unit(key + "#lon") * 2 - 1) * self.spread
            lat = round(self.base_lat + lat_off, 6)
            lon = round(self.base_lon + lon_off, 6)
            self._cache[key] = (lat, lon)
        lat, lon = self._cache[key]
        return lat, lon


# --------------------------------------------------------------------------- #
# 资源 / 拓扑索引
# --------------------------------------------------------------------------- #
def build_indexes(topic):
    resources = topic.get("resources") or []
    # 以 resourceVid 为准去重（test.json 里 resources 可能有重复行）
    vid_to_resource = OrderedDict()
    for res in resources:
        if not isinstance(res, dict):
            continue
        vid = _text(res.get("resourceVid"))
        if vid and vid not in vid_to_resource:
            vid_to_resource[vid] = res

    # resourceVid -> neName（拓扑连边解析用）
    vid_to_nename = {}
    # neName -> 代表资源
    resource_by_nename = {}
    for vid, res in vid_to_resource.items():
        nename = _text(res.get("neName")) or _text(res.get("name"))
        if not nename:
            continue
        vid_to_nename[vid] = nename
        resource_by_nename.setdefault(nename, res)

    # 网元-网元无向邻接（以 neName 为节点）；resourceRelations 的 src/dst 不作为方向。
    adjacency = defaultdict(dict)  # ne -> {peer_ne: {layers:set}}
    for rel in topic.get("resourceRelations") or []:
        if not isinstance(rel, dict):
            continue
        src = vid_to_nename.get(_text(rel.get("srcVid")))
        dst = vid_to_nename.get(_text(rel.get("dstVid")))
        if not src or not dst or src == dst:
            continue
        layer = _text(rel.get("linkLayer")) or "link"
        src_meta = adjacency[src].setdefault(dst, {"layers": set()})
        dst_meta = adjacency[dst].setdefault(src, {"layers": set()})
        src_meta["layers"].add(layer)
        dst_meta["layers"].add(layer)

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
# 多源优先最低公共上游，无公共上游时逐站取最远上游；Data 只在这些上游路径内优先。
# --------------------------------------------------------------------------- #
def _text_has_token(text, token):
    # 逐字来自 complete_group_topology._text_has_token
    if len(token) <= 2:
        return re.search(rf"(?<![A-Z0-9]){token}(?![A-Z0-9])", text) is not None
    return token in text


def _device_role(domain_text):
    """逐字来自 complete_group_topology._device_role（只看 domain）。"""
    text = (domain_text or "").strip().upper()
    if any(_text_has_token(text, t) for t in ("DATA", "IP", "ROUTER", "METRO")):
        return "Data"
    if any(_text_has_token(text, t) for t in ("MICROWAVE", "MW", "RTN", "TRANSMISSION", "DWDM", "OTN", "OPTICAL", "WDM")):
        return "Microwave"
    if any(_text_has_token(text, t) for t in ("RAN", "WIRELESS", "NODEB", "BTS", "LTE")):
        return "Ran"
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


def _source_upstream_sites(start_site, site_links, site_has_data):
    """从源站 BFS 到最近 Data 边界，返回上游路径内的跳数、前驱和 Data 终点。

    能到达 Data：只保留通往所有最近 Data 站点的最短路径并在 Data 处停止。
    到不了任何 Data：方向没有锚点，视作“无上游”（只返回自身），由上层按连通分量决定
    是自身为根因（分量内仅此一个断站）还是判不了上游（分量内多断站且连通）。
    """
    if start_site in site_has_data:
        return {start_site: 0}, {start_site: None}, {start_site}

    predecessors = defaultdict(set)
    hops = {start_site: 0}
    q = deque([start_site])
    nearest_data_hop = None
    nearest_data_sites = set()
    while q:
        cur = q.popleft()
        cur_hop = hops[cur]
        if nearest_data_hop is not None and cur_hop >= nearest_data_hop:
            continue
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
            if nb in site_has_data:
                if nearest_data_hop is None:
                    nearest_data_hop = candidate_hop
                if candidate_hop == nearest_data_hop:
                    nearest_data_sites.add(nb)

    if not nearest_data_sites:
        # 该连通分量内没有任何路由(Data)，无方向锚点 -> 无上游。
        return {start_site: 0}, {start_site: None}, set()

    # 从最近 Data 终点反向回溯，仅保留至少一条最短上游路径上的站点。
    kept_sites = set(nearest_data_sites)
    stack = list(nearest_data_sites)
    while stack:
        current = stack.pop()
        for previous in predecessors.get(current, ()):
            if previous not in kept_sites:
                kept_sites.add(previous)
                stack.append(previous)
    kept_sites.add(start_site)
    kept_hops = {site_id: hops[site_id] for site_id in kept_sites}
    parents = {start_site: None}
    for site_id in kept_sites - {start_site}:
        valid_predecessors = predecessors.get(site_id, set()) & kept_sites
        parents[site_id] = min(valid_predecessors) if valid_predecessors else None
    return kept_hops, parents, nearest_data_sites


def _site_path(start_site, target_site, parents):
    if target_site not in parents:
        return []
    path = []
    current = target_site
    while current is not None:
        path.append(current)
        if current == start_site:
            break
        current = parents.get(current)
    return list(reversed(path))


def _promote_to_data_site(candidate_site, allowed_data_sites, site_links):
    """非 Data 候选只在共同上游范围内提升到最近 Data；没有则保留原候选。"""
    allowed_data_sites = set(allowed_data_sites or ())
    if candidate_site in allowed_data_sites:
        return candidate_site, None
    hops = {candidate_site: 0}
    q = deque([candidate_site])
    while q:
        current = q.popleft()
        for neighbor in sorted(site_links.get(current, ())):
            if neighbor in hops:
                continue
            hops[neighbor] = hops[current] + 1
            q.append(neighbor)
    candidates = sorted((set(hops) - {candidate_site}) & allowed_data_sites)
    if not candidates:
        return candidate_site, None
    promoted_site = min(candidates, key=lambda site_id: (hops[site_id], site_id))
    return promoted_site, {
        "from_site_id": candidate_site,
        "to_site_id": promoted_site,
        "hop": hops[promoted_site],
    }


def _stringify_like_js(value):
    """与标注页面的字符串化规则一致：整数 float 不保留小数点。"""
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


def _pick_site_root_cause(site_id, site_nes, ne_alarms, ne_role, ne_site, adjacency, downstream_sites):
    """逐案照搬 complete_group_topology._pick_site_root_cause：
    最早非断站告警 > 最早断站告警 > 无告警时挑“下游连接最多的 Transmission(Microwave) 设备”。
    返回 (root_ne, kind, primary_occurrence_uuid, alarm_key)。"""
    non_offline = []
    offline = []
    for ne_id in site_nes:
        for index, alarm in enumerate(ne_alarms.get(ne_id) or []):
            ts = alarm.get("ts")
            sort_key = (ts is None, ts if ts is not None else 0.0, ne_id, index)
            record = (sort_key, ne_id, index, alarm)
            if _is_offline_alarm(alarm.get("alarm_type")):
                offline.append(record)
            else:
                non_offline.append(record)
    for records, kind in ((non_offline, "non_offline_alarm"), (offline, "offline_alarm")):
        if records:
            _key, ne_id, index, alarm = min(records)
            return (
                ne_id,
                kind,
                alarm.get("occurrence_uuid", ""),
                _alarm_root_cause_key(alarm, index),
            )

    # 无告警：挑与下游站点连接最多的 Microwave(传输) 设备，平局取 ne_id 最小
    best = None
    for ne_id in sorted(site_nes):
        if ne_role.get(ne_id) != "Microwave":
            continue
        connected = set()
        for peer in adjacency.get(ne_id, {}):
            ps = ne_site.get(peer)
            if ps and ps in downstream_sites:
                connected.add(ps)
        if best is None or len(connected) > best[0]:
            best = (len(connected), ne_id)
    if best is not None:
        return best[1], "transmission_device", "", ""
    return "", "", "", ""


def _build_root_cause_annotations(site_root_cause):
    """把站点根因选择结果转换成标注页面识别的标准结构。"""
    annotations = {}
    for info in (site_root_cause or {}).values():
        root_ne = info.get("root_ne", "")
        if not root_ne:
            continue
        annotation = annotations.setdefault(root_ne, {"device": False, "alarms": {}})
        alarm_key = info.get("alarm_key", "")
        if alarm_key:
            annotation["alarms"][alarm_key] = True
        else:
            annotation["device"] = True
    return annotations


def _build_auto_root_cause_summary(site_root_cause):
    """生成 complete_group_topology.topology_completion 使用的自动根因摘要。"""
    summary = []
    for site_id, info in (site_root_cause or {}).items():
        root_ne = info.get("root_ne", "")
        if not root_ne:
            continue
        summary.append({
            "site_id": site_id,
            "ne_id": root_ne,
            "kind": info.get("root_cause_kind", ""),
            # complete_group_topology 对设备级根因写 JSON null，而不是空字符串。
            "alarm_key": info.get("alarm_key") or None,
        })
    return summary


def _connected_components(sites, site_links):
    """返回 {site_id: component_id}。孤立站点(无边)自成一个分量。"""
    comp_of = {}
    cid = 0
    for start in sites:
        if start in comp_of:
            continue
        comp_of[start] = cid
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in site_links.get(cur, ()):
                if nb not in comp_of:
                    comp_of[nb] = cid
                    stack.append(nb)
        cid += 1
    return comp_of


def find_upstream_roots(core_nes, ne_alarms, display_nes, ne_site, ne_role, adjacency):
    """只根据 Offline/断站站点寻找公共/最远可达点，再优先提升到 Data 站点。"""
    site_links, site_has_data = _build_site_topology(
        display_nes, ne_site, ne_role, adjacency
    )
    all_alarm_sites = sorted({ne_site.get(ne) for ne in core_nes if ne_site.get(ne)})
    offline_nes = {
        ne
        for ne in core_nes
        if any(_is_offline_alarm(alarm.get("alarm_type")) for alarm in ne_alarms.get(ne) or [])
    }
    alarm_sites = sorted({ne_site.get(ne) for ne in offline_nes if ne_site.get(ne)})

    # 按连通分量分别判：统计每个分量里有几个断站站点，用于“无路由+多断站+连通”判空。
    all_sites = set(site_links) | set(alarm_sites)
    for ne in display_nes:
        s = ne_site.get(ne)
        if s:
            all_sites.add(s)
    comp_of = _connected_components(all_sites, site_links)
    offline_count_in_comp = defaultdict(int)
    for site_id in alarm_sites:
        offline_count_in_comp[comp_of.get(site_id)] += 1

    reach_by_site = {}
    parents_by_site = {}
    nearest_data_by_site = {}
    common_source_sites = []      # 能朝路由定向、且有上游（非自身）的断站站点
    no_upstream_sites = []        # 自身即最上游（路由自身，或无路由但分量内仅此一个断站）
    undetermined_sites = []       # 无路由 + 分量内多断站且连通 -> 上游判不了
    for site_id in alarm_sites:
        hops, parents, nearest_data_sites = _source_upstream_sites(
            site_id, site_links, site_has_data
        )
        reach_by_site[site_id] = hops
        parents_by_site[site_id] = parents
        nearest_data_by_site[site_id] = nearest_data_sites
        if set(hops) - {site_id}:
            # 能到达路由，且路径上有上游站点
            common_source_sites.append(site_id)
        elif site_id in site_has_data:
            # 路由站点自身 -> 自身为根因
            no_upstream_sites.append(site_id)
        elif offline_count_in_comp.get(comp_of.get(site_id), 0) >= 2:
            # 无路由 + 该连通分量内 >=2 个断站 -> 无方向锚点，判不了
            undetermined_sites.append(site_id)
        else:
            # 无路由 + 分量内仅此一个断站 -> 自身为根因
            no_upstream_sites.append(site_id)

    common_candidates = None
    for site_id in common_source_sites:
        candidates = set(reach_by_site[site_id])
        common_candidates = candidates if common_candidates is None else common_candidates & candidates
    common_candidates = common_candidates or set()

    candidate_roots = []
    data_ancestor_promotions = []

    def _add_candidate(base_site, source_sites, selection_kind, allowed_data_sites, base_hop=None):
        # 与本家一致：根因站点保持在最低公共祖先(base_site)，即使它不是 Data。
        # 若其上游存在 Data 站点，只作为 router_ancestor 注记，不替换根因站点。
        promoted_site, promotion = _promote_to_data_site(
            base_site, allowed_data_sites, site_links
        )
        if promotion:
            data_ancestor_promotions.append(promotion)
        candidate_roots.append({
            "site_id": base_site,
            "base_ancestor_site": base_site,
            "selection_kind": selection_kind,
            "source_sites": sorted(set(source_sites)),
            "base_hop": base_hop,
            "router_promoted": bool(promotion),
            "router_ancestor_site_id": promoted_site if promotion else "",
        })

    if common_candidates:
        def _common_rank(candidate):
            return (
                sum(reach_by_site[site_id][candidate] for site_id in common_source_sites),
                max(reach_by_site[site_id][candidate] for site_id in common_source_sites),
            )

        best_rank = min(_common_rank(candidate) for candidate in common_candidates)
        lowest_common_sites = sorted(
            candidate for candidate in common_candidates if _common_rank(candidate) == best_rank
        )
        for base_site in lowest_common_sites:
            _add_candidate(
                base_site,
                common_source_sites,
                "common_upstream",
                common_candidates & site_has_data,
            )
    else:
        for site_id in common_source_sites:
            upstream_hops = {
                candidate: hop
                for candidate, hop in reach_by_site[site_id].items()
                if candidate != site_id
            }
            max_hop = max(upstream_hops.values())
            farthest_site = min(
                candidate for candidate, hop in upstream_hops.items() if hop == max_hop
            )
            _add_candidate(
                farthest_site,
                [site_id],
                "farthest_upstream",
                set(reach_by_site[site_id]) & site_has_data,
                max_hop,
            )

    # 无上游 -> 自身为根因：路由站点自身(data_self)，或无路由但分量内仅此一个断站(isolated_self)。
    for site_id in no_upstream_sites:
        kind = "data_self_fallback" if site_id in site_has_data else "isolated_offline_self"
        _add_candidate(site_id, [site_id], kind, {site_id} if site_id in site_has_data else set(), 0)

    root_sites = sorted({item["site_id"] for item in candidate_roots if item.get("site_id")})

    # 每个断站源站记录一个确定性的最终候选和无向最短路径，兼容原输出结构。
    undetermined_set = set(undetermined_sites)
    upstream_of = {
        site_id: {
            "upstream_site": "",
            "hop": -1,
            "path": [],
            "selection_kind": (
                "upstream_undetermined_no_router"
                if site_id in undetermined_set
                else "no_upstream"
            ),
        }
        for site_id in alarm_sites
    }
    downstream_by_root = defaultdict(set)
    for candidate in sorted(candidate_roots, key=lambda item: (item["site_id"], item["base_ancestor_site"])):
        root_site = candidate["site_id"]
        for source_site in candidate["source_sites"]:
            hops = reach_by_site.get(source_site, {source_site: 0})
            path = _site_path(source_site, root_site, parents_by_site.get(source_site, {source_site: None}))
            hop = hops.get(root_site, len(path) - 1 if path else candidate.get("base_hop", -1))
            current = upstream_of[source_site]
            rank = (hop < 0, hop, root_site)
            current_rank = (
                current["hop"] < 0,
                current["hop"] if current["hop"] >= 0 else 0,
                current["upstream_site"],
            )
            if not current["upstream_site"] or rank < current_rank:
                upstream_of[source_site] = {
                    "upstream_site": root_site,
                    "hop": hop,
                    "path": path,
                    "selection_kind": candidate["selection_kind"],
                    "base_ancestor_site": candidate["base_ancestor_site"],
                    "router_promoted": candidate["router_promoted"],
                    "router_ancestor_site_id": candidate["router_ancestor_site_id"],
                }
            if root_site != source_site:
                downstream_by_root[root_site].add(source_site)

    # 站点内网元
    site_to_nes = defaultdict(list)
    for ne in display_nes:
        s = ne_site.get(ne)
        if s:
            site_to_nes[s].append(ne)

    # 每个上游根因站点挑一个根因网元
    site_root_cause = {}
    for root_site in root_sites:
        ne_id, kind, primary_uuid, alarm_key = _pick_site_root_cause(
            root_site, site_to_nes.get(root_site, []), ne_alarms,
            ne_role, ne_site, adjacency, downstream_by_root.get(root_site, set()),
        )
        site_root_cause[root_site] = {
            "root_ne": ne_id,
            "root_cause_kind": kind,
            "primary_occurrence_uuid": primary_uuid,
            "alarm_key": alarm_key,
        }

    return {
        "root_sites": root_sites,
        "site_has_data": sorted(site_has_data),
        "alarm_sites": alarm_sites,
        "all_alarm_sites": all_alarm_sites,
        "offline_alarm_sites": alarm_sites,
        "common_upstream_source_sites": common_source_sites,
        "no_upstream_sites": no_upstream_sites,
        # 无路由 + 分量内多断站且连通：上游无法判定，不产出根因
        "upstream_undetermined_sites": sorted(undetermined_sites),
        "nearest_data_sites_by_source": {
            site_id: sorted(data_sites)
            for site_id, data_sites in nearest_data_by_site.items()
        },
        "candidate_roots": candidate_roots,
        "data_ancestor_promotions": data_ancestor_promotions,
        "upstream_of": upstream_of,
        "site_root_cause": site_root_cause,
    }


# --------------------------------------------------------------------------- #
# 单个故障组构建
# --------------------------------------------------------------------------- #
def build_group_object(
    group_id,
    alarms,
    resource_by_nename,
    adjacency,
    coord_gen,
    all_ne_names,
):
    # 1) 收集本组所有告警对应的核心网元（有告警的 NE）及其告警
    ne_alarms = OrderedDict()          # neName -> [node_alarm]
    ne_alarm_fallback = {}             # neName -> 一条原始告警（补字段用）
    symptoms = []
    site_ids = []
    site_names = {}                    # site_id -> site_name
    timestamps = []

    for index, alarm in enumerate(alarms, start=1):
        nename = _text(alarm.get("neName")) or _text(alarm.get("neVid"))
        if not nename:
            continue
        ne_alarm_fallback.setdefault(nename, alarm)

        ts = _ms_to_ts(alarm.get("firstOccurrence")) or _ms_to_ts(alarm.get("lastOccurrence"))
        clear_ts = _ms_to_ts(alarm.get("clearTime"))
        if ts is not None:
            timestamps.append(ts)

        alarm_type = _text(alarm.get("alarmName"))
        alarm_id = _text(alarm.get("identifier")) or f"{group_id}-{index}"
        occurrence_uuid = _text(alarm.get("alarmVertexVid")) or alarm_id
        domain = _text(alarm.get("domain"))
        site_id, site_name = _site_of(nename, resource_by_nename, alarm)
        if site_id:
            _append_unique(site_ids, site_id)
            if site_name:
                site_names.setdefault(site_id, site_name)

        node_alarm = {
            "alarm_id": alarm_id,
            "occurrence_uuid": occurrence_uuid,
            "alarm_type": alarm_type,
            "alarm_time": _fmt_ts(ts),
            "alarm_clear_time": _fmt_ts(clear_ts),
            "domain": domain,
            "site_id": site_id,
            "matched_role": "alarm_group",
            "matched_rule": "fault_group_id_rule",
            "matched_role_key": "alarm_group",
            "owner_type": _text(alarm.get("ownerType")),
            "ne_type": _text(alarm.get("neType")),
            "severity": _text(alarm.get("severity")),
            "vendor": _text(alarm.get("vendor")),
            "faultGroupId": group_id,
            "ts": ts,
        }
        ne_alarms.setdefault(nename, []).append(node_alarm)

        symptoms.append({
            "node": site_id,
            "alarm": alarm_type,
            "alarm_source": nename,
            "ts": ts,
            "eid": alarm_id,
            "occurrence_uuid": occurrence_uuid,
            "matched_role": "alarm_group",
            "matched_rule": "fault_group_id_rule",
            "matched_role_key": "alarm_group",
            "faultGroupId": group_id,
            "domain": domain,
            "告警清除时间": _fmt_ts(clear_ts),
        })

    core_nes = list(ne_alarms.keys())

    # 2) 不裁剪：所有网元 + 本组核心（核心一般已是子集）都进图。
    # 上游探索也据此拿到完整拓扑，与显示无关，避免路径被裁断。
    display_nes = list(core_nes)
    seen = set(core_nes)
    for ne in all_ne_names:
        if ne not in seen:
            seen.add(ne)
            display_nes.append(ne)

    display_set = set(display_nes)

    # 2.5) 找上游根因：Offline/断站站点 + resourceRelations 无向拓扑 + Data 优先
    ne_site = {}
    ne_role = {}
    for ne in display_nes:
        s, _ = _site_of(ne, resource_by_nename, ne_alarm_fallback.get(ne))
        ne_site[ne] = s
        res = resource_by_nename.get(ne) or {}
        fallback = ne_alarm_fallback.get(ne)
        domain = _text(res.get("domain")) or (_text(fallback.get("domain")) if fallback else "")
        ne_role[ne] = _device_role(domain)

    root = find_upstream_roots(core_nes, ne_alarms, display_nes, ne_site, ne_role, adjacency)
    root_sites = root["root_sites"]
    root_site_set = set(root_sites)
    root_ne_set = {info["root_ne"] for info in root["site_root_cause"].values() if info["root_ne"]}
    primary_uuids = {info["primary_occurrence_uuid"] for info in root["site_root_cause"].values()
                     if info["primary_occurrence_uuid"]}

    # 标注页面的标准根因协议：告警根因精确标记 alarm key；无告警根因标记整台设备。
    root_cause_annotations = _build_root_cause_annotations(root["site_root_cause"])
    auto_root_cause_annotations = _build_auto_root_cause_summary(root["site_root_cause"])

    # 在根因网元的告警上打标记：根因网元上的告警算根因症状，主根因单独标一条
    for rn in root_ne_set:
        for na in ne_alarms.get(rn, []):
            na["is_root_cause"] = True
            na["matched_role"] = "root_cause"
            na["is_primary_root_cause"] = (na.get("occurrence_uuid") in primary_uuids)

    # 3) 组装 ne_info（含 link，只保留两端都在展示集合里的边）
    ne_info = OrderedDict()
    for nename in display_nes:
        res = resource_by_nename.get(nename) or {}
        fallback = ne_alarm_fallback.get(nename)
        alarms_here = ne_alarms.get(nename, [])
        site_id, site_name = _site_of(nename, resource_by_nename, fallback)
        if site_id and site_name:
            site_names.setdefault(site_id, site_name)

        links = {}
        for peer, meta in adjacency.get(nename, {}).items():
            if peer not in display_set or peer == nename:
                continue
            layers = ",".join(sorted(meta.get("layers", set())))
            links[peer] = {
                "connection_type": layers,
                "distance": "",
                "topology": layers,
                "time_window": "",
                "left_alarm": {},
                "right_alarm": {},
            }

        ne_type = _text(res.get("neType")) or (_text(fallback.get("neType")) if fallback else "")
        network_type = _text(res.get("networkType")) or (_text(fallback.get("networkType")) if fallback else "")
        vendor = _text(res.get("vendor")) or (_text(fallback.get("vendor")) if fallback else "")
        domain = _text(res.get("domain")) or (_text(fallback.get("domain")) if fallback else "")

        lat, lon = coord_gen.coords(site_id)

        ne_info[nename] = {
            "link": links,
            "group": group_id,
            "name": _text(res.get("name")) or nename,
            "site_id": site_id,
            "site_name": site_name,
            "site_type": "",
            "type": ne_type.upper(),
            "network_type": network_type.upper(),
            "manufacturer": vendor.upper(),
            "running_status": "",
            "domain": domain.upper(),
            "region_id": "",
            "longitude": lon,
            "latitude": lat,
            "alarm": alarms_here,
            # 邻居节点（本身无告警）标记为拓扑补充节点
            "supplemental_fault_pattern_context": nename not in ne_alarms,
            # 上游根因网元/站点标记
            "is_root_cause_device": nename in root_ne_set,
            "is_upstream_root_site": site_id in root_site_set,
            "device_role": ne_role.get(nename, ""),
        }

    # 同步把根因标记写进 symptoms（详情/导出可用）
    for sym in symptoms:
        if sym.get("alarm_source") in root_ne_set:
            sym["matched_role"] = "root_cause"
            sym["is_root_cause"] = True
            sym["is_primary_root_cause"] = (sym.get("occurrence_uuid") in primary_uuids)

    # 4) group_info / match_info / 头部字段
    group_site_ids = sorted(site_ids)
    # 站点角色：上游根因站点 -> associated_site（节点绿框）；其余告警站点 -> context_site（青框）
    other_alarm_sites = [s for s in group_site_ids if s not in root_site_set]
    role_mapping = {
        "associated_site": root_sites,
        "context_site": other_alarm_sites,
        # 上游站点在地图上标红（visualizer isTopologyHighlightSite 读取此键）
        "common_upstream_site": root_sites,
    }
    # 让上游根因站点在地图上高亮成红色（站点光晕/标签）
    managed_sites = root_sites
    anchor_ts = min(timestamps) if timestamps else None
    inferred_roots = {
        info_site: [rc["root_ne"]] if rc["root_ne"] else []
        for info_site, rc in root["site_root_cause"].items()
    }
    root_cause_summary = dict(root)
    selected_site_ids = sorted({site_id for site_id in ne_site.values() if site_id})
    added_site_ids = sorted(set(selected_site_ids) - set(group_site_ids))
    added_ne_ids = sorted(ne_id for ne_id in display_nes if ne_id not in set(core_nes))
    highlight_sites = []
    role_by_selection = {
        "common_upstream": "common_upstream_site",
        "farthest_upstream": "farthest_upstream_site",
        "data_self_fallback": "no_upstream_site",
        "isolated_offline_self": "no_upstream_site",
    }
    for candidate in root.get("candidate_roots", []):
        site_id = candidate.get("site_id", "")
        if not site_id:
            continue
        item = {
            "site_id": site_id,
            "role": role_by_selection.get(candidate.get("selection_kind"), "common_upstream_site"),
            "source_sites": candidate.get("source_sites", []),
        }
        if candidate.get("router_promoted"):
            item["router_promoted"] = True
            item["router_ancestor_site_ids"] = [candidate.get("router_ancestor_site_id")]
        highlight_sites.append(item)
    topology_completion = {
        "mode": "topic_data_boundary_bfs",
        "original_alarm_ne_ids": sorted(core_nes),
        "original_alarm_site_ids": root.get("all_alarm_sites", []),
        "ancestor_source_site_ids": root.get("offline_alarm_sites", []),
        "non_offline_alarm_site_ids": sorted(
            set(root.get("all_alarm_sites", [])) - set(root.get("offline_alarm_sites", []))
        ),
        "selected_site_ids": selected_site_ids,
        "added_site_ids": added_site_ids,
        "added_ne_ids": added_ne_ids,
        "common_upstream_source_site_ids": root.get("common_upstream_source_sites", []),
        "no_upstream_sites": root.get("no_upstream_sites", []),
        "upstream_undetermined_sites": root.get("upstream_undetermined_sites", []),
        "nearest_data_sites_by_source": root.get("nearest_data_sites_by_source", {}),
        "candidate_roots": root.get("candidate_roots", []),
        "data_ancestor_promotions": root.get("data_ancestor_promotions", []),
        "highlight_site_ids": root_sites,
        "highlight_sites": highlight_sites,
        "auto_root_cause_annotations": auto_root_cause_annotations,
        "site_level_connected": bool(root_sites) or len(root.get("offline_alarm_sites", [])) <= 1,
    }
    match_info = {
        "uuid": group_id,
        "rule": "fault_group_id_rule",
        "merged_rules": ["fault_group_id_rule"],
        "related_group_uuids": [],
        "inferred_roots": inferred_roots,
        "root_cause_annotations": root_cause_annotations,
        "role_mapping": role_mapping,
        "symptoms": symptoms,
        "uses_missing_topology": False,
        "missing_topology_edges": [],
        "root_cause": root_cause_summary,
    }

    return {
        "uuid": group_id,
        "rule": "fault_group_id_rule",
        "merged_rules": ["fault_group_id_rule"],
        "related_group_uuids": [],
        "role_mapping": role_mapping,
        "symptoms": symptoms,
        "match_info": match_info,
        "ne_info": ne_info,
        # 根因站点在地图上标红（visualizer 读取 fault_pattern_managed_sites）
        "fault_pattern_managed_sites": managed_sites,
        "inferred_roots": inferred_roots,
        "root_cause_annotations": root_cause_annotations,
        "topology_completion": topology_completion,
        "root_cause": root_cause_summary,
        "group_info": {
            group_id: {
                "ne_list": sorted(display_nes),
                "site_list": group_site_ids,
                "core_ne_list": sorted(core_nes),
                "root_sites": root_sites,
                "root_nes": sorted(root_ne_set),
                "upstream_undetermined_sites": root.get("upstream_undetermined_sites", []),
            }
        },
        "group_anchor_ts": anchor_ts,
        "group_anchor_time": _fmt_ts(anchor_ts),
        "alarm_count": sum(len(v) for v in ne_alarms.values()),
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_groups(topic, coord_gen):
    _, _, resource_by_nename, adjacency = build_indexes(topic)
    # 全量网元名（resources 里所有网元，按名排序保证确定性）
    all_ne_names = sorted(resource_by_nename.keys())

    grouped = OrderedDict()
    skipped_no_group = 0
    for alarm in topic.get("alarms") or []:
        if not isinstance(alarm, dict):
            continue
        group_id = _text(alarm.get("faultGroupId"))
        if not group_id:
            skipped_no_group += 1
            continue
        grouped.setdefault(group_id, []).append(alarm)

    groups = []
    for group_id, alarms in grouped.items():
        groups.append(build_group_object(
            group_id, alarms, resource_by_nename, adjacency,
            coord_gen, all_ne_names,
        ))
    return groups, skipped_no_group


def _safe_filename(name, fallback):
    """把故障组ID转成文件系统安全的文件名（逐字来自 complete_group_topology._safe_filename）。"""
    text = _text(name) or fallback
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    text = text.strip().strip(".").strip()  # Windows 不允许以空格或点结尾
    if not text:
        text = fallback
    return text[:120]


def write_outputs(groups, out_dir):
    """每个故障组写一个单行 <uuid>.jsonl 到目录（与 complete_group_topology.py --per-file 命名一致，
    可直接放进 visualization/data/ 用 start_labeling.bat 加载）。返回写出的文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = {}
    written = []
    for index, group in enumerate(groups):
        base = _safe_filename(group.get("uuid", ""), f"group_{index}")
        if base in used:
            used[base] += 1
            name = f"{base}_{used[base]}"
        else:
            used[base] = 0
            name = base
        path = out_dir / f"{name}.jsonl"
        line = json.dumps(group, ensure_ascii=False, separators=(",", ":"))
        path.write_text(line + "\n", encoding="utf-8")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="把 microwave_topic/test.json 转成 ne_propagation_visualizer.html 可展示的故障组"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(Path(__file__).with_name("test.json")),
        help="拓扑导出 JSON，默认: microwave_topic/test.json",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).with_name("fault_groups")),
        help=(
            "输出目录，每个故障组一个 <uuid>.jsonl（单行，字段与 complete_group_topology.py --per-file 一致）。"
            "默认 microwave_topic/fault_groups/。直接指向 visualization/data/ 即可用 start_labeling.bat 加载"
        ),
    )
    parser.add_argument(
        "--base-lat",
        type=float,
        default=DEFAULT_BASE_LAT,
        help=f"合成站点经纬度的中心纬度，默认 {DEFAULT_BASE_LAT}",
    )
    parser.add_argument(
        "--base-lon",
        type=float,
        default=DEFAULT_BASE_LON,
        help=f"合成站点经纬度的中心经度，默认 {DEFAULT_BASE_LON}",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=DEFAULT_SPREAD,
        help=f"站点相对中心的最大散布（度），越小越聚拢，默认 {DEFAULT_SPREAD}（约6~7km）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as fr:
        topic = json.load(fr)
    if not isinstance(topic, dict):
        parser.error("输入顶层必须是对象（含 alarms/resources/resourceRelations）")

    coord_gen = SiteCoordGenerator(
        base_lat=args.base_lat, base_lon=args.base_lon, spread=args.spread
    )

    groups, skipped_no_group = build_groups(
        topic, coord_gen
    )

    if not groups:
        print("⚠️ 没有解析出任何故障组（检查 alarms[*].faultGroupId 是否存在）", file=sys.stderr)

    written = write_outputs(groups, args.output)

    stats = {
        "input": str(input_path),
        "group_count": len(groups),
        "skipped_alarms_without_group": skipped_no_group,
        "output_dir": str(Path(args.output)),
        "output_files": [str(p) for p in written],
        "synthesized_site_coords": len(coord_gen._cache),
        "coord_center": [args.base_lat, args.base_lon],
        "coord_spread_deg": args.spread,
        "groups": [
            {
                "faultGroupId": g["uuid"],
                "alarm_count": g["alarm_count"],
                "ne_count": len(g["group_info"][g["uuid"]]["ne_list"]),
                "core_ne_count": len(g["group_info"][g["uuid"]]["core_ne_list"]),
                "site_count": len(g["group_info"][g["uuid"]]["site_list"]),
                "root_sites": g["group_info"][g["uuid"]]["root_sites"],
                "root_nes": g["group_info"][g["uuid"]]["root_nes"],
                "upstream_undetermined_sites": g["group_info"][g["uuid"]]["upstream_undetermined_sites"],
                "upstream_of": {
                    s: {"upstream_site": i["upstream_site"], "hop": i["hop"]}
                    for s, i in (g.get("root_cause") or {}).get("upstream_of", {}).items()
                },
                "site_root_cause": (g.get("root_cause") or {}).get("site_root_cause", {}),
            }
            for g in groups
        ],
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if written:
        print(
            f"\n➡️ 已写出 {len(written)} 个 <uuid>.jsonl 到 {Path(args.output)}。"
            f"\n   把它们（或直接 -o 指向 visualization/data/）放进 visualization/data/ 后运行 start_labeling.bat 加载。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
