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

输出：每个 faultGroupId 一个「原始格式」故障组对象（含 ne_info / group_info / symptoms /
match_info），可被 ne_propagation_visualizer.html 的 loadOriginalFormat 直接加载。
  - <output>.jsonl        每行一个故障组，供故障组总览页浏览
  - 当只有一个故障组时，额外写出同名 <output>.json 单对象，方便直接拖进 NE 传播图页面
"""

import argparse
import hashlib
import json
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

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

    # 网元-网元邻接（以 neName 为节点）
    adjacency = defaultdict(dict)  # ne -> {peer_ne: {layers:set}}
    for rel in topic.get("resourceRelations") or []:
        if not isinstance(rel, dict):
            continue
        src = vid_to_nename.get(_text(rel.get("srcVid")))
        dst = vid_to_nename.get(_text(rel.get("dstVid")))
        if not src or not dst or src == dst:
            continue
        layer = _text(rel.get("linkLayer")) or "link"
        adjacency[src].setdefault(dst, {"layers": set()})["layers"].add(layer)
        adjacency[dst].setdefault(src, {"layers": set()})["layers"].add(layer)

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
# 单个故障组构建
# --------------------------------------------------------------------------- #
def build_group_object(
    group_id,
    alarms,
    resource_by_nename,
    adjacency,
    coord_gen,
    include_neighbors=True,
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

    # 2) 可选：把核心网元的直接拓扑邻居也纳入，便于展示传播路径
    display_nes = list(core_nes)
    if include_neighbors:
        core_set = set(core_nes)
        neighbors = []
        for ne in core_nes:
            for peer in adjacency.get(ne, {}):
                if peer not in core_set and peer not in neighbors:
                    neighbors.append(peer)
        display_nes.extend(neighbors)

    display_set = set(display_nes)

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
        }

    # 4) group_info / match_info / 头部字段
    group_site_ids = sorted(site_ids)
    role_mapping = {"associated_site": group_site_ids}
    anchor_ts = min(timestamps) if timestamps else None
    match_info = {
        "uuid": group_id,
        "rule": "fault_group_id_rule",
        "merged_rules": ["fault_group_id_rule"],
        "related_group_uuids": [],
        "inferred_roots": {},
        "role_mapping": role_mapping,
        "symptoms": symptoms,
        "uses_missing_topology": False,
        "missing_topology_edges": [],
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
        "group_info": {
            group_id: {
                "ne_list": sorted(display_nes),
                "site_list": group_site_ids,
                "core_ne_list": sorted(core_nes),
            }
        },
        "group_anchor_ts": anchor_ts,
        "group_anchor_time": _fmt_ts(anchor_ts),
        "alarm_count": sum(len(v) for v in ne_alarms.values()),
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_groups(topic, coord_gen, include_neighbors=True):
    _, _, resource_by_nename, adjacency = build_indexes(topic)

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
            coord_gen, include_neighbors=include_neighbors,
        ))
    return groups, skipped_no_group


def write_outputs(groups, output_path):
    output_path = Path(output_path)
    jsonl_path = output_path if output_path.suffix == ".jsonl" else output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for group in groups:
            fw.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
            fw.write("\n")

    single_path = None
    if len(groups) == 1:
        single_path = jsonl_path.with_suffix(".json")
        with open(single_path, "w", encoding="utf-8") as fw:
            json.dump(groups[0], fw, ensure_ascii=False, indent=2)
    return jsonl_path, single_path


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
        default=str(Path(__file__).with_name("test_fault_groups.jsonl")),
        help="输出路径，默认 microwave_topic/test_fault_groups.jsonl（单故障组时另出同名 .json）",
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
    parser.add_argument(
        "--no-neighbors",
        action="store_true",
        help="只展示有告警的网元，不纳入其直接拓扑邻居（默认纳入邻居以展示传播路径）",
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
        topic, coord_gen, include_neighbors=not args.no_neighbors
    )

    if not groups:
        print("⚠️ 没有解析出任何故障组（检查 alarms[*].faultGroupId 是否存在）", file=sys.stderr)

    jsonl_path, single_path = write_outputs(groups, args.output)

    stats = {
        "input": str(input_path),
        "group_count": len(groups),
        "skipped_alarms_without_group": skipped_no_group,
        "jsonl_output": str(jsonl_path),
        "single_json_output": str(single_path) if single_path else None,
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
            }
            for g in groups
        ],
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if single_path:
        print(
            f"\n➡️ 直接把 {single_path} 拖进 visualization/ne_propagation_visualizer.html 即可查看该故障组。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
