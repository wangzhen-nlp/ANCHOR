#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按站点 upstream_site_hops 信息补齐故障组拓扑。"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fault_grouping.site_topology import build_site_to_ne_ids, load_site_chain_index
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_CHAINS_JSON,
    SITE_GRAPH_JSON,
    resource_display,
)


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


def _load_site_chain_index(site_chains_path):
    if site_chains_path and Path(site_chains_path).exists():
        return load_site_chain_index(site_chains_path)[0]
    if site_chains_path:
        print(f"⚠️ site_chains 文件不存在，将只保留原始告警站点: {site_chains_path}", file=sys.stderr)
    return {}


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


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _upstream_site_hops(site_id, site_chain_index, include_self=False):
    site_id = _normalize_text(site_id)
    hops = {site_id: 0} if include_self and site_id else {}
    info = site_chain_index.get(site_id, {}) if isinstance(site_chain_index, dict) else {}
    if not isinstance(info, dict):
        return hops
    for upstream_site, hop in (info.get("upstream_site_hops") or {}).items():
        upstream_site = _normalize_text(upstream_site)
        hop = _safe_int(hop)
        if upstream_site and hop is not None:
            hops[upstream_site] = min(hop, hops.get(upstream_site, hop))
    return hops


def _select_nearest_common_upstream(alarm_sites, site_chain_index):
    hops_by_site = {
        site_id: _upstream_site_hops(site_id, site_chain_index, include_self=True)
        for site_id in alarm_sites
    }
    common_candidates = None
    for site_id in alarm_sites:
        candidates = set(hops_by_site.get(site_id, {}))
        common_candidates = candidates if common_candidates is None else common_candidates & candidates
    if not common_candidates:
        return None, hops_by_site, {}

    common_upstream_site = min(
        common_candidates,
        key=lambda candidate: (
            sum(hops_by_site[site_id][candidate] for site_id in alarm_sites),
            max(hops_by_site[site_id][candidate] for site_id in alarm_sites),
            candidate,
        ),
    )
    return common_upstream_site, hops_by_site, {
        site_id: hops_by_site[site_id][common_upstream_site]
        for site_id in alarm_sites
    }


def _select_farthest_upstreams(alarm_sites, site_chain_index):
    farthest_by_site = {}
    for site_id in alarm_sites:
        hops = _upstream_site_hops(site_id, site_chain_index, include_self=False)
        if not hops:
            continue
        max_hop = max(hops.values())
        farthest_site = min(candidate for candidate, hop in hops.items() if hop == max_hop)
        farthest_by_site[site_id] = {
            "site_id": farthest_site,
            "hop": max_hop,
        }
    return farthest_by_site


def _build_site_completion(alarm_sites, site_chain_index):
    alarm_sites = sorted(set(site for site in alarm_sites if site))
    selected_sites = set(alarm_sites)
    common_upstream_site, hops_by_site, common_upstream_hops = _select_nearest_common_upstream(
        alarm_sites,
        site_chain_index,
    )
    farthest_upstream_sites = {}
    if common_upstream_site:
        selected_sites.add(common_upstream_site)
    else:
        farthest_upstream_sites = _select_farthest_upstreams(alarm_sites, site_chain_index)
        for selected in farthest_upstream_sites.values():
            selected_sites.add(selected["site_id"])

    return {
        "selected_sites": selected_sites,
        "common_upstream_site": common_upstream_site,
        "common_upstream_hops": common_upstream_hops,
        "farthest_upstream_sites": farthest_upstream_sites,
        "upstream_site_hops": hops_by_site,
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
    return result


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

    ancestor_count = len(completion.get("highlight_site_ids") or [])
    if ancestor_output == "one":
        return ancestor_count == 1
    if ancestor_output == "multiple":
        return ancestor_count > 1
    raise ValueError(f"未知 ancestor_output: {ancestor_output}")


def complete_group_topology(group, ne_graph_data, site_graph_data, site_to_ne_ids, site_chain_index):
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
    completion = _build_site_completion(alarm_sites, site_chain_index)
    selected_sites = completion["selected_sites"]
    topology_highlight_sites = _build_topology_highlight_sites(completion)
    topology_highlight_site_ids = sorted(
        item["site_id"]
        for item in topology_highlight_sites
        if item.get("site_id")
    )

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
    for derived_role in ("context_site", "common_upstream_site", "farthest_upstream_site"):
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
    if common_upstream_sites:
        existing_role_mapping["common_upstream_site"] = sorted(common_upstream_sites)
    if farthest_upstream_sites:
        existing_role_mapping["farthest_upstream_site"] = sorted(farthest_upstream_sites)
    group["role_mapping"] = existing_role_mapping

    match_info = copy.deepcopy(match_info)
    match_info.setdefault("uuid", group_id)
    match_info.setdefault("rule", group.get("rule", "alarm_group_id_rule"))
    match_info.setdefault("merged_rules", group.get("merged_rules", ["alarm_group_id_rule"]))
    match_info["role_mapping"] = existing_role_mapping
    group["match_info"] = match_info

    group["topology_completion"] = {
        "mode": "site_upstream_hops",
        "original_alarm_ne_ids": sorted(alarm_ne_ids),
        "original_alarm_site_ids": alarm_sites,
        "selected_site_ids": all_site_ids,
        "added_site_ids": context_sites,
        "added_ne_ids": sorted(ne_id for ne_id in included_ne_ids if ne_id not in set(alarm_ne_ids)),
        "common_upstream_site": completion["common_upstream_site"],
        "common_upstream_hops": completion["common_upstream_hops"],
        "farthest_upstream_sites": completion["farthest_upstream_sites"],
        "upstream_site_hops": completion["upstream_site_hops"],
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
    site_chain_index = _load_site_chain_index(site_chains_path)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)

    stats = {
        "input_group_count": 0,
        "output_group_count": 0,
        "common_upstream_group_count": 0,
        "fallback_upstream_group_count": 0,
        "one_ancestor_group_count": 0,
        "multiple_ancestor_group_count": 0,
        "skipped_by_ancestor_output_group_count": 0,
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
                )
                completion = completed.get("topology_completion", {})
                if completion.get("common_upstream_site"):
                    stats["common_upstream_group_count"] += 1
                elif len(completion.get("original_alarm_site_ids") or []) > 1:
                    stats["fallback_upstream_group_count"] += 1
                ancestor_count = len(completion.get("highlight_site_ids") or [])
                if ancestor_count == 1:
                    stats["one_ancestor_group_count"] += 1
                elif ancestor_count > 1:
                    stats["multiple_ancestor_group_count"] += 1
                if not _should_output_by_ancestor_count(completion, ancestor_output):
                    stats["skipped_by_ancestor_output_group_count"] += 1
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
