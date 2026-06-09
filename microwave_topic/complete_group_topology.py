#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为按故障组ID聚合的输出补齐最小站点级联通拓扑。"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display


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


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fw.write("\n")


def _site_of_ne(ne_id, ne_graph_data, group_site_by_ne=None):
    group_site_by_ne = group_site_by_ne or {}
    ne_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    if isinstance(ne_info, dict):
        site_id = _normalize_text(ne_info.get("site_id", ""))
        if site_id:
            return site_id
    return _normalize_text(group_site_by_ne.get(ne_id, ""))


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


def _canonical_site_edge(site_a, site_b):
    return tuple(sorted((_normalize_text(site_a), _normalize_text(site_b))))


def _build_topology_index(ne_graph_data):
    ne_adj = defaultdict(set)
    intra_adj = defaultdict(lambda: defaultdict(set))
    site_adj = defaultdict(set)
    site_edge_reps = defaultdict(list)

    for source_ne, source_info in ne_graph_data.items():
        if not isinstance(source_info, dict):
            continue
        source_site = _normalize_text(source_info.get("site_id", ""))
        raw_links = source_info.get("link", {})
        if not isinstance(raw_links, dict):
            continue
        for target_ne, link_meta in raw_links.items():
            target_info = ne_graph_data.get(target_ne, {})
            if not isinstance(target_info, dict):
                target_info = {}
            target_site = _normalize_text(target_info.get("site_id", ""))
            if not target_site or not source_site:
                continue

            ne_adj[source_ne].add(target_ne)
            ne_adj[target_ne].add(source_ne)
            if source_site == target_site:
                intra_adj[source_site][source_ne].add(target_ne)
                intra_adj[source_site][target_ne].add(source_ne)
                continue

            site_adj[source_site].add(target_site)
            site_adj[target_site].add(source_site)
            edge_key = _canonical_site_edge(source_site, target_site)
            site_edge_reps[edge_key].append({
                "source_ne": source_ne,
                "target_ne": target_ne,
                "source_site": source_site,
                "target_site": target_site,
                "link_meta": link_meta,
            })

    for reps in site_edge_reps.values():
        reps.sort(key=lambda item: (item["source_ne"], item["target_ne"]))

    return {
        "ne_adj": ne_adj,
        "intra_adj": intra_adj,
        "site_adj": site_adj,
        "site_edge_reps": site_edge_reps,
    }


def _group_site_by_ne(group):
    mapping = {}
    ne_info = group.get("ne_info", {})
    if isinstance(ne_info, dict):
        for ne_id, info in ne_info.items():
            if isinstance(info, dict):
                site_id = _normalize_text(info.get("site_id", ""))
                if site_id:
                    mapping[ne_id] = site_id

    symptoms = group.get("symptoms", [])
    if isinstance(symptoms, list):
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            ne_id = _normalize_text(
                symptom.get("alarm_source")
                or symptom.get("ne_id")
                or symptom.get("source")
                or ""
            )
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
        if not isinstance(alarm, dict):
            continue
        ne_id = _normalize_text(alarm.get("告警源", ""))
        if ne_id and ne_id not in ne_ids:
            ne_ids.append(ne_id)

    symptoms = group.get("symptoms", [])
    if isinstance(symptoms, list):
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            ne_id = _normalize_text(
                symptom.get("alarm_source")
                or symptom.get("ne_id")
                or symptom.get("source")
                or ""
            )
            if ne_id and ne_id not in ne_ids:
                ne_ids.append(ne_id)

    ne_info = group.get("ne_info", {})
    if isinstance(ne_info, dict):
        for ne_id, info in ne_info.items():
            alarms = info.get("alarm") if isinstance(info, dict) else None
            if isinstance(alarms, list) and alarms and ne_id not in ne_ids:
                ne_ids.append(ne_id)

    return ne_ids


def _shortest_site_path(start_sites, target_sites, site_adj):
    start_sites = sorted(_normalize_text(site) for site in start_sites if _normalize_text(site))
    target_sites = {_normalize_text(site) for site in target_sites if _normalize_text(site)}
    if not start_sites or not target_sites:
        return []
    for site in start_sites:
        if site in target_sites:
            return [site]

    queue = deque((site, [site]) for site in start_sites)
    seen = set(start_sites)
    while queue:
        site, path = queue.popleft()
        for neighbor in sorted(site_adj.get(site, ())):
            if neighbor in seen:
                continue
            next_path = path + [neighbor]
            if neighbor in target_sites:
                return next_path
            seen.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def _shortest_intra_site_ne_path(starts, targets, site_id, intra_adj):
    starts = sorted(set(starts))
    targets = set(targets)
    if not starts or not targets:
        return []
    for ne_id in starts:
        if ne_id in targets:
            return [ne_id]

    queue = deque((ne_id, [ne_id]) for ne_id in starts)
    seen = set(starts)
    adj = intra_adj.get(site_id, {})
    while queue:
        ne_id, path = queue.popleft()
        for neighbor in sorted(adj.get(ne_id, ())):
            if neighbor in seen:
                continue
            next_path = path + [neighbor]
            if neighbor in targets:
                return next_path
            seen.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def _orient_site_edge_rep(rep, left_site, right_site):
    if rep["source_site"] == left_site and rep["target_site"] == right_site:
        oriented = dict(rep)
        oriented["left_site"] = left_site
        oriented["right_site"] = right_site
        oriented["left_ne"] = rep["source_ne"]
        oriented["right_ne"] = rep["target_ne"]
        return oriented
    if rep["source_site"] == right_site and rep["target_site"] == left_site:
        oriented = dict(rep)
        oriented["left_site"] = left_site
        oriented["right_site"] = right_site
        oriented["left_ne"] = rep["target_ne"]
        oriented["right_ne"] = rep["source_ne"]
        return oriented
    return None


def _site_edge_candidates(left_site, right_site, site_edge_reps):
    edge = _canonical_site_edge(left_site, right_site)
    candidates = []
    for rep in site_edge_reps.get(edge, []):
        oriented = _orient_site_edge_rep(rep, left_site, right_site)
        if oriented is not None:
            candidates.append(oriented)
    candidates.sort(key=lambda item: (item["left_ne"], item["right_ne"], item["source_ne"], item["target_ne"]))
    return candidates


def _intra_connect_cost(site_id, source_ne, target_ne, included_ne_ids, intra_adj):
    if not source_ne or not target_ne or source_ne == target_ne:
        return (0, 0, [])
    path = _shortest_intra_site_ne_path({source_ne}, {target_ne}, site_id, intra_adj)
    if not path:
        return (1, 10**9, [])
    new_count = sum(1 for ne_id in path if ne_id not in set(included_ne_ids))
    return (0, new_count, path)


def _attach_endpoint_cost(site_id, endpoint_ne, included_ne_ids, ne_graph_data, group_site_by_ne, intra_adj):
    site_included = sorted(
        ne_id
        for ne_id in included_ne_ids
        if _site_of_ne(ne_id, ne_graph_data, group_site_by_ne) == site_id
    )
    if not site_included or endpoint_ne in site_included:
        return (0, 0, [])
    path = _shortest_intra_site_ne_path(site_included, {endpoint_ne}, site_id, intra_adj)
    if not path:
        return (1, 10**9, [])
    new_count = sum(1 for ne_id in path if ne_id not in set(included_ne_ids))
    return (0, new_count, path)


def _select_site_path_reps(site_path, site_edge_reps, included_ne_ids, alarm_ne_ids, ne_graph_data, group_site_by_ne, intra_adj):
    if len(site_path) < 2:
        return []

    edge_candidates = [
        _site_edge_candidates(left_site, right_site, site_edge_reps)
        for left_site, right_site in zip(site_path, site_path[1:])
    ]
    if any(not candidates for candidates in edge_candidates):
        return []

    included_ne_ids = set(included_ne_ids)
    alarm_ne_ids = set(alarm_ne_ids)

    # DP state: right_ne of previous edge -> (score_tuple, selected_reps)
    states = {}
    first_left_site = site_path[0]
    for candidate in edge_candidates[0]:
        attach_bad, attach_new, _path = _attach_endpoint_cost(
            first_left_site,
            candidate["left_ne"],
            included_ne_ids,
            ne_graph_data,
            group_site_by_ne,
            intra_adj,
        )
        endpoint_new = int(candidate["left_ne"] not in included_ne_ids) + int(candidate["right_ne"] not in included_ne_ids)
        non_alarm_new = int(candidate["left_ne"] not in alarm_ne_ids) + int(candidate["right_ne"] not in alarm_ne_ids)
        score = (attach_bad, attach_new, endpoint_new, non_alarm_new, candidate["left_ne"], candidate["right_ne"])
        existing = states.get(candidate["right_ne"])
        if existing is None or score < existing[0]:
            states[candidate["right_ne"]] = (score, [candidate])

    for edge_index in range(1, len(edge_candidates)):
        site_id = site_path[edge_index]
        next_states = {}
        for previous_right_ne, (previous_score, previous_reps) in states.items():
            for candidate in edge_candidates[edge_index]:
                connect_bad, connect_new, _path = _intra_connect_cost(
                    site_id,
                    previous_right_ne,
                    candidate["left_ne"],
                    included_ne_ids,
                    intra_adj,
                )
                endpoint_new = int(candidate["left_ne"] not in included_ne_ids) + int(candidate["right_ne"] not in included_ne_ids)
                non_alarm_new = int(candidate["left_ne"] not in alarm_ne_ids) + int(candidate["right_ne"] not in alarm_ne_ids)
                step_score = (connect_bad, connect_new, endpoint_new, non_alarm_new, candidate["left_ne"], candidate["right_ne"])
                score = tuple(previous_score[idx] + step_score[idx] for idx in range(4)) + (
                    previous_score[4],
                    previous_score[5],
                    candidate["left_ne"],
                    candidate["right_ne"],
                )
                existing = next_states.get(candidate["right_ne"])
                if existing is None or score < existing[0]:
                    next_states[candidate["right_ne"]] = (score, previous_reps + [candidate])
        states = next_states

    final_site = site_path[-1]
    best = None
    for previous_right_ne, (score, reps) in states.items():
        attach_bad, attach_new, _path = _attach_endpoint_cost(
            final_site,
            previous_right_ne,
            included_ne_ids,
            ne_graph_data,
            group_site_by_ne,
            intra_adj,
        )
        final_score = (score[0] + attach_bad, score[1] + attach_new) + score[2:]
        if best is None or final_score < best[0]:
            best = (final_score, reps)
    if best is None:
        return []
    # 不引入无法接回当前故障组拓扑的站间端点；否则会把非关键设备挂进输出。
    if best[0][0] > 0:
        return []
    return best[1]


def _shortest_feasible_site_path(
    start_sites,
    target_sites,
    site_adj,
    site_edge_reps,
    included_ne_ids,
    alarm_ne_ids,
    ne_graph_data,
    group_site_by_ne,
    intra_adj,
):
    start_sites = sorted(_normalize_text(site) for site in start_sites if _normalize_text(site))
    target_sites = {_normalize_text(site) for site in target_sites if _normalize_text(site)}
    if not start_sites or not target_sites:
        return [], []

    queue = deque((site, [site]) for site in start_sites)
    while queue:
        site, path = queue.popleft()
        for neighbor in sorted(site_adj.get(site, ())):
            if neighbor in path:
                continue
            next_path = path + [neighbor]
            if neighbor in target_sites:
                reps = _select_site_path_reps(
                    next_path,
                    site_edge_reps,
                    included_ne_ids,
                    alarm_ne_ids,
                    ne_graph_data,
                    group_site_by_ne,
                    intra_adj,
                )
                if reps:
                    return next_path, reps
                continue
            queue.append((neighbor, next_path))
    return [], []


def _connect_intra_site_devices(included_ne_ids, ne_graph_data, group_site_by_ne, intra_adj):
    included_ne_ids = set(included_ne_ids)
    site_to_included = defaultdict(list)
    for ne_id in included_ne_ids:
        site_id = _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
        if site_id:
            site_to_included[site_id].append(ne_id)

    added_paths = []
    for site_id, site_ne_ids in sorted(site_to_included.items()):
        site_ne_ids = sorted(set(site_ne_ids))
        if len(site_ne_ids) <= 1:
            continue
        connected = {site_ne_ids[0]}
        remaining = set(site_ne_ids[1:])
        while remaining:
            path = _shortest_intra_site_ne_path(connected, remaining, site_id, intra_adj)
            if not path:
                break
            included_ne_ids.update(path)
            connected.update(path)
            remaining.difference_update(connected)
            if len(path) > 1:
                added_paths.append({
                    "site_id": site_id,
                    "ne_path": path,
                })

    return included_ne_ids, added_paths


def _connected_ne_ids(start_ne, included_ne_ids, ne_adj):
    if not start_ne:
        return set()
    included_ne_ids = set(included_ne_ids)
    seen = {start_ne}
    queue = deque([start_ne])
    while queue:
        ne_id = queue.popleft()
        for neighbor in sorted(ne_adj.get(ne_id, ())):
            if neighbor not in included_ne_ids or neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _is_included_ne_connected(included_ne_ids, ne_adj):
    included_ne_ids = sorted(set(included_ne_ids))
    if len(included_ne_ids) <= 1:
        return True
    connected = _connected_ne_ids(included_ne_ids[0], included_ne_ids, ne_adj)
    return len(connected) == len(included_ne_ids)


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


class _NullProgress:
    def update(self, _stats):
        pass

    def close(self):
        pass


class _TqdmGroupProgress:
    def __init__(self, total):
        from tqdm import tqdm

        self._bar = tqdm(
            total=total,
            desc="补齐拓扑",
            unit="组",
            dynamic_ncols=True,
            file=sys.stderr,
        )

    def update(self, stats):
        self._bar.update(1)
        self._bar.set_postfix({
            "新增设备": stats["added_ne_count"],
            "NE联通": stats["ne_level_connected_group_count"],
        })

    def close(self):
        self._bar.close()


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _StderrGroupProgress:
    def __init__(self, total):
        self.total = max(int(total), 0)
        self.current = 0
        self.start_time = time.time()
        self._render({"added_ne_count": 0, "ne_level_connected_group_count": 0}, force=True)

    def update(self, stats):
        self.current += 1
        self._render(stats)

    def close(self):
        self._render({"added_ne_count": "", "ne_level_connected_group_count": ""}, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, stats, force=False):
        elapsed = max(time.time() - self.start_time, 1e-6)
        rate = self.current / elapsed
        if self.total > 0:
            percent = min(self.current / self.total, 1.0) * 100
            remaining = max(self.total - self.current, 0)
            eta = _format_duration(remaining / rate) if rate > 0 else "00:00"
            message = (
                f"\r补齐拓扑: {self.current}/{self.total} "
                f"{percent:6.2f}% ({rate:.1f}组/s, ETA {eta})"
            )
        else:
            message = f"\r补齐拓扑: {self.current} ({rate:.1f}组/s)"

        added_ne_count = stats.get("added_ne_count", "")
        ne_connected_count = stats.get("ne_level_connected_group_count", "")
        if added_ne_count != "" or ne_connected_count != "":
            message += f" | 新增设备 {added_ne_count}，NE联通 {ne_connected_count}"

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


def _build_filtered_link_info(ne_id, included_ne_ids, ne_graph_data):
    ne_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    raw_links = ne_info.get("link", {}) if isinstance(ne_info, dict) else {}
    if not isinstance(raw_links, dict):
        return {}
    included_ne_ids = set(included_ne_ids)
    return {
        target_ne: _format_link_meta(link_meta)
        for target_ne, link_meta in sorted(raw_links.items())
        if target_ne in included_ne_ids and target_ne != ne_id
    }


def _build_ne_info_entry(ne_id, group, included_ne_ids, alarm_ne_ids, ne_graph_data, site_graph_data, group_site_by_ne):
    existing = {}
    if isinstance(group.get("ne_info"), dict) and isinstance(group["ne_info"].get(ne_id), dict):
        existing = copy.deepcopy(group["ne_info"][ne_id])

    ne_info = ne_graph_data.get(ne_id, {}) if isinstance(ne_graph_data, dict) else {}
    if not isinstance(ne_info, dict):
        ne_info = {}
    site_id = _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    site_ctx = _site_context(site_id, site_graph_data, ne_info)
    is_alarm_ne = ne_id in set(alarm_ne_ids)

    entry = {
        "link": _build_filtered_link_info(ne_id, included_ne_ids, ne_graph_data),
        "group": group.get("uuid") or group.get("故障组ID") or group.get("match_info", {}).get("uuid", ""),
        "name": ne_info.get("name", existing.get("name", ne_id)),
        "site_id": site_id or existing.get("site_id", ""),
        "site_name": site_ctx["site_name"] or existing.get("site_name", ""),
        "site_type": site_ctx["site_type"] or existing.get("site_type", ""),
        "type": str(ne_info.get("type", existing.get("type", ""))).upper(),
        "network_type": str(ne_info.get("network_type", existing.get("network_type", ""))).upper(),
        "manufacturer": str(ne_info.get("manufacturer", existing.get("manufacturer", ""))).upper(),
        "running_status": ne_info.get("running_status", ne_info.get("status", existing.get("running_status", ""))),
        "domain": str(ne_info.get("domain", existing.get("domain", ""))).upper(),
        "region_id": site_ctx["region_id"] or existing.get("region_id", ""),
        "longitude": site_ctx["longitude"] if site_ctx["longitude"] != "" else existing.get("longitude", ""),
        "latitude": site_ctx["latitude"] if site_ctx["latitude"] != "" else existing.get("latitude", ""),
        "alarm": existing.get("alarm", []) if is_alarm_ne else [],
    }
    if not is_alarm_ne:
        entry["topology_added"] = True
    return entry


def complete_group_topology(group, ne_graph_data, site_graph_data, topo_index):
    group = copy.deepcopy(group)
    group_id = group.get("uuid") or group.get("故障组ID") or group.get("match_info", {}).get("uuid", "")
    group["uuid"] = group_id

    group_site_by_ne = _group_site_by_ne(group)
    alarm_ne_ids = _extract_alarm_ne_ids(group)
    included_ne_ids = set(alarm_ne_ids)
    alarm_sites = sorted({
        _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
        for ne_id in alarm_ne_ids
        if _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    })

    connected_sites = {alarm_sites[0]} if alarm_sites else set()
    remaining_seed_sites = set(alarm_sites[1:])
    site_paths = []
    selected_site_edges = []
    selected_site_edge_reps = []
    failed_site_paths = []
    seen_selected_rep_keys = set()
    seen_selected_site_edges = set()

    while remaining_seed_sites:
        site_path, path_reps = _shortest_feasible_site_path(
            connected_sites,
            remaining_seed_sites,
            topo_index["site_adj"],
            topo_index["site_edge_reps"],
            included_ne_ids,
            alarm_ne_ids,
            ne_graph_data,
            group_site_by_ne,
            topo_index["intra_adj"],
        )
        if not site_path or not path_reps:
            fallback_path = _shortest_site_path(connected_sites, remaining_seed_sites, topo_index["site_adj"])
            if fallback_path:
                failed_site_paths.append(fallback_path)
            break
        site_paths.append(site_path)
        for rep in path_reps:
            rep_key = (rep["source_ne"], rep["target_ne"], rep["source_site"], rep["target_site"])
            if rep_key in seen_selected_rep_keys:
                continue
            seen_selected_rep_keys.add(rep_key)
            selected_site_edge_reps.append(rep)
            included_ne_ids.add(rep["source_ne"])
            included_ne_ids.add(rep["target_ne"])
            site_edge = _canonical_site_edge(rep["source_site"], rep["target_site"])
            if site_edge not in seen_selected_site_edges:
                seen_selected_site_edges.add(site_edge)
                selected_site_edges.append(site_edge)
        # 只更新 included_ne_ids，供后续站点路径选择看到已引入的站内关键设备；
        # intra_site_paths 元数据在最终拓扑稳定后统一记录，避免重复/重叠路径噪声。
        included_ne_ids, _ = _connect_intra_site_devices(
            included_ne_ids,
            ne_graph_data,
            group_site_by_ne,
            topo_index["intra_adj"],
        )
        connected_sites.update(site_path)
        remaining_seed_sites.difference_update(connected_sites)

    included_ne_ids, intra_site_paths = _connect_intra_site_devices(
        included_ne_ids,
        ne_graph_data,
        group_site_by_ne,
        topo_index["intra_adj"],
    )
    ne_level_connected = _is_included_ne_connected(included_ne_ids, topo_index["ne_adj"])
    added_ne_ids = sorted(ne_id for ne_id in included_ne_ids if ne_id not in set(alarm_ne_ids))
    all_site_ids = sorted({
        _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
        for ne_id in included_ne_ids
        if _site_of_ne(ne_id, ne_graph_data, group_site_by_ne)
    })

    ne_info = {}
    for ne_id in sorted(included_ne_ids):
        ne_info[ne_id] = _build_ne_info_entry(
            ne_id,
            group,
            included_ne_ids,
            alarm_ne_ids,
            ne_graph_data,
            site_graph_data,
            group_site_by_ne,
        )

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

    alarm_site_set = set(alarm_sites)
    existing_role_mapping["associated_site"] = sorted(alarm_site_set)
    context_sites = sorted(set(all_site_ids) - alarm_site_set)
    if context_sites:
        existing_role_mapping["context_site"] = context_sites
    group["role_mapping"] = existing_role_mapping

    match_info = copy.deepcopy(match_info)
    match_info.setdefault("uuid", group_id)
    match_info.setdefault("rule", group.get("rule", "alarm_group_id_rule"))
    match_info.setdefault("merged_rules", group.get("merged_rules", ["alarm_group_id_rule"]))
    match_info["role_mapping"] = existing_role_mapping
    group["match_info"] = match_info

    group["topology_completion"] = {
        "site_level_connected": not remaining_seed_sites and not failed_site_paths,
        "ne_level_connected": ne_level_connected,
        "original_alarm_ne_ids": sorted(alarm_ne_ids),
        "added_ne_ids": added_ne_ids,
        "added_site_ids": context_sites,
        "site_paths": site_paths,
        "selected_site_edges": [
            {"source_site": left, "target_site": right}
            for left, right in selected_site_edges
        ],
        "selected_ne_edges": [
            {
                "source_ne": rep["source_ne"],
                "target_ne": rep["target_ne"],
                "source_site": rep["source_site"],
                "target_site": rep["target_site"],
            }
            for rep in selected_site_edge_reps
        ],
        "intra_site_paths": intra_site_paths,
        "failed_site_paths": failed_site_paths,
        "unconnected_seed_sites": sorted(remaining_seed_sites),
    }
    return group


def complete_groups(input_path, output_path, ne_graph_path, site_graph_path, show_progress=True):
    ne_graph_data = _load_json_object(ne_graph_path, "ne_graph", warn_if_missing=True)
    site_graph_data = _load_json_object(site_graph_path, "site_graph", warn_if_missing=True)
    topo_index = _build_topology_index(ne_graph_data)

    stats = {
        "input_group_count": 0,
        "output_group_count": 0,
        "site_level_connected_group_count": 0,
        "site_level_unconnected_group_count": 0,
        "ne_level_connected_group_count": 0,
        "ne_level_unconnected_group_count": 0,
        "added_ne_count": 0,
    }
    progress = _build_group_progress(input_path, show_progress)
    with open(output_path, "w", encoding="utf-8") as fw:
        try:
            for group in _iter_jsonl(input_path):
                stats["input_group_count"] += 1
                completed = complete_group_topology(group, ne_graph_data, site_graph_data, topo_index)
                completion = completed.get("topology_completion", {})
                if completion.get("site_level_connected"):
                    stats["site_level_connected_group_count"] += 1
                else:
                    stats["site_level_unconnected_group_count"] += 1
                if completion.get("ne_level_connected"):
                    stats["ne_level_connected_group_count"] += 1
                else:
                    stats["ne_level_unconnected_group_count"] += 1
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
    return stats


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="根据故障组 JSONL 补齐原始告警设备之间的最小站点级联通拓扑"
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
        show_progress=not args.no_progress,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
