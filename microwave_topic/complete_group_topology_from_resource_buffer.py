#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 resource_buffer.jsonl 为故障组补齐站点级拓扑。"""

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL
from fault_grouping.site_topology import build_site_to_ne_ids, normalize_site_chain_hops
from microwave_topic.complete_group_topology import (
    _blocked_ancestor_site_ids,
    _build_group_progress,
    _build_site_chain_component_index,
    _build_site_data_and_link_index,
    _build_weighted_upstream_adjacency,
    _normalize_site_id_set,
    _check_group_alarm_topology,
    _detect_restrict_relation,
    _group_uuid,
    _iter_jsonl,
    _safe_filename,
    _ancestor_highlight_count,
    _should_output_by_ancestor_count,
    complete_group_topology,
)


def _load_site_chain_index_from_data(site_chains_data):
    if not isinstance(site_chains_data, dict):
        raise ValueError("resource_buffer 中的 site_chains 顶层必须是对象")
    restrict_relation = _detect_restrict_relation(site_chains_data.get("meta", {}))
    raw_sites = site_chains_data.get("sites", {})
    site_chain_index = {}
    if isinstance(raw_sites, dict):
        for raw_site_id, raw_info in raw_sites.items():
            site_id = str(raw_site_id or "").strip()
            if not site_id or not isinstance(raw_info, dict):
                continue
            site_chain_index[site_id] = {
                "upstream_site_hops": normalize_site_chain_hops(
                    raw_info.get("upstream_site_hops")
                ),
                "downstream_site_hops": normalize_site_chain_hops(
                    raw_info.get("downstream_site_hops")
                ),
                "bidirectional_sites": _normalize_site_id_set(
                    raw_info.get("bidirectional_sites")
                ),
            }
    return site_chain_index, restrict_relation


def _load_topology_resources(resource_buffer_path):
    resources = load_resource_buffer(
        resource_buffer_path,
        wanted_types=("ne_graph", "site_graph", "site_chains"),
    )
    ne_graph_data = resources["ne_graph"]
    site_graph_data = resources["site_graph"]
    site_chain_index, restrict_relation = _load_site_chain_index_from_data(resources["site_chains"])
    if not isinstance(ne_graph_data, dict):
        raise ValueError("resource_buffer 中的 ne_graph 顶层必须是对象")
    if not isinstance(site_graph_data, dict):
        raise ValueError("resource_buffer 中的 site_graph 顶层必须是对象")
    return ne_graph_data, site_graph_data, site_chain_index, restrict_relation


def complete_groups_from_resource_buffer(
    input_path,
    output_path,
    resource_buffer_path,
    show_progress=True,
    ancestor_output="all",
    per_file=False,
    offline_duration_filter=False,
):
    (
        ne_graph_data,
        site_graph_data,
        site_chain_index,
        restrict_relation,
    ) = _load_topology_resources(resource_buffer_path)
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    site_has_data, site_has_ran, site_links, directed_edge_types = (
        _build_site_data_and_link_index(ne_graph_data)
    )
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
        "skipped_by_missing_alarm_topology_group_count": 0,
        "missing_alarm_source_group_count": 0,
        "missing_ne_graph_group_count": 0,
        "missing_ne_site_group_count": 0,
        "missing_site_graph_group_count": 0,
        "added_site_count": 0,
        "added_ne_count": 0,
    }

    progress = _build_group_progress(input_path, show_progress)
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
                    copy.deepcopy(group),
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
                if offline_duration_filter and not completion.get(
                    "offline_duration_filter", {}
                ).get("passes", False):
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
    stats["resource_buffer"] = resource_buffer_path
    stats["resources"] = ("ne_graph", "site_graph", "site_chains")
    stats["restrict_relation"] = restrict_relation
    stats["ancestor_output"] = ancestor_output
    stats["offline_duration_filter"] = offline_duration_filter
    return stats


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="使用 resource_buffer.jsonl 为故障组补齐站点级拓扑"
    )
    parser.add_argument("input", help="输入故障组 JSONL")
    parser.add_argument(
        "output",
        help="输出位置：默认为单个多行 JSONL 文件；加 --per-file 时为输出目录（每组一个单行 jsonl）",
    )
    parser.add_argument(
        "--resource-buffer",
        default=RESOURCE_BUFFER_JSONL,
        help=f"build_resource_buffer.py 生成的资源缓冲 JSONL，默认: {RESOURCE_BUFFER_JSONL}",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="每个故障组输出为单独的单行 jsonl 文件到 output 目录",
    )
    parser.add_argument(
        "--ancestor-output",
        choices=("all", "one", "multiple"),
        default="all",
        help="按补出的祖先站点数量筛选输出，默认 all",
    )
    parser.add_argument(
        "--offline-duration-filter",
        "--filter",
        action="store_true",
        dest="offline_duration_filter",
        help="按每站最长 offline 告警持续时间筛选故障组",
    )
    parser.add_argument("--no-progress", action="store_true", help="关闭处理进度输出")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    stats = complete_groups_from_resource_buffer(
        args.input,
        args.output,
        args.resource_buffer,
        show_progress=not args.no_progress,
        ancestor_output=args.ancestor_output,
        per_file=args.per_file,
        offline_duration_filter=args.offline_duration_filter,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
