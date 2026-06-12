#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the health of alarm-native fault groups.

This tool loads raw/sorted alarms with the same region-filtering path used by
train_alarm_mhp.py, groups alarms by their native group id field (default:
故障组ID), converts those groups to the visual JSONL shape, and reuses
analyze_visual_group_metrics.py so baseline and stream metrics are comparable.

Example:
    python fault_grouping/tools/analyze_alarm_group_baseline_metrics.py ./0301-0306/ \
        --regions "EAST JAVA" \
        -o alarm_group_baseline.metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from fault_grouping.tools.alarm_group_baseline import (
    build_baseline_records,
    load_json_if_exists,
    write_json,
    write_jsonl,
)
from fault_grouping.tools.analyze_visual_group_metrics import analyze, print_summary
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON, resource_display
from topology_tools.region_utils import parse_regions


def _print_region_filter(metadata):
    payload = (metadata or {}).get("region_filter") or {}
    if payload.get("enabled"):
        kept = payload.get("kept_event_count", payload.get("raw_kept_alarm_count", 0))
        total = payload.get("input_event_count", payload.get("raw_checked_alarm_count", 0))
        print(
            "[baseline] region filter: "
            f"regions={payload.get('regions', [])}, "
            f"events={kept}/{total}, "
            f"allowed_devices={payload.get('allowed_device_count', 0)}",
            flush=True,
        )
    else:
        print(
            f"[baseline] region filter: disabled; events={payload.get('input_event_count', 'n/a')}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="按告警自带故障组ID评估 baseline 健康度")
    parser.add_argument("alarms", help="原始告警输入或 prepare_sorted_alarms 缓存")
    parser.add_argument("-o", "--output", default="", help="指标 JSON 输出；为空则只打印摘要")
    parser.add_argument(
        "--baseline-output",
        default="",
        help="可选：保存构造出来的 baseline visual JSONL，便于后续复查",
    )
    parser.add_argument("--group-field", default="故障组ID", help="告警中的故障组字段名，默认: 故障组ID")
    parser.add_argument("--min-group-events", type=int, default=2, help="丢弃小于该告警数的原始故障组，默认 2")
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help="只保留指定 region 的告警；可重复传入或用逗号分隔，口径与 train_alarm_mhp.py 一致",
    )
    parser.add_argument("--start-time", default="", help="开始时间，传给告警加载器")
    parser.add_argument("--end-time", default="", help="结束时间，传给告警加载器")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0, help="清除告警延迟参数，传给告警加载器")
    parser.add_argument(
        "--topo",
        default=SITE_GRAPH_BY_NE_JSON,
        help=f"Site topology for raw alarm inputs. Default: {resource_display('site_graph_by_ne.json')}.",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"NE graph JSON，默认 {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--site-graph",
        default=SITE_GRAPH_JSON,
        help=f"Site graph JSON，默认 {resource_display('site_graph.json')}",
    )
    parser.add_argument("--topo-max-hops", type=int, default=3, help="拓扑关系计算的最大 hop，默认 3")
    parser.add_argument("--max-pairwise-ne", type=int, default=200, help="每组最多对多少个 NE 做全 pair 拓扑统计，默认 200；0 表示关闭")
    parser.add_argument("--risk-duration-sec", type=float, default=2 * 3600, help="long_duration 风险阈值，默认 7200")
    parser.add_argument("--risk-site-count", type=int, default=10, help="many_sites 风险阈值，默认 10")
    parser.add_argument("--risk-unknown-pair-ratio", type=float, default=0.5, help="high_unknown_pair_ratio 风险阈值，默认 0.5")
    parser.add_argument("--health-target-duration-sec", type=float, default=3600.0, help="健康度 time_compactness 的 p90 目标时长，默认 3600")
    parser.add_argument("--health-target-virtual-ratio", type=float, default=0.2, help="健康度 virtual_reasonableness 的目标虚拟告警比例，默认 0.2")
    parser.add_argument("--health-target-size-p50", type=float, default=2.0, help="健康度 size_reasonableness 的 p50 目标真实告警数，默认 2")
    parser.add_argument("--health-target-size-p90", type=float, default=20.0, help="健康度 size_reasonableness 的 p90 合理上限，默认 20")
    parser.add_argument("--health-target-size-p99", type=float, default=100.0, help="健康度 size_reasonableness 的 p99 合理上限，默认 100")
    parser.add_argument("--no-detail", action="store_true", help="输出指标 JSON 不包含逐组 detail，文件更小")
    parser.add_argument("--progress-every", type=int, default=1000, help="指标阶段每处理 N 个组打印一次进度；0 表示关闭")
    parser.add_argument("--quiet", action="store_true", help="不打印加载/构造阶段日志")
    args = parser.parse_args()

    if args.min_group_events < 1:
        parser.error("--min-group-events must be >= 1")
    if args.topo_max_hops < 1:
        parser.error("--topo-max-hops must be >= 1")
    if args.max_pairwise_ne < 0:
        parser.error("--max-pairwise-ne must be >= 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    selected_regions = parse_regions(args.regions)
    if not args.quiet:
        print(f"[baseline] loading alarms: {args.alarms}", flush=True)
    alarm_events, alarm_metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
        regions=selected_regions,
        show_progress=not args.quiet,
    )
    if not args.quiet:
        print(f"[baseline] loaded alarm events: {len(alarm_events)}", flush=True)
        _print_region_filter(alarm_metadata)

    ne_graph_data = load_json_if_exists(args.ne_graph)
    records = build_baseline_records(
        alarm_events,
        group_field=args.group_field,
        ne_graph_data=ne_graph_data,
        min_group_events=args.min_group_events,
    )
    if not args.quiet:
        print(
            f"[baseline] native groups: {len(records)} "
            f"(group_field={args.group_field!r}, min_group_events={args.min_group_events})",
            flush=True,
        )

    temp_path = None
    visual_path = args.baseline_output
    try:
        if visual_path:
            write_jsonl(visual_path, records)
            if not args.quiet:
                print(f"baseline visual groups written to: {visual_path}; groups={len(records)}", flush=True)
        else:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".alarm_group_baseline.visual.jsonl",
                encoding="utf-8",
                delete=False,
            )
            temp_path = handle.name
            with handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            visual_path = temp_path

        result = analyze(
            visual_path,
            ne_graph_path=args.ne_graph,
            site_graph_path=args.site_graph,
            topo_max_hops=args.topo_max_hops,
            max_pairwise_ne=args.max_pairwise_ne,
            risk_duration_sec=args.risk_duration_sec,
            risk_site_count=args.risk_site_count,
            risk_unknown_pair_ratio=args.risk_unknown_pair_ratio,
            health_target_duration_sec=args.health_target_duration_sec,
            health_target_virtual_ratio=args.health_target_virtual_ratio,
            health_target_size_p50=args.health_target_size_p50,
            health_target_size_p90=args.health_target_size_p90,
            health_target_size_p99=args.health_target_size_p99,
            include_details=not args.no_detail,
            progress_every=args.progress_every,
            verbose=not args.quiet,
        )
        result.setdefault("meta", {})["baseline"] = {
            "alarm_input": os.path.abspath(args.alarms),
            "group_field": args.group_field,
            "min_group_events": args.min_group_events,
            "region_filter": (alarm_metadata or {}).get("region_filter") or {},
            "alarm_metadata": alarm_metadata,
            "baseline_visual_output": os.path.abspath(args.baseline_output) if args.baseline_output else "",
        }
        print_summary(result)
        if args.output:
            write_json(args.output, result)
            print(f"指标已保存: {args.output}", flush=True)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
