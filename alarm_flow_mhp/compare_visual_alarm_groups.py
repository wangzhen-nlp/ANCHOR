#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare MHP visual fault groups with alarm-native group ids.

The MHP online stream writes match-rules-compatible visual JSONL. This wrapper
reuses the bidirectional site-coverage evaluator from ticket_recall, treating:

  - MHP visual groups as one clustering
  - the alarms' native group id field (default: 故障组ID) as the other clustering

Both sides are evaluated as gold labels in turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display
from ticket_recall.evaluation.compute_ultimate_group_alarm_group_metrics import (
    compute_ultimate_group_alarm_group_metrics,
)


def _is_disabled(value) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "none", "off"}


def _derive_case_output_path(output_file, suffix):
    base, _ext = os.path.splitext(output_file)
    return f"{base}.{suffix}.cases.jsonl"


def _rename_result(raw_result, *, visual_output, alarms):
    result = dict(raw_result)
    result["method"] = "alarm_flow_mhp_visual_alarm_group_comparison"
    result["mhp_visual_output"] = os.path.abspath(visual_output)
    result["alarm_input"] = os.path.abspath(alarms)

    if "ultimate_group_count" in result:
        result["mhp_group_count"] = result.pop("ultimate_group_count")
    if "ultimate_group_as_gold" in result:
        result["mhp_group_as_gold"] = result.pop("ultimate_group_as_gold")
    if "ultimate_group_as_gold_case_jsonl_output" in result:
        result["mhp_group_as_gold_case_jsonl_output"] = result.pop(
            "ultimate_group_as_gold_case_jsonl_output"
        )
    if "ultimate_group_as_gold_case_count" in result:
        result["mhp_group_as_gold_case_count"] = result.pop(
            "ultimate_group_as_gold_case_count"
        )
    return result


def _print_direction(label, payload):
    print(f"【{label}】")
    print(f"样本数: {payload.get('sample_count', 0)}")
    print(f"gold站点数分布: {payload.get('gold_site_count_distribution', {})}")
    print(f"平均召回率: {payload.get('average_recall', 0.0):.6f}")
    print(f"平均准确率: {payload.get('average_precision', 0.0):.6f}")
    print(f"平均F1: {payload.get('average_f1', 0.0):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="双向比较 alarm_flow_mhp visual 输出与告警自带故障组ID 的站点覆盖结果"
    )
    parser.add_argument("visual_output", help="alarm_flow_mhp 在线输出的 visual JSONL")
    parser.add_argument("alarms", help="原始告警输入，支持 jsonl/csv/zip/目录")
    parser.add_argument(
        "--group-field",
        default="故障组ID",
        help="告警/visual symptom 中的原始故障组字段名，默认: 故障组ID",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"用于通过告警源回填站点的 ne_graph 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--min-site-num",
        type=int,
        default=0,
        help="仅统计 gold label 站点数 >= 该值的样本；默认: 0（不过滤）",
    )
    parser.add_argument(
        "--no-domain-alarm",
        metavar="DOMAIN",
        help="如果当前 gold label 中出现来自指定 domain 的告警，则跳过该样本",
    )
    parser.add_argument(
        "--no-domain-site",
        metavar="DOMAIN",
        help="如果当前 gold label 的任一站点在 ne_graph 中包含指定 domain 设备，则跳过该样本",
    )
    parser.add_argument(
        "--require-domain-per-site",
        metavar="DOMAIN",
        help="先剔除不包含指定 domain 设备的 gold 站点，再做 min-site-num 过滤",
    )
    parser.add_argument(
        "--only-offline",
        action="store_true",
        help="仅统计包含 OFFLINE_ALARMS 的 gold label 样本",
    )
    parser.add_argument(
        "--only-one",
        action="store_true",
        help="只保留覆盖当前 gold 站点最多的单个预测 group，用它计算指标",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="允许在当前 gold 站点范围内，按时间窗把其它预测 group 做 loose 扩张",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=900,
        help="loose 模式使用的前后对称时间窗，单位秒，默认: 900",
    )
    parser.add_argument(
        "--no-potential",
        action="store_true",
        help="关闭默认的告警ID关联；默认开启以兼容旧版未透传故障组ID的 visual 文件",
    )
    parser.add_argument(
        "--only-unrecalled-predictions",
        action="store_true",
        help="输出 JSON 中两类 details 仅保留召回率不足 100%% 的预测；平均指标仍基于全部样本计算",
    )
    parser.add_argument(
        "--mhp-case-jsonl-output",
        default=None,
        help="MHP group 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成 sidecar，none 关闭",
    )
    parser.add_argument(
        "--alarm-group-case-jsonl-output",
        default=None,
        help="告警故障组ID 作为 gold 的未满召回样本可视化 jsonl；默认随主输出生成 sidecar，none 关闭",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mhp_visual_alarm_group_comparison.json",
        help="输出 JSON 文件，默认: mhp_visual_alarm_group_comparison.json",
    )
    args = parser.parse_args()

    mhp_case_output = args.mhp_case_jsonl_output
    alarm_case_output = args.alarm_group_case_jsonl_output
    if mhp_case_output is None and args.output:
        mhp_case_output = _derive_case_output_path(args.output, "mhp_group_as_gold")
    if alarm_case_output is None and args.output:
        alarm_case_output = _derive_case_output_path(args.output, "alarm_group_as_gold")
    if _is_disabled(mhp_case_output):
        mhp_case_output = None
    if _is_disabled(alarm_case_output):
        alarm_case_output = None

    raw_result = compute_ultimate_group_alarm_group_metrics(
        group_output_input=args.visual_output,
        alarm_input=args.alarms,
        group_field=args.group_field,
        ne_graph_file=args.ne_graph,
        min_site_num=args.min_site_num,
        no_domain_alarm=args.no_domain_alarm,
        no_domain_site=args.no_domain_site,
        require_domain_per_site=args.require_domain_per_site,
        only_offline=args.only_offline,
        only_one=args.only_one,
        loose=args.loose,
        window_seconds=args.window_seconds,
        potential=not args.no_potential,
        only_unrecalled_predictions=args.only_unrecalled_predictions,
        output_file=None,
        ultimate_case_jsonl_output_file=mhp_case_output,
        alarm_group_case_jsonl_output_file=alarm_case_output,
    )
    result = _rename_result(
        raw_result,
        visual_output=args.visual_output,
        alarms=args.alarms,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    _print_direction("MHP visual group 作为 gold", result.get("mhp_group_as_gold", {}))
    _print_direction("告警故障组ID 作为 gold", result.get("alarm_group_as_gold", {}))
    if args.output:
        print(f"结果已输出到: {args.output}")
    if result.get("mhp_group_as_gold_case_jsonl_output"):
        print(
            f"MHP group-case jsonl: {result['mhp_group_as_gold_case_jsonl_output']} "
            f"({result.get('mhp_group_as_gold_case_count', 0)} 条)"
        )
    if result.get("alarm_group_as_gold_case_jsonl_output"):
        print(
            f"告警故障组ID-case jsonl: {result['alarm_group_as_gold_case_jsonl_output']} "
            f"({result.get('alarm_group_as_gold_case_count', 0)} 条)"
        )


if __name__ == "__main__":
    main()
