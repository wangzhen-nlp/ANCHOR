import json
import os
import sys
import time

from argparse import ArgumentParser

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alarm_tools.alarm_types import CRITICAL_ALARMS
from alarm_tools.progress_utils import ProgressBar
from fault_csm_codex.engine import FaultCSMEngine, SUPPORTED_ALGORITHMS
from fault_grouping.alarm_events.io import (
    count_alarm_event_types,
    load_sorted_alarm_cache_with_stats,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
    warn_sorted_alarm_cache_option_mismatch,
)
from fault_grouping.alarm_events.sorted_cache import is_sorted_alarm_cache_file, write_sorted_alarm_cache
from fault_grouping.matching.group_output_builder import (
    build_alarm_metadata_index,
    build_jsonl_match_output,
)
from fault_grouping.rule_config import (
    data_link_adjacent_link_rule,
    data_link_adjacent_no_offline_rule,
    data_link_adjacent_offline_rule,
    data_no_offline_adjacent_optional_offline_rule,
    data_offline_adjacent_offline_rule,
    data_rule,
    link_rule,
    power_rule,
    transmission_rule,
)
from fault_grouping.site_topology import (
    build_site_to_ne_ids,
    build_site_topology_from_ne_graph,
    load_site_chain_index,
)
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_DEVICE_COUNTS_JSON,
    SITE_GRAPH_JSON,
    SITE_GRAPH_BY_NE_JSON,
    resource_display,
)


def build_arg_parser():
    parser = ArgumentParser(
        description=(
            "Continuous Subgraph Matching fault grouping. "
            "It keeps the same alarm/rule/output formats as match_rules.py, "
            "but maps alarm arrivals to dynamic graph edge updates."
        )
    )
    parser.add_argument("alarms", help="alarm stream input")
    parser.add_argument("output", help="output JSONL file")
    parser.add_argument("--topo", default=SITE_GRAPH_BY_NE_JSON, help=f"站点拓扑文件，默认: {resource_display('site_graph_by_ne.json')}")
    parser.add_argument("--site-chains", default="", help=f"可选 generate_site_chains.py 输出文件，默认不使用；推荐: {resource_display('site_chains.json')}")
    parser.add_argument("--site-domain", default=SITE_DEVICE_COUNTS_JSON, help=f"站点画像文件，默认: {resource_display('site_device_counts.json')}")
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=f"site_graph.json 文件，默认: {resource_display('site_graph.json')}")
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}")
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_ALGORITHMS,
        default="graphflow",
        help=(
            "CSM匹配算法：incisomatch/sjtree/graphflow/iedyn/turboflux/symbi；"
            "默认 graphflow"
        ),
    )
    parser.add_argument("--clear-delay-sec", type=float, default=420.0, help="清除告警最小延迟时间，默认 420 秒")
    parser.add_argument(
        "--alarm-active-sec",
        type=float,
        default=0.0,
        help="告警自动活跃时间；发生时间+该值后自动删除动态告警边，默认 0 表示不启用",
    )
    parser.add_argument("--start_time", help="仅处理告警首次发生时间 >= 该时间")
    parser.add_argument("--end_time", help="仅处理告警首次发生时间 <= 该时间")
    parser.add_argument("--rule", action="append", default=[], help="仅启用指定规则，可重复传入，也支持逗号分隔")
    parser.add_argument("--sorted-alarms-input", default="", help="直接加载 prepare_sorted_alarms.py 生成的排序告警缓存")
    parser.add_argument("--sorted-alarms-output", default="", help="从原始告警加载并排序后，额外写出排序告警缓存")
    parser.add_argument("--compact-output", action="store_true", help="输出轻量化 JSONL，兼容可视化页面")
    return parser


def parse_selected_rules(rule_args, valid_rule_names):
    if not rule_args:
        return []
    valid = set(valid_rule_names)
    selected = []
    seen = set()
    for raw_value in rule_args:
        for part in str(raw_value).replace("，", ",").split(","):
            rule_name = part.strip()
            if not rule_name or rule_name in seen:
                continue
            if rule_name not in valid:
                raise ValueError(f"未知规则名: {rule_name}；可选值: {', '.join(sorted(valid))}")
            seen.add(rule_name)
            selected.append(rule_name)
    return selected


def build_rules_config(args):
    all_rules = {
        "transmission_rule": transmission_rule,
        "link_rule": link_rule,
        "power_rule": power_rule,
        "data_rule": data_rule,
        "data_link_adjacent_no_offline_rule": data_link_adjacent_no_offline_rule,
        "data_link_adjacent_offline_rule": data_link_adjacent_offline_rule,
        "data_link_adjacent_link_rule": data_link_adjacent_link_rule,
        "data_no_offline_adjacent_optional_offline_rule": data_no_offline_adjacent_optional_offline_rule,
        "data_offline_adjacent_offline_rule": data_offline_adjacent_offline_rule,
    }
    selected = parse_selected_rules(args.rule, all_rules)
    rules = {name: all_rules[name] for name in selected} if selected else all_rules
    print("启用规则: " + ", ".join(rules))
    return rules


def normalize_topology(raw_topology):
    normalized = {}
    for site_id, downstreams in raw_topology.items():
        if isinstance(downstreams, dict):
            normalized[str(site_id)] = [str(item) for item in downstreams.keys()]
        elif isinstance(downstreams, list):
            normalized[str(site_id)] = [str(item) for item in downstreams]
        else:
            normalized[str(site_id)] = []
    return normalized


def load_static_context(args):
    ne_graph_data = json.load(open(args.ne_graph, "r", encoding="utf-8"))
    if args.topo:
        print(f"加载显式站点拓扑: {args.topo}")
        topo_downstream_map = normalize_topology(json.load(open(args.topo, "r", encoding="utf-8")))
        valid_sites = set(topo_downstream_map)
        for downstreams in topo_downstream_map.values():
            valid_sites.update(downstreams)
    else:
        print(f"基于 ne_graph 原始连边构建站点传播拓扑: {args.ne_graph}")
        topo_downstream_map, valid_sites = build_site_topology_from_ne_graph(ne_graph_data)
        topo_downstream_map = normalize_topology(topo_downstream_map)

    site_chain_index = None
    if args.site_chains:
        print(f"加载预计算站点链路: {args.site_chains}")
        site_chain_index, site_chain_valid_sites = load_site_chain_index(args.site_chains)
        valid_sites.update(site_chain_valid_sites)

    site_domain_map = json.load(open(args.site_domain, "r", encoding="utf-8"))
    site_graph_data = json.load(open(args.site_graph, "r", encoding="utf-8"))
    ne_to_site = {
        ne_id: str(ne_info.get("site_id", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("site_id", "")).strip()
    }
    alarm_source_domain_map = {
        ne_id: str(ne_info.get("domain", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("domain", "")).strip()
    }
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    return {
        "ne_graph_data": ne_graph_data,
        "topo_downstream_map": topo_downstream_map,
        "valid_sites": valid_sites,
        "site_chain_index": site_chain_index,
        "site_domain_map": site_domain_map,
        "site_graph_data": site_graph_data,
        "ne_to_site": ne_to_site,
        "alarm_source_domain_map": alarm_source_domain_map,
        "site_to_ne_ids": site_to_ne_ids,
        "ne_link_info_cache": {},
    }


def validate_args(parser, args):
    start_ts = parse_datetime_text(args.start_time, "start_time").timestamp() if args.start_time else None
    end_ts = parse_datetime_text(args.end_time, "end_time").timestamp() if args.end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        parser.error("start_time 不能晚于 end_time")
    if args.site_chains and not os.path.exists(args.site_chains):
        parser.error(f"site_chains 文件不存在: {args.site_chains}")
    return start_ts, end_ts


def load_alarm_data(args, parser, static_context, start_ts, end_ts):
    sorted_input = args.sorted_alarms_input.strip()
    if not sorted_input and is_sorted_alarm_cache_file(args.alarms):
        sorted_input = args.alarms
    if sorted_input:
        if not os.path.exists(sorted_input):
            parser.error(f"排序告警缓存不存在: {sorted_input}")
        processed_count, valid_alarms, normal_count, clear_count, metadata = load_sorted_alarm_cache_with_stats(sorted_input)
        warn_sorted_alarm_cache_option_mismatch(metadata, args)
        return processed_count, valid_alarms, normal_count, clear_count

    processed_count, valid_alarms, normal_count, clear_count = load_valid_alarms(
        args.alarms,
        CRITICAL_ALARMS,
        static_context["valid_sites"],
        static_context["ne_to_site"],
        start_ts=start_ts,
        end_ts=end_ts,
        clear_delay_sec=args.clear_delay_sec,
    )
    print("⏳ 正在按时间排序有效告警...")
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)
    if args.sorted_alarms_output:
        cached_normal_count, cached_clear_count = count_alarm_event_types(valid_alarms)
        metadata = {
            "source_alarms": os.path.abspath(args.alarms),
            "topo": os.path.abspath(args.topo) if args.topo else "",
            "ne_graph": os.path.abspath(args.ne_graph),
            "start_time": args.start_time or "",
            "end_time": args.end_time or "",
            "clear_delay_sec": float(args.clear_delay_sec),
            "processed_count": processed_count,
            "normal_alarm_count": normal_count,
            "clear_alarm_count": clear_count,
            "cached_normal_alarm_count": cached_normal_count,
            "cached_clear_alarm_count": cached_clear_count,
        }
        write_sorted_alarm_cache(args.sorted_alarms_output, valid_alarms, metadata)
        print(f"💾 排序告警缓存已写出: {args.sorted_alarms_output}")
    return processed_count, valid_alarms, normal_count, clear_count


def write_matches(fw, matches, static_context, alarm_metadata_index, compact_output):
    count = 0
    for match in matches:
        record = build_jsonl_match_output(
            match,
            static_context["ne_graph_data"],
            static_context["site_graph_data"],
            alarm_metadata_index,
            site_to_ne_ids=static_context["site_to_ne_ids"],
            ne_link_info_cache=static_context["ne_link_info_cache"],
            compact_output=compact_output,
            include_eid_list=False,
        )
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    return count


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    start_time = time.time()
    start_ts, end_ts = validate_args(parser, args)
    static_context = load_static_context(args)
    rules_config = build_rules_config(args)
    processed_count, valid_alarms, normal_count, clear_count = load_alarm_data(args, parser, static_context, start_ts, end_ts)
    print(f"有效告警数: {len(valid_alarms)}，正常告警数: {normal_count}，清除告警数: {clear_count}")
    print(f"CSM算法: {args.algorithm}")

    engine = FaultCSMEngine(
        static_context["topo_downstream_map"],
        rules_config,
        static_context["site_domain_map"],
        alarm_source_domain_map=static_context["alarm_source_domain_map"],
        site_chain_index=static_context["site_chain_index"],
        algorithm=args.algorithm,
        alarm_active_sec=args.alarm_active_sec,
    )
    alarm_metadata_index = build_alarm_metadata_index(valid_alarms)
    match_count = 0
    progress = ProgressBar(len(valid_alarms), "CSM处理有效告警")
    with open(args.output, "w", encoding="utf-8") as fw:
        for item in valid_alarms:
            matches = engine.process_event(item)
            if matches:
                match_count += write_matches(fw, matches, static_context, alarm_metadata_index, args.compact_output)
            progress.update()
        final_matches = engine.flush()
        if final_matches:
            match_count += write_matches(fw, final_matches, static_context, alarm_metadata_index, args.compact_output)
    progress.close()

    elapsed = time.time() - start_time
    print(
        f"🏁 CSM故障聚类完成。共处理 {processed_count} 条告警，过滤后 {len(valid_alarms)} 条，"
        f"生成 {match_count} 个故障组，耗时 {elapsed:.4f} 秒。"
    )


if __name__ == "__main__":
    main()
