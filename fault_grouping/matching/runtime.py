import json
import os
import time

from dataclasses import dataclass

from alarm_tools.alarm_types import CRITICAL_ALARMS
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.alarm_events.io import (
    count_alarm_event_types,
    load_sorted_alarm_cache_with_stats,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
    warn_sorted_alarm_cache_option_mismatch,
)
from fault_grouping.alarm_events.stream import (
    build_simulated_now_ts_getter,
    process_alarm,
    refresh_process_progress,
    stream_alarms_by_ts,
)
from fault_grouping.matching.debug import (
    match_debug_trigger,
    print_debug_match_details,
    run_debug_mode,
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
from fault_grouping.site_merge_helper import (
    AdaptiveDensitySiteMergeHelper,
    BatchSiteMergeHelper,
)
from fault_grouping.alarm_events.sorted_cache import (
    SortedAlarmCacheStream,
    is_sorted_alarm_cache_file,
    read_sorted_alarm_cache_header,
    write_sorted_alarm_cache,
)
from fault_grouping.temporal_engine.engine import TemporalGraphEngine
from fault_grouping.site_topology import (
    apply_missing_topology_predictions,
    build_site_to_ne_ids,
    build_site_topology_from_ne_graph,
    load_missing_topology_predictions,
    load_site_chain_index,
)


@dataclass
class LoadedStaticContext:
    ne_graph_data: dict
    topo_downstream_map: dict
    valid_sites: set
    site_chain_index: object
    site_domain_map: dict
    site_graph_data: dict
    ne_to_site: dict
    alarm_source_domain_map: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
    missing_topology_edges: dict


@dataclass
class AlarmLoadResult:
    processed_count: int
    valid_alarms: list
    normal_alarm_count: int
    clear_alarm_count: int
    sort_elapsed: float

    @property
    def filtered_count(self):
        return len(self.valid_alarms)


def parse_selected_rule_names(rule_args, valid_rule_names):
    if not rule_args:
        return []

    selected_rule_names = []
    seen_rule_names = set()
    available_rule_name_set = set(valid_rule_names)

    for raw_value in rule_args:
        for part in str(raw_value).replace("，", ",").split(","):
            rule_name = part.strip()
            if not rule_name or rule_name in seen_rule_names:
                continue
            if rule_name not in available_rule_name_set:
                raise ValueError(
                    f"未知规则名: {rule_name}；可选值: {', '.join(sorted(available_rule_name_set))}"
                )
            seen_rule_names.add(rule_name)
            selected_rule_names.append(rule_name)

    return selected_rule_names


def validate_main_args(parser, args):
    start_ts = None
    end_ts = None
    if args.start_time:
        start_ts = parse_datetime_text(args.start_time, "start_time").timestamp()
    if args.end_time:
        end_ts = parse_datetime_text(args.end_time, "end_time").timestamp()
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        parser.error("start_time 不能晚于 end_time")
    if args.batch_merge_density_knn < 0:
        parser.error("batch-merge-density-knn 不能小于 0")
    if args.batch_merge_density_scale < 0:
        parser.error("batch-merge-density-scale 不能小于 0")
    if args.batch_merge_density_min_meters < 0 or args.batch_merge_density_max_meters < 0:
        parser.error("batch-merge-density-min-meters / max-meters 不能小于 0")
    if (
        args.batch_merge_density_max_meters > 0
        and args.batch_merge_density_min_meters > 0
        and args.batch_merge_density_max_meters < args.batch_merge_density_min_meters
    ):
        parser.error("batch-merge-density-max-meters 不能小于 batch-merge-density-min-meters")
    if not args.no_output and not args.output:
        parser.error("未指定 output；正常输出模式必须提供 output，或使用 --no-output 跳过故障组输出")
    if args.no_output and args.compute_ticket_recall:
        parser.error("--no-output 不能与 --compute-ticket-recall 同时使用，因为工单召回计算需要读取故障组输出文件")
    if (
        args.stream_sorted_alarms
        and not args.sorted_alarms_input.strip()
        and not is_sorted_alarm_cache_file(args.alarms)
    ):
        parser.error(
            "--stream-sorted-alarms 只能用于排序告警缓存；"
            "请先运行 fault_grouping/tools/prepare_sorted_alarms.py 生成缓存，"
            "或通过 --sorted-alarms-input 指定缓存"
        )
    if args.batch_merge_density_knn > 0 and args.batch_merge_density_scale <= 0:
        parser.error("启用 batch-merge-density-knn 时，batch-merge-density-scale 必须大于 0")
    if args.site_chains and not os.path.exists(args.site_chains):
        parser.error(f"site_chains 文件不存在: {args.site_chains}")
    if args.missing_topology and not os.path.exists(args.missing_topology):
        parser.error(f"missing_topology 文件不存在: {args.missing_topology}")
    return start_ts, end_ts


def load_static_context(args):
    ne_graph_data = json.load(open(args.ne_graph, 'r', encoding='utf-8'))
    if args.topo:
        print(f"加载显式站点拓扑: {args.topo}")
        topo_downstream_map = json.load(open(args.topo, 'r', encoding='utf-8'))
        valid_sites = set(topo_downstream_map.keys())
        for _, connected_sites in topo_downstream_map.items():
            if isinstance(connected_sites, list):
                valid_sites.update(connected_sites)
            elif isinstance(connected_sites, dict):
                valid_sites.update(connected_sites.keys())
    else:
        print(f"基于 ne_graph 原始连边构建站点传播拓扑: {args.ne_graph}")
        topo_downstream_map, valid_sites = build_site_topology_from_ne_graph(ne_graph_data)

    site_chain_index = None
    if args.site_chains:
        print(f"加载预计算站点链路: {args.site_chains}")
        site_chain_index, site_chain_valid_sites = load_site_chain_index(args.site_chains)
        valid_sites.update(site_chain_valid_sites)
        print(f"预计算站点链路站点数: {len(site_chain_index)}")

    missing_topology_edges = {}
    if args.missing_topology:
        print(f"加载弱拓扑缺边预测: {args.missing_topology}")
        missing_topology_predictions = load_missing_topology_predictions(
            args.missing_topology,
            min_score=args.missing_topology_min_score,
        )
        topo_downstream_map, site_chain_index, missing_topology_edges = apply_missing_topology_predictions(
            topo_downstream_map,
            site_chain_index,
            missing_topology_predictions,
        )
        for source_site, target_site in missing_topology_edges:
            valid_sites.add(source_site)
            valid_sites.add(target_site)
        print(f"弱拓扑补偿边数: {len(missing_topology_edges)}")

    site_domain_map = json.load(open(args.site_domain, 'r', encoding='utf-8'))
    site_graph_data = json.load(open(args.site_graph, 'r', encoding='utf-8'))

    print("加载有效站点集合...")
    print(f"有效站点数: {len(valid_sites)}")
    print(f"站点拓扑起点数: {len(topo_downstream_map)}")

    print("构建 ne -> site 映射...")
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
    print(f"NE 数量: {len(ne_to_site)}")
    print("构建 site -> NE 输出索引...")
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    ne_link_info_cache = {}
    print(f"site -> NE 索引站点数: {len(site_to_ne_ids)}")

    return LoadedStaticContext(
        ne_graph_data=ne_graph_data,
        topo_downstream_map=topo_downstream_map,
        valid_sites=valid_sites,
        site_chain_index=site_chain_index,
        site_domain_map=site_domain_map,
        site_graph_data=site_graph_data,
        ne_to_site=ne_to_site,
        alarm_source_domain_map=alarm_source_domain_map,
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
        missing_topology_edges=missing_topology_edges,
    )


def print_run_configuration(args, static_context, valid_alarm_titles):
    if args.start_time or args.end_time:
        print(
            "告警首次发生时间过滤: "
            f"start_time={args.start_time or '-'}, "
            f"end_time={args.end_time or '-'}"
        )
    if args.clear_delay_sec > 0:
        print(f"清除告警最小延迟: {args.clear_delay_sec:g} 秒")
    if args.no_output:
        print("故障组输出: 关闭（--no-output，仅统计不写 JSONL）")
    if args.batch_merge_site_hops > 0:
        print(f"批内站点邻接合并: 开启，hop={args.batch_merge_site_hops}")
    if args.batch_merge_density_knn > 0:
        print(
            "批内站点密度合并: 开启，"
            f"k={args.batch_merge_density_knn}, "
            f"scale={args.batch_merge_density_scale:g}, "
            f"min_radius={args.batch_merge_density_min_meters:g}m, "
            f"max_radius={args.batch_merge_density_max_meters:g}m"
        )
    if args.enable_support_pruning:
        print("候选 support 剪枝: 开启")
    if args.enable_support_count_sort:
        print("候选 support count 排序: 开启")
    if args.missing_topology:
        print(
            "弱拓扑补偿: 开启，"
            f"方向化边数={len(static_context.missing_topology_edges)}, "
            f"min_score={args.missing_topology_min_score:g}"
        )
    print(f"有效告警类型数: {len(valid_alarm_titles)}")


def build_rules_config(args, parser):
    all_rules_config = {
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
    try:
        selected_rule_names = parse_selected_rule_names(args.rule, all_rules_config.keys())
    except ValueError as exc:
        parser.error(str(exc))
    rules_config = (
        {
            rule_name: all_rules_config[rule_name]
            for rule_name in selected_rule_names
        }
        if selected_rule_names else all_rules_config
    )
    print("启用规则: " + ", ".join(rules_config.keys()))
    return rules_config


def build_batch_site_merge_helper(args, topo_downstream_map):
    density_site_merge_helper = None
    if args.batch_merge_density_knn > 0:
        density_site_merge_helper = AdaptiveDensitySiteMergeHelper(
            args.site_graph,
            density_knn=args.batch_merge_density_knn,
            density_scale=args.batch_merge_density_scale,
            min_radius_meters=args.batch_merge_density_min_meters,
            max_radius_meters=args.batch_merge_density_max_meters,
        )

    batch_site_merge_helper = None
    if args.batch_merge_site_hops > 0 or density_site_merge_helper is not None:
        batch_site_merge_helper = BatchSiteMergeHelper(
            topo_downstream_map,
            site_neighbor_hops=args.batch_merge_site_hops,
            density_helper=density_site_merge_helper,
        )
        if density_site_merge_helper is not None:
            print("⏳ 正在准备站点批内合并辅助器...")
            batch_site_merge_helper.warmup()
            print("✅ 站点批内合并辅助器就绪")
    return batch_site_merge_helper


def initialize_engine(args, static_context, rules_config, batch_site_merge_helper):
    print("⏳ 正在初始化时序图引擎与拓扑映射...")
    print(f"聚合等待时间: {args.aggregation_wait_sec:g} 秒")
    print(
        "event_cache 模式: "
        + ("设备告警时段(period)" if args.use_alarm_period_cache else "逐条活跃告警(raw, 默认)")
    )
    engine = TemporalGraphEngine(
        static_context.topo_downstream_map,
        rules_config,
        static_context.site_domain_map,
        alarm_source_domain_map=static_context.alarm_source_domain_map,
        aggregation_wait_sec=args.aggregation_wait_sec,
        site_merge_helper=batch_site_merge_helper,
        site_chain_index=static_context.site_chain_index,
        use_alarm_period_cache=args.use_alarm_period_cache,
        enable_support_pruning=args.enable_support_pruning,
        enable_support_count_sort=args.enable_support_count_sort,
        missing_topology_edges=static_context.missing_topology_edges,
        ne_graph_data=static_context.ne_graph_data,
        site_to_ne_ids=static_context.site_to_ne_ids,
    )
    print("✅ 引擎启动就绪，开始监听告警流...\n")
    return engine


def resolve_sorted_alarm_cache_input(parser, args):
    alarm_file_path = args.alarms
    sorted_alarm_cache_input = args.sorted_alarms_input.strip()
    if sorted_alarm_cache_input:
        if not os.path.exists(sorted_alarm_cache_input):
            parser.error(f"排序告警缓存不存在: {sorted_alarm_cache_input}")
        if not is_sorted_alarm_cache_file(sorted_alarm_cache_input):
            parser.error(f"不是有效的排序告警缓存: {sorted_alarm_cache_input}")
    elif is_sorted_alarm_cache_file(alarm_file_path):
        sorted_alarm_cache_input = alarm_file_path
    return sorted_alarm_cache_input


def load_alarm_data(args, parser, static_context, valid_alarm_titles, start_ts, end_ts):
    sorted_alarm_cache_input = resolve_sorted_alarm_cache_input(parser, args)
    if args.stream_sorted_alarms and not sorted_alarm_cache_input:
        parser.error(
            "--stream-sorted-alarms 只能用于排序告警缓存；"
            "请先运行 fault_grouping/tools/prepare_sorted_alarms.py 生成缓存，"
            "或通过 --sorted-alarms-input 指定缓存"
        )
    if sorted_alarm_cache_input:
        if args.stream_sorted_alarms:
            print(f"⚡ 流式读取排序告警缓存: {sorted_alarm_cache_input}")
            load_start_time = time.time()
            sorted_alarm_cache_metadata = read_sorted_alarm_cache_header(sorted_alarm_cache_input)
            warn_sorted_alarm_cache_option_mismatch(sorted_alarm_cache_metadata, args)
            valid_alarms = SortedAlarmCacheStream(
                sorted_alarm_cache_input,
                metadata=sorted_alarm_cache_metadata,
            )
            sort_elapsed = time.time() - load_start_time
            normal_alarm_count = int(
                sorted_alarm_cache_metadata.get(
                    "cached_normal_alarm_count",
                    sorted_alarm_cache_metadata.get("normal_alarm_count", 0),
                )
            )
            clear_alarm_count = int(
                sorted_alarm_cache_metadata.get(
                    "cached_clear_alarm_count",
                    sorted_alarm_cache_metadata.get("clear_alarm_count", 0),
                )
            )
            processed_count = int(
                sorted_alarm_cache_metadata.get(
                    "processed_count",
                    len(valid_alarms),
                )
            )
            return AlarmLoadResult(
                processed_count=processed_count,
                valid_alarms=valid_alarms,
                normal_alarm_count=normal_alarm_count,
                clear_alarm_count=clear_alarm_count,
                sort_elapsed=sort_elapsed,
            )

        print(f"⚡ 直接加载排序告警缓存: {sorted_alarm_cache_input}")
        load_start_time = time.time()
        (
            processed_count,
            valid_alarms,
            normal_alarm_count,
            clear_alarm_count,
            sorted_alarm_cache_metadata,
        ) = load_sorted_alarm_cache_with_stats(sorted_alarm_cache_input)
        warn_sorted_alarm_cache_option_mismatch(sorted_alarm_cache_metadata, args)
        sort_elapsed = time.time() - load_start_time
        return AlarmLoadResult(
            processed_count=processed_count,
            valid_alarms=valid_alarms,
            normal_alarm_count=normal_alarm_count,
            clear_alarm_count=clear_alarm_count,
            sort_elapsed=sort_elapsed,
        )

    processed_count, valid_alarms, normal_alarm_count, clear_alarm_count = load_valid_alarms(
        args.alarms,
        valid_alarm_titles,
        static_context.valid_sites,
        static_context.ne_to_site,
        start_ts=start_ts,
        end_ts=end_ts,
        clear_delay_sec=args.clear_delay_sec,
    )

    print("⏳ 正在按时间排序有效告警...")
    sort_start_time = time.time()
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)
    sort_elapsed = time.time() - sort_start_time
    if args.sorted_alarms_output:
        cached_normal_alarm_count, cached_clear_alarm_count = count_alarm_event_types(valid_alarms)
        cache_metadata = {
            "source_alarms": os.path.abspath(args.alarms),
            "topo": os.path.abspath(args.topo) if args.topo else "",
            "ne_graph": os.path.abspath(args.ne_graph),
            "start_time": args.start_time or "",
            "end_time": args.end_time or "",
            "clear_delay_sec": float(args.clear_delay_sec),
            "processed_count": processed_count,
            "normal_alarm_count": normal_alarm_count,
            "clear_alarm_count": clear_alarm_count,
            "cached_normal_alarm_count": cached_normal_alarm_count,
            "cached_clear_alarm_count": cached_clear_alarm_count,
            "valid_site_count": len(static_context.valid_sites),
            "ne_to_site_count": len(static_context.ne_to_site),
            "valid_alarm_title_count": len(valid_alarm_titles),
        }
        write_sorted_alarm_cache(args.sorted_alarms_output, valid_alarms, cache_metadata)
        print(f"💾 排序告警缓存已写出: {args.sorted_alarms_output}")

    return AlarmLoadResult(
        processed_count=processed_count,
        valid_alarms=valid_alarms,
        normal_alarm_count=normal_alarm_count,
        clear_alarm_count=clear_alarm_count,
        sort_elapsed=sort_elapsed,
    )


def print_alarm_load_summary(alarm_load_result):
    print(
        f"有效告警数: {alarm_load_result.filtered_count}，排序/加载耗时: "
        f"{alarm_load_result.sort_elapsed:.4f} 秒"
    )
    print(
        f"正常告警数: {alarm_load_result.normal_alarm_count}，"
        f"清除告警数: {alarm_load_result.clear_alarm_count}"
    )


def default_valid_alarm_titles():
    return CRITICAL_ALARMS


def run_live_mode(engine, valid_alarms, speedup, real_harvest_interval_sec, on_matches, process_progress):
    """按 ts 差值模拟实时告警流，并由后台定时线程异步收割成熟故障组。"""
    print(
        f"⏱️ 运行模式: live, speedup={speedup:g}x, "
        f"模拟收割周期={real_harvest_interval_sec * speedup:g}s, "
        f"真实收割周期={real_harvest_interval_sec:.3f}s"
    )
    now_ts_getter = build_simulated_now_ts_getter(valid_alarms, speedup)
    engine.start_periodic_harvest(
        interval_sec=real_harvest_interval_sec,
        on_matches=on_matches,
        now_ts_getter=now_ts_getter
    )

    try:
        for item in stream_alarms_by_ts(valid_alarms, speedup=speedup):
            process_alarm(engine, item, collect_matches=False)
            refresh_process_progress(process_progress)
    finally:
        process_progress.close()
        engine.stop_periodic_harvest()


def run_offline_mode(engine, valid_alarms, on_matches, process_progress):
    """按时间排序顺序处理告警，并在每条告警后立即同步收割一次成熟故障组。"""
    print("⏱️ 运行模式: offline, 每条告警到来时直接触发检查")
    try:
        for item in valid_alarms:
            matches = process_alarm(engine, item, collect_matches=True)
            if matches:
                on_matches(matches)
            refresh_process_progress(process_progress)
    finally:
        process_progress.close()


def run_matching_pipeline(
    args,
    engine,
    valid_alarms,
    output_session,
    debug_targets,
    rules_config,
):
    debug_enabled = bool(debug_targets)
    if debug_enabled:
        print("🔎 Debug 模式已开启:")
        for site_id, alarm_name in sorted(debug_targets):
            print(f"   - {site_id} / {alarm_name}")

    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)
    process_progress = ProgressBar(len(valid_alarms), "处理有效告警")
    process_progress._refresh_extra_text = output_session.refresh_progress_extra_text
    output_session.process_progress = process_progress
    output_session.refresh_progress_extra_text(force=True)

    if debug_enabled:
        run_debug_mode(
            engine,
            valid_alarms,
            output_session.write_matches,
            process_progress,
            debug_targets,
            rules_config,
            args.mode,
            speedup,
            real_harvest_interval_sec,
        )
    elif args.mode == 'live':
        run_live_mode(
            engine,
            valid_alarms,
            speedup,
            real_harvest_interval_sec,
            output_session.write_matches,
            process_progress
        )
    else:
        run_offline_mode(engine, valid_alarms, output_session.write_matches, process_progress)

    output_session.process_progress = None

    print("⏳ 数据流读取完毕，正在清空并计算延迟聚合队列...")
    final_matches = engine.flush_pending()
    if final_matches:
        if debug_enabled:
            debug_final_matches = [
                match for match in final_matches
                if match_debug_trigger(match, debug_targets, rules_config)
            ]
            if debug_final_matches:
                print(f"🔎 Flush 阶段额外产出 {len(debug_final_matches)} 个故障组")
                for match in debug_final_matches:
                    print_debug_match_details(match)
        output_session.write_matches(final_matches)

    if debug_enabled:
        engine.debug_observer = None


def print_final_summary(args, engine, processed_count, filtered_count, match_count, elapsed):
    final_merge_stats = engine.get_batch_merge_stats_snapshot().get("total", {})
    primary_merge_count = (
        final_merge_stats.get('alarm_overlap_merge_group_count', 0)
        if args.use_alarm_period_cache
        else final_merge_stats.get('eid_merge_group_count', 0)
    )
    primary_merge_label = "告警时段合并组数" if args.use_alarm_period_cache else "eid合并组数"
    print(
        f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，"
        f"生成 {match_count} 个故障组，"
        f"{primary_merge_label} {primary_merge_count}，"
        f"hop合并组数 {final_merge_stats.get('hop_merge_group_count', 0)}，"
        f"距离合并组数 {final_merge_stats.get('distance_merge_group_count', 0)}，"
        f"耗时 {elapsed:.4f} 秒。"
    )
