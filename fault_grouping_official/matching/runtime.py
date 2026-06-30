import os
import time
import zipfile

from dataclasses import dataclass

from fault_grouping_official.alarm_types import CRITICAL_ALARMS
from fault_grouping_official.tools.progress_utils import ProgressBar
from fault_grouping_official.alarm_events.io import (
    load_sorted_alarm_cache_with_stats,
    load_valid_alarms,
    trim_trailing_clear_alarms,
    warn_sorted_alarm_cache_option_mismatch,
)
from fault_grouping_official.alarm_events.stream import (
    build_simulated_now_ts_getter,
    process_alarm,
    refresh_process_progress,
    stream_alarms_by_ts,
)
from fault_grouping_official.rule_config import (
    OUTPUT_ELIGIBLE_RULE_FIELD,
    data_link_adjacent_no_offline_rule,
    data_link_adjacent_offline_rule,
    data_no_offline_adjacent_optional_offline_rule,
    data_offline_adjacent_offline_rule,
)
from fault_grouping_official.alarm_events.sorted_cache import (
    SortedAlarmCacheStream,
    read_sorted_alarm_cache_header,
    try_read_sorted_alarm_cache_header,
)
from fault_grouping_official.temporal_engine.engine import TemporalGraphEngine
from fault_grouping_official.site_topology import (
    build_site_domain_map,
    build_site_to_ne_ids,
    build_site_chain_index,
    build_site_topology_from_ne_graph,
)
from fault_grouping_official.link_peer_index import build_peer_index
from fault_grouping_official.resource_buffer import load_resource_buffer


@dataclass
class LoadedStaticContext:
    ne_graph_data: dict
    site_graph_data: dict
    valid_sites: set
    topo_downstream_map: dict
    site_chain_index: object
    site_domain_map: dict
    ne_to_site: dict
    alarm_source_domain_map: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
    link_peer_index: dict


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


def validate_main_args(parser, args):
    if args.stream_sorted_alarms:
        try:
            sorted_alarm_cache_metadata = read_sorted_alarm_cache_header(args.alarms)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            parser.error(
                "--stream-sorted-alarms 只能用于排序告警缓存；"
                f"缓存识别失败: {exc}"
            )
    else:
        sorted_alarm_cache_metadata = try_read_sorted_alarm_cache_header(args.alarms)
    if not os.path.exists(args.resource_buffer):
        parser.error(
            f"资源缓冲文件不存在: {args.resource_buffer}；"
            "请先运行 build_resource_buffer.py 生成 resources/resource_buffer.jsonl，"
            "或通过 --resource-buffer 指定已有缓冲文件"
        )
    return sorted_alarm_cache_metadata


def load_static_context(args):
    print(f"加载资源缓冲文件: {args.resource_buffer}")
    resources = load_resource_buffer(
        args.resource_buffer,
        wanted_types=("ne_graph", "site_graph", "site_chains", "link_peer_index"),
    )
    ne_graph_data = resources["ne_graph"]
    site_graph_data = resources["site_graph"]
    site_chain_index, valid_sites = build_site_chain_index(resources["site_chains"])
    topo_downstream_map, topology_sites = build_site_topology_from_ne_graph(ne_graph_data)
    valid_sites.update(topology_sites)
    site_domain_map = build_site_domain_map(ne_graph_data)
    print(f"预计算站点链路站点数: {len(site_chain_index)}")
    print(f"ne_graph 站点拓扑起点数: {len(topo_downstream_map)}")

    print("加载有效站点集合...")
    print(f"有效站点数: {len(valid_sites)}")

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
    link_peer_index = build_peer_index(resources["link_peer_index"])
    print(f"link peer_index 记录数: {len(link_peer_index)}")

    return LoadedStaticContext(
        ne_graph_data=ne_graph_data,
        site_graph_data=site_graph_data,
        valid_sites=valid_sites,
        topo_downstream_map=topo_downstream_map,
        site_chain_index=site_chain_index,
        site_domain_map=site_domain_map,
        ne_to_site=ne_to_site,
        alarm_source_domain_map=alarm_source_domain_map,
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
        link_peer_index=link_peer_index,
    )


def print_run_configuration(args, valid_alarm_titles):
    if args.clear_delay_sec > 0:
        print(f"清除告警最小延迟: {args.clear_delay_sec:g} 秒")
    print(f"有效告警类型数: {len(valid_alarm_titles)}")
    print(f"落盘规则检查: {'启用' if args.rule_check else '关闭'}")
    print(f"故障模式检查与增强: {'启用' if args.pattern_check else '关闭'}")


def build_rules_config():
    rules_config = {
        "data_link_adjacent_no_offline_rule": data_link_adjacent_no_offline_rule,
        "data_link_adjacent_offline_rule": data_link_adjacent_offline_rule,
        "data_no_offline_adjacent_optional_offline_rule": data_no_offline_adjacent_optional_offline_rule,
        "data_offline_adjacent_offline_rule": data_offline_adjacent_offline_rule,
    }
    print("启用规则: " + ", ".join(rules_config.keys()))
    return rules_config


def collect_output_eligible_rules(rules_config):
    """返回“可落盘规则”名集合：rule_config 中标记 output_eligible=True 的规则。

    match_rules.py 输出故障组前据此过滤——只有 merged_rules 命中其中任意一个
    规则的故障组才会写入输出文件。若没有任何规则被标记，返回 None 表示不过滤
    （全部落盘）。
    """
    eligible = frozenset(
        rule_name
        for rule_name, rule in rules_config.items()
        if rule.get(OUTPUT_ELIGIBLE_RULE_FIELD)
    )
    if not eligible:
        print("未标记可落盘规则(output_eligible)，输出不做规则过滤")
        return None
    print("仅落盘包含以下规则的故障组: " + ", ".join(sorted(eligible)))
    return eligible


def build_fault_pattern_filter(static_context):
    """构建落盘前故障模式过滤器（filter-others + one-component-only，默认启用）。

    过滤器仅在启用时使用，因此在函数内延迟导入。
    """
    from fault_grouping_official.matching.fault_pattern_filter import FaultPatternFilter

    fault_pattern_filter = FaultPatternFilter.from_static_context(
        static_context.ne_graph_data,
        static_context.site_chain_index,
        static_context.ne_to_site,
        static_context.site_to_ne_ids,
        site_graph_data=static_context.site_graph_data,
    )
    print("已启用落盘前故障模式过滤+增强: filter-others + one-component-only")
    return fault_pattern_filter


def initialize_engine(args, static_context, rules_config):
    print("⏳ 正在初始化时序图引擎与拓扑映射...")
    print(f"聚合等待时间: {args.aggregation_wait_sec:g} 秒")
    engine = TemporalGraphEngine(
        rules_config,
        static_context.site_domain_map,
        alarm_source_domain_map=static_context.alarm_source_domain_map,
        aggregation_wait_sec=args.aggregation_wait_sec,
        topo_downstream_map=static_context.topo_downstream_map,
        site_chain_index=static_context.site_chain_index,
        ne_graph_data=static_context.ne_graph_data,
        site_to_ne_ids=static_context.site_to_ne_ids,
        link_peer_index=static_context.link_peer_index,
    )
    print("✅ 引擎启动就绪，开始监听告警流...\n")
    return engine


def load_alarm_data(args, static_context, valid_alarm_titles, sorted_alarm_cache_metadata):
    if sorted_alarm_cache_metadata is not None:
        sorted_alarm_cache_input = args.alarms
        if args.stream_sorted_alarms:
            print(f"⚡ 流式读取排序告警缓存: {sorted_alarm_cache_input}")
            load_start_time = time.time()
            warn_sorted_alarm_cache_option_mismatch(sorted_alarm_cache_metadata, args)
            valid_alarms = SortedAlarmCacheStream(
                sorted_alarm_cache_input,
                metadata=sorted_alarm_cache_metadata,
            )
            sort_elapsed = time.time() - load_start_time
            normal_alarm_count = int(sorted_alarm_cache_metadata["cached_normal_alarm_count"])
            clear_alarm_count = int(sorted_alarm_cache_metadata["cached_clear_alarm_count"])
            processed_count = int(sorted_alarm_cache_metadata["processed_count"])
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
        ) = load_sorted_alarm_cache_with_stats(
            sorted_alarm_cache_input,
            sorted_alarm_cache_metadata,
        )
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
        clear_delay_sec=args.clear_delay_sec,
    )

    print("⏳ 正在按时间排序有效告警...")
    sort_start_time = time.time()
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)
    sort_elapsed = time.time() - sort_start_time

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


def run_live_mode(
    engine,
    valid_alarms,
    speedup,
    real_harvest_interval_sec,
    on_matches,
    process_progress,
    refresh_extra_text,
):
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
            refresh_process_progress(process_progress, refresh_extra_text)
    finally:
        process_progress.close()
        engine.stop_periodic_harvest()


def run_offline_mode(engine, valid_alarms, on_matches, process_progress, refresh_extra_text):
    """按时间排序顺序处理告警，并在每条告警后立即同步收割一次成熟故障组。"""
    print("⏱️ 运行模式: offline, 每条告警到来时直接触发检查")
    try:
        for item in valid_alarms:
            matches = process_alarm(engine, item, collect_matches=True)
            if matches:
                on_matches(matches)
            refresh_process_progress(process_progress, refresh_extra_text)
    finally:
        process_progress.close()


def run_matching_pipeline(
    args,
    engine,
    valid_alarms,
    output_session,
):
    speedup = max(float(args.speedup), 1e-9)
    real_harvest_interval_sec = max(args.harvest_interval_sec / speedup, 0.001)
    process_progress = ProgressBar(len(valid_alarms), "处理有效告警")
    output_session.process_progress = process_progress
    output_session.refresh_progress_extra_text(force=True)

    if args.mode == 'live':
        run_live_mode(
            engine,
            valid_alarms,
            speedup,
            real_harvest_interval_sec,
            output_session.write_matches,
            process_progress,
            output_session.refresh_progress_extra_text,
        )
    else:
        run_offline_mode(
            engine,
            valid_alarms,
            output_session.write_matches,
            process_progress,
            output_session.refresh_progress_extra_text,
        )

    output_session.process_progress = None

    print("⏳ 数据流读取完毕，正在清空并计算延迟聚合队列...")
    final_matches = engine.flush_pending()
    if final_matches:
        output_session.write_matches(final_matches)


def print_final_summary(engine, processed_count, filtered_count, match_count, elapsed):
    final_merge_stats = engine.get_batch_merge_stats_snapshot().get("total", {})
    primary_merge_count = final_merge_stats.get('eid_merge_group_count', 0)
    primary_merge_label = "eid合并组数"
    print(
        f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，"
        f"生成 {match_count} 个故障组，"
        f"{primary_merge_label} {primary_merge_count}，"
        f"耗时 {elapsed:.4f} 秒。"
    )
