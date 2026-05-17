"""
fault_csm_claude 增量故障匹配管道入口。

静态资源加载、规则配置、告警加载、输出会话全部复用 fault_grouping 组件；
唯一差异是以 IncrementalFaultEngine 替代 TemporalGraphEngine，
并采用"逐条立即匹配"流程（无聚合等待，无 live/offline 模式之分）。
"""

import time
from argparse import ArgumentParser

from topology_resources import (
    NE_GRAPH_JSON,
    SITE_DEVICE_COUNTS_JSON,
    SITE_GRAPH_JSON,
    SITE_GRAPH_BY_NE_JSON,
    resource_display,
)
from fault_grouping.matching.runtime import (
    build_batch_site_merge_helper,
    build_rules_config,
    default_valid_alarm_titles,
    load_alarm_data,
    load_static_context,
    print_alarm_load_summary,
    print_run_configuration,
    validate_main_args,
)
from fault_grouping.matching.group_output_session import MatchOutputSession
from fault_grouping.matching.group_output_builder import build_alarm_metadata_index
from alarm_tools.progress_utils import ProgressBar
from fault_grouping.alarm_events.stream import process_alarm, refresh_process_progress

from fault_csm_claude.engine import IncrementalFaultEngine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = ArgumentParser(
        description=(
            "fault_csm_claude: TurboFlux-inspired incremental fault grouping. "
            "每条告警到达后立即执行增量匹配，告警失效后执行失效匹配，"
            "无聚合等待窗口。"
        )
    )
    parser.add_argument("alarms", type=str, help="告警流文件")
    parser.add_argument("output", type=str, help="输出 JSONL 文件")
    parser.add_argument(
        "--topo", type=str, default=SITE_GRAPH_BY_NE_JSON,
        help=f"站点拓扑文件，默认: {resource_display('site_graph_by_ne.json')}"
    )
    parser.add_argument(
        "--site-chains", type=str, default="",
        help=f"可选预计算站点链路文件，推荐: {resource_display('site_chains.json')}"
    )
    parser.add_argument(
        "--site-domain", type=str, default=SITE_DEVICE_COUNTS_JSON,
        help=f"站点画像文件，默认: {resource_display('site_device_counts.json')}"
    )
    parser.add_argument(
        "--site-graph", type=str, default=SITE_GRAPH_JSON,
        help=f"site_graph.json，默认: {resource_display('site_graph.json')}"
    )
    parser.add_argument(
        "--ne-graph", type=str, default=NE_GRAPH_JSON,
        help=f"ne_graph.json，默认: {resource_display('ne_graph.json')}"
    )
    parser.add_argument(
        "--rule", action="append", default=[],
        help="仅启用指定规则，可重复传入"
    )
    parser.add_argument(
        "--clear-delay-sec", type=float, default=420.0,
        help="清除告警最小延迟时间（秒），默认 420"
    )
    parser.add_argument(
        "--batch-merge-site-hops", type=int, default=0,
        help="批内候选组额外按站点邻接合并的 hop 数；0 表示关闭"
    )
    parser.add_argument(
        "--batch-merge-density-knn", type=int, default=0,
        help="批内站点局部密度自适应合并近邻数；0 表示关闭"
    )
    parser.add_argument(
        "--batch-merge-density-scale", type=float, default=1.0,
        help="局部密度半径放大倍数，默认 1.0"
    )
    parser.add_argument(
        "--batch-merge-density-min-meters", type=float, default=0.0,
        help="局部密度自适应半径下限（米），0 表示不设下限"
    )
    parser.add_argument(
        "--batch-merge-density-max-meters", type=float, default=0.0,
        help="局部密度自适应半径上限（米），0 表示不设上限"
    )
    parser.add_argument(
        "--start_time", type=str,
        help="仅处理 >= 该时间的告警，格式: 2025-01-01 00:00:00"
    )
    parser.add_argument(
        "--end_time", type=str,
        help="仅处理 <= 该时间的告警，格式: 2025-01-31 23:59:59"
    )
    parser.add_argument(
        "--sorted-alarms-input", type=str, default="",
        help="直接加载排序告警缓存（JSONL/ZIP）"
    )
    parser.add_argument(
        "--sorted-alarms-output", type=str, default="",
        help="排序告警缓存写出路径"
    )
    parser.add_argument(
        "--compact-output", action="store_true",
        help="输出轻量化 JSONL"
    )
    parser.add_argument(
        "--verbose-groups", action="store_true",
        help="打印每个故障组的详细报告"
    )
    return parser


# ---------------------------------------------------------------------------
# Engine 初始化
# ---------------------------------------------------------------------------

def initialize_engine(args, static_context, rules_config, batch_site_merge_helper):
    """创建 IncrementalFaultEngine 实例。"""
    print("⏳ 正在初始化增量匹配引擎与拓扑索引（含 RoleSiteIndex 预构建）...")
    engine = IncrementalFaultEngine(
        topo_downstream_map=static_context.topo_downstream_map,
        rules_config=rules_config,
        site_domain_map=static_context.site_domain_map,
        alarm_source_domain_map=static_context.alarm_source_domain_map,
        site_merge_helper=batch_site_merge_helper,
        site_chain_index=static_context.site_chain_index,
    )
    print("✅ 增量匹配引擎就绪，开始监听告警流...\n")
    return engine


# ---------------------------------------------------------------------------
# 增量匹配管道（无聚合等待，逐条立即匹配）
# ---------------------------------------------------------------------------

def run_incremental_pipeline(engine, valid_alarms, output_session):
    """
    逐条处理告警，每条告警到达后立即执行增量匹配。

    与 fault_grouping 的 run_offline_mode 不同之处：
    - 无 aggregation_wait_sec 延迟
    - process_event 返回结果即为当前告警触发的故障组
    - 告警清除同样立即执行失效匹配
    """
    print("⏱️  运行模式: incremental（逐条立即匹配，无聚合等待）")
    process_progress = ProgressBar(len(valid_alarms), "处理有效告警")
    process_progress._refresh_extra_text = output_session.refresh_progress_extra_text
    output_session.process_progress = process_progress
    output_session.refresh_progress_extra_text(force=True)

    try:
        for item in valid_alarms:
            matches = process_alarm(engine, item, collect_matches=True)
            if matches:
                output_session.write_matches(matches)
            refresh_process_progress(process_progress)
    finally:
        process_progress.close()

    output_session.process_progress = None

    # 增量模式无 pending_triggers，flush_pending 返回空列表
    # 仍调用一次以保持接口一致性
    final_matches = engine.flush_pending()
    if final_matches:
        output_session.write_matches(final_matches)


# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------

def print_final_summary(engine, processed_count, filtered_count, match_count, elapsed):
    final_merge_stats = engine.get_batch_merge_stats_snapshot().get("total", {})
    print(
        f"🏁 告警流处理完毕。共处理 {processed_count} 条告警，过滤后 {filtered_count} 条，"
        f"生成 {match_count} 个故障组，"
        f"eid合并组数 {final_merge_stats.get('eid_merge_group_count', 0)}，"
        f"hop合并组数 {final_merge_stats.get('hop_merge_group_count', 0)}，"
        f"耗时 {elapsed:.4f} 秒。"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    start_ts, end_ts = validate_main_args(parser, args)
    static_context = load_static_context(args)
    valid_alarm_titles = default_valid_alarm_titles()
    print_run_configuration(args, static_context, valid_alarm_titles)

    rules_config = build_rules_config(args, parser)
    batch_site_merge_helper = build_batch_site_merge_helper(
        args, static_context.topo_downstream_map
    )
    engine = initialize_engine(args, static_context, rules_config, batch_site_merge_helper)

    start_time = time.time()
    alarm_load_result = load_alarm_data(
        args, parser, static_context, valid_alarm_titles, start_ts, end_ts
    )
    alarm_metadata_index = build_alarm_metadata_index(alarm_load_result.valid_alarms)
    print_alarm_load_summary(alarm_load_result)

    output_session = MatchOutputSession(
        args=args,
        engine=engine,
        output_path=args.output,
        ne_graph_data=static_context.ne_graph_data,
        site_graph_data=static_context.site_graph_data,
        alarm_metadata_index=alarm_metadata_index,
        site_to_ne_ids=static_context.site_to_ne_ids,
        ne_link_info_cache=static_context.ne_link_info_cache,
    )
    output_session.reset_output_file()

    run_incremental_pipeline(engine, alarm_load_result.valid_alarms, output_session)

    elapsed = time.time() - start_time
    print_final_summary(
        engine,
        alarm_load_result.processed_count,
        alarm_load_result.filtered_count,
        output_session.match_count,
        elapsed,
    )


if __name__ == "__main__":
    main()
