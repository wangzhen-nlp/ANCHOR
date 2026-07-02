import time

from argparse import ArgumentParser
from dataclasses import dataclass

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.tools.topology_resources import (
    RESOURCE_BUFFER_JSONL,
    resource_display,
)
from anchor_grouping_online.matching.group_output_session import MatchOutputSession
from anchor_grouping_online.matching.runtime import (
    AlarmLoadResult,
    build_fault_pattern_filter,
    build_rules_config,
    collect_output_eligible_rules,
    default_valid_alarm_titles,
    initialize_engine,
    load_alarm_data,
    load_static_context,
    print_alarm_load_summary,
    print_final_summary,
    print_run_configuration,
    run_matching_pipeline,
    validate_main_args,
)
from anchor_grouping_online.temporal_engine.engine import TemporalGraphEngine


@dataclass
class RuntimeExecutionPlan:
    engine: TemporalGraphEngine
    alarm_load_result: AlarmLoadResult
    output_session: MatchOutputSession
    run_started_at: float


def _build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        'alarms',
        type=str,
        help='alarm stream',
    )
    parser.add_argument(
        'output',
        type=str,
        help='output jsonl file',
    )
    parser.add_argument(
        '--resource-buffer',
        type=str,
        default=RESOURCE_BUFFER_JSONL,
        help=(
            'build_resource_buffer.py 生成的资源缓冲文件（含 ne_graph / '
            'site_chains / link_peer_index），默认: '
            f'{resource_display("resource_buffer.jsonl")}'
        ),
    )
    parser.add_argument(
        '--stream-sorted-alarms',
        action='store_true',
        help=(
            '从排序告警缓存流式读取，不把全部 valid_alarms 加载到内存；'
            '仅当 alarms 本身为 prepare_sorted_alarms.py 生成的排序缓存'
            '(JSONL/ZIP)时生效'
        ),
    )
    return parser


def _build_output_session(
    args, engine, static_context, output_eligible_rules, fault_pattern_filter
):
    output_session = MatchOutputSession(
        engine=engine,
        output_path=args.output,
        ne_graph_data=static_context.ne_graph_data,
        site_graph_data=static_context.site_graph_data,
        site_to_ne_ids=static_context.site_to_ne_ids,
        ne_link_info_cache=static_context.ne_link_info_cache,
        output_eligible_rules=output_eligible_rules,
        fault_pattern_filter=fault_pattern_filter,
    )
    output_session.reset_output_file()
    return output_session


def _prepare_runtime_execution(parser, args):
    sorted_alarm_cache_metadata = validate_main_args(parser, args)
    static_context = load_static_context(args)
    valid_alarm_titles = default_valid_alarm_titles()
    print_run_configuration(args, valid_alarm_titles)
    rules_config = build_rules_config()
    output_eligible_rules = collect_output_eligible_rules(rules_config)
    fault_pattern_filter = build_fault_pattern_filter(static_context)
    engine = initialize_engine(args, static_context, rules_config)
    run_started_at = time.time()
    alarm_load_result = load_alarm_data(
        args,
        static_context,
        valid_alarm_titles,
        sorted_alarm_cache_metadata,
    )
    print_alarm_load_summary(alarm_load_result)
    output_session = _build_output_session(
        args, engine, static_context, output_eligible_rules, fault_pattern_filter
    )
    return RuntimeExecutionPlan(
        engine=engine,
        alarm_load_result=alarm_load_result,
        output_session=output_session,
        run_started_at=run_started_at,
    )


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    runtime_plan = _prepare_runtime_execution(parser, args)

    try:
        run_matching_pipeline(
            runtime_plan.engine,
            runtime_plan.alarm_load_result.alarm_generator,
            runtime_plan.output_session,
        )
    finally:
        # 关闭持久输出文件句柄，确保 flush 落盘。
        runtime_plan.output_session.close()

    elapsed = time.time() - runtime_plan.run_started_at
    print_final_summary(
        runtime_plan.engine,
        runtime_plan.alarm_load_result.processed_count,
        runtime_plan.alarm_load_result.filtered_count,
        runtime_plan.output_session.match_count,
        elapsed,
    )
    _print_fault_pattern_filter_summary(runtime_plan.output_session.fault_pattern_filter)


def _print_fault_pattern_filter_summary(fault_pattern_filter):
    """打印落盘前故障模式过滤中，按阈值归因的丢弃数量。"""
    if fault_pattern_filter is None:
        return
    from anchor_grouping_online.matching.fault_pattern_analysis import (
        LONGEST_PATH_EXACT_MAX_SITES,
        MAX_ANALYSIS_SITES,
    )

    stats = fault_pattern_filter.stats
    print(
        f"落盘前故障模式过滤丢弃: "
        f"总站点数>{MAX_ANALYSIS_SITES} {stats.dropped_by_max_analysis_sites} 组 | "
        f"断站簇>{LONGEST_PATH_EXACT_MAX_SITES}(放弃精确搜索) {stats.dropped_by_longest_path_cap} 组"
    )


if __name__ == "__main__":
    main()
