import time

from argparse import ArgumentParser
from dataclasses import dataclass

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from fault_grouping_official.tools.topology_resources import (
    RESOURCE_BUFFER_JSONL,
    resource_display,
)
from fault_grouping_official.matching.group_output_session import MatchOutputSession
from fault_grouping_official.matching.runtime import (
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
from fault_grouping_official.temporal_engine.engine import TemporalGraphEngine
from fault_grouping_official.time_config import (
    DEFAULT_AGGREGATION_WAIT_SEC,
    DEFAULT_CLEAR_DELAY_SEC,
    DEFAULT_HARVEST_INTERVAL_SEC,
)


@dataclass
class RuntimeExecutionPlan:
    engine: TemporalGraphEngine
    alarm_load_result: AlarmLoadResult
    output_session: MatchOutputSession
    run_started_at: float


def _build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, help='output jsonl file')
    parser.add_argument('--resource-buffer', type=str, default=RESOURCE_BUFFER_JSONL, help=f'build_resource_buffer.py 生成的资源缓冲文件（含 ne_graph / site_chains / link_peer_index），默认: {resource_display("resource_buffer.jsonl")}')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='offline', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=DEFAULT_HARVEST_INTERVAL_SEC, help=f'模拟时间下的定时收割周期，单位秒，默认 {DEFAULT_HARVEST_INTERVAL_SEC:g}')
    parser.add_argument('--aggregation-wait-sec', type=float, default=DEFAULT_AGGREGATION_WAIT_SEC, help=f'trigger 成熟前的聚合等待时间，单位秒，默认 {DEFAULT_AGGREGATION_WAIT_SEC:g}')
    parser.add_argument('--clear-delay-sec', type=float, default=DEFAULT_CLEAR_DELAY_SEC, help=f'清除告警最小延迟时间，清除生效时间=max(clear_delay_sec, 清除时间-发生时间)+发生时间，默认 {DEFAULT_CLEAR_DELAY_SEC:g}')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    parser.add_argument('--stream-sorted-alarms', action='store_true', help='从排序告警缓存流式读取，不把全部 valid_alarms 加载到内存；仅当 alarms 本身为 prepare_sorted_alarms.py 生成的排序缓存(JSONL/ZIP)时生效')
    parser.add_argument(
        '--no-rule-check',
        dest='rule_check',
        action='store_false',
        default=True,
        help='关闭落盘规则检查，允许所有引擎匹配结果进入后续处理',
    )
    parser.add_argument(
        '--no-pattern-check',
        dest='pattern_check',
        action='store_false',
        default=True,
        help='关闭落盘故障模式检查，同时跳过模式字段增强',
    )
    return parser


def _build_output_session(
    args, engine, static_context, output_eligible_rules, fault_pattern_filter
):
    output_session = MatchOutputSession(
        args=args,
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
    output_eligible_rules = (
        collect_output_eligible_rules(rules_config)
        if args.rule_check
        else None
    )
    fault_pattern_filter = (
        build_fault_pattern_filter(static_context)
        if args.pattern_check
        else None
    )
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
            args,
            runtime_plan.engine,
            runtime_plan.alarm_load_result.valid_alarms,
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


if __name__ == "__main__":
    main()
