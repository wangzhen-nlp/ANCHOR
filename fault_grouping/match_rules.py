import time

from argparse import ArgumentParser
from dataclasses import dataclass

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import (
    NE_GRAPH_JSON,
    SITE_DEVICE_COUNTS_JSON,
    SITE_GRAPH_JSON,
    SITE_GRAPH_BY_NE_JSON,
    resource_display,
)
from ticket_recall.evaluation.compute_group_output_ticket_recall import compute_group_output_ticket_recall
from fault_grouping.matching.debug import parse_debug_targets
from fault_grouping.matching.group_output_builder import (
    build_alarm_metadata_index,
)
from fault_grouping.matching.group_output_session import MatchOutputSession
from fault_grouping.matching.profiling import PhaseTimer, enable_engine_profiling, enable_output_profiling
from fault_grouping.matching.runtime import (
    AlarmLoadResult,
    LoadedStaticContext,
    build_batch_site_merge_helper,
    build_rules_config,
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
from fault_grouping.temporal_engine.engine import TemporalGraphEngine


@dataclass
class RuntimeExecutionPlan:
    static_context: LoadedStaticContext
    rules_config: dict
    engine: TemporalGraphEngine
    alarm_load_result: AlarmLoadResult
    alarm_metadata_index: dict
    debug_targets: set
    output_session: MatchOutputSession
    start_time: float


def _build_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('alarms', type=str, help='alarm stream')
    parser.add_argument('output', type=str, nargs='?', help='output jsonl file；--no-output 时可省略')
    parser.add_argument('--topo', type=str, default=SITE_GRAPH_BY_NE_JSON, help=f'站点拓扑文件，默认: {resource_display("site_graph_by_ne.json")}；若传空值则退回为基于 ne_graph.json 原始连边自动构建')
    parser.add_argument('--site-chains', type=str, default='', help=f'可选 generate_site_chains.py 输出文件；提供后无 path 约束的上下游遍历优先使用预计算 hop，推荐: {resource_display("site_chains.json")}')
    parser.add_argument('--site-domain', type=str, default=SITE_DEVICE_COUNTS_JSON, help=f'站点画像文件，默认: {resource_display("site_device_counts.json")}')
    parser.add_argument('--site-graph', type=str, default=SITE_GRAPH_JSON, help=f'site_graph.json 文件，默认: {resource_display("site_graph.json")}')
    parser.add_argument('--ne-graph', type=str, default=NE_GRAPH_JSON, help=f'ne_graph.json 文件，默认: {resource_display("ne_graph.json")}')
    parser.add_argument('--mode', type=str, choices=('live', 'offline'), default='offline', help='live: 按 ts 模拟实时流并启动后台定时收割; offline: 每条告警到来时直接触发检查')
    parser.add_argument('--harvest-interval-sec', type=float, default=300.0, help='模拟时间下的定时收割周期，单位秒')
    parser.add_argument('--aggregation-wait-sec', type=float, default=420.0, help='trigger 成熟前的聚合等待时间，单位秒，默认 420')
    parser.add_argument('--clear-delay-sec', type=float, default=420.0, help='清除告警最小延迟时间，清除生效时间=max(clear_delay_sec, 清除时间-发生时间)+发生时间')
    parser.add_argument('--batch-merge-site-hops', type=int, default=0, help='批内候选组额外按站点邻接合并的 hop 数；0 表示关闭，2 表示两跳内可合并')
    parser.add_argument('--batch-merge-density-knn', type=int, default=0, help='批内候选组额外按站点局部密度自适应合并时使用的近邻数；0 表示关闭')
    parser.add_argument('--batch-merge-density-scale', type=float, default=1.0, help='局部密度半径放大倍数，实际阈值=scale * 第k近邻距离')
    parser.add_argument('--batch-merge-density-min-meters', type=float, default=0.0, help='局部密度自适应半径下限，单位米；0 表示不设下限')
    parser.add_argument('--batch-merge-density-max-meters', type=float, default=0.0, help='局部密度自适应半径上限，单位米；0 表示不设上限')
    parser.add_argument('--speedup', type=float, default=1.0, help='按 ts 模拟实时流时的加速倍数，1 表示真实时间，60 表示 1 分钟压到 1 秒')
    parser.add_argument('--debug-trigger', action='append', help='debug: 指定一个 trigger，格式为 站点ID::告警名，可重复传多次')
    parser.add_argument('--verbose-groups', action='store_true', help='打印每个故障组的详细报告；默认静默，仅输出进度与汇总')
    parser.add_argument('--start_time', type=str, help='仅处理告警首次发生时间 >= 该时间的告警，格式如 2025-01-01 00:00:00')
    parser.add_argument('--end_time', type=str, help='仅处理告警首次发生时间 <= 该时间的告警，格式如 2025-01-31 23:59:59')
    parser.add_argument('--compute-ticket-recall', action='store_true', help='在主故障组输出完成后额外计算工单站点召回率')
    parser.add_argument('--ticket-sites', type=str, help='工单站点映射 JSON。不提供时，可退化为从 alarms 自身回推工单站点')
    parser.add_argument('--ticket-field', type=str, default='工单号', help='工单字段名，默认: 工单号')
    parser.add_argument('--ticket-recall-output', type=str, help='工单站点召回率输出文件。默认: <output>.ticket_recall.json')
    parser.add_argument('--rule', action='append', default=[], help='仅启用指定规则；可重复传入，也支持逗号分隔，如 --rule transmission_rule --rule link_rule 或 --rule transmission_rule,link_rule')
    parser.add_argument('--sorted-alarms-input', type=str, default='', help='直接加载 prepare_sorted_alarms.py 生成的排序告警缓存(JSONL/ZIP)；若 alarms 本身是该缓存格式，也会自动识别')
    parser.add_argument('--stream-sorted-alarms', action='store_true', help='从排序告警缓存流式读取，不把全部 valid_alarms 加载到内存；仅对 --sorted-alarms-input 或 alarms 本身为排序缓存时生效')
    parser.add_argument('--sorted-alarms-output', type=str, default='', help='从原始告警加载并排序后，额外写出排序告警缓存；后缀为 .zip 时写压缩包，供后续快速加载')
    parser.add_argument('--compact-output', action='store_true', help='输出轻量化 JSONL：省略 ne_info 内重复告警列表，并压缩空 link 字段；可视化页会从 symptoms 补回节点告警')
    parser.add_argument('--no-output', action='store_true', help='只运行匹配与统计，不创建、不截断、不写入故障组输出 JSONL')
    parser.add_argument('--use-alarm-period-cache', action='store_true', help='可选：把 event_cache 切换为“设备告警时段”模式；默认关闭，保持旧版逐条活跃告警缓存逻辑')
    parser.add_argument('--missing-topology', type=str, default='', help='可选：site_relation_learning/infer.py --mode topology-errors 输出 JSONL；加载其中 error_type=missing 的高置信缺边作为弱拓扑补偿边')
    parser.add_argument('--missing-topology-min-score', type=float, default=0.95, help='加载 --missing-topology 时的最低 score 阈值；默认 0.95，与 infer.py --mode topology-errors 默认缺边阈值一致')
    parser.add_argument('--enable-support-pruning', action='store_true', help='可选：启用候选邻接 role 支撑剪枝；默认关闭以避免额外邻居扫描开销')
    parser.add_argument('--enable-support-count-sort', action='store_true', help='可选：按候选支撑数量排序；默认关闭，因为会提前计算所有候选支撑，可能变慢')
    parser.add_argument('--profile', action='store_true', help='打开后在结束时打印主要阶段耗时（init/ingest/harvest/evaluate/merge/finalize/output/flush），便于定位瓶颈')
    return parser


def _build_output_session(args, engine, static_context, alarm_metadata_index):
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
    return output_session


def _prepare_runtime_execution(parser, args, timer=None):
    def _phase(name):
        # 没开 profile 时退化为 contextlib.nullcontext，零开销。
        return timer.time(name) if timer is not None else _NullCtx()

    start_ts, end_ts = validate_main_args(parser, args)
    with _phase("init.load_static_context"):
        static_context = load_static_context(args)
    valid_alarm_titles = default_valid_alarm_titles()
    print_run_configuration(args, static_context, valid_alarm_titles)
    rules_config = build_rules_config(args, parser)
    batch_site_merge_helper = build_batch_site_merge_helper(args, static_context.topo_downstream_map)
    with _phase("init.initialize_engine"):
        engine = initialize_engine(args, static_context, rules_config, batch_site_merge_helper)
    start_time = time.time()
    with _phase("init.load_alarm_data"):
        alarm_load_result = load_alarm_data(
            args,
            parser,
            static_context,
            valid_alarm_titles,
            start_ts,
            end_ts,
        )
    with _phase("init.build_alarm_metadata_index"):
        alarm_metadata_index = build_alarm_metadata_index(alarm_load_result.valid_alarms)
    print_alarm_load_summary(alarm_load_result)
    debug_targets = parse_debug_targets(args)
    with _phase("init.build_output_session"):
        output_session = _build_output_session(args, engine, static_context, alarm_metadata_index)
    return RuntimeExecutionPlan(
        static_context=static_context,
        rules_config=rules_config,
        engine=engine,
        alarm_load_result=alarm_load_result,
        alarm_metadata_index=alarm_metadata_index,
        debug_targets=debug_targets,
        output_session=output_session,
        start_time=start_time,
    )


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def _maybe_compute_ticket_recall(args):
    if not args.compute_ticket_recall:
        return

    ticket_recall_output = args.ticket_recall_output or f"{args.output}.ticket_recall.json"
    print("⏳ 正在基于当前故障组输出计算工单站点召回率...")
    try:
        recall_result = compute_group_output_ticket_recall(
            args.output,
            args.ticket_sites,
            ticket_field=args.ticket_field,
            alarms_input=args.alarms,
            ne_graph_file=args.ne_graph,
            output_file=ticket_recall_output,
        )
        print(
            f"✅ 工单站点召回率计算完成。工单数: {recall_result['ticket_count']}，"
            f"平均召回率: {recall_result['average_recall']:.6f}，"
            f"输出: {ticket_recall_output}"
        )
    except ValueError as exc:
        print(f"⚠️ 工单站点召回率计算跳过: {exc}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    timer = PhaseTimer() if args.profile else None
    if timer is not None:
        timer.mark_wall_start()

    runtime_plan = _prepare_runtime_execution(parser, args, timer=timer)

    # 启用引擎/输出关键路径的方法级计时（monkey-patch；仅 --profile 下生效）
    if timer is not None:
        enable_engine_profiling(timer, runtime_plan.engine, runtime_plan.output_session)
        enable_output_profiling(timer, runtime_plan.output_session)

    try:
        with (timer.time("pipeline.run_matching_pipeline") if timer else _NullCtx()):
            run_matching_pipeline(
                args,
                runtime_plan.engine,
                runtime_plan.alarm_load_result.valid_alarms,
                runtime_plan.output_session,
                runtime_plan.debug_targets,
                runtime_plan.rules_config,
            )
    finally:
        # 关闭持久输出文件句柄，确保 flush 落盘；后续 ticket_recall 等读路径才能拿到完整内容。
        runtime_plan.output_session.close()

    elapsed = time.time() - runtime_plan.start_time
    print_final_summary(
        args,
        runtime_plan.engine,
        runtime_plan.alarm_load_result.processed_count,
        runtime_plan.alarm_load_result.filtered_count,
        runtime_plan.output_session.match_count,
        elapsed,
    )
    _maybe_compute_ticket_recall(args)

    if timer is not None:
        timer.mark_wall_end()
        timer.print_summary()


if __name__ == "__main__":
    main()
