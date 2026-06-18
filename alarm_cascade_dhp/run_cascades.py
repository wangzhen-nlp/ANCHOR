import json
import sys
import time

from argparse import ArgumentParser
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alarm_cascade_dhp.config import AlarmDHPConfig, StreamPolicyConfig
from alarm_cascade_dhp.engine import AlarmCascadeEngine
from alarm_cascade_dhp.profiling import PhaseTimer, enable_engine_profiling
from alarm_cascade_dhp.visual_output import CascadeVisualOutputSession
from alarm_tools.alarm_inputs import stream_alarm_inputs
from fault_grouping.alarm_events.identity import alarm_content_uuid
from alarm_tools.progress_utils import ProgressBar
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_BY_NE_JSON, SITE_GRAPH_JSON


_OFFLINE_REORDER_LAG_SEC = 0.0
_LIVE_REORDER_LAG_SEC = 300.0


def _build_arg_parser():
    parser = ArgumentParser(
        description="Cluster streaming alarms into topology-aware powered-DHP cascades."
    )
    parser.add_argument("alarms", type=str, help="alarm CSV, JSONL, ZIP, or directory")
    parser.add_argument("output", type=str, help="decision JSONL output")
    parser.add_argument(
        "--groups-output",
        type=str,
        default="",
        help="final cascade snapshot JSON; default: replace output suffix with .groups.json unless --visual-output is set",
    )
    parser.add_argument(
        "--visual-output",
        type=str,
        default="",
        help="finalized cascade JSONL for the fault group browser and propagation visualizer",
    )
    parser.add_argument(
        "--topo",
        type=str,
        default=SITE_GRAPH_BY_NE_JSON,
        help="site topology JSON used for undirected site-hop affinity",
    )
    parser.add_argument(
        "--ne-graph",
        type=str,
        default=NE_GRAPH_JSON,
        help="NE metadata JSON used to resolve source site/domain/type",
    )
    parser.add_argument(
        "--site-graph",
        type=str,
        default=SITE_GRAPH_JSON,
        help="site metadata JSON used by --visual-output",
    )
    parser.add_argument(
        "--visual-ne-scope",
        choices=("alarm-only", "site-context"),
        default="alarm-only",
        help="NEs in --visual-output: clustered alarm devices only, or all devices at cascade sites",
    )
    parser.add_argument("--particles", type=int, default=4, help="particle count")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument(
        "--base-intensity",
        type=float,
        default=0.0001,
        help="DHP new-cascade mass; raise it when the stream should split more often",
    )
    parser.add_argument(
        "--assignment",
        choices=("sample", "map"),
        default="map",
        help="sample particles online or use MAP assignment",
    )
    parser.add_argument(
        "--time-power",
        type=float,
        default=1.0,
        help="PDHP temporal prior power; >1 sharpens burst affinity",
    )
    parser.add_argument(
        "--topology-strength",
        type=float,
        default=1.0,
        help="multiplier for explicit topology affinity",
    )
    parser.add_argument(
        "--topology-max-hops",
        type=int,
        default=2,
        help="maximum explicit NE/site topology hops scored for an alarm",
    )
    parser.add_argument(
        "--require-topology-candidate",
        action="store_true",
        help="let existing cascades compete only when same-NE or ne_graph links support them",
    )
    parser.add_argument(
        "--active-window-sec",
        type=float,
        default=7200.0,
        help="maximum Hawkes support age for an active cascade",
    )
    parser.add_argument(
        "--cooling-after-sec",
        type=float,
        default=1800.0,
        help="mark quiet cascades cooling after this many seconds",
    )
    parser.add_argument(
        "--close-after-sec",
        type=float,
        default=7200.0,
        help="close quiet cascades after this many seconds",
    )
    parser.add_argument(
        "--max-candidate-cascades",
        type=int,
        default=1024,
        help="score only the most recently updated N active cascades; 0 disables the limit",
    )
    parser.add_argument(
        "--reorder-lag-sec",
        type=float,
        default=None,
        help="event-time reorder buffer lag; default: 0 after offline sorting, 300 with --preserve-input-order",
    )
    parser.add_argument(
        "--late-tolerance-sec",
        type=float,
        default=30.0,
        help="lateness tolerated after reorder buffering",
    )
    parser.add_argument(
        "--duplicate-window-sec",
        type=float,
        default=120.0,
        help="compress repeated active raises in this window",
    )
    parser.add_argument(
        "--flap-window-sec",
        type=float,
        default=300.0,
        help="compress clear-followed-by-reopen flaps in this window",
    )
    parser.add_argument(
        "--emit-orphan-clears",
        action="store_true",
        help="emit clear controls even when no matching active raise is known",
    )
    parser.add_argument(
        "--debug-skips",
        action="store_true",
        help="print skipped alarm details and stream-policy collision context",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show source file read progress in --preserve-input-order mode; offline sorting shows it by default",
    )
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="skip the default offline event-time sort and consume source order as a live stream",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print phase timings for input, features, stream policy, model scoring, and output",
    )
    return parser


def _build_engine(args):
    model_config = AlarmDHPConfig(
        particle_count=args.particles,
        seed=args.seed,
        assignment_strategy=args.assignment,
        base_intensity=args.base_intensity,
        time_power=args.time_power,
        topology_strength=args.topology_strength,
        topology_max_hops=args.topology_max_hops,
        require_topology_candidate=args.require_topology_candidate,
        active_window_sec=args.active_window_sec,
        cooling_after_sec=args.cooling_after_sec,
        close_after_sec=args.close_after_sec,
        max_candidate_cascades=args.max_candidate_cascades,
    )
    stream_config = StreamPolicyConfig(
        reorder_lag_sec=args.reorder_lag_sec,
        late_tolerance_sec=args.late_tolerance_sec,
        duplicate_window_sec=args.duplicate_window_sec,
        flap_window_sec=args.flap_window_sec,
        emit_orphan_clears=args.emit_orphan_clears,
        debug_skips=args.debug_skips,
    )
    return AlarmCascadeEngine.from_topology_files(
        site_graph_path=args.topo,
        ne_graph_path=args.ne_graph,
        model_config=model_config,
        stream_config=stream_config,
    )


def _apply_input_mode_defaults(args):
    if args.reorder_lag_sec is None:
        args.reorder_lag_sec = (
            _LIVE_REORDER_LAG_SEC
            if args.preserve_input_order
            else _OFFLINE_REORDER_LAG_SEC
        )
    return args


def _write_decisions(handle, decisions, counts, timer=None, debug_skips=False):
    if timer is None:
        return _write_decisions_now(handle, decisions, counts, debug_skips=debug_skips)
    with timer.time("output.write_decisions"):
        return _write_decisions_now(handle, decisions, counts, debug_skips=debug_skips)


def _resolve_groups_output(args):
    if args.groups_output:
        return args.groups_output
    if args.visual_output:
        return ""
    return str(Path(args.output).with_suffix(".groups.json"))


def _write_decisions_now(handle, decisions, counts, debug_skips=False):
    for decision in decisions:
        payload = decision.to_dict()
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        counts[payload["status"]] = counts.get(payload["status"], 0) + 1
        if debug_skips and decision.status == "skipped":
            _print_skip_debug(decision)


def _print_skip_debug(decision):
    debug = (decision.details or {}).get("skip_debug") or {
        "event_key": decision.event.event_key,
        "current_event": decision.event.compact(),
    }
    payload = {
        "reason": decision.reason,
        **debug,
    }
    print("跳过告警 debug: " + json.dumps(payload, ensure_ascii=False, default=str))


def _status_count(counts, status):
    return counts.get(status, 0)


def _progress_extra_text(engine, counts):
    snapshot = engine.progress_snapshot()
    return (
        f"聚类 {_status_count(counts, 'clustered')}，"
        f"清除 {_status_count(counts, 'clear')}，"
        f"跳过 {_status_count(counts, 'skipped')}，"
        f"cascade {snapshot['cascade_count']}，"
        f"乱序缓冲 {snapshot['pending_event_count']}"
    )


def _refresh_process_progress(progress, engine, counts, force=False):
    progress.set_extra_text(_progress_extra_text(engine, counts), force=force)


class _StreamProcessProgress:
    """Throttle processing progress for sorted files or a real stream."""

    def __init__(self, total=0, interval_sec=0.2):
        self.bar = ProgressBar(total, "处理告警流")
        self.interval_sec = interval_sec
        self.last_refresh = 0.0

    def refresh(self, processed_count, engine, counts, force=False):
        now = time.monotonic()
        if not force and now - self.last_refresh < self.interval_sec:
            return
        _refresh_process_progress(self.bar, engine, counts, force=force)
        self.bar.set(processed_count)
        self.last_refresh = now

    def close(self):
        self.bar.close()


def _print_run_configuration(args):
    print("⏳ 正在初始化告警 cascade 聚类器与拓扑映射...")
    print(
        "聚类配置: "
        f"particles={args.particles}, "
        f"assignment={args.assignment}, "
        f"time_power={args.time_power:g}, "
        f"topology_strength={args.topology_strength:g}, "
        f"topology_max_hops={args.topology_max_hops}"
    )
    if args.require_topology_candidate:
        print("拓扑候选 gate: 开启，仅同 NE 或 ne_graph 明确可达设备簇可参与已有 cascade 打分")
    candidate_limit = (
        "不限制"
        if args.max_candidate_cascades == 0
        else str(args.max_candidate_cascades)
    )
    print(
        "候选窗口: "
        f"active={args.active_window_sec:g}s, "
        f"cooling={args.cooling_after_sec:g}s, "
        f"close={args.close_after_sec:g}s, "
        f"最近 cascade 上限={candidate_limit}"
    )
    print(
        "流清洗: "
        f"reorder_lag={args.reorder_lag_sec:g}s, "
        f"late_tolerance={args.late_tolerance_sec:g}s, "
        f"duplicate_window={args.duplicate_window_sec:g}s, "
        f"flap_window={args.flap_window_sec:g}s"
    )
    print(
        "输入顺序: "
        + (
            "保留源顺序，按实时流入口处理"
            if args.preserve_input_order
            else "离线文件默认按事件时间排序后聚类"
        )
    )
    if args.profile:
        print("性能分析: 开启，结束后打印主要阶段累计耗时")
    if args.debug_skips:
        print("跳过告警 debug: 开启，逐条打印当前告警与清洗碰撞上下文")
    if args.visual_output:
        print("可视化输出: cascade 关闭时写 JSONL，输入结束时补写仍未关闭的 cascade")
        print(
            "可视化设备范围: "
            + (
                "仅 cascade 中发生告警的设备"
                if args.visual_ne_scope == "alarm-only"
                else "cascade 站点内全部设备"
            )
        )


def _time_phase(timer, phase_name):
    if timer is None:
        return _NullContext()
    return timer.time(phase_name)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _next_alarm(alarm_stream, timer):
    if timer is None:
        return next(alarm_stream)
    with timer.time("input.next_alarm"):
        return next(alarm_stream)


def _iter_input_alarm_records(args, timer, show_progress=None):
    if show_progress is None:
        show_progress = args.show_progress
    alarm_stream = iter(stream_alarm_inputs(args.alarms, show_progress=show_progress))
    while True:
        try:
            alarm = dict(_next_alarm(alarm_stream, timer))
            alarm["occurrence_uuid"] = alarm_content_uuid(alarm)
            yield alarm
        except StopIteration:
            break


def _load_sorted_events(args, engine, timer=None):
    print("⏳ 正在加载告警并构造 cascade 事件...")
    events = []
    with _time_phase(timer, "input.load_events"):
        for alarm in _iter_input_alarm_records(args, timer, show_progress=True):
            events.append(engine.features.from_alarm_record(alarm))

    print("⏳ 正在按事件时间排序 cascade 告警...")
    with _time_phase(timer, "input.sort_events"):
        events.sort(key=lambda event: event.ts)
    print(f"✅ 已准备 {len(events)} 条按事件时间排序的告警事件")
    return events


def _process_alarm_records(args, engine, output_handle, counts, timer, visual_output=None):
    processed_count = 0
    process_progress = _StreamProcessProgress()
    process_progress.refresh(processed_count, engine, counts, force=True)
    try:
        with _time_phase(timer, "pipeline.process_stream"):
            for alarm in _iter_input_alarm_records(args, timer):
                decisions = engine.observe(alarm)
                _write_decisions(
                    output_handle,
                    decisions,
                    counts,
                    timer=timer,
                    debug_skips=args.debug_skips,
                )
                _emit_closed_visual_output(
                    visual_output,
                    engine,
                    decisions,
                    timer=timer,
                )
                processed_count += 1
                process_progress.refresh(processed_count, engine, counts)
    finally:
        process_progress.refresh(processed_count, engine, counts, force=True)
        process_progress.close()
    return processed_count


def _process_sorted_events(
    events,
    engine,
    output_handle,
    counts,
    timer,
    visual_output=None,
    debug_skips=False,
):
    processed_count = 0
    process_progress = _StreamProcessProgress(total=len(events))
    process_progress.refresh(processed_count, engine, counts, force=True)
    try:
        with _time_phase(timer, "pipeline.process_stream"):
            for event in events:
                decisions = engine.observe_event(event)
                _write_decisions(
                    output_handle,
                    decisions,
                    counts,
                    timer=timer,
                    debug_skips=debug_skips,
                )
                _emit_closed_visual_output(
                    visual_output,
                    engine,
                    decisions,
                    timer=timer,
                )
                processed_count += 1
                process_progress.refresh(processed_count, engine, counts)
    finally:
        process_progress.refresh(processed_count, engine, counts, force=True)
        process_progress.close()
    return processed_count


def _decision_now_ts(engine, decisions):
    now_ts = engine.model.last_ts
    for decision in decisions:
        now_ts = max(now_ts, decision.event.ts)
    return now_ts


def _emit_closed_visual_output(visual_output, engine, decisions, timer=None):
    if visual_output is None or not decisions:
        return 0
    with _time_phase(timer, "output.write_visual_groups"):
        return visual_output.emit_closed(
            engine,
            now_ts=_decision_now_ts(engine, decisions),
        )


def _emit_remaining_visual_output(visual_output, engine, decisions, timer=None):
    if visual_output is None:
        return 0
    with _time_phase(timer, "output.write_visual_groups"):
        return visual_output.emit_remaining(
            engine,
            now_ts=_decision_now_ts(engine, decisions),
        )


def main():
    parser = _build_arg_parser()
    args = _apply_input_mode_defaults(parser.parse_args())
    groups_output = _resolve_groups_output(args)
    timer = PhaseTimer() if args.profile else None
    if timer is not None:
        timer.mark_wall_start()
    _print_run_configuration(args)
    with _time_phase(timer, "init.build_engine"):
        engine = _build_engine(args)
    visual_output = None
    if args.visual_output:
        with _time_phase(timer, "init.build_visual_output"):
            visual_output = CascadeVisualOutputSession.from_files(
                args.visual_output,
                args.ne_graph,
                args.site_graph,
                ne_scope=args.visual_ne_scope,
            )
            visual_output.reset_output_file()
    if timer is not None:
        enable_engine_profiling(timer, engine)
    counts = {}
    start_time = time.time()
    sorted_events = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.preserve_input_order:
            sorted_events = _load_sorted_events(args, engine, timer=timer)

        with output_path.open("w", encoding="utf-8") as output_handle:
            if sorted_events is None:
                processed_count = _process_alarm_records(
                    args,
                    engine,
                    output_handle,
                    counts,
                    timer,
                    visual_output=visual_output,
                )
            else:
                processed_count = _process_sorted_events(
                    sorted_events,
                    engine,
                    output_handle,
                    counts,
                    timer,
                    visual_output=visual_output,
                    debug_skips=args.debug_skips,
                )

            print("⏳ 数据流读取完毕，正在清空乱序缓冲并输出剩余 cascade 决策...")
            with _time_phase(timer, "pipeline.flush"):
                flush_decisions = engine.flush()
                _write_decisions(
                    output_handle,
                    flush_decisions,
                    counts,
                    timer=timer,
                    debug_skips=args.debug_skips,
                )
                _emit_remaining_visual_output(
                    visual_output,
                    engine,
                    flush_decisions,
                    timer=timer,
                )
    finally:
        if visual_output is not None:
            visual_output.close()

    snapshots = engine.cascade_snapshots()
    group_path = None
    if groups_output:
        group_path = Path(groups_output)
        group_path.parent.mkdir(parents=True, exist_ok=True)
        with _time_phase(timer, "output.write_groups"):
            with group_path.open("w", encoding="utf-8") as group_handle:
                json.dump(
                    {
                        "decision_counts": counts,
                        "cascade_count": len(snapshots),
                        "cascades": snapshots,
                    },
                    group_handle,
                    ensure_ascii=False,
                    indent=2,
                )
                group_handle.write("\n")

    elapsed = time.time() - start_time
    print(
        f"🏁 告警流处理完毕。共读取 {processed_count} 条告警，"
        f"聚类 {_status_count(counts, 'clustered')} 条，"
        f"清除控制 {_status_count(counts, 'clear')} 条，"
        f"跳过 {_status_count(counts, 'skipped')} 条，"
        f"汇聚 {len(snapshots)} 个 cascade，"
        f"耗时 {elapsed:.4f} 秒。"
    )
    print(f"决策输出: {output_path}")
    if group_path is not None:
        print(f"cascade 输出: {group_path}")
    if visual_output is not None:
        print(f"可视化输出: {visual_output.output_path} ({visual_output.emitted_count} 个 cascade)")
    if timer is not None:
        timer.mark_wall_end()
        timer.print_summary()


if __name__ == "__main__":
    main()
