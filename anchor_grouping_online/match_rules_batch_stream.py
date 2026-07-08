"""滑动窗口二次汇聚测试：排序告警流 -> 逐窗口 aggregate_alarm_groups -> jsonl。

输入为 prepare_sorted_alarms.py 生成的排序告警缓存（JSONL/ZIP）。预处理按
发生时间做滑动窗口切段（默认每段
7 分钟、步长 1 分钟，一条告警最多落进 7 个段），全程流式，不整体加载：

1. 逐条读取排序告警并转成生成器输出字典；清除告警与「故障组ID」为空的
   告警跳过（故障组成员只考虑已归组的上报告警）；
2. 窗口缓冲只保留当前窗口内的告警，告警时间越过窗口右界即触发一次结算：
   把缓冲按「故障组ID」组装为 {故障组id: [告警, ...]}，调用
   BatchFaultGroupMatcher.aggregate_alarm_groups；
3. 每个非空窗口的结果作为一行 JSON 追加写入输出文件。

用法：
    python match_rules_batch_stream.py <sorted_alarms> <output.jsonl> \
        [--resource-buffer ...] [--window-minutes 7] [--step-minutes 1] \
        [--associate-time 7] [--max-group-time 10] [--max-group-member 1000]
"""

import json
import math
import time

from argparse import ArgumentParser
from collections import deque
from datetime import datetime

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.alarm_events.generator import generate_alarm
from anchor_grouping_online.alarm_events.io import is_clear_alarm
from anchor_grouping_online.alarm_events.sorted_cache import iter_sorted_alarm_cache_items
from anchor_grouping_online.match_rules_batch import BatchFaultGroupMatcher
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


def _build_arg_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("alarms", type=str, help="排序告警缓存（JSONL/ZIP）")
    parser.add_argument("output", type=str, help="输出 jsonl 文件，每窗口一行")
    parser.add_argument(
        "--resource-buffer",
        type=str,
        default=RESOURCE_BUFFER_JSONL,
        help="build_resource_buffer.py 生成的资源缓冲文件",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=7.0, help="每段告警的时间窗，分钟"
    )
    parser.add_argument(
        "--step-minutes", type=float, default=1.0, help="窗口滑动步长，分钟"
    )
    parser.add_argument(
        "--associate-time", type=float, default=7.0,
        help="aggregate_alarm_groups 的关联时间窗，分钟",
    )
    parser.add_argument(
        "--max-group-time", type=float, default=10.0,
        help="aggregate_alarm_groups 的汇聚组最大时间窗，分钟",
    )
    parser.add_argument(
        "--max-group-member", type=int, default=1000,
        help="汇聚组允许包含的最大原始故障组个数",
    )
    parser.add_argument(
        "--batch-isolated", action="store_true",
        help="批隔离：不使用跨窗口信息，只做本窗口内的告警汇聚",
    )
    return parser


def _format_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _iter_window_alarms(alarms_path):
    """流式产出 (ts, 故障组ID, 生成器格式告警)；跳过清除告警与未归组告警。

    返回的生成器带 skip 统计，结束后可读 .clear_skipped / .ungrouped_skipped。
    """
    def _generate():
        for item in iter_sorted_alarm_cache_items(
            alarms_path, show_progress=True
        ):
            # 生成器输出不再携带清除标记，清除告警在这里按原始
            # 载荷的「清除告警」字段过滤，不进入汇聚。
            if is_clear_alarm(item.get("alarm", {})):
                stats["clear_skipped"] += 1
                continue
            generated_alarm = generate_alarm(item)
            group_id = str(item.get("alarm", {}).get("故障组ID", "") or "").strip()
            if not group_id:
                stats["ungrouped_skipped"] += 1
                continue
            yield float(item["ts"]), group_id, generated_alarm

    stats = {"clear_skipped": 0, "ungrouped_skipped": 0}
    return _generate(), stats


def _settle_window(
    buffer, matcher, output_file, start, window_sec,
    associate_time, max_group_time, max_group_member,
    window_index, emitted_row_number,
):
    """结算 [start, start + window_sec) 窗口：组装 -> 汇聚 -> 追加一行。

    缓冲为空时不输出，返回是否写出了记录。
    """
    if not buffer:
        return False
    alarm_groups = {}
    for _ts, group_id, generated_alarm in buffer:
        alarm_groups.setdefault(group_id, []).append(generated_alarm)
    aggregation_started_at = time.perf_counter()
    agg_alarm_groups = matcher.aggregate_alarm_groups(
        alarm_groups,
        associate_time=associate_time,
        max_group_time=max_group_time,
        max_group_member=max_group_member,
    )
    aggregation_elapsed_seconds = time.perf_counter() - aggregation_started_at
    merged_input_group_count = sum(
        len(member_entries)
        for member_entries in agg_alarm_groups.values()
    )
    record = {
        "window_start": start,
        "window_end": start + window_sec,
        "window_start_time": _format_ts(start),
        "window_end_time": _format_ts(start + window_sec),
        "input_group_count": len(alarm_groups),
        "input_alarm_count": len(buffer),
        "agg_group_count": len(agg_alarm_groups),
        "merged_input_group_count": merged_input_group_count,
        "aggregation_elapsed_seconds": round(
            aggregation_elapsed_seconds, 6
        ),
        "agg_alarm_groups": agg_alarm_groups,
    }
    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    # 每个窗口结果立即落盘，方便运行期间通过 tail -f
    # 查看完整中间结果，不必等到文件关闭。
    output_file.flush()
    print(
        f"  [窗口 {window_index}] "
        f"{record['window_start_time']} ~ {record['window_end_time']}："
        f"{record['input_alarm_count']} 条告警 / "
        f"{record['input_group_count']} 个原始组 -> "
        f"{record['agg_group_count']} 个汇聚组，"
        f"其中合并原始组 {merged_input_group_count} 个，"
        f"汇聚耗时 {aggregation_elapsed_seconds:.3f} 秒"
        f"（已写入第 "
        f"{emitted_row_number} 行）",
        flush=True,
    )
    return True


def run_sliding_window_aggregation(
    alarms_path,
    output_path,
    resource_buffer=RESOURCE_BUFFER_JSONL,
    window_minutes=7.0,
    step_minutes=1.0,
    associate_time=7.0,
    max_group_time=10.0,
    max_group_member=1000,
    batch_isolated=False,
):
    """按滑动窗口切段并逐窗口二次汇聚，返回运行统计。

    默认 matcher 跨窗口持久（有状态）：本窗口的故障组可以与之前窗口喂入
    的告警/故障组建立汇聚关系。
    batch_isolated=True 时只做本窗口内的告警汇聚，不使用跨窗口信息，
    汇聚组 ID（UUID）每窗口独立生成，跨窗口比较无意义。
    """
    window_sec = float(window_minutes) * 60.0
    step_sec = float(step_minutes) * 60.0
    matcher = BatchFaultGroupMatcher(
        resource_buffer=resource_buffer, batch_isolated=batch_isolated
    )

    print(
        f"开始滑动窗口汇聚：窗口 {window_minutes:g} 分钟，"
        f"步长 {step_minutes:g} 分钟，"
        f"{'批隔离（不使用跨窗口信息）' if batch_isolated else '跨窗口有状态'}，"
        f"输出 {output_path}",
        flush=True,
    )

    alarm_stream, skip_stats = _iter_window_alarms(alarms_path)
    # 缓冲近似保有当前窗口的告警。对输入顺序不做假设：乱序只影响告警
    # 被切进哪一批，聚合判定由 matcher 按告警自身时间完成，与顺序无关。
    buffer = deque()
    window_start = None
    window_count = 0
    emitted_window_count = 0

    def _align_to_step(ts):
        return math.floor(ts / step_sec) * step_sec

    with open(output_path, "w", encoding="utf-8") as output_file:

        def settle_window(start):
            nonlocal window_count, emitted_window_count
            window_count += 1
            if _settle_window(
                buffer, matcher, output_file, start, window_sec,
                associate_time, max_group_time, max_group_member,
                window_count, emitted_window_count + 1,
            ):
                emitted_window_count += 1

        for ts, group_id, generated_alarm in alarm_stream:
            if window_start is None:
                window_start = _align_to_step(ts)
            # 新告警越过当前窗口右界：依次结算已完整的窗口并滑动。
            while ts >= window_start + window_sec:
                settle_window(window_start)
                window_start += step_sec
                while buffer and buffer[0][0] < window_start:
                    buffer.popleft()
                if not buffer and ts >= window_start + window_sec:
                    # 空档期跳到第一个能覆盖到该告警的窗口起点
                    # （align(ts - window) + step），中间被跳过的窗口在
                    # 理想网格中必然为空，不影响滑动语义。
                    window_start = max(
                        window_start,
                        _align_to_step(ts - window_sec) + step_sec,
                    )
            buffer.append((ts, group_id, generated_alarm))

        # 流结束后把缓冲内剩余告警覆盖到的窗口全部结算完。
        while buffer:
            settle_window(window_start)
            window_start += step_sec
            while buffer and buffer[0][0] < window_start:
                buffer.popleft()

    return {
        "window_count": window_count,
        "emitted_window_count": emitted_window_count,
        "clear_skipped": skip_stats["clear_skipped"],
        "ungrouped_skipped": skip_stats["ungrouped_skipped"],
    }


def main():
    args = _build_arg_parser().parse_args()
    stats = run_sliding_window_aggregation(
        args.alarms,
        args.output,
        resource_buffer=args.resource_buffer,
        window_minutes=args.window_minutes,
        step_minutes=args.step_minutes,
        associate_time=args.associate_time,
        max_group_time=args.max_group_time,
        max_group_member=args.max_group_member,
        batch_isolated=args.batch_isolated,
    )
    print(
        f"完成：结算窗口 {stats['window_count']} 个，"
        f"输出非空窗口 {stats['emitted_window_count']} 行，"
        f"跳过清除告警 {stats['clear_skipped']} 条、"
        f"未归组告警 {stats['ungrouped_skipped']} 条",
        flush=True,
    )


if __name__ == "__main__":
    main()
