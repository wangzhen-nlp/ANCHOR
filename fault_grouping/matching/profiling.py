"""极简阶段计时工具：聚合每个命名阶段的耗时和调用次数，最后打印汇总表。

- 仅在 --profile 开启时由 match_rules.py 装配，平时零侵入。
- 通过 monkey-patch 给 engine / output 关键方法包一层计时，不修改业务代码。
"""
import time
from contextlib import contextmanager


class PhaseTimer:
    def __init__(self):
        self._phases = {}  # name -> [total_sec, count]
        self._wall_start = None
        self._wall_end = None

    def mark_wall_start(self):
        self._wall_start = time.perf_counter()

    def mark_wall_end(self):
        self._wall_end = time.perf_counter()

    @property
    def wall_elapsed(self):
        if self._wall_start is None:
            return 0.0
        end = self._wall_end if self._wall_end is not None else time.perf_counter()
        return end - self._wall_start

    @contextmanager
    def time(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - t0)

    def _record(self, name, sec):
        slot = self._phases.get(name)
        if slot is None:
            self._phases[name] = [sec, 1]
        else:
            slot[0] += sec
            slot[1] += 1

    def wrap_method(self, owner, attr, phase_name):
        """把 owner.attr 这个 bound 方法/函数替换成计时版。返回原始可调用对象。"""
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(phase_name, time.perf_counter() - t0)

        setattr(owner, attr, wrapped)
        return original

    def print_summary(self, title="match_rules 性能分析"):
        """按嵌套结构分组打印：init / pipeline / harvest 三个块。

        嵌套关系（硬编码，与 enable_engine_profiling 对应）：
          pipeline.run_matching_pipeline 是 wall 容器
            ├── ingest.process_event          ─┐
            ├── output.write_matches           │
            └── flush.total                   ─┤
          harvest.total（被 ingest 和 flush 共同嵌套调用）
            ├── harvest.evaluate
            ├── harvest.merge_expand
            └── harvest.finalize

        每个阶段的 cum(累计) 含其子阶段；同时给出 self(本阶段自身排除子阶段)，
        互斥的叶子节点之和 ≈ wall。
        """
        wall = self.wall_elapsed
        if not self._phases and wall <= 0:
            return

        # 拆桶
        init_rows = sorted(
            [(n, *kv) for n, kv in self._phases.items() if n.startswith("init.")],
            key=lambda x: -x[1],
        )
        pipeline_total = self._phases.get("pipeline.run_matching_pipeline")
        pipeline_children = []
        for n, kv in self._phases.items():
            if n.startswith(("ingest.", "output.", "flush.")):
                pipeline_children.append((n, *kv))
        pipeline_children.sort(key=lambda x: -x[1])
        harvest_total = self._phases.get("harvest.total")
        harvest_children_names = ("harvest.evaluate", "harvest.merge_expand", "harvest.finalize")
        harvest_children = [
            (n, *self._phases[n]) for n in harvest_children_names if n in self._phases
        ]

        def pct(t):
            return (t / wall * 100) if wall > 0 else 0.0

        def fmt_row(name, total, count, indent=0, suffix=""):
            prefix = " " * indent
            if count is None:
                # 派生行（如 harvest.self / init 合计），不展示调用次数与 avg。
                count_str = f"{'—':>5}"
                avg_str = f"{'—':>15}"
            else:
                count_str = f"{count:>5}次"
                avg_ms = total / max(count, 1) * 1000 if count else 0.0
                avg_str = f"avg={avg_ms:>10.3f}ms"
            return f"{prefix}{name:<{40 - indent}} {total:>9.3f}s {count_str} {pct(total):>6.1f}%  {avg_str}{suffix}"

        line_w = 96
        print()
        print("=" * line_w)
        print(f"{title}（wall={wall:.3f}s）")
        print("=" * line_w)

        # [init] 块
        if init_rows:
            print()
            print("[init] 准备阶段（互不嵌套）")
            init_sum = 0.0
            for name, total, count in init_rows:
                print(fmt_row(name, total, count, indent=2))
                init_sum += total
            print(f"  {'─' * 70}")
            print(fmt_row("init 合计", init_sum, None, indent=2))

        # [pipeline] 块
        if pipeline_total is not None or pipeline_children:
            print()
            print("[pipeline] 主流程（wall 容器；三个子阶段互不重叠，但分别向 [harvest] 调用）")
            if pipeline_total is not None:
                p_total, p_count = pipeline_total
                print(fmt_row("pipeline.run_matching_pipeline", p_total, p_count, indent=2))
            for name, total, count in pipeline_children:
                marker = ""
                if name == "ingest.process_event":
                    marker = "  ← 每条事件内嵌一次 harvest（offline 模式）"
                elif name == "flush.total":
                    marker = "  ← 内嵌一次 harvest（force=True）"
                print(fmt_row(name, total, count, indent=4, suffix=marker))

        # [harvest] 块
        if harvest_total is not None:
            h_total, h_count = harvest_total
            children_sum = sum(t for _, t, _ in harvest_children)
            self_time = max(h_total - children_sum, 0.0)
            print()
            print("[harvest] 收割路径（嵌套在 ingest.process_event / flush.total 内，时间双重计入它们）")
            print(fmt_row("harvest.total", h_total, h_count, indent=2,
                          suffix=f"  ← 累计被调用 {h_count} 次"))
            for name, total, count in harvest_children:
                # 在 harvest 内的占比，给一个更直观的"占 harvest" 数字
                pct_in_h = (total / h_total * 100) if h_total > 0 else 0.0
                print(fmt_row(name, total, count, indent=4,
                              suffix=f"  ← 占 harvest {pct_in_h:>5.1f}%"))
            print(fmt_row("(harvest.self = total − 子阶段)", self_time, None, indent=4,
                          suffix="  ← _collect_pending_matches 自身框架开销"))

        print()
        print("-" * line_w)
        print("说明：")
        print("  • cum(s) = 累计耗时（含其子阶段）；wall% 同口径")
        print("  • harvest.* 不在 pipeline 之外，它内嵌在 ingest 和 flush 里，时间被双重计入")
        print("  • 互斥（叶子）耗时:  init.* + harvest.evaluate + harvest.merge_expand")
        print("                     + harvest.finalize + harvest.self + ingest.self + output.* + flush.self")
        print("                     之和 ≈ wall")
        print("=" * line_w)


def enable_engine_profiling(timer, engine, output_session):
    """给 engine 和 output_session 的关键方法包一层计时。

    覆盖的关键路径（offline 模式下每条告警都会走）：
      ingest:      engine.process_event 内部（事件入 cache + trigger 入 pending）
      harvest:     engine._collect_pending_matches（事件触发同步收割）
        evaluate:    _evaluate_mature_pending_items（最重，规则匹配）
        merge:       _merge_and_expand_raw_matches（批内合并 + pending 上下文扩展）
        finalize:    _finalize_expanded_matches_for_output（历史合并 + 可见性过滤）
      output:      output_session.write_matches（落盘）
      flush:       engine.flush_pending（流末尾强制收割）
    """
    timer.wrap_method(engine, "process_event", "ingest.process_event")
    timer.wrap_method(engine, "_collect_pending_matches", "harvest.total")
    timer.wrap_method(engine, "_evaluate_mature_pending_items", "harvest.evaluate")
    timer.wrap_method(engine, "_merge_and_expand_raw_matches", "harvest.merge_expand")
    timer.wrap_method(engine, "_finalize_expanded_matches_for_output", "harvest.finalize")
    timer.wrap_method(engine, "flush_pending", "flush.total")
    if output_session is not None:
        timer.wrap_method(output_session, "write_matches", "output.write_matches")
