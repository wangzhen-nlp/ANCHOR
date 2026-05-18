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
        wall = self.wall_elapsed
        rows = sorted(self._phases.items(), key=lambda kv: -kv[1][0])
        if not rows and wall <= 0:
            return

        name_w = max([len(n) for n, _ in rows] + [16])
        line_w = name_w + 50
        print()
        print("=" * line_w)
        print(f"{title}（wall={wall:.3f}s）")
        print("=" * line_w)
        print(f"{'阶段':<{name_w}} {'总耗时(s)':>10} {'次数':>9} {'占比wall':>9} {'avg(ms)':>10}")
        print("-" * line_w)
        for name, (total_sec, count) in rows:
            avg_ms = total_sec / max(count, 1) * 1000
            pct = total_sec / wall * 100 if wall > 0 else 0.0
            print(f"{name:<{name_w}} {total_sec:>10.3f} {count:>9d} {pct:>8.1f}% {avg_ms:>10.4f}")
        print("-" * line_w)
        print("说明：阶段之间可能嵌套（例如 harvest 包含 evaluate+merge+finalize），")
        print("      因此各项之和可能超过 wall。重点看占比 wall 高的阶段。")
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
