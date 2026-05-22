import time

from contextlib import contextmanager

from alarm_cascade_dhp import model as model_module


class PhaseTimer:
    """Aggregate optional cascade profiling timings by named phase."""

    def __init__(self):
        self._phases = {}
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
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started_at)

    def record(self, name, elapsed):
        slot = self._phases.get(name)
        if slot is None:
            self._phases[name] = [elapsed, 1]
        else:
            slot[0] += elapsed
            slot[1] += 1

    def phase_snapshot(self):
        return {
            name: {"total_sec": total, "count": count}
            for name, (total, count) in self._phases.items()
        }

    def wrap_method(self, owner, attr, phase_name):
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            started_at = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.record(phase_name, time.perf_counter() - started_at)

        setattr(owner, attr, wrapped)
        return original

    def print_summary(self, title="alarm cascade DHP 性能分析"):
        wall = self.wall_elapsed
        if not self._phases and wall <= 0:
            return

        line_width = 98
        print()
        print("=" * line_width)
        print(f"{title}（wall={wall:.3f}s）")
        print("=" * line_width)

        for prefix, label in (
            ("init.", "准备阶段"),
            ("pipeline.", "主流程阶段"),
            ("input.", "输入读取"),
            ("ingest.", "告警入口"),
            ("features.", "告警特征"),
            ("stream.", "流清洗与缓冲"),
            ("model.", "模型分配"),
            ("score.", "候选打分"),
            ("update.", "簇状态更新"),
            ("topology.", "拓扑调用"),
            ("output.", "结果输出"),
        ):
            rows = self._rows_for_prefix(prefix)
            if not rows:
                continue
            print()
            print(f"[{prefix[:-1]}] {label}")
            for name, total, count in rows:
                average_ms = total / max(count, 1) * 1000.0
                wall_pct = total / wall * 100.0 if wall > 0 else 0.0
                print(
                    f"  {name:<42} {total:>9.3f}s "
                    f"{count:>7}次 {wall_pct:>6.1f}% "
                    f"avg={average_ms:>10.3f}ms"
                )

        print()
        print("-" * line_width)
        print("说明：")
        print("  • 各行是累计耗时；父阶段包含子阶段，跨 block 相加会重复计时。")
        print("  • input.next_alarm 包含原始 CSV/JSONL/ZIP 读取与解析。")
        print("  • score.* 是候选 cascade 精排热点；候选上限由 max_candidate_cascades 控制。")
        print("=" * line_width)

    def _rows_for_prefix(self, prefix):
        rows = [
            (name, total, count)
            for name, (total, count) in self._phases.items()
            if name.startswith(prefix)
        ]
        return sorted(rows, key=lambda row: -row[1])


def enable_engine_profiling(timer, engine):
    """Wrap the online cascade path only when the CLI enables profiling."""
    timer.wrap_method(engine, "observe", "ingest.observe")
    timer.wrap_method(engine, "observe_event", "ingest.observe_event")
    timer.wrap_method(engine, "flush", "stream.flush")
    timer.wrap_method(engine, "_consume_sanitized", "stream.consume_sanitized")
    timer.wrap_method(engine.features, "from_alarm_record", "features.from_alarm_record")
    timer.wrap_method(
        engine.features,
        "from_match_rules_item",
        "features.from_match_rules_item",
    )
    timer.wrap_method(engine.sanitizer, "push", "stream.sanitizer_push")
    timer.wrap_method(engine.sanitizer, "flush", "stream.sanitizer_flush")

    model = engine.model
    timer.wrap_method(model, "observe_raise", "model.observe_raise")
    timer.wrap_method(model, "observe_clear", "model.observe_clear")
    timer.wrap_method(model, "_step_particle", "model.particle_step")
    timer.wrap_method(model, "_normalize_weights", "model.normalize_weights")
    timer.wrap_method(model, "_resample_if_needed", "model.resample_check")
    timer.wrap_method(model, "_new_cluster_proposal", "score.new_cascade")
    timer.wrap_method(model, "_existing_cluster_proposal", "score.existing_cascade")
    timer.wrap_method(model, "cascade_snapshots", "output.cascade_snapshots")

    timer.wrap_method(model_module._Cluster, "time_rate", "score.time_rate")
    timer.wrap_method(
        model_module._Cluster,
        "content_log_predictive",
        "score.content_predictive",
    )
    timer.wrap_method(
        model_module._Cluster,
        "topology_affinity",
        "score.topology_affinity",
    )
    timer.wrap_method(model_module._Cluster, "add", "update.cluster_add")
    timer.wrap_method(
        model_module._Cluster,
        "_update_time_kernel",
        "update.time_kernel",
    )
    timer.wrap_method(
        model_module._Cluster,
        "_update_topology_counts",
        "update.topology_counts",
    )
    timer.wrap_method(engine.topology, "relation", "topology.relation")
    timer.wrap_method(engine.topology, "hop_distance", "topology.hop_distance")
