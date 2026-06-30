import json
import threading

from dataclasses import dataclass, field

from fault_grouping_official.matching.group_output_builder import build_jsonl_match_output
from fault_grouping_official.temporal_engine.engine import TemporalGraphEngine


# orjson 比 stdlib json 在 dumps 上快约 3~5×，且默认 UTF-8 二进制输出（省去 encode）。
# 输出格式仅差在分隔符紧凑（无空格），仍是合法 JSONL，任何 JSON 解析器都能读。
# orjson 未安装时自动回退到 stdlib 实现。
try:
    import orjson
    _ORJSON_OPTS = orjson.OPT_NON_STR_KEYS  # 容忍 dict 里非字符串 key（保持与 stdlib 行为一致）
    _NEWLINE_BYTES = b"\n"

    def _dumps_line(obj):
        return orjson.dumps(obj, option=_ORJSON_OPTS) + _NEWLINE_BYTES
except ImportError:
    def _dumps_line(obj):
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@dataclass
class MatchOutputSession:
    args: object
    engine: TemporalGraphEngine
    output_path: str
    ne_graph_data: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
    # 可落盘规则名集合（frozenset）。None 表示不做规则过滤，全部落盘。
    # 只有 merged_rules 命中其中任意一个规则的故障组才会写入输出文件。
    output_eligible_rules: object = None
    # 落盘前故障模式过滤器（FaultPatternFilter）。None 表示不做故障模式过滤。
    # 等价 analyze_case_fault_patterns.py --filter-others --one-component-only。
    fault_pattern_filter: object = None
    match_count: int = 0
    process_progress: object = None
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    # 持久 append-mode 文件句柄，避免每批 open+close 的 syscall 开销。
    # reset_output_file() 截断 + 打开；close() 显式收尾。多线程下由 output_lock 保护。
    _fw: object = field(default=None, init=False, repr=False)

    def reset_output_file(self):
        # 先关掉已有句柄，再截断文件并打开新句柄。
        with self.output_lock:
            self._close_fw_locked()
            with open(self.output_path, 'wb'):
                pass
            self._fw = open(self.output_path, 'ab')

    def close(self):
        with self.output_lock:
            self._close_fw_locked()

    def _close_fw_locked(self):
        fw = self._fw
        if fw is None:
            return
        # 无论 flush/close 是否抛异常，都把 _fw 清空，避免下次 write 复用已损坏句柄
        self._fw = None
        try:
            fw.flush()
        except Exception:
            pass
        try:
            fw.close()
        except Exception:
            pass

    def build_progress_extra_text(self):
        merge_stats = self.engine.get_batch_merge_stats_snapshot().get("total", {})
        primary_merge_count = merge_stats.get('eid_merge_group_count', 0)
        primary_merge_label = "eid合并组数"
        return (
            f"已汇聚故障组数: {self.match_count} | "
            f"{primary_merge_label}: {primary_merge_count}"
        )

    def refresh_progress_extra_text(self, force=False):
        if self.process_progress is None:
            return
        self.process_progress.set_extra_text(self.build_progress_extra_text(), force=force)

    def _match_is_output_eligible(self, match):
        """故障组是否满足落盘规则要求。

        output_eligible_rules 为 None 时全部放行；否则要求 merged_rules（单个规则名
        列表，权威来源）与可落盘规则集合有交集。merged_rules 缺失时退回到 rule 字段，
        但合并组的 rule 形如 "a + b"，无法直接命中，故以 merged_rules 为准。
        """
        eligible = self.output_eligible_rules
        if eligible is None:
            return True
        merged_rules = match.get("merged_rules")
        if isinstance(merged_rules, list):
            for rule_name in merged_rules:
                if str(rule_name).strip() in eligible:
                    return True
            return False
        rule = match.get("rule")
        return isinstance(rule, str) and rule.strip() in eligible

    def write_matches(self, matches):
        with self.output_lock:
            fw = self._fw
            if fw is None:
                raise RuntimeError("output file is not initialized; call reset_output_file() first")
            output_lines = []
            written_count = 0
            for match in matches:
                # 落盘前过滤：不含可落盘规则的故障组直接跳过，不再构建输出对象。
                if not self._match_is_output_eligible(match):
                    continue
                enriched_match = build_jsonl_match_output(
                    match,
                    self.ne_graph_data,
                    site_to_ne_ids=self.site_to_ne_ids,
                    ne_link_info_cache=self.ne_link_info_cache,
                )
                # 故障模式过滤+增强需要增强后的 ne_info/group_info，故在构建之后处理。
                # process() 返回追加了模式备注/相关站点的记录；被过滤掉时返回 None。
                if self.fault_pattern_filter is not None:
                    enriched_match = self.fault_pattern_filter.process(enriched_match)
                    if enriched_match is None:
                        continue
                output_lines.append(_dumps_line(enriched_match))
                written_count += 1
            if output_lines:
                fw.writelines(output_lines)
                fw.flush()
            self.match_count += written_count
            self.refresh_progress_extra_text()
