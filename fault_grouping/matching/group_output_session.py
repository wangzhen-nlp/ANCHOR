import json
import threading

from dataclasses import dataclass, field

from fault_grouping.matching.group_output_builder import build_jsonl_match_output
from fault_grouping.matching.reports import generate_incident_report
from fault_grouping.temporal_engine.engine import TemporalGraphEngine


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
    site_graph_data: dict
    alarm_metadata_index: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
    match_count: int = 0
    process_progress: object = None
    output_lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_output_file(self):
        with open(self.output_path, 'w', encoding='utf-8'):
            pass

    def build_progress_extra_text(self):
        merge_stats = self.engine.get_batch_merge_stats_snapshot().get("total", {})
        primary_merge_count = (
            merge_stats.get('alarm_overlap_merge_group_count', 0)
            if self.args.use_alarm_period_cache
            else merge_stats.get('eid_merge_group_count', 0)
        )
        primary_merge_label = "告警时段合并组数" if self.args.use_alarm_period_cache else "eid合并组数"
        return (
            f"已汇聚故障组数: {self.match_count} | "
            f"{primary_merge_label}: {primary_merge_count} | "
            f"hop合并组数: {merge_stats.get('hop_merge_group_count', 0)} | "
            f"距离合并组数: {merge_stats.get('distance_merge_group_count', 0)}"
        )

    def refresh_progress_extra_text(self, force=False):
        if self.process_progress is None:
            return
        self.process_progress.set_extra_text(self.build_progress_extra_text(), force=force)

    def write_matches(self, matches):
        with self.output_lock:
            # 二进制 append：与 _dumps_line 的 bytes 输出对接，避免 UTF-8 编码两次。
            with open(self.output_path, 'ab') as fw:
                output_lines = []
                for match in matches:
                    if self.args.verbose_groups:
                        generate_incident_report(match)
                    enriched_match = build_jsonl_match_output(
                        match,
                        self.ne_graph_data,
                        self.site_graph_data,
                        self.alarm_metadata_index,
                        site_to_ne_ids=self.site_to_ne_ids,
                        ne_link_info_cache=self.ne_link_info_cache,
                        compact_output=self.args.compact_output,
                        include_eid_list=self.args.use_alarm_period_cache,
                    )
                    output_lines.append(_dumps_line(enriched_match))
                fw.writelines(output_lines)
            self.match_count += len(matches)
            self.refresh_progress_extra_text()
