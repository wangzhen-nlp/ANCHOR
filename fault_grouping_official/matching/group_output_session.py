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
    alarm_metadata_index: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict
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

    def write_matches(self, matches):
        with self.output_lock:
            fw = self._fw
            if fw is None:
                raise RuntimeError("output file is not initialized; call reset_output_file() first")
            output_lines = []
            for match in matches:
                enriched_match = build_jsonl_match_output(
                    match,
                    self.ne_graph_data,
                    self.alarm_metadata_index,
                    site_to_ne_ids=self.site_to_ne_ids,
                    ne_link_info_cache=self.ne_link_info_cache,
                )
                output_lines.append(_dumps_line(enriched_match))
            fw.writelines(output_lines)
            fw.flush()
            self.match_count += len(matches)
            self.refresh_progress_extra_text()
