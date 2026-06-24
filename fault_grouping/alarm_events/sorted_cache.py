import json
import zipfile
from datetime import datetime

from fault_grouping.alarm_events.identity import ALARM_IDENTITY_SCHEME, require_alarm_identity


SORTED_ALARM_CACHE_TYPE = "fault_grouping.sorted_alarms.v3"
SORTED_ALARM_CACHE_MEMBER = "sorted_alarms.jsonl"


def _require_cached_alarm_identity(item):
    alarm = item.get("alarm") if isinstance(item, dict) else None
    eid = alarm.get("告警编码ID") if isinstance(alarm, dict) else None
    return require_alarm_identity({
        "eid": eid,
        "occurrence_uuid": item.get("occurrence_uuid") if isinstance(item, dict) else None,
    })


def build_sorted_alarm_cache_metadata(**kwargs):
    metadata = dict(kwargs)
    metadata["cache_type"] = SORTED_ALARM_CACHE_TYPE
    metadata["alarm_identity_scheme"] = ALARM_IDENTITY_SCHEME
    metadata["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return metadata


def is_sorted_alarm_cache_header(record):
    return (
        isinstance(record, dict)
        and record.get("cache_type") == SORTED_ALARM_CACHE_TYPE
        and record.get("alarm_identity_scheme") == ALARM_IDENTITY_SCHEME
    )


def consume_sorted_alarm_cache_header(record):
    """Return True for the current cache header and reject older cache schemas."""
    if is_sorted_alarm_cache_header(record):
        return True
    cache_type = record.get("cache_type") if isinstance(record, dict) else None
    if str(cache_type or "").startswith("fault_grouping.sorted_alarms."):
        raise ValueError(f"不支持的排序告警缓存格式: {cache_type}")
    return False


def is_sorted_alarm_cache_file(path):
    if str(path).lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                member_name = _find_sorted_alarm_cache_member(zf)
                if not member_name:
                    return False
                with zf.open(member_name, "r") as raw:
                    first_line = raw.readline().decode("utf-8").strip()
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError):
            return False

        if not first_line:
            return False
        try:
            record = json.loads(first_line)
        except json.JSONDecodeError:
            return False
        return is_sorted_alarm_cache_header(record)

    try:
        with open(path, "r", encoding="utf-8") as fr:
            first_line = fr.readline().strip()
    except OSError:
        return False

    if not first_line:
        return False

    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return False

    return is_sorted_alarm_cache_header(record)


def _find_sorted_alarm_cache_member(zf):
    if SORTED_ALARM_CACHE_MEMBER in zf.namelist():
        return SORTED_ALARM_CACHE_MEMBER

    candidates = [
        info.filename for info in zf.infolist()
        if not info.is_dir() and info.filename.lower().endswith((".jsonl", ".json"))
    ]
    return candidates[0] if len(candidates) == 1 else None


def _iter_sorted_alarm_cache_lines(path):
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            member_name = _find_sorted_alarm_cache_member(zf)
            if not member_name:
                raise ValueError(f"排序告警缓存 zip 中找不到唯一 JSONL 成员: {path}")
            with zf.open(member_name, "r") as raw:
                for line in raw:
                    yield line.decode("utf-8")
        return

    with open(path, "r", encoding="utf-8") as fr:
        yield from fr


def read_sorted_alarm_cache_header(path):
    line_iter = iter(_iter_sorted_alarm_cache_lines(path))
    try:
        first_line = next(line_iter).strip()
    except StopIteration:
        first_line = ""

    if not first_line:
        raise ValueError(f"排序告警缓存为空: {path}")

    metadata = json.loads(first_line)
    if not is_sorted_alarm_cache_header(metadata):
        raise ValueError(f"不是当前身份方案的有效排序告警缓存: {path}")
    return metadata


def iter_sorted_alarm_cache_items(path, show_progress=False):
    read_sorted_alarm_cache_header(path)
    line_iter = iter(_iter_sorted_alarm_cache_lines(path))
    next(line_iter, None)
    for idx, line in enumerate(line_iter, start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        _require_cached_alarm_identity(item)
        yield item
        if show_progress and idx % 100000 == 0:
            print(f"  已流式读取排序告警 {idx} 条...")


class SortedAlarmCacheStream:
    def __init__(self, path, metadata=None):
        self.path = path
        self.metadata = metadata or read_sorted_alarm_cache_header(path)
        self.alarm_count = int(self.metadata.get("alarm_count", 0))
        self._first_ts = None
        self._first_ts_loaded = False

    def __len__(self):
        return self.alarm_count

    def __iter__(self):
        return iter_sorted_alarm_cache_items(self.path)

    def first_ts(self):
        if not self._first_ts_loaded:
            self._first_ts = None
            for item in self:
                self._first_ts = item.get("ts")
                break
            self._first_ts_loaded = True
        return self._first_ts


def _write_sorted_alarm_cache_jsonl(stream, sorted_alarms, header):
    stream.write(json.dumps(header, ensure_ascii=False) + "\n")
    for item in sorted_alarms:
        _require_cached_alarm_identity(item)
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_sorted_alarm_cache(path, sorted_alarms, metadata=None):
    metadata = metadata or {}
    header = build_sorted_alarm_cache_metadata(**metadata)
    header["alarm_count"] = len(sorted_alarms)

    if str(path).lower().endswith(".zip"):
        zip_member = header.get("zip_member") or SORTED_ALARM_CACHE_MEMBER
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            with zf.open(zip_member, "w") as raw:
                raw.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
                for item in sorted_alarms:
                    _require_cached_alarm_identity(item)
                    raw.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8") as fw:
            _write_sorted_alarm_cache_jsonl(fw, sorted_alarms, header)

    return header


def load_sorted_alarm_cache(path, show_progress=False):
    alarms = []

    metadata = read_sorted_alarm_cache_header(path)
    for item in iter_sorted_alarm_cache_items(path, show_progress=show_progress):
        alarms.append(item)

    return metadata, alarms
