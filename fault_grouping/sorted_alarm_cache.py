import json
from datetime import datetime


SORTED_ALARM_CACHE_TYPE = "fault_grouping.sorted_alarms.v1"


def build_sorted_alarm_cache_metadata(**kwargs):
    metadata = dict(kwargs)
    metadata["cache_type"] = SORTED_ALARM_CACHE_TYPE
    metadata["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return metadata


def is_sorted_alarm_cache_file(path):
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

    return record.get("cache_type") == SORTED_ALARM_CACHE_TYPE


def write_sorted_alarm_cache(path, sorted_alarms, metadata=None):
    metadata = metadata or {}
    header = build_sorted_alarm_cache_metadata(**metadata)
    header["alarm_count"] = len(sorted_alarms)

    with open(path, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(header, ensure_ascii=False) + "\n")
        for item in sorted_alarms:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    return header


def load_sorted_alarm_cache(path, show_progress=False):
    metadata = None
    alarms = []

    with open(path, "r", encoding="utf-8") as fr:
        first_line = fr.readline().strip()
        if not first_line:
            raise ValueError(f"排序告警缓存为空: {path}")

        metadata = json.loads(first_line)
        if metadata.get("cache_type") != SORTED_ALARM_CACHE_TYPE:
            raise ValueError(f"不是有效的排序告警缓存: {path}")

        for idx, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            alarms.append(json.loads(line))
            if show_progress and idx % 100000 == 0:
                print(f"  已加载排序告警 {idx} 条...")

    return metadata, alarms
