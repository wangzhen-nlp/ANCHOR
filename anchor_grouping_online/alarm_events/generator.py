from anchor_grouping_online.alarm_events.identity import require_alarm_identity
from anchor_grouping_online.alarm_events.io import is_clear_alarm, parse_datetime_text


class AlarmGenerator:
    """将已过滤、已排序的告警数据惰性转换为对外告警字典。"""

    def __init__(self, alarm_data):
        self._alarm_data = alarm_data

    def __len__(self):
        return len(self._alarm_data)

    def __iter__(self):
        for item in self._alarm_data:
            yield generate_alarm(item)


def generate_alarm(item):
    if not isinstance(item, dict):
        raise ValueError("alarm data item must be a dict")

    alarm = item.get("alarm")
    if not isinstance(alarm, dict):
        raise ValueError("alarm data item is missing required alarm payload")

    event_id, occurrence_uuid = require_alarm_identity(item)
    generated_alarm = {
        "站点ID": str(item["site_id"]).strip(),
        "告警标题": str(item["alarm_title"]).strip(),
        "告警首次发生时间": alarm.get("告警首次发生时间", item["ts"]),
        "告警编码ID": compose_alarm_id(event_id, occurrence_uuid),
        "告警源": str(item.get("alarm_source", "") or "").strip(),
        "物理端口名称": str(alarm.get("物理端口名称", "") or "").strip(),
    }
    if is_clear_alarm(alarm):
        generated_alarm["是否清除"] = True
    return generated_alarm


def compose_alarm_id(event_id, occurrence_uuid):
    """将原事件 ID 和发生实例 UUID 合成算法唯一使用的 ID。"""
    return f"{event_id}::{occurrence_uuid}"


def to_matching_alarm(generated_alarm):
    """将生成器对外字段转为时序引擎的内部字段。"""
    if not isinstance(generated_alarm, dict):
        raise ValueError("generated alarm must be a dict")

    alarm_id = generated_alarm["告警编码ID"]
    if not isinstance(alarm_id, str) or not alarm_id.strip():
        raise ValueError("告警编码ID must be a non-empty str")

    event_time = generated_alarm["告警首次发生时间"]
    if isinstance(event_time, (int, float)):
        ts = float(event_time)
    else:
        ts = parse_datetime_text(event_time, "告警首次发生时间").timestamp()

    return {
        "site_id": str(generated_alarm["站点ID"]).strip(),
        "alarm_title": str(generated_alarm["告警标题"]).strip(),
        "ts": ts,
        "alarm_id": alarm_id.strip(),
        "alarm_source": str(generated_alarm["告警源"] or "").strip(),
        "physical_port_name": str(
            generated_alarm.get("物理端口名称", "") or ""
        ).strip(),
        "is_clear": bool(generated_alarm.get("是否清除", False)),
    }
