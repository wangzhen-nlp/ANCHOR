from anchor_grouping_online.alarm_events.identity import require_alarm_identity
from anchor_grouping_online.alarm_events.io import parse_datetime_text

_PORT_VID_PREFIX = "portVid:"


def generate_alarm(item):
    if not isinstance(item, dict):
        raise ValueError("alarm data item must be a dict")

    alarm = item.get("alarm")
    if not isinstance(alarm, dict):
        raise ValueError("alarm data item is missing required alarm payload")

    event_id, occurrence_uuid = require_alarm_identity(item)
    port_name = str(alarm.get("物理端口名称", "") or "").strip()
    generated_alarm = {
        "alarmName": str(item["alarm_title"]).strip(),
        "firstOccurrence": alarm.get("告警首次发生时间", item["ts"]),
        "vid": compose_alarm_id(event_id, occurrence_uuid),
        "neVid": str(item.get("alarm_source", "") or "").strip(),
        # ownerVid 暂以 site_id 占位；实际来源待定（可能换成其它字段，
        # 或作为 neVid 缺失时的资源兜底 ID），消费侧只把它当
        # alarm_source 的兜底值，不假设它是站点 ID。
        "ownerVid": str(item.get("site_id", "") or "").strip(),
        "extendedattr": f"{_PORT_VID_PREFIX}{port_name}" if port_name else "",
    }
    return generated_alarm


def compose_alarm_id(event_id, occurrence_uuid):
    """将原事件 ID 和发生实例 UUID 合成算法唯一使用的 ID。"""
    return f"{event_id}::{occurrence_uuid}"


def port_name_from_extendedattr(extendedattr):
    """从 extendedattr 自由键值文本中解析物理端口名称。

    extendedattr 是分号分隔的 key:value 条目文本，本生成器只写入
    "portVid:物理端口名称"；其他生成器可能写入其他条目（如
    "startAtTime:2026-06-24 03:52:36;connectAtTime:..."），
    没有 portVid 条目时返回空串。
    """
    for entry in str(extendedattr or "").split(";"):
        entry = entry.strip()
        if entry.startswith(_PORT_VID_PREFIX):
            return entry[len(_PORT_VID_PREFIX):].strip()
    return ""


def to_matching_alarm(generated_alarm, ne_to_site):
    """将生成器对外字段转为时序引擎的内部字段。

    alarm_source 取 neVid，为空时退回 ownerVid；站点用 alarm_source 在
    ne_to_site（网元 ID -> 站点 ID）中反查得到，查不到时 site_id 为空串。
    物理端口名称从 extendedattr 的 portVid 条目解析。
    是否清除 为可选字段：generate_alarm 不再输出它（清除告警在流式
    入口按原始载荷过滤），调用方自带时仍按清除告警处理。
    """
    if not isinstance(generated_alarm, dict):
        raise ValueError("generated alarm must be a dict")

    alarm_id = generated_alarm["vid"]
    if not isinstance(alarm_id, str) or not alarm_id.strip():
        raise ValueError("vid must be a non-empty str")

    event_time = generated_alarm["firstOccurrence"]
    if isinstance(event_time, (int, float)):
        ts = float(event_time)
    else:
        ts = parse_datetime_text(event_time, "firstOccurrence").timestamp()

    ne_vid = str(generated_alarm.get("neVid", "") or "").strip()
    owner_vid = str(generated_alarm.get("ownerVid", "") or "").strip()
    alarm_source = ne_vid or owner_vid
    return {
        "site_id": ne_to_site.get(alarm_source, ""),
        "alarm_title": str(generated_alarm["alarmName"]).strip(),
        "ts": ts,
        "alarm_id": alarm_id.strip(),
        "alarm_source": alarm_source,
        "physical_port_name": port_name_from_extendedattr(
            generated_alarm.get("extendedattr")
        ),
        "is_clear": bool(generated_alarm.get("是否清除", False)),
    }
