from anchor_grouping_online.alarm_events.identity import require_alarm_identity
from anchor_grouping_online.alarm_events.io import parse_datetime_text
from anchor_grouping_online.peer_index_keys import make_key

_PORT_VID_PREFIX = "portVid:"


def _normalize_ne_vid_for_generator(value) -> str:
    return str(value or "").strip().upper()


def _normalize_owner_vid_for_generator(value) -> str:
    return str(value or "").strip().upper()


def generate_alarm(item):
    if not isinstance(item, dict):
        raise ValueError("alarm data item must be a dict")

    alarm = item.get("alarm")
    if not isinstance(alarm, dict):
        raise ValueError("alarm data item is missing required alarm payload")

    event_id, occurrence_uuid = require_alarm_identity(item)
    physical_port_name = str(alarm.get("物理端口名称", "") or "").strip()
    ne_vid = _normalize_ne_vid_for_generator(item.get("alarm_source", ""))
    port_vid = (
        make_key(ne_vid, physical_port_name)
        if ne_vid and physical_port_name
        else ""
    )
    generated_alarm = {
        "alarmName": str(item["alarm_title"]).strip(),
        "firstOccurrence": alarm.get("告警首次发生时间", item["ts"]),
        "vid": compose_alarm_id(event_id, occurrence_uuid),
        "neVid": ne_vid,
        # ownerVid 暂以 site_id 占位；实际来源待定（可能换成其它字段，
        # 或作为 neVid 缺失时的资源兜底 ID），消费侧只把它当
        # alarm_source 的兜底值，不假设它是站点 ID。
        "ownerVid": _normalize_owner_vid_for_generator(item.get("site_id", "")),
        "extendedattr": f"{_PORT_VID_PREFIX}{port_vid}" if port_vid else "",
    }
    return generated_alarm


def compose_alarm_id(event_id, occurrence_uuid):
    """将原事件 ID 和发生实例 UUID 合成算法唯一使用的 ID。"""
    return f"{event_id}::{occurrence_uuid}"


def _extendedattr_value(extendedattr, prefix):
    for entry in str(extendedattr or "").split(";"):
        entry = entry.strip()
        if entry.startswith(prefix):
            return entry[len(prefix):].strip()
    return ""


def port_vid_from_extendedattr(extendedattr):
    """从 extendedattr 自由键值文本中解析全局端口 VID。"""
    return _extendedattr_value(extendedattr, _PORT_VID_PREFIX)


def to_matching_alarm(generated_alarm, ne_to_site):
    """将生成器对外字段转为时序引擎的内部字段。

    alarm_source 取 neVid，为空时退回 ownerVid；站点用 alarm_source 在
    ne_to_site（网元 ID -> 站点 ID）中反查得到，查不到时 site_id 为空串。
    全局端口 VID 从 extendedattr 的 portVid 条目解析。
    是否清除 为可选字段：generate_alarm 不再输出它（清除告警在流式
    入口按原始载荷过滤），调用方自带时仍按清除告警处理。
    """
    if not isinstance(generated_alarm, dict):
        raise ValueError("generated alarm must be a dict")

    alarm_id = generated_alarm["vid"]
    if not isinstance(alarm_id, str) or not alarm_id:
        raise ValueError("vid must be a non-empty str")

    event_time = generated_alarm["firstOccurrence"]
    if isinstance(event_time, (int, float)):
        ts = float(event_time)
    else:
        ts = parse_datetime_text(event_time, "firstOccurrence").timestamp()

    ne_vid = generated_alarm.get("neVid", "") or ""
    owner_vid = generated_alarm.get("ownerVid", "") or ""
    alarm_source = ne_vid or owner_vid
    extendedattr = generated_alarm.get("extendedattr")
    return {
        "site_id": ne_to_site.get(alarm_source, ""),
        "alarm_title": str(generated_alarm["alarmName"]).strip(),
        "ts": ts,
        "alarm_id": alarm_id,
        "alarm_source": alarm_source,
        "extendedattr": str(extendedattr or ""),
        "is_clear": bool(generated_alarm.get("是否清除", False)),
    }
