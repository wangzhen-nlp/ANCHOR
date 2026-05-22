import re

from collections import Counter
from datetime import datetime
from itertools import count

from alarm_cascade_dhp.types import AlarmEvent


_TITLE_PIECES = re.compile(r"[A-Za-z0-9_.:/-]+|[\u4e00-\u9fff]+")


def _text(value):
    return str(value or "").strip()


def _first_text(mapping, *names):
    for name in names:
        value = _text(mapping.get(name))
        if value:
            return value
    return ""


def _is_clear_alarm(raw_alarm):
    value = raw_alarm.get("清除告警", raw_alarm.get("is_clear", ""))
    return _text(value).lower() in {"是", "yes", "true", "1", "y"}


def _parse_timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = _text(value)
    if not text:
        raise ValueError("alarm timestamp is empty")
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return datetime.fromisoformat(text.replace("T", " ")).timestamp()


class AlarmFeatureBuilder:
    """Convert alarm records or match_rules stream items to AlarmEvent."""

    _FIELD_GROUPS = (
        ("alarm_code", ("告警码", "告警编码", "标准告警标识", "alarm_code")),
        ("severity", ("告警级别", "告警等级", "级别", "severity")),
        ("network", ("网络专业", "告警源专业", "专业", "network_domain")),
        ("network_type", ("网络类型", "network_type")),
        ("device_type", ("设备类型", "设备类型名称", "device_type")),
        ("manufacturer", ("设备厂家名称", "厂家", "manufacturer")),
        ("resource", ("告警资源类型", "资源类型", "resource_type")),
        ("resource_id", ("告警资源标识", "资源标识", "resource_id")),
    )

    def __init__(self, topology=None, topology_context_hops=1, topology_context_limit=16):
        self.topology = topology
        self.topology_context_hops = topology_context_hops
        self.topology_context_limit = topology_context_limit
        self._fallback_ids = count(1)

    def from_match_rules_item(self, item):
        raw = dict(item.get("alarm") or {})
        title = _text(item.get("alarm_title")) or _first_text(raw, "告警标题", "alarm_title")
        source = _text(item.get("alarm_source")) or _first_text(raw, "告警源", "alarm_source")
        site_id = _text(item.get("site_id")) or _first_text(raw, "站点ID", "site_id")
        ts = item.get("ts", raw.get("ts", ""))
        return self._build_event(raw, ts, title=title, source=source, site_id=site_id)

    def from_alarm_record(self, raw_record):
        raw = dict(raw_record or {})
        title = _first_text(raw, "告警标题", "alarm_title", "title")
        source = _first_text(raw, "告警源", "alarm_source", "source")
        site_id = _first_text(raw, "站点ID", "site_id", "site")
        ts_value = (
            raw.get("ts")
            or raw.get("timestamp")
            or raw.get("告警首次发生时间")
            or raw.get("告警发生时间")
        )
        return self._build_event(raw, ts_value, title=title, source=source, site_id=site_id)

    def _build_event(self, raw, ts_value, title, source, site_id):
        if self.topology is not None:
            site_id = self.topology.resolve_site(site_id, source)
        ts = _parse_timestamp(ts_value)
        event_id = _first_text(raw, "告警编码ID", "event_id", "alarm_id")
        if not event_id:
            event_id = f"alarm-event-{next(self._fallback_ids)}"
        domain = _first_text(raw, "网络专业", "告警源专业", "专业", "network_domain")
        if self.topology is not None:
            domain = self.topology.resolve_domain(source, domain)
        event_key = self._event_key(raw, title, source, site_id)
        return AlarmEvent(
            event_id=event_id,
            ts=ts,
            alarm_title=title,
            alarm_source=source,
            site_id=site_id,
            feature_counts=self._feature_counts(raw, title, source, site_id),
            event_key=event_key,
            is_clear=_is_clear_alarm(raw),
            device_domain=domain,
            raw=raw,
        )

    def _event_key(self, raw, title, source, site_id):
        code = _first_text(raw, "告警码", "告警编码", "标准告警标识", "alarm_code")
        resource_id = _first_text(raw, "告警资源标识", "资源标识", "resource_id")
        pieces = (source, site_id, code, title, resource_id)
        return "|".join(_text(piece) for piece in pieces)

    def _feature_counts(self, raw, title, source, site_id):
        counts = Counter()
        if title:
            counts[f"title:{title}"] += 2
            for piece in _TITLE_PIECES.findall(title):
                normalized = piece.lower()
                if normalized and normalized != title.lower():
                    counts[f"title_piece:{normalized}"] += 1
        if source:
            counts[f"device:{source}"] += 1
        if site_id:
            counts[f"site:{site_id}"] += 1

        for label, field_names in self._FIELD_GROUPS:
            value = _first_text(raw, *field_names)
            if value:
                counts[f"{label}:{value}"] += 1

        if self.topology is not None:
            for token in self.topology.content_context_tokens(
                site_id,
                source,
                max_hops=self.topology_context_hops,
                limit=self.topology_context_limit,
            ):
                counts[token] += 1
        return counts
