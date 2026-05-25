from __future__ import annotations

from collections import Counter
import json


REGION_KEYS = (
    "region_id",
    "regionId",
    "regionId1",
    "region",
    "area_id",
    "area",
    "区域",
    "地市",
)


def normalize_text(value) -> str:
    return str(value or "").strip()


def parse_regions(value) -> tuple[str, ...]:
    """Parse one or more region labels from CLI text or a Python collection."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_parts = value.replace("，", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_parts = []
        for item in value:
            if isinstance(item, str):
                raw_parts.extend(item.replace("，", ",").split(","))
            else:
                raw_parts.append(item)
    else:
        raw_parts = [value]

    regions = []
    seen = set()
    for item in raw_parts:
        region = normalize_text(item)
        if not region or region in seen:
            continue
        seen.add(region)
        regions.append(region)
    return tuple(regions)


def get_region(record, *, default: str = "") -> str:
    if not isinstance(record, dict):
        return default
    for key in REGION_KEYS:
        region = normalize_text(record.get(key))
        if region:
            return region
    return default


def load_ne_graph(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def build_ne_region_map(ne_graph_data) -> dict[str, str]:
    if not isinstance(ne_graph_data, dict):
        return {}
    ne_regions = {}
    for ne_id, ne_info in ne_graph_data.items():
        ne_id = normalize_text(ne_id)
        region = get_region(ne_info)
        if ne_id and region:
            ne_regions[ne_id] = region
    return ne_regions


def _event_alarm_source(event) -> str:
    if not isinstance(event, dict):
        return ""
    alarm = event.get("alarm", {})
    if not isinstance(alarm, dict):
        alarm = {}
    return (
        normalize_text(event.get("alarm_source"))
        or normalize_text(event.get("告警源"))
        or normalize_text(alarm.get("告警源"))
    )


def event_region(event, ne_region_map=None) -> str:
    ne_region_map = ne_region_map or {}
    alarm_source = _event_alarm_source(event)
    if alarm_source:
        region = ne_region_map.get(alarm_source)
        if region:
            return region
    region = get_region(event)
    if region:
        return region
    alarm = event.get("alarm", {}) if isinstance(event, dict) else {}
    return get_region(alarm)


def filter_alarm_events_by_regions(sorted_alarm_events, regions, *, ne_graph_data=None):
    selected_regions = frozenset(parse_regions(regions))
    events = list(sorted_alarm_events)
    stats = {
        "enabled": bool(selected_regions),
        "regions": sorted(selected_regions),
        "input_event_count": len(events),
        "kept_event_count": len(events),
        "dropped_event_count": 0,
        "ne_graph_device_count": len(ne_graph_data) if isinstance(ne_graph_data, dict) else 0,
        "allowed_device_count": 0,
        "unknown_region_event_count": 0,
        "kept_region_counts": {},
        "dropped_region_counts": {},
    }
    if not selected_regions:
        return events, stats

    ne_region_map = build_ne_region_map(ne_graph_data)
    allowed_devices = {
        ne_id
        for ne_id, region in ne_region_map.items()
        if region in selected_regions
    }
    stats["allowed_device_count"] = len(allowed_devices)

    kept_events = []
    kept_region_counts = Counter()
    dropped_region_counts = Counter()
    unknown_region_event_count = 0
    for event in events:
        region = event_region(event, ne_region_map)
        if not region:
            unknown_region_event_count += 1
            dropped_region_counts["<unknown>"] += 1
            continue
        if region in selected_regions:
            kept_events.append(event)
            kept_region_counts[region] += 1
        else:
            dropped_region_counts[region] += 1

    stats.update(
        {
            "kept_event_count": len(kept_events),
            "dropped_event_count": len(events) - len(kept_events),
            "unknown_region_event_count": unknown_region_event_count,
            "kept_region_counts": dict(sorted(kept_region_counts.items())),
            "dropped_region_counts": dict(sorted(dropped_region_counts.items())),
        }
    )
    return kept_events, stats


def filter_ne_graph_by_regions(ne_graph_data, regions):
    selected_regions = frozenset(parse_regions(regions))
    if not isinstance(ne_graph_data, dict):
        return {}, {
            "enabled": bool(selected_regions),
            "regions": sorted(selected_regions),
            "original_device_count": 0,
            "allowed_device_count": 0,
            "dropped_device_count": 0,
            "original_link_count": 0,
            "kept_link_count": 0,
            "dropped_link_count": 0,
        }

    original_device_count = len(ne_graph_data)
    if not selected_regions:
        original_link_count = sum(
            len(info.get("link", {}))
            for info in ne_graph_data.values()
            if isinstance(info, dict) and isinstance(info.get("link", {}), dict)
        )
        return ne_graph_data, {
            "enabled": False,
            "regions": [],
            "original_device_count": original_device_count,
            "allowed_device_count": original_device_count,
            "dropped_device_count": 0,
            "original_link_count": original_link_count,
            "kept_link_count": original_link_count,
            "dropped_link_count": 0,
        }

    allowed_devices = {
        normalize_text(ne_id)
        for ne_id, info in ne_graph_data.items()
        if get_region(info) in selected_regions
    }
    filtered_graph = {}
    original_link_count = sum(
        len(info.get("link", {}))
        for info in ne_graph_data.values()
        if isinstance(info, dict) and isinstance(info.get("link", {}), dict)
    )
    kept_link_count = 0
    for raw_ne_id, info in ne_graph_data.items():
        ne_id = normalize_text(raw_ne_id)
        if ne_id not in allowed_devices:
            continue
        if not isinstance(info, dict):
            filtered_graph[raw_ne_id] = info
            continue
        filtered_info = dict(info)
        links = info.get("link", {})
        if isinstance(links, dict):
            filtered_links = {}
            for raw_target_ne, link_meta in links.items():
                if normalize_text(raw_target_ne) in allowed_devices:
                    filtered_links[raw_target_ne] = link_meta
                    kept_link_count += 1
            filtered_info["link"] = filtered_links
        filtered_graph[raw_ne_id] = filtered_info

    return filtered_graph, {
        "enabled": True,
        "regions": sorted(selected_regions),
        "original_device_count": original_device_count,
        "allowed_device_count": len(allowed_devices),
        "dropped_device_count": original_device_count - len(allowed_devices),
        "original_link_count": original_link_count,
        "kept_link_count": kept_link_count,
        "dropped_link_count": original_link_count - kept_link_count,
    }
