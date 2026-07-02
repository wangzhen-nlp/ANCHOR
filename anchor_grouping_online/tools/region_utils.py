from __future__ import annotations


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
    """Parse region labels from CLI text or a Python collection."""
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


def allowed_devices_for_regions(ne_graph_data, regions) -> set[str]:
    selected_regions = frozenset(parse_regions(regions))
    if not selected_regions:
        return set()
    return {
        ne_id
        for ne_id, region in build_ne_region_map(ne_graph_data).items()
        if region in selected_regions
    }


__all__ = [
    "REGION_KEYS",
    "normalize_text",
    "parse_regions",
    "get_region",
    "build_ne_region_map",
    "allowed_devices_for_regions",
]
