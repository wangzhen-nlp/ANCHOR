from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent
LEGACY_RESOURCE_DIR = RESOURCE_DIR.parent
RESOURCE_DIR_NAME = "topology_resources"


def resource_path(name: str) -> str:
    preferred_path = RESOURCE_DIR / name
    if preferred_path.exists():
        return str(preferred_path)

    legacy_path = LEGACY_RESOURCE_DIR / name
    if legacy_path.exists():
        return str(legacy_path)

    return str(preferred_path)


def resource_display(name: str) -> str:
    return f"{RESOURCE_DIR_NAME}/{name}"


NE_GRAPH_JSON = resource_path("ne_graph.json")
SITE_GRAPH_JSON = resource_path("site_graph.json")
SITE_GRAPH_BY_NE_JSON = resource_path("site_graph_by_ne.json")
SITE_DEVICE_COUNTS_JSON = resource_path("site_device_counts.json")
SYS_LINK_JSONL = resource_path("sys_link_1231.jsonl")
SYS_NE_DIR = resource_path("SYS_NE_0306")
SYS_SITE_DIR = resource_path("SYS_SITE_0306")


__all__ = [
    "RESOURCE_DIR",
    "LEGACY_RESOURCE_DIR",
    "RESOURCE_DIR_NAME",
    "resource_path",
    "resource_display",
    "NE_GRAPH_JSON",
    "SITE_GRAPH_JSON",
    "SITE_GRAPH_BY_NE_JSON",
    "SITE_DEVICE_COUNTS_JSON",
    "SYS_LINK_JSONL",
    "SYS_NE_DIR",
    "SYS_SITE_DIR",
]
