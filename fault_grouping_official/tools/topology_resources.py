from pathlib import Path


# 资源目录：本工具目录的上一级目录下的 resources（fault_grouping_official/resources）
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
RESOURCE_DIR_NAME = "resources"


def resource_path(name: str) -> str:
    return str(RESOURCE_DIR / name)


def resource_display(name: str) -> str:
    return f"{RESOURCE_DIR_NAME}/{name}"


NE_GRAPH_JSON = resource_path("ne_graph.json")
SITE_CHAINS_JSON = resource_path("site_chains.json")
SYS_LINK_DIR = resource_path("SYS_LINK_0306")
LINK_PEER_INDEX_JSON = resource_path("link_peer_index.json")
SYS_NE_DIR = resource_path("SYS_NE_0306")
SYS_SITE_DIR = resource_path("SYS_SITE_0306")


__all__ = [
    "RESOURCE_DIR",
    "RESOURCE_DIR_NAME",
    "resource_path",
    "resource_display",
    "NE_GRAPH_JSON",
    "SITE_CHAINS_JSON",
    "SYS_LINK_DIR",
    "LINK_PEER_INDEX_JSON",
    "SYS_NE_DIR",
    "SYS_SITE_DIR",
]
