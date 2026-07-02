from pathlib import Path


# 资源目录：本工具目录的上一级目录下的 resources（anchor_grouping_online/resources）
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
RESOURCE_DIR_NAME = "resources"


def resource_path(name: str) -> str:
    return str(RESOURCE_DIR / name)


def resource_display(name: str) -> str:
    return f"{RESOURCE_DIR_NAME}/{name}"


SYS_LINK_DIR = resource_path("SYS_LINK_20260525")
SYS_NE_DIR = resource_path("SYS_NE_20260525")
SYS_SITE_DIR = resource_path("SYS_SITE_20260525")
RESOURCE_BUFFER_JSONL = resource_path("resource_buffer.jsonl")


__all__ = [
    "RESOURCE_DIR",
    "RESOURCE_DIR_NAME",
    "resource_path",
    "resource_display",
    "SYS_LINK_DIR",
    "SYS_NE_DIR",
    "SYS_SITE_DIR",
    "RESOURCE_BUFFER_JSONL",
]
