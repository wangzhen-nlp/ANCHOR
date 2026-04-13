from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent
LEGACY_RESOURCE_DIR = RESOURCE_DIR.parent
LEGACY_TOPOLOGY_RESOURCE_DIR = RESOURCE_DIR.parent / "topology_resources"
RESOURCE_DIR_NAME = "ticket_resources"


def resource_path(name: str) -> str:
    preferred_path = RESOURCE_DIR / name
    if preferred_path.exists():
        return str(preferred_path)

    legacy_path = LEGACY_RESOURCE_DIR / name
    if legacy_path.exists():
        return str(legacy_path)

    legacy_topology_path = LEGACY_TOPOLOGY_RESOURCE_DIR / name
    if legacy_topology_path.exists():
        return str(legacy_topology_path)

    return str(preferred_path)


def resource_display(name: str) -> str:
    return f"{RESOURCE_DIR_NAME}/{name}"


DEFAULT_INCIDENT_TICKET_XLSX = resource_path("Incident Ticket_20260201-20260318.xlsx")


__all__ = [
    "RESOURCE_DIR",
    "LEGACY_RESOURCE_DIR",
    "LEGACY_TOPOLOGY_RESOURCE_DIR",
    "RESOURCE_DIR_NAME",
    "resource_path",
    "resource_display",
    "DEFAULT_INCIDENT_TICKET_XLSX",
]
