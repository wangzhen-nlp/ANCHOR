import uuid
from pathlib import Path


def new_occurrence_uuid():
    return str(uuid.uuid4())


def deterministic_occurrence_uuid(namespace, ordinal):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{ordinal}"))


def input_occurrence_uuid(source, ordinal):
    source_id = str(Path(source).expanduser().resolve())
    return deterministic_occurrence_uuid(f"alarm-input:{source_id}", int(ordinal))


def require_occurrence_uuid(record):
    value = record.get("occurrence_uuid") if isinstance(record, dict) else None
    if value in (None, ""):
        raise ValueError("alarm occurrence is missing required occurrence_uuid")
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid occurrence_uuid: {value!r}") from exc


def require_eid(record):
    if not isinstance(record, dict):
        raise ValueError("alarm occurrence must be a dict")
    candidates = [record]
    if isinstance(record.get("alarm"), dict):
        candidates.append(record["alarm"])
    for candidate in candidates:
        for field_name in ("eid", "alarm_id", "告警编码ID", "event_id"):
            value = candidate.get(field_name)
            if value not in (None, ""):
                return str(value)
    raise ValueError("alarm occurrence is missing required eid")


def require_alarm_identity(record):
    return require_eid(record), require_occurrence_uuid(record)
