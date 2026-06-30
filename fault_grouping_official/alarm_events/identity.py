import json
import uuid


ALARM_CONTENT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "fault-grouping:alarm-content:v1",
)
ALARM_IDENTITY_SCHEME = "eid+canonical-json-uuid5:v1"
_IDENTITY_INTERNAL_FIELDS = frozenset({"occurrence_uuid"})


def alarm_content_uuid(record):
    if not isinstance(record, dict):
        raise ValueError("alarm record must be a dict")
    identity_record = {
        key: value
        for key, value in record.items()
        if key not in _IDENTITY_INTERNAL_FIELDS
    }
    try:
        canonical_json = json.dumps(
            identity_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("alarm record cannot be serialized as canonical JSON") from exc
    return str(uuid.uuid5(ALARM_CONTENT_NAMESPACE, canonical_json))


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
        for field_name in ("eid", "告警编码ID"):
            value = candidate.get(field_name)
            if value not in (None, ""):
                return str(value)
    raise ValueError("alarm occurrence is missing required eid")


def require_alarm_identity(record):
    return require_eid(record), require_occurrence_uuid(record)
