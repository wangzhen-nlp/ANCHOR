"""Device-domain enrichment for alarm events.

The MHP event type is defined by ``type_fields`` (see
:func:`alarm_flow_isahp.sequences.event_type_label`). Most fields read straight
off the raw event dict, but ``device_domain`` is a property of the device's NE
in the topology graph, not of the alarm record. This module stamps that field
onto each event once, so the pure ``event_type_label`` machinery can treat it
like any other field.

The SAME annotation must run on both the training and inference event streams
(the model persists ``type_fields``, and inference rebuilds the type label from
the raw event) — otherwise inference labels would carry ``<empty>`` for the
domain and never match the trained vocabulary.
"""

from ne_link_learning.core import normalize_domain_bucket, normalize_text


DEVICE_DOMAIN_FIELD = "device_domain"
MODELED_DOMAINS = frozenset({"RAN", "TRANSMISSION", "DATA"})
# Backwards-compatible descriptive alias. Keep a single source of truth so
# observed-event filtering and missing-event enumeration cannot drift apart.
SUPPORTED_DEVICE_DOMAINS = MODELED_DOMAINS


def build_ne_domain_bucket_map(ne_graph_data):
    """``ne_id -> domain_bucket`` (RAN / TRANSMISSION / DATA / OTHER / ...).

    Uses the SAME ``normalize_domain_bucket`` derivation that the feature-mode
    μ attributes use (see :class:`ne_link_learning.core.NodeInfo`), so the
    domain bucketing is identical across the codebase.
    """
    out = {}
    for ne_id, info in (ne_graph_data or {}).items():
        if not isinstance(info, dict):
            continue
        normalized_ne_id = normalize_text(ne_id)
        if not normalized_ne_id:
            continue
        raw_domain = info.get("domain") or info.get("Domain") or info.get("DOMAIN") or ""
        out[normalized_ne_id] = normalize_domain_bucket(raw_domain)
    return out


def annotate_device_domain(events, ne_graph_data):
    """Stamp ``event['device_domain']`` from the event's ``alarm_source`` NE.

    Mutates ``events`` in place. Events whose NE is unknown get ``""`` (which
    :func:`event_type_label` renders as ``<empty>``, consistent on both sides).
    Returns the number of events annotated.
    """
    domain_map = build_ne_domain_bucket_map(ne_graph_data)
    count = 0
    for event in events:
        ne_id = normalize_text(event.get("alarm_source", ""))
        event[DEVICE_DOMAIN_FIELD] = domain_map.get(ne_id, "")
        count += 1
    return count


def filter_and_annotate_device_domain(events, ne_graph_data):
    """Keep only events from the supported modeled device domains.

    The site×domain×alarm_type model intentionally covers only RAN,
    TRANSMISSION and DATA devices. Devices mapped to OTHER/MISSING, and alarm
    sources absent from the NE graph, are excluded rather than turned into
    additional event types.

    Returns ``(filtered_events, stats)``. Event dictionaries are annotated in
    place, while the input iterable itself is not mutated.
    """
    domain_map = build_ne_domain_bucket_map(ne_graph_data)
    input_events = list(events)
    kept = []
    dropped_by_domain = {}
    for event in input_events:
        ne_id = normalize_text(event.get("alarm_source", ""))
        domain = domain_map.get(ne_id, "")
        event[DEVICE_DOMAIN_FIELD] = domain
        if domain in SUPPORTED_DEVICE_DOMAINS:
            kept.append(event)
            continue
        reason = domain or "UNKNOWN_DEVICE"
        dropped_by_domain[reason] = dropped_by_domain.get(reason, 0) + 1
    return kept, {
        "enabled": True,
        "supported_domains": sorted(SUPPORTED_DEVICE_DOMAINS),
        "input_event_count": len(input_events),
        "kept_event_count": len(kept),
        "dropped_event_count": len(input_events) - len(kept),
        "dropped_by_domain": dropped_by_domain,
    }
