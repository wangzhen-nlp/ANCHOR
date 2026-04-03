import json
from collections import defaultdict


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null", "undefined"}:
        return ""
    return text


def dedupe_alarm_records(records):
    seen = set()
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = (
            normalize_text(record.get("告警编码ID", "")),
            normalize_text(record.get("故障组ID", "")),
            normalize_text(record.get("来源故障组UUID", "")),
            normalize_text(record.get("工单号", "")),
            normalize_text(record.get("站点ID", "")),
            normalize_text(record.get("关联站点ID", "")),
            normalize_text(record.get("告警源", "")),
            normalize_text(record.get("告警标题", "")),
            normalize_text(record.get("告警首次发生时间", "")),
            normalize_text(record.get("告警最后发生时间", "")),
            normalize_text(record.get("告警清除时间", "")),
            normalize_text(record.get("node", "")),
            normalize_text(record.get("matched_role", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def load_upper_bound_index(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("召回率上限结果 JSON 顶层必须是对象")

    ticket_index = {}
    for item in data.get("details", []):
        if not isinstance(item, dict):
            continue

        ticket_id = normalize_text(item.get("ticket_id", ""))
        if not ticket_id:
            continue

        ticket_site_count = int(item.get("ticket_site_count", 0) or 0)
        associated_site_count = int(item.get("associated_site_count", 0) or 0)

        merged_site_evidence = defaultdict(list)
        evidence = item.get("evidence", {})
        if isinstance(evidence, dict):
            for bucket_name in ("direct_site_alarms", "inferred_site_alarms"):
                bucket = evidence.get(bucket_name, {})
                if not isinstance(bucket, dict):
                    continue
                for site_id, alarms in bucket.items():
                    normalized_site_id = normalize_text(site_id)
                    if not normalized_site_id or not isinstance(alarms, list):
                        continue
                    merged_site_evidence[normalized_site_id].extend(
                        alarm for alarm in alarms if isinstance(alarm, dict)
                    )

        ticket_index[ticket_id] = {
            "ticket_site_count": ticket_site_count,
            "associated_site_count": associated_site_count,
            "fully_associable": ticket_site_count > 0 and associated_site_count == ticket_site_count,
            "site_evidence": {
                site_id: dedupe_alarm_records(alarms)
                for site_id, alarms in sorted(merged_site_evidence.items())
            },
        }

    return ticket_index


def build_site_alarm_map_for_sites(site_alarm_map, site_ids):
    result = {}
    for site_id in sorted(site_ids):
        result[site_id] = dedupe_alarm_records(site_alarm_map.get(site_id, []))
    return result
