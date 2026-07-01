from fault_grouping_official.alarm_events.io import is_clear_alarm


def process_alarm(engine, item):
    alarm = item["alarm"]
    return engine.process_event(
        node=item["site_id"],
        alarm_source=item.get("alarm_source", ""),
        alarm_type=item["alarm_title"],
        ts=item["ts"],
        event_id=alarm["告警编码ID"],
        occurrence_uuid=item["occurrence_uuid"],
        alarm_payload=alarm,
        is_clear=is_clear_alarm(alarm),
    )


def refresh_process_progress(process_progress, refresh_extra_text):
    refresh_extra_text()
    process_progress.update()
