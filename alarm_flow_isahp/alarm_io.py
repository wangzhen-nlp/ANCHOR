from fault_grouping.alarm_events.sorted_cache import (
    is_sorted_alarm_cache_file,
    load_sorted_alarm_cache,
)
from fault_grouping.tools.prepare_sorted_alarms import build_sorted_alarms


def load_ordered_alarm_events(
    alarm_input,
    *,
    topo_path,
    ne_graph_path,
    start_time=None,
    end_time=None,
    clear_delay_sec=0.0,
):
    """Load alarms with the same filtering and ordering rules as match_rules."""
    if is_sorted_alarm_cache_file(alarm_input):
        metadata, events = load_sorted_alarm_cache(alarm_input, show_progress=True)
        return events, metadata

    return build_sorted_alarms(
        alarm_input,
        topo_path=topo_path,
        ne_graph_path=ne_graph_path,
        start_time=start_time,
        end_time=end_time,
        clear_delay_sec=clear_delay_sec,
    )
