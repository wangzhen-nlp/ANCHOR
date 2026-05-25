from fault_grouping.alarm_events.sorted_cache import (
    is_sorted_alarm_cache_file,
    load_sorted_alarm_cache,
)
from fault_grouping.alarm_events.io import count_alarm_event_types
from fault_grouping.tools.prepare_sorted_alarms import build_sorted_alarms
from topology_tools.region_utils import filter_alarm_events_by_regions, load_ne_graph, parse_regions


def load_ordered_alarm_events(
    alarm_input,
    *,
    topo_path,
    ne_graph_path,
    start_time=None,
    end_time=None,
    clear_delay_sec=0.0,
    regions=None,
):
    """Load alarms with the same filtering and ordering rules as match_rules."""
    selected_regions = parse_regions(regions)
    if is_sorted_alarm_cache_file(alarm_input):
        metadata, events = load_sorted_alarm_cache(alarm_input, show_progress=True)
        if selected_regions:
            ne_graph_data = load_ne_graph(ne_graph_path)
            events, region_filter_stats = filter_alarm_events_by_regions(
                events,
                selected_regions,
                ne_graph_data=ne_graph_data,
            )
            normal_alarm_count, clear_alarm_count = count_alarm_event_types(events)
            metadata = dict(metadata or {})
            metadata["region_filter"] = {
                "stage": "sorted_cache",
                **region_filter_stats,
                "cached_normal_alarm_count": normal_alarm_count,
                "cached_clear_alarm_count": clear_alarm_count,
            }
            metadata["cached_normal_alarm_count"] = normal_alarm_count
            metadata["cached_clear_alarm_count"] = clear_alarm_count
        return events, metadata

    return build_sorted_alarms(
        alarm_input,
        topo_path=topo_path,
        ne_graph_path=ne_graph_path,
        start_time=start_time,
        end_time=end_time,
        clear_delay_sec=clear_delay_sec,
        regions=selected_regions,
    )
