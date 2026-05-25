import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alarm_tools.alarm_types import CRITICAL_ALARMS
from fault_grouping.alarm_events.io import (
    is_clear_alarm,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
)
from fault_grouping.site_topology import (
    build_site_topology_from_ne_graph,
)
from fault_grouping.alarm_events.sorted_cache import write_sorted_alarm_cache
from topology_resources import (
    NE_GRAPH_JSON,
    SITE_GRAPH_BY_NE_JSON,
    resource_display,
)
from topology_tools.region_utils import allowed_devices_for_regions, load_ne_graph, parse_regions


def _load_valid_sites_and_ne_mapping(topo_path, ne_graph_path):
    ne_graph_data = load_ne_graph(ne_graph_path)
    if topo_path:
        topo_downstream_map = json.load(open(topo_path, "r", encoding="utf-8"))
        valid_sites = set(topo_downstream_map.keys())
        for _, connected_sites in topo_downstream_map.items():
            if isinstance(connected_sites, list):
                valid_sites.update(connected_sites)
            elif isinstance(connected_sites, dict):
                valid_sites.update(connected_sites.keys())
    else:
        _topo_downstream_map, valid_sites = build_site_topology_from_ne_graph(ne_graph_data)

    ne_to_site = {
        ne_id: str(ne_info.get("site_id", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("site_id", "")).strip()
    }
    return valid_sites, ne_to_site, ne_graph_data


def build_sorted_alarms(
    alarm_input,
    *,
    topo_path=SITE_GRAPH_BY_NE_JSON,
    ne_graph_path=NE_GRAPH_JSON,
    start_time=None,
    end_time=None,
    clear_delay_sec=0.0,
    regions=None,
):
    start_ts = parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time 不能晚于 end_time")

    selected_regions = parse_regions(regions)
    valid_sites, ne_to_site, ne_graph_data = _load_valid_sites_and_ne_mapping(topo_path, ne_graph_path)
    allowed_alarm_sources = None
    region_filter_stats = {
        "stage": "raw_input",
        "enabled": bool(selected_regions),
        "regions": sorted(selected_regions),
        "ne_graph_device_count": len(ne_graph_data) if isinstance(ne_graph_data, dict) else 0,
        "allowed_device_count": 0,
        "raw_checked_alarm_count": 0,
        "raw_kept_alarm_count": 0,
        "raw_dropped_alarm_count": 0,
    }
    if selected_regions:
        allowed_alarm_sources = allowed_devices_for_regions(ne_graph_data, selected_regions)
        region_filter_stats["allowed_device_count"] = len(allowed_alarm_sources)
    processed_count, valid_alarms, normal_alarm_count, clear_alarm_count = load_valid_alarms(
        alarm_input,
        CRITICAL_ALARMS,
        valid_sites,
        ne_to_site,
        start_ts=start_ts,
        end_ts=end_ts,
        clear_delay_sec=clear_delay_sec,
        allowed_alarm_sources=allowed_alarm_sources,
        region_filter_stats=region_filter_stats,
    )
    region_filter_stats["pre_sort_event_count"] = len(valid_alarms)
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)

    cached_normal_alarm_count = sum(
        1 for item in valid_alarms if not is_clear_alarm(item.get("alarm", {}))
    )
    cached_clear_alarm_count = len(valid_alarms) - cached_normal_alarm_count
    metadata = {
        "source_alarms": os.path.abspath(alarm_input),
        "topo": os.path.abspath(topo_path) if topo_path else "",
        "ne_graph": os.path.abspath(ne_graph_path),
        "start_time": start_time or "",
        "end_time": end_time or "",
        "clear_delay_sec": float(clear_delay_sec),
        "processed_count": processed_count,
        "normal_alarm_count": normal_alarm_count,
        "clear_alarm_count": clear_alarm_count,
        "cached_normal_alarm_count": cached_normal_alarm_count,
        "cached_clear_alarm_count": cached_clear_alarm_count,
        "valid_site_count": len(valid_sites),
        "ne_to_site_count": len(ne_to_site),
        "valid_alarm_title_count": len(CRITICAL_ALARMS),
    }
    if selected_regions:
        region_filter_stats.update(
            {
                "input_event_count": region_filter_stats["pre_sort_event_count"],
                "kept_event_count": len(valid_alarms),
                "dropped_event_count": (
                    region_filter_stats["pre_sort_event_count"] - len(valid_alarms)
                ),
                "cached_normal_alarm_count": cached_normal_alarm_count,
                "cached_clear_alarm_count": cached_clear_alarm_count,
            }
        )
        metadata["region_filter"] = region_filter_stats
    return valid_alarms, metadata


def main():
    parser = ArgumentParser(description="预处理 match_rules.py 输入，生成已排序告警缓存(JSONL/ZIP，包含清除告警)")
    parser.add_argument("alarms", help="原始告警输入，支持 jsonl/csv/zip/目录，与 match_rules.py 一致")
    parser.add_argument("output", help="排序告警缓存输出；后缀为 .zip 时写压缩包，否则写 JSONL")
    parser.add_argument(
        "--topo",
        type=str,
        default=SITE_GRAPH_BY_NE_JSON,
        help=(
            f"站点拓扑文件，默认: {resource_display('site_graph_by_ne.json')}；"
            "若传空值则退回为基于 ne_graph.json 原始连边自动构建"
        ),
    )
    parser.add_argument(
        "--ne-graph",
        type=str,
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument("--start_time", type=str, help="仅处理告警首次发生时间 >= 该时间")
    parser.add_argument("--end_time", type=str, help="仅处理告警首次发生时间 <= 该时间")
    parser.add_argument(
        "--clear-delay-sec",
        type=float,
        default=0.0,
        help="清除告警最小延迟时间，清除生效时间=max(clear_delay_sec, 清除时间-发生时间)+发生时间",
    )
    parser.add_argument(
        "--regions",
        "--region",
        dest="regions",
        action="append",
        default=None,
        help="仅保留这些 region 下设备的告警；可重复传入或用逗号分隔",
    )
    args = parser.parse_args()

    start_time = time.time()
    valid_alarms, metadata = build_sorted_alarms(
        args.alarms,
        topo_path=args.topo,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time,
        end_time=args.end_time,
        clear_delay_sec=args.clear_delay_sec,
        regions=args.regions,
    )
    header = write_sorted_alarm_cache(args.output, valid_alarms, metadata)
    elapsed = time.time() - start_time
    print(
        f"✅ 排序告警缓存已写入: {args.output}\n"
        f"   缓存告警数: {header['alarm_count']}\n"
        f"   正常告警数: {metadata['cached_normal_alarm_count']}，"
        f"清除告警数: {metadata['cached_clear_alarm_count']}\n"
        f"   耗时: {elapsed:.2f} 秒"
    )


if __name__ == "__main__":
    main()
