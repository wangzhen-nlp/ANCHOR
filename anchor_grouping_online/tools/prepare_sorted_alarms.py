import os
import time
from argparse import ArgumentParser

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.alarm_types import CRITICAL_ALARMS
from anchor_grouping_online.alarm_events.io import (
    is_clear_alarm,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
)
from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.site_topology import (
    build_site_chain_index,
    build_site_topology_from_ne_graph,
)
from anchor_grouping_online.alarm_events.sorted_cache import write_sorted_alarm_cache
from anchor_grouping_online.tools.topology_resources import (
    RESOURCE_BUFFER_JSONL,
    resource_display,
)
from anchor_grouping_online.tools.region_utils import (
    allowed_devices_for_regions,
    parse_regions,
)


def _load_valid_sites_and_ne_mapping(resource_buffer_path):
    """从 resource_buffer.jsonl 取 ne_graph / site_chains，与 match_rules.py 口径一致。"""
    resources = load_resource_buffer(
        resource_buffer_path,
        wanted_types=("ne_graph", "site_chains"),
    )
    ne_graph_data = resources["ne_graph"]
    _site_chain_index, valid_sites = build_site_chain_index(resources["site_chains"])
    _topo_downstream_map, topology_sites = build_site_topology_from_ne_graph(ne_graph_data)
    valid_sites.update(topology_sites)

    ne_to_site = {
        ne_id: str(ne_info.get("site_id", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("site_id", "")).strip()
    }
    return valid_sites, ne_to_site, ne_graph_data


def build_sorted_alarms(
    alarm_input,
    *,
    resource_buffer_path=RESOURCE_BUFFER_JSONL,
    start_time=None,
    end_time=None,
    clear_delay_sec=0.0,
    regions=None,
    show_progress=True,
):
    start_ts = parse_datetime_text(start_time, "start_time").timestamp() if start_time else None
    end_ts = parse_datetime_text(end_time, "end_time").timestamp() if end_time else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start_time 不能晚于 end_time")

    selected_regions = parse_regions(regions)
    valid_sites, ne_to_site, ne_graph_data = _load_valid_sites_and_ne_mapping(resource_buffer_path)
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
        show_progress=show_progress,
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
        "resource_buffer": os.path.abspath(resource_buffer_path),
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
        "--resource-buffer",
        type=str,
        default=RESOURCE_BUFFER_JSONL,
        help=(
            "build_resource_buffer.py 生成的资源缓冲文件（含 ne_graph / site_chains），"
            f"与 match_rules.py 一致，默认: {resource_display('resource_buffer.jsonl')}"
        ),
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
        resource_buffer_path=args.resource_buffer,
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
