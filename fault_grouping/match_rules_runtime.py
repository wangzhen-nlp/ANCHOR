import json
import os
import time

from collections import defaultdict
from dataclasses import dataclass

from alarm_tools.alarm_types import CRITICAL_ALARMS
from fault_grouping.match_rules_alarm_io import (
    count_alarm_event_types,
    load_sorted_alarm_cache_with_stats,
    load_valid_alarms,
    parse_datetime_text,
    trim_trailing_clear_alarms,
    warn_sorted_alarm_cache_option_mismatch,
)
from fault_grouping.rule_config import (
    data_link_adjacent_link_rule,
    data_link_adjacent_no_offline_rule,
    data_link_adjacent_offline_rule,
    data_no_offline_adjacent_optional_offline_rule,
    data_offline_adjacent_offline_rule,
    data_rule,
    link_rule,
    power_rule,
    transmission_rule,
)
from fault_grouping.site_merge_helper import (
    AdaptiveDensitySiteMergeHelper,
    BatchSiteMergeHelper,
)
from fault_grouping.sorted_alarm_cache import (
    is_sorted_alarm_cache_file,
    write_sorted_alarm_cache,
)
from fault_grouping.temporal_graph_engine import TemporalGraphEngine


@dataclass
class LoadedStaticContext:
    ne_graph_data: dict
    topo_downstream_map: dict
    valid_sites: set
    site_chain_index: object
    site_domain_map: dict
    site_graph_data: dict
    ne_to_site: dict
    alarm_source_domain_map: dict
    site_to_ne_ids: dict
    ne_link_info_cache: dict


@dataclass
class AlarmLoadResult:
    processed_count: int
    valid_alarms: list
    normal_alarm_count: int
    clear_alarm_count: int
    sort_elapsed: float

    @property
    def filtered_count(self):
        return len(self.valid_alarms)


def _extract_link_direction_values(link_meta):
    if isinstance(link_meta, dict):
        raw_values = link_meta.values()
    elif isinstance(link_meta, (list, tuple, set)):
        raw_values = link_meta
    else:
        raw_values = [link_meta]

    direction_values = set()
    for raw_value in raw_values:
        text = str(raw_value).strip()
        if text:
            direction_values.add(text)
    return direction_values


def build_site_topology_from_ne_graph(ne_graph_data):
    """基于 ne_graph 原始连边构建站点级 downstream 拓扑。"""
    ne_to_site = {}
    all_sites = set()

    for ne_id, ne_info in ne_graph_data.items():
        site_id = str(ne_info.get("site_id", "")).strip()
        if not site_id:
            continue
        ne_to_site[ne_id] = site_id
        all_sites.add(site_id)

    topo_downstream_map = defaultdict(set)
    for site_id in all_sites:
        topo_downstream_map[site_id]

    for source_ne, source_info in ne_graph_data.items():
        source_site = ne_to_site.get(source_ne)
        if not source_site:
            continue

        raw_links = source_info.get("link", {})
        if not isinstance(raw_links, dict):
            continue

        for target_ne, link_meta in raw_links.items():
            target_site = ne_to_site.get(target_ne)
            if not target_site or target_site == source_site:
                continue

            direction_values = _extract_link_direction_values(link_meta)
            if not direction_values:
                continue

            if any("<-" in direction for direction in direction_values):
                topo_downstream_map[source_site].add(target_site)
            if any("->" in direction for direction in direction_values):
                topo_downstream_map[target_site].add(source_site)

    return {
        site_id: sorted(downstream_sites)
        for site_id, downstream_sites in topo_downstream_map.items()
    }, all_sites


def _normalize_site_chain_hops(hops_map):
    normalized = {}
    if not isinstance(hops_map, dict):
        return normalized
    for related_site, hop_value in hops_map.items():
        related_site_id = str(related_site).strip()
        if not related_site_id:
            continue
        try:
            hop = int(hop_value)
        except (TypeError, ValueError):
            continue
        if hop <= 0:
            continue
        normalized[related_site_id] = hop
    return normalized


def load_site_chain_index(site_chains_path):
    """加载 generate_site_chains.py 产出的预计算上下游 hop 索引。"""
    data = json.load(open(site_chains_path, 'r', encoding='utf-8'))
    raw_sites = data.get("sites", {}) if isinstance(data, dict) else {}
    site_chain_index = {}
    valid_sites = set()

    for raw_site_id, raw_info in raw_sites.items():
        site_id = str(raw_site_id or "").strip()
        if not site_id or not isinstance(raw_info, dict):
            continue

        downstream_hops = _normalize_site_chain_hops(raw_info.get("downstream_site_hops"))
        upstream_hops = _normalize_site_chain_hops(raw_info.get("upstream_site_hops"))
        bidirectional_sites = {
            str(neighbor_site or "").strip()
            for neighbor_site in raw_info.get("bidirectional_sites", [])
            if str(neighbor_site or "").strip()
        }

        site_chain_index[site_id] = {
            "downstream_site_hops": downstream_hops,
            "upstream_site_hops": upstream_hops,
            "bidirectional_sites": bidirectional_sites,
        }
        valid_sites.add(site_id)
        valid_sites.update(downstream_hops)
        valid_sites.update(upstream_hops)
        valid_sites.update(bidirectional_sites)

    return site_chain_index, valid_sites


def build_site_to_ne_ids(ne_graph_data):
    site_to_ne_ids = defaultdict(list)
    for ne_id, ne_info in ne_graph_data.items():
        if not isinstance(ne_info, dict):
            continue
        site_id = str(ne_info.get("site_id", "")).strip()
        if site_id:
            site_to_ne_ids[site_id].append(ne_id)
    return {
        site_id: tuple(sorted(ne_ids))
        for site_id, ne_ids in site_to_ne_ids.items()
    }


def parse_selected_rule_names(rule_args, valid_rule_names):
    if not rule_args:
        return []

    selected_rule_names = []
    seen_rule_names = set()
    available_rule_name_set = set(valid_rule_names)

    for raw_value in rule_args:
        for part in str(raw_value).replace("，", ",").split(","):
            rule_name = part.strip()
            if not rule_name or rule_name in seen_rule_names:
                continue
            if rule_name not in available_rule_name_set:
                raise ValueError(
                    f"未知规则名: {rule_name}；可选值: {', '.join(sorted(available_rule_name_set))}"
                )
            seen_rule_names.add(rule_name)
            selected_rule_names.append(rule_name)

    return selected_rule_names


def validate_main_args(parser, args):
    start_ts = None
    end_ts = None
    if args.start_time:
        start_ts = parse_datetime_text(args.start_time, "start_time").timestamp()
    if args.end_time:
        end_ts = parse_datetime_text(args.end_time, "end_time").timestamp()
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        parser.error("start_time 不能晚于 end_time")
    if args.batch_merge_density_knn < 0:
        parser.error("batch-merge-density-knn 不能小于 0")
    if args.batch_merge_density_scale < 0:
        parser.error("batch-merge-density-scale 不能小于 0")
    if args.batch_merge_density_min_meters < 0 or args.batch_merge_density_max_meters < 0:
        parser.error("batch-merge-density-min-meters / max-meters 不能小于 0")
    if (
        args.batch_merge_density_max_meters > 0
        and args.batch_merge_density_min_meters > 0
        and args.batch_merge_density_max_meters < args.batch_merge_density_min_meters
    ):
        parser.error("batch-merge-density-max-meters 不能小于 batch-merge-density-min-meters")
    if args.batch_merge_density_knn > 0 and args.batch_merge_density_scale <= 0:
        parser.error("启用 batch-merge-density-knn 时，batch-merge-density-scale 必须大于 0")
    if args.site_chains and not os.path.exists(args.site_chains):
        parser.error(f"site_chains 文件不存在: {args.site_chains}")
    return start_ts, end_ts


def load_static_context(args):
    ne_graph_data = json.load(open(args.ne_graph, 'r', encoding='utf-8'))
    if args.topo:
        print(f"加载显式站点拓扑: {args.topo}")
        topo_downstream_map = json.load(open(args.topo, 'r', encoding='utf-8'))
        valid_sites = set(topo_downstream_map.keys())
        for _, connected_sites in topo_downstream_map.items():
            if isinstance(connected_sites, list):
                valid_sites.update(connected_sites)
            elif isinstance(connected_sites, dict):
                valid_sites.update(connected_sites.keys())
    else:
        print(f"基于 ne_graph 原始连边构建站点传播拓扑: {args.ne_graph}")
        topo_downstream_map, valid_sites = build_site_topology_from_ne_graph(ne_graph_data)

    site_chain_index = None
    if args.site_chains:
        print(f"加载预计算站点链路: {args.site_chains}")
        site_chain_index, site_chain_valid_sites = load_site_chain_index(args.site_chains)
        valid_sites.update(site_chain_valid_sites)
        print(f"预计算站点链路站点数: {len(site_chain_index)}")

    site_domain_map = json.load(open(args.site_domain, 'r', encoding='utf-8'))
    site_graph_data = json.load(open(args.site_graph, 'r', encoding='utf-8'))

    print("加载有效站点集合...")
    print(f"有效站点数: {len(valid_sites)}")
    print(f"站点拓扑起点数: {len(topo_downstream_map)}")

    print("构建 ne -> site 映射...")
    ne_to_site = {
        ne_id: str(ne_info.get("site_id", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("site_id", "")).strip()
    }
    alarm_source_domain_map = {
        ne_id: str(ne_info.get("domain", "")).strip()
        for ne_id, ne_info in ne_graph_data.items()
        if str(ne_info.get("domain", "")).strip()
    }
    print(f"NE 数量: {len(ne_to_site)}")
    print("构建 site -> NE 输出索引...")
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    ne_link_info_cache = {}
    print(f"site -> NE 索引站点数: {len(site_to_ne_ids)}")

    return LoadedStaticContext(
        ne_graph_data=ne_graph_data,
        topo_downstream_map=topo_downstream_map,
        valid_sites=valid_sites,
        site_chain_index=site_chain_index,
        site_domain_map=site_domain_map,
        site_graph_data=site_graph_data,
        ne_to_site=ne_to_site,
        alarm_source_domain_map=alarm_source_domain_map,
        site_to_ne_ids=site_to_ne_ids,
        ne_link_info_cache=ne_link_info_cache,
    )


def print_run_configuration(args, static_context, valid_alarm_titles):
    if args.start_time or args.end_time:
        print(
            "告警首次发生时间过滤: "
            f"start_time={args.start_time or '-'}, "
            f"end_time={args.end_time or '-'}"
        )
    if args.clear_delay_sec > 0:
        print(f"清除告警最小延迟: {args.clear_delay_sec:g} 秒")
    if args.batch_merge_site_hops > 0:
        print(f"批内站点邻接合并: 开启，hop={args.batch_merge_site_hops}")
    if args.batch_merge_density_knn > 0:
        print(
            "批内站点密度合并: 开启，"
            f"k={args.batch_merge_density_knn}, "
            f"scale={args.batch_merge_density_scale:g}, "
            f"min_radius={args.batch_merge_density_min_meters:g}m, "
            f"max_radius={args.batch_merge_density_max_meters:g}m"
        )
    print(f"有效告警类型数: {len(valid_alarm_titles)}")


def build_rules_config(args, parser):
    all_rules_config = {
        "transmission_rule": transmission_rule,
        "link_rule": link_rule,
        "power_rule": power_rule,
        "data_rule": data_rule,
        "data_link_adjacent_no_offline_rule": data_link_adjacent_no_offline_rule,
        "data_link_adjacent_offline_rule": data_link_adjacent_offline_rule,
        "data_link_adjacent_link_rule": data_link_adjacent_link_rule,
        "data_no_offline_adjacent_optional_offline_rule": data_no_offline_adjacent_optional_offline_rule,
        "data_offline_adjacent_offline_rule": data_offline_adjacent_offline_rule,
    }
    try:
        selected_rule_names = parse_selected_rule_names(args.rule, all_rules_config.keys())
    except ValueError as exc:
        parser.error(str(exc))
    rules_config = (
        {
            rule_name: all_rules_config[rule_name]
            for rule_name in selected_rule_names
        }
        if selected_rule_names else all_rules_config
    )
    print("启用规则: " + ", ".join(rules_config.keys()))
    return rules_config


def build_batch_site_merge_helper(args, topo_downstream_map):
    density_site_merge_helper = None
    if args.batch_merge_density_knn > 0:
        density_site_merge_helper = AdaptiveDensitySiteMergeHelper(
            args.site_graph,
            density_knn=args.batch_merge_density_knn,
            density_scale=args.batch_merge_density_scale,
            min_radius_meters=args.batch_merge_density_min_meters,
            max_radius_meters=args.batch_merge_density_max_meters,
        )

    batch_site_merge_helper = None
    if args.batch_merge_site_hops > 0 or density_site_merge_helper is not None:
        batch_site_merge_helper = BatchSiteMergeHelper(
            topo_downstream_map,
            site_neighbor_hops=args.batch_merge_site_hops,
            density_helper=density_site_merge_helper,
        )
        if density_site_merge_helper is not None:
            print("⏳ 正在准备站点批内合并辅助器...")
            batch_site_merge_helper.warmup()
            print("✅ 站点批内合并辅助器就绪")
    return batch_site_merge_helper


def initialize_engine(args, static_context, rules_config, batch_site_merge_helper):
    print("⏳ 正在初始化时序图引擎与拓扑映射...")
    print(f"聚合等待时间: {args.aggregation_wait_sec:g} 秒")
    print(
        "event_cache 模式: "
        + ("设备告警时段(period)" if args.use_alarm_period_cache else "逐条活跃告警(raw, 默认)")
    )
    engine = TemporalGraphEngine(
        static_context.topo_downstream_map,
        rules_config,
        static_context.site_domain_map,
        alarm_source_domain_map=static_context.alarm_source_domain_map,
        aggregation_wait_sec=args.aggregation_wait_sec,
        site_merge_helper=batch_site_merge_helper,
        site_chain_index=static_context.site_chain_index,
        use_alarm_period_cache=args.use_alarm_period_cache,
    )
    print("✅ 引擎启动就绪，开始监听告警流...\n")
    return engine


def resolve_sorted_alarm_cache_input(parser, args):
    alarm_file_path = args.alarms
    sorted_alarm_cache_input = args.sorted_alarms_input.strip()
    if sorted_alarm_cache_input:
        if not os.path.exists(sorted_alarm_cache_input):
            parser.error(f"排序告警缓存不存在: {sorted_alarm_cache_input}")
        if not is_sorted_alarm_cache_file(sorted_alarm_cache_input):
            parser.error(f"不是有效的排序告警缓存: {sorted_alarm_cache_input}")
    elif is_sorted_alarm_cache_file(alarm_file_path):
        sorted_alarm_cache_input = alarm_file_path
    return sorted_alarm_cache_input


def load_alarm_data(args, parser, static_context, valid_alarm_titles, start_ts, end_ts):
    sorted_alarm_cache_input = resolve_sorted_alarm_cache_input(parser, args)
    if sorted_alarm_cache_input:
        print(f"⚡ 直接加载排序告警缓存: {sorted_alarm_cache_input}")
        load_start_time = time.time()
        (
            processed_count,
            valid_alarms,
            normal_alarm_count,
            clear_alarm_count,
            sorted_alarm_cache_metadata,
        ) = load_sorted_alarm_cache_with_stats(sorted_alarm_cache_input)
        warn_sorted_alarm_cache_option_mismatch(sorted_alarm_cache_metadata, args)
        sort_elapsed = time.time() - load_start_time
        return AlarmLoadResult(
            processed_count=processed_count,
            valid_alarms=valid_alarms,
            normal_alarm_count=normal_alarm_count,
            clear_alarm_count=clear_alarm_count,
            sort_elapsed=sort_elapsed,
        )

    processed_count, valid_alarms, normal_alarm_count, clear_alarm_count = load_valid_alarms(
        args.alarms,
        valid_alarm_titles,
        static_context.valid_sites,
        static_context.ne_to_site,
        start_ts=start_ts,
        end_ts=end_ts,
        clear_delay_sec=args.clear_delay_sec,
    )

    print("⏳ 正在按时间排序有效告警...")
    sort_start_time = time.time()
    valid_alarms.sort(key=lambda item: item["ts"])
    valid_alarms = trim_trailing_clear_alarms(valid_alarms)
    sort_elapsed = time.time() - sort_start_time
    if args.sorted_alarms_output:
        cached_normal_alarm_count, cached_clear_alarm_count = count_alarm_event_types(valid_alarms)
        cache_metadata = {
            "source_alarms": os.path.abspath(args.alarms),
            "topo": os.path.abspath(args.topo) if args.topo else "",
            "ne_graph": os.path.abspath(args.ne_graph),
            "start_time": args.start_time or "",
            "end_time": args.end_time or "",
            "clear_delay_sec": float(args.clear_delay_sec),
            "processed_count": processed_count,
            "normal_alarm_count": normal_alarm_count,
            "clear_alarm_count": clear_alarm_count,
            "cached_normal_alarm_count": cached_normal_alarm_count,
            "cached_clear_alarm_count": cached_clear_alarm_count,
            "valid_site_count": len(static_context.valid_sites),
            "ne_to_site_count": len(static_context.ne_to_site),
            "valid_alarm_title_count": len(valid_alarm_titles),
        }
        write_sorted_alarm_cache(args.sorted_alarms_output, valid_alarms, cache_metadata)
        print(f"💾 排序告警缓存已写出: {args.sorted_alarms_output}")

    return AlarmLoadResult(
        processed_count=processed_count,
        valid_alarms=valid_alarms,
        normal_alarm_count=normal_alarm_count,
        clear_alarm_count=clear_alarm_count,
        sort_elapsed=sort_elapsed,
    )


def print_alarm_load_summary(alarm_load_result):
    print(
        f"有效告警数: {alarm_load_result.filtered_count}，排序/加载耗时: "
        f"{alarm_load_result.sort_elapsed:.4f} 秒"
    )
    print(
        f"正常告警数: {alarm_load_result.normal_alarm_count}，"
        f"清除告警数: {alarm_load_result.clear_alarm_count}"
    )


def default_valid_alarm_titles():
    return CRITICAL_ALARMS
