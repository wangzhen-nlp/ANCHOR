from dataclasses import dataclass

from anchor_grouping_online.link_peer_index import build_peer_index
from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.rule_config import (
    OUTPUT_ELIGIBLE_RULE_FIELD,
    data_link_adjacent_no_offline_rule,
    data_link_adjacent_offline_rule,
    data_no_offline_adjacent_optional_offline_rule,
    data_offline_adjacent_offline_rule,
)
from anchor_grouping_online.site_topology import (
    build_site_chain_index,
    build_site_domain_map,
    build_site_to_ne_ids,
    site_chain_upstream_hops_are_complete,
)


@dataclass
class LoadedStaticContext:
    ne_graph_data: dict
    site_chain_index: object
    site_chain_upstream_hops_complete: bool
    site_domain_map: dict
    ne_to_site: dict
    alarm_source_domain_map: dict
    site_to_ne_ids: dict
    link_peer_index: dict


def load_static_context(args):
    print(f"加载资源缓冲文件: {args.resource_buffer}")
    resources = load_resource_buffer(
        args.resource_buffer,
        wanted_types=("ne_graph", "site_chains", "link_peer_index"),
    )
    ne_graph_data = resources["ne_graph"]
    site_chain_resource = resources["site_chains"]
    site_chain_index = build_site_chain_index(site_chain_resource)
    site_chain_upstream_hops_complete = site_chain_upstream_hops_are_complete(
        site_chain_resource
    )
    site_domain_map = build_site_domain_map(ne_graph_data)
    print(f"预计算站点链路站点数: {len(site_chain_index)}")
    hop_completeness_label = (
        "完整，故障模式分析可跳过 BFS 补齐"
        if site_chain_upstream_hops_complete
        else "未知/截断，仅故障模式分析保留 BFS 补齐"
    )
    print("预计算上游 hop 完整性: " + hop_completeness_label)

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
    print("构建 site -> NE 索引...")
    site_to_ne_ids = build_site_to_ne_ids(ne_graph_data)
    print(f"site -> NE 索引站点数: {len(site_to_ne_ids)}")
    link_peer_index = build_peer_index(resources["link_peer_index"])
    print(f"link peer_index 记录数: {len(link_peer_index)}")

    return LoadedStaticContext(
        ne_graph_data=ne_graph_data,
        site_chain_index=site_chain_index,
        site_chain_upstream_hops_complete=site_chain_upstream_hops_complete,
        site_domain_map=site_domain_map,
        ne_to_site=ne_to_site,
        alarm_source_domain_map=alarm_source_domain_map,
        site_to_ne_ids=site_to_ne_ids,
        link_peer_index=link_peer_index,
    )


def build_rules_config():
    rules_config = {
        "data_link_adjacent_no_offline_rule": data_link_adjacent_no_offline_rule,
        "data_link_adjacent_offline_rule": data_link_adjacent_offline_rule,
        "data_no_offline_adjacent_optional_offline_rule": (
            data_no_offline_adjacent_optional_offline_rule
        ),
        "data_offline_adjacent_offline_rule": data_offline_adjacent_offline_rule,
    }
    print("启用规则: " + ", ".join(rules_config.keys()))
    return rules_config


def collect_output_eligible_rules(rules_config):
    """返回标记为 output_eligible=True 的可输出规则名集合。"""
    eligible = frozenset(
        rule_name
        for rule_name, rule in rules_config.items()
        if rule.get(OUTPUT_ELIGIBLE_RULE_FIELD)
    )
    if not eligible:
        print("未标记可输出规则(output_eligible)，结果不做规则过滤")
        return None
    print("仅输出包含以下规则的故障组: " + ", ".join(sorted(eligible)))
    return eligible


def build_fault_pattern_filter(static_context):
    """构建二次汇聚前的故障模式过滤器。"""
    from anchor_grouping_online.matching.fault_pattern_filter import (
        FaultPatternFilter,
    )

    fault_pattern_filter = FaultPatternFilter.from_static_context(
        static_context.ne_graph_data,
        static_context.site_chain_index,
        static_context.ne_to_site,
        precomputed_upstream_hops_complete=(
            static_context.site_chain_upstream_hops_complete
        ),
    )
    print("已启用故障模式过滤: filter-others + one-component-only")
    return fault_pattern_filter
