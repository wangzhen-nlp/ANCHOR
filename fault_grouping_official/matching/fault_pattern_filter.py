"""落盘前的故障模式过滤适配器。

复用（移植自 ticket_recall/evaluation/analyze_case_fault_patterns.py 的）故障模式
分析，等价于在该脚本上同时启用 --filter-others 与 --one-component-only，并作为默认
行为：

  仅当故障组拆分后 component_count == 1，且剔除 ip_ring_others 之后仍保留至少一个
  可识别故障模式时，才允许写入输出文件；否则在落盘前丢弃。

分析在 build_jsonl_match_output() 产出的“增强后”记录上进行（含 match_info /
ne_info / group_info / symptoms），与该脚本作用于 case JSONL 的结构一致。
"""

# 分析逻辑已移植进官方包（fault_pattern_analysis），不再依赖 ticket_recall /
# 旧 fault_grouping / alarm_tools 等外部包，保证 fault_grouping_official 自满足。
from fault_grouping_official.matching.fault_pattern_analysis import (
    SiteRelationIndex,
    analyze_case_record,
    build_site_has_router_device_map,
    filter_other_patterns,
)


class FaultPatternFilter:
    """对增强后的故障组记录做 filter-others + one-component-only 过滤。"""

    def __init__(self, relation_index, ne_to_site, site_has_router_device):
        self._relation_index = relation_index
        self._ne_to_site = ne_to_site
        self._site_has_router_device = site_has_router_device

    @classmethod
    def from_static_context(cls, ne_graph_data, site_chain_index, ne_to_site):
        """用官方 static_context 已有的数据构建过滤器，无需额外 site_chains 文件。

        官方 site_chain_index 与 analyze 脚本里 load_site_chain_index 的结构一致
        （downstream_site_hops / upstream_site_hops / bidirectional_sites），可直接
        注入 SiteRelationIndex 并展开为直接上下游/双向邻接关系；缺失时退化为仅凭
        ne_graph 拓扑（此时没有双向环关系，无法识别 ip_ring_* 模式）。
        """
        if site_chain_index:
            relation_index = SiteRelationIndex()
            relation_index.site_chains = site_chain_index
            relation_index._load_direct_relations_from_site_chains()
        else:
            relation_index = SiteRelationIndex(ne_graph_data=ne_graph_data)
        site_has_router_device = build_site_has_router_device_map(ne_graph_data)
        return cls(relation_index, ne_to_site, site_has_router_device)

    def should_keep(self, record):
        """等价 analyze_case_fault_patterns.py --filter-others --one-component-only。

        - one-component-only：拆分后必须是单连通分量（component_count == 1）。
        - filter-others + 无条件兜底：剔除 ip_ring_others 后必须仍有可识别模式。
          （脚本中“had_other 且过滤后为空则丢弃”与“过滤后为空则丢弃”两条分支，
          合并即：过滤 others 后无模式 -> 丢弃。）
        """
        result = analyze_case_record(
            record,
            self._relation_index,
            self._ne_to_site,
            self._site_has_router_device,
        )
        if result.get("component_count") != 1:
            return False
        result = filter_other_patterns(result)
        return bool(result.get("patterns"))
