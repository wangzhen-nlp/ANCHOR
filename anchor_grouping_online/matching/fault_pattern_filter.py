"""二次汇聚前的故障模式过滤。

仅当故障组拆分后 component_count == 1，且剔除 ip_ring_others 之后仍保留
至少一个可识别故障模式时，匹配结果才可作为原始故障组之间的关联证据。
"""

from dataclasses import dataclass

from anchor_grouping_online.matching.fault_pattern_analysis import (
    MAX_ANALYSIS_SITES,
    SiteRelationIndex,
    analyze_prepared_case,
    build_site_has_router_device_map,
    extract_case_sites,
    filter_other_patterns,
    normalize_text,
    prepare_case_record,
)


@dataclass
class FaultPatternFilterStats:
    """二次汇聚前过滤丢弃统计（按阈值归因）。"""

    # 整组总站点数 > MAX_ANALYSIS_SITES(200) 直接拒绝的故障组数。
    dropped_by_max_analysis_sites: int = 0
    # 因某个断站簇 > LONGEST_PATH_EXACT_MAX_SITES(18) 放弃精确搜索、
    # 最终导致整组无可识别模式而被丢弃的故障组数。
    dropped_by_longest_path_cap: int = 0


class FaultPatternFilter:
    """对匹配结果执行 filter-others + one-component-only 过滤。"""

    def __init__(
        self,
        relation_index,
        ne_to_site,
        site_has_router_device,
    ):
        self._relation_index = relation_index
        self._ne_to_site = ne_to_site
        self._site_has_router_device = site_has_router_device
        self.stats = FaultPatternFilterStats()

    @classmethod
    def from_static_context(
        cls,
        ne_graph_data,
        site_chain_index,
        ne_to_site,
        precomputed_upstream_hops_complete=False,
    ):
        """用 static_context 已有的数据构建过滤器，无需额外 site_chains 文件。

        site_chain_index 包含 downstream_site_hops、upstream_site_hops 和
        bidirectional_sites，可直接注入 SiteRelationIndex 并展开为直接上下游/双向
        邻接关系；缺失时退化为仅凭 ne_graph 拓扑（此时没有双向环关系，无法识别
        ip_ring_* 模式）。
        """
        if site_chain_index:
            relation_index = SiteRelationIndex()
            relation_index.site_chains = site_chain_index
            relation_index.precomputed_upstream_hops_complete = bool(
                precomputed_upstream_hops_complete
            )
            relation_index._load_direct_relations_from_site_chains()
        else:
            relation_index = SiteRelationIndex(ne_graph_data=ne_graph_data)
        site_has_router_device = build_site_has_router_device_map(ne_graph_data)
        return cls(
            relation_index,
            ne_to_site,
            site_has_router_device,
        )

    def _extract_match_sites(self, match):
        """从引擎原始 match 及告警源静态映射提取站点集合。"""
        site_ids = set(extract_case_sites(match))
        for symptom in match.get("symptoms", []) or []:
            if not isinstance(symptom, dict):
                continue
            alarm_source = symptom.get("alarm_source")
            static_site_id = normalize_text(self._ne_to_site.get(alarm_source, ""))
            if static_site_id:
                site_ids.add(static_site_id)
        return sorted(site_ids)

    def analyze_match(self, match):
        """返回可用于二次汇聚的模式分析；不满足条件时返回 None。"""
        site_ids = self._extract_match_sites(match)
        if len(site_ids) > MAX_ANALYSIS_SITES:
            self.stats.dropped_by_max_analysis_sites += 1
            return None

        prepared_case = prepare_case_record(
            match,
            self._relation_index,
            self._ne_to_site,
            self._site_has_router_device,
            site_ids=site_ids,
            # 只接受单分量；发现第二个投影分量即可停止遍历。
            component_limit=2,
        )
        if len(prepared_case.projected_components) != 1:
            return None

        cap_hits = [0]
        analysis = analyze_prepared_case(
            match,
            prepared_case,
            self._relation_index,
            recognized_patterns_only=True,
            cap_hits=cap_hits,
        )
        analysis = filter_other_patterns(analysis)
        if not analysis.get("patterns"):
            # 归因：仅当本组分析确实触发了 >LONGEST_PATH_EXACT_MAX_SITES 的精确搜索
            # 放弃（cap_hits>0），才计入被 18 上限丢弃；其它无模式丢弃不计入。
            if cap_hits[0] > 0:
                self.stats.dropped_by_longest_path_cap += 1
            return None
        return analysis
