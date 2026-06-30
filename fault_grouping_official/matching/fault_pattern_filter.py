"""落盘前的故障模式过滤与记录增强。

默认同时启用模式过滤与单连通分量约束，并在保留的记录上追加模式信息：

  - 过滤：仅当故障组拆分后 component_count == 1，且剔除 ip_ring_others 之后仍保留
    至少一个可识别故障模式时，才允许写入输出文件；否则在落盘前丢弃。
  - 增强：对保留下来的记录追加 note 模式备注、fault_pattern_* 字段，以及补充相关
    站点/网元/链路（supplemental_fault_pattern_context，供 ne_propagation_
    visualizer.html 标红展示）。

分析与增强都作用在 build_jsonl_match_output() 产出的记录上（含 match_info /
ne_info / group_info / symptoms）。
"""

from fault_grouping_official.matching.fault_pattern_analysis import (
    SiteRelationIndex,
    absorb_unmanaged_downstream_sites,
    analyze_case_record,
    append_note,
    augment_case_with_supplemental_fault_pattern_sites,
    build_pattern_note,
    build_site_has_router_device_map,
    extract_case_sites,
    extract_offline_sites,
    filter_other_patterns,
    projected_active_components_by_original_graph,
)


class FaultPatternFilter:
    """对增强后的故障组记录做 filter-others + one-component-only 过滤与模式增强。"""

    def __init__(
        self,
        relation_index,
        ne_to_site,
        site_has_router_device,
        ne_graph_data,
        site_to_ne_ids,
        site_graph_data=None,
    ):
        self._relation_index = relation_index
        self._ne_to_site = ne_to_site
        self._site_has_router_device = site_has_router_device
        self._ne_graph_data = ne_graph_data
        self._site_to_ne_ids = site_to_ne_ids
        self._site_graph_data = site_graph_data or {}

    @classmethod
    def from_static_context(
        cls,
        ne_graph_data,
        site_chain_index,
        ne_to_site,
        site_to_ne_ids,
        site_graph_data=None,
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
            relation_index._load_direct_relations_from_site_chains()
        else:
            relation_index = SiteRelationIndex(ne_graph_data=ne_graph_data)
        site_has_router_device = build_site_has_router_device_map(ne_graph_data)
        return cls(
            relation_index,
            ne_to_site,
            site_has_router_device,
            ne_graph_data,
            site_to_ne_ids,
            site_graph_data=site_graph_data,
        )

    def process(self, record):
        """对单条增强后记录做过滤与模式增强。

        返回增强后的记录（保留并写盘）；若被过滤掉则返回 None（不落盘）。

        过滤判定：
          - one-component-only：拆分后必须是单连通分量（component_count == 1）。
          - filter-others + 无条件兜底：剔除 ip_ring_others 后必须仍有可识别模式，
            否则丢弃。
        增强使用“剔除 others 之后”的分析结果。
        """
        # ① one-component-only 提前短路：先用便宜的方式算出投影连通分量数，!= 1 的组
        #    必被丢弃，可跳过 analyze_case_record 里逐分量的 classify_component
        #    （其中 longest_path 的穷举 DFS 是热路径上的指数级开销来源）。被丢弃的组
        #    本就不落盘，最终 keep/drop 与增强结果与"先 analyze 再判断"完全一致。
        if self._projected_component_count(record) != 1:
            return None
        analysis = analyze_case_record(
            record,
            self._relation_index,
            self._ne_to_site,
            self._site_has_router_device,
        )
        if analysis.get("component_count") != 1:
            # 与上面短路一致（同一确定性计算），仅作健壮性兜底。
            return None
        analysis = filter_other_patterns(analysis)
        if not analysis.get("patterns"):
            return None
        return self._augment(record, analysis)

    def _projected_component_count(self, record):
        """与 analyze_case_record 一致地算出投影连通分量数，但不做逐分量分类。

        由 extract_case_sites / extract_offline_sites /
        absorb_unmanaged_downstream_sites / projected_active_components_by_original_graph），
        按 analyze_case_record 计算 component_count 的同一流程执行，但省掉昂贵的
        classify_component。
        """
        site_ids = extract_case_sites(record)
        offline_sites = extract_offline_sites(record, self._ne_to_site) & set(site_ids)
        active_sites, _unmanaged, _absorbed_by, _steps = absorb_unmanaged_downstream_sites(
            site_ids,
            offline_sites,
            self._relation_index,
        )
        projected_components = projected_active_components_by_original_graph(
            site_ids,
            active_sites,
            self._relation_index,
        )
        return len(projected_components)

    def _augment(self, record, analysis):
        """把分析结果写入记录，省去不必要的 deepcopy。

        record 是每个故障组新构建、未被共享的记录，可安全原地改写。
        """
        pattern_note = build_pattern_note(analysis)

        record["note"] = append_note(record.get("note", ""), pattern_note)
        match_info = record.setdefault("match_info", {})
        if isinstance(match_info, dict):
            match_info["note"] = append_note(match_info.get("note", ""), pattern_note)

        record["fault_pattern_analysis"] = analysis
        record["fault_patterns"] = analysis.get("patterns", [])
        record["fault_pattern_count"] = analysis.get("pattern_count", 0)
        record["fault_pattern_managed_sites"] = analysis.get("managed_sites", [])
        record["fault_pattern_active_unmanaged_sites"] = analysis.get(
            "active_unmanaged_sites", []
        )
        augment_case_with_supplemental_fault_pattern_sites(
            record,
            analysis,
            self._ne_graph_data,
            self._site_to_ne_ids,
            site_has_router_device=self._site_has_router_device,
            site_graph_data=self._site_graph_data,
        )
        return record
