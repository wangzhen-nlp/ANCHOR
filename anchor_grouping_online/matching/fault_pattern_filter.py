"""落盘前的故障模式过滤与记录增强。

默认同时启用模式过滤与单连通分量约束，并在保留的记录上追加模式信息：

  - 过滤：仅当故障组拆分后 component_count == 1，且剔除 ip_ring_others 之后仍保留
    至少一个可识别故障模式时，才允许写入输出文件；否则在落盘前丢弃。
  - 增强：对保留下来的记录追加 note 模式备注、fault_pattern_* 字段，以及补充相关
    站点/网元/链路（supplemental_fault_pattern_context，供 ne_propagation_
    visualizer.html 标红展示）。

过滤分析可直接作用于引擎原始 match，避免为被丢弃记录构建完整输出；增强仍作用于
build_jsonl_match_output() 产出的记录（含 match_info / ne_info / group_info / symptoms）。
"""

from dataclasses import dataclass

from anchor_grouping_online.matching.fault_pattern_analysis import (
    MAX_ANALYSIS_SITES,
    SiteRelationIndex,
    analyze_prepared_case,
    append_note,
    augment_case_with_supplemental_fault_pattern_sites,
    build_pattern_note,
    build_site_has_router_device_map,
    extract_case_sites,
    filter_other_patterns,
    normalize_text,
    prepare_case_record,
)


@dataclass
class FaultPatternFilterStats:
    """落盘前过滤丢弃统计（按阈值归因）。"""

    # 整组总站点数 > MAX_ANALYSIS_SITES(200) 直接拒绝的故障组数。
    dropped_by_max_analysis_sites: int = 0
    # 因某个断站簇 > LONGEST_PATH_EXACT_MAX_SITES(18) 放弃精确搜索、
    # 最终导致整组无可识别模式而被丢弃的故障组数。
    dropped_by_longest_path_cap: int = 0


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
        self.stats = FaultPatternFilterStats()

    @classmethod
    def from_static_context(
        cls,
        ne_graph_data,
        site_chain_index,
        ne_to_site,
        site_to_ne_ids,
        site_graph_data=None,
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
        analysis = self.analyze(record)
        if analysis is None:
            return None
        return self.augment(record, analysis)

    def _extract_match_sites(self, match):
        """从引擎原始 match 提取与完整输出记录等价的站点集合。

        build_group_output 还会把 symptom.alarm_source 对应的静态 site_id 加入
        group_info；这里轻量复刻该补齐，避免为了模式过滤提前构建全部 NE/链路。
        """
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
        """在构建完整 JSONL 输出前，对引擎原始 match 做等价模式判定。"""
        return self.analyze(match, site_ids=self._extract_match_sites(match))

    def analyze(self, record, site_ids=None):
        """返回可保留记录的模式分析；不满足输出条件时返回 None。"""
        if site_ids is None:
            site_ids = extract_case_sites(record)
        if len(site_ids) > MAX_ANALYSIS_SITES:
            self.stats.dropped_by_max_analysis_sites += 1
            return None

        prepared_case = prepare_case_record(
            record,
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
            record,
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

    def augment(self, record, analysis):
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
