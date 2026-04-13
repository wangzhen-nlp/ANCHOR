# Site Pair Features

本文档对应 `site_link_learning/core.py` 当前站点对版本的数据构造逻辑，描述样本里实际写入 `features` 的全部特征。

说明：

- 样本粒度：有向站点对 `(u_site_id, v_site_id)`
- 正样本定义：`u_site -> v_site` 至少存在一条已观测跨站 NE 有向边
- 当前文档只描述 `features` 字段
- `candidate_reasons`、`supporting_ne_edge_count`、`supporting_link_types` 等属于样本元信息，不属于模型输入特征

## 1. 站点基础关系

- `same_region`
- `same_dominant_domain`
- `geo_distance_km`
- `geo_distance_missing`
- `geo_distance_log1p`

## 2. 站点规模

- `left_site_size`
- `right_site_size`
- `site_size_ratio_min_max`
- `site_size_diff_abs`

## 3. 候选边排除后的上下行完整性

这些特征会先把当前候选对 `(left_site_id, right_site_id)` 从左右站点的出入邻居里临时排除，再看站点本身是否还具备上下游连接。

- `left_site_out_degree_excl_pair`
- `left_site_in_degree_excl_pair`
- `right_site_out_degree_excl_pair`
- `right_site_in_degree_excl_pair`
- `left_missing_outgoing_excl_pair`
- `left_missing_incoming_excl_pair`
- `right_missing_outgoing_excl_pair`
- `right_missing_incoming_excl_pair`
- `left_has_both_in_out_excl_pair`
- `right_has_both_in_out_excl_pair`

## 4. 候选边是否补齐当前方向缺口

这里把候选方向固定为 `left -> right`。

- `candidate_fills_left_outgoing_gap`
- `candidate_fills_right_incoming_gap`
- `candidate_fills_forward_zero_gap_both`
- `candidate_fills_reverse_zero_gap_both`
- `candidate_completes_left_bidirectional_role`
- `candidate_completes_right_bidirectional_role`
- `candidate_completes_bidirectional_roles_for_both`

含义：

- `candidate_fills_left_outgoing_gap`：如果去掉当前候选对后，`left` 没有任何出站点连接，则该特征为 1
- `candidate_fills_right_incoming_gap`：如果去掉当前候选对后，`right` 没有任何入站点连接，则该特征为 1
- `candidate_fills_forward_zero_gap_both`：当前候选同时填补 `left` 的出向缺口和 `right` 的入向缺口
- `candidate_fills_reverse_zero_gap_both`：如果反向看，当前对更像是在补 `left` 的入向和 `right` 的出向
- `candidate_completes_left_bidirectional_role`：候选边会让 `left` 从“仅有入向、无出向”变成“上下游都有”
- `candidate_completes_right_bidirectional_role`：候选边会让 `right` 从“仅有出向、无入向”变成“上下游都有”
- `candidate_completes_bidirectional_roles_for_both`：候选边同时让左右两侧都达到各自的上下行完整

## 5. 同类站点模板缺口

这里会给每个站点找一个“同类站点 peer group”，优先级如下：

1. `region + dominant_domain + size_bucket`
2. `dominant_domain + size_bucket`
3. `dominant_domain`
4. 如果都没有同类站点，则为 `none`

对应特征：

- `left_peer_site_count`
- `right_peer_site_count`
- `left_peer_out_degree_median`
- `left_peer_in_degree_median`
- `right_peer_out_degree_median`
- `right_peer_in_degree_median`
- `left_out_degree_gap_to_peer_median`
- `left_in_degree_gap_to_peer_median`
- `right_out_degree_gap_to_peer_median`
- `right_in_degree_gap_to_peer_median`
- `left_out_degree_gap_ratio_to_peer_median`
- `left_in_degree_gap_ratio_to_peer_median`
- `right_out_degree_gap_ratio_to_peer_median`
- `right_in_degree_gap_ratio_to_peer_median`
- `forward_gap_fill_score`
- `reverse_gap_fill_score`
- `forward_minus_reverse_gap_fill_score`

peer group 级别 one-hot：

- `left_peer_level_is__region_domain_size`
- `left_peer_level_is__domain_size`
- `left_peer_level_is__domain_only`
- `left_peer_level_is__none`
- `right_peer_level_is__region_domain_size`
- `right_peer_level_is__domain_size`
- `right_peer_level_is__domain_only`
- `right_peer_level_is__none`

## 6. 站点级图结构

- `left_site_out_degree`
- `left_site_in_degree`
- `left_site_undirected_degree`
- `right_site_out_degree`
- `right_site_in_degree`
- `right_site_undirected_degree`
- `common_out_count`
- `common_in_count`
- `common_neighbor_count`
- `jaccard_out`
- `jaccard_in`
- `jaccard_neighbor`
- `two_hop_left_to_right_count`
- `two_hop_right_to_left_count`

## 7. 候选方向和邻居 domain 的匹配程度

- `left_neighbor_target_domain_match_count`
- `right_neighbor_source_domain_match_count`
- `left_site_receives_from_right_domain_count`
- `right_site_receives_from_left_domain_count`
- `left_site_sends_to_right_domain_count`
- `right_site_sends_to_left_domain_count`

## 8. 站点画像多样性和类别重叠

多样性：

- `left_site_type_diversity`
- `right_site_type_diversity`
- `left_site_network_type_diversity`
- `right_site_network_type_diversity`
- `left_site_manufacturer_diversity`
- `right_site_manufacturer_diversity`

类别集合重叠：

- `type_key_jaccard`
- `network_type_key_jaccard`
- `manufacturer_key_jaccard`

分布相似度：

- `domain_ratio_cosine_similarity`
- `type_ratio_cosine_similarity`
- `network_type_ratio_cosine_similarity`
- `manufacturer_ratio_cosine_similarity`

## 9. 图相似性分数

- `adamic_adar_neighbor`
- `resource_allocation_neighbor`
- `adamic_adar_two_hop_left_to_right`
- `resource_allocation_two_hop_left_to_right`

## 10. 缺失信息标记

- `left_region_missing`
- `right_region_missing`

## 11. Domain 比例与 One-Hot 编码

站点内各 domain bucket 占比：

- `left_site_domain_ratio__ran`
- `left_site_domain_ratio__transmission`
- `left_site_domain_ratio__data`
- `left_site_domain_ratio__other`
- `left_site_domain_ratio__missing`
- `right_site_domain_ratio__ran`
- `right_site_domain_ratio__transmission`
- `right_site_domain_ratio__data`
- `right_site_domain_ratio__other`
- `right_site_domain_ratio__missing`

左右站点 dominant domain 的 one-hot：

- `left_dominant_domain_is__ran`
- `left_dominant_domain_is__transmission`
- `left_dominant_domain_is__data`
- `left_dominant_domain_is__other`
- `left_dominant_domain_is__missing`
- `right_dominant_domain_is__ran`
- `right_dominant_domain_is__transmission`
- `right_dominant_domain_is__data`
- `right_dominant_domain_is__other`
- `right_dominant_domain_is__missing`

dominant domain 组合 one-hot：

- `dominant_domain_pair__ran__ran`
- `dominant_domain_pair__ran__transmission`
- `dominant_domain_pair__ran__data`
- `dominant_domain_pair__ran__other`
- `dominant_domain_pair__ran__missing`
- `dominant_domain_pair__transmission__ran`
- `dominant_domain_pair__transmission__transmission`
- `dominant_domain_pair__transmission__data`
- `dominant_domain_pair__transmission__other`
- `dominant_domain_pair__transmission__missing`
- `dominant_domain_pair__data__ran`
- `dominant_domain_pair__data__transmission`
- `dominant_domain_pair__data__data`
- `dominant_domain_pair__data__other`
- `dominant_domain_pair__data__missing`
- `dominant_domain_pair__other__ran`
- `dominant_domain_pair__other__transmission`
- `dominant_domain_pair__other__data`
- `dominant_domain_pair__other__other`
- `dominant_domain_pair__other__missing`
- `dominant_domain_pair__missing__ran`
- `dominant_domain_pair__missing__transmission`
- `dominant_domain_pair__missing__data`
- `dominant_domain_pair__missing__other`
- `dominant_domain_pair__missing__missing`
