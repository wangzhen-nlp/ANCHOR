#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成相邻站点对的方向先验（pairwise 证据融合版）。

核心思路：
1. 先从 ne_graph 提取站点级无向邻接，只回答“站点是否相连”；
2. 再按连通分量挑选更像汇聚层的 anchor 站点；
3. 计算每个站点到 anchor 的 hops 与层级分数；
4. 只对相邻站点对输出 A->B / B->A / <-> 三态方向关系。

这里不强行给所有站点排全序；当局部与全局证据都不够时，保留双向边。
"""

from collections import Counter, defaultdict, deque

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.tools.site_pair_order_common import (
    ProgressReporter,
    _get_site_id,
    apply_strict_ring_pairwise_override,
    build_strict_ring_context,
    build_transmission_misconnection_pairs,
    cross_domain_priority,
    iter_unique_cross_site_links,
    normalize_domain,
)


def compact_pairwise_prediction(pair_result):
    """把 pairwise 内部方向结果转换成统一的上下行预测格式。"""
    if pair_result.get("relation") == "<->":
        prediction = "bidirectional"
        upstream_site = None
        downstream_site = None
    else:
        upstream_site = pair_result.get("preferred_source")
        downstream_site = pair_result.get("preferred_target")
        prediction = (
            f"{upstream_site}->{downstream_site}"
            if downstream_site and upstream_site
            else None
        )

    return {
        "site_a": pair_result.get("site_a"),
        "site_b": pair_result.get("site_b"),
        "prediction": prediction,
        "upstream_site": upstream_site,
        "downstream_site": downstream_site,
    }


def _site_cross_domain_priority(site_id, site_domain_counts):
    """站点自身最高 domain 优先级；未知/其它 domain 不参与。"""
    return max(
        (
            cross_domain_priority(domain)
            for domain, count in site_domain_counts.get(site_id, {}).items()
            if count > 0
        ),
        default=0,
    )


def build_site_pair_inputs(
    ne_graph,
    show_progress=False,
    collect_cross_domain=False,
    transmission_misconnection_pairs=None,
):
    site_domain_counts = defaultdict(Counter)
    site_neighbors = defaultdict(set)
    site_external_edge_count = Counter()
    site_data_edge_count = Counter()
    site_transmission_edge_count = Counter()
    pair_edge_count = Counter()
    pair_site_domain_counts = defaultdict(lambda: defaultdict(Counter))
    # 跨类型连边证据：{pair_key: {site_priority, direction_votes, link_count}}。
    # 收集口径为全部跨 domain 连边（含被拓扑过滤的），只用于方向约束，不进拓扑。
    cross_domain_pair_evidence = defaultdict(
        lambda: {"site_priority": {}, "direction_votes": Counter(), "link_count": 0}
    )
    all_sites = set()

    with ProgressReporter(len(ne_graph), "pairwise: 聚合站点设备", show_progress) as progress:
        for ne_info in ne_graph.values():
            progress.update()
            if not isinstance(ne_info, dict):
                continue
            site_id = _get_site_id(ne_info)
            if not site_id:
                continue
            all_sites.add(site_id)
            site_domain_counts[site_id][normalize_domain(ne_info.get("domain", ""))] += 1

    with ProgressReporter(0, "pairwise: 扫描跨站链路", show_progress) as progress:
        for link in iter_unique_cross_site_links(
            ne_graph,
            include_filtered_cross_domain=collect_cross_domain,
            transmission_misconnection_pairs=transmission_misconnection_pairs,
        ):
            progress.update()
            left_site = link["source_site"]
            right_site = link["target_site"]
            pair_key = tuple(sorted((left_site, right_site)))
            left_domain = link["source_domain"]
            right_domain = link["target_domain"]

            if collect_cross_domain and left_domain != right_domain:
                left_priority = cross_domain_priority(left_domain)
                right_priority = cross_domain_priority(right_domain)
                left_site_priority = _site_cross_domain_priority(
                    left_site, site_domain_counts
                )
                right_site_priority = _site_cross_domain_priority(
                    right_site, site_domain_counts
                )
                endpoint_winner = (
                    left_site if left_priority > right_priority else right_site
                )
                site_winner = (
                    left_site
                    if left_site_priority > right_site_priority
                    else right_site
                )
                # 只有两端设备优先级都已知且不同，并且站点自身最高优先级支持
                # 同一方向时，才构成跨 domain 方向证据。
                if left_priority and right_priority and left_priority != right_priority:
                    if (
                        left_site_priority
                        and right_site_priority
                        and left_site_priority != right_site_priority
                        and endpoint_winner == site_winner
                    ):
                        evidence = cross_domain_pair_evidence[pair_key]
                        evidence["link_count"] += 1
                        site_priority = evidence["site_priority"]
                        if left_site_priority > site_priority.get(left_site, 0):
                            site_priority[left_site] = left_site_priority
                        if right_site_priority > site_priority.get(right_site, 0):
                            site_priority[right_site] = right_site_priority
                        evidence["direction_votes"][site_winner] += 1

            # 被拓扑过滤的跨 domain 连边只贡献约束证据，不参与任何拓扑聚合
            if not link.get("included_in_topology", True):
                continue

            site_neighbors[left_site].add(right_site)
            site_neighbors[right_site].add(left_site)
            pair_edge_count[pair_key] += 1

            pair_site_domain_counts[pair_key][left_site][left_domain] += 1
            pair_site_domain_counts[pair_key][right_site][right_domain] += 1

            site_external_edge_count[left_site] += 1
            site_external_edge_count[right_site] += 1
            if left_domain == "Data":
                site_data_edge_count[left_site] += 1
            elif left_domain == "Transmission":
                site_transmission_edge_count[left_site] += 1
            if right_domain == "Data":
                site_data_edge_count[right_site] += 1
            elif right_domain == "Transmission":
                site_transmission_edge_count[right_site] += 1

    with ProgressReporter(len(all_sites), "pairwise: 初始化孤立站点邻接", show_progress) as progress:
        for site_id in all_sites:
            progress.update()
            site_neighbors[site_id]

    return {
        "all_sites": all_sites,
        "site_domain_counts": site_domain_counts,
        "site_neighbors": site_neighbors,
        "site_external_edge_count": site_external_edge_count,
        "site_data_edge_count": site_data_edge_count,
        "site_transmission_edge_count": site_transmission_edge_count,
        "pair_edge_count": pair_edge_count,
        "pair_site_domain_counts": pair_site_domain_counts,
        "cross_domain_pair_evidence": cross_domain_pair_evidence,
    }


def _find_directed_cycle(constraint_adjacency):
    """在约束有向图中找一个环，返回环上约束的 pair_key 列表；无环返回 None。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    for root in sorted(constraint_adjacency):
        if color.get(root, WHITE) != WHITE:
            continue
        color[root] = GRAY
        stack = [(root, iter(sorted(constraint_adjacency.get(root, ()))))]
        path_nodes = [root]
        path_edges = []
        while stack:
            node, edge_iter = stack[-1]
            next_entry = None
            for next_site, pair_key in edge_iter:
                state = color.get(next_site, WHITE)
                if state == GRAY:
                    cycle_start = path_nodes.index(next_site)
                    return path_edges[cycle_start:] + [pair_key]
                if state == WHITE:
                    next_entry = (next_site, pair_key)
                    break
            if next_entry is None:
                stack.pop()
                color[node] = BLACK
                path_nodes.pop()
                if path_edges:
                    path_edges.pop()
            else:
                next_site, pair_key = next_entry
                color[next_site] = GRAY
                stack.append(
                    (next_site, iter(sorted(constraint_adjacency.get(next_site, ()))))
                )
                path_nodes.append(next_site)
                path_edges.append(pair_key)
    return None


def _break_constraint_cycles(constraints):
    """在约束有向图上原地消圈：每次找到一个环，丢弃环上证据最少的约束。

    约束成环说明跨类型证据互相矛盾；层级投影要求约束集无环，否则不收敛。
    返回被丢弃的 pair_key 列表。
    """
    dropped_pairs = []
    while constraints:
        constraint_adjacency = defaultdict(list)
        for pair_key, constraint in constraints.items():
            constraint_adjacency[constraint["upstream_site"]].append(
                (constraint["downstream_site"], pair_key)
            )
        cycle_pairs = _find_directed_cycle(constraint_adjacency)
        if not cycle_pairs:
            break
        weakest_pair = min(
            cycle_pairs,
            key=lambda pk: (constraints[pk]["evidence_link_count"], pk),
        )
        dropped_pairs.append(weakest_pair)
        del constraints[weakest_pair]
    return dropped_pairs


def build_cross_domain_constraints(pair_evidence, pair_edge_count):
    """从跨类型连边证据构建站点对方向约束：优先级高的一侧为上行。

    方向证据必须同时满足：连边端点 domain 优先级给出的方向，与两端站点自身
    最高 domain 优先级给出的方向一致。对内冲突消解时比较两端站点优先级，
    高者为上行；平局则该对不产生约束（计入 tie_pair_count）。跨对约束成环时
    按证据数丢弃最弱者。

    Returns:
        (constraints, stats)；constraints 为 {pair_key: {upstream_site,
        downstream_site, evidence_link_count, total_cross_link_count,
        has_topology_edge}}。
    """
    constraints = {}
    tie_pair_count = 0
    for pair_key in sorted(pair_evidence):
        evidence = pair_evidence[pair_key]
        site_a, site_b = pair_key
        priority_a = evidence["site_priority"].get(site_a, 0)
        priority_b = evidence["site_priority"].get(site_b, 0)
        if not priority_a or not priority_b or priority_a == priority_b:
            tie_pair_count += 1
            continue
        upstream_site, downstream_site = (
            (site_a, site_b) if priority_a > priority_b else (site_b, site_a)
        )
        constraints[pair_key] = {
            "upstream_site": upstream_site,
            "downstream_site": downstream_site,
            "evidence_link_count": int(evidence["direction_votes"].get(upstream_site, 0)),
            "total_cross_link_count": int(evidence["link_count"]),
            "has_topology_edge": pair_key in pair_edge_count,
        }
    dropped_pairs = _break_constraint_cycles(constraints)
    stats = {
        "tie_pair_count": tie_pair_count,
        "cycle_dropped_pair_count": len(dropped_pairs),
    }
    return constraints, stats


def compute_connected_components(all_sites, site_neighbors, show_progress=False):
    visited = set()
    components = []

    # 全程不排序：分量划分与成员集合是图不变量，遍历/输出顺序不影响边方向判断
    # （anchor 选择内部自带 site_id 全序，core_distance 为 BFS 最短路，均与顺序无关）。
    # 代价：component_id 编号与 full-output 顺序可能不跨运行稳定，但二者都不参与决策。
    with ProgressReporter(len(all_sites), "pairwise: 计算连通分量", show_progress) as progress:
        for start_site in all_sites:
            progress.update()
            if start_site in visited:
                continue
            queue = deque([start_site])
            visited.add(start_site)
            component = []

            while queue:
                current_site = queue.popleft()
                component.append(current_site)
                for neighbor_site in site_neighbors.get(current_site, ()):
                    if neighbor_site in visited:
                        continue
                    visited.add(neighbor_site)
                    queue.append(neighbor_site)

            components.append(component)

    return components


def find_bridge_pairs(all_sites, site_neighbors, show_progress=False):
    """非递归 Tarjan 算法识别无向站点图中的桥边。"""
    discovery_time = {}
    low_link = {}
    parent = {}
    bridges = set()
    current_time = 0

    # 不排序：桥集是图不变量，DFS 根/邻居顺序不改变返回的桥边集合（is_bridge 因此稳定）
    with ProgressReporter(len(all_sites), "pairwise: 识别桥边", show_progress) as progress:
        for start_site in all_sites:
            if start_site in discovery_time:
                continue

            parent[start_site] = None
            current_time += 1
            discovery_time[start_site] = low_link[start_site] = current_time
            progress.update()
            # 邻居无需排序：桥集是图不变量，Tarjan 对任意 DFS 邻居顺序返回相同结果
            stack = [(start_site, iter(site_neighbors.get(start_site, ())))]

            while stack:
                site_id, neighbors_iter = stack[-1]

                try:
                    neighbor_site = next(neighbors_iter)
                except StopIteration:
                    stack.pop()
                    parent_site = parent.get(site_id)
                    if parent_site is not None:
                        low_link[parent_site] = min(low_link[parent_site], low_link[site_id])
                        if low_link[site_id] > discovery_time[parent_site]:
                            bridges.add(tuple(sorted((parent_site, site_id))))
                    continue

                if neighbor_site == parent.get(site_id):
                    continue

                if neighbor_site not in discovery_time:
                    parent[neighbor_site] = site_id
                    current_time += 1
                    discovery_time[neighbor_site] = low_link[neighbor_site] = current_time
                    progress.update()
                    stack.append((neighbor_site, iter(site_neighbors.get(neighbor_site, ()))))
                else:
                    low_link[site_id] = min(low_link[site_id], discovery_time[neighbor_site])

    return bridges


def build_pairwise_graph_metrics(inputs, show_progress=False):
    bridge_pairs = find_bridge_pairs(
        inputs["all_sites"],
        inputs["site_neighbors"],
        show_progress=show_progress,
    )
    pair_graph_metrics = {}

    with ProgressReporter(len(inputs["pair_edge_count"]), "pairwise: 计算站点对图指标", show_progress) as progress:
        for left_site, right_site in sorted(inputs["pair_edge_count"].keys()):
            progress.update()
            pair_key = tuple(sorted((left_site, right_site)))
            left_neighbors = set(inputs["site_neighbors"].get(left_site, ()))
            right_neighbors = set(inputs["site_neighbors"].get(right_site, ()))
            left_neighbors.discard(right_site)
            right_neighbors.discard(left_site)
            shared_neighbors = sorted(left_neighbors & right_neighbors)

            is_bridge = pair_key in bridge_pairs
            pair_graph_metrics[pair_key] = {
                "is_bridge": is_bridge,
                "has_alternative_path": not is_bridge,
                "shared_neighbor_count": len(shared_neighbors),
                "shared_neighbors": shared_neighbors,
            }

    return pair_graph_metrics


def compute_base_core_score(site_id, inputs, args):
    domain_counts = inputs["site_domain_counts"].get(site_id, Counter())
    has_data = 1.0 if domain_counts.get("Data", 0) > 0 else 0.0
    data_ne_count = float(domain_counts.get("Data", 0))
    neighbor_count = float(len(inputs["site_neighbors"].get(site_id, ())))
    external_edge_count = float(inputs["site_external_edge_count"].get(site_id, 0))
    data_edge_count = float(inputs["site_data_edge_count"].get(site_id, 0))
    transmission_edge_count = float(
        inputs["site_transmission_edge_count"].get(site_id, 0)
    )

    score = 0.0
    score += args.data_site_bonus * has_data
    score += args.data_ne_weight * data_ne_count
    score += args.neighbor_weight * neighbor_count
    score += args.external_edge_weight * external_edge_count
    score += args.data_edge_weight * data_edge_count
    score += args.transmission_edge_weight * transmission_edge_count
    return score


def select_component_anchors(component_sites, inputs, base_scores, args):
    data_sites = [
        site_id
        for site_id in component_sites
        if inputs["site_domain_counts"].get(site_id, Counter()).get("Data", 0) > 0
    ]
    ranked_candidates = sorted(
        data_sites if data_sites else component_sites,
        key=lambda site_id: (
            base_scores[site_id],
            len(inputs["site_neighbors"].get(site_id, ())),
            site_id,
        ),
        reverse=True,
    )

    if not ranked_candidates:
        return []

    anchors = []
    max_score = base_scores[ranked_candidates[0]]
    threshold = max_score * args.anchor_score_ratio

    for site_id in ranked_candidates:
        if len(anchors) >= args.max_anchor_sites_per_component:
            break
        if data_sites:
            if base_scores[site_id] + 1e-9 >= threshold:
                anchors.append(site_id)
        elif not anchors:
            anchors.append(site_id)

    if not anchors:
        anchors.append(ranked_candidates[0])

    return anchors


def compute_component_core_distance(component_sites, anchors, site_neighbors):
    distance_map = {site_id: None for site_id in component_sites}
    if not anchors:
        return distance_map

    queue = deque()
    for anchor_site in anchors:
        if anchor_site not in distance_map:
            continue
        distance_map[anchor_site] = 0
        queue.append(anchor_site)

    while queue:
        current_site = queue.popleft()
        current_distance = distance_map[current_site]
        for neighbor_site in site_neighbors.get(current_site, ()):
            if neighbor_site not in distance_map:
                continue
            if distance_map[neighbor_site] is not None:
                continue
            distance_map[neighbor_site] = current_distance + 1
            queue.append(neighbor_site)

    return distance_map


def smooth_level_scores(
    site_metrics,
    site_neighbors,
    anchor_sites,
    alpha,
    max_iter,
    tol,
    constraints=None,
    constraint_gap=1.0,
    show_progress=False,
):
    """对站点 level_score 做标签传播平滑，得到全局一致的层级场。

    - anchor 站点被钉住（保持原始 level_score），充当稳定参照系；
    - 其余站点按 new = alpha * 自身原始分 + (1-alpha) * 邻居均值 迭代，直到收敛。
    - 传入 constraints（跨类型方向约束）时，每轮迭代后做一次软投影：对每条约束
      upstream->downstream 保证 level(up) >= level(down) + constraint_gap，不足则
      抬升上行侧/压低下行侧各半（锚点侧不动）。抬压量随平滑扩散到邻居，让约束
      周边站点的层级顺势重排。收尾再用“只抬升上行侧”的单调投影补齐链式约束
      （A->B->C 需逐级传导）。

    注意自锚项用的是**原始 level_score**（固定值，软 Dirichlet 约束），而非当前迭代值；
    否则不动点退化为纯调和场，非 anchor 站点会全部塌缩到最近 anchor，抹掉层级落差。

    复杂度 O(max_iter * (E + C))，纯线性迭代，无路径枚举/优化器。

    Returns:
        (scores, stats)；stats 含 unsatisfied_constraint_count——收尾后仍不满足的
        约束数，仅在两端都被锚点钉死等无法调整的情形下非零。
    """
    base_level = {site_id: metrics["level_score"] for site_id, metrics in site_metrics.items()}
    scores = dict(base_level)
    anchors = set(anchor_sites)

    constraint_pairs = []
    if constraints:
        for pair_key in sorted(constraints):
            constraint = constraints[pair_key]
            upstream_site = constraint["upstream_site"]
            downstream_site = constraint["downstream_site"]
            if upstream_site in scores and downstream_site in scores:
                constraint_pairs.append((upstream_site, downstream_site))

    def _project_constraints(target_scores):
        """软投影：缺口两侧各补一半；锚点侧不动，双锚点无法调整则跳过。"""
        for upstream_site, downstream_site in constraint_pairs:
            deficit = (
                target_scores[downstream_site] + constraint_gap
                - target_scores[upstream_site]
            )
            if deficit <= 0.0:
                continue
            upstream_pinned = upstream_site in anchors
            downstream_pinned = downstream_site in anchors
            if upstream_pinned and downstream_pinned:
                continue
            if upstream_pinned:
                target_scores[downstream_site] -= deficit
            elif downstream_pinned:
                target_scores[upstream_site] += deficit
            else:
                half = deficit / 2.0
                target_scores[upstream_site] += half
                target_scores[downstream_site] -= half

    if constraint_pairs:
        _project_constraints(scores)

    # 预先算好每个非 anchor 站点的有效邻居列表，避免每轮重复过滤/成员判断
    pending = {}
    for site_id in scores:
        if site_id in anchors:
            continue
        neighbors = [n for n in site_neighbors.get(site_id, ()) if n in scores]
        if neighbors:
            pending[site_id] = neighbors

    with ProgressReporter(max_iter, "pairwise: 平滑层级分", show_progress) as progress:
        for _ in range(max_iter):
            progress.update()
            new_scores = dict(scores)
            max_delta = 0.0
            for site_id, neighbors in pending.items():
                neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
                value = alpha * base_level[site_id] + (1.0 - alpha) * neighbor_avg
                new_scores[site_id] = value
                max_delta = max(max_delta, abs(value - scores[site_id]))
            if constraint_pairs:
                _project_constraints(new_scores)
            scores = new_scores
            if max_delta < tol:
                break

    unsatisfied_constraint_count = 0
    if constraint_pairs:
        # 收尾单调投影：优先抬升上行侧（上行被锚点钉住则压低下行侧），
        # 反复扫描直至稳定，保证链式约束在最终场上逐级满足。
        for _ in range(min(len(constraint_pairs), 200) + 1):
            changed = False
            for upstream_site, downstream_site in constraint_pairs:
                deficit = (
                    scores[downstream_site] + constraint_gap - scores[upstream_site]
                )
                if deficit <= 1e-9:
                    continue
                if upstream_site not in anchors:
                    scores[upstream_site] += deficit
                elif downstream_site not in anchors:
                    scores[downstream_site] -= deficit
                else:
                    continue
                changed = True
            if not changed:
                break
        unsatisfied_constraint_count = sum(
            1
            for upstream_site, downstream_site in constraint_pairs
            if scores[upstream_site] < scores[downstream_site] + constraint_gap - 1e-6
        )

    return scores, {"unsatisfied_constraint_count": unsatisfied_constraint_count}


def build_pairwise_site_metrics(inputs, args, show_progress=False, constraints=None):
    base_scores = {
        site_id: compute_base_core_score(site_id, inputs, args)
        for site_id in inputs["all_sites"]
    }
    components = compute_connected_components(
        inputs["all_sites"],
        inputs["site_neighbors"],
        show_progress=show_progress,
    )

    site_metrics = {}
    component_summaries = []
    with ProgressReporter(len(components), "pairwise: 计算分量层级指标", show_progress) as progress:
        for component_index, component_sites in enumerate(components):
            progress.update()
            anchors = select_component_anchors(component_sites, inputs, base_scores, args)
            distance_map = compute_component_core_distance(
                component_sites,
                anchors,
                inputs["site_neighbors"],
            )
            component_summaries.append(
                {
                    "component_id": component_index,
                    "site_count": len(component_sites),
                    "anchor_sites": anchors,
                }
            )

            for site_id in component_sites:
                core_distance = distance_map.get(site_id)
                level_score = base_scores[site_id]
                if core_distance is not None:
                    level_score -= args.core_distance_penalty * core_distance
                site_metrics[site_id] = {
                    "component_id": component_index,
                    "domain_counts": dict(inputs["site_domain_counts"].get(site_id, {})),
                    "neighbor_count": len(inputs["site_neighbors"].get(site_id, ())),
                    "external_edge_count": int(
                        inputs["site_external_edge_count"].get(site_id, 0)
                    ),
                    "data_edge_count": int(inputs["site_data_edge_count"].get(site_id, 0)),
                    "transmission_edge_count": int(
                        inputs["site_transmission_edge_count"].get(site_id, 0)
                    ),
                    "base_core_score": round(base_scores[site_id], 6),
                    "core_distance": core_distance,
                    "level_score": round(level_score, 6),
                }

    # 标签传播平滑：得到全局一致的层级场，供 gap-first 定向与严格环入口排序使用
    anchor_sites = set()
    for summary in component_summaries:
        anchor_sites.update(summary["anchor_sites"])
    smoothed, smoothing_stats = smooth_level_scores(
        site_metrics,
        inputs["site_neighbors"],
        anchor_sites,
        args.smooth_alpha,
        args.smooth_iters,
        args.smooth_tol,
        constraints=constraints,
        constraint_gap=float(getattr(args, "constraint_level_gap", 1.0)),
        show_progress=show_progress,
    )
    for site_id, metrics in site_metrics.items():
        metrics["level_score_smoothed"] = round(smoothed[site_id], 6)

    return site_metrics, component_summaries, smoothing_stats


def _add_direction_evidence(score_container, winner, loser, feature, amount, detail):
    score_container[winner]["score"] += amount
    score_container[winner]["breakdown"].append(
        {
            "feature": feature,
            "amount": round(amount, 6),
            "detail": detail,
            "towards": f"{winner}->{loser}",
        }
    )


def build_global_gap_result(
    left_site, right_site, left_level, right_level, gap, effective_threshold, graph_metrics, args
):
    """全局层级差决定方向时的结果，字段对齐常规 evaluate 输出。"""
    if gap > 0:
        preferred_source, preferred_target = left_site, right_site
    else:
        preferred_source, preferred_target = right_site, left_site
    return {
        "site_a": left_site,
        "site_b": right_site,
        "relation": "->",
        "preferred_source": preferred_source,
        "preferred_target": preferred_target,
        "score_a_to_b": round(max(gap, 0.0), 6),
        "score_b_to_a": round(max(-gap, 0.0), 6),
        "score_gap": round(gap, 6),
        "decision_margin": round(effective_threshold, 6),
        "base_global_gap_threshold": args.global_gap_threshold,
        "base_direction_margin": args.direction_margin,
        "decision_method": "global_level_gap",
        "level_smoothed_a": round(left_level, 6),
        "level_smoothed_b": round(right_level, 6),
        "is_bridge": graph_metrics["is_bridge"],
        "has_alternative_path": graph_metrics["has_alternative_path"],
        "shared_neighbor_count": graph_metrics["shared_neighbor_count"],
        "shared_neighbors": graph_metrics["shared_neighbors"],
        "uncertainty_adjustments": [],
        "score_breakdown_a_to_b": [],
        "score_breakdown_b_to_a": [],
    }


def _vote_core_distance(score_container, left_site, right_site, left_metrics, right_metrics, args):
    """特征投票：谁离汇聚 anchor 更近。"""
    left_distance = left_metrics.get("core_distance")
    right_distance = right_metrics.get("core_distance")
    if left_distance is None or right_distance is None or left_distance == right_distance:
        return
    delta = min(abs(left_distance - right_distance), args.max_core_distance_delta)
    amount = delta * args.core_distance_weight
    if left_distance < right_distance:
        _add_direction_evidence(
            score_container,
            left_site,
            right_site,
            "core_distance",
            amount,
            f"{left_site} 更接近汇聚 anchor ({left_distance} < {right_distance})",
        )
    else:
        _add_direction_evidence(
            score_container,
            right_site,
            left_site,
            "core_distance",
            amount,
            f"{right_site} 更接近汇聚 anchor ({right_distance} < {left_distance})",
        )


def _vote_score_gaps(score_container, left_site, right_site, left_metrics, right_metrics, args):
    """特征投票：层级分与汇聚候选分差距。"""
    level_gap = left_metrics["level_score"] - right_metrics["level_score"]
    if abs(level_gap) >= args.min_level_gap:
        amount = min(abs(level_gap), args.max_level_score_delta) * args.level_score_weight
        if level_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "level_score",
                amount,
                f"{left_site} 层级分更高 ({left_metrics['level_score']:.3f} > {right_metrics['level_score']:.3f})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "level_score",
                amount,
                f"{right_site} 层级分更高 ({right_metrics['level_score']:.3f} > {left_metrics['level_score']:.3f})",
            )

    base_score_gap = left_metrics["base_core_score"] - right_metrics["base_core_score"]
    if abs(base_score_gap) >= args.min_base_score_gap:
        amount = min(abs(base_score_gap), args.max_base_score_delta) * args.base_score_weight
        if base_score_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "base_core_score",
                amount,
                f"{left_site} 汇聚候选分更高 "
                f"({left_metrics['base_core_score']:.3f} > {right_metrics['base_core_score']:.3f})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "base_core_score",
                amount,
                f"{right_site} 汇聚候选分更高 "
                f"({right_metrics['base_core_score']:.3f} > {left_metrics['base_core_score']:.3f})",
            )


def _vote_data_domains(
    score_container, left_site, right_site, left_metrics, right_metrics,
    pair_domain_counts, args,
):
    """特征投票：Data 设备存在性与站点对连接中的 Data 暴露量。"""
    left_has_data = left_metrics["domain_counts"].get("Data", 0) > 0
    right_has_data = right_metrics["domain_counts"].get("Data", 0) > 0
    if left_has_data != right_has_data:
        if left_has_data:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "data_presence",
                args.data_presence_weight,
                f"{left_site} 存在 Data 设备而 {right_site} 不存在",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "data_presence",
                args.data_presence_weight,
                f"{right_site} 存在 Data 设备而 {left_site} 不存在",
            )

    left_pair_data = pair_domain_counts.get(left_site, Counter()).get("Data", 0)
    right_pair_data = pair_domain_counts.get(right_site, Counter()).get("Data", 0)
    if left_pair_data != right_pair_data:
        amount = min(abs(left_pair_data - right_pair_data), args.max_pair_domain_delta)
        amount *= args.pair_data_weight
        if left_pair_data > right_pair_data:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "pair_data_exposure",
                amount,
                f"{left_site} 在该站点对连接中出现更多 Data 设备 ({left_pair_data} > {right_pair_data})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "pair_data_exposure",
                amount,
                f"{right_site} 在该站点对连接中出现更多 Data 设备 ({right_pair_data} > {left_pair_data})",
            )


def _vote_topology_shape(score_container, left_site, right_site, left_metrics, right_metrics, args):
    """特征投票：邻接站点数量与叶子站点倾向。"""
    neighbor_gap = left_metrics["neighbor_count"] - right_metrics["neighbor_count"]
    if abs(neighbor_gap) >= args.min_neighbor_gap:
        amount = min(abs(neighbor_gap), args.max_neighbor_delta) * args.neighbor_direction_weight
        if neighbor_gap > 0:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "neighbor_count",
                amount,
                f"{left_site} 的邻接站点更多 ({left_metrics['neighbor_count']} > {right_metrics['neighbor_count']})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "neighbor_count",
                amount,
                f"{right_site} 的邻接站点更多 ({right_metrics['neighbor_count']} > {left_metrics['neighbor_count']})",
            )

    left_is_leaf = left_metrics["neighbor_count"] <= args.leaf_neighbor_threshold
    right_is_leaf = right_metrics["neighbor_count"] <= args.leaf_neighbor_threshold
    if left_is_leaf != right_is_leaf:
        if right_is_leaf:
            _add_direction_evidence(
                score_container,
                left_site,
                right_site,
                "leaf_bias",
                args.leaf_bias_weight,
                f"{right_site} 更像叶子站点 (neighbor_count <= {args.leaf_neighbor_threshold})",
            )
        else:
            _add_direction_evidence(
                score_container,
                right_site,
                left_site,
                "leaf_bias",
                args.leaf_bias_weight,
                f"{left_site} 更像叶子站点 (neighbor_count <= {args.leaf_neighbor_threshold})",
            )


def _effective_direction_margin(graph_metrics, args):
    """按桥边/共享邻居情况放大判向 margin，返回 (margin, 不确定性说明)。"""
    effective_margin = args.direction_margin
    uncertainty_reasons = []
    if not graph_metrics["is_bridge"]:
        effective_margin += args.non_bridge_margin_bonus
        uncertainty_reasons.append(
            {
                "feature": "non_bridge",
                "amount": round(args.non_bridge_margin_bonus, 6),
                "detail": "该站点对存在替代路径，环路/多路径下方向判断需更保守",
            }
        )
    shared_neighbor_bonus = (
        min(
            graph_metrics["shared_neighbor_count"],
            args.max_shared_neighbor_bonus_count,
        )
        * args.shared_neighbor_margin_bonus
    )
    if shared_neighbor_bonus > 0:
        effective_margin += shared_neighbor_bonus
        uncertainty_reasons.append(
            {
                "feature": "shared_neighbors",
                "amount": round(shared_neighbor_bonus, 6),
                "detail": f"两端共享 {graph_metrics['shared_neighbor_count']} 个邻居站点，局部多路径更明显",
            }
        )
    return effective_margin, uncertainty_reasons


def evaluate_pair_direction(left_site, right_site, site_metrics, inputs, pair_graph_metrics, args):
    left_metrics = site_metrics[left_site]
    right_metrics = site_metrics[right_site]
    pair_key = tuple(sorted((left_site, right_site)))
    pair_domain_counts = inputs["pair_site_domain_counts"].get(pair_key, {})
    graph_metrics = pair_graph_metrics.get(
        pair_key,
        {
            "is_bridge": False,
            "has_alternative_path": True,
            "shared_neighbor_count": 0,
            "shared_neighbors": [],
        },
    )

    # 全局层级场清晰时直接按层级差定向，跳过局部 7 特征投票。
    # 短路阈值与投票的 effective_margin 一样对非桥边/共享邻居更保守：层级差不够大时
    # 不短路，交回投票，由其保守 margin 把不确定的环路/多路径边判成双向。
    left_level = left_metrics.get("level_score_smoothed", left_metrics["level_score"])
    right_level = right_metrics.get("level_score_smoothed", right_metrics["level_score"])
    gap = left_level - right_level
    effective_threshold = args.global_gap_threshold
    if not graph_metrics["is_bridge"]:
        effective_threshold += args.global_gap_nonbridge_bonus
    effective_threshold += (
        min(graph_metrics["shared_neighbor_count"], args.max_shared_neighbor_bonus_count)
        * args.global_gap_shared_neighbor_bonus
    )
    if gap != 0.0 and abs(gap) >= effective_threshold:
        return build_global_gap_result(
            left_site, right_site, left_level, right_level, gap,
            effective_threshold, graph_metrics, args,
        )

    score_container = {
        left_site: {"score": 0.0, "breakdown": []},
        right_site: {"score": 0.0, "breakdown": []},
    }

    _vote_core_distance(score_container, left_site, right_site, left_metrics, right_metrics, args)
    _vote_score_gaps(score_container, left_site, right_site, left_metrics, right_metrics, args)
    _vote_data_domains(
        score_container, left_site, right_site, left_metrics, right_metrics,
        pair_domain_counts, args,
    )
    _vote_topology_shape(score_container, left_site, right_site, left_metrics, right_metrics, args)

    left_score = score_container[left_site]["score"]
    right_score = score_container[right_site]["score"]
    score_gap = left_score - right_score
    effective_margin, uncertainty_reasons = _effective_direction_margin(graph_metrics, args)

    if abs(score_gap) < effective_margin:
        relation = "<->"
        preferred_source = None
        preferred_target = None
    elif score_gap > 0:
        relation = "->"
        preferred_source = left_site
        preferred_target = right_site
    else:
        relation = "->"
        preferred_source = right_site
        preferred_target = left_site

    return {
        "site_a": left_site,
        "site_b": right_site,
        "relation": relation,
        "preferred_source": preferred_source,
        "preferred_target": preferred_target,
        "score_a_to_b": round(left_score, 6),
        "score_b_to_a": round(right_score, 6),
        "score_gap": round(score_gap, 6),
        "decision_margin": effective_margin,
        "base_direction_margin": args.direction_margin,
        "pair_edge_count": int(inputs["pair_edge_count"].get(pair_key, 0)),
        "is_bridge": graph_metrics["is_bridge"],
        "has_alternative_path": graph_metrics["has_alternative_path"],
        "shared_neighbor_count": graph_metrics["shared_neighbor_count"],
        "shared_neighbors": graph_metrics["shared_neighbors"],
        "uncertainty_adjustments": uncertainty_reasons,
        "score_breakdown_a_to_b": score_container[left_site]["breakdown"],
        "score_breakdown_b_to_a": score_container[right_site]["breakdown"],
    }


def apply_cross_domain_constraint_override(pair_result, constraint):
    """按跨类型方向约束覆盖 pairwise 输出：强制 upstream->downstream。

    约束优先级高于严格环与投票结果；原判定保留在 original_* 字段。
    """
    upstream_site = constraint["upstream_site"]
    downstream_site = constraint["downstream_site"]
    changed = (
        pair_result.get("relation") != "->"
        or pair_result.get("preferred_source") != upstream_site
        or pair_result.get("preferred_target") != downstream_site
    )
    updated = dict(pair_result)
    updated["original_relation"] = pair_result.get("relation")
    updated["original_preferred_source"] = pair_result.get("preferred_source")
    updated["original_preferred_target"] = pair_result.get("preferred_target")
    updated["relation"] = "->"
    updated["preferred_source"] = upstream_site
    updated["preferred_target"] = downstream_site
    updated["cross_domain_constraint"] = True
    updated["cross_domain_constraint_evidence_link_count"] = constraint[
        "evidence_link_count"
    ]
    updated["uncertainty_adjustments"] = list(
        updated.get("uncertainty_adjustments", [])
    ) + [{
        "feature": "cross_domain_priority_constraint",
        "amount": 0.0,
        "detail": (
            f"跨类型连边约束：{upstream_site} 端设备 domain 优先级更高，"
            f"强制 {upstream_site}->{downstream_site}"
        ),
    }]
    return updated, changed


def build_pairwise_orders(
    inputs,
    site_metrics,
    pair_graph_metrics,
    args,
    show_progress=False,
    compact_output=False,
    constraints=None,
):
    pair_orders = {}
    compact_edges = [] if compact_output else None
    downstream_map = defaultdict(set)
    before_directed_pair_count = 0
    before_bidirectional_pair_count = 0
    directed_pair_count = 0
    bidirectional_pair_count = 0
    strict_ring_context = {"pair_context": {}, "components": []}
    strict_ring_forced_pair_count = 0
    strict_ring_entry_direction_pair_count = 0
    strict_ring_changed_pair_count = 0

    strict_ring_context = build_strict_ring_context(
        inputs["pair_edge_count"].keys(),
        [
            pair_key
            for pair_key, graph_metrics in pair_graph_metrics.items()
            if graph_metrics.get("is_bridge")
        ],
        site_scores={
            # 与 gap-first 一致地使用平滑层级
            site_id: metrics.get(
                "level_score_smoothed", metrics.get("level_score", 0.0)
            )
            for site_id, metrics in site_metrics.items()
        },
    )
    strict_ring_pair_context = strict_ring_context["pair_context"]

    # 环块解除（constraint_ring_release，默认关闭）：约束两端同环块时解除该块的
    # 严格环覆盖。默认不启用——环块会把共边环与双归下游一并融合进来，
    # “约束所在的环”在共边网状结构上没有唯一定义，解除范围无法正确圈定；
    # 约束仍通过直连对硬覆盖与势场投影生效。
    constraints = constraints or {}
    constraint_ring_release = bool(getattr(args, "constraint_ring_release", False))
    # 直连约束对硬覆盖开关：关闭后约束不再改写任何边的判向（观察/评估模式）
    constraint_hard_override = bool(getattr(args, "constraint_hard_override", True))
    constraint_forced_pair_count = 0
    constraint_changed_pair_count = 0
    strict_ring_released_pair_count = 0
    released_component_ids = set()
    if constraints and constraint_ring_release:
        site_component_ids = {}
        for component in strict_ring_context["components"]:
            for component_site in component["sites"]:
                site_component_ids[component_site] = component["component_id"]
        for constraint in constraints.values():
            upstream_component_id = site_component_ids.get(constraint["upstream_site"])
            if upstream_component_id is None:
                continue
            if upstream_component_id == site_component_ids.get(
                constraint["downstream_site"]
            ):
                released_component_ids.add(upstream_component_id)
        for component in strict_ring_context["components"]:
            if component["component_id"] in released_component_ids:
                component["released_by_constraint"] = True
    if released_component_ids:
        retained_pair_context = {}
        for ring_pair_key, ring_context in strict_ring_pair_context.items():
            if ring_context["component_id"] in released_component_ids:
                strict_ring_released_pair_count += 1
            else:
                retained_pair_context[ring_pair_key] = ring_context
        strict_ring_pair_context = retained_pair_context

    with ProgressReporter(len(inputs["pair_edge_count"]), "pairwise: 判断站点对方向", show_progress) as progress:
        for left_site, right_site in sorted(inputs["pair_edge_count"].keys()):
            progress.update()
            pair_key = tuple(sorted((left_site, right_site)))
            pair_result = evaluate_pair_direction(
                left_site,
                right_site,
                site_metrics,
                inputs,
                pair_graph_metrics,
                args,
            )
            if pair_result.get("relation") == "<->":
                before_bidirectional_pair_count += 1
            else:
                before_directed_pair_count += 1
            ring_pair_context = strict_ring_pair_context.get(pair_key)
            pair_result, strict_ring_changed = apply_strict_ring_pairwise_override(
                pair_result,
                ring_pair_context,
            )
            if ring_pair_context:
                if ring_pair_context.get("force_bidirectional"):
                    strict_ring_forced_pair_count += 1
                elif ring_pair_context.get("force_entry_direction"):
                    strict_ring_entry_direction_pair_count += 1
                if strict_ring_changed:
                    strict_ring_changed_pair_count += 1
            constraint_info = constraints.get(pair_key)
            if constraint_info is not None and constraint_hard_override:
                pair_result, constraint_changed = apply_cross_domain_constraint_override(
                    pair_result,
                    constraint_info,
                )
                constraint_forced_pair_count += 1
                if constraint_changed:
                    constraint_changed_pair_count += 1
            if compact_output:
                compact_edges.append(compact_pairwise_prediction(pair_result))
            else:
                pair_orders[f"{left_site}||{right_site}"] = pair_result

            relation = pair_result["relation"]
            if relation == "<->":
                bidirectional_pair_count += 1
                downstream_map[left_site].add(right_site)
                downstream_map[right_site].add(left_site)
            else:
                directed_pair_count += 1
                downstream_map[pair_result["preferred_source"]].add(
                    pair_result["preferred_target"]
                )

    output = {
        "pair_orders": pair_orders,
        "downstream_map": {
            site_id: sorted(neighbors)
            for site_id, neighbors in sorted(downstream_map.items())
        },
        "before_directed_pair_count": before_directed_pair_count,
        "before_bidirectional_pair_count": before_bidirectional_pair_count,
        "directed_pair_count": directed_pair_count,
        "bidirectional_pair_count": bidirectional_pair_count,
        "strict_ring_components": strict_ring_context["components"],
        "strict_ring_forced_pair_count": strict_ring_forced_pair_count,
        "strict_ring_entry_direction_pair_count": strict_ring_entry_direction_pair_count,
        "strict_ring_changed_pair_count": strict_ring_changed_pair_count,
        "strict_ring_released_component_count": len(released_component_ids),
        "strict_ring_released_pair_count": strict_ring_released_pair_count,
        "cross_domain_constraint_forced_pair_count": constraint_forced_pair_count,
        "cross_domain_constraint_changed_pair_count": constraint_changed_pair_count,
    }
    if compact_output:
        output["compact_edges"] = compact_edges
    return output


def build_pairwise_meta(args, inputs, component_summaries, pair_outputs, bridge_pair_count, pair_graph_metrics):
    """组装 pairwise 输出的 meta（main 与内存调用共用，避免字段漂移）。"""
    return {
        "algorithm": "pairwise_evidence",
        "ne_graph": args.ne_graph,
        "site_count": len(inputs["all_sites"]),
        "adjacent_pair_count": len(inputs["pair_edge_count"]),
        "component_count": len(component_summaries),
        "strict_ring_before_directed_pair_count": pair_outputs["before_directed_pair_count"],
        "strict_ring_before_bidirectional_pair_count": pair_outputs["before_bidirectional_pair_count"],
        "directed_pair_count": pair_outputs["directed_pair_count"],
        "bidirectional_pair_count": pair_outputs["bidirectional_pair_count"],
        "bridge_pair_count": bridge_pair_count,
        "non_bridge_pair_count": len(pair_graph_metrics) - bridge_pair_count,
        "smooth_alpha": args.smooth_alpha,
        "smooth_iters": args.smooth_iters,
        "smooth_tol": args.smooth_tol,
        "global_gap_threshold": args.global_gap_threshold,
        "global_gap_nonbridge_bonus": args.global_gap_nonbridge_bonus,
        "global_gap_shared_neighbor_bonus": args.global_gap_shared_neighbor_bonus,
        "strict_ring_component_count": len(pair_outputs["strict_ring_components"]),
        "strict_ring_forced_pair_count": pair_outputs["strict_ring_forced_pair_count"],
        "strict_ring_entry_direction_pair_count": pair_outputs["strict_ring_entry_direction_pair_count"],
        "strict_ring_changed_pair_count": pair_outputs["strict_ring_changed_pair_count"],
    }


def build_pairwise_prediction(ne_graph, args, show_progress=False):
    """在内存中跑完整 pairwise 流程，返回 compact 输出（meta / edges / downstream_map）。

    与 main 的非 full-output 路径产物一致，可直接作为下游 site_chains 的 prediction 输入，
    省去落盘再读盘。

    args.cross_domain_priority_constraint 开启时，跨类型连边（如 Data-Ran）在
    连边端点优先级与站点自身优先级方向一致时产生硬方向约束：注入层级平滑
    （周边站点顺势重排）、解除含约束端点的严格环块、并对有拓扑直连边的约束对
    强制判向；约束集随结果的 cross_domain_constraints 字段导出，供 site_chains
    侧校验。
    """
    constraint_enabled = bool(getattr(args, "cross_domain_priority_constraint", False))

    # 预处理：疑似误连接的站点对（Data 站点 <-> Trans+Ran 站点之间只有传输连边、
    # 无 Data-Ran 佐证），其传输类连边从拓扑/约束/裁剪口径整体剔除
    misconnection_enabled = bool(
        getattr(args, "transmission_misconnection_filter", False)
    )
    misconnection_pairs = set()
    misconnection_stats = {"candidate_pair_count": 0, "misconnection_pair_count": 0}
    if misconnection_enabled:
        misconnection_pairs, misconnection_stats = (
            build_transmission_misconnection_pairs(ne_graph)
        )

    inputs = build_site_pair_inputs(
        ne_graph,
        show_progress=show_progress,
        collect_cross_domain=constraint_enabled,
        transmission_misconnection_pairs=misconnection_pairs,
    )
    constraints = {}
    constraint_stats = {"tie_pair_count": 0, "cycle_dropped_pair_count": 0}
    if constraint_enabled:
        constraints, constraint_stats = build_cross_domain_constraints(
            inputs["cross_domain_pair_evidence"],
            inputs["pair_edge_count"],
        )
    pair_graph_metrics = build_pairwise_graph_metrics(inputs, show_progress=show_progress)
    bridge_pair_count = sum(
        1 for graph_metrics in pair_graph_metrics.values() if graph_metrics["is_bridge"]
    )
    # 势场投影开关：关闭后约束不再注入层级平滑（多跳约束将失去全部判向影响）
    constraint_level_projection = bool(
        getattr(args, "constraint_level_projection", True)
    )
    site_metrics, component_summaries, smoothing_stats = build_pairwise_site_metrics(
        inputs,
        args,
        show_progress=show_progress,
        constraints=constraints if constraint_level_projection else None,
    )
    pair_outputs = build_pairwise_orders(
        inputs,
        site_metrics,
        pair_graph_metrics,
        args,
        show_progress=show_progress,
        compact_output=True,
        constraints=constraints,
    )
    meta = build_pairwise_meta(
        args, inputs, component_summaries, pair_outputs, bridge_pair_count, pair_graph_metrics
    )
    meta["cross_domain_priority_constraint"] = constraint_enabled
    meta["transmission_misconnection_filter"] = misconnection_enabled
    if misconnection_enabled:
        meta["transmission_misconnection_candidate_pair_count"] = (
            misconnection_stats["candidate_pair_count"]
        )
        meta["transmission_misconnection_pair_count"] = (
            misconnection_stats["misconnection_pair_count"]
        )
    if constraint_enabled:
        direct_pair_count = sum(
            1 for constraint in constraints.values() if constraint["has_topology_edge"]
        )
        meta.update({
            "cross_domain_constraint_hard_override": bool(
                getattr(args, "constraint_hard_override", True)
            ),
            "cross_domain_constraint_level_projection": constraint_level_projection,
            "cross_domain_constraint_ring_release": bool(
                getattr(args, "constraint_ring_release", False)
            ),
            "cross_domain_constraint_pair_count": len(constraints),
            "cross_domain_constraint_direct_pair_count": direct_pair_count,
            "cross_domain_constraint_multi_hop_pair_count": len(constraints) - direct_pair_count,
            "cross_domain_constraint_tie_pair_count": constraint_stats["tie_pair_count"],
            "cross_domain_constraint_cycle_dropped_pair_count": constraint_stats["cycle_dropped_pair_count"],
            "cross_domain_constraint_level_unsatisfied_count": smoothing_stats["unsatisfied_constraint_count"],
            "cross_domain_constraint_forced_pair_count": pair_outputs["cross_domain_constraint_forced_pair_count"],
            "cross_domain_constraint_changed_pair_count": pair_outputs["cross_domain_constraint_changed_pair_count"],
            "strict_ring_released_component_count": pair_outputs["strict_ring_released_component_count"],
            "strict_ring_released_pair_count": pair_outputs["strict_ring_released_pair_count"],
        })
    result = {
        "meta": meta,
        "edges": pair_outputs["compact_edges"],
        "downstream_map": pair_outputs["downstream_map"],
    }
    if constraint_enabled:
        result["cross_domain_constraints"] = [
            {
                "upstream_site": constraint["upstream_site"],
                "downstream_site": constraint["downstream_site"],
                "evidence_link_count": constraint["evidence_link_count"],
                "total_cross_link_count": constraint["total_cross_link_count"],
                "has_topology_edge": constraint["has_topology_edge"],
            }
            for _, constraint in sorted(constraints.items())
        ]
    if misconnection_enabled:
        # 随产物导出，供 site_chains 的 restrict 裁剪按同一口径剔除误连接
        result["transmission_misconnection_pairs"] = [
            list(pair_key) for pair_key in sorted(misconnection_pairs)
        ]
    return result
