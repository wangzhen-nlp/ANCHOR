#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ne_graph.json 生成相邻站点对的方向先验（global 无路径全局层级推断版）。

与 generate_site_pair_order_pairwise.py 的主要区别：
1. 先把站点映射到 wireless / microwave / router 三类抽象角色；
2. 选择 core / access anchors；
3. 在站点图上计算 distance score，并做全图平滑；
4. 最终再由站点全局分数反推边方向。
"""

import argparse
import json

from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from topology_resources import NE_GRAPH_JSON, resource_display, resource_path
from topology_tools.site_pair_order_common import (
    ROLE_SCORE,
    _get_site_id,
    build_downstream_map,
    classify_device_role,
    compute_distance_scores,
    extract_primary_upstream_map,
    find_bridges,
    iter_unique_cross_site_links,
    score_to_level,
)


# =========================
# 2) 从 NE 图构建站点图
# =========================

def build_site_topology(ne_graph):
    """
    将 ne_graph 聚合成站点级无向图。
    返回:
        site_stats: {
            site_id: {
                "site_id": ...,
                "nes": set(...),
                "role_counts": Counter(...),
                "degree": int,
                "neighbors": set(...),
                ...
            }
        }
        site_edges: {
            (site_a, site_b): {
                "site_a": ...,
                "site_b": ...,
                "ne_pairs": [(source_ne, target_ne), ...],
                "link_types": set(...),
                "link_count": int,
            }
        }
        adjacency: {site_id: set(neighbor_site_ids)}
    """
    site_stats = {}
    adjacency = defaultdict(set)
    site_edges = {}

    # 2.1 聚合站点内设备
    for ne_id, ne_info in ne_graph.items():
        if not isinstance(ne_info, dict):
            continue

        site_id = _get_site_id(ne_info)
        if not site_id:
            continue

        site_rec = site_stats.setdefault(site_id, {
            "site_id": site_id,
            "nes": set(),
            "role_counts": Counter(),
            "neighbors": set(),
            "degree": 0,
        })
        site_rec["nes"].add(ne_id)

        role = classify_device_role(ne_info.get("domain", ""))
        site_rec["role_counts"][role] += 1

    # 2.2 聚合跨站点链路
    for link in iter_unique_cross_site_links(ne_graph):
        a = link["source_site"]
        b = link["target_site"]
        if not a or not b or a == b:
            continue

        key = tuple(sorted((a, b)))
        rec = site_edges.setdefault(key, {
            "site_a": key[0],
            "site_b": key[1],
            "ne_pairs": [],
            "link_types": set(),
            "link_count": 0,
        })
        rec["ne_pairs"].append((link["source_ne"], link["target_ne"]))
        rec["link_types"].add(link.get("link_type", "__unknown__"))
        rec["link_count"] += 1

        adjacency[a].add(b)
        adjacency[b].add(a)

    # 2.3 写回 degree / neighbors
    for site_id, rec in site_stats.items():
        rec["neighbors"] = set(adjacency.get(site_id, set()))
        rec["degree"] = len(rec["neighbors"])

    return site_stats, site_edges, adjacency


# =========================
# 3) 站点初始层级先验
# =========================

def compute_site_priors(site_stats):
    """
    给每个站点计算:
    - raw_prior: 仅根据设备类型组成得到的初始分
    - anchor_strength: 该站点自身类型先验的可信度
    - predominant_role: 主导类型
    """
    for site_id, rec in site_stats.items():
        counts = rec["role_counts"]
        known_total = counts["wireless"] + counts["microwave"] + counts["router"]

        if known_total > 0:
            raw_prior = (
                counts["wireless"] * ROLE_SCORE["wireless"] +
                counts["microwave"] * ROLE_SCORE["microwave"] +
                counts["router"] * ROLE_SCORE["router"]
            ) / known_total
        else:
            raw_prior = 0.0

        degree = rec["degree"]
        wireless_ratio = counts["wireless"] / known_total if known_total else 0.0
        microwave_ratio = counts["microwave"] / known_total if known_total else 0.0
        router_ratio = counts["router"] / known_total if known_total else 0.0

        # 轻量拓扑修正:
        # - 无线叶子更偏下
        # - 路由汇聚点更偏上
        if degree <= 1 and wireless_ratio >= 0.5:
            raw_prior -= 0.5
        if degree >= 3 and router_ratio >= 0.5:
            raw_prior += 0.5

        # 主导类型
        predominant_role = "unknown"
        if known_total > 0:
            predominant_role = max(
                ("wireless", "microwave", "router"),
                key=lambda x: counts[x]
            )

        # 锚点强度：类型越纯，先验越可信
        purity = 0.0
        if known_total > 0:
            purity = max(wireless_ratio, microwave_ratio, router_ratio)

        if router_ratio >= 0.7 and wireless_ratio == 0:
            anchor_strength = 0.85
        elif wireless_ratio >= 0.7 and router_ratio == 0:
            anchor_strength = 0.85
        elif purity >= 0.6:
            anchor_strength = 0.65
        else:
            anchor_strength = 0.45

        rec["known_total"] = known_total
        rec["wireless_ratio"] = wireless_ratio
        rec["microwave_ratio"] = microwave_ratio
        rec["router_ratio"] = router_ratio
        rec["raw_prior"] = raw_prior
        rec["anchor_strength"] = anchor_strength
        rec["predominant_role"] = predominant_role


def select_anchor_sites(site_stats):
    """
    选择核心锚点 / 接入锚点。
    """
    core_anchors = []
    access_anchors = []

    for site_id, rec in site_stats.items():
        if rec["known_total"] == 0:
            continue

        # 核心锚点：路由占比较高
        if rec["router_ratio"] >= 0.6 and rec["role_counts"]["router"] > 0:
            core_anchors.append(site_id)

        # 接入锚点：无线占比较高，且通常更接近叶子
        if rec["wireless_ratio"] >= 0.6 and rec["role_counts"]["wireless"] > 0:
            access_anchors.append(site_id)

    # Fallback: 避免没有锚点
    all_sites = list(site_stats.keys())
    if not core_anchors and all_sites:
        core_anchors = [
            max(
                all_sites,
                key=lambda s: (
                    site_stats[s]["router_ratio"],
                    site_stats[s]["raw_prior"],
                    site_stats[s]["degree"],
                )
            )
        ]

    if not access_anchors and all_sites:
        access_anchors = [
            max(
                all_sites,
                key=lambda s: (
                    site_stats[s]["wireless_ratio"],
                    -site_stats[s]["raw_prior"],
                    -site_stats[s]["degree"],
                )
            )
        ]

    # 防止只剩一个站点同时当 core/access
    if set(core_anchors) == set(access_anchors) and len(all_sites) > 1:
        candidates = sorted(
            all_sites,
            key=lambda s: (
                site_stats[s]["wireless_ratio"],
                -site_stats[s]["raw_prior"],
                -site_stats[s]["degree"],
            ),
            reverse=True,
        )
        for s in candidates:
            if s not in core_anchors:
                access_anchors = [s]
                break

    return list(dict.fromkeys(core_anchors)), list(dict.fromkeys(access_anchors))


def smooth_site_scores(site_stats, adjacency, max_iter=100, tol=1e-4):
    """
    在站点图上做一个带锚点的平滑迭代:
        new = anchor_strength * base_score + (1-anchor_strength) * (0.7 * neighbor_avg + 0.3 * base_score)

    这样做的效果:
    - 类型先验不会丢
    - 拓扑位置会通过邻居传播
    """
    scores = {site: rec["base_score"] for site, rec in site_stats.items()}

    for _ in range(max_iter):
        new_scores = {}
        max_delta = 0.0

        for site, rec in site_stats.items():
            neighbors = adjacency.get(site, set())
            base = rec["base_score"]
            anchor_strength = rec["anchor_strength"]

            if neighbors:
                neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
                structural = 0.7 * neighbor_avg + 0.3 * base
            else:
                structural = base

            new_score = anchor_strength * base + (1.0 - anchor_strength) * structural
            new_scores[site] = new_score
            max_delta = max(max_delta, abs(new_score - scores[site]))

        scores = new_scores
        if max_delta < tol:
            break

    return scores


# =========================
# 7) 主函数：站点上下行预测
# =========================

def predict_site_directions_global(
    ne_graph,
    *,
    base_margin=0.35,
    ring_margin=0.75,
    same_role_margin=0.60,
):
    """
    基于设备类型 + 全局站点拓扑，预测站点间上下行关系。

    返回:
    {
        "sites": {
            site_id: {
                "score": ...,
                "level": "access/backhaul/core",
                "predominant_role": ...,
                "role_counts": ...,
                "degree": ...,
                ...
            }
        },
        "edges": [
            {
                "site_a": ...,
                "site_b": ...,
                "prediction": "A->B" / "B->A" / "bidirectional",
                "upstream_site": ... or None,
                "downstream_site": ... or None,
                "confidence": float in [0,1],
                "reasons": [...],
                "link_types": [...],
                "link_count": ...,
                "is_bridge": bool,
            },
            ...
        ]
    }

    约定:
    - A->B 表示 "A 向 B 上行"，即 B 更靠核心，A 更靠接入
    - 下行方向是反向
    - bidirectional 表示无法稳定判断，常见于同级环链/保护链路/证据不足
    """
    site_stats, site_edges, adjacency = build_site_topology(ne_graph)

    if not site_stats:
        return {"sites": {}, "edges": []}

    compute_site_priors(site_stats)
    core_anchors, access_anchors = select_anchor_sites(site_stats)

    distance_scores = compute_distance_scores(
        site_stats, adjacency, core_anchors, access_anchors
    )

    # 基础分 = 类型先验 + 距离先验
    for site_id, rec in site_stats.items():
        dist_score = distance_scores.get(site_id, 0.0)
        base_score = 0.7 * rec["raw_prior"] + 0.9 * dist_score
        rec["distance_score"] = dist_score
        rec["base_score"] = base_score

    final_scores = smooth_site_scores(site_stats, adjacency)
    bridges = find_bridges(adjacency)

    # 写回站点信息
    site_output = {}
    for site_id, rec in site_stats.items():
        score = final_scores.get(site_id, rec["base_score"])
        level = score_to_level(score)
        site_output[site_id] = {
            "score": round(score, 6),
            "level": level,
            "predominant_role": rec["predominant_role"],
            "role_counts": dict(rec["role_counts"]),
            "degree": rec["degree"],
            "raw_prior": round(rec["raw_prior"], 6),
            "distance_score": round(rec["distance_score"], 6),
            "anchor_strength": round(rec["anchor_strength"], 6),
            "neighbors": sorted(rec["neighbors"]),
            "is_core_anchor": site_id in core_anchors,
            "is_access_anchor": site_id in access_anchors,
        }

    # 边方向预测
    edge_output = []
    for key in sorted(site_edges.keys()):
        a, b = key
        edge_rec = site_edges[key]

        sa = final_scores.get(a, 0.0)
        sb = final_scores.get(b, 0.0)
        diff = sb - sa  # >0 表示 b 比 a 更靠核心

        level_a = site_output[a]["level"]
        level_b = site_output[b]["level"]
        role_a = site_output[a]["predominant_role"]
        role_b = site_output[b]["predominant_role"]

        same_level = (level_a == level_b)
        same_role = (role_a == role_b and role_a != "unknown")
        is_bridge = key in bridges

        # 不同情况使用不同阈值
        margin = base_margin
        if same_role:
            margin = max(margin, same_role_margin)
        if same_level and not is_bridge:
            margin = max(margin, ring_margin)

        reasons = []
        if same_role:
            reasons.append(f"same_predominant_role={role_a}")
        if same_level:
            reasons.append(f"same_level={level_a}")
        if not is_bridge:
            reasons.append("edge_in_cycle_or_ring_like")
        else:
            reasons.append("bridge_edge")

        if abs(diff) < margin:
            prediction = "bidirectional"
            upstream_site = None
            downstream_site = None
            confidence = max(0.05, min(0.50, abs(diff) / max(margin, 1e-6)))
            reasons.append(f"score_gap={diff:.3f} < margin={margin:.3f}")
        else:
            if diff > 0:
                # a -> b 为上行
                prediction = f"{a}->{b}"
                downstream_site = a
                upstream_site = b
            else:
                # b -> a 为上行
                prediction = f"{b}->{a}"
                downstream_site = b
                upstream_site = a

            # gap 越大，置信度越高
            confidence = min(0.99, abs(diff) / (margin + 1.5))
            reasons.append(f"score_gap={diff:.3f} >= margin={margin:.3f}")

        edge_output.append({
            "site_a": a,
            "site_b": b,
            "prediction": prediction,
            "upstream_site": upstream_site,
            "downstream_site": downstream_site,
            "confidence": round(confidence, 6),
            "reasons": reasons,
            "link_types": sorted(edge_rec["link_types"]),
            "link_count": edge_rec["link_count"],
            "ne_pairs": edge_rec["ne_pairs"],
            "is_bridge": is_bridge,
            "score_a": round(sa, 6),
            "score_b": round(sb, 6),
            "level_a": level_a,
            "level_b": level_b,
        })

    return {
        "sites": site_output,
        "edges": edge_output,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 ne_graph.json 生成相邻站点对的方向先验（global 无路径全局层级推断版）"
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=resource_path("site_pair_order_global.json"),
        help=f"输出 JSON，默认: {resource_display('site_pair_order_global.json')}",
    )
    parser.add_argument(
        "--base-margin",
        type=float,
        default=0.35,
        help="边方向基础判定门槛",
    )
    parser.add_argument(
        "--ring-margin",
        type=float,
        default=0.75,
        help="同层且非桥边时的更高判定门槛",
    )
    parser.add_argument(
        "--same-role-margin",
        type=float,
        default=0.60,
        help="同主导角色站点对的更高判定门槛",
    )
    args = parser.parse_args()

    if args.base_margin < 0:
        parser.error("base-margin 不能小于 0")
    if args.ring_margin < 0:
        parser.error("ring-margin 不能小于 0")
    if args.same_role_margin < 0:
        parser.error("same-role-margin 不能小于 0")
    return args


def main():
    args = parse_args()

    ne_graph_path = Path(args.ne_graph)
    if not ne_graph_path.exists():
        raise SystemExit(f"未找到 ne_graph.json: {args.ne_graph}")

    print(f"加载 ne_graph: {args.ne_graph}")
    with open(ne_graph_path, "r", encoding="utf-8") as f:
        ne_graph = json.load(f)

    prediction_result = predict_site_directions_global(
        ne_graph,
        base_margin=args.base_margin,
        ring_margin=args.ring_margin,
        same_role_margin=args.same_role_margin,
    )
    primary_upstream_map = extract_primary_upstream_map(prediction_result)
    downstream_map = build_downstream_map(prediction_result)

    bidirectional_edge_count = sum(
        1
        for edge in prediction_result["edges"]
        if edge.get("prediction") == "bidirectional"
    )
    directed_edge_count = len(prediction_result["edges"]) - bidirectional_edge_count
    bridge_edge_count = sum(
        1
        for edge in prediction_result["edges"]
        if edge.get("is_bridge")
    )

    print(f"站点数: {len(prediction_result['sites'])}")
    print(f"边数: {len(prediction_result['edges'])}")
    print(f"单向边数: {directed_edge_count}")
    print(f"双向边数: {bidirectional_edge_count}")
    print(f"桥边数: {bridge_edge_count}")

    output_data = {
        "meta": {
            "algorithm": "global_hierarchy",
            "ne_graph": args.ne_graph,
            "site_count": len(prediction_result["sites"]),
            "edge_count": len(prediction_result["edges"]),
            "directed_edge_count": directed_edge_count,
            "bidirectional_edge_count": bidirectional_edge_count,
            "bridge_edge_count": bridge_edge_count,
            "base_margin": args.base_margin,
            "ring_margin": args.ring_margin,
            "same_role_margin": args.same_role_margin,
        },
        "sites": prediction_result["sites"],
        "edges": prediction_result["edges"],
        "primary_upstream_map": primary_upstream_map,
        "downstream_map": downstream_map,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
