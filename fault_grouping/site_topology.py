import json

from collections import defaultdict


def extract_link_direction_values(link_meta):
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

            direction_values = extract_link_direction_values(link_meta)
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


def _iter_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def _normalize_predicted_relation(row):
    relation = str(
        row.get("site_a_to_site_b_relation")
        or row.get("predicted_relation")
        or ""
    ).strip()
    if relation in {"downstream", "upstream", "bidirection"}:
        return relation
    return ""


def load_missing_topology_predictions(path, min_score=0.0):
    """加载 site_relation_learning/infer.py --mode topology-errors 输出的 missing 预测。"""
    predictions = []
    if not path:
        return predictions

    for row in _iter_json_or_jsonl(path):
        if str(row.get("error_type", "")).strip() != "missing":
            continue
        try:
            score = float(row.get("score", row.get("predicted_score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            continue

        site_a = str(row.get("site_a", "") or "").strip()
        site_b = str(row.get("site_b", "") or "").strip()
        relation = _normalize_predicted_relation(row)
        if not site_a or not site_b or site_a == site_b or not relation:
            continue

        predictions.append({
            "site_a": site_a,
            "site_b": site_b,
            "relation": relation,
            "score": score,
            "sample_id": row.get("sample_id", f"{site_a}__{site_b}"),
        })
    return predictions


def apply_missing_topology_predictions(topo_downstream_map, site_chain_index, predictions):
    """把高置信缺边预测注入站点拓扑，并返回方向化弱拓扑边索引。"""
    if not predictions:
        return topo_downstream_map, site_chain_index, {}

    augmented_topo = defaultdict(set)
    for site_id, downstream_sites in (topo_downstream_map or {}).items():
        site_id = str(site_id).strip()
        if not site_id:
            continue
        augmented_topo[site_id].update(str(item).strip() for item in downstream_sites if str(item).strip())

    augmented_chain = {}
    if isinstance(site_chain_index, dict):
        for site_id, info in site_chain_index.items():
            augmented_chain[site_id] = {
                "downstream_site_hops": dict(info.get("downstream_site_hops", {})),
                "upstream_site_hops": dict(info.get("upstream_site_hops", {})),
                "bidirectional_sites": set(info.get("bidirectional_sites", set())),
            }

    weak_edges = {}
    direct_downstream_edges = set()

    def ensure_chain(site_id):
        if not isinstance(site_chain_index, dict):
            return None
        return augmented_chain.setdefault(site_id, {
            "downstream_site_hops": {},
            "upstream_site_hops": {},
            "bidirectional_sites": set(),
        })

    def upsert_weak_edge(edge_key, edge_meta):
        existing = weak_edges.get(edge_key)
        if existing and float(existing.get("score", 0.0) or 0.0) > float(edge_meta.get("score", 0.0) or 0.0):
            return
        weak_edges[edge_key] = edge_meta

    def add_downstream(source_site, target_site, prediction, relation_label, propagate=True):
        if not source_site or not target_site or source_site == target_site:
            return
        augmented_topo[source_site].add(target_site)
        augmented_topo[target_site]
        source_chain = ensure_chain(source_site)
        target_chain = ensure_chain(target_site)
        if source_chain is not None:
            previous_hop = source_chain["downstream_site_hops"].get(target_site)
            if previous_hop is None or previous_hop > 1:
                source_chain["downstream_site_hops"][target_site] = 1
        if target_chain is not None:
            previous_hop = target_chain["upstream_site_hops"].get(source_site)
            if previous_hop is None or previous_hop > 1:
                target_chain["upstream_site_hops"][source_site] = 1
        if propagate:
            direct_downstream_edges.add((source_site, target_site))
        edge_key = (source_site, target_site)
        upsert_weak_edge(edge_key, {
            "source_site": source_site,
            "target_site": target_site,
            "relation": relation_label,
            "score": prediction.get("score", 0.0),
            "sample_id": prediction.get("sample_id", ""),
            "inferred_from_missing_topology": False,
        })

    for prediction in predictions:
        site_a = prediction["site_a"]
        site_b = prediction["site_b"]
        relation = prediction["relation"]
        if relation == "downstream":
            add_downstream(site_a, site_b, prediction, relation)
        elif relation == "upstream":
            add_downstream(site_b, site_a, prediction, relation)
        elif relation == "bidirection":
            add_downstream(site_a, site_b, prediction, relation, propagate=False)
            add_downstream(site_b, site_a, prediction, relation, propagate=False)
            ensure_a = ensure_chain(site_a)
            ensure_b = ensure_chain(site_b)
            if ensure_a is not None:
                ensure_a["bidirectional_sites"].add(site_b)
            if ensure_b is not None:
                ensure_b["bidirectional_sites"].add(site_a)

    def update_chain_hop(source_site, target_site, hop):
        if not isinstance(site_chain_index, dict):
            return False
        if not source_site or not target_site or source_site == target_site or hop <= 0:
            return False
        source_chain = ensure_chain(source_site)
        target_chain = ensure_chain(target_site)
        changed = False
        previous_downstream_hop = source_chain["downstream_site_hops"].get(target_site)
        if previous_downstream_hop is None or hop < previous_downstream_hop:
            source_chain["downstream_site_hops"][target_site] = hop
            changed = True
        previous_upstream_hop = target_chain["upstream_site_hops"].get(source_site)
        if previous_upstream_hop is None or hop < previous_upstream_hop:
            target_chain["upstream_site_hops"][source_site] = hop
            changed = True
        return changed

    if isinstance(site_chain_index, dict) and direct_downstream_edges:
        # 新增 u->v 后，所有 u 的上游都应可达 v 的所有下游：
        # ancestor -> u -> v -> descendant。循环到稳定以覆盖连续多条补边。
        changed = True
        while changed:
            changed = False
            for source_site, target_site in direct_downstream_edges:
                source_chain = ensure_chain(source_site)
                target_chain = ensure_chain(target_site)
                ancestor_hops = {source_site: 0, **source_chain["upstream_site_hops"]}
                descendant_hops = {target_site: 0, **target_chain["downstream_site_hops"]}
                for ancestor_site, ancestor_hop in ancestor_hops.items():
                    for descendant_site, descendant_hop in descendant_hops.items():
                        inferred_hop = int(ancestor_hop) + 1 + int(descendant_hop)
                        if update_chain_hop(ancestor_site, descendant_site, inferred_hop):
                            changed = True
                            direct_meta = weak_edges.get((source_site, target_site), {})
                            direct_edge = {
                                "source_site": source_site,
                                "target_site": target_site,
                                "relation": direct_meta.get("relation", "downstream"),
                                "score": direct_meta.get("score", 0.0),
                                "sample_id": direct_meta.get("sample_id", ""),
                            }
                            edge_key = (ancestor_site, descendant_site)
                            if edge_key != (source_site, target_site):
                                upsert_weak_edge(edge_key, {
                                    "source_site": ancestor_site,
                                    "target_site": descendant_site,
                                    "relation": "downstream",
                                    "score": direct_edge["score"],
                                    "sample_id": f"{ancestor_site}__{descendant_site}",
                                    "inferred_from_missing_topology": True,
                                    "inferred_hops": inferred_hop,
                                    "via_missing_edges": [direct_edge],
                                })

    normalized_topo = {
        site_id: sorted(downstream_sites)
        for site_id, downstream_sites in augmented_topo.items()
    }
    return normalized_topo, augmented_chain if isinstance(site_chain_index, dict) else site_chain_index, weak_edges


def normalize_site_chain_hops(hops_map):
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

        downstream_hops = normalize_site_chain_hops(raw_info.get("downstream_site_hops"))
        upstream_hops = normalize_site_chain_hops(raw_info.get("upstream_site_hops"))
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
