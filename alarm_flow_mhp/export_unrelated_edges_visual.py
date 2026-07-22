#!/usr/bin/env python3
"""Decode a ``--candidate-scope unrelated`` association cache (.npz) into a
visual JSONL that the fault-group overview page can browse.

Every edge in an ``unrelated`` cache is, by construction, a topologically
UNRELATED device pair whose peak association score cleared the immigrant
threshold — i.e. exactly the "unrelated devices that nonetheless look related"
question. This tool turns each such edge into one overview record:

* one record == one edge between exactly two devices;
* ``ne_info`` carries both devices (coords/site/domain) so the NE-propagation
  map page can place them;
* ``missing_topology_edges`` carries the single predicted propagation edge with
  the source/target alarm types and the trigger strength (``score``);
* the record ``uuid`` is what the overview page shows in its list.

Load the output in ``visualization/fault_group_browser.html`` (故障组总览);
clicking 查看详情 opens ``ne_propagation_visualizer.html`` on the map.

The signature ids stored in the cache are indices into
``sorted(graph_period_types(model, ne_graph), key=(entity, alarm_type))``, so
only the model + NE/site graphs are needed to decode names — the compile-time
candidate policy/config is NOT required.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_mhp.feature_spec import domain_of, split_entity, topo_node_of
from alarm_flow_mhp.stream_alarm_period_mhp import (
    CACHE_STATE_LAYOUT_TARGET_ONLY,
    _build_runtime_scorers,
    graph_period_types,
    load_association_cache,
)
from fault_grouping.matching.group_output_builder import (
    _get_ne_static_info,
    build_group_link_info,
    resolve_ne_site_context,
)
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display

RELATION_LABEL = "unrelated"
EDGE_SOURCE = "alarm_period_unrelated"


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Decode an unrelated-scope AlarmPeriod association cache (.npz) into "
            "a browsable propagation JSONL (one edge between two devices per line)."
        )
    )
    parser.add_argument("cache", help="Association cache .npz (compiled with --candidate-scope unrelated).")
    parser.add_argument("model", help="The feature-mode alarm-flow MHP artifact JSON used to compile the cache.")
    parser.add_argument("output", help="Output JSONL for the fault-group overview page.")
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=resource_display("ne_graph.json"))
    parser.add_argument("--site-graph", default=SITE_GRAPH_JSON, help=resource_display("site_graph.json"))
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop edges whose peak base score is below this value.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Drop edges whose (base_score - threshold) margin is below this value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Keep only the strongest K edges (by score); 0 keeps all. Default 1000.",
    )
    parser.add_argument(
        "--include-self-node",
        action="store_true",
        help="Keep edges whose two entities map to the same topology node (default drops them).",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _load_site_graph(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _decode_period_types(artifact, scorer):
    """Reproduce the exact signature-id ordering the compiler used."""
    period_types, _entities, _alarm_types = graph_period_types(artifact, scorer)
    return tuple(sorted(period_types, key=lambda pt: (pt.entity, pt.alarm_type)))


def _node_info(node_id, ne_graph_data, site_graph_data, alarm_types):
    """Build one ne_info entry matching the schema the visualizer consumes."""
    static = _get_ne_static_info(ne_graph_data, node_id)
    site_ctx = resolve_ne_site_context(node_id, [], ne_graph_data, site_graph_data)
    site_id = site_ctx["site_id"] or static["site_id_from_ne"]
    site_name = static["site_name_from_ne"] or site_ctx["site_name"]
    alarms = [
        {
            "alarm_id": "",
            "alarm_type": alarm_type,
            "alarm_time": "",
            "domain": static["domain_raw"],
            "site_id": site_id,
            "site_name": site_name,
            "matched_role": "",
            "matched_rule": EDGE_SOURCE,
        }
        for alarm_type in sorted(alarm_types)
    ]
    return {
        "link": build_group_link_info(node_id, set(), ne_graph_data),
        "name": static["name"],
        "site_id": site_id,
        "site_name": site_name,
        "type": static["type"],
        "network_type": static["network_type"],
        "manufacturer": static["manufacturer"],
        "running_status": static["running_status"],
        "domain": static["domain_upper"],
        "region_id": site_ctx["region_id"],
        "longitude": site_ctx["longitude"],
        "latitude": site_ctx["latitude"],
        "alarm": alarms,
    }


def _build_record(agg, ne_graph_data, site_graph_data):
    src_node = agg["source_node"]
    tgt_node = agg["target_node"]
    src_alarm = agg["source_alarm"]
    tgt_alarm = agg["target_alarm"]
    uuid = f"unrelated::{src_node}[{src_alarm}]->{tgt_node}[{tgt_alarm}]"

    src_static = _get_ne_static_info(ne_graph_data, src_node)
    tgt_static = _get_ne_static_info(ne_graph_data, tgt_node)

    edge = {
        "source_ne": src_node,
        "target_ne": tgt_node,
        "source_alarm": src_alarm,
        "target_alarm": tgt_alarm,
        "source_type": src_static["type"],
        "target_type": tgt_static["type"],
        "source_domain": agg["source_domain"],
        "target_domain": agg["target_domain"],
        "relation": RELATION_LABEL,
        "predicted_relation": RELATION_LABEL,
        "edge_source": EDGE_SOURCE,
        "score": agg["score"],
        "threshold": agg["threshold"],
        "margin": agg["score"] - agg["threshold"],
        "past_window_sec": agg["past_window_sec"],
        "future_window_sec": agg["future_window_sec"],
        "state_pair_count": agg["state_pair_count"],
        "peak_target_state": agg["peak_target_state"],
        "peak_source_state": agg["peak_source_state"],
        "sample_id": uuid,
    }

    ne_info = {
        src_node: _node_info(src_node, ne_graph_data, site_graph_data, {src_alarm}),
        tgt_node: _node_info(tgt_node, ne_graph_data, site_graph_data, {tgt_alarm}),
    }
    for node_id, info in ne_info.items():
        info["group"] = uuid

    return {
        "uuid": uuid,
        "match_info": {
            "uuid": uuid,
            "rule": EDGE_SOURCE,
            "merged_rules": [],
            "related_group_uuids": [],
            "role_mapping": {},
            "missing_topology_edges": [edge],
        },
        "missing_topology_edges": [edge],
        "ne_info": ne_info,
        "group_info": {
            uuid: {
                "ne_list": sorted(ne_info.keys()),
                "site_list": sorted(
                    {info["site_id"] for info in ne_info.values() if info["site_id"]}
                ),
            }
        },
    }


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not str(args.output).lower().endswith((".jsonl", ".json", ".txt")):
        parser.error("output must be a .jsonl (or .json/.txt) file")

    artifact = load_alarm_mhp_artifact(args.model)
    scorer, _mu_scorer, ne_graph_data = _build_runtime_scorers(
        artifact, args.ne_graph, args.site_graph, quiet=args.quiet
    )
    site_graph_data = _load_site_graph(args.site_graph)
    period_types = _decode_period_types(artifact, scorer)
    type_count = len(period_types)

    cache = load_association_cache(args.cache)
    header_scope = (cache.get("fingerprint") or {}).get("candidate_scope")
    if header_scope and header_scope != "unrelated" and not args.quiet:
        print(
            f"[unrelated-export] WARNING: cache candidate_scope={header_scope!r}, "
            "not 'unrelated'; every edge will still be exported as-is.",
            flush=True,
        )
    metadata = cache.get("metadata") or {}
    state_layout = str(metadata.get("state_layout", ""))
    target_only = state_layout == CACHE_STATE_LAYOUT_TARGET_ONLY

    arrays = cache["arrays"]
    target_sig = np.asarray(arrays["target_signature_ids"], dtype=np.int64)
    source_sig = np.asarray(arrays["source_signature_ids"], dtype=np.int64)
    base_scores = np.asarray(arrays["base_scores"], dtype=np.float64)
    thresholds = np.asarray(arrays["thresholds"], dtype=np.float64)
    past_windows = np.asarray(arrays["past_windows"], dtype=np.float64)
    future_windows = np.asarray(arrays["future_windows"], dtype=np.float64)
    edge_count = len(base_scores)

    target_type_id = target_sig // 8
    target_state = target_sig % 8
    if target_only:
        source_type_id = source_sig
        source_state = np.full(edge_count, -1, dtype=np.int64)
    else:
        source_type_id = source_sig // 8
        source_state = source_sig % 8

    # Aggregate the (up to 8/64) per-state edges of one device-alarm pair into a
    # single connecting edge, keeping the strongest state as the representative.
    aggregates = {}
    kept_edges = 0
    dropped_self = 0
    for i in range(edge_count):
        t_id = int(target_type_id[i])
        s_id = int(source_type_id[i])
        if t_id >= type_count or s_id >= type_count:
            continue
        target_pt = period_types[t_id]
        source_pt = period_types[s_id]
        src_node = topo_node_of(source_pt.entity)
        tgt_node = topo_node_of(target_pt.entity)
        if not args.include_self_node and src_node == tgt_node:
            dropped_self += 1
            continue
        score = float(base_scores[i])
        key = (src_node, source_pt.alarm_type, tgt_node, target_pt.alarm_type)
        current = aggregates.get(key)
        if current is None:
            current = {
                "source_node": src_node,
                "target_node": tgt_node,
                "source_alarm": source_pt.alarm_type,
                "target_alarm": target_pt.alarm_type,
                "source_domain": domain_of(source_pt.entity, scorer.node_infos),
                "target_domain": domain_of(target_pt.entity, scorer.node_infos),
                "score": score,
                "threshold": float(thresholds[i]),
                "past_window_sec": float(past_windows[i]),
                "future_window_sec": float(future_windows[i]),
                "peak_target_state": int(target_state[i]),
                "peak_source_state": int(source_state[i]),
                "state_pair_count": 1,
            }
            aggregates[key] = current
        else:
            current["state_pair_count"] += 1
            if score > current["score"]:
                current["score"] = score
                current["threshold"] = float(thresholds[i])
                current["past_window_sec"] = float(past_windows[i])
                current["future_window_sec"] = float(future_windows[i])
                current["peak_target_state"] = int(target_state[i])
                current["peak_source_state"] = int(source_state[i])
        kept_edges += 1

    records = list(aggregates.values())
    if args.min_score is not None:
        records = [r for r in records if r["score"] >= args.min_score]
    if args.min_margin is not None:
        records = [r for r in records if (r["score"] - r["threshold"]) >= args.min_margin]
    records.sort(key=lambda r: r["score"], reverse=True)
    if args.top_k and args.top_k > 0:
        records = records[: args.top_k]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as stream:
        for agg in records:
            record = _build_record(agg, ne_graph_data, site_graph_data)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if not args.quiet:
        print(
            f"[unrelated-export] cache_edges={edge_count}, state_layout={state_layout}, "
            f"device_pairs={len(aggregates)}, dropped_self_node={dropped_self}, "
            f"written={written}; output={os.path.abspath(args.output)}",
            flush=True,
        )
        if written:
            top = records[0]
            print(
                f"[unrelated-export] strongest: {top['source_node']}[{top['source_alarm']}] "
                f"-> {top['target_node']}[{top['target_alarm']}] "
                f"score={top['score']:.4f} margin={top['score'] - top['threshold']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
