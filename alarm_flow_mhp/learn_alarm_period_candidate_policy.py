#!/usr/bin/env python3
"""Learn and validate an indexable AlarmPeriod candidate policy.

The current global MHP scorer acts as the teacher.  Calibration and validation
target entities are disjoint, and every sampled target is scored against the
entire graph source universe.  The learner then chooses a low-cost union of
indexable rules independently for each directed alarm-type pair.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import random
from statistics import NormalDist
import time

import numpy as np

if __package__ in (None, ""):
    from _script_env import ensure_repo_root

    ensure_repo_root(1)

from alarm_flow_mhp.aggregator import load_alarm_mhp_artifact
from alarm_flow_isahp.alarm_io import load_ordered_alarm_events
from alarm_flow_isahp.event_domain import (
    DEVICE_DOMAIN_FIELD,
    filter_and_annotate_device_domain,
)
from alarm_flow_mhp.candidate_policy import (
    CandidatePolicy,
    RELATED_MASK,
    RELATED_RULES,
    RULES,
    RULE_BITS,
    adaptive_candidate_count,
    build_candidate_indices,
    candidate_policy_fingerprint,
    candidate_rule_mask,
    rule_candidates,
    sha256_file,
    write_candidate_policy,
)
from alarm_flow_mhp.feature_spec import runtime_ne_at, topo_node_of
from alarm_flow_mhp.stream_alarm_period_mhp import (
    EPS,
    CompiledAssociationPlan,
    PeriodStreamConfig,
    PeriodType,
    _association_plan_config,
    _build_runtime_scorers,
    _combo_state,
    graph_period_universe,
)
from alarm_flow_mhp.topology_relation_prior import (
    parse_topology_relation_prior,
    topology_relation_weights,
)
from topology_resources import NE_GRAPH_JSON, SITE_GRAPH_JSON, resource_display
from fault_grouping.alarm_events.io import is_clear_alarm


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Distill the exact global AlarmPeriod scorer into a validated, "
            "indexable candidate policy."
        )
    )
    parser.add_argument("model", help="Trained feature-mode alarm-flow MHP artifact JSON.")
    parser.add_argument("output", help="Candidate policy JSON output.")
    parser.add_argument("--ne-graph", default=NE_GRAPH_JSON, help=resource_display("ne_graph.json"))
    parser.add_argument(
        "--site-graph", default=SITE_GRAPH_JSON, help=resource_display("site_graph.json")
    )
    parser.add_argument("--history-window-sec", type=float, default=None)
    parser.add_argument("--time-slack-sec", type=float, default=None)
    parser.add_argument("--late-penalty-half-life-sec", type=float, default=None)
    parser.add_argument("--immigrant-bias", type=float, default=1.0)
    parser.add_argument("--feature-alpha-floor", type=float, default=None)
    parser.add_argument("--attach-threshold-ratio", type=float, default=1.0)
    parser.add_argument(
        "--topology-relation-prior",
        default="",
        help="Comma-separated relation multipliers used by the teacher scorer.",
    )
    parser.add_argument(
        "--alarms",
        default="",
        help=(
            "Optional representative alarm batch. Target entities are sampled "
            "from its non-clear events while every target is still validated "
            "against the full graph source universe."
        ),
    )
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--regions", default="")
    parser.add_argument("--clear-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--target-entities-file",
        default="",
        help=(
            "Optional representative target universe as a JSON string array "
            "or one entity per line. Calibration/validation targets are "
            "sampled from its intersection with the graph."
        ),
    )
    parser.add_argument(
        "--calibration-target-entities",
        type=int,
        default=128,
        help="Disjoint graph entities used to select rules.",
    )
    parser.add_argument(
        "--validation-target-entities",
        type=int,
        default=128,
        help="Disjoint graph entities used only for final recall validation.",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.99,
        help="Required one-sided confidence lower bound for validation recall.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the Wilson recall lower bound.",
    )
    parser.add_argument(
        "--min-pair-positives",
        type=int,
        default=20,
        help=(
            "Alarm-type pairs with at least this many validation positives "
            "must also meet point recall."
        ),
    )
    parser.add_argument("--source-chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _runtime_config(args, artifact, relation_prior):
    history = (
        float(args.history_window_sec)
        if args.history_window_sec is not None
        else float(artifact.config.history_window_sec)
    )
    slack = (
        float(args.time_slack_sec)
        if args.time_slack_sec is not None
        else float(getattr(artifact.config, "time_slack_sec", 0.0))
    )
    late_half_life = (
        float(args.late_penalty_half_life_sec)
        if args.late_penalty_half_life_sec is not None
        else float(getattr(artifact.config, "late_penalty_half_life_sec", 1.0))
    )
    floor = (
        float(args.feature_alpha_floor)
        if args.feature_alpha_floor is not None
        else float(getattr(artifact.config, "edge_threshold", 0.0))
    )
    config = PeriodStreamConfig(
        aggregation_wait_sec=max(30.0, slack),
        history_window_sec=history,
        time_slack_sec=slack,
        late_penalty_half_life_sec=late_half_life,
        time_scale_sec=float(artifact.config.time_scale_sec),
        immigrant_bias=args.immigrant_bias,
        feature_alpha_floor=floor,
        attach_threshold_ratio=args.attach_threshold_ratio,
        candidate_scope="global",
        topology_relation_prior=relation_prior,
    )
    config.validate()
    return config


def _sample_entity_splits(entities, calibration_count, validation_count, seed):
    needed = calibration_count + validation_count
    if calibration_count < 1 or validation_count < 1:
        raise ValueError("calibration and validation target entity counts must be >= 1")
    if len(entities) < needed:
        raise ValueError(
            f"graph has {len(entities)} entities but {needed} disjoint targets were requested"
        )
    selected = random.Random(seed).sample(list(entities), needed)
    return selected[:calibration_count], selected[calibration_count:]


def _load_target_entities(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as stream:
        text = stream.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list):
        raise ValueError("target entities file must be a JSON array or one entity per line")
    return tuple(dict.fromkeys(str(value) for value in payload if str(value)))


def _input_digest(path):
    if os.path.isfile(path):
        return sha256_file(path)
    digest = hashlib.sha256()
    root = os.path.abspath(path)
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            file_path = os.path.join(directory, name)
            digest.update(os.path.relpath(file_path, root).encode("utf-8"))
            digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def _target_entities_from_alarms(args, artifact, ne_graph_data):
    events, _metadata = load_ordered_alarm_events(
        args.alarms,
        topo_path=args.site_graph,
        ne_graph_path=args.ne_graph,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        clear_delay_sec=args.clear_delay_sec,
        regions=args.regions,
        show_progress=not args.quiet,
    )
    if DEVICE_DOMAIN_FIELD in tuple(artifact.config.type_fields):
        events, _stats = filter_and_annotate_device_domain(events, ne_graph_data)
    entities = []
    seen = set()
    for event in events:
        payload = event.get("alarm", {}) if isinstance(event, dict) else {}
        if is_clear_alarm(payload):
            continue
        entity, alarm_type = runtime_ne_at(
            event,
            artifact.config.type_fields,
            artifact.config.topology_node_field,
        )
        if entity and alarm_type and entity not in seen:
            seen.add(entity)
            entities.append(str(entity))
    return tuple(entities)


def _teacher_positive_masks(
    plan,
    target_entities,
    alarm_types,
    source_entities,
    chunk_size,
    quiet=False,
    label="teacher",
):
    """Exact positive type-pair rule-mask histograms for sampled targets.

    Dynamic mark contributions are additive in the learned logit, so using the
    independently maximal source and target mark terms is exactly equivalent
    to asking whether *any* of the 8×8 state pairs can produce an edge.
    """
    if chunk_size < 1:
        raise ValueError("source chunk size must be >= 1")
    d = plan.decomposed
    max_source_state = int(np.argmax(d.src_mark_table))
    max_target_term = max(d.tgt_term(_combo_state(state)) for state in range(8))
    counts = defaultdict(Counter)
    total_chunks = math.ceil(len(source_entities) / chunk_size)
    started = time.monotonic()
    for chunk_number, start in enumerate(
        range(0, len(source_entities), chunk_size), 1
    ):
        entities = source_entities[start:start + chunk_size]
        static_table = d.entity_static_table(entities)
        source_count = len(entities)
        source_mark_indices = np.full(
            source_count, max_source_state, dtype=np.int64
        )
        source_at_id_arrays = {
            str(source_at): np.full(
                source_count,
                plan.scorer.at_to_id.get(str(source_at), -1),
                dtype=np.int64,
            )
            for source_at in alarm_types
        }
        source_nodes = (
            [topo_node_of(entity) for entity in entities]
            if plan.config.topology_relation_prior
            else None
        )
        for target_entity in target_entities:
            target_node = topo_node_of(target_entity)
            parts = d.entity_parts_from_table(target_entity, static_table)
            if plan.config.topology_relation_prior:
                relation = topology_relation_weights(
                    source_nodes,
                    target_node,
                    plan.scorer.topology_index,
                    plan.scorer.node_infos,
                    plan.config.topology_relation_prior,
                )
            else:
                relation = 1.0
            for target_at in alarm_types:
                target = PeriodType(target_entity, target_at)
                threshold = plan._mu(target)
                target_at_id = plan.scorer.at_to_id.get(str(target_at), -1)
                for source_at in alarm_types:
                    pair_key = (str(target_at), str(source_at))
                    logits = d.logits_from_parts(
                        target_at_id,
                        source_at_id_arrays[str(source_at)],
                        src_mark_idx=source_mark_indices,
                        tgt_term=max_target_term,
                        **parts,
                    )
                    alpha = d._softplus(logits)
                    if d.alpha_scale != 1.0:
                        alpha = alpha * d.alpha_scale
                    base_scores = alpha * plan.beta * relation
                    keep = (
                        (alpha >= plan.config.feature_alpha_floor)
                        & (base_scores + EPS >= threshold)
                        & (base_scores > 0)
                    )
                    for index in np.flatnonzero(keep):
                        source = PeriodType(entities[int(index)], source_at)
                        mask = candidate_rule_mask(target, source, plan.scorer)
                        counts[pair_key][int(mask)] += 1
            # entity_parts_from_table caches one (source_count, 7) site-pair
            # matrix per target site. Learning samples many sites, so retain no
            # cross-target rows and keep memory bounded by one source chunk.
            static_table.site_pair_rows.clear()
            plan.scorer._site_pair_cache.clear()
        if not quiet:
            elapsed = time.monotonic() - started
            print(
                f"[candidate-policy] {label}: source_chunks="
                f"{chunk_number}/{total_chunks}, positives="
                f"{sum(sum(row.values()) for row in counts.values()):,}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return counts


def _rule_costs(target_entities, alarm_types, prepared):
    costs = defaultdict(Counter)
    for target_entity in target_entities:
        for target_at in alarm_types:
            target = PeriodType(target_entity, target_at)
            for source_at in alarm_types:
                pair = (str(target_at), str(source_at))
                for rule in RULES:
                    costs[pair][rule] += len(
                        rule_candidates(target, source_at, rule, prepared)
                    )
    return costs


# Rules the ``unrelated`` branch is allowed to learn: everything the related
# predicate does not already provide. The related branch owns its pairs from a
# separate cache, so the policy only distills the non-local delta.
NONLOCAL_RULES = tuple(rule for rule in RULES if rule not in RELATED_RULES)


def _covered_union(mask_counts, selected_mask):
    """Positives the two-branch system recalls: related-covered ∪ policy-covered.

    A positive is recalled if the related predicate would retrieve it (any
    ``RELATED_MASK`` bit set — the related cache owns it) or a selected non-local
    rule retrieves it. This is the recall the online union actually achieves.
    """
    return sum(
        count
        for mask, count in mask_counts.items()
        if (int(mask) & RELATED_MASK) or (int(mask) & selected_mask)
    )


def _choose_rules(mask_counts, rule_costs, recall_target):
    """Cheapest non-local rule union whose related∪delta recall clears target.

    Related-covered positives are free (served by the related cache), so the
    delta only has to recall what related misses. An empty selection is valid
    and means the related branch alone already meets the target for this pair.
    """
    total = sum(mask_counts.values())
    if not total:
        return ()
    best = None
    for extra in range(1 << len(NONLOCAL_RULES)):
        selected_mask = 0
        for index, rule in enumerate(NONLOCAL_RULES):
            if extra & (1 << index):
                selected_mask |= RULE_BITS[rule]
        if _covered_union(mask_counts, selected_mask) / total + 1e-15 < recall_target:
            continue
        selected = tuple(
            rule for rule in NONLOCAL_RULES if selected_mask & RULE_BITS[rule]
        )
        # Sum is a conservative overlap-blind candidate-cost estimate.  Exact
        # union size is measured on the held-out targets below.
        cost = sum(rule_costs.get(rule, 0) for rule in selected)
        key = (cost, len(selected), selected)
        if best is None or key < best[0]:
            best = (key, selected)
    return best[1] if best is not None else tuple(NONLOCAL_RULES)


def _learn_rules(calibration_masks, costs, alarm_types, recall_target):
    rows = {}
    for target_at in alarm_types:
        row = {}
        for source_at in alarm_types:
            key = (str(target_at), str(source_at))
            row[str(source_at)] = _choose_rules(
                calibration_masks.get(key, {}),
                costs.get(key, {}),
                recall_target,
            )
        rows[str(target_at)] = row
    return rows


def _wilson_lower_bound(successes, total, confidence):
    if total <= 0:
        return 0.0
    z = NormalDist().inv_cdf(float(confidence))
    p = successes / total
    z2 = z * z
    center = p + z2 / (2 * total)
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)
    return max(0.0, (center - radius) / (1 + z2 / total))


def _validation_report(policy, masks, confidence, recall_target, min_pair_positives):
    rows = {}
    total = covered = 0
    pair_failures = []
    for target_at, row in policy.rules_by_alarm_pair.items():
        for source_at, rules in row.items():
            key = (str(target_at), str(source_at))
            mask_counts = masks.get(key, {})
            pair_total = sum(mask_counts.values())
            selected_mask = sum(RULE_BITS[rule] for rule in rules)
            pair_covered = _covered_union(mask_counts, selected_mask)
            recall = pair_covered / pair_total if pair_total else None
            rows[f"{target_at}->{source_at}"] = {
                "positive_count": pair_total,
                "covered_positive_count": pair_covered,
                "recall": recall,
                "rules": list(rules),
            }
            total += pair_total
            covered += pair_covered
            if (
                pair_total >= min_pair_positives
                and recall is not None
                and recall + 1e-15 < recall_target
            ):
                pair_failures.append(f"{target_at}->{source_at}")
    recall = covered / total if total else 0.0
    lower = _wilson_lower_bound(covered, total, confidence)
    return {
        "positive_count": total,
        "covered_positive_count": covered,
        "recall": recall,
        "confidence": confidence,
        "recall_lower_bound": lower,
        "recall_target": recall_target,
        "min_pair_positives": min_pair_positives,
        "pair_failures": pair_failures,
        "alarm_pairs": rows,
        "approved": bool(total and lower >= recall_target and not pair_failures),
    }


def _feature_importance(artifact, limit=30):
    metadata = artifact.training_metadata or {}
    kernel = metadata.get("feature_kernel") or {}
    names = kernel.get("feature_names") or []
    weights = kernel.get("weights") or []
    ranked = sorted(
        ((str(name), float(weight)) for name, weight in zip(names, weights)),
        key=lambda item: -abs(item[1]),
    )
    return [
        {"feature": name, "weight": weight, "abs_weight": abs(weight)}
        for name, weight in ranked[:limit]
    ]


def main():
    parser = _parser()
    args = parser.parse_args()
    if not 0 < args.recall_target < 1:
        parser.error("--recall-target must be in (0, 1)")
    if not 0.5 < args.confidence < 1:
        parser.error("--confidence must be in (0.5, 1)")
    if args.min_pair_positives < 1:
        parser.error("--min-pair-positives must be >= 1")
    if args.source_chunk_size < 1:
        parser.error("--source-chunk-size must be >= 1")
    try:
        relation_prior = parse_topology_relation_prior(args.topology_relation_prior)
    except ValueError as exc:
        parser.error(str(exc))

    artifact = load_alarm_mhp_artifact(args.model)
    scorer, mu_scorer, ne_graph_data = _build_runtime_scorers(
        artifact, args.ne_graph, args.site_graph, quiet=args.quiet
    )
    try:
        config = _runtime_config(args, artifact, relation_prior)
    except ValueError as exc:
        parser.error(str(exc))
    entities, alarm_types = graph_period_universe(artifact, scorer)
    graph_entity_count = len(entities)
    alarm_type_count = len(alarm_types)
    prepared = build_candidate_indices(
        (), scorer, entities=entities, alarm_types=alarm_types
    )
    eligible_entities = entities
    target_data_metadata = {}
    if args.alarms and args.target_entities_file:
        parser.error("--alarms and --target-entities-file are mutually exclusive")
    if args.alarms:
        try:
            requested_entities = _target_entities_from_alarms(
                args, artifact, ne_graph_data
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load --alarms: {exc}")
        eligible_entities = tuple(
            value for value in requested_entities if value in prepared["attributes"]
        )
        target_data_metadata = {
            "alarms": os.path.abspath(args.alarms),
            "alarms_sha256": _input_digest(args.alarms),
            "requested_target_entity_count": len(requested_entities),
            "eligible_target_entity_count": len(eligible_entities),
        }
    elif args.target_entities_file:
        try:
            requested_entities = _load_target_entities(args.target_entities_file)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load --target-entities-file: {exc}")
        eligible_entities = tuple(
            value for value in requested_entities if value in prepared["attributes"]
        )
        target_data_metadata = {
            "target_entities_file": os.path.abspath(args.target_entities_file),
            "target_entities_file_sha256": sha256_file(args.target_entities_file),
            "requested_target_entity_count": len(requested_entities),
            "eligible_target_entity_count": len(eligible_entities),
        }
    try:
        calibration_entities, validation_entities = _sample_entity_splits(
            eligible_entities,
            args.calibration_target_entities,
            args.validation_target_entities,
            args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            f"[candidate-policy] graph_entities={graph_entity_count:,}, "
            f"alarm_types={alarm_type_count}, "
            f"source_types={graph_entity_count * alarm_type_count:,}, "
            f"calibration_entities={len(calibration_entities)}, "
            f"validation_entities={len(validation_entities)}",
            flush=True,
    )
    plan = CompiledAssociationPlan(scorer, mu_scorer, artifact, config)
    calibration_masks = _teacher_positive_masks(
        plan,
        calibration_entities,
        alarm_types,
        entities,
        args.source_chunk_size,
        quiet=args.quiet,
        label="calibration",
    )
    costs = _rule_costs(calibration_entities, alarm_types, prepared)
    rules = _learn_rules(
        calibration_masks, costs, alarm_types, args.recall_target
    )
    provisional = CandidatePolicy(rules_by_alarm_pair=rules)
    calibration_report = _validation_report(
        provisional,
        calibration_masks,
        args.confidence,
        args.recall_target,
        args.min_pair_positives,
    )
    validation_masks = _teacher_positive_masks(
        plan,
        validation_entities,
        alarm_types,
        entities,
        args.source_chunk_size,
        quiet=args.quiet,
        label="validation",
    )
    validation = _validation_report(
        provisional,
        validation_masks,
        args.confidence,
        args.recall_target,
        args.min_pair_positives,
    )
    calibration_point_ok = bool(
        calibration_report["positive_count"]
        and calibration_report["recall"] + 1e-15 >= args.recall_target
        and not calibration_report["pair_failures"]
    )
    validation["approved"] = bool(
        validation["approved"] and calibration_point_ok
    )
    validation["calibration"] = {
        "positive_count": calibration_report["positive_count"],
        "covered_positive_count": calibration_report["covered_positive_count"],
        "recall": calibration_report["recall"],
        "pair_failures": calibration_report["pair_failures"],
    }

    validation_targets = [
        PeriodType(entity, alarm_type)
        for entity in validation_entities
        for alarm_type in alarm_types
    ]
    prepared["policy"] = provisional
    selected_pairs = sum(
        adaptive_candidate_count(target, provisional, prepared, exclude_related=True)
        for target in validation_targets
    )
    global_pairs = (
        len(validation_targets) * graph_entity_count * alarm_type_count
    )
    validation.update(
        {
            "validation_target_entity_count": len(validation_entities),
            "validation_target_type_count": len(validation_targets),
            "selected_candidate_pair_count": selected_pairs,
            "global_candidate_pair_count": global_pairs,
            "candidate_ratio": selected_pairs / global_pairs if global_pairs else 0.0,
            "candidate_reduction": (
                global_pairs / selected_pairs if selected_pairs else None
            ),
        }
    )
    fingerprint = candidate_policy_fingerprint(
        args.model,
        args.ne_graph,
        args.site_graph,
        _association_plan_config(config),
        artifact.config.topology_node_field,
    )
    validation["calibration_target_entity_count"] = len(calibration_entities)
    validation["seed"] = args.seed
    validation.update(target_data_metadata)
    validation["top_features_by_absolute_weight"] = _feature_importance(artifact)
    policy = CandidatePolicy(
        rules_by_alarm_pair=rules,
        fallback_rules=RELATED_RULES,
        approved=validation["approved"],
        fingerprint=fingerprint,
        validation=validation,
    )
    write_candidate_policy(args.output, policy)
    print(
        f"[candidate-policy] approved={policy.approved}, "
        f"recall={validation['recall']:.6f}, "
        f"lower_bound={validation['recall_lower_bound']:.6f}, "
        f"candidate_ratio={validation['candidate_ratio']:.6%}, "
        f"output={os.path.abspath(args.output)}",
        flush=True,
    )
    if not policy.approved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
