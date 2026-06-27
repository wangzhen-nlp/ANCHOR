import collections
import uuid

from collections.abc import Iterable

from fault_grouping.alarm_events.identity import require_alarm_identity
from fault_grouping.time_config import RULE_DEFAULT_EDGE_TIME_WINDOW_SEC


def _normalize_edge_directions(direction):
    if direction is None:
        return ("downstream",)
    if isinstance(direction, str):
        text = direction.strip()
        return (text,) if text else ("downstream",)
    if isinstance(direction, Iterable):
        directions = []
        seen = set()
        for item in direction:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            directions.append(text)
        return tuple(directions) if directions else ("downstream",)
    return (str(direction).strip() or "downstream",)


def _reverse_edge_direction(direction):
    if direction == "downstream":
        return "upstream"
    if direction == "upstream":
        return "downstream"
    if direction == "self":
        return "self"
    if direction == "either":
        return "either"
    if direction in {"bidirection", "bidirectional"}:
        return "bidirectional"
    return "bidirectional"


def _format_edge_directions(directions):
    return directions[0] if len(directions) == 1 else directions


def node_has_required_alarm_anchor(node_cfg):
    """判断节点配置是否同时具备 alarm_source_ne_anchor 与 required_alarms。

    mutual NE anchor 仅在边两端节点都"既配 anchor、又有 required 告警"时才有意义。
    """
    if not isinstance(node_cfg, dict) or not node_cfg.get("alarm_source_ne_anchor"):
        return False
    for site_rule in node_cfg.get("site_rules") or ():
        expected = site_rule.get("expected_alarms") if isinstance(site_rule, dict) else None
        if isinstance(expected, dict) and expected.get("required_alarms"):
            return True
    return False


def build_pattern_adj(edges_cfg):
    """把规则边展开成支持双向遍历的模式邻接表。"""
    pattern_adj = collections.defaultdict(list)
    for edge in edges_cfg:
        source, target = edge["source"], edge["target"]
        fwd_dirs = _normalize_edge_directions(edge.get("direction", "downstream"))
        rev_dirs = tuple(dict.fromkeys(_reverse_edge_direction(direction) for direction in fwd_dirs))
        fwd_dir = _format_edge_directions(fwd_dirs)
        rev_dir = _format_edge_directions(rev_dirs)
        hops = edge.get("max_hops")
        win = edge.get("time_window_sec", RULE_DEFAULT_EDGE_TIME_WINDOW_SEC)
        rev_win = win
        if isinstance(win, dict):
            rev_win = {
                "before_sec": win.get("after_sec", win.get("forward_sec", 0)),
                "after_sec": win.get("before_sec", win.get("backward_sec", 0))
            }
        constraints = edge.get("constraints", {})
        path_requirements = constraints.get("path_node_requirements")
        source_candidate_selector = constraints.get("source_candidate_selector")
        target_candidate_selector = constraints.get("target_candidate_selector")
        source_node = edge.get("source_node")
        target_node = edge.get("target_node")
        dedupe_symmetric_pair = bool(
            edge.get("dedupe_symmetric_pair")
            or constraints.get("dedupe_symmetric_pair")
        )
        mutual_alarm_source_ne_anchor = constraints.get("mutual_alarm_source_ne_anchor")
        if mutual_alarm_source_ne_anchor is None:
            source_anchor_cfg = (
                source_node.get("alarm_source_ne_anchor")
                if isinstance(source_node, dict) else None
            )
            target_anchor_cfg = (
                target_node.get("alarm_source_ne_anchor")
                if isinstance(target_node, dict) else None
            )
            if (
                source_anchor_cfg and target_anchor_cfg
                and node_has_required_alarm_anchor(source_node)
                and node_has_required_alarm_anchor(target_node)
            ):
                mutual_alarm_source_ne_anchor = {
                    "max_ne_hops": min(
                        int(source_anchor_cfg.get("max_ne_hops", 1)),
                        int(target_anchor_cfg.get("max_ne_hops", 1)),
                    )
                }
        optional = bool(edge.get("optional", False))

        pattern_adj[source].append({
            "role": target,
            "source_role": source,
            "target_role": target,
            "traverse_dir": fwd_dir,
            "hops": hops,
            "win": win,
            "path_requirements": path_requirements,
            "candidate_selector": target_candidate_selector,
            "dedupe_symmetric_pair": dedupe_symmetric_pair,
            "mutual_alarm_source_ne_anchor": mutual_alarm_source_ne_anchor,
            "optional": optional,
        })
        pattern_adj[target].append({
            "role": source,
            "source_role": source,
            "target_role": target,
            "traverse_dir": rev_dir,
            "hops": hops,
            "win": rev_win,
            "path_requirements": path_requirements,
            "candidate_selector": source_candidate_selector,
            "dedupe_symmetric_pair": dedupe_symmetric_pair,
            "mutual_alarm_source_ne_anchor": mutual_alarm_source_ne_anchor,
            "optional": optional,
        })
    return pattern_adj


def _normalize_domain_filter(domain_filter):
    if domain_filter is None:
        return None
    if isinstance(domain_filter, str):
        return [domain_filter]
    if isinstance(domain_filter, Iterable):
        return list(domain_filter)
    return None


def _has_domain(source_domain, expected_domain):
    if isinstance(source_domain, dict):
        value = source_domain.get(expected_domain)
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value not in ("", "0")
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return bool(value)
    if isinstance(source_domain, (list, tuple, set)):
        return expected_domain in source_domain
    return str(expected_domain).strip().lower() == str(source_domain or "").strip().lower()


def _matches_source_domain(source_domain, domain_filter):
    domains = _normalize_domain_filter(domain_filter)
    if domains is None:
        return domain_filter is None
    return any(_has_domain(source_domain, domain) for domain in domains)


def matches_expected_alarm(alarm_type, expected, alarm_source_domain=None):
    """判断单条告警类型是否满足 expected_alarms 定义。"""
    if expected in (None, "NONE"):
        return False
    if expected == "ANY":
        return True
    if isinstance(expected, dict):
        required_alarms = expected.get("required_alarms")
        if not (
            isinstance(required_alarms, Iterable)
            and not isinstance(required_alarms, str)
            and alarm_type in required_alarms
        ):
            return False
        required_source_domains = expected.get("required_alarm_source_domains")
        if required_source_domains is None:
            return True
        return _matches_source_domain(alarm_source_domain, required_source_domains)
    return isinstance(expected, Iterable) and alarm_type in expected


def get_symptom_alarm_identity(symptom, use_alarm_period_cache=False):
    del use_alarm_period_cache
    return require_alarm_identity(symptom)


def get_symptom_strong_occurrence_identity(symptom):
    return require_alarm_identity(symptom)


def get_match_alarm_keys(match_result, use_alarm_period_cache=False):
    alarm_keys = set()
    for symptom in match_result.get("symptoms", []):
        alarm_keys.add(get_symptom_alarm_identity(
            symptom,
            use_alarm_period_cache=use_alarm_period_cache,
        ))
    return alarm_keys


def get_symptom_overlap_base_key(symptom):
    node = symptom.get("node")
    alarm = symptom.get("alarm")
    alarm_source = symptom.get("alarm_source", "")
    if node not in (None, "") and alarm not in (None, ""):
        return (node, alarm_source, alarm)

    fallback_key = symptom.get("_segment_key")
    if fallback_key in (None, ""):
        fallback_key = symptom.get("eid")
    if fallback_key in (None, ""):
        return None
    return ("__fallback__", str(fallback_key))


def get_match_symptom_overlap_keys(match_result):
    overlap_keys = set()
    for symptom in match_result.get("symptoms", []):
        overlap_key = get_symptom_overlap_base_key(symptom)
        if overlap_key is not None:
            overlap_keys.add(overlap_key)
    return overlap_keys


def get_symptom_interval(symptom):
    start_ts = symptom.get("_segment_start_ts")
    if start_ts is None:
        start_ts = symptom.get("ts")
    if start_ts is None:
        return None, None

    end_ts = symptom.get("_segment_end_ts")
    if end_ts is None and "_segment_start_ts" not in symptom:
        end_ts = symptom.get("ts")
    return start_ts, end_ts


def _interval_end_sort_value(end_ts):
    return float("inf") if end_ts is None else end_ts


def _append_unique_value(values, value):
    if value in (None, "") or value in values:
        return
    values.append(value)


def _collect_symptom_values(symptom, single_field, list_field):
    values = []
    raw_values = symptom.get(list_field)
    if isinstance(raw_values, list):
        for value in raw_values:
            _append_unique_value(values, value)
    _append_unique_value(values, symptom.get(single_field))
    return values


def merge_symptom_role_metadata(existing_symptom, incoming_symptom):
    """合并同一告警/时段被多个规则或 role 命中的归属信息。"""
    merged = dict(existing_symptom)

    for single_field, list_field in (
        ("matched_rule", "matched_rule_list"),
        ("matched_role", "matched_role_list"),
        ("matched_role_key", "matched_role_key_list"),
    ):
        values = []
        for symptom in (existing_symptom, incoming_symptom):
            for value in _collect_symptom_values(symptom, single_field, list_field):
                _append_unique_value(values, value)
        if values:
            merged[list_field] = values
            if merged.get(single_field) in (None, ""):
                merged[single_field] = values[0]

    for field_name in (
        "node",
        "alarm",
        "alarm_source",
        "alarm_source_domain",
        "time_str",
    ):
        if merged.get(field_name) in (None, "") and incoming_symptom.get(field_name) not in (None, ""):
            merged[field_name] = incoming_symptom[field_name]

    return merged


def _merge_interval_end(left_end_ts, right_end_ts):
    if left_end_ts is None or right_end_ts is None:
        return None
    return max(left_end_ts, right_end_ts)


def symptoms_overlap(left_symptom, right_symptom):
    if get_symptom_strong_occurrence_identity(left_symptom) != get_symptom_strong_occurrence_identity(right_symptom):
        return False

    if get_symptom_overlap_base_key(left_symptom) != get_symptom_overlap_base_key(right_symptom):
        return False

    left_start_ts, left_end_ts = get_symptom_interval(left_symptom)
    right_start_ts, right_end_ts = get_symptom_interval(right_symptom)
    if left_start_ts is None or right_start_ts is None:
        return False

    return (
        left_start_ts <= _interval_end_sort_value(right_end_ts)
        and right_start_ts <= _interval_end_sort_value(left_end_ts)
    )


def symptom_covers(cover_symptom, target_symptom):
    if get_symptom_strong_occurrence_identity(cover_symptom) != get_symptom_strong_occurrence_identity(target_symptom):
        return False

    if get_symptom_overlap_base_key(cover_symptom) != get_symptom_overlap_base_key(target_symptom):
        return False

    cover_start_ts, cover_end_ts = get_symptom_interval(cover_symptom)
    target_start_ts, target_end_ts = get_symptom_interval(target_symptom)
    if cover_start_ts is None or target_start_ts is None:
        return False

    return (
        cover_start_ts <= target_start_ts
        and _interval_end_sort_value(cover_end_ts) >= _interval_end_sort_value(target_end_ts)
    )


def _build_merged_segment_key(symptom):
    base_key = get_symptom_overlap_base_key(symptom)
    if not base_key:
        return symptom.get("_segment_key") or symptom.get("eid")

    start_ts, end_ts = get_symptom_interval(symptom)
    if start_ts is None:
        return symptom.get("_segment_key") or symptom.get("eid")

    if len(base_key) == 3 and base_key[0] != "__fallback__":
        node, alarm_source, alarm = base_key
        return f"{node}|{alarm_source}|{alarm}|{start_ts:.6f}|{'open' if end_ts is None else f'{end_ts:.6f}'}"

    return f"{base_key[0]}|{base_key[1]}|{start_ts:.6f}|{'open' if end_ts is None else f'{end_ts:.6f}'}"


def merge_symptom_records(existing_symptom, incoming_symptom):
    existing_start_ts, existing_end_ts = get_symptom_interval(existing_symptom)
    incoming_start_ts, incoming_end_ts = get_symptom_interval(incoming_symptom)

    if existing_start_ts is None:
        base_symptom, other_symptom = incoming_symptom, existing_symptom
    elif incoming_start_ts is None:
        base_symptom, other_symptom = existing_symptom, incoming_symptom
    elif incoming_start_ts < existing_start_ts:
        base_symptom, other_symptom = incoming_symptom, existing_symptom
    else:
        base_symptom, other_symptom = existing_symptom, incoming_symptom
    merged = dict(base_symptom)

    merged_start_ts_candidates = [
        ts for ts in (existing_start_ts, incoming_start_ts)
        if ts is not None
    ]
    merged_start_ts = min(merged_start_ts_candidates) if merged_start_ts_candidates else None
    merged_end_ts = _merge_interval_end(existing_end_ts, incoming_end_ts)

    if merged_start_ts is not None:
        merged["ts"] = merged_start_ts
        merged["_segment_start_ts"] = merged_start_ts
    merged["_segment_end_ts"] = merged_end_ts

    for field_name in (
        "node",
        "alarm",
        "alarm_source",
        "alarm_source_domain",
        "matched_rule",
        "matched_role",
        "matched_role_key",
        "time_str",
    ):
        if not merged.get(field_name) and other_symptom.get(field_name):
            merged[field_name] = other_symptom[field_name]

    merged = merge_symptom_role_metadata(merged, other_symptom)

    if merged.get("eid") in (None, "") and other_symptom.get("eid") not in (None, ""):
        merged["eid"] = other_symptom["eid"]

    merged_eid_list = []
    for symptom in (existing_symptom, incoming_symptom):
        raw_eid_list = symptom.get("eid_list")
        if not isinstance(raw_eid_list, list):
            raw_eid_list = [symptom.get("eid")] if symptom.get("eid") not in (None, "") else []
        for event_id in raw_eid_list:
            if event_id in (None, "") or event_id in merged_eid_list:
                continue
            merged_eid_list.append(event_id)
    if merged_eid_list:
        merged["eid_list"] = merged_eid_list
        if merged.get("eid") in (None, ""):
            merged["eid"] = merged_eid_list[0]

    merged["_segment_key"] = _build_merged_segment_key(merged)
    return merged


def merge_overlapping_symptoms(symptoms):
    if len(symptoms) <= 1:
        return list(symptoms)

    grouped_symptoms = collections.defaultdict(list)
    passthrough_symptoms = []
    for symptom in symptoms:
        overlap_key = get_symptom_overlap_base_key(symptom)
        start_ts, _end_ts = get_symptom_interval(symptom)
        if overlap_key is None or start_ts is None:
            passthrough_symptoms.append(dict(symptom))
            continue
        # 同一个 node/source/alarm 下可能存在重复 eid 的多次独立发生。
        # occurrence 身份必须成为分组的一部分，否则 X/Y/X 交错排序时，
        # 中间的 Y 会打断 X 的归并，最终把同一发生输出两次。
        occurrence_identity = get_symptom_strong_occurrence_identity(symptom)
        grouped_symptoms[(overlap_key, occurrence_identity)].append(dict(symptom))

    passthrough_by_identity = {}
    for symptom in passthrough_symptoms:
        identity = get_symptom_alarm_identity(symptom)
        existing = passthrough_by_identity.get(identity)
        passthrough_by_identity[identity] = (
            symptom
            if existing is None
            else merge_symptom_role_metadata(existing, symptom)
        )
    merged_symptoms = list(passthrough_by_identity.values())
    for grouped_items in grouped_symptoms.values():
        grouped_items.sort(
            key=lambda symptom: (
                get_symptom_interval(symptom)[0],
                _interval_end_sort_value(get_symptom_interval(symptom)[1]),
                str(symptom.get("eid", "")),
            )
        )

        current = None
        for symptom in grouped_items:
            if current is None:
                current = symptom
                continue
            if symptoms_overlap(current, symptom):
                current = merge_symptom_records(current, symptom)
                continue
            merged_symptoms.append(current)
            current = symptom

        if current is not None:
            merged_symptoms.append(current)

    merged_symptoms.sort(
        key=lambda symptom: (
            symptom.get("_segment_start_ts", symptom.get("ts", float("inf"))),
            str(symptom.get("eid", "")),
        )
    )
    return merged_symptoms


def get_match_site_keys(match_result):
    site_keys = set()

    for nodes in match_result.get("role_mapping", {}).values():
        for node in nodes:
            if node not in (None, ""):
                site_keys.add(node)

    for symptom in match_result.get("symptoms", []):
        node = symptom.get("node")
        if node not in (None, ""):
            site_keys.add(node)

    return site_keys


def qualify_role_key(rule_name, role):
    rule_name = str(rule_name or "").strip()
    role = str(role or "").strip()
    if not role:
        return role
    if not rule_name:
        return role
    qualified_prefix = f"{rule_name}."
    if role.startswith(qualified_prefix):
        return role
    return f"{qualified_prefix}{role}"


def _role_key_for_merged_source(source, role):
    role = str(role or "").strip()
    if not role:
        return role

    merged_rules = [
        str(rule).strip()
        for rule in source.get("merged_rules", [source.get("rule")])
        if str(rule or "").strip()
    ]
    for rule_name in merged_rules:
        if role.startswith(f"{rule_name}."):
            return role

    if len(merged_rules) == 1:
        return qualify_role_key(merged_rules[0], role)

    source_rule = str(source.get("rule") or "").strip()
    if source_rule and " + " not in source_rule:
        return qualify_role_key(source_rule, role)

    return role


def _add_nodes_to_role_mapping(role_mapping, role, nodes):
    role_mapping.setdefault(role, [])
    role_mapping[role] = sorted(set(role_mapping[role]) | set(nodes))


def merge_match_component(component_matches, use_alarm_period_cache=False):
    """合并一组通过 eid 或告警时段 overlap 连通的候选组。"""
    merged_rules = sorted({
        rule_name
        for match in component_matches
        for rule_name in match.get("merged_rules", [match.get("rule")])
        if rule_name
    })
    combined_rule_name = " + ".join(merged_rules) if merged_rules else "UNKNOWN_RULE"
    merged = {
        "uuid": str(uuid.uuid4()),
        "rule": combined_rule_name,
        "merged_rules": merged_rules,
        "inferred_roots": {},
        "role_mapping": {},
        "symptoms": []
    }

    related_group_uuids = set()
    missing_topology_edges = {}
    expire_ts_hint = None
    collected_symptoms = []
    symptom_map = {}

    for source in component_matches:
        source_expire_ts_hint = source.get("_expire_ts_hint")
        if source_expire_ts_hint is not None:
            expire_ts_hint = source_expire_ts_hint if expire_ts_hint is None else max(expire_ts_hint, source_expire_ts_hint)

        for role, nodes in source.get("inferred_roots", {}).items():
            role_key = _role_key_for_merged_source(source, role)
            _add_nodes_to_role_mapping(merged["inferred_roots"], role_key, nodes)

        for role, nodes in source.get("role_mapping", {}).items():
            role_key = _role_key_for_merged_source(source, role)
            _add_nodes_to_role_mapping(merged["role_mapping"], role_key, nodes)

        if use_alarm_period_cache:
            collected_symptoms.extend(source.get("symptoms", []))
        else:
            for symptom in source.get("symptoms", []):
                alarm_key = get_symptom_alarm_identity(symptom, use_alarm_period_cache=False)
                existing_symptom = symptom_map.get(alarm_key)
                if existing_symptom is None:
                    symptom_map[alarm_key] = symptom
                else:
                    symptom_map[alarm_key] = merge_symptom_role_metadata(existing_symptom, symptom)

        related_group_uuids.update(source.get("related_group_uuids", []))
        for edge in source.get("missing_topology_edges", []):
            if not isinstance(edge, dict):
                continue
            edge_key = (
                str(edge.get("source_site", "")),
                str(edge.get("target_site", "")),
                str(edge.get("relation", "")),
                str(edge.get("sample_id", "")),
            )
            missing_topology_edges[edge_key] = dict(edge)

    if use_alarm_period_cache:
        merged["symptoms"] = merge_overlapping_symptoms(collected_symptoms)
    else:
        merged["symptoms"] = list(symptom_map.values())
    if related_group_uuids:
        merged["related_group_uuids"] = sorted(related_group_uuids)
    if missing_topology_edges:
        merged["uses_missing_topology"] = True
        merged["missing_topology_edges"] = sorted(
            missing_topology_edges.values(),
            key=lambda item: (
                str(item.get("source_site", "")),
                str(item.get("target_site", "")),
                str(item.get("relation", "")),
                str(item.get("sample_id", "")),
            ),
        )
    if expire_ts_hint is not None:
        merged["_expire_ts_hint"] = expire_ts_hint

    return merged


def build_empty_merge_stats():
    return {
        "alarm_overlap_merge_group_count": 0,
        "eid_merge_group_count": 0,
        "shared_site_merge_group_count": 0,
        "hop_merge_group_count": 0,
        "distance_merge_group_count": 0,
    }


def add_merge_stats(*stats_list):
    total = build_empty_merge_stats()
    for stats in stats_list:
        if not stats:
            continue
        for key in total:
            total[key] += int(stats.get(key, 0) or 0)
    return total


def normalize_match_symptoms(match, use_alarm_period_cache=False):
    """去掉单个候选组内的重复发生，同时保留重复 eid 的不同 occurrence。"""
    symptoms = match.get("symptoms", [])
    if len(symptoms) <= 1:
        return match

    if use_alarm_period_cache:
        normalized_symptoms = merge_overlapping_symptoms(symptoms)
    else:
        normalized_symptoms = []
        symptom_indexes = {}
        for symptom in symptoms:
            alarm_key = get_symptom_alarm_identity(symptom, use_alarm_period_cache=False)
            existing_idx = symptom_indexes.get(alarm_key)
            if existing_idx is None:
                symptom_indexes[alarm_key] = len(normalized_symptoms)
                normalized_symptoms.append(symptom)
            else:
                normalized_symptoms[existing_idx] = merge_symptom_role_metadata(
                    normalized_symptoms[existing_idx],
                    symptom,
                )

    if len(normalized_symptoms) == len(symptoms) and all(
        normalized is original
        for normalized, original in zip(normalized_symptoms, symptoms)
    ):
        return match

    normalized_match = dict(match)
    normalized_match["symptoms"] = normalized_symptoms
    return normalized_match


def merge_match_batch(matches, site_merge_helper=None, return_stats=False, use_alarm_period_cache=False):
    """在同一轮收割内，先把共享 eid 或告警时段重叠的候选组合并后再输出。"""
    merge_stats = build_empty_merge_stats()
    if len(matches) <= 1:
        normalized_matches = [
            normalize_match_symptoms(match, use_alarm_period_cache=use_alarm_period_cache)
            for match in matches
        ]
        return (normalized_matches, merge_stats) if return_stats else normalized_matches

    parent = list(range(len(matches)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right, reason=None):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root
            if reason:
                stat_key = f"{reason}_merge_group_count"
                if stat_key in merge_stats:
                    merge_stats[stat_key] += 1
            return True
        return False

    if use_alarm_period_cache:
        match_primary_keys = [get_match_symptom_overlap_keys(match) for match in matches]
        overlap_key_to_entries = collections.defaultdict(list)
        for idx, match in enumerate(matches):
            for symptom in match.get("symptoms", []):
                overlap_key = get_symptom_overlap_base_key(symptom)
                start_ts, end_ts = get_symptom_interval(symptom)
                if overlap_key is None or start_ts is None:
                    continue
                occurrence_identity = get_symptom_strong_occurrence_identity(symptom)
                overlap_key_to_entries[overlap_key].append((start_ts, end_ts, idx, occurrence_identity))

        for entries in overlap_key_to_entries.values():
            if len(entries) < 2:
                continue

            entries_by_occurrence = collections.defaultdict(list)
            for entry in entries:
                entries_by_occurrence[entry[3]].append(entry)

            # 各 occurrence 身份独立扫描。不能用一条全局的
            # component_head：不同 occurrence 交错排列会把同一发生的重叠链打断。
            for same_occurrence_entries in entries_by_occurrence.values():
                same_occurrence_entries.sort(
                    key=lambda item: (item[0], _interval_end_sort_value(item[1]), item[2])
                )
                component_head = None
                component_end_ts = None
                for start_ts, end_ts, idx, _occurrence_identity in same_occurrence_entries:
                    if component_head is None or start_ts > _interval_end_sort_value(component_end_ts):
                        component_head = idx
                        component_end_ts = end_ts
                        continue
                    union(component_head, idx, reason="alarm_overlap")
                    component_end_ts = _merge_interval_end(component_end_ts, end_ts)
    else:
        match_primary_keys = [get_match_alarm_keys(match, use_alarm_period_cache=False) for match in matches]
        eid_to_match_indexes = collections.defaultdict(list)
        for idx, alarm_keys in enumerate(match_primary_keys):
            if not alarm_keys:
                continue
            for alarm_key in alarm_keys:
                eid_to_match_indexes[alarm_key].append(idx)

        for indexes in eid_to_match_indexes.values():
            if len(indexes) < 2:
                continue
            head = indexes[0]
            for idx in indexes[1:]:
                union(head, idx, reason="eid")

    if site_merge_helper is not None and site_merge_helper.enabled:
        match_site_keys = [get_match_site_keys(match) for match in matches]
        active_indexes = [idx for idx, site_keys in enumerate(match_site_keys) if site_keys]
        for pos, left_idx in enumerate(active_indexes):
            left_sites = match_site_keys[left_idx]
            for right_idx in active_indexes[pos + 1:]:
                if find(left_idx) == find(right_idx):
                    continue
                reason = site_merge_helper.classify_component_adjacency(
                    left_sites,
                    match_site_keys[right_idx],
                )
                if reason:
                    union(left_idx, right_idx, reason=reason)

    groups = collections.defaultdict(list)
    group_indexes = collections.defaultdict(list)
    for idx, match in enumerate(matches):
        root_idx = find(idx)
        groups[root_idx].append(match)
        group_indexes[root_idx].append(idx)

    merged_matches = []
    for root_idx, component_matches in groups.items():
        indexes = group_indexes[root_idx]
        if len(component_matches) == 1 and not match_primary_keys[indexes[0]]:
            merged_matches.append(matches[indexes[0]])
            continue
        merged_matches.append(
            merge_match_component(
                component_matches,
                use_alarm_period_cache=use_alarm_period_cache,
            )
        )
    return (merged_matches, merge_stats) if return_stats else merged_matches


def clone_instance_with_updates(inst, curr_role, surviving_curr_phys, tgt_role, tgt_nodes):
    """只复制当前需要修改的角色分支，避免整棵实例 deepcopy。"""
    new_inst = dict(inst)
    roles = inst.get("roles", {})
    new_inst["roles"] = {
        role: {
            "nodes": dict(role_state["nodes"]),
            "checked": role_state["checked"]
        }
        for role, role_state in roles.items()
    }
    if "_dependencies" in inst:
        new_inst["_dependencies"] = {
            dep_key: {
                "src_to_dst": {
                    src_node: set(dst_nodes)
                    for src_node, dst_nodes in dep_value.get("src_to_dst", {}).items()
                },
                "dst_to_src": {
                    dst_node: set(src_nodes)
                    for dst_node, src_nodes in dep_value.get("dst_to_src", {}).items()
                },
            }
            for dep_key, dep_value in inst["_dependencies"].items()
        }
    curr_entry = roles[curr_role]
    new_inst["roles"][curr_role] = {
        "nodes": dict(surviving_curr_phys),
        "checked": curr_entry["checked"]
    }
    new_inst["roles"][tgt_role] = {
        "nodes": dict(tgt_nodes),
        "checked": True
    }
    return new_inst
