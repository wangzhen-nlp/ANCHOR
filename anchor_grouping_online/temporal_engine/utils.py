import collections

from anchor_grouping_online.alarm_events.identity import require_eid


def _normalize_edge_directions(direction):
    if isinstance(direction, str):
        return (direction,)
    return tuple(dict.fromkeys(direction))


def _reverse_edge_direction(direction):
    if direction == "downstream":
        return "upstream"
    if direction == "upstream":
        return "downstream"
    if direction == "bidirectional":
        return "bidirectional"
    raise ValueError(f"不支持的边方向: {direction!r}")


def _format_edge_directions(directions):
    return directions[0] if len(directions) == 1 else directions


def build_pattern_adj(edges_cfg):
    """把规则边展开成支持双向遍历的模式邻接表。"""
    pattern_adj = collections.defaultdict(list)
    for edge in edges_cfg:
        source, target = edge["source"], edge["target"]
        fwd_dirs = _normalize_edge_directions(edge["direction"])
        rev_dirs = tuple(dict.fromkeys(_reverse_edge_direction(direction) for direction in fwd_dirs))
        fwd_dir = _format_edge_directions(fwd_dirs)
        rev_dir = _format_edge_directions(rev_dirs)
        hops = edge.get("max_hops")
        win = edge["time_window_sec"]
        constraints = edge.get("constraints", {})
        dedupe_symmetric_pair = bool(constraints.get("dedupe_symmetric_pair"))
        optional = bool(edge.get("optional", False))

        pattern_adj[source].append({
            "role": target,
            "source_role": source,
            "target_role": target,
            "traverse_dir": fwd_dir,
            "hops": hops,
            "win": win,
            "dedupe_symmetric_pair": dedupe_symmetric_pair,
            "optional": optional,
        })
        pattern_adj[target].append({
            "role": source,
            "source_role": source,
            "target_role": target,
            "traverse_dir": rev_dir,
            "hops": hops,
            "win": win,
            "dedupe_symmetric_pair": dedupe_symmetric_pair,
            "optional": optional,
        })
    return pattern_adj


def _has_domain(source_domain, expected_domain):
    if isinstance(source_domain, dict):
        return source_domain.get(expected_domain, 0) > 0
    return str(expected_domain).strip().lower() == str(source_domain or "").strip().lower()


def _matches_source_domain(source_domain, domain_filter):
    if domain_filter is None:
        return True
    return any(_has_domain(source_domain, domain) for domain in domain_filter)


def matches_expected_alarm(alarm_type, expected, alarm_source_domain=None):
    """判断单条告警类型是否满足 expected_alarms 定义。"""
    required_alarms = expected.get("required_alarms")
    if required_alarms is None or alarm_type not in required_alarms:
        return False
    required_source_domains = expected.get("required_alarm_source_domains")
    if required_source_domains is None:
        return True
    return _matches_source_domain(alarm_source_domain, required_source_domains)


def get_symptom_alarm_identity(symptom):
    return require_eid(symptom)


def get_match_alarm_keys(match_result):
    alarm_keys = set()
    for symptom in match_result["symptoms"]:
        alarm_keys.add(get_symptom_alarm_identity(symptom))
    return alarm_keys


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
    """合并同一告警被多个规则或 role 命中的归属信息。"""
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
        for rule in source["merged_rules"]
        if str(rule or "").strip()
    ]
    for rule_name in merged_rules:
        if role.startswith(f"{rule_name}."):
            return role

    if len(merged_rules) == 1:
        return qualify_role_key(merged_rules[0], role)

    return role


def _add_nodes_to_role_mapping(role_mapping, role, nodes):
    role_mapping.setdefault(role, [])
    role_mapping[role] = sorted(set(role_mapping[role]) | set(nodes))


def merge_match_component(component_matches):
    """合并一组通过共享 eid 连通的候选组。"""
    merged_rules = sorted({
        rule_name
        for match in component_matches
        for rule_name in match["merged_rules"]
    })
    combined_rule_name = " + ".join(merged_rules)
    merged = {
        "rule": combined_rule_name,
        "merged_rules": merged_rules,
        "inferred_roots": {},
        "role_mapping": {},
        "symptoms": []
    }

    expire_ts_hint = max(source["_expire_ts_hint"] for source in component_matches)
    symptom_map = {}

    for source in component_matches:
        for role, nodes in source["inferred_roots"].items():
            role_key = _role_key_for_merged_source(source, role)
            _add_nodes_to_role_mapping(merged["inferred_roots"], role_key, nodes)

        for role, nodes in source["role_mapping"].items():
            role_key = _role_key_for_merged_source(source, role)
            _add_nodes_to_role_mapping(merged["role_mapping"], role_key, nodes)

        for symptom in source["symptoms"]:
            alarm_key = get_symptom_alarm_identity(symptom)
            existing_symptom = symptom_map.get(alarm_key)
            if existing_symptom is None:
                symptom_map[alarm_key] = symptom
            else:
                symptom_map[alarm_key] = merge_symptom_role_metadata(existing_symptom, symptom)

    merged["symptoms"] = list(symptom_map.values())
    merged["_expire_ts_hint"] = expire_ts_hint

    return merged


def normalize_match_symptoms(match):
    """按唯一告警 ID 去掉单个候选组内的重复发生。"""
    symptoms = match["symptoms"]
    if len(symptoms) <= 1:
        return match

    normalized_symptoms = []
    symptom_indexes = {}
    for symptom in symptoms:
        alarm_key = get_symptom_alarm_identity(symptom)
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


def merge_match_batch(matches):
    """在同一轮收割内，先把共享 eid 的候选组合并后再输出。"""
    if len(matches) <= 1:
        return [
            normalize_match_symptoms(match)
            for match in matches
        ]

    parent = list(range(len(matches)))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root
            return True
        return False

    match_primary_keys = [get_match_alarm_keys(match) for match in matches]
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
            union(head, idx)

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
        merged_matches.append(merge_match_component(component_matches))
    return merged_matches


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
