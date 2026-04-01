import collections
import uuid

from collections.abc import Iterable


def build_pattern_adj(edges_cfg):
    """把规则边展开成支持双向遍历的模式邻接表。"""
    pattern_adj = collections.defaultdict(list)
    for edge in edges_cfg:
        source, target = edge["source"], edge["target"]
        fwd_dir = edge.get("direction", "downstream")
        rev_dir = "upstream" if fwd_dir == "downstream" else (
            "downstream" if fwd_dir == "upstream" else "bidirectional")
        hops = edge.get("max_hops")
        win = edge.get("time_window_sec", 300)
        constraints = edge.get("constraints", {})
        path_requirements = constraints.get("path_node_requirements")
        source_candidate_selector = constraints.get("source_candidate_selector")
        target_candidate_selector = constraints.get("target_candidate_selector")

        pattern_adj[source].append({
            "role": target,
            "traverse_dir": fwd_dir,
            "hops": hops,
            "win": win,
            "path_requirements": path_requirements,
            "candidate_selector": target_candidate_selector
        })
        pattern_adj[target].append({
            "role": source,
            "traverse_dir": rev_dir,
            "hops": hops,
            "win": win,
            "path_requirements": path_requirements,
            "candidate_selector": source_candidate_selector
        })
    return pattern_adj


def matches_expected_alarm(alarm_type, expected):
    """判断单条告警类型是否满足 expected_alarms 定义。"""
    if expected in (None, "NONE"):
        return False
    if expected == "ANY":
        return True
    return isinstance(expected, Iterable) and alarm_type in expected


def get_match_alarm_keys(match_result):
    alarm_keys = set()
    for symptom in match_result.get("symptoms", []):
        alarm_key = symptom.get("eid")
        if alarm_key not in (None, ""):
            alarm_keys.add(alarm_key)
    return alarm_keys


def merge_match_component(component_matches):
    """合并一组通过 eid 连通的候选组。"""
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

    symptom_map = {}
    related_group_uuids = set()

    for source in component_matches:
        for role, nodes in source.get("inferred_roots", {}).items():
            merged["inferred_roots"].setdefault(role, [])
            merged["inferred_roots"][role] = sorted(set(merged["inferred_roots"][role]) | set(nodes))

        for role, nodes in source.get("role_mapping", {}).items():
            merged["role_mapping"].setdefault(role, [])
            merged["role_mapping"][role] = sorted(set(merged["role_mapping"][role]) | set(nodes))

        for symptom in source.get("symptoms", []):
            alarm_key = symptom.get("eid")
            if alarm_key in (None, ""):
                continue
            symptom_map[alarm_key] = symptom

        related_group_uuids.update(source.get("related_group_uuids", []))

    merged["symptoms"] = list(symptom_map.values())
    if related_group_uuids:
        merged["related_group_uuids"] = sorted(related_group_uuids)

    return merged


def merge_match_batch(matches):
    """在同一轮收割内，先把共享 eid 的候选组合并后再输出。"""
    if len(matches) <= 1:
        return matches

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

    eid_to_match_indexes = collections.defaultdict(list)
    standalone_indexes = []
    for idx, match in enumerate(matches):
        alarm_keys = get_match_alarm_keys(match)
        if not alarm_keys:
            standalone_indexes.append(idx)
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
    for idx, match in enumerate(matches):
        alarm_keys = get_match_alarm_keys(match)
        if not alarm_keys:
            continue
        groups[find(idx)].append(match)

    merged_matches = [merge_match_component(component_matches) for component_matches in groups.values()]
    merged_matches.extend(matches[idx] for idx in standalone_indexes)
    return merged_matches


def clone_instance_with_updates(inst, curr_role, surviving_curr_phys, tgt_role, tgt_nodes):
    """只复制当前需要修改的角色分支，避免整棵实例 deepcopy。"""
    new_inst = dict(inst)
    curr_entry = inst[curr_role]
    new_inst[curr_role] = {
        "nodes": dict(surviving_curr_phys),
        "checked": curr_entry["checked"]
    }
    new_inst[tgt_role] = {
        "nodes": dict(tgt_nodes),
        "checked": True
    }
    return new_inst
