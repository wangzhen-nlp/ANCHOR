import collections
import uuid

from collections.abc import Iterable


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
        win = edge.get("time_window_sec", 300)
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
        dedupe_symmetric_pair = bool(
            edge.get("dedupe_symmetric_pair")
            or constraints.get("dedupe_symmetric_pair")
        )
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
            "optional": optional,
        })
    return pattern_adj


def matches_expected_alarm(alarm_type, expected):
    """判断单条告警类型是否满足 expected_alarms 定义。"""
    if expected in (None, "NONE"):
        return False
    if expected == "ANY":
        return True
    if isinstance(expected, dict):
        required_alarms = expected.get("required_alarms")
        return (
            isinstance(required_alarms, Iterable)
            and not isinstance(required_alarms, str)
            and alarm_type in required_alarms
        )
    return isinstance(expected, Iterable) and alarm_type in expected


def get_match_alarm_keys(match_result):
    alarm_keys = set()
    for symptom in match_result.get("symptoms", []):
        alarm_key = symptom.get("eid")
        if alarm_key not in (None, ""):
            alarm_keys.add(alarm_key)
    return alarm_keys


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
    expire_ts_hint = None

    for source in component_matches:
        source_expire_ts_hint = source.get("_expire_ts_hint")
        if source_expire_ts_hint is not None:
            expire_ts_hint = source_expire_ts_hint if expire_ts_hint is None else max(expire_ts_hint, source_expire_ts_hint)

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
    if expire_ts_hint is not None:
        merged["_expire_ts_hint"] = expire_ts_hint

    return merged


def build_empty_merge_stats():
    return {
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


def merge_match_batch(matches, site_merge_helper=None, return_stats=False):
    """在同一轮收割内，先把共享 eid 的候选组合并后再输出。"""
    merge_stats = build_empty_merge_stats()
    if len(matches) <= 1:
        return (matches, merge_stats) if return_stats else matches

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

    match_alarm_keys = [get_match_alarm_keys(match) for match in matches]
    eid_to_match_indexes = collections.defaultdict(list)
    for idx, alarm_keys in enumerate(match_alarm_keys):
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
        if len(component_matches) == 1 and not match_alarm_keys[indexes[0]]:
            merged_matches.append(matches[indexes[0]])
            continue
        merged_matches.append(merge_match_component(component_matches))
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
