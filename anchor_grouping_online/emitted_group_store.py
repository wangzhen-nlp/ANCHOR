import collections

from anchor_grouping_online.temporal_engine.utils import (
    _add_nodes_to_role_mapping,
    _role_key_for_merged_source,
    get_match_alarm_keys,
    get_symptom_alarm_identity,
    merge_symptom_role_metadata,
)


# 已知会出现在 match dict 里的顶层 dict-of-list 字段。
_MATCH_TOP_DICT_OF_LIST_FIELDS = ("inferred_roots", "role_mapping")
# 已知会出现在 symptom dict 里的 list 字段。
_SYMPTOM_LIST_FIELDS = (
    "matched_rule_list",
    "matched_role_list",
    "matched_role_key_list",
)


def _shallow_copy_symptom(symptom):
    """与 dict(symptom) 等价但额外复制已知 list 字段；不递归 list 元素（都是 immutable str）。"""
    copied = dict(symptom)
    for field in _SYMPTOM_LIST_FIELDS:
        value = copied.get(field)
        if value is not None:
            copied[field] = list(value)
    return copied


def _structured_shallow_copy_match(match):
    """copy.deepcopy(match) 的快速等价实现。

    覆盖所有当前 codebase 里会被存进 EmittedGroupStore 的 match 字段，只在容器
    层做隔离（顶层 dict、merged_rules 列表、inferred_roots /
    role_mapping 的 list value、symptoms 内的 dict 与已知 list 子字段）。
    其余字段都是 str / number / bool 等
    immutable 值，不需要复制。

    经实测，copy.deepcopy 在 5000 事件 benchmark 上累计耗时约 0.85s（21%），
    该函数把它降到几十毫秒。
    """
    copied = dict(match)
    copied["merged_rules"] = list(match["merged_rules"])
    for field in _MATCH_TOP_DICT_OF_LIST_FIELDS:
        copied[field] = {
            role: list(nodes)
            for role, nodes in match[field].items()
        }
    copied["symptoms"] = [_shallow_copy_symptom(symptom) for symptom in match["symptoms"]]
    return copied


class EmittedGroupStore:
    """管理历史故障组的保留，并按 eid 重新合并落库。"""

    def __init__(self):
        self.groups = []
        self.eid_to_group_indexes = collections.defaultdict(set)
        self.deleted_group_count = 0

    def prune_expired(self, current_time):
        """移除已经超过停留时间的历史故障组。"""
        pruned_groups = [
            item for item in self.groups
            if item is not None and current_time <= item["expire_ts"]
        ]
        if len(pruned_groups) != len(self.groups):
            self.groups = pruned_groups
            self.deleted_group_count = 0
            self._rebuild_alarm_index()

    def merge_with_related(self, match_result):
        """按 eid 与历史故障组合并，生成更完整的当前故障组。"""
        return self._merge_with_related_by_eid(match_result)

    @staticmethod
    def _qualified_role_mapping(match_result, field_name):
        qualified = {}
        for role, nodes in match_result[field_name].items():
            role_key = _role_key_for_merged_source(match_result, role)
            _add_nodes_to_role_mapping(qualified, role_key, nodes)
        return qualified

    def _merge_with_related_by_eid(self, match_result):
        related_groups = []
        current_alarm_keys = self._get_alarm_keys(match_result["symptoms"])

        related_indexes = sorted({
            idx
            for alarm_key in current_alarm_keys
            for idx in self.eid_to_group_indexes.get(alarm_key, set())
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None
        })
        for idx in related_indexes:
            related_groups.append((idx, self.groups[idx]))

        if not related_groups:
            return match_result, set(), True

        merged = {
            "rule": match_result["rule"],
            "merged_rules": list(match_result["merged_rules"]),
            "inferred_roots": self._qualified_role_mapping(match_result, "inferred_roots"),
            "role_mapping": self._qualified_role_mapping(match_result, "role_mapping"),
            "symptoms": list(match_result["symptoms"]),
            "_expire_ts_hint": match_result["_expire_ts_hint"],
        }

        symptom_map = {}
        for symptom in merged["symptoms"]:
            alarm_key = self._get_alarm_key(symptom)
            existing_symptom = symptom_map.get(alarm_key)
            if existing_symptom is None:
                symptom_map[alarm_key] = symptom
            else:
                symptom_map[alarm_key] = merge_symptom_role_metadata(
                    existing_symptom,
                    symptom,
                )

        merged_group_indexes = set()
        fully_containing_history_exists = False
        for idx, item in related_groups:
            merged_group_indexes.add(idx)
            previous_match = item["match"]
            previous_alarm_keys = item["alarm_keys"]
            if current_alarm_keys.issubset(previous_alarm_keys):
                fully_containing_history_exists = True
            previous_merged_rules = previous_match["merged_rules"]
            merged["merged_rules"] = sorted(set(merged["merged_rules"]) | {rule for rule in previous_merged_rules if rule})
            for role, nodes in previous_match["inferred_roots"].items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["inferred_roots"], role_key, nodes)

            for role, nodes in previous_match["role_mapping"].items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["role_mapping"], role_key, nodes)

            for symptom in previous_match["symptoms"]:
                alarm_key = self._get_alarm_key(symptom)
                existing_symptom = symptom_map.get(alarm_key)
                if existing_symptom is None:
                    symptom_map[alarm_key] = symptom
                else:
                    # 同一发生同时出现在当前组与历史组：保留当前轮的 symptom 为主，
                    # 但合并历史组的 role/规则归属，避免丢失某一侧的命中信息。
                    symptom_map[alarm_key] = merge_symptom_role_metadata(existing_symptom, symptom)

        merged["symptoms"] = list(symptom_map.values())

        if fully_containing_history_exists:
            return merged, merged_group_indexes, False

        return merged, merged_group_indexes, True

    def replace_and_store(self, merged_group_indexes, match_result):
        """删除被吸收的历史组，并把当前组作为新的历史版本落库。"""
        current_expire_ts = match_result.pop("_expire_ts_hint")
        merged_expire_ts = max(
            (
                self.groups[idx]["expire_ts"]
                for idx in merged_group_indexes
                if 0 <= idx < len(self.groups) and self.groups[idx] is not None
            ),
            default=current_expire_ts
        )

        if merged_group_indexes:
            for idx in merged_group_indexes:
                if 0 <= idx < len(self.groups) and self.groups[idx] is not None:
                    self._remove_group_from_index(idx, self.groups[idx])
                    self.groups[idx] = None
                    self.deleted_group_count += 1

        stored_match = _structured_shallow_copy_match(match_result)
        group_item = {
            "expire_ts": max(current_expire_ts, merged_expire_ts),
            "match": stored_match,
            "alarm_keys": self._get_alarm_keys(stored_match["symptoms"]),
        }
        self.groups.append(group_item)
        self._add_group_to_index(len(self.groups) - 1, group_item)
        self._maybe_compact_groups()

    def extend_related_expire_ts(self, merged_group_indexes, match_result):
        """当当前结果不需要再次输出时，延长相关历史组的过期时间。"""
        if not merged_group_indexes:
            return

        current_expire_ts = match_result["_expire_ts_hint"]

        for idx in merged_group_indexes:
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None:
                self.groups[idx]["expire_ts"] = max(self.groups[idx]["expire_ts"], current_expire_ts)

    def _add_group_to_index(self, idx, group_item):
        for alarm_key in group_item["alarm_keys"]:
            self.eid_to_group_indexes[alarm_key].add(idx)

    def _remove_group_from_index(self, idx, group_item):
        for alarm_key in group_item["alarm_keys"]:
            indexes = self.eid_to_group_indexes.get(alarm_key)
            if not indexes:
                continue
            indexes.discard(idx)
            if not indexes:
                self.eid_to_group_indexes.pop(alarm_key, None)

    def _rebuild_alarm_index(self):
        self.eid_to_group_indexes = collections.defaultdict(set)
        for idx, group_item in enumerate(self.groups):
            if group_item is None:
                continue
            self._add_group_to_index(idx, group_item)

    def _maybe_compact_groups(self):
        if self.deleted_group_count <= 0:
            return
        if len(self.groups) < 1024:
            return
        if self.deleted_group_count <= len(self.groups) // 2:
            return

        self.groups = [group_item for group_item in self.groups if group_item is not None]
        self.deleted_group_count = 0
        self._rebuild_alarm_index()

    @staticmethod
    def _get_alarm_keys(symptoms):
        return set(get_match_alarm_keys({"symptoms": list(symptoms)}))

    @staticmethod
    def _get_alarm_key(symptom):
        return get_symptom_alarm_identity(symptom)
