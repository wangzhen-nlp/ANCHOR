import copy
import collections

from fault_grouping.temporal_engine.utils import (
    _add_nodes_to_role_mapping,
    _role_key_for_merged_source,
    get_match_alarm_keys,
    get_match_symptom_overlap_keys,
    get_symptom_overlap_base_key,
    merge_overlapping_symptoms,
    symptom_covers,
    symptoms_overlap,
)


class EmittedGroupStore:
    """管理历史故障组的保留，并按 eid 或告警时段重新合并落库。"""

    def __init__(self, rules, default_stay_time, use_alarm_period_cache=False):
        self.rules = rules
        self.default_stay_time = default_stay_time
        self.use_alarm_period_cache = bool(use_alarm_period_cache)
        self.groups = []
        self.eid_to_group_indexes = collections.defaultdict(set)
        self.symptom_overlap_to_group_indexes = collections.defaultdict(set)
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

    def get_group_anchor_ts(self, match_result, fallback_ts):
        """提取故障组自身最早告警时间，作为停留时间锚点。"""
        timestamps = [symptom["ts"] for symptom in match_result.get("symptoms", []) if "ts" in symptom]
        if timestamps:
            return min(timestamps)
        return fallback_ts

    def get_rule_max_stay_time(self, rule_name):
        """读取规则配置中的最大停留时间。"""
        rule = self.rules.get(rule_name, {})
        return rule.get("max_stay_time_sec", self.default_stay_time)

    def merge_with_related(self, match_result):
        """按当前模式与历史故障组合并，生成更完整的当前故障组。"""
        if not self.use_alarm_period_cache:
            return self._merge_with_related_by_eid(match_result)
        return self._merge_with_related_by_overlap(match_result)

    @staticmethod
    def _qualified_role_mapping(match_result, field_name):
        qualified = {}
        for role, nodes in match_result.get(field_name, {}).items():
            role_key = _role_key_for_merged_source(match_result, role)
            _add_nodes_to_role_mapping(qualified, role_key, nodes)
        return qualified

    def _merge_with_related_by_eid(self, match_result):
        related_groups = []
        current_alarm_keys = self._get_alarm_keys(match_result.get("symptoms", []))

        if not current_alarm_keys:
            return match_result, set(), set(), True, "no_alarm_keys"

        related_indexes = sorted({
            idx
            for alarm_key in current_alarm_keys
            for idx in self.eid_to_group_indexes.get(alarm_key, set())
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None
        })
        for idx in related_indexes:
            related_groups.append((idx, self.groups[idx]))

        if not related_groups:
            return match_result, set(), set(), True, "no_related_history"

        merged = {
            "uuid": match_result.get("uuid"),
            "rule": match_result.get("rule"),
            "merged_rules": list(match_result.get("merged_rules", [match_result.get("rule")])),
            "inferred_roots": self._qualified_role_mapping(match_result, "inferred_roots"),
            "role_mapping": self._qualified_role_mapping(match_result, "role_mapping"),
            "symptoms": list(match_result.get("symptoms", []))
        }
        if "_expire_ts_hint" in match_result:
            merged["_expire_ts_hint"] = match_result["_expire_ts_hint"]

        symptom_map = {}
        for symptom in merged["symptoms"]:
            alarm_key = self._get_alarm_key(symptom)
            if alarm_key is not None:
                symptom_map[alarm_key] = symptom

        merged_group_indexes = set()
        related_group_uuids = set()
        fully_containing_history_exists = False
        for idx, item in related_groups:
            merged_group_indexes.add(idx)
            previous_match = item["match"]
            previous_alarm_keys = item.get("alarm_keys")
            if previous_alarm_keys is None:
                previous_alarm_keys = self._get_alarm_keys(previous_match.get("symptoms", []))
                item["alarm_keys"] = previous_alarm_keys
            if current_alarm_keys.issubset(previous_alarm_keys):
                fully_containing_history_exists = True
            previous_uuid = previous_match.get("uuid")
            if previous_uuid:
                related_group_uuids.add(previous_uuid)
            related_group_uuids.update(previous_match.get("related_group_uuids", []))
            previous_merged_rules = previous_match.get("merged_rules", [previous_match.get("rule")])
            merged["merged_rules"] = sorted(set(merged["merged_rules"]) | {rule for rule in previous_merged_rules if rule})
            for role, nodes in previous_match.get("inferred_roots", {}).items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["inferred_roots"], role_key, nodes)

            for role, nodes in previous_match.get("role_mapping", {}).items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["role_mapping"], role_key, nodes)

            for symptom in previous_match.get("symptoms", []):
                alarm_key = self._get_alarm_key(symptom)
                if alarm_key is not None:
                    symptom_map[alarm_key] = symptom

        merged["symptoms"] = list(symptom_map.values())

        if fully_containing_history_exists:
            return merged, merged_group_indexes, related_group_uuids, False, "suppressed_by_fully_containing_history"

        return merged, merged_group_indexes, related_group_uuids, True, "merged_with_related_history"

    def _merge_with_related_by_overlap(self, match_result):
        related_groups = []
        current_symptoms = list(match_result.get("symptoms", []))
        current_overlap_keys = get_match_symptom_overlap_keys(match_result)

        if not current_overlap_keys:
            return match_result, set(), set(), True, "no_alarm_keys"

        related_indexes = sorted({
            idx
            for overlap_key in current_overlap_keys
            for idx in self.symptom_overlap_to_group_indexes.get(overlap_key, set())
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None
        })
        for idx in related_indexes:
            group_item = self.groups[idx]
            previous_match = group_item["match"]
            if self._matches_any_overlap(current_symptoms, previous_match.get("symptoms", [])):
                related_groups.append((idx, group_item))

        if not related_groups:
            return match_result, set(), set(), True, "no_related_history"

        merged = {
            "uuid": match_result.get("uuid"),
            "rule": match_result.get("rule"),
            "merged_rules": list(match_result.get("merged_rules", [match_result.get("rule")])),
            "inferred_roots": self._qualified_role_mapping(match_result, "inferred_roots"),
            "role_mapping": self._qualified_role_mapping(match_result, "role_mapping"),
            "symptoms": list(match_result.get("symptoms", []))
        }
        if "_expire_ts_hint" in match_result:
            merged["_expire_ts_hint"] = match_result["_expire_ts_hint"]

        merged_group_indexes = set()
        related_group_uuids = set()
        fully_containing_history_exists = False
        for idx, item in related_groups:
            merged_group_indexes.add(idx)
            previous_match = item["match"]
            previous_overlap_keys = item.get("symptom_overlap_keys")
            if previous_overlap_keys is None:
                previous_overlap_keys = get_match_symptom_overlap_keys(previous_match)
                item["symptom_overlap_keys"] = previous_overlap_keys
            if current_overlap_keys.issubset(previous_overlap_keys) and self._all_symptoms_covered(
                current_symptoms,
                previous_match.get("symptoms", []),
            ):
                fully_containing_history_exists = True
            previous_uuid = previous_match.get("uuid")
            if previous_uuid:
                related_group_uuids.add(previous_uuid)
            related_group_uuids.update(previous_match.get("related_group_uuids", []))
            previous_merged_rules = previous_match.get("merged_rules", [previous_match.get("rule")])
            merged["merged_rules"] = sorted(set(merged["merged_rules"]) | {rule for rule in previous_merged_rules if rule})
            for role, nodes in previous_match.get("inferred_roots", {}).items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["inferred_roots"], role_key, nodes)

            for role, nodes in previous_match.get("role_mapping", {}).items():
                role_key = _role_key_for_merged_source(previous_match, role)
                _add_nodes_to_role_mapping(merged["role_mapping"], role_key, nodes)

            merged["symptoms"].extend(previous_match.get("symptoms", []))

        merged["symptoms"] = merge_overlapping_symptoms(merged["symptoms"])

        if fully_containing_history_exists:
            return merged, merged_group_indexes, related_group_uuids, False, "suppressed_by_fully_containing_history"

        return merged, merged_group_indexes, related_group_uuids, True, "merged_with_related_history"

    def replace_and_store(self, merged_group_indexes, anchor_ts, match_result):
        """删除被吸收的历史组，并把当前组作为新的历史版本落库。"""
        current_expire_ts = match_result.pop("_expire_ts_hint", None)
        if current_expire_ts is None:
            current_expire_ts = anchor_ts + self.get_rule_max_stay_time(match_result.get("rule"))
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

        stored_match = copy.deepcopy(match_result)
        if self.use_alarm_period_cache:
            group_item = {
                "anchor_ts": anchor_ts,
                "expire_ts": max(current_expire_ts, merged_expire_ts),
                "match": stored_match,
                "symptom_overlap_keys": get_match_symptom_overlap_keys(stored_match),
            }
        else:
            group_item = {
                "anchor_ts": anchor_ts,
                "expire_ts": max(current_expire_ts, merged_expire_ts),
                "match": stored_match,
                "alarm_keys": self._get_alarm_keys(stored_match.get("symptoms", [])),
            }
        self.groups.append(group_item)
        self._add_group_to_index(len(self.groups) - 1, group_item)
        self._maybe_compact_groups()

    def extend_related_expire_ts(self, merged_group_indexes, match_result, anchor_ts):
        """当当前结果不需要再次输出时，延长相关历史组的过期时间。"""
        if not merged_group_indexes:
            return

        current_expire_ts = match_result.get("_expire_ts_hint")
        if current_expire_ts is None:
            current_expire_ts = anchor_ts + self.get_rule_max_stay_time(match_result.get("rule"))

        for idx in merged_group_indexes:
            if 0 <= idx < len(self.groups) and self.groups[idx] is not None:
                self.groups[idx]["expire_ts"] = max(self.groups[idx]["expire_ts"], current_expire_ts)

    def _add_group_to_index(self, idx, group_item):
        if self.use_alarm_period_cache:
            for overlap_key in group_item.get("symptom_overlap_keys", set()):
                self.symptom_overlap_to_group_indexes[overlap_key].add(idx)
        else:
            for alarm_key in group_item.get("alarm_keys", set()):
                self.eid_to_group_indexes[alarm_key].add(idx)

    def _remove_group_from_index(self, idx, group_item):
        if self.use_alarm_period_cache:
            for overlap_key in group_item.get("symptom_overlap_keys", set()):
                indexes = self.symptom_overlap_to_group_indexes.get(overlap_key)
                if not indexes:
                    continue
                indexes.discard(idx)
                if not indexes:
                    self.symptom_overlap_to_group_indexes.pop(overlap_key, None)
        else:
            for alarm_key in group_item.get("alarm_keys", set()):
                indexes = self.eid_to_group_indexes.get(alarm_key)
                if not indexes:
                    continue
                indexes.discard(idx)
                if not indexes:
                    self.eid_to_group_indexes.pop(alarm_key, None)

    def _rebuild_alarm_index(self):
        self.eid_to_group_indexes = collections.defaultdict(set)
        self.symptom_overlap_to_group_indexes = collections.defaultdict(set)
        for idx, group_item in enumerate(self.groups):
            if group_item is None:
                continue
            if self.use_alarm_period_cache:
                overlap_keys = group_item.get("symptom_overlap_keys")
                if overlap_keys is None:
                    overlap_keys = get_match_symptom_overlap_keys(group_item.get("match", {}))
                    group_item["symptom_overlap_keys"] = overlap_keys
            else:
                alarm_keys = group_item.get("alarm_keys")
                if alarm_keys is None:
                    alarm_keys = self._get_alarm_keys(group_item.get("match", {}).get("symptoms", []))
                    group_item["alarm_keys"] = alarm_keys
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
    def _matches_any_overlap(left_symptoms, right_symptoms):
        right_grouped = collections.defaultdict(list)
        for symptom in right_symptoms:
            overlap_key = get_symptom_overlap_base_key(symptom)
            if overlap_key is not None:
                right_grouped[overlap_key].append(symptom)

        for left_symptom in left_symptoms:
            overlap_key = get_symptom_overlap_base_key(left_symptom)
            if overlap_key is None:
                continue
            for right_symptom in right_grouped.get(overlap_key, []):
                if symptoms_overlap(left_symptom, right_symptom):
                    return True
        return False

    @classmethod
    def _all_symptoms_covered(cls, source_symptoms, candidate_cover_symptoms):
        cover_grouped = collections.defaultdict(list)
        for symptom in candidate_cover_symptoms:
            overlap_key = get_symptom_overlap_base_key(symptom)
            if overlap_key is not None:
                cover_grouped[overlap_key].append(symptom)

        for source_symptom in source_symptoms:
            overlap_key = get_symptom_overlap_base_key(source_symptom)
            if overlap_key is None:
                return False

            if not any(
                symptom_covers(cover_symptom, source_symptom)
                for cover_symptom in cover_grouped.get(overlap_key, [])
            ):
                return False

        return True

    @staticmethod
    def _get_alarm_keys(symptoms):
        return set(get_match_alarm_keys({"symptoms": list(symptoms)}, use_alarm_period_cache=False))

    @staticmethod
    def _get_alarm_key(symptom):
        eid = symptom.get("eid")
        if eid in (None, ""):
            return None
        return eid
