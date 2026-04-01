import copy


class EmittedGroupStore:
    """管理历史故障组的保留、按 eid 合并和重新落库。"""

    def __init__(self, rules, default_stay_time):
        self.rules = rules
        self.default_stay_time = default_stay_time
        self.groups = []

    def prune_expired(self, current_time):
        """移除已经超过停留时间的历史故障组。"""
        self.groups = [
            item for item in self.groups
            if current_time <= item["expire_ts"]
        ]

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
        """按 eid 与历史故障组合并，生成更完整的当前故障组。"""
        related_groups = []
        current_alarm_keys = self._get_alarm_keys(match_result.get("symptoms", []))

        if not current_alarm_keys:
            return match_result, set(), set(), True

        for idx, item in enumerate(self.groups):
            previous_match = item["match"]
            previous_alarm_keys = self._get_alarm_keys(previous_match.get("symptoms", []))
            if current_alarm_keys & previous_alarm_keys:
                related_groups.append((idx, item))

        if not related_groups:
            return match_result, set(), set(), True

        merged = {
            "uuid": match_result.get("uuid"),
            "rule": match_result.get("rule"),
            "merged_rules": list(match_result.get("merged_rules", [match_result.get("rule")])),
            "inferred_roots": copy.deepcopy(match_result.get("inferred_roots", {})),
            "role_mapping": copy.deepcopy(match_result.get("role_mapping", {})),
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
            previous_alarm_keys = self._get_alarm_keys(previous_match.get("symptoms", []))
            if current_alarm_keys.issubset(previous_alarm_keys):
                fully_containing_history_exists = True
            previous_uuid = previous_match.get("uuid")
            if previous_uuid:
                related_group_uuids.add(previous_uuid)
            related_group_uuids.update(previous_match.get("related_group_uuids", []))
            previous_merged_rules = previous_match.get("merged_rules", [previous_match.get("rule")])
            merged["merged_rules"] = sorted(set(merged["merged_rules"]) | {rule for rule in previous_merged_rules if rule})
            for role, nodes in previous_match.get("inferred_roots", {}).items():
                merged["inferred_roots"].setdefault(role, [])
                merged["inferred_roots"][role] = sorted(set(merged["inferred_roots"][role]) | set(nodes))

            for role, nodes in previous_match.get("role_mapping", {}).items():
                merged["role_mapping"].setdefault(role, [])
                merged["role_mapping"][role] = sorted(set(merged["role_mapping"][role]) | set(nodes))

            symptom_map = {}
            for symptom in merged["symptoms"]:
                alarm_key = self._get_alarm_key(symptom)
                if alarm_key is not None:
                    symptom_map[alarm_key] = symptom
            for symptom in previous_match.get("symptoms", []):
                alarm_key = self._get_alarm_key(symptom)
                if alarm_key is not None:
                    symptom_map[alarm_key] = symptom
            merged["symptoms"] = list(symptom_map.values())

        if fully_containing_history_exists:
            return merged, merged_group_indexes, related_group_uuids, False

        return merged, merged_group_indexes, related_group_uuids, True

    def replace_and_store(self, merged_group_indexes, anchor_ts, match_result):
        """删除被吸收的历史组，并把当前组作为新的历史版本落库。"""
        current_expire_ts = match_result.pop("_expire_ts_hint", None)
        if current_expire_ts is None:
            current_expire_ts = anchor_ts + self.get_rule_max_stay_time(match_result.get("rule"))
        merged_expire_ts = max(
            (self.groups[idx]["expire_ts"] for idx in merged_group_indexes),
            default=current_expire_ts
        )

        if merged_group_indexes:
            self.groups = [
                item for idx, item in enumerate(self.groups)
                if idx not in merged_group_indexes
            ]

        self.groups.append({
            "anchor_ts": anchor_ts,
            "expire_ts": max(current_expire_ts, merged_expire_ts),
            "match": copy.deepcopy(match_result)
        })

    def extend_related_expire_ts(self, merged_group_indexes, match_result, anchor_ts):
        """当当前结果不需要再次输出时，延长相关历史组的过期时间。"""
        if not merged_group_indexes:
            return

        current_expire_ts = match_result.get("_expire_ts_hint")
        if current_expire_ts is None:
            current_expire_ts = anchor_ts + self.get_rule_max_stay_time(match_result.get("rule"))

        for idx in merged_group_indexes:
            if 0 <= idx < len(self.groups):
                self.groups[idx]["expire_ts"] = max(self.groups[idx]["expire_ts"], current_expire_ts)

    def _get_alarm_keys(self, symptoms):
        """提取一组症状中的有效 eid 集合。"""
        alarm_keys = set()
        for symptom in symptoms:
            alarm_key = self._get_alarm_key(symptom)
            if alarm_key is not None:
                alarm_keys.add(alarm_key)
        return alarm_keys

    def _get_alarm_key(self, symptom):
        """从单条症状中提取唯一告警键；当前只认 eid。"""
        eid = symptom.get("eid")
        if eid in (None, ""):
            return None
        return eid
