class TemporalGraphEngineConstraintMixin:
    def _validate_result_constraints(self, rule, match_result):
        """对已成型的候选故障组做规则级后置约束校验。"""
        result_constraints = rule.get("result_constraints") or {}
        if not result_constraints:
            return True, None

        role_alarm_requirements_any = result_constraints.get("role_alarm_requirements_any", [])
        for requirement in role_alarm_requirements_any:
            roles = {
                str(role).strip()
                for role in requirement.get("roles", [])
                if str(role).strip()
            }
            alarms = {
                str(alarm).strip()
                for alarm in requirement.get("alarms", [])
                if str(alarm).strip()
            }
            min_roles = max(1, int(requirement.get("min_roles", 1) or 1))
            if not roles or not alarms:
                continue

            matched_roles = {
                symptom.get("matched_role")
                for symptom in match_result.get("symptoms", [])
                if symptom.get("matched_role") in roles
                and str(symptom.get("alarm", "")).strip() in alarms
            }
            if len(matched_roles) < min_roles:
                return (
                    False,
                    (
                        f"后置约束失败：角色 {sorted(roles)} 中至少 {min_roles} 个需要命中告警 "
                        f"{sorted(alarms)}，实际命中角色={sorted(role for role in matched_roles if role)}"
                    ),
                )

        role_alarm_or_presence_any = result_constraints.get("role_alarm_or_presence_any", [])
        for requirement in role_alarm_or_presence_any:
            alarm_roles = {
                str(role).strip()
                for role in requirement.get("alarm_roles", [])
                if str(role).strip()
            }
            alarms = {
                str(alarm).strip()
                for alarm in requirement.get("alarms", [])
                if str(alarm).strip()
            }
            presence_roles = {
                str(role).strip()
                for role in requirement.get("presence_roles", [])
                if str(role).strip()
            }
            min_matches = max(1, int(requirement.get("min_matches", 1) or 1))

            matched_alarm_roles = {
                symptom.get("matched_role")
                for symptom in match_result.get("symptoms", [])
                if symptom.get("matched_role") in alarm_roles
                and str(symptom.get("alarm", "")).strip() in alarms
            }
            role_mapping = match_result.get("role_mapping", {})
            matched_presence_roles = {
                role
                for role in presence_roles
                if role_mapping.get(role)
            }
            matched_items = matched_alarm_roles | matched_presence_roles
            if len(matched_items) < min_matches:
                return (
                    False,
                    (
                        "后置约束失败：需要满足至少 "
                        f"{min_matches} 个条件，告警角色={sorted(alarm_roles)} 命中告警={sorted(alarms)} "
                        f"或存在角色={sorted(presence_roles)}；"
                        f"实际告警命中={sorted(role for role in matched_alarm_roles if role)}，"
                        f"实际存在角色={sorted(role for role in matched_presence_roles if role)}"
                    ),
                )

        return True, None
