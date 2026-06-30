class TemporalGraphEngineConstraintMixin:
    @staticmethod
    def _matched_alarm_roles(match_result, roles, alarms):
        return {
            symptom.get("matched_role")
            for symptom in match_result.get("symptoms", [])
            if symptom.get("matched_role") in roles
            and str(symptom.get("alarm", "")).strip() in alarms
        }

    def _validate_result_constraints(self, rule, match_result):
        result_constraints = rule.get("result_constraints") or {}

        for requirement in result_constraints.get("role_alarm_requirements_any", []):
            roles = set(requirement["roles"])
            alarms = set(requirement["alarms"])
            min_roles = int(requirement["min_roles"])
            if len(self._matched_alarm_roles(match_result, roles, alarms)) < min_roles:
                return False

        return True
