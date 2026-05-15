import collections


class RoleSiteIndex:
    """Static structural role/site index.

    It precomputes two reusable maps:
    - (rule_name, role) -> frozenset(site_id)
    - id(node_config) -> frozenset(site_id)

    The index only uses static site profile / node structure predicates, so it
    does not affect trigger timing, alarm clear handling, or group lifecycle.
    """

    def __init__(self, rules, sites_domain_map, node_rule_helper):
        self._by_rule_role = {}
        self._by_config_id = {}
        self._by_site = collections.defaultdict(set)
        self._build(rules, sites_domain_map, node_rule_helper)

    def _build(self, rules, sites_domain_map, node_rule_helper):
        for rule_name, rule in rules.items():
            for role, node_config in rule.get("nodes", {}).items():
                config_id = id(node_config)
                candidates = self._by_config_id.get(config_id)
                if candidates is None:
                    candidates = frozenset(
                        site_id
                        for site_id, domain in sites_domain_map.items()
                        if node_rule_helper.matches_node_structure(domain, node_config)
                    )
                    self._by_config_id[config_id] = candidates
                self._by_rule_role[(rule_name, role)] = candidates
                for site_id in candidates:
                    self._by_site[site_id].add((rule_name, role))

    def matches_config(self, site_id, node_config):
        candidates = self._by_config_id.get(id(node_config))
        if candidates is None:
            return None
        return site_id in candidates

    def matches_role(self, rule_name, role, site_id):
        candidates = self._by_rule_role.get((rule_name, role))
        if candidates is None:
            return None
        return site_id in candidates

    def role_candidates(self, rule_name, role):
        return self._by_rule_role.get((rule_name, role), frozenset())

    def config_candidates(self, node_config):
        return self._by_config_id.get(id(node_config), frozenset())

    def site_rule_roles(self, site_id):
        return self._by_site.get(site_id, frozenset())
