"""Central timing configuration for fault grouping.

Only static/default time values live here. Timestamps derived from input data
or wall-clock runtime stay in the code paths that compute them.
"""

# CLI defaults used by fault_grouping/match_rules.py.
DEFAULT_HARVEST_INTERVAL_SEC = 300
DEFAULT_AGGREGATION_WAIT_SEC = 420
DEFAULT_CLEAR_DELAY_SEC = 420

# Temporal engine defaults.
DEFAULT_EVENT_TTL_SEC = 3600
DEFAULT_POWER_ALARM_TTL_SEC = 10800
DEFAULT_PERIODIC_HARVEST_INTERVAL_SEC = 10

# Rule timing defaults used by fault_grouping/rule_config.py.
RULE_DEFAULT_MAX_STAY_TIME_SEC = 3600
RULE_POWER_MAX_STAY_TIME_SEC = 10800
RULE_DEFAULT_EDGE_TIME_WINDOW_SEC = 900
RULE_POWER_EDGE_BEFORE_SEC = 900
RULE_POWER_EDGE_AFTER_SEC = 10800
