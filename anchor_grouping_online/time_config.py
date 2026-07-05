"""Central timing configuration for fault grouping.

This module contains only static/default time values. Timestamps derived from
input data or wall-clock runtime stay in the code paths that compute them.
"""

# Temporal engine defaults.
DEFAULT_EVENT_TTL_SEC = 3600
DEFAULT_POWER_ALARM_TTL_SEC = 10800

# Rule timing defaults used by anchor_grouping_online/rule_config.py.
RULE_DEFAULT_MAX_STAY_TIME_SEC = 600
RULE_DEFAULT_EDGE_TIME_WINDOW_SEC = 420
