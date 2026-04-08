from alarm_types import OFFLINE_ALARMS, POWER_ALARMS, LINK_ALARMS

TRANSMISSION_SITE_RULES = [
  {
    "include": ["Transmission"],
    "exclude": ["Ran"],
    "expected_alarms": "ANY"
  },
  {
    "include": ["Transmission", "Ran"],
    "expected_alarms": OFFLINE_ALARMS
  },
  {
    "include": ["Data", "Ran"],
    "expected_alarms": OFFLINE_ALARMS
  }
]

transmission_rule = {
  "pattern_name": "bounded_silent_cross_domain_storm",
  "description": "无告警 -> 断站? -> 断站?",
  "max_stay_time_sec": 3600,
  "trigger_role": "downstream_compound_node",
  "nodes": {
    "grandparent_node": {
      "type": "compound",
      "match": "ALL",
      "min_count": 1,
      "patterns": [
        {
          "type": "primitive",
          "site_rules": [
            {
              "include": ["Data"],
              "expected_alarms": {
                "forbidden_alarms": OFFLINE_ALARMS
              }
            },
            {
              "include": ["Transmission"],
              "expected_alarms": {
                "forbidden_alarms": OFFLINE_ALARMS
              }
            }
          ]
        }
      ]
    },
    "parent_microwave_node": {
      "type": "primitive",
      "site_rules": TRANSMISSION_SITE_RULES
    },
    "downstream_compound_node": {
      "type": "compound",
      "min_count": 1,
      "patterns": [
        {
          "type": "primitive",
          "site_rules": TRANSMISSION_SITE_RULES
        }
      ]
    }
  },
  "edges": [
    {
      "source": "parent_microwave_node",
      "target": "grandparent_node",
      "direction": "upstream",
      "constraints": {
        "target_candidate_selector": {
          "mode": "nearest_matching"
        }
      },
      "time_window_sec": 600
    },
    {
      "source": "parent_microwave_node",
      "target": "downstream_compound_node",
      "direction": "downstream",
      "time_window_sec": 600,
      "constraints": {
        "path_node_requirements": {
          "site_rules": TRANSMISSION_SITE_RULES
        }
      }
    }
  ]
}

link_rule = {
  "pattern_name": "upstream_link_to_offline",
  "description": "父节点传输告警 -> 儿子节点断站",
  "max_stay_time_sec": 3600,
  "trigger_role": "link_child_offline_node",
  "nodes": {
    "link_child_offline_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Transmission"],
          "expected_alarms": OFFLINE_ALARMS
        }
      ]
    },
    "link_parent_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Transmission"],
          "expected_alarms": LINK_ALARMS
        }
      ]
    }
  },
  "edges": [
    {
      "source": "link_parent_node",
      "target": "link_child_offline_node",
      "direction": "downstream",
      "max_hops": 1,
      "time_window_sec": 600
    }
  ]
}

power_rule = {
  "pattern_name": "local_power_to_offline",
  "description": "同站点离线告警 -> 同站点电源根因",
  "max_stay_time_sec": 10800,
  "trigger_role": "offline_node",
  "nodes": {
    "offline_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Transmission"],
          "expected_alarms": OFFLINE_ALARMS
        }
      ]
    },
    "power_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Transmission"],
          "expected_alarms": POWER_ALARMS
        }
      ]
    }
  },
  "edges": [
    {
      "source": "power_node",
      "target": "offline_node",
      "direction": "self",
      "time_window_sec": {
        "before_sec": 600,
        "after_sec": 10800
      }
    }
  ]
}
