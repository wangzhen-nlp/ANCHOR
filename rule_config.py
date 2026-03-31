from alarm_types import OFFLINE_ALARMS

TRANSMISSION_SITE_RULES = [
  {
    "include": ["Transmission"],
    "exclude": ["Ran"],
    "expected_alarms": "ANY"
  },
  {
    "include": ["Transmission", "Ran"],
    "expected_alarms": OFFLINE_ALARMS
  }
]

transmission_rule = {
  "pattern_name": "bounded_silent_cross_domain_storm",
  "description": "无告警 -> 断站? -> 断站",
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
              "expected_alarms": "NONE"
            },
            {
              "include": ["Transmission"],
              "expected_alarms": "NONE"
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
      "min_count": 3,
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
      "time_window_sec": 300,
      "constraints": {
        "path_node_requirements": {
          "site_rules": TRANSMISSION_SITE_RULES
        }
      }
    }
  ]
}
