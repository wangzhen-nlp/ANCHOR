from alarm_tools.alarm_types import OFFLINE_ALARMS, POWER_ALARMS, LINK_ALARMS

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

OPTIONAL_LINK_NO_OFFLINE_DATA_NODE = {
  "type": "compound",
  "patterns": [
    {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Data"],
          "expected_alarms": {
            "required_alarms": LINK_ALARMS,
            "forbidden_alarms": OFFLINE_ALARMS
          }
        }
      ]
    },
    {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Data"],
          "expected_alarms": {
            "forbidden_alarms": OFFLINE_ALARMS
          }
        }
      ]
    }
  ]
}

REQUIRED_OFFLINE_DATA_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Data"],
      "expected_alarms": {
        "required_alarms": OFFLINE_ALARMS
      }
    }
  ]
}

UNDERNEATH_TRANSMISSION_COMPOUND_NODE = {
  "type": "compound",
  "min_count": 1,
  "patterns": [
    {
      "type": "primitive",
      "site_rules": TRANSMISSION_SITE_RULES
    }
  ]
}

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
      "time_window_sec": 900
    },
    {
      "source": "parent_microwave_node",
      "target": "downstream_compound_node",
      "direction": "downstream",
      "time_window_sec": 900,
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
      "time_window_sec": 900
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
        "before_sec": 900,
        "after_sec": 10800
      }
    }
  ]
}

data_rule = {
  "pattern_name": "cross_domain_storm_under_data",
  "description": "无断站 -> 断站",
  "max_stay_time_sec": 3600,
  "trigger_role": "underneath_compound_node",
  "nodes": {
    "parent_data_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Data"],
          "expected_alarms": {
            "forbidden_alarms": OFFLINE_ALARMS
          }
        }
      ]
    },
    "underneath_compound_node": {
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
      "source": "underneath_compound_node",
      "target": "parent_data_node",
      "direction": "upstream",
      "time_window_sec": 900,
      "max_hops": 1
    }
  ]
}

data_link_neighbor_rule = {
  "pattern_name": "offline_under_data_with_neighbor_link_context",
  "description": "本路由/上下游相邻路由至少一侧有link(均无断站) -> 下挂断站",
  "max_stay_time_sec": 3600,
  "trigger_role": "underneath_compound_node",
  "nodes": {
    "parent_data_node": OPTIONAL_LINK_NO_OFFLINE_DATA_NODE,
    "adjacent_data_neighbor_node": OPTIONAL_LINK_NO_OFFLINE_DATA_NODE,
    "underneath_compound_node": {
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
      "source": "underneath_compound_node",
      "target": "parent_data_node",
      "direction": "upstream",
      "time_window_sec": 900,
      "max_hops": 1
    },
    {
      "source": "parent_data_node",
      "target": "adjacent_data_neighbor_node",
      "direction": "either",
      "time_window_sec": 900,
      "max_hops": 1
    }
  ],
  "result_constraints": {
    "role_alarm_requirements_any": [
      {
        "roles": ["parent_data_node", "adjacent_data_neighbor_node"],
        "alarms": LINK_ALARMS,
        "min_roles": 1
      }
    ]
  }
}

data_adjacent_router_rule = {
  "pattern_name": "offline_under_adjacent_data_router_context",
  "description": "本路由和相邻路由均存在下挂断站，且相邻路由站点需有link告警（路由站点本身均无断站）",
  "max_stay_time_sec": 3600,
  "trigger_role": "current_underneath_compound_node",
  "nodes": {
    "current_parent_data_node": OPTIONAL_LINK_NO_OFFLINE_DATA_NODE,
    "current_underneath_compound_node": UNDERNEATH_TRANSMISSION_COMPOUND_NODE,
    "adjacent_data_neighbor_node": REQUIRED_OFFLINE_DATA_NODE,
    "adjacent_underneath_compound_node": UNDERNEATH_TRANSMISSION_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "current_underneath_compound_node",
      "target": "current_parent_data_node",
      "direction": "upstream",
      "time_window_sec": 900,
      "max_hops": 1
    },
    {
      "source": "current_parent_data_node",
      "target": "adjacent_data_neighbor_node",
      "direction": "either",
      "time_window_sec": 900,
      "max_hops": 1
    },
    {
      "source": "adjacent_underneath_compound_node",
      "target": "adjacent_data_neighbor_node",
      "direction": "upstream",
      "time_window_sec": 900,
      "max_hops": 1
    }
  ]
}
