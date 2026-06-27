from alarm_tools.alarm_types import OFFLINE_ALARMS, POWER_ALARMS, LINK_ALARMS
from fault_grouping.time_config import (
  RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
  RULE_DEFAULT_MAX_STAY_TIME_SEC,
  RULE_POWER_EDGE_AFTER_SEC,
  RULE_POWER_EDGE_BEFORE_SEC,
  RULE_POWER_MAX_STAY_TIME_SEC,
)

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

NO_OFFLINE_DATA_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Data"],
      "expected_alarms": {
        "optional_alarms": LINK_ALARMS | POWER_ALARMS,
        "forbidden_alarms": OFFLINE_ALARMS
      }
    }
  ]
}

REQUIRED_OFFLINE_DATA_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Data"],
      "expected_alarms": {
        "required_alarms": OFFLINE_ALARMS,
        "optional_alarms": LINK_ALARMS | POWER_ALARMS,
      }
    }
  ]
}

OPTIONAL_OFFLINE_DATA_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Data"],
      "expected_alarms": {
        "optional_alarms": OFFLINE_ALARMS | LINK_ALARMS | POWER_ALARMS
      }
    }
  ]
}

REQUIRED_LINK_NO_OFFLINE_DATA_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Data"],
      "expected_alarms": {
        "required_alarms": LINK_ALARMS,
        "required_alarm_source_domains": ["Data"],
        "forbidden_alarms": OFFLINE_ALARMS
      }
    }
  ]
}

UNDERNEATH_SITE_RULES = [
  {
    "expected_alarms": {
      "required_alarms": OFFLINE_ALARMS
    }
  }
]

UNDERNEATH_COMPOUND_NODE = {
  "type": "compound",
  "hide_if_no_alarms": True,
  "min_count": 1,
  "patterns": [
    {"type": "primitive", "site_rules": UNDERNEATH_SITE_RULES},
  ]
}

TRANSMISSION_OFFLINE_NODE = {
  "type": "primitive",
  "site_rules": [
    {
      "include": ["Transmission"],
      "expected_alarms": OFFLINE_ALARMS
    }
  ]
}

TRANSMISSION_DOWNSTREAM_COMPOUND_NODE = {
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
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "downstream_compound_node",
  "exclusive_site_roles": ["grandparent_node", "downstream_compound_node"],
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
                "forbidden_alarms": OFFLINE_ALARMS,
                "forbidden_alarm_source_domains": ["Data"]
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
    "downstream_compound_node": TRANSMISSION_DOWNSTREAM_COMPOUND_NODE
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
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC
    },
    {
      "source": "parent_microwave_node",
      "target": "downstream_compound_node",
      "direction": "downstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
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
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "link_child_offline_node",
  "nodes": {
    "link_child_offline_node": TRANSMISSION_OFFLINE_NODE,
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
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC
    }
  ]
}

power_rule = {
  "pattern_name": "local_power_to_offline",
  "description": "同站点离线告警 -> 同站点电源根因",
  "max_stay_time_sec": RULE_POWER_MAX_STAY_TIME_SEC,
  "trigger_role": "offline_node",
  "nodes": {
    "offline_node": TRANSMISSION_OFFLINE_NODE,
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
        "before_sec": RULE_POWER_EDGE_BEFORE_SEC,
        "after_sec": RULE_POWER_EDGE_AFTER_SEC
      }
    }
  ]
}

data_rule = {
  "pattern_name": "cross_domain_storm_under_data",
  "description": "无断站 -> 断站",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "data_underneath_compound_node",
  "nodes": {
    "data_parent_data_node": {
      "type": "primitive",
      "site_rules": [
        {
          "include": ["Data"],
          "expected_alarms": {
            "forbidden_alarms": OFFLINE_ALARMS,
            "forbidden_alarm_source_domains": ["Data"]
          }
        }
      ]
    },
    "data_underneath_compound_node": TRANSMISSION_DOWNSTREAM_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "data_underneath_compound_node",
      "target": "data_parent_data_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1
    }
  ]
}

REQUIRED_LINK_NO_OFFLINE_DATA_NODE_NE_ANCHORED = {
  **REQUIRED_LINK_NO_OFFLINE_DATA_NODE,
  # 仅召回 alarm_source 在"与入边 source 角色站点的 NE 拓扑相邻"的 NE 集合内的告警。
  # anchor_role 用 "<edge_source>" 在编译期由 evaluator 解析到唯一入边的 source。
  "alarm_source_ne_anchor": {
    "anchor_role": "<edge_source>",
    "max_ne_hops": 1
  }
}

data_link_adjacent_no_offline_rule = {
  "pattern_name": "data_link_adjacent_no_offline_context",
  "description": "本路由Data link且无Data offline，邻接路由无Data offline -> 下挂断站",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "data_link_underneath_compound_node",
  "exclusive_site_roles": [
    "data_link_adjacent_data_neighbor_node",
    "data_link_parent_data_node",
    "data_link_underneath_compound_node"
  ],
  "nodes": {
    "data_link_adjacent_data_neighbor_node": REQUIRED_LINK_NO_OFFLINE_DATA_NODE_NE_ANCHORED,
    "data_link_parent_data_node": NO_OFFLINE_DATA_NODE,
    "data_link_underneath_compound_node": UNDERNEATH_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "data_link_underneath_compound_node",
      "target": "data_link_parent_data_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC
    },
    {
      "source": "data_link_parent_data_node",
      "target": "data_link_adjacent_data_neighbor_node",
      "direction": ["bidirection", "upstream", "downstream"],
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1
    }
  ],
  "result_constraints": {
    "role_alarm_requirements_any": [
      {
        "roles": ["data_link_underneath_compound_node"],
        "alarms": OFFLINE_ALARMS,
        "min_roles": 1
      }
    ]
  }
}

data_link_adjacent_offline_rule = {
  "pattern_name": "data_link_adjacent_offline_context",
  "description": "本路由Data link且无Data offline，邻接路由Data offline，下挂断站可有可无",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "data_link_offline_parent_data_node",
  "exclusive_site_roles": [
    "data_link_offline_adjacent_data_node",
    "data_link_offline_parent_data_node",
    "data_link_offline_underneath_compound_node"
  ],
  "nodes": {
    "data_link_offline_adjacent_data_node": REQUIRED_LINK_NO_OFFLINE_DATA_NODE_NE_ANCHORED,
    "data_link_offline_parent_data_node": REQUIRED_OFFLINE_DATA_NODE,
    "data_link_offline_underneath_compound_node": UNDERNEATH_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "data_link_offline_parent_data_node",
      "target": "data_link_offline_adjacent_data_node",
      "direction": ["bidirection", "upstream", "downstream"],
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1
    },
    {
      "source": "data_link_offline_underneath_compound_node",
      "target": "data_link_offline_parent_data_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "optional": True
    }
  ]
}

data_link_adjacent_link_rule = {
  "pattern_name": "data_link_adjacent_link_context",
  "description": "本路由Data link且无Data offline，邻接路由Data link且无Data offline",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "data_link_pair_current_data_node",
  "exclusive_site_roles": [
    "data_link_pair_current_data_node",
    "data_link_pair_adjacent_data_node"
  ],
  "nodes": {
    "data_link_pair_current_data_node": REQUIRED_LINK_NO_OFFLINE_DATA_NODE_NE_ANCHORED,
    "data_link_pair_adjacent_data_node": REQUIRED_LINK_NO_OFFLINE_DATA_NODE_NE_ANCHORED
  },
  "edges": [
    {
      "source": "data_link_pair_current_data_node",
      "target": "data_link_pair_adjacent_data_node",
      "direction": ["bidirection", "upstream", "downstream"],
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1,
      "constraints": {
        "dedupe_symmetric_pair": True
      }
    }
  ]
}

data_no_offline_adjacent_optional_offline_rule = {
  "pattern_name": "data_no_offline_adjacent_optional_offline_context",
  "description": "本路由存在下挂断站，双向相邻路由自身Data offline或其下游存在offline",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "current_underneath_compound_node",
  "exclusive_site_roles": [
    "current_parent_data_node",
    "current_underneath_compound_node",
    "adjacent_router_data_neighbor_node",
    "adjacent_router_underneath_compound_node"
  ],
  "nodes": {
    "current_parent_data_node": NO_OFFLINE_DATA_NODE,
    "current_underneath_compound_node": UNDERNEATH_COMPOUND_NODE,
    "adjacent_router_data_neighbor_node": OPTIONAL_OFFLINE_DATA_NODE,
    "adjacent_router_underneath_compound_node": UNDERNEATH_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "current_underneath_compound_node",
      "target": "current_parent_data_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC
    },
    {
      "source": "current_parent_data_node",
      "target": "adjacent_router_data_neighbor_node",
      "direction": ["bidirection", "upstream", "downstream"],
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1
    },
    {
      "source": "adjacent_router_underneath_compound_node",
      "target": "adjacent_router_data_neighbor_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "optional": True
    }
  ],
  "result_constraints": {
    "role_alarm_requirements_any": [
      {
        "roles": ["current_underneath_compound_node"],
        "alarms": OFFLINE_ALARMS,
        "min_roles": 1
      }
    ],
    "role_alarm_or_presence_any": [
      {
        "alarm_roles": ["adjacent_router_data_neighbor_node", "adjacent_router_underneath_compound_node"],
        "alarms": OFFLINE_ALARMS,
        "min_matches": 1
      }
    ]
  }
}

data_offline_adjacent_offline_rule = {
  "pattern_name": "data_offline_adjacent_offline_context",
  "description": "本路由Data offline，双向相邻路由Data offline，本路由/相邻路由下挂offline可有可无",
  "max_stay_time_sec": RULE_DEFAULT_MAX_STAY_TIME_SEC,
  "trigger_role": "offline_current_parent_data_node",
  "exclusive_site_roles": [
    "offline_current_parent_data_node",
    "offline_current_underneath_compound_node",
    "offline_adjacent_router_data_neighbor_node",
    "offline_adjacent_router_underneath_compound_node"
  ],
  "nodes": {
    "offline_current_parent_data_node": REQUIRED_OFFLINE_DATA_NODE,
    "offline_current_underneath_compound_node": UNDERNEATH_COMPOUND_NODE,
    "offline_adjacent_router_data_neighbor_node": REQUIRED_OFFLINE_DATA_NODE,
    "offline_adjacent_router_underneath_compound_node": UNDERNEATH_COMPOUND_NODE
  },
  "edges": [
    {
      "source": "offline_current_parent_data_node",
      "target": "offline_adjacent_router_data_neighbor_node",
      "direction": ["bidirection", "upstream", "downstream"],
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "max_hops": 1,
      "constraints": {
        "dedupe_symmetric_pair": True
      }
    },
    {
      "source": "offline_adjacent_router_underneath_compound_node",
      "target": "offline_adjacent_router_data_neighbor_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "optional": True
    },
    {
      "source": "offline_current_underneath_compound_node",
      "target": "offline_current_parent_data_node",
      "direction": "upstream",
      "time_window_sec": RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
      "optional": True
    }
  ]
}
