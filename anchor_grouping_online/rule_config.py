from anchor_grouping_online.alarm_types import OFFLINE_ALARMS, POWER_ALARMS, LINK_ALARMS
from anchor_grouping_online.time_config import (
  RULE_DEFAULT_EDGE_TIME_WINDOW_SEC,
  RULE_DEFAULT_MAX_STAY_TIME_SEC,
)

# 规则字典上的可选布尔字段：标记为 True 的规则才算“可落盘规则”。
# match_rules.py 输出故障组前会做一次过滤——只有 merged_rules 命中任意一个
# 带 output_eligible=True 的规则的故障组，才会写入输出文件。
# 若所有规则都没标记该字段，则不做过滤（全部落盘）。
# 引擎只读取 nodes/edges/trigger_role 等已知字段，多出来的该字段会被安全忽略。
OUTPUT_ELIGIBLE_RULE_FIELD = "output_eligible"

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
  # 终极/可落盘规则：包含此规则的故障组才会被 match_rules.py 写入输出文件。
  "output_eligible": True,
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
      "direction": ["bidirectional", "upstream", "downstream"],
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
      "direction": ["bidirectional", "upstream", "downstream"],
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

data_no_offline_adjacent_optional_offline_rule = {
  "pattern_name": "data_no_offline_adjacent_optional_offline_context",
  "description": "本路由存在下挂断站，双向相邻路由自身Data offline或其下游存在offline",
  # 终极/可落盘规则：包含此规则的故障组才会被 match_rules.py 写入输出文件。
  "output_eligible": True,
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
      "direction": ["bidirectional", "upstream", "downstream"],
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
      },
      {
        "roles": ["adjacent_router_data_neighbor_node", "adjacent_router_underneath_compound_node"],
        "alarms": OFFLINE_ALARMS,
        "min_roles": 1
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
      "direction": ["bidirectional", "upstream", "downstream"],
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
