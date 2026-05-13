# fault_csm_claude 增量匹配说明

`fault_csm_claude` 是一个 TurboFlux-inspired 的实验性增量故障匹配流程。它仍然复用 `fault_grouping` 的规则、拓扑、告警缓存、规则评估和输出格式，但把 `match_rules.py` 里的“pending 聚合后收割”改成“告警到达后立即局部增量评估”。

## 和 match_rules.py 的核心区别

`match_rules.py` 的主流程是延迟聚合：

```text
告警到达
  -> 写入 event_cache
  -> 如果命中 trigger_role，则进入 pending_triggers
  -> 等 aggregation_wait_sec 成熟
  -> 收割 pending trigger 并调用 _evaluate_rule()
```

`fault_csm_claude` 的主流程是即时增量：

```text
告警到达
  -> 写入 event_cache
  -> 立即找受影响的 rule/trigger
  -> 局部调用 _evaluate_rule()
```

因此，`fault_csm_claude` 的重点不是重写规则匹配语义，而是减少每条新告警到达后需要重新评估的规则和 trigger 数量。

## TurboFlux-inspired 数据结构

### RoleSiteIndex

`RoleSiteIndex` 是静态结构索引，启动时预先构建两类映射：

```text
(rule_name, role) -> {结构上能匹配该 role 的 site_id}
site_id -> {(rule_name, role)}
```

这两类索引表示同一份关系的两个查询方向。第一类用于“给定规则角色，找候选站点”；第二类用于“某站点来了告警后，找它可能影响哪些规则角色”。

它主要减少重复的 `matches_node_structure()` 判断。

### ActiveTriggerTracker

`ActiveTriggerTracker` 维护当前仍活跃的 trigger 告警：

```text
(site, rule_name) -> {eid: ts}
rule_name -> {site: latest_ts}
```

第一类索引用于精确增删。清除某个 `eid` 时，只删除该 eid；如果同一站点同一规则下还有其它 trigger eid，该站点仍然是活跃 trigger。

第二类索引用于快速枚举某条规则当前有哪些活跃 trigger 站点。这样 non-trigger 告警到达时，不需要扫历史告警，也不用等 pending 成熟。

### Non-trigger Role Index

非 trigger 索引用来回答：

```text
当前站点结构上可能作为哪些规则的 non-trigger role？
```

形式是：

```text
site_id -> [(rule_name, non_trigger_role)]
```

当站点 `B` 来了一条新告警时，可以直接查出 `B` 可能补齐哪些规则，而不是扫描所有规则。

### NonTriggerAlarmSpecsIndex

这个索引用来在拓扑检查前先过滤告警类型：

```text
(site_id, rule_name, role) -> expected_alarms
```

如果 `B` 的新告警类型不满足该 non-trigger role 的期望告警，就直接跳过，不做拓扑遍历，也不调用完整 `_evaluate_rule()`。

### RoleFilteredNeighborCache

这个缓存把“拓扑可达 + role 结构匹配”的结果保存下来：

```text
(source_site, direction, max_hops, rule_name, target_role)
  -> {可达且结构上能扮演 target_role 的站点}
```

拓扑和站点画像是静态的，因此这个缓存不需要随告警增删失效。

它用于快速判断：

```text
活跃 trigger A 和新告警站点 B 是否可能组成规则中的某条边？
```

这里的“可能相关”只表示结构和拓扑边约束可能成立，并不代表完整规则已经匹配成功。最终仍然要由 `_evaluate_rule()` 判断时间窗口、其它 role、result_constraints 等完整条件。

### DCS-style Support Cache

邻域支持缓存用于在进入完整 `_evaluate_rule()` 前做更便宜的剪枝：

```text
(alarm_generation, trigger_site, rule_name) -> bool
```

它检查某个 trigger 站点在当前活跃告警状态下，规则需要的必选 non-trigger role 是否至少有一个邻域告警支撑。

每次告警新增或清除都会推进 `alarm_generation`，旧缓存自然失效。

## 增量匹配策略

### Strategy 1: 直接触发

当新告警站点 `A` 满足某条规则的 `trigger_role` 告警谓词时，直接以 `A` 为 trigger 调用 `_evaluate_rule()`。

```text
A 新告警
  -> A 满足 rule.trigger_role
  -> _evaluate_rule(rule, A)
```

理论上 Strategy 1 也可以使用邻域支持检查进行剪枝，但需要非常保守。因为规则可能包含 optional/context-only/compound role，如果 support 判断不完整，可能误剪掉本该匹配的规则。

### Strategy 2: 间接触发

当新告警站点 `B` 可能作为某规则的 non-trigger role 时，流程是：

```text
B 新告警
  -> 查 B 可能扮演哪些 non-trigger role
  -> 检查 B 的告警类型是否满足该 role
  -> 查询该 rule 当前有哪些活跃 trigger A
  -> 检查 A 和 B 是否满足规则边的方向/跳数/role 约束
  -> 检查 A 的邻域是否有基本告警支撑
  -> _evaluate_rule(rule, A)
```

这个策略用于捕获：

```text
trigger A 已经活跃，但当时缺少邻居 B 的告警；
后来 B 告警到达，可能补齐整个故障模式。
```

## 关于非直接相关的 role

如果规则边本身允许多跳，例如：

```text
grandparent_role -> child_role, max_hops = 2
```

那么 `grandparent` 和 `child` 可以通过 RoleFilteredNeighborCache 找到，因为它按 `direction + max_hops` 做拓扑可达判断。

但如果规则图是链式的：

```text
grandparent_role -> parent_role -> child_role
```

且没有 `grandparent_role -> child_role` 的直接规则边，那么当前 Strategy 2 不保证能由 `child` 新告警直接找到 `grandparent` trigger。它主要检查 trigger role 与当前 non-trigger role 是否通过规则边直接相关。

## 清除告警处理

清除告警时，`fault_csm_claude` 会：

```text
从 event_cache 删除 eid
从 ActiveTriggerTracker 删除 eid
将历史故障组中包含该 eid 的组标记为 tombstone
推进 alarm_generation，使 support cache 失效
```

这更强调“失效已有匹配”。它不会主动因为 `NO_OFFLINE` 或 `forbidden_alarms` 变成立而重新产出新匹配。

## 总结

`fault_csm_claude` 的几个索引本质上都在服务同一个目标：

```text
当新告警到达时，快速找到值得重新评估的 trigger，而不是扫描所有规则或等待 pending 聚合。
```

其中：

```text
RoleSiteIndex: 站点和规则角色的结构候选索引
ActiveTriggerTracker: 当前活跃 trigger 索引
Non-trigger Role Index: 新告警站点可能补齐哪些非 trigger role
NonTriggerAlarmSpecsIndex: 非 trigger 告警类型预过滤
RoleFilteredNeighborCache: trigger 与 non-trigger 的拓扑相关性预过滤
Support Cache: 进入完整评估前的邻域告警支撑剪枝
```

完整规则语义仍由 `TemporalGraphEngine._evaluate_rule()` 负责。
