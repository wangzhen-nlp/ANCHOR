# Index/cache-assisted 方法说明

本文档说明 `fault_csm_codex` 中三种 index/cache-assisted 方法：

```text
iedyn
symbi
turboflux
```

这里的 “index/cache-assisted” 指的是：它们仍然保持 `rule_config.py` 的规则语义和输出格式不变，但在事件锚定 backtracking 的基础上，增加了动态 support check、support cache、候选排序等剪枝逻辑。

需要注意：当前实现不是完整复刻论文系统里的 IEDyn、SymBi 或 TurboFlux。它们在当前代码中更接近：

```text
event-anchored incremental backtracking
  + query order 差异
  + dynamic support pruning
  + support cache
```

## 和原 match_rules.py 的共同区别

原 `match_rules.py` 的主流程是 trigger 延迟聚合：

```text
trigger_role 告警
  -> 进入 pending_triggers
  -> 等 aggregation_wait_sec
  -> 从 trigger_role 调 _evaluate_rule()
```

`fault_csm_codex` 的 index/cache-assisted 方法改成即时增量：

```text
任意相关 role 告警到达
  -> 立即作为一次图更新边
  -> 从当前告警 role 出发
  -> 做局部 backtracking
```

因此它们不再严格要求必须由 `trigger_role` 告警触发。

匹配入口也不同：

```text
match_rules.py:
  成熟 trigger: (trigger_site, rule_name, trigger_ts)

fault_csm_codex:
  新增 HAS_ALARM 边: site --alarm--> eid
```

## 共同的剪枝机制

这三种方法都使用同一类轻量 DCS-style support check。

当某个候选站点准备扮演某个 role 时，会先检查：

```text
该 role 在规则图里有哪些相邻 role？
当前候选站点按规则边方向/跳数能到达的邻域中，
每个必选相邻 role 是否至少有一个可行候选？
```

如果某个必选邻接 role 完全没有 support，则该候选不可能形成完整匹配，可以提前剪掉。

示例：

```text
规则图:
A -- B
A -- C

当前判断:
S1 是否可以作为 A
```

support check 会看：

```text
S1 的 B 邻域里有没有至少一个满足 B role 的站点？
S1 的 C 邻域里有没有至少一个满足 C role 的站点？
```

如果 `C` 没有任何支撑，则 `S1 as A` 直接被剪掉，不进入更深 backtracking。

这个检查是必要条件，不是充分条件。完整匹配仍然由后续 backtracking 和结果约束判断。

## support check 是怎么实现的

当前实现是：

```text
拓扑 + role 结构候选:
  使用静态索引/cache

当前是否有活跃告警支撑:
  动态查 active alarm/event cache

结果:
  按当前动态状态缓存
```

也就是说，它不是完整维护：

```text
(role, site) -> 当前支持计数
```

而是在需要判断某个候选时动态计算：

```text
(rule, role, site_id, trigger_ts) -> True/False
```

然后写入 `support_cache`。

告警新增、清除、过期后会使动态 support cache 失效，避免使用旧状态。

## iedyn

当前 `iedyn` 可以理解为：

```text
DAG/选择性顺序
  + dynamic support check
```

它会先从规则图里选择一个相对有选择性的 query root，然后把规则图序列化成一个 DAG/BFS 顺序。

随后按如下因素排序 role：

```text
role_selectivity_rank
-degree
role_name
```

含义是：

```text
告警约束越强，越优先
规则图连接度越高，越优先
role 名字作为稳定 tie-break
```

不过它不会无视当前 partial match 的连通性。真正 backtracking 时仍然要求：

```text
优先扩展当前已经和 visited roles 相邻的 role
```

例如规则图：

```text
A -- B
A -- C
C -- D
```

如果当前告警锚定 `A`，即使全局选择性排序中 `D` 更靠前，也不能直接扩展 `D`。它会先在 `A` 可连接的 `B/C` 中选择更有选择性的 role，例如：

```text
A -> C -> D -> B
```

而不是：

```text
A -> B -> C -> D
```

因此 `iedyn` 的收益主要来自：

```text
更好的 role 扩展顺序
提前剪掉缺少邻接支撑的候选
```

适合：

```text
规则有多个必选邻接 role
告警比较稀疏
很多候选局部看似可行，但缺少其它必要支撑
```

## symbi

当前 `symbi` 可以理解为：

```text
query DAG 序列化顺序
  + dynamic support pruning
```

它更强调：

```text
选一个 query DAG root
按 DAG/BFS 序列化 role
每个更新边从当前 role 开始
沿预生成的 DAG/order 做 backtracking
```

和 `iedyn/turboflux` 一样，它也会使用 support check 和 support cache。

它和完整 SymBi 的差距在于：

```text
没有完整维护论文里的双向动态候选索引
没有完整实现双向 DAG 上的动态插入/删除传播
```

当前更像：

```text
固定 query DAG order
+ event-anchored backtracking
+ support pruning
```

如果规则是简单链，`symbi` 和其它 index/cache-assisted 方法差异可能很小。

如果规则图有多个方向的连接，它的 DAG 顺序可能会比普通规则顺序更稳定。

## turboflux

当前 `turboflux` 可以理解为：

```text
选择性 DAG/order
  + support check
  + support_count 候选排序
```

和原 `match_rules.py` 的最大区别仍然是：

```text
match_rules.py:
  trigger 延迟聚合后完整评估

turboflux:
  任意相关告警边即时触发
  并用 TurboFlux 风格的选择性顺序和 support 剪枝减少无效 backtracking
```

`turboflux` 会更强调：

```text
选择性 query root / DAG order
候选扩展时优先 support 更少的候选
```

也就是：

```text
更少 support 的候选更可能快速失败
优先检查它们可以更早剪枝
```

但当前实现不是完整 TurboFlux：

```text
没有持续维护完整 DCS
没有完整 d1/d2 状态传播
没有物化所有 candidate support counter
没有完整负匹配传播机制
```

当前实现是：

```text
动态寻找 support
缓存 support 结果
告警新增/清除/过期后失效动态 support cache
```

## iedyn、symbi、turboflux 的区别

三者共同点：

```text
任意相关告警 role 即时触发
使用静态 role/site 候选缓存
使用 dynamic support check
使用 support_cache
使用 support pruning
```

主要差异：

```text
iedyn:
  更偏 DAG/选择性 role 顺序 + support check

symbi:
  更偏 query DAG 序列化顺序 + support pruning

turboflux:
  更偏选择性 DAG/order + support_count 候选排序
```

更直白地说：

```text
iedyn:
  先扩展更有选择性、更中心的 role

symbi:
  按 query DAG 生成稳定匹配顺序

turboflux:
  选择性顺序之外，更强调 support 数量和更早失败的候选
```

由于当前没有完整维护论文级动态 index，这三者在简单规则上差异可能不大。规则越复杂、分叉越多、候选越多，差异才越明显。

## 与完整论文算法的关系

当前实现借用的是算法思想，而不是完整系统复刻：

```text
IEDyn:
  借用 DAG/候选支持剪枝思想。

SymBi:
  借用 query DAG 序列化和双向约束思想。

TurboFlux:
  借用 DCS/support/path-count 风格的选择性顺序和候选排序思想。
```

实现目标仍然是：

```text
保持 rule_config.py 规则语义不变
保持输出格式不变
只改变流式告警到达后的匹配触发、顺序和剪枝策略
```
