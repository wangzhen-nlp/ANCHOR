# No-index 方法说明

本文档说明 `fault_csm_codex` 中三种 no-index 类方法：

```text
incisomatch
sjtree
graphflow
```

这里的 “no-index” 指的是：它们不长期维护复杂的动态候选支持结构，不物化 partial match，也不维护完整 DCS。它们主要依靠当前告警事件作为锚点，临时做 backtracking。

## 和原 match_rules.py 的共同区别

原 `match_rules.py` 的主流程是 trigger 延迟收割：

```text
告警到达
  -> 写入 event_cache
  -> 如果命中 trigger_role，则进入 pending_triggers
  -> 等 aggregation_wait_sec 成熟
  -> 从 trigger_role 站点调用 _evaluate_rule()
```

`fault_csm_codex` 的 no-index 方法改成事件锚定增量匹配：

```text
任意相关 role 的告警到达
  -> 如果该站点/告警能匹配某个 rule role
  -> 立即从当前告警 role 出发
  -> 沿规则边做局部 backtracking
```

因此它们不再严格要求匹配必须由 `trigger_role` 告警触发。

需要注意：

```text
不要求 trigger_role 作为触发入口
不等于 trigger_role 在规则里完全无意义
```

`trigger_role` 仍然保留在规则配置、根因语义和输出语义中；只是它不再是唯一能启动匹配的入口。

## incisomatch

`incisomatch` 是最直接的事件锚定 backtracking。

流程可以理解为：

```text
当前告警站点 S
  -> S 能扮演某个 updated_role
  -> 固定 updated_role = S
  -> 从 updated_role 相邻规则边开始扩展
  -> 后续 role 基本按规则原始顺序扩展
```

它和原 `match_rules.py` 的区别主要是触发时机和起点：

```text
match_rules.py:
  trigger_role 告警进入 pending，成熟后从 trigger_role 评估

incisomatch:
  任意相关 role 告警立即触发，从当前告警 role 局部 DFS/backtracking
```

`incisomatch` 适合规则比较简单、链状结构较多、分支不大的情况。

## sjtree

当前实现中的 `sjtree` 不是严格意义上的经典 SJ-Tree。它没有长期物化 partial match，也没有维护 join tree。

更准确地说，它是：

```text
event-anchored backtracking + selective role ordering
```

它仍然从当前告警站点开始：

```text
当前告警站点 S
  -> S 能扮演某个 updated_role
  -> 固定 updated_role = S
  -> 从 updated_role 相邻规则边开始扩展
```

和 `incisomatch` 的区别在后续扩展顺序：

```text
incisomatch:
  更接近规则原始 role 顺序

sjtree:
  遇到多个可扩展 role 时，优先扩展更有选择性的规则边/role
```

例如规则图为：

```text
A(trigger) -- B(context)
A(trigger) -- C(data_offline)
A(trigger) -- D(data_link)
```

当前告警命中 `A`。

`incisomatch` 可能按原始顺序：

```text
A -> B -> C -> D
```

如果 `B` 是 context role，候选很多，就会先展开大量分支。

`sjtree` 会倾向于：

```text
A -> C -> D -> B
```

因为 `C`、`D` 有更强告警约束，候选更少，更容易提前剪枝。

所以 `sjtree` 的收益主要出现在：

```text
规则图有分叉
当前锚点周围有多个可扩展 role
这些 role 的候选数量差异明显
```

如果规则是简单链：

```text
A -- B -- C
```

每一步只有一个可扩展方向，那么 `sjtree` 和 `incisomatch` 的差异通常很小。

## graphflow

`graphflow` 也是事件锚定 backtracking，但它比 `sjtree` 更动态。

核心思想是：

```text
扩展某个新 role 时，
查看它连接了哪些已经匹配的 role，
从候选集最小的已匹配邻居生成候选，
再用其它已匹配邻居做 join 检查。
```

例如规则图是三角形：

```text
A -- B
A -- C
B -- C
```

当前已经匹配：

```text
A = S1
B = S2
```

准备扩展 `C`。

如果：

```text
从 A 可达的 C 候选有 100 个
从 B 可达的 C 候选有 3 个
```

`graphflow` 会优先从 `B` 生成 `C` 候选，只检查 3 个，再验证这些候选是否也满足和 `A` 的边约束。

因此：

```text
sjtree:
  更偏静态扩展顺序

graphflow:
  每一步根据当前 partial match 动态选择候选来源
```

`graphflow` 的优势主要出现在：

```text
规则图有多边约束
某个待扩展 role 同时连接多个已匹配 role
不同已匹配邻居产生的候选数量差异很大
```

对于链状规则，每一步通常只有一个已匹配邻居，`graphflow` 的动态选择空间很小，因此和普通 backtracking 差异也小。

## 三者的相对特点

```text
incisomatch:
  最简单，基本按规则顺序扩展。

sjtree:
  仍是普通 backtracking，但在多个可扩展分支中优先选择更有选择性的 role。

graphflow:
  在每一步根据当前 partial match 动态选择候选最少的已匹配邻居，并做 join pruning。
```

经验上：

```text
链状规则:
  incisomatch / sjtree / graphflow 差距通常较小

分叉规则:
  sjtree 可能优于 incisomatch

多边 join 规则:
  graphflow 通常更有优势
```

## 与完整论文算法的关系

当前实现没有完整复刻这些系统的所有数据结构：

```text
incisomatch:
  借用事件增量 backtracking 思路。

sjtree:
  借用选择性子结构优先思想，但没有物化 SJ-Tree partial match join。

graphflow:
  借用动态候选来源选择和 join pruning 思想，但没有完整实现论文系统的所有优化。
```

实现目标是：

```text
保持 rule_config.py 规则语义和输出格式不变，
只改变流式告警到达后的匹配触发方式与 backtracking 计划。
```
