# Note

## 1. 当前预定义/配置时间窗的含义

**【逻辑】**

当前代码里涉及多类时间参数，它们作用在不同阶段：

- `aggregation_wait_sec`：trigger 进入 pending 后，需要等待多久才算成熟并可正式收割。
- `time_window_sec`：规则边上的关联时间窗，表示从一个角色传播/关联到另一个角色时，允许匹配的告警时间范围。
  既可以是数字，表示对称窗口；也可以是字典，表示非对称窗口（如前后不同容忍时间）。
- `global_ttl`：默认告警在 `event_cache` 中保留多久，超时后会被清理。
- `power_alarm_ttl`：电源类告警在 `event_cache` 中单独保留多久，用于支持更长时间窗的根因回看。
- `max_stay_time_sec`：故障组在历史组缓存中最多保留多久，用于后续历史组合并与延期。
- `harvest_interval_sec`：定时收割周期，表示多久推进一次 watermark 并检查成熟 pending。
- `speedup`：仅用于 `match_rules.py` 的模拟实时流回放，表示把告警时间差按多少倍压缩到真实运行时间；
  同时也会影响 live 模式下定时收割线程的真实运行周期。

**【意义】**

这些时间参数分别控制“触发等待”“规则关联”“事件缓存保留”“历史故障组保留”和“定时收割频率”；
它们名称相近但语义不同，区分清楚后，才能避免把规则窗口、缓存时长和故障组留存时间混用。

## 2. 当前 live / offline 模式的区别

**【逻辑】**

`offline` 模式按告警时间顺序逐条处理，每条告警后都会立即触发一次同步检查；
`live` 模式则按告警时间差模拟实时流输入，同时启动后台定时线程定期推进收割检查。
当前这版里，两种模式都会把告警加入 `event_cache` 并注册 trigger，
但在时间推进上有所不同：

- `offline` 更接近“每条告警到来就立刻看当前能不能出组”；
- `live` 更接近“告警持续流入，后台周期性检查是否有组成熟”。

在 TTL 和 trigger 成熟时间基准上，两者当前又有一层共同点：

- `event_cache` 的 TTL 清理看的是“已到达事件时间” `latest_arrived_event_ts`，不是后台模拟水印；
- 历史故障组的过期清理也优先看 `latest_arrived_event_ts`；
- pending trigger 是否成熟，当前也优先看 `latest_arrived_event_ts`，而不是单纯看 `current_watermark`。

这意味着：

- `offline` 下，TTL 和成熟判断基本都跟随当前处理到的事件时间推进；
- `live` 下，后台线程虽然仍会推进 `current_watermark`，但 TTL 和成熟判断不会因为模拟时钟跑得过快而提前触发，仍以真实已进入引擎的事件时间为准。

**【意义】**

两种模式适合不同场景：

- `offline` 适合离线验证、排查和对照，因为输出更接近逐条事件驱动；
- `live` 适合模拟真实流式接入、观察定时收割行为以及 speedup 对整体时间轴的影响。

另外，这种统一到“已到达事件时间”的做法可以避免 live 模式里因模拟时间推进过快而把缓存或故障组提前清掉；
但相应地，也削弱了“长时间无新告警时，仅靠后台时间流逝就让 pending 自然成熟”的语义。

## 3. trigger 的含义与相关变量

**【逻辑】**

trigger 表示“某条规则开始被激活和等待聚合的起点告警”。
一条告警命中某规则的 `trigger_role` 后，会先进入 `trigger_event_index` 作为可触发候选；
若该 `(node, rule)` 当前还没有 pending，就会在 `pending_triggers` 中登记一个等待聚合的 trigger 锚点；
等到成熟后，再以这个 trigger 为起点进入规则评估与故障组汇聚。

相关变量作用如下：

- `trigger_role`：规则里定义的触发角色，表示哪类节点告警可以启动这条规则。
- `trigger_specs_by_node`：预编译的站点触发索引，表示某个站点可能触发哪些规则、对应哪些告警类型。
- `trigger_event_index`：按 `(node, rule)` 保存还能作为 trigger 候选的事件索引。
- `pending_triggers`：按 `(node, rule)` 保存当前仍在等待聚合窗口成熟的 trigger 锚点。
- `pending_trigger_heap`：按成熟时间排序的最小堆，用来高效摘取已成熟的 pending trigger。
- `consumed_trigger_rules`：表示这条告警已经被哪些规则当作 trigger 消费过；后续它不能再作为这些规则的 trigger，但仍可作为其它规则的 trigger。

**【意义】**

把 trigger 候选、pending 等待态、按 rule 维度的已消费状态拆开管理，
能同时满足三件事：一是避免同一条告警被同一条规则重复触发；二是保留聚合等待窗口；三是允许告警在失去某条规则的 trigger 身份后，仍作为其它规则的 trigger 或普通证据参与匹配。

## 4. `transmission_rule` 的逻辑与意义

**【逻辑】**

`transmission_rule` 的 `trigger_role` 是 `downstream_compound_node`。
它要求：

- 同一个 `parent_microwave_node` 下，至少有 `3` 个 `downstream_compound_node` 命中传输/传输+无线离线类告警；
- 同时这个 `parent_microwave_node` 往上还要能找到最近的 `grandparent_node`；
- 这个 `grandparent_node` 在 `Data` 或 `Transmission` 画像上不能出现 `OFFLINE_ALARMS`；
  也就是说，这里现在不是“完全 `NONE`”，而是通过 `forbidden_alarms`
  只禁止更上游出现离线类告警；
- 关联时间窗主要是 `600s`，表示上下游传播关系在 10 分钟内成立。

**【意义】**

这条规则表达的是一种“表层多点断站，向上传播收敛到同一个父层传输节点，并进一步校验更上游没有离线类告警”的传播语义，
用于识别跨域传播、父层传输异常或上游静默导致的成片站点断站场景。

## 5. `power_rule` 的逻辑与意义

**【逻辑】**

`power_rule` 的 `trigger_role` 是 `offline_node`。
它在同一个站点内建立 `power_node -> offline_node` 的自关联：

- `offline_node` 要命中 `OFFLINE_ALARMS`；
- `power_node` 要命中 `POWER_ALARMS`；
- 时间窗是非对称的：电源告警最多可以比离线告警早 `10800s`，也允许比离线告警晚 `600s`；
- 规则保留时长 `max_stay_time_sec` 也是 `10800s`，方便后续历史组合并。

**【意义】**

这条规则表达的是“同站点离线表象，回看电源类根因”的本地根因关联语义，
用于把离线类告警与同站点电源异常串起来，补足根因解释。

## 5.1 `link_rule` 的逻辑与意义

**【逻辑】**

`link_rule` 的 `trigger_role` 是 `link_child_offline_node`。
它要求：

- 儿子节点命中 `OFFLINE_ALARMS`；
- 其直接上游 `1` 跳的 `link_parent_node` 命中 `LINK_ALARMS`；
- 两者的关联时间窗是 `600s`。

也就是说，这条规则当前表达的是：

- “父节点出现链路/传输类告警”
- “随后儿子节点在时间窗内出现离线”

并且 role 命名已经和其它规则解耦，不会和 `offline_node`、`power_node` 等旧 role 混用。

**【意义】**

这条规则用于表达一种更直接的“上游链路异常 -> 下游断站”传播关系，
适合补充那些不一定会形成 `transmission_rule` 那种多分支聚集，
但确实存在明显父子传播迹象的场景。

## 6. 配置中 `ALL / ANY / NONE` 的含义

**【逻辑】**

当前配置里这几个关键词会出现在不同位置，但语义并不完全一样：

- 当它出现在 `expected_alarms` 里时：
  - `ANY` 表示“这个节点在窗口内出现任意告警都算满足”；
  - `NONE` 表示“这个节点在窗口内不能出现关键告警”；
- 当它出现在节点配置的 `match` 字段里时：
  - `ANY` 表示“候选节点里只要有一部分通过校验即可”；
  - `ALL` 表示“候选节点里所有被纳入这一分支的节点都必须通过校验，只要有一个失败，这一分支就失败”。

也就是说，`ANY / ALL` 在 `match` 上控制的是“同一批候选节点怎么判定通过”，
而 `ANY / NONE` 在 `expected_alarms` 上控制的是“单个节点在时间窗里需要出现什么告警”。

**【意义】**

虽然它们都叫 `ALL / ANY / NONE`，但作用层级不同：

- `expected_alarms` 是“单节点告警条件”；
- `match` 是“多候选节点的通过方式”。

把这两层分清楚后，后续新增规则时才不容易把“单节点要匹配什么告警”和“多个候选节点怎么共同通过”混为一谈。

## 7. `_evaluate_rule()` 的逻辑与意义

**【逻辑】**

`_evaluate_rule()` 以一个已成熟的 trigger 为起点，
按规则图中的边逐步向外扩展，去验证各个角色上的候选节点是否满足站点结构与告警窗口条件。
它的过程可以概括为：

- 先校验 trigger 节点自身是否满足 `trigger_role` 的要求；
- 再根据规则边逐步遍历上下游/自身节点；
- 对每个候选节点调用 `validate_node()` 做站点画像与告警窗口校验；
- 在存在多个可能路径时，通过“实例分叉”的方式保留不同匹配分支；
- 最终把一个满足规则图的匹配实例整理成故障组候选，产出 `inferred_roots`、`role_mapping`、`symptoms` 等信息。

**【意义】**

它是整套时序图引擎里真正负责“把 trigger 扩展成故障组”的核心过程；
前面的 `pending`、收割、历史组合并只是决定什么时候触发和如何落库，
而 `_evaluate_rule()` 决定了一条规则在当前 trigger 上到底能否匹配成功，以及故障组里具体包含哪些根因、站点和告警。

## 8. `_evaluate_rule()` 中 `checked` 字段的作用

**【逻辑】**

`checked` 是匹配实例里每个角色的一个状态位，用来表示“这个角色是否已经完成过一轮有效校验/展开”。
在 `_evaluate_rule()` 中：

- trigger 角色初始化时 `checked=False`；
- 当某个目标角色在一条边扩展中成功拿到候选节点后，会写入 `checked=True`；
- 如果后续再次遍历到这个角色，且它已经 `checked=True`，就不会重复把它当成未展开节点继续处理；
- 同时在回溯校验 `min_count` 时，也会结合 `checked` 判断：只有已经真正展开过的角色，才会因为收缩后节点数不足而让分支失效。

**【意义】**

`checked` 的存在是为了区分“角色暂时还没被展开过”和“角色已经展开过，但后续候选被收缩了”这两种状态。
这样可以避免重复展开同一个角色，也可以让 `min_count` 等数量约束在正确时机生效，保持整条规则图匹配过程稳定。

## 9. `_evaluate_rule()` 中 dependency 的作用

**【逻辑】**

当前在 `_evaluate_rule()` 的实例状态里，会额外记录一份 `dependency` 信息：

- `roles` 保存各个 role 当前还保留的候选节点与 `checked` 状态；
- `_dependencies` 保存“哪条边上的哪些上游节点依赖哪些下游节点”。

每当一条边扩展成功后，就会把这条边上的支撑关系记录下来；
然后运行一次收敛裁剪：

- 如果某个上游节点已经没有任何仍然存活的下游支撑，就把这个上游节点删掉；
- 如果某个下游节点已经没有任何仍然存活的上游支撑，也把这个下游节点删掉；
- 对已经 `checked=True` 的 role，再继续检查 `min_count` 是否仍然满足；
- 上述过程会反复迭代，直到这一轮不再发生节点收缩为止。

**【意义】**

这层 dependency 是为了让“深层后验失效”能够回传到更上游的角色。
也就是说，某个中间 role 看起来一开始满足条件，但如果它真正依赖的更下游节点在后续又被裁掉，
这次失效可以通过 dependency 回传回来，把中间 role 乃至更上游 role 一起收缩掉，
避免保留实际上已经无法支撑完整规则链的伪匹配分支。

## 10. `consumed_trigger_rules`

**【逻辑】**

当前不再用一个全局布尔值表示“这条告警是否已经被消费成 trigger”，
而是在 `event_cache` 里为每条事件保存一个 `consumed_trigger_rules` 集合。

一轮故障组最终成型后，会先按 `(node, alarm_type, rule)` 计算各自的 trigger cutoff；
然后对同一个 `node + alarm_type` 下的缓存事件逐条判断：

- 如果某条事件时间 `<=` 某个 `rule` 的 cutoff，就把该 `rule` 加进这条事件的 `consumed_trigger_rules`；
- 同时只从该 `(node, rule)` 对应的 `trigger_event_index` 里移除这条事件；
- 如果这些删除影响到了某个 `(node, rule)` 当前挂着的 pending，也只会刷新这个 `rule` 自己的 `pending_triggers` 锚点，不会把同站点其它 rule 的 pending 一起重算；
- 后续 `validate_node()` 在校验 trigger 角色时，也只会排除“已被当前 rule 消费过”的事件。

这样一条告警可以表现成：

- 对 `rule_A` 已经不能再作为 trigger；
- 但对 `rule_B` 仍然还可以继续作为 trigger。

**【意义】**

这样可以避免“某条规则消费得更晚，就把其它规则也一起误伤”的问题，
也能让同一条告警在不同规则上的 trigger 生命周期彼此独立；
同时又不丢失它在后续结构匹配中作为普通证据节点的价值。

## 11. 当前的 pending 扩充

**【逻辑】**

在一轮收割中，先对原始候选故障组做一次批内合并；
然后从这一批已合并故障组里，收集所有非 trigger 角色上的节点；
如果这些节点上存在 `pending_triggers`，就把对应 pending 评估出的故障组补充进来；
补充后的结果再与当前批统一做一次批内合并，最后再进入历史组合并；
整个扩充过程是只读的，不会消费、删除或修改这些 pending 的原始 trigger 状态，也不会影响它们后续继续按正常流程成熟触发。

**【意义】**

这样可以把当前批内已经成熟的故障组，与其内部非 trigger 节点上尚未成熟但已挂起的 pending 关系一起纳入考虑，
提前补齐本轮批次视图，减少因触发时机不同导致的故障组割裂；
同时又不会破坏这些 pending 未来作为原始 trigger 正常成熟和汇聚的时序语义。

## 12. debug 模式的流程与作用

**【逻辑】**

当前 `match_rules.py` 里的 debug 模式通过重复传入 `--debug-trigger SITE::ALARM` 启用。
它现在是“纯观测模式”，不会改变引擎内部的正常聚合行为：

- 所有告警仍然会像正常模式一样进入 `event_cache`；
- 所有 trigger 仍然会像正常模式一样注册、成熟、汇聚和进入历史组合并；
- debug 只是额外筛出和指定 `site + alarm` 相关的过程信息做打印。

当前 debug 主要会输出几类信息：

- 指定站点的最近事件缓存、`trigger_event_index`、`pending_triggers`；
- 指定站点收到 `POWER_ALARMS` 时，立即打印“电告警进入缓存”及缓存快照；
- 指定站点上的事件因为 `TTL` 或清除告警被从 `event_cache` 移除时，打印删除原因；
- 一轮收割中的阶段性结果，按真实流水线顺序输出：
  - 原始候选组；
  - 当前批次合并后；
  - pending 扩充后；
  - 历史组合并后。

需要注意的是，当前 debug 模式虽然不改引擎内部行为，但主输出文件仍然只写和 debug 目标相关的故障组；
因此 debug 更适合做问题定位，不适合直接当成全量结果文件使用。

**【意义】**

debug 模式的作用，是在不改变正常匹配语义的前提下，
把“某个站点 + 某类告警”在缓存、trigger、pending、批内合并、历史组合并等阶段的变化过程暴露出来。
这样在排查“为什么没聚起来”“为什么被提前清掉”“为什么历史合并后看不到”这类问题时，
可以直接看到中间状态，而不需要完全靠最终故障组倒推。

## 13. 故障组批内合并的逻辑与意义

**【逻辑】**

一轮收割里，多个成熟 trigger 可能会各自产生原始候选故障组。
这些候选组在正式进入历史组合并前，会先做一次“批内合并”：

- 每个原始候选组在生成时，会先根据“该组最早 symptom 时间 + 当前规则 `max_stay_time_sec`”算出自己的 `_expire_ts_hint`；
- 合并判断的核心依据是 `symptoms` 里的 `eid` 是否有交集；
- 只要两个候选组共享相同 `eid`，就认为它们属于同一批次内的同一故障上下文；
- 合并后会对以下信息做并集或去重合并：
  - `symptoms`
  - `role_mapping`
  - `inferred_roots`
  - `merged_rules`
  - `related_group_uuids`
- 批内合并后的 `_expire_ts_hint`，取参与合并的当前候选组里最大的那个；
- 在当前实现里，pending 扩充出的故障组也会回到这一层，再与当前批已有故障组统一做一次批内合并。

**【意义】**

批内合并的作用，是把“同一轮收割中、其实已经通过共享告警连在一起”的多个候选故障组先收敛成更完整的一组，
避免把同一批次里本应属于同一个传播上下文的结果拆成多条输出。
它解决的是“同批内部的碎片化”问题，让后面的历史组合并面对的是更稳定、更完整的当前批结果。
同时，按最大 `_expire_ts_hint` 保留当前批里最长的留存时间，也能避免批内合并后把本来应该保留更久的故障组缩短。

## 14. 故障组历史组合并的逻辑与意义

**【逻辑】**

批内合并后的故障组，会再与历史故障组缓存做“历史组合并”：

- 首先查找与当前组在 `eid` 上有交集的历史组；
- 如果存在某一个单独历史组，其 `eid` 集合完整包含当前组的 `eid` 集合，
  则当前组判定为“没有带来新的告警信息”，`should_emit=False`；
- 否则，当前组会与所有相关历史组做合并，并替换这些历史组为新的结果；
- 合并时会继续并集或去重合并：
  - `symptoms`
  - `role_mapping`
  - `inferred_roots`
  - `merged_rules`
  - `related_group_uuids`
- 当前组自己的 `expire_ts` 会优先沿用批内阶段算好的 `_expire_ts_hint`；
- 真正落历史缓存时，最终 `expire_ts` 取“当前组 `_expire_ts_hint`”与“所有相关历史组已有 `expire_ts`”中的最大值；
- 如果 `should_emit=False`，当前实现还会顺手延长相关历史组的 `expire_ts`：
  会拿“这次当前组自己的 `_expire_ts_hint`”去和这些相关历史组原本的 `expire_ts` 比较，
  然后把每个历史组的 `expire_ts` 更新成两者中的更大值；
  这样即使当前组因为被单个历史组完整包含而不再重新输出，也不会丢掉这次带来的更长留存时间。

**【意义】**

历史组合并的作用，是把“跨收割轮次、跨时间批次、但本质上属于同一故障演化链”的结果连起来，
避免同一批根因传播在不同时间点反复生成大量近似重复的故障组。
它解决的是“跨批次重复输出”和“历史上下文延续”问题，使故障组具备时间上的连续性。
同时，按当前组与历史组中最大的 `expire_ts` 继续保留，也能避免新一轮合并把历史上本来应该保留更久的故障组提前收尾。

## 15. `traverse_graph()` 的流程与作用

**【逻辑】**

`traverse_graph()` 的作用，是从当前已知的物理节点出发，沿规则边要求的方向去图里找可到达的候选节点。
它本身不负责做告警窗口校验，只负责“拓扑上哪些节点可达”这件事。

当前流程大致是：

- 输入起点节点 `start_node`、方向 `direction`、以及可选的路径节点要求等信息；
- 如果是 `self` 方向，则直接返回当前节点自己；
- 如果是普通拓扑方向（如 `upstream` / `downstream`），就到预构建的传播图中做遍历；
- 遍历过程中会结合 `path_node_requirements` 对路径上的中间节点做过滤；
- 最终返回一批可到达候选节点，以及它们对应的跳数 / 路径层级信息；
- 这些候选节点随后会进入 `validate_node()`，再继续做站点画像和告警时间窗校验。

当前实现里，这一步还带有 topo cache：

- cache 实体是 `global_topo_cache`，底层用 `OrderedDict` 维护，容量上限由 `max_topo_cache_size` 控制；
- cache key 当前是 `(start_node, direction, max_hops)`；
- 只有 `path_requirements is None` 时才会查缓存和写缓存；
- 也就是说，只有“纯拓扑可达性”的 BFS 结果才会被缓存复用；
- 一旦带了 `path_requirements`，就会跳过缓存，重新遍历；
- 命中缓存时，会把该项移动到尾部，保持近似 LRU 的淘汰顺序；
- 写入新结果后，如果缓存超过上限，会弹掉最旧的一项。

之所以“带路径约束时不缓存”，是因为这类遍历不仅依赖拓扑结构，
还依赖路径节点的站点画像、`reference_ts`、`edge_window` 等动态条件；
如果仍然直接复用纯拓扑结果，容易把本轮本不应该穿透的路径错误地算进去。

**【意义】**

`traverse_graph()` 解决的是“从当前节点出发，结构上有哪些节点可能属于下一个 role”这个问题，
也就是先做一层拓扑候选筛选，再把这些候选交给告警和时间窗逻辑处理。
这样可以把“结构可达性”和“告警是否满足”两件事拆开：

- `traverse_graph()` 管结构；
- `validate_node()` 管画像和告警窗口；
- `_evaluate_rule()` 管整条规则链能否成立。

这种拆分可以让规则匹配过程更清楚，也便于后续对拓扑遍历和告警校验分别做优化。

## 16. 规则评估过程中的缓存层次

**【逻辑】**

除了 `traverse_graph()` 使用的全局拓扑缓存外，当前规则评估过程里还有两层“局部缓存”：

1. 事件快照缓存  
   在进入一轮 `_evaluate_rule()` 之前，会先从当前 `event_cache` 拍一个与种子节点相关的快照；
   后续如果规则评估中又访问到新的节点，会通过 `_build_snapshot_helper()` 按需把这些节点的事件读出来并补进同一个快照。
   这样 `NodeRuleHelper.events_in_window()` 读到的是“本轮评估视角下的一致快照”，而不是边算边直接读实时 `event_cache`。

2. `validate_node()` 结果缓存  
   在 `_evaluate_rule()` 内部有一个 `validation_cache`；
   当某个候选节点在同一轮规则评估里，被用相同的 role、参考时间和窗口重复校验时，会直接复用上一次 `validate_node()` 的结果，
   避免重复做站点画像匹配和窗口告警过滤。

因此当前缓存层次可以理解为：

- `global_topo_cache`：缓存纯拓扑可达结果；
- `event snapshot cache`：缓存本轮评估要看的事件视图；
- `validation_cache`：缓存本轮评估里节点校验结果。

**【意义】**

这三层缓存分别解决不同层面的重复计算：

- 拓扑缓存减少重复 BFS；
- 事件快照保证本轮评估看到的事件集合一致，并减少反复加锁读实时缓存；
- `validate_node()` 结果缓存减少相同候选节点在同一轮评估中的重复校验。

这样既能提升性能，也能让一轮 `_evaluate_rule()` 在并发环境下保持更稳定、可解释的输入视图。

## 17. `match_rules.py` 中原始告警加载与标准化流程

**【逻辑】**

`match_rules.py` 的入口不是直接把原始文件喂给引擎，而是会先做一轮告警加载、过滤和标准化：

- 通过 `stream_alarm_inputs()` 顺序读取原始告警；
- 只保留 `告警标题` 落在有效告警集合里的告警；
- 站点优先取原始 `站点ID`，如果没有或不在有效站点集合里，再回退用 `告警源 -> site_id` 映射补站点；
- 每条有效告警都会先生成一条“正常告警事件”；
- 如果存在 `告警清除时间`，还会再额外生成一条“清除告警事件”：
  - 时间改成 `告警清除时间`
  - 并在事件里补 `清除告警 = 是`
- 这些标准化后的事件会统一落到 `valid_alarms`，字段主要包括：
  - `alarm`
  - `site_id`
  - `alarm_source`
  - `alarm_title`
  - `ts`
- 最后会按 `ts` 排序，并裁掉尾部“仅由清除告警组成”的那一段。

**【意义】**

这一步把原始告警流先统一转换成引擎能直接处理的事件格式，
并且把“首次发生”和“清除”两个时点都显式化。
这样后面的 live / offline 回放、trigger 注册、清除事件处理和最终故障组时间线，都会基于同一套标准化事件输入。

## 18. `match_rules.py` 主入口执行流程

**【逻辑】**

`main()` 的整体执行顺序可以概括为：

1. 解析命令行参数  
   读取输入告警文件、拓扑文件、站点画像文件、运行模式、speedup、debug 目标等参数。

2. 加载静态数据  
   包括：
   - 站点拓扑 `site_graph_by_ne.json`
   - 站点画像 `site_device_counts.json`
   - NE 画像 `ne_graph.json`
   - 有效站点集合
   - `ne -> site` 映射

3. 初始化引擎  
   用拓扑、规则配置、站点画像构造 `TemporalGraphEngine`。

4. 加载并标准化告警  
   调用 `_load_valid_alarms()` 生成 `valid_alarms`，排序后再做尾部清除告警裁剪。

5. 根据模式选择运行方式  
   - debug 模式：走 `_run_debug_mode()`
   - live 模式：走 `_run_live_mode()`
   - offline 模式：走 `_run_offline_mode()`

6. 数据流结束后做最终收尾  
   调用 `engine.flush_pending()`，把最后还挂着的 pending 做一次收尾收割。

7. 输出统计信息  
   最后打印：
   - 原始处理告警数
   - 过滤后有效告警数
   - 最终故障组数
   - 总耗时

**【意义】**

这一层把“静态数据准备”“告警标准化”“引擎驱动”“结果落盘”串成了一条完整入口链，
是整个项目从命令行运行到故障组生成的最外层调度器。
理解这条主入口流程后，排查问题时就能更容易区分：

- 是输入数据阶段的问题；
- 是引擎匹配阶段的问题；
- 还是最后输出与展示阶段的问题。

## 19. 故障组输出构建与落盘流程

**【逻辑】**

引擎内部生成的故障组，最终不会原样直接写文件，而是会在 `match_rules.py` 里再做一层输出增强：

1. `on_matches(matches)` 统一处理每一批新产出的故障组  
   对每个故障组会先调用 `generate_incident_report(match)` 生成控制台报告，
   再调用 `_build_jsonl_match_output(match, ne_graph_data)` 组装最终落盘结构。

2. `_build_jsonl_match_output()` 会在原始 match 基础上补充：
   - `group_anchor_ts`
   - `group_anchor_time`
   - `match_info`
   - `ne_info`
   - `group_info`

3. 其中 `_build_group_output()` 会把故障组内部信息整理成更适合前端展示的结构：
   - `match_info`：保存 `uuid / rule / merged_rules / related_group_uuids / inferred_roots / role_mapping`
   - `ne_info`：按设备组织告警、设备画像和组内链路
   - `group_info`：按故障组组织 `ne_list / site_list`

4. 最终按 `jsonl` 一行一个故障组追加写入输出文件。

**【意义】**

这一步把“引擎内部的匹配结果”转换成了“既能程序消费、又能被前端页面直接加载展示”的输出格式。
也就是说：

- `TemporalGraphEngine` 更偏匹配与聚合；
- `match_rules.py` 的输出构建更偏落盘与展示适配。

把这层分开后，规则引擎本身可以保持聚焦，而前端页面又能直接复用 `jsonl` 做故障组总览和单组传播图展示。

## 20. `compute_ticket_site_recall.py` 的流程与作用

**【逻辑】**

这份脚本是“基于原始告警流里的 `工单号 + 故障组ID`”来计算工单站点召回率。
整体过程可以概括为：

1. 先确定每个工单的目标站点列表  
   - 如果显式提供了 `--ticket-sites`，就直接使用这份 `{工单号: [站点列表]}`；
   - 如果没有提供，就从原始告警里反推出 `工单 -> 站点`。

2. 流式读取告警，建立两层索引  
   - `ticket -> fault_groups`：根据告警里的 `工单号` 和 `故障组ID` 建立；
   - `fault_group -> sites`：根据带有该 `故障组ID` 的告警，把站点并起来。

3. 过滤分母  
   只有“在原始告警里真实出现过”的工单才会参与平均召回率计算。

4. 逐工单计算召回率  
   - 先拿到该工单的目标站点；
   - 再把该工单关联到的所有 `故障组ID` 对应站点做并集；
   - 最后同时计算：
     - `recall = |ticket_sites ∩ recalled_sites| / |ticket_sites|`
     - `precision = |ticket_sites ∩ recalled_sites| / |group_sites|`
     - `f1 = 2PR / (P + R)`。

5. 输出明细  
   每个工单会输出：
   - 工单站点数与站点列表；
   - 告警数；
   - 关联故障组数与故障组列表；
   - 召回站点数与召回站点列表；
   - group 覆盖到的全部站点；
   - 最终 `recall / precision / f1`。

**【意义】**

这份脚本描述的是“如果只看原始告警里已经存在的 `工单号 + 故障组ID` 关系，
理论上能把工单目标站点召回到什么程度”。
它更像一条“原始数据口径”的基线，用来衡量：

- 告警里的故障组字段本身是否已经足够把工单相关站点串起来；
- 后续 `match_rules.py` 聚出来的故障组，相比这条原始基线是更好还是更差。

## 21. `compute_group_output_ticket_recall.py` 的流程与作用

**【逻辑】**

这份脚本是“基于 `match_rules.py` 聚合输出的故障组”来计算工单站点召回率。
它与上一份脚本的召回率公式相同，但构图来源不同：

1. 先确定工单目标站点  
   - 优先使用 `--ticket-sites`；
   - 如果没有提供，就从 `--alarms` 反推出 `工单 -> 站点`。

2. 读取故障组输出，建立两层索引  
   - `ticket -> group_output groups`：从故障组 `symptoms[*].工单号` 提取；
   - `group -> sites`：优先取故障组里的 `group_info.site_list`，没有时再退回 `symptoms[*].node`。

3. 确定分母口径  
   - 如果提供了 `--alarms`，则仍以“工单是否在原始告警里出现过”为准；
   - 否则退化成“工单是否在故障组输出里出现过”为准。

4. 复用 `compute_ticket_site_recall.py` 的同一套召回率计算函数  
   所以最终会沿用同一套站点指标：
   - `recall`
   - `precision`
   - `f1`

5. 输出明细  
   与原始告警口径类似，但这里统计的是：
   - `ticket_occurrence_count`
   - `fault_group_count`
   - `recalled_sites`
   - `group_sites`
   - `recall / precision / f1`

**【意义】**

这份脚本用来评估“当前 `match_rules.py` 聚出来的故障组，对工单站点的覆盖情况到底如何”。
它是最直接的结果评估工具之一，因为它回答的是：

- 聚类后的故障组，是否真的把工单目标站点串起来了；
- 聚合结果相比原始 `故障组ID` 字段，是提升了召回，还是损失了召回。

## 22. `compute_ticket_site_recall_upper_bound.py` 的流程与作用

**【逻辑】**

这份脚本不是算当前方法的真实召回，而是算“基于原始告警流与时间窗关系，工单站点召回率的上限”。
当前逻辑是：

1. 输入 `{工单: [站点列表]}` 和原始告警流  
   先确定每个工单的目标站点集合。

2. 第一遍流式扫描告警  
   - 过滤掉 `FAN FAIL`；
   - 只把真实在告警里出现过的工单纳入分母；
   - 收集每个工单自己的 anchor 告警时间；
   - 收集这些工单已经显式命中的站点。

3. 基于 anchor 时间，为每个工单生成 `±window_seconds` 的时间窗  
   然后找出该工单目标站点里还没有显式出现的缺失站点。

4. 第二遍流式扫描告警  
   检查缺失站点上的告警时间，是否能落入该工单任一 anchor 时间窗内；
   如果可以，就把这个站点视为“通过时间窗补关联成功”。

5. 计算上限召回率  
   - 直接命中的站点 + 时间窗补回的站点 = `associated_sites`；
   - 按：
     `associated_site_count / ticket_site_count`
     直接计算 `recall_upper_bound`；
   - 同时也会输出：
     - `precision_upper_bound`
     - `f1_upper_bound`
     作为与其它评测脚本一致的补充指标。

6. 第三遍流式扫描告警，补证据  
   输出：
   - `direct_site_alarms`
   - `inferred_site_alarms`
   作为“这些站点为什么能被关联上”的证据。

**【意义】**

这份脚本给出的不是当前方法已经做到的结果，而是“在现有告警流里，理论上还能做到多好”。
它的价值主要有两点：

- 为真实召回率提供一个上界参考，判断当前方法离上限还有多远；
- 把“直接命中”与“时间窗补关联”的证据一起输出，便于后续追查哪些站点本来就有希望被召回，但当前方法没做到。

## 22.1 `compute_ticket_site_recall_upper_bound.py` 中 evidence 与时间窗的计算细节

**【逻辑】**

当前 upper bound 里的 `evidence` 不是随便把工单相关告警都存下来，而是分成两桶：

- `direct_site_alarms`
- `inferred_site_alarms`

它们都依赖前面先算好的“工单时间窗”。

### 1. 工单时间窗怎么计算

1. 先扫描原始告警流，只看“带当前工单号”的告警  
   这些告警的时间来自 `time_field`，默认是 `告警首次发生时间`。

2. 把这些时间解析成时间戳后，按工单聚合成 `ticket_alarm_times[ticket_id]`。

3. 对每个工单的时间点做排序去重，然后把每个时间点 `ts` 扩成：
   `[ts - window_seconds, ts + window_seconds]`

4. 如果相邻两个窗口重叠，就合并成一个更大的窗口；
   如果不重叠，就保留成两段独立窗口。

最终 `ticket_windows[ticket_id]` 保存的是：

- `starts`
- `ends`

这两列一一对应，表示该工单所有合并后的时间窗。

### 2. 时间窗合并举例

如果 `window_seconds = 600`，也就是前后各 `10` 分钟：

- `10:00` 会扩成 `[09:50, 10:10]`
- `10:05` 会扩成 `[09:55, 10:15]`

因为两段重叠，所以会合并成：

- `[09:50, 10:15]`

但如果是：

- `10:00` -> `[09:50, 10:10]`
- `10:30` -> `[10:20, 10:40]`

两段不重叠，就不会合并，最终保留为两段。

因此像 `10:15` 这种时间点：

- 不会落在 `[09:50, 10:10]` 中
- 也不会落在 `[10:20, 10:40]` 中

所以它不会被判定为“落入工单时间窗”。

### 3. `direct_site_alarms` 怎么收

`direct_site_alarms` 收的是：

- 告警所在站点属于该工单的 `direct_sites`
- 并且这条告警本身的工单号就等于当前工单号

也就是说，这一桶代表的是：

- “这个站点本来就直接带出了该工单号”
- “所以它是工单的直接命中站点”

### 4. `inferred_site_alarms` 怎么收

`inferred_site_alarms` 收的是：

- 告警所在站点属于该工单通过时间窗补出来的 `inferred_sites`
- 且这条告警的时间落进该工单的 `ticket_windows`

这里不要求这条告警本身带当前工单号。

也就是说，这一桶代表的是：

- “这个站点并不是靠工单字段直接命中的”
- “而是因为站点上的告警时间和该工单的时间窗对得上，所以被补关联进来”

### 5. 当前 evidence 的整体含义

所以当前 upper bound 输出里的 `evidence` 可以理解为：

- `direct_site_alarms`：证明“这个站点为什么能直接算到这个工单头上”
- `inferred_site_alarms`：证明“这个站点为什么能通过时间窗被补到这个工单头上”

它不是“工单相关的所有告警全集”，而是：

- 只保留 direct / inferred 两类已关联站点上的证据告警
- 并且仍然会过滤掉 `FAN FAIL`

**【意义】**

把 evidence 拆成 direct / inferred 两桶之后，后续分析会更清楚：

- 如果一个站点出现在 `direct_site_alarms`，说明它本来就被工单字段显式标出来了；
- 如果一个站点只出现在 `inferred_site_alarms`，说明它完全依赖时间窗补关联；
- 后面的 `only-offline`、`potential` 等逻辑，本质上都是建立在这两桶 evidence 的统一读取之上。

### 6. 为什么 `precision_upper_bound` 往往是 `1`

当前 upper bound 的“预测站点集合”本身就是从工单 `target_sites` 里长出来的：

- `direct_sites` 先取的是“带工单号告警命中的目标站点”；
- `inferred_sites` 也是只在“目标站点里的缺失站点”上做时间窗补关联；
- 最终 `associated_sites` 仍然是 `target_sites` 的子集。

因此在当前定义下：

- false positive site 基本不存在；
- 只要 `associated_sites` 非空，
  `precision_upper_bound = |associated_sites ∩ target_sites| / |associated_sites|`
  就会等于 `1.0`。

所以 `precision_upper_bound` 目前更像是一个“口径补充字段”，
而不是像真实评测那样有明显区分度的指标。

## 23. `compute_filtered_real_recall.py` 的流程与作用

**【逻辑】**

这份脚本的作用是：只挑出“在上限分析里，本来就能把全部站点关联出来”的工单，
再去看当前真实方法在这些工单上的平均召回率。

具体过程是：

1. 读取真实召回率结果  
   来源可以是：
   - `compute_ticket_site_recall.py`
   - 或 `compute_group_output_ticket_recall.py`

2. 读取上限结果  
   来源是 `compute_ticket_site_recall_upper_bound.py`。

3. 用上限结果筛工单  
   只保留满足：
   `associated_site_count == ticket_site_count`
   的工单，也就是“理论上所有目标站点都已经可以被关联出来”的工单。

4. 对这些工单重新计算真实平均召回率  
   输出：
   - 原始工单数；
   - 具备比较资格的工单数；
   - `filtered_average_recall`；
   - 对应工单明细。

**【意义】**

这份脚本的重点不是再算一遍召回率，而是把“不可能被召回的工单”和“本来可以全召回、但当前方法没做到的工单”分开。
这样能更准确地回答一个问题：

- 当前方法的不足，到底是数据先天上限造成的；
- 还是方法本身还没有把本来可以做到的部分做出来。

## 24. `compute_ticket_site_recall_v2.py` 与 `compute_group_output_ticket_recall_v2.py` 的流程与作用

**【逻辑】**

这两份 `v2` 脚本是在前面真实召回率脚本的基础上，再叠加上限结果做“差异对照”。
它们共同的思路是：

1. 先读取 `compute_ticket_site_recall_upper_bound.py` 的输出  
   只保留“上限里能把全部站点关联出来”的工单。

2. 再跑各自的真实方法  
   - `compute_ticket_site_recall_v2.py`：
     仍然基于原始告警里的 `工单号 + 故障组ID`；
   - `compute_group_output_ticket_recall_v2.py`：
     仍然基于 `match_rules.py` 聚合输出的故障组。

3. 对每个保留下来的工单，把站点拆成两类  
   - `associated_sites`：当前方法已经关联上的站点；
   - `missing_sites`：当前方法还没关联上的站点。

4. 同时输出两类站点对应的告警证据  
   - `associated_site_alarms`：当前方法已经真正关联到的站点告警；
   - `missing_site_alarms`：上限里明明可以补出来、但当前方法仍未关联的站点告警。

5. 两份脚本统一输出字段  
   包括：
   - `associated_site_count / associated_sites / associated_site_alarms`
   - `missing_site_count / missing_sites / missing_site_alarms`
   - `group_site_count / group_sites`
   - `recall / precision / f1`

**【意义】**

`v2` 的重点不是只看一个平均分，而是把“已经召回的部分”和“还没召回但其实有证据可以召回的部分”显式分开。
这样在分析时可以直接回答：

- 当前方法已经把哪些站点串起来了；
- 剩下哪些站点其实有时间窗/故障组证据，但当前方法没用上；
- 原始告警口径和聚合故障组口径，到底分别漏掉了哪些站点。

## 24.1 `v2` 的默认模式、`loose`、`only-offline`、`potential` 的当前逻辑

**【逻辑】**

当前两份 `v2` 脚本都支持：

- 默认模式（不开额外选项）
- `--loose`
- `--only-offline`
- `--potential`
- `--only-one`

另外：

- `compute_group_output_ticket_recall_v2.py` 还额外支持 `--ultimate-only`

它们的作用层级并不一样：

- 默认模式决定“当前方法本来能关联到哪些 group / 故障组ID”
- `loose` 决定“是否允许 group 之间按时间窗进一步扩张”
- `potential` 决定“是否允许 upper bound evidence 里的告警把额外 group 直接吸附进来”
- `only-one` 决定“最终是否只保留一个最优 group 来算站点指标”
- `only-offline` 决定“这个工单样本最终要不要保留到分母里”
- `ultimate-only` 决定“对 group output 口径，是否先排除所有只是作为关联 group 被别人引用的 group”

### 1. 默认模式

默认模式下，两份脚本都只使用 `base groups`：

- `compute_ticket_site_recall_v2.py`
  - `base groups` 来自原始告警里的 `工单号 + 故障组ID`
- `compute_group_output_ticket_recall_v2.py`
  - `base groups` 来自聚合输出里的 `symptoms[*].工单号 + 当前 group id`

然后用这些 `base groups` 覆盖到的站点去计算：

- `associated_sites`
- `missing_sites`
- `group_sites`
- `recall / precision / f1`

此时不会引入任何额外的 group。

### 2. `--loose`

`loose` 是“group 间按时间窗做闭包扩张”。

它的过程是：

1. 先拿当前工单已经有的 `base groups`
2. 再只看这个工单 `target_sites` 上出现过的其它 group
3. 收集这些 group 在 `target_sites` 上的告警时间
4. 用当前已纳入 group 的时间，构造 `±window_seconds` 的合并窗口
5. 只要某个候选 group 在 `target_sites` 上的任意时间点落进窗口，就把这个 group 并进来
6. 新并进来的 group 又继续提供时间，再做下一轮扩张，直到不能继续扩

所以 `loose` 的本质是：

- 从 `base groups` 出发
- 在工单目标站点范围内
- 用时间窗做 group 间传递闭包

当前它用的时间窗大小，来自 `compute_ticket_site_recall_upper_bound.py` 输出里的 `window_seconds`。

### 3. `--only-offline`

`only-offline` 不会改变 group 的关联过程，而是一个样本过滤开关。

当前逻辑是：

1. 先读取 upper bound 输出里的 `evidence`
2. 把其中：
   - `direct_site_alarms`
   - `inferred_site_alarms`
   合并成统一的 `site_evidence`
3. 如果这个工单的整份 `site_evidence` 里没有出现 `OFFLINE_ALARMS`
4. 就把这个工单样本直接跳过：
   - 不进入 `details`
   - 不进入 `ticket_count`
   - 不进入 `average_recall` 的分母

所以 `only-offline` 的作用是：

- 不是改“怎么关联”
- 而是改“哪些样本值得统计”

### 4. `--potential`

`potential` 现在已经不是“按时间窗吸附额外 group”，而是更直接的：

- 看 upper bound evidence 里的告警
- 这些告警分别出现在哪些 group / 故障组ID
- 命中的 group 就直接和当前工单建立关联

也就是说，`potential` 当前只回答一个问题：

- “upper bound 里已经出现过的这些证据告警，本身属于哪些 group？”

如果某个 group 里包含了这些 evidence 告警，它就会被吸附进来。

当前实现里：

- `compute_ticket_site_recall_v2.py`
  - 是按原始告警里的 `告警编码ID -> 故障组ID`
    去反查
- `compute_group_output_ticket_recall_v2.py`
  - 是按 group output 的 `symptoms` 里的 `eid/告警编码ID -> group`
    去反查

这里不会再额外做时间窗判断。

### 5. 这些模式如何叠加

当前这几个模式可以组合使用，它们的顺序是：

1. 先算 `base_fault_groups`
2. 如果开了 `--loose`，得到 `loose_fault_groups`
3. 如果开了 `--potential`，再得到 `potential_fault_groups`
4. 对 `compute_group_output_ticket_recall_v2.py` 来说，如果开了 `--ultimate-only`，
   则以上几步都只在“最终 group”范围内进行
5. 如果开了 `--only-one`，则会在
   `base ∪ loose ∪ potential`
   中只保留“覆盖当前工单目标站点最多的那一个 group”
6. 最终真正参与站点指标计算的是：
   `fault_groups = base ∪ loose ∪ potential`
   或者在 `only-one` 模式下的那一个 `selected_fault_group`
7. 如果开了 `--only-offline`，则在输出前再决定这个工单样本是否直接跳过

所以：

- `loose` 和 `potential` 都会改变 `fault_groups`
- `only-one` 会改变最终拿来计算指标的有效 group 集合
- `only-offline` 不改 `fault_groups`，只改最终样本集
- `ultimate-only` 会先收窄 group output 口径里可参与计算的 group 范围

**【意义】**

把这几个模式区分清楚之后，就能更准确地理解 `v2` 输出里每一部分差异是怎么来的：

- 默认模式：当前方法本来的能力
- `loose`：当前方法内部是否还能通过 group 间时间关系再扩一层
- `potential`：如果直接借用 upper bound 的证据告警，还能把哪些额外 group 拉进来
- `only-one`：如果只允许保留单个最优 group，当前方法还能保住多少站点
- `only-offline`：最终统计时是否只看“确实有离线证据”的工单样本
- `ultimate-only`：对于 group output 结果，只看最终 group 后，评测结果会变成什么样

## 25. `compute_ultimate_group_alarm_group_metrics.py` 的流程与作用

**【逻辑】**

这份脚本不是按“工单 -> 站点”评测，而是把：

- `match_rules.py` 输出里的**终极 group**
- 和原始告警流里的**告警故障组ID**

当成两套可以互相对照的站点覆盖集合，分别做双向评测。

### 1. 基础对象怎么定义

1. 先读取 `match_rules.py` 输出，并按 `related_group_uuids` 排除“被别的 group 作为关联 group 引用过的 group”；
   剩下的 group 视为**终极 group**。

2. 对每个终极 group，提取：
   - `group_info.site_list` / `symptoms.node` 覆盖到的站点；
   - `symptoms[*].故障组ID` 中出现过的告警故障组ID；
   - `symptoms[*].eid/告警编码ID` 中出现过的告警ID。

3. 对原始告警流，提取：
   - 每个告警故障组ID覆盖到的站点；
   - 每个告警故障组ID出现过的告警ID。

这样就得到两套对象：

- 终极 group -> 站点 / 告警故障组ID / 告警ID
- 告警故障组ID -> 站点 / 告警ID

### 2. 正向与反向怎么评测

当前会同时算两组指标：

1. `ultimate_group_as_gold`
   - gold label = 终极 group 覆盖到的站点
   - base prediction = 这个终极 group 的 `symptoms[*].故障组ID` 对应的告警故障组ID
   - 预测站点 = 这些告警故障组ID 在原始告警流里覆盖到的站点并集

2. `alarm_group_as_gold`
   - gold label = 告警故障组ID 覆盖到的站点
   - base prediction = 包含这个告警故障组ID 的终极 group
   - 预测站点 = 这些终极 group 覆盖到的站点并集

然后按站点集合计算：

- `recall`
- `precision`
- `f1`

### 3. `--min-site-num`

`--min-site-num` 作用在**当前作为 gold label 的那一侧**。

只有当：

- `gold_site_count >= min_site_num`

这个样本才会进入均值统计。

所以：

- 在 `ultimate_group_as_gold` 方向，过滤的是终极 group 的站点数；
- 在 `alarm_group_as_gold` 方向，过滤的是告警故障组ID 的站点数。

### 4. gold label 的 domain 过滤

当前这份脚本额外有一个约束：

- 如果某个 gold label 里出现过 `domain` 不属于 `Ran / Transmission` 的告警，
  这个 gold 样本直接跳过。

两边的 domain 来源不同：

1. `ultimate_group_as_gold`
   - 从 group output 的 `ne_info[*].alarm[*].domain` 判断

2. `alarm_group_as_gold`
   - 先读原始告警里的 `domain/Domain/DOMAIN`
   - 如果没有，再用 `告警源 -> ne_graph.json -> domain` 回填

所以：

- `Data`
- 未知类型
- 或任何不属于 `Ran / Transmission` 的告警

只要出现在当前 gold label 里，这个样本就不会被统计。

### 5. `--only-offline`

`--only-offline` 是一个**gold 样本过滤开关**，不会改变 prediction group 的关联过程。

它的语义是：

- 只统计**包含离线告警**的 gold label 样本

两边的判断口径是：

1. `ultimate_group_as_gold`
   - 看终极 group 的 `symptoms` 里是否出现 `OFFLINE_ALARMS`

2. `alarm_group_as_gold`
   - 看该告警故障组ID对应的原始告警里是否出现 `OFFLINE_ALARMS`

所以 `only-offline` 的作用是：

- 不改 `base / loose / potential / only-one`
- 只决定当前 gold 样本是否进入最终分母

### 6. `--require-transmission-per-site`

`--require-transmission-per-site` 不是直接整条过滤 gold label，
而是先对 gold label 的站点做一次**按站点裁剪**：

- 只保留在 `ne_graph.json` 中至少存在一个 `domain=Transmission` 设备的站点

裁剪之后：

- 如果 gold 站点集合变空，则该样本跳过
- 如果裁剪后的 gold 站点数 `< min-site-num`，则该样本跳过

所以这个开关的真实语义是：

- 先把 gold label 收成“有 Transmission 设备支撑的站点子集”
- 再用这个收缩后的 gold 站点集合去计算后续指标与 case

这会影响：

- `gold_sites`
- `gold_site_count`
- `gold_site_count_distribution`
- `recall / precision / f1`
- cases 里的 gold 站点范围

### 7. `--loose`

`--loose` 的语义和 `v2` 脚本一致，也是“group 间按时间窗做闭包扩张”，只是这里扩的是：

- 终极 group <-> 告警故障组ID

它的过程是：

1. 先拿当前 gold 样本已有的 base prediction group
2. 只在当前 gold 站点范围内，收集其它候选预测 group
3. 提取这些 group 在 gold 站点上的告警时间
4. 用当前已纳入预测 group 的时间构造 `±window_seconds` 的合并窗口
5. 只要候选 group 在这些站点上的任意时间点落进窗口，就把它并进来
6. 新并进来的 group 又继续提供时间，再做下一轮扩张

所以 `loose` 本质上是：

- 不改变 gold label
- 只在当前 gold 站点范围内，把 prediction group 按时间窗再扩一层

### 8. `--potential`

`--potential` 不是按时间窗扩张，而是按**告警ID命中**来吸附额外 prediction group。

它的过程是：

1. 先收集当前 gold 样本里已经出现过的告警ID
2. 看这些告警ID还落在哪些另一侧的 group 里
3. 命中的 group 直接并入 prediction group

所以 `potential` 回答的问题是：

- “如果直接用同一批告警ID做桥，另一侧还有哪些 group 可以被吸进来？”

### 9. `--only-one`

`--only-one` 会在：

- `base ∪ loose ∪ potential`

这批 prediction group 里，只保留**覆盖当前 gold 站点最多的单个 group**。

随后：

- `predicted_sites`
- `recall / precision / f1`

都只基于这个单 group 来计算。

### 10. 可选项叠加顺序

当前 prediction group 的组合顺序是：

1. `base`
2. `loose`
3. `potential`
4. `only-one`

样本过滤顺序是：

1. `domain` 过滤
2. `only-offline`
3. `require-transmission-per-site` 对 gold 站点做裁剪
4. `min-site-num`

也就是说：

- `base / loose / potential / only-one` 决定“预测结果长什么样”
- `domain / only-offline / require-transmission-per-site / min-site-num` 决定“当前 gold 样本算不算进分母，以及 gold 站点集合会不会先被收缩”

### 11. 两个方向的 `cases.jsonl`

这份脚本会像 `v2` 一样，额外输出两份 sidecar 可视化文件：

- `ultimate_group_as_gold.cases.jsonl`
- `alarm_group_as_gold.cases.jsonl`

它们的用途是：

- 把每个未满召回样本整理成可以直接给现有 HTML 页面加载的故障组样式 JSONL
- 便于在页面上直接查看：
  - gold 站点
  - 已命中站点
  - 未命中站点
  - 相关告警

这里要注意，当前 case 的展示口径是：

1. 展示站点范围只取 **gold 站点**
   - 不会把“只在预测里有、但不在 gold 里的站点”并进 case

2. 节点范围不是只保留“有告警的设备”
   - 会先根据 gold 站点去 `ne_graph.json` 里拉这个站点上的设备
   - 所以同站点其它设备也会进入 `ne_info`
   - 这些设备之间的 link 也会按 `ne_graph.json` 正常生成

3. 告警展示是“命中侧看 prediction，漏召侧看 gold”
   - `associated_site_alarms`：来自 prediction group，在已命中站点上的告警
   - `missing_site_alarms`：来自 gold label，在未命中站点上的告警

所以 case 的真实语义是：

- 站点范围按 gold 来定
- 节点和连边按 `ne_graph.json` 来补
- 命中站点的告警证据更偏 prediction 侧
- 漏召站点的告警证据更偏 gold 侧

### 12. 输出字段补充

除了两组方向性的平均指标之外，当前还会输出：

- `ultimate_group_as_gold.sample_count`
- `alarm_group_as_gold.sample_count`
- `ultimate_group_as_gold.gold_site_count_distribution`
- `alarm_group_as_gold.gold_site_count_distribution`
- `ultimate_group_as_gold_case_jsonl_output`
- `alarm_group_as_gold_case_jsonl_output`

需要注意：

- 顶层的 `ultimate_group_count` / `alarm_group_count`
  代表的是原始索引规模
- 真正进入平均值计算的样本数，要看两个方向各自的 `sample_count`

**【意义】**

这份脚本的重点不是评测“工单召回率”，而是回答这样两个问题：

- `match_rules.py` 聚出来的终极 group，和原始告警里的故障组ID，在站点覆盖上到底有多一致？
- 如果把视角反过来，原始告警故障组ID 又能在多大程度上被终极 group 解释？

再加上 `min-site-num / loose / potential / only-one / domain过滤` 这些开关后，
它可以帮助分析：

- 纯 base 关系下两侧的重合程度；
- 时间窗扩张或告警ID吸附后是否能显著提升一致性；
- 如果只允许一个“最优” group，匹配能力会不会明显下降；
- Data/未知类型告警过滤后，剩下的 Ran/Transmission 场景是否更一致。
