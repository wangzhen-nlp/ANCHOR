# TODO

- [ ] 评估并设计“pending 扩充”的循环收敛版本  
  当前只做一轮扩充，可能遗漏“扩充后新引入的非 trigger 节点上也存在 pending”的场景。  
  后续如果实现循环扩充，需要明确收敛条件，避免重复评估或进入无效循环。

- [ ] 评估告警流时间语义  
  当前处理默认按“告警首次发生时间”排序与推进，  
  后续需要考虑按“告警首次采集时间”或“告警首次入库时间”作为告警流到达顺序，  
  同时继续以“告警首次发生时间”参与故障组聚合。

- [ ] 评估 live 模式下的成熟判断语义  
  当前为避免模拟时钟推进过快导致 `event_cache` 和历史故障组被提前清理，  
  已将 TTL / 历史组过期 / 成熟判断统一切到“已到达事件时间”语义；  
  后续需要再评估是否要兼顾“长时间无新告警时，pending 也能自然成熟收割”的 live 模式需求。

- [ ] 评估 dependency 与复杂规则图的 matcher 边界  
  当前 `_dependencies` 以 `(curr_role, tgt_role)` 作为 key；  
  如果未来一条规则里出现“同一对 role 之间存在两条不同语义边”的情况，需要确认这里是否会互相覆盖，  
  以及是否要把 edge 本身的身份也纳入 dependency key。  
  同时，对于多条独立入边、菱形/网状结构这类更复杂规则图，也需要继续评估当前 matcher 是否足够通用，  
  或者是否需要更系统的图约束求解方式。

- [ ] 评估单条规则支持多个 `trigger_role` 的改造方案  
  当前引擎默认一条规则只有一个 `trigger_role`；  
  `trigger` 索引构建、规则起算、trigger 消费回收、debug 识别等逻辑都按单 trigger 入口实现。  
  如果未来需要在一个 rule 里配置多个 `trigger_role`，需要统一改造上述链路，  
  或者明确继续采用“拆成多条 rule”的约束。

- [ ] 评估 symptom 的跨 rule 归属表达  
  当前 trigger 消费已经按 `rule` 粒度拆开，但故障组 `symptoms` 仍主要依赖 `matched_role` 表达其角色信息；  
  如果未来不同规则共用相同 `trigger_role`，或者同一条 `eid` 同时参与多条规则，  
  需要考虑为 symptom 显式记录 `matched_rule(s)` / `matched_roles_by_rule`，避免 trigger 回收时的归属歧义。

- [ ] 优化 `nearest_matching` 的拓扑遍历路径  
  当前 `_traverse_graph()` 会先做完整 BFS，再由 `select_candidates_by_rule()` 从全量候选里筛最近一层；  
  后续可评估把 `nearest_matching` 直接并入 BFS，改成“按层遍历、命中首层后扫完当前层即早停”，  
  以保持现有语义不变的前提下降低深层无效拓扑扫描成本。  
  同时建议保留现有 `global_topo_cache` 作为“纯拓扑可达结果”缓存，  
  不直接改变其语义；如后续确认 `nearest_matching` 仍是热点，再考虑为这条早停路径单独设计新缓存。

- [ ] 评估事件查询按告警类型索引的优化方案  
  当前 `event_cache[node]` 是节点级总事件序列，`events_in_window()` 查询某类告警时需要扫描该节点整条缓存；  
  后续可评估为事件查询增加“按 `alarm_type` 分桶”的索引，以减少 `OFFLINE_ALARMS / POWER_ALARMS / LINK_ALARMS / forbidden_alarms` 这类场景下的无效扫描。  
  需要重点比较两条路线：
  - 双维护方案：保留当前节点级总队列，同时额外维护按告警类型分桶；  
    优点是对现有 TTL / clear / snapshot / debug / consumed 逻辑侵入较小；  
    缺点是写入、删除、过期清理都要双份维护，内存和维护复杂度都会上升。
  - 主存储重构方案：直接把按告警类型分桶作为主存储，再提供统一的全量迭代接口；  
    优点是结构更干净，避免长期双维护；  
    缺点是改动面更大，需要同步重构当前依赖“节点总事件序列”的多处逻辑。
  在落地前还需要确认：这类查询是否真是主要热点；否则这条优化未必比 `nearest_matching` 早停或规则执行计划预编译更划算。

- [ ] 评估“静态结构匹配”和“动态告警匹配”拆分缓存  
  当前 `validate_node()` 同时做站点画像结构判断与时间窗告警判断，缓存粒度仍偏粗；  
  后续可评估先把“`site + role` 是否满足结构约束”单独做成更长生命周期的静态缓存，再把时间窗告警判断保留为动态部分。  
  优点是可以明显减少同一站点在同一 role 上被反复做 `site_rules / compound patterns` 匹配的成本，  
  尤其对当前 `transmission_rule` 这种重复校验较多的规则更可能见效。  
  缺点是需要设计稳定的 role/结构签名，避免缓存键不一致；  
  同时结构缓存和时间窗缓存拆开后，代码路径会更分层，调试时也更容易出现“结构命中但事件没命中”的双层排查成本。

- [ ] 评估减少 `_evaluate_rule()` 分支复制成本的方案  
  当前 `clone_instance_with_updates()` 在状态分叉时会复制整份 `roles` 和 `_dependencies`，  
  后续可评估更强的 copy-on-write 或更细粒度的局部复制方式，只复制当前真正被修改的 role / dependency 子集。  
  优点是规则更深、候选更多时，能明显降低实例分叉阶段的 CPU 和内存开销；  
  对 `primitive` 目标节点的多分叉场景尤其有帮助。  
  缺点是实现复杂度会上升，  
  一旦共享结构处理不好，就更容易引入“分支之间意外串状态”的隐蔽 bug；  
  同时 dependency 目前是可变嵌套结构，这块如果做局部共享，需要格外小心副作用。

- [ ] 评估把 dependency 收敛从“全量 fixpoint”改成“增量收敛”  
  当前 `_stabilize_instance_dependencies()` 每次新边扩展后都会对整份 `_dependencies` 反复全量扫描直到稳定；  
  后续可评估只从本轮受影响的 role / 节点开始向上下游传播删除，  
  或维护更直接的支撑计数，在节点失去最后一个支撑时再触发局部回传。  
  优点是有机会显著减少每次分叉后的重复全表扫描，  
  尤其在规则边更多、依赖关系更复杂时，理论收益会比较明显。  
  缺点是实现门槛更高，  
  当前这套全量收敛虽然偏重，但语义直观、正确性更容易验证；  
  一旦改成增量传播，需要重新严格证明不会漏掉跨层回传或收敛顺序问题。

- [ ] 评估预编译“站点是否满足 role 结构”的静态候选表  
  当前 `validate_node()` 每次评估都会重复做 `site_rules / compound patterns` 这类站点结构判断；  
  后续可评估把这层静态条件提前预编译成：
  - `(site, rule, role) -> 是否满足结构`
  - 或 `rule/role -> 满足结构的候选站点集合`
  以减少同一站点在同一 role 上被反复做画像匹配的成本。  
  优点是结构判断天然与时间窗无关，适合做长生命周期缓存；  
  对当前 `transmission_rule` 这类会大量重复校验相同站点结构的规则，预期收益较稳定。  
  缺点是需要设计稳定的缓存 key / role 签名，  
  并确认规则配置变更后如何同步失效；  
  同时如果后续 role 语义里混入更多动态条件，这层“纯静态候选表”的边界也要重新划清。

## CaLiG / CSM 启发的匹配提效方向

- [ ] 低索引开销：按更新告警锚定局部匹配顺序  
  借鉴 CaLiG 从更新边出发做 `searchMatch` 的思路，在当前引擎里从本次触发 / 新增告警对应的 `role + site` 出发，优先扩展候选少、约束强的邻接 role。  
  预期收益：减少无效分支展开，尤其是规则图里存在多个可扩展方向时。  
  索引开销：很低，主要依赖已有规则执行计划、候选数和 support count，不需要新增长期动态索引。  
  风险：需要确保不改变 `trigger_role`、`match_mode=ALL`、`optional` 和 dependency 收敛语义。

- [ ] 低到中等索引开销：KSS 风格 kernel / shell 分层  
  借鉴 CaLiG 将查询拆成 kernel 和 shell 的做法，把决定规则成立的核心 role 先完成匹配，把可选上下文、下挂 compound、批量吸收节点放到 shell 阶段做集合 join / 批量补充。  
  预期收益：减少 compound/context-heavy 规则里的递归 backtracking 和实例分叉。  
  索引开销：低到中等，主要是规则执行计划层面的分层信息和局部候选集合，不需要维护全局动态图索引。  
  风险：需要逐类确认 primitive / compound / optional role 的输出归属、`hide_if_no_alarms`、result constraints 不被改变。

- [ ] 中等索引开销：增强 batch 内 support 缓存  
  当前已有 `support_cache` / `support_count_cache`，后续可进一步把同一批评估里的 `(rule, role, site, neighbor_role, reference_ts/window)` 支撑关系缓存得更细，减少 `_candidate_has_required_support()` 重复扫邻居。  
  预期收益：候选多、相同站点反复作为同一 role 被验证时收益明显。  
  索引开销：中等，但可以先限定为 eval-batch 内缓存，避免长期一致性维护。  
  风险：缓存 key 必须包含 `rule`、`role`、窗口、reference ts、已绑定 role 等信息，避免跨规则或跨窗口误复用。

- [ ] 较高索引开销：dynamic active role index  
  借鉴 CaLiG `LI` 的“当前候选是否仍可行”思想，维护 `(rule, role) -> 当前告警窗口内可能满足该 role 动态告警条件的站点集合`，让候选搜索同时具备静态结构过滤和动态告警过滤。  
  预期收益：对 `offline/link/power` 这类告警条件强、活跃站点远少于全量站点的规则，剪枝可能非常明显。  
  索引开销：较高，需要在告警新增、清除、TTL 过期、period cache/raw cache 两种模式下维护一致性。  
  风险：窗口语义比较敏感；`events_in_window()` 依赖 reference ts 和 edge window，不能简单用“当前 active”替代，需要设计清楚按时间窗查询或延迟失效策略。

- [ ] 高索引开销 / 暂不优先：CaLiG 式删除传播与全量 LI 维护  
  完整 CaLiG 会在边删除 / 新增时传播更新候选可行性，避免后续搜索访问已失效候选。  
  预期收益：如果动态图更新频繁且规则图复杂，理论上可显著减少后续 match generation。  
  索引开销：高，需要维护每个站点对每个 role 的局部可行性、邻接候选和反向支撑。  
  风险：当前故障汇聚还有 `pending trigger`、等待窗口、clear-delay、period cache、历史故障组合并等生命周期语义，硬搬 LI 传播容易引入一致性问题；建议只在确认 support check 仍是热点后再评估。

- [ ] 推荐落地顺序  
  优先从“不改变业务生命周期、索引开销低”的方向开始：  
  `按更新告警锚定局部匹配顺序` -> `kernel/shell 分层` -> `batch 内 support 缓存增强` -> `dynamic active role index`。  
  暂不建议优先做完整 CaLiG `LI` 删除传播，除非 profile 明确显示现有 support check / backtracking 仍是主热点。

## MQ-Match 启发的多规则共享提效方向

- [ ] 低索引开销：规则 / 边约束签名归并  
  借鉴 MQ-Match 用 `q_labelEdge` 按标签边反查受影响 query edge 的思路，为当前规则边预编译稳定签名：  
  `(source_role_structure_signature, target_role_structure_signature, direction, hops, window, selector/path_requirements)`。  
  多条规则如果共享同一类边约束，可以复用候选遍历、role 结构过滤和部分 support check。  
  预期收益：减少相似 `data_*_adjacent_*`、`transmission_*` 规则之间重复做拓扑遍历和节点结构校验。  
  索引开销：低，主要是 rule 编译阶段的静态字典，不随告警流动态变化。  
  风险：签名必须覆盖影响语义的字段，尤其是 `candidate_selector`、`path_requirements`、`match_mode=ALL`、窗口配置，避免误把语义不同的边合并。

- [ ] 低到中等索引开销：公共 pattern fragment 识别  
  借鉴 MQ-Match 的 `commSubgraph / RemainQ`，把多条规则里相同的公共片段识别出来，例如“当前 Data 路由站点 + 邻接 Data 路由站点”这类基础结构。  
  后续每条规则只在公共片段命中后继续评估自己的剩余约束。  
  预期收益：相似规则越多、公共前缀越长，重复匹配减少越明显。  
  索引开销：低到中等，需要维护 fragment signature、fragment 内 role 映射、rule 到 fragment 的反向引用。  
  风险：当前规则存在 `trigger_role`、`optional`、`compound`、result constraints 和输出归属逻辑，公共片段必须只复用“中间匹配事实”，不能提前改变最终规则是否成立。

- [ ] 中等索引开销：batch 内公共片段 partial match 缓存  
  借鉴 MQ-Match `Auxiliary` 复用 common subgraph 中间结果的做法，在一次收割 / 一次评估批次中缓存：  
  `(fragment_signature, anchor_role, anchor_site, reference_ts/window) -> partial instances / role_mapping candidates`。  
  后续共享该 fragment 的规则直接 join 剩余约束。  
  预期收益：多条规则共享同一 trigger 附近的公共拓扑片段时，能明显减少重复 `_evaluate_rule()` 分支展开。  
  索引开销：中等，但建议先限定为 batch 内缓存，批次结束即释放，避免长期失效维护。  
  风险：缓存内容需要包含 `rule` 无关的通用信息；如果缓存了 rule-specific 的 node validation / consumed trigger 状态，容易跨规则误复用。

- [ ] 中等索引开销：多规则共享 matching order  
  MQ-Match 会为公共子图和剩余查询分别生成匹配顺序。当前每条规则已有 `rule_execution_plan`，后续可把结构相同或片段相同的规则合并生成共享扩展顺序。  
  预期收益：减少重复编译和重复排序，同时让相似规则使用一致的低分支扩展顺序。  
  索引开销：中等，主要是共享 execution plan / fragment plan 缓存。  
  风险：不同规则的 `trigger_role`、`exclusive_site_roles`、result constraints 可能不同，共享顺序只能作用于公共拓扑片段，不能替代整条规则执行计划。

- [ ] 较高索引开销 / 暂不优先：MQ-Match 风格 `self_LI / nbr_LI` 完整局部可行性维护  
  MQ-Match 通过 `self_LI` 表示数据点能匹配哪些 query 节点，通过 `nbr_LI` 表示邻接支撑关系，并在边新增 / 删除时增量维护。  
  这类结构理论上能把 support check 变成索引查表。  
  预期收益：如果大量规则共享结构且动态更新频繁，剪枝收益可能很高。  
  索引开销：较高，需要为每个站点维护 role 可行性和邻接支撑；告警清除、TTL、period cache/raw cache 都要联动更新。  
  风险：我们当前动态图主要来自告警窗口而不是拓扑边增删，直接维护 `self_LI / nbr_LI` 容易把时间窗语义复杂化；建议先做静态签名归并和 batch 内公共片段缓存。

- [ ] 推荐落地顺序  
  先做静态、低风险的多规则共享：  
  `规则/边约束签名归并` -> `公共 pattern fragment 识别` -> `batch 内公共片段 partial match 缓存` -> `多规则共享 matching order`。  
  暂不建议优先做完整 `self_LI / nbr_LI` 动态维护，除非 profile 证明多规则重复 support check 已经成为主热点。
