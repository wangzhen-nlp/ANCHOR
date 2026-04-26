# 仓库结构说明

当前仓库按功能大致分成下面几块：

## `alarm_tools/`

和告警输入流、告警类型、基础告警处理直接相关的模块。

- `alarm_inputs.py`：读取告警文件、流式遍历、NE 到站点映射
- `alarm_types.py`：告警类型常量
- `progress_utils.py`：命令行进度显示
- `extract_*` / `get_alarm_time_range.py`：基础告警字段提取工具

## `fault_grouping/`

故障组匹配主链路。

- `match_rules.py`：按规则跑故障组匹配
- `rule_config.py`：规则配置
- `alarm_events/`：告警事件读取、流式回放和已排序缓存
- `matching/`：匹配运行时、debug、输出构建和报告生成
- `temporal_engine/`：时序图引擎实现包，包含主引擎、遍历、输出、告警时段缓存、约束和工具函数
- `tools/`：与主匹配流程解耦的故障组分析、过滤、提取、后处理和输入预处理脚本
- `node_rule_helper.py` / `emitted_group_store.py`：配套能力

## `ticket_recall/`

围绕工单站点召回、上界分析、group 输出评测、case 导出的脚本。

- `evaluation/`
  - 纯评测和指标计算脚本
  - `compute_ticket_site_recall*.py`
  - `compute_group_output_ticket_recall*.py`
  - `compute_ultimate_group_alarm_group_metrics.py`
  - `compute_filtered_real_recall.py`
- `ticket_recall_utils.py`
- `filter_incident_tickets.py`
- `extract_alarm_group_reference_json.py`
- `extract_ultimate_fault_groups.py`

## `ticket_resources/`

脚本默认依赖的工单资源目录。

- `Incident Ticket_20260201-20260318.xlsx`

## `alarm_resources/`

脚本默认依赖的告警资源目录。

- `CROSS_alarm_propagation.xlsx`

## `topology_tools/`

NE / Site 拓扑提取和拓扑分析工具。

- `extract_ne_graph.py`
- `extract_site_graph.py`
- `extract_transmission_isolated_sites.py`
- `build_ne_propagation.py`

## `topology_resources/`

脚本默认依赖的资源目录。

- `ne_graph.json`
- `site_graph.json`
- `site_graph_by_ne.json`
- `site_device_counts.json`
- `sys_link_1231.jsonl`
- `SYS_NE_0306/`
- `SYS_SITE_0306/`

## `ne_link_learning/`

NE 对级别的拓扑补边学习包，同时承载通用的数据集拆分、训练、评估逻辑。

- `core.py`
- `build_link_dataset.py`
- `rank_link_candidates.py`
- `split_dataset.py`
- `train_model.py`
- `test_model.py`

## `site_link_learning/`

站点对级别的拓扑补边学习包。

- `core.py`
- `build_link_dataset.py`
- `rank_link_candidates.py`
- `split_dataset.py`
- `train_model.py`
- `test_model.py`
- `feature_reference.md`

## `misc_tools/`

不直接属于主链路的小工具。

- `count_ids.py`
- `excel_to_json_by_key.py`

## `visualization/`

前端页面和可视化资源。

- `fault_group_browser.html`
- `fault_group_detail.html`
- `reference_record_viewer.html`
- `ne_propagation_visualizer.html`

## `docs/`

文档和说明。

- `note.md`
- `todo.md`
- `repo_structure.md`

## 调用方式

不再保留根目录同名脚本入口。

建议直接按包内路径调用脚本，例如：

- `python fault_grouping/match_rules.py ...`
- `python ticket_recall/evaluation/compute_group_output_ticket_recall.py ...`
- `python ne_link_learning/train_model.py ...`
- `python site_link_learning/train_model.py ...`

后续新增功能建议优先直接放到对应包内，而不是再回到根目录铺脚本。
