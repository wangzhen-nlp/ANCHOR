# resources

`anchor_grouping_online` 的资源目录。所有工具与 matcher 默认依赖的资源文件、
数据目录都集中放在这里，由 [`tools/topology_resources.py`](../tools/topology_resources.py)
统一解析（`RESOURCE_DIR = anchor_grouping_online/resources`）。

资源路径固定在本目录，把真实数据直接放到这里即可。

## 默认约定的资源

匹配器读取（运行时由上游产物或部署填入）：

- `resource_buffer.jsonl` —— 包含 `ne_graph`、`site_chains`、
  `link_peer_index` 等资源的缓冲文件

资源缓冲构建工具读取的原始输入目录（仓库里只放占位 `.gitkeep`，不强制提交真实数据）：

- `SYS_NE_20260525/`
- `SYS_SITE_20260525/`
- `SYS_LINK_20260525/`

## 说明

- 仓库里这些只是默认引用位置，不强制提交实际数据；部署时把对应文件放进本目录。
- 所有脚本都可用命令行参数覆盖为其它路径。
- `build_resource_buffer.py` 默认把缓冲产物（`resource_buffer.jsonl`）也写到本目录。
