# resources

`fault_grouping_official` 的自包含资源目录。所有工具与 matcher 默认依赖的资源文件、
数据目录都集中放在这里，由 [`tools/topology_resources.py`](../tools/topology_resources.py)
统一解析（`RESOURCE_DIR = fault_grouping_official/resources`）。

本包不依赖仓库其它目录：不再回退到仓库根目录或仓库级 `topology_resources/` 的旧位置。
把真实数据直接放到这里即可。

## 默认约定的资源

匹配/链路工具读取（运行时由上游产物或部署填入）：

- `ne_graph.json` —— NE 邻接图
- `site_chains.json` —— `generate_site_chains.py` 预计算的站点链路
- `link_peer_index.json` —— 设备端口对端索引
- `sys_link_1231.jsonl` —— 链路记录

资源缓冲构建工具读取的原始输入目录（仓库里只放占位 `.gitkeep`，不强制提交真实数据）：

- `SYS_NE_0306/`
- `SYS_SITE_0306/`

## 说明

- 仓库里这些只是默认引用位置，不强制提交实际数据；部署时把对应文件放进本目录。
- 所有脚本都可用命令行参数覆盖为其它路径。
- `build_resource_buffer.py` 默认把缓冲产物（`resource_buffer.jsonl`）也写到本目录。
