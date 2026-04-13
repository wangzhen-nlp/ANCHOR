# topology_resources

这里集中放脚本默认依赖的资源文件和目录，后续直接把真实数据放到这里即可。

当前默认约定包括：

- `ne_graph.json`
- `site_graph.json`
- `site_graph_by_ne.json`
- `site_device_counts.json`
- `sys_link_1231.jsonl`
- `SYS_NE_0306/`
- `SYS_SITE_0306/`

说明：

- 这些文件目前只是默认引用位置，仓库里不强制提交实际数据。
- 相关脚本已经统一改成优先使用这里的默认路径。
- 如果 `topology_resources/` 下没有对应文件，但仓库根目录下仍有旧同名资源，当前也会自动兼容旧位置。
- 告警规则类默认资源已拆到 `alarm_resources/`。
- 工单输入类默认资源已拆到 `ticket_resources/`。
- 仍然可以通过命令行参数覆盖为其它路径。
