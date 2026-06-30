import collections


class TemporalGraphEngineDependencyMixin:
    def _record_instance_dependency(self, inst, curr_role, tgt_role, curr_support_targets):
        """记录一条已处理边上的节点依赖关系，供后续收敛裁剪使用。"""
        if not curr_support_targets:
            return

        src_to_dst = {
            curr_node: set(target_nodes)
            for curr_node, target_nodes in curr_support_targets.items()
            if target_nodes
        }
        if not src_to_dst:
            return

        dst_to_src = collections.defaultdict(set)
        for curr_node, target_nodes in src_to_dst.items():
            for target_node in target_nodes:
                dst_to_src[target_node].add(curr_node)

        dependencies = inst.setdefault("_dependencies", {})
        dependencies[(curr_role, tgt_role)] = {
            "src_to_dst": src_to_dst,
            "dst_to_src": {target_node: set(src_nodes) for target_node, src_nodes in dst_to_src.items()},
        }
        dependencies[(tgt_role, curr_role)] = {
            "src_to_dst": {
                target_node: set(src_nodes)
                for target_node, src_nodes in dst_to_src.items()
            },
            "dst_to_src": {
                curr_node: set(target_nodes)
                for curr_node, target_nodes in src_to_dst.items()
            },
        }

    def _stabilize_instance_dependencies(self, inst, nodes_cfg):
        """基于已记录的边依赖做收敛裁剪，把深层失效回传到上下游角色。"""
        dependencies = inst.get("_dependencies")
        if not dependencies:
            return inst

        stabilized_inst = dict(inst)
        stabilized_inst["roles"] = {
            role: {
                "nodes": dict(role_state["nodes"]),
                "checked": role_state["checked"]
            }
            for role, role_state in inst.get("roles", {}).items()
        }
        stabilized_roles = stabilized_inst["roles"]

        while True:
            changed = False

            for (src_role, dst_role), dep in dependencies.items():
                src_state = stabilized_roles.get(src_role)
                dst_state = stabilized_roles.get(dst_role)
                if src_state is None or dst_state is None:
                    continue

                live_src_nodes = src_state["nodes"]
                live_dst_nodes = dst_state["nodes"]
                live_dst_set = set(live_dst_nodes)
                live_src_set = set(live_src_nodes)

                kept_src_nodes = {
                    src_node: live_src_nodes[src_node]
                    for src_node in live_src_nodes
                    if dep.get("src_to_dst", {}).get(src_node, set()) & live_dst_set
                }
                if len(kept_src_nodes) != len(live_src_nodes):
                    src_state["nodes"] = kept_src_nodes
                    live_src_nodes = kept_src_nodes
                    live_src_set = set(kept_src_nodes)
                    changed = True

                kept_dst_nodes = {
                    dst_node: live_dst_nodes[dst_node]
                    for dst_node in live_dst_nodes
                    if dep.get("dst_to_src", {}).get(dst_node, set()) & live_src_set
                }
                if len(kept_dst_nodes) != len(live_dst_nodes):
                    dst_state["nodes"] = kept_dst_nodes
                    changed = True

            for role, role_state in stabilized_roles.items():
                if role_state["checked"] and len(role_state["nodes"]) < nodes_cfg.get(role, {}).get("min_count", 1):
                    return None

            if not changed:
                return stabilized_inst
