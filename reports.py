from datetime import datetime


def generate_incident_report(match_result):
    """
    将引擎输出的字典，格式化为带有角色映射的结构化工单
    """
    rule_name = match_result.get("rule", "UNKNOWN_RULE")
    merged_rules = match_result.get("merged_rules", [])
    group_uuid = match_result.get("uuid", "")
    related_group_uuids = match_result.get("related_group_uuids", [])
    inferred_roots = match_result.get("inferred_roots", {})
    role_mapping = match_result.get("role_mapping", {})
    symptoms = match_result.get("symptoms", [])

    timestamps = [s["ts"] for s in symptoms]
    start_time = datetime.fromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M:%S') if timestamps else "静默断站 (无告警事件)"

    impact_summary = {}
    for sym in symptoms:
        alarm = sym["alarm"]
        impact_summary[alarm] = impact_summary.get(alarm, 0) + 1

    print("\n" + "=" * 60)
    print("🚨 【AIOps 根因定位报告】")
    print("=" * 60)
    print(f"📍 命中规则 : {rule_name}")
    if len(merged_rules) > 1 and rule_name != " + ".join(merged_rules):
        print(f"🧬 合并规则 : {merged_rules}")
    if group_uuid:
        print(f"🆔 故障组ID : {group_uuid}")
    if related_group_uuids:
        print(f"🔗 关联故障组 : {related_group_uuids}")

    root_strs = []
    for r_role, r_nodes in inferred_roots.items():
        if r_nodes:
            root_strs.append(f"{r_role} -> {r_nodes}")
    print(f"🎯 推断根因 : {', '.join(root_strs)}")
    print(f"🕒 爆发时间 : {start_time}")
    print(f"💥 衍生告警 : 共 {len(symptoms)} 条")

    print("-" * 60)
    print("🧩 模式角色映射详情 (Topology Mapping):")
    for role, nodes in role_mapping.items():
        node_str = ", ".join([str(n) for n in nodes])
        print(f"   🔹 [{role:<25}] 对应实体: {node_str}")

    print("-" * 60)
    print("📝 详细证据链 (Evidence Symptoms):")

    symptoms_sorted = sorted(symptoms, key=lambda x: x["ts"])
    for sym in symptoms_sorted:
        ts_str = datetime.fromtimestamp(sym["ts"]).strftime('%H:%M:%S')
        role_tag = f"[{sym.get('matched_role', 'UNKNOWN')}]"
        alarm_source = sym.get("alarm_source", "")
        device_str = alarm_source if alarm_source else "-"
        print(f"   [{ts_str}] {role_tag:<28} | 节点: {sym['node']:<15} | 设备: {device_str:<20} | 告警: {sym['alarm']}")

    print("=" * 60 + "\n")

    return match_result
