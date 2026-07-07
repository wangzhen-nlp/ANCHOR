"""把 match_rules_batch_stream.py 输出转换为 visualization 可加载的 JSONL。

输入文件每行是一个滑动窗口记录，其中 ``agg_alarm_groups`` 为本窗口的
二次汇聚增量输出。脚本按二次汇聚 ID 跨窗口累积：

1. 同一二次汇聚组的原始故障组全部合并到一条可视化记录；
2. 同一原始故障组跨窗口重复出现时，按告警编码 ID 去重并补齐新增告警；
3. 从 resource_buffer 加载 NE/站点拓扑，构造总览页和传播页需要的
   ``match_info / group_info / ne_info / symptoms``；
4. 输出仍为 JSONL，可直接加载到 visualization/fault_group_browser.html，
   再从总览页打开 visualization/ne_propagation_visualizer.html。

用法：
    python anchor_grouping_online/tools/build_batch_stream_visualization.py \
        batch_windows.jsonl secondary_aggregates_visualization.jsonl \
        [--resource-buffer anchor_grouping_online/resources/resource_buffer.jsonl]
"""

import argparse
import copy
import json

from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    from _script_env import ensure_package_parent

    ensure_package_parent()

from anchor_grouping_online.alarm_events.generator import to_matching_alarm
from anchor_grouping_online.resource_buffer import load_resource_buffer
from anchor_grouping_online.tools.topology_resources import RESOURCE_BUFFER_JSONL


SECONDARY_AGGREGATION_RULE = "secondary_aggregation_rule"


def _normalize_text(value):
    return str(value or "").strip()


def _format_ts(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _update_min(current, value):
    if value is None:
        return current
    return value if current is None or value < current else current


def _update_max(current, value):
    if value is None:
        return current
    return value if current is None or value > current else current


def _require_mapping(value, label, line_number):
    if not isinstance(value, dict):
        raise ValueError(f"第 {line_number} 行 {label} 必须是对象")
    return value


def _matching_alarm_signature(matching_alarm):
    return (
        matching_alarm["site_id"],
        matching_alarm["alarm_title"],
        matching_alarm["ts"],
        matching_alarm["alarm_source"],
        matching_alarm["physical_port_name"],
        matching_alarm["is_clear"],
    )


def _new_aggregate_state(agg_id):
    return {
        "agg_id": agg_id,
        "groups": {},
        "alarm_group_by_id": {},
        "windows": set(),
        "first_window_start": None,
        "last_window_end": None,
        "occurrence_count": 0,
    }


def _load_window_aggregates(input_path, ne_to_site):
    """流式读取滑窗输出，返回按首次出现顺序保存的二次汇聚状态。"""
    aggregates = {}
    window_record_count = 0

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            line = line.strip()
            if not line:
                continue
            window_record_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第 {line_number} 行 JSON 解析失败: {exc}"
                ) from exc
            _require_mapping(record, "窗口记录", line_number)
            agg_alarm_groups = _require_mapping(
                record.get("agg_alarm_groups", {}),
                "agg_alarm_groups",
                line_number,
            )
            window_start = record.get("window_start")
            window_end = record.get("window_end")

            for raw_agg_id, member_entries in agg_alarm_groups.items():
                agg_id = _normalize_text(raw_agg_id)
                if not agg_id:
                    raise ValueError(f"第 {line_number} 行存在空的二次汇聚组 ID")
                if not isinstance(member_entries, list):
                    raise ValueError(
                        f"第 {line_number} 行汇聚组 {agg_id!r} 成员必须是列表"
                    )

                state = aggregates.setdefault(agg_id, _new_aggregate_state(agg_id))
                state["occurrence_count"] += 1
                window_key = (
                    window_start,
                    window_end,
                    line_number if window_start is None and window_end is None else None,
                )
                state["windows"].add(window_key)
                state["first_window_start"] = _update_min(
                    state["first_window_start"], window_start
                )
                state["last_window_end"] = _update_max(
                    state["last_window_end"], window_end
                )

                for member_entry in member_entries:
                    _require_mapping(member_entry, "原始故障组成员", line_number)
                    for raw_group_id, generated_alarms in member_entry.items():
                        group_id = _normalize_text(raw_group_id)
                        if not group_id:
                            raise ValueError(
                                f"第 {line_number} 行汇聚组 {agg_id!r} "
                                "存在空的原始故障组 ID"
                            )
                        if not isinstance(generated_alarms, list):
                            raise ValueError(
                                f"第 {line_number} 行原始组 {group_id!r} "
                                "的告警必须是列表"
                            )
                        group_state = state["groups"].setdefault(
                            group_id,
                            {"group_id": group_id, "alarms": {}},
                        )

                        for generated_alarm in generated_alarms:
                            if not isinstance(generated_alarm, dict):
                                raise ValueError(
                                    f"第 {line_number} 行原始组 {group_id!r} "
                                    "包含非对象告警"
                                )
                            matching_alarm = to_matching_alarm(
                                generated_alarm, ne_to_site
                            )
                            if matching_alarm["is_clear"]:
                                raise ValueError(
                                    f"第 {line_number} 行原始组 {group_id!r} "
                                    "包含清除告警"
                                )
                            alarm_id = matching_alarm["alarm_id"]
                            existing_owner = state["alarm_group_by_id"].get(alarm_id)
                            if existing_owner is not None and existing_owner != group_id:
                                raise ValueError(
                                    f"第 {line_number} 行告警 {alarm_id!r} 同时属于"
                                    f" {existing_owner!r} 和 {group_id!r}"
                                )
                            state["alarm_group_by_id"][alarm_id] = group_id

                            existing_alarm = group_state["alarms"].get(alarm_id)
                            if existing_alarm is not None:
                                if (
                                    _matching_alarm_signature(existing_alarm["matching"])
                                    != _matching_alarm_signature(matching_alarm)
                                ):
                                    raise ValueError(
                                        f"第 {line_number} 行告警 {alarm_id!r} "
                                        "跨窗口内容不一致"
                                    )
                                continue
                            group_state["alarms"][alarm_id] = {
                                "generated": dict(generated_alarm),
                                "matching": matching_alarm,
                            }

    return aggregates, window_record_count


def _build_site_placeholder_ne_id(site_id):
    return f"SITE::{site_id}"


def _resolve_ne_id(matching_alarm):
    alarm_source = matching_alarm["alarm_source"]
    if alarm_source:
        return alarm_source
    site_id = matching_alarm["site_id"]
    return _build_site_placeholder_ne_id(site_id) if site_id else ""


def _split_composed_alarm_id(alarm_id):
    if "::" not in alarm_id:
        return alarm_id, ""
    return tuple(alarm_id.rsplit("::", 1))


def _build_ne_meta(ne_id, site_id, agg_id, ne_graph, site_graph):
    ne_meta = ne_graph.get(ne_id, {}) if isinstance(ne_graph, dict) else {}
    site_meta = site_graph.get(site_id, {}) if isinstance(site_graph, dict) else {}
    is_placeholder = ne_id.startswith("SITE::")
    links = ne_meta.get("link", {}) if isinstance(ne_meta, dict) else {}
    return {
        "link": copy.deepcopy(links) if isinstance(links, dict) else {},
        "group": agg_id,
        "name": ne_meta.get("name", ne_id if not is_placeholder else site_id),
        "site_id": site_id,
        "site_name": (
            ne_meta.get("site_name", "")
            or site_meta.get("site_name", "")
            or site_id
        ),
        "type": _normalize_text(ne_meta.get("type", "")).upper(),
        "network_type": _normalize_text(
            ne_meta.get("network_type", "")
        ).upper(),
        "manufacturer": _normalize_text(
            ne_meta.get("manufacturer", "")
        ).upper(),
        "running_status": ne_meta.get(
            "running_status", ne_meta.get("status", "")
        ),
        "domain": _normalize_text(ne_meta.get("domain", "")).upper(),
        "region_id": ne_meta.get("region_id", "") or site_meta.get("region_id", ""),
        "longitude": ne_meta.get("longitude", "") or site_meta.get("longitude", ""),
        "latitude": ne_meta.get("latitude", "") or site_meta.get("latitude", ""),
        "alarm": [],
    }


def _build_visualization_record(state, ne_graph, site_graph):
    agg_id = state["agg_id"]
    symptoms = []
    ne_info = {}
    all_site_ids = set()
    all_ne_ids = set()
    raw_group_summaries = []

    for group_id, group_state in state["groups"].items():
        group_site_ids = set()
        group_ne_ids = set()
        group_timestamps = []
        alarm_items = sorted(
            group_state["alarms"].values(),
            key=lambda item: (
                item["matching"]["ts"],
                item["matching"]["alarm_id"],
            ),
        )
        for alarm_item in alarm_items:
            generated_alarm = alarm_item["generated"]
            matching_alarm = alarm_item["matching"]
            alarm_id = matching_alarm["alarm_id"]
            event_id, occurrence_uuid = _split_composed_alarm_id(alarm_id)
            site_id = matching_alarm["site_id"]
            ne_id = _resolve_ne_id(matching_alarm)
            ts = matching_alarm["ts"]
            group_timestamps.append(ts)
            if site_id:
                group_site_ids.add(site_id)
                all_site_ids.add(site_id)
            if ne_id:
                group_ne_ids.add(ne_id)
                all_ne_ids.add(ne_id)

            ne_meta = ne_graph.get(ne_id, {}) if ne_id else {}
            domain = _normalize_text(ne_meta.get("domain", "")).upper()
            symptom = {
                "node": site_id,
                "alarm": matching_alarm["alarm_title"],
                "ts": ts,
                "eid": alarm_id,
                "alarm_id": alarm_id,
                "event_id": event_id,
                "occurrence_uuid": occurrence_uuid,
                "alarm_source": matching_alarm["alarm_source"] or ne_id,
                "domain": domain,
                "matched_role": "secondary_aggregate_member",
                "工单号": "",
                "故障组ID": group_id,
                "来源故障组UUID": f"alarm-{group_id}",
                "告警清除时间": "",
                "物理端口名称": matching_alarm["physical_port_name"],
            }
            symptoms.append(symptom)

            if ne_id:
                if ne_id not in ne_info:
                    ne_info[ne_id] = _build_ne_meta(
                        ne_id, site_id, agg_id, ne_graph, site_graph
                    )
                ne_info[ne_id]["alarm"].append({
                    "alarm_id": alarm_id,
                    "event_id": event_id,
                    "occurrence_uuid": occurrence_uuid,
                    "alarm_type": matching_alarm["alarm_title"],
                    "alarm_time": _format_ts(ts),
                    "alarm_clear_time": "",
                    "domain": domain,
                    "site_id": site_id,
                    "site_name": ne_info[ne_id]["site_name"],
                    "matched_role": "secondary_aggregate_member",
                    "工单号": "",
                    "故障组ID": group_id,
                    "来源故障组UUID": f"alarm-{group_id}",
                    "物理端口名称": generated_alarm.get(
                        "物理端口名称", ""
                    ),
                    "ts": ts,
                })

        raw_group_summaries.append({
            "group_id": group_id,
            "alarm_count": len(alarm_items),
            "site_list": sorted(group_site_ids),
            "ne_list": sorted(group_ne_ids),
            "first_alarm_ts": min(group_timestamps) if group_timestamps else None,
            "last_alarm_ts": max(group_timestamps) if group_timestamps else None,
        })

    symptoms.sort(key=lambda item: (item["ts"], item["eid"]))
    for ne_meta in ne_info.values():
        ne_meta["alarm"].sort(
            key=lambda item: (item.get("ts", 0), item.get("alarm_id", ""))
        )

    timestamps = [symptom["ts"] for symptom in symptoms]
    anchor_ts = min(timestamps) if timestamps else None
    last_alarm_ts = max(timestamps) if timestamps else None
    role_mapping = {"secondary_aggregate_member": sorted(all_site_ids)}
    match_info = {
        "uuid": agg_id,
        "rule": SECONDARY_AGGREGATION_RULE,
        "merged_rules": [SECONDARY_AGGREGATION_RULE],
        "related_group_uuids": [],
        "inferred_roots": {},
        "role_mapping": role_mapping,
    }
    return {
        "uuid": agg_id,
        "rule": SECONDARY_AGGREGATION_RULE,
        "merged_rules": [SECONDARY_AGGREGATION_RULE],
        "related_group_uuids": [],
        "inferred_roots": {},
        "role_mapping": role_mapping,
        "match_info": match_info,
        "group_info": {
            agg_id: {
                "site_list": sorted(all_site_ids),
                "ne_list": sorted(all_ne_ids),
            }
        },
        "ne_info": ne_info,
        "symptoms": symptoms,
        "group_anchor_ts": anchor_ts,
        "group_anchor_time": _format_ts(anchor_ts),
        "group_last_ts": last_alarm_ts,
        "group_last_time": _format_ts(last_alarm_ts),
        "secondary_aggregation": {
            "aggregate_id": agg_id,
            "raw_group_count": len(raw_group_summaries),
            "alarm_count": len(symptoms),
            "window_count": len(state["windows"]),
            "aggregate_occurrence_count": state["occurrence_count"],
            "first_window_start": state["first_window_start"],
            "last_window_end": state["last_window_end"],
            "raw_fault_groups": raw_group_summaries,
        },
    }


def build_visualization_jsonl(
    input_path,
    output_path,
    resource_buffer=RESOURCE_BUFFER_JSONL,
):
    """转换滑窗二次汇聚输出，返回写出统计。"""
    input_resolved = Path(input_path).expanduser().resolve()
    output_resolved = Path(output_path).expanduser().resolve()
    if input_resolved == output_resolved:
        raise ValueError("输入文件和输出文件不能是同一路径")

    print(f"加载可视化拓扑资源: {resource_buffer}", flush=True)
    resources = load_resource_buffer(
        resource_buffer,
        wanted_types=("ne_graph", "site_graph"),
    )
    ne_graph = resources["ne_graph"]
    site_graph = resources["site_graph"]
    # 告警不携带站点字段：解析时用 告警源 在网元拓扑中反查站点。
    ne_to_site = {
        ne_id: str(ne_info.get("site_id", "")).strip()
        for ne_id, ne_info in ne_graph.items()
        if str(ne_info.get("site_id", "")).strip()
    }

    print(f"读取滑窗二次汇聚输出: {input_path}", flush=True)
    aggregates, window_record_count = _load_window_aggregates(
        input_path, ne_to_site
    )

    aggregate_states = sorted(
        aggregates.values(),
        key=lambda state: (
            state["first_window_start"] is None,
            state["first_window_start"]
            if state["first_window_start"] is not None
            else 0,
            state["agg_id"],
        ),
    )
    raw_group_count = 0
    alarm_count = 0
    with open(output_path, "w", encoding="utf-8") as output_file:
        for state in aggregate_states:
            visual_record = _build_visualization_record(
                state, ne_graph, site_graph
            )
            raw_group_count += visual_record["secondary_aggregation"][
                "raw_group_count"
            ]
            alarm_count += visual_record["secondary_aggregation"]["alarm_count"]
            output_file.write(
                json.dumps(visual_record, ensure_ascii=False) + "\n"
            )

    stats = {
        "window_record_count": window_record_count,
        "aggregate_count": len(aggregate_states),
        "raw_group_count": raw_group_count,
        "alarm_count": alarm_count,
    }
    print(
        "完成："
        f"{stats['window_record_count']} 个窗口 -> "
        f"{stats['aggregate_count']} 个二次汇聚可视化组，"
        f"包含 {stats['raw_group_count']} 个原始故障组 / "
        f"{stats['alarm_count']} 条去重告警；输出 {output_path}",
        flush=True,
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="match_rules_batch_stream.py 输出 JSONL")
    parser.add_argument("output", help="visualization 可加载的 JSONL")
    parser.add_argument(
        "--resource-buffer",
        default=RESOURCE_BUFFER_JSONL,
        help="build_resource_buffer.py 生成的资源缓冲文件",
    )
    args = parser.parse_args()
    build_visualization_jsonl(
        args.input,
        args.output,
        resource_buffer=args.resource_buffer,
    )


if __name__ == "__main__":
    main()
