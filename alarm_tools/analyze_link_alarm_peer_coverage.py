import argparse
import json

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from alarm_tools.alarm_inputs import stream_alarm_inputs
from alarm_tools.alarm_types import LINK_ALARMS
from topology_resources import LINK_PEER_INDEX_JSON, NE_GRAPH_JSON, resource_display
from topology_tools.link_peer_index import (
    load_peer_index,
    resolve_link_alarm_endpoints_from_peer_index,
)


def load_ne_ids(ne_graph_path):
    with open(ne_graph_path, "r", encoding="utf-8") as fr:
        ne_graph_data = json.load(fr)
    return {
        str(ne_id or "").strip().upper()
        for ne_id in ne_graph_data.keys()
        if str(ne_id or "").strip()
    }


def analyze_link_alarm_peer_coverage(alarms_input, peer_index, ne_ids, show_progress=True):
    total_link_alarms = 0
    found_peer_in_ne_graph = 0

    for alarm in stream_alarm_inputs(alarms_input, show_progress=show_progress):
        if alarm.get("告警标题", "") not in LINK_ALARMS:
            continue
        total_link_alarms += 1
        endpoints = resolve_link_alarm_endpoints_from_peer_index(
            alarm,
            peer_index=peer_index,
            alarm_source=alarm.get("告警源", ""),
        )
        if endpoints.remote_ne and endpoints.remote_ne in ne_ids:
            found_peer_in_ne_graph += 1

    ratio = found_peer_in_ne_graph / total_link_alarms if total_link_alarms else 0.0
    return {
        "total_link_alarms": total_link_alarms,
        "found_peer_in_ne_graph": found_peer_in_ne_graph,
        "ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="统计 link 告警通过 peer-index 找到 ne_graph 对端设备的覆盖率"
    )
    parser.add_argument("alarms", help="告警输入，支持 JSONL/CSV/ZIP/目录，格式同 match_rules.py")
    parser.add_argument(
        "--link-peer-index",
        default=LINK_PEER_INDEX_JSON,
        help=f"设备端口对端索引 JSON，默认: {resource_display('link_peer_index.json')}",
    )
    parser.add_argument(
        "--ne-graph",
        default=NE_GRAPH_JSON,
        help=f"ne_graph.json 文件，默认: {resource_display('ne_graph.json')}",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭读取进度显示",
    )
    args = parser.parse_args()

    print(f"加载 peer-index: {args.link_peer_index}")
    peer_index = load_peer_index(args.link_peer_index)
    print(f"peer-index 记录数: {len(peer_index)}")
    print(f"加载 ne_graph: {args.ne_graph}")
    ne_ids = load_ne_ids(args.ne_graph)
    print(f"ne_graph NE 数: {len(ne_ids)}")

    result = analyze_link_alarm_peer_coverage(
        args.alarms,
        peer_index,
        ne_ids,
        show_progress=not args.no_progress,
    )
    print(f"link 告警总条数: {result['total_link_alarms']}")
    print(f"可找到 ne_graph 对端设备条数: {result['found_peer_in_ne_graph']}")
    print(f"比例: {result['ratio']:.6f}")


if __name__ == "__main__":
    main()
