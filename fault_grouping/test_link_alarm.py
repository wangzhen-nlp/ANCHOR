import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fault_grouping.temporal_engine.evaluator import link_alarm_points_to_site
from topology_tools.link_peer_index import (
    LinkAlarmEndpoints,
    build_peer_index_from_sys_link,
    resolve_link_alarm_endpoints,
    resolve_link_alarm_endpoints_from_peer_index,
)
from fault_grouping.rule_config import link_rule
from fault_grouping.temporal_engine.engine import TemporalGraphEngine
from alarm_tools.analyze_link_alarm_peer_coverage import analyze_link_alarm_peer_coverage


def _write_sys_link_csv(path):
    fieldnames = [
        "nativeId",
        "last_Modified",
        "a_end_ne_nativeId",
        "z_end_ne_nativeId",
        "a_end_port_name",
        "z_end_port_name",
        "a_end_port_ip",
        "z_end_port_ip",
        "a_end_ne_manager_name",
        "z_end_ne_manager_name",
    ]
    rows = [
        {
            "nativeId": "link-1",
            "last_Modified": "1780272000000",
            "a_end_ne_nativeId": "P1",
            "z_end_ne_nativeId": "C1",
            "a_end_port_name": "PORT-TO-CHILD",
            "z_end_port_name": "PORT-TO-PARENT",
            "a_end_port_ip": "10.0.0.1",
            "z_end_port_ip": "10.0.0.2",
            "a_end_ne_manager_name": "mgr-a",
            "z_end_ne_manager_name": "mgr-z",
        },
        {
            "nativeId": "link-2",
            "last_Modified": "1780272000000",
            "a_end_ne_nativeId": "P1",
            "z_end_ne_nativeId": "O1",
            "a_end_port_name": "PORT-TO-OTHER",
            "z_end_port_name": "PORT-TO-PARENT",
            "a_end_port_ip": "10.0.1.1",
            "z_end_port_ip": "10.0.1.2",
            "a_end_ne_manager_name": "mgr-a",
            "z_end_ne_manager_name": "mgr-o",
        },
    ]
    with open(path, "w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_peer_index():
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "sys_link.csv"
        _write_sys_link_csv(csv_path)
        return build_peer_index_from_sys_link(str(csv_path))


def test_resolve_link_alarm_endpoints_uses_alarm_source_and_physical_port():
    peer_index = _build_peer_index()

    endpoints = resolve_link_alarm_endpoints_from_peer_index(
        {"告警源": "P1", "物理端口": "PORT-TO-CHILD"},
        peer_index=peer_index,
    )

    assert endpoints == LinkAlarmEndpoints(
        local_ne="P1",
        local_port="PORT-TO-CHILD",
        remote_ne="C1",
        remote_port="PORT-TO-PARENT",
    )
    assert link_alarm_points_to_site(
        {"告警源": "P1", "物理端口": "PORT-TO-CHILD"},
        "CHILD",
        {"C1": "CHILD"},
        peer_index=peer_index,
    ) is True
    assert link_alarm_points_to_site(
        {"告警源": "P1", "物理端口": "PORT-TO-OTHER"},
        "CHILD",
        {"O1": "OTHER"},
        peer_index=peer_index,
    ) is False
    assert resolve_link_alarm_endpoints({"告警源": "P1"}).remote_ne == ""


def test_process_event_preserves_raw_alarm_payload_for_later_link_endpoint_parsing():
    engine = TemporalGraphEngine({}, {}, {})
    alarm_payload = {
        "告警源": "NE-A",
        "告警标题": "Link Down",
        "待补字段": "后续用于解析对端设备",
    }

    engine.process_event(
        "SITE-A",
        "Link Down",
        10,
        "event-1",
        "00000000-0000-0000-0000-000000000001",
        alarm_source="NE-A",
        alarm_payload=alarm_payload,
    )

    cached_event = engine.event_cache["SITE-A"][0]
    assert cached_event["alarm_payload"] == alarm_payload


def _build_link_rule_engine(peer_index):
    ne_graph = {
        "P1": {"site_id": "PARENT", "domain": "Transmission", "link": {"C1": {"MW": "->"}}},
        "C1": {"site_id": "CHILD", "domain": "Transmission", "link": {"P1": {"MW": "<-"}}},
        "O1": {"site_id": "OTHER", "domain": "Transmission", "link": {"P1": {"MW": "<-"}}},
    }
    return TemporalGraphEngine(
        {"PARENT": ["CHILD"], "CHILD": []},
        {"link_rule": link_rule},
        {
            "PARENT": {"Transmission": 1},
            "CHILD": {"Transmission": 1},
            "OTHER": {"Transmission": 1},
        },
        alarm_source_domain_map={
            ne_id: ne_info["domain"]
            for ne_id, ne_info in ne_graph.items()
        },
        aggregation_wait_sec=0,
        ne_graph_data=ne_graph,
        site_to_ne_ids={
            "PARENT": ("P1",),
            "CHILD": ("C1",),
            "OTHER": ("O1",),
        },
        link_peer_index=peer_index,
    )


def _run_link_rule_with_port(port_name):
    engine = _build_link_rule_engine(_build_peer_index())
    engine.process_event(
        "PARENT",
        "Link Down",
        10,
        f"parent-{port_name}",
        "00000000-0000-0000-0000-000000000001",
        alarm_source="P1",
        alarm_payload={"告警源": "P1", "物理端口": port_name},
    )
    return engine.process_event(
        "CHILD",
        "BTS Down",
        20,
        "child-offline",
        "00000000-0000-0000-0000-000000000002",
        alarm_source="C1",
        alarm_payload={"告警源": "C1"},
        collect_matches=True,
    )


def test_link_rule_requires_alarm_physical_port_to_point_to_child_site():
    assert len(_run_link_rule_with_port("PORT-TO-OTHER")) == 0
    assert len(_run_link_rule_with_port("PORT-TO-CHILD")) == 1


def test_analyze_link_alarm_peer_coverage_requires_remote_ne_in_ne_graph():
    peer_index = _build_peer_index()
    with tempfile.TemporaryDirectory() as temp_dir:
        alarms_path = Path(temp_dir) / "alarms.jsonl"
        alarms = [
            {"告警标题": "Link Down", "告警源": "P1", "物理端口": "PORT-TO-CHILD"},
            {"告警标题": "Link Down", "告警源": "P1", "物理端口": "PORT-TO-OTHER"},
            {"告警标题": "BTS Down", "告警源": "C1", "物理端口": ""},
        ]
        with open(alarms_path, "w", encoding="utf-8") as fw:
            for alarm in alarms:
                fw.write(json.dumps(alarm, ensure_ascii=False) + "\n")

        result = analyze_link_alarm_peer_coverage(
            str(alarms_path),
            peer_index,
            {"C1"},
            show_progress=False,
        )

    assert result["total_link_alarms"] == 2
    assert result["found_peer_in_ne_graph"] == 1
    assert result["ratio"] == 0.5


def test_analyze_link_alarm_peer_coverage_debug_prints_missing_peer_and_exits():
    peer_index = _build_peer_index()
    with tempfile.TemporaryDirectory() as temp_dir:
        alarms_path = Path(temp_dir) / "alarms.jsonl"
        alarm = {
            "告警标题": "Link Down",
            "告警源": "P1",
            "物理端口": "PORT-NOT-IN-SYS-LINK",
            "自定义字段": "debug-visible",
        }
        alarms_path.write_text(
            json.dumps(alarm, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        output = io.StringIO()
        try:
            with redirect_stdout(output):
                analyze_link_alarm_peer_coverage(
                    str(alarms_path),
                    peer_index,
                    {"C1"},
                    show_progress=False,
                    debug_missing_peer=True,
                )
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("debug_missing_peer should exit on the first missing peer")

    debug_text = output.getvalue()
    assert "未找到 link 告警对端设备" in debug_text
    assert "告警源=P1" in debug_text
    assert "物理端口=PORT-NOT-IN-SYS-LINK" in debug_text
    assert "自定义字段" in debug_text
    assert "debug-visible" in debug_text


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    for name, test_func in sorted(globals().items()):
        if name.startswith("test_") and callable(test_func):
            suite.addTest(unittest.FunctionTestCase(test_func, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
