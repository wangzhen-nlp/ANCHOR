import json
import tempfile
import unittest
from pathlib import Path

from fault_grouping.site_topology import build_site_to_ne_ids
from microwave_topic.complete_group_topology import complete_group_topology
from microwave_topic.complete_group_topology_from_resource_buffer import (
    complete_groups_from_resource_buffer,
)


def _write_resource_buffer(path, resources):
    with open(path, "w", encoding="utf-8") as file_obj:
        for resource_type, data in resources:
            file_obj.write('{"resource_type":')
            json.dump(resource_type, file_obj, ensure_ascii=False)
            file_obj.write(',"data":')
            json.dump(data, file_obj, ensure_ascii=False, separators=(",", ":"))
            file_obj.write("}\n")


class CompleteGroupTopologyTest(unittest.TestCase):
    def test_expands_all_devices_at_non_offline_alarm_site(self):
        ne_graph = {
            "NE-ALARM": {
                "site_id": "SITE-1",
                "domain": "MICROWAVE",
                "link": {},
            },
            "NE-PEER": {
                "site_id": "SITE-1",
                "domain": "DATA",
                "link": {},
            },
        }
        site_graph = {"SITE-1": {"site_name": "Site 1"}}
        group = {
            "故障组ID": "GROUP-1",
            "alarms": [
                {
                    "告警源": "NE-ALARM",
                    "告警标题": "普通设备告警",
                }
            ],
        }

        completed = complete_group_topology(
            group,
            ne_graph,
            site_graph,
            build_site_to_ne_ids(ne_graph),
            site_chain_index={},
        )

        self.assertEqual(set(completed["ne_info"]), {"NE-ALARM", "NE-PEER"})
        self.assertEqual(completed["ne_info"]["NE-PEER"]["alarm"], [])
        self.assertTrue(completed["ne_info"]["NE-PEER"]["topology_added"])
        self.assertEqual(completed["topology_completion"]["selected_site_ids"], ["SITE-1"])
        self.assertEqual(completed["topology_completion"]["added_ne_ids"], ["NE-PEER"])

    def test_copies_site_distance_to_device_link_metadata(self):
        ne_graph = {
            "NE-A": {
                "site_id": "SITE-A",
                "domain": "MICROWAVE",
                "link": {"NE-B": {"MW": "->"}},
            },
            "NE-B": {
                "site_id": "SITE-B",
                "domain": "MICROWAVE",
                "link": {"NE-A": {"MW": "<-"}},
            },
        }
        site_graph = {
            "SITE-A": {
                "site_name": "Site A",
                "link_distance_km": {"SITE-B": 12.345},
            },
            "SITE-B": {"site_name": "Site B"},
        }
        group = {
            "故障组ID": "GROUP-DISTANCE",
            "alarms": [
                {"告警源": "NE-A", "告警标题": "普通设备告警"},
                {"告警源": "NE-B", "告警标题": "普通设备告警"},
            ],
        }

        completed = complete_group_topology(
            group,
            ne_graph,
            site_graph,
            build_site_to_ne_ids(ne_graph),
            site_chain_index={},
        )

        self.assertEqual(completed["ne_info"]["NE-A"]["link"]["NE-B"]["distance"], 12.35)
        self.assertEqual(completed["ne_info"]["NE-B"]["link"]["NE-A"]["distance"], 12.35)

    def test_same_site_device_link_distance_is_zero(self):
        ne_graph = {
            "NE-A": {"site_id": "SITE-A", "link": {"NE-B": {"IP": "->"}}},
            "NE-B": {"site_id": "SITE-A", "link": {"NE-A": {"IP": "<-"}}},
        }
        group = {
            "故障组ID": "GROUP-SAME-SITE",
            "alarms": [{"告警源": "NE-A", "告警标题": "普通设备告警"}],
        }

        completed = complete_group_topology(
            group,
            ne_graph,
            {"SITE-A": {"site_name": "Site A"}},
            build_site_to_ne_ids(ne_graph),
            site_chain_index={},
        )

        self.assertEqual(completed["ne_info"]["NE-A"]["link"]["NE-B"]["distance"], 0.0)

    def test_resource_buffer_script_uses_topology_resources(self):
        ne_graph = {
            "NE-A": {"site_id": "SITE-A", "domain": "Ran", "link": {}},
            "NE-B": {"site_id": "SITE-B", "domain": "Ran", "link": {}},
            "NE-UP": {"site_id": "SITE-UP", "domain": "Data", "link": {}},
        }
        site_graph = {
            "SITE-A": {"site_name": "Site A", "is_hub": False},
            "SITE-B": {"site_name": "Site B", "is_hub": False},
            "SITE-UP": {"site_name": "Upstream", "is_hub": True},
        }
        site_chains = {
            "meta": {
                "input_config": {"max_depth": None, "restrict_relation": False},
                "relation_options": {"restrict_relation_effective": False},
            },
            "sites": {
                "SITE-A": {"upstream_site_hops": {"SITE-UP": 1}, "downstream_site_hops": {}},
                "SITE-B": {"upstream_site_hops": {"SITE-UP": 1}, "downstream_site_hops": {}},
                "SITE-UP": {"upstream_site_hops": {}, "downstream_site_hops": {"SITE-A": 1, "SITE-B": 1}},
            },
        }
        group = {
            "故障组ID": "GROUP-1",
            "alarms": [
                {"告警源": "NE-A", "告警标题": "Offline"},
                {"告警源": "NE-B", "告警标题": "Offline"},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "groups.jsonl"
            output_path = temp_path / "completed.jsonl"
            resource_buffer_path = temp_path / "resource_buffer.jsonl"
            input_path.write_text(json.dumps(group, ensure_ascii=False) + "\n", encoding="utf-8")
            _write_resource_buffer(
                resource_buffer_path,
                (
                    ("site_graph", site_graph),
                    ("ne_graph", ne_graph),
                    ("site_chains", site_chains),
                ),
            )

            stats = complete_groups_from_resource_buffer(
                str(input_path),
                str(output_path),
                str(resource_buffer_path),
                show_progress=False,
            )

            self.assertEqual(stats["output_group_count"], 1)
            completed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["topology_completion"]["common_upstream_site"], "SITE-UP")
            self.assertEqual(completed["role_mapping"]["common_upstream_site"], ["SITE-UP"])
            self.assertIn("NE-UP", completed["ne_info"])


if __name__ == "__main__":
    unittest.main()
