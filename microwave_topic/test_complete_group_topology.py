import unittest

from fault_grouping.site_topology import build_site_to_ne_ids
from microwave_topic.complete_group_topology import complete_group_topology


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


if __name__ == "__main__":
    unittest.main()
