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


if __name__ == "__main__":
    unittest.main()
