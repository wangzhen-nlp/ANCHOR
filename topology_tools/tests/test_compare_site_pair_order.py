#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_site_pair_order 的核心统计逻辑单测。

重点验证：
- upstream/downstream 字段而非 prediction 字符串决定方向（跨 pairwise/global 可比）；
- 方向反转、双向<->有向 转换的分类；
- 混淆矩阵、一致率与 Cohen's kappa；
- downstream_map 有向关系对的 Jaccard。
"""

import unittest
import json
import tempfile

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topology_tools.compare_site_pair_order import (
    build_relation_map,
    compare,
    directed_relation_pairs,
    load_prediction,
    prediction_from_site_chains,
    relation_counts,
)


# 左：pairwise 风格（prediction 写 upstream->downstream）
LEFT = {
    "edges": [
        {"site_a": "A", "site_b": "B", "prediction": "A->B",
         "upstream_site": "A", "downstream_site": "B"},
        {"site_a": "B", "site_b": "C", "prediction": "B->C",
         "upstream_site": "B", "downstream_site": "C"},
        {"site_a": "C", "site_b": "D", "prediction": "bidirectional",
         "upstream_site": None, "downstream_site": None},
        {"site_a": "D", "site_b": "E", "prediction": "D->E",
         "upstream_site": "D", "downstream_site": "E"},
    ],
    "downstream_map": {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["C", "E"]},
}

# 右：global 风格（prediction 写 downstream->upstream，方向语义靠字段）
RIGHT = {
    "edges": [
        {"site_a": "A", "site_b": "B", "prediction": "B->A",
         "upstream_site": "A", "downstream_site": "B"},
        {"site_a": "B", "site_b": "C", "prediction": "C->B",
         "upstream_site": "C", "downstream_site": "B"},
        {"site_a": "C", "site_b": "D", "prediction": "D->C",
         "upstream_site": "D", "downstream_site": "C"},
        {"site_a": "E", "site_b": "F", "prediction": "bidirectional",
         "upstream_site": None, "downstream_site": None},
    ],
    "downstream_map": {"A": ["B"], "C": ["B"], "D": ["C"]},
}

SITE_CHAINS = {
    "meta": {"source": "test"},
    "sites": {
        "A": {
            "bidirectional_sites": [],
            "downstream_site_hops": {"B": 1, "C": 2},
            "upstream_site_hops": {},
        },
        "B": {
            "bidirectional_sites": ["C"],
            "downstream_site_hops": {},
            "upstream_site_hops": {"A": 1},
        },
        "C": {
            "bidirectional_sites": ["B"],
            "downstream_site_hops": {},
            "upstream_site_hops": {"A": 2},
        },
    },
}


class CompareSitePairOrderTest(unittest.TestCase):
    def setUp(self):
        self.left_map, _, _ = build_relation_map(LEFT)
        self.right_map, _, _ = build_relation_map(RIGHT)
        self.result = compare(self.left_map, self.right_map)

    def test_prediction_string_does_not_drive_direction(self):
        # A-B：左字符串 "A->B"、右字符串 "B->A"，但 upstream 都是 A → 同为 s0_up
        self.assertEqual(self.left_map[("A", "B")], "s0_up")
        self.assertEqual(self.right_map[("A", "B")], "s0_up")

    def test_relation_counts(self):
        left_stats = relation_counts(self.left_map)
        self.assertEqual(left_stats["edge_count"], 4)
        self.assertEqual(left_stats["directed_count"], 3)
        self.assertEqual(left_stats["bidirectional_count"], 1)
        self.assertAlmostEqual(left_stats["directed_ratio"], 0.75)

    def test_edge_set_split(self):
        self.assertEqual(self.result["common_count"], 3)
        self.assertEqual(self.result["left_only"], [("D", "E")])
        self.assertEqual(self.result["right_only"], [("E", "F")])

    def test_reversal_and_transitions(self):
        # B-C：左 upstream=B(s0_up)，右 upstream=C(s1_up) → 反转
        self.assertEqual(len(self.result["reversals"]), 1)
        self.assertEqual(self.result["reversals"][0][0], ("B", "C"))
        # C-D：左双向、右有向
        self.assertEqual(len(self.result["bidir_to_directed"]), 1)
        self.assertEqual(len(self.result["directed_to_bidir"]), 0)

    def test_agreement_and_confusion(self):
        self.assertEqual(self.result["agree_count"], 1)
        self.assertAlmostEqual(self.result["agreement_rate"], 1 / 3)
        confusion = self.result["confusion"]
        self.assertEqual(confusion[("s0_up", "s0_up")], 1)
        self.assertEqual(confusion[("s0_up", "s1_up")], 1)
        self.assertEqual(confusion[("bidir", "s1_up")], 1)

    def test_cohen_kappa(self):
        # observed=1/3, expected=2/9 → kappa=(1/3-2/9)/(1-2/9)=1/7
        self.assertAlmostEqual(self.result["kappa"], 1 / 7, places=6)

    def test_downstream_pairs_jaccard(self):
        left_pairs = directed_relation_pairs(LEFT)
        right_pairs = directed_relation_pairs(RIGHT)
        intersection = left_pairs & right_pairs
        union = left_pairs | right_pairs
        self.assertEqual(len(left_pairs), 5)
        self.assertEqual(len(right_pairs), 3)
        self.assertEqual(len(intersection), 2)  # (A,B) 和 (D,C)
        self.assertAlmostEqual(len(intersection) / len(union), 1 / 3)

    def test_downstream_map_fallback_from_edges(self):
        # 没有 downstream_map 时应从 edges 重建
        no_map = {"edges": RIGHT["edges"]}
        pairs = directed_relation_pairs(no_map)
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("D", "C"), pairs)

    def test_prediction_from_site_chains_uses_only_first_hop(self):
        prediction = prediction_from_site_chains(SITE_CHAINS)
        relation_map, duplicate_count, invalid_count = build_relation_map(prediction)
        self.assertEqual(duplicate_count, 0)
        self.assertEqual(invalid_count, 0)
        self.assertEqual(relation_map, {
            ("A", "B"): "s0_up",
            ("B", "C"): "bidir",
        })
        self.assertNotIn(("A", "C"), relation_map)
        self.assertEqual(
            directed_relation_pairs(prediction),
            {("A", "B"), ("B", "C"), ("C", "B")},
        )

    def test_load_prediction_from_resource_buffer_jsonl(self):
        records = [
            {"resource_type": "site_graph", "data": {"ignored": True}},
            {"resource_type": "site_chains", "data": SITE_CHAINS},
        ]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
            f.flush()
            prediction, path = load_prediction(f.name)

        relation_map, _, _ = build_relation_map(prediction)
        self.assertEqual(str(path), f.name)
        self.assertEqual(relation_map[("A", "B")], "s0_up")
        self.assertEqual(relation_map[("B", "C")], "bidir")


if __name__ == "__main__":
    unittest.main()
