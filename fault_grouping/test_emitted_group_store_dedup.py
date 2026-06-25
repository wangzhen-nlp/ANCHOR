import os
import sys
import unittest
import uuid as uuid_module
import random

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fault_grouping.emitted_group_store import EmittedGroupStore
from fault_grouping.temporal_engine.utils import (
    get_symptom_alarm_identity,
    merge_match_batch,
    merge_symptom_records,
)


# 一个足够大的停留时间，确保测试期间历史组不会因为过期被裁剪，
# 这样我们检验的是"汇聚/去重"逻辑本身，而不是过期逻辑。
_LONG_STAY = 10 ** 9
_RULES = {"r": {"max_stay_time_sec": _LONG_STAY}}


def _uuid(value):
    try:
        return str(uuid_module.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, f"test-emitted-group:{value}"))


def _symptom(eid, ts, occurrence_uuid, node="S1", alarm_source="NE1", alarm="A"):
    return {
        "node": node,
        "alarm_source": alarm_source,
        "alarm": alarm,
        "eid": eid,
        "ts": ts,
        "occurrence_uuid": _uuid(occurrence_uuid),
    }


def _period_symptom(eid, ts, occurrence_uuid, node="S1", alarm_source="NE1", alarm="A"):
    symptom = _symptom(eid, ts, occurrence_uuid, node, alarm_source, alarm)
    symptom["_segment_start_ts"] = ts
    symptom["_segment_end_ts"] = ts
    return symptom


def _match(symptoms, node="S1"):
    return {
        "uuid": str(uuid_module.uuid4()),
        "rule": "r",
        "role_mapping": {"cascade": [node]},
        "symptoms": list(symptoms),
    }


class EmittedGroupStoreDedupTest(unittest.TestCase):
    """直接驱动 EmittedGroupStore，复刻 engine._finalize_matches_with_history 的调用序列，

    验证跨轮汇聚在"重复 eid / 多次发生"下做到不多（无重复落库）不漏（无丢失发生）。
    """

    def _finalize(self, store, match, current_time):
        """复刻 engine 的最终合并+落库循环，返回真正会输出的故障组（或 None 表示被抑制）。"""
        store.prune_expired(current_time)
        anchor_ts = store.get_group_anchor_ts(match, current_time)
        (
            merged_match,
            merged_group_indexes,
            _related_group_uuids,
            should_emit,
            _reason,
        ) = store.merge_with_related(match)
        if not should_emit:
            store.extend_related_expire_ts(merged_group_indexes, merged_match, anchor_ts)
            return None
        store.replace_and_store(merged_group_indexes, anchor_ts, merged_match)
        return merged_match

    def _emitted_identities(self, emitted_group):
        return [
            get_symptom_alarm_identity(symptom)
            for symptom in emitted_group["symptoms"]
        ]

    def _assert_no_duplicate_within_group(self, emitted_group):
        identities = self._emitted_identities(emitted_group)
        self.assertEqual(
            len(identities),
            len(set(identities)),
            f"同一故障组内出现重复告警发生: {identities}",
        )

    # ---- eid 模式（use_alarm_period_cache=False）----

    def test_eid_mode_distinct_occurrences_not_lost(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)

        first = self._finalize(store, _match([_symptom("E", 1, "raw-1")]), current_time=1)
        self.assertIsNotNone(first)
        self.assertEqual(len(self._emitted_identities(first)), 1)

        # 同 eid、不同发生（不同 occurrence_uuid + 不同时间）→ 不能被前一组吸收，必须照常输出。
        second = self._finalize(store, _match([_symptom("E", 2, "raw-2")]), current_time=2)
        self.assertIsNotNone(second, "重复 eid 的不同发生被错误抑制，造成漏报")
        self.assertNotEqual(
            self._emitted_identities(first),
            self._emitted_identities(second),
        )

    def test_numeric_zero_eid_and_timestamp_are_valid_identity_fields(self):
        first = _symptom(0, 0, "raw-0")
        second = _symptom(0, 1, "raw-1")

        self.assertIsNotNone(get_symptom_alarm_identity(first))
        self.assertNotEqual(
            get_symptom_alarm_identity(first),
            get_symptom_alarm_identity(second),
        )
        merged = merge_match_batch([_match([first, second])])
        self.assertEqual([symptom["eid"] for symptom in merged[0]["symptoms"]], [0, 0])

    def test_eid_mode_identical_reemission_suppressed(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)

        self.assertIsNotNone(self._finalize(store, _match([_symptom("E", 1, "raw-1")]), 1))
        # 完全相同的发生再次进来 → 必须被抑制，避免重复落库（不多）。
        again = self._finalize(store, _match([_symptom("E", 1, "raw-1")]), 2)
        self.assertIsNone(again, "完全相同的发生被重复输出，造成多报")
        self.assertEqual(len([g for g in store.groups if g is not None]), 1)

    def test_eid_mode_superset_absorbs_history_without_duplication(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)

        self._finalize(store, _match([_symptom("E", 1, "raw-1")]), 1)
        self._finalize(store, _match([_symptom("E", 2, "raw-2")]), 2)

        # 超集组：包含两个历史发生 + 一个新发生 → 应输出，并吸收两个历史组。
        superset = self._finalize(
            store,
            _match([
                _symptom("E", 1, "raw-1"),
                _symptom("E", 2, "raw-2"),
                _symptom("E", 3, "raw-3"),
            ]),
            3,
        )
        self.assertIsNotNone(superset)
        self._assert_no_duplicate_within_group(superset)
        self.assertEqual(len(superset["symptoms"]), 3)
        # 两个历史组被吸收删除，只剩当前这一个超集组。
        self.assertEqual(len([g for g in store.groups if g is not None]), 1)

    def test_eid_mode_no_loss_across_many_rounds(self):
        """端到端不变量：所有输出组覆盖的发生 = 喂入的全部不同发生，且无重复。"""
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)

        rounds = [
            [_symptom("E", 1, "raw-1")],
            [_symptom("E", 2, "raw-2")],
            [_symptom("E", 1, "raw-1"), _symptom("E", 2, "raw-2")],  # 子集组合，应被抑制
            [_symptom("F", 5, "raw-5"), _symptom("E", 3, "raw-3")],
            [_symptom("E", 1, "raw-1")],  # 旧发生重放，应被抑制
        ]
        fed_identities = set()
        for round_idx, symptoms in enumerate(rounds, start=1):
            fed_identities.update(get_symptom_alarm_identity(s) for s in symptoms)
            self._finalize(store, _match(symptoms), current_time=round_idx)

        live_groups = [g for g in store.groups if g is not None]
        seen = []
        for group in live_groups:
            for symptom in group["match"]["symptoms"]:
                seen.append(get_symptom_alarm_identity(symptom))

        # 不漏：每个喂入的不同发生都仍能在某个历史组中找到。
        self.assertEqual(set(seen) & fed_identities, fed_identities)
        # 不多：历史组之间不重复保存同一个发生。
        self.assertEqual(len(seen), len(set(seen)), f"历史组间存在重复发生: {seen}")

    def test_eid_mode_merges_role_metadata_for_same_occurrence(self):
        """同一发生在当前组与历史组以不同 role 命中时，归属信息要合并而不是被覆盖。"""
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)

        history = _match([_symptom("E", 1, "raw-1")])
        history["symptoms"][0]["matched_role"] = "root"
        history["symptoms"][0]["matched_role_list"] = ["root"]
        self._finalize(store, history, 1)

        # 当前组里同一个发生以另一个 role 命中，并带一个新发生触发输出。
        current = _match([
            dict(_symptom("E", 1, "raw-1"), matched_role="cascade", matched_role_list=["cascade"]),
            _symptom("F", 2, "raw-2"),
        ])
        emitted = self._finalize(store, current, 2)
        self.assertIsNotNone(emitted)

        by_eid = {s["eid"]: s for s in emitted["symptoms"]}
        self.assertEqual(
            sorted(by_eid["E"].get("matched_role_list", [])),
            ["cascade", "root"],
            "同一发生的 role 归属被覆盖丢失，未做合并",
        )

    def test_single_candidate_dedupes_occurrence_without_dropping_duplicate_eid(self):
        first = dict(
            _symptom("E", 1, "raw-1"),
            matched_role="root",
            matched_role_list=["root"],
        )
        same_occurrence = dict(
            _symptom("E", 1, "raw-1"),
            matched_role="cascade",
            matched_role_list=["cascade"],
        )
        different_occurrence = _symptom("E", 2, "raw-2")

        merged = merge_match_batch([
            _match([first, same_occurrence, different_occurrence])
        ])

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["symptoms"]), 2)
        by_occurrence = {
            symptom["occurrence_uuid"]: symptom
            for symptom in merged[0]["symptoms"]
        }
        self.assertEqual(set(by_occurrence), {_uuid("raw-1"), _uuid("raw-2")})
        self.assertEqual(
            sorted(by_occurrence[_uuid("raw-1")]["matched_role_list"]),
            ["cascade", "root"],
        )

    def test_single_candidate_dedupes_identical_anonymous_symptoms(self):
        unidentified = {
            "node": "S2",
            "alarm": "NO_ID_ALARM",
            "eid": "NO_ID",
            "occurrence_uuid": _uuid("anonymous-single"),
            "details": {"values": [1, 2]},
        }

        merged = merge_match_batch([
            _match([dict(unidentified), dict(unidentified)])
        ])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["symptoms"], [unidentified])

    def test_period_single_candidate_dedupes_identical_passthrough_symptoms(self):
        unidentified = {
            "eid": "NO_ID",
            "occurrence_uuid": _uuid("anonymous-period"),
            "metadata": {"values": [1, 2]},
        }

        merged = merge_match_batch(
            [_match([dict(unidentified), dict(unidentified)])],
            use_alarm_period_cache=True,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["symptoms"], [unidentified])

    def test_batch_merge_preserves_symptoms_without_identity_key(self):
        shared = _symptom("E", 1, "raw-1")
        unidentified = {
            "node": "S2",
            "alarm": "NO_ID_ALARM",
            "alarm_source": "NE2",
            "ts": 2,
            "eid": "NO_ID",
            "occurrence_uuid": _uuid("anonymous-batch"),
        }

        merged = merge_match_batch([
            _match([shared, dict(unidentified)]),
            _match([dict(shared), unidentified]),
        ])

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["symptoms"]), 2)
        self.assertIn(unidentified, merged[0]["symptoms"])

    def test_batch_merge_keeps_distinct_unidentified_symptoms(self):
        shared = _symptom("E", 1, "raw-1")
        first = {"node": "S2", "alarm": "NO_ID_ALARM", "ts": 2, "eid": "NO_ID", "occurrence_uuid": _uuid("anonymous-a"), "details": ["a"]}
        second = {"node": "S2", "alarm": "NO_ID_ALARM", "ts": 2, "eid": "NO_ID", "occurrence_uuid": _uuid("anonymous-b"), "details": ["b"]}

        merged = merge_match_batch([
            _match([shared, first]),
            _match([dict(shared), second]),
        ])

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["symptoms"]), 3)
        self.assertIn(first, merged[0]["symptoms"])
        self.assertIn(second, merged[0]["symptoms"])

    def test_history_does_not_suppress_new_unidentified_symptom(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)
        shared = _symptom("E", 1, "raw-1")
        self._finalize(store, _match([shared]), 1)

        unidentified = {
            "node": "S2",
            "alarm": "NO_ID_ALARM",
            "alarm_source": "NE2",
            "ts": 2,
            "eid": "NO_ID",
            "occurrence_uuid": _uuid("anonymous-history-new"),
        }
        emitted = self._finalize(
            store,
            _match([dict(shared), unidentified]),
            2,
        )

        self.assertIsNotNone(emitted)
        self.assertEqual(len(emitted["symptoms"]), 2)
        self.assertIn(unidentified, emitted["symptoms"])

    def test_history_merge_dedupes_identical_unidentified_symptom(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)
        shared = _symptom("E", 1, "raw-1")
        unidentified = {
            "node": "S2",
            "alarm": "NO_ID_ALARM",
            "alarm_source": "NE2",
            "ts": 2,
            "eid": "NO_ID",
            "occurrence_uuid": _uuid("anonymous-history-same"),
            "nested": {"values": [1, 2]},
        }
        self._finalize(store, _match([shared, dict(unidentified)]), 1)

        emitted = self._finalize(
            store,
            _match([dict(shared), dict(unidentified)]),
            2,
        )

        self.assertIsNone(emitted)

    def test_randomized_batch_occurrence_conservation_with_duplicate_eids(self):
        rng = random.Random(20260618)
        occurrence_pool = [
            _symptom(f"E{idx % 3}", idx + 1, f"raw-{idx}")
            for idx in range(12)
        ]

        for _case_idx in range(100):
            matches = []
            input_identities = set()
            for _match_idx in range(rng.randint(1, 10)):
                symptoms = []
                for occurrence in rng.sample(
                    occurrence_pool,
                    rng.randint(1, len(occurrence_pool)),
                ):
                    symptoms.append(dict(occurrence))
                    input_identities.add(get_symptom_alarm_identity(occurrence))
                    if rng.random() < 0.25:
                        symptoms.append(dict(occurrence))
                matches.append(_match(symptoms))

            merged = merge_match_batch(matches)
            output_identities = [
                get_symptom_alarm_identity(symptom)
                for match in merged
                for symptom in match["symptoms"]
            ]

            self.assertEqual(set(output_identities), input_identities)
            self.assertEqual(len(output_identities), len(input_identities))

    def test_randomized_history_occurrence_conservation_with_duplicate_eids(self):
        rng = random.Random(20260619)
        occurrence_pool = [
            _symptom(f"E{idx % 4}", idx + 1, f"raw-{idx}")
            for idx in range(16)
        ]

        for _case_idx in range(100):
            store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=False)
            fed_identities = set()
            for round_idx in range(1, rng.randint(5, 25)):
                symptoms = []
                for occurrence in rng.sample(
                    occurrence_pool,
                    rng.randint(1, len(occurrence_pool)),
                ):
                    symptoms.append(dict(occurrence))
                    fed_identities.add(get_symptom_alarm_identity(occurrence))
                    if rng.random() < 0.2:
                        symptoms.append(dict(occurrence))
                normalized = merge_match_batch([_match(symptoms)])[0]
                self._finalize(store, normalized, round_idx)

            stored_identities = [
                get_symptom_alarm_identity(symptom)
                for item in store.groups
                if item is not None
                for symptom in item["match"]["symptoms"]
            ]
            self.assertEqual(set(stored_identities), fed_identities)
            self.assertEqual(len(stored_identities), len(fed_identities))

    def test_randomized_period_batch_conserves_overlapping_duplicate_eids(self):
        rng = random.Random(20260620)
        occurrence_pool = []
        for idx in range(12):
            symptom = _period_symptom(f"E{idx % 3}", idx % 4 + 1, f"raw-{idx}")
            symptom["_segment_end_ts"] = symptom["_segment_start_ts"] + 3
            occurrence_pool.append(symptom)

        for _case_idx in range(100):
            matches = []
            input_identities = set()
            for _match_idx in range(rng.randint(1, 10)):
                symptoms = []
                for occurrence in rng.sample(
                    occurrence_pool,
                    rng.randint(1, len(occurrence_pool)),
                ):
                    symptoms.append(dict(occurrence))
                    input_identities.add(get_symptom_alarm_identity(occurrence))
                    if rng.random() < 0.25:
                        symptoms.append(dict(occurrence))
                matches.append(_match(symptoms))

            merged = merge_match_batch(matches, use_alarm_period_cache=True)
            output_identities = [
                get_symptom_alarm_identity(symptom)
                for match in merged
                for symptom in match["symptoms"]
            ]
            self.assertEqual(set(output_identities), input_identities)
            self.assertEqual(len(output_identities), len(input_identities))

    def test_randomized_period_history_conserves_overlapping_duplicate_eids(self):
        rng = random.Random(20260621)
        occurrence_pool = []
        for idx in range(12):
            symptom = _period_symptom(f"E{idx % 3}", idx % 4 + 1, f"raw-{idx}")
            symptom["_segment_end_ts"] = symptom["_segment_start_ts"] + 3
            occurrence_pool.append(symptom)

        for _case_idx in range(50):
            store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=True)
            fed_identities = set()
            for round_idx in range(1, rng.randint(5, 20)):
                symptoms = [
                    dict(occurrence)
                    for occurrence in rng.sample(
                        occurrence_pool,
                        rng.randint(1, len(occurrence_pool)),
                    )
                ]
                fed_identities.update(get_symptom_alarm_identity(s) for s in symptoms)
                normalized = merge_match_batch(
                    [_match(symptoms)],
                    use_alarm_period_cache=True,
                )[0]
                self._finalize(store, normalized, round_idx)

            stored_identities = [
                get_symptom_alarm_identity(symptom)
                for item in store.groups
                if item is not None
                for symptom in item["match"]["symptoms"]
            ]
            self.assertEqual(set(stored_identities), fed_identities)
            self.assertEqual(len(stored_identities), len(fed_identities))

    # ---- 告警时段重叠模式（use_alarm_period_cache=True）----

    def test_period_mode_distinct_occurrences_not_lost(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=True)

        first = self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 1)
        self.assertIsNotNone(first)
        # 同 eid、同 (node, source, alarm)、时间重叠，但 occurrence_uuid 不同 → 不同发生，不能合并。
        second = self._finalize(store, _match([_period_symptom("E", 1, "raw-2")]), 2)
        self.assertIsNotNone(second, "时段模式下不同发生被错误抑制，造成漏报")

    def test_period_mode_identical_occurrence_suppressed(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=True)

        self.assertIsNotNone(self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 1))
        again = self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 2)
        self.assertIsNone(again, "时段模式下相同发生被重复输出，造成多报")
        self.assertEqual(len([g for g in store.groups if g is not None]), 1)

    def test_period_batch_interleaved_duplicate_eids_merge_only_same_occurrence(self):
        """不同 occurrence 交错时，不能打断同一 occurrence 的重叠链。"""
        first = _period_symptom("E", 1, "raw-1")
        first["_segment_end_ts"] = 1
        other = _period_symptom("E", 1, "raw-2")
        other["_segment_end_ts"] = 2
        same_as_first = _period_symptom("E", 1, "raw-1")
        same_as_first["_segment_end_ts"] = 3

        merged, stats = merge_match_batch(
            [_match([first]), _match([other]), _match([same_as_first])],
            return_stats=True,
            use_alarm_period_cache=True,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["alarm_overlap_merge_group_count"], 1)
        occurrence_sets = {
            frozenset(
                symptom.get("occurrence_uuid")
                for symptom in match["symptoms"]
                if symptom.get("occurrence_uuid")
            )
            for match in merged
        }
        self.assertEqual(occurrence_sets, {frozenset({_uuid("raw-1")}), frozenset({_uuid("raw-2")})})

    def test_period_single_candidate_dedupes_interleaved_same_occurrence(self):
        first = _period_symptom("E", 1, "raw-1")
        first["_segment_end_ts"] = 1
        other = _period_symptom("E", 1, "raw-2")
        other["_segment_end_ts"] = 2
        same_as_first = _period_symptom("E", 1, "raw-1")
        same_as_first["_segment_end_ts"] = 3

        merged = merge_match_batch(
            [_match([first, other, same_as_first])],
            use_alarm_period_cache=True,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["symptoms"]), 2)
        by_occurrence = {
            symptom["occurrence_uuid"]: symptom
            for symptom in merged[0]["symptoms"]
        }
        self.assertEqual(set(by_occurrence), {_uuid("raw-1"), _uuid("raw-2")})
        self.assertEqual(by_occurrence[_uuid("raw-1")]["_segment_end_ts"], 3)

    def test_period_record_merge_keeps_metadata_from_later_base_argument(self):
        existing = dict(
            _period_symptom("E", 2, "raw-1"),
            matched_role="root",
            matched_role_list=["root"],
        )
        incoming = dict(
            _period_symptom("E", 1, "raw-1"),
            matched_role="cascade",
            matched_role_list=["cascade"],
        )

        merged = merge_symptom_records(existing, incoming)

        self.assertEqual(merged["_segment_start_ts"], 1)
        self.assertEqual(
            sorted(merged["matched_role_list"]),
            ["cascade", "root"],
        )


if __name__ == "__main__":
    unittest.main()
