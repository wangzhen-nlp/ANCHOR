import unittest
import uuid as uuid_module

from fault_grouping.emitted_group_store import EmittedGroupStore
from fault_grouping.temporal_engine.utils import get_symptom_alarm_identity


# 一个足够大的停留时间，确保测试期间历史组不会因为过期被裁剪，
# 这样我们检验的是"汇聚/去重"逻辑本身，而不是过期逻辑。
_LONG_STAY = 10 ** 9
_RULES = {"r": {"max_stay_time_sec": _LONG_STAY}}


def _symptom(eid, ts, occurrence_id=None, node="S1", alarm_source="NE1", alarm="A"):
    symptom = {
        "node": node,
        "alarm_source": alarm_source,
        "alarm": alarm,
        "eid": eid,
        "ts": ts,
    }
    if occurrence_id is not None:
        symptom["occurrence_id"] = occurrence_id
    return symptom


def _period_symptom(eid, ts, occurrence_id, node="S1", alarm_source="NE1", alarm="A"):
    symptom = _symptom(eid, ts, occurrence_id, node, alarm_source, alarm)
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
        self.assertEqual(self._emitted_identities(first), self._emitted_identities(first))

        # 同 eid、不同发生（不同 occurrence_id + 不同时间）→ 不能被前一组吸收，必须照常输出。
        second = self._finalize(store, _match([_symptom("E", 2, "raw-2")]), current_time=2)
        self.assertIsNotNone(second, "重复 eid 的不同发生被错误抑制，造成漏报")
        self.assertNotEqual(
            self._emitted_identities(first),
            self._emitted_identities(second),
        )

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

    # ---- 告警时段重叠模式（use_alarm_period_cache=True）----

    def test_period_mode_distinct_occurrences_not_lost(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=True)

        first = self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 1)
        self.assertIsNotNone(first)
        # 同 eid、同 (node, source, alarm)、时间重叠，但 occurrence_id 不同 → 不同发生，不能合并。
        second = self._finalize(store, _match([_period_symptom("E", 1, "raw-2")]), 2)
        self.assertIsNotNone(second, "时段模式下不同发生被错误抑制，造成漏报")

    def test_period_mode_identical_occurrence_suppressed(self):
        store = EmittedGroupStore(_RULES, _LONG_STAY, use_alarm_period_cache=True)

        self.assertIsNotNone(self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 1))
        again = self._finalize(store, _match([_period_symptom("E", 1, "raw-1")]), 2)
        self.assertIsNone(again, "时段模式下相同发生被重复输出，造成多报")
        self.assertEqual(len([g for g in store.groups if g is not None]), 1)


if __name__ == "__main__":
    unittest.main()
