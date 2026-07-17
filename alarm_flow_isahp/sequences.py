from dataclasses import asdict, dataclass
from typing import Optional

from alarm_tools.alarm_types import LINK_ALARMS, OFFLINE_ALARMS, POWER_ALARMS
from fault_grouping.alarm_events.io import is_clear_alarm


SUPPORTED_TYPE_FIELDS = frozenset(
    {"alarm_title", "alarm_type", "site_id", "alarm_source", "device_domain"}
)


@dataclass(frozen=True)
class AlarmSequenceConfig:
    type_fields: tuple = ("alarm_source", "alarm_type")
    history_window_sec: float = 900.0
    max_history_events: int = 128
    min_events: Optional[int] = 2
    time_scale_sec: float = 60.0
    include_clear: bool = False

    def __post_init__(self):
        if not self.type_fields:
            raise ValueError("type_fields must not be empty")
        unknown_fields = set(self.type_fields) - SUPPORTED_TYPE_FIELDS
        if unknown_fields:
            raise ValueError(f"unsupported alarm type fields: {sorted(unknown_fields)}")
        if self.history_window_sec <= 0:
            raise ValueError("history_window_sec must be > 0")
        if self.max_history_events < 1:
            raise ValueError("max_history_events must be >= 1")
        if self.min_events is not None and self.min_events < 2:
            raise ValueError("min_events must be >= 2")
        if self.time_scale_sec <= 0:
            raise ValueError("time_scale_sec must be > 0")

    def to_dict(self):
        payload = asdict(self)
        payload["type_fields"] = list(self.type_fields)
        return payload

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        if "type_fields" in payload:
            payload["type_fields"] = tuple(payload["type_fields"])
        return cls(**payload)


@dataclass
class AlarmSequence:
    sequence_id: str
    type_ids: list
    type_labels: list
    alarm_source_ids: list
    alarm_type_ids: list
    times: list
    events: list
    target_windows: list

    def __len__(self):
        return len(self.type_ids)


@dataclass
class AlarmTargetWindow:
    sequence_id: str
    target_index: int
    target_type_id: int
    target_type_label: str
    target_time: float
    target_event: dict
    interval_dt: float
    query_dt: float
    query_alarm_source_id: int
    query_alarm_type_id: int
    history_indices: list
    history_type_ids: list
    history_type_labels: list
    history_times: list
    history_dts: list
    history_alarm_source_ids: list
    history_alarm_type_ids: list
    history_events: list
    topology_pair_features: list

    def __len__(self):
        return len(self.history_indices)


class AlarmTypeVocab:
    def __init__(self, labels=()):
        self.labels = []
        self._label_to_id = {}
        for label in labels:
            self.add(label)

    def __len__(self):
        return len(self.labels)

    def add(self, label):
        label = str(label)
        if label not in self._label_to_id:
            self._label_to_id[label] = len(self.labels)
            self.labels.append(label)
        return self._label_to_id[label]

    def get(self, label):
        return self._label_to_id.get(str(label))

    def encode(self, label, *, add_missing=False):
        encoded = self.get(label)
        if encoded is None and add_missing:
            encoded = self.add(label)
        return encoded

    def to_dict(self):
        return {"labels": list(self.labels)}

    @classmethod
    def from_dict(cls, payload):
        return cls((payload or {}).get("labels", ()))


@dataclass
class AlarmVocabs:
    type_vocab: AlarmTypeVocab
    alarm_source_vocab: AlarmTypeVocab
    alarm_type_vocab: AlarmTypeVocab

    def to_dict(self):
        return {
            "type_vocab": self.type_vocab.to_dict(),
            "alarm_source_vocab": self.alarm_source_vocab.to_dict(),
            "alarm_type_vocab": self.alarm_type_vocab.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            type_vocab=AlarmTypeVocab.from_dict(payload["type_vocab"]),
            alarm_source_vocab=AlarmTypeVocab.from_dict(payload["alarm_source_vocab"]),
            alarm_type_vocab=AlarmTypeVocab.from_dict(payload["alarm_type_vocab"]),
        )


def parse_type_fields(text):
    fields = tuple(part.strip() for part in str(text or "").split(",") if part.strip())
    return fields or ("alarm_source", "alarm_type")


def alarm_type_from_title(alarm_title):
    if alarm_title in LINK_ALARMS:
        return "link"
    if alarm_title in POWER_ALARMS:
        return "power"
    if alarm_title in OFFLINE_ALARMS:
        return "offline"
    return None


def _event_type_field_value(item, field):
    if field == "alarm_type":
        return alarm_type_from_title(item.get("alarm_title", ""))
    return item.get(field, "")


def _label_value(value):
    return str(value or "").strip() or "<empty>"


def alarm_source_label(item):
    return _label_value(item.get("alarm_source", ""))


def alarm_type_label(item):
    return alarm_type_from_title(item.get("alarm_title", ""))


def event_type_label(item, type_fields):
    values = []
    for field in type_fields:
        values.append(_label_value(_event_type_field_value(item, field)))
    return " | ".join(values)


def _iter_model_events(sorted_alarm_events, config):
    for item in sorted(sorted_alarm_events, key=lambda event: float(event["ts"])):
        if not config.include_clear and is_clear_alarm(item.get("alarm", {})):
            continue
        if alarm_type_label(item) is None:
            continue
        yield item


def build_alarm_vocabs(sorted_alarm_events, config):
    vocabs = AlarmVocabs(
        type_vocab=AlarmTypeVocab(),
        alarm_source_vocab=AlarmTypeVocab(),
        alarm_type_vocab=AlarmTypeVocab(),
    )
    event_count = 0
    for item in _iter_model_events(sorted_alarm_events, config):
        vocabs.type_vocab.add(event_type_label(item, config.type_fields))
        vocabs.alarm_source_vocab.add(alarm_source_label(item))
        vocabs.alarm_type_vocab.add(alarm_type_label(item))
        event_count += 1
    return vocabs, event_count


def build_alarm_type_vocab(sorted_alarm_events, config):
    vocabs, event_count = build_alarm_vocabs(sorted_alarm_events, config)
    return vocabs.type_vocab, event_count


def _history_indexes(times, target_index, config):
    target_time = times[target_index]
    max_age = config.history_window_sec / config.time_scale_sec
    history_indexes = []
    for source_index in range(target_index - 1, -1, -1):
        age = target_time - times[source_index]
        if age <= 0:
            continue
        if age > max_age:
            break
        history_indexes.append(source_index)
        if len(history_indexes) >= config.max_history_events:
            break
    return list(reversed(history_indexes))


def _build_topology_features(events, history_indexes, target_index, topology_index):
    if topology_index is None:
        return []
    target_ne = events[target_index].get("alarm_source", "")
    return [
        topology_index.pair_features(events[source_index].get("alarm_source", ""), target_ne)
        for source_index in history_indexes
    ]


def _build_target_windows(sequence, event_dts, topology_index, config):
    target_windows = []
    for target_index in range(1, len(sequence)):
        history_indexes = _history_indexes(sequence.times, target_index, config)
        # Query 始终用 target_index - 1 与 interval_dt 的积分区间对齐；
        # 即便它被 history_window_sec 截断在 history 外，也比窗口里更早的事件更能代表 target 前一刻状态。
        query_index = target_index - 1
        target_windows.append(
            AlarmTargetWindow(
                sequence_id=sequence.sequence_id,
                target_index=target_index,
                target_type_id=sequence.type_ids[target_index],
                target_type_label=sequence.type_labels[target_index],
                target_time=sequence.times[target_index],
                target_event=sequence.events[target_index],
                interval_dt=event_dts[target_index],
                query_dt=event_dts[query_index],
                query_alarm_source_id=sequence.alarm_source_ids[query_index],
                query_alarm_type_id=sequence.alarm_type_ids[query_index],
                history_indices=history_indexes,
                history_type_ids=[sequence.type_ids[index] for index in history_indexes],
                history_type_labels=[sequence.type_labels[index] for index in history_indexes],
                history_times=[sequence.times[index] for index in history_indexes],
                history_dts=[event_dts[index] for index in history_indexes],
                history_alarm_source_ids=[
                    sequence.alarm_source_ids[index] for index in history_indexes
                ],
                history_alarm_type_ids=[sequence.alarm_type_ids[index] for index in history_indexes],
                history_events=[sequence.events[index] for index in history_indexes],
                topology_pair_features=_build_topology_features(
                    sequence.events,
                    history_indexes,
                    target_index,
                    topology_index,
                ),
            )
        )
    return target_windows


def _emit_sequence(
    sequence_id,
    raw_events,
    vocabs,
    config,
    add_missing_types,
    topology_index,
    *,
    build_target_windows: bool = True,
):
    type_ids = []
    type_labels = []
    alarm_source_ids = []
    alarm_type_ids = []
    events = []
    for item in raw_events:
        label = event_type_label(item, config.type_fields)
        type_id = vocabs.type_vocab.encode(label, add_missing=add_missing_types)
        source_id = vocabs.alarm_source_vocab.encode(
            alarm_source_label(item),
            add_missing=add_missing_types,
        )
        alarm_type_id = vocabs.alarm_type_vocab.encode(
            alarm_type_label(item),
            add_missing=add_missing_types,
        )
        if type_id is None or source_id is None or alarm_type_id is None:
            continue
        type_ids.append(type_id)
        type_labels.append(label)
        alarm_source_ids.append(source_id)
        alarm_type_ids.append(alarm_type_id)
        events.append(item)

    if not events:
        return None
    if config.min_events is not None and len(events) < config.min_events:
        return None

    start_ts = float(events[0]["ts"])
    times = [(float(item["ts"]) - start_ts) / config.time_scale_sec for item in events]
    event_dts = [0.0]
    event_dts.extend(
        max(0.0, times[index] - times[index - 1])
        for index in range(1, len(times))
    )
    sequence = AlarmSequence(
        sequence_id=sequence_id,
        type_ids=type_ids,
        type_labels=type_labels,
        alarm_source_ids=alarm_source_ids,
        alarm_type_ids=alarm_type_ids,
        times=times,
        events=events,
        target_windows=[],
    )
    # target_windows are an isahp-specific structure (each window pre-materializes
    # the per-event candidate history) and can occupy several GB for million-
    # event streams. Callers that don't need them (brunch training only needs
    # the flat event list) can opt out via build_target_windows=False.
    if build_target_windows:
        sequence.target_windows = _build_target_windows(
            sequence,
            event_dts,
            topology_index,
            config,
        )
    return sequence


def build_alarm_sequences(
    sorted_alarm_events,
    vocabs,
    config,
    *,
    add_missing_types=False,
    topology_index=None,
    build_target_windows: bool = True,
):
    model_events = list(_iter_model_events(sorted_alarm_events, config))
    sequence = _emit_sequence(
        "__global__",
        model_events,
        vocabs,
        config,
        add_missing_types,
        topology_index,
        build_target_windows=build_target_windows,
    )
    sequences = [sequence] if sequence is not None else []
    stats = {
        "sequence_count": 0,
        "target_window_count": 0,
        "history_pair_count": 0,
        "max_window_history_count": 0,
        "input_event_count": len(model_events),
        "sequence_event_position_count": 0,
        "dropped_event_count": 0,
    }

    represented_event_ids = set()
    for sequence in sequences:
        stats["sequence_event_position_count"] += len(sequence)
        stats["target_window_count"] += len(sequence.target_windows)
        stats["history_pair_count"] += sum(len(window) for window in sequence.target_windows)
        stats["max_window_history_count"] = max(
            stats["max_window_history_count"],
            max((len(window) for window in sequence.target_windows), default=0),
        )
        represented_event_ids.update(id(event) for event in sequence.events)
    stats["dropped_event_count"] = sum(1 for event in model_events if id(event) not in represented_event_ids)
    stats["sequence_count"] = len(sequences)
    return sequences, stats
