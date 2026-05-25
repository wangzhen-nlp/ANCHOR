"""BRUNCH-style fault aggregation for ordered alarm streams."""

from alarm_flow_brunch.aggregator import (
    AlarmBRUNCHArtifact,
    AlarmBRUNCHConfig,
    AlarmBRUNCHOutput,
    aggregate_alarm_flow,
    infer_alarm_flow,
    load_alarm_brunch_artifact,
    save_alarm_brunch_artifact,
    train_alarm_brunch,
)

__all__ = [
    "AlarmBRUNCHArtifact",
    "AlarmBRUNCHConfig",
    "AlarmBRUNCHOutput",
    "aggregate_alarm_flow",
    "infer_alarm_flow",
    "load_alarm_brunch_artifact",
    "save_alarm_brunch_artifact",
    "train_alarm_brunch",
]
