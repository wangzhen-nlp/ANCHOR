from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlarmEvent:
    event_id: str
    ts: float
    alarm_title: str
    alarm_source: str
    site_id: str
    feature_counts: Counter
    event_key: str
    is_clear: bool = False
    device_domain: str = ""
    raw: dict = field(default_factory=dict)

    def compact(self):
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "alarm_title": self.alarm_title,
            "alarm_source": self.alarm_source,
            "site_id": self.site_id,
            "event_key": self.event_key,
            "is_clear": self.is_clear,
        }


@dataclass
class CascadeDecision:
    status: str
    event: AlarmEvent
    cascade_id: str = ""
    reason: str = ""
    probability: float = 0.0
    candidate_count: int = 0
    log_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        output = {
            "status": self.status,
            "cascade_id": self.cascade_id,
            "event_id": self.event.event_id,
            "ts": self.event.ts,
            "alarm_title": self.event.alarm_title,
            "alarm_source": self.event.alarm_source,
            "site_id": self.event.site_id,
            "reason": self.reason,
        }
        if self.status == "clustered":
            output.update(
                {
                    "probability": self.probability,
                    "candidate_count": self.candidate_count,
                    "log_score": self.log_score,
                }
            )
        if self.details:
            output["details"] = self.details
        return output


@dataclass
class SanitizedEvent:
    action: str
    event: AlarmEvent
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
