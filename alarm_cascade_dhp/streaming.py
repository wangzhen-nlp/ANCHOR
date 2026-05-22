import heapq

from alarm_cascade_dhp.config import StreamPolicyConfig
from alarm_cascade_dhp.types import SanitizedEvent


class AlarmStreamSanitizer:
    """Reorder and compress operational noise in a streaming alarm feed."""

    def __init__(self, config=None):
        self.config = config or StreamPolicyConfig()
        self._heap = []
        self._sequence = 0
        self._max_seen_ts = None
        self._last_emitted_ts = None
        self._last_raise_by_key = {}
        self._active_by_key = {}
        self._last_clear_by_key = {}

    def push(self, event):
        self._sequence += 1
        heapq.heappush(self._heap, (event.ts, self._sequence, event))
        self._max_seen_ts = event.ts if self._max_seen_ts is None else max(self._max_seen_ts, event.ts)
        watermark = self._max_seen_ts - self.config.reorder_lag_sec
        return self._drain_until(watermark)

    def flush(self):
        return self._drain_until(float("inf"))

    def pending_count(self):
        return len(self._heap)

    def _drain_until(self, watermark):
        output = []
        while self._heap and self._heap[0][0] <= watermark:
            _, _, event = heapq.heappop(self._heap)
            output.append(self._sanitize_ordered(event))
        return output

    def _sanitize_ordered(self, event):
        if (
            self._last_emitted_ts is not None
            and event.ts + self.config.late_tolerance_sec < self._last_emitted_ts
        ):
            return SanitizedEvent("skip", event, "late_after_reorder_watermark")

        self._last_emitted_ts = (
            event.ts if self._last_emitted_ts is None else max(self._last_emitted_ts, event.ts)
        )
        key = event.event_key or event.event_id
        if event.is_clear:
            return self._handle_clear(key, event)
        return self._handle_raise(key, event)

    def _handle_raise(self, key, event):
        previous_clear_ts = self._last_clear_by_key.get(key)
        if (
            previous_clear_ts is not None
            and 0 <= event.ts - previous_clear_ts <= self.config.flap_window_sec
        ):
            self._last_raise_by_key[key] = event.ts
            return SanitizedEvent("skip", event, "flap_reopen_compressed")

        previous_raise_ts = self._last_raise_by_key.get(key)
        if (
            key in self._active_by_key
            and previous_raise_ts is not None
            and 0 <= event.ts - previous_raise_ts <= self.config.duplicate_window_sec
        ):
            self._last_raise_by_key[key] = event.ts
            return SanitizedEvent("skip", event, "duplicate_raise_compressed")

        self._last_raise_by_key[key] = event.ts
        self._active_by_key[key] = event
        return SanitizedEvent("raise", event)

    def _handle_clear(self, key, event):
        self._last_clear_by_key[key] = event.ts
        active_event = self._active_by_key.pop(key, None)
        if active_event is None and not self.config.emit_orphan_clears:
            return SanitizedEvent("skip", event, "orphan_clear")
        return SanitizedEvent("clear", event)
