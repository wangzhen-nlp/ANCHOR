import sys
import time


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    def __init__(self, total, label, width=30, min_interval=0.2):
        self.total = max(int(total), 0)
        self.label = label
        self.width = width
        self.min_interval = min_interval
        self.current = 0
        self.start_time = time.time()
        self.last_render_time = 0.0
        self._render(force=True)

    def update(self, step=1):
        self.current += step
        self._render()

    def set(self, current):
        self.current = max(0, int(current))
        self._render()

    def close(self):
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, force=False):
        now = time.time()
        if not force and (now - self.last_render_time) < self.min_interval and self.current < self.total:
            return

        self.last_render_time = now
        elapsed = max(now - self.start_time, 1e-6)

        if self.total > 0:
            ratio = min(self.current / self.total, 1.0)
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            percent = ratio * 100
            speed = self.current / elapsed
            remaining = max(self.total - self.current, 0)
            eta_sec = (remaining / speed) if speed > 0 else 0
            eta_str = _format_duration(eta_sec)
            msg = (
                f"\r{self.label}: [{bar}] {self.current}/{self.total} "
                f"{percent:6.2f}% ({speed:.1f}/s, ETA {eta_str})"
            )
        else:
            msg = f"\r{self.label}: {self.current}"

        sys.stdout.write(msg)
        sys.stdout.flush()
