from __future__ import annotations

import time


class TerminalProgress:
    """Throttled, line-oriented progress suitable for long CLI jobs."""

    def __init__(self, label: str, updates: int = 100) -> None:
        self.label = label
        self.updates = max(1, updates)
        self.started_at = time.monotonic()
        self.last_reported = 0

    def __call__(
        self,
        completed: int,
        total: int,
        playlist_id: str,
        split_id: int,
        strategy: str,
    ) -> None:
        interval = max(1, total // self.updates)
        if completed < total and completed - self.last_reported < interval:
            return
        self.last_reported = completed
        elapsed = time.monotonic() - self.started_at
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = max(total - completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        percent = completed / total * 100 if total else 100.0
        print(
            f"[{self.label}] {completed}/{total} ({percent:5.1f}%) "
            f"split={split_id + 1} strategy={strategy} "
            f"elapsed={_duration(elapsed)} eta={_duration(eta)}",
            flush=True,
        )


def _duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"
