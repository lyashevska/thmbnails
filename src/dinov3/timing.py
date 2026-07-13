"""Execution-time tracking for DINOv3 pipeline scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {rem}s"


@dataclass
class RunTimer:
    started_at: str = field(default_factory=utc_now)
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start

    def manifest_dict(self, *, finished: bool = True) -> dict[str, Any]:
        elapsed = self.elapsed_seconds()
        data: dict[str, Any] = {
            "started_at": self.started_at,
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": format_duration(elapsed),
        }
        if finished:
            data["finished_at"] = utc_now()
        return data

    def log_line(self, label: str = "Elapsed") -> str:
        return f"{label}: {format_duration(self.elapsed_seconds())} ({self.elapsed_seconds():.1f}s)"