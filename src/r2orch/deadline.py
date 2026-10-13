from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class WorkerWallClockLimitExceeded(RuntimeError):
    """An end-to-end rollout deadline violation."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        stage: str,
        elapsed_s: float,
        idle_s: float,
    ) -> None:
        self.reason = reason
        self.stage = stage
        self.elapsed_s = float(elapsed_s)
        self.idle_s = float(idle_s)
        self.partial_state: dict[str, Any] | None = None
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "elapsed_s": self.elapsed_s,
            "idle_s": self.idle_s,
        }


@dataclass
class RolloutDeadline:
    """A hard rollout bound plus a control-idle clock and request budget.

    Wall-clock diagnostics always advance. When explicitly enabled, the task
    budget excludes the union of failed request waits, with a separate absolute
    safety limit of 900 seconds. The no-progress clock measures
    controller idle time and pauses while backend requests queue or execute.
    """

    total_timeout_s: float
    no_progress_timeout_s: float
    request_timeout_s: float
    request_timeout_overrides_s: dict[str, float] = field(default_factory=dict)
    started_monotonic: float = field(default_factory=time.monotonic)
    cancellation_event: threading.Event | None = field(default=None, repr=False)
    cancellation_reason: str = "backend_circuit_open"
    exclude_failed_request_time: bool = False
    absolute_wall_timeout_s: float = 900.0
    _failed_intervals: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _last_progress_active_elapsed_s: float = field(default=0.0, init=False, repr=False)
    _last_progress_kind: str = field(default="rollout_started", init=False, repr=False)
    _paused_total_s: float = field(default=0.0, init=False, repr=False)
    _pause_started_monotonic: float | None = field(default=None, init=False, repr=False)
    _pause_depth: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            min(
                self.total_timeout_s,
                self.no_progress_timeout_s,
                self.request_timeout_s,
                self.absolute_wall_timeout_s,
            )
            <= 0
        ):
            raise ValueError("rollout deadline values must be positive")
        if any(value <= 0 for value in self.request_timeout_overrides_s.values()):
            raise ValueError("request timeout overrides must be positive")

    def record_failed_request(self, started: float, finished: float) -> None:
        """Credit the union of failed request intervals, never summed parallel waits."""
        if not self.exclude_failed_request_time:
            return
        with self._lock:
            intervals = sorted(
                [*self._failed_intervals, (max(started, self.started_monotonic), finished)]
            )
            merged: list[tuple[float, float]] = []
            for begin, end in intervals:
                if end <= begin:
                    continue
                if merged and begin <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((begin, end))
            self._failed_intervals = merged

    def _failed_elapsed_locked(self) -> float:
        return sum(end - begin for begin, end in self._failed_intervals)

    def _active_elapsed_locked(self, now: float) -> float:
        paused = self._paused_total_s
        if self._pause_started_monotonic is not None:
            paused += now - self._pause_started_monotonic
        return max(0.0, now - self.started_monotonic - paused)

    def _timing_locked(self, now: float) -> tuple[float, float, str, bool]:
        elapsed = now - self.started_monotonic
        active_elapsed = self._active_elapsed_locked(now)
        idle = active_elapsed - self._last_progress_active_elapsed_s
        return elapsed, max(0.0, idle), self._last_progress_kind, self._pause_depth > 0

    def mark_progress(self, kind: str) -> None:
        with self._lock:
            self._last_progress_active_elapsed_s = self._active_elapsed_locked(time.monotonic())
            self._last_progress_kind = str(kind)

    @contextmanager
    def pause_no_progress(self, kind: str = "active_request") -> Iterator[None]:
        """Pause only the control-idle clock; never pause the hard deadline."""

        del kind
        with self._lock:
            if self._pause_depth == 0:
                self._pause_started_monotonic = time.monotonic()
            self._pause_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._pause_depth -= 1
                if self._pause_depth < 0:
                    raise RuntimeError("deadline pause depth became negative")
                if self._pause_depth == 0:
                    now = time.monotonic()
                    assert self._pause_started_monotonic is not None
                    self._paused_total_s += now - self._pause_started_monotonic
                    self._pause_started_monotonic = None

    def check(self, stage: str) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed, idle, last_kind, idle_paused = self._timing_locked(now)
            effective = elapsed - self._failed_elapsed_locked()
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            message = "rollout collection was cancelled after the backend failure circuit opened"
            raise WorkerWallClockLimitExceeded(
                message + f" ({elapsed:.1f}s at {stage})",
                reason=self.cancellation_reason,
                stage=stage,
                elapsed_s=elapsed,
                idle_s=idle,
            )
        if self.exclude_failed_request_time and elapsed >= self.absolute_wall_timeout_s:
            raise WorkerWallClockLimitExceeded(
                "rollout reached its separate absolute safety limit",
                reason="absolute_wall_deadline",
                stage=stage,
                elapsed_s=elapsed,
                idle_s=idle,
            )
        if effective >= self.total_timeout_s:
            raise WorkerWallClockLimitExceeded(
                "complete rollout exceeded its effective execution budget "
                f"({effective:.1f}/{self.total_timeout_s:.1f}s at {stage}; wall={elapsed:.1f}s)",
                reason="hard_deadline",
                stage=stage,
                elapsed_s=elapsed,
                idle_s=idle,
            )
        if not idle_paused and idle >= self.no_progress_timeout_s:
            raise WorkerWallClockLimitExceeded(
                "complete rollout made no effective progress within its wall-clock budget "
                f"({idle:.1f}/{self.no_progress_timeout_s:.1f}s at {stage}; "
                f"last_progress={last_kind})",
                reason="no_progress",
                stage=stage,
                elapsed_s=elapsed,
                idle_s=idle,
            )

    def remaining_s(self, stage: str) -> float:
        self.check(stage)
        now = time.monotonic()
        with self._lock:
            elapsed, idle, _, idle_paused = self._timing_locked(now)
            hard_remaining = self.total_timeout_s - elapsed + self._failed_elapsed_locked()
            if self.exclude_failed_request_time:
                hard_remaining = min(hard_remaining, self.absolute_wall_timeout_s - elapsed)
            progress_remaining = (
                hard_remaining if idle_paused else self.no_progress_timeout_s - idle
            )
        return max(0.001, min(hard_remaining, progress_remaining))

    def hard_remaining_s(self, stage: str) -> float:
        """Return total rollout time left, independent of the refreshable idle clock."""

        self.check(stage)
        with self._lock:
            remaining = self.total_timeout_s - (time.monotonic() - self.started_monotonic)
            remaining += self._failed_elapsed_locked()
            if self.exclude_failed_request_time:
                remaining = min(
                    remaining,
                    self.absolute_wall_timeout_s - (time.monotonic() - self.started_monotonic),
                )
        return max(0.001, remaining)

    def request_budget_s(self, stage: str, *, route: str = "") -> float:
        request_timeout_s = self.request_timeout_overrides_s.get(str(route), self.request_timeout_s)
        return max(0.001, min(request_timeout_s, self.hard_remaining_s(stage)))

    def for_request(
        self, role: str, configured_timeout_s: float
    ) -> RolloutDeadline | DirectorRequestDeadline:
        """Isolate Director request limits without mutating the shared Worker clock."""
        if role != "graph-director":
            return self
        if configured_timeout_s <= 0:
            raise ValueError("Director request timeout must be positive")
        return DirectorRequestDeadline(self, float(configured_timeout_s))

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            elapsed, idle, _, idle_paused = self._timing_locked(now)
            return {
                "elapsed_s": elapsed,
                "failed_request_wait_s": self._failed_elapsed_locked(),
                "effective_elapsed_s": max(0.0, elapsed - self._failed_elapsed_locked()),
                "exclude_failed_request_time": self.exclude_failed_request_time,
                "absolute_wall_limit_s": (
                    self.absolute_wall_timeout_s
                    if self.exclude_failed_request_time
                    else self.total_timeout_s
                ),
                "idle_s": idle,
                "last_progress_kind": self._last_progress_kind,
                "no_progress_paused": idle_paused,
                "paused_total_s": self._paused_total_s,
                "total_timeout_s": self.total_timeout_s,
                "no_progress_timeout_s": self.no_progress_timeout_s,
                "request_timeout_s": self.request_timeout_s,
                "request_timeout_overrides_s": dict(self.request_timeout_overrides_s),
            }


@dataclass(frozen=True)
class DirectorRequestDeadline:
    """Request-local view: independent Director timeout, shared hard/idle clocks.

    No state is copied or reset. Queueing and bounded retries still consume the
    same logical request budget in the gateway. Worker/Judge requests keep the
    original RolloutDeadline and route-specific overrides.
    """

    parent: RolloutDeadline
    configured_timeout_s: float

    def request_budget_s(self, stage: str, *, route: str = "") -> float:
        del route
        return max(0.001, min(self.configured_timeout_s, self.parent.hard_remaining_s(stage)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parent, name)
