from __future__ import annotations

import json
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteLatencyEstimate:
    route: str
    seconds: float
    quantile: float
    sample_count: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "seconds": self.seconds,
            "quantile": self.quantile,
            "sample_count": self.sample_count,
            "source": self.source,
        }


class RouteLatencyTracker:
    """Thread-safe rolling successful Worker-call latency observations by route."""

    def __init__(self, *, window_size: int = 64) -> None:
        if int(window_size) <= 0:
            raise ValueError("route latency window_size must be positive")
        self.window_size = int(window_size)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._lock = threading.Lock()

    def record(self, route: str, duration_s: float) -> None:
        route = str(route or "default").strip() or "default"
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("route latency duration must be finite and non-negative")
        with self._lock:
            self._samples[route].append(duration)

    def estimate(
        self,
        route: str,
        *,
        quantile: float,
        minimum_samples: int,
        cold_start_s: float,
    ) -> RouteLatencyEstimate:
        route = str(route or "default").strip() or "default"
        quantile = float(quantile)
        minimum_samples = int(minimum_samples)
        cold_start_s = float(cold_start_s)
        if not 0 < quantile <= 1:
            raise ValueError("route latency quantile must be in (0, 1]")
        if minimum_samples <= 0:
            raise ValueError("route latency minimum_samples must be positive")
        if cold_start_s <= 0:
            raise ValueError("route latency cold_start_s must be positive")
        with self._lock:
            samples = sorted(self._samples.get(route, ()))
        if len(samples) < minimum_samples:
            return RouteLatencyEstimate(
                route=route,
                seconds=cold_start_s,
                quantile=quantile,
                sample_count=len(samples),
                source="cold_start",
            )
        rank = max(0, math.ceil(quantile * len(samples)) - 1)
        return RouteLatencyEstimate(
            route=route,
            seconds=float(samples[rank]),
            quantile=quantile,
            sample_count=len(samples),
            source="rolling_quantile",
        )

    def snapshot(self) -> dict[str, list[float]]:
        with self._lock:
            return {route: list(samples) for route, samples in sorted(self._samples.items())}


@dataclass(frozen=True)
class RouteTokenEstimate:
    route: str
    tokens: int
    quantile: float
    sample_count: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "tokens": self.tokens,
            "quantile": self.quantile,
            "sample_count": self.sample_count,
            "source": self.source,
        }


class RouteTokenTracker:
    """Thread-safe, optionally persistent Worker token observations by route."""

    def __init__(
        self,
        *,
        window_size: int = 64,
        path: str | Path | None = None,
    ) -> None:
        if int(window_size) <= 0:
            raise ValueError("route token window_size must be positive")
        self.window_size = int(window_size)
        self.path = Path(path) if path is not None else None
        self._samples: dict[str, deque[dict[str, int]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._lock = threading.Lock()
        self._load()

    def record(self, route: str, token_in: int, token_out: int) -> None:
        route = str(route or "default").strip() or "default"
        token_in = int(token_in)
        token_out = int(token_out)
        if token_in < 0 or token_out < 0:
            raise ValueError("route token usage must be non-negative")
        sample = {
            "token_in": token_in,
            "token_out": token_out,
            "total_tokens": token_in + token_out,
        }
        with self._lock:
            self._samples[route].append(sample)
            self._persist_locked()

    def estimate(
        self,
        route: str,
        *,
        quantile: float,
        minimum_samples: int,
        cold_start_tokens: int,
    ) -> RouteTokenEstimate:
        route = str(route or "default").strip() or "default"
        quantile = float(quantile)
        minimum_samples = int(minimum_samples)
        cold_start_tokens = int(cold_start_tokens)
        if not 0 < quantile <= 1:
            raise ValueError("route token quantile must be in (0, 1]")
        if minimum_samples <= 0:
            raise ValueError("route token minimum_samples must be positive")
        if cold_start_tokens <= 0:
            raise ValueError("route token cold_start_tokens must be positive")
        with self._lock:
            samples = sorted(int(item["total_tokens"]) for item in self._samples.get(route, ()))
        if len(samples) < minimum_samples:
            return RouteTokenEstimate(
                route=route,
                tokens=cold_start_tokens,
                quantile=quantile,
                sample_count=len(samples),
                source="cold_start",
            )
        rank = max(0, math.ceil(quantile * len(samples)) - 1)
        return RouteTokenEstimate(
            route=route,
            tokens=int(samples[rank]),
            quantile=quantile,
            sample_count=len(samples),
            source="rolling_quantile",
        )

    def snapshot(self) -> dict[str, list[dict[str, int]]]:
        with self._lock:
            return {
                route: [dict(sample) for sample in samples]
                for route, samples in sorted(self._samples.items())
            }

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"invalid route token state: {self.path}")
        routes = payload.get("routes", {})
        if not isinstance(routes, dict):
            raise ValueError(f"invalid route token routes: {self.path}")
        for route, raw_samples in routes.items():
            if not isinstance(raw_samples, list):
                raise ValueError(f"invalid route token samples for {route}")
            for raw in raw_samples[-self.window_size :]:
                if not isinstance(raw, dict):
                    raise ValueError(f"invalid route token sample for {route}")
                token_in = int(raw.get("token_in", 0))
                token_out = int(raw.get("token_out", 0))
                if token_in < 0 or token_out < 0:
                    raise ValueError(f"negative route token sample for {route}")
                self._samples[str(route)].append(
                    {
                        "token_in": token_in,
                        "token_out": token_out,
                        "total_tokens": token_in + token_out,
                    }
                )

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "window_size": self.window_size,
            "routes": {route: list(samples) for route, samples in sorted(self._samples.items())},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
