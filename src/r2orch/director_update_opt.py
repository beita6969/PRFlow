"""Speed/diagnostic helpers for the R2Orch Director update (training.py).

Everything here is opt-in plumbing.  Nothing changes the objective: the helpers
time phases, toggle gradient checkpointing for short calls and rewrite the LoRA
dropout modules.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from typing import Any

MATMUL_PRECISIONS = ("highest", "high", "medium")


class _Span:
    """Mutable counters handed to the body of ``PhaseClock.phase``."""

    __slots__ = ("tokens", "calls")

    def __init__(self, tokens: int, calls: int) -> None:
        self.tokens = int(tokens)
        self.calls = int(calls)


class NullSpan(_Span):
    def __init__(self) -> None:
        super().__init__(0, 0)


class PhaseClock:
    """Accumulate wall seconds, forwarded tokens and call counts per named phase.

    CUDA is synchronised at both edges so the numbers are real device time (the
    update is dominated by multi-hundred-millisecond kernels, so the few
    thousand synchronisations per update are noise).  With ``enabled=False``
    every method is a no-op.
    """

    def __init__(self, torch_module: Any, device: Any, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.torch = torch_module
        self.device = device
        self.cuda = bool(
            self.enabled
            and torch_module.cuda.is_available()
            and str(device).startswith("cuda")
        )
        self.seconds: dict[str, float] = defaultdict(float)
        self.tokens: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if self.cuda:
            # An asynchronous OOM must surface from the real call site, not from here.
            with suppress(RuntimeError):
                self.torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name: str, *, tokens: int = 0, calls: int = 1):
        if not self.enabled:
            yield NullSpan()
            return
        span = _Span(tokens, calls)
        self._sync()
        started = time.perf_counter()
        try:
            yield span
        finally:
            self._sync()
            self.seconds[name] += time.perf_counter() - started
            self.tokens[name] += span.tokens
            self.calls[name] += span.calls

    def drain(self) -> dict[str, dict[str, float]]:
        """Return and reset ``{"seconds", "tokens", "calls"}`` accumulated so far."""

        snapshot = {
            "seconds": {key: round(value, 4) for key, value in self.seconds.items()},
            "tokens": dict(self.tokens),
            "calls": dict(self.calls),
        }
        self.seconds.clear()
        self.tokens.clear()
        self.calls.clear()
        return snapshot


def lora_dropout_modules(model: Any) -> Iterable[tuple[Any, Any]]:
    """Yield ``(lora_layer, dropout_module_dict)`` for every PEFT LoRA layer."""

    import torch

    for module in model.modules():
        dropouts = getattr(module, "lora_dropout", None)
        if isinstance(dropouts, torch.nn.ModuleDict):
            yield module, dropouts


def effective_lora_dropout(model: Any) -> float | None:
    """Largest active LoRA dropout probability; ``None`` when the model has no LoRA layer."""

    import torch

    values: list[float] = []
    for _layer, dropouts in lora_dropout_modules(model):
        for dropout in dropouts.values():
            values.append(float(dropout.p) if isinstance(dropout, torch.nn.Dropout) else 0.0)
    return max(values) if values else None


def set_lora_dropout(model: Any, probability: float) -> int:
    """Force one dropout probability on every LoRA layer, whatever the saved adapter says.

    ``PeftModel.from_pretrained`` restores the dropout stored in the adapter
    config, so a resumed run would silently keep the old value.  Returns the
    number of rewritten adapter dropouts.
    """

    import torch

    if not 0.0 <= probability < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    changed = 0
    for _layer, dropouts in lora_dropout_modules(model):
        for name in list(dropouts.keys()):
            dropouts[name] = (
                torch.nn.Dropout(p=probability) if probability > 0 else torch.nn.Identity()
            )
            changed += 1
    return changed


class CheckpointingToggle:
    """Disable gradient checkpointing for short calls without touching the weights.

    ``GradientCheckpointingLayer.__call__`` consults ``self.gradient_checkpointing``
    at forward time only (a checkpointed layer's recompute reads the function it
    captured), so flipping the flag around one forward leaves the autograd graph
    of that call plain and every other call checkpointed.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._layers: list[Any] | None = None

    def layers(self) -> list[Any]:
        if self._layers is None:
            self._layers = [
                module
                for module in self.model.modules()
                if getattr(module, "gradient_checkpointing", False) is True
            ]
        return self._layers

    @contextmanager
    def suspended(self):
        layers = self.layers()
        for layer in layers:
            layer.gradient_checkpointing = False
        try:
            yield len(layers)
        finally:
            for layer in layers:
                layer.gradient_checkpointing = True
