"""Sample-level data parallelism for the R2Orch Director update (opt-in).

The Director loss is a token-mean over one PPO mini-batch (``/ token_count`` is applied
per sample before ``backward``), so splitting a mini-batch's samples over ranks
and SUMMING the per-rank gradients reproduces the single-process gradient
exactly; no rank ever re-normalises.  Replicas stay in sync because every rank
then applies the identical clip + optimizer step to identical gradients.

This module only holds the model-free pieces: the deterministic shard planner and
the collective-communication adapter.  The update itself lives in
``TransformersGRPOTrainer._update_r2orch``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def sample_cost(sample: Any) -> float:
    """Relative compute of one sample: tokens forwarded per pass plus a quadratic attention term.

    ``timeline`` evaluation processes the last call once when every earlier call is a
    prefix of it; otherwise every call is evaluated separately.
    """

    from .policy_timeline import timeline_positions

    lengths = [len(call.token_ids) for call in sample.policy_calls] or [len(sample.token_ids)]
    try:
        merged = (
            bool(sample.policy_calls)
            and timeline_positions(sample.policy_calls, max(lengths) + 1) is not None
        )
    except ValueError:
        merged = False
    if merged:
        lengths = [lengths[-1]]
    return float(sum(n + n * n / 8192 for n in lengths))


def plan_shards(costs: Sequence[float], indices: Sequence[int], world: int) -> list[list[int]]:
    """Split ``indices`` over ``world`` ranks, balancing ``costs`` (greedy longest-first).

    Deterministic (every rank computes the same plan); ranks may receive nothing.
    """

    if world < 1:
        raise ValueError("world size must be positive")
    shards: list[list[int]] = [[] for _ in range(world)]
    loads = [0.0] * world
    for index in sorted(indices, key=lambda i: (-costs[i], i)):
        rank = min(range(world), key=lambda r: (loads[r], len(shards[r]), r))
        shards[rank].append(index)
        loads[rank] += costs[index]
    return [sorted(shard) for shard in shards]


class TorchDistributedGroup:
    """``torch.distributed`` adapter (process group must already be initialised)."""

    def __init__(self) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialised")
        self.dist = dist
        self.rank = int(dist.get_rank())
        self.world = int(dist.get_world_size())
        self.backend = str(dist.get_backend())

    def all_gather_object(self, value: Any) -> list[Any]:
        gathered: list[Any] = [None] * self.world
        self.dist.all_gather_object(gathered, value)
        return gathered

    def barrier(self) -> None:
        self.dist.barrier()

    def all_reduce_sum_gradients(self, model: Any) -> None:
        """SUM gradients over ranks (None stays None only if no rank produced a gradient)."""

        import torch

        dist = self.dist
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("no trainable parameters to synchronise")
        device = parameters[0].device
        comm_device = device if self.backend == "nccl" else torch.device("cpu")
        used = torch.tensor(
            [p.grad is not None for p in parameters], device=comm_device, dtype=torch.int32
        )
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        active = [p for p, flag in zip(parameters, used.tolist(), strict=True) if flag]
        if not active:
            return
        flat = torch.cat(
            [
                (
                    p.grad.detach().float()
                    if p.grad is not None
                    else torch.zeros(p.shape, device=device, dtype=torch.float32)
                ).reshape(-1)
                for p in active
            ]
        ).to(comm_device)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("non-finite synchronised gradient")
        flat = flat.to(device)
        offset = 0
        for parameter in active:
            gradient = flat[offset : offset + parameter.numel()].view_as(parameter)
            gradient = gradient.to(parameter.dtype)
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                parameter.grad.copy_(gradient)
            offset += parameter.numel()
