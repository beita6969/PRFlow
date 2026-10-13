"""Opt-in two-GPU Director updates in isolated distributed workers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

from .training import (
    PolicyTrainingConfig,
    PolicyUpdateResult,
    TransformersGRPOTrainer,
    _atomic_json,
    _jsonable,
    load_training_batch,
)


class SubprocessSolverTrainer:
    def __init__(self, role, config, *, seed=0):
        if role != "solver":
            raise ValueError("distributed trainer is Solver-only")
        config.validate()
        self.config, self.seed = config, seed

    def update(self, batch, *, step):
        # The Director shards every PPO mini-batch across the two ranks inside
        # TransformersGRPOTrainer._update_r2orch (multi-step, global token-mean
        # denominators).
        if len(batch.samples) < 2:
            raise ValueError("distributed Solver requires at least two samples")
        root = self.config.checkpoint_root
        if (root / f"step-{step:08d}").exists():
            raise FileExistsError("refusing to overwrite an existing distributed checkpoint")
        run = root / "distributed-updates" / f"step-{step:08d}-{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True)
        _atomic_json(run / "batch.json", batch.to_dict())
        payload = _jsonable(asdict(self.config))
        payload["base_model_path"] = str(self.config.base_model_path.resolve())
        payload["checkpoint_root"] = str(root.resolve())
        _atomic_json(run / "request.json", dict(config=payload, seed=self.seed, step=step))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.data_parallel_gpu_ids))
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "r2orch.solver_parallel_trainer",
            str(run.resolve()),
        ]
        with (run / "workers.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = PolicyUpdateResult(**json.loads((run / "result.json").read_text()))
        destination = Path(result.checkpoint)
        for name in (
            "trainer_state.pt",
            "training_config.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            if not (destination / name).is_file():
                raise RuntimeError(f"incomplete distributed checkpoint: {name}")
        _atomic_json(destination / "training_config.json", payload)
        if self.config.publish_latest_checkpoint:
            _atomic_json(root / "latest.json", dict(path=str(destination), step=step))
        return result

    def close(self):
        pass


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def config_from_request(policy):
    """Rebuild a ``PolicyTrainingConfig`` from the JSON the launcher wrote."""

    policy = dict(policy)
    for key in ("checkpoint_root", "base_model_path"):
        policy[key] = Path(policy[key])
    for key in ("lora_target_modules", "data_parallel_gpu_ids"):
        if key in policy:
            policy[key] = tuple(policy[key])
    return PolicyTrainingConfig(**policy)


def _agree(dist, value):
    """Every rank must hold the same value (policy state, step counters)."""

    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != values[0] for item in values):
        raise RuntimeError("distributed policy state mismatch")


def _worker(run, rank, config, batch, request):
    """Sample-level data parallelism of the Director update (see director_data_parallel).

    Every rank holds a full bf16 base + fp32 LoRA replica, processes a cost-balanced
    shard of each PPO mini-batch and SUMS gradients before the replicated optimizer
    step.  Rank 0 writes the checkpoint and the (already global) update result.
    """

    import torch
    import torch.distributed as dist

    from .director_data_parallel import TorchDistributedGroup

    trainer = None
    try:
        local_config = replace(
            config,
            device=f"cuda:{rank}",
            data_parallel_gpu_ids=(),
            publish_latest_checkpoint=False,
        )
        trainer = TransformersGRPOTrainer("solver", local_config, seed=request["seed"])
        trainer._dp = TorchDistributedGroup()
        initial = parameter_hash(trainer.model)
        _agree(dist, (initial, trainer.optimizer_step_count))
        before_step = trainer.optimizer_step_count
        dist.barrier()
        result = trainer.update(batch, step=request["step"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        final = parameter_hash(trainer.model)
        _agree(dist, (final, trainer.optimizer_step_count))
        if trainer.optimizer_step_count != before_step + result.optimizer_steps:
            raise RuntimeError("optimizer state did not advance by the update's step count")
        if rank == 0:
            _atomic_json(run / "result.json", _jsonable(asdict(result)))
        dist.barrier()
    finally:
        if trainer is not None:
            trainer.close()


def worker(run):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=50))
    try:
        request = json.loads((run / "request.json").read_text())
        config = config_from_request(request["config"])
        batch = load_training_batch(run / "batch.json")
        _worker(run, rank, config, batch, request)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
