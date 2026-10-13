from __future__ import annotations

import gc
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .director_update_opt import (
    MATMUL_PRECISIONS,
    CheckpointingToggle,
    NullSpan,
    PhaseClock,
    effective_lora_dropout,
    set_lora_dropout,
)
from .qwen_compat import load_training_model
from .rollouts import (
    TokenizedPolicyCall,
    TrainingBatch,
    TrainingSample,
)

_NULL_PHASE = nullcontext(NullSpan())


@dataclass(frozen=True)
class PolicyTrainingConfig:
    base_model_path: Path
    checkpoint_root: Path
    learning_rate: float = 1e-5
    epochs: int = 1
    clip_range: float = 0.2
    kl_coefficient: float = 0.005
    entropy_coefficient: float = 0.0
    loss_variant: str = "kl005_entropy0"
    # SESA/verl separates the logical PPO mini-batch from the device micro-batch.
    # Keeping the defaults at one preserves the small standalone trainer API; the
    # experiment CLI supplies the SESA-style 64/1 recipe.
    mini_batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_micro_batch_tokens: int = 16_384
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 10
    warmup_start_factor: float = 0.0
    total_optimizer_steps: int = 300
    learning_rate_schedule: str = "cosine"
    max_sequence_length: int = 4096
    dtype: str = "bfloat16"
    device: str = "cuda"
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    gradient_checkpointing: bool = True
    # Optional memory-only execution policy.  Saved autograd tensors (including
    # tensors produced by gradient-checkpoint recomputation) are staged on CPU;
    # the objective, sample set, and optimizer step are unchanged.
    activation_cpu_offload: bool = False
    # Zero preserves all-call offload.  A positive threshold stages only
    # micro-batches containing at least one sequence of this length, allowing
    # short calls to remain entirely on GPU while protecting true long-call
    # peaks.
    activation_cpu_offload_min_tokens: int = 0
    # Release raw-call graphs promptly; logical optimizer boundaries are unchanged.
    raw_policy_backward_mode: str = "micro"
    # Parallel role training writes checkpoints first and publishes the two role
    # pointers only after both updates succeed.
    publish_latest_checkpoint: bool = True
    data_parallel_gpu_ids: tuple[int, ...] = ()
    # The Proposer update needs several independent lists to estimate a stable
    # direction.  A partial collection with only one or two valid pairs can
    # otherwise turn one large reward difference into the whole Proposer update.
    # The guard is consumed only by the Proposer pair objective.
    min_proposer_pair_samples: int = 4
    # Proposer pair update only (``r2orch.proposer``): optimizer steps per cycle and the
    # gradient-norm cap.  Zero keeps the shared ``ppo_mini_batches`` / ``max_grad_norm``.
    proposer_optimizer_steps: int = 0
    proposer_max_grad_norm: float = 0.0
    proposer_kl_beta: float = 0.05
    proposer_gap_clip: float = 0.0
    tis_cap: float = 2.0
    ppo_mini_batches: int = 2
    clip_low: float = 0.2
    clip_high: float = 0.28
    # --- Director-update speed/diagnostic knobs.  Learning-dynamics notes sit next
    # to each field.
    # Take log pi_old for the FIRST PPO mini-batch from the differentiated forward
    # itself (detached) instead of a separate no-grad pass.  Exact only when the
    # LoRA dropout is 0 (otherwise it falls back and says so in the metrics).  The
    # remaining mini-batches still need log pi_old at theta-bar, i.e. before the
    # first optimizer step, so their no-grad pass stays.
    fuse_old_logprobs: bool = False
    # Frozen-reference (LoRA-off) KL pass cadence: 1 = every sample,
    # k > 1 = every k-th sample (counted within each scaffold condition, offset
    # rotating with the update step) with its KL weight multiplied by k, i.e. an
    # unbiased subsample of an already tiny term (kl coefficient 0.001-0.005);
    # 0 = never (KL term and KL monitor dropped).
    kl_reference_stride: int = 1
    # Upper bound of one token's sampled KL term in the Director loss (0 = unbounded).
    # 10 = verl's low_var_kl clamp: tokens past it keep their policy gradient
    # but no KL gradient, so a single near-impossible sampled token cannot dominate an update.
    kl_term_cap: float = 0.0
    # torch.set_float32_matmul_precision for the fp32 LoRA matmuls (PEFT keeps the
    # adapter in fp32 even with a bf16 base).  "high" = TF32.
    matmul_precision: str = "highest"
    # Rewrite every LoRA dropout after the adapter is created/restored (a resumed
    # adapter otherwise keeps the dropout stored in its adapter_config.json).
    lora_dropout_override: float | None = None
    # Do not gradient-checkpoint calls shorter than this many tokens (0 = always
    # checkpoint).  Saves the recompute forward at the price of activation memory.
    checkpoint_skip_below_tokens: int = 0
    # Per-phase wall-time / token / memory telemetry in step_metrics.
    phase_timing: bool = True

    def validate(self) -> None:
        if self.kl_reference_stride < 0:
            raise ValueError("kl_reference_stride must be non-negative")
        if self.kl_term_cap < 0:
            raise ValueError("kl_term_cap must be non-negative")
        if self.matmul_precision not in MATMUL_PRECISIONS:
            raise ValueError(f"matmul_precision must be one of {MATMUL_PRECISIONS}")
        if self.lora_dropout_override is not None and not 0.0 <= self.lora_dropout_override < 1.0:
            raise ValueError("lora_dropout_override must be in [0, 1)")
        if self.checkpoint_skip_below_tokens < 0:
            raise ValueError("checkpoint_skip_below_tokens must be non-negative")
        if self.min_proposer_pair_samples < 1:
            raise ValueError("min_proposer_pair_samples must be positive")
        if self.proposer_optimizer_steps < 0 or self.proposer_max_grad_norm < 0:
            raise ValueError("proposer optimizer steps and gradient cap must be non-negative")
        if self.ppo_mini_batches <= 0 or self.tis_cap <= 0:
            raise ValueError("Director loss needs positive mini-batches and TIS cap")
        if self.learning_rate_schedule not in {"cosine", "constant"}:
            raise ValueError("learning_rate_schedule must be cosine or constant")
        if self.data_parallel_gpu_ids and (
            len(self.data_parallel_gpu_ids) != 2
            or len(set(self.data_parallel_gpu_ids)) != 2
            or min(self.data_parallel_gpu_ids) < 0
        ):
            raise ValueError("Solver data parallel requires two distinct physical GPUs")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be in (0, 1)")
        if (
            min(
                self.mini_batch_size,
                self.micro_batch_size,
                self.gradient_accumulation_steps,
            )
            <= 0
        ):
            raise ValueError("mini-batch, micro-batch, and gradient accumulation must be positive")
        if self.micro_batch_size > self.mini_batch_size:
            raise ValueError("micro_batch_size may not exceed mini_batch_size")
        if self.max_micro_batch_tokens <= 0:
            raise ValueError("max_micro_batch_tokens must be positive")
        if self.max_micro_batch_tokens < self.max_sequence_length:
            raise ValueError(
                "max_micro_batch_tokens must fit at least one max_sequence_length sample"
            )
        if self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("weight_decay and warmup_steps must be non-negative")
        if not math.isfinite(self.warmup_start_factor) or not 0 <= self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be finite and in [0, 1]")
        if self.total_optimizer_steps <= 0 or self.max_sequence_length < 2:
            raise ValueError("total_optimizer_steps must be positive and sequence length >= 2")
        if self.activation_cpu_offload_min_tokens < 0:
            raise ValueError("activation_cpu_offload_min_tokens must be non-negative")
        if self.raw_policy_backward_mode not in ("micro", "call", "timeline"):
            raise ValueError("raw_policy_backward_mode must be micro, call or timeline")
        if min(self.kl_coefficient, self.entropy_coefficient) < 0:
            raise ValueError("KL and entropy coefficients must be non-negative")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")


def r2orch_policy_config(config: PolicyTrainingConfig, r2orch: Any) -> PolicyTrainingConfig:
    """Map ``config.r2orch`` (``director.*`` and ``lora.*``) onto one role."""

    director, lora, proposer = r2orch.director, r2orch.lora, r2orch.proposer
    return replace(
        config,
        proposer_optimizer_steps=int(proposer.optimizer_steps_per_cycle),
        proposer_max_grad_norm=float(proposer.max_grad_norm),
        proposer_kl_beta=float(proposer.kl_beta),
        proposer_gap_clip=float(proposer.gap_clip),
        clip_range=float(director.clip_low),
        kl_coefficient=float(director.kl),
        tis_cap=float(director.tis_cap),
        ppo_mini_batches=int(director.ppo_mini_batches),
        clip_low=float(director.clip_low),
        clip_high=float(director.clip_high),
        lora_rank=int(lora.rank),
        lora_alpha=int(lora.alpha),
        lora_dropout=float(lora.dropout),
        lora_target_modules=tuple(str(name) for name in lora.target_modules),
    )


def even_mini_batches(count: int, parts: int) -> tuple[tuple[int, ...], ...]:
    """Split ``range(count)`` into ``parts`` contiguous, near-equal, non-empty chunks."""

    parts = max(1, min(int(parts), count))
    base, extra = divmod(count, parts)
    chunks, start = [], 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        chunks.append(tuple(range(start, start + size)))
        start += size
    return tuple(chunk for chunk in chunks if chunk)


@dataclass(frozen=True)
class AlternatingTrainingConfig:
    proposer: PolicyTrainingConfig
    solver: PolicyTrainingConfig
    state_path: Path
    seed: int = 0
    parallel_roles: bool = False

    def validate(self) -> None:
        self.proposer.validate()
        self.solver.validate()
        if self.proposer.data_parallel_gpu_ids:
            raise ValueError("data parallel is supported for Solver only")
        if self.parallel_roles and self.solver.data_parallel_gpu_ids:
            raise ValueError("two-card Solver requires sequential role updates")

        def batch_shape(policy: PolicyTrainingConfig) -> tuple[int, int, int]:
            return (
                policy.mini_batch_size,
                policy.micro_batch_size,
                policy.gradient_accumulation_steps,
                policy.max_micro_batch_tokens,
            )

        if batch_shape(self.proposer) != batch_shape(self.solver):
            raise ValueError(
                "proposer and solver must use the same mini-batch, micro-batch, and "
                "gradient-accumulation configuration; they consume different role batches, "
                "not different batch configurations"
            )
        if self.parallel_roles and self.proposer.device == self.solver.device:
            raise ValueError("parallel role training requires distinct proposer and solver devices")


@dataclass(frozen=True)
class PolicyUpdateResult:
    role: str
    step: int
    loss: float
    policy_loss: float
    kl: float
    masked_tokens: int
    checkpoint: str
    entropy: float = 0.0
    samples: int = 0
    optimizer_steps: int = 0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    clip_fraction: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    duration_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    step_metrics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    kl_loss_contribution: float = 0.0
    entropy_loss_contribution: float = 0.0
    loss_variant: str = ""
    reference_provenance: str = ""
    status: str = "updated"
    skip_reason: str = ""
    cycle: int | None = None


@dataclass
class AlternatingTrainingState:
    cycle: int = 0
    global_step: int = 0
    proposer_step: int = 0
    solver_step: int = 0
    proposer_checkpoint: str = ""
    solver_checkpoint: str = ""
    phase: str = "proposer"
    history: list[dict[str, Any]] = field(default_factory=list)
    active_batch_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AlternatingTrainingState:
        return cls(**payload)


class PolicyTrainer(Protocol):
    def update(self, batch: TrainingBatch, *, step: int) -> PolicyUpdateResult: ...

    def close(self) -> None: ...


class UnsafeTrainingBatchError(ValueError):
    """Training was stopped before either optimizer because the batch is unsafe."""


def training_device_for_gpu(physical_gpu_id: int, *, visible_devices: str | None = None) -> str:
    """Resolve a configured physical GPU to its process-local CUDA index."""

    if physical_gpu_id < 0:
        raise ValueError("physical_gpu_id must be non-negative")
    value = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if value is None or not value.strip():
        return f"cuda:{physical_gpu_id}"
    try:
        visible = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain numeric GPU ids") from exc
    if physical_gpu_id not in visible:
        raise ValueError(
            f"configured physical GPU {physical_gpu_id} is not visible in CUDA_VISIBLE_DEVICES={value}"
        )
    return f"cuda:{visible.index(physical_gpu_id)}"


def training_batch_from_dict(payload: dict[str, Any]) -> TrainingBatch:
    samples = tuple(
        TrainingSample(
            rollout_id=str(item["rollout_id"]),
            task_id=str(item["task_id"]),
            token_ids=tuple(int(value) for value in item["token_ids"]),
            action_mask=tuple(int(value) for value in item["action_mask"]),
            reward=float(item["reward"]),
            advantage=float(item["advantage"]),
            canonical_graph_key=str(item.get("canonical_graph_key", "")),
            graph_features=tuple(float(value) for value in item.get("graph_features", [])),
            metadata=dict(item.get("metadata", {})),
            policy_calls=tuple(
                TokenizedPolicyCall(
                    call_id=str(call["call_id"]),
                    token_ids=tuple(int(value) for value in call["token_ids"]),
                    action_mask=tuple(int(value) for value in call["action_mask"]),
                    behavior_log_probs=tuple(
                        float(value) for value in call.get("behavior_log_probs", ())
                    ),
                    action_token_span=(
                        tuple(call["action_token_span"])
                        if call.get("action_token_span") is not None
                        else None
                    ),
                    relation_token_span=(
                        tuple(call["relation_token_span"])
                        if call.get("relation_token_span") is not None
                        else None
                    ),
                    metadata=dict(call.get("metadata", {})),
                )
                for call in item.get("policy_calls", ())
            ),
        )
        for item in payload.get("samples", [])
    )
    return TrainingBatch(
        role=str(payload["role"]),
        samples=samples,
        objective=str(payload.get("objective", "masked_graph_level_grpo")),
        optimizer_steps=0,
        metadata=dict(payload.get("metadata", {})),
    )


def load_training_batch(path: str | Path) -> TrainingBatch:
    path = Path(path)
    batch = training_batch_from_dict(json.loads(path.read_text(encoding="utf-8")))
    if batch.objective != "proposer_pair" or not batch.samples:
        return batch
    from .proposer_reward import filter_proposer_pair_task_bindings

    manifest = path.parent / "tasks.jsonl"
    current_tasks = (
        [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if manifest.exists()
        else []
    )
    return filter_proposer_pair_task_bindings(batch, current_tasks=current_tasks)


def token_advantages(sample: TrainingSample) -> tuple[float, ...]:
    """Use the graph-level credit for every token."""

    values = [float(sample.advantage)] * len(sample.token_ids)
    return tuple(values)


def _is_relation_call(call: TokenizedPolicyCall) -> bool:
    return call.relation_token_span is not None and bool(call.metadata.get("relation_policy"))


def _sampled_kl_terms(current_log_probs: Any, reference_log_probs: Any, cap: float = 0.0) -> Any:
    """Compute the sampled KL estimator in fp32 to avoid bf16 cancellation.

    ``cap > 0`` bounds every token's term, as verl's ``low_var_kl`` does at 10.  The k3 term
    grows like exp(log ref - log pi): one sampled token the policy has all but ruled out
    (log pi = -12 where the reference gives -3) is worth thousands of ordinary tokens and sets
    the direction of the whole update.
    """

    log_ratio = (reference_log_probs.float() - current_log_probs.float()).clamp(-20.0, 20.0)
    terms = log_ratio.exp() - log_ratio - 1.0
    return terms.clamp(max=cap) if cap > 0 else terms


def _categorical_kl(current_log_probs: Any, reference_log_probs: Any) -> Any:
    """Compute exact finite-support KL in fp32 even when model logits are bf16."""

    current = current_log_probs.float()
    reference = reference_log_probs.float()
    return (current.exp() * (current - reference)).sum()


class TransformersGRPOTrainer:
    """LoRA trainer for the Director and Proposer objectives (one checkpoint chain per role)."""

    def __init__(self, role: str, config: PolicyTrainingConfig, *, seed: int = 0) -> None:
        config.validate()
        self.role = role
        self.config = config
        self.seed = int(seed)
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'train' extra for parameter training") from exc
        self.torch = torch
        # Model/kernel initialization may consult the thread's current CUDA
        # device before model.to(). Bind it to the allocated role GPU rather
        # than accidentally creating a context on the default physical GPU 0.
        if str(config.device).startswith("cuda"):
            torch.cuda.set_device(config.device)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        init_started = time.perf_counter()
        latest = self._latest_checkpoint()
        source = config.base_model_path
        dtype = getattr(torch, config.dtype)
        self.model = load_training_model(source, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(latest or config.base_model_path), trust_remote_code=True
        )
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError("PEFT is required") from exc
        if latest:
            self.model = PeftModel.from_pretrained(self.model, str(latest), is_trainable=True)
        else:
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    task_type="CAUSAL_LM",
                    target_modules=list(config.lora_target_modules),
                ),
            )
        if config.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.to(config.device)
        if role == "solver" and config.raw_policy_backward_mode == "timeline":
            from .policy_timeline import stabilize_gated_delta_projections

            stabilize_gated_delta_projections(self.model)
            if config.dtype == "float32":
                from .policy_timeline import enable_expanded_fp32_sdpa

                enable_expanded_fp32_sdpa(self.model)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = _build_learning_rate_scheduler(self.optimizer, config)
        self.optimizer_step_count = 0
        self._restore_optimizer(latest or source)
        self._active_cuda_rng_state = (
            torch.cuda.get_rng_state(config.device)
            if torch.cuda.is_available() and str(config.device).startswith("cuda")
            else None
        )
        self._apply_runtime_options()
        self._model_load_seconds = time.perf_counter() - init_started

    def _apply_runtime_options(self) -> dict[str, Any]:
        """Apply the execution-only knobs of ``PolicyTrainingConfig`` to the live model."""

        config = self.config
        applied: dict[str, Any] = {"matmul_precision": config.matmul_precision}
        # "highest" is torch's default; it is set explicitly as well.
        self.torch.set_float32_matmul_precision(config.matmul_precision)
        if config.lora_dropout_override is not None:
            applied["lora_dropout_rewritten"] = set_lora_dropout(
                self.model, config.lora_dropout_override
            )
        self._checkpoint_toggle = CheckpointingToggle(self.model)
        applied["lora_dropout_effective"] = effective_lora_dropout(self.model)
        self._runtime_options = applied
        return applied

    def activate_rng(self) -> None:
        """Restore this role's device RNG after the other role is initialized."""

        if self._active_cuda_rng_state is not None:
            self.torch.cuda.set_rng_state(self._active_cuda_rng_state, self.config.device)

    def update(self, batch: TrainingBatch, *, step: int) -> PolicyUpdateResult:
        if batch.role != self.role:
            raise ValueError(f"expected {self.role} batch, got {batch.role}")
        if batch.objective == "proposer_pair":
            from .proposer_pair_training import proposer_pair_update

            return proposer_pair_update(self, batch, step=step)
        if batch.objective == "r2orch_director":
            return self._update_r2orch(batch, step=step)
        raise ValueError(f"unsupported training objective: {batch.objective}")

    # ------------------------------------------------------------------
    # R2Orch Director objective.

    def _update_r2orch(self, batch: TrainingBatch, *, step: int) -> PolicyUpdateResult:
        """Token-mean PPO on the outcome advantage with TIS weights and a KL anchor.

        With ``self._dp`` (a ``TorchDistributedGroup``) every rank holds a full replica and
        processes a disjoint, cost-balanced shard of each PPO mini-batch.  The loss of
        a sample is already divided by the *global* mini-batch token count, so SUMMING
        the rank gradients before the (replicated) clip + optimizer step is exactly the
        single-process update.
        """

        started = time.perf_counter()
        torch = self.torch
        config = self.config
        dp = getattr(self, "_dp", None)
        rank = dp.rank if dp is not None else 0
        world = dp.world if dp is not None else 1
        cuda_metrics = torch.cuda.is_available() and str(config.device).startswith("cuda")
        self._clock = PhaseClock(torch, config.device, enabled=config.phase_timing)
        peaks: list[float] = []

        def reset_peak() -> None:
            if cuda_metrics:
                torch.cuda.reset_peak_memory_stats(config.device)

        def read_peak() -> float:
            if not cuda_metrics:
                return 0.0
            torch.cuda.synchronize(config.device)
            return torch.cuda.max_memory_allocated(config.device) / 2**20

        reset_peak()
        if not batch.samples:
            raise ValueError(f"cannot update {self.role} from an empty training batch")
        for sample in batch.samples:
            if not sample.policy_calls:
                raise ValueError(f"Director sample {sample.rollout_id} lacks raw policy calls")
            for call in sample.policy_calls:
                if len(call.token_ids) > config.max_sequence_length:
                    raise ValueError(
                        f"policy call {call.call_id} has {len(call.token_ids)} tokens; "
                        f"limit is {config.max_sequence_length}"
                    )
                if not call.behavior_log_probs:
                    raise ValueError(
                        f"raw policy call {call.call_id} lacks rollout-time behavior log-probs"
                    )
        # One no-grad pass at θ̄ (the learner before this update): log π_old and the
        # LoRA-off reference.  With ``fuse_old_logprobs`` the first mini-batch skips its
        # separate log π_old pass: the differentiated forward is at θ̄ too, so its detached
        # values are π_old.
        self.model.eval()
        fuse, fuse_reason = self._fuse_decision()
        mini_batches = even_mini_batches(len(batch.samples), config.ppo_mini_batches)
        if dp is not None:
            from .director_data_parallel import plan_shards, sample_cost

            costs = [sample_cost(sample) for sample in batch.samples]
            window_shards = [
                set(plan_shards(costs, indices, world)[rank]) for indices in mini_batches
            ]
        else:
            window_shards = [set(indices) for indices in mini_batches]
        local = set().union(*window_shards)
        deferred = (set(mini_batches[0]) & local) if fuse and mini_batches else set()
        # KL-reference cadence: rank within the sample's condition (so the rarer
        # scaffold-on condition is always represented) rotated by the update step
        # (so every sample position is selected equally often across updates).
        seen: dict[str, int] = {}
        kl_ranks = []
        for sample in batch.samples:
            condition = str(
                (sample.metadata.get("r2orch") or {}).get("scaffold_condition", "on")
            )
            kl_ranks.append(seen.get(condition, 0) + int(step))
            seen[condition] = seen.get(condition, 0) + 1
        prepare_started = time.perf_counter()
        prepare_failure: Exception | None = None
        try:
            prepared = [
                self._prepare(sample, kl_rank=kl_ranks[index], defer_old=index in deferred)
                if index in local
                else self._remote_placeholder(sample)
                for index, sample in enumerate(batch.samples)
            ]
        except Exception as exc:
            if dp is None:
                raise
            prepare_failure = exc
            prepared = []
        if dp is not None:
            self._agree_no_failure(dp, prepare_failure, "prepare")
        prepare_seconds = time.perf_counter() - prepare_started
        prepare_profile = self._clock.drain()
        peaks.append(read_peak())
        stats = {
            "kl_on": 0.0,
            "kl_on_tokens": 0,
            "kl_off": 0.0,
            "kl_off_tokens": 0,
            "policy_sum": 0.0,
            "clipped": 0,
            "pg_tokens": 0,
            "logratio_abs_sum": 0.0,
            "logratio_tokens": 0,
            "logratio_abs_max": 0.0,
            "kl_capped": 0,
        }
        total_loss = total_policy = total_kl = 0.0
        kl_samples = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        self.model.train()
        for _epoch in range(config.epochs):
            for window, indices in enumerate(mini_batches):
                window_started = time.perf_counter()
                reset_peak()
                self.optimizer.zero_grad(set_to_none=True)
                # Global denominators: static per-sample counts, identical on every rank.
                token_count = max(1, sum(prepared[index]["pg_tokens"] for index in indices))
                records = []
                ratio_mark = (stats["logratio_abs_sum"], stats["logratio_tokens"])
                stats["logratio_abs_max_window"] = 0.0
                mine = [index for index in indices if index in window_shards[window]]
                window_failure: Exception | None = None
                try:
                    for index in mine:
                        sample = batch.samples[index]
                        terms = self._sample_backward(
                            sample,
                            prepared[index],
                            token_count=token_count,
                            stats=stats,
                        )
                        records.append(
                            (
                                index,
                                terms["loss"],
                                terms["policy"],
                                terms["kl"],
                                terms["kl_valid"],
                                float(sample.reward),
                                terms["clipped"],
                                prepared[index]["pg_tokens"],
                            )
                        )
                except Exception as exc:
                    if dp is None:
                        raise
                    window_failure = exc
                if dp is not None:
                    self._agree_no_failure(dp, window_failure, "window")
                    with self._phase("grad_sync"):
                        dp.all_reduce_sum_gradients(self.model)
                with self._phase("clip"):
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )
                    gradient_norms.append(float(norm))
                with self._phase("optimizer"):
                    self.optimizer.step()
                    self.scheduler.step()
                optimizer_steps += 1
                self.optimizer_step_count += 1
                window_profile = self._clock.drain()
                window_peak = read_peak()
                peaks.append(window_peak)
                ratio_local = (
                    stats["logratio_abs_sum"] - ratio_mark[0],
                    stats["logratio_tokens"] - ratio_mark[1],
                    stats["logratio_abs_max_window"],
                )
                window_seconds = time.perf_counter() - window_started
                shared = (records, ratio_local, window_seconds, len(mine), window_peak)
                gathered = dp.all_gather_object(shared) if dp is not None else [shared]
                # Rebuild the single-process view: records in sample order.
                ordered = sorted(
                    (record for part in gathered for record in part[0]), key=lambda r: r[0]
                )
                window_loss = [record[1] for record in ordered]
                window_policy = [record[2] for record in ordered]
                window_kl = [record[3] for record in ordered if record[4]]
                window_rewards = [record[5] for record in ordered]
                window_clipped = sum(record[6] for record in ordered)
                window_masked = sum(record[7] for record in ordered)
                total_loss += sum(window_loss)
                total_policy += sum(window_policy)
                total_kl += sum(window_kl)
                kl_samples += len(window_kl)
                ratio = (
                    sum(part[1][0] for part in gathered),
                    sum(part[1][1] for part in gathered),
                    max(part[1][2] for part in gathered),
                )
                metric = self._step_metrics(
                    alternating_update_step=step,
                    epoch=_epoch,
                    losses=window_loss,
                    policy_losses=window_policy,
                    kls=window_kl or [0.0],
                    entropies=[0.0] * len(window_loss),
                    rewards=window_rewards,
                    masked_tokens=window_masked,
                    clipped_tokens=window_clipped,
                    grad_norm=float(norm),
                    duration_seconds=window_seconds,
                    microbatch_sizes=[1] * len(indices),
                )
                metric.update(
                    self._window_profile(
                        window_profile,
                        window_peak_mb=max(part[4] for part in gathered),
                        ratio=ratio,
                        fused=bool(fuse and window == 0 and _epoch == 0),
                    )
                )
                if dp is not None:
                    metric["data_parallel"] = {
                        "world": world,
                        "rank": rank,
                        "rank_samples": [part[3] for part in gathered],
                        "rank_window_seconds": [round(part[2], 3) for part in gathered],
                    }
                step_metrics.append(metric)
        if dp is not None:
            # Make every rank's `prepared` / `stats` the global view (cheap, tiny payloads).
            part = {
                index: {
                    key: prepared[index][key]
                    for key in ("tis_weights",)
                }
                for index in local
            }
            for gathered_part in dp.all_gather_object(part):
                for index, values in gathered_part.items():
                    if prepared[index].get("remote"):
                        prepared[index].update(values)
            merged = dp.all_gather_object(stats)
            stats = {
                key: (
                    max(item[key] for item in merged)
                    if key.startswith("logratio_abs_max")
                    else sum(item[key] for item in merged)
                )
                for key in stats
                if key != "logratio_abs_max_window"
            }
        step_metrics[-1].update(self._metrics(batch, prepared, stats, total_loss))
        save_started = time.perf_counter()
        with self._phase("save"):
            if rank == 0:
                checkpoint = self._save(step)
            else:
                checkpoint = config.checkpoint_root / f"step-{step:08d}"
            if dp is not None:
                dp.barrier()  # no rank leaves (and frees its replica) before the save is done
        save_seconds = time.perf_counter() - save_started
        self._clock.drain()
        if cuda_metrics:
            torch.cuda.synchronize(config.device)
            peaks.append(read_peak())
            peak_gpu_memory_mb = max(peaks)
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(config.device) / 2**20
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(config.device) / 2**20
        else:
            peak_gpu_memory_mb = gpu_memory_allocated_mb = gpu_memory_reserved_mb = 0.0
        duration_seconds = time.perf_counter() - started
        step_metrics[-1]["phase_update"] = {
            "total_seconds": round(duration_seconds, 4),
            "prepare_seconds": round(prepare_seconds, 4),
            "prepare_phase_seconds": prepare_profile["seconds"],
            "prepare_phase_tokens": prepare_profile["tokens"],
            "prepare_phase_calls": prepare_profile["calls"],
            "prepare_peak_gpu_memory_mb": round(peaks[0], 1),
            "save_seconds": round(save_seconds, 4),
            "model_load_seconds": getattr(self, "_model_load_seconds", None),
            "fuse_old_requested": bool(config.fuse_old_logprobs),
            "fuse_old_active": bool(fuse),
            "fuse_old_reason": fuse_reason,
            "fused_samples": len(mini_batches[0]) if fuse and mini_batches else 0,
            "fused_samples_local": len(deferred),
            "kl_reference_stride": int(config.kl_reference_stride),
            "kl_term_cap": float(config.kl_term_cap),
            "checkpoint_skip_below_tokens": int(config.checkpoint_skip_below_tokens),
            "matmul_precision": config.matmul_precision,
            "lora_dropout_effective": getattr(self, "_runtime_options", {}).get(
                "lora_dropout_effective"
            ),
            "data_parallel_world": world,
        }
        self._clock = None
        divisor = max(1, len(batch.samples) * config.epochs)
        rewards = [float(sample.reward) for sample in batch.samples]
        masked_tokens = sum(item["pg_tokens"] for item in prepared)
        return PolicyUpdateResult(
            self.role,
            step,
            total_loss / divisor,
            total_policy / divisor,
            total_kl / max(1, kl_samples),
            masked_tokens,
            str(checkpoint),
            0.0,
            samples=len(batch.samples),
            optimizer_steps=optimizer_steps,
            learning_rate=float(self.scheduler.get_last_lr()[0]),
            grad_norm=(sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0),
            clip_fraction=stats["clipped"] / max(1, stats["pg_tokens"]),
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            duration_seconds=duration_seconds,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            step_metrics=tuple(step_metrics),
            loss_variant="r2orch_director",
            reference_provenance="lora_disabled_base",
        )

    def _fuse_decision(self) -> tuple[bool, str]:
        """Whether the first mini-batch may take log π_old from its own forward."""

        if not self.config.fuse_old_logprobs:
            return False, "disabled"
        dropout = effective_lora_dropout(self.model)
        if dropout:
            return False, f"lora_dropout={dropout:g}"
        return True, ""

    def _remote_placeholder(self, sample: TrainingSample) -> dict[str, Any]:
        """Stand-in for a sample that another data-parallel rank prepares and back-propagates.

        It carries exactly what every rank needs for the *global* token-mean denominators
        and the objective telemetry; the per-sample lists are filled after the update.
        """

        r2orch = sample.metadata.get("r2orch", {})
        kinds, pg_tokens = self._plan(sample)
        return {
            "calls": [],
            "kinds": kinds,
            "condition": str(r2orch.get("scaffold_condition", "on")),
            "kl_coefficient": 0.0,
            "kl_scale": 1.0,
            "kl_reference": False,
            "pg_tokens": pg_tokens,
            "tis_weights": [],
            "pending_old": False,
            "remote": True,
        }

    @staticmethod
    def _agree_no_failure(dp: Any, failure: Exception | None, stage: str) -> None:
        """Raise on EVERY rank if any rank failed, instead of leaving the others in a collective."""

        message = None if failure is None else f"rank {dp.rank}: {type(failure).__name__}: {failure}"
        failures = [item for item in dp.all_gather_object(message) if item]
        if failures:
            if failure is not None:
                raise failure
            raise RuntimeError(f"data-parallel {stage} failed on another rank: {'; '.join(failures)}")

    def _window_profile(
        self,
        profile: dict[str, dict[str, float]],
        *,
        window_peak_mb: float,
        ratio: tuple[float, int, float],
        fused: bool,
    ) -> dict[str, Any]:
        """Per-optimizer-step phase telemetry merged into ``step_metrics``.

        ``ratio`` = (sum |log ratio|, token count, max |log ratio|) of the window.
        """

        seconds, tokens = profile["seconds"], profile["tokens"]
        train_seconds = seconds.get("train_forward", 0.0) + seconds.get("train_backward", 0.0)
        return {
            "phase_seconds": seconds,
            "phase_tokens": tokens,
            "phase_calls": profile["calls"],
            "train_tokens_per_s": (
                tokens.get("train_forward", 0) / train_seconds if train_seconds > 0 else 0.0
            ),
            "peak_gpu_memory_mb": round(window_peak_mb, 1),
            "fused_old_logprobs": fused,
            "ratio_log_abs_mean": ratio[0] / ratio[1] if ratio[1] else 0.0,
            "ratio_log_abs_max": ratio[2],
        }

    def _call_outputs(self, calls: Any, *, reference: bool = False) -> list[Any]:
        """No-grad outputs per call: [off, on] log-probs for relation calls, else action log-probs."""

        torch = self.torch
        with self._phase("reference" if reference else "old", calls=0) as span, torch.no_grad():
            timeline = (
                self._timeline_policy_outputs(calls, reference=reference)
                if self.config.raw_policy_backward_mode == "timeline"
                else None
            )
            if timeline is not None:
                span.tokens += len(calls[-1].token_ids)
                span.calls += 1
            outputs = []
            for index, call in enumerate(calls):
                relation_policy = call.metadata.get("relation_policy", {})
                if timeline is not None:
                    value = timeline[index]
                elif _is_relation_call(call):
                    value = self._binary_choice_log_probs(
                        call, relation_policy, reference=reference
                    )
                    span.tokens += len(call.token_ids)
                    span.calls += 1
                elif any(call.action_mask[1:]):
                    value = self._masked_policy_call_log_probs(call, reference=reference)
                    span.tokens += len(call.token_ids)
                    span.calls += 1
                else:
                    value = torch.empty(0)
                outputs.append(value.detach().float().cpu())
        return outputs

    def _plan(self, sample: TrainingSample) -> tuple[list[str], int]:
        """Model-free classification of every call plus the loss-token count.

        Returns ``(kinds, pg_tokens)``: ``empty|relation|ordinary`` per call and the number
        of token-mean policy-gradient terms.  Needed before the first forward so a
        mini-batch's token-mean denominator is known without any model pass.
        """

        calls = sample.policy_calls
        kinds: list[str] = []
        pg_tokens = 0
        for call in calls:
            masked = sum(
                1
                for _value, keep in zip(
                    call.behavior_log_probs, call.action_mask[1:], strict=True
                )
                if keep
            )
            if not masked:
                kinds.append("empty")
            elif _is_relation_call(call):
                kinds.append("relation")
                pg_tokens += 1
            else:
                kinds.append("ordinary")
                pg_tokens += masked
        return kinds, pg_tokens

    def _prepare(
        self, sample: TrainingSample, *, kl_rank: int = 0, defer_old: bool = False
    ) -> dict[str, Any]:
        """Per-call PPO targets and TIS weights.

        ``defer_old`` leaves log π_old to the differentiated forward
        (``_sample_backward`` finalizes the calls); everything that does not
        depend on it (reference, loss-token counts) is still computed here.
        ``kl_rank`` only matters for ``kl_reference_stride > 1``: the sample gets its
        LoRA-off KL pass when ``kl_rank % stride == 0``.
        """

        config = self.config
        calls = sample.policy_calls
        r2orch = sample.metadata.get("r2orch", {})
        condition = str(r2orch.get("scaffold_condition", "on"))
        kl_coefficient = float(r2orch.get("kl_coefficient", config.kl_coefficient))
        kinds, pg_tokens = self._plan(sample)
        # ``reference_indices``: None = every call, () = no reference pass at all.
        stride = int(config.kl_reference_stride)
        if stride == 1:
            reference_indices = None if kl_coefficient > 0 else ()
            kl_scale = 1.0
        else:
            selected = stride > 1 and kl_rank % stride == 0 and kl_coefficient > 0
            reference_indices = None if selected else ()
            kl_scale = float(stride) if selected else 1.0
        reference = (
            self._call_outputs(calls, reference=True)
            if reference_indices != ()
            else None
        )
        state: dict[str, Any] = {
            "calls": [{"kind": "empty"} if kind == "empty" else None for kind in kinds],
            "kinds": kinds,
            "condition": condition,
            "kl_coefficient": kl_coefficient,
            "kl_scale": kl_scale,
            "kl_reference": reference_indices is None and reference is not None,
            "pg_tokens": pg_tokens,
            "tis_weights": [],
            "pending_old": True,
            "_context": {"reference": reference},
        }
        if not defer_old:
            old = self._call_outputs(calls)
            for call_index, kind in enumerate(kinds):
                if kind != "empty":
                    self._finalize_call(sample, state, call_index, old[call_index])
            state["pending_old"] = False
        return state

    def _finalize_call(
        self, sample: TrainingSample, state: dict[str, Any], index: int, current: Any
    ) -> dict[str, Any]:
        """Build one call's PPO targets from its log π_old (``current``, detached, CPU)."""

        torch = self.torch
        config = self.config
        call = sample.policy_calls[index]
        reference = state["_context"]["reference"]
        reference_row = reference[index] if reference is not None else None
        behavior = torch.tensor(
            [
                value
                for value, keep in zip(call.behavior_log_probs, call.action_mask[1:], strict=True)
                if keep
            ],
            dtype=torch.float32,
        )
        checked = [behavior, current] + ([reference_row] if reference_row is not None else [])
        if any(item.numel() and not bool(torch.isfinite(item).all()) for item in checked):
            raise ValueError(f"policy call {call.call_id} has non-finite log-probabilities")
        chosen = None
        if _is_relation_call(call):
            off_id = int(call.metadata["relation_policy"]["token_ids"]["off"])
            chosen = 0 if int(call.token_ids[call.relation_token_span[0]]) == off_id else 1
            selected_old = current[chosen : chosen + 1]
            advantage = torch.full((1,), float(sample.advantage))
        else:
            selected_old = current
            advantage = torch.full_like(current, float(sample.advantage))
        weight = (selected_old - behavior).exp().clamp(max=config.tis_cap)
        state["tis_weights"].extend(float(value) for value in weight.tolist())
        state["calls"][index] = {
            "kind": "relation" if chosen is not None else "ordinary",
            "chosen": chosen,
            "old": selected_old,
            "advantage": advantage,
            "weight": weight,
            "reference": reference_row if state["kl_reference"] else None,
        }
        return state["calls"][index]

    def _sample_backward(
        self,
        sample: TrainingSample,
        prepared: dict[str, Any],
        *,
        token_count: int,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Backward one sample's token-mean PG + KL contribution."""

        torch = self.torch
        config = self.config
        calls = sample.policy_calls
        timeline = None
        if config.raw_policy_backward_mode == "timeline":
            with self._phase("train_forward", calls=0) as span:
                timeline = self._timeline_policy_outputs(calls)
                if timeline is not None:
                    span.tokens += len(calls[-1].token_ids)
                    span.calls += 1
        kl_coefficient = prepared["kl_coefficient"]
        kl_weight = kl_coefficient * prepared.get("kl_scale", 1.0)
        policy_sum = kl_sum = 0.0
        clipped = 0
        pending = None
        for index, call in enumerate(calls):
            item = prepared["calls"][index]
            kind = prepared["kinds"][index] if item is None else item["kind"]
            if kind == "empty":
                continue
            relation = _is_relation_call(call)
            if timeline is not None:
                current = timeline[index]
            else:
                with self._phase("train_forward", tokens=len(call.token_ids)):
                    if relation:
                        current = self._binary_choice_log_probs(
                            call, call.metadata["relation_policy"], require_grad=True
                        )
                    else:
                        current = self._masked_policy_call_log_probs(call, require_grad=True)
            if item is None:
                # Deferred π_old: this forward is at θ̄ and LoRA dropout is off, so
                # its detached values are exactly what the no-grad pass would give.
                item = self._finalize_call(
                    sample, prepared, index, current.detach().float().cpu()
                )
            with self._phase("train_loss"):
                log_probs = current[item["chosen"]].reshape(1) if relation else current
                device = log_probs.device
                reference = item["reference"]
                if reference is None:
                    kl_terms = torch.zeros_like(log_probs, dtype=torch.float32)
                elif relation:
                    kl_terms = _categorical_kl(current, reference.to(device)).reshape(1)
                else:
                    kl_terms = _sampled_kl_terms(
                        log_probs, reference.to(device), config.kl_term_cap
                    )
                    if config.kl_term_cap > 0:
                        stats["kl_capped"] += int(
                            (kl_terms.detach() >= config.kl_term_cap).sum()
                        )
                log_ratio = (log_probs.float() - item["old"].to(device)).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                advantage = item["advantage"].to(device)
                surrogate = torch.maximum(
                    -advantage * ratio,
                    -advantage * ratio.clamp(1.0 - config.clip_low, 1.0 + config.clip_high),
                )
                call_policy = (item["weight"].to(device) * surrogate).sum()
                call_kl = kl_terms.sum()
                contribution = (call_policy + kl_weight * call_kl) / token_count
                clipped += int(
                    ((ratio < 1.0 - config.clip_low) | (ratio > 1.0 + config.clip_high)).sum()
                )
                policy_sum += float(call_policy.detach())
                kl_sum += float(call_kl.detach())
                abs_log_ratio = log_ratio.detach().abs()
                stats["logratio_abs_sum"] += float(abs_log_ratio.sum())
                stats["logratio_tokens"] += int(abs_log_ratio.numel())
                call_max = float(abs_log_ratio.max()) if abs_log_ratio.numel() else 0.0
                stats["logratio_abs_max"] = max(stats["logratio_abs_max"], call_max)
                stats["logratio_abs_max_window"] = max(
                    stats.get("logratio_abs_max_window", 0.0), call_max
                )
            if timeline is None:
                if contribution.requires_grad:
                    with self._phase("train_backward"):
                        contribution.backward()
            else:
                pending = contribution if pending is None else pending + contribution
        if pending is not None and pending.requires_grad:
            with self._phase("train_backward"):
                pending.backward()
        prepared["pending_old"] = False
        del timeline
        tokens = prepared["pg_tokens"]
        condition_key = "kl_on" if prepared["condition"] == "on" else "kl_off"
        kl_valid = bool(prepared.get("kl_reference", True)) or config.kl_reference_stride == 1
        stats[condition_key] += kl_sum
        stats[condition_key + "_tokens"] += tokens if kl_valid else 0
        stats["policy_sum"] += policy_sum
        stats["clipped"] += clipped
        stats["pg_tokens"] += tokens
        return {
            "loss": (policy_sum + kl_weight * kl_sum) / max(1, tokens),
            "policy": policy_sum / max(1, tokens),
            "kl": kl_sum / max(1, tokens),
            "kl_valid": kl_valid,
            "clipped": clipped,
        }

    def _metrics(
        self,
        batch: TrainingBatch,
        prepared: list[dict[str, Any]],
        stats: dict[str, Any],
        total_loss: float,
    ) -> dict[str, Any]:
        """Director loss telemetry, prefixed ``r2orch_``."""

        weights = [value for item in prepared for value in item["tis_weights"]]
        conditions = [item["condition"] for item in prepared]
        return {
            "r2orch_outcome_adv_abs_mean": statistics.fmean(
                abs(float(sample.advantage)) for sample in batch.samples
            ),
            # The logged `loss`/`policy_loss` average per-sample token means (every sample counts the
            # same); the gradient uses the mini-batch token mean, whose whitened advantages average to 0.
            # This is the matching token-weighted policy loss (about 0 at ratio 1).
            "r2orch_policy_loss_token_mean": stats["policy_sum"] / max(1, stats["pg_tokens"]),
            "r2orch_kl_on": stats["kl_on"] / max(1, stats["kl_on_tokens"]),
            "r2orch_kl_off": stats["kl_off"] / max(1, stats["kl_off_tokens"]),
            "r2orch_kl_capped_tokens": stats.get("kl_capped", 0),
            "r2orch_tis_weight_mean": statistics.fmean(weights) if weights else 0.0,
            "r2orch_tis_capped_fraction": (
                sum(1 for value in weights if value >= self.config.tis_cap) / len(weights)
                if weights
                else 0.0
            ),
            "r2orch_samples_on": conditions.count("on"),
            "r2orch_samples_off": conditions.count("off"),
            "r2orch_loss": total_loss / max(1, len(batch.samples) * self.config.epochs),
            "r2orch_logratio_abs_mean": stats["logratio_abs_sum"] / max(1, stats["logratio_tokens"]),
            "r2orch_logratio_abs_max": stats["logratio_abs_max"],
        }

    def _step_metrics(
        self,
        *,
        alternating_update_step: int,
        epoch: int,
        losses: list[float],
        policy_losses: list[float],
        kls: list[float],
        entropies: list[float],
        rewards: list[float],
        masked_tokens: int,
        clipped_tokens: int,
        grad_norm: float,
        duration_seconds: float,
        microbatch_sizes: list[int],
    ) -> dict[str, Any]:
        cuda_metrics = self.torch.cuda.is_available() and str(self.config.device).startswith("cuda")
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "role": self.role,
            "role_optimizer_step": self.optimizer_step_count,
            "alternating_update_step": alternating_update_step,
            "epoch": epoch,
            "microbatch_count": len(microbatch_sizes),
            "microbatch_sizes": list(microbatch_sizes),
            "max_microbatch_size": max(microbatch_sizes, default=0),
            "loss": statistics.fmean(losses),
            "policy_loss": statistics.fmean(policy_losses),
            "kl": statistics.fmean(kls),
            "entropy": statistics.fmean(entropies),
            "kl_loss_contribution": self.config.kl_coefficient * statistics.fmean(kls),
            "entropy_loss_contribution": (
                -self.config.entropy_coefficient * statistics.fmean(entropies)
            ),
            "loss_variant": self.config.loss_variant,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "masked_tokens": masked_tokens,
            "clip_fraction": clipped_tokens / max(1, masked_tokens),
            "grad_norm": grad_norm,
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
            "duration_seconds": duration_seconds,
            "gpu_memory_allocated_mb": (
                self.torch.cuda.memory_allocated(self.config.device) / 2**20
                if cuda_metrics
                else 0.0
            ),
            "gpu_memory_reserved_mb": (
                self.torch.cuda.memory_reserved(self.config.device) / 2**20 if cuda_metrics else 0.0
            ),
        }

    def _binary_choice_log_probs(
        self,
        call: TokenizedPolicyCall,
        policy: dict[str, Any],
        *,
        require_grad: bool = False,
        reference: bool = False,
    ):
        """Evaluate current/reference policy on the same constrained off/on support."""

        torch = self.torch
        if call.relation_token_span is None:
            raise ValueError("binary relation call lacks a token span")
        start, end = call.relation_token_span
        if end - start != 1 or start <= 0:
            raise ValueError("binary relation call must contain one predicted token")
        token_map = policy.get("token_ids", {})
        try:
            support = [int(token_map[choice]) for choice in ("off", "on")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("binary relation policy lacks off/on tokenizer IDs") from exc
        inputs = torch.tensor([call.token_ids], dtype=torch.long, device=self.config.device)
        attention_mask = torch.ones_like(inputs)
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = nullcontext() if require_grad else torch.no_grad()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if require_grad and self._activation_cpu_offload_enabled(len(call.token_ids))
            else nullcontext()
        )
        checkpoint_context = (
            self._checkpoint_context(len(call.token_ids)) if require_grad else nullcontext()
        )
        with adapter_context, gradient_context, offload_context, checkpoint_context:
            position = torch.tensor([start - 1], dtype=torch.long, device=self.config.device)
            logits = self._selected_sequence_logits(inputs, attention_mask, position)
            selected = logits[0, 0, support]
            values = torch.log_softmax(selected.float(), dim=-1)
        return values if require_grad else values.detach()

    def _timeline_policy_outputs(self, calls, *, reference=False):
        """Share exact causal prefixes; return None for per-call streaming fallback.

        All action targets retain their original positions. Reference/behavior
        probabilities, PPO weighting and optimizer boundaries remain unchanged.
        """
        from .policy_timeline import timeline_positions

        positions = timeline_positions(calls, self.config.max_sequence_length)
        if positions is None:
            return None
        torch = self.torch
        ids = calls[-1].token_ids
        inputs = torch.tensor([ids], dtype=torch.long, device=self.config.device)
        selected = sorted(p for row in positions for p in row)
        lookup = {p: i for i, p in enumerate(selected)}
        offload = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if self._activation_cpu_offload_enabled(len(ids))
            else nullcontext()
        )
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = torch.no_grad() if reference else nullcontext()
        checkpoint_context = nullcontext() if reference else self._checkpoint_context(len(ids))
        with adapter_context, gradient_context, offload, checkpoint_context:
            logits = self._selected_sequence_logits(
                inputs,
                torch.ones_like(inputs),
                torch.tensor(selected, dtype=torch.long, device=inputs.device),
            )[0]
            outputs = []
            for call, row in zip(calls, positions, strict=True):
                local = logits[[lookup[p] for p in row]]
                relation = call.metadata.get("relation_policy", {})
                if call.relation_token_span is not None and relation:
                    support = [int(relation["token_ids"][k]) for k in ("off", "on")]
                    outputs.append(torch.log_softmax(local[0, support].float(), dim=-1))
                else:
                    lp = torch.log_softmax(local, dim=-1)
                    labels = torch.tensor(
                        [call.token_ids[p + 1] for p in row], dtype=torch.long, device=inputs.device
                    )
                    outputs.append(lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
        return outputs

    def _masked_policy_call_log_probs(
        self,
        call: TokenizedPolicyCall,
        *,
        require_grad: bool = False,
        reference: bool = False,
    ):
        """Evaluate only sampled action positions while retaining the full prefix."""

        torch = self.torch
        positions = [index for index, keep in enumerate(call.action_mask[1:]) if keep]
        if not positions:
            raise ValueError(f"policy call {call.call_id} has no next-token action target")
        inputs = torch.tensor([call.token_ids], dtype=torch.long, device=self.config.device)
        attention_mask = torch.ones_like(inputs)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.config.device)
        labels = torch.tensor(
            [call.token_ids[position + 1] for position in positions],
            dtype=torch.long,
            device=self.config.device,
        )
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        gradient_context = nullcontext() if require_grad else torch.no_grad()
        offload_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
            if require_grad and self._activation_cpu_offload_enabled(len(call.token_ids))
            else nullcontext()
        )
        checkpoint_context = (
            self._checkpoint_context(len(call.token_ids)) if require_grad else nullcontext()
        )
        with adapter_context, gradient_context, offload_context, checkpoint_context:
            logits = self._selected_sequence_logits(inputs, attention_mask, position_tensor)[0]
            all_log_probs = torch.log_softmax(logits, dim=-1)
            values = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if not require_grad:
            values = values.detach().cpu()
        return values

    def _selected_sequence_logits(self, inputs: Any, attention_mask: Any, positions: Any):
        """Use model-side logit slicing."""

        return self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            logits_to_keep=positions,
        ).logits

    def _activation_cpu_offload_enabled(self, token_count: int) -> bool:
        threshold = self.config.activation_cpu_offload_min_tokens
        return bool(
            self.config.activation_cpu_offload
            and str(self.config.device).startswith("cuda")
            and (threshold == 0 or token_count >= threshold)
        )

    def _checkpoint_context(self, token_count: int):
        """Skip gradient checkpointing for calls shorter than the configured threshold."""

        threshold = self.config.checkpoint_skip_below_tokens
        if not (
            self.config.gradient_checkpointing
            and threshold > 0
            and token_count < threshold
        ):
            return nullcontext()
        toggle = getattr(self, "_checkpoint_toggle", None)
        if toggle is None:
            toggle = self._checkpoint_toggle = CheckpointingToggle(self.model)
        return toggle.suspended()

    def _phase(self, name: str, *, tokens: int = 0, calls: int = 1):
        """Timing context (``PhaseClock``) when an update is being profiled, else a no-op."""

        clock = getattr(self, "_clock", None)
        if clock is None:
            return _NULL_PHASE
        return clock.phase(name, tokens=tokens, calls=calls)

    def _save(self, step: int) -> Path:
        destination = self.config.checkpoint_root / f"step-{step:08d}"
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        self.torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_step_count": self.optimizer_step_count,
                "torch_rng": self.torch.get_rng_state(),
                "cuda_rng": (
                    self.torch.cuda.get_rng_state(self.config.device)
                    if self.torch.cuda.is_available() and str(self.config.device).startswith("cuda")
                    else None
                ),
                "cuda_rng_scope": "configured_device",
                "python_rng": random.getstate(),
                "step": step,
            },
            destination / "trainer_state.pt",
        )
        if self.config.publish_latest_checkpoint:
            _atomic_json(
                self.config.checkpoint_root / "latest.json",
                {"path": str(destination), "step": step},
            )
        _atomic_json(destination / "training_config.json", _jsonable(asdict(self.config)))
        return destination

    def _latest_checkpoint(self) -> Path | None:
        latest = self.config.checkpoint_root / "latest.json"
        if not latest.exists():
            return None
        path = Path(json.loads(latest.read_text(encoding="utf-8"))["path"])
        return path if path.exists() else None

    def _restore_optimizer(self, source: Path) -> None:
        state_path = source / "trainer_state.pt"
        if not state_path.exists():
            return
        saved_total_optimizer_steps = None
        saved_schedule = "cosine"
        saved_warmup_start_factor = 0.0
        saved_config_path = source / "training_config.json"
        if saved_config_path.exists():
            saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            saved_total_optimizer_steps = int(saved_config["total_optimizer_steps"])
            saved_schedule = saved_config["learning_rate_schedule"]
            saved_warmup_start_factor = float(saved_config["warmup_start_factor"])
        if saved_schedule != self.config.learning_rate_schedule:
            raise ValueError("checkpoint learning-rate schedule differs from requested schedule")
        if saved_warmup_start_factor != self.config.warmup_start_factor:
            raise ValueError("checkpoint warmup start factor differs from requested schedule")
        state = self.torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer_step_count = int(state["optimizer_step_count"])
        if (
            saved_total_optimizer_steps is not None
            and saved_total_optimizer_steps != self.config.total_optimizer_steps
        ):
            if self.config.total_optimizer_steps <= self.optimizer_step_count:
                raise ValueError(
                    "extended training requires total_optimizer_steps greater than the "
                    f"restored optimizer step {self.optimizer_step_count}"
                )
            resumed_lrs = _scheduler_lrs_at_step(self.scheduler, self.optimizer_step_count)
            for group, learning_rate in zip(self.optimizer.param_groups, resumed_lrs, strict=True):
                group["lr"] = learning_rate
            self.scheduler.last_epoch = self.optimizer_step_count
            self.scheduler._last_lr = resumed_lrs
        self.torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state(state["cuda_rng"], self.config.device)
        random.setstate(state["python_rng"])

    def close(self) -> None:
        del self.scheduler
        del self.optimizer
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _build_learning_rate_scheduler(optimizer: Any, config: PolicyTrainingConfig) -> Any:
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    if config.warmup_steps > 0 and config.warmup_start_factor > 0:
        from torch.optim.lr_scheduler import LambdaLR

        def factor(step: int) -> float:
            if step < config.warmup_steps:
                return (
                    config.warmup_start_factor
                    + (1.0 - config.warmup_start_factor) * step / config.warmup_steps
                )
            if config.learning_rate_schedule == "constant":
                return 1.0
            progress = (step - config.warmup_steps) / max(
                1, config.total_optimizer_steps - config.warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, factor)
    if config.learning_rate_schedule == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_optimizer_steps,
    )


def _scheduler_lrs_at_step(scheduler: Any, step: int) -> list[float]:
    """Rebase a LambdaLR when a resumed experiment explicitly extends its horizon."""

    return [
        float(base_lr) * float(schedule(step))
        for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)
    ]


def create_policy_trainer(role: str, policy: PolicyTrainingConfig, seed: int) -> PolicyTrainer:
    if policy.data_parallel_gpu_ids:
        from .solver_parallel_trainer import SubprocessSolverTrainer

        return SubprocessSolverTrainer(role, policy, seed=seed)
    return TransformersGRPOTrainer(role, policy, seed=seed)


class AlternatingGRPOTrainer:
    def __init__(
        self,
        config: AlternatingTrainingConfig,
        *,
        trainer_factory: Callable[[str, PolicyTrainingConfig, int], PolicyTrainer] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.trainer_factory = trainer_factory or create_policy_trainer
        self.state = self._load_state()
        self._recover_interrupted_parallel_publish()

    def train_cycle(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
        *,
        relation_credits: Iterable[Any] = (),
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult]:
        self._validate_cycle_batches(proposer_batch, solver_batch)
        # A low-sample Proposer pair batch is an intentional no-op.  Keep
        # that cycle on the serial path so the strict parallel-result
        # validation (which requires two committed checkpoints) cannot treat
        # the no-op as a failed optimizer update.  The Solver still trains in
        # the same cycle, matching the ordinary alternating path.
        proposer_parallel_ready = bool(proposer_batch.samples) and not (
            proposer_batch.objective == "proposer_pair"
            and len(proposer_batch.samples) < self.config.proposer.min_proposer_pair_samples
        )
        if self.config.parallel_roles and proposer_parallel_ready and solver_batch.samples:
            return self._train_cycle_parallel(proposer_batch, solver_batch)
        if self.state.phase not in {"proposer", "solver", "cycle_complete"}:
            raise ValueError(f"unknown alternating training phase: {self.state.phase}")
        proposer = (
            self._train_role(
                "proposer",
                proposer_batch,
                self.config.proposer,
                next_phase="solver",
            )
            if self.state.phase == "proposer"
            else self._latest_result("proposer")
        )
        solver = (
            self._train_role(
                "solver",
                solver_batch,
                self.config.solver,
                next_phase="cycle_complete",
            )
            if self.state.phase == "solver"
            else self._latest_result("solver")
        )
        self.state.cycle += 1
        self.state.active_batch_sha256 = ""
        self.state.phase = "proposer"
        self._save_state()
        return proposer, solver

    def committed_cycle_results(
        self, cycle: int
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult] | None:
        """Return a fully committed cycle without running either optimizer again.

        The experiment progress file is written after service restoration and
        metrics.  A failure in those steps can therefore leave durable policy
        checkpoints for a cycle which is not yet present in experiment_progress.
        Resume must use those checkpoints instead of interpreting the fresh
        ``phase=proposer`` boundary as permission to train the same batch twice.
        """

        if self.state.cycle <= cycle:
            return None
        results: dict[str, PolicyUpdateResult] = {}
        for payload in self.state.history:
            if payload.get("cycle") != cycle:
                continue
            normalized = dict(payload)
            normalized["step_metrics"] = tuple(normalized.get("step_metrics", ()))
            result = PolicyUpdateResult(**normalized)
            results[result.role] = result
        if set(results) != {"proposer", "solver"}:
            raise RuntimeError(
                f"training state marks cycle {cycle} committed but does not contain both roles"
            )
        return results["proposer"], results["solver"]

    def _train_cycle_parallel(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
    ) -> tuple[PolicyUpdateResult, PolicyUpdateResult]:
        if self.state.phase != "proposer":
            raise RuntimeError(
                "parallel role training can start only at a jointly committed cycle boundary"
            )
        proposer_step = self.state.global_step + 1
        solver_step = self.state.global_step + 2
        proposer_config = replace(self.config.proposer, publish_latest_checkpoint=False)
        solver_config = replace(self.config.solver, publish_latest_checkpoint=False)
        proposer_trainer = self.trainer_factory(
            "proposer", proposer_config, self.config.seed + proposer_step
        )
        try:
            solver_trainer = self.trainer_factory(
                "solver", solver_config, self.config.seed + solver_step
            )
        except BaseException:
            proposer_trainer.close()
            raise
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="policy-update") as pool:
            proposer_future = pool.submit(
                self._run_role_update,
                proposer_trainer,
                proposer_batch,
                proposer_step,
            )
            solver_future = pool.submit(
                self._run_role_update,
                solver_trainer,
                solver_batch,
                solver_step,
            )
            proposer = proposer_future.result()
            solver = solver_future.result()
        proposer = replace(proposer, cycle=self.state.cycle)
        solver = replace(solver, cycle=self.state.cycle)
        self._validate_parallel_results(proposer, solver)
        self._publish_parallel_results(proposer, solver)
        # Both trainers still have live local references until this method
        # returns.  Starting vLLM from here races their CUDA teardown and can
        # fail even though both checkpoints have already been committed.  The
        # experiment coordinator restores services after train_cycle returns.
        return proposer, solver

    def _run_role_update(
        self,
        trainer: PolicyTrainer,
        batch: TrainingBatch,
        step: int,
    ) -> PolicyUpdateResult:
        activate_rng = getattr(trainer, "activate_rng", None)
        if activate_rng is not None:
            activate_rng()
        try:
            return trainer.update(batch, step=step)
        finally:
            trainer.close()

    def _validate_parallel_results(
        self,
        proposer: PolicyUpdateResult,
        solver: PolicyUpdateResult,
    ) -> None:
        for expected_role, result in (("proposer", proposer), ("solver", solver)):
            if result.role != expected_role:
                raise RuntimeError(f"parallel trainer returned {result.role!r} for {expected_role}")
            if result.optimizer_steps <= 0 or not Path(result.checkpoint).exists():
                raise RuntimeError(f"{expected_role} produced an incomplete pending checkpoint")
            numeric = (result.loss, result.policy_loss, result.kl, result.grad_norm)
            if not all(math.isfinite(float(value)) for value in numeric):
                raise RuntimeError(f"{expected_role} produced non-finite training metrics")

    @property
    def _parallel_transaction_path(self) -> Path:
        return self.config.state_path.with_suffix(".parallel-transaction.json")

    def _latest_payload(self, config: PolicyTrainingConfig) -> dict[str, Any] | None:
        path = config.checkpoint_root / "latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _restore_latest_payload(
        self,
        config: PolicyTrainingConfig,
        payload: dict[str, Any] | None,
    ) -> None:
        path = config.checkpoint_root / "latest.json"
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_json(path, payload)

    def _publish_parallel_results(
        self,
        proposer: PolicyUpdateResult,
        solver: PolicyUpdateResult,
    ) -> None:
        previous = {
            "proposer": self._latest_payload(self.config.proposer),
            "solver": self._latest_payload(self.config.solver),
        }
        next_state = AlternatingTrainingState.from_dict(self.state.to_dict())
        next_state.global_step = solver.step
        next_state.proposer_step = proposer.step
        next_state.solver_step = solver.step
        next_state.proposer_checkpoint = proposer.checkpoint
        next_state.solver_checkpoint = solver.checkpoint
        next_state.history.extend((asdict(proposer), asdict(solver)))
        next_state.cycle += 1
        next_state.active_batch_sha256 = ""
        next_state.phase = "proposer"
        transaction = {
            "schema_version": 1,
            "target_cycle": next_state.cycle,
            "previous_latest": previous,
            "next_checkpoints": {
                "proposer": proposer.checkpoint,
                "solver": solver.checkpoint,
            },
        }
        _atomic_json(self._parallel_transaction_path, transaction)
        try:
            _atomic_json(
                self.config.proposer.checkpoint_root / "latest.json",
                {"path": proposer.checkpoint, "step": proposer.step},
            )
            _atomic_json(
                self.config.solver.checkpoint_root / "latest.json",
                {"path": solver.checkpoint, "step": solver.step},
            )
            self.state = next_state
            self._save_state()
        except BaseException:
            self._restore_latest_payload(self.config.proposer, previous["proposer"])
            self._restore_latest_payload(self.config.solver, previous["solver"])
            raise
        finally:
            self._parallel_transaction_path.unlink(missing_ok=True)

    def _recover_interrupted_parallel_publish(self) -> None:
        path = self._parallel_transaction_path
        if not path.exists():
            return
        transaction = json.loads(path.read_text(encoding="utf-8"))
        target_cycle = int(transaction.get("target_cycle", -1))
        if self.state.cycle < target_cycle:
            previous = dict(transaction.get("previous_latest", {}))
            self._restore_latest_payload(self.config.proposer, previous.get("proposer"))
            self._restore_latest_payload(self.config.solver, previous.get("solver"))
        path.unlink(missing_ok=True)

    def _validate_cycle_batches(
        self,
        proposer_batch: TrainingBatch,
        solver_batch: TrainingBatch,
    ) -> None:
        from .execution_contract import validate_training_contract

        try:
            validate_training_contract(proposer_batch, solver_batch)
        except ValueError as exc:
            raise UnsafeTrainingBatchError(str(exc)) from exc
        if proposer_batch.role != "proposer" or solver_batch.role != "solver":
            raise UnsafeTrainingBatchError("training batch roles must be proposer then solver")
        lineage = solver_batch.metadata.get("policy_lineage")
        if lineage is not None:
            if proposer_batch.metadata.get("policy_lineage") != lineage:
                raise UnsafeTrainingBatchError("role batches have different policy lineage")
            target_cycle = int(lineage.get("target_cycle", -1))
            behavior_index = int(lineage.get("behavior_update_index", -1))
            if target_cycle != self.state.cycle or self.state.cycle - behavior_index not in {0, 1}:
                raise UnsafeTrainingBatchError(
                    "rollout policy lineage exceeds the current one-update boundary"
                )
        schema = solver_batch.metadata.get("training_selection_schema")
        if schema == "r2orch":
            # Admission is sample-local (training_selection.trajectory_exclusions in
            # build_director_batch); partial groups are expected.
            if (
                proposer_batch.objective != "proposer_pair"
                or solver_batch.objective != "r2orch_director"
            ):
                raise UnsafeTrainingBatchError("r2orch batches carry unexpected objectives")
            return
        raise UnsafeTrainingBatchError("unsupported training selection schema")

    def _train_role(
        self,
        role: str,
        batch: TrainingBatch,
        config: PolicyTrainingConfig,
        *,
        next_phase: str,
    ) -> PolicyUpdateResult:
        skip_reason = None
        if not batch.samples:
            skip_reason = "no_eligible_samples"
        elif (
            role == "proposer"
            and batch.objective == "proposer_pair"
            and len(batch.samples) < config.min_proposer_pair_samples
        ):
            skip_reason = (
                "insufficient_proposer_pair_samples:"
                f"{len(batch.samples)}<{config.min_proposer_pair_samples}"
            )
        if skip_reason is not None:
            result = PolicyUpdateResult(
                role=role,
                step=self.state.global_step,
                loss=0.0,
                policy_loss=0.0,
                kl=0.0,
                masked_tokens=0,
                checkpoint="",
                status="skipped",
                skip_reason=skip_reason,
                cycle=self.state.cycle,
            )
            self.state.history.append(asdict(result))
            self.state.phase = next_phase
            self._save_state()
            return result
        step = self.state.global_step + 1
        trainer = self.trainer_factory(role, config, self.config.seed + step)
        try:
            result = trainer.update(batch, step=step)
        finally:
            trainer.close()
        result = replace(result, cycle=self.state.cycle)
        self.state.global_step = step
        setattr(self.state, f"{role}_step", step)
        setattr(self.state, f"{role}_checkpoint", result.checkpoint)
        self.state.history.append(asdict(result))
        self.state.phase = next_phase
        self._save_state()
        return result

    def _latest_result(self, role: str) -> PolicyUpdateResult:
        for payload in reversed(self.state.history):
            if str(payload.get("role", "")) == role:
                normalized = dict(payload)
                normalized["step_metrics"] = tuple(normalized.get("step_metrics", ()))
                return PolicyUpdateResult(**normalized)
        raise RuntimeError(f"training state says {role} is complete but contains no saved result")

    def _load_state(self) -> AlternatingTrainingState:
        if not self.config.state_path.exists():
            return AlternatingTrainingState()
        return AlternatingTrainingState.from_dict(
            json.loads(self.config.state_path.read_text(encoding="utf-8"))
        )

    def _save_state(self) -> None:
        _atomic_json(self.config.state_path, self.state.to_dict())


class MockPolicyTrainer:
    """Small deterministic backend used to exercise orchestration without Torch."""

    def __init__(self, role: str, config: PolicyTrainingConfig, *, seed: int = 0) -> None:
        self.role = role
        self.config = config
        self.seed = seed

    def update(self, batch: TrainingBatch, *, step: int) -> PolicyUpdateResult:
        if batch.objective == "proposer_pair":
            from .proposer_pair_training import mock_proposer_pair_update

            return mock_proposer_pair_update(self, batch, step=step)
        started = time.perf_counter()
        masked = sum(sum(sample.action_mask) for sample in batch.samples)
        values = [
            value
            for sample in batch.samples
            for value, selected in zip(token_advantages(sample), sample.action_mask, strict=True)
            if selected
        ]
        loss = -sum(values) / max(1, len(values))
        destination = self.config.checkpoint_root / f"step-{step:08d}"
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            destination / "mock_policy.json",
            {"role": self.role, "step": step, "loss": loss, "seed": self.seed},
        )
        if self.config.publish_latest_checkpoint:
            _atomic_json(
                self.config.checkpoint_root / "latest.json",
                {"path": str(destination), "step": step},
            )
        rewards = [float(sample.reward) for sample in batch.samples]
        duration = time.perf_counter() - started
        optimizer_steps = len(even_mini_batches(len(batch.samples), self.config.ppo_mini_batches))
        step_metric = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "role": self.role,
            "role_optimizer_step": step,
            "alternating_update_step": step,
            "epoch": 0,
            "microbatch_count": len(batch.samples),
            "loss": loss,
            "policy_loss": loss,
            "kl": 0.0,
            "entropy": 0.0,
            "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "masked_tokens": masked,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "learning_rate": self.config.learning_rate,
            "duration_seconds": duration,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
        }
        step_metric.update(_mock_metrics(batch, loss))
        return PolicyUpdateResult(
            self.role,
            step,
            loss,
            loss,
            0.0,
            masked,
            str(destination),
            samples=len(batch.samples),
            optimizer_steps=optimizer_steps,
            learning_rate=self.config.learning_rate,
            reward_mean=statistics.fmean(rewards) if rewards else 0.0,
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            duration_seconds=duration,
            step_metrics=(step_metric,),
        )

    def close(self) -> None:
        return None


def _mock_metrics(batch: TrainingBatch, loss: float) -> dict[str, Any]:
    conditions = [
        sample.metadata.get("r2orch", {}).get("scaffold_condition") for sample in batch.samples
    ]
    return {
        "r2orch_outcome_adv_abs_mean": (
            statistics.fmean(abs(float(sample.advantage)) for sample in batch.samples)
            if batch.samples
            else 0.0
        ),
        "r2orch_policy_loss_token_mean": 0.0,
        "r2orch_kl_on": 0.0,
        "r2orch_kl_off": 0.0,
        "r2orch_kl_capped_tokens": 0,
        "r2orch_tis_weight_mean": 1.0,
        "r2orch_tis_capped_fraction": 0.0,
        "r2orch_samples_on": conditions.count("on"),
        "r2orch_samples_off": conditions.count("off"),
        "r2orch_loss": loss,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
