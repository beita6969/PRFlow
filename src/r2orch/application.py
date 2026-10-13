from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shlex
import tomllib
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .adaptive import AdaptiveSolverResult, AdaptiveWorkflowSolver
from .agent_tools import (
    FiniteSearchTool,
    PythonExecutionTool,
    SearchServiceTool,
    SymbolicComputeTool,
)
from .alfworld import (
    ALFWorldEnvironmentVerifier,
    ALFWorldSessionLifecycle,
    ALFWorldStepTool,
)
from .config import (
    DEFAULT_DATASET_MAX_TOTAL_TOKENS,
    CanvasConfig,
    ModelGatewayConfig,
    ModelRoleConfig,
    canonical_dataset_name,
    canvas_config_record,
)
from .dataset_actions import default_dataset_action_registry
from .deadline import RolloutDeadline, WorkerWallClockLimitExceeded
from .benchmark_scoring import benchmark_verifier_adapters
from .latency import RouteLatencyTracker, RouteTokenTracker
from .llm import (
    BinaryChoiceResponse,
    ChatBackend,
    MockBackend,
    OpenAICompatibleBackend,
)
from .observability import (
    AutoVerifier,
    ExactMatchVerifier,
    FlowSteerQAVerifier,
    JSONLTraceStore,
    MultiAnswerExactMatchVerifier,
    MultipleChoiceVerifier,
    NumericVerifier,
    TaskSpec,
    TokenF1Verifier,
    Verifier,
    task_requires_reference,
    task_to_public_dict,
)
from .output_contract import OUTPUT_CONTRACT_VERSION
from .pats import PatsConfig
from .run_config import R2OrchConfig
from .rollouts import Tokenizer
from .runtime import (
    ModelAgentExecutor,
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
)
from .selfplay import AlternatingSnapshots
from .skills import E5SkillEmbedder


REMOTE_RUNTIME_MAX_CONCURRENCY = 16


def _allowed_physical_gpu_ids() -> set[int]:
    """Return the explicit per-run physical GPU allowlist (default: GPU 0)."""

    raw = os.environ.get("R2ORCH_ALLOWED_PHYSICAL_GPUS", "0")
    try:
        allowed = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError("R2ORCH_ALLOWED_PHYSICAL_GPUS must contain integer GPU ids") from exc
    if not allowed or any(gpu < 0 for gpu in allowed):
        raise ValueError("R2ORCH_ALLOWED_PHYSICAL_GPUS must contain non-negative GPU ids")
    return allowed


@dataclass(frozen=True)
class RoleModelConfig:
    """One independently served and checkpointed model-role boundary."""

    base_url: str
    served_model: str
    base_model_path: Path
    checkpoint_path: Path
    api_key: str = "EMPTY"
    timeout_s: float = 120.0
    trainable: bool = True
    enable_thinking: bool | None = None
    # Director sampling (graph-director role only); None keeps 0.6 / 0.95 / 20.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    def validate(self, name: str) -> None:
        if not self.base_url.strip():
            raise ValueError(f"models.{name}.base_url cannot be empty")
        if not self.served_model.strip():
            raise ValueError(f"models.{name}.served_model cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError(f"models.{name}.timeout_s must be positive")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError(f"models.{name}.enable_thinking must be a boolean")

    def identity(self) -> tuple[str, str]:
        return self.base_url.rstrip("/"), self.served_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "served_model": self.served_model,
            "base_model_path": str(self.base_model_path),
            "checkpoint_path": str(self.checkpoint_path),
            "trainable": self.trainable,
            "enable_thinking": self.enable_thinking,
            **{key: getattr(self, key) for key in ("temperature", "top_p", "top_k")
               if getattr(self, key) is not None},
        }


@dataclass(frozen=True)
class FixedRuntimeConfig:
    """Fixed inference environment used by Workers and skill distillation."""

    base_url: str
    served_model: str
    model_path: Path | None
    api_key: str = "EMPTY"
    timeout_s: float = 120.0
    request_profile: str = "qwen"
    enable_thinking: bool = False
    max_tokens: int = 2048
    max_concurrency: int = 4
    managed_locally: bool = True
    frozen: bool = True
    # Worker sampling temperature; the skill distiller stays at 0.
    temperature: float = 0.0
    # A self-hosted Qwen service (for example on another node): no provider quota, so the
    # per-endpoint concurrency cap of externally managed API providers does not apply.
    self_hosted: bool = False

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("runtime.max_tokens must be positive")
        if not self.base_url.strip() or not self.served_model.strip():
            raise ValueError("runtime endpoint and served_model cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("runtime.max_concurrency must be positive")
        remote_limit = REMOTE_RUNTIME_MAX_CONCURRENCY
        # A separately launched local Qwen service has no provider account quota;
        # managed_locally controls service lifecycle, not location.
        local_qwen = self.request_profile == "qwen" and (
            self.self_hosted or urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        )
        enforce_provider_limit = not self.managed_locally and not local_qwen
        if enforce_provider_limit and self.max_concurrency > remote_limit:
            raise ValueError(
                f"externally managed runtime.max_concurrency must not exceed {remote_limit}"
            )
        if self.request_profile not in {"qwen", "generic"}:
            raise ValueError("runtime.request_profile must be qwen or generic")
        if self.managed_locally and self.model_path is None:
            raise ValueError("locally managed runtime requires model_path")
        if not self.frozen:
            raise ValueError("runtime must remain fixed during Proposer/Solver collection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "served_model": self.served_model,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "max_concurrency": self.max_concurrency,
            "managed_locally": self.managed_locally,
            "frozen": self.frozen,
            "enable_thinking": self.enable_thinking,
            "max_tokens": self.max_tokens,
            **({"temperature": self.temperature} if self.temperature else {}),
            **({"self_hosted": True} if self.self_hosted else {}),
        }


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = False
    service_url: str = "http://127.0.0.1:8010/retrieve"
    top_k: int = 3
    timeout_s: float = 120.0
    max_tool_rounds: int = 3
    # HotpotQA may also search in addition to its supplied context.
    hotpotqa_search_enabled: bool = False
    # Search-R1 passage cut; 0 keeps whole passages.
    passage_chars: int = 0

    def validate(self) -> None:
        if self.hotpotqa_search_enabled and not self.enabled:
            raise ValueError("HotpotQA search requires retrieval.enabled")
        if not self.enabled:
            return
        if not self.service_url.startswith(("http://", "https://")):
            raise ValueError("retrieval.service_url must be an HTTP(S) URL")
        if min(self.top_k, self.max_tool_rounds) <= 0 or self.timeout_s <= 0:
            raise ValueError("retrieval limits and timeout must be positive")


@dataclass(frozen=True)
class AIMEActionConfig:
    enabled: bool = False
    timeout_s: float = 5.0
    max_output_chars: int = 8000
    max_code_chars: int = 12000
    memory_limit_mb: int = 1024
    max_initial_calls: int = 3
    max_revision_calls: int = 1
    max_total_calls: int = 4

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.timeout_s <= 0:
            raise ValueError("aime_actions.timeout_s must be positive")
        if min(self.max_output_chars, self.max_code_chars, self.memory_limit_mb) <= 0:
            raise ValueError("aime_actions limits must be positive")
        if min(self.max_initial_calls, self.max_revision_calls, self.max_total_calls) < 0:
            raise ValueError("aime_actions call budgets must be non-negative")
        if self.max_initial_calls + self.max_revision_calls > self.max_total_calls:
            raise ValueError("aime_actions phase call budgets exceed max_total_calls")


@dataclass(frozen=True)
class ALFWorldConfig:
    enabled: bool = False
    data_root: Path = Path("datasets/alfworld/data_assets/json_2.1.1")
    max_episode_steps: int = 50
    max_rollout_steps: int = 400
    max_observation_chars: int = 12_000
    max_initial_calls: int = 50
    max_revision_calls: int = 50
    max_total_calls: int = 100
    task_prompt_source: str = "dataset"

    def validate(self) -> None:
        if self.task_prompt_source not in {"dataset", "environment_reset"}:
            raise ValueError("alfworld.task_prompt_source must be dataset or environment_reset")
        if not self.enabled:
            return
        if not self.data_root.is_dir():
            raise ValueError("alfworld.data_root must be an existing directory")
        if (
            min(
                self.max_episode_steps,
                self.max_rollout_steps,
                self.max_observation_chars,
            )
            <= 0
        ):
            raise ValueError("alfworld limits must be positive")
        budgets = (
            self.max_initial_calls,
            self.max_revision_calls,
            self.max_total_calls,
        )
        if min(budgets) < 0:
            raise ValueError("alfworld call budgets must be non-negative")
        if budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("alfworld phase call budgets exceed max_total_calls")


def _default_role(name: str, port: int, *, trainable: bool = True) -> RoleModelConfig:
    return RoleModelConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        served_model="Qwen3.5-9B",
        base_model_path=Path("models/Qwen3.5-9B"),
        checkpoint_path=Path(f"state/checkpoints/{name}"),
        trainable=trainable,
    )


@dataclass(frozen=True)
class AdaptiveApplicationConfig:
    proposer_model: RoleModelConfig = field(default_factory=lambda: _default_role("proposer", 8001))
    solver_model: RoleModelConfig = field(default_factory=lambda: _default_role("solver", 8002))
    runtime: FixedRuntimeConfig = field(
        default_factory=lambda: FixedRuntimeConfig(
            base_url="http://127.0.0.1:8003/v1",
            served_model="Qwen3.5-9B",
            model_path=Path("models/Qwen3.5-9B"),
        )
    )
    runtime_name: str = "default"
    additional_runtimes: dict[str, FixedRuntimeConfig] = field(default_factory=dict)
    worker_runtime_routes: tuple[str, ...] = ("default",)
    # Logical model choices; physical endpoint selection stays inside each pool.
    runtime_endpoint_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    endpoint_pool_retry_attempts: int = 2
    endpoint_pool_retry_backoff_s: float = 1.0
    endpoint_pool_member_queue_wait_s: float = 0.5
    skill_distiller_runtime: str = "default"
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    verifier: str = "none"
    seed: int = 0
    skillbank_enabled: bool = True
    pats: PatsConfig = field(default_factory=PatsConfig)
    skillbank_prompt_token_budget: int = 1024
    skillbank_retrieval_min_score: float | None = None
    skillbank_path: Path = Path("state/solver_skillbank.json")
    skill_cases_path: Path = Path("state/solver_skill_cases.json")
    skillbank_retrieve_top_k: int = 3
    skillbank_embedding_model_path: str | Path | None = "intfloat/e5-base-v2"
    trace_path: Path = Path("state/traces.jsonl")
    allocated_gpu_ids: tuple[int, ...] = (0,)
    proposer_gpu_id: int = 0
    solver_gpu_id: int = 0
    runtime_gpu_id: int = 0
    allow_policy_gpu_colocation: bool = True
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    aime_actions: AIMEActionConfig = field(default_factory=AIMEActionConfig)
    alfworld: ALFWorldConfig = field(default_factory=ALFWorldConfig)
    director_prompt_variant: str = "r2orch"
    # Parsed from [r2orch].
    r2orch: R2OrchConfig = field(default_factory=R2OrchConfig)

    def validate(self) -> None:
        self.proposer_model.validate("proposer")
        self.solver_model.validate("solver")
        runtime_pool = self.runtime_pool()
        for runtime in runtime_pool.values():
            runtime.validate()
        self.retrieval.validate()
        self.aime_actions.validate()
        self.alfworld.validate()
        from .director import DIRECTOR_PROMPT_VARIANTS
        if self.director_prompt_variant not in DIRECTOR_PROMPT_VARIANTS:
            raise ValueError("director.prompt_variant must be 'r2orch'")
        if not self.worker_runtime_routes:
            raise ValueError("runtime_routing.worker_routes cannot be empty")
        unknown_routes = set(self.all_worker_routes()) - set(runtime_pool)
        if unknown_routes:
            raise ValueError("unknown worker runtime routes: " + ", ".join(sorted(unknown_routes)))
        for logical, members in self.runtime_endpoint_pools.items():
            if (
                logical not in runtime_pool
                or len(members) < 2
                or len(set(members)) != len(members)
                or set(members) - set(runtime_pool)
            ):
                raise ValueError("invalid endpoint pool: " + logical)
        if (
            self.endpoint_pool_retry_attempts < 0
            or self.endpoint_pool_retry_backoff_s < 0
            or self.endpoint_pool_member_queue_wait_s < 0
        ):
            raise ValueError("endpoint pool retry settings must be non-negative")
        if self.skill_distiller_runtime not in runtime_pool:
            raise ValueError(
                f"unknown skill distiller runtime route: {self.skill_distiller_runtime}"
            )
        if any(runtime.managed_locally for runtime in self.additional_runtimes.values()):
            raise ValueError("additional runtimes must be externally managed")
        if self.proposer_model.identity() == self.solver_model.identity():
            raise ValueError("Proposer and Solver must use independent served model instances")
        if self.proposer_model.checkpoint_path == self.solver_model.checkpoint_path:
            raise ValueError("Proposer and Solver checkpoint paths must be different")
        if not self.proposer_model.trainable or not self.solver_model.trainable:
            raise ValueError("Proposer and Solver model roles must be trainable")
        if self.canvas.max_agents <= 0 or self.canvas.max_rounds <= 0:
            raise ValueError("canvas agent/round budgets must be positive")
        if (
            self.canvas.max_total_tokens <= 0
            or self.canvas.relay_max_chars <= 0
            or self.canvas.feedback_max_chars <= 0
            or self.canvas.artifact_summary_max_chars <= 0
        ):
            raise ValueError("canvas token and context budgets must be positive")
        dataset_token_budgets = {
            canonical_dataset_name(dataset): int(limit)
            for dataset, limit in self.canvas.max_total_tokens_by_dataset.items()
        }
        if any(not dataset or limit <= 0 for dataset, limit in dataset_token_budgets.items()):
            raise ValueError(
                "canvas.max_total_tokens_by_dataset requires non-empty dataset keys "
                "and positive limits"
            )
        minimum_canvas_token_budget = min(
            (self.canvas.max_total_tokens, *dataset_token_budgets.values())
        )
        if not 0 <= self.canvas.graph_growth_token_reserve < minimum_canvas_token_budget:
            raise ValueError(
                "canvas graph_growth_token_reserve must be non-negative and below "
                "every configured max_total_tokens budget"
            )
        if self.canvas.worker_latency_window <= 0:
            raise ValueError("canvas Worker latency sample limits must be positive")
        if not 0 < self.canvas.worker_token_quantile <= 1:
            raise ValueError("canvas.worker_token_quantile must be in (0, 1]")
        if (
            min(
                self.canvas.worker_token_window,
                self.canvas.worker_token_min_samples,
                self.canvas.worker_token_cold_start,
                self.canvas.finalization_token_reserve,
            )
            <= 0
        ):
            raise ValueError("canvas Worker token sample limits and reserves must be positive")
        if self.canvas.worker_token_min_samples > self.canvas.worker_token_window:
            raise ValueError("canvas.worker_token_min_samples cannot exceed worker_token_window")
        if self.canvas.finalization_token_reserve >= minimum_canvas_token_budget:
            raise ValueError(
                "canvas.finalization_token_reserve must be below every configured "
                "max_total_tokens budget"
            )
        if (
            min(
                self.canvas.repair_recent_action_limit,
                self.canvas.semantic_no_progress_limit,
                self.canvas.output_selection_budget,
            )
            <= 0
        ):
            raise ValueError(
                "canvas repair history, progress, and selection limits must be positive"
            )
        if not 0.0 <= self.canvas.bidirectional_revision_confidence_threshold <= 1.0:
            raise ValueError("canvas.bidirectional_revision_confidence_threshold must be in [0, 1]")
        if self.verifier not in {
            "none",
            "auto",
            "exact_match",
            "multi_answer_exact_match",
            "flowsteer_qa",
            "token_f1",
            "numeric",
            "multiple_choice",
        }:
            raise ValueError("unsupported verifier.mode")
        self.pats.validate()
        if self.skillbank_prompt_token_budget <= 0:
            raise ValueError("skill prompt token budget must be positive")
        if (
            self.skillbank_retrieval_min_score is not None
            and not -1 <= self.skillbank_retrieval_min_score <= 1
        ):
            raise ValueError("skill retrieval cutoff must be in [-1,1]")
        if self.skillbank_retrieve_top_k <= 0:
            raise ValueError("solver_skillbank.retrieve_top_k must be positive")
        allowed_physical_gpus = _allowed_physical_gpu_ids()
        if not self.allocated_gpu_ids or any(
            gpu not in allowed_physical_gpus for gpu in self.allocated_gpu_ids
        ):
            raise ValueError(
                "resources.allocated_gpu_ids must be a subset of "
                "R2ORCH_ALLOWED_PHYSICAL_GPUS="
                + ",".join(str(gpu) for gpu in sorted(allowed_physical_gpus))
            )
        policy_gpus = (self.proposer_gpu_id, self.solver_gpu_id)
        if self.proposer_gpu_id == self.solver_gpu_id:
            if not self.allow_policy_gpu_colocation:
                raise ValueError("locally managed policy GPU assignments must be distinct")
        assigned = policy_gpus
        if self.runtime.managed_locally:
            if self.runtime_gpu_id in policy_gpus and not self.allow_policy_gpu_colocation:
                raise ValueError(
                    "locally managed runtime GPU overlap requires explicit GPU colocation"
                )
            assigned = (*assigned, self.runtime_gpu_id)
        if any(gpu not in self.allocated_gpu_ids for gpu in assigned):
            raise ValueError("all service GPUs must be included in resources.allocated_gpu_ids")

    def __post_init__(self) -> None:
        # PATS audits/refines the actual Director variant, including dataclass
        # replacements made by async collection. It has no independent override.
        object.__setattr__(self, "pats", replace(
            self.pats,
            director_prompt_variant=self.director_prompt_variant,
            director_prompt_variant_by_dataset={
                canonical_dataset_name(dataset): "r2orch"
                for dataset in self.canvas.submission_protocol_by_dataset
            },
        ))

    def model_manifest(self) -> dict[str, Any]:
        runtime_pool = self.runtime_pool()
        from .execution_contract import execution_semantics
        from .run_config import r2orch_semantics_payload

        return {
            "execution_semantics": execution_semantics(
                self.director_prompt_variant,
                admission_config={"canvas": canvas_config_record(self.canvas),
                                  "alfworld_task_prompt_source": self.alfworld.task_prompt_source},
                r2orch=r2orch_semantics_payload(self),
            ),
            "policies": {
                "proposer": self.proposer_model.to_dict(),
                "solver": self.solver_model.to_dict(),
            },
            "runtime_environment": self.runtime.to_dict(),
            "runtime_environments": {
                name: runtime.to_dict() for name, runtime in runtime_pool.items()
            },
            "runtime_routing": {
                "peer_selection_policy": "removed_direct_neighbors",
                "model_selection_policy": "director_set_model",
                "action_protocol": "director_model",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "counterfactual_execution": "full_graph",
                "worker_routes": list(self.worker_runtime_routes),
                "endpoint_pools": {
                    key: list(value) for key, value in self.runtime_endpoint_pools.items()
                },
                "endpoint_pool_retry_attempts": self.endpoint_pool_retry_attempts,
                "endpoint_pool_retry_backoff_s": self.endpoint_pool_retry_backoff_s,
                "endpoint_pool_member_queue_wait_s": (self.endpoint_pool_member_queue_wait_s),
                "skill_distiller": self.skill_distiller_runtime,
            },
            "canvas_execution": {
                "worker_protocol_status_version": "worker_protocol_status",
                "fallback_max_total_tokens": self.canvas.max_total_tokens,
                "max_total_tokens_by_dataset": {
                    canonical_dataset_name(dataset): int(limit)
                    for dataset, limit in sorted(self.canvas.max_total_tokens_by_dataset.items())
                },
                "bidirectional_revision_confidence_threshold": (
                    self.canvas.bidirectional_revision_confidence_threshold
                ),
                "bidirectional_revision_wave_budget": 1,
            },
            "retrieval": asdict(self.retrieval),
            "aime_actions": asdict(self.aime_actions),
            "alfworld": {
                **asdict(self.alfworld),
                "data_root": str(self.alfworld.data_root),
                "session_scope": "per_task",
                "worker_protocol": "alfworld_protocol",
            },
            "director_prompt": {
                "variant": self.director_prompt_variant,
                "variant_by_dataset": dict(self.pats.director_prompt_variant_by_dataset),
            },
        }

    def all_worker_routes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.worker_runtime_routes))

    def runtime_pool(self) -> dict[str, FixedRuntimeConfig]:
        name = self.runtime_name.strip()
        if not name:
            raise ValueError("runtime.name cannot be empty")
        if name in self.additional_runtimes:
            raise ValueError(f"duplicate runtime name: {name}")
        return {name: self.runtime, **self.additional_runtimes}


def load_adaptive_config(path: str | Path) -> AdaptiveApplicationConfig:
    source = Path(path).resolve()
    _load_project_env(source)
    with source.open("rb") as handle:
        payload = tomllib.load(handle)
    root = source.parent.parent if source.parent.name == "configs" else source.parent
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("adaptive config requires [models.proposer] and [models.solver]")
    r2orch = payload.get("r2orch")
    if not isinstance(r2orch, dict):
        raise ValueError("adaptive config requires a [r2orch] table")
    missing = [name for name in ("proposer", "solver") if name not in models]
    if missing:
        raise ValueError("missing model sections: " + ", ".join(missing))
    canvas = payload.get("canvas", {})
    if not isinstance(canvas, dict):
        raise ValueError("canvas must be a TOML table")
    raw_dataset_token_budgets = canvas.get(
        "max_total_tokens_by_dataset",
        DEFAULT_DATASET_MAX_TOTAL_TOKENS,
    )
    if not isinstance(raw_dataset_token_budgets, dict):
        raise ValueError("canvas.max_total_tokens_by_dataset must be a TOML table")
    skills = payload.get("solver_skillbank", {})
    trace = payload.get("trace", {})
    verifier = payload.get("verifier", {})
    resources = payload.get("resources", {})
    retrieval = payload.get("retrieval", {})
    if not isinstance(retrieval, dict):
        raise ValueError("retrieval must be a TOML table")
    aime_actions = payload.get("aime_actions", {})
    alfworld = payload.get("alfworld", {})
    director = payload.get("director", {})
    if not isinstance(director, dict):
        raise ValueError("director must be a TOML table")
    runtime_payload = payload.get("runtime", {})
    if not isinstance(runtime_payload, dict):
        raise ValueError("runtime must be a TOML table")
    runtime_name = str(runtime_payload.get("name", "default")).strip()
    additional_runtime_payload = payload.get("runtimes", {})
    if not isinstance(additional_runtime_payload, dict):
        raise ValueError("runtimes must be a TOML table")
    runtime_routing = payload.get("runtime_routing", {})
    if not isinstance(runtime_routing, dict):
        raise ValueError("runtime_routing must be a TOML table")
    config = AdaptiveApplicationConfig(
        proposer_model=_role_model(models["proposer"], root, "proposer", trainable=True),
        solver_model=_role_model(models["solver"], root, "solver", trainable=True),
        runtime=_fixed_runtime(runtime_payload, root),
        runtime_name=runtime_name,
        additional_runtimes={
            str(name): _fixed_runtime(value, root)
            for name, value in additional_runtime_payload.items()
        },
        worker_runtime_routes=tuple(
            str(value) for value in runtime_routing.get("worker_routes", [runtime_name])
        ),
        runtime_endpoint_pools={
            str(key): tuple(str(member) for member in value)
            for key, value in runtime_routing.get("endpoint_pools", {}).items()
        },
        endpoint_pool_retry_attempts=int(runtime_routing.get("pool_retry_attempts", 2)),
        endpoint_pool_retry_backoff_s=float(runtime_routing.get("pool_retry_backoff_s", 1.0)),
        endpoint_pool_member_queue_wait_s=float(
            runtime_routing.get("pool_member_queue_wait_s", 0.5)
        ),
        skill_distiller_runtime=str(runtime_routing.get("skill_distiller", runtime_name)),
        canvas=CanvasConfig(
            submission_protocol_by_dataset={
                canonical_dataset_name(dataset): str(protocol)
                for dataset, protocol in canvas.get("submission_protocol_by_dataset", {}).items()
            },
            submission_journal_dir=str(_path(canvas.get("submission_journal_dir"), root, "state/submissions")),
            max_recovery_executions=int(canvas.get("max_recovery_executions", 2)),
            max_agents=int(canvas.get("max_agents", 8)),
            max_rounds=int(canvas.get("max_rounds", 20)),
            max_total_tokens=int(canvas.get("max_total_tokens", 32_768)),
            max_total_tokens_by_dataset={
                canonical_dataset_name(dataset): int(limit)
                for dataset, limit in raw_dataset_token_budgets.items()
            },
            relay_max_chars=int(canvas.get("relay_max_chars", 4000)),
            feedback_max_chars=int(canvas.get("feedback_max_chars", 6000)),
            artifact_summary_max_chars=int(canvas.get("artifact_summary_max_chars", 320)),
            graph_growth_token_reserve=int(canvas.get("graph_growth_token_reserve", 8192)),
            worker_latency_window=int(canvas.get("worker_latency_window", 64)),
            worker_token_quantile=float(canvas.get("worker_token_quantile", 0.95)),
            worker_token_window=int(canvas.get("worker_token_window", 64)),
            worker_token_min_samples=int(canvas.get("worker_token_min_samples", 3)),
            worker_token_cold_start=int(canvas.get("worker_token_cold_start", 4096)),
            finalization_token_reserve=int(canvas.get("finalization_token_reserve", 2048)),
            repair_recent_action_limit=int(canvas.get("repair_recent_action_limit", 5)),
            semantic_no_progress_limit=int(canvas.get("semantic_no_progress_limit", 2)),
            output_selection_budget=int(canvas.get("output_selection_budget", 1)),
            bidirectional_revision_confidence_threshold=float(
                canvas.get("bidirectional_revision_confidence_threshold", 0.8)
            ),
            parallel_components_max=canvas.get("parallel_components_max", 1),
        ),
        verifier=str(verifier.get("mode", "none")),
        seed=int(runtime_routing.get("seed", 0)),
        skillbank_enabled=bool(skills.get("enabled", True)),
        pats=PatsConfig(**skills.get("pats", {})),
        skillbank_prompt_token_budget=int(skills.get("prompt_token_budget", 1024)),
        skillbank_retrieval_min_score=(
            float(skills["retrieval_min_score"]) if "retrieval_min_score" in skills else None
        ),
        skillbank_path=_path(skills.get("path"), root, "state/solver_skillbank.json"),
        skill_cases_path=_path(skills.get("cases_path"), root, "state/solver_skill_cases.json"),
        skillbank_retrieve_top_k=int(skills.get("retrieve_top_k", 3)),
        skillbank_embedding_model_path=_model_reference(
            skills.get("embedding_model_path", "intfloat/e5-base-v2"), root
        ),
        trace_path=_path(trace.get("path"), root, "state/traces.jsonl"),
        allocated_gpu_ids=tuple(int(value) for value in resources.get("allocated_gpu_ids", [0])),
        proposer_gpu_id=int(resources.get("proposer_gpu_id", 0)),
        solver_gpu_id=int(resources.get("solver_gpu_id", 0)),
        runtime_gpu_id=int(resources.get("runtime_gpu_id", 0)),
        allow_policy_gpu_colocation=bool(resources.get("allow_policy_gpu_colocation", True)),
        retrieval=RetrievalConfig(
            enabled=bool(retrieval.get("enabled", False)),
            service_url=str(retrieval.get("service_url", "http://127.0.0.1:8010/retrieve")),
            top_k=int(retrieval.get("top_k", 3)),
            timeout_s=float(retrieval.get("timeout_s", 120.0)),
            max_tool_rounds=int(retrieval.get("max_tool_rounds", 3)),
            hotpotqa_search_enabled=bool(retrieval.get("hotpotqa_search_enabled", False)),
            passage_chars=int(retrieval.get("passage_chars", 0)),
        ),
        aime_actions=AIMEActionConfig(
            enabled=bool(aime_actions.get("enabled", False)),
            timeout_s=float(aime_actions.get("timeout_s", 5.0)),
            max_output_chars=int(aime_actions.get("max_output_chars", 8000)),
            max_code_chars=int(aime_actions.get("max_code_chars", 12000)),
            memory_limit_mb=int(aime_actions.get("memory_limit_mb", 1024)),
            max_initial_calls=int(aime_actions.get("max_initial_calls", 3)),
            max_revision_calls=int(aime_actions.get("max_revision_calls", 1)),
            max_total_calls=int(aime_actions.get("max_total_calls", 4)),
        ),
        alfworld=ALFWorldConfig(
            task_prompt_source=str(alfworld.get("task_prompt_source", "dataset")),
            enabled=bool(alfworld.get("enabled", False)),
            data_root=_path(
                alfworld.get("data_root"),
                root,
                "datasets/alfworld/data_assets/json_2.1.1",
            ),
            max_episode_steps=int(alfworld.get("max_episode_steps", 50)),
            max_rollout_steps=int(alfworld.get("max_rollout_steps", 400)),
            max_observation_chars=int(alfworld.get("max_observation_chars", 12_000)),
            max_initial_calls=int(alfworld.get("max_initial_calls", 50)),
            max_revision_calls=int(alfworld.get("max_revision_calls", 50)),
            max_total_calls=int(alfworld.get("max_total_calls", 100)),
        ),
        director_prompt_variant=str(director.get("prompt_variant", "r2orch")).strip().casefold(),
        r2orch=R2OrchConfig.from_dict(r2orch),
    )
    config.validate()
    return config


def _role_model(
    payload: object,
    root: Path,
    name: str,
    *,
    trainable: bool,
) -> RoleModelConfig:
    if not isinstance(payload, dict):
        raise ValueError(f"models.{name} must be a TOML table")
    default_port = {"proposer": 8001, "solver": 8002}[name]
    return RoleModelConfig(
        base_url=str(payload.get("base_url", f"http://127.0.0.1:{default_port}/v1")),
        api_key=str(payload.get("api_key", "EMPTY")),
        served_model=str(payload.get("served_model", "Qwen3.5-9B")),
        base_model_path=_path(payload.get("base_model_path"), root, "models/Qwen3.5-9B"),
        checkpoint_path=_path(payload.get("checkpoint_path"), root, f"state/checkpoints/{name}"),
        timeout_s=float(payload.get("timeout_s", 120.0)),
        trainable=bool(payload.get("trainable", trainable)),
        enable_thinking=payload.get("enable_thinking"),
        temperature=(float(payload["temperature"]) if "temperature" in payload else None),
        top_p=(float(payload["top_p"]) if "top_p" in payload else None),
        top_k=(int(payload["top_k"]) if "top_k" in payload else None),
    )


def _fixed_runtime(payload: object, root: Path) -> FixedRuntimeConfig:
    if not isinstance(payload, dict):
        raise ValueError("runtime must be a TOML table")
    managed_locally = bool(payload.get("managed_locally", True))
    raw_model_path = payload.get("model_path")
    model_path = (
        _path(raw_model_path, root, "models/Qwen3.5-9B")
        if raw_model_path is not None or managed_locally
        else None
    )
    default_max_concurrency = 4 if managed_locally else REMOTE_RUNTIME_MAX_CONCURRENCY
    return FixedRuntimeConfig(
        base_url=str(payload.get("base_url", "http://127.0.0.1:8003/v1")),
        api_key=str(payload.get("api_key", "EMPTY")),
        served_model=str(payload.get("served_model", "Qwen3.5-9B")),
        model_path=model_path,
        timeout_s=float(payload.get("timeout_s", 120.0)),
        request_profile=str(payload.get("request_profile", "qwen")).strip().lower(),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        max_tokens=int(payload.get("max_tokens", 2048)),
        max_concurrency=int(payload.get("max_concurrency", default_max_concurrency)),
        managed_locally=managed_locally,
        frozen=bool(payload.get("frozen", True)),
        temperature=float(payload.get("temperature", 0.0)),
        self_hosted=bool(payload.get("self_hosted", False)),
    )


def _load_project_env(config_path: Path) -> None:
    """Load the nearest project .env without overriding the caller environment."""

    env_path = next(
        (parent / ".env" for parent in config_path.parents if (parent / ".env").is_file()),
        None,
    )
    if env_path is None:
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry at {env_path}:{line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid .env variable name at {env_path}:{line_number}")
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid .env value at {env_path}:{line_number}") from exc
        if len(parsed) > 1:
            raise ValueError(f"unquoted whitespace in .env value at {env_path}:{line_number}")
        value = parsed[0] if parsed else ""
        os.environ.setdefault(name, value)


def _path(value: object, root: Path, default: str) -> Path:
    path = Path(str(value if value is not None else default)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _model_reference(value: object, root: Path) -> str | Path | None:
    source = str(value or "").strip()
    if not source:
        return None
    path = Path(source).expanduser()
    if path.is_absolute() or source.startswith(".") or (root / path).exists():
        return path if path.is_absolute() else (root / path).resolve()
    return source


@dataclass(frozen=True)
class AdaptiveApplicationResult:
    run_id: str
    task: TaskSpec
    solver_result: AdaptiveSolverResult
    skills_used: tuple[str, ...]
    trace_store_path: str

    def to_dict(self) -> dict[str, Any]:
        result = self.solver_result.to_dict()
        submission = self.solver_result.answer_submission
        worker_token_in, worker_token_out = _unique_worker_token_totals(
            self.solver_result.trace.events,
            run_id=self.run_id,
        )
        return {
            "run_id": self.run_id,
            "task": task_to_public_dict(self.task),
            "answer": (
                submission.submitted_answer
                if submission
                else self.solver_result.director_run.output
            ),
            "raw_answer": self.solver_result.director_run.output,
            "candidate_output": self.solver_result.director_run.candidate_output,
            "submission_receipt": result["submission_receipt"],
            "submission_contract_version": result["submission_contract_version"],
            "outcome_decision": result["outcome_decision"],
            "submission_status": self.task.metadata.get("submission_status"),
            "answer_submission": submission.to_dict() if submission else None,
            "finished": self.solver_result.director_run.finished,
            "verification": result["verification"],
            "final_graph": self.solver_result.director_run.graph,
            "skills_used": list(self.skills_used),
            "trace_run_id": self.solver_result.trace.run_id,
            "trace_store_path": self.trace_store_path,
            "token_in": worker_token_in,
            "token_out": worker_token_out,
            "model_roles": self.task.metadata.get("model_roles", {}),
        }


def _unique_worker_token_totals(trace_events: list[Any], *, run_id: str) -> tuple[int, int]:
    """Sum each Worker execution once, including overwritten peer initial artifacts.

    Canvas emits the latest ExecutionReport with a final event so downstream
    audit can see the selected artifact. That event may contain no new Worker
    execution. The per-execution ledger retains initial and revision usage even
    when the report's latest-artifact map has replaced the initial artifact.
    Artifact identity, scoped by run, deduplicates final-event replay and caches.
    """

    seen_artifacts: set[str] = set()
    token_in = token_out = 0
    for event in trace_events:
        execution = event.payload.get("execution")
        if not isinstance(execution, dict):
            continue
        usage_events = [
            item for item in execution.get("execution_events", [])
            if isinstance(item, dict) and item.get("usage_schema") == "worker_execution_usage"
        ]
        for item in usage_events:
            identity = f"{run_id}:{item['artifact_id']}"
            if item.get("cache_hit"):
                seen_artifacts.add(identity)
                continue
            if identity in seen_artifacts:
                continue
            seen_artifacts.add(identity)
            token_in += int(item.get("token_in", 0) or 0)
            token_out += int(item.get("token_out", 0) or 0)
    return token_in, token_out


class AdaptiveSolverApplication:
    """Runnable composition root for the complete inference-only Adaptive Solver."""

    def __init__(
        self,
        *,
        config: AdaptiveApplicationConfig,
        solver: AdaptiveWorkflowSolver,
        runtime: MultiAgentRuntime,
        skillbank: Any,
        owned_backends: tuple[ChatBackend, ...] = (),
    ) -> None:
        self.config = config
        self.solver = solver
        self.runtime = runtime
        self.skillbank = skillbank
        self.owned_backends = owned_backends

    def close(self) -> None:
        closed: set[int] = set()
        backends, self.owned_backends = self.owned_backends, ()
        first_error = None
        for backend in backends:
            if id(backend) in closed:
                continue
            closed.add(id(backend))
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def set_rollout_deadline(self, deadline: RolloutDeadline | None) -> None:
        """Install one deadline across Director, Canvas, Worker, and API gates."""

        self.solver.rollout_deadline = deadline
        set_executor_deadline = getattr(self.runtime.executor, "set_deadline_context", None)
        if callable(set_executor_deadline):
            set_executor_deadline(deadline)
        for backend in self.owned_backends:
            setter = getattr(backend, "set_deadline_context", None)
            if callable(setter):
                setter(deadline)

    def solve(
        self,
        prompt: str,
        *,
        task_id: str = "task",
        task_type: str = "general",
        reference: Any = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        private_verifier_payload: dict[str, Any] | None = None,
    ) -> AdaptiveApplicationResult:
        if self.solver.rollout_deadline is not None:
            self.solver.rollout_deadline.check("application_solve_start")
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("task prompt cannot be empty")
        prospective_task = TaskSpec(
            task_id,
            prompt,
            reference=reference,
            task_type=task_type,
            metadata=dict(metadata or {}),
            private_verifier_payload=dict(private_verifier_payload or {}),
        )
        needs_reference = self.config.verifier in {
            "exact_match",
            "multi_answer_exact_match",
            "flowsteer_qa",
            "token_f1",
            "numeric",
            "multiple_choice",
        } or (self.config.verifier == "auto" and task_requires_reference(prospective_task))
        if needs_reference and reference is None:
            raise ValueError(f"{self.config.verifier} verifier requires a reference answer")
        self.runtime.reset()

        task = TaskSpec(
            task_id,
            prompt,
            reference=reference,
            task_type=task_type,
            metadata={
                **dict(metadata or {}),
                "model_roles": self.config.model_manifest(),
                "skills_enabled": self.config.skillbank_enabled,
                "action_protocol": "director_model",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
            },
            private_verifier_payload=dict(private_verifier_payload or {}),
        )
        resolved_run_id = run_id or f"adaptive-{uuid.uuid4().hex}"
        result = self.solver.solve(task, run_id=resolved_run_id)
        if self.solver.rollout_deadline is not None:
            try:
                self.solver.rollout_deadline.check("application_solve_complete")
            except WorkerWallClockLimitExceeded as exc:
                # The solver has already returned and persisted a verifier result.
                # Preserve that evidence and the policy record; downstream reward
                # and training admission still decide whether either is usable.
                verification = result.trace.verification
                if verification is None or not math.isfinite(verification.score):
                    raise
                task.metadata["postsolve_deadline_exceeded"] = exc.to_dict()
                result.trace.task.metadata["postsolve_deadline_exceeded"] = exc.to_dict()
        result.trace.task.metadata["executor_bundle_signature"] = hashlib.sha256(
            json.dumps(self.config.model_manifest(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return AdaptiveApplicationResult(
            run_id=resolved_run_id,
            task=task,
            solver_result=result,
            skills_used=result.skills_used,
            trace_store_path=str(self.config.trace_path),
        )


def create_adaptive_application(
    config: AdaptiveApplicationConfig,
    *,
    mock: bool = False,
    route_latency_tracker: RouteLatencyTracker | None = None,
    route_token_tracker: RouteTokenTracker | None = None,
    director_tokenizer: Tokenizer | None = None,
) -> AdaptiveSolverApplication:
    config.validate()
    owned_backends: list[ChatBackend] = []
    tools: dict[str, Any] = {}
    if config.retrieval.enabled:
        tools["search"] = SearchServiceTool(
            config.retrieval.service_url,
            top_k=config.retrieval.top_k,
            timeout_s=config.retrieval.timeout_s,
            passage_chars=config.retrieval.passage_chars,
        )
    if config.aime_actions.enabled:
        tools["symbolic_compute"] = SymbolicComputeTool(
            timeout_s=min(config.aime_actions.timeout_s, 3.0),
            max_output_chars=config.aime_actions.max_output_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
        tools["finite_search"] = FiniteSearchTool(
            timeout_s=min(config.aime_actions.timeout_s, 3.0),
            max_output_chars=config.aime_actions.max_output_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
        tools["python_exec"] = PythonExecutionTool(
            timeout_s=config.aime_actions.timeout_s,
            max_output_chars=config.aime_actions.max_output_chars,
            max_code_chars=config.aime_actions.max_code_chars,
            memory_limit_mb=config.aime_actions.memory_limit_mb,
        )
    alfworld_lifecycle = None
    if config.alfworld.enabled:
        from .alfworld_bridge import ALFWorldBridgeClient

        alfworld_client = ALFWorldBridgeClient(config.r2orch.paths.alfworld_python)
        owned_backends.append(alfworld_client)
        alfworld_lifecycle = ALFWorldSessionLifecycle(
            alfworld_client,
            data_root=config.alfworld.data_root,
            task_prompt_source=config.alfworld.task_prompt_source,
            max_episode_steps=config.alfworld.max_episode_steps,
            max_rollout_steps=config.alfworld.max_rollout_steps,
            max_observation_chars=config.alfworld.max_observation_chars,
        )
        tools["alfworld_step"] = ALFWorldStepTool(alfworld_lifecycle)
    action_registry = default_dataset_action_registry(
        tools,
        hotpotqa_search_enabled=config.retrieval.hotpotqa_search_enabled,
        aime_budgets=(
            config.aime_actions.max_initial_calls,
            config.aime_actions.max_revision_calls,
            config.aime_actions.max_total_calls,
        ),
        retrieval_initial_budget=config.retrieval.max_tool_rounds,
        alfworld_budgets=(
            config.alfworld.max_initial_calls,
            config.alfworld.max_revision_calls,
            config.alfworld.max_total_calls,
        ),
    )
    worker_executor: ModelAgentExecutor | RoutedModelAgentExecutor
    if mock:
        director_backend, worker_backend, distiller_backend = _mock_adaptive_backends(
            runtime_route=config.worker_runtime_routes[0],
        )
        worker_executor = ModelAgentExecutor(
            worker_backend,
            tools=tools,
            action_registry=action_registry,
            max_tool_rounds=config.retrieval.max_tool_rounds,
        )
    else:
        director_backend = OpenAICompatibleBackend(
            _gateway_config(
                config.solver_model,
                {"graph-director": (
                    0.6 if config.solver_model.temperature is None
                    else config.solver_model.temperature
                )},
                sampling_seed=config.seed,
            )
        )
        owned_backends.append(director_backend)
        runtime_backends = {
            name: _create_runtime_backend(runtime, route_name=name)
            for name, runtime in config.runtime_pool().items()
        }
        owned_backends.extend(runtime_backends.values())
        from .endpoint_pool import EndpointPoolBackend

        physical_backends = dict(runtime_backends)
        for logical, members in config.runtime_endpoint_pools.items():
            runtime_backends[logical] = EndpointPoolBackend(
                logical,
                {name: physical_backends[name] for name in members},
                config.trace_path.parent / "endpoint_pools",
                pool_retry_attempts=config.endpoint_pool_retry_attempts,
                retry_backoff_s=config.endpoint_pool_retry_backoff_s,
                member_queue_wait_s=config.endpoint_pool_member_queue_wait_s,
            )
            # Install the shared rollout clock on the pool wrapper too,
            # not only on its physical clients.
            owned_backends.append(runtime_backends[logical])
        worker_executor = RoutedModelAgentExecutor(
            runtime_backends,
            config.all_worker_routes(),
            tools=tools,
            action_registry=action_registry,
            max_tool_rounds=config.retrieval.max_tool_rounds,
        )
        distiller_backend = runtime_backends[config.skill_distiller_runtime]

    runtime = MultiAgentRuntime(
        worker_executor,
        relay_max_chars=config.canvas.relay_max_chars,
        seed=config.seed,
        bidirectional_revision_confidence_threshold=(
            config.canvas.bidirectional_revision_confidence_threshold
        ),
        parallel_components_max=config.canvas.parallel_components_max,
        route_latency_tracker=(
            route_latency_tracker
            or RouteLatencyTracker(window_size=config.canvas.worker_latency_window)
        ),
        route_token_tracker=(
            route_token_tracker or RouteTokenTracker(window_size=config.canvas.worker_token_window)
        ),
    )
    skillbank = None
    if config.skillbank_enabled:
        from .skill_store import load_bank

        skillbank = load_bank(
            config,
            embedder=(
                E5SkillEmbedder(config.skillbank_embedding_model_path)
                if config.skillbank_embedding_model_path and not mock
                else None
            ),
        )
    selected_verifier = _verifier(
        config.verifier,
        adapters={
            "alfworld_environment": ALFWorldEnvironmentVerifier(),
            **benchmark_verifier_adapters(python=config.r2orch.paths.evalplus_python or None),
        },
    )
    solver = AdaptiveWorkflowSolver(
        director_backend=director_backend,
        runtime=runtime,
        verifier=selected_verifier,
        skillbank=skillbank,
        trace_store=JSONLTraceStore(config.trace_path),
        canvas_config=config.canvas,
        runtime_routes=config.worker_runtime_routes,
        action_registry=action_registry,
        director_enable_thinking=config.solver_model.enable_thinking,
        director_tokenizer=director_tokenizer,
        worker_cost=config.r2orch.cost,
    )
    return AdaptiveSolverApplication(
        config=config,
        solver=solver,
        runtime=runtime,
        skillbank=skillbank,
        owned_backends=tuple(owned_backends),
    )


def _gateway_config(
    model: RoleModelConfig,
    temperatures: dict[str, float],
    *,
    sampling_seed: int | None = None,
) -> ModelGatewayConfig:
    return ModelGatewayConfig(
        base_url=model.base_url,
        api_key=model.api_key,
        timeout_s=model.timeout_s,
        sampling_seed=sampling_seed,
        roles={
            role: ModelRoleConfig(
                model=model.served_model,
                temperature=temperature,
                top_p=(
                    (0.95 if model.top_p is None else model.top_p)
                    if role == "graph-director" else 1.0
                ),
                top_k=(
                    (20 if model.top_k is None else model.top_k)
                    if role == "graph-director" else None
                ),
                enable_thinking=(
                    model.enable_thinking
                    if model.enable_thinking is not None
                    else role == "graph-director"
                ),
            )
            for role, temperature in temperatures.items()
        },
    )


def _runtime_gateway_config(
    runtime: FixedRuntimeConfig,
    temperatures: dict[str, float],
    *,
    route_name: str = "",
) -> ModelGatewayConfig:
    return ModelGatewayConfig(
        base_url=runtime.base_url,
        api_key=runtime.api_key,
        timeout_s=runtime.timeout_s,
        request_profile=runtime.request_profile,
        route_name=route_name,
        max_concurrency=runtime.max_concurrency,
        roles={
            role: ModelRoleConfig(
                model=runtime.served_model,
                temperature=temperature,
                enable_thinking=runtime.enable_thinking,
                max_tokens=runtime.max_tokens,
            )
            for role, temperature in temperatures.items()
        },
    )


def _create_runtime_backend(runtime: FixedRuntimeConfig, *, route_name: str = "") -> ChatBackend:
    gateway = _runtime_gateway_config(
        runtime,
        {
            "skill-distiller": 0.0,
            "worker": runtime.temperature,
        },
        route_name=route_name,
    )
    return OpenAICompatibleBackend(gateway)


def create_selfplay_snapshots(config: AdaptiveApplicationConfig) -> AlternatingSnapshots:
    """Bind self-play state to distinct Proposer and Solver checkpoint roots."""

    config.validate()
    return AlternatingSnapshots(
        proposer_snapshot=_current_policy_snapshot(
            config.proposer_model.checkpoint_path, config.proposer_model.base_model_path
        ),
        solver_snapshot=_current_policy_snapshot(
            config.solver_model.checkpoint_path, config.solver_model.base_model_path
        ),
    )


def consolidate_selfplay_skills(
    config: AdaptiveApplicationConfig,
    result: Any,
    *,
    mock: bool,
    step: int,
    cycle_dir: Path | None = None,
    tokenizer: Any | None = None,
) -> None:
    """Update the Director skill bank (``skill_store``) from one cycle's rollouts."""

    if not config.skillbank_enabled:
        return
    from .skill_store import atomic_json, consolidate

    try:
        kwargs = {"tokenizer": tokenizer} if config.pats.enabled else {}
        consolidate(config, result, step=step, mock=mock, cycle_dir=cycle_dir, **kwargs)
    except Exception as exc:
        # Raw rollout evidence remains authoritative and can be re-ingested.
        # Skill maintenance must not cancel an otherwise valid PPO update.
        import warnings

        warnings.warn(
            f"SkillBank evidence maintenance failed: {type(exc).__name__}",
            RuntimeWarning,
            stacklevel=2,
        )
        if cycle_dir is not None:
            with contextlib.suppress(OSError):
                atomic_json(
                    cycle_dir / "skillbank_maintenance_error.json",
                    {
                        "step": step,
                        "error_type": type(exc).__name__,
                        "raw_rollouts_preserved": True,
                    },
                )


def _current_policy_snapshot(checkpoint_root: Path, base_model_path: Path) -> str:
    latest = checkpoint_root / "latest.json"
    if not latest.exists():
        return str(base_model_path.resolve())
    payload = json.loads(latest.read_text(encoding="utf-8"))
    checkpoint = Path(str(payload["path"]))
    return str(checkpoint.resolve()) if checkpoint.exists() else str(base_model_path.resolve())


def _verifier(
    name: str,
    *,
    adapters: dict[str, Verifier] | None = None,
) -> Verifier | None:
    builtins: dict[str, Verifier | None] = {
        "none": None,
        "auto": AutoVerifier(adapters),
        "exact_match": ExactMatchVerifier(),
        "multi_answer_exact_match": MultiAnswerExactMatchVerifier(),
        "flowsteer_qa": FlowSteerQAVerifier(),
        "token_f1": TokenF1Verifier(),
        "numeric": NumericVerifier(),
        "multiple_choice": MultipleChoiceVerifier(),
    }
    if name in builtins:
        return builtins[name]
    selected = (adapters or {}).get(name)
    if selected is None:
        raise ValueError(f"verifier {name!r} requires a registered dataset adapter")
    return selected


# Offline R2Orch Director graphs, chosen by the sibling a mock run serves (see
# TopologyMockDirector): an ``off`` relation whose endpoint is then deleted, a 2-agent peer
# pair (its only edge is a bridge), a 3-agent chain and a triangle (no bridge
# edge).  Sibling i builds MOCK_TOPOLOGIES[i % 4].
MOCK_TOPOLOGIES = ("off_delete", "peer_pair", "chain", "triangle")


def _mock_binary(choice: str, p_on: float) -> BinaryChoiceResponse:
    return BinaryChoiceResponse(
        choice=choice,
        model="mock:graph-director",
        probabilities={"off": 1.0 - p_on, "on": p_on},
        log_probabilities={"off": math.log(1.0 - p_on), "on": math.log(p_on)},
        token_ids={"off": 256, "on": 257},
        metadata={"mock": True},
    )


def _mock_scripts(runtime_route: str) -> dict[str, tuple[list[str], list[BinaryChoiceResponse]]]:
    """Unified-protocol Director scripts (actions, off/on choices) by topology name."""

    def add(agent_id: str) -> str:
        return json.dumps({"action": "add_agent", "agent_id": agent_id})

    def prompt(target: str, scope: str) -> str:
        return json.dumps(
            {
                "action": "set_prompt", "target": target, "role": f"Mock {target}",
                "objective": "Solve the assigned task.",
                "scope": "Develop and verify a candidate result.",
                "expected_output": "Return a concise answer with essential justification.",
                "result_scope": scope,
            }
        )

    def model(target: str) -> str:
        return json.dumps({"action": "set_model", "target": target, "runtime_route": runtime_route})

    def layer(target: str, value: int) -> str:
        return json.dumps({"action": "set_layer", "target": target, "layer": value})

    def consider(source: str, target: str) -> str:
        return json.dumps({"action": "consider_relation", "source": source, "target": target})

    def finish(target: str) -> str:
        return json.dumps({"action": "finish", "target": target})

    return {
        # The only edge is a bridge: off would leave agent_2 unable to influence the output.
        "peer_pair": (
            [
                add("agent_1"), prompt("agent_1", "task_result"), model("agent_1"),
                add("agent_2"), prompt("agent_2", "subtask"), model("agent_2"),
                consider("agent_1", "agent_2"), finish("agent_1"),
            ],
            [_mock_binary("on", 0.5)],
        ),
        "chain": (
            [
                add("agent_1"), prompt("agent_1", "subtask"), model("agent_1"),
                add("agent_2"), prompt("agent_2", "subtask"), model("agent_2"),
                layer("agent_2", 1), consider("agent_1", "agent_2"),
                add("agent_3"), prompt("agent_3", "task_result"), model("agent_3"),
                layer("agent_3", 2), consider("agent_2", "agent_3"), finish("agent_3"),
            ],
            [_mock_binary("on", 0.5), _mock_binary("on", 0.5)],
        ),
        # off, then the Director repairs the disconnected graph by deleting an endpoint.
        "off_delete": (
            [
                add("agent_1"), prompt("agent_1", "subtask"), model("agent_1"),
                add("agent_2"), prompt("agent_2", "task_result"), model("agent_2"),
                consider("agent_1", "agent_2"),
                json.dumps({"action": "delete_agent", "target": "agent_1"}), finish("agent_2"),
            ],
            [_mock_binary("off", 0.5)],
        ),
    }


class TopologyMockDirector(MockBackend):
    """Offline Director whose script depends on the sibling it serves.

    ``AdaptiveWorkflowSolver.solve`` calls ``prepare_for_task`` with the task metadata
    before each rollout: the siblings cycle through ``MOCK_TOPOLOGIES`` by index.
    Without an R2Orch sibling it keeps the triangle.
    """

    def __init__(
        self,
        scripts: dict[str, tuple[list[str], list[BinaryChoiceResponse]]],
        default: str,
    ) -> None:
        super().__init__()
        self.scripts = scripts
        self.default = default
        self.topology = default
        self.prepare_for_task({})

    def prepare_for_task(self, metadata: dict[str, Any]) -> None:
        sibling = metadata.get("r2orch") or {}
        if "sibling_index" in sibling:
            topology = MOCK_TOPOLOGIES[int(sibling["sibling_index"]) % len(MOCK_TOPOLOGIES)]
        else:
            topology = self.default
        responses, binaries = self.scripts[topology]
        self.responses = deque(responses)
        self.binary_responses = deque(binaries)
        self.topology = topology


def _mock_adaptive_backends(
    *,
    runtime_route: str = "default",
) -> tuple[MockBackend, MockBackend, MockBackend]:
    def scoped(action: str, result_scope: str) -> str:
        return json.dumps({**json.loads(action), "result_scope": result_scope})

    director = MockBackend(
        [
            '{"action":"add_agent","agent_id":"analyst"}',
            scoped(
                '{"action":"set_prompt","target":"analyst",'
                '"role":"Independent analyst",'
                '"objective":"Solve the assigned task independently.",'
                '"scope":"Develop and verify a candidate result.",'
                '"expected_output":"Return a concise answer with essential justification."}',
                "subtask",
            ),
            json.dumps(
                {"action": "set_model", "target": "analyst", "runtime_route": runtime_route}
            ),
            '{"action":"add_agent","agent_id":"verifier"}',
            scoped(
                '{"action":"set_prompt","target":"verifier",'
                '"role":"Independent verifier",'
                '"objective":"Check the assigned task and produce the best answer.",'
                '"scope":"Verify correctness independently.",'
                '"expected_output":"Return the corrected direct final answer."}',
                "subtask",
            ),
            json.dumps(
                {"action": "set_model", "target": "verifier", "runtime_route": runtime_route}
            ),
            json.dumps({"action": "consider_relation", "source": "analyst", "target": "verifier"}),
            '{"action":"add_agent","agent_id":"formatter"}',
            scoped(
                '{"action":"set_prompt","target":"formatter",'
                '"role":"Final formatter",'
                '"objective":"Synthesize visible artifacts into the requested final form.",'
                '"scope":"Resolve disagreement using only visible artifacts.",'
                '"expected_output":"Return one concise direct final answer."}',
                "task_result",
            ),
            json.dumps(
                {"action": "set_model", "target": "formatter", "runtime_route": runtime_route}
            ),
            '{"action":"set_layer","target":"formatter","layer":1}',
            json.dumps({"action": "consider_relation", "source": "analyst", "target": "formatter"}),
            json.dumps({"action": "consider_relation", "source": "verifier", "target": "formatter"}),
            '{"action":"finish","target":"formatter"}',
        ],
        binary_responses=[
            BinaryChoiceResponse(
                choice="on",
                model="mock:graph-director",
                probabilities={"off": 0.5, "on": 0.5},
                log_probabilities={"off": -0.6931471805599453, "on": -0.6931471805599453},
                token_ids={"off": 256, "on": 257},
                metadata={"mock": True},
            )
            for _ in range(3)
        ],
    )
    scripts = _mock_scripts(runtime_route)
    scripts["triangle"] = (list(director.responses), list(director.binary_responses))
    director = TopologyMockDirector(scripts, default="triangle")
    worker = MockBackend(
        handler=lambda _messages, _role: json.dumps(
            {
                "answer": "mock adaptive output",
                "summary": "mock adaptive output",
                "confidence": 1.0,
                "evidence": [],
                "unresolved_issues": [],
                "tool_summary": [],
            }
        )
    )
    distiller = MockBackend(
        handler=lambda _messages, _role: json.dumps(
            {
                "name": "Verify before finalizing",
                "description": "Add an explicit verification pass after a failed workflow.",
                "trigger": "A prior workflow of the same task type failed verification.",
                "plan": "Compare the failed trace with a successful trace and verify the output.",
                "pitfall": "Do not copy an answer without checking it.",
                "constraint": "Workflow Solver only.",
                "kind": "verification",
            }
        )
    )
    return director, worker, distiller
