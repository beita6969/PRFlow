from __future__ import annotations

from dataclasses import asdict, dataclass, field

DEFAULT_DATASET_MAX_TOTAL_TOKENS = {
    "aime": 240_000,
    "nq_open": 240_000,
    "hotpotqa": 240_000,
    "alfworld": 350_000,
}
# R2Orch datasets: valid submission-protocol override keys. They are not
# in the default budget table above.
R2ORCH_EXTRA_DATASETS = frozenset({"medqa", "mbppplus"})

_DATASET_ALIASES = {
    "aime": "aime",
    "nq": "nq_open",
    "nq-open": "nq_open",
    "nq_open": "nq_open",
    "natural_questions": "nq_open",
    "hotpot": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "alfworld": "alfworld",
    # The six benchmarks.
    "aime_2026": "aime",
    "medqa": "medqa",
    "med_qa": "medqa",
    "mbppplus": "mbppplus",
    "mbpp_plus": "mbppplus",
    "mbpp+": "mbppplus",
}


def canonical_dataset_name(dataset: object) -> str:
    key = str(dataset or "").strip().casefold()
    return _DATASET_ALIASES.get(key, key)


@dataclass(frozen=True)
class ModelRoleConfig:
    model: str = "Qwen3.5-9B"
    system_prompt: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 2048
    enable_thinking: bool = False
    reasoning_effort: str | None = None


@dataclass
class ModelGatewayConfig:
    """Configuration for one model instance, never a cross-policy model bundle."""

    base_url: str = "http://127.0.0.1:8003/v1"
    api_key: str = "EMPTY"
    timeout_s: float = 120.0
    request_profile: str = "qwen"
    # Empty for local policy models. Runtime routes set this so a rollout can
    # apply a narrowly scoped route/dataset timeout override.
    route_name: str = ""
    # Independent requests remain separate at the protocol layer.  This limit
    # only controls how many may be in flight to one endpoint so vLLM/SGLang can
    # continuously batch local generations without flooding remote providers.
    max_concurrency: int = 64
    # Local policy rollouts use an explicit provider seed so the recorded
    # rollout seed is the seed that actually governed sampling.  Fixed remote
    # runtime models leave this unset.
    sampling_seed: int | None = None
    roles: dict[str, ModelRoleConfig] = field(
        default_factory=lambda: {
            "worker": ModelRoleConfig(model="Qwen3.5-9B", temperature=0.0),
        }
    )


# CanvasConfig fields that only schedule work (throughput) and never change what
# a run computes. They are excluded from every execution-semantics / frozen
# protocol payload (see canvas_config_record) so toggling them neither
# invalidates a resumed run nor changes the action-protocol hash.
CANVAS_SCHEDULING_FIELDS = frozenset({"parallel_components_max"})


@dataclass
class CanvasConfig:
    submission_protocol_by_dataset: dict[str, str] = field(default_factory=dict)
    submission_journal_dir: str = "state/submissions"
    max_recovery_executions: int = 2
    max_agents: int = 8
    max_rounds: int = 20
    max_total_tokens: int = 32_768
    max_total_tokens_by_dataset: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_DATASET_MAX_TOTAL_TOKENS)
    )
    relay_max_chars: int = 4000
    feedback_max_chars: int = 6000
    artifact_summary_max_chars: int = 320
    graph_growth_token_reserve: int = 8192
    worker_latency_window: int = 64
    worker_token_quantile: float = 0.95
    worker_token_window: int = 64
    worker_token_min_samples: int = 3
    worker_token_cold_start: int = 4096
    finalization_token_reserve: int = 2048
    repair_recent_action_limit: int = 5
    semantic_no_progress_limit: int = 2
    output_selection_budget: int = 1
    bidirectional_revision_confidence_threshold: float = 0.8
    # Scheduling knob: how many independent components of one graph layer may
    # have their Worker calls in flight at once. 1 keeps the strictly serial
    # execution (no threads). Results are identical for every value, so this is
    # excluded from execution semantics (CANVAS_SCHEDULING_FIELDS).
    parallel_components_max: int = 1

    def __post_init__(self) -> None:
        if type(self.parallel_components_max) is not int or self.parallel_components_max < 1:
            raise ValueError("canvas.parallel_components_max must be a positive integer")
        protocols = {canonical_dataset_name(key): value
                     for key, value in self.submission_protocol_by_dataset.items()}
        for dataset, protocol in protocols.items():
            if (dataset not in DEFAULT_DATASET_MAX_TOTAL_TOKENS
                    and dataset not in R2ORCH_EXTRA_DATASETS):
                raise ValueError("dataset submission protocol override requires a supported dataset")
            if protocol not in {"unified_task_result"}:
                raise ValueError("unsupported dataset submission_protocol")
        if self.max_recovery_executions < 0:
            raise ValueError("max_recovery_executions must be non-negative")

    def token_budget_for_dataset(self, dataset: object) -> tuple[str, int]:
        dataset_key = canonical_dataset_name(dataset)
        normalized = {
            canonical_dataset_name(name): int(limit)
            for name, limit in self.max_total_tokens_by_dataset.items()
        }
        return dataset_key, int(normalized.get(dataset_key, self.max_total_tokens))


def canvas_config_record(canvas: CanvasConfig) -> dict:
    """``asdict(canvas)`` for manifests and execution semantics."""

    return {key: value for key, value in asdict(canvas).items() if key not in CANVAS_SCHEDULING_FIELDS}
