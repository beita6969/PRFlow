"""R2Orch configuration (`[r2orch]` in the adaptive TOML).

Only the orchestration layer reads this object; pure functions receive explicit values.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name

R2ORCH_DATASETS = ("aime", "hotpotqa", "nq_open", "medqa", "mbppplus", "alfworld")
JOB_ORDERS = ("fifo", "longest_first")
CUDAGRAPH_MODES = ("eager", "decode", "breakable", "default")
# Deployment-only sub-tables: changing an allocation, GPU layout or path must not
# change the execution semantics of a run.
DEPLOYMENT_TABLES = ("paths", "serving", "launch")


@dataclass(frozen=True)
class Dataset:
    task_type: str
    verifier: str
    adapter: str
    stateful: bool = False


DEFAULT_DATASETS: dict[str, Dataset] = {
    "aime": Dataset("math", "integer_answer", "aime"),
    "hotpotqa": Dataset("multi_hop_qa", "qa_exact_match", "retrieval_qa"),
    "nq_open": Dataset("factual_qa", "qa_exact_match", "retrieval_qa"),
    "medqa": Dataset("medical_mcq", "letter_choice", "retrieval_qa"),
    "mbppplus": Dataset("code_function", "mbppplus_tests", "mbppplus"),
    "alfworld": Dataset("alfworld", "alfworld_environment", "alfworld", stateful=True),
}


@dataclass(frozen=True)
class SiblingsConfig:
    independent_siblings: int = 4
    # Keep the local vLLM queue bounded.  Fifty-six concurrent rollouts can
    # starve a hard AIME request long enough to hit its absolute deadline;
    # twenty-four still overlaps independent siblings while leaving headroom
    # for multi-turn tool traffic.
    rollout_workers: int = 24
    rollout_wall_time_s: float = 900.0
    # --- tail control.  A cycle is gated by its slowest AIME/ALFWorld rollout (decode latency
    # x tokens), not by vLLM queueing.
    # Scheduling only (never changes a rollout, popped from the execution-semantics payload):
    #   job_order = "fifo"          sibling jobs run in proposal order
    #   job_order = "longest_first" datasets in ``longest_first_datasets`` are submitted first so
    #                               the known long tails start at t=0 instead of behind a queue
    job_order: str = "fifo"
    longest_first_datasets: tuple[str, ...] = ("aime", "alfworld")
    # Semantics-affecting (recorded in the execution-semantics payload):
    #   aime_rollout_wall_time_s / alfworld_rollout_wall_time_s: 0 keeps the 2400 s floors;
    #       a positive value is used as given.
    # A wall-clock timeout is recorded as a failed slot; the rollout is not re-run.
    aime_rollout_wall_time_s: float = 0.0
    alfworld_rollout_wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.job_order not in JOB_ORDERS:
            raise ValueError(f"siblings.job_order must be one of {JOB_ORDERS}, got {self.job_order!r}")
        if float(self.aime_rollout_wall_time_s) < 0 or float(self.alfworld_rollout_wall_time_s) < 0:
            raise ValueError("siblings.*_rollout_wall_time_s must be >= 0 (0 keeps the 2400 s floor)")


# Reference solve of the cost normalisation: the Worker tokens (input, output) of ONE non-thinking
# solve of a task of the dataset with a 1024-token answer (a typical single-node rollout).  These
# are estimates from what each Worker prompt holds (instruction ~600 tokens, plus):
#   aime      ~150-token problem + the Director's delegation                       -> 900
#   hotpotqa  question + the 10 distractor paragraphs it keeps in the prompt + up to 3 search rounds -> 2500
#   nq_open   question + up to 3 rounds of 3 search hits of 600 characters          -> 2000
#   medqa     question + 4-5 options + search rounds                                -> 1500
#   mbppplus  problem + the tests shown to the Worker                               -> 1000
#   alfworld  an episode (~25 Worker calls of ~2.4k growing prompt, ~150 output tokens each), not
#             one request                                                           -> 60000 in, 4000 out
# Re-fit them from the ``worker_cost_record`` of a run (median token_in / token_out of single-node
# rollouts per dataset); they only rescale the cost, never the accuracy.
DEFAULT_COST_REFERENCE_TOKENS: dict[str, tuple[int, int]] = {
    "aime": (900, 1024),
    "hotpotqa": (2500, 1024),
    "nq_open": (2000, 1024),
    "medqa": (1500, 1024),
    "mbppplus": (1000, 1024),
    "alfworld": (60000, 4000),
}


@dataclass(frozen=True)
class CostConfig:
    """Worker cost of a rollout, and its weight in the Director's outcome reward.

    ``worker_cost = sum over Worker requests (token_in * c_in + token_out * c_out) / reference``,
    ``reference = reference_in * c_in + reference_out * c_out`` of the task's dataset (one
    non-thinking 1024-token solve, see ``DEFAULT_COST_REFERENCE_TOKENS``).  The cost is recorded
    for every rollout whatever ``lambda`` is.

    ``lambda`` (TOML key; ``lambda_`` here) is the weight in the Director reward:
    ``reward - lambda * worker_cost`` feeds the group-relative advantage of the Director only.
    The default 0 leaves every reward, advantage and manifest as it was.  Only the Director uses
    the cost-adjusted reward: the Proposer's reward F~, the scaffold success rates and the
    curriculum keep the raw correctness outcome (``proposer_reward``).

    ``c_in`` / ``c_out`` are relative prices per token: 0.25 / 1.0 makes an output token four
    input tokens, the usual API price ratio and about the prefill / decode throughput ratio of a
    prefix-cached 9B service.
    """

    lambda_: float = 0.0
    c_in: float = 0.25
    c_out: float = 1.0
    reference_tokens: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COST_REFERENCE_TOKENS)
    )
    # Datasets absent from ``reference_tokens``.
    default_reference_tokens: tuple[int, int] = (1500, 1024)

    def __post_init__(self) -> None:
        if float(self.lambda_) < 0.0:
            raise ValueError("cost.lambda must be >= 0")
        if float(self.c_in) < 0.0 or float(self.c_out) <= 0.0:
            raise ValueError("cost.c_in must be >= 0 and cost.c_out > 0")
        normalized = {
            canonical_dataset_name(name): self._pair(name, pair)
            for name, pair in dict(self.reference_tokens).items()
        }
        object.__setattr__(self, "reference_tokens", normalized)
        object.__setattr__(
            self, "default_reference_tokens", self._pair("default", self.default_reference_tokens)
        )

    @staticmethod
    def _pair(name: str, pair: Any) -> tuple[int, int]:
        values = tuple(int(item) for item in pair)
        if len(values) != 2 or values[0] < 0 or values[1] <= 0:
            raise ValueError(f"cost.reference_tokens.{name} must be [input >= 0, output > 0]")
        return values

    def reference_for(self, dataset: str) -> tuple[int, int]:
        return self.reference_tokens.get(canonical_dataset_name(dataset), self.default_reference_tokens)

    def reference_cost(self, dataset: str) -> float:
        token_in, token_out = self.reference_for(dataset)
        return token_in * float(self.c_in) + token_out * float(self.c_out)

    def cost(self, token_in: int, token_out: int, dataset: str) -> float:
        """Normalised cost of ``token_in`` / ``token_out`` Worker tokens on ``dataset``."""

        return (
            int(token_in) * float(self.c_in) + int(token_out) * float(self.c_out)
        ) / self.reference_cost(dataset)


@dataclass(frozen=True)
class ArmsConfig:
    """Arm-coupled scaffold evaluation and retirement (``[r2orch.arms]``).

    Every task runs ``hidden + visible`` independent siblings
    (``siblings.independent_siblings`` must equal the sum).  One card k among the cards retrieval
    shows for the task is chosen by Thompson sampling from the skill ledger (``skill_ledger.py``);
    the hidden siblings see the retrieved context without k, the visible siblings all of it.  Each
    task with both arms completed adds ``z = ybar_v - ybar_h`` (raw correctness) to card k; at the
    end of a cycle the PATS lifecycle retires cards whose effect E[Delta^+] fell to ``epsilon``,
    admits trial cards whose ``admit_quantile`` quantile is above 0 and retires trial cards still
    not admitted after ``3 * min_tasks`` tasks.  The Proposer reward is the arms Bell reward
    (``proposer_reward.arms_bell_reward``).

    Delta_k is the effect of k on the current Director policy, so a card the policy had
    internalised (retired) can matter again after the policy drifts:

    * ``window_cycles`` W: every posterior used for a decision or a Thompson draw reads only the
      records of the last W cycles of evidence.
    * ``audit_share`` a: a task whose scope holds ledger-retired cards shows, with probability
      a, one of them (uniform) to its visible siblings instead of the Thompson pick among the
      active cards; the hidden siblings keep the retrieved context.  The card is rendered from its
      archived record (retrieval no longer shows it).
    * ``reinstate_quantile`` q: at each lifecycle step a retired card whose (windowed) posterior
      has n >= min_tasks, E[Delta^+] > epsilon and Q_q(Delta) > 0 is reinstated into the scopes
      it was retired from.
    """

    hidden: int = 3  # siblings that do NOT see the candidate card k
    visible: int = 2  # siblings that see the full retrieved context including k
    epsilon: float = 0.02  # retire threshold on E[Delta_k^+ | D]
    admit_quantile: float = 0.05  # admit when this posterior quantile of Delta_k is > 0
    min_tasks: int = 12  # tasks with both arms completed before any decision on a card
    bootstrap_draws: int = 2000
    gamma: float = 0.5  # weight of the scaffold-assisted frontier term of the Proposer reward
    endpoint_penalty: float = 0.2  # lambda_P: reward of a group whose outcomes are all equal
    shrink_strength: float = 2.0  # Beta shrinkage of an arm's success toward the scope's hidden mean
    # EMA decay (per cycle) of the scope's hidden-arm success (the shrinkage target) and of the
    # dataset's mean task duration (the compute normaliser below).
    scope_prior_decay: float = 0.9
    # Standard deviation of the wide prior of a card's effect in the Thompson draw.
    thompson_prior_sd: float = 0.25
    # Learning progress per unit of compute (Graves et al. 2017): the positive (Bell) part of the
    # arms_bell reward is multiplied by (Tbar_d / T_q) ** time_exponent, T_q the mean duration_s
    # of the task's completed siblings, Tbar_d the dataset's EMA of it.  0 disables.
    time_exponent: float = 0.5
    window_cycles: int = 30  # posteriors read only the last window_cycles cycles of records
    audit_share: float = 0.3  # share of tasks (in scopes with retired cards) that audit one
    reinstate_quantile: float = 0.05  # reinstate a retired card when this quantile of Delta > 0

    def __post_init__(self) -> None:
        if int(self.hidden) < 1 or int(self.visible) < 1:
            raise ValueError("arms.hidden and arms.visible must be >= 1")
        if float(self.epsilon) < 0.0:
            raise ValueError("arms.epsilon must be >= 0")
        if not 0.0 < float(self.admit_quantile) < 1.0:
            raise ValueError("arms.admit_quantile must be in (0, 1)")
        if int(self.min_tasks) < 1:
            raise ValueError("arms.min_tasks must be >= 1")
        if int(self.bootstrap_draws) < 100:
            raise ValueError("arms.bootstrap_draws must be >= 100")
        if float(self.gamma) < 0.0 or float(self.endpoint_penalty) < 0.0:
            raise ValueError("arms.gamma and arms.endpoint_penalty must be >= 0")
        if float(self.shrink_strength) < 0.0:
            raise ValueError("arms.shrink_strength must be >= 0")
        if not 0.0 <= float(self.scope_prior_decay) < 1.0:
            raise ValueError("arms.scope_prior_decay must be in [0, 1)")
        if float(self.thompson_prior_sd) <= 0.0:
            raise ValueError("arms.thompson_prior_sd must be > 0")
        if float(self.time_exponent) < 0.0:
            raise ValueError("arms.time_exponent must be >= 0 (0 disables the compute factor)")
        if isinstance(self.window_cycles, bool) or int(self.window_cycles) != self.window_cycles:
            raise ValueError("arms.window_cycles must be an integer")
        if int(self.window_cycles) < 1:
            raise ValueError("arms.window_cycles must be >= 1")
        if not 0.0 < float(self.audit_share) <= 1.0:
            raise ValueError("arms.audit_share must be in (0, 1]")
        if not 0.0 < float(self.reinstate_quantile) < 1.0:
            raise ValueError("arms.reinstate_quantile must be in (0, 1)")


@dataclass(frozen=True)
class ProposerConfig:
    candidates_per_list: int = 8
    ads_candidates: int = 4
    tsds_candidates: int = 4
    # Number of anchor lists sampled per dataset in each proposer cycle.
    lists_per_dataset: int = 1
    label_alphabet: str = "ABCDEFGH"
    temperature: float = 1.0
    # Second pick of a list.  "independent": two independent draws (40-45 % of
    # lists end up on the same candidate and carry no gradient).  "distinct" redraws a second pick
    # that lands on the first candidate from pi(.)/(1 - pi(a)) over the other candidates (sampling
    # without replacement); proposition 4 makes the minimiser independent of the pair-sampling
    # distribution, so the objective is unchanged.  Only changes which tasks run; hence part of
    # the execution semantics when not "independent".
    second_pick: str = "independent"
    # ---- learner-side knobs (never part of the execution semantics) ----
    # Optimizer steps per Proposer cycle; 0 = shared ``director.ppo_mini_batches``.  One step on
    # the whole pair set averages every pair instead of splitting 2-6 pairs over two steps; it is
    # also the knob that sets the cosine schedule length (see runtime).
    optimizer_steps_per_cycle: int = 0
    # Gradient-norm cap of the Proposer update; 0 = shared trainer ``max_grad_norm`` (1.0).  The
    # recorded pre-clip norms are 10-180, so 1.0 turns every step into a unit-norm step.
    max_grad_norm: float = 0.0
    # Proposer loss: plain policy gradient on the reward of the two picks (advantage = half the
    # reward difference F~a - F~b, scaled to unit RMS over the batch) plus ``kl_beta`` times the KL
    # of the label distribution to the reference.
    kl_beta: float = 0.05
    # Trust region on the pair's tilt relative to the reference (0 = off); see
    # proposer_pair_training.reinforce_pair_loss.
    gap_clip: float = 0.0

    def __post_init__(self) -> None:
        if float(self.kl_beta) < 0.0:
            raise ValueError("proposer.kl_beta must be >= 0")
        if int(self.lists_per_dataset) < 1:
            raise ValueError("proposer.lists_per_dataset must be >= 1")
        if self.second_pick not in ("independent", "distinct"):
            raise ValueError("proposer.second_pick must be 'independent' or 'distinct'")
        if int(self.optimizer_steps_per_cycle) < 0:
            raise ValueError("proposer.optimizer_steps_per_cycle must be >= 0")
        if float(self.max_grad_norm) < 0.0:
            raise ValueError("proposer.max_grad_norm must be >= 0")


@dataclass(frozen=True)
class DirectorConfig:
    tis_cap: float = 2.0
    kl: float = 0.005
    whiten_eps: float = 1e-8
    ppo_mini_batches: int = 2
    clip_low: float = 0.2
    clip_high: float = 0.28
    # Learner-side speed knob (not part of the semantics): a sample whose group-centered outcome advantage
    # (|reward - group mean|, BEFORE whitening) is below ``low_signal_abs_advantage``
    # carries no policy gradient.  They are dropped from the Director batch before the whitening, except
    # for a deterministic ``low_signal_keep_fraction`` (KL anchor).  0 = keep everything.
    low_signal_abs_advantage: float = 0.0
    low_signal_keep_fraction: float = 0.15


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 64
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class ServingConfig:
    layout: str = "4gpu"
    vllm_port: int = 18600
    vllm_gpu_ids: tuple[int, ...] = (0, 1)
    vllm_data_parallel_size: int = 2
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 32768
    base_served_name: str = "r2orch-base"
    proposer_lora_name: str = "r2orch-proposer"
    director_lora_name: str = "r2orch-director"
    max_loras: int = 4
    generation_config: str = "vllm"
    retrieval_port: int = 18610
    retrieval_gpu_id: int = 3
    retrieval_device: str = "cuda:3"
    director_train_gpu_id: int = 2
    proposer_train_gpu_id: int = 3
    # --- decode-latency knobs (deployment only).
    # "default" (vLLM's own CUDA-graph config) is measured 2.7-3.3x faster than "eager"
    # (--enforce-eager).  Modes (vLLM 0.27.1):
    #   "decode"     FULL decode graphs, no torch.compile (-cc mode NONE, cudagraph FULL_DECODE_ONLY)
    #   "breakable"  FULL_AND_PIECEWISE through VLLM_USE_BREAKABLE_CUDAGRAPH=1 (no torch.compile)
    #   "default"    vLLM's default torch.compile + FULL_AND_PIECEWISE (validated on A100)
    vllm_cudagraph_mode: str = "default"
    vllm_prefix_caching: bool = True
    vllm_max_num_seqs: int = 0  # 0 = vLLM default
    vllm_max_num_batched_tokens: int = 0  # 0 = vLLM default
    vllm_extra_args: tuple[str, ...] = ()
    # Optional second vLLM that serves only the frozen Worker base (no LoRA).  worker_port 0
    # keeps the single resident service; with a port the Worker/reviewer route moves to it.
    worker_port: int = 0
    worker_gpu_ids: tuple[int, ...] = ()
    worker_gpu_memory_utilization: float = 0.90
    worker_cudagraph_mode: str = "default"
    worker_prefix_caching: bool = True
    worker_max_num_seqs: int = 0
    worker_max_num_batched_tokens: int = 0
    worker_extra_args: tuple[str, ...] = ()
    # Cross-node serving: a non-empty host that is not this node means the service runs on another node
    # (started there); the driver only waits for it and hot-loads adapters.
    vllm_host: str = ""
    worker_host: str = ""
    remote_wait_s: float = 3600.0

    def __post_init__(self) -> None:
        for name in ("vllm_cudagraph_mode", "worker_cudagraph_mode"):
            if getattr(self, name) not in CUDAGRAPH_MODES:
                raise ValueError(f"serving.{name} must be one of {CUDAGRAPH_MODES}, got {getattr(self, name)!r}")
        if int(self.worker_port) < 0:
            raise ValueError("serving.worker_port must be >= 0")
        if int(self.worker_port) and not self.worker_gpu_ids:
            raise ValueError("serving.worker_port requires serving.worker_gpu_ids")
        if int(self.worker_port) == int(self.vllm_port) and int(self.worker_port):
            raise ValueError("serving.worker_port must differ from serving.vllm_port")


@dataclass(frozen=True)
class LaunchConfig:
    stop_env: str = "R2ORCH_STOP_AT_EPOCH"
    stop_file: str = "STOP"
    collect_stop_margin_s: float = 120.0
    train_stop_margin_s: float = 480.0
    resume_exit_code: int = 75


# Defaults for a config that leaves [r2orch.paths] unset; configs/r2orch.toml sets every path.
_DATA = "/data/r2orch"  # data, models and run outputs
_CODE = "/opt/R2Orch"  # this repository


@dataclass(frozen=True)
class PathsConfig:
    run_root: str = "@RUN_ROOT@"
    runs_dir: str = f"{_DATA}/runs"
    code_root: str = _CODE
    venv: str = f"{_CODE}/.venv"
    base_model: str = f"{_DATA}/models/Qwen3.5-9B"
    e5_model: str = f"{_DATA}/models/e5-base-v2"
    wiki18_dir: str = f"{_DATA}/wiki18"
    train_pool: str = f"{_DATA}/pools/train_pool.jsonl"
    alfworld_data_root: str = f"{_DATA}/alfworld"
    alfworld_python: str = f"{_DATA}/envs/alfworld/bin/python"
    evalplus_python: str = f"{_DATA}/envs/evalplus/bin/python"
    # The wiki-18 retrieval service shipped with this repository.
    retrieval_service_script: str = f"{_CODE}/scripts/r2orch/retrieval_service.py"


_SUBTABLES = {
    "siblings": SiblingsConfig,
    "cost": CostConfig,
    "arms": ArmsConfig,
    "proposer": ProposerConfig,
    "director": DirectorConfig,
    "lora": LoraConfig,
    "serving": ServingConfig,
    "launch": LaunchConfig,
    "paths": PathsConfig,
}


@dataclass(frozen=True)
class R2OrchConfig:
    base_seed: int = 0
    datasets_enabled: tuple[str, ...] = R2ORCH_DATASETS
    datasets: dict[str, Dataset] = field(default_factory=lambda: dict(DEFAULT_DATASETS))
    siblings: SiblingsConfig = field(default_factory=SiblingsConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    arms: ArmsConfig = field(default_factory=ArmsConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self) -> None:
        arms = self.arms
        siblings = int(self.siblings.independent_siblings)
        if siblings != int(arms.hidden) + int(arms.visible):
            raise ValueError(
                "siblings.independent_siblings must equal arms.hidden + arms.visible "
                f"({siblings} != {int(arms.hidden)} + {int(arms.visible)})"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> R2OrchConfig:
        values = dict(payload)
        raw_datasets = values.pop("datasets", {})
        enabled = tuple(
            canonical_dataset_name(name)
            for name in values.pop("datasets_enabled", R2ORCH_DATASETS)
        )
        datasets = dict(DEFAULT_DATASETS)
        for name, table in raw_datasets.items():
            key = canonical_dataset_name(name)
            datasets[key] = (
                replace(DEFAULT_DATASETS[key], **table) if key in DEFAULT_DATASETS
                else Dataset(**table)
            )
        missing = [name for name in enabled if name not in datasets]
        if missing:
            raise ValueError("r2orch.datasets lacks enabled datasets: " + ", ".join(missing))
        tables = {}
        for name, kind in _SUBTABLES.items():
            table = dict(values.pop(name, {}))
            if name == "cost" and "lambda" in table:  # a Python keyword cannot be a field name
                table["lambda_"] = table.pop("lambda")
            for item in fields(kind):
                if item.name in table and isinstance(table[item.name], list):
                    table[item.name] = tuple(table[item.name])
            tables[name] = kind(**table)
        return cls(
            datasets_enabled=enabled,
            datasets={name: datasets[name] for name in enabled},
            **tables,
            **values,
        )

    def dataset(self, name: str) -> Dataset:
        return self.datasets[canonical_dataset_name(name)]

    @property
    def scopes(self) -> tuple[str, ...]:
        from .pats import resolve_scope

        return tuple(
            resolve_scope(self.datasets[name].task_type, {"dataset": name})
            for name in self.datasets_enabled
        )


_PROPOSER_LEARNER_KEYS = ("optimizer_steps_per_cycle", "max_grad_norm", "kl_beta", "gap_clip")


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _canonical_seed_skills_path(value: Any) -> str:
    """Keep the semantic manifest stable across rendered absolute paths.

    ``render_config.py`` expands repository-relative paths for remote jobs.  The
    seed-card identity is the repository-relative path, so retain that portion
    when it is present and avoid making resumability depend on the allocation's
    mount point.
    """

    raw = str(value or "")
    parts = Path(raw).parts
    if "configs" in parts:
        return Path(*parts[parts.index("configs"):]).as_posix()
    return raw


def r2orch_semantics_payload(config: Any) -> dict[str, Any]:
    """Execution-semantics record for `execution_semantics(..., r2orch=...)`.

    `config` is the AdaptiveApplicationConfig. Deployment sub-tables are left out
    so a run resumes after a change of allocation, layout or path.
    """

    r2orch = asdict(config.r2orch)
    for name in DEPLOYMENT_TABLES:
        r2orch.pop(name)
    # Learner-side Proposer knobs never reach a rollout, so they stay out of the payload; the
    # one collection-side option (second_pick) is recorded.
    director = dict(r2orch.get("director", {}))
    for name in ("low_signal_abs_advantage", "low_signal_keep_fraction"):
        director.pop(name, None)
    r2orch["director"] = director
    proposer = dict(r2orch.get("proposer", {}))
    for name in _PROPOSER_LEARNER_KEYS:
        proposer.pop(name, None)
    r2orch["proposer"] = proposer
    # Worker-pool widths affect scheduling/throughput only.  Treat them like
    # deployment knobs so a resumed run can reduce queue pressure without
    # invalidating the algorithmic execution contract or its artifacts.
    siblings = dict(r2orch.get("siblings", {}))
    siblings.pop("rollout_workers", None)
    # Pure scheduling knobs never change what a rollout is.
    siblings.pop("job_order", None)
    siblings.pop("longest_first_datasets", None)
    r2orch["siblings"] = siblings
    # The cost term changes the Director's reward only when lambda != 0.  At lambda = 0 the table
    # (weights, reference tokens) only scales a recorded metric, so it stays out of the payload;
    # with lambda != 0 the whole table is semantics.
    cost = r2orch.pop("cost", None)
    if cost is not None and float(cost.get("lambda_", 0.0)) != 0.0:
        cost = dict(cost)
        cost["lambda"] = cost.pop("lambda_")
        r2orch["cost"] = cost
    pats = asdict(config.pats)
    pats["seed_skills_path"] = _canonical_seed_skills_path(pats.get("seed_skills_path"))
    return _json_value({"r2orch": r2orch, "pats": pats})
