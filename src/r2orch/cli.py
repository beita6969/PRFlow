from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .ads_preprocessing import ADSPreprocessingConfig, prepare_ads_pool
from .application import load_adaptive_config
from .curriculum import ADSBoundaryScheduler, CurriculumProfile, FixedTaskPool, TSDSRetriever
from .services import VLLMServiceManager
from .training import PolicyTrainingConfig, training_device_for_gpu

VERIFIERS = (
    "none",
    "auto",
    "exact_match",
    "multi_answer_exact_match",
    "flowsteer_qa",
    "token_f1",
    "numeric",
    "multiple_choice",
)

DEFAULT_CURRICULUM_PROFILE = (
    Path(__file__).resolve().parents[2] / "configs" / "curriculum" / "r2orch.toml"
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2Orch runtime")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config", help="parse and validate a TOML config")
    validate.add_argument("path", type=Path)

    services = commands.add_parser("model-services", help="manage owned vLLM model services")
    services.add_argument("action", choices=("status", "start", "stop"))
    services.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "r2orch.toml",
    )
    services.add_argument("--state-dir", type=Path, default=Path("state/services"))
    services.add_argument("--wait-s", type=float, default=120.0)

    ads = commands.add_parser(
        "prepare-ads-pool",
        help="extract base-policy embeddings/NLL and add ADS K-Means metadata",
    )
    ads.add_argument("--input", type=Path, action="append", required=True)
    ads.add_argument("--output", type=Path, required=True)
    ads.add_argument("--artifacts-dir", type=Path, required=True)
    ads.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    ads.add_argument("--num-clusters", type=int, required=True)
    ads.add_argument("--device", default="cuda:0")
    ads.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ads.add_argument("--batch-size", type=int, default=1)
    ads.add_argument("--max-length", type=int, default=4096)
    ads.add_argument("--pca-dim", type=int, default=128)
    ads.add_argument("--seed", type=int, default=42)
    ads.add_argument("--overwrite", action="store_true")

    experiment = commands.add_parser(
        "selfplay-experiment",
        help="run repeated rollout -> Proposer update -> Solver update cycles",
    )
    experiment.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "r2orch.toml",
    )
    experiment.add_argument(
        "--task-pool",
        type=Path,
        action="append",
        default=[],
        help="train JSONL pool(s) used by ADS+TSDS fixed-pool selection",
    )
    experiment.add_argument(
        "--curriculum-profile",
        type=Path,
        default=DEFAULT_CURRICULUM_PROFILE,
        help="curriculum profile (ADS clusters, TSDS candidates)",
    )
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--cycles", type=int, default=1)
    experiment.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip completed cycles and resume training from complete rollout batches; "
            "incomplete rollout collection is preserved but not recollected"
        ),
    )
    experiment.add_argument(
        "--async-next-cycle-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "collect cycle N+1 from the frozen cycle-N policy snapshots while cycle N "
            "updates; requires a distinct inference GPU and permits at most one stale update"
        ),
    )
    experiment.add_argument("--rollout-no-progress-time-s", type=float, default=120.0)
    experiment.add_argument("--request-wall-time-s", type=float, default=180.0)
    experiment.add_argument("--verifier", choices=VERIFIERS, default="auto")
    experiment.add_argument("--proposer-learning-rate", type=float, default=1e-5)
    experiment.add_argument("--solver-learning-rate", type=float, default=1e-5)
    experiment.add_argument("--epochs", type=int, default=1)
    _add_grpo_arguments(experiment)
    experiment.add_argument(
        "--device",
        help="explicit device override; default/cuda uses the role GPU from [resources]",
    )
    experiment.add_argument("--mock", action="store_true")
    experiment.add_argument("--mock-trainer", action="store_true")

    return parser


def _add_grpo_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the optimizer recipe without coupling the two policy LRs."""

    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=0.0,
        help="initial LR / peak LR during warmup; zero = linear warmup from zero",
    )
    parser.add_argument(
        "--learning-rate-schedule", choices=("cosine", "constant"), default="cosine"
    )
    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        help="shared scheduler horizon; experiment mode computes role horizons when omitted",
    )
    parser.add_argument(
        "--proposer-total-optimizer-steps",
        type=int,
        help="Proposer scheduler horizon; does not change shared batch/accumulation settings",
    )
    parser.add_argument(
        "--solver-total-optimizer-steps",
        type=int,
        help="Solver scheduler horizon; does not change shared batch/accumulation settings",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=64,
        help="logical PPO/GRPO mini-batch size (SESA recipe: 64)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="maximum per-device dynamic micro-batch size",
    )
    parser.add_argument(
        "--max-micro-batch-tokens",
        type=int,
        default=16_384,
        help="maximum padded tokens in one dynamic device micro-batch",
    )
    parser.add_argument(
        "--parallel-role-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train Proposer and Solver concurrently on distinct configured GPUs",
    )
    parser.add_argument(
        "--solver-data-parallel-gpus",
        type=int,
        nargs=2,
        default=None,
        help="two physical GPUs for one global Solver update; roles run sequentially",
    )
    parser.add_argument(
        "--raw-policy-backward-mode", choices=("micro", "call", "timeline"), default="micro"
    )
    parser.add_argument(
        "--solver-training-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Solver learner precision; leaves Proposer and serving precision unchanged",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--activation-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stage saved autograd tensors on CPU to reduce training-only GPU peak memory",
    )
    parser.add_argument(
        "--activation-cpu-offload-min-tokens",
        type=int,
        default=0,
        help=(
            "when activation CPU offload is enabled, stage only micro-batches "
            "containing a sequence at least this long; zero stages every call"
        ),
    )
    # R2Orch Director-update speed knobs.  Every default is the exact update.
    parser.add_argument(
        "--fuse-old-logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "first PPO mini-batch takes log pi_old from its own differentiated forward "
            "(exact at theta-bar); needs LoRA dropout 0, else it falls back and logs why"
        ),
    )
    parser.add_argument(
        "--kl-reference-stride",
        type=int,
        default=1,
        help=(
            "frozen-reference KL pass every k-th sample (weight x k, unbiased); "
            "1 = every sample, 0 = no KL pass"
        ),
    )
    parser.add_argument(
        "--kl-term-cap",
        type=float,
        default=0.0,
        help=(
            "upper bound of one token's sampled KL term in the Director loss "
            "(0 = unbounded; 10 = verl low_var_kl)"
        ),
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="torch float32 matmul precision for the fp32 LoRA adapter ('high' = TF32)",
    )
    parser.add_argument(
        "--lora-dropout-override",
        type=float,
        default=None,
        help="force this LoRA dropout on the live adapter (also on a resumed checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-skip-below-tokens",
        type=int,
        default=0,
        help=(
            "do not gradient-checkpoint policy calls shorter than this many tokens "
            "(saves the recompute forward; 0 = always checkpoint)"
        ),
    )
    parser.add_argument(
        "--phase-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write per-phase seconds/tokens/peak memory into step_metrics",
    )


def validate_config(path: Path) -> int:
    config = load_adaptive_config(path)
    print(
        json.dumps(
            {
                "valid": True,
                "proposer": config.proposer_model.served_model,
                "solver": config.solver_model.served_model,
                "runtime": config.runtime.served_model,
            },
            ensure_ascii=False,
        )
    )
    return 0


def model_services(args: argparse.Namespace) -> int:
    config = load_adaptive_config(args.config)
    manager = VLLMServiceManager(args.state_dir)
    return _r2orch_model_services(config, manager, args)


def _r2orch_model_services(config, manager: VLLMServiceManager, args) -> int:
    """One resident vLLM serves the base model and both LoRA adapters."""
    from .vllm_serving import (
        ROLE,
        WORKER_ROLE,
        ensure_rollout_service,
        latest_adapter,
        rollout_service_spec,
        worker_service_spec,
    )

    if args.action == "start":
        ensure_rollout_service(config, manage=True, state_dir=args.state_dir)
    elif args.action == "stop":
        manager.stop(ROLE, wait_s=args.wait_s)
        manager.stop(WORKER_ROLE, wait_s=args.wait_s)  # no-op without a dedicated Worker vLLM
    spec, _rollout_config = rollout_service_spec(
        config,
        proposer_adapter=latest_adapter(config.proposer_model.checkpoint_path)[0],
        director_adapter=latest_adapter(config.solver_model.checkpoint_path)[0],
    )
    specs = [spec]
    worker = worker_service_spec(config)
    if worker is not None:
        specs.append(worker)
    print(
        json.dumps(
            [asdict(manager.status(item)) for item in specs],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


def prepare_ads_pool_command(args: argparse.Namespace) -> int:
    manifest = prepare_ads_pool(
        args.input,
        ADSPreprocessingConfig(
            model_path=args.model_path,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
            num_clusters=args.num_clusters,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pca_dim=args.pca_dim,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _training_devices(config, override: str | None) -> tuple[str, str]:
    if override and override != "cuda":
        return override, override
    return (
        training_device_for_gpu(config.proposer_gpu_id),
        training_device_for_gpu(config.solver_gpu_id),
    )


def selfplay_experiment(args: argparse.Namespace) -> int:
    # Cycle metrics stay in the run directory; there is no external metrics sink.
    tracker = SimpleNamespace(start=lambda config: None, log_cycle=lambda record: None)
    try:
        return _selfplay_experiment(args, tracker)
    except BaseException:
        try:
            from .outcome_metrics import record_interrupted_collection

            record_interrupted_collection(args.output, 5)
        except Exception as telemetry_error:
            import warnings

            warnings.warn(
                f"Interruption telemetry failed ({type(telemetry_error).__name__}); original error preserved",
                stacklevel=2,
            )
        raise


def _selfplay_experiment(args: argparse.Namespace, tracker) -> int:
    from .training_cycle import run_r2orch_experiment

    return run_r2orch_experiment(args, tracker)


def _policy_training_config(
    base_model_path: Path,
    checkpoint_root: Path,
    learning_rate: float,
    device: str,
    args: argparse.Namespace,
    *,
    total_optimizer_steps: int,
    r2orch: Any,
) -> PolicyTrainingConfig:
    from .training import r2orch_policy_config

    effective_gradient_accumulation = math.ceil(args.mini_batch_size / args.micro_batch_size)
    policy = PolicyTrainingConfig(
        base_model_path,
        checkpoint_root,
        learning_rate=learning_rate,
        epochs=args.epochs,
        entropy_coefficient=args.entropy_coefficient,
        mini_batch_size=args.mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=effective_gradient_accumulation,
        max_micro_batch_tokens=args.max_micro_batch_tokens,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_start_factor=args.warmup_start_factor,
        learning_rate_schedule=args.learning_rate_schedule,
        total_optimizer_steps=total_optimizer_steps,
        max_sequence_length=args.max_sequence_length,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
        activation_cpu_offload=args.activation_cpu_offload,
        activation_cpu_offload_min_tokens=args.activation_cpu_offload_min_tokens,
        raw_policy_backward_mode=args.raw_policy_backward_mode,
        fuse_old_logprobs=args.fuse_old_logprobs,
        kl_reference_stride=args.kl_reference_stride,
        kl_term_cap=args.kl_term_cap,
        matmul_precision=args.matmul_precision,
        lora_dropout_override=args.lora_dropout_override,
        checkpoint_skip_below_tokens=args.checkpoint_skip_below_tokens,
        phase_timing=args.phase_timing,
    )
    # R2Orch: director.* / lora.* from config (no new CLI flags).
    return r2orch_policy_config(policy, r2orch)


class _MockTaskEmbedder:
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]:
        del query
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            values = [0.0] * 16
            for index, byte in enumerate(text.encode("utf-8")):
                values[index % len(values)] += float(byte) / 255.0
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            vectors.append(tuple(value / norm for value in values))
        return vectors


def _fixed_pool_components(
    paths: list[Path],
    *,
    config: Any,
    mock: bool,
    seed: int,
    profile: CurriculumProfile | None = None,
    require_validation_manifest: bool = True,
) -> tuple[FixedTaskPool, ADSBoundaryScheduler, TSDSRetriever]:
    embedder = _MockTaskEmbedder() if mock else None
    pool = FixedTaskPool.from_jsonl(
        paths,
        embedder=embedder,
        require_ads_metadata=not mock,
        # Mock mode may replace embeddings and model calls, but it must not
        # bypass the fixed-pool validity contract exercised by production.
        # R2Orch pools come from build_pool.py without an audit manifest.
        require_validation_manifest=require_validation_manifest,
    )
    scheduler = ADSBoundaryScheduler(
        pool,
        **(
            profile.scheduler_kwargs()
            if profile
            else {"active_clusters": 4, "mini_cluster_size": 32, "boundary_eps": 0.17,
                  "alpha": 0.3, "cooldown": 64}
        ),
        seed=seed,
    )
    if profile is not None and not mock:
        if len(scheduler.dataset_cluster_ids) != profile.expected_datasets:
            raise ValueError(
                f"curriculum profile {profile.name!r} expects "
                f"{profile.expected_datasets} datasets, loaded "
                f"{len(scheduler.dataset_cluster_ids)}"
            )
        mismatched = {
            dataset: len(cluster_ids)
            for dataset, cluster_ids in scheduler.dataset_cluster_ids.items()
            if len(cluster_ids) != profile.num_clusters_per_dataset
        }
        if mismatched:
            raise ValueError(
                f"curriculum profile {profile.name!r} expects "
                f"{profile.num_clusters_per_dataset} ADS clusters per dataset; got {mismatched}"
            )
    retriever = TSDSRetriever(pool)
    return pool, scheduler, retriever


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "validate-config": lambda: validate_config(args.path),
        "model-services": lambda: model_services(args),
        "prepare-ads-pool": lambda: prepare_ads_pool_command(args),
        "selfplay-experiment": lambda: selfplay_experiment(args),
    }
    return handlers[args.command]()
