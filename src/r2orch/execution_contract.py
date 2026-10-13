"""Bind prompt, state-machine and skill semantics across collection and learning."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


R2ORCH_CONTRACT_VERSION = "r2orch"
R2ORCH_COUNTERFACTUAL_EXECUTION = "closure_replay"
# Recorded in cycle_dir/source_hashes.json only; the semantics records carry no source hashes.
R2ORCH_SOURCE_MODULES = (
    "training_cycle",
    "proposer_reward",
    "proposer",
    "proposer_pair_training",
    "director_update",
    "skill_ledger",
    "run_config",
    "benchmark_scoring",
    "graph_learning",
    "training_selection",
    "curriculum",
    "alfworld_bridge",
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def execution_semantics(
    prompt_variant: str,
    *,
    admission_config: dict | None = None,
    r2orch: dict,
) -> dict:
    from . import runtime
    from .director import director_prompt_components
    from .output_contract import OUTPUT_CONTRACT_VERSION, WORKER_OUTPUT_ROLE_VERSION
    from .pats_semantics import SEMANTIC_REVISION, contract_hash

    base, hints = director_prompt_components(prompt_variant)
    from .config import canonical_dataset_name
    scoped_protocols = (admission_config or {}).get("canvas", {}).get("submission_protocol_by_dataset", {})
    variants_by_dataset = {canonical_dataset_name(dataset): "r2orch" for dataset in scoped_protocols}
    scoped_prompt_hashes = {}
    for dataset, variant in variants_by_dataset.items():
        scoped_base, scoped_hints = director_prompt_components(variant)
        scoped_prompt_hashes[dataset] = {
            key: _digest(scoped_base.rstrip() + "\n\n" + hint.strip() + "\n")
            for key, hint in sorted(scoped_hints.items())
        }
    rendered = {}
    render_table = [
        ("hotpotqa", "hotpotqa_context"),
        ("nq_open", "retrieval_qa"),
        ("aime", "aime"),
        ("alfworld", "alfworld"),
        ("", ""),
        ("hotpotqa", "retrieval_qa"),
        ("medqa", "retrieval_qa"),
        ("mbppplus", "mbppplus"),
    ]
    for dataset, adapter in render_table:
        for selected in (False, True):
            args = dict(
                dataset=dataset,
                short_answer_qa=dataset in {"hotpotqa", "nq_open"},
                is_output_agent=selected,
            )
            instruction = runtime._worker_output_instruction([], action_adapter=adapter, **args)
            recovery = runtime._finalization_recovery_messages(
                instruction=instruction,
                react_trace=[],
                previous_attempt_issue="invalid_json",
                visible_context={
                    "public_task_context": "PUBLIC TASK",
                    "assigned_task": "DELEGATION",
                    "action_environment": {"adapter": adapter},
                },
                **args,
            )
            from .contracts import AgentNode
            from .unified_contract import PROTOCOL, result_instruction

            node = AgentNode(agent_id="manifest", metadata={
                "submission_protocol": PROTOCOL,
                "result_scope": "task_result" if selected else "subtask",
            })
            responsibility = result_instruction(node, dataset)
            key = f"{dataset}:{selected}"
            if key in rendered:
                key = f"{dataset}:{adapter}:{selected}"
            rendered[key] = _digest([instruction, responsibility, recovery])
    semantics = {
        "director_action_protocol_version": "director_action_json",
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "submission_contract_version": "unified_submission",
        "submission_admission_config": json.loads(json.dumps(admission_config or {})),
        "worker_output_role_version": WORKER_OUTPUT_ROLE_VERSION,
        "director_prompt_variant": prompt_variant,
        "director_prompt_variant_by_dataset": variants_by_dataset,
        "director_prompt_template_sha256_by_dataset": scoped_prompt_hashes,
        "director_action_protocol_version_by_dataset": {
            dataset: "director_action_json" for dataset in variants_by_dataset
        },
        "submission_contract_version_by_dataset": {
            dataset: "unified_submission" for dataset in variants_by_dataset
        },
        "director_prompt_template_sha256_by_problem_type": {
            key: _digest(base.rstrip() + "\n\n" + hint.strip() + "\n")
            for key, hint in sorted(hints.items())
        },
        "worker_and_recovery_sha256": rendered,
        "pats_semantic_revision": SEMANTIC_REVISION,
        "pats_runtime_contract_sha256": contract_hash(prompt_variant),
        "pats_runtime_contract_sha256_by_dataset": {
            dataset: contract_hash(variant) for dataset, variant in variants_by_dataset.items()
        },
    }
    # Semantics are versioned by contract version, config and prompt hashes only;
    # source hashes are recorded per cycle (source_hashes.json), never bound.
    semantics.update(
        r2orch_contract_version=R2ORCH_CONTRACT_VERSION,
        r2orch_config=json.loads(json.dumps(r2orch)),
        r2orch_config_sha256=canonical_json_sha256(r2orch),
        counterfactual_execution=R2ORCH_COUNTERFACTUAL_EXECUTION,
    )
    return semantics


def contract_source_hashes() -> dict[str, str]:
    """Conservative code provenance covering tool-dependent prompts and templates."""
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest()
        for name in (*_SOURCE_MODULES, *R2ORCH_SOURCE_MODULES)
    }


_SOURCE_MODULES = (
    "execution_contract",
    "actions",
    "output_contract",
    "hotpot_answer_contract",
    "runtime",
    "llm",
    "artifact_protocol",
    "endpoint_pool",
    "director",
    "canvas",
    "graph",
    "delegation",
    "adaptive",
    "answer_submission",
    "submission_contract",
    "unified_contract",
    "unified_submission",
    "alfworld",
    "outcome_admission",
    "selfplay_runtime",
    "protocol_reward",
    "application",
    "observability",
    "qa_metrics",
    "qa_submission",
    "aime_submission",
    "outcome_metrics",
    "rollouts",
    "training",
    "selfplay",
    "async_cycle",
    "config",
    "pats_semantics",
    "pats_refiner",
    "pats",
    "skill_store",
)


def manifest_semantics(manifest: dict) -> dict | None:
    return manifest.get("execution_semantics")


_TAIL_CONTROL_SIBLING_KEYS = (
    "rollout_wall_time_s",
    "aime_rollout_wall_time_s",
    "alfworld_rollout_wall_time_s",
)
_LEARNER_DIRECTOR_KEYS = (
    "low_signal_abs_advantage",
    "low_signal_keep_fraction",
)


def _comparison_semantics(value):
    """Return the resume-comparable view of an execution-semantics payload.

    Collection-tail time boxes and learner-side loss weights never reach a rollout, so a
    resumed run may tune them.  Keep the on-disk manifest byte-for-byte intact, but remove
    those fields from both sides of an equality check and recompute the config digest over
    that normalized view.
    """

    if not isinstance(value, dict):
        return value
    normalized = deepcopy(value)
    r2orch_config = normalized.get("r2orch_config")
    if not isinstance(r2orch_config, dict):
        return normalized
    # ``r2orch_config`` is the two-table payload returned by
    # ``r2orch_semantics_payload`` (``r2orch`` + ``pats``).
    target = r2orch_config["r2orch"]
    siblings = target.get("siblings")
    if isinstance(siblings, dict):
        siblings = dict(siblings)
        # Time boxes and failure tolerance bound the collection tail (which slots are given up);
        # they change neither the action protocol nor the Director's context, so tuning them on a
        # resumed run is allowed.
        for key in _TAIL_CONTROL_SIBLING_KEYS:
            siblings.pop(key, None)
        target["siblings"] = siblings
    director = target.get("director")
    if isinstance(director, dict):
        # Learner-side loss weights never reach a rollout.
        director = dict(director)
        for key in _LEARNER_DIRECTOR_KEYS:
            director.pop(key, None)
        target["director"] = director
    normalized["r2orch_config_sha256"] = canonical_json_sha256(r2orch_config)
    return normalized


def require_same_semantics(expected, actual) -> None:
    # Transport/policy-lag overrides must never waive this equality.
    if _comparison_semantics(expected) != _comparison_semantics(actual):
        raise ValueError("execution semantics changed; start a fresh collection")


def bind_rollout_contract(batches, rollouts):
    from dataclasses import replace

    values = [manifest_semantics(r.trajectory.metadata.get("model_roles", {})) for r in rollouts]
    versions = {r.trajectory.metadata.get("submission_contract_version") for r in rollouts}
    scoped = bool(values and values[0] and values[0].get("submission_contract_version_by_dataset"))
    if not scoped and len(versions) > 1:
        raise ValueError("submission contract changed within rollout group")
    for rollout, value in zip(rollouts, values):
        require_same_semantics(values[0], value)
        _validate_dataset_submission_contract(rollout.trajectory.metadata, value)
    return tuple(
        replace(batch, metadata={**batch.metadata, "execution_semantics": values[0]})
        for batch in batches
    )


def _validate_dataset_submission_contract(metadata: dict, semantics: dict | None) -> None:
    """A mixed batch may use different protocols only as declared by its manifest."""
    from .config import canonical_dataset_name

    scoped = (semantics or {}).get("submission_contract_version_by_dataset", {})
    if not scoped:
        return
    dataset = canonical_dataset_name(metadata.get("dataset", ""))
    if not dataset:
        raise ValueError("dataset required for mixed submission contracts")
    expected = scoped.get(dataset, semantics["submission_contract_version"])
    if metadata.get("submission_contract_version") != expected:
        raise ValueError(f"submission contract does not match dataset: {dataset}")


def validate_training_contract(proposer_batch, solver_batch, *, expected=None):
    recorded = solver_batch.metadata.get("execution_semantics")
    require_same_semantics(recorded, proposer_batch.metadata.get("execution_semantics"))
    if expected is not None:
        require_same_semantics(expected, recorded)
    if recorded is not None:
        require_same_semantics(recorded, execution_semantics(
            recorded["director_prompt_variant"],
            admission_config=recorded.get("submission_admission_config"),
            r2orch=recorded["r2orch_config"],
        ))
    for sample in solver_batch.samples:
        from .submission_contract import validate_primary_training_outcome

        validate_primary_training_outcome(sample.metadata)
        observed = manifest_semantics(sample.metadata.get("model_roles", {}))
        if recorded is not None or observed is not None:
            require_same_semantics(recorded, observed)
        _validate_dataset_submission_contract(sample.metadata, recorded)
