from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .application import AdaptiveApplicationResult, AdaptiveSolverApplication
from .config import canonical_dataset_name
from .deadline import RolloutDeadline, WorkerWallClockLimitExceeded
from .graph import MultiAgentGraph
from .observability import TaskSpec, task_to_public_dict
from .output_contract import OUTPUT_CONTRACT_VERSION
from .protocol_reward import calculate_director_reward
from .qwen_compat import encode_chat_trajectory
from .rollouts import (
    DirectorTurn,
    TokenizedDirectorTrajectory,
    TokenizedPolicyCall,
    Tokenizer,
    tokenize_director_policy_calls,
)
from .selfplay import AlternatingSnapshots, ProposedTask, SolverRollout

OUTCOME_ONLY_REWARD_VERSION = "outcome_only"


def _stable_execution_seed(base_seed: int, task_id: str, *, phase: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}\0{task_id}\0{phase}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _rollout_sampling_seed(
    base_seed: int,
    task_id: str,
    rollout_index: int,
    replacement_attempt: int,
) -> int:
    """Keep primary seeds stable while giving a recovery a new policy sample."""

    if replacement_attempt == 0:
        return int(base_seed) + int(rollout_index)
    return _stable_execution_seed(
        base_seed,
        f"{task_id}\0{int(rollout_index)}\0{int(replacement_attempt)}",
        phase="policy_recovery",
    )


def _outcome_task_reward(dataset: str, verification: Any, *, prediction: str):
    from .outcome_admission import task_reward_from_verification

    return task_reward_from_verification(dataset, verification, prediction=prediction)


_JSONL_APPEND_LOCK = threading.Lock()


class WorkerBackendUnavailableError(RuntimeError):
    """A primary rollout is invalid because a required Worker backend failed."""

    def __init__(self, failure: dict[str, Any]) -> None:
        self.routes = tuple(str(value) for value in failure.get("routes", ()))
        self.failure_types = tuple(str(value) for value in failure.get("failure_types", ()))
        super().__init__(
            "required Worker backend failed; "
            f"routes={','.join(self.routes) or 'unknown'}; "
            f"failure_types={','.join(self.failure_types) or 'unknown'}"
        )


class CollectionInfrastructureIncidentError(RuntimeError):
    """A backend, tool, or environment incident invalidated the active cycle."""


@dataclass
class _PrimaryCollection:
    """Live state retained only after the primary rollout is durable."""

    application: AdaptiveSolverApplication
    result: AdaptiveApplicationResult
    rollout: SolverRollout
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.application.close()
        self.closed = True


class ByteTokenizer:
    approximate_token_count = True

    """Deterministic tokenizer for smoke tests only."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "off":
            return [256]
        if text == "on":
            return [257]
        return list(text.encode("utf-8"))


class HuggingFaceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'selfplay' extra to use a real tokenizer") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def encode_chat_trajectory(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        return encode_chat_trajectory(self.tokenizer, messages, enable_thinking=False)


def adaptive_result_to_rollout(
    result: AdaptiveApplicationResult,
    tokenizer: Tokenizer,
    *,
    rollout_index: int,
    seed: int,
    max_tokens: int = 4096,
    duration_s: float | None = None,
) -> SolverRollout:
    run = result.solver_result.director_run
    prefixes = {
        int(event.payload["director_turn_index"]): event.payload.get("graph_before", {})
        for event in result.solver_result.trace.events
        if event.kind == "canvas_step" and event.payload.get("director_turn_index") is not None
    }
    turns = [
        DirectorTurn(
            model_response=turn.model_action,
            feedback=turn.feedback,
            accepted=turn.accepted,
            graph_prefix=prefixes.get(index, {}),
            prompt_messages=tuple(turn.prompt_messages),
            trainable=turn.trainable,
            call_id=turn.call_id,
            completion_token_ids=tuple(turn.completion_token_ids),
            behavior_log_probs=tuple(turn.behavior_log_probs),
            raw_reasoning_text=turn.raw_reasoning_text,
            raw_action_text=turn.raw_action_text,
            prompt_token_ids=tuple(turn.prompt_token_ids),
            action_character_span=turn.action_character_span,
            turn_kind=turn.turn_kind,
            metadata={
                "model_id": turn.model_id,
                "route_name": turn.route_name,
                "thinking_requested": turn.thinking_requested,
                "thinking_effective": turn.thinking_effective,
                "token_provenance": turn.token_provenance,
                "trajectory_schema": turn.trajectory_schema,
                "relation_policy": dict(turn.relation_decision.get("policy", {})),
                "accepted": bool(turn.accepted),
                "action_type": (
                    "relation_choice"
                    if turn.turn_kind == "relation_choice"
                    else turn.action_diagnostics.get("action_type")
                ),
                "rejection_code": turn.rejection_code,
            },
        )
        for index, turn in enumerate(run.turns)
    ]
    policy_encoding_error = None
    try:
        policy_calls = tokenize_director_policy_calls(turns, tokenizer, max_tokens=max_tokens)
    except ValueError as exc:
        # Keep outcome evidence durable without synthesizing missing policy
        # probabilities or repairing token alignment. This record is audit-only.
        policy_calls = ()
        policy_encoding_error = str(exc)
    flattened_ids: list[int] = []
    flattened_mask: list[int] = []
    action_spans: list[tuple[int, int] | None] = []
    for turn in turns:
        completion_ids = list(
            turn.completion_token_ids
            or tokenizer.encode(
                turn.raw_reasoning_text + turn.raw_action_text, add_special_tokens=False
            )
        )
        start = len(flattened_ids)
        flattened_ids.extend(completion_ids)
        flattened_mask.extend([int(turn.trainable)] * len(completion_ids))
        action_spans.append((start, len(flattened_ids)) if completion_ids else None)
    token_ids, mask = tuple(flattened_ids), tuple(flattened_mask)
    relation_choice_spans: list[tuple[int, int] | None] = list(action_spans)
    for index, turn in enumerate(run.turns):
        if turn.turn_kind != "relation_choice":
            relation_choice_spans[index] = None
            continue
        span = action_spans[index]
        policy = turn.relation_decision.get("policy", {})
        token_map = policy.get("token_ids", {}) if isinstance(policy, dict) else {}
        try:
            expected_id = int(token_map[turn.model_action])
        except (KeyError, TypeError, ValueError):
            relation_choice_spans[index] = None
            continue
        positions = (
            [position for position in range(span[0], span[1]) if token_ids[position] == expected_id]
            if span is not None
            else []
        )
        relation_choice_spans[index] = (
            (positions[0], positions[0] + 1) if len(positions) == 1 else None
        )
    verification = result.solver_result.verification
    worker_backend_failure = result.task.metadata.get("worker_backend_failure")
    worker_artifact_integrity = result.task.metadata.get("worker_artifact_integrity", {})
    worker_output_integrity_risks = result.task.metadata.get("worker_output_integrity_risks", [])
    worker_artifact_integrity_failure = result.task.metadata.get(
        "worker_artifact_integrity_failure"
    )
    alfworld_output_progress = result.task.metadata.get("alfworld_output_progress", {})
    dataset_key = canonical_dataset_name(result.task.metadata.get("dataset", ""))
    from .submission_contract import (
        OutcomeDecision,
        is_text_submission_dataset,
        receipt_error,
    )

    text_primary = is_text_submission_dataset(dataset_key)
    outcome = result.solver_result.outcome_decision
    runtime_outcome = isinstance(outcome, OutcomeDecision) and outcome.runtime_owned
    receipt = run.submission_receipt
    submission_error = receipt_error(
        receipt, run=run, events=result.solver_result.trace.events,
        run_id=result.run_id, dataset=dataset_key,
    )
    text_score_valid = bool(
        runtime_outcome and outcome.status == "scored"
        and submission_error is None and outcome.receipt_ref == receipt.receipt_ref
        and verification == outcome.verification
    )
    if not text_score_valid:
        # A caller cannot restore a candidate score by populating verification/metadata.
        verification = None
    stateful_environment_result = (
        result.task.metadata.get("alfworld_environment_result", {})
        if dataset_key == "alfworld"
        else {}
    )
    # ALFWorld owns the final answer: the canonical episode receipt is the
    # result of the environment, while the Worker text field may still contain
    # WORKER_PROTOCOL_FAILURE after its optional textual repair was exhausted.
    # Accept that boundary only for a fully successful, runtime-owned episode
    # with no backend/tool failure.  The expected terminal protocol sentinel
    # is the one integrity risk this boundary is allowed to repair; any other
    # integrity risk keeps the result fail-closed. Other datasets and
    # unsuccessful episodes continue to fail closed on the sentinel.
    alfworld_receipt_payload = (
        receipt.payload if receipt is not None and isinstance(receipt.payload, dict) else {}
    )
    alfworld_tool_failure = any(
        isinstance(artifact, dict)
        and isinstance(artifact.get("runtime_tool_evidence"), dict)
        and (
            int(artifact["runtime_tool_evidence"].get("failed_count", 0) or 0) > 0
            or bool(artifact["runtime_tool_evidence"].get("failure_codes"))
        )
        for artifact in worker_artifact_integrity.values()
    )
    alfworld_integrity_risks = {
        str(value)
        for value in (
            worker_artifact_integrity_failure.get("risks", ())
            if isinstance(worker_artifact_integrity_failure, dict)
            else ()
        )
    }
    alfworld_integrity_safe = bool(
        not worker_artifact_integrity_failure
        or alfworld_integrity_risks <= {"terminal_protocol_failure"}
    )
    alfworld_official_success = bool(
        dataset_key == "alfworld"
        and receipt is not None
        and receipt.payload_kind == "episode"
        and submission_error is None
        and not worker_backend_failure
        and alfworld_integrity_safe
        and not alfworld_tool_failure
        and isinstance(stateful_environment_result, dict)
        and stateful_environment_result.get("environment_completed") is True
        and stateful_environment_result.get("done") is True
        and stateful_environment_result.get("won") is True
        and float(stateful_environment_result.get("official_score", 0.0) or 0.0) == 1.0
        and alfworld_receipt_payload.get("environment_completed") is True
        and alfworld_receipt_payload.get("done") is True
        and alfworld_receipt_payload.get("won") is True
        and float(alfworld_receipt_payload.get("official_score", 0.0) or 0.0) == 1.0
    )
    answer_score = float(verification.score) if verification else None if text_primary else 0.0
    result_payload = result.to_dict()
    submission = result.solver_result.answer_submission
    raw_submitted_output = submission.submitted_answer if submission else run.output
    raw_submission_valid = bool(submission.valid if submission else run.output.strip())
    alfworld_protocol_repair_diagnostic = None
    if alfworld_official_success and not raw_submission_valid:
        alfworld_protocol_repair_diagnostic = {
            "schema_version": "alfworld_environment_receipt_repair",
            "reason": "official_environment_success_overrides_worker_text_sentinel",
            "raw_worker_output": run.output,
            "raw_submitted_answer": raw_submitted_output,
            "raw_submission_valid": raw_submission_valid,
            "receipt_payload_kind": receipt.payload_kind,
        }
    submitted_output = (
        receipt.submitted_answer_snapshot
        if alfworld_official_success
        else raw_submitted_output
    )
    protocol_submission_valid = True if alfworld_official_success else raw_submission_valid
    protocol_reward = calculate_director_reward(
        task=result.task.prompt,
        graph=run.graph,
        events=result.solver_result.trace.events,
        finished=run.finished,
        output=submitted_output,
        answer_score=answer_score if answer_score is not None else 0.0,
        submission_valid=protocol_submission_valid,
        worker_backend_failure=bool(worker_backend_failure),
        submission_receipt=(receipt.to_dict() if submission_error is None else None),
    )
    answer_reward_released = protocol_reward.answer_reward_released
    reward_semantics = "outcome_only"
    director_reward = 0.0
    task_reward_breakdown: dict[str, Any] = {"source": "pending_eligibility"}
    graph = MultiAgentGraph.from_dict(run.graph)
    canvas_events = [
        event for event in result.solver_result.trace.events if event.kind == "canvas_step"
    ]
    cross_agent_read_audit = {
        "schema_version": 1,
        "policy": "record_only",
        "pair_count": 0,
        "flagged_pair_count": 0,
        "duplicate_read_only_exploration": False,
        "pairs": [],
    }
    # A final Canvas event can carry the previous ExecutionReport solely to
    # expose the selected artifact. It is not another Worker execution and must
    # not inflate Worker/revision metrics.
    execution_payloads = [
        execution
        for event in canvas_events
        if isinstance((execution := event.payload.get("execution")), dict)
        and (
            bool(execution.get("execution_events"))
            or bool(execution.get("executed_agents"))
            or bool(execution.get("reused_agents"))
        )
    ]
    bidirectional_revision_decisions = [
        dict(decision)
        for payload in execution_payloads
        for decision in payload.get("revision_decisions", ())
        if isinstance(decision, dict)
    ]
    bidirectional_revision_reason_codes = sorted(
        {
            str(reason)
            for decision in bidirectional_revision_decisions
            for reason in decision.get("reason_codes", ())
            if str(reason)
        }
    )
    prompt_revision_payloads = [
        dict(event.payload["prompt_revision"])
        for event in canvas_events
        if isinstance(event.payload.get("prompt_revision"), dict)
        and event.payload["prompt_revision"]
    ]
    prompt_revision_bases = sorted(
        {str(payload.get("basis", "")) for payload in prompt_revision_payloads} - {""}
    )
    responsibility_overlap_checks = [
        dict(event.payload["responsibility_overlap_check"])
        for event in canvas_events
        if isinstance(event.payload.get("responsibility_overlap_check"), dict)
        and event.payload["responsibility_overlap_check"]
    ]
    duplicate_responsibility_detection_count = sum(
        int(
            any(
                bool(comparison.get("high_confidence_duplicate"))
                for comparison in check.get("comparisons", ())
                if isinstance(comparison, dict)
            )
        )
        for check in responsibility_overlap_checks
    )
    duplicate_responsibility_decision_counts = {
        decision: sum(
            int(str(check.get("decision", "")) == decision)
            for check in responsibility_overlap_checks
        )
        for decision in (
            "record_only",
            "rewrite_requested",
            "accepted_after_warning",
            "rejected",
        )
    }
    executed_agent_count = sum(
        len(payload.get("executed_agents", ())) for payload in execution_payloads
    )
    reused_agent_count = sum(
        len(payload.get("reused_agents", ())) for payload in execution_payloads
    )
    execution_agent_count = executed_agent_count + reused_agent_count
    worker_model_call_count = sum(
        int(payload.get("worker_model_calls_total", len(payload.get("executed_agents", ()))))
        for payload in execution_payloads
    )
    worker_initial_model_call_count = sum(
        int(payload.get("initial_model_calls", len(payload.get("executed_agents", ()))))
        for payload in execution_payloads
    )
    worker_revision_model_call_count = sum(
        int(payload.get("revision_model_calls", 0)) for payload in execution_payloads
    )
    worker_cache_hit_count = sum(
        int(payload.get("cache_hits", 0)) for payload in execution_payloads
    )
    worker_component_execution_count = sum(
        int(payload.get("component_execution_count", 0)) for payload in execution_payloads
    )
    worker_scheduled_agent_count = sum(
        len(payload.get("scheduled_agents", ())) for payload in execution_payloads
    )
    worker_attempt_count = worker_model_call_count + worker_cache_hit_count
    repair_payloads = [
        event.payload.get("structural_repair")
        for event in canvas_events
        if isinstance(event.payload.get("structural_repair"), dict)
    ]
    audit_payloads = [
        event.payload.get("topology_audit")
        for event in canvas_events
        if isinstance(event.payload.get("topology_audit"), dict)
    ]
    final_graph_errors = graph.validate(final=True)
    bounded_director_terminal = next(
        (
            str(event.payload.get("rejection_code"))
            for event in reversed(canvas_events)
            if event.payload.get("rejection_code")
            in {
                "director_no_legal_continuation",
                "director_no_progress_exhausted",
                "director_action_protocol_exhausted",
            }
        ),
        None,
    )
    terminal_execution_codes = sorted(
        {
            str(event.payload.get("rejection_code"))
            for event in canvas_events
            if event.payload.get("rejection_code")
            in {
                "execution_budget_exceeded",
                "worker_backend_unavailable",
                "execution_failure",
            }
        }
    )
    training_exclusion_reasons: list[str] = []
    if result.task.metadata.get("output_contract_failure"):
        training_exclusion_reasons.append("output_contract_failure")
    if submission_error:
        training_exclusion_reasons.append(submission_error)
    if not run.turns or any(not turn.trainable for turn in run.turns):
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if policy_encoding_error:
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if any(
        turn.token_provenance != "mock_text"
        and (
            not turn.prompt_token_ids
            or not turn.completion_token_ids
            or len(turn.behavior_log_probs) != len(turn.completion_token_ids)
            or any(not math.isfinite(value) or value > 0 for value in turn.behavior_log_probs)
        )
        for turn in run.turns
    ):
        training_exclusion_reasons.append("director_policy_call_ineligible")
    if not run.finished:
        training_exclusion_reasons.append("not_finished")
    if final_graph_errors:
        training_exclusion_reasons.append("invalid_final_graph")
    if not protocol_reward.execution_complete:
        training_exclusion_reasons.append("execution_incomplete")
    if worker_backend_failure:
        training_exclusion_reasons.append("worker_backend_failure")
    if worker_artifact_integrity_failure and not alfworld_official_success:
        training_exclusion_reasons.append("worker_artifact_integrity_failure")
    training_exclusion_reasons.extend(terminal_execution_codes)
    if bounded_director_terminal:
        # Runtime stopping is not by itself proof of a trainable model failure.
        training_exclusion_reasons.append(bounded_director_terminal)
    training_exclusion_reasons = list(dict.fromkeys(training_exclusion_reasons))
    # Reward attribution and policy-data integrity are independent. A known
    # failure remains a known zero even when its policy record cannot enter PPO.
    policy_data_exclusions = [
        reason
        for reason in training_exclusion_reasons
        if reason == "director_policy_call_ineligible"
    ]
    from .outcome_admission import model_tool_errors

    terminal = result.task.metadata.get("runtime_terminal_policy_failure")
    runtime_terminal_score = bool(
        verification and verification.verifier == "runtime_policy_terminal"
    )
    unresolved_tool_failure = any(
        bool(
            set(artifact.get("runtime_tool_evidence", {}).get("failure_codes", ()))
            - model_tool_errors(dataset_key)
        )
        and not artifact.get("runtime_tool_evidence", {}).get("recovered_failure")
        for artifact in worker_artifact_integrity.values()
        if isinstance(artifact, dict)
    )
    if unresolved_tool_failure:
        training_exclusion_reasons.append("tool_failure_attribution_unresolved")
    clearable_terminal_reasons = {
        "not_finished",
        "invalid_final_graph",
        "execution_incomplete",
        "worker_artifact_integrity_failure",
        "execution_budget_exceeded",
    }
    # Only the solver's runtime-owned decision is authoritative.
    trusted_result = text_score_valid or alfworld_official_success
    typed_policy_failure = None
    if (runtime_outcome and outcome.status == "policy_failure"
            and isinstance(terminal, dict)
            and terminal.get("source") == "runtime_terminal_ledger"
            and terminal.get("code") == outcome.reason
            and not worker_backend_failure and not unresolved_tool_failure
            and not result.task.metadata.get("output_contract_failure")):
        typed_policy_failure = terminal
    if runtime_outcome and outcome.status == "scoring_pending":
        training_exclusion_reasons.append("scoring_pending")
    if (
        not text_primary
        and not alfworld_official_success
        and isinstance(terminal, dict)
        and terminal.get("source") == "runtime_terminal_ledger"
        and not worker_backend_failure
        and not unresolved_tool_failure
        and set(training_exclusion_reasons)
        <= clearable_terminal_reasons | {"director_policy_call_ineligible"}
        and (runtime_terminal_score or not (submission.valid if submission else run.output.strip()))
    ):
        typed_policy_failure = terminal
    if typed_policy_failure is not None:
        training_exclusion_reasons = list(policy_data_exclusions)
        director_reward = 0.0
    if (
        verification is None
        and typed_policy_failure is None
        and dataset_key in {"aime", "nq_open", "hotpotqa", "alfworld"}
    ):
        training_exclusion_reasons.append("missing_trusted_verification")
    elif (
        verification is not None
        and not trusted_result
        and typed_policy_failure is None
        and not training_exclusion_reasons
    ):
        training_exclusion_reasons.append("outcome_attribution_unresolved")
    training_eligible = not training_exclusion_reasons
    if typed_policy_failure is not None:
        task_reward_breakdown = {
            "source": "typed_model_policy_failure",
            "failure": typed_policy_failure,
        }
    elif trusted_result:
        director_reward, task_reward_breakdown = _outcome_task_reward(
            str(result.task.metadata.get("dataset", "")),
            verification,
            prediction=submitted_output,
        )
    else:
        task_reward_breakdown = {
            "source": "training_ineligible",
            "reasons": list(training_exclusion_reasons),
        }
    answer_reward_released = bool(trusted_result and typed_policy_failure is None)
    if runtime_outcome:
        outcome = replace(
            outcome, training_eligible=training_eligible,
            training_exclusion_reasons=tuple(training_exclusion_reasons),
        )
    last_safe_graph: dict[str, Any] = {}
    for event in canvas_events:
        payload = event.payload
        raw_graph = payload.get("graph")
        audit = payload.get("topology_audit")
        repair = payload.get("structural_repair")
        if (
            not payload.get("accepted")
            or not isinstance(raw_graph, dict)
            or not isinstance(audit, dict)
            or not isinstance(repair, dict)
            or repair.get("required")
            or int(audit.get("agent_count", 0)) <= 0
            or int(audit.get("configured_agent_count", 0)) != int(audit.get("agent_count", 0))
            or int(audit.get("weak_component_count", 0)) > 1
        ):
            continue
        last_safe_graph = dict(raw_graph)

    def final_repair_counter(name: str) -> int:
        return max(
            (int(payload.get(name, 0) or 0) for payload in repair_payloads),
            default=0,
        )

    return SolverRollout(
        TokenizedDirectorTrajectory(
            rollout_id=f"{result.task.task_id}-r{rollout_index}",
            task_id=result.task.task_id,
            token_ids=token_ids,
            action_mask=mask,
            reward=director_reward,
            graph=run.graph,
            seed=seed,
            executor_version="adaptive",
            metadata={
                "run_id": result.run_id,
                "dataset": dataset_key,
                "finished": run.finished,
                "interactive_turns": len(run.turns),
                "accepted_turns": sum(int(turn.accepted) for turn in run.turns),
                "rejected_turns": sum(int(not turn.accepted) for turn in run.turns),
                "responsibility_violation_count": sum(
                    int(turn.rejection_code == "responsibility_violation") for turn in run.turns
                ),
                "duplicate_responsibility_rejection_count": sum(
                    int(turn.rejection_code == "duplicate_responsibility") for turn in run.turns
                ),
                "duplicate_responsibility_detection_count": (
                    duplicate_responsibility_detection_count
                ),
                "duplicate_responsibility_decision_counts": (
                    duplicate_responsibility_decision_counts
                ),
                "duplicate_responsibility_policy": next(
                    (
                        str(check.get("policy"))
                        for check in reversed(responsibility_overlap_checks)
                        if check.get("policy")
                    ),
                    "",
                ),
                "cross_agent_exact_read_overlap": cross_agent_read_audit,
                "duplicate_read_only_exploration": bool(
                    cross_agent_read_audit["duplicate_read_only_exploration"]
                ),
                "delegation_field_repair_count": sum(
                    len(event.payload.get("delegation_field_repairs", ()))
                    for event in canvas_events
                ),
                "managed_delegation_contract_count": sum(
                    int(
                        bool(
                            node.get("metadata", {})
                            .get("system_managed_contract", {})
                            .get("version")
                        )
                    )
                    for node in run.graph.get("nodes", ())
                ),
                "director_action_diagnostics": [
                    dict(turn.action_diagnostics) for turn in run.turns
                ],
                "postsolve_deadline_exceeded": result.task.metadata.get(
                    "postsolve_deadline_exceeded"
                ),
                "director_action_repairs": sum(
                    int(bool(turn.action_diagnostics.get("repair_attempted"))) for turn in run.turns
                ),
                "director_action_repair_successes": sum(
                    int(bool(turn.action_diagnostics.get("repair_succeeded"))) for turn in run.turns
                ),
                "director_discarded_output_chars": sum(
                    int(turn.action_diagnostics.get("discarded_output_chars", 0) or 0)
                    for turn in run.turns
                ),
                "protocol_recovery_count": sum(
                    int(bool(step.payload.get("protocol_recovery")))
                    for step in result.solver_result.trace.events
                    if step.kind == "canvas_step"
                ),
                "skills_used": list(result.skills_used),
                "skill_context": dict(result.solver_result.skill_context),
                "verification": asdict(verification) if verification else None,
                "answer_score": answer_score,
                "evaluation_score": answer_score,
                "answer_reward_released": answer_reward_released,
                "director_reward": director_reward,
                "base_director_reward": director_reward,
                "task_reward": (director_reward if not text_primary or trusted_result or typed_policy_failure is not None else None),
                "task_reward_breakdown": task_reward_breakdown,
                "reward_semantics": reward_semantics,
                "director_reward_version": OUTCOME_ONLY_REWARD_VERSION,
                "protocol_audit_version": protocol_reward.version,
                "protocol_reward_diagnostic": protocol_reward.to_dict(),
                "protocol_reward": protocol_reward.to_dict(),
                "alfworld_protocol_repair_diagnostic": alfworld_protocol_repair_diagnostic,
                "protocol_score": protocol_reward.protocol_score,
                "protocol_qualified": protocol_reward.qualified,
                "delegation_fidelity": protocol_reward.delegation_fidelity,
                "delegation_issues": list(protocol_reward.delegation_issues),
                "skills_enabled": result.solver_result.trace.task.metadata.get("skills_enabled"),
                "action_protocol": "director_model",
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "submission_contract_version": result.task.metadata.get("submission_contract_version"),
                "submission_receipt": receipt.to_dict() if receipt else None,
                "submission_status": ("submitted" if (text_primary and submission_error is None) or receipt
                                      else "unsubmitted"),
                "outcome_decision": outcome.to_dict() if runtime_outcome else None,
                "candidate_output": run.candidate_output,
                "diagnostic_qa_metrics": result.task.metadata.get("diagnostic_qa_metrics"),
                "environment_request_events": result.task.metadata.get(
                    "environment_request_events", []
                ),
                "deadline_accounting": result.task.metadata.get("deadline_accounting", {}),
                "model_roles": result.task.metadata.get("model_roles", {}),
                "output_contract_failure": result.task.metadata.get("output_contract_failure"),
                "output_artifact_binding": result.task.metadata.get("output_artifact_binding"),
                "canvas_prefixes": prefixes,
                "action_token_spans": action_spans,
                "relation_choice_token_spans": tuple(relation_choice_spans),
                "trajectory_schema": "director_trajectory_raw_policy_calls",
                "director_sampling_seed": seed,
                "primary_executor_seed": seed,
                "executor_bundle_signature": result.solver_result.trace.task.metadata.get(
                    "executor_bundle_signature"
                ),
                "solver_answer": run.output,
                "raw_solver_answer": run.output,
                "submitted_answer": (
                    submitted_output
                    if alfworld_official_success
                    else (submission.submitted_answer if submission else run.output)
                ),
                "answer_submission": submission.to_dict() if submission else None,
                "qa_token_f1": result.task.metadata.get("qa_token_f1"),
                "qa_official_metrics": result.task.metadata.get("qa_official_metrics"),
                "environment_result_metrics": {
                    key: stateful_environment_result[key]
                    for key in (
                        "steps",
                        "success",
                        "terminal",
                        "reward",
                    )
                    if key in stateful_environment_result
                    and isinstance(stateful_environment_result[key], (bool, int, float))
                },
                "solver_trace": result.solver_result.trace.to_dict(),
                "backend_request_events": result.task.metadata.get("backend_request_events", []),
                "director_request_events": [
                    event
                    for turn in run.turns
                    for event in turn.action_diagnostics.get("backend_request_events", [])
                ],
                "worker_backend_failure": worker_backend_failure,
                "worker_artifact_integrity": worker_artifact_integrity,
                "worker_output_integrity_risks": worker_output_integrity_risks,
                "worker_artifact_integrity_failure": (worker_artifact_integrity_failure),
                "alfworld_output_progress": alfworld_output_progress,
                "training_eligible": training_eligible,
                "reward_known": trusted_result or typed_policy_failure is not None,
                "task_outcome_passed": (bool(verification.passed) if trusted_result
                                        else False if typed_policy_failure is not None else None),
                "reward_admission_reason": (
                    "trusted_task_result"
                    if trusted_result
                    else "explicit_policy_terminal"
                    if typed_policy_failure is not None
                    else "infrastructure_or_unresolved_attribution"
                ),
                "policy_data_exclusion_reasons": policy_data_exclusions,
                "policy_encoding_error": policy_encoding_error,
                "runtime_terminal_policy_failure": terminal,
                "bounded_director_terminal": bounded_director_terminal,
                "training_exclusion_reasons": training_exclusion_reasons,
                "typed_policy_failure": typed_policy_failure,
                "terminal_status": (
                    "typed_policy_failure"
                    if typed_policy_failure is not None
                    else ("completed" if training_eligible else "excluded")
                ),
                "terminal_graph_status": (
                    "typed_policy_failure"
                    if typed_policy_failure is not None
                    else (
                        "unfinished_with_trusted_result"
                        if trusted_result and not run.finished
                        else "valid_finished"
                        if training_eligible
                        # A finished graph with a trusted result stays outcome evidence
                        # (Frontier, PATS, ADS) when only a Director call is untrainable,
                        # e.g. a thinking-budget boundary; training still excludes it.
                        or (
                            trusted_result
                            and run.finished
                            and set(training_exclusion_reasons) <= {"director_policy_call_ineligible"}
                        )
                        else "unsafe_partial"
                    )
                ),
                "final_graph_validation_errors": final_graph_errors,
                "last_safe_graph": last_safe_graph,
                # This field is consumed only for zero-reward skill cases, but it is
                # also part of the persisted rollout metadata used for audits.  Do
                # not label successful trajectories as failures.
                "failure_mode": (
                    "worker_backend_failure"
                    if worker_backend_failure
                    else (
                        "typed_policy_failure"
                        if typed_policy_failure is not None
                        else (
                            "worker_artifact_integrity_failure"
                            if worker_artifact_integrity_failure
                            else (
                                None
                                if director_reward > 0.0
                                else (
                                    "director_protocol_failure"
                                    if not protocol_reward.qualified
                                    else "task_verification_failure"
                                )
                            )
                        )
                    )
                ),
                "duration_s": duration_s,
                "token_in": int(result_payload.get("token_in", 0)),
                "token_out": int(result_payload.get("token_out", 0)),
                # Worker token usage of the rollout and its normalised worker_cost
                # (worker_cost.py); always recorded, see CostConfig.
                "worker_cost_record": result.task.metadata.get("worker_cost_record"),
                "final_execution_round": next(
                    (
                        int(event.payload.get("director_turn_index", 0)) + 1
                        for event in result.solver_result.trace.events
                        if event.kind == "canvas_step"
                        and event.payload.get("final_execution")
                        and event.payload.get("director_turn_index") is not None
                    ),
                    None,
                ),
                "incremental_execution_count": sum(
                    int(
                        event.payload.get("execution") is not None
                        and not bool(event.payload.get("final_execution"))
                    )
                    for event in result.solver_result.trace.events
                    if event.kind == "canvas_step"
                ),
                "worker_executed_agent_count": executed_agent_count,
                "worker_reused_agent_count": reused_agent_count,
                "worker_scheduled_agent_count": worker_scheduled_agent_count,
                "worker_model_call_count": worker_model_call_count,
                "worker_initial_model_call_count": worker_initial_model_call_count,
                "worker_revision_model_call_count": worker_revision_model_call_count,
                "worker_cache_hit_count": worker_cache_hit_count,
                "worker_initial_cache_hit_count": sum(
                    int(payload.get("initial_cache_hits", 0)) for payload in execution_payloads
                ),
                "worker_revision_cache_hit_count": sum(
                    int(payload.get("revision_cache_hits", 0)) for payload in execution_payloads
                ),
                "worker_mandatory_revision_call_count": sum(
                    int(payload.get("mandatory_revision_calls", 0))
                    for payload in execution_payloads
                ),
                "worker_incomplete_bidirectional_components": [
                    dict(item)
                    for payload in execution_payloads
                    for item in payload.get("incomplete_bidirectional_components", ())
                    if isinstance(item, dict)
                ],
                "worker_component_execution_count": worker_component_execution_count,
                "worker_bidirectional_revision_gate_count": len(bidirectional_revision_decisions),
                "worker_bidirectional_revision_required_count": sum(
                    int(bool(decision.get("revision_required")))
                    for decision in bidirectional_revision_decisions
                ),
                "worker_bidirectional_revision_skipped_agent_count": sum(
                    len(payload.get("revision_skipped_agents", ()))
                    for payload in execution_payloads
                ),
                "worker_bidirectional_revision_wave_count": sum(
                    int(payload.get("revision_wave_count", 0)) for payload in execution_payloads
                ),
                "worker_bidirectional_revision_reason_counts": {
                    reason: sum(
                        int(reason in decision.get("reason_codes", ()))
                        for decision in bidirectional_revision_decisions
                    )
                    for reason in bidirectional_revision_reason_codes
                },
                "worker_bidirectional_revision_decisions": (bidirectional_revision_decisions),
                "worker_attempt_cache_hit_rate": (
                    worker_cache_hit_count / worker_attempt_count if worker_attempt_count else 0.0
                ),
                "prompt_revision_count": len(prompt_revision_payloads),
                "prompt_revision_basis_counts": {
                    basis: sum(
                        int(str(payload.get("basis", "")) == basis)
                        for payload in prompt_revision_payloads
                    )
                    for basis in prompt_revision_bases
                },
                "prompt_revision_target_model_call_count": sum(
                    int(payload.get("target_model_calls", 0))
                    for payload in prompt_revision_payloads
                ),
                "prompt_revision_worker_model_call_count": sum(
                    int(payload.get("worker_model_calls_total", 0))
                    for payload in prompt_revision_payloads
                ),
                "prompt_revision_evidence_rejection_count": sum(
                    int(event.payload.get("rejection_code") == "prompt_revision_evidence_required")
                    for event in canvas_events
                ),
                "prompt_revision_events": prompt_revision_payloads,
                "worker_cache_reuse_rate": (
                    reused_agent_count / execution_agent_count if execution_agent_count else 0.0
                ),
                "feedback_truncation_count": sum(
                    int("[feedback truncated]" in str(event.payload.get("feedback", "")))
                    for event in canvas_events
                ),
                "structural_repair_entry_count": final_repair_counter("entries_total"),
                "structural_repair_resolution_count": final_repair_counter("resolutions_total"),
                "structural_repair_blocked_action_count": final_repair_counter(
                    "blocked_actions_total"
                ),
                "semantic_no_progress_recovery_count": final_repair_counter(
                    "semantic_no_progress_recoveries_total"
                ),
                "output_switch_without_progress_count": final_repair_counter(
                    "output_switches_without_progress_total"
                ),
                "output_lifecycle_recovery_count": final_repair_counter(
                    "output_lifecycle_recoveries_total"
                ),
                "finish_reachability_rejection_count": final_repair_counter(
                    "finish_reachability_rejections_total"
                ),
                "duplicate_agent_rejection_count": final_repair_counter(
                    "duplicate_agent_rejections_total"
                ),
                "disconnected_multi_agent_step_count": sum(
                    int(bool(payload.get("disconnected_multi_agent"))) for payload in audit_payloads
                ),
                "token_admission_rejection_count": sum(
                    int(bool(event.payload.get("token_admission", {}).get("blocked")))
                    for event in canvas_events
                    if isinstance(event.payload.get("token_admission"), dict)
                ),
                "token_admission_events": [
                    dict(event.payload["token_admission"])
                    for event in canvas_events
                    if isinstance(event.payload.get("token_admission"), dict)
                    and event.payload["token_admission"].get("blocked")
                ],
                "final_execution_count": sum(
                    int(bool(event.payload.get("final_execution")))
                    for event in result.solver_result.trace.events
                    if event.kind == "canvas_step"
                ),
            },
            policy_calls=policy_calls,
        ),
        graph,
    )


@dataclass(frozen=True)
class SelfPlayRunConfig:
    rollouts_per_task: int = 5
    base_seed: int = 0
    max_tokens: int = 4096
    workers: int = 1
    # Every primary trajectory owns the same complete wall-clock budget.
    rollout_wall_time_s: float = 900.0
    # A larger AIME budget for hard competition problems.
    aime_rollout_wall_time_s: float = 900.0
    stateful_rollout_wall_time_s: float = 900.0
    rollout_slot_wall_time_s: float = 900.0
    stateful_slot_wall_time_s: float = 900.0
    rollout_no_progress_time_s: float = 120.0
    request_wall_time_s: float = 180.0
    aime_request_wall_time_s: float = 240.0
    # Stateful ALFWorld episodes can spend several minutes in one free-text
    # Worker request.
    alfworld_request_wall_time_s: float = 180.0
    recovery_attempts: int = 1


class SelfPlayRolloutRunner:
    """Collects one primary Director rollout (one sibling slot) for a proposed task."""

    def __init__(
        self,
        *,
        proposer: Any,
        application_factory: Any,
        tokenizer: Tokenizer,
        snapshots: AlternatingSnapshots,
        output_dir: str | Path,
        config: SelfPlayRunConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.application_factory = application_factory
        self.tokenizer = tokenizer
        self.snapshots = snapshots
        self.output_dir = Path(output_dir)
        self.config = config or SelfPlayRunConfig()

        if self.config.rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two")
        if self.config.rollout_wall_time_s <= 0:
            raise ValueError("rollout_wall_time_s must be positive")
        if self.config.aime_rollout_wall_time_s <= 0:
            raise ValueError("aime_rollout_wall_time_s must be positive")
        if self.config.stateful_rollout_wall_time_s <= 0:
            raise ValueError("stateful_rollout_wall_time_s must be positive")
        if self.config.rollout_no_progress_time_s <= 0:
            raise ValueError("rollout_no_progress_time_s must be positive")
        if self.config.request_wall_time_s <= 0:
            raise ValueError("request_wall_time_s must be positive")
        if self.config.aime_request_wall_time_s <= 0:
            raise ValueError("aime_request_wall_time_s must be positive")
        if self.config.alfworld_request_wall_time_s <= 0:
            raise ValueError("alfworld_request_wall_time_s must be positive")
        if self.config.recovery_attempts < 0:
            raise ValueError("recovery_attempts must be non-negative")

    def _collect_primary(
        self,
        job: tuple[ProposedTask, int],
        *,
        replacement_attempt: int = 0,
        cancellation_event: threading.Event | None = None,
        sibling_index: int | None = None,
        execution_seed: int | None = None,
        r2orch_metadata: dict[str, Any] | None = None,
    ) -> _PrimaryCollection:
        proposal, rollout_index = job
        rollout_seed = _rollout_sampling_seed(
            self.config.base_seed,
            proposal.task.task_id,
            rollout_index,
            replacement_attempt,
        )
        executor_seed = (
            int(execution_seed)
            if execution_seed is not None
            else _stable_execution_seed(
                self.config.base_seed, proposal.task.task_id, phase="primary"
            )
        )
        started = time.monotonic()
        total_timeout_s, request_timeout_s, request_overrides = self._deadline_profile(proposal)
        deadline = RolloutDeadline(
            total_timeout_s=total_timeout_s,
            no_progress_timeout_s=min(
                self.config.rollout_no_progress_time_s,
                total_timeout_s,
            ),
            request_timeout_s=min(request_timeout_s, total_timeout_s),
            request_timeout_overrides_s=request_overrides,
            started_monotonic=started,
            cancellation_event=cancellation_event,
            exclude_failed_request_time=True,
            # The absolute guard must not be shorter than a dataset-specific
            # effective budget.
            absolute_wall_timeout_s=max(900.0, total_timeout_s),
        )
        application: AdaptiveSolverApplication = self.application_factory(rollout_seed)
        application.runtime.seed = executor_seed
        solve_metadata = dict(proposal.task.metadata)
        solve_metadata["same_slot_recovery_attempt"] = replacement_attempt
        if r2orch_metadata is not None:
            # R2Orch sibling identity; adaptive.solve resolves the effective
            # scaffold condition and the reference context into this mapping.
            solve_metadata["r2orch"] = {
                **r2orch_metadata,
                "sibling_index": rollout_index if sibling_index is None else int(sibling_index),
                "execution_seed": executor_seed,
                "director_sampling_seed": rollout_seed,
            }
        application.set_rollout_deadline(deadline)
        try:
            solve_kwargs: dict[str, Any] = {
                "task_id": proposal.task.task_id,
                "task_type": proposal.task.task_type,
                "reference": proposal.task.reference,
                "run_id": (
                    f"{proposal.task.task_id}-r{rollout_index}"
                    if replacement_attempt == 0
                    else (
                        f"{proposal.task.task_id}-r{rollout_index}"
                        f"-replacement-{replacement_attempt}"
                    )
                ),
                "metadata": solve_metadata,
            }
            if proposal.task.private_verifier_payload:
                solve_kwargs["private_verifier_payload"] = proposal.task.private_verifier_payload
            result = application.solve(proposal.task.prompt, **solve_kwargs)
            environment_events = []
            seen_clients = set()
            for tool in application.runtime.executor.tools.values():
                client = getattr(getattr(tool, "lifecycle", None), "client", None)
                if client is not None and id(client) not in seen_clients:
                    seen_clients.add(id(client))
                    environment_events.extend(getattr(client, "retry_events", ()))
            result.task.metadata["environment_request_events"] = environment_events
            result.task.metadata["deadline_accounting"] = deadline.diagnostics()
            backend_failure = result.task.metadata.get("worker_backend_failure")
            if backend_failure:
                raise WorkerBackendUnavailableError(backend_failure)
            duration_s = time.monotonic() - started
            rollout = adaptive_result_to_rollout(
                result,
                self.tokenizer,
                rollout_index=rollout_index,
                seed=rollout_seed,
                max_tokens=self.config.max_tokens,
                duration_s=duration_s,
            )
            rollout.trajectory.metadata.update(
                {
                    "replacement_attempt": replacement_attempt,
                    "policy_sampling_attempt_offset": 0,
                    "policy_sampling_attempt": replacement_attempt,
                    "original_rollout_seed": self.config.base_seed + rollout_index,
                    "effective_rollout_seed": rollout_seed,
                    "director_sampling_seed": rollout_seed,
                    "primary_executor_seed": executor_seed,
                    "same_slot_seed_preserved": True,
                    "same_slot_executor_seed_preserved": True,
                    "policy_resampled_on_replacement": replacement_attempt > 0,
                    "deadline_profile": self._deadline_profile_payload(proposal),
                }
            )
            if r2orch_metadata is not None:
                rollout.trajectory.metadata["r2orch"] = dict(
                    result.task.metadata["r2orch"]
                )
            return _PrimaryCollection(application=application, result=result, rollout=rollout)
        except BaseException as exc:
            if isinstance(exc, WorkerWallClockLimitExceeded):
                exc.partial_state = {
                    **(getattr(exc, "partial_state", None) or {}),
                    **self._partial_timeout_state(application, deadline),
                }
            application.close()
            raise

    def _deadline_profile(self, proposal: ProposedTask) -> tuple[float, float, dict[str, float]]:
        """Return the compact timeout profile selected from public task metadata."""

        dataset = str(proposal.task.metadata.get("dataset", "")).strip().lower()
        if dataset == "alfworld":
            total_timeout_s = self.config.stateful_rollout_wall_time_s
        else:
            total_timeout_s = self.config.rollout_wall_time_s
        if dataset == "aime":
            return (
                max(total_timeout_s, self.config.aime_rollout_wall_time_s),
                self.config.aime_request_wall_time_s,
                {},
            )
        if dataset == "alfworld":
            return total_timeout_s, self.config.alfworld_request_wall_time_s, {}
        return total_timeout_s, self.config.request_wall_time_s, {}

    def _deadline_profile_payload(self, proposal: ProposedTask) -> dict[str, Any]:
        total_timeout_s, request_timeout_s, overrides = self._deadline_profile(proposal)
        return {
            "dataset": str(proposal.task.metadata.get("dataset", "")),
            "total_timeout_s": total_timeout_s,
            "control_idle_timeout_s": min(self.config.rollout_no_progress_time_s, total_timeout_s),
            "request_timeout_s": min(request_timeout_s, total_timeout_s),
            "request_timeout_overrides_s": dict(overrides),
        }

    @staticmethod
    def _partial_timeout_state(
        application: AdaptiveSolverApplication,
        deadline: RolloutDeadline,
    ) -> dict[str, Any]:
        canvas = application.solver.active_canvas
        if canvas is None:
            return {"deadline": deadline.diagnostics()}
        return {
            "deadline": deadline.diagnostics(),
            "canvas_state": canvas.state.value,
            "round_index": canvas.round_index,
            "total_worker_tokens": canvas.total_tokens,
            "pending_agent_id": canvas.pending_agent_id,
            "dirty_agents": sorted(canvas.dirty_agents),
            "graph": canvas.graph.to_dict(),
            "history": [step.to_dict() for step in canvas.history],
        }


def _proposal_dict(value: ProposedTask) -> dict[str, Any]:
    return {
        "task": task_to_public_dict(value.task),
        "response": value.response,
        "token_ids": list(value.token_ids),
        "action_mask": list(value.action_mask),
        "metadata": value.metadata,
        "policy_calls": [asdict(call) for call in value.policy_calls],
    }


def _proposal_from_dict(value: dict[str, Any]) -> ProposedTask:
    return ProposedTask(
        TaskSpec(**value["task"]),
        str(value["response"]),
        tuple(value["token_ids"]),
        tuple(value["action_mask"]),
        dict(value.get("metadata", {})),
        tuple(
            TokenizedPolicyCall(
                call_id=str(call["call_id"]),
                token_ids=tuple(call["token_ids"]),
                action_mask=tuple(call["action_mask"]),
                behavior_log_probs=tuple(call.get("behavior_log_probs", ())),
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
            for call in value.get("policy_calls", ())
        ),
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    with _JSONL_APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _proposal_extraction_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(bool(item.get("success"))) for item in attempts)
    failure_kinds: dict[str, int] = {}
    for item in attempts:
        kind = item.get("failure_kind")
        if not item.get("success") and kind:
            failure_kinds[str(kind)] = failure_kinds.get(str(kind), 0) + 1
    return {
        "attempts": len(attempts),
        "successes": successes,
        "failures": len(attempts) - successes,
        "success_rate": successes / max(1, len(attempts)),
        "failure_kinds": failure_kinds,
    }
