"""Runtime-owned evidence for terminal policy failures (never a default zero)."""

from __future__ import annotations

from typing import Any

from .config import canonical_dataset_name

DATASETS = frozenset(
    {
        "aime",
        "nq_open",
        "hotpotqa",
        "alfworld",
        "medqa",
        "mbppplus",
    }
)
# Do not include generic tool_action_failed/action_execution_failed: these can
# also describe a broken retrieval service, sandbox, or environment transport.
MODEL_TOOL_ERRORS = frozenset(
    {
        "action_arguments_must_be_an_object",
        "invalid_action_arguments",
        "invalid_action_arguments_encoding",
        "action_not_visible",
        "SecurityError",
        "ImportError",
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
    }
)


def model_tool_errors(dataset: str) -> frozenset[str]:
    dataset = canonical_dataset_name(dataset)
    # The runtime deliberately discards extra stateful calls from one model
    # response; no environment request failed for those discarded calls.
    stateful_errors = (
        frozenset({"stateful_action_deferred"})
        if dataset in {"alfworld"}
        else frozenset()
    )
    if dataset in {"aime", "mbppplus"}:
        # Both use only the local python_exec sandbox.
        return MODEL_TOOL_ERRORS
    # Python exception names in a retrieval/environment tool may describe the
    # service itself. Only AIME's code-sandbox path attributes them to user code.
    return (
        frozenset(
            value
            for value in MODEL_TOOL_ERRORS
            if value.startswith("action_") or value.startswith("invalid_action_")
        )
        | stateful_errors
    )


def terminal_policy_failure(
    dataset: str,
    *,
    terminal: bool,
    rejection_codes: list[str],
    artifacts: dict[str, Any],
    output_agent: str | None,
    rounds: int,
    max_rounds: int,
    worker_tokens: int,
    worker_token_limit: int,
    infrastructure_failure: bool = False,
    historical_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Only finite action/token/protocol evidence can establish attribution.

    An elapsed-time limit is deliberately NOT evidence of model failure.
    Failed tools of unknown origin poison this decision, including retrieval
    failures preceding otherwise normal model calls.
    """
    dataset = canonical_dataset_name(dataset)
    codes = set(rejection_codes)
    if dataset not in DATASETS or not terminal or infrastructure_failure:
        return None
    if codes & {
        "execution_failure",
        "worker_backend_unavailable",
        "time_budget_consolidation_required",
    }:
        return None
    # An overrun is an admission/accounting defect, not evidence that the model
    # knowingly spent a correctly enforced budget.
    if worker_token_limit > 0 and worker_tokens > worker_token_limit:
        return None
    for artifact in [*artifacts.values(), *(historical_artifacts or {}).values()]:
        if artifact.get("backend_failure"):
            return None
        evidence = artifact.get("runtime_tool_evidence", {})
        failures = set(evidence.get("failure_codes", ()))
        if failures - model_tool_errors(dataset):
            return None
        if evidence.get("failed_count", 0) and not failures:
            return None
    selected = artifacts.get(output_agent, {}) if output_agent else {}
    # Do not guess which of several local responsibilities was the final answer.
    if not selected and len(artifacts) == 1:
        selected = next(iter(artifacts.values()))
    risks = set(selected.get("integrity_risks", ()))
    selected_evidence = selected.get("runtime_tool_evidence", {})
    selected_errors = set(selected_evidence.get("failure_codes", ()))
    reason = ""
    if risks == {"terminal_protocol_failure"}:
        reason = ("aime_worker_final_protocol_policy_failure" if dataset == "aime"
                  else "answer_protocol_repair_exhausted")
    elif (dataset == "aime" and "terminal_tool_failure" in risks
          and selected_evidence.get("terminal_failure")
          and selected_errors and selected_errors <= model_tool_errors(dataset)):
        reason = "aime_model_tool_policy_failure"
    elif max_rounds > 0 and rounds >= max_rounds and "max_rounds_exhausted" in codes:
        reason = "director_action_budget_exhausted"
    elif (max_rounds > 0 and 0 <= max_rounds - rounds < 4
          and "director_round_budget_dead_end" in codes):
        reason = "director_round_budget_dead_end"
    elif rounds >= 4 and "director_action_protocol_exhausted" in codes:
        reason = "director_action_protocol_exhausted"
    elif ("director_no_progress_exhausted" in codes
          and rejection_codes.count("director_action_not_allowed") >= 3):
        reason = "director_repeated_illegal_actions"
    elif rounds > 0 and "director_context_budget_exhausted" in codes:
        reason = "director_context_budget_exhausted"
    elif worker_tokens > 0 and codes & {
        "token_budget_admission_required",
        "execution_budget_exceeded",
        "token_budget_consolidation_required",
    }:
        reason = "worker_token_budget_exhausted"
    if not reason:
        return None
    return {
        "status": "typed_policy_failure",
        "code": reason,
        "dataset": dataset,
        "attribution": "model_policy",
        "source": "runtime_terminal_ledger",
        "budget": {
            "director_rounds": rounds,
            "director_round_limit": max_rounds,
            "worker_tokens": worker_tokens,
            "worker_token_limit": worker_token_limit,
        },
        "rejection_codes": sorted(codes),
    }


def trusted_environment_outcome(dataset: str, result: object) -> bool:
    """Only the canonical runtime result, never best-of hidden Agent artifacts."""
    if not isinstance(result, dict):
        return False
    dataset = canonical_dataset_name(dataset)
    if dataset == "alfworld":
        return result.get("environment_completed") is True
    return False


def task_reward_from_verification(
    dataset: str,
    verification: Any,
    *,
    prediction: str,
) -> tuple[float, dict[str, Any]]:
    if verification is None:
        return 0.0, {"source": "empty_or_missing_verification"}
    dataset_key = canonical_dataset_name(dataset)
    reward = float(verification.score)
    if not 0.0 <= reward <= 1.0:
        raise ValueError(f"trusted task outcome for {dataset_key or dataset!r} must be in [0, 1]")
    if (dataset_key in frozenset({"aime", "alfworld", "medqa", "mbppplus"})
            and reward not in {0.0, 1.0}):
        raise ValueError(
            f"trusted task outcome for binary dataset {dataset_key!r} must be exactly 0 or 1"
        )
    return reward, {"source": "trusted_verifier_outcome", "evaluation_score": reward}
