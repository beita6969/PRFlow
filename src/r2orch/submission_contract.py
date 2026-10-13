"""Runtime-owned text submissions, separate from answer formatting and policy data."""

from __future__ import annotations

import hashlib
import copy
import json
from dataclasses import dataclass, field, fields
from typing import Any

from .config import canonical_dataset_name

TEXT_DATASETS = frozenset(
    {"aime", "nq_open", "hotpotqa", "medqa", "mbppplus"}
)


class _RuntimeAuthority:
    def __deepcopy__(self, memo):
        return self


_AUTHORITY = _RuntimeAuthority()


def is_text_submission_dataset(dataset: str) -> bool:
    return canonical_dataset_name(dataset) in TEXT_DATASETS


def snapshot_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DirectorCallContext:
    run_id: str
    call_id: str
    _authority: Any = field(default=None, repr=False, compare=False)

    @property
    def runtime_owned(self) -> bool:
        return self._authority is _AUTHORITY and bool(self.run_id and self.call_id)


def _director_call_context(run_id: str, call_id: str) -> DirectorCallContext:
    """Called by the driver after an actual backend response, never by action parsing."""
    return DirectorCallContext(run_id, call_id, _AUTHORITY)


@dataclass(frozen=True)
class SubmissionReceipt:
    version: str
    run_id: str
    dataset: str
    completion_source: str
    director_call_id: str
    accepted_event_id: str
    output_agent_id: str
    graph_snapshot_hash: str
    artifact_id: str
    input_signature: str
    execution_generation: int
    raw_answer_snapshot: str
    submitted_answer_snapshot: str
    answer_hash: str
    normalization_version: str
    worker_tokens_used: int
    worker_token_limit: int
    payload_kind: str = "text"
    payload: dict[str, Any] = field(default_factory=dict)
    transaction_id: str = ""
    _authority: Any = field(default=None, repr=False, compare=False)

    @property
    def runtime_owned(self) -> bool:
        return self._authority is _AUTHORITY

    @property
    def receipt_ref(self) -> str:
        return snapshot_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {item.name: copy.deepcopy(getattr(self, item.name)) for item in fields(self)
                if not item.name.startswith("_")}


def _issue_receipt(*, context: DirectorCallContext, **values) -> SubmissionReceipt:
    if not isinstance(context, DirectorCallContext) or not context.runtime_owned:
        raise ValueError("a runtime-owned Director call is required for text submission")
    values["dataset"] = canonical_dataset_name(values["dataset"])
    return SubmissionReceipt(
        version=values.pop("version"), run_id=context.run_id,
        director_call_id=context.call_id, completion_source="director_finish",
        _authority=_AUTHORITY, **values,
    )


def receipt_error(receipt: Any, *, run: Any, events: Any, run_id: str, dataset: str) -> str | None:
    """Validate runtime evidence; task/model metadata and finished alone confer no authority."""
    if not isinstance(receipt, SubmissionReceipt) or not receipt.runtime_owned:
        return "missing_runtime_submission_receipt"
    if (receipt.version != "unified_submission" or receipt.run_id != run_id
            or receipt.dataset != canonical_dataset_name(dataset)
            or receipt.completion_source != "director_finish"):
        return "submission_origin_mismatch"
    if (not run.finished or snapshot_hash(run.graph) != receipt.graph_snapshot_hash
            or run.graph.get("output_agent") != receipt.output_agent_id):
        return "submission_graph_mismatch"
    if (not receipt.artifact_id or not receipt.input_signature
            or not receipt.submitted_answer_snapshot
            or answer_hash(receipt.submitted_answer_snapshot) != receipt.answer_hash
            or receipt.worker_tokens_used > receipt.worker_token_limit):
        return "submission_snapshot_invalid"
    calls = [turn for turn in run.turns if turn.call_id == receipt.director_call_id]
    if len(calls) != 1 or not calls[0].accepted:
        return "submission_director_call_missing"
    try:
        action = json.loads(calls[0].model_action)
    except (ValueError, TypeError):
        return "submission_director_action_invalid"
    if str(action.get("action", "")).casefold() != "finish":
        return "submission_director_action_invalid"
    # Use precisely the action parser's ID normalization. A numeric JSON ID
    # may have been accepted as the string node ID; validation must bind to
    # that same target instead of revoking an already accepted submission.
    from .actions import ActionParser
    parsed = ActionParser().parse_policy_output(calls[0].model_action).action
    if not parsed.valid or parsed.target != receipt.output_agent_id:
        return "submission_target_mismatch"
    matched = []
    for event in events:
        payload = getattr(event, "payload", None)
        if payload is None:
            payload = event.to_dict() if hasattr(event, "to_dict") else event
        if isinstance(payload, dict) and payload.get("event_id") == receipt.accepted_event_id:
            matched.append(payload)
    if len(matched) != 1:
        return "submission_accept_event_missing"
    event = matched[0]
    if (not event.get("accepted") or not event.get("final_execution")
            or event.get("protocol_recovery")
            or event.get("director_call_id") != receipt.director_call_id
            or event.get("submission_receipt") != receipt.to_dict()):
        return "submission_accept_event_invalid"
    return None


@dataclass(frozen=True)
class OutcomeDecision:
    status: str
    source: str
    reason: str
    receipt_ref: str | None = None
    score_known: bool = False
    verification: Any = None
    task_reward: float | None = None
    training_eligible: bool = False
    training_exclusion_reasons: tuple[str, ...] = ()
    _authority: Any = field(default=None, repr=False, compare=False)

    @property
    def runtime_owned(self) -> bool:
        return self._authority is _AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = {item.name: getattr(self, item.name) for item in fields(self)
                   if not item.name.startswith("_")}
        payload["verification"] = asdict(self.verification) if self.verification else None
        payload["training_exclusion_reasons"] = list(self.training_exclusion_reasons)
        return payload


def decide_text_outcome(*, receipt: SubmissionReceipt | None, verification: Any,
                        terminal_failure: dict | None, reason: str = "") -> OutcomeDecision:
    from .outcome_admission import task_reward_from_verification

    if receipt is not None and receipt.runtime_owned:
        if verification is None:
            return OutcomeDecision(
                "scoring_pending", "submitted_answer", reason or "verification_unavailable",
                receipt.receipt_ref, training_exclusion_reasons=("scoring_pending",),
                _authority=_AUTHORITY,
            )
        reward, _ = task_reward_from_verification(
            receipt.dataset, verification, prediction=receipt.submitted_answer_snapshot
        )
        return OutcomeDecision(
            "scored", "submitted_answer", reason or "accepted_director_finish",
            receipt.receipt_ref, True, verification, reward,
            training_exclusion_reasons=("policy_data_not_checked",), _authority=_AUTHORITY,
        )
    if terminal_failure is not None:
        return OutcomeDecision(
            "policy_failure", "runtime_terminal_ledger", terminal_failure["code"],
            score_known=True, task_reward=0.0,
            training_exclusion_reasons=("policy_data_not_checked",), _authority=_AUTHORITY,
        )
    return OutcomeDecision(
        "unsubmitted_unknown", "runtime_submission_gate", reason or "no_accepted_finish",
        training_exclusion_reasons=("unsubmitted_unknown",), _authority=_AUTHORITY,
    )


def validate_primary_training_outcome(metadata: dict[str, Any]) -> None:
    """Assert the serialized primary reward source again at learning boundaries."""
    if (metadata.get("submission_contract_version") != "unified_submission"
            or not is_text_submission_dataset(metadata.get("dataset", ""))
            or metadata.get("training_eligible") is False):
        return
    outcome = metadata.get("outcome_decision") or {}
    if not metadata.get("reward_known") or not outcome.get("score_known"):
        raise ValueError("unsubmitted/unknown text outcome cannot enter training")
    if outcome.get("status") == "policy_failure":
        if (outcome.get("source") != "runtime_terminal_ledger"
                or metadata.get("task_reward") != 0.0
                or not metadata.get("typed_policy_failure")
                or metadata.get("answer_reward_released")):
            raise ValueError("text policy failure must retain its runtime-owned zero reward")
        return
    receipt = metadata.get("submission_receipt") or {}
    if (outcome.get("status") != "scored" or outcome.get("source") != "submitted_answer"
            or receipt.get("version") != "unified_submission"
            or receipt.get("completion_source") != "director_finish"
            or outcome.get("receipt_ref") != snapshot_hash(receipt)
            or receipt.get("answer_hash") != answer_hash(str(metadata.get("submitted_answer", "")))
            or metadata.get("submission_status") != "submitted"):
        raise ValueError("text answer reward lacks a matching FINISH submission")
    accepted = [event.get("payload", {}) for event in metadata.get("solver_trace", {}).get("events", [])
                if event.get("payload", {}).get("event_id") == receipt.get("accepted_event_id")]
    if (len(accepted) != 1 or not accepted[0].get("accepted")
            or accepted[0].get("protocol_recovery")
            or accepted[0].get("submission_receipt") != receipt):
        raise ValueError("text answer reward lacks its accepted submission event")
