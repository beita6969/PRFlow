from __future__ import annotations

import json
import math
import re
import string
import threading
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .aime_submission import is_aime_dataset, parse_aime_answer
from .canvas import GraphCanvas
from .graph import MultiAgentGraph


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    reference: Any = None
    task_type: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    private_verifier_payload: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        private_payload = dict(self.private_verifier_payload)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "private_verifier_payload", private_payload)


def task_to_public_dict(task: TaskSpec) -> dict[str, Any]:
    """Serialize a TaskSpec without its verifier-only payload."""

    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "reference": task.reference,
        "task_type": task.task_type,
        "metadata": dict(task.metadata),
    }


@dataclass(frozen=True)
class VerificationResult:
    score: float
    passed: bool
    verifier: str
    detail: str = ""

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("verification score must be finite")
        object.__setattr__(self, "score", score)


class Verifier(Protocol):
    name: str

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult: ...


class ExactMatchVerifier:
    name = "exact_match"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        references = (
            task.reference if isinstance(task.reference, (list, tuple, set)) else [task.reference]
        )
        expected = [
            _normalize_answer("" if reference is None else str(reference))
            for reference in references
        ]
        actual_candidates = _normalized_qa_candidates(prediction)
        passed = any(actual and actual in expected for actual in actual_candidates)
        return VerificationResult(
            float(passed),
            passed,
            self.name,
            f"actual_candidates={actual_candidates!r}",
        )


class MultiAnswerExactMatchVerifier(ExactMatchVerifier):
    """Match one prediction against dataset-provided aliases, without adding aliases."""

    name = "multi_answer_exact_match"


class FlowSteerQAVerifier:
    """FlowSteer's ``eval_only.py`` QA metric for NQ-open and HotpotQA.

    The score is the bucketed QA reward; evaluation passes at token-F1 >= 0.5.
    Both values are retained in the detail string. Official answer EM/F1 are
    computed independently by the Solver audit and do not replace this reward.
    """

    name = "flowsteer_qa"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        references = (
            task.reference if isinstance(task.reference, (list, tuple, set)) else [task.reference]
        )
        actual_candidates = _normalized_qa_candidates(prediction)
        best = 0.0
        best_reward = 0.0
        for actual in actual_candidates:
            for reference in references:
                expected = _normalize_answer("" if reference is None else str(reference))
                if not actual or not expected:
                    continue
                if actual == expected:
                    best_reward = max(best_reward, 1.0)
                elif actual in expected or expected in actual:
                    best_reward = max(best_reward, 0.7)
                predicted_tokens = set(actual.split())
                expected_tokens = set(expected.split())
                common = len(predicted_tokens & expected_tokens)
                if common:
                    precision = common / len(predicted_tokens)
                    recall = common / len(expected_tokens)
                    f1 = 2.0 * precision * recall / (precision + recall)
                    best = max(best, f1)
                    if f1 >= 0.8:
                        best_reward = max(best_reward, 1.0)
                    elif f1 >= 0.5:
                        best_reward = max(best_reward, 0.7)
                    elif f1 >= 0.3:
                        best_reward = max(best_reward, 0.4)
                    elif f1 >= 0.1:
                        best_reward = max(best_reward, 0.2)
        passed = best >= 0.5
        return VerificationResult(
            best_reward,
            passed,
            self.name,
            f"token_f1={best:.12g}; flowsteer_reward={best_reward:.12g}; "
            f"eval_pass_f1_ge_0.5={int(passed)}",
        )


class TokenF1Verifier:
    """Official SQuAD/HotpotQA answer Token-F1 over dataset aliases."""

    name = "token_f1"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        references = (
            task.reference if isinstance(task.reference, (list, tuple, set)) else [task.reference]
        )
        predicted = _normalize_answer(prediction).split()
        best = 0.0
        for reference in references:
            expected = _normalize_answer("" if reference is None else str(reference)).split()
            if not predicted or not expected:
                continue
            common = sum((Counter(predicted) & Counter(expected)).values())
            if not common:
                continue
            precision = common / len(predicted)
            recall = common / len(expected)
            best = max(best, 2.0 * precision * recall / (precision + recall))
        passed = math.isclose(best, 1.0, rel_tol=0.0, abs_tol=1e-12)
        return VerificationResult(best, passed, self.name, f"token_f1={best:.12g}")


class NumericVerifier:
    name = "numeric"

    def __init__(self, *, absolute_tolerance: float = 1e-6) -> None:
        self.absolute_tolerance = absolute_tolerance

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        expected = _final_number(str("" if task.reference is None else task.reference))
        dataset = str(task.metadata.get("dataset", "")).casefold()
        if is_aime_dataset(dataset):
            parsed = parse_aime_answer(prediction)
            if not parsed.valid:
                return VerificationResult(
                    0.0, False, self.name, f"invalid_answer_submission:{parsed.reason}"
                )
            actual = Decimal(parsed.answer)
            passed = (
                expected is not None
                and actual is not None
                and expected == actual
                and expected == expected.to_integral_value()
                and Decimal(0) <= actual <= Decimal(999)
            )
        else:
            actual = _final_number(prediction)
            passed = (
                expected is not None
                and actual is not None
                and math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=0.0,
                    abs_tol=self.absolute_tolerance,
                )
            )
        return VerificationResult(float(passed), passed, self.name, f"actual={actual!r}")


class MultipleChoiceVerifier:
    name = "multiple_choice"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        expected = _choice(str("" if task.reference is None else task.reference))
        actual = _choice(prediction)
        passed = expected is not None and actual == expected
        return VerificationResult(float(passed), passed, self.name, f"actual={actual!r}")


class AutoVerifier:
    """Choose a deterministic verifier from per-task metadata and task type."""

    name = "auto"

    def __init__(
        self,
        adapters: Mapping[str, Verifier] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.adapters = {
            str(name).strip().casefold(): verifier for name, verifier in (adapters or {}).items()
        }
        self.aliases = {
            str(name).strip().casefold(): str(target).strip().casefold()
            for name, target in (aliases or {}).items()
        }

    def resolve(self, task: TaskSpec) -> Verifier:
        """Resolve the concrete verifier for a task.

        Capability checks must use the concrete adapter.  Treating the outer
        ``AutoVerifier`` as universally capable of scoring arbitrary text can
        accidentally enable intermediate rewards for stateful, final-only
        environments.
        """

        requested = str(task.metadata.get("verifier", "")).strip().casefold()
        if not requested:
            task_type = task.task_type.casefold()
            if any(value in task_type for value in ("math", "numeric", "number")):
                requested = "numeric"
            elif any(value in task_type for value in ("multiple_choice", "choice", "mcq")):
                requested = "multiple_choice"
            else:
                requested = "exact_match"
        requested = self.aliases.get(requested, requested)
        verifier: Verifier | None = {
            "exact_match": ExactMatchVerifier(),
            "multi_answer_exact_match": MultiAnswerExactMatchVerifier(),
            "flowsteer_qa": FlowSteerQAVerifier(),
            "token_f1": TokenF1Verifier(),
            "numeric": NumericVerifier(),
            "multiple_choice": MultipleChoiceVerifier(),
            **self.adapters,
        }.get(requested)
        if verifier is None:
            raise ValueError(
                f"verifier {requested!r} requires a dataset-specific environment adapter"
            )
        return verifier

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        return self.resolve(task).verify(task, prediction)

    def score_intermediate(self, task: TaskSpec, prediction: str) -> float:
        verifier = self.resolve(task)
        if not bool(getattr(verifier, "supports_intermediate_scoring", True)):
            raise ValueError(f"verifier {verifier.name!r} does not support intermediate scoring")
        scorer = getattr(verifier, "score_intermediate", None)
        if callable(scorer):
            return float(scorer(task, prediction))
        return float(verifier.verify(task, prediction).score)


def task_requires_reference(task: TaskSpec) -> bool:
    """Only answer-comparison verifiers require a textual reference answer."""

    requested = str(task.metadata.get("verifier", "")).strip().casefold()
    if requested:
        return requested in {
            "exact_match",
            "multi_answer_exact_match",
            "flowsteer_qa",
            "token_f1",
            "numeric",
            "multiple_choice",
        }
    task_type = task.task_type.casefold()
    return not any(value in task_type for value in ("alfworld", "environment"))


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    kind: str
    payload: dict[str, Any]


@dataclass
class ExecutionTrace:
    run_id: str
    task: TaskSpec
    events: list[TraceEvent]
    final_graph: dict[str, Any]
    output: str = ""
    verification: VerificationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": task_to_public_dict(self.task),
            "events": [asdict(event) for event in self.events],
            "final_graph": self.final_graph,
            "output": self.output,
            "verification": asdict(self.verification) if self.verification else None,
        }


def trace_from_canvas(
    *,
    run_id: str,
    task: TaskSpec,
    canvas: GraphCanvas,
    verification: VerificationResult | None = None,
) -> ExecutionTrace:
    events: list[TraceEvent] = []
    director_turn_index = 0
    graph_before = MultiAgentGraph(max_agents=canvas.graph.max_agents).to_dict()
    for sequence, step in enumerate(canvas.history):
        # A protocol-recovery step is an environment mutation, not a sampled
        # Director action.  Persist the mapping explicitly so Canvas event
        # sequence numbers are never confused with trainable model turns.
        model_turn = None if step.protocol_recovery else director_turn_index
        if model_turn is not None:
            director_turn_index += 1
        events.append(
            TraceEvent(
                sequence=sequence,
                kind="canvas_step",
                payload={
                    "raw_action": step.action.raw_text,
                    "event_id": step.event_id,
                    "director_call_id": step.director_call_id,
                    "submission_receipt": step.submission_receipt,
                    "accepted": step.accepted,
                    "feedback": step.feedback,
                    "graph_version": step.graph.get("version"),
                    "graph_before": graph_before,
                    "graph": step.graph,
                    "dirty_agents": list(step.dirty_agents),
                    "invalidated_agents": list(step.invalidated_agents),
                    "scheduled_agents": list(step.scheduled_agents),
                    "executed_agents": list(step.executed_agents),
                    "reused_agents": list(step.reused_agents),
                    "remaining_dirty": list(step.remaining_dirty),
                    "invalidation_reasons": {
                        agent_id: list(reasons)
                        for agent_id, reasons in step.invalidation_reasons.items()
                    },
                    "prompt_revision": dict(step.prompt_revision),
                    "execution": step.execution.to_dict() if step.execution else None,
                    "rejection_code": step.rejection_code,
                    "protocol_recovery": step.protocol_recovery,
                    "director_turn_index": model_turn,
                    "final_execution": step.final_execution,
                    "topology_audit": dict(step.topology_audit),
                    "structural_repair": dict(step.structural_repair),
                    "responsibility_issue": dict(step.responsibility_issue),
                    "responsibility_overlap_check": dict(step.responsibility_overlap_check),
                    "delegation_field_repairs": list(step.delegation_field_repairs),
                    "time_admission": dict(step.time_admission),
                    "token_admission": dict(step.token_admission),
                    "control_snapshot": dict(step.control_snapshot),
                    "rejection_details": dict(step.rejection_details),
                    "relation_decision": dict(step.relation_decision),
                    "invalid_repeat_count": step.invalid_repeat_count,
                    "topology_edits_frozen": step.topology_edits_frozen,
                },
            )
        )
        graph_before = step.graph
    output = ""
    if canvas.graph.output_agent:
        artifact = canvas.runtime.artifacts.get(canvas.graph.output_agent)
        output = artifact.answer if artifact else ""
    from .submission_contract import is_text_submission_dataset

    if is_text_submission_dataset(canvas.dataset):
        output = (canvas.submission_receipt.submitted_answer_snapshot
                  if canvas.submission_receipt else "")
    return ExecutionTrace(
        run_id=run_id,
        task=task,
        events=events,
        final_graph=canvas.graph.to_dict(),
        output=output,
        verification=verification,
    )


class JSONLTraceStore:
    _append_lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: ExecutionTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def _normalize_answer(value: str) -> str:
    """SQuAD/NQ-style exact-match normalization with Unicode punctuation."""

    lowered = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if char in string.punctuation or unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return re.sub(r"\s+", " ", without_articles).strip()


def _normalized_qa_candidates(value: str) -> list[str]:
    """At most one reference-blind candidate."""
    from .qa_submission import extract_qa_answer

    candidate = _normalize_answer(extract_qa_answer(value))
    return [candidate] if candidate else []


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _final_number(value: str) -> Decimal | None:
    """Prefer an explicit final/boxed number, then fall back to the last number."""

    cleaned = value.replace(",", "")
    candidates: list[str] = []
    for pattern in (
        rf"\\boxed\s*\{{\s*({_NUMBER_PATTERN})\s*\}}",
        rf"(?:final\s+answer|answer|答案)\s*(?:is|=|:|：)?\s*({_NUMBER_PATTERN})",
    ):
        candidates.extend(re.findall(pattern, cleaned, flags=re.IGNORECASE))
    if not candidates:
        candidates = re.findall(_NUMBER_PATTERN, cleaned)
    if not candidates:
        return None
    try:
        return Decimal(candidates[-1])
    except InvalidOperation:
        return None


def _choice(value: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", value.upper())
    return matches[-1] if matches else None
