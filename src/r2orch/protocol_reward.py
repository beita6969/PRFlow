from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .graph import MultiAgentGraph

PROTOCOL_REWARD_VERSION = "protocol_reward"


@dataclass(frozen=True)
class ProtocolReward:
    version: str
    delegation_complete: bool
    delegation_fidelity: bool
    delegation_issues: tuple[str, ...]
    graph_complete: bool
    finish_complete: bool
    execution_complete: bool
    protocol_score: float
    qualified: bool
    answer_reward_released: bool
    answer_score: float
    reward: float
    excluded_external_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("delegation_fidelity", None)
        payload.pop("delegation_issues", None)
        return payload


def calculate_director_reward(
    *,
    graph: dict[str, Any],
    events: list[Any] | tuple[Any, ...],
    finished: bool,
    output: str,
    answer_score: float,
    task: str = "",
    submission_valid: bool = True,
    worker_backend_failure: bool = False,
    submission_receipt: dict[str, Any] | None = None,
) -> ProtocolReward:
    score = float(answer_score)
    if not math.isfinite(score):
        raise ValueError("answer_score must be finite")

    components = _protocol_components(
        graph=graph,
        events=events,
        finished=finished,
        output=output,
        submission_valid=submission_valid,
        worker_backend_failure=worker_backend_failure,
        submission_receipt=submission_receipt,
    )
    released = bool(finished and not worker_backend_failure)
    reward = 0.0 if worker_backend_failure else (score if released else -1.0)
    return ProtocolReward(
        version=PROTOCOL_REWARD_VERSION,
        **components,
        answer_reward_released=released,
        answer_score=score,
        reward=reward,
        excluded_external_failure=bool(worker_backend_failure),
    )


def _protocol_components(
    *,
    graph: dict[str, Any],
    events: list[Any] | tuple[Any, ...],
    finished: bool,
    output: str,
    submission_valid: bool,
    worker_backend_failure: bool,
    submission_receipt: dict[str, Any] | None = None,
) -> dict[str, bool | float]:
    parsed = MultiAgentGraph.from_dict(graph)
    configured = bool(parsed.nodes) and all(node.configured for node in parsed.nodes.values())
    delegation_complete = configured
    graph_complete = not parsed.validate(final=True)
    final_events = [event for event in events if _event_value(event, "final_execution")]
    receipt = submission_receipt or {}
    final_events = [
        event for event in final_events
        if receipt and _event_value(event, "accepted")
        and not _event_value(event, "protocol_recovery")
        and _event_value(event, "event_id") == receipt.get("accepted_event_id")
        and _event_value(event, "director_call_id") == receipt.get("director_call_id")
        and _event_value(event, "submission_receipt") == receipt
    ]
    finish_complete = len(final_events) == 1
    final_event = final_events[0] if finish_complete else None
    execution_complete = bool(
        finished
        and finish_complete
        and _event_value(final_event, "accepted")
        and (_event_value(final_event, "execution") is not None
             or bool(submission_receipt))
        and str(output or "").strip()
        and not _is_failure_output(output)
        and submission_valid
        and not worker_backend_failure
    )
    checks = (
        delegation_complete,
        graph_complete,
        finish_complete,
        execution_complete,
    )
    protocol_score = sum(int(value) for value in checks) / len(checks)
    return {
        "delegation_complete": delegation_complete,
        "delegation_fidelity": configured,
        "delegation_issues": (),
        "graph_complete": graph_complete,
        "finish_complete": finish_complete,
        "execution_complete": execution_complete,
        "protocol_score": protocol_score,
        "qualified": all(checks),
    }


def _event_value(event: Any, key: str) -> Any:
    if event is None:
        return None
    if isinstance(event, dict):
        payload = event.get("payload", event)
        return payload.get(key) if isinstance(payload, dict) else None
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(event, key, None)


def _is_failure_output(output: str) -> bool:
    normalized = str(output or "").strip().upper()
    return normalized in {
        "WORKER_BACKEND_FAILURE",
        "WORKER_PROTOCOL_FAILURE",
    }
