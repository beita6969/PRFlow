from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RelationType(StrEnum):
    DIRECTED = "directed"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class AgentNode:
    """An Agent with a scoped delegation prompt."""

    agent_id: str
    prompt: str = ""
    layer: int = 0
    allowed_tools: tuple[str, ...] = ()
    operation_policy_configured: bool = False
    initial_tool_budget: int = 0
    revision_tool_budget: int = 0
    total_tool_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return self.prompt_configured and (
            not self.metadata.get("model_selection_required", False)
            or bool(str(self.metadata.get("runtime_route", "")).strip())
        )

    @property
    def prompt_configured(self) -> bool:
        return bool(self.prompt.strip())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_tools"] = list(self.allowed_tools)
        return payload


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation_type: RelationType

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation_type.value,
        }


@dataclass
class AgentArtifact:
    """Structured agent output adapted from MANTA's ArtifactRecord contract."""

    artifact_id: str
    agent_id: str
    answer: str
    summary: str = ""
    confidence: float = 0.5
    # `claimed_confidence` preserves the model-authored value while `confidence`
    # is the runtime-enforced effective value used by routing and control.
    claimed_confidence: float | None = None
    unresolved_issues: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tool_summary: list[str] = field(default_factory=list)
    # Model-authored tool prose is kept separate from the runtime-owned Action
    # ledger so downstream components never mistake a claim for an Observation.
    model_tool_summary: list[str] = field(default_factory=list)
    runtime_tool_evidence: dict[str, Any] = field(default_factory=dict)
    integrity_risks: list[str] = field(default_factory=list)
    # Runtime-owned ALFWorld progress evidence. This remains separate from the
    # environment's official outcome: it records a model-policy semantic stall
    # only from public state/action observations.
    alfworld_progress: dict[str, Any] = field(default_factory=dict)
    react_trace: list[dict[str, Any]] = field(default_factory=list)
    source_artifact_ids: list[str] = field(default_factory=list)
    raw_response: str = ""
    revision: bool = False
    token_in: int = 0
    token_out: int = 0
    model: str = ""
    model_route: str = ""
    protocol_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    # Local control-plane telemetry only. It is persisted for route diagnosis but
    # deliberately omitted from RelayPacket and all subsequent model prompts.
    backend_request_events: list[dict[str, Any]] = field(default_factory=list)
    # Runtime-owned environment evidence. Model output parsing never populates it.
    environment_result: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.claimed_confidence = (
            self.confidence
            if self.claimed_confidence is None
            else max(0.0, min(1.0, float(self.claimed_confidence)))
        )
        if not self.summary:
            self.summary = self.answer[:400]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_model_text(
        cls,
        *,
        text: str,
        artifact_id: str,
        agent_id: str,
        source_artifact_ids: list[str] | None = None,
        revision: bool = False,
        token_in: int = 0,
        token_out: int = 0,
        model: str = "",
        model_route: str = "",
        validated_payload: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        payload = validated_payload if validated_payload is not None else _extract_json_object(text)
        if payload is None:
            return cls(
                artifact_id=artifact_id,
                agent_id=agent_id,
                answer=text.strip(),
                summary=text.strip()[:400],
                raw_response=text,
                source_artifact_ids=list(source_artifact_ids or []),
                revision=revision,
                token_in=token_in,
                token_out=token_out,
                model=model,
                model_route=model_route,
            )
        answer = payload.get("answer", payload.get("answer_artifact", ""))
        if isinstance(answer, (dict, list)):
            answer = json.dumps(answer, ensure_ascii=False, sort_keys=True)
        claimed_confidence = _coerce_confidence(payload.get("confidence"))
        model_tool_summary = _coerce_string_list(payload.get("tool_summary"))
        return cls(
            artifact_id=artifact_id,
            agent_id=agent_id,
            answer=str(answer if answer is not None else "").strip(),
            summary=str(payload.get("summary", "")).strip(),
            confidence=claimed_confidence,
            claimed_confidence=claimed_confidence,
            unresolved_issues=_coerce_string_list(payload.get("unresolved_issues")),
            evidence=_coerce_string_list(payload.get("evidence", payload.get("evidence_summary"))),
            tool_summary=list(model_tool_summary),
            model_tool_summary=model_tool_summary,
            source_artifact_ids=list(source_artifact_ids or []),
            raw_response=text,
            revision=revision,
            token_in=token_in,
            token_out=token_out,
            model=model,
            model_route=model_route,
        )


@dataclass(frozen=True)
class RelayPacket:
    """Bounded structured communication adapted from MANTA's RelayPacket."""

    message_id: str
    sender: str
    recipients: tuple[str, ...]
    artifact_id: str
    answer: str
    summary: str
    confidence: float
    claimed_confidence: float | None = None
    unresolved_issues: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    tool_summary: tuple[str, ...] = ()
    runtime_tool_evidence: dict[str, Any] = field(default_factory=dict)
    integrity_risks: tuple[str, ...] = ()
    alfworld_progress: dict[str, Any] = field(default_factory=dict)
    phase: str = "final"

    @classmethod
    def from_artifact(
        cls,
        artifact: AgentArtifact,
        *,
        recipients: list[str],
        message_id: str,
        phase: str = "final",
    ) -> RelayPacket:
        return cls(
            message_id=message_id,
            sender=artifact.agent_id,
            recipients=tuple(recipients),
            artifact_id=artifact.artifact_id,
            answer=artifact.answer,
            summary=artifact.summary,
            confidence=artifact.confidence,
            claimed_confidence=artifact.claimed_confidence,
            unresolved_issues=tuple(artifact.unresolved_issues),
            evidence=tuple(artifact.evidence),
            tool_summary=tuple(artifact.tool_summary),
            runtime_tool_evidence=dict(artifact.runtime_tool_evidence),
            integrity_risks=tuple(artifact.integrity_risks),
            alfworld_progress=dict(artifact.alfworld_progress),
            phase=phase,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recipients"] = list(self.recipients)
        payload["unresolved_issues"] = list(self.unresolved_issues)
        payload["evidence"] = list(self.evidence)
        payload["tool_summary"] = list(self.tool_summary)
        payload["integrity_risks"] = list(self.integrity_risks)
        return payload


@dataclass
class ExecutionReport:
    artifacts: dict[str, AgentArtifact] = field(default_factory=dict)
    packets: list[RelayPacket] = field(default_factory=list)
    scheduled_agents: list[str] = field(default_factory=list)
    executed_agents: list[str] = field(default_factory=list)
    reused_agents: list[str] = field(default_factory=list)
    cache_reused_agents: list[str] = field(default_factory=list)
    skipped_clean_agents: list[str] = field(default_factory=list)
    invalidation_reasons: dict[str, list[str]] = field(default_factory=dict)
    execution_events: list[dict[str, Any]] = field(default_factory=list)
    attempt_artifacts: list[AgentArtifact] = field(default_factory=list)
    initial_model_calls: int = 0
    revision_model_calls: int = 0
    cache_hits: int = 0
    component_execution_count: int = 0
    worker_model_calls_total: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    output: str = ""
    token_in: int = 0
    token_out: int = 0
    revision_decisions: list[dict[str, Any]] = field(default_factory=list)
    revision_skipped_agents: list[str] = field(default_factory=list)
    revision_wave_count: int = 0
    initial_cache_hits: int = 0
    revision_cache_hits: int = 0
    mandatory_revision_calls: int = 0
    incomplete_bidirectional_components: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {key: value.to_dict() for key, value in self.artifacts.items()},
            "packets": [packet.to_dict() for packet in self.packets],
            "scheduled_agents": list(self.scheduled_agents),
            "executed_agents": list(self.executed_agents),
            "reused_agents": list(self.reused_agents),
            "cache_reused_agents": list(self.cache_reused_agents),
            "skipped_clean_agents": list(self.skipped_clean_agents),
            "invalidation_reasons": {
                key: list(value) for key, value in self.invalidation_reasons.items()
            },
            "execution_events": [dict(value) for value in self.execution_events],
            "attempt_artifacts": [value.to_dict() for value in self.attempt_artifacts],
            "worker_usage_schema": "worker_execution_usage",
            "initial_model_calls": self.initial_model_calls,
            "revision_model_calls": self.revision_model_calls,
            "cache_hits": self.cache_hits,
            "component_execution_count": self.component_execution_count,
            "worker_model_calls_total": self.worker_model_calls_total,
            "errors": dict(self.errors),
            "output": self.output,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "revision_decisions": [dict(value) for value in self.revision_decisions],
            "revision_skipped_agents": list(self.revision_skipped_agents),
            "revision_wave_count": self.revision_wave_count,
            "initial_cache_hits": self.initial_cache_hits,
            "revision_cache_hits": self.revision_cache_hits,
            "mandatory_revision_calls": self.mandatory_revision_calls,
            "incomplete_bidirectional_components": [
                dict(value) for value in self.incomplete_bidirectional_components
            ],
        }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        return payload

    # Some reasoning models wrap the requested JSON in <think> text, which can
    # itself contain example JSON objects. Decode every balanced object and
    # prefer the last artifact-shaped payload instead of slicing from the first
    # opening brace to the final closing brace.
    decoder = json.JSONDecoder()
    decoded: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            candidate, _ = decoder.raw_decode(stripped[match.start() :])
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict):
            decoded.append(candidate)
    artifacts = [
        candidate
        for candidate in decoded
        if "answer" in candidate or "answer_artifact" in candidate
    ]
    if artifacts:
        return artifacts[-1]
    return decoded[-1] if decoded else None


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in raw if str(item).strip()][:8]
