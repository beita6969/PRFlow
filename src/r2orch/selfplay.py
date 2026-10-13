from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .config import canonical_dataset_name
from .graph import MultiAgentGraph
from .observability import TaskSpec
from .rollouts import TokenizedDirectorTrajectory, TokenizedPolicyCall, TrainingBatch


@dataclass(frozen=True)
class ProposedTask:
    task: TaskSpec
    response: str
    token_ids: tuple[int, ...] = ()
    action_mask: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()


@dataclass(frozen=True)
class SelfPlaySeed:
    """A Proposer anchor: one pool task prompt with its pool id and metadata."""

    content: str
    seed_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("self-play seed content must not be empty")


def _selection_prompt_preview(
    prompt: str, metadata: dict[str, Any], *, dataset: str, max_chars: int
) -> str:
    """Keep the public question visible when evidence precedes it."""
    if canonical_dataset_name(dataset) == "hotpotqa" and prompt.startswith(
        "Based on the following passages, answer the question."
    ):
        context, separator, question = prompt.rpartition("\nQuestion:")
        if separator and question.strip():
            # FlowSteer places the question after all passages. Reorder only
            # this selector view; the immutable Solver task keeps full context.
            context = context.partition("\n")[2].strip()
            prompt = f"Question: {question.strip()}\n\nContext preview:\n{context}"
    question = metadata.get("original_question")
    if (
        canonical_dataset_name(dataset) == "nq_open"
        and isinstance(question, str)
        and question.strip()
    ):
        prompt = question.strip()
    return prompt[:max_chars]


@dataclass(frozen=True)
class SolverRollout:
    trajectory: TokenizedDirectorTrajectory
    graph: MultiAgentGraph


@dataclass(frozen=True)
class FrontierScore:
    task_id: str
    validity: float
    graph_local: float
    rewards: tuple[float, ...]
    stable_graph_local: float | None = None
    reverify_status: str = "not_triggered"
    pair_stability: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SelfPlayPhase(StrEnum):
    PROPOSER_COLLECTION = "proposer_collection"
    SOLVER_COLLECTION = "solver_collection"
    BATCH_READY = "batch_ready"


@dataclass
class AlternatingSnapshots:
    proposer_snapshot: str
    solver_snapshot: str
    phase: SelfPlayPhase = SelfPlayPhase.PROPOSER_COLLECTION
    cycle: int = 0

    def advance(self) -> SelfPlayPhase:
        if self.phase is SelfPlayPhase.PROPOSER_COLLECTION:
            self.phase = SelfPlayPhase.SOLVER_COLLECTION
        elif self.phase is SelfPlayPhase.SOLVER_COLLECTION:
            self.phase = SelfPlayPhase.BATCH_READY
        else:
            self.cycle += 1
            self.phase = SelfPlayPhase.PROPOSER_COLLECTION
        return self.phase


@dataclass(frozen=True)
class DryRunSelfPlayResult:
    tasks: tuple[TaskSpec, ...]
    frontier_scores: tuple[FrontierScore, ...]
    proposer_batch: TrainingBatch
    solver_batch: TrainingBatch
    snapshots: dict[str, Any]
    optimizer_steps: int = 0
    proposal_extraction: dict[str, Any] = field(default_factory=dict)
