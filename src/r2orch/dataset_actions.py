from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observability import TaskSpec


class EnvironmentState(StrEnum):
    STATELESS = "stateless"
    STATEFUL = "stateful"


class SessionScope(StrEnum):
    NONE = "none"
    SHARED = "shared"


class ActionExecution(StrEnum):
    BATCH = "batch"
    SEQUENTIAL = "sequential"


class CommitPolicy(StrEnum):
    NONE = "none"


class CommitActivation(StrEnum):
    NONE = "none"


@dataclass(frozen=True)
class DatasetActionAdapter:
    """Dataset-owned Worker action visibility and budget policy.

    The Director never sees or edits this policy. The Canvas applies it to every
    Worker node created for the task, and the runtime enforces the resulting
    action names before dispatching to an executor.
    """

    adapter_id: str
    datasets: tuple[str, ...]
    action_names: tuple[str, ...]
    initial_action_budget: int
    revision_action_budget: int
    total_action_budget: int
    environment_state: EnvironmentState | str = EnvironmentState.STATELESS
    session_scope: SessionScope | str = SessionScope.NONE
    action_execution: ActionExecution | str = ActionExecution.BATCH
    commit_policy: CommitPolicy | str = CommitPolicy.NONE
    commit_activation: CommitActivation | str = CommitActivation.NONE
    supports_parallel_reads: bool = True

    def __post_init__(self) -> None:
        adapter_id = self.adapter_id.strip().casefold()
        datasets = tuple(dict.fromkeys(value.strip().casefold() for value in self.datasets))
        actions = tuple(dict.fromkeys(value.strip() for value in self.action_names))
        environment_state = EnvironmentState(self.environment_state)
        session_scope = SessionScope(self.session_scope)
        action_execution = ActionExecution(self.action_execution)
        commit_policy = CommitPolicy(self.commit_policy)
        commit_activation = CommitActivation(self.commit_activation)
        if not adapter_id:
            raise ValueError("dataset action adapter_id cannot be empty")
        if not datasets or any(not value for value in datasets):
            raise ValueError("dataset action adapter requires dataset names")
        if any(not value for value in actions):
            raise ValueError("dataset action names cannot be empty")
        budgets = (
            int(self.initial_action_budget),
            int(self.revision_action_budget),
            int(self.total_action_budget),
        )
        if min(budgets) < 0:
            raise ValueError("dataset action budgets must be non-negative")
        if budgets[0] + budgets[1] > budgets[2]:
            raise ValueError("dataset phase action budgets exceed the total budget")
        if actions and budgets[2] <= 0:
            raise ValueError("visible dataset actions require a positive total budget")
        if not actions and any(budgets):
            raise ValueError("an empty dataset action set requires zero budgets")
        if environment_state is EnvironmentState.STATELESS and (
            session_scope is not SessionScope.NONE
            or action_execution is not ActionExecution.BATCH
        ):
            raise ValueError("stateless adapters cannot declare sessions")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "action_names", actions)
        object.__setattr__(self, "environment_state", environment_state)
        object.__setattr__(self, "session_scope", session_scope)
        object.__setattr__(self, "action_execution", action_execution)
        object.__setattr__(self, "commit_policy", commit_policy)
        object.__setattr__(self, "commit_activation", commit_activation)

    def capability_policy(self) -> dict[str, object]:
        return {
            "environment_state": self.environment_state.value,
            "session_scope": self.session_scope.value,
            "action_execution": self.action_execution.value,
            "commit_policy": self.commit_policy.value,
            "commit_activation": self.commit_activation.value,
            "supports_parallel_reads": bool(self.supports_parallel_reads),
        }


class DatasetActionRegistry:
    """Resolve exactly one Action Adapter from trusted task dataset metadata."""

    def __init__(
        self,
        adapters: Iterable[DatasetActionAdapter] = (),
        *,
        available_actions: Iterable[str] = (),
    ) -> None:
        self.available_actions = frozenset(str(value) for value in available_actions)
        self._by_dataset: dict[str, DatasetActionAdapter] = {}
        self._by_id: dict[str, DatasetActionAdapter] = {}
        for adapter in adapters:
            if adapter.adapter_id in self._by_id:
                raise ValueError(f"duplicate dataset action adapter: {adapter.adapter_id}")
            unavailable = set(adapter.action_names) - self.available_actions
            if unavailable:
                raise ValueError(
                    f"adapter {adapter.adapter_id} references unavailable actions: "
                    + ", ".join(sorted(unavailable))
                )
            self._by_id[adapter.adapter_id] = adapter
            for dataset in adapter.datasets:
                if dataset in self._by_dataset:
                    raise ValueError(f"multiple action adapters registered for dataset {dataset}")
                self._by_dataset[dataset] = adapter

    def resolve(self, task: TaskSpec) -> DatasetActionAdapter | None:
        dataset = str(task.metadata.get("dataset", "")).strip().casefold()
        return self._by_dataset.get(dataset)

    def get(self, adapter_id: str) -> DatasetActionAdapter | None:
        return self._by_id.get(str(adapter_id).strip().casefold())


def default_dataset_action_registry(
    available_actions: Iterable[str],
    *,
    aime_budgets: tuple[int, int, int] = (3, 1, 4),
    retrieval_initial_budget: int = 3,
    hotpotqa_search_enabled: bool = False,
    alfworld_budgets: tuple[int, int, int] = (50, 50, 100),
) -> DatasetActionRegistry:
    """Build zero-action adapters and executable adapters whose actions exist."""

    available = tuple(dict.fromkeys(str(value) for value in available_actions))
    available_set = set(available)
    if hotpotqa_search_enabled and "search" not in available_set:
        raise ValueError("HotpotQA search baseline requires the search Action")
    adapters: list[DatasetActionAdapter] = [
        DatasetActionAdapter(
            adapter_id="hotpotqa_context",
            datasets=("hotpotqa_context",) if hotpotqa_search_enabled else ("hotpotqa",),
            action_names=(),
            initial_action_budget=0,
            revision_action_budget=0,
            total_action_budget=0,
            environment_state=EnvironmentState.STATELESS,
            session_scope=SessionScope.NONE,
            action_execution=ActionExecution.BATCH,
            commit_policy=CommitPolicy.NONE,
            commit_activation=CommitActivation.NONE,
            supports_parallel_reads=True,
        ),
    ]
    aime_actions = ("symbolic_compute", "finite_search", "python_exec")
    if set(aime_actions) <= available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="aime",
                datasets=("aime",),
                action_names=aime_actions,
                initial_action_budget=aime_budgets[0],
                revision_action_budget=aime_budgets[1],
                total_action_budget=aime_budgets[2],
            )
        )
    if "search" in available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="retrieval_qa",
                # MedQA searches wiki-18 like NQ-Open and HotpotQA.
                datasets=(
                    ("nq_open", "hotpotqa", "medqa") if hotpotqa_search_enabled
                    else ("nq_open", "medqa")
                ),
                action_names=("search",),
                initial_action_budget=retrieval_initial_budget,
                revision_action_budget=1,
                total_action_budget=retrieval_initial_budget + 1,
            )
        )
    if "python_exec" in available_set:
        # MBPP+: one stateless python tool.
        adapters.append(
            DatasetActionAdapter(
                adapter_id="mbppplus",
                datasets=("mbppplus",),
                action_names=("python_exec",),
                initial_action_budget=aime_budgets[0],
                revision_action_budget=aime_budgets[1],
                total_action_budget=aime_budgets[2],
            )
        )
    if "alfworld_step" in available_set:
        adapters.append(
            DatasetActionAdapter(
                adapter_id="alfworld",
                datasets=("alfworld",),
                action_names=("alfworld_step",),
                initial_action_budget=alfworld_budgets[0],
                revision_action_budget=alfworld_budgets[1],
                total_action_budget=alfworld_budgets[2],
                environment_state=EnvironmentState.STATEFUL,
                # One ALFWorld question owns one TextWorld episode.  Worker
                # nodes and revisions continue that same episode instead of
                # silently resetting a fresh scene per agent.
                session_scope=SessionScope.SHARED,
                action_execution=ActionExecution.SEQUENTIAL,
            )
        )
    return DatasetActionRegistry(
        adapters,
        available_actions=available,
    )
