from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .actions import (
    ActionParser,
    ActionType,
    CanvasAction,
    PromptRevisionBasis,
)
from .aime_submission import is_aime_dataset, parse_aime_answer
from .artifact_protocol import summarize_worker_protocol
from .config import CanvasConfig
from .contracts import ExecutionReport, RelationType
from .dataset_actions import DatasetActionAdapter
from .deadline import RolloutDeadline
from .delegation import (
    DelegationCompilation,
    DelegationValidationError,
    compare_responsibilities,
    compile_delegation,
    responsibility_signature,
)
from .graph import GraphValidationError, MultiAgentGraph, MutationResult
from .qa_submission import is_short_qa_dataset
from .runtime import (
    WORKER_BACKEND_FAILURE_SENTINEL,
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    MultiAgentRuntime,
    artifact_backend_failure_records,
)
from .submission_contract import (
    DirectorCallContext,
    SubmissionReceipt,
)


class CanvasState(StrEnum):
    """FlowSteer's edit state with an explicit prompt-completion barrier."""

    BUILDING = "building"
    AWAITING_PROMPT = "awaiting_prompt"
    AWAITING_MODEL = "awaiting_model"
    AWAITING_RELATION_CHOICE = "awaiting_relation_choice"
    FINISHED = "finished"
    FAILED = "failed"


class TokenBudgetExceeded(RuntimeError):
    """Terminal aggregate-cost failure retaining the committed execution report."""

    def __init__(self, report: ExecutionReport, total: int, limit: int) -> None:
        self.report = report
        self.total = int(total)
        self.limit = int(limit)
        super().__init__(f"token budget exceeded ({self.total}/{self.limit})")


class RequiredWorkerBackendFailure(RuntimeError):
    """Terminal Worker outage retaining the execution that exposed it."""

    def __init__(self, report: ExecutionReport) -> None:
        self.report = report
        failures = [
            artifact
            for artifact in report.artifacts.values()
            if artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
        ]
        self.agents = tuple(sorted(artifact.agent_id for artifact in failures))
        self.routes = tuple(sorted({artifact.model_route or "unassigned" for artifact in failures}))
        self.failure_details = tuple(
            record for artifact in failures for record in artifact_backend_failure_records(artifact)
        )
        self.failure_types = tuple(
            sorted({str(record.get("kind", "unknown")) for record in self.failure_details})
        )
        super().__init__(
            "required Worker backend unavailable "
            f"(agents={','.join(self.agents) or 'unknown'}; "
            f"routes={','.join(self.routes) or 'unknown'}; "
            f"failure_types={','.join(self.failure_types) or 'unknown'})"
        )


@dataclass
class CanvasStep:
    round_index: int
    action: CanvasAction
    accepted: bool
    active: bool
    feedback: str
    graph: dict[str, Any]
    dirty_agents: list[str] = field(default_factory=list)
    invalidated_agents: list[str] = field(default_factory=list)
    scheduled_agents: list[str] = field(default_factory=list)
    executed_agents: list[str] = field(default_factory=list)
    reused_agents: list[str] = field(default_factory=list)
    remaining_dirty: list[str] = field(default_factory=list)
    invalidation_reasons: dict[str, list[str]] = field(default_factory=dict)
    prompt_revision: dict[str, Any] = field(default_factory=dict)
    execution: ExecutionReport | None = None
    rejection_code: str | None = None
    protocol_recovery: bool = False
    final_execution: bool = False
    topology_audit: dict[str, Any] = field(default_factory=dict)
    structural_repair: dict[str, Any] = field(default_factory=dict)
    responsibility_issue: dict[str, Any] = field(default_factory=dict)
    responsibility_overlap_check: dict[str, Any] = field(default_factory=dict)
    delegation_field_repairs: list[dict[str, Any]] = field(default_factory=list)
    time_admission: dict[str, Any] = field(default_factory=dict)
    token_admission: dict[str, Any] = field(default_factory=dict)
    control_snapshot: dict[str, Any] = field(default_factory=dict)
    rejection_details: dict[str, Any] = field(default_factory=dict)
    relation_decision: dict[str, Any] = field(default_factory=dict)
    invalid_repeat_count: int = 0
    topology_edits_frozen: bool = False
    event_id: str = ""
    director_call_id: str = ""
    submission_receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "event_id": self.event_id,
            "director_call_id": self.director_call_id,
            "submission_receipt": self.submission_receipt,
            "action": self.action.to_dict(),
            "accepted": self.accepted,
            "active": self.active,
            "feedback": self.feedback,
            "graph": self.graph,
            "dirty_agents": list(self.dirty_agents),
            "invalidated_agents": list(self.invalidated_agents),
            "scheduled_agents": list(self.scheduled_agents),
            "executed_agents": list(self.executed_agents),
            "reused_agents": list(self.reused_agents),
            "remaining_dirty": list(self.remaining_dirty),
            "invalidation_reasons": {
                key: list(value) for key, value in self.invalidation_reasons.items()
            },
            "prompt_revision": dict(self.prompt_revision),
            "execution": self.execution.to_dict() if self.execution else None,
            "rejection_code": self.rejection_code,
            "protocol_recovery": self.protocol_recovery,
            "final_execution": self.final_execution,
            "topology_audit": dict(self.topology_audit),
            "structural_repair": dict(self.structural_repair),
            "responsibility_issue": dict(self.responsibility_issue),
            "responsibility_overlap_check": dict(self.responsibility_overlap_check),
            "delegation_field_repairs": list(self.delegation_field_repairs),
            "time_admission": dict(self.time_admission),
            "token_admission": dict(self.token_admission),
            "control_snapshot": dict(self.control_snapshot),
            "rejection_details": dict(self.rejection_details),
            "relation_decision": dict(self.relation_decision),
            "invalid_repeat_count": self.invalid_repeat_count,
            "topology_edits_frozen": self.topology_edits_frozen,
        }


@dataclass(frozen=True)
class RepairProgressSignature:
    """Version-independent measure of whether a repair became easier to finish."""

    reason: str
    output_health: str
    final_error_codes: tuple[str, ...]
    unconfigured_count: int
    unreachable_count: int
    dirty_count: int
    final_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "output_health": self.output_health,
            "final_error_codes": list(self.final_error_codes),
            "unconfigured_count": self.unconfigured_count,
            "unreachable_count": self.unreachable_count,
            "dirty_count": self.dirty_count,
            "final_ready": self.final_ready,
        }


@dataclass(frozen=True)
class PendingRelationDecision:
    source: str
    target: str
    relation_type: RelationType
    graph_version: int
    proposal_round_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "graph_version": self.graph_version,
            "proposal_round_index": self.proposal_round_index,
        }


from .unified_submission import UnifiedSubmissionMixin
from .unified_contract import PROTOCOL, RESULT_SCOPES


class GraphCanvas(UnifiedSubmissionMixin):
    """Transactional graph editor plus factual execution feedback.

    Invalid mutations never alter the accepted graph. Like FlowSteer, adding an
    agent enters a mandatory AWAITING_PROMPT state so empty placeholder agents
    cannot accumulate on the canvas.
    """

    def __init__(
        self,
        *,
        task: str,
        worker_task: str | None = None,
        director_task: str | None = None,
        runtime: MultiAgentRuntime,
        config: CanvasConfig | None = None,
        runtime_routes: tuple[str, ...] = (),
        task_type: str = "general",
        action_adapter: DatasetActionAdapter | None = None,
        dataset: str = "",
        rollout_deadline: RolloutDeadline | None = None,
    ) -> None:
        self.run_id = ""
        self.submission_receipt: SubmissionReceipt | None = None
        self._director_call_context: DirectorCallContext | None = None
        self.task = task
        self.worker_task = worker_task if worker_task is not None else task
        self.director_task = director_task if director_task is not None else task
        self.runtime = runtime
        self.config = config or CanvasConfig()
        self.parser = ActionParser()
        self._init_unified()
        self.runtime_routes = tuple(runtime_routes)
        self.task_type = str(task_type)
        self.action_adapter = action_adapter
        inferred_dataset = (
            action_adapter.datasets[0]
            if action_adapter is not None and len(action_adapter.datasets) == 1
            else ""
        )
        self.dataset = str(dataset or inferred_dataset).strip().casefold()
        self.rollout_deadline = rollout_deadline
        self.graph = MultiAgentGraph(max_agents=self.config.max_agents, runtime_routes=self.runtime_routes)
        self.state = CanvasState.BUILDING
        self.pending_agent_id: str | None = None
        self.pending_relation_decision: PendingRelationDecision | None = None
        self._last_relation_decision: dict[str, Any] = {}
        self._considered_relation_pairs: set[tuple[int, tuple[str, str]]] = set()
        self.dirty_agents: set[str] = set()
        self.dirty_reasons: dict[str, set[str]] = {}
        self._step_invalidated_agents: set[str] = set()
        self._step_scheduled_agents: set[str] = set()
        self._step_invalidation_reasons: dict[str, set[str]] = {}
        self._prompt_revision_checkpoints: dict[str, dict[str, Any]] = {}
        self.prompt_revision_counts: dict[str, int] = {}
        self._step_responsibility_overlap_check: dict[str, Any] = {}
        self.round_index = 0
        self.total_tokens = 0
        self.history: list[CanvasStep] = []
        self.structural_repair_reason: str | None = None
        self.structural_repair_agents: tuple[str, ...] = ()
        self.structural_repair_entries = 0
        self.structural_repair_resolutions = 0
        self.structural_repair_blocked_actions = 0
        self.finish_reachability_rejections = 0
        self.duplicate_agent_rejections = 0
        self.time_budget_rejections = 0
        self.token_budget_admission_rejections = 0
        self._token_admission_event: dict[str, Any] = {}
        self._repair_transition: str | None = None
        # Rejection fusion is keyed by the authoritative Canvas state, not by
        # the model-authored action. Different invalid actions against the
        # same unchanged state are one semantic no-progress streak.
        self._last_invalid_signature: str | None = None
        self._invalid_repeat_count = 0
        self.topology_edits_frozen = False
        self.repair_epoch = 0
        self.semantic_no_progress_streak = 0
        self.semantic_no_progress_recovery_count = 0
        self.output_switches_without_progress = 0
        self.output_lifecycle_recovery_count = 0
        self._last_repair_signature: RepairProgressSignature | None = None
        self._repair_entry_pending = False
        self._recent_repair_actions: list[dict[str, Any]] = []
        # Connectivity-repair state is tracked separately from output-lifecycle
        # recovery. Its history survives intermediate structural reasons (for
        # example a layer edit before the relation is added).
        self._connectivity_repair_active = False
        self._connectivity_repair_states: set[str] = set()
        self._connectivity_cycle_blocks = 0
        self._consolidation_output_locked = False
        self._output_selection_budget_remaining = self.config.output_selection_budget
        self._output_incumbent: dict[str, Any] = {}

    @property
    def director_round_limit(self) -> int:
        return self.config.max_rounds

    @property
    def active(self) -> bool:
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return False
        return self.round_index < self.director_round_limit

    def step(
        self,
        raw_action: str | CanvasAction,
        *,
        count_round: bool = True,
        authoritative_director: bool = False,
        director_context: DirectorCallContext | None = None,
    ) -> CanvasStep:
        self._director_call_context = director_context
        self._token_admission_event = {}
        self._step_invalidated_agents = set()
        self._step_scheduled_agents = set()
        self._step_invalidation_reasons = {}
        self._step_responsibility_overlap_check = {}
        action = (
            raw_action if isinstance(raw_action, CanvasAction) else self.parser.parse(raw_action)
        )
        self._restore_submission_lock()
        if self.rollout_deadline is not None:
            self.rollout_deadline.check("canvas_step_start")
        # Internal deterministic recovery actions do not consume Director
        # rounds and must remain possible after the model-turn budget is
        # exhausted. Terminal Canvas states are never reopened.
        if self.state in {CanvasState.FINISHED, CanvasState.FAILED} or (
            count_round
            and self.round_index >= self.director_round_limit
        ):
            return self._record(
                action,
                accepted=False,
                feedback="Canvas is not active.",
                rejection_code="canvas_inactive",
            )

        # Validate the admitted action against its pre-action budget. Charging
        # the final round must not invalidate FINISH/RUN_AGENT (or ADD_AGENT's
        # completion reserve). Post-action snapshots still use the charged
        # counter, and the entry guard above rejects any extra Director turn.
        action_snapshot = (
            self.control_snapshot()
            if authoritative_director and action.valid else None
        )
        if count_round:
            self.round_index += 1
        if (
            self.runtime_routes
            and action.action_type is ActionType.ADD_AGENT
            and self.config.max_rounds - self.round_index < 3
        ):
            return self._record(
                action,
                accepted=False,
                feedback="Insufficient Director turns for prompt, model, output and finish.",
                rejection_code="configuration_turn_budget_exhausted",
            )
        if not action.valid:
            if action.parse_error_code:
                return self._record(
                    action, accepted=False,
                    feedback=f"Rejected action encoding: {action.parse_error}",
                    rejection_code=action.parse_error_code,
                    rejection_details={"action_protocol_error": dict(action.parse_error_details)},
                )
            awaiting = self.state is CanvasState.AWAITING_PROMPT
            return self._record(
                action,
                accepted=False,
                feedback=f"Rejected action: {action.parse_error}",
                rejection_code=("responsibility_violation" if awaiting else "invalid_action"),
                responsibility_issue=(
                    {
                        "code": "prompt_action_schema",
                        "field": None,
                        "message": str(action.parse_error or "invalid action"),
                    }
                    if awaiting
                    else None
                ),
            )
        protected = self._protected_alfworld_candidates()
        if protected and (action.action_type is not ActionType.FINISH or action.target not in protected):
            return self._reject_graph_action(action, code="alfworld_terminal_candidate_protected",
                message="A current trusted successful episode is ready. Submit it with FINISH(target).",
                rejection_details={"targets": protected})
        if action.action_type is ActionType.SET_OUTPUT:
            return self._reject_graph_action(action, code="retired_action", message="Use result_scope in SET_PROMPT and FINISH(target).")
        if action.action_type is ActionType.SET_PROMPT and action.result_scope not in RESULT_SCOPES:
            return self._reject_graph_action(action, code="result_scope_required", message="SET_PROMPT requires result_scope: subtask or task_result.")
        if action.action_type in {ActionType.RUN_AGENT, ActionType.FINISH} and not action.target:
            return self._reject_graph_action(action, code="target_required", message="An explicit target is required.")
        if authoritative_director:
            # The Director chooses graph semantics, not concurrency metadata.
            # External Canvas callers retain the normal optimistic-version
            # check; internal Director actions always bind to current state.
            action.expected_version = self.graph.version
        if action.expected_version is not None and action.expected_version != self.graph.version:
            return self._reject_graph_action(
                action,
                code="stale_canvas_version",
                message=(
                    f"expected Canvas version {action.expected_version}, but current version is "
                    f"{self.graph.version}"
                ),
            )
        # FINISH may submit an existing result while abandoning an isolated draft.
        # Admission, provenance and transaction checks still run in _unified_finish.
        if (action.action_type is ActionType.FINISH
                and self._isolated_submission_draft(str(action.target))):
            return self._unified_finish(action)
        if self.state is CanvasState.AWAITING_MODEL and not (
            action.action_type is ActionType.SET_MODEL and action.target == self.pending_agent_id
        ) and not (
            action.action_type is ActionType.DELETE_AGENT and action.target == self.pending_agent_id
        ):
            return self._record(
                action,
                accepted=False,
                feedback=f"{self.pending_agent_id} requires SET_MODEL before another graph edit.",
                rejection_code="model_selection_required",
            )
        if self.state is CanvasState.AWAITING_RELATION_CHOICE:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: the pending relation proposal requires one constrained "
                    "off/on policy choice before another graph action."
                ),
                rejection_code="relation_choice_required",
                relation_decision=(
                    self.pending_relation_decision.to_dict()
                    if self.pending_relation_decision is not None
                    else {}
                ),
            )
        frozen_recovery_actions = {ActionType.SET_OUTPUT, ActionType.FINISH}
        frozen_recovery_actions.update({ActionType.RUN_AGENT, ActionType.SET_PROMPT, ActionType.SET_MODEL, ActionType.SET_LAYER, ActionType.CONSIDER_RELATION, ActionType.SET_RELATION, ActionType.REMOVE_RELATION, ActionType.DELETE_AGENT})
        if self.topology_edits_frozen and action.action_type not in frozen_recovery_actions:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: topology edits are frozen after repeated invalid "
                    "mutations; use one of the bounded recovery actions exposed "
                    "by the control snapshot."
                ),
                rejection_code="topology_edits_frozen",
                rejection_details=self._rejection_details("topology_edits_frozen"),
            )
        discards_pending_agent = bool(
            (
                self.state is CanvasState.AWAITING_PROMPT
                or self.state is CanvasState.AWAITING_MODEL
            )
            and action.action_type is ActionType.DELETE_AGENT
            and str(action.target or action.agent_id) == self.pending_agent_id
        )
        if (
            self.state is CanvasState.AWAITING_PROMPT
            and not self._completes_prompt(action)
            and not discards_pending_agent
        ):
            if (
                action.action_type is ActionType.SET_PROMPT
                and action.target not in self.graph.nodes
            ):
                return self._reject_graph_action(
                    action,
                    code="unknown_agent",
                    message=(
                        f"unknown agent: {action.target}; the pending runtime-allocated Agent is "
                        f"{self.pending_agent_id}"
                    ),
                )
            return self._record(
                action,
                accepted=False,
                feedback=(
                    f"Rejected action: {self.pending_agent_id} requires SET_PROMPT before "
                    "another graph edit."
                ),
                rejection_code="responsibility_violation",
                responsibility_issue={
                    "code": "prompt_order",
                    "field": None,
                    "message": (
                        f"{self.pending_agent_id} requires SET_PROMPT before another graph edit"
                    ),
                },
            )
        if (
            action.action_type is ActionType.ADD_AGENT
            and self.graph.nodes
            and self.config.max_total_tokens - self.total_tokens
            < self.config.graph_growth_token_reserve
        ):
            self._enter_structural_repair(
                "token_budget_consolidation",
                agents=tuple(self.graph.nodes),
            )
            self.structural_repair_blocked_actions += 1
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: STRUCTURAL_REPAIR_REQUIRED "
                    "(reason=token_budget_consolidation; remaining Worker tokens "
                    f"{self.config.max_total_tokens - self.total_tokens} are reserved "
                    "for selecting an existing output and FINISH)"
                ),
                rejection_code="token_budget_consolidation_required",
            )
        is_prompt_revision = bool(
            action.action_type is ActionType.SET_PROMPT
            and str(action.target) in self.graph.nodes
            and self.graph.nodes[str(action.target)].configured
        )
        if authoritative_director and not is_prompt_revision:
            legality_error = self._director_action_legality_error(action, snapshot=action_snapshot)
            if legality_error is not None:
                code, message = legality_error
                return self._reject_graph_action(action, code=code, message=message)
        if action.action_type is ActionType.CONSIDER_RELATION:
            return self._begin_relation_decision(action)
        if action.action_type is ActionType.RUN_AGENT:
            return self._unified_run(action)
        if action.action_type is ActionType.FINISH:
            return self._unified_finish(action)

        graph_before = self.graph.to_dict()
        graph_before.pop("version", None)
        candidate = self.graph.clone()
        before = set(candidate.nodes)
        rejection_code: str | None = None
        compilation: DelegationCompilation | None = None
        responsibility_issue: dict[str, Any] = {}
        try:
            if action.action_type is ActionType.SET_PROMPT:
                compilation = self._validate_delegation(action)
            mutation = self._apply(candidate, action, compilation=compilation)
            if not mutation.changed:
                field_repairs = (
                    [repair.to_dict() for repair in compilation.field_repairs]
                    if compilation is not None
                    else []
                )
                return self._reject_graph_action(
                    action,
                    code="no_effective_change",
                    message=(
                        f"{mutation.message}; the Canvas version and Worker inputs "
                        "were left unchanged"
                    ),
                    rejection_details={
                        "change_reason": mutation.change_reason,
                        "mutation_changed": False,
                    },
                    delegation_field_repairs=field_repairs,
                )
            candidate.assert_valid(final=False)
            # One accepted Canvas action is one committed topology version even
            # when its implementation performs several internal graph updates.
            candidate.version = self.graph.version + 1
        except (GraphValidationError, ValueError, TypeError) as exc:
            message = str(exc)
            if isinstance(exc, DelegationValidationError):
                responsibility_issue = {
                    "code": exc.issue.code,
                    "field": exc.issue.field,
                    "message": exc.issue.message,
                }
                if exc.issue.details:
                    responsibility_issue["details"] = dict(exc.issue.details)
            rejection_code = self._classify_mutation_error(action, message)
            if action.action_type is ActionType.ADD_AGENT and "already exists" in message:
                self.duplicate_agent_rejections += 1
                rejection_code = "duplicate_agent_id"
            return self._reject_graph_action(
                action,
                code=rejection_code,
                message=message,
                responsibility_issue=responsibility_issue,
            )

        new_agent = next(iter(set(candidate.nodes) - before), None)
        previous_graph = self.graph
        self.graph = candidate
        if action.action_type in self._DIRECTOR_EDIT_ACTIONS:
            # Charge the committed mutation even when its subsequent Worker
            # execution fails. Validation failures and no-ops never reach here.
            self.director_edits_used += 1
        if action.action_type is ActionType.DELETE_AGENT:
            self.runtime.discard_environment_candidate(str(action.target or action.agent_id))
        graph_after = self.graph.to_dict()
        graph_after.pop("version", None)
        if self.rollout_deadline is not None and graph_after != graph_before:
            self.rollout_deadline.mark_progress(f"canvas_{action.action_type.value}")
        invalidated = set(mutation.dirty_agents)
        self.dirty_agents.update(invalidated)
        self._step_invalidated_agents.update(invalidated)
        invalidation_reasons = self._mutation_invalidation_reasons(
            action,
            mutation,
            previous_graph=previous_graph,
            current_graph=self.graph,
        )
        for agent_id, reasons in invalidation_reasons.items():
            self.dirty_reasons.setdefault(agent_id, set()).update(reasons)
            self._step_invalidation_reasons.setdefault(agent_id, set()).update(reasons)
        self.dirty_reasons = {
            agent_id: reasons
            for agent_id, reasons in self.dirty_reasons.items()
            if agent_id in self.graph.nodes
        }
        if action.action_type is ActionType.ADD_AGENT:
            self.pending_agent_id = new_agent
            self.state = CanvasState.AWAITING_PROMPT
        elif (
            self._completes_prompt(action)
            and self.runtime_routes
            and not self.graph.nodes[str(action.target)].configured
        ):
            self.state = CanvasState.AWAITING_MODEL
        elif (
            self._completes_prompt(action)
            or discards_pending_agent
            or (
                self.state is CanvasState.AWAITING_MODEL
                and action.action_type is ActionType.SET_MODEL
                and action.target == self.pending_agent_id
            )
        ):
            self.pending_agent_id = None
            self.state = CanvasState.BUILDING

        self._refresh_structural_repair(action)

        field_repairs = (
            [repair.to_dict() for repair in compilation.field_repairs]
            if compilation is not None
            else []
        )
        report: ExecutionReport | None = None
        if action.action_type is not ActionType.ADD_AGENT:
            try:
                # DELETE_AGENT may leave no dirty configured nodes, but Runtime
                # still needs a zero-cost synchronization pass to drop deleted
                # artifacts.  Other edits execute only their dirty closure.
                report = self._execute_dirty(force=action.action_type is ActionType.DELETE_AGENT)
            except TokenBudgetExceeded as exc:
                report = exc.report
                self._finalize_prompt_assignment(action)
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    # The graph edit itself was legal and is already committed;
                    # only the cumulative execution budget ended the rollout.
                    accepted=True,
                    feedback=self._feedback(
                        f"Execution stopped after the accepted graph edit: {exc}",
                        report,
                    ),
                    execution=report,
                    rejection_code="execution_budget_exceeded",
                    delegation_field_repairs=field_repairs,
                )
            except RequiredWorkerBackendFailure as exc:
                report = exc.report
                self._finalize_prompt_assignment(action)
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    # The graph mutation and diagnostic execution are durable,
                    # but the outage must end the trajectory before the Director
                    # can react by adding unrelated Agents.
                    accepted=True,
                    feedback=self._feedback(
                        f"Execution stopped after the accepted graph edit: {exc}",
                        report,
                    ),
                    execution=report,
                    rejection_code="worker_backend_unavailable",
                    delegation_field_repairs=field_repairs,
                )
            except ValueError as exc:
                self._finalize_prompt_assignment(action)
                self.state = CanvasState.FAILED
                return self._record(
                    action,
                    accepted=True,
                    feedback=self._feedback(
                        f"Execution failed after the accepted graph edit: {exc}",
                        report,
                    ),
                    execution=report,
                    rejection_code="execution_failure",
                    delegation_field_repairs=field_repairs,
                )
        self._finalize_prompt_assignment(action)
        return self._record(
            action,
            accepted=True,
            feedback=self._feedback(mutation.message, report),
            execution=report,
            delegation_field_repairs=field_repairs,
        )

    def terminate_round_limit_without_graph_repair(self) -> CanvasStep | None:
        """Terminate an exhausted policy loop without editing the Director's graph."""

        if self.state in {CanvasState.FINISHED, CanvasState.FAILED}:
            return None
        if self.round_index < self.config.max_rounds:
            return None
        self.topology_edits_frozen = True
        return self._fail_terminal_recovery(
            code="max_rounds_exhausted",
            message="Director round budget ended before a final-valid graph was finished",
        )

    def terminate_context_limit_without_graph_repair(self) -> CanvasStep:
        """Record exact-context exhaustion without selecting output or inventing tokens."""
        return self._fail_terminal_recovery(
            code="director_context_budget_exhausted",
            message="exact Director history exhausted the 32768-token context budget",
        )

    def _fail_terminal_recovery(self, *, code: str, message: str) -> CanvasStep:
        self.state = CanvasState.FAILED
        return self._record(
            self.parser.parse(
                json.dumps({"action": "finish", "expected_version": self.graph.version})
            ),
            accepted=False,
            feedback=f"Deterministic terminal recovery failed: {message}.",
            rejection_code=code,
            protocol_recovery=True,
        )

    def selected_output_is_current(self) -> bool:
        """One submission gate shared by Director, finalization and scoring."""
        selected = self.graph.output_agent
        if not selected or selected not in self.runtime.artifacts:
            return False
        if self.submission_receipt is not None and self.state is CanvasState.FINISHED:
            return self.runtime.artifacts[selected].artifact_id == self.submission_receipt.artifact_id
        return selected not in self.dirty_agents and self.runtime.artifact_matches_current_input_signature(
            selected, task=self.worker_task, graph=self.graph
        )

    def _execute_dirty(self, *, force: bool = False) -> ExecutionReport | None:
        # A deleted Agent can remain in the mutation's historical dirty set but
        # must never survive as executable state.
        self.dirty_agents.intersection_update(self.graph.nodes)
        self.dirty_reasons = {
            agent_id: reasons
            for agent_id, reasons in self.dirty_reasons.items()
            if agent_id in self.graph.nodes
        }
        configured = {key for key, node in self.graph.nodes.items() if node.configured}
        executable_dirty = self.dirty_agents & configured
        if not executable_dirty and not force:
            return None
        if executable_dirty:
            # The estimator is a scheduling hint, not permission to spend past
            # the budget. Give each scheduled execution/revision a bounded share
            # and enforce it against serialized requests in the gateway.
            calls = max(
                1,
                int(
                    self.runtime.estimate_execution_tokens(
                        self.graph,
                        executable_dirty,
                        quantile=self.config.worker_token_quantile,
                        minimum_samples=self.config.worker_token_min_samples,
                        cold_start_tokens=self.config.worker_token_cold_start,
                    )["call_count"]
                ),
            )
            remaining = max(0, self.config.max_total_tokens - self.total_tokens)
            credit = remaining // calls
            budget_kind = (
                "short_qa_request_credit" if is_short_qa_dataset(self.dataset)
                else "unified_request_credit"
            )
            self._token_admission_event.update(
                {
                    "budget_schema": budget_kind,
                    "remaining_worker_tokens": remaining,
                    "finalization_output_reserve_minimum": self.config.finalization_token_reserve,
                    "call_count": calls,
                    "per_execution_credit": credit,
                    "request_admission": "authoritative_after_request_serialization",
                }
            )
            for agent_id in configured:
                self.graph.nodes[agent_id].metadata.update(
                    {
                        "_runtime_token_credit": credit,
                        "_runtime_budget_kind": budget_kind,
                        "_runtime_finalization_output_reserve": self.config.finalization_token_reserve,
                    }
                )
        report = self.runtime.execute(
            task=self.worker_task,
            graph=self.graph,
            dirty_agents=executable_dirty if executable_dirty else set(),
            invalidation_reasons={
                agent_id: set(self.dirty_reasons.get(agent_id, ())) for agent_id in executable_dirty
            },
            token_credit=max(0, self.config.max_total_tokens - self.total_tokens),
        )
        self._step_scheduled_agents.update(report.scheduled_agents)
        self.total_tokens += report.token_in + report.token_out
        # Execution has already committed artifacts to the runtime. Clear the
        # corresponding dirty bits before reporting a budget violation; otherwise
        # every later graph edit re-executes the same Agents and compounds the
        # over-budget trajectory without producing new feedback.
        completed = set(report.scheduled_agents)
        self.dirty_agents -= completed
        for agent_id in completed:
            self.dirty_reasons.pop(agent_id, None)
        if any(
            artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
            and not self.runtime.artifact_matches_current_input_signature(
                artifact.agent_id, task=self.worker_task, graph=self.graph)
            for artifact in report.artifacts.values()
        ):
            raise RequiredWorkerBackendFailure(report)
        if self.total_tokens > self.config.max_total_tokens:
            raise TokenBudgetExceeded(
                report,
                self.total_tokens,
                self.config.max_total_tokens,
            )
        return report

    def _artifact_is_usable_output(self, agent_id: str) -> bool:
        """Return whether an existing configured Agent has a real answer artifact."""

        node = self.graph.nodes.get(agent_id)
        artifact = self.runtime.artifacts.get(agent_id)
        if node is None or not node.configured or artifact is None:
            return False
        if (
            (
                self.dataset == "alfworld"
                or (
                    self.action_adapter is not None
                    and self.action_adapter.adapter_id == "alfworld"
                )
            )
            and isinstance(artifact.environment_result, dict)
            and isinstance(artifact.environment_result.get("environment_completed"), bool)
            and "won" in artifact.environment_result
        ):
            return True
        answer = str(artifact.answer or "").strip()
        return bool(
            answer
            and answer
            not in {
                WORKER_BACKEND_FAILURE_SENTINEL,
                WORKER_PROTOCOL_FAILURE_SENTINEL,
            }
        )

    def _usable_output_agents(self) -> tuple[str, ...]:
        return tuple(
            agent_id
            for agent_id in sorted(self.graph.nodes)
            if self._artifact_is_usable_output(agent_id)
        )

    def _output_health(self) -> str:
        output = self.graph.output_agent
        if output is None:
            return "missing"
        return "usable" if self._artifact_is_usable_output(output) else "unusable"

    def _final_validation_error_codes(self) -> tuple[str, ...]:
        codes: set[str] = set()
        for error in self.graph.validate(final=True):
            if error == "graph is empty":
                codes.add("empty_graph")
            elif error.startswith("agents missing prompts:"):
                codes.add("unconfigured_agents")
            elif error == "output agent is not set":
                codes.add("output_not_set")
            elif error.startswith("agents cannot influence output:"):
                codes.add("output_unreachable")
            elif error.startswith("agent budget exceeded"):
                codes.add("agent_budget_exceeded")
            elif "relation" in error or "layer" in error:
                codes.add("invalid_relation")
            else:
                codes.add("other_graph_error")
        if self.graph.output_agent is not None and self._output_health() == "unusable":
            codes.add("output_artifact_unusable")
        return tuple(sorted(codes))

    def _repair_progress_signature(self) -> RepairProgressSignature | None:
        reason = self.structural_repair_reason
        if reason is None:
            return None
        unconfigured_count = sum(int(not node.configured) for node in self.graph.nodes.values())
        final_error_codes = self._final_validation_error_codes()
        return RepairProgressSignature(
            reason=reason,
            output_health=self._output_health(),
            final_error_codes=final_error_codes,
            unconfigured_count=unconfigured_count,
            unreachable_count=len(self._unreachable_to_output()),
            dirty_count=len(self.dirty_agents),
            final_ready=bool(
                not final_error_codes
                and not self.dirty_agents
                and self.state is CanvasState.BUILDING
            ),
        )

    @staticmethod
    def _repair_progress_score(signature: RepairProgressSignature) -> tuple[int, ...]:
        return (
            int(signature.final_ready),
            int(signature.output_health == "usable"),
            -len(signature.final_error_codes),
            -signature.unconfigured_count,
            -signature.unreachable_count,
            -signature.dirty_count,
        )

    def _record_repair_progress(
        self,
        action: CanvasAction,
        *,
        accepted: bool,
        rejection_code: str | None,
    ) -> None:
        connectivity_repair = (
            self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}
            or (self._connectivity_repair_active and not self._is_consolidation_repair())
        )
        if connectivity_repair:
            self._connectivity_repair_active = True
            signature = self._connectivity_state_signature()
            was_visited = signature in self._connectivity_repair_states
            if accepted:
                if was_visited:
                    self.semantic_no_progress_streak += 1
                else:
                    # A novel state may be a legitimate intermediate step, such
                    # as SET_LAYER before relation selection. Do not freeze it.
                    self.semantic_no_progress_streak = 0
                    self._connectivity_repair_states.add(signature)
            self._recent_repair_actions.append({
                "action": action.action_type.value,
                "accepted": accepted,
                "rejection_code": rejection_code,
                "semantic_progress": not was_visited if accepted else False,
                "state_revisited": was_visited,
                "signature": {"state_hash": signature},
                "unreachable_agents": list(self._unreachable_to_output()),
            })
            self._recent_repair_actions = self._recent_repair_actions[
                -self.config.repair_recent_action_limit :
            ]
            if self.structural_repair_reason is None:
                self._connectivity_repair_active = False
                self._connectivity_repair_states.clear()
                self.semantic_no_progress_streak = 0
            return

        # Other repair classes retain their existing progress policy.
        if not self._is_output_lifecycle_repair():
            return
        current = self._repair_progress_signature()
        if current is None:
            return
        previous = self._last_repair_signature
        progress: bool | None
        if self._repair_entry_pending or previous is None:
            progress = None
            self._repair_entry_pending = False
            self.semantic_no_progress_streak = 0
        else:
            progress = self._repair_progress_score(current) > self._repair_progress_score(previous)
            if progress:
                self.semantic_no_progress_streak = 0
            else:
                self.semantic_no_progress_streak += 1
        self._last_repair_signature = current
        self._recent_repair_actions.append(
            {
                "action": action.action_type.value,
                "accepted": accepted,
                "rejection_code": rejection_code,
                "semantic_progress": progress,
                "signature": current.to_dict(),
            }
        )
        self._recent_repair_actions = self._recent_repair_actions[
            -self.config.repair_recent_action_limit :
        ]
        if (
            self.semantic_no_progress_streak >= self.config.semantic_no_progress_limit
            and self.state is CanvasState.BUILDING
            and not self.topology_edits_frozen
        ):
            self.topology_edits_frozen = True
            self.semantic_no_progress_recovery_count += 1

    def _begin_relation_decision(self, action: CanvasAction) -> CanvasStep:
        source, target, relation_type = self._canonical_relation_candidate(
            str(action.source), str(action.target)
        )
        key = (self.graph.version, tuple(sorted((source, target))))
        if key in self._considered_relation_pairs:
            return self._record(
                action,
                accepted=False,
                feedback=(
                    "Rejected action: this Agent pair was already considered at the current "
                    "Canvas version; make another graph change before considering it again."
                ),
                rejection_code="relation_pair_already_considered",
            )
        pending = PendingRelationDecision(
            source=source,
            target=target,
            relation_type=relation_type,
            graph_version=self.graph.version,
            proposal_round_index=self.round_index,
        )
        self._considered_relation_pairs.add(key)
        self.pending_relation_decision = pending
        self.state = CanvasState.AWAITING_RELATION_CHOICE
        payload = {"phase": "proposal", **pending.to_dict()}
        self._last_relation_decision = dict(payload)
        return self._record(
            action,
            accepted=True,
            feedback=(
                f"Accepted relation proposal for {source} and {target}; Canvas inferred "
                f"{relation_type.value}. A constrained off/on policy choice is now required."
            ),
            relation_decision=payload,
        )

    def resolve_relation_choice(
        self,
        choice: str,
        *,
        policy_audit: dict[str, Any] | None = None,
        count_round: bool = True,
    ) -> CanvasStep:
        """Resolve one pending proposal without treating a no-op choice as failure."""

        pending = self.pending_relation_decision
        normalized = str(choice).strip().casefold()
        if pending is None or self.state is not CanvasState.AWAITING_RELATION_CHOICE:
            return self._record(
                CanvasAction(
                    ActionType.INVALID, raw_text=str(choice), parse_error="no pending relation"
                ),
                accepted=False,
                feedback="Rejected relation choice: no relation proposal is pending.",
                rejection_code="no_pending_relation_choice",
            )
        if count_round:
            if self.round_index >= self.director_round_limit:
                return self._record(
                    CanvasAction(
                        ActionType.INVALID,
                        raw_text=str(choice),
                        parse_error="relation choice exceeds Director round budget",
                    ),
                    accepted=False,
                    feedback="Rejected relation choice: Director round budget is exhausted.",
                    rejection_code="canvas_inactive",
                    relation_decision={"phase": "choice", **pending.to_dict()},
                )
            self.round_index += 1
        if normalized not in {"off", "on"}:
            return self._record(
                CanvasAction(
                    ActionType.INVALID,
                    raw_text=str(choice),
                    parse_error="relation choice must be exactly off or on",
                ),
                accepted=False,
                feedback="Rejected relation choice: expected exactly off or on.",
                rejection_code="invalid_relation_choice",
                relation_decision={"phase": "choice", **pending.to_dict()},
            )
        if pending.graph_version != self.graph.version:
            self.pending_relation_decision = None
            self.state = CanvasState.BUILDING
            return self._record(
                CanvasAction(
                    ActionType.INVALID, raw_text=normalized, parse_error="stale relation choice"
                ),
                accepted=False,
                feedback="Rejected relation choice: the Canvas version changed after proposal.",
                rejection_code="stale_relation_choice",
                relation_decision={"phase": "choice", **pending.to_dict()},
            )

        desired_present = normalized == "on"
        pair = tuple(sorted((pending.source, pending.target)))
        currently_present = (
            pair in self.graph.bidirectional_edges
            if pending.relation_type is RelationType.BIDIRECTIONAL
            else (pending.source, pending.target) in self.graph.directed_edges
        )
        decision_payload = {
            "phase": "choice",
            **pending.to_dict(),
            "choice": normalized,
            "chosen_present": desired_present,
            "previously_present": currently_present,
            "policy": dict(policy_audit or {}),
        }
        self._last_relation_decision = dict(decision_payload)
        self.pending_relation_decision = None
        self.state = CanvasState.BUILDING
        if desired_present == currently_present:
            action = CanvasAction(
                ActionType.SET_RELATION if desired_present else ActionType.REMOVE_RELATION,
                source=pending.source,
                target=pending.target,
                relation=pending.relation_type,
                expected_version=pending.graph_version,
                raw_text=normalized,
            )
            return self._record(
                action,
                accepted=True,
                feedback=(
                    f"Accepted relation choice {normalized}; the inferred "
                    f"{pending.relation_type.value} relation already had that state, so the "
                    "Canvas version and Worker inputs are unchanged."
                ),
                relation_decision=decision_payload,
            )

        action_name = "set_relation" if desired_present else "remove_relation"
        raw_action = json.dumps(
            {
                "action": action_name,
                "source": pending.source,
                "target": pending.target,
                "relation": pending.relation_type.value,
                "expected_version": pending.graph_version,
            },
            separators=(",", ":"),
        )
        step = self.step(raw_action, count_round=False, authoritative_director=False)
        step.action.raw_text = normalized
        step.relation_decision = decision_payload
        return step

    def abandon_relation_choice(self, reason: str) -> CanvasStep:
        """Fail closed when a backend cannot provide auditable binary probabilities."""

        pending = self.pending_relation_decision
        if pending is None:
            raise RuntimeError("no pending relation decision to abandon")
        self.pending_relation_decision = None
        self.state = CanvasState.BUILDING
        return self._record(
            CanvasAction(ActionType.INVALID, raw_text="", parse_error=str(reason)),
            accepted=False,
            feedback=(
                "Relation proposal was abandoned without changing the graph because the backend "
                "could not provide an auditable constrained off/on policy choice."
            ),
            rejection_code="relation_binary_policy_unavailable",
            protocol_recovery=True,
            relation_decision={"phase": "abandoned", **pending.to_dict(), "reason": str(reason)},
        )

    def _canonical_relation_candidate(
        self, source: str, target: str
    ) -> tuple[str, str, RelationType]:
        source_node = self.graph.require_node(source)
        target_node = self.graph.require_node(target)
        if source == target:
            raise GraphValidationError("self relations are not allowed")
        if source_node.layer == target_node.layer:
            first, second = sorted((source, target))
            return first, second, RelationType.BIDIRECTIONAL
        if source_node.layer < target_node.layer:
            return source, target, RelationType.DIRECTED
        return target, source, RelationType.DIRECTED

    def _apply(
        self,
        graph: MultiAgentGraph,
        action: CanvasAction,
        *,
        compilation: DelegationCompilation | None = None,
    ) -> MutationResult:
        kind = action.action_type
        if kind is ActionType.ADD_AGENT:
            mutation = graph.add_agent(action.agent_id)
            agent_id = next(iter(mutation.dirty_agents))
            from uuid import uuid4
            graph.nodes[agent_id].metadata.update(submission_protocol=PROTOCOL, result_scope="subtask", incarnation_id=uuid4().hex, task_dataset=self.dataset)
            if self.action_adapter is not None:
                configured = graph.configure_action_environment(
                    agent_id,
                    adapter_id=self.action_adapter.adapter_id,
                    action_names=self.action_adapter.action_names,
                    initial_action_budget=self.action_adapter.initial_action_budget,
                    revision_action_budget=self.action_adapter.revision_action_budget,
                    total_action_budget=self.action_adapter.total_action_budget,
                    capability_policy=self.action_adapter.capability_policy(),
                )
                mutation.dirty_agents.update(configured.dirty_agents)
                mutation.message += "; " + configured.message
            return mutation
        if kind is ActionType.SET_PROMPT:
            prompt = (
                compilation.prompt if compilation is not None else self._delegation_prompt(action)
            )
            if action.runtime_route is not None:
                raise GraphValidationError("use SET_MODEL to select a Worker model")
            metadata_updates = compilation.metadata() if compilation is not None else {}
            metadata_updates["result_scope"] = action.result_scope
            mutation = graph.set_prompt(
                str(action.target),
                prompt,
                metadata_updates=metadata_updates,
            )
            if compilation is not None and compilation.field_repairs:
                repaired_fields = ", ".join(
                    sorted({repair.field for repair in compilation.field_repairs})
                )
                mutation.message += f"; deterministically repaired fields: {repaired_fields}"
            if compilation is not None and compilation.contract_version:
                mutation.message += (
                    f"; appended {compilation.contract_version} for {compilation.dataset}"
                )
            return mutation
        if kind is ActionType.SET_MODEL:
            if action.runtime_route not in self.runtime_routes:
                raise GraphValidationError(f"unknown Worker runtime_route: {action.runtime_route}")
            return graph.set_model(str(action.target), str(action.runtime_route))
        if kind is ActionType.SET_LAYER:
            return graph.set_layer(str(action.target), int(action.layer))
        if kind is ActionType.SET_RELATION:
            return graph.set_relation(str(action.source), str(action.target), action.relation)
        if kind is ActionType.REMOVE_RELATION:
            return graph.remove_relation(str(action.source), str(action.target), action.relation)
        if kind is ActionType.DELETE_AGENT:
            return graph.delete_agent(str(action.target or action.agent_id))
        raise GraphValidationError(f"unsupported canvas action: {kind.value}")

    @staticmethod
    def _mutation_invalidation_reasons(
        action: CanvasAction,
        mutation: MutationResult,
        *,
        previous_graph: MultiAgentGraph,
        current_graph: MultiAgentGraph,
    ) -> dict[str, set[str]]:
        dirty = set(mutation.dirty_agents) & set(current_graph.nodes)
        if not dirty:
            return {}
        if action.action_type is ActionType.ADD_AGENT:
            return {agent_id: {"new_agent_initial"} for agent_id in dirty}
        if action.action_type is ActionType.SET_PROMPT:
            target = str(action.target)
            component = current_graph.bidirectional_component(target)
            reasons: dict[str, set[str]] = {}
            for agent_id in dirty:
                if agent_id == target:
                    reasons[agent_id] = {"prompt_changed"}
                elif agent_id in component:
                    reasons[agent_id] = {"peer_changed"}
                else:
                    reasons[agent_id] = {"upstream_changed"}
            return reasons
        if action.action_type is ActionType.SET_LAYER:
            target = str(action.target)
            peer_agents: set[str] = set()
            for graph in (previous_graph, current_graph):
                if target in graph.nodes:
                    peer_agents.update(graph.bidirectional_component(target))
            return {
                agent_id: {
                    "layer_changed"
                    if agent_id == target
                    else "peer_changed"
                    if agent_id in peer_agents
                    else "upstream_changed"
                }
                for agent_id in dirty
            }
        if action.action_type in {
            ActionType.SET_RELATION,
            ActionType.REMOVE_RELATION,
        }:
            if action.relation is RelationType.BIDIRECTIONAL:
                peer_agents: set[str] = set()
                for graph in (previous_graph, current_graph):
                    for endpoint in (str(action.source), str(action.target)):
                        if endpoint in graph.nodes:
                            peer_agents.update(graph.bidirectional_component(endpoint))
                return {
                    agent_id: {"peer_changed" if agent_id in peer_agents else "upstream_changed"}
                    for agent_id in dirty
                }
            return {agent_id: {"upstream_changed"} for agent_id in dirty}
        if action.action_type is ActionType.DELETE_AGENT:
            target = str(action.target or action.agent_id)
            former_peers = (
                previous_graph.bidirectional_component(target) - {target}
                if target in previous_graph.nodes
                else set()
            )
            return {
                agent_id: {"peer_changed" if agent_id in former_peers else "upstream_changed"}
                for agent_id in dirty
            }
        return {agent_id: {mutation.change_reason or "environment_changed"} for agent_id in dirty}

    def _prompt_revision_checkpoint(self, target: str) -> dict[str, Any]:
        upstream_ids = sorted(self.graph.directed_predecessors(target))
        peer_ids = sorted(self.graph.bidirectional_component(target) - {target})

        def artifact_ids(agent_ids: list[str]) -> dict[str, str]:
            return {
                agent_id: (
                    self.runtime.artifacts[agent_id].artifact_id
                    if agent_id in self.runtime.artifacts
                    else ""
                )
                for agent_id in agent_ids
            }

        return {
            "upstream_artifacts": artifact_ids(upstream_ids),
            "peer_artifacts": artifact_ids(peer_ids),
            "topology": {
                "layer": self.graph.nodes[target].layer,
                "upstream_agent_ids": upstream_ids,
                "peer_agent_ids": peer_ids,
            },
            "repair_epoch": self.repair_epoch,
        }

    @staticmethod
    def _revision_evidence_signature(kind: str, payload: object) -> str:
        encoded = json.dumps(
            {"kind": kind, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
        return f"{kind}:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def _stable_tool_evidence(cls, value: object) -> object:
        volatile = {
            "call_id",
            "elapsed_ms",
            "remaining_budget",
            "round_index",
            "token_in",
            "token_out",
            "raw_response",
        }
        if isinstance(value, dict):
            return {
                str(key): cls._stable_tool_evidence(item)
                for key, item in value.items()
                if str(key) not in volatile
            }
        if isinstance(value, (list, tuple)):
            return [cls._stable_tool_evidence(item) for item in value]
        return value

    def _eligible_prompt_revision_evidence(self, target: str) -> dict[str, Any]:
        if target not in self.graph.nodes or not self.graph.nodes[target].configured:
            return {"public": {}, "internal": {}}
        current = self._prompt_revision_checkpoint(target)
        checkpoint = self._prompt_revision_checkpoints.get(target, current)
        internal: dict[str, dict[str, list[str]]] = {}

        def add(kind: PromptRevisionBasis, agent_id: str, payload: object) -> None:
            signature = self._revision_evidence_signature(kind.value, payload)
            internal.setdefault(kind.value, {}).setdefault(agent_id, []).append(signature)

        for kind, checkpoint_field in (
            (PromptRevisionBasis.UPSTREAM_ARTIFACT_CHANGED, "upstream_artifacts"),
            (PromptRevisionBasis.PEER_ARTIFACT_CHANGED, "peer_artifacts"),
        ):
            before = checkpoint[checkpoint_field]
            after = current[checkpoint_field]
            for agent_id, artifact_id in after.items():
                if before.get(agent_id) != artifact_id:
                    add(
                        kind,
                        agent_id,
                        {"agent_id": agent_id, "artifact_id": artifact_id},
                    )

        artifact = self.runtime.artifacts.get(target)
        if artifact is not None:
            for trace_item in artifact.react_trace:
                if not isinstance(trace_item, dict):
                    continue
                observation = trace_item.get("observation")
                if not isinstance(observation, dict):
                    continue
                output = observation.get("output")
                output_error = bool(
                    isinstance(output, dict) and str(output.get("status", "")).casefold() == "error"
                )
                if str(observation.get("status", "")).casefold() == "error" or output_error:
                    add(
                        PromptRevisionBasis.TOOL_ERROR,
                        target,
                        {
                            "action": self._stable_tool_evidence(trace_item.get("action")),
                            "observation": self._stable_tool_evidence(observation),
                        },
                    )
            for issue in artifact.unresolved_issues:
                add(PromptRevisionBasis.UNRESOLVED_ISSUE, target, str(issue))
            protocol_evidence: list[object] = []
            if is_aime_dataset(self.dataset) and target == self.graph.output_agent:
                submitted = parse_aime_answer(artifact.answer)
                if not submitted.valid:
                    # Public answer-format evidence permits an optional revision;
                    # it is not a Worker protocol failure or a FINISH prerequisite.
                    add(
                        PromptRevisionBasis.UNRESOLVED_ISSUE,
                        target,
                        {
                            "stage": "answer_format",
                            "artifact_id": artifact.artifact_id,
                            "format_reason": submitted.reason,
                        }
                    )
            protocol_status = summarize_worker_protocol(
                answer=artifact.answer,
                raw_response=artifact.raw_response,
                diagnostics=artifact.protocol_diagnostics,
            )
            if protocol_status["status"] == "failed":
                protocol_evidence.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "stage": protocol_status["stage"],
                        "error_code": protocol_status["error_code"],
                    }
                )
            for evidence in protocol_evidence:
                add(PromptRevisionBasis.PROTOCOL_FAILURE, target, evidence)

        if current["topology"] != checkpoint["topology"]:
            add(
                PromptRevisionBasis.STRUCTURAL_ROLE_CHANGE,
                target,
                current["topology"],
            )
        if int(current["repair_epoch"]) > int(checkpoint["repair_epoch"]):
            add(
                PromptRevisionBasis.CONTROLLER_REPAIR,
                target,
                {
                    "repair_epoch": current["repair_epoch"],
                    "target": target,
                },
            )

        public = {
            basis: {
                "evidence_agent_ids": sorted(by_agent),
                "revision_count": self.prompt_revision_counts.get(target, 0),
            }
            for basis, by_agent in sorted(internal.items())
            if by_agent
        }
        return {"public": public, "internal": internal}

    def _finalize_prompt_assignment(self, action: CanvasAction) -> None:
        if action.action_type not in {ActionType.SET_PROMPT, ActionType.SET_MODEL}:
            return
        target = str(action.target)
        if target not in self.graph.nodes or not self.graph.nodes[target].configured:
            return
        self._prompt_revision_checkpoints[target] = self._prompt_revision_checkpoint(target)

    def _completes_prompt(self, action: CanvasAction) -> bool:
        return (
            action.action_type is ActionType.SET_PROMPT and action.target == self.pending_agent_id
        )

    def _delegation_prompt(self, action: CanvasAction) -> str:
        if action.prompt is not None:
            return action.prompt
        return "\n".join(
            (
                f"Role: {action.role}",
                f"Objective: {action.objective}",
                f"Scope: {action.scope}",
                f"Expected output: {action.expected_output}",
            )
        )

    def _validate_delegation(self, action: CanvasAction) -> DelegationCompilation | None:
        if action.prompt is not None:
            raise GraphValidationError(
                "SET_PROMPT must use short role, objective, scope, and expected_output fields; "
                "free-form prompt is forbidden"
            )
        fields = {
            "role": action.role,
            "objective": action.objective,
            "scope": action.scope,
            "expected_output": action.expected_output,
        }
        action_names = set(self.action_adapter.action_names if self.action_adapter else ())
        compilation, issue = compile_delegation(
            fields,
            public_task=self.task,
            dataset=self.dataset,
            action_names=action_names,
        )
        if issue is not None:
            raise DelegationValidationError(issue)
        assert compilation is not None
        self._duplicate_responsibility_issue(
            target=str(action.target),
            fields=compilation.director_fields,
        )
        # Keep the accepted Canvas action aligned with the compiled responsibility.
        # ``raw_text`` remains the exact model output for observability.
        for name, value in compilation.director_fields.items():
            setattr(action, name, value)
        return compilation

    def _duplicate_responsibility_issue(
        self,
        *,
        target: str,
        fields: dict[str, str],
    ) -> None:
        """Record how a new responsibility overlaps those of the other Agents (never rejects)."""

        candidate_signature = responsibility_signature(fields)
        comparisons: list[dict[str, Any]] = []
        for agent_id, node in sorted(self.graph.nodes.items()):
            if agent_id == target:
                continue
            existing_fields = node.metadata.get("director_delegation")
            if not isinstance(existing_fields, dict):
                continue
            comparison = compare_responsibilities(existing_fields, fields)
            comparisons.append(
                {
                    "agent_id": agent_id,
                    **comparison.to_dict(),
                }
            )
        comparisons.sort(key=lambda item: float(item["similarity"]), reverse=True)
        strongest = comparisons[0] if comparisons else None
        decision = (
            "overlap_warning"
            if strongest and float(strongest["scope_overlap"]) >= 0.55
            else "accepted"
        )
        self._step_responsibility_overlap_check = {
            "target": target,
            "policy": "record_only",
            "candidate_primary_type": candidate_signature.primary_type,
            "decision": decision,
            "comparisons": comparisons[:8],
        }
        return None

    def _feedback(self, event: str, report: ExecutionReport | None) -> str:
        facts = [event, self.graph.describe()]
        if self.dirty_agents:
            facts.append("Dirty agents: " + ", ".join(sorted(self.dirty_agents)))
        if report is not None:
            facts.append(
                "Execution: ran=[{}], reused=[{}], tokens={}/{}.".format(
                    ", ".join(report.executed_agents),
                    ", ".join(report.reused_agents),
                    self.total_tokens,
                    self.config.max_total_tokens,
                )
            )
            if report.errors:
                facts.append(
                    "Execution errors: "
                    + "; ".join(
                        f"{key}={self._one_line(value, 160)}"
                        for key, value in sorted(report.errors.items())
                    )
                    + "."
                )
            if report.revision_decisions:
                revision_facts = []
                for decision in report.revision_decisions:
                    agents = ",".join(decision.get("component_agents", []))
                    outcome = "ran" if bool(decision.get("revision_required")) else "skipped"
                    reasons = ",".join(decision.get("reason_codes", []))
                    revision_facts.append(f"[{agents}]={outcome}({reasons})")
                facts.append("Peer revision decisions: " + "; ".join(revision_facts) + ".")
            signals = []
            for agent_id in sorted(set(report.executed_agents)):
                artifact = report.artifacts.get(agent_id)
                if artifact is None:
                    continue
                if self.dataset == "alfworld":
                    public_task = (artifact.environment_result or {}).get("public_task_statement")
                    if public_task:
                        facts.append(
                            f"ALFWorld public reset task for {agent_id}: "
                            + json.dumps(public_task, ensure_ascii=False)
                            + ". This is the environment's public task statement; "
                            "the original dataset task remains unchanged."
                        )
                tool_errors = int(artifact.runtime_tool_evidence.get("failed_count", 0) or 0)
                summary = self._one_line(
                    artifact.summary or artifact.answer,
                    self.config.artifact_summary_max_chars,
                )
                unresolved = [
                    self._one_line(value, 120) for value in artifact.unresolved_issues[:2]
                ]
                signals.append(
                    f"{agent_id}(summary={json.dumps(summary, ensure_ascii=False)}, "
                    f"confidence={artifact.confidence:.2f}, "
                    f"claimed_confidence={float(artifact.claimed_confidence or 0.0):.2f}, "
                    f"unresolved={len(artifact.unresolved_issues)}, "
                    f"unresolved_preview={json.dumps(unresolved, ensure_ascii=False)}, "
                    f"evidence={len(artifact.evidence)}, tool_errors={tool_errors}, "
                    f"integrity_risks={json.dumps(artifact.integrity_risks)}, "
                    f"backend_failure={artifact.answer == 'WORKER_BACKEND_FAILURE'})"
                )
            if signals:
                facts.append("Process signals: " + "; ".join(signals) + ".")
        return "\n".join(facts)

    def _unreachable_to_output(self) -> tuple[str, ...]:
        output = self.graph.output_agent
        if output is None or output not in self.graph.nodes:
            return ()
        return tuple(
            sorted(
                agent_id
                for agent_id in self.graph.nodes
                if agent_id != output and output not in self.graph.reachable_from(agent_id)
            )
        )

    def _weak_component_count(self) -> int:
        remaining = set(self.graph.nodes)
        count = 0
        adjacency = {agent_id: set() for agent_id in remaining}
        for source, target in self.graph.directed_edges | self.graph.bidirectional_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        while remaining:
            count += 1
            frontier = [remaining.pop()]
            while frontier:
                current = frontier.pop()
                unseen = adjacency[current] & remaining
                remaining -= unseen
                frontier.extend(unseen)
        return count

    @staticmethod
    def _unreachable_for_graph(graph: MultiAgentGraph, output: str | None) -> tuple[str, ...]:
        if output is None or output not in graph.nodes:
            return ()
        return tuple(sorted(
            agent_id for agent_id in graph.nodes
            if agent_id != output and output not in graph.reachable_from(agent_id)
        ))

    def _connectivity_state_signature(self) -> str:
        """Stable repair state; excludes counters and output-role bookkeeping metadata."""
        graph = self.graph.clone()
        payload = graph.to_dict()
        payload.pop("version", None)
        layer_rank = {
            value: rank for rank, value in enumerate(
                sorted({node.layer for node in graph.nodes.values()})
            )
        }
        for node in payload.get("nodes", []):
            node["layer"] = layer_rank.get(int(node.get("layer", 0)), 0)
            metadata = node.get("metadata", {})
            metadata.pop("_runtime_is_output_agent", None)
        artifacts = {}
        for agent_id, artifact in sorted(self.runtime.artifacts.items()):
            artifacts[agent_id] = {
                "answer": artifact.answer,
                "summary": artifact.summary,
                "evidence": list(artifact.evidence),
                "environment_result": artifact.environment_result,
            }
        return hashlib.sha256(json.dumps(
            {"graph": payload, "artifacts": artifacts},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()

    def topology_audit(self) -> dict[str, Any]:
        relation_count = len(self.graph.directed_edges) + len(self.graph.bidirectional_edges)
        unreachable = self._unreachable_to_output()
        return {
            "agent_count": len(self.graph.nodes),
            "configured_agent_count": sum(
                int(node.configured) for node in self.graph.nodes.values()
            ),
            "relation_count": relation_count,
            "weak_component_count": self._weak_component_count(),
            "weakly_disconnected": self._weak_component_count() > 1,
            "actual_relations": (
                [
                    {"source": source, "target": target, "relation": "directed"}
                    for source, target in sorted(self.graph.directed_edges)
                ]
                + [
                    {"source": source, "target": target, "relation": "bidirectional"}
                    for source, target in sorted(self.graph.bidirectional_edges)
                ]
            ),
            "output_agent": self.graph.output_agent,
            "unreachable_agents": list(unreachable),
            "all_agents_reach_output": (
                not unreachable if self.graph.output_agent is not None else None
            ),
            "disconnected_multi_agent": (len(self.graph.nodes) > 1 and relation_count == 0),
        }

    def _graph_state_snapshot(self) -> dict[str, Any]:
        # Snapshots also enter feedback/history sent to the Director. Keep the
        # sampled choice here, but retain probability/request audit details only
        # in the original relation_decision event and DirectorTurn records.
        relation_facts = {
            key: value for key, value in self._last_relation_decision.items()
            if key != "policy"
        }
        nodes = [
            {"id": agent_id, "layer": self.graph.nodes[agent_id].layer,
             "configured": self.graph.nodes[agent_id].configured}
            for agent_id in sorted(self.graph.nodes)
        ]
        actual_relations = self.topology_audit()["actual_relations"]
        targets = [key for key, node in self.graph.nodes.items()
                   if node.metadata.get("result_scope") == "task_result"]
        validation = {}
        for target in targets:
            candidate = self.graph.clone()
            candidate.output_agent = target
            validation[target] = {
                "unreachable_nodes": list(self._unreachable_for_graph(candidate, target)),
                "graph_blockers": list(candidate.validate(final=True)),
            }
        return {
            "nodes": [{**node, "result_scope": self.graph.nodes[node["id"]].metadata.get("result_scope")} for node in nodes],
            "actual_relations": actual_relations,
            "candidate_relations_are_not_edges": True,
            "submission_target": self.graph.output_agent,
            "task_result_agents": sorted(targets),
            "reachability_by_target": validation,
            "weak_components": self._weak_component_count(),
            "weakly_disconnected": self._weak_component_count() > 1,
            "last_relation_decision": relation_facts,
        }

    def _enter_structural_repair(
        self,
        reason: str,
        *,
        agents: tuple[str, ...] = (),
    ) -> None:
        previous_reason = self.structural_repair_reason
        if previous_reason is None:
            self.structural_repair_entries += 1
            self._repair_transition = "entered"
        elif previous_reason != reason:
            self._repair_transition = "retargeted"
        if previous_reason != reason:
            self.repair_epoch += 1
            self.semantic_no_progress_streak = 0
            self._last_repair_signature = None
            self._repair_entry_pending = True
            self._recent_repair_actions = []
            self._consolidation_output_locked = False
            self._output_selection_budget_remaining = self.config.output_selection_budget
        self.structural_repair_reason = reason
        if self._is_consolidation_repair():
            self._connectivity_repair_active = False
            self._connectivity_repair_states.clear()
        self.structural_repair_agents = tuple(
            sorted(agent_id for agent_id in agents if agent_id in self.graph.nodes)
        )
        if reason in {"output_reachability", "disconnected_multi_agent"}:
            self._connectivity_repair_active = True
            self._connectivity_repair_states.add(self._connectivity_state_signature())
            if not self._is_consolidation_repair():
                # A generic invalid-action fuse must not strand a graph whose
                # known structural defect requires layer/relation edits.
                self.topology_edits_frozen = False
                self._invalid_repeat_count = 0
                self._last_invalid_signature = None

    def _clear_structural_repair(self) -> None:
        if self.structural_repair_reason is None:
            return
        self.structural_repair_reason = None
        self.structural_repair_agents = ()
        self.structural_repair_resolutions += 1
        self._repair_transition = "resolved"
        self._consolidation_output_locked = False

    def _is_consolidation_repair(self) -> bool:
        return self.structural_repair_reason in {
            "token_budget_consolidation",
            "time_budget_consolidation",
        }

    def _is_output_lifecycle_repair(self) -> bool:
        return self._is_consolidation_repair() or self.structural_repair_reason in {
            "output_not_set",
            "output_artifact_unusable",
        }

    def _refresh_structural_repair(self, action: CanvasAction) -> None:
        if self.state is CanvasState.AWAITING_MODEL:
            return
        reason = self.structural_repair_reason
        if reason is None:
            # Construction remains free until an output is selected. Once one
            # exists, a newly configured disconnected Agent must be connected,
            # deleted, or made the output before the graph can grow again.
            # ADD_AGENT itself must still reach the mandatory SET_PROMPT
            # barrier; auditing it one turn early would deadlock that barrier.
            unreachable = (
                () if action.action_type is ActionType.ADD_AGENT else self._unreachable_to_output()
            )
            if unreachable:
                self._enter_structural_repair(
                    "output_reachability",
                    agents=unreachable,
                )
            elif (
                action.action_type is not ActionType.ADD_AGENT
                and self.graph.output_agent is None
                and self.topology_audit()["disconnected_multi_agent"]
            ):
                self._enter_structural_repair(
                    "disconnected_multi_agent",
                    agents=tuple(self.graph.nodes),
                )
            return
        if reason in {"token_budget_consolidation", "time_budget_consolidation"}:
            # Consolidation is a terminal budget lock, not a structural defect.
            # Keep it latched until FINISH terminates the graph.
            self.structural_repair_agents = tuple(sorted(self.graph.nodes))
            return
        if reason == "disconnected_multi_agent" and self.graph.output_agent is None:
            if self.topology_audit()["disconnected_multi_agent"]:
                return
            self._clear_structural_repair()
            return
        if self.graph.output_agent is None:
            if reason == "output_not_set":
                return
            self._clear_structural_repair()
            return
        unreachable = self._unreachable_to_output()
        if unreachable:
            self._enter_structural_repair(
                "output_reachability",
                agents=unreachable,
            )
        else:
            self._clear_structural_repair()

    def _structural_repair_snapshot(self) -> dict[str, Any]:
        # Report the same state/protocol/parameter mask that admission uses.
        # The repair-type superset alone can advertise disabled relation actions.
        allowed = (
            self.control_snapshot()["allowed_actions"] if self.structural_repair_reason else []
        )
        guidance = ""
        if self.structural_repair_reason == "token_budget_consolidation":
            guidance = (
                "The remaining Worker token budget does not admit another execution. The "
                "allowed actions and their valid targets are listed separately."
            )
        if self.structural_repair_reason in {"output_reachability", "disconnected_multi_agent"}:
            guidance = (
                f"Actual edges are {self.topology_audit()['actual_relations']}; optional relation "
                "parameters are only candidates. Check topology_action_previews before switching "
                "output. A legal edge may still point away from the selected output. Keep layer "
                "and relation edits available until the graph validates."
            )
        return {
            "required": self.structural_repair_reason is not None,
            "reason": self.structural_repair_reason,
            "agents": list(self.structural_repair_agents),
            "guidance": guidance,
            "allowed_actions": allowed,
            "transition": self._repair_transition,
            "entries_total": self.structural_repair_entries,
            "resolutions_total": self.structural_repair_resolutions,
            "blocked_actions_total": self.structural_repair_blocked_actions,
            "finish_reachability_rejections_total": self.finish_reachability_rejections,
            "duplicate_agent_rejections_total": self.duplicate_agent_rejections,
            "time_budget_rejections_total": self.time_budget_rejections,
            "token_budget_admission_rejections_total": (self.token_budget_admission_rejections),
            "repair_epoch": self.repair_epoch,
            "progress_signature": (
                self._last_repair_signature.to_dict()
                if self._last_repair_signature is not None
                else None
            ),
            "semantic_no_progress_streak": self.semantic_no_progress_streak,
            "semantic_no_progress_limit": self.config.semantic_no_progress_limit,
            "semantic_no_progress_recoveries_total": (self.semantic_no_progress_recovery_count),
            "connectivity_repair_active": self._connectivity_repair_active,
            "connectivity_repair_visited_states": len(self._connectivity_repair_states),
            "connectivity_output_cycle_blocks": self._connectivity_cycle_blocks,
            "output_switches_without_progress_total": (self.output_switches_without_progress),
            "output_lifecycle_recoveries_total": self.output_lifecycle_recovery_count,
            "recent_actions": list(self._recent_repair_actions),
            "output_lifecycle": {
                "health": self._output_health(),
                "usable_candidates": list(self._usable_output_agents()),
                "locked": self._consolidation_output_locked,
                "selection_budget_remaining": self._output_selection_budget_remaining,
                "incumbent": dict(self._output_incumbent),
            },
        }

    def _audit_feedback(self) -> str:
        return "Actual topology: " + json.dumps(self._graph_state_snapshot(), ensure_ascii=False, sort_keys=True) + ". FINISH(target) eligibility is recorded in result_assessments."

    @staticmethod
    def _one_line(value: object, limit: int) -> str:
        compact = " ".join(str(value or "").split())
        if limit <= 0 or len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 15)] + "... [truncated]"

    def _bound_feedback(self, feedback: str) -> str:
        limit = self.config.feedback_max_chars
        if limit <= 0 or len(feedback) <= limit:
            return feedback
        marker = "\n... [feedback truncated] ...\n"
        available = max(0, limit - len(marker))
        head = available * 2 // 3
        tail = available - head
        return feedback[:head] + marker + (feedback[-tail:] if tail else "")

    def _record(
        self,
        action: CanvasAction,
        *,
        accepted: bool,
        feedback: str,
        execution: ExecutionReport | None = None,
        rejection_code: str | None = None,
        protocol_recovery: bool = False,
        final_execution: bool = False,
        responsibility_issue: dict[str, Any] | None = None,
        delegation_field_repairs: list[dict[str, Any]] | None = None,
        rejection_details: dict[str, Any] | None = None,
        relation_decision: dict[str, Any] | None = None,
        invalid_repeat_count: int | None = None,
    ) -> CanvasStep:
        resolved_rejection_details = dict(rejection_details or {})
        if not accepted:
            rejection_code = rejection_code or "rejected_action"
        self._record_unified_progress(accepted=accepted)
        self._record_repair_progress(
            action,
            accepted=accepted,
            rejection_code=rejection_code,
        )
        if not accepted:
            self._register_no_progress_rejection()
            common_details = self._rejection_details(rejection_code, action=action)
            common_details.update(resolved_rejection_details)
            common_details["repeat_count"] = self._invalid_repeat_count
            common_details["deterministic_recovery"] = (
                "refresh_and_choose_legal_action"
                if not self.topology_edits_frozen
                else "freeze_topology_edits_continue_existing_graph"
            )
            resolved_rejection_details = common_details
            recovery = json.dumps(
                resolved_rejection_details,
                ensure_ascii=False,
                sort_keys=True,
            )
            if "Recovery:" not in feedback:
                feedback = f"{feedback} Recovery: {recovery}"
        audit = self.topology_audit()
        repair = self._structural_repair_snapshot()
        decorated_feedback = self._bound_feedback(
            "\n".join(
                (
                    feedback,
                    self._audit_feedback(),
                    f"Canvas state: {self.state.value}; "
                    + f"round {self.round_index}/{self.director_round_limit}; "
                    + f"cumulative Worker tokens={self.total_tokens}/"
                    f"{self.config.max_total_tokens}.",
                )
            )
        )
        step = CanvasStep(
            event_id=f"{self.run_id}:canvas:{len(self.history)}",
            director_call_id=(self._director_call_context.call_id
                              if isinstance(self._director_call_context, DirectorCallContext)
                              and self._director_call_context.runtime_owned else ""),
            submission_receipt=(self.submission_receipt.to_dict()
                                if accepted and action.action_type is ActionType.FINISH
                                and self.submission_receipt is not None else None),
            round_index=self.round_index,
            action=action,
            accepted=accepted,
            active=self.active,
            feedback=decorated_feedback,
            graph=self.graph.to_dict(),
            dirty_agents=sorted(self.dirty_agents),
            invalidated_agents=sorted(self._step_invalidated_agents),
            scheduled_agents=sorted(self._step_scheduled_agents),
            executed_agents=(list(execution.executed_agents) if execution else []),
            reused_agents=(list(execution.reused_agents) if execution else []),
            remaining_dirty=sorted(self.dirty_agents),
            invalidation_reasons={
                agent_id: sorted(reasons)
                for agent_id, reasons in self._step_invalidation_reasons.items()
                if reasons
            },
            execution=execution,
            rejection_code=rejection_code,
            protocol_recovery=protocol_recovery,
            final_execution=final_execution,
            topology_audit=audit,
            structural_repair=repair,
            responsibility_issue=dict(responsibility_issue or {}),
            responsibility_overlap_check=dict(self._step_responsibility_overlap_check),
            delegation_field_repairs=list(delegation_field_repairs or []),
            token_admission=dict(self._token_admission_event),
            control_snapshot=self.control_snapshot(),
            rejection_details=resolved_rejection_details,
            relation_decision=dict(relation_decision or {}),
            invalid_repeat_count=(
                self._invalid_repeat_count
                if invalid_repeat_count is None
                else int(invalid_repeat_count)
            ),
            topology_edits_frozen=self.topology_edits_frozen,
        )
        self.history.append(step)
        if accepted:
            self._last_invalid_signature = None
            self._invalid_repeat_count = 0
        self._repair_transition = None
        return step

    def _register_no_progress_rejection(self) -> None:
        """Count every rejection made against the same authoritative state."""

        signature = json.dumps(
            {
                "version": self.graph.version,
                "state": self.state.value,
                "pending_agent_id": self.pending_agent_id,
                "pending_relation_decision": (
                    self.pending_relation_decision.to_dict()
                    if self.pending_relation_decision is not None
                    else None
                ),
                "dirty_agents": sorted(self.dirty_agents),
                "structural_repair_reason": self.structural_repair_reason,
                "structural_repair_agents": list(self.structural_repair_agents),
            },
            sort_keys=True,
        )
        if signature == self._last_invalid_signature:
            self._invalid_repeat_count += 1
        else:
            self._last_invalid_signature = signature
            self._invalid_repeat_count = 1

    def director_progress_signature(self) -> str:
        """Effective state, excluding rounds, feedback and rejection counters."""
        return self._unified_progress_signature()

    def terminate_director_stall(self, code: str) -> CanvasStep:
        """Record runtime closure only; no model call or graph repair is invented."""
        if code not in {"director_no_legal_continuation", "director_no_progress_exhausted",
                        "director_action_protocol_exhausted"}:
            raise ValueError(f"unsupported Director stall: {code}")
        if (code == "director_no_legal_continuation"
                and not self.graph.nodes and self.runtime_routes
                and not self.topology_edits_frozen and self._unified_transaction is None
                and 0 <= self.config.max_rounds - self.round_index < 4):
            # ADD_AGENT, SET_PROMPT, SET_MODEL and FINISH require four policy
            # actions. Deleting the last node can exhaust that completion path
            # before the numeric round counter reaches its absolute ceiling.
            code = "director_round_budget_dead_end"
        return self._fail_terminal_recovery(
            code=code,
            message="no legal continuation or bounded Director recovery exhausted",
        )

    def control_snapshot(self) -> dict[str, Any]:
        """Bounded authoritative state supplied to the Director after every turn."""
        return self._unified_control_snapshot()

    def _legal_action_parameters(self) -> dict[str, dict[str, Any]]:
        legal_ids = sorted(self.graph.nodes)
        revision_evidence = {
            agent_id: self._eligible_prompt_revision_evidence(agent_id)["public"]
            for agent_id in legal_ids
            if self.graph.nodes[agent_id].configured
        }
        if self.state is CanvasState.AWAITING_PROMPT and self.pending_agent_id:
            prompt_targets = [self.pending_agent_id]
        else:
            prompt_targets = [
                agent_id
                for agent_id in legal_ids
                if not self.graph.nodes[agent_id].configured
                or bool(revision_evidence.get(agent_id))
            ]
        layer_targets = list(legal_ids)
        relation_candidates: list[dict[str, str]] = []
        for source in legal_ids:
            for target in legal_ids:
                if source < target:
                    canonical_source, canonical_target, inferred = (
                        self._canonical_relation_candidate(source, target)
                    )
                    considered_key = (self.graph.version, (source, target))
                    if considered_key not in self._considered_relation_pairs:
                        relation_candidates.append(
                            {
                                "source": canonical_source,
                                "target": canonical_target,
                                "relation": inferred.value,
                            }
                        )
        return {
            ActionType.ADD_AGENT.value: {
                "system_allocated_id": True,
                "remaining_slots": max(0, self.graph.max_agents - len(legal_ids)),
            },
            ActionType.SET_PROMPT.value: {
                "targets": prompt_targets,
                "revision_evidence_by_target": {
                    agent_id: revision_evidence[agent_id]
                    for agent_id in prompt_targets
                    if revision_evidence.get(agent_id)
                },
                "revision_counts": {
                    agent_id: self.prompt_revision_counts.get(agent_id, 0) for agent_id in legal_ids
                },
            },
            ActionType.SET_MODEL.value: {
                "targets": (
                    [self.pending_agent_id]
                    if self.state is CanvasState.AWAITING_MODEL
                    else [key for key in legal_ids if self.graph.nodes[key].prompt_configured]
                )
                if self.runtime_routes
                else [],
                "runtime_routes": list(self.runtime_routes),
            },
            ActionType.SET_LAYER.value: {
                "targets": layer_targets,
                "minimum_layer": 0,
                "current_layers": {
                    agent_id: self.graph.nodes[agent_id].layer for agent_id in layer_targets
                },
            },
            ActionType.CONSIDER_RELATION.value: {"relations": relation_candidates},
            # The unified control snapshot sets the DELETE_AGENT targets and adds FINISH and
            # RUN_AGENT.
            ActionType.DELETE_AGENT.value: {"targets": list(legal_ids)},
        }

    def _director_action_legality_error(
        self,
        action: CanvasAction,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> tuple[str, str] | None:
        if snapshot is None:
            snapshot = self.control_snapshot()
        action_name = action.action_type.value
        if action_name not in snapshot["allowed_actions"]:
            return (
                "director_action_not_allowed",
                f"{action_name} is not allowed in the current Canvas state",
            )
        parameters = snapshot["legal_action_parameters"][action_name]
        target_actions = {
            ActionType.SET_PROMPT,
            ActionType.SET_MODEL,
            ActionType.SET_LAYER,
            ActionType.DELETE_AGENT,
            ActionType.RUN_AGENT,
            ActionType.FINISH,
        }
        if action.action_type in target_actions:
            target = str(action.target or action.agent_id or "")
            if target not in parameters["targets"]:
                code = (
                    "unknown_agent"
                    if target not in self.graph.nodes
                    else "director_parameter_not_allowed"
                )
                return (code, f"target {target or '<missing>'} is not legal for {action_name}")
        if action.action_type is ActionType.CONSIDER_RELATION:
            # This action proposes an unordered pair. Use exactly the same
            # layer-based inference as _begin_relation_decision; raw name
            # order must not reject an otherwise legal binary proposal.
            source, target, relation = self._canonical_relation_candidate(
                str(action.source), str(action.target)
            )
            requested = {"source": source, "target": target, "relation": relation.value}
            if requested not in parameters["relations"]:
                return (
                    "director_parameter_not_allowed",
                    f"relation tuple is not legal for {action_name}",
                )
        return None

    @staticmethod
    def _classify_mutation_error(action: CanvasAction, message: str) -> str:
        lowered = message.casefold()
        if "unknown agent" in lowered:
            return "unknown_agent"
        if "agent budget exceeded" in lowered:
            return "agent_cap_exceeded"
        if "already exists" in lowered and action.action_type is ActionType.ADD_AGENT:
            return "duplicate_agent_id"
        if "relation" in lowered or "layer" in lowered:
            return "invalid_relation"
        if action.action_type is ActionType.SET_PROMPT:
            return "responsibility_violation"
        return "invalid_graph_edit"

    def _rejection_details(
        self,
        code: str,
        *,
        action: CanvasAction | None = None,
    ) -> dict[str, Any]:
        snapshot = self.control_snapshot()
        # ``legal_agent_ids`` contains every graph node, while a
        # parameterized action such as RUN_AGENT or FINISH usually accepts
        # only a subset.  Returning only the former made a rejected action
        # look retryable with the same target and caused repeated
        # ``director_parameter_not_allowed`` turns in stateful rollouts.
        # Preserve the full result assessments, but expose the exact
        # action-level mask that admission used for this rejection.
        parameters = snapshot.get("legal_action_parameters", {})
        action_name = action.action_type.value if action is not None else None
        details = {
            "reason_code": code,
            "current_version": self.graph.version,
            "legal_agent_ids": list(snapshot["legal_agent_ids"]),
            "allowed_actions": list(snapshot["allowed_actions"]),
            "result_assessments": snapshot["result_assessments"],
            "legal_action_parameters": {
                name: value
                for name, value in parameters.items()
                if name in snapshot["allowed_actions"] or name == action_name
            },
        }
        if action_name:
            details["attempted_action"] = action_name
            details["attempted_action_parameters"] = dict(
                parameters.get(action_name, {})
            )
        return details

    def _reject_graph_action(
        self,
        action: CanvasAction,
        *,
        code: str,
        message: str,
        responsibility_issue: dict[str, Any] | None = None,
        rejection_details: dict[str, Any] | None = None,
        delegation_field_repairs: list[dict[str, Any]] | None = None,
    ) -> CanvasStep:
        return self._record(
            action,
            accepted=False,
            feedback=f"Rejected action: {message}.",
            rejection_code=code,
            responsibility_issue=responsibility_issue,
            rejection_details=rejection_details,
            delegation_field_repairs=delegation_field_repairs,
        )
