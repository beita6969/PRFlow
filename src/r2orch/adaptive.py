from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from .aime_submission import is_aime_dataset
from .alfworld import alfworld_lifecycles
from .answer_submission import AnswerFinalizer, AnswerSubmission, qa_token_f1
from .canvas import GraphCanvas
from .config import CanvasConfig, canonical_dataset_name
from .dataset_actions import DatasetActionRegistry
from .dataset_adapters import solver_task_text
from .deadline import RolloutDeadline
from .director import DirectorRun, GraphDirector
from .llm import ChatBackend
from .observability import (
    ExecutionTrace,
    JSONLTraceStore,
    NumericVerifier,
    TaskSpec,
    VerificationResult,
    Verifier,
    trace_from_canvas,
)
from .outcome_admission import terminal_policy_failure, trusted_environment_outcome
from .qa_metrics import hotpot_evidence_metrics, qa_official_metrics
from .runtime import (
    WORKER_BACKEND_FAILURE_SENTINEL,
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    MultiAgentRuntime,
    artifact_backend_failure_records,
    artifact_integrity_failure_risks,
)
from .skill_store import DirectorSkillBank
from .submission_contract import (
    OutcomeDecision,
    decide_text_outcome,
    is_text_submission_dataset,
    receipt_error,
)
from .unified_contract import SUBMISSION_VERSION


def _deduplicate_backend_request_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        signature = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(dict(event))
    return unique


@dataclass
class AdaptiveSolverResult:
    director_run: DirectorRun
    verification: VerificationResult | None
    trace: ExecutionTrace
    skills_used: tuple[str, ...] = ()
    skill_context: dict[str, Any] = field(default_factory=dict)
    answer_submission: AnswerSubmission | None = None
    outcome_decision: OutcomeDecision | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "director_run": self.director_run.to_dict(),
            "submission_contract_version": SUBMISSION_VERSION,
            "outcome_decision": self.outcome_decision.to_dict() if self.outcome_decision else None,
            "submission_receipt": (self.director_run.submission_receipt.to_dict()
                                   if self.director_run.submission_receipt else None),
            "candidate_output": self.director_run.candidate_output,
            "verification": (
                {
                    "score": self.verification.score,
                    "passed": self.verification.passed,
                    "verifier": self.verification.verifier,
                    "detail": self.verification.detail,
                }
                if self.verification
                else None
            ),
            "trace": self.trace.to_dict(),
            "skills_used": list(self.skills_used),
            "skill_context": self.skill_context,
            "answer_submission": (
                self.answer_submission.to_dict() if self.answer_submission else None
            ),
        }


class AdaptiveWorkflowSolver:
    """Stage-two inference integration; it never updates model parameters."""

    def __init__(
        self,
        *,
        director_backend: ChatBackend,
        runtime: MultiAgentRuntime,
        verifier: Verifier | None = None,
        skillbank: DirectorSkillBank | None = None,
        trace_store: JSONLTraceStore | None = None,
        canvas_config: CanvasConfig | None = None,
        runtime_routes: tuple[str, ...] = (),
        action_registry: DatasetActionRegistry | None = None,
        rollout_deadline: RolloutDeadline | None = None,
        director_enable_thinking: bool | None = None,
        director_tokenizer: Any | None = None,
        worker_cost: Any,
    ) -> None:
        # R2Orch ``cost`` table (CostConfig).
        self.worker_cost = worker_cost
        self.director_backend = director_backend
        self.runtime = runtime
        self.verifier = verifier
        self.skillbank = skillbank
        self.trace_store = trace_store
        self.canvas_config = canvas_config
        self.runtime_routes = tuple(runtime_routes)
        self.action_registry = action_registry or DatasetActionRegistry()
        self.answer_finalizer = AnswerFinalizer()
        self.rollout_deadline = rollout_deadline
        self.director_enable_thinking = director_enable_thinking
        self.director_tokenizer = director_tokenizer
        self.active_canvas: GraphCanvas | None = None

    def _record_worker_cost(self, task: TaskSpec, canvas: GraphCanvas) -> None:
        """Record this rollout's Worker token usage and its normalised cost.

        ``task.metadata["worker_cost_record"]`` carries the full record (dataset, input / output
        tokens, reference, cost); an R2Orch sibling also gets the scalar in its ``r2orch``
        mapping, which is what the Director's batch builder reads.  Always written; it changes no
        reward unless ``cost.lambda`` is positive.
        """

        from .worker_cost import worker_cost_record, worker_usage_from_history

        record = worker_cost_record(
            worker_usage_from_history(canvas.history),
            dataset=str(task.metadata.get("dataset", "")),
            cost=self.worker_cost,
            canvas_total_tokens=canvas.total_tokens,
        )
        task.metadata["worker_cost_record"] = record
        sibling = task.metadata.get("r2orch")
        if isinstance(sibling, dict):
            task.metadata["r2orch"] = {**sibling, "worker_cost": record["worker_cost"]}

    def _apply_skill_arm(
        self,
        sibling: dict[str, Any],
        selected_skills: list[Any],
        skill_context: str,
        skill_manifest: dict[str, Any],
        reference_selected: list[dict[str, Any]],
    ) -> tuple[list[Any], str, dict[str, Any], dict[str, Any]]:
        """Resolve a ``[r2orch.arms]`` sibling against the retrieved ("on") context.

        Card k is the retrieved card with the smallest Thompson draw of the plan
        (``skill_ledger.choose_arm_card``); retrieval is deterministic per snapshot and query, so
        every sibling of the task resolves the same k.  A visible sibling keeps the full context;
        a hidden sibling gets the same cards without k (no other card is pulled in), rendered by
        the bank that rendered the full context.  Without a retrieved card, or when the bank cannot
        reproduce its own context, the sibling runs plainly with the full context and arm "none"
        (the task then adds nothing to the ledger).  Returns the possibly reduced
        (skills, context, manifest) and the arm metadata.
        """

        from .skill_ledger import choose_arm_card

        if isinstance(sibling.get("arm_audit_card"), dict):
            return self._apply_audit_arm(
                sibling, selected_skills, skill_context, skill_manifest, reference_selected
            )
        requested = str(sibling["skill_arm_requested"])
        candidates = [item["id"] for item in reference_selected]
        card = choose_arm_card(reference_selected, sibling.get("arm_draws") or {})
        if card is None:
            reason = "no_retrieved_card" if not reference_selected else "no_planned_draw"
            return selected_skills, skill_context, skill_manifest, {
                "skill_arm": "none",
                "arm_card": None,
                "arm_candidates": candidates,
                "arm_reason": reason,
            }
        arm_meta = {"skill_arm": requested, "arm_card": card, "arm_candidates": candidates}
        if requested == "visible":
            return selected_skills, skill_context, skill_manifest, arm_meta
        remaining = [skill for skill in selected_skills if skill.skill_id != card["id"]]
        hidden_context = self._render_without_card(
            selected_skills, remaining, skill_context, skill_manifest
        )
        if hidden_context is None:
            return selected_skills, skill_context, skill_manifest, {
                "skill_arm": "none",
                "arm_card": None,
                "arm_candidates": candidates,
                "arm_reason": "hidden_context_unavailable",
            }
        manifest = dict(skill_manifest)
        if manifest:
            tokens: int | None = 0
            if hidden_context:
                tokens = (
                    len(self.director_tokenizer.encode(hidden_context, add_special_tokens=False))
                    if self.director_tokenizer is not None
                    else None
                )
            manifest.update(
                selected=[
                    item for item in manifest.get("selected", ()) if str(item["id"]) != card["id"]
                ],
                context=hidden_context,
                prompt_tokens=tokens,
            )
            if "context_sha256" in manifest:
                manifest["context_sha256"] = hashlib.sha256(hidden_context.encode()).hexdigest()
        return remaining, hidden_context, manifest, arm_meta

    def _apply_audit_arm(
        self,
        sibling: dict[str, Any],
        selected_skills: list[Any],
        skill_context: str,
        skill_manifest: dict[str, Any],
        reference_selected: list[dict[str, Any]],
    ) -> tuple[list[Any], str, dict[str, Any], dict[str, Any]]:
        """Audit sibling (``[r2orch.arms] audit_share``): k is a ledger-retired card.

        Retrieval no longer shows a retired card, so the contrast is taken the other way round:
        a hidden sibling keeps the retrieved context, a visible sibling sees it plus k, rendered
        from the archived card record the arm plan carries (``arm_audit_card.record``) in the
        format that rendered the retrieved context (``skill_store.render_skill_context``),
        after the retrieved cards (in addition to them: retrieval's top-k and token budget do not
        apply to k).  Both arms check that this rendering is possible, so either both run the
        audit or both run plainly with arm "none" (the task then adds nothing to the ledger), as
        when retrieval itself shows a card with k's id.
        """

        audit = sibling["arm_audit_card"]
        requested = str(sibling["skill_arm_requested"])
        candidates = [item["id"] for item in reference_selected]

        def plain(reason: str) -> tuple[list[Any], str, dict[str, Any], dict[str, Any]]:
            return selected_skills, skill_context, skill_manifest, {
                "skill_arm": "none",
                "arm_card": None,
                "arm_candidates": candidates,
                "arm_reason": reason,
                "arm_audit": True,
            }

        card = {
            "id": str(audit["id"]),
            "version": int(audit["version"]),
            "key": str(audit["key"]),
            "audit": True,
        }
        if card["id"] in candidates or any(
            str(skill.skill_id) == card["id"] for skill in selected_skills
        ):
            return plain("audit_card_retrieved")
        rendered = self._render_with_card(
            selected_skills, skill_context, skill_manifest, audit.get("record")
        )
        if rendered is None:
            return plain("audit_context_unavailable")
        arm_meta = {
            "skill_arm": requested,
            "arm_card": card,
            "arm_candidates": candidates,
            "arm_audit": True,
        }
        if requested != "visible":
            return selected_skills, skill_context, skill_manifest, arm_meta
        skills, context = rendered
        manifest = dict(skill_manifest)
        manifest.update(
            selected=[
                *manifest.get("selected", ()),
                {"id": card["id"], "version": card["version"]},
            ],
            context=context,
            prompt_tokens=(
                len(self.director_tokenizer.encode(context, add_special_tokens=False))
                if self.director_tokenizer is not None
                else None
            ),
        )
        if "context_sha256" in manifest:
            manifest["context_sha256"] = hashlib.sha256(context.encode()).hexdigest()
        return skills, context, manifest, arm_meta

    def _render_with_card(
        self,
        selected_skills: list[Any],
        skill_context: str,
        skill_manifest: dict[str, Any],
        record: Any,
    ) -> tuple[list[Any], str] | None:
        """``(skills, context)``: the retrieved cards plus the card of ``record``, or None.

        The retrieved context must be exactly the ``render_skill_context`` rendering of the
        manifest's (id, version) list (DirectorSkillBank's format); the extra card is appended
        to it with the version of its record.  Anything else (a manifest that does not match
        the skills, a malformed record) returns None.
        """

        from .skill_store import render_skill_context
        from .skills import SkillCard

        selected = skill_manifest.get("selected") if isinstance(skill_manifest, dict) else None
        if not isinstance(record, dict) or not isinstance(selected, list):
            return None
        try:
            entries = []
            for skill, item in zip(selected_skills, selected, strict=True):
                if str(skill.skill_id) != str(item["id"]):
                    return None
                entries.append((skill, item["version"]))
            if render_skill_context(entries) != skill_context:
                return None
            extra = SkillCard.from_dict(record["card"])
            context = render_skill_context([*entries, (extra, record["version"])])
        except Exception:  # a manifest or record this renderer cannot read
            return None
        return [*selected_skills, extra], context

    def _render_without_card(
        self,
        selected_skills: list[Any],
        remaining: list[Any],
        skill_context: str,
        skill_manifest: dict[str, Any],
    ) -> str | None:
        """Render ``remaining`` with the renderer that reproduces ``skill_context``, else None."""

        renderers = []
        views = self.skillbank._pats_views
        scope = skill_manifest.get("pats_scope")
        if scope in views:
            renderers.append(views[scope].format_prompt_context)
        renderers.append(self.skillbank.format_prompt_context)
        for render in renderers:
            try:
                if render(list(selected_skills)) != skill_context:
                    continue
                return render(list(remaining)) if remaining else ""
            except Exception:  # a renderer that cannot handle these cards is skipped
                continue
        return None

    def finalize_answer(
        self,
        task: TaskSpec,
        raw_answer: str,
        *,
        raw_summary: str = "",
    ) -> AnswerSubmission:
        return self.answer_finalizer.finalize(
            task,
            raw_answer,
            raw_summary=raw_summary,
            verifier=self.verifier,
        )

    def solve(self, task: TaskSpec, *, run_id: str) -> AdaptiveSolverResult:
        try:
            return self._solve(task, run_id=run_id)
        except BaseException:
            # Preparing an environment goal starts a real episode before the
            # Director. Configuration/skill failures must close it too.
            for lifecycle in alfworld_lifecycles(self.runtime.executor.tools):
                lifecycle.close_all()
            raise

    def _solve(self, task: TaskSpec, *, run_id: str) -> AdaptiveSolverResult:
        task.metadata["judge_evaluation_scope"] = f"primary:{run_id}:{self.runtime.seed}"

        action_adapter = self.action_registry.resolve(task)
        lifecycle_tools = self.runtime.executor.tools
        active_alfworld_lifecycles = alfworld_lifecycles(lifecycle_tools)
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            if not active_alfworld_lifecycles:
                raise ValueError("ALFWorld task requires configured ALFWorld Actions")
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.bind_task(task)
            self.runtime.environment_fingerprint = active_alfworld_lifecycles[
                0
            ].environment_fingerprint
        effective_task = task.prompt
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            effective_task = active_alfworld_lifecycles[0].effective_task
        worker_task = (effective_task if action_adapter is not None and action_adapter.adapter_id == "alfworld"
                       else solver_task_text(task))
        skill_manifest = {}
        if self.skillbank:
            selected_skills, skill_context, skill_manifest = self.skillbank.select_context(
                effective_task,
                task_type=task.task_type,
                tokenizer=self.director_tokenizer,
                tools=self.runtime.executor.tools.keys(),
                task_metadata=task.metadata,
            )
        else:
            selected_skills = []
            skill_context = ""
        sibling = task.metadata.get("r2orch")
        if isinstance(sibling, dict) and "skill_arm_requested" in sibling:
            # R2Orch ([r2orch.arms]): retrieval is deterministic per snapshot and query;
            # the hidden arm sees the retrieved context minus card k.  The scaffold condition is
            # "on" when the Director is shown any skill context, else "off".
            reference_context = skill_context
            reference_selected = [
                {"id": str(item["id"]), "version": int(item["version"])}
                for item in skill_manifest.get("selected", ())
            ]
            selected_skills, skill_context, skill_manifest, arm_meta = self._apply_skill_arm(
                sibling, selected_skills, skill_context, skill_manifest, reference_selected
            )
            effective = "on" if skill_context else "off"
            task.metadata["r2orch"] = {
                **sibling,
                "scaffold_condition": effective,
                "scaffold_on_empty": not reference_context,
                "reference_selected": reference_selected,
                **arm_meta,
            }
            if effective == "off":
                selected_skills, skill_context = [], ""
                if skill_manifest:
                    skill_manifest = {
                        **skill_manifest,
                        "selected": [],
                        "context": "",
                        "prompt_tokens": 0,
                        **(
                            {"context_sha256": hashlib.sha256(b"").hexdigest()}
                            if "context_sha256" in skill_manifest
                            else {}
                        ),
                    }
        base_canvas_config = self.canvas_config or CanvasConfig()
        dataset_key, selected_token_budget = base_canvas_config.token_budget_for_dataset(
            task.metadata.get("dataset", "")
        )
        task.metadata["canvas_token_budget"] = {
            "dataset": dataset_key,
            "max_total_tokens": selected_token_budget,
            "fallback_max_total_tokens": base_canvas_config.max_total_tokens,
        }
        canvas = GraphCanvas(
            task=worker_task,
            worker_task=worker_task,
            director_task=effective_task,
            runtime=self.runtime,
            config=replace(base_canvas_config, max_total_tokens=selected_token_budget),
            runtime_routes=self.runtime_routes,
            task_type=task.task_type,
            action_adapter=action_adapter,
            dataset=str(task.metadata.get("dataset", "")),
            rollout_deadline=self.rollout_deadline,
        )
        canvas.run_id = run_id
        if is_aime_dataset(dataset_key):
            # FINISH normalizes only the selected immutable answer. Supply the
            # actual scoring adapter without invoking recovery or a Worker.
            canvas._unified_answer_finalizer = lambda raw: self.finalize_answer(task, raw)
        self.active_canvas = canvas
        # Offline Directors that script their graph per sibling (application.TopologyMockDirector).
        prepare_director = getattr(self.director_backend, "prepare_for_task", None)
        if callable(prepare_director):
            prepare_director(task.metadata)
        try:
            run = GraphDirector(
                backend=self.director_backend,
                canvas=canvas,
                solver_skill_context=skill_context,
                enable_thinking=self.director_enable_thinking,
                tokenizer=self.director_tokenizer,
                call_namespace=run_id,
            ).run()
        except Exception:
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.close_all()
            raise
        receipt = canvas.submission_receipt
        self._record_worker_cost(task, canvas)
        output_artifact = (
            self.runtime.artifacts.get(canvas.graph.output_agent)
            if canvas.graph.output_agent
            else None
        )
        output_artifact = (
            self.runtime.artifacts.get(canvas.graph.output_agent)
            if canvas.graph.output_agent
            else None
        )
        artifact_integrity = {
            agent_id: {
                "claimed_confidence": artifact.claimed_confidence,
                "effective_confidence": artifact.confidence,
                "integrity_risks": list(artifact.integrity_risks),
                "runtime_tool_evidence": dict(artifact.runtime_tool_evidence),
                "alfworld_progress": dict(artifact.alfworld_progress),
            }
            for agent_id, artifact in self.runtime.artifacts.items()
        }
        historical_integrity = {}
        for step in canvas.history:
            if step.execution is None:
                continue
            for artifact in [*step.execution.artifacts.values(),
                             *step.execution.attempt_artifacts]:
                historical_integrity[artifact.artifact_id] = {
                    'agent_id': artifact.agent_id,
                    'backend_failure': artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL,
                    'runtime_tool_evidence': dict(artifact.runtime_tool_evidence),
                }
        task.metadata['worker_artifact_history_integrity'] = historical_integrity
        output_integrity_failure_risks = (
            artifact_integrity_failure_risks(output_artifact) if output_artifact is not None else []
        )
        task.metadata["worker_artifact_integrity"] = artifact_integrity
        task.metadata["worker_output_integrity_risks"] = list(
            output_artifact.integrity_risks if output_artifact is not None else []
        )
        task.metadata["alfworld_output_progress"] = (
            dict(output_artifact.alfworld_progress)
            if output_artifact is not None and output_artifact.alfworld_progress
            else {}
        )
        task.metadata["worker_artifact_integrity_failure"] = (
            {
                "output_agent": output_artifact.agent_id,
                "risks": output_integrity_failure_risks,
                "claimed_confidence": output_artifact.claimed_confidence,
                "effective_confidence": output_artifact.confidence,
            }
            if output_artifact is not None and output_integrity_failure_risks
            else None
        )
        unified_binding_error = receipt_error(receipt, run=run, events=canvas.history,
                                              run_id=run_id, dataset=dataset_key)
        if unified_binding_error:
            receipt = None
            task.metadata["submission_binding_error"] = unified_binding_error
        task.metadata["submission_status"] = "submitted" if receipt else "unsubmitted"
        task.metadata["submission_receipt"] = receipt.to_dict() if receipt else None
        task.metadata["submission_transaction"] = copy.deepcopy(canvas._unified_transaction)
        if action_adapter is not None and action_adapter.adapter_id == "alfworld":
            winning_artifacts = [
                artifact
                for artifact in self.runtime.artifacts.values()
                if artifact.environment_result.get("environment_completed") is True
                and artifact.environment_result.get("won") is True
            ]
            selected_output = canvas.graph.output_agent
            selected_result = (
                dict(output_artifact.environment_result)
                if output_artifact is not None and output_artifact.environment_result
                else active_alfworld_lifecycles[0].result_for(selected_output)
            )
            environment_result = dict(selected_result or {})
            task.metadata["alfworld_unselected_winning_episodes"] = [
                {
                    "agent_id": artifact.agent_id,
                    "environment_result": dict(artifact.environment_result),
                }
                for artifact in sorted(winning_artifacts, key=lambda item: item.agent_id)
                if artifact.agent_id != selected_output
            ]
            task.metadata["alfworld_environment_result"] = environment_result
            for lifecycle in active_alfworld_lifecycles:
                lifecycle.close_all()
        backend_failures = [
            artifact
            for artifact in self.runtime.artifacts.values()
            if artifact.answer == WORKER_BACKEND_FAILURE_SENTINEL
        ]
        backend_request_events = _deduplicate_backend_request_events(
            event
            for artifact in self.runtime.artifacts.values()
            for event in artifact.backend_request_events
        )
        task.metadata["backend_request_events"] = backend_request_events
        stale_output = bool(
            output_artifact is not None
            and output_artifact.answer not in {
                WORKER_BACKEND_FAILURE_SENTINEL, WORKER_PROTOCOL_FAILURE_SENTINEL
            }
            and not canvas.selected_output_is_current()
        )
        task.metadata["output_artifact_binding"] = self.runtime.artifact_input_binding(
            str(canvas.graph.output_agent or "")
        )
        if stale_output:
            task.metadata["output_contract_failure"] = {
                "reason": "stale_output_artifact", "output_agent": canvas.graph.output_agent,
                "artifact_id": output_artifact.artifact_id,
            }
        text_primary = is_text_submission_dataset(dataset_key)
        candidate_output = run.candidate_output
        task.metadata["submission_contract_version"] = "unified_submission"
        task.metadata["candidate_output"] = candidate_output
        candidate_submission = self.finalize_answer(
            task, candidate_output,
            raw_summary=output_artifact.summary if output_artifact else "",
        ) if text_primary else None
        task.metadata["candidate_answer_submission"] = (
            candidate_submission.to_dict() if candidate_submission else None
        )
        if text_primary:
            binding_error = unified_binding_error
            if binding_error:
                receipt = None
                run.output = ""
                answer_submission = AnswerSubmission(
                    raw_answer="", submitted_answer="", method="runtime_submission_gate",
                    changed=False, valid=False, detail=binding_error,
                )
            else:
                answer_submission = AnswerSubmission(
                    raw_answer=receipt.raw_answer_snapshot,
                    submitted_answer=receipt.submitted_answer_snapshot,
                    method=receipt.normalization_version,
                    changed=receipt.raw_answer_snapshot != receipt.submitted_answer_snapshot,
                    valid=True,
                )
            task.metadata["submission_status"] = "submitted" if receipt else "unsubmitted"
            task.metadata["submission_receipt"] = receipt.to_dict() if receipt else None
            task.metadata["diagnostic_qa_metrics"] = (
                qa_official_metrics(dataset_key, candidate_submission.submitted_answer, task.reference)
                if candidate_submission and candidate_submission.valid and not receipt else None
            )
            if dataset_key == "aime" and candidate_submission.valid and receipt is None:
                diagnostic = NumericVerifier().verify(task, candidate_submission.submitted_answer)
                task.metadata["diagnostic_qa_metrics"] = {
                    "schema": "aime_candidate_answer_em", "em": diagnostic.score,
                }
        elif stale_output:
            run.output = ""
            task.metadata["output_contract_failure"] = {
                "reason": "stale_output_artifact", "output_agent": canvas.graph.output_agent,
                "artifact_id": output_artifact.artifact_id,
            }
            task.metadata["outcome_exclusion_reason"] = "stale_output_artifact"
            answer_submission = AnswerSubmission(
                raw_answer="", submitted_answer="", method="runtime_output_gate",
                changed=False, valid=False, detail="stale_output_artifact",
            )
        else:
            answer_submission = self.finalize_answer(
                task, run.output, raw_summary=output_artifact.summary if output_artifact else "",
            )
        if run.output.strip() in {WORKER_PROTOCOL_FAILURE_SENTINEL, WORKER_BACKEND_FAILURE_SENTINEL}:
            answer_submission = replace(answer_submission, submitted_answer="", valid=False,
                                        detail="runtime_failure_sentinel_not_an_answer")
        task.metadata["answer_submission"] = answer_submission.to_dict()
        scored_answer_available = bool(receipt) if text_primary else not stale_output
        task.metadata["qa_token_f1"] = (
            qa_token_f1(task, answer_submission.submitted_answer) if scored_answer_available else None
        )
        task.metadata["qa_official_metrics"] = (
            qa_official_metrics(task.metadata.get("dataset"), answer_submission.submitted_answer,
                                task.reference) if scored_answer_available else None
        )
        if (task.metadata["qa_official_metrics"] or {}).get("schema") == "hotpot_official_answer":
            task.metadata["qa_official_metrics"]["evidence"] = hotpot_evidence_metrics(
                answer_submission.raw_answer, task.private_verifier_payload.get("supporting_facts"),
                task.metadata["qa_official_metrics"],
            )
        terminal_failure = terminal_policy_failure(
            str(task.metadata.get("dataset", "")),
            terminal=not canvas.active,
            rejection_codes=[step.rejection_code for step in canvas.history if step.rejection_code],
            artifacts=artifact_integrity,
            historical_artifacts=historical_integrity,
            output_agent=canvas.graph.output_agent,
            rounds=canvas.round_index,
            max_rounds=canvas.director_round_limit,
            worker_tokens=canvas.total_tokens,
            worker_token_limit=canvas.config.max_total_tokens,
            infrastructure_failure=bool(
                stale_output or backend_failures or task.metadata.get("output_contract_failure")
            ),
        )
        task.metadata["runtime_terminal_policy_failure"] = terminal_failure
        trusted_environment = any(
            trusted_environment_outcome(dataset, task.metadata.get(key))
            for dataset, key in (
                ("alfworld", "alfworld_environment_result"),
            )
        )
        outcome_decision = None
        if text_primary:
            verification = None
            scoring_error = ""
            if receipt is not None and self.verifier is not None:
                # Retry only the immutable submitted answer, never the Worker graph.
                for attempt in range(2):
                    try:
                        verification = self.verifier.verify(task, receipt.submitted_answer_snapshot)
                        break
                    except Exception as exc:
                        scoring_error = "scoring_exception:" + type(exc).__name__
                        task.metadata.setdefault("submission_scoring_errors", []).append(
                            {"attempt": attempt + 1, "error_type": type(exc).__name__,
                             "receipt_ref": receipt.receipt_ref}
                        )
            outcome_decision = decide_text_outcome(
                receipt=receipt, verification=verification,
                terminal_failure=terminal_failure if receipt is None else None,
                reason=scoring_error or (
                    "execution_token_budget_overrun"
                    if receipt is None
                    and canvas.total_tokens > canvas.config.max_total_tokens
                    else binding_error if receipt is None else ""
                ),
            )
            task.metadata["outcome_decision"] = outcome_decision.to_dict()
        elif receipt is None:
            verification = None
            if terminal_failure is not None:
                verification = VerificationResult(0.0, False, "runtime_policy_terminal",
                    json.dumps(terminal_failure, ensure_ascii=False, sort_keys=True))
            else:
                task.metadata["outcome_exclusion_reason"] = unified_binding_error or "unsubmitted_unknown"
        elif backend_failures:
            # Backend availability is not task correctness. Keep the trace for
            # diagnosis, but do not verify the sentinel or commit a zero reward.
            # The rollout runner will leave this ID missing.
            failure_details = [
                {**record, "agent_id": artifact.agent_id}
                for artifact in backend_failures
                for record in artifact_backend_failure_records(artifact)
            ]
            task.metadata["worker_backend_failure"] = {
                "count": len(backend_failures),
                "agents": sorted(artifact.agent_id for artifact in backend_failures),
                "routes": sorted(
                    {artifact.model_route or "unassigned" for artifact in backend_failures}
                ),
                "failure_types": sorted(
                    {str(record.get("kind", "unknown")) for record in failure_details}
                ),
                "failure_details": failure_details,
                "request_events": backend_request_events,
                "retryable": any(bool(record.get("retryable")) for record in failure_details),
                "counts_toward_route_circuit": any(
                    bool(record.get("counts_toward_route_circuit")) for record in failure_details
                ),
                "disable_route": any(
                    bool(record.get("disable_route")) for record in failure_details
                ),
            }
            # A later backend failure cannot erase an already committed
            # official environment result. Keep the incident as a separate
            # training exclusion; never score the sentinel.
            verification = (
                self.verifier.verify(task, answer_submission.submitted_answer)
                if trusted_environment and self.verifier
                else None
            )

        elif stale_output:
            verification = None
        elif (
            terminal_failure
            and not trusted_environment
            and (
                not answer_submission.valid
                or canonical_dataset_name(task.metadata.get("dataset", ""))
                in {"alfworld"}
            )
        ):
            verification = VerificationResult(
                0.0,
                False,
                "runtime_policy_terminal",
                json.dumps(terminal_failure, ensure_ascii=False, sort_keys=True),
            )
        else:
            verification = (
                self.verifier.verify(task, answer_submission.submitted_answer)
                if self.verifier
                else None
            )
        if not text_primary:
            outcome_decision = decide_text_outcome(
                receipt=receipt, verification=verification,
                terminal_failure=terminal_failure if receipt is None else None,
                reason=unified_binding_error or "",
            )
            task.metadata["outcome_decision"] = outcome_decision.to_dict()
        # Model selection is a Director action trained from final graph reward.
        # Never score a local responsibility against the original task answer.
        trace = trace_from_canvas(
            run_id=run_id,
            task=task,
            canvas=canvas,
            verification=verification,
        )
        if self.trace_store:
            self.trace_store.append(trace)
        return AdaptiveSolverResult(
            director_run=run,
            verification=verification,
            trace=trace,
            skills_used=tuple(skill.skill_id for skill in selected_skills),
            skill_context=skill_manifest,
            answer_submission=answer_submission,
            outcome_decision=outcome_decision,
        )
