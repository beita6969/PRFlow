from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .actions import POLICY_PARSE_ERROR_CODES
from .canvas import CanvasState, GraphCanvas
from .director_timeline import (
    HISTORY_THINKING_VISIBILITY,
    TIMELINE_CONTEXT_MODES,
    director_context_mode,
    timeline_assistant_content,
    timeline_prefix_audit,
)
from .llm import (
    BinaryChoiceUnavailable,
    ChatBackend,
    DirectorContextExhausted,
    MockBackend,
    director_recovery_budget,
)
from .submission_contract import (
    SubmissionReceipt,
    _director_call_context,
    is_text_submission_dataset,
)

# Per-action ceiling. The gateway separately admits the exact templated prompt
# plus completion within the service context; a constant alone cannot ensure fit.
DIRECTOR_ACTION_MAX_TOKENS = 1000
DIRECTOR_PROMPT_MAX_TOKENS = 1000
DIRECTOR_INVALID_ACTION = '{"action":"invalid"}'
DIRECTOR_CONTEXT_SCHEMA = "task_once_action_feedback_history_with_thinking"

DIRECTOR_PROMPT_VARIANTS = frozenset({"r2orch"})


def director_prompt_components(variant: str) -> tuple[str, dict[str, str]]:
    """Base prompt and problem-type hints."""

    normalized = str(variant).strip().casefold()
    if normalized not in DIRECTOR_PROMPT_VARIANTS:
        raise ValueError(
            f"unsupported Director prompt variant {variant!r}; "
            f"use one of {sorted(DIRECTOR_PROMPT_VARIANTS)}"
        )
    from .unified_contract import DIRECTOR_HINTS, DIRECTOR_PROMPT

    return DIRECTOR_PROMPT, DIRECTOR_HINTS


_PROBLEM_TYPE_BY_DATASET = {
    "aime": "math",
    "nq": "retrieval_qa",
    "nq_open": "retrieval_qa",
    "natural_questions": "retrieval_qa",
    "hotpotqa": "retrieval_qa",
    "alfworld": "environment",
    "medqa": "retrieval_qa",
    "mbppplus": "code_function",
}

_PROBLEM_TYPE_BY_ADAPTER = {
    "aime": "math",
    "retrieval_qa": "retrieval_qa",
    "alfworld": "environment",
    "mbppplus": "code_function",
}

_PROBLEM_TYPE_BY_TASK_TYPE = {
    "math": "math",
    "mathematical_reasoning": "math",
    "exact_reasoning": "math",
    "retrieval_qa": "retrieval_qa",
    "open_qa": "retrieval_qa",
    "environment": "environment",
    "stateful_environment": "environment",
    "alfworld": "environment",
    "multi_hop_qa": "retrieval_qa",
    "factual_qa": "retrieval_qa",
    "medical_mcq": "retrieval_qa",
    "code_function": "code_function",
}


def render_skill_system_suffix(context: str) -> str:
    """System-message suffix carrying scaffold cards."""

    if not context:
        return ""
    return "\n## Optional Orchestration Knowledge\n" + context + "\n"


def infer_director_problem_type(
    *,
    dataset: str = "",
    task_type: str = "",
    action_adapter_id: str = "",
) -> str:
    """Infer a coarse reusable prompt type without exposing a dataset policy."""

    adapter_key = str(action_adapter_id or "").strip().casefold()
    dataset_key = str(dataset or "").strip().casefold()
    task_key = str(task_type or "").strip().casefold()
    return (
        _PROBLEM_TYPE_BY_ADAPTER.get(adapter_key)
        or _PROBLEM_TYPE_BY_DATASET.get(dataset_key)
        or _PROBLEM_TYPE_BY_TASK_TYPE.get(task_key)
        or "general"
    )


@dataclass
class DirectorTurn:
    round_index: int
    model_action: str
    feedback: str
    accepted: bool
    graph_version: int
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    trainable: bool = True
    rejection_code: str | None = None
    action_diagnostics: dict[str, Any] = field(default_factory=dict)
    turn_kind: str = "graph_action"
    relation_decision: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw_reasoning_text: str = ""
    raw_action_text: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    action_character_span: tuple[int, int] | None = None
    model_id: str = ""
    route_name: str = ""
    thinking_requested: bool | None = None
    thinking_effective: bool = False
    token_provenance: str = "unavailable"
    trajectory_schema: str = "director_trajectory_raw_policy_calls"

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "model_action": self.model_action,
            "feedback": self.feedback,
            "accepted": self.accepted,
            "graph_version": self.graph_version,
            "prompt_messages": list(self.prompt_messages),
            "trainable": self.trainable,
            "rejection_code": self.rejection_code,
            "action_diagnostics": dict(self.action_diagnostics),
            "turn_kind": self.turn_kind,
            "relation_decision": dict(self.relation_decision),
            "call_id": self.call_id,
            "raw_reasoning_text": self.raw_reasoning_text,
            "raw_action_text": self.raw_action_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "behavior_log_probs": list(self.behavior_log_probs),
            "action_character_span": list(self.action_character_span)
            if self.action_character_span
            else None,
            "model_id": self.model_id,
            "route_name": self.route_name,
            "thinking_requested": self.thinking_requested,
            "thinking_effective": self.thinking_effective,
            "token_provenance": self.token_provenance,
            "trajectory_schema": self.trajectory_schema,
        }


@dataclass
class DirectorRun:
    task: str
    finished: bool
    output: str
    graph: dict[str, Any]
    turns: list[DirectorTurn] = field(default_factory=list)
    candidate_output: str = ""
    submission_receipt: SubmissionReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "finished": self.finished,
            "output": self.output,
            "candidate_output": self.candidate_output,
            "submission_receipt": self.submission_receipt.to_dict() if self.submission_receipt else None,
            "graph": self.graph,
            "turns": [turn.to_dict() for turn in self.turns],
        }


class GraphDirector:
    """Multi-turn Canvas driver adapted from FlowSteer's interactive workflow loop."""

    def __init__(
        self,
        *,
        backend: ChatBackend,
        canvas: GraphCanvas,
        role: str = "graph-director",
        solver_skill_context: str = "",
        enable_thinking: bool | None = None,
        tokenizer: Any | None = None,
        call_namespace: str = "",
    ) -> None:
        self.context_mode = director_context_mode()
        self.context_schema = (
            "append_only_action_feedback_history_with_thinking"
            if self.context_mode in TIMELINE_CONTEXT_MODES
            else DIRECTOR_CONTEXT_SCHEMA
        )
        self.backend = backend
        self.canvas = canvas
        self.role = role
        self.solver_skill_context = solver_skill_context
        self.prompt_variant = "r2orch"
        self.enable_thinking = enable_thinking
        self.relation_token_ids = _binary_relation_token_ids(tokenizer)
        self.relation_tokenizer_attestation = _tokenizer_attestation(tokenizer)
        self.tokenizer = tokenizer
        self.call_namespace = str(call_namespace).strip() or str(self.canvas.runtime.seed)
        if not self.canvas.run_id:
            self.canvas.run_id = self.call_namespace

    def run(self) -> DirectorRun:
        self._completed_turns: list[DirectorTurn] = []
        try:
            return self._run()
        except Exception as exc:
            # Preserve completed policy tokens and factual state, never execute
            # an incomplete/unknown response or convert an exception to success.
            exc.partial_state = {
                **(getattr(exc, "partial_state", None) or {}),
                "director_turns": [turn.to_dict() for turn in self._completed_turns],
                "canvas_state": self.canvas.state.value,
                "round_index": self.canvas.round_index,
                "graph": self.canvas.graph.to_dict(),
                "history": [step.to_dict() for step in self.canvas.history],
                "deadline": (
                    self.canvas.rollout_deadline.diagnostics()
                    if self.canvas.rollout_deadline is not None
                    else None
                ),
            }
            raise

    def _run(self) -> DirectorRun:
        feedback = "Graph is empty."
        turns = self._completed_turns
        problem_type = infer_director_problem_type(
            dataset=self.canvas.dataset,
            task_type=self.canvas.task_type,
            action_adapter_id=(
                self.canvas.action_adapter.adapter_id
                if self.canvas.action_adapter is not None
                else ""
            ),
        )
        base_prompt, problem_type_hints = director_prompt_components(self.prompt_variant)
        base_prompt = base_prompt.replace(
            "at most 20 Director turns",
            f"at most {self.canvas.config.max_rounds} Director turns",
        )
        hint = problem_type_hints.get(problem_type) or problem_type_hints["general"]
        system_prompt = base_prompt.rstrip() + "\n\n" + hint.strip() + "\n"
        # The per-turn conversation is deliberately fresh, so persistent
        # Solver-only orchestration context belongs in the system message.
        system_prompt += render_skill_system_suffix(self.solver_skill_context)
        system_message = {"role": "system", "content": system_prompt}
        # Keep the policy history without replaying obsolete control snapshots.
        # Every prior sampled thinking/action and its factual environment feedback remain
        # in the conversation; the immutable task and current authoritative
        # snapshot are each supplied exactly once in snapshot mode. Append-only
        # mode also retains original control messages.
        history_turns: list[tuple[str, str]] = []
        chronological_messages: list[dict[str, str]] = [dict(system_message)]
        previous_policy_ids: tuple[int, ...] = ()
        timeline_chain_valid = True

        def prepare_timeline_turn(messages, prompt_ids, completion_ids):
            nonlocal previous_policy_ids, timeline_chain_valid
            if self.context_mode not in TIMELINE_CONTEXT_MODES:
                return {}
            audit = timeline_prefix_audit(previous_policy_ids, prompt_ids, completion_ids)
            assistant = timeline_assistant_content(self.tokenizer, prompt_ids, completion_ids)
            timeline_chain_valid = timeline_chain_valid and audit["timeline_merge_candidate"]
            audit["timeline_merge_candidate"] = timeline_chain_valid
            chronological_messages[:] = [dict(m) for m in messages]
            chronological_messages.append({"role": "assistant", "content": assistant})
            previous_policy_ids = tuple(prompt_ids) + tuple(completion_ids)
            return audit

        def prompt_messages_for(current_user_content: str) -> list[dict[str, str]]:
            if self.context_mode in TIMELINE_CONTEXT_MODES:
                return [
                    *[dict(m) for m in chronological_messages],
                    {"role": "user", "content": current_user_content},
                ]
            messages = [dict(system_message)]
            if history_turns:
                messages.append({"role": "user", "content": f"Task:\n{self.canvas.director_task}"})
                for index, (raw_action_text, prior_feedback) in enumerate(history_turns):
                    messages.append({"role": "assistant", "content": raw_action_text})
                    if index + 1 < len(history_turns):
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Canvas feedback:\n{prior_feedback}",
                            }
                        )
            return [*messages, {"role": "user", "content": current_user_content}]

        def current_user_prefix() -> str:
            return "" if history_turns else f"Task:\n{self.canvas.director_task}\n\n"

        responsibility_failures = 0
        responsibility_issue: dict[str, Any] = {}
        protocol_error: dict[str, Any] = {}
        responsibility_rejections = {
            "responsibility_violation",
            "duplicate_responsibility",
        }
        observed_turns = 0
        stalled_turns = 0
        seen_progress_states = {self.canvas.director_progress_signature()}
        submission_recoveries = len(self.canvas._unified_submission_recoveries)
        while True:
            # Count completed policy calls, including accepted semantic no-ops.
            # Feedback, inference activity and automatic recovery are not progress.
            current_signature = self.canvas.director_progress_signature()
            if len(turns) != observed_turns:
                repeated = current_signature in seen_progress_states
                current_recoveries = len(self.canvas._unified_submission_recoveries)
                if current_recoveries > submission_recoveries:
                    repeated = False
                submission_recoveries = current_recoveries
                stalled_turns = stalled_turns + 1 if repeated else 0
                seen_progress_states.add(current_signature)
                observed_turns = len(turns)
            if not self.canvas.active:
                self.canvas.terminate_round_limit_without_graph_repair()
                break
            # Worker execution is synchronous with Canvas.step: by this boundary
            # there are no in-flight Workers to wait for. Pending configuration is
            # represented by legal SET_PROMPT/SET_MODEL/relation actions.
            if stalled_turns >= 4:
                # Complete model-authored invalid objects have known protocol
                # attribution. Missing/truncated responses retain the existing
                # unknown path; no answer or infrastructure failure is invented.
                protocol_exhausted = all(
                    turn.rejection_code in POLICY_PARSE_ERROR_CODES
                    and turn.action_diagnostics.get("json_objects_found", 0) > 0
                    and turn.action_diagnostics.get("finish_reason") in {None, "stop", "end_turn"}
                    for turn in turns[-4:]
                )
                feedback = self.canvas.terminate_director_stall(
                    "director_action_protocol_exhausted" if protocol_exhausted
                    else "director_no_progress_exhausted"
                ).feedback
                break
            if not self.canvas.control_snapshot()["allowed_actions"]:
                feedback = self.canvas.terminate_director_stall(
                    "director_no_legal_continuation"
                ).feedback
                break
            if stalled_turns == 3:
                feedback += (
                    "\nBounded recovery: this is the last call without effective progress. "
                    "Choose one legal action from the snapshot. Keep reasoning concise; "
                    "return a complete JSON action. Workers solve the task."
                )
            if self.canvas.rollout_deadline is not None:
                self.canvas.rollout_deadline.check("director_turn_start")
            if self.canvas.state is CanvasState.AWAITING_RELATION_CHOICE:
                pending = self.canvas.pending_relation_decision
                assert pending is not None
                binary_messages = prompt_messages_for(
                    current_user_prefix()
                    + "Authoritative Canvas control snapshot:\n"
                    + _snapshot_text(self.canvas.control_snapshot())
                    + "\n\nCanvas feedback:\n"
                    + feedback
                    + "\n\nChoose off or on for this relation."
                )
                try:
                    choose_binary = self.backend.choose_binary
                    if self.relation_token_ids is None:
                        raise BinaryChoiceUnavailable(
                            "the active Director tokenizer does not encode off/on as one token each"
                        )
                    binary = choose_binary(
                        binary_messages,
                        role=self.role,
                        token_ids=self.relation_token_ids,
                    )
                except DirectorContextExhausted:
                    self.canvas.terminate_context_limit_without_graph_repair()
                    break
                except (AttributeError, BinaryChoiceUnavailable, TypeError, ValueError) as exc:
                    abandoned = self.canvas.abandon_relation_choice(str(exc))
                    feedback = abandoned.feedback
                    continue
                audit = binary.to_policy_audit()
                audit.update(
                    {
                        "behavior_policy_version": "relation_binary",
                        "tokenizer_attestation": dict(self.relation_tokenizer_attestation),
                        "rollout_seed": int(self.canvas.runtime.seed),
                    }
                )
                timeline_audit = prepare_timeline_turn(
                    binary_messages,
                    binary.prompt_token_ids,
                    (binary.token_ids[binary.choice],),
                )
                step = self.canvas.resolve_relation_choice(binary.choice, policy_audit=audit)
                feedback = step.feedback
                history_turns.append((binary.choice, step.feedback))
                turns.append(
                    DirectorTurn(
                        round_index=step.round_index,
                        model_action=binary.choice,
                        feedback=step.feedback,
                        accepted=step.accepted,
                        graph_version=self.canvas.graph.version,
                        prompt_messages=binary_messages,
                        # The sampled off/on token is a genuine policy action even
                        # when Canvas later rejects the requested graph mutation
                        # (for example, a remaining-time admission gate).  As with
                        # rejected JSON actions below, factual rejection is part of
                        # the environment response; it must not erase the action or
                        # make an otherwise exact trajectory ineligible.
                        trainable=True,
                        rejection_code=step.rejection_code,
                        action_diagnostics={
                            "action_type": "relation_choice",
                            "no_progress_streak_before_call": stalled_turns,
                            "bounded_recovery_call": stalled_turns == 3,
                            "generated_action": True,
                            "director_context_schema": self.context_schema,
                            "history_thinking_visibility": HISTORY_THINKING_VISIBILITY,
                            "timeline_prefix_audit": timeline_audit,
                            "binary_policy_audit": audit,
                            "cleaned_action_chars": len(binary.choice),
                            "backend_request_events": binary.metadata.get(
                                "backend_request_events", []
                            ),
                        },
                        turn_kind="relation_choice",
                        relation_decision=dict(step.relation_decision),
                        call_id=f"{self.call_namespace}:{len(turns)}:relation",
                        raw_action_text=binary.choice,
                        prompt_token_ids=tuple(binary.prompt_token_ids),
                        completion_token_ids=(int(binary.token_ids[binary.choice]),),
                        behavior_log_probs=(float(binary.log_probabilities[binary.choice]),),
                        action_character_span=(0, len(binary.choice)),
                        model_id=binary.model,
                        route_name=str(binary.metadata.get("route", "")),
                        thinking_requested=False,
                        thinking_effective=False,
                        token_provenance=(
                            "mock_text"
                            if isinstance(self.backend, MockBackend)
                            else "provider_prompt_and_binary_token_ids_and_logprobs"
                        ),
                    )
                )
                continue
            if responsibility_failures == 1 and self.canvas.pending_agent_id:
                issue_field = str(responsibility_issue.get("field") or "violating field")
                issue_code = str(responsibility_issue.get("code") or "responsibility_violation")
                if issue_code == "prompt_action_schema":
                    retry_instruction = (
                        "Repair the JSON syntax or required action fields, preserving the "
                        "intended responsibility. Return one complete JSON object with all "
                        "four short fields as properly quoted and escaped strings. "
                        "A schema rejection does not by itself indicate an unsafe responsibility."
                    )
                elif issue_code == "duplicate_responsibility":
                    conflicting_agent = str(
                        responsibility_issue.get("details", {}).get(
                            "conflicting_agent_id", "an existing Agent"
                        )
                    )
                    retry_instruction = (
                        f"The responsibility overlaps {conflicting_agent}. Keep the task "
                        "subject, but make this Agent's contribution genuinely distinct: "
                        "narrow it to a non-overlapping implementation scope or assign "
                        "diagnosis, testing, review, or synthesis with a distinct deliverable. "
                        "Do not merely rename the role or paraphrase the same ownership."
                    )
                else:
                    retry_instruction = (
                        "Preserve the original task subject and intended contribution; rewrite "
                        "only that field and keep the other fields semantically unchanged. "
                        "Task entities and goals are allowed, but answers, procedural solution "
                        "steps, and explicit Action control are not."
                    )
                user_content = (
                    "Your previous SET_PROMPT was rejected. Retry only the currently required "
                    f"SET_PROMPT for {self.canvas.pending_agent_id}. Use the four short fields "
                    "role, objective, scope, and expected_output. The rejection was "
                    f"{issue_code} in {issue_field}. {retry_instruction}\n\n"
                    + current_user_prefix()
                    + "Authoritative Canvas control snapshot:\n"
                    + _snapshot_text(self.canvas.control_snapshot())
                    + "\n\nCanvas feedback:\n"
                    + feedback
                )
            else:
                user_content = (
                    current_user_prefix() + "Authoritative Canvas control snapshot:\n"
                    f"{_snapshot_text(self.canvas.control_snapshot())}\n\n"
                    f"Canvas feedback:\n{feedback}\n\n"
                    "Return the next single JSON action."
                )
            if protocol_error:
                user_content += (
                    "\n\nAction encoding failure (separate from responsibility or graph legality):\n"
                    + json.dumps(protocol_error, ensure_ascii=False, sort_keys=True)
                    + "\nThe previous response executed no graph action. Use the current "
                    "allowed_actions and action_field_requirements. Correct the encoding or "
                    "field structure and return exactly one object, with no prose or examples. "
                    "This is a normal Director turn within the existing budget."
                )
            self.canvas.observe_submission_candidates(f"{self.call_namespace}:{len(turns)}:action")
            prompt_messages = prompt_messages_for(user_content)
            awaiting_prompt = self.canvas.state is CanvasState.AWAITING_PROMPT
            try:
                with director_recovery_budget(stalled_turns == 3):
                    response = self.backend.generate(
                        prompt_messages,
                        role=self.role,
                        max_tokens=(
                            DIRECTOR_PROMPT_MAX_TOKENS
                            if awaiting_prompt
                            else DIRECTOR_ACTION_MAX_TOKENS
                        ),
                        enable_thinking=self.enable_thinking,
                    )
            except DirectorContextExhausted:
                self.canvas.terminate_context_limit_without_graph_repair()
                break
            if self.canvas.rollout_deadline is not None:
                self.canvas.rollout_deadline.check("director_turn_response")
            raw_reasoning = str(response.raw_reasoning_text or "")
            raw_action = str(
                response.raw_action_text or ("" if raw_reasoning else response.text) or ""
            )
            raw_policy_text = raw_reasoning + raw_action
            # The complete reasoning+action completion remains the trainable
            # policy trajectory. The typed-action parser consumes only the
            # provider-authored action channel (or the exact suffix after an
            # inline Qwen </think> tag), so JSON examples mentioned in thought
            # cannot be mistaken for extra Canvas actions.
            parsed = self.canvas.parser.parse_policy_output(raw_action)
            # Canvas receives only the uniquely parsed typed action.  The marker is
            # deliberately invalid and cannot mutate the graph; the authoritative
            # raw policy response remains stored separately below.
            action_text = parsed.action_text or DIRECTOR_INVALID_ACTION
            completion_ids = tuple(int(value) for value in response.completion_token_ids)
            mock_policy = bool(response.metadata.get("mock"))
            if not completion_ids and mock_policy and self.tokenizer is not None:
                chat_encoder = getattr(self.tokenizer, "encode_chat_trajectory", None)
                if callable(chat_encoder):
                    policy_ids, _policy_mask, policy_spans = chat_encoder(
                        [
                            *[dict(message) for message in prompt_messages],
                            {"role": "assistant", "content": raw_policy_text},
                        ]
                    )
                    if not policy_spans:
                        raise ValueError(
                            "rollout-time chat tokenizer did not expose the policy completion"
                        )
                    completion_ids = tuple(
                        int(value) for value in policy_ids[slice(*policy_spans[-1])]
                    )
                else:
                    completion_ids = tuple(
                        int(value)
                        for value in self.tokenizer.encode(
                            raw_policy_text, add_special_tokens=False
                        )
                    )
            behavior_log_probs = tuple(float(value) for value in response.behavior_log_probs)
            exact_behavior = bool(
                completion_ids
                and behavior_log_probs
                and len(completion_ids) == len(behavior_log_probs)
            )
            training_eligible = bool(response.training_eligible and (exact_behavior or mock_policy))
            timeline_audit = prepare_timeline_turn(
                prompt_messages,
                response.prompt_token_ids,
                completion_ids,
            )
            diagnostics = {
                "no_progress_streak_before_call": stalled_turns,
                "bounded_recovery_call": stalled_turns == 3,
                "raw_output_chars": len(raw_policy_text),
                "backend_request_events": response.metadata.get("backend_request_events", []),
                "qwen_thinking_boundary_restored": response.metadata.get(
                    "qwen_thinking_boundary_restored", False
                ),
                "qwen_thinking_boundary_source": response.metadata.get(
                    "qwen_thinking_boundary_source"
                ),
                "json_objects_found": parsed.candidate_count,
                "director_action_protocol_version": "director_action_json",
                "action_type": parsed.action.action_type.value,
                "parse_error_code": parsed.action.parse_error_code,
                "parse_error_details": dict(parsed.action.parse_error_details),
                "initial_token_out": response.token_out,
                "finish_reason": response.metadata.get("finish_reason"),
                "repair_attempted": parsed.repair_applied,
                "repair_succeeded": False,
                "generated_action": True,
                "behavior_logprobs_exact": exact_behavior,
                "trajectory_training_eligible": training_eligible,
                "director_context_schema": self.context_schema,
                "history_thinking_visibility": HISTORY_THINKING_VISIBILITY,
                "timeline_prefix_audit": timeline_audit,
            }
            parsed_model_action = parsed.action
            diagnostics["model_expected_version"] = parsed_model_action.expected_version
            diagnostics["bound_canvas_version"] = self.canvas.graph.version
            # Pass the already parsed typed action (valid or invalid) so Canvas
            # reports the exact parser failure instead of reparsing a synthetic
            # placeholder. The raw policy response remains separately immutable.
            call_id = f"{self.call_namespace}:{len(turns)}:action"
            step = self.canvas.step(
                parsed.action, authoritative_director=True,
                director_context=_director_call_context(self.canvas.run_id, call_id),
            )
            diagnostics["repair_succeeded"] = bool(parsed.repair_applied and step.accepted)
            feedback = step.feedback
            protocol_error = (
                {"code": step.rejection_code, **parsed.action.parse_error_details}
                if step.rejection_code in POLICY_PARSE_ERROR_CODES else {}
            )
            history_turns.append((raw_policy_text, step.feedback))
            turns.append(
                DirectorTurn(
                    round_index=step.round_index,
                    model_action=action_text,
                    feedback=step.feedback,
                    accepted=step.accepted,
                    graph_version=self.canvas.graph.version,
                    prompt_messages=prompt_messages,
                    trainable=training_eligible,
                    rejection_code=step.rejection_code,
                    action_diagnostics=diagnostics,
                    relation_decision=dict(step.relation_decision),
                    call_id=call_id,
                    raw_reasoning_text=raw_reasoning,
                    raw_action_text=raw_action,
                    prompt_token_ids=tuple(response.prompt_token_ids),
                    completion_token_ids=completion_ids,
                    behavior_log_probs=behavior_log_probs,
                    action_character_span=parsed.character_span,
                    model_id=response.model,
                    route_name=str(response.metadata.get("route", "")),
                    thinking_requested=response.metadata.get("enable_thinking"),
                    thinking_effective=bool(
                        raw_reasoning or re.search(r"<think>.*?</think>", raw_action, re.DOTALL)
                    ),
                    token_provenance=response.token_provenance,
                )
            )
            if step.rejection_code in responsibility_rejections:
                responsibility_issue = dict(step.responsibility_issue)
                if self.canvas.pending_agent_id:
                    responsibility_failures += 1
                else:
                    # A rejected rewrite of an already configured Agent must not enter
                    # the mandatory-prompt recovery path: there is no pending Agent to
                    # recover, and referring to it as ``None`` terminates valid graphs.
                    responsibility_failures = 0
                    responsibility_issue = {}
            elif step.accepted or step.rejection_code in POLICY_PARSE_ERROR_CODES:
                responsibility_failures = 0
                responsibility_issue = {}
            if step.topology_edits_frozen:
                continue
            # Loop once more even when this model turn exhausted max_rounds so
            # Canvas can record budget termination without repairing the graph.
        output = ""
        candidate_output = ""
        if self.canvas.graph.output_agent:
            selected = self.canvas.runtime.artifacts.get(self.canvas.graph.output_agent)
            candidate_output = selected.answer if selected else ""
        if self.canvas.graph.output_agent and self.canvas.selected_output_is_current():
            artifact = self.canvas.runtime.artifacts.get(self.canvas.graph.output_agent)
            if artifact is not None:
                output = artifact.answer
        if is_text_submission_dataset(self.canvas.dataset):
            output = (self.canvas.submission_receipt.raw_answer_snapshot
                      if self.canvas.submission_receipt else "")
        return DirectorRun(
            task=self.canvas.task,
            finished=self.canvas.state.value == "finished",
            output=output,
            graph=self.canvas.graph.to_dict(),
            turns=turns,
            candidate_output=candidate_output,
            submission_receipt=self.canvas.submission_receipt,
        )


def _snapshot_text(snapshot: dict[str, Any]) -> str:
    import json

    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _binary_relation_token_ids(tokenizer: Any | None) -> dict[str, int] | None:
    """Attest at startup that each constrained choice is exactly one policy token."""

    if tokenizer is None:
        return None
    resolved: dict[str, int] = {}
    for choice in ("off", "on"):
        token_ids = list(tokenizer.encode(choice, add_special_tokens=False))
        if len(token_ids) != 1:
            return None
        resolved[choice] = int(token_ids[0])
    if resolved["off"] == resolved["on"]:
        return None
    return resolved


def _tokenizer_attestation(tokenizer: Any | None) -> dict[str, Any]:
    if tokenizer is None:
        return {}
    underlying = getattr(tokenizer, "tokenizer", tokenizer)
    name = str(getattr(underlying, "name_or_path", type(underlying).__qualname__))
    chat_template = str(getattr(underlying, "chat_template", "") or "")
    payload = (
        f"{type(underlying).__module__}.{type(underlying).__qualname__}\n{name}\n{chat_template}"
    )
    return {
        "name_or_path": name,
        "implementation": f"{type(underlying).__module__}.{type(underlying).__qualname__}",
        "chat_template_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
