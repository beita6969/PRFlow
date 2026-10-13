"""Shared result assessment and zero-Worker submission for every task adapter."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
from contextlib import contextmanager
import fcntl

from .actions import ActionType, UNIFIED_ACTION_FIELDS
from .submission_contract import DirectorCallContext, _issue_receipt, answer_hash, snapshot_hash
from .unified_contract import ACTION_PROTOCOL, PROTOCOL, SUBMISSION_VERSION, is_task_result


class SubmissionJournal:
    """Durable intent before any externally visible commit. Unknown is never retried blindly."""

    def __init__(self, directory: str, run_id: str):
        self.path = Path(directory) / (snapshot_hash(run_id) + ".json")

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class UnifiedSubmissionMixin:
    _DIRECTOR_EDIT_ACTIONS = frozenset({
        ActionType.ADD_AGENT, ActionType.SET_PROMPT, ActionType.SET_MODEL,
        ActionType.SET_LAYER, ActionType.SET_RELATION, ActionType.REMOVE_RELATION,
        ActionType.DELETE_AGENT,
    })

    def _init_unified(self):
        self.director_edits_used = 0
        self._submission_lock = threading.RLock()
        self._unified_recovery_used = 0
        self._unified_transaction = None
        self._unified_observed_snapshots = {}
        self._unified_seen_states = set()
        self._unified_no_progress = 0
        self._unified_previous_submit_ready = False
        self._unified_submission_recoveries = set()
        self._unified_answer_finalizer = None

    def director_edit_budget(self):
        # Edits are counted, not limited.
        return {"used": self.director_edits_used, "max": None, "remaining": None,
                "counting": "successful_graph_mutations_including_initial_construction",
                "reset_on_node_edit": False}

    def _record_unified_progress(self, *, accepted):
        signature = self.director_progress_signature()
        repeated = signature in self._unified_seen_states
        ready = any(self.submission_assessment(target)["submit_ready"] for target in self.graph.nodes)
        # Deleting an isolated node can restore a previously seen, now valid
        # candidate. Give the Director a bounded chance to submit that repair.
        # The same semantic state receives this allowance only once, so repeated
        # add/delete cycles still exhaust the ordinary no-progress limit.
        if (accepted and repeated and ready and not self._unified_previous_submit_ready
                and signature not in self._unified_submission_recoveries):
            self._unified_submission_recoveries.add(signature)
            repeated = False
        self._unified_previous_submit_ready = ready
        self._unified_no_progress = self._unified_no_progress + 1 if repeated else 0
        self._unified_seen_states.add(signature)

    def _restore_submission_lock(self):
        if self.run_id and self._unified_transaction is None:
            existing = SubmissionJournal(self.config.submission_journal_dir, self.run_id).read()
            if existing:
                from .canvas import CanvasState
                self._unified_transaction = existing
                self.state = CanvasState.FAILED
                return True
        return False

    def _unified_progress_signature(self):
        graph = self.graph.to_dict()
        graph.pop("version", None)
        for node in graph.get("nodes", {}).values() if isinstance(graph.get("nodes"), dict) else graph.get("nodes", []):
            if isinstance(node, dict):
                node["metadata"] = {k: v for k, v in node.get("metadata", {}).items() if not k.startswith("_runtime_")}
        artifacts = {key: {"answer": value.answer, "summary": value.summary,
                    "issues": value.unresolved_issues, "evidence": value.evidence,
                    "resource": self.runtime.resource_signature(key)}
            for key, value in self.runtime.artifacts.items() if key in self.graph.nodes}
        return snapshot_hash({"graph": graph, "state": self.state.value,
                              "pending": self.pending_agent_id,
                              "dirty": sorted(self.dirty_agents), "results": artifacts})

    def observe_submission_candidates(self, call_id):
        self._unified_observed_snapshots[call_id] = {
            target: [self.graph.version, artifact.artifact_id,
                     self.runtime.artifact_input_binding(target).get("input_hash")]
            for target, artifact in self.runtime.artifacts.items()
            if target in self.graph.nodes
        }

    def _isolated_submission_draft(self, target):
        """An unexecuted, isolated pending node is not part of a result's dependencies."""
        pending = self.pending_agent_id
        node = self.graph.nodes.get(pending)
        return bool(pending and pending != target and node and not node.configured
            and pending not in self.runtime.artifacts
            and not any(pending in edge for edge in
                        self.graph.directed_edges | self.graph.bidirectional_edges))

    def _submission_graph(self, target):
        graph = self.graph.clone()
        if self._isolated_submission_draft(target):
            graph.delete_agent(self.pending_agent_id)
        graph.output_agent = target
        return graph

    def submission_assessment(self, target, *, include_payload=False):
        from .runtime import artifact_integrity_failure_risks

        if self.submission_receipt is not None and self.submission_receipt.output_agent_id == target:
            result = copy.deepcopy(self._unified_transaction["assessment"])
            result.update(submitted=True, submit_ready=False, blockers=[])
            if not include_payload:
                result["payload_hash"] = snapshot_hash(result.pop("payload"))
            return result

        node = self.graph.nodes.get(target)
        artifact = self.runtime.artifacts.get(target)
        blockers = []
        kind = "text"
        payload = {}
        if node is None:
            blockers.append("unknown_agent")
        elif not node.configured:
            blockers.append("configuration_incomplete")
        elif not is_task_result(node):
            blockers.append("local_result_only")
        if artifact is None:
            blockers.append("missing_artifact")
        else:
            if target in self.dirty_agents or not self.runtime.artifact_matches_current_input_signature(
                target, task=self.worker_task, graph=self.graph
            ):
                blockers.append("stale_artifact")
            risks = artifact_integrity_failure_risks(artifact)
            if self.dataset == "alfworld":
                kind = "episode"
                payload = self.runtime.environment_result_for(target)
                if not payload or payload.get("environment_completed") is not True:
                    blockers.append("environment_result_unavailable")
            else:
                if not str(artifact.answer).strip() or str(artifact.answer) in {"WORKER_BACKEND_FAILURE", "WORKER_PROTOCOL_FAILURE"}:
                    blockers.append("answer_missing")
                payload = {"answer": artifact.answer}
            trusted_environment = (
                self.dataset == "alfworld" and payload.get("environment_completed") is True
            )
            if risks and not trusted_environment:
                blockers.extend("integrity:" + risk for risk in risks)
        if node is not None:
            candidate = self._submission_graph(target)
            blockers.extend("graph:" + error for error in candidate.validate(final=True))
        if ((self.pending_agent_id and not self._isolated_submission_draft(target))
                or self.pending_relation_decision):
            blockers.append("pending_configuration")
        if self.total_tokens > self.config.max_total_tokens:
            blockers.append("worker_budget_exceeded")
        binding = self.runtime.artifact_input_binding(target) if artifact is not None else {}
        return {
            "target": target,
            "result_scope": node.metadata.get("result_scope") if node else None,
            "artifact_id": artifact.artifact_id if artifact else None,
            "input_signature": binding.get("input_hash"),
            "resource_signature": binding.get("resource_signature"),
            "agent_incarnation": node.metadata.get("incarnation_id") if node else None,
            "payload_kind": kind,
            **({"payload": payload} if include_payload else {"payload_hash": snapshot_hash(payload)}),
            "submit_ready": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
        }

    def _unified_can_run(self, target, *, admitted_action=False):
        from .canvas import CanvasState
        if (self.state in {CanvasState.FINISHED, CanvasState.FAILED}
                or (not admitted_action and not self.active)):
            return False
        if self.total_tokens >= self.config.max_total_tokens:
            return False
        node = self.graph.nodes.get(target)
        if node is None or not node.configured:
            return False
        artifact = self.runtime.artifacts.get(target)
        if self.dataset == "alfworld":
            resource = self.runtime.environment_result_for(target)
            if resource.get("resource_status") == "unknown" or resource.get("termination_reason") == "environment_step_failed":
                return False
            if self.runtime.environment_continuation_status(target).get("can_continue") is False:
                return False
        if target in self.dirty_agents or artifact is None:
            return True
        from .runtime import artifact_integrity_failure_risks
        if artifact_integrity_failure_risks(artifact):
            return self._unified_recovery_used < self.config.max_recovery_executions
        if self.dataset == "alfworld":
            result = self.runtime.environment_result_for(target)
            remaining = result.get("remaining_rollout_env_steps", result.get("remaining_env_steps", result.get("remaining_steps", 1)))
            return (bool(result) and remaining > 0
                    and result.get("termination_reason") != "environment_step_failed"
                    and not result.get("done"))
        return False

    def _unified_run(self, action):
        from .canvas import CanvasState
        # step() has admitted and charged this action's Director round already.
        # Keep every Worker/resource check, without demanding another round.
        if self.state is not CanvasState.BUILDING or not self._unified_can_run(str(action.target), admitted_action=True):
            message = "No pending work or bounded recovery is available for this target."
            return self._reject_graph_action(action, code="execution_not_admissible", message=message)
        target = str(action.target)
        from .runtime import artifact_integrity_failure_risks
        artifact = self.runtime.artifacts.get(target)
        if artifact and artifact_integrity_failure_risks(artifact):
            self._unified_recovery_used += 1
        self.graph.nodes[target].metadata["_runtime_explicit_recovery"] = True
        self.dirty_agents.add(target)
        self.dirty_reasons.setdefault(target, set()).add("explicit_continuation")
        try:
            report = self._execute_dirty()
        except Exception as exc:
            from .canvas import RequiredWorkerBackendFailure, TokenBudgetExceeded
            if not isinstance(exc, (RequiredWorkerBackendFailure, TokenBudgetExceeded, ValueError)):
                raise
            self.state = CanvasState.FAILED
            return self._record(
                action, accepted=True, execution=getattr(exc, "report", None),
                feedback=f"Explicit execution stopped: {exc}",
                rejection_code=("worker_backend_unavailable" if isinstance(exc, RequiredWorkerBackendFailure)
                                else "execution_budget_exceeded" if isinstance(exc, TokenBudgetExceeded)
                                else "execution_failure"),
            )
        finally:
            self.graph.nodes[target].metadata.pop("_runtime_explicit_recovery", None)
        return self._record(action, accepted=True, feedback="Executed the bounded continuation on current inputs and resources.", execution=report)

    def _unified_finish(self, action):
        from .answer_submission import AnswerFinalizer
        from .canvas import CanvasState
        from .observability import TaskSpec

        journal = SubmissionJournal(self.config.submission_journal_dir, self.run_id)
        with self._submission_lock, journal.locked():
            context = self._director_call_context
            if not isinstance(context, DirectorCallContext) or not context.runtime_owned or context.run_id != self.run_id:
                return self._reject_graph_action(action, code="director_finish_source_required", message="A real Director FINISH call is required for every dataset.")
            assessment = self.submission_assessment(str(action.target), include_payload=True)
            if not assessment["submit_ready"]:
                return self._reject_graph_action(action, code="submission_not_ready", message="; ".join(assessment["blockers"]), rejection_details={"result_assessment": self.submission_assessment(str(action.target))})
            observed = self._unified_observed_snapshots.get(context.call_id)
            current_ref = [self.graph.version, assessment["artifact_id"], assessment["input_signature"]]
            if observed is not None and observed.get(str(action.target)) != current_ref:
                return self._reject_graph_action(action, code="stale_submission_observation", message="The candidate changed after the Director observed it.")
            artifact = self.runtime.artifacts[str(action.target)]
            binding = self.runtime.artifact_input_binding(str(action.target))
            if not binding.get("input_hash"):
                return self._reject_graph_action(action, code="submission_input_binding_missing", message="The candidate lacks runtime input provenance.")
            graph = self._submission_graph(str(action.target))
            transaction_id = snapshot_hash([self.run_id, action.target, artifact.artifact_id, binding["input_hash"]])
            existing = journal.read()
            if existing:
                self._unified_transaction = existing
                self.state = CanvasState.FAILED
                return self._reject_graph_action(action, code="submission_transaction_exists", message="A persisted submission exists; reconcile that transaction before any new operation.")
            transaction = {
                "version": SUBMISSION_VERSION, "transaction_id": transaction_id,
                "run_id": self.run_id, "director_call_id": context.call_id,
                "target": action.target, "assessment": assessment,
                "graph": graph.to_dict(), "state": "prepared",
            }
            journal.write(transaction)
            self._unified_transaction = transaction
            try:
                raw_answer = str(artifact.answer or "")
                if assessment["payload_kind"] == "text":
                    if self._unified_answer_finalizer is not None:
                        submission = self._unified_answer_finalizer(raw_answer)
                    else:
                        submission = AnswerFinalizer().finalize(
                            TaskSpec(self.run_id, self.worker_task, metadata={"dataset": self.dataset}), raw_answer
                        )
                    submitted_answer = submission.submitted_answer
                    normalization = submission.method
                else:
                    submitted_answer = json.dumps(assessment["payload"], ensure_ascii=False, sort_keys=True)
                    normalization = "runtime_result_payload"
                self.graph = graph
                if self.pending_agent_id not in graph.nodes:
                    self.dirty_agents.discard(self.pending_agent_id)
                    self.dirty_reasons.pop(self.pending_agent_id, None)
                    self.pending_agent_id = None
                self.submission_receipt = _issue_receipt(
                    context=context, version=SUBMISSION_VERSION, dataset=self.dataset,
                    accepted_event_id=f"{self.run_id}:canvas:{len(self.history)}",
                    output_agent_id=str(action.target), graph_snapshot_hash=snapshot_hash(graph.to_dict()),
                    artifact_id=artifact.artifact_id, input_signature=binding["input_hash"],
                    execution_generation=int(binding.get("generation", 0)),
                    raw_answer_snapshot=raw_answer, submitted_answer_snapshot=submitted_answer,
                    answer_hash=answer_hash(submitted_answer), normalization_version=normalization,
                    worker_tokens_used=self.total_tokens, worker_token_limit=self.config.max_total_tokens,
                    payload_kind=assessment["payload_kind"], payload=copy.deepcopy(assessment["payload"]),
                    transaction_id=transaction_id,
                )
                transaction.update(state="committed", receipt=self.submission_receipt.to_dict())
                journal.write(transaction)
                self.state = CanvasState.FINISHED
                return self._record(action, accepted=True, feedback="Submitted the selected current task result without a Worker call.", final_execution=True)
            except Exception as exc:
                transaction.update(state="commit_unknown", error_type=type(exc).__name__, error=str(exc)[:500])
                try:
                    journal.write(transaction)
                except OSError:
                    # The last durable PREPARED record still prevents replay.
                    # Preserve factual confirmation in the in-memory audit.
                    transaction["persistence_failed"] = True
                self.submission_receipt = None
                self.state = CanvasState.FAILED
                return self._record(action, accepted=False, feedback="Submission status is uncertain; the persisted transaction is locked and must not be replayed with another target.", rejection_code="submission_commit_unknown", final_execution=True)

    def _protected_alfworld_candidates(self):
        if self.dataset != "alfworld":
            return []
        candidates = []
        for target in sorted(self.graph.nodes):
            assessment = self.submission_assessment(target, include_payload=True)
            payload = assessment.get("payload", {})
            if assessment["submit_ready"] and payload.get("done") is True and payload.get("won") is True:
                candidates.append(target)
        return candidates

    def _unified_control_snapshot(self):
        from .canvas import CanvasState
        self._restore_submission_lock()
        parameters = self._legal_action_parameters()
        parameters.pop("set_output", None)
        assessments = {key: self.submission_assessment(key) for key in sorted(self.graph.nodes)}
        ready = [key for key, value in assessments.items() if value["submit_ready"]]
        executable = [key for key in sorted(self.graph.nodes) if self._unified_can_run(key)]
        ids = sorted(self.graph.nodes)
        parameters["finish"] = {"targets": ready, "ready": bool(ready),
            "discards_isolated_unconfigured_draft": self.pending_agent_id if any(
                self._isolated_submission_draft(key) for key in ready) else None}
        parameters["run_agent"] = {"targets": executable}
        parameters["delete_agent"]["targets"] = (
            [self.pending_agent_id] if self.pending_agent_id and self.state in
            {CanvasState.AWAITING_PROMPT, CanvasState.AWAITING_MODEL} else ids)
        parameters["set_layer"]["targets"] = ids
        parameters["set_prompt"]["targets"] = [self.pending_agent_id] if self.pending_agent_id else ids
        parameters["set_prompt"]["result_scopes"] = ["subtask", "task_result"]
        allowed = ["add_agent", "set_prompt", "set_model", "set_layer", "consider_relation", "delete_agent", "run_agent", "finish"]
        if len(ids) >= self.config.max_agents or self.topology_edits_frozen or (
                self.runtime_routes
                and self.director_round_limit - self.round_index < 4):
            allowed.remove("add_agent")
        if not self.active:
            allowed = []
        elif self.state is CanvasState.AWAITING_PROMPT:
            allowed = ["set_prompt", "delete_agent"] + (["finish"] if ready else [])
        elif self.state is CanvasState.AWAITING_MODEL:
            allowed = ["set_model", "delete_agent"] + (["finish"] if ready else [])
        elif self.state is CanvasState.AWAITING_RELATION_CHOICE:
            allowed = ["relation_choice"]
        allowed = [name for name in allowed if name in {"add_agent", "relation_choice"} or bool(parameters.get(name, {}).get("targets", parameters.get(name, {}).get("relations")))]
        protected = self._protected_alfworld_candidates()
        if protected:
            allowed = ["finish"] if "finish" in allowed else []
            parameters["finish"]["targets"] = protected
        return {
            "canvas_version": self.graph.version, "director_action_protocol_version": ACTION_PROTOCOL,
            "submission_contract_version": SUBMISSION_VERSION, "submission_protocol": PROTOCOL,
            "submission_status": self._unified_transaction["state"] if self._unified_transaction else "working",
            "state": self.state.value, "legal_agent_ids": ids, "pending_agent_id": self.pending_agent_id,
            "pending_relation_decision": self.pending_relation_decision.to_dict() if self.pending_relation_decision else None,
            "allowed_actions": allowed,
            "worker_usage": None,
            "terminal_candidate_protection": {
                "policy": "finish_only",
                "targets": protected,
                "instruction": "Submit an existing trusted terminal candidate with FINISH(target); no more Worker calls or graph edits." if protected else "",
            } if self.dataset == "alfworld" else None,
            "action_field_requirements": {
                **{name: list(UNIFIED_ACTION_FIELDS[name]) for name in allowed if name in UNIFIED_ACTION_FIELDS},
            },
            "legal_action_parameters": parameters, "graph_state": self._graph_state_snapshot(),
            "result_assessments": assessments,
            "agent_budget": {"used": len(ids), "max": self.config.max_agents, "remaining": self.config.max_agents - len(ids)},
            "progress": {"no_progress_count": self._unified_no_progress,
                         "semantic_states_seen": len(self._unified_seen_states),
                         "submission_recovery_count": len(self._unified_submission_recoveries)},
            "recovery_budget": {"used": self._unified_recovery_used, "max": self.config.max_recovery_executions},
            "director_edit_budget": self.director_edit_budget(),
            "action_budget": None,
            "round_budget": {"used": self.round_index, "max": self.director_round_limit,
                             "remaining": max(0, self.director_round_limit - self.round_index)},
            "token_budget": {"used": self.total_tokens, "max": self.config.max_total_tokens, "remaining": max(0, self.config.max_total_tokens - self.total_tokens)},
            "output_agent": self.graph.output_agent,
            "environment_commit_resolution": "FINISH(target) commits only that current result, without executing Workers. RUN_AGENT continues unfinished work within the same resource limits.",
        }
