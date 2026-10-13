from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from .observability import TaskSpec, VerificationResult


class ALFWorldClient(Protocol):
    def create_session(self, game_path: str, *, max_steps: int) -> dict[str, Any]: ...

    def step(self, session_id: str, command: str) -> dict[str, Any]: ...

    def close_session(self, session_id: str) -> None: ...


@dataclass
class ALFWorldSessionLifecycle:
    """Keep one continuous episode per bound task across its Worker nodes and revisions."""

    client: ALFWorldClient
    data_root: Path
    max_episode_steps: int = 50
    max_rollout_steps: int = 400
    max_observation_chars: int = 4_000
    adapter_id: str = "alfworld"
    task_prompt_source: str = "dataset"
    _prepared_payload: dict[str, Any] | None = None
    _effective_task: str = ""
    _task: TaskSpec | None = None
    _game_path: Path | None = None
    _game_fingerprint: str = ""
    _internal_goal_contract: dict[str, Any] = field(default_factory=dict)
    _active_agent: str | None = None
    _active_session: str | None = None
    _active_state: dict[str, Any] = field(default_factory=dict)
    _action_map: dict[str, str] = field(default_factory=dict)
    _results: dict[str, dict[str, Any]] = field(default_factory=dict)
    _rollout_steps: int = 0
    _attempt_index: int = 0
    _active_attempt_index: int = 0
    _shared_session: tuple[str, dict[str, Any], dict[str, str], int] | None = None
    _shared_owner: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def environment_fingerprint(self) -> str:
        return self._game_fingerprint

    @property
    def effective_task(self) -> str:
        return self._effective_task

    def bind_task(self, task: TaskSpec) -> None:
        with self._lock:
            self.close_all()
            game_path = _trusted_game_path(task, self.data_root)
            self._task = task
            self._game_path = game_path
            self._game_fingerprint = _game_fingerprint(game_path)
            self._internal_goal_contract = _alfworld_internal_goal_contract(task)
            self._results = {}
            self._rollout_steps = 0
            self._attempt_index = 0
            if self.task_prompt_source not in {"dataset", "environment_reset"}:
                raise ValueError("unknown ALFWorld task_prompt_source")
            self._effective_task = task.prompt
            if self.task_prompt_source == "environment_reset":
                payload = self.client.create_session(str(game_path), max_steps=self.max_episode_steps)
                self._prepared_payload = payload
                try:
                    if not str(payload.get("session_id", "")).strip():
                        raise RuntimeError("ALFWorld reset returned no session_id")
                    self._effective_task = self._reset_goal(payload)
                    if not self._effective_task:
                        raise RuntimeError("ALFWorld reset has no public task statement")
                    task.metadata["alfworld_task_binding"] = {
                        "version": "environment_reset", "source": self.task_prompt_source,
                        "original_task": task.prompt, "effective_task": self._effective_task,
                        "game_fingerprint": self._game_fingerprint,
                        "prepared_session_id": str(payload["session_id"]),
                    }
                except BaseException:
                    self.close_all()
                    raise

    @staticmethod
    def _reset_goal(payload: dict[str, Any]) -> str:
        _, marker, goal = str(payload.get("observation", "")).partition("Your task is to:")
        return goal.strip() if marker else ""

    def begin_execution(self, *, agent_id: str, seed: int, revision: bool) -> dict[str, Any]:
        del seed
        with self._lock:
            if self._task is None or self._game_path is None:
                raise RuntimeError("ALFWorld lifecycle has no bound task")
            if self._active_session is not None:
                raise RuntimeError("ALFWorld lifecycle already has an active Worker execution")
            agent_id = str(agent_id)
            if self._shared_session is not None:
                if self._shared_owner and self._results.get(self._shared_owner, {}).get(
                    "termination_reason"
                ) == "environment_step_failed":
                    raise RuntimeError("ALFWorld session state is uncertain after an environment failure")
                session_id, state, action_map, attempt_index = self._shared_session
                self._shared_session = None
                self._shared_owner = agent_id
                self._active_agent = agent_id
                self._active_session = session_id
                self._active_state = state
                self._action_map = action_map
                self._active_attempt_index = attempt_index
                prior = self._results.get(agent_id)
                if prior is None:
                    prior = self._result(
                        state,
                        revision=revision,
                        termination_reason=(
                            "official_success" if state.get("done") and state.get("success")
                            else "official_done_without_success" if state.get("done")
                            else "running"
                        ),
                        environment_completed=True,
                    )
                prior["revision"] = bool(revision)
                prior["session_id"] = session_id
                prior["attempt_index"] = attempt_index
                if not state.get("done"):
                    prior["termination_reason"] = "running"
                self._results[agent_id] = prior
                state["remaining_rollout_env_steps"] = max(
                    0, self.max_rollout_steps - self._rollout_steps
                )
                return dict(state)
            if self._rollout_steps >= self.max_rollout_steps:
                raise RuntimeError("ALFWorld rollout environment-step budget is exhausted")
            payload = self._prepared_payload
            if payload is None:
                payload = self.client.create_session(str(self._game_path), max_steps=self.max_episode_steps)
            self._prepared_payload = None
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                raise RuntimeError("ALFWorld session creation returned no session_id")
            if self.task_prompt_source == "environment_reset" and self._reset_goal(payload) != self._effective_task:
                self.client.close_session(session_id)
                raise RuntimeError("ALFWorld reset public task differs from the prepared task")
            self._active_agent = str(agent_id)
            self._active_session = session_id
            self._attempt_index += 1
            self._active_attempt_index = self._attempt_index
            state = self._public_state(payload, step=0)
            self._active_state = state
            self._results[self._active_agent] = self._result(
                state,
                revision=revision,
                termination_reason="running",
                environment_completed=True,
            )
            return dict(state)

    def step(self, action_id: str) -> dict[str, Any]:
        with self._lock:
            session_id, agent_id = self._active()
            if self._active_state.get("done"):
                raise RuntimeError("ALFWorld episode is already done")
            command = self._action_map.get(str(action_id))
            if command is None:
                raise ValueError("stale_or_unknown_action_id")
            if self._rollout_steps >= self.max_rollout_steps:
                self._results[agent_id] = self._result(
                    self._active_state,
                    revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                    termination_reason="rollout_environment_step_budget_exhausted",
                    environment_completed=False,
                )
                raise RuntimeError("ALFWorld rollout environment-step budget is exhausted")
            try:
                payload = self.client.step(session_id, command)
            except Exception:
                self._results[agent_id] = self._result(
                    self._active_state,
                    revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                    termination_reason="environment_step_failed",
                    environment_completed=False,
                )
                raise
            self._rollout_steps += 1
            step = int(self._active_state.get("step", 0)) + 1
            state = self._public_state(payload, step=step, executed_command=command)
            self._active_state = state
            reason = (
                "official_success"
                if state["done"] and state["success"]
                else "episode_step_budget_exhausted"
                if state["done"] and step >= self.max_episode_steps
                else "official_done_without_success"
                if state["done"]
                else "running"
            )
            self._results[agent_id] = self._result(
                state,
                revision=bool(self._results.get(agent_id, {}).get("revision", False)),
                termination_reason=reason,
                environment_completed=True,
            )
            return dict(state)

    def discard_pending(self, agent_id: str) -> None:
        """Delete the node's live episode without refunding the rollout ledger."""
        with self._lock:
            if self._active_agent == agent_id:
                self._close_active()
            if (
                self._shared_session is not None
                and self._shared_owner == agent_id
            ):
                self.client.close_session(self._shared_session[0])
                self._shared_session = None
                self._shared_owner = None
            self._results.pop(agent_id, None)

    def result_for(self, agent_id: str | None) -> dict[str, Any]:
        with self._lock:
            if not agent_id:
                return _empty_result("missing_output_agent")
            return dict(self._results.get(agent_id, _empty_result("agent_never_executed")))

    def continuation_status(self, agent_id: str) -> dict[str, Any]:
        """Current execution allowance, separate from immutable episode evidence."""
        with self._lock:
            result = self._results.get(agent_id, {})
            remaining = min(
                max(0, self.max_episode_steps - int(result.get("steps", 0))),
                max(0, self.max_rollout_steps - self._rollout_steps),
            )
            return {
                "remaining_steps": remaining,
                "can_continue": remaining > 0 and not result.get("done", False)
                    and result.get("termination_reason") != "environment_step_failed",
            }

    def end_execution(self) -> None:
        with self._lock:
            if self._active_agent and not self._active_state.get("done", False):
                current = self._results.get(self._active_agent, {})
                if current.get("termination_reason") == "running":
                    current.update(
                        {
                            "termination_reason": "worker_final_before_environment_done",
                            "environment_completed": True,
                        }
                    )
                self._results[self._active_agent] = current
            if self._active_agent and self._active_session:
                saved = (
                    self._active_session, self._active_state,
                    self._action_map, self._active_attempt_index,
                )
                self._shared_session = saved
                self._shared_owner = self._active_agent
                self._active_session = None
                self._active_agent = None
                self._active_state = {}
                self._action_map = {}

    def close_all(self) -> None:
        with self._lock:
            self._close_active()
            if self._prepared_payload is not None:
                session_id = str(self._prepared_payload.get("session_id", ""))
                self._prepared_payload = None
                if session_id:
                    self.client.close_session(session_id)
            self._effective_task = ""
            if self._shared_session is not None:
                self.client.close_session(self._shared_session[0])
                self._shared_session = None
                self._shared_owner = None
            self._task = None
            self._game_path = None
            self._game_fingerprint = ""
            self._internal_goal_contract = {}

    def _active(self) -> tuple[str, str]:
        if not self._active_session or not self._active_agent:
            raise RuntimeError("ALFWorld Action called outside an active Worker execution")
        return self._active_session, self._active_agent

    def _close_active(self) -> None:
        if self._active_session:
            try:
                self.client.close_session(self._active_session)
            finally:
                self._active_session = None
                self._active_agent = None
                self._active_state = {}
                self._action_map = {}

    def _public_state(
        self,
        payload: dict[str, Any],
        *,
        step: int,
        executed_command: str = "",
    ) -> dict[str, Any]:
        # TextWorld can return stale admissible commands together with done=True.
        # The trusted terminal result ends the Action surface, not the budget.
        commands = (
            []
            if payload.get("done")
            else [str(value) for value in payload.get("admissible_commands", [])]
        )
        self._action_map = {
            f"s{step:04d}:a{index:03d}": command for index, command in enumerate(commands)
        }
        observation = str(payload.get("observation", ""))
        if step == 0:
            initial_observation = observation[: self.max_observation_chars]
            # Extract before truncation: the public reset goal follows the room description.
            _, marker, public_task = observation.partition("Your task is to:")
            public_task_statement = public_task.strip()[: self.max_observation_chars] if marker else ""
        else:
            initial_observation = str(self._active_state.get("initial_observation", ""))
            public_task_statement = str(self._active_state.get("public_task_statement", ""))
        state: dict[str, Any] = {
            "status": (
                "success"
                if bool(payload.get("done")) and bool(payload.get("won"))
                else "done"
                if bool(payload.get("done"))
                else "running"
            ),
            "step": step,
            "observation": observation[: self.max_observation_chars],
            "initial_observation": initial_observation,
            "public_task_statement": public_task_statement,
            "observation_truncated": len(observation) > self.max_observation_chars,
            "reward": float(payload.get("reward", 0.0)),
            "score": float(payload.get("score", payload.get("reward", 0.0))),
            "done": bool(payload.get("done", False)),
            "success": bool(payload.get("won", False)),
            "remaining_env_steps": max(0, self.max_episode_steps - step),
            "remaining_rollout_env_steps": max(0, self.max_rollout_steps - self._rollout_steps),
            "admissible_actions": [
                {"action_id": action_id, "command": command}
                for action_id, command in self._action_map.items()
            ],
            # Runtime-internal goal interpretation. Model-visible prompt projections
            # remove this. It contains no demonstration, object location, or
            # solution trajectory.
            "goal_contract": dict(self._internal_goal_contract),
        }
        if executed_command:
            state["executed_command"] = executed_command
        state["factual_memory"] = _update_factual_memory(
            self._active_state.get("factual_memory", {}) if step else {},
            command=executed_command,
            observation=observation[: self.max_observation_chars],
            step=step,
        )
        return state

    def _result(
        self,
        state: dict[str, Any],
        *,
        revision: bool,
        termination_reason: str,
        environment_completed: bool,
    ) -> dict[str, Any]:
        return {
            "adapter": self.adapter_id,
            "task_id": self._task.task_id if self._task else None,
            "session_id": self._active_session,
            "task_prompt_source": self.task_prompt_source,
            "game_fingerprint": self._game_fingerprint,
            "attempt_index": self._active_attempt_index,
            "revision": bool(revision),
            "done": bool(state.get("done", False)),
            "won": bool(state.get("success", False)),
            "official_score": float(state.get("score", 0.0)),
            "steps": int(state.get("step", 0)),
            "termination_reason": termination_reason,
            "budget_truncated": "budget_exhausted" in termination_reason,
            "environment_completed": bool(environment_completed),
            "public_task_statement": str(state.get("public_task_statement", "")),
        }


@dataclass(frozen=True)
class ALFWorldStepTool:
    lifecycle: ALFWorldSessionLifecycle
    name: str = "alfworld_step"
    stateful: bool = True
    description: str = (
        "Execute exactly one currently admissible ALFWorld command. Copy action_id from the "
        "latest admissible_actions list; stale or invented ids are rejected."
    )
    error_codes: tuple[str, ...] = (
        "stale_or_unknown_action_id",
        "episode_already_done",
        "episode_not_active",
        "environment_step_budget_exhausted",
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"action_id": {"type": "string", "minLength": 1}},
            "required": ["action_id"],
            "additionalProperties": False,
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": [
                "status",
                "step",
                "observation",
                "reward",
                "done",
                "success",
                "admissible_actions",
            ],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        action_id = arguments.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("alfworld_step requires a non-empty action_id")
        return json.dumps(self.lifecycle.step(action_id.strip()), ensure_ascii=False)


class ALFWorldEnvironmentVerifier:
    name = "alfworld_environment"
    supports_intermediate_scoring = False

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        del prediction
        result = task.metadata.get("alfworld_environment_result")
        if not isinstance(result, dict):
            raise ValueError("ALFWorld verification requires a trusted environment result")
        if not bool(result.get("environment_completed", False)):
            # An incomplete environment result is still a trustworthy runtime
            # observation.  Return a zero-score verification so the rollout can
            # be persisted and the runner can either recover it or, under the
            # narrow model-policy contract, retain the final failed attempt as a
            # negative sample.  Raising here used to discard the full Director
            # trajectory before the trace store could write it.
            diagnostic = {
                key: result.get(key)
                for key in (
                    "done",
                    "won",
                    "official_score",
                    "steps",
                    "termination_reason",
                    "budget_truncated",
                    "attempt_index",
                    "revision",
                )
                if key in result
            }
            return VerificationResult(
                score=0.0,
                passed=False,
                verifier=self.name,
                detail=json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
            )
        won = bool(result.get("won", False))
        detail = {
            key: result.get(key)
            for key in (
                "done",
                "won",
                "official_score",
                "steps",
                "termination_reason",
                "budget_truncated",
                "game_fingerprint",
                "attempt_index",
                "revision",
            )
            if key in result
        }
        return VerificationResult(
            score=float(won),
            passed=won,
            verifier=self.name,
            detail=json.dumps(detail, ensure_ascii=False),
        )


def alfworld_lifecycles(tools: dict[str, Any]) -> tuple[ALFWorldSessionLifecycle, ...]:
    values: dict[int, ALFWorldSessionLifecycle] = {}
    for tool in tools.values():
        lifecycle = getattr(tool, "lifecycle", None)
        if isinstance(lifecycle, ALFWorldSessionLifecycle):
            values[id(lifecycle)] = lifecycle
    return tuple(values.values())


def _trusted_game_path(task: TaskSpec, data_root: Path) -> Path:
    value = task.metadata.get("game_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ALFWorld task requires trusted metadata.game_path")
    root = Path(data_root).expanduser().resolve()
    # Released task files name the game relative to the ALFWorld data root.
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("ALFWorld game_path escapes configured data_root")
    if path.name != "game.tw-pddl" or not path.is_file():
        raise ValueError("ALFWorld game_path must reference an existing game.tw-pddl")
    source_split = str(task.metadata.get("source_split", task.metadata.get("split", "")))
    if source_split and source_split not in path.parts:
        raise ValueError("ALFWorld game_path does not match trusted source_split")
    return path


def _game_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    for package in ("alfworld", "textworld"):
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "unknown"
        digest.update(f"\0{package}:{package_version}".encode())
    return digest.hexdigest()


def _alfworld_internal_goal_contract(task: TaskSpec) -> dict[str, Any]:
    """Build the trusted declared-goal shape without reading a solution trajectory."""

    metadata = task.metadata
    task_type = str(metadata.get("alfworld_task_type", "")).strip().casefold()
    raw_params = metadata.get("pddl_params", {})
    params = raw_params if isinstance(raw_params, dict) else {}
    transform = ""
    for candidate in ("heat", "cool", "clean"):
        if candidate in task_type:
            transform = candidate
            break
    return {
        "version": "alfworld-public-goal",
        "task_type": task_type,
        "object_target": str(params.get("object_target", "")).strip(),
        "destination_target": str(params.get("parent_target", "")).strip(),
        "toggle_target": str(params.get("toggle_target", "")).strip(),
        "required_object_count": 2 if "pick_two" in task_type else 1,
        "required_transform": transform,
        "requires_placement": bool(
            "place" in task_type or str(params.get("parent_target", "")).strip()
        ),
        "requires_lit_examination": task_type == "look_at_obj_in_light",
        "requires_slicing": bool(params.get("object_sliced", False)),
    }


def _update_factual_memory(
    previous: dict[str, Any], *, command: str, observation: str, step: int
) -> dict[str, Any]:
    """Keep public observations, never infer task goals or unseen object state."""
    memory = copy.deepcopy(previous) if previous else {
        "current_location": "",
        "visited_locations": [],
        "opened_receptacles": [],
        "location_observations": {},
        "object_interactions": {},
        "recent_interactions": [],
    }
    arrival = re.search(r"You arrive at (.+?)\.", observation)
    if arrival:
        location = arrival.group(1)
        memory["current_location"] = location
        if location not in memory["visited_locations"]:
            memory["visited_locations"].append(location)
    opened = re.search(r"You open the (.+?)\.", observation)
    if opened and opened.group(1) not in memory["opened_receptacles"]:
        memory["opened_receptacles"].append(opened.group(1))
    location = memory["current_location"]
    # These are dated observations, not assertions that objects remain there.
    # Keep inspection evidence separate from subsequent manipulation feedback.
    if location and (arrival or opened or command.startswith("examine ")):
        memory["location_observations"][location] = {
            "step": step, "command": command, "observation": observation,
        }
    if command:
        if command.startswith(("take ", "move ", "clean ", "cool ", "heat ")):
            # Retain each object's dated feedback even after recent history rotates.
            object_name = re.split(r" from | to | with ", command.split(" ", 1)[1])[0]
            memory["object_interactions"].setdefault(object_name, []).append({
                "step": step, "command": command, "observation": observation,
            })
        memory["recent_interactions"].append({
            "step": step, "location": location,
            "command": command, "observation": observation,
        })
        memory["recent_interactions"] = memory["recent_interactions"][-32:]
    return memory


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "adapter": "alfworld",
        "done": False,
        "won": False,
        "official_score": 0.0,
        "steps": 0,
        "termination_reason": reason,
        "budget_truncated": "budget_exhausted" in reason,
        "environment_completed": False,
    }
