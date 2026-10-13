"""R2Orch collection orchestration.

One cycle: Proposer pair selection per dataset, ``hidden + visible`` independent siblings per task
around one Thompson-chosen card k (``_arm_plans``; the hidden arm sees the retrieved context
without k), then the Proposer reward (the arms Bell reward), the Proposer pair batch and the
Director batch.  The cycle's arm contrasts go to the skill ledger (``_record_arms``).  With
probability ``arms.audit_share`` a task audits a ledger-retired card instead (k = that card, shown to the
visible arm only, ``audit: true`` in the plan, the rollouts and the ledger record).  Cross-workstream
modules are imported lazily inside functions.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name
from .graph import MultiAgentGraph
from .observability import TaskSpec
from .rollouts import TokenizedDirectorTrajectory, TokenizedPolicyCall
from .selfplay import DryRunSelfPlayResult, FrontierScore, ProposedTask, SolverRollout
from .selfplay_runtime import (
    CollectionInfrastructureIncidentError,
    SelfPlayRolloutRunner,
    SelfPlayRunConfig,
    _append_jsonl,
    _proposal_dict,
    _proposal_extraction_summary,
    _proposal_from_dict,
    _read_jsonl,
    _stable_execution_seed,
    _write_json,
)
from .training_selection import trajectory_exclusions

SIBLING_SCHEMA = "r2orch_sibling@1"
SELECTION_SCHEMA = "r2orch"
ACTION_PROTOCOL = "director_model"
# AIME's free-text Worker responses can legitimately take longer than the
# generic 180 s request budget when Qwen is serving a long reasoning trace.
# Give R2Orch enough per-request headroom while the 900 s rollout
# deadline still bounds a complete trajectory.  The generic floor matches the
# full rollout budget because one Worker trajectory can contain several model
# requests (a 600 s per-call floor still left only 181 s for a follow-up).
R2ORCH_REQUEST_WALL_TIME_S = 900.0
R2ORCH_AIME_REQUEST_WALL_TIME_S = 900.0
R2ORCH_ALFWORLD_REQUEST_WALL_TIME_S = 900.0
R2ORCH_AIME_ROLLOUT_WALL_TIME_S = 2400.0
# ALFWorld episodes can legitimately require several environment actions.  A
# 1,800 s effective budget leaves too little room when a long Qwen request is
# retried, and the deadline's absolute guard then aborts every sibling before
# it can produce a usable training trajectory.  Give stateful episodes 2,400 s.
R2ORCH_ALFWORLD_ROLLOUT_WALL_TIME_S = 2400.0


class R2OrchStopped(RuntimeError):
    """A stop boundary was reached; the run continues with ``--resume``."""


def r2orch_request_wall_time_s(request_wall_time_s: float) -> float:
    """Return R2Orch's generic Worker request budget.

    The shared runner defaults to 180 s, but R2Orch uses long free-text Qwen
    generations for every dataset.  A 180 s client timeout routinely closes an
    otherwise healthy backend response and opens the endpoint failure circuit.
    Keep explicitly larger values intact while applying a 900 s R2Orch floor (the
    900 s rollout deadline remains the outer bound for ordinary datasets).
    """

    configured = float(request_wall_time_s)
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("request_wall_time_s must be positive and finite")
    return max(configured, R2ORCH_REQUEST_WALL_TIME_S)


def r2orch_aime_request_wall_time_s(request_wall_time_s: float) -> float:
    """Return the R2Orch AIME request budget.

    R2Orch's long free-text Worker requests need a larger request-level budget
    than the ``SelfPlayRunConfig`` default.  Preserve an explicitly larger CLI
    value while raising the R2Orch floor to 900 s.
    """

    configured = float(request_wall_time_s)
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("request_wall_time_s must be positive and finite")
    return max(configured, R2ORCH_AIME_REQUEST_WALL_TIME_S)


def r2orch_alfworld_request_wall_time_s(request_wall_time_s: float) -> float:
    """Return R2Orch's ALFWorld request budget."""

    configured = float(request_wall_time_s)
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("request_wall_time_s must be positive and finite")
    return max(configured, R2ORCH_ALFWORLD_REQUEST_WALL_TIME_S)


def aime_rollout_wall_time_s(r2orch: Any, wall: float) -> float:
    """AIME rollout deadline: ``siblings.aime_rollout_wall_time_s`` when set, else the
    2400 s R2Orch floor."""

    override = float(r2orch.siblings.aime_rollout_wall_time_s)
    return override if override > 0 else max(float(wall), R2ORCH_AIME_ROLLOUT_WALL_TIME_S)


def alfworld_rollout_wall_time_s(r2orch: Any, wall: float) -> float:
    """Stateful (ALFWorld) rollout/slot deadline, same convention as the AIME one."""

    override = float(r2orch.siblings.alfworld_rollout_wall_time_s)
    return override if override > 0 else max(float(wall), R2ORCH_ALFWORLD_ROLLOUT_WALL_TIME_S)


# --------------------------------------------------------------------------- seeds


def round_base_seed(base_seed: int, cycle: int) -> int:
    return int(base_seed) + int(cycle) * 100_000


def sibling_execution_seed(base_seed: int, task_id: str, sibling_index: int) -> int:
    return _stable_execution_seed(base_seed, task_id, phase=f"sibling:{int(sibling_index)}")


def proposer_choice_seed(base_seed: int, list_id: str, k: int) -> int:
    return _stable_execution_seed(base_seed, list_id, phase=f"proposer_choice:{int(k)}")


# ------------------------------------------------------------------ stop / resume


def stop_requested(
    output_dir: Path,
    *,
    margin_s: float,
    env: str = "R2ORCH_STOP_AT_EPOCH",
    stop_file: str = "STOP",
) -> bool:
    if (Path(output_dir) / stop_file).exists():
        return True
    value = os.environ.get(env, "").strip()
    return bool(value) and float(value) - float(margin_s) <= time.time()


def action_protocol_record(execution_semantics: Mapping[str, Any]) -> dict[str, Any]:
    from .execution_contract import R2ORCH_CONTRACT_VERSION, R2ORCH_COUNTERFACTUAL_EXECUTION
    from .output_contract import OUTPUT_CONTRACT_VERSION

    return {
        "execution_semantics": json.loads(json.dumps(execution_semantics)),
        "action_protocol": ACTION_PROTOCOL,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "counterfactual_execution": R2ORCH_COUNTERFACTUAL_EXECUTION,
        "r2orch_contract_version": R2ORCH_CONTRACT_VERSION,
    }


def validate_action_protocol(directory: Path, execution_semantics: Mapping[str, Any]) -> None:
    """Record the action protocol of ``directory`` once; afterwards it must match."""
    from .execution_contract import _comparison_semantics

    path = Path(directory) / "action_protocol.json"
    expected = action_protocol_record(execution_semantics)
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        # Do not rewrite recorded manifests: only the comparison view drops the
        # fields a resumed run may retune and recomputes the config digest in memory.
        recorded_view = dict(recorded)
        expected_view = dict(expected)
        recorded_view["execution_semantics"] = _comparison_semantics(
            recorded.get("execution_semantics")
        )
        expected_view["execution_semantics"] = _comparison_semantics(
            expected.get("execution_semantics")
        )
        if recorded_view != expected_view:
            raise RuntimeError("incompatible action protocol; start a fresh run")
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    _write_json(path, expected)


def run_fork(output_dir: Path) -> dict[str, Any] | None:
    """``fork.json`` of a run forked from another run after one of its cycles, or ``None``.

    A forked run carries the parent's cycles ``0 .. fork_after_cycle`` (their files, the training history, the
    curriculum, stats, ledger and PATS state as they were after that cycle) and continues at the next cycle with
    its own configuration.  ``{"parent": <run root>, "fork_after_cycle": <int>, ...}``.
    """
    path = Path(output_dir) / "fork.json"
    if not path.exists():
        return None
    fork = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fork.get("fork_after_cycle"), int) or fork["fork_after_cycle"] < 0:
        raise ValueError("fork.json needs a nonnegative integer fork_after_cycle")
    return fork


def check_run_protocol(output_dir: Path, execution_semantics: Mapping[str, Any]) -> None:
    """Reject a run directory written under another action protocol before any new work.

    The first cycle of the run is checked; for a forked run that is the first cycle after the fork, because the
    inherited cycles keep the parent's protocol record.
    """
    output_dir = Path(output_dir)
    fork = run_fork(output_dir)
    first = output_dir / ("cycle-0000" if fork is None else f"cycle-{fork['fork_after_cycle'] + 1:04d}")
    if (first / "action_protocol.json").exists():
        validate_action_protocol(first, execution_semantics)


# ---------------------------------------------------------------------- helpers


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_id(proposal: ProposedTask) -> str:
    return str(proposal.metadata.get("pool_id") or proposal.task.metadata.get("pool_id") or "")


def _task_info(proposal: ProposedTask) -> dict[str, Any]:
    return {
        **dict(proposal.metadata.get("r2orch") or {}),
        **dict(proposal.task.metadata.get("r2orch") or {}),
    }


def _retrievable_cards(skill_snapshot: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    """Card records the frozen collection snapshot lets retrieval rank for ``scope``.

    Mirrors ``DirectorSkillBank.select_context``: the PATS scope view when the scope exists
    (an empty view is a withdrawal), the global seed/active cards otherwise; one record per id.
    """
    pats = skill_snapshot.get("pats")
    scoped = ((pats or {}).get("scopes") or {}).get(scope) if pats is not None else None
    records = scoped["cards"] if scoped is not None else skill_snapshot.get("cards") or []
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") in {"seed", "active"}:
            by_id[str(record["card"]["skill_id"])] = record
    return list(by_id.values())


def _retired_audit_cards(skill_snapshot: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    """Ledger-retired cards of ``scope`` in the frozen snapshot (``[r2orch.arms]`` audits).

    ``PatsController.snapshot`` lists them (with the archived record).
    """
    from .skill_ledger import RETIRED_VIEW_KEY

    view = (skill_snapshot.get("pats") or {}).get(RETIRED_VIEW_KEY) or {}
    return [entry for entry in view.get(scope) or [] if isinstance(entry, Mapping)]


def _outcome_exclusions(rollout: SolverRollout) -> list[str]:
    """Reasons a sibling's outcome may not enter the outcome statistics (Frontier, skill ledger)."""
    from .proposer_learning import frontier_evidence_exclusions

    return frontier_evidence_exclusions(rollout)


def _trajectory_from_row(item: Mapping[str, Any]) -> TokenizedDirectorTrajectory:
    return TokenizedDirectorTrajectory(
        rollout_id=str(item["rollout_id"]),
        task_id=str(item["task_id"]),
        token_ids=tuple(item["token_ids"]),
        action_mask=tuple(item["action_mask"]),
        reward=float(item["reward"]),
        graph=dict(item["graph"]),
        seed=int(item.get("seed", 0)),
        executor_version=str(item.get("executor_version", "adaptive")),
        metadata=dict(item.get("metadata", {})),
        policy_calls=tuple(
            TokenizedPolicyCall(
                call_id=str(call["call_id"]),
                token_ids=tuple(call["token_ids"]),
                action_mask=tuple(call["action_mask"]),
                behavior_log_probs=tuple(call.get("behavior_log_probs", ())),
                action_token_span=(
                    tuple(call["action_token_span"])
                    if call.get("action_token_span") is not None
                    else None
                ),
                relation_token_span=(
                    tuple(call["relation_token_span"])
                    if call.get("relation_token_span") is not None
                    else None
                ),
                metadata=dict(call.get("metadata", {})),
            )
            for call in item.get("policy_calls", ())
        ),
    )


def _rollout_from_row(item: Mapping[str, Any]) -> SolverRollout:
    trajectory = _trajectory_from_row(item)
    return SolverRollout(trajectory, MultiAgentGraph.from_dict(trajectory.graph))


def _persisted_rollout_quality(item: Mapping[str, Any]) -> tuple[int, ...]:
    """Rank duplicate durable rows so a bad retry cannot hide a good sample.

    A driver can be interrupted while a previous collection worker is still
    flushing its final row.  On the next ``--resume`` that leaves duplicate
    rollout IDs in ``solver_rollouts.jsonl``.  A last-row-wins loader would
    then replace a valid finished trajectory with a later unsafe partial
    retry and abort the whole cycle.  Prefer a training-eligible, finished
    row, while retaining deterministic last-row preference among equal rows.
    """

    metadata = item.get("metadata") or {}
    try:
        trajectory = _trajectory_from_row(item)
        eligible = int(not trajectory_exclusions(trajectory))
    except Exception:
        eligible = 0
    return (
        eligible,
        int(bool(metadata.get("finished"))),
        int(metadata.get("reward_known") is True),
        int(metadata.get("submission_status") == "submitted"),
        int(metadata.get("terminal_graph_status") == "valid_finished"),
    )


def _deduplicate_persisted_rollouts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one best row per rollout ID, preserving first-seen order."""

    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in rows:
        row = dict(raw)
        rollout_id = str(row.get("rollout_id", ""))
        if not rollout_id:
            continue
        previous = chosen.get(rollout_id)
        if previous is None:
            chosen[rollout_id] = row
            order.append(rollout_id)
        elif _persisted_rollout_quality(row) > _persisted_rollout_quality(previous):
            chosen[rollout_id] = row
    return [chosen[rollout_id] for rollout_id in order]


def _resume_proposal_quality(
    task_id: str, persisted: Mapping[str, Mapping[str, Any]]
) -> tuple[int, ...]:
    """Score a duplicate proposal by the quality of rows it can resume.

    A cancelled driver can leave two proposal rows with the same durable task
    id.  The rows are otherwise indistinguishable to the scheduler, so prefer
    the proposal whose persisted sibling rows are already eligible.  Returning
    a tuple keeps the comparison deterministic and lets the caller retain the
    first row when the evidence is tied.
    """

    qualities = [
        _persisted_rollout_quality(row)
        for rollout_id, row in persisted.items()
        if str(row.get("task_id", "")) == task_id
        and str(rollout_id).startswith(f"{task_id}-r")
    ]
    if not qualities:
        return (0, 0, 0, 0, 0)
    return tuple(sum(quality[index] for quality in qualities) for index in range(5))


def _deduplicate_resume_task_rows(
    rows: Sequence[Mapping[str, Any]],
    persisted: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate task rows left by overlapping driver instances.

    The durable task id is the rollout key.  Replaying duplicate rows would
    reset ``slots[task_id]`` in ``_collect`` and hide valid siblings from the
    effective-sample check.  Choose the row with the strongest persisted
    evidence; ties keep first-seen order so a clean cycle remains stable.
    """

    chosen: dict[str, tuple[tuple[int, ...], dict[str, Any]]] = {}
    order: list[str] = []
    for raw in rows:
        row = dict(raw)
        task_id = str(dict(row.get("task") or {}).get("task_id", ""))
        if not task_id:
            continue
        quality = _resume_proposal_quality(task_id, persisted)
        previous = chosen.get(task_id)
        if previous is None:
            chosen[task_id] = (quality, row)
            order.append(task_id)
        elif quality > previous[0]:
            chosen[task_id] = (quality, row)
    return [chosen[task_id][1] for task_id in order]


def _selection_from_dict(row: Mapping[str, Any]) -> Any:
    from .proposer import CandidateList, PairSelection

    candidate_list = CandidateList(
        list_id=str(row["list_id"]),
        cycle=int(row["cycle"]),
        dataset=str(row["dataset"]),
        scope=str(row["scope"]),
        anchor_pool_id=str(row["anchor_pool_id"]),
        candidates=tuple(dict(item) for item in row["candidates"]),
        messages=(),
        label_token_ids={str(k): int(v) for k, v in row["label_token_ids"].items()},
    )
    return PairSelection(
        list=candidate_list,
        choices=tuple(dict(item) for item in row["choices"]),
        same_choice=bool(row["same_choice"]),
        status=str(row["status"]),
        failure=row.get("failure"),
        prompt_token_ids=tuple(int(x) for x in row.get("prompt_token_ids", ())),
    )


@dataclass
class _Sibling:
    index: int
    plan: dict[str, Any]
    rollout: SolverRollout | None = None
    failure: dict[str, Any] | None = None

    @property
    def resolved(self) -> bool:
        return self.rollout is not None or self.failure is not None


def _sibling_rollout_recovery_reason(
    rollout: SolverRollout, *, dataset: str
) -> str | None:
    """Return a bounded-retry reason for a completed but unusable rollout.

    R2Orch deliberately keeps Worker output free-form.  A malformed or
    empty final answer is therefore a recoverable collection problem, rather
    than a reason to abort the whole cycle.  We only reopen output-contract
    failures and missing stateful output; ordinary verified zeroes remain valid
    negatives and are never resampled here.
    """

    metadata = rollout.trajectory.metadata
    if metadata.get("output_contract_failure"):
        return "output_contract_failure"
    dataset_key = canonical_dataset_name(dataset)
    if dataset_key in {"aime", "mbppplus"}:
        submission = metadata.get("answer_submission")
        if isinstance(submission, dict) and not submission.get("valid", True):
            submitted = str(submission.get("submitted_answer") or "").strip()
            if not submitted and metadata.get("reward_known") is not True:
                return "empty_or_invalid_submission"
        candidate = str(metadata.get("candidate_output") or "").strip()
        if not candidate and metadata.get("reward_known") is not True:
            return "empty_candidate_output"
    if dataset_key == "alfworld":
        environment = metadata.get("alfworld_environment_result", {})
        if isinstance(environment, dict) and str(environment.get("termination_reason", "")) in {
            "missing_output_agent",
            "agent_never_executed",
        }:
            return "missing_stateful_output"
    return None


# ------------------------------------------------------------------------ runner


class R2OrchRunner:
    def __init__(
        self,
        *,
        pair_proposer: Any,
        application_factory: Any,
        tokenizer: Any,
        snapshots: Any,
        output_dir: Path,
        cycle: int,
        base_seed: int,
        r2orch: Any,
        skill_snapshot: Mapping[str, Any],
        pool: Any,
        graph_feature_extractor: Any = None,
        execution_semantics: Mapping[str, Any] | None = None,
        run_config: SelfPlayRunConfig | None = None,
        stop_root: Path | None = None,
        stop_event: threading.Event | None = None,
        skill_ledger: Any,
    ) -> None:
        self.pair_proposer = pair_proposer
        self.application_factory = application_factory
        self.tokenizer = tokenizer
        self.snapshots = snapshots
        self.output_dir = Path(output_dir)
        self.cycle = int(cycle)
        self.base_seed = int(base_seed)
        self.r2orch = r2orch
        self.skill_snapshot = dict(skill_snapshot or {})
        self.pool = pool
        self.graph_feature_extractor = graph_feature_extractor
        self.execution_semantics = execution_semantics
        self.siblings = int(r2orch.siblings.independent_siblings)
        self.stop_root = Path(stop_root) if stop_root is not None else (
            self.output_dir.parent if self.output_dir.name.startswith("cycle-") else self.output_dir
        )
        self.stop_event = stop_event or threading.Event()
        self.cancel = threading.Event()
        # [r2orch.arms]: hidden/visible sibling arms around one card per task.
        self.arms = r2orch.arms
        self.skill_ledger = skill_ledger
        wall = float(r2orch.siblings.rollout_wall_time_s)
        if run_config is None:
            effective_run_config = SelfPlayRunConfig(
                rollouts_per_task=self.siblings,
                base_seed=self.base_seed,
                workers=int(r2orch.siblings.rollout_workers),
                rollout_wall_time_s=wall,
                aime_rollout_wall_time_s=aime_rollout_wall_time_s(r2orch, wall),
                stateful_rollout_wall_time_s=alfworld_rollout_wall_time_s(r2orch, wall),
                rollout_slot_wall_time_s=wall,
                stateful_slot_wall_time_s=alfworld_rollout_wall_time_s(r2orch, wall),
                request_wall_time_s=R2ORCH_REQUEST_WALL_TIME_S,
                aime_request_wall_time_s=R2ORCH_AIME_REQUEST_WALL_TIME_S,
                alfworld_request_wall_time_s=R2ORCH_ALFWORLD_REQUEST_WALL_TIME_S,
            )
        else:
            # The R2Orch driver passes an explicit config so it can add
            # CLI-specific budgets.  Preserve those fields, but still apply
            # R2Orch's long AIME/ALFWorld floors.
            effective_run_config = replace(
                run_config,
                request_wall_time_s=r2orch_request_wall_time_s(
                    run_config.request_wall_time_s
                ),
                aime_request_wall_time_s=r2orch_aime_request_wall_time_s(
                    run_config.aime_request_wall_time_s
                ),
                alfworld_request_wall_time_s=r2orch_alfworld_request_wall_time_s(
                    run_config.alfworld_request_wall_time_s
                ),
                aime_rollout_wall_time_s=aime_rollout_wall_time_s(
                    r2orch, float(run_config.aime_rollout_wall_time_s)
                ),
                stateful_rollout_wall_time_s=alfworld_rollout_wall_time_s(
                    r2orch, float(run_config.stateful_rollout_wall_time_s)
                ),
                stateful_slot_wall_time_s=alfworld_rollout_wall_time_s(
                    r2orch, float(run_config.stateful_slot_wall_time_s)
                ),
            )
        self.collector = SelfPlayRolloutRunner(
            proposer=pair_proposer,
            application_factory=application_factory,
            tokenizer=tokenizer,
            snapshots=snapshots,
            output_dir=self.output_dir,
            config=effective_run_config,
        )

    # -- stop ------------------------------------------------------------------

    def _stop_now(self) -> bool:
        launch = self.r2orch.launch
        return self.stop_event.is_set() or stop_requested(
            self.stop_root,
            margin_s=float(launch.collect_stop_margin_s),
            env=str(launch.stop_env),
            stop_file=str(launch.stop_file),
        )

    # -- main ------------------------------------------------------------------

    def run(self, anchors: Sequence[Any], *, resume: bool = False) -> DryRunSelfPlayResult:
        """Collect one cycle.  Durable rows in the cycle directory are always reused,
        so ``resume`` on an empty directory is an ordinary first start."""
        del resume
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        if self.execution_semantics is not None:
            validate_action_protocol(out, self.execution_semantics)
        selection_path = out / "training_selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()).get(
            "postcollection_complete"
        ):
            return self._load_complete()
        if self._stop_now():
            raise R2OrchStopped("stop requested before collection")
        from .pats import scope_view

        view = scope_view(self.skill_snapshot)
        selections, proposals = self._select(anchors, view)
        plans = self._plan(proposals)
        slots = self._collect(proposals, plans)
        return self._finalize(proposals, selections, slots)

    # -- selection -------------------------------------------------------------

    def _select(self, anchors: Sequence[Any], view: Mapping[str, Any]):
        out = self.output_dir
        lists_path, tasks_path = out / "proposer_lists.jsonl", out / "tasks.jsonl"
        state_path = out / "curriculum_state.json"
        pp = self.pair_proposer
        if lists_path.exists() and tasks_path.exists():
            selections = [_selection_from_dict(row) for row in _read_jsonl(lists_path)]
            task_rows = _read_jsonl(tasks_path)
            persisted_rows = _deduplicate_persisted_rollouts(
                _read_jsonl(out / "solver_rollouts.jsonl")
            )
            persisted = {str(row["rollout_id"]): row for row in persisted_rows}
            deduplicated_task_rows = _deduplicate_resume_task_rows(task_rows, persisted)
            proposals = [
                self._rehydrate(_proposal_from_dict(row)) for row in deduplicated_task_rows
            ]
            if len(deduplicated_task_rows) != len(task_rows):
                # Keep the durable manifest deterministic for subsequent
                # resumes.  This also prevents the duplicate task from
                # resetting its in-memory slot map in ``_collect``.
                _write_jsonl(tasks_path, [_proposal_dict(proposal) for proposal in proposals])
            if state_path.exists():
                pp.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
            else:
                # The scheduler was reloaded from the previous cycle; replay this
                # cycle's reservations exactly once.
                for selection in selections:
                    pp.reserve_choices(selection)
            return selections, proposals
        selections, proposals = [], []
        seen_task_ids: set[str] = set()
        for anchor in anchors:
            candidate_list = pp.build_candidate_list(anchor, cycle=self.cycle, scope_view=view)
            seeds = tuple(
                proposer_choice_seed(self.base_seed, candidate_list.list_id, k) for k in (0, 1)
            )
            selection = pp.propose_pair(candidate_list, seeds=seeds)
            pp.reserve_choices(selection)
            selections.append(selection)
            for k in (0, 1):
                proposal = pp.task_for_choice(selection, k)
                if proposal is not None:
                    # Multiple anchor lists can select the same underlying
                    # pool item.  Its task ID is the durable rollout key, so
                    # admitting it twice would launch two concurrent sibling
                    # groups that race to write the same rollout IDs (one may
                    # then overwrite a valid row with an unsafe partial).
                    if proposal.task.task_id in seen_task_ids:
                        continue
                    proposals.append(proposal)
                    seen_task_ids.add(proposal.task.task_id)
        _write_jsonl(tasks_path, [_proposal_dict(proposal) for proposal in proposals])
        _write_jsonl(lists_path, [selection.to_dict() for selection in selections])
        return selections, proposals

    def _rehydrate(self, proposal: ProposedTask) -> ProposedTask:
        source = self.pool.tasks[_pool_id(proposal)].task
        return replace(
            proposal,
            task=replace(proposal.task, private_verifier_payload=source.private_verifier_payload),
        )

    # -- sibling plans -----------------------------------------------------------

    def _plan(self, proposals: Sequence[ProposedTask]):
        from .pats import resolve_scope

        plans: dict[str, list[dict[str, Any]]] = {}
        assignments: dict[str, dict[str, Any]] = {}
        arm_plans = self._arm_plans(proposals)
        for proposal in proposals:
            task = proposal.task
            info = _task_info(proposal)
            dataset = canonical_dataset_name(task.metadata.get("dataset", info.get("dataset", "")))
            scope = str(info.get("scope") or resolve_scope(task.task_type, task.metadata))
            # Every sibling asks for the retrieved context; the hidden arm loses only card k
            # (resolved at retrieval, adaptive.py).
            arm_plan = arm_plans[task.task_id]
            base = {
                "schema": SIBLING_SCHEMA,
                "cycle": self.cycle,
                "task_id": task.task_id,
                "list_id": info.get("list_id"),
                "choice_index": info.get("choice_index"),
                "dataset": dataset,
                "task_type": task.task_type,
                "scope": scope,
                "skill_snapshot_id": str(self.skill_snapshot.get("snapshot_id", "")),
            }
            plans[task.task_id] = [
                {
                    **base,
                    "sibling_index": index,
                    "execution_seed": sibling_execution_seed(self.base_seed, task.task_id, index),
                    "skill_arm_requested": arm_plan["arms"][index],
                    "skill_arm": arm_plan["arms"][index],
                    "arm_draws": arm_plan["draws"],
                    # audit_share: k is this retired card, not the Thompson pick.
                    **(
                        {"arm_audit_card": arm_plan["audit_card"]}
                        if arm_plan.get("audit")
                        else {}
                    ),
                }
                for index in range(self.siblings)
            ]
            assignments[task.task_id] = {
                "scope": scope,
                "skill_arms": list(arm_plan["arms"]),
                "arm_scope": arm_plan["scope"],
                "arm_candidate_cards": len(arm_plan["draws"]),
                **(
                    {"arm_audit": True, "arm_audit_card": arm_plan["audit_card"]["key"]}
                    if arm_plan.get("audit")
                    else {}
                ),
            }
        _write_json(self.output_dir / "scaffold_assignments.json", assignments)
        return plans

    def _arm_plans(self, proposals: Sequence[ProposedTask]) -> dict[str, dict[str, Any]]:
        """Arm plan of every task, written once to ``skill_arm_plan.json``.

        ``arms[i]`` is sibling i's arm (seeded by cycle and task id: the first ``arms.hidden`` slots
        of a permutation are hidden).  ``draws`` holds one Thompson draw of Delta for every card
        the task's retrieval can show (the scope's cards of the frozen snapshot, the pool
        ``DirectorSkillBank.select_context`` ranks), from the ledger records of earlier cycles
        (the last ``window_cycles`` of them); card k is the retrieved card with the
        smallest draw.  A task whose scope holds ledger-retired cards is, with probability
        ``audit_share``, an audit: ``audit_card`` (a retired card, uniform) replaces the Thompson
        pick as k.  A resumed cycle reuses the stored plan, so its siblings keep their arms, draws
        and audits whatever the ledger holds by then.
        """
        from .skill_ledger import ARM_PLAN_SCHEMA

        path = self.output_dir / "skill_arm_plan.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if stored and (
            stored.get("schema") != ARM_PLAN_SCHEMA or int(stored.get("cycle", -1)) != self.cycle
        ):
            raise ValueError(f"{path} does not belong to cycle {self.cycle}")
        plans = dict(stored.get("tasks") or {})
        missing = [
            proposal
            for proposal in proposals
            if proposal.task.task_id not in plans
        ]
        if missing:
            # Records of the cycles before this one (cycle <= self.cycle - 1), windowed.
            contrasts = self.skill_ledger.decision_contrasts(through_cycle=self.cycle - 1)
            for proposal in missing:
                plans[proposal.task.task_id] = self._new_arm_plan(proposal.task, contrasts)
            _write_json(
                path,
                {
                    "schema": ARM_PLAN_SCHEMA,
                    "cycle": self.cycle,
                    "hidden": int(self.arms.hidden),
                    "visible": int(self.arms.visible),
                    "tasks": plans,
                },
            )
        return plans

    def _new_arm_plan(
        self, task: TaskSpec, contrasts: Mapping[str, Sequence[float]]
    ) -> dict[str, Any]:
        from .pats import resolve_scope
        from .skill_ledger import arm_assignment, ledger_rng, record_card_key, thompson_pick

        # The scope retrieval resolves for this task (DirectorSkillBank.select_context).
        scope = resolve_scope(task.task_type, task.metadata)
        cards = {record_card_key(record): record for record in _retrievable_cards(self.skill_snapshot, scope)}
        _pick, draws = thompson_pick(
            {key: contrasts.get(key, []) for key in cards},
            ledger_rng(self.base_seed, self.cycle, "thompson", task.task_id),
            prior_sd=float(self.skill_ledger.thompson_prior_sd),
        )
        plan = {
            "scope": scope,
            "arms": list(
                arm_assignment(self.cycle, task.task_id, int(self.arms.hidden), siblings=self.siblings)
            ),
            "draws": {
                str(cards[key]["card"]["skill_id"]): {
                    "version": int(cards[key]["version"]),
                    "key": key,
                    "draw": draws[key],
                    "n": len(contrasts.get(key, [])),
                }
                for key in sorted(cards)
            },
        }
        audit = self._audit_card(task, scope, cards, contrasts)
        if audit is not None:
            plan.update(audit=True, audit_card=audit)
        return plan

    def _audit_card(
        self,
        task: TaskSpec,
        scope: str,
        cards: Mapping[str, Mapping[str, Any]],
        contrasts: Mapping[str, Sequence[float]],
    ) -> dict[str, Any] | None:
        """``arms.audit_share``: the retired card this task audits, or None.

        Candidates are the scope's ledger-retired cards of the frozen snapshot whose id no active
        card of the scope uses.  The draw comes from its own generator (seeded by cycle and task,
        like the Thompson draw, which it leaves unchanged); nothing is drawn when the scope has no
        retired card.  The plan carries the archived record, so the visible siblings render the
        exact retired version.
        """
        from .skill_ledger import audit_choice, ledger_rng

        share = float(self.arms.audit_share)
        active_ids = {str(record["card"]["skill_id"]) for record in cards.values()}
        retired = {
            str(entry["key"]): entry
            for entry in _retired_audit_cards(self.skill_snapshot, scope)
            if str(entry.get("id")) not in active_ids and isinstance(entry.get("record"), Mapping)
        }
        if not retired:
            return None
        key = audit_choice(retired, share, ledger_rng(self.base_seed, self.cycle, "audit", task.task_id))
        if key is None:
            return None
        entry = retired[key]
        return {
            "id": str(entry["id"]),
            "version": int(entry["version"]),
            "key": key,
            "n": len(contrasts.get(key, [])),
            "candidates": len(retired),
            "reason": entry.get("reason"),
            "record": json.loads(json.dumps(entry["record"])),
        }

    # -- collection ----------------------------------------------------------------

    def _collect_sibling(self, proposal: ProposedTask, plan: dict[str, Any]) -> _Sibling:
        index = int(plan["sibling_index"])
        sibling = _Sibling(index=index, plan=plan)
        if self.cancel.is_set():
            return sibling
        rollout_id = f"{proposal.task.task_id}-r{index}"
        # R2Orch calls _collect_primary directly for each sibling, so apply one
        # small, slot-local recovery budget here. A timeout or empty final
        # artifact must not cancel unrelated siblings; after the bounded retry
        # the slot is recorded and the group gate below decides whether enough
        # valid samples remain.
        configured_recoveries = self.collector.config.recovery_attempts
        # The driver may build this runtime without the CLI's run config, so ``recovery_attempts``
        # does not always reach it; the environment variable wins.
        # A replacement re-runs a whole sibling after an empty submission, which doubles the
        # slowest (AIME) slot of a cycle.  Scheduling only: not part of the execution semantics.
        override = os.environ.get("R2ORCH_RECOVERY_ATTEMPTS", "").strip()
        if override:
            configured_recoveries = int(override)
        recovery_budget = max(0, int(configured_recoveries))
        attempt_events: list[dict[str, Any]] = []
        for replacement_attempt in range(recovery_budget + 1):
            if self.cancel.is_set():
                return sibling
            attempt_plan = {
                **plan,
                "recovery_attempt": replacement_attempt,
                "execution_seed": plan["execution_seed"],
            }
            collection = None
            try:
                collection = self.collector._collect_primary(
                    (proposal, index),
                    replacement_attempt=replacement_attempt,
                    cancellation_event=self.cancel,
                    sibling_index=index,
                    execution_seed=plan["execution_seed"],
                    r2orch_metadata=attempt_plan,
                )
                issue = _sibling_rollout_recovery_reason(
                    collection.rollout, dataset=str(plan["dataset"])
                )
                if issue is not None:
                    attempt_event = {
                        "rollout_id": rollout_id,
                        "task_id": proposal.task.task_id,
                        "dataset": plan["dataset"],
                        "replacement_attempt": replacement_attempt,
                        "error_type": "RecoverableRolloutOutputError",
                        "recovery_reason": issue,
                        "accepted_as_primary": False,
                        "trajectory": collection.rollout.trajectory.to_dict(),
                    }
                    if replacement_attempt < recovery_budget:
                        attempt_events.append(attempt_event)
                        _append_jsonl(
                            self.output_dir / "rollout_attempts.jsonl", attempt_event
                        )
                        collection.close()
                        continue
                    # Preserve the malformed/empty response as an attempt for
                    # diagnosis, but do not admit it as a training sibling.
                    attempt_events.append(attempt_event)
                    _append_jsonl(self.output_dir / "rollout_attempts.jsonl", attempt_event)
                    failure = {
                        "rollout_id": rollout_id,
                        "task_id": proposal.task.task_id,
                        "dataset": plan["dataset"],
                        "error_type": "RecoverableRolloutOutputError",
                        "phase": "independent",
                        "recovery_attempts": len(attempt_events),
                        "recovery_reasons": [
                            str(event.get("recovery_reason", ""))
                            for event in attempt_events
                        ],
                        "error": issue,
                    }
                    sibling.failure = failure
                    _append_jsonl(self.output_dir / "rollout_errors.jsonl", failure)
                    collection.close()
                    return sibling
                _append_jsonl(
                    self.output_dir / "solver_rollouts.jsonl",
                    collection.rollout.trajectory.to_dict(),
                )
                sibling.rollout = collection.rollout
                collection.close()
                return sibling
            except Exception as exc:
                if collection is not None:
                    collection.close()
                if self.cancel.is_set():
                    return sibling
                # Time-boxed tail: a slot that fails (for example at its wall-clock limit) is recorded
                # as failed (the other siblings still train) instead of re-running the straggler.
                failure = {
                    "rollout_id": rollout_id,
                    "task_id": proposal.task.task_id,
                    "dataset": plan["dataset"],
                    "error_type": type(exc).__name__,
                    "phase": "independent",
                    "error": str(exc)[:2000],
                }
                if attempt_events:
                    failure["recovery_attempts"] = len(attempt_events)
                    failure["recovery_reasons"] = [
                        str(event.get("recovery_reason", ""))
                        for event in attempt_events
                    ]
                sibling.failure = failure
                _append_jsonl(self.output_dir / "rollout_errors.jsonl", failure)
                return sibling
        return sibling

    def _order_jobs(
        self, jobs: list[tuple[ProposedTask, dict[str, Any]]]
    ) -> list[tuple[ProposedTask, dict[str, Any]]]:
        """Submission order of the sibling jobs (scheduling only).

        ``longest_first`` submits the datasets of ``siblings.longest_first_datasets`` (AIME and
        ALFWorld, whose slowest rollout gates the whole cycle) before the short ones; within a
        class the proposal order is kept (the sort is stable), so the siblings of a task
        still start together."""

        if self.r2orch.siblings.job_order == "fifo":
            return jobs
        priority = {
            canonical_dataset_name(name): rank
            for rank, name in enumerate(self.r2orch.siblings.longest_first_datasets)
        }
        rest = len(priority)
        return sorted(
            jobs, key=lambda job: priority.get(canonical_dataset_name(job[1]["dataset"]), rest)
        )

    def _collect(self, proposals: Sequence[ProposedTask], plans: Mapping[str, list[dict]]):
        out = self.output_dir
        persisted_rows = _deduplicate_persisted_rollouts(
            _read_jsonl(out / "solver_rollouts.jsonl")
        )
        # Compact interrupted/resumed files before reconstructing slots.  This
        # is safe at the single-driver boundary and prevents a later invalid
        # retry with the same rollout ID from masking a valid earlier sample.
        if (out / "solver_rollouts.jsonl").exists():
            _write_jsonl(out / "solver_rollouts.jsonl", persisted_rows)
        persisted = {row["rollout_id"]: row for row in persisted_rows}
        failed = {row["rollout_id"]: row for row in _read_jsonl(out / "rollout_errors.jsonl")}
        slots: dict[str, dict[int, _Sibling]] = {}
        jobs: list[tuple[ProposedTask, dict[str, Any]]] = []
        for proposal in proposals:
            task_id = proposal.task.task_id
            slots[task_id] = {}
            for plan in plans[task_id]:
                sibling = _Sibling(index=plan["sibling_index"], plan=plan)
                rollout_id = f"{task_id}-r{sibling.index}"
                if rollout_id in persisted:
                    sibling.rollout = _rollout_from_row(persisted[rollout_id])
                elif rollout_id in failed:
                    sibling.failure = failed[rollout_id]
                else:
                    jobs.append((proposal, plan))
                slots[task_id][sibling.index] = sibling
        try:
            self._check_failures(proposals, slots)
        except CollectionInfrastructureIncidentError as exc:
            self._record_abort(exc)
            raise
        jobs = self._order_jobs(jobs)
        stopping = False
        incident: Exception | None = None
        with ThreadPoolExecutor(
            max_workers=max(1, int(self.r2orch.siblings.rollout_workers)),
            thread_name_prefix="r2orch-sibling",
        ) as sibling_pool:
            pending: dict[Future, str] = {}
            for proposal, plan in jobs:
                future = sibling_pool.submit(self._collect_sibling, proposal, plan)
                pending[future] = proposal.task.task_id
            while pending:
                done, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
                if not stopping and self._stop_now():
                    stopping = True
                    self.cancel.set()
                for future in done:
                    task_id = pending.pop(future)
                    sibling = future.result()
                    if not sibling.resolved:
                        continue  # cancelled before completion; recollected on resume
                    slots[task_id][sibling.index] = sibling
                    if sibling.failure is not None and incident is None:
                        try:
                            self._check_failures(proposals, slots)
                        except CollectionInfrastructureIncidentError as exc:
                            incident = exc
                            self.cancel.set()
        # A rollout can finish without raising while still being unusable for
        # training (for example an unknown/empty answer with no valid reward).
        # Apply the same effective-sample boundary after the pool drains so we
        # do not silently produce an empty update in that case.
        if incident is None:
            try:
                self._check_failures(proposals, slots)
            except CollectionInfrastructureIncidentError as exc:
                incident = exc
        if incident is not None:
            self._record_abort(incident)
            raise incident
        if stopping:
            raise R2OrchStopped("stop requested during collection")
        return slots

    def _record_abort(self, incident: Exception) -> None:
        """Keep the failed slots for audit; a later --resume recollects them."""
        out = self.output_dir
        errors = out / "rollout_errors.jsonl"
        index = len(list(out.glob("rollout_errors.abort-*.jsonl")))
        if errors.exists():
            os.replace(errors, out / f"rollout_errors.abort-{index}.jsonl")
        _append_jsonl(
            out / "collection_abort.jsonl",
            {"error_type": type(incident).__name__, "error": str(incident), "abort_index": index},
        )

    def _check_failures(self, proposals, slots) -> None:
        planned: dict[str, int] = {}
        failures: dict[str, int] = {}
        eligible_by_task: dict[str, int] = {}
        for proposal in proposals:
            task_id = proposal.task.task_id
            for sibling in slots[proposal.task.task_id].values():
                dataset = sibling.plan["dataset"]
                planned[dataset] = planned.get(dataset, 0) + 1
                failures[dataset] = failures.get(dataset, 0) + int(sibling.failure is not None)
                if sibling.rollout is not None and not trajectory_exclusions(
                    sibling.rollout.trajectory
                ):
                    eligible_by_task[task_id] = eligible_by_task.get(task_id, 0) + 1
        # A small number of bad slots is tolerated.  Collection should only
        # abort when a task has no usable trajectory left and unresolved slots
        # cannot recover one.  The training-selection layer already skips a
        # task with fewer than two usable siblings, while retaining every
        # other task's valid pair.  Requiring two here made one malformed or
        # policy-ineligible sibling abort an otherwise trainable cycle.
        minimum_effective_samples = 1
        insufficient: list[tuple[str, str, int]] = []
        for proposal in proposals:
            task_id = proposal.task.task_id
            planned_task = len(slots[task_id])
            resolved_task = sum(int(sibling.resolved) for sibling in slots[task_id].values())
            eligible_task = eligible_by_task.get(task_id, 0)
            # Do not abort on the first failure while other siblings are still
            # running.  Abort only once the task is complete, or once the
            # unresolved slots can no longer bring it to the minimum pair size.
            remaining_task = max(0, planned_task - resolved_task)
            if planned_task and eligible_task + remaining_task < minimum_effective_samples:
                insufficient.append(
                    (
                        task_id,
                        str(_task_info(proposal).get("dataset") or slots[task_id][0].plan["dataset"]),
                        eligible_task,
                    )
                )
        # One task can legitimately lose every sibling to a model-policy or
        # formatting failure while the other task groups still contain usable
        # trajectories.  Skip that task in training; abort only when the
        # entire cycle has no usable trajectory at all.  This keeps a single
        # bad AIME/MBPP+ group from discarding otherwise valid work.
        if insufficient and not any(eligible_by_task.values()):
            details = ", ".join(
                f"{dataset}/{task_id}: {effective}/{minimum_effective_samples} usable"
                for task_id, dataset, effective in insufficient
            )
            totals = ", ".join(
                f"{dataset}: {failures.get(dataset, 0)}/{planned.get(dataset, 0)} failed"
                for dataset in sorted(planned)
            )
            raise CollectionInfrastructureIncidentError(
                "insufficient effective rollout samples; " + details + "; " + totals
            )

    # -- post-collection ---------------------------------------------------------------

    def _record_arms(self, proposals, slots) -> dict[str, Any]:
        """Write this cycle's skill-ledger records and the per-task arm report ``skill_arms.json``.

        A task contributes one record when at least one outcome-trusted sibling of each arm
        completed (``skill_ledger.task_arm_record``); the reward is the raw correctness outcome.
        """
        from .skill_ledger import (
            ARM_REPORT_SCHEMA,
            task_arm_record,
            task_duration,
        )

        records: list[dict[str, Any]] = []
        tasks: dict[str, dict[str, Any]] = {}
        durations: dict[str, list[float]] = {}
        for proposal in proposals:
            task_id = proposal.task.task_id
            info = _task_info(proposal)
            siblings, rows = [], []
            for index, sibling in sorted(slots[task_id].items()):
                if sibling.rollout is None:
                    rows.append({"index": index, "status": "failed"})
                    continue
                meta = sibling.rollout.trajectory.metadata.get("r2orch") or {}
                trusted = not _outcome_exclusions(sibling.rollout)
                rows.append(
                    {
                        "index": index,
                        "arm": meta.get("skill_arm"),
                        "condition": meta.get("scaffold_condition"),
                        "card": (meta.get("arm_card") or {}).get("key"),
                        "reward": float(sibling.rollout.trajectory.reward),
                        "trusted": trusted,
                        "duration_s": sibling.rollout.trajectory.metadata.get("duration_s"),
                    }
                )
                if trusted:
                    siblings.append((float(sibling.rollout.trajectory.reward), meta))
            # T_q of a Proposer-chosen task feeds its dataset's duration EMA (compute factor).
            seconds = task_duration(row.get("duration_s") for row in rows)
            if seconds is not None:
                dataset = canonical_dataset_name(
                    proposal.task.metadata.get("dataset", info.get("dataset", ""))
                )
                durations.setdefault(dataset, []).append(seconds)
            scope = next(
                (str(meta["scope"]) for _reward, meta in siblings if meta.get("scope")),
                str(info.get("scope") or ""),
            )
            record, status = task_arm_record(
                cycle=self.cycle, task_id=task_id, scope=scope, siblings=siblings
            )
            entry: dict[str, Any] = {"status": status, "scope": scope, "siblings": rows}
            if record is not None:
                records.append(record)
                entry.update(
                    {key: record[key] for key in ("card", "n_h", "n_v", "ybar_h", "ybar_v", "z")}
                )
            if any((sibling.plan or {}).get("arm_audit_card") for sibling in slots[task_id].values()):
                entry["audit"] = True  # audit_share: k was a ledger-retired card
            tasks[task_id] = entry
        statuses: dict[str, int] = {}
        for entry in tasks.values():
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        report = {
            "schema": ARM_REPORT_SCHEMA,
            "cycle": self.cycle,
            "records": len(records),
            "statuses": dict(sorted(statuses.items())),
            "tasks": tasks,
        }
        ledger_report: dict[str, Any] = {"statuses": report["statuses"]}
        report["audit_tasks"] = sum(1 for entry in tasks.values() if entry.get("audit"))
        report["audit_records"] = sum(1 for record in records if record.get("audit"))
        ledger_report["audit_records"] = report["audit_records"]
        self.skill_ledger.append_cycle(
            self.cycle, records, report=ledger_report, durations=durations
        )
        _write_json(self.output_dir / "skill_arms.json", report)
        return report

    def _finalize(self, proposals, selections, slots):
        from .director_update import DirectorParams, build_director_batch
        from .execution_contract import bind_rollout_contract, contract_source_hashes
        from .features import execution_policy_features_many
        from .proposer_reward import build_frontier_scores, build_proposer_pair_batch
        from .skill_ledger import DEFAULT_SCOPE_PRIOR

        out = self.output_dir
        rollouts = [
            slots[proposal.task.task_id][index].rollout
            for proposal in proposals
            for index in sorted(slots[proposal.task.task_id])
            if slots[proposal.task.task_id][index].rollout is not None
        ]
        trusted_rewards = {
            proposal.task.task_id: [
                sibling.rollout.trajectory.reward
                for _index, sibling in sorted(slots[proposal.task.task_id].items())
                if sibling.rollout is not None
                and not trajectory_exclusions(sibling.rollout.trajectory)
            ]
            for proposal in proposals
        }
        state_path = out / "curriculum_state.json"
        if not state_path.exists():
            # ADS sees only the independent siblings.
            observed = [
                {
                    "task_id": proposal.task.task_id,
                    "pool_id": _pool_id(proposal),
                    "rewards": trusted_rewards[proposal.task.task_id],
                }
                for proposal in proposals
                if trusted_rewards[proposal.task.task_id]
            ]
            self.pair_proposer.observe_many([(row["pool_id"], row["rewards"]) for row in observed])
            _write_jsonl(out / "curriculum_observations.jsonl", observed)
            _write_json(state_path, self.pair_proposer.state_dict())
        graphs = [rollout.graph for rollout in rollouts]
        features = (
            self.graph_feature_extractor.extract_many(graphs)
            if self.graph_feature_extractor is not None
            else execution_policy_features_many(graphs)
        )
        features_by_rollout = {
            rollout.trajectory.rollout_id: feature
            for rollout, feature in zip(rollouts, features, strict=True)
        }
        self._record_arms(proposals, slots)
        frontier_scores = build_frontier_scores(
            tasks=proposals,
            rollouts=rollouts,
            arms={
                "gamma": float(self.arms.gamma),
                "endpoint_penalty": float(self.arms.endpoint_penalty),
                "shrink_strength": float(self.arms.shrink_strength),
                # Shrinkage target: the scope's hidden success over earlier cycles only.
                "scope_prior": self.skill_ledger.scope_hidden_priors(before_cycle=self.cycle),
                "default_prior": DEFAULT_SCOPE_PRIOR,
                # Compute normaliser: the dataset's T_q EMA through this cycle.
                "time_exponent": float(self.arms.time_exponent),
                "dataset_duration": self.skill_ledger.dataset_durations(through_cycle=self.cycle),
            },
        )
        proposer_batch = build_proposer_pair_batch(
            selections=selections,
            current_tasks=proposals,
            frontier_by_task={score.task_id: score for score in frontier_scores},
        )
        solver_batch = build_director_batch(
            rollouts=rollouts,
            features_by_rollout=features_by_rollout,
            # One mapping config -> params (director_update.DirectorParams.from_config); the Director
            # only: the frontier and curriculum keep the raw correctness reward.
            params=DirectorParams.from_config(self.r2orch),
        )
        self.snapshots.advance()
        snapshots = {
            "proposer": self.snapshots.proposer_snapshot,
            "solver": self.snapshots.solver_snapshot,
            "phase": self.snapshots.phase.value,
            "cycle": self.snapshots.cycle,
        }
        if rollouts:
            proposer_batch, solver_batch = bind_rollout_contract(
                (proposer_batch, solver_batch), rollouts
            )
        _write_json(out / "frontier_scores.json", [asdict(score) for score in frontier_scores])
        _write_json(out / "proposer_batch.json", proposer_batch.to_dict())
        _write_json(out / "solver_batch.json", solver_batch.to_dict())
        _write_json(out / "snapshots.json", snapshots)
        (out / "training_relation_credits.jsonl").write_text("", encoding="utf-8")
        _write_json(out / "source_hashes.json", contract_source_hashes())
        choices = [
            choice for selection in selections for choice in selection.to_dict()["choices"]
        ]
        extraction = _proposal_extraction_summary(
            [
                {
                    "task_id": choice.get("task_id"),
                    "success": bool(choice.get("valid")),
                    "failure_kind": choice.get("failure"),
                }
                for choice in choices
            ]
        )
        _write_json(out / "proposal_extraction.json", extraction)
        _write_json(
            out / "training_selection.json",
            {
                "schema_version": SELECTION_SCHEMA,
                "postcollection_complete": True,
                "artifacts_sha256": {
                    name: _sha256_file(out / name)
                    for name in (
                        "solver_batch.json",
                        "proposer_batch.json",
                        "frontier_scores.json",
                        "snapshots.json",
                        "training_relation_credits.jsonl",
                    )
                },
            },
        )
        return DryRunSelfPlayResult(
            tasks=tuple(proposal.task for proposal in proposals),
            frontier_scores=tuple(frontier_scores),
            proposer_batch=proposer_batch,
            solver_batch=solver_batch,
            snapshots=snapshots,
            proposal_extraction=extraction,
        )

    def _load_complete(self) -> DryRunSelfPlayResult:
        from .training import load_training_batch

        out = self.output_dir
        proposals = [_proposal_from_dict(row) for row in _read_jsonl(out / "tasks.jsonl")]
        state_path = out / "curriculum_state.json"
        if state_path.exists():
            self.pair_proposer.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
        extraction_path = out / "proposal_extraction.json"
        return DryRunSelfPlayResult(
            tasks=tuple(proposal.task for proposal in proposals),
            frontier_scores=tuple(
                FrontierScore(**row)
                for row in json.loads((out / "frontier_scores.json").read_text(encoding="utf-8"))
            ),
            proposer_batch=load_training_batch(out / "proposer_batch.json"),
            solver_batch=load_training_batch(out / "solver_batch.json"),
            snapshots=json.loads((out / "snapshots.json").read_text(encoding="utf-8")),
            proposal_extraction=(
                json.loads(extraction_path.read_text(encoding="utf-8"))
                if extraction_path.exists()
                else {}
            ),
        )


# -------------------------------------------------------------------- experiment


def _with_served_names(config: Any, names: Mapping[str, str]) -> Any:
    return replace(
        config,
        proposer_model=replace(config.proposer_model, served_model=names["proposer"]),
        solver_model=replace(config.solver_model, served_model=names["director"]),
    )


def run_r2orch_experiment(args: Any, tracker: Any) -> int:
    """`selfplay-experiment`."""
    from . import cli
    from .application import (
        consolidate_selfplay_skills,
        create_adaptive_application,
        create_selfplay_snapshots,
        load_adaptive_config,
    )
    from .async_cycle import (
        bind_rollout_result_lineage,
        pats_skill_context_lineage,
        validate_batch_lineage,
    )
    from .curriculum import CurriculumProfile
    from .execution_contract import R2ORCH_CONTRACT_VERSION
    from .features import E5DelegationEncoder, SemanticGraphFeatureExtractor
    from .latency import RouteLatencyTracker, RouteTokenTracker
    from .outcome_metrics import collect_outcome_metrics, record_provenance
    from .cycle_metrics import write_dashboard
    from .vllm_serving import ensure_rollout_service, publish_latest_adapters
    from .proposer import create_pair_proposer
    from .selfplay_runtime import ByteTokenizer, HuggingFaceTokenizer
    from .skill_store import freeze_collection
    from .training import AlternatingGRPOTrainer, AlternatingTrainingConfig, MockPolicyTrainer
    from .training_metrics import TrainingMetricsStore, collect_cycle_metrics

    if args.cycles <= 0:
        raise ValueError("cycles must be positive")
    if not args.task_pool:
        raise ValueError("R2Orch selfplay-experiment requires --task-pool")
    config = replace(load_adaptive_config(args.config), verifier=args.verifier)
    config.validate()
    r2orch = config.r2orch
    launch = r2orch.launch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def stop_now(margin_s: float) -> bool:
        return stop_requested(
            output, margin_s=margin_s, env=str(launch.stop_env), stop_file=str(launch.stop_file)
        )

    expected_semantics = config.model_manifest()["execution_semantics"]
    check_run_protocol(output, expected_semantics)
    profile = CurriculumProfile.from_toml(args.curriculum_profile)
    pool, scheduler, retriever = cli._fixed_pool_components(
        args.task_pool,
        config=config,
        mock=args.mock,
        seed=config.seed,
        profile=profile,
        require_validation_manifest=False,
    )
    proposer_device, solver_device = cli._training_devices(config, args.device)
    args.gradient_accumulation_steps = math.ceil(args.mini_batch_size / args.micro_batch_size)
    steps_per_cycle = int(r2orch.director.ppo_mini_batches)
    # The Proposer's cosine schedule must span the steps it really takes: with
    # ``proposer.optimizer_steps_per_cycle`` set, that replaces the Director's step count.
    proposer_steps = int(r2orch.proposer.optimizer_steps_per_cycle)
    training_config = AlternatingTrainingConfig(
        cli._policy_training_config(
            config.proposer_model.base_model_path,
            config.proposer_model.checkpoint_path,
            args.proposer_learning_rate,
            proposer_device,
            args,
            total_optimizer_steps=(
                args.proposer_total_optimizer_steps
                or (args.cycles * proposer_steps if proposer_steps else 0)
                or args.total_optimizer_steps
                or args.cycles * steps_per_cycle
            ),
            r2orch=r2orch,
        ),
        cli._policy_training_config(
            config.solver_model.base_model_path,
            config.solver_model.checkpoint_path,
            args.solver_learning_rate,
            solver_device,
            args,
            total_optimizer_steps=(
                args.solver_total_optimizer_steps
                or args.total_optimizer_steps
                or args.cycles * steps_per_cycle
            ),
            r2orch=r2orch,
        ),
        output / "training_state.json",
        parallel_roles=args.parallel_role_training,
    )
    training_config = replace(
        training_config,
        solver=replace(
            training_config.solver,
            dtype=args.solver_training_dtype,
            data_parallel_gpu_ids=tuple(args.solver_data_parallel_gpus or ()),
        ),
    )
    training_config.validate()
    factory = (
        (lambda role, policy, seed: MockPolicyTrainer(role, policy, seed=seed))
        if args.mock_trainer
        else None
    )
    progress_path = output / "experiment_progress.json"
    previous_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if args.resume and progress_path.exists()
        else {}
    )
    summaries: list[dict[str, Any]] = list(previous_progress.get("cycles", []))
    completed_cycles = int(previous_progress.get("completed_cycles", 0))
    if completed_cycles >= args.cycles:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    if completed_cycles:
        previous_state = output / f"cycle-{completed_cycles - 1:04d}" / "curriculum_state.json"
        if previous_state.exists():
            scheduler.load_state_dict(json.loads(previous_state.read_text(encoding="utf-8")))
    served = {
        "proposer": config.proposer_model.served_model,
        "director": config.solver_model.served_model,
    }
    if completed_cycles:
        served_path = output / f"cycle-{completed_cycles - 1:04d}" / "served_adapters.json"
        if served_path.exists():
            saved = json.loads(served_path.read_text(encoding="utf-8"))
            served = {"proposer": saved["proposer"], "director": saved["director"]}
    config = _with_served_names(config, served)
    service = ensure_rollout_service(
        config, manage=not args.mock, mock=args.mock, state_dir=output / "services"
    )
    config = service
    # The resident service names adapters by optimizer step, as publish does.
    served = {
        "proposer": config.proposer_model.served_model,
        "director": config.solver_model.served_model,
    }
    provenance_id = record_provenance(output, args.config, args.task_pool)
    tracker.start(
        {
            "planned_cycles": args.cycles,
            "r2orch_contract_version": R2ORCH_CONTRACT_VERSION,
            "provenance_id": provenance_id,
            "independent_siblings": r2orch.siblings.independent_siblings,
            "rollout_workers": r2orch.siblings.rollout_workers,
            "datasets": list(r2orch.datasets_enabled),
            "mock": args.mock,
            "mock_trainer": args.mock_trainer,
        }
    )
    from .skill_ledger import SkillLedger, skill_ledger_path

    # Shared with the PATS lifecycle (skill_store.pats_controller): same file.
    skill_ledger = SkillLedger.from_arms(
        skill_ledger_path(r2orch), r2orch.arms, base_seed=int(r2orch.base_seed)
    )
    route_latency_tracker = RouteLatencyTracker(
        window_size=config.canvas.worker_latency_window
    )
    route_token_tracker = RouteTokenTracker(
        window_size=config.canvas.worker_token_window, path=output / "route_token_usage.json"
    )
    tokenizer = (
        ByteTokenizer()
        if args.mock
        else HuggingFaceTokenizer(config.proposer_model.base_model_path)
    )
    graph_feature_extractor = (
        None
        if args.mock
        else SemanticGraphFeatureExtractor(
            E5DelegationEncoder(config.skillbank_embedding_model_path or "intfloat/e5-base-v2")
        )
    )
    stop_event = threading.Event()

    def collect(
        cycle: int,
        *,
        collection_config: Any,
        behavior_update_index: int,
        mode: str,
        snapshots: Any,
    ):
        started = time.monotonic()
        cycle_dir = output / f"cycle-{cycle:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        collection_config = freeze_collection(collection_config, cycle_dir, step=cycle + 1)
        snapshot_path = cycle_dir / "director_skill_snapshot.json"
        skill_snapshot = (
            json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
        )
        pair_proposer = create_pair_proposer(
            collection_config, pool, scheduler, retriever, tokenizer=tokenizer, mock=args.mock
        )

        def application_factory(seed: int, cfg: Any = collection_config):
            return create_adaptive_application(
                replace(cfg, seed=seed),
                mock=args.mock,
                route_latency_tracker=route_latency_tracker,
                route_token_tracker=route_token_tracker,
                director_tokenizer=tokenizer,
            )

        base_seed = round_base_seed(r2orch.base_seed, cycle)
        wall = float(r2orch.siblings.rollout_wall_time_s)
        runner = R2OrchRunner(
            pair_proposer=pair_proposer,
            application_factory=application_factory,
            tokenizer=tokenizer,
            snapshots=snapshots,
            output_dir=cycle_dir,
            cycle=cycle,
            base_seed=base_seed,
            r2orch=r2orch,
            skill_snapshot=skill_snapshot,
            pool=pool,
            graph_feature_extractor=graph_feature_extractor,
            execution_semantics=expected_semantics,
            run_config=SelfPlayRunConfig(
                rollouts_per_task=int(r2orch.siblings.independent_siblings),
                base_seed=base_seed,
                max_tokens=args.max_sequence_length,
                workers=int(r2orch.siblings.rollout_workers),
                rollout_wall_time_s=wall,
                aime_rollout_wall_time_s=max(wall, R2ORCH_AIME_ROLLOUT_WALL_TIME_S),
                stateful_rollout_wall_time_s=max(
                    wall, R2ORCH_ALFWORLD_ROLLOUT_WALL_TIME_S
                ),
                rollout_slot_wall_time_s=wall,
                stateful_slot_wall_time_s=max(
                    wall, R2ORCH_ALFWORLD_ROLLOUT_WALL_TIME_S
                ),
                rollout_no_progress_time_s=args.rollout_no_progress_time_s,
                request_wall_time_s=r2orch_request_wall_time_s(args.request_wall_time_s),
                aime_request_wall_time_s=r2orch_aime_request_wall_time_s(
                    args.request_wall_time_s
                ),
                alfworld_request_wall_time_s=r2orch_alfworld_request_wall_time_s(
                    args.request_wall_time_s
                ),
            ),
            stop_root=output,
            stop_event=stop_event,
            skill_ledger=skill_ledger,
        )
        # Sample multiple independent anchor lists per dataset when requested.
        # This expands proposer opportunities without changing the two-choice
        # pair protocol or sharing a mutable driver across workers.
        lists_per_dataset = max(1, int(r2orch.proposer.lists_per_dataset))
        if scheduler.anchor_mode == "coverage_random":
            # Seeded draws over the whole pool (curriculum profile ``anchor_mode``); the sequential
            # schedule reads pool rows in file order and only ever scans the first ~cycle rows.
            anchors = pool.seeds_for_ids(
                scheduler.anchor_ids(
                    per_dataset=lists_per_dataset, cycle=cycle, seed=int(r2orch.base_seed)
                )
            )
        else:
            anchors = pool.balanced_selection_seeds(
                len(scheduler.dataset_cluster_ids) * lists_per_dataset,
                offset_per_dataset=cycle,
            )
        result = runner.run(anchors, resume=True)
        result = bind_rollout_result_lineage(
            result,
            output_dir=cycle_dir,
            target_cycle=cycle,
            behavior_update_index=behavior_update_index,
            collection_mode=mode,
        )
        return result, tokenizer, time.monotonic() - started

    pipeline = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2orch-next-cycle")
        if args.async_next_cycle_rollouts
        else None
    )
    try:
        try:
            collected = collect(
                completed_cycles,
                collection_config=config,
                behavior_update_index=completed_cycles,
                mode="synchronous",
                snapshots=create_selfplay_snapshots(config),
            )
        except R2OrchStopped:
            return int(launch.resume_exit_code)
        collection_wait_s = 0.0
        for cycle in range(completed_cycles, args.cycles):
            cycle_started = time.monotonic()
            cycle_dir = output / f"cycle-{cycle:04d}"
            rollout_result, director_tokenizer, collection_elapsed_s = collected
            # Where the learner's wall time goes outside the optimizer (diagnostic only).
            timings: dict[str, float] = {
                "collection_wait_before_cycle_s": round(collection_wait_s, 2)
            }
            if stop_now(float(launch.train_stop_margin_s)):
                return int(launch.resume_exit_code)
            phase_started = time.monotonic()
            consolidate_selfplay_skills(
                config,
                rollout_result,
                mock=args.mock,
                step=cycle + 1,
                cycle_dir=cycle_dir,
                tokenizer=director_tokenizer,
            )
            timings["consolidate_skills_s"] = round(time.monotonic() - phase_started, 2)
            next_future = None
            if pipeline is not None and cycle + 1 < args.cycles:
                next_future = pipeline.submit(
                    collect,
                    cycle + 1,
                    collection_config=config,
                    behavior_update_index=cycle,
                    mode="async_one_step_stale",
                    snapshots=create_selfplay_snapshots(config),
                )
            if stop_now(float(launch.train_stop_margin_s)):
                stop_event.set()
                if next_future is not None:
                    with contextlib.suppress(R2OrchStopped):
                        next_future.result()
                return int(launch.resume_exit_code)
            learner_snapshots = create_selfplay_snapshots(config)
            training_state = (
                json.loads(training_config.state_path.read_text(encoding="utf-8"))
                if training_config.state_path.exists()
                else {}
            )
            if int(training_state.get("cycle", 0)) <= cycle:
                binding = validate_batch_lineage(
                    rollout_result.proposer_batch,
                    rollout_result.solver_batch,
                    learner_update_index=cycle,
                    learner_proposer_snapshot=learner_snapshots.proposer_snapshot,
                    learner_solver_snapshot=learner_snapshots.solver_snapshot,
                    expected_skill_context=pats_skill_context_lineage(cycle_dir),
                    expected_execution_semantics=expected_semantics,
                )
                cli._atomic_write_json(cycle_dir / "learner_policy_binding.json", binding)
            trainer = AlternatingGRPOTrainer(training_config, trainer_factory=factory)
            training_started = time.monotonic()
            updates = trainer.committed_cycle_results(cycle) or trainer.train_cycle(
                rollout_result.proposer_batch, rollout_result.solver_batch, relation_credits=()
            )
            training_elapsed_s = time.monotonic() - training_started
            timings["train_cycle_s"] = round(training_elapsed_s, 2)
            # Hot-load the trainers' latest
            # adapters; names carry the optimizer step, matching ensure_rollout_service.
            phase_started = time.monotonic()
            published = publish_latest_adapters(
                config,
                previous=served,
                mock=args.mock,
            )
            timings["publish_adapters_s"] = round(time.monotonic() - phase_started, 2)
            served = {"proposer": published["proposer"], "director": published["director"]}
            _write_json(cycle_dir / "served_adapters.json", published)
            config = _with_served_names(config, served)
            phase_started = time.monotonic()
            outcomes = collect_outcome_metrics(
                cycle_dir,
                cycle=cycle,
                tasks=rollout_result.tasks,
                solver_batch=rollout_result.solver_batch,
                k=int(r2orch.siblings.independent_siblings),
                snapshots=rollout_result.snapshots,
            )
            timings["outcome_metrics_s"] = round(time.monotonic() - phase_started, 2)
            cycle_metrics = collect_cycle_metrics(
                cycle=cycle,
                proposer_batch=rollout_result.proposer_batch,
                solver_batch=rollout_result.solver_batch,
                updates=updates,
                frontier_scores=rollout_result.frontier_scores,
                skillbank_path=config.skillbank_path,
                cycle_dir=cycle_dir,
                context={
                    "config": str(Path(args.config).resolve()),
                    "provenance_id": provenance_id,
                    "r2orch_contract_version": R2ORCH_CONTRACT_VERSION,
                    "collection_elapsed_s": collection_elapsed_s,
                    "training_compute_elapsed_s": training_elapsed_s,
                    "cycle_elapsed_s": time.monotonic() - cycle_started + collection_elapsed_s,
                    "async_next_cycle_rollouts": bool(args.async_next_cycle_rollouts),
                    "mock_trainer": args.mock_trainer,
                    "mock_rollouts": args.mock,
                    "served_adapters": dict(served),
                    "phase_elapsed_s": dict(timings),
                },
                proposal_extraction=rollout_result.proposal_extraction,
            )
            cycle_metrics["outcomes"] = outcomes
            TrainingMetricsStore(output).append(cycle_metrics)
            tracker.log_cycle(cycle_metrics)
            write_dashboard(output)
            summaries.append(
                {
                    "cycle": cycle,
                    "tasks": len(rollout_result.tasks),
                    "mode": "train",
                    "updates": [asdict(update) for update in updates],
                    "served_adapters": dict(served),
                }
            )
            _write_json(progress_path, {"completed_cycles": cycle + 1, "cycles": summaries})
            if cycle + 1 >= args.cycles:
                break
            wait_started = time.monotonic()
            try:
                collected = (
                    next_future.result()
                    if next_future is not None
                    else collect(
                        cycle + 1,
                        collection_config=config,
                        behavior_update_index=cycle + 1,
                        mode="synchronous",
                        snapshots=create_selfplay_snapshots(config),
                    )
                )
            except R2OrchStopped:
                return int(launch.resume_exit_code)
            # Learner idle time: the collection (async or not) was the critical path.
            collection_wait_s = time.monotonic() - wait_started
    finally:
        if pipeline is not None:
            stop_event.set()  # an abandoned next-cycle collection stops at its next check
            pipeline.shutdown(wait=True, cancel_futures=True)
    write_dashboard(output)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0
