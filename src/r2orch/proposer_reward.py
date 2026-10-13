"""R2Orch Proposer reward and Proposer pair batches.

The Proposer reward of a task is the arms Bell reward (``arms_bell_reward``, ``[r2orch.arms]``);
it is stored as ``F_tilde`` and the pair batch consumes it.

Rewards here are ``trajectory.reward``: the raw correctness outcome.  The Worker-cost term
(``[r2orch.cost] lambda``) is deliberately NOT applied: the reward measures how much the outcome
of a task depends on the Director / scaffold (it is built from 0/1 results), so a continuous cost
would make every task look contested and would send the Proposer towards tasks where the Director
can save tokens rather than tasks where its choices change the answer.  The cost enters only the
Director's advantage (``director_update``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .rollouts import TrainingBatch, TrainingSample

PAIR_OBJECTIVE = "proposer_pair"


def arms_bell_reward(
    hidden: Sequence[float],
    visible: Sequence[float],
    *,
    prior: float,
    gamma: float = 0.5,
    endpoint_penalty: float = 0.2,
    shrink_strength: float = 2.0,
    siblings: Sequence[float] | None = None,
    time_factor: float = 1.0,
) -> dict[str, Any]:
    """Arm-coupled Proposer reward of one task.

    B(p) = 4 p (1 - p) and p~ = (s + m * prior) / (n + m), m = ``shrink_strength``, ``prior`` the
    scope's hidden-arm success (EMA).  With both arms observed:

    * all hidden + visible outcomes equal (all wrong or all right)  ->  -endpoint_penalty
    * the hidden arm has a success (mixed or partially successful)  ->  B(p_h~)
    * the hidden arm all wrong, the visible arm has a success       ->  gamma * B(p_v~)
      (the scaffold-assisted frontier: hard, but learnable with the card's help)

    Without arm data (no card was retrieved, or an arm lost every sibling) the task falls back to
    B(p~) over ``siblings`` (every trusted outcome; default hidden + visible) under the same
    endpoint rule.  Fewer than two outcomes give None.  Rewards are raw correctness outcomes.
    ``time_factor`` ((Tbar_d / T_q) ** time_exponent, learning progress per unit of compute)
    multiplies the positive Bell part only, never the endpoint penalty.
    """
    from .skill_ledger import bell, shrunk_rate

    hidden = [float(value) for value in hidden]
    visible = [float(value) for value in visible]
    arm_data = bool(hidden and visible)
    outcomes = (
        [*hidden, *visible] if arm_data or siblings is None else [float(v) for v in siblings]
    )
    m = float(shrink_strength)
    record: dict[str, Any] = {
        "arm_data": arm_data,
        "n": len(outcomes),
        "s": math.fsum(outcomes),
        "n_h": len(hidden),
        "s_h": math.fsum(hidden),
        "n_v": len(visible),
        "s_v": math.fsum(visible),
        "prior": float(prior),
        "shrink_strength": m,
        "gamma": float(gamma),
        "endpoint_penalty": float(endpoint_penalty),
        "p_h_tilde": None,
        "p_v_tilde": None,
        "p_tilde": None,
        "bell": None,
        "time_factor": float(time_factor),
    }
    factor = float(time_factor)
    if len(outcomes) < 2:
        return {**record, "case": "insufficient_outcomes", "value": None}
    equal = all(value == outcomes[0] for value in outcomes)
    if not arm_data:
        record["p_tilde"] = shrunk_rate(record["s"], len(outcomes), prior=prior, strength=m)
        if equal:
            return {**record, "case": "fallback_endpoint", "value": -float(endpoint_penalty)}
        record["bell"] = bell(record["p_tilde"])
        return {**record, "case": "fallback_bell", "value": factor * record["bell"]}
    record["p_h_tilde"] = shrunk_rate(record["s_h"], len(hidden), prior=prior, strength=m)
    record["p_v_tilde"] = shrunk_rate(record["s_v"], len(visible), prior=prior, strength=m)
    if equal:
        return {**record, "case": "endpoint", "value": -float(endpoint_penalty)}
    if record["s_h"] > 0.0:
        record["bell"] = bell(record["p_h_tilde"])
        return {**record, "case": "hidden_bell", "value": factor * record["bell"]}
    # Hidden all wrong and not every outcome equal: the visible arm has a success.
    record["bell"] = bell(record["p_v_tilde"])
    return {
        **record,
        "case": "assisted_bell",
        "value": factor * float(gamma) * record["bell"],
    }


def _arm_split(trusted: Sequence[Any]) -> tuple[list[float], list[float], dict[str, Any] | None]:
    """Hidden / visible raw rewards of a task's trusted siblings and their common card k.

    Arm data requires both arms and one card k for every armed sibling; otherwise both lists are
    returned empty so the task falls back to B over all siblings.
    """
    hidden, visible, cards = [], [], set()
    card = None
    for item in trusted:
        meta = item.trajectory.metadata["r2orch"]
        arm = meta.get("skill_arm")
        if arm not in ("hidden", "visible"):
            continue
        card = dict(meta.get("arm_card") or {})
        cards.add(str(card.get("key", "")))
        (hidden if arm == "hidden" else visible).append(float(item.trajectory.reward))
    if len(cards) != 1 or "" in cards or not hidden or not visible:
        return [], [], None
    return hidden, visible, card


def build_frontier_scores(
    *,
    tasks: Sequence[Any],
    rollouts: Sequence[Any],
    arms: Mapping[str, Any],
) -> list[Any]:
    """Return the FrontierScore of every task.

    ``arms`` = {"gamma", "endpoint_penalty", "shrink_strength", "scope_prior": {scope: {"value",
    "cycles"}}, "default_prior", "time_exponent", "dataset_duration"}: the record's ``F_tilde`` is
    ``arms_bell_reward`` (components under ``arms_bell``).
    """
    from .config import canonical_dataset_name
    from .proposer_learning import frontier_evidence_exclusions
    from .selfplay import FrontierScore
    from .skill_ledger import compute_factor, task_duration

    by_task: dict[str, list[Any]] = defaultdict(list)
    for rollout in rollouts:
        by_task[rollout.trajectory.task_id].append(rollout)
    scores = []
    for proposal in tasks:
        task = proposal.task
        task_r2orch = task.metadata.get("r2orch", {})
        siblings = sorted(
            by_task.get(task.task_id, []),
            key=lambda item: item.trajectory.metadata["r2orch"]["sibling_index"],
        )
        first = siblings[0].trajectory.metadata["r2orch"] if siblings else {}
        dataset = first.get("dataset") or canonical_dataset_name(task.metadata.get("dataset", ""))
        scope = first.get("scope") or task_r2orch.get("scope")
        trusted = [item for item in siblings if not frontier_evidence_exclusions(item)]
        rollout_ids = [item.trajectory.rollout_id for item in trusted]
        rewards = [float(item.trajectory.reward) for item in trusted]
        hidden, visible, card = _arm_split(trusted)
        prior_entry = dict((arms.get("scope_prior") or {}).get(scope) or {})
        # Learning progress per compute: T_q over the task's completed siblings against
        # the dataset's EMA (both picks of a Proposer list come from one dataset).
        task_seconds = task_duration(
            item.trajectory.metadata.get("duration_s") for item in siblings
        )
        dataset_entry = dict((arms.get("dataset_duration") or {}).get(dataset) or {})
        exponent = float(arms.get("time_exponent", 0.0))
        factor = compute_factor(task_seconds, dataset_entry.get("value"), exponent)
        reward = arms_bell_reward(
            hidden,
            visible,
            prior=float(prior_entry.get("value", arms.get("default_prior", 0.5))),
            gamma=float(arms["gamma"]),
            endpoint_penalty=float(arms["endpoint_penalty"]),
            shrink_strength=float(arms["shrink_strength"]),
            siblings=[float(item.trajectory.reward) for item in trusted],
            time_factor=factor,
        )
        active = reward["value"]
        metadata = {
            "schema": "r2orch_frontier@1",
            "task_id": task.task_id,
            "dataset": dataset,
            "scope": scope,
            "list_id": task_r2orch.get("list_id"),
            "choice_index": task_r2orch.get("choice_index"),
            "n": len(rewards),
            "rollout_ids": rollout_ids,
            "rewards": rewards,
            "F_tilde": active,
            "excluded_reason": "insufficient_trusted_siblings" if active is None else None,
            "arms_bell": {
                **reward,
                "arm_card": card,
                "prior_cycles": int(prior_entry.get("cycles", 0)),
                "T_q": task_seconds,
                "Tbar_d": dataset_entry.get("value"),
                "Tbar_d_cycles": int(dataset_entry.get("cycles", 0)),
                "time_exponent": exponent,
            },
        }
        scores.append(
            FrontierScore(
                task_id=task.task_id,
                validity=1.0,
                graph_local=active if active is not None else 0.0,
                rewards=tuple(rewards),
                stable_graph_local=active,
                reverify_status="r2orch",
                pair_stability=(),
                metadata={"r2orch": metadata},
            )
        )
    return scores


def _collected_pool_ids(current_tasks: Sequence[Any]) -> dict[str, str | None]:
    actual_pool_ids: dict[str, str | None] = {}
    for proposal in current_tasks:
        if isinstance(proposal, Mapping):
            task = proposal.get("task", proposal)
        else:
            task = getattr(proposal, "task", proposal)
        if isinstance(task, Mapping):
            task_id = str(task["task_id"])
            pool_id = (task.get("metadata") or {}).get("pool_id")
        else:
            task_id = str(task.task_id)
            pool_id = task.metadata.get("pool_id")
        if not isinstance(pool_id, str) or not pool_id:
            pool_id = None
        if task_id in actual_pool_ids and actual_pool_ids[task_id] != pool_id:
            # An ambiguous manifest cannot certify either choice.
            actual_pool_ids[task_id] = None
        else:
            actual_pool_ids[task_id] = pool_id
    return actual_pool_ids


def _matches_collected_task(choice: Mapping[str, Any], pool_ids: Mapping[str, str | None]) -> bool:
    pool_id = pool_ids.get(str(choice.get("task_id", "")))
    return pool_id is not None and choice.get("pool_id") == pool_id


def build_proposer_pair_batch(
    *,
    selections: Sequence[Any],
    current_tasks: Sequence[Any],
    frontier_by_task: Mapping[str, Any],
) -> TrainingBatch:
    """Pair samples: the target of a pair is the reward difference F~a - F~b.

    a == b, a failed list, a missing F~, or a choice that does not match the
    actual collected task's pool identity on either side yields no sample.
    """

    actual_pool_ids = _collected_pool_ids(current_tasks)

    def f_tilde(task_id: str) -> float | None:
        score = frontier_by_task.get(task_id)
        return None if score is None else score.metadata["r2orch"]["F_tilde"]

    samples: list[TrainingSample] = []
    skipped: dict[str, str] = {}
    for selection in selections:
        candidate_list = selection.list
        if selection.status != "ok":
            skipped[candidate_list.list_id] = "list_failed"
            continue
        if selection.same_choice:
            skipped[candidate_list.list_id] = "same_choice"
            continue
        if any(
            not _matches_collected_task(choice, actual_pool_ids)
            for choice in selection.choices[:2]
        ):
            skipped[candidate_list.list_id] = "task_pool_mismatch"
            continue
        sides = []
        for choice in selection.choices[:2]:
            sides.append(
                {
                    "label": choice["label"],
                    "index": candidate_list.labels.index(choice["label"]),
                    "pool_id": choice["pool_id"],
                    "task_id": choice["task_id"],
                    "F_tilde": f_tilde(choice["task_id"]),
                }
            )
        if any(side["F_tilde"] is None for side in sides):
            skipped[candidate_list.list_id] = "missing_f_tilde"
            continue
        a, b = sides
        target = a["F_tilde"] - b["F_tilde"]
        token_ids = tuple(int(value) for value in selection.prompt_token_ids)
        samples.append(
            TrainingSample(
                rollout_id=f"pair-{candidate_list.list_id}",
                task_id=candidate_list.list_id,
                token_ids=token_ids,
                action_mask=(0,) * (len(token_ids) - 1) + (1,),
                reward=target,
                advantage=target,
                metadata={
                    "r2orch": {
                        "schema": "r2orch_proposer_pair@1",
                        "cycle": candidate_list.cycle,
                        "list_id": candidate_list.list_id,
                        "dataset": candidate_list.dataset,
                        "label_token_ids": [
                            candidate_list.label_token_ids[label]
                            for label in candidate_list.labels
                        ],
                        "labels": list(candidate_list.labels),
                        "choice_a": a,
                        "choice_b": b,
                        "target": target,
                        # Candidate annotations for monitoring only (never used by the loss):
                        # P(source) drift and the policy-vs-previous-success proxy.
                        "label_sources": [
                            str(candidate.get("source", "")) for candidate in candidate_list.candidates
                        ],
                        "label_previous_success": [
                            candidate.get("previous_success") for candidate in candidate_list.candidates
                        ],
                        # The behaviour policy that sampled the picks; the loss ignores it (no
                        # importance ratio).
                        "behavior_logprobs": dict(selection.choices[0]["behavior_logprobs"]),
                    }
                },
            )
        )
    return TrainingBatch(
        role="proposer",
        samples=tuple(samples),
        objective=PAIR_OBJECTIVE,
        metadata={
            "training_selection_schema": "r2orch",
            "r2orch_proposer": {
                "schema": "r2orch_proposer_batch@1",
                "lists": len(selections),
                "pairs": len(samples),
                "nonzero_target_pairs": sum(
                    1 for sample in samples if abs(sample.advantage) > 1e-12
                ),
                "skipped": skipped,
            },
        },
    )


def filter_proposer_pair_task_bindings(
    batch: TrainingBatch, *, current_tasks: Sequence[Any]
) -> TrainingBatch:
    """Remove task-identity mismatches from an already-written batch, preserving valid targets.

    Used by the common batch loader so a ready collection cannot bypass the
    identity check on resume.
    """
    if batch.objective != PAIR_OBJECTIVE:
        return batch
    pool_ids = _collected_pool_ids(current_tasks)
    kept, removed = [], []
    for sample in batch.samples:
        pair = sample.metadata.get("r2orch", {})
        if all(
            _matches_collected_task(pair.get(side, {}), pool_ids)
            for side in ("choice_a", "choice_b")
        ):
            kept.append(sample)
        else:
            removed.append(sample.rollout_id)
    if not removed:
        return batch
    metadata = dict(batch.metadata)
    metadata["r2orch_proposer"] = {
        **metadata.get("r2orch_proposer", {}),
        "pairs": len(kept),
        "task_pool_binding_removed": removed,
    }
    return replace(batch, samples=tuple(kept), metadata=metadata)
