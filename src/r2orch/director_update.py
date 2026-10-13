"""R2Orch Director batch construction.

Group independent siblings by ``(task_id, scaffold_condition)``; groups of two or more
subtract the group mean, then whiten over the whole batch (verl ``masked_whiten``,
token-weighted).  A single-sibling group has no group-centered advantage and stays out of the
batch.  The torch loss lives in ``training.py``.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .rollouts import TrainingBatch, TrainingSample

DIRECTOR_OBJECTIVE = "r2orch_director"
SELECTION_SCHEMA = "r2orch"


@dataclass(frozen=True)
class DirectorParams:
    whiten_eps: float = 1e-8
    kl: float = 0.005
    low_signal_abs_advantage: float = 0.0
    low_signal_keep_fraction: float = 0.15
    # Weight of the Worker cost in the Director's outcome reward (``[r2orch.cost] lambda``).
    # 0 leaves the raw correctness reward, and every number derived from it, untouched.
    cost_lambda: float = 0.0

    @classmethod
    def from_config(cls, r2orch: Any) -> DirectorParams:
        """Map ``config.r2orch`` (``director.*``) onto the batch parameters."""

        director = r2orch.director
        return cls(
            whiten_eps=float(director.whiten_eps),
            kl=float(director.kl),
            low_signal_abs_advantage=float(director.low_signal_abs_advantage),
            low_signal_keep_fraction=float(director.low_signal_keep_fraction),
            cost_lambda=float(r2orch.cost.lambda_),
        )


def _keep_low_signal(rollout_id: str, fraction: float) -> bool:
    """Deterministic (seed-free) choice of the low-signal samples that stay as a KL anchor."""

    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(f"{rollout_id}|low_signal".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def masked_whiten_stats(
    values: Sequence[float], weights: Sequence[int]
) -> tuple[float, float, int]:
    """Token-weighted mean and unbiased variance, matching verl ``masked_whiten``."""

    tokens = int(sum(weights))
    if tokens == 0:
        return 0.0, 0.0, 0
    mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / tokens
    variance = (
        sum((value - mean) ** 2 * weight for value, weight in zip(values, weights, strict=True))
        / tokens
    )
    if tokens > 1:
        variance *= tokens / (tokens - 1)
    return mean, variance, tokens


def _action_token_count(trajectory: Any) -> int:
    calls = trajectory.policy_calls
    if calls:
        return sum(sum(1 for keep in call.action_mask[1:] if keep) for call in calls)
    return sum(1 for keep in trajectory.action_mask[1:] if keep)


def worker_cost_of(trajectory: Any) -> float | None:
    """The rollout's normalised Worker cost (``worker_cost.py``), or None when none was recorded."""

    value = (trajectory.metadata.get("r2orch") or {}).get("worker_cost")
    if value is None:
        record = trajectory.metadata.get("worker_cost_record") or {}
        value = record.get("worker_cost")
    return None if value is None else float(value)


def director_outcome_reward(trajectory: Any, cost_lambda: float) -> float:
    """The primary outcome reward the Director's group-relative advantage is built from.

    ``reward - lambda * worker_cost``.  With ``lambda == 0`` (the default) this is exactly
    ``float(trajectory.reward)``: the cost is not read, so no number changes.  Only the Director
    uses it.  The Proposer's reward F~ (``proposer_reward``), the scaffold success rates and the
    curriculum keep ``trajectory.reward``, the raw correctness: F~
    measures how much the *outcome* of a task depends on the Director, and a continuous cost would
    make every task look contested.
    """

    reward = float(trajectory.reward)
    if not cost_lambda:
        return reward
    cost = worker_cost_of(trajectory)
    return reward if cost is None else reward - float(cost_lambda) * cost


# ---------------------------------------------------------------------------
# Batch construction.


def build_director_batch(
    *,
    rollouts: Sequence[Any],
    features_by_rollout: Mapping[str, Any],
    params: DirectorParams,
) -> TrainingBatch:
    from .features import execution_policy_features_many
    from .graph_learning import canonical_graph_key
    from .training_selection import trajectory_exclusions

    by_id = {rollout.trajectory.rollout_id: rollout for rollout in rollouts}
    eligible = [
        rollout
        for rollout in rollouts
        if not trajectory_exclusions(rollout.trajectory)
    ]
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for rollout in eligible:
        r2orch = rollout.trajectory.metadata["r2orch"]
        groups[(rollout.trajectory.task_id, str(r2orch["scaffold_condition"]))].append(rollout)

    centered_by_id: dict[str, float] = {}
    baseline: dict[str, float] = {}
    features: dict[str, Any] = {}
    for members in groups.values():
        fallback = execution_policy_features_many([member.graph for member in members])
        group_features = [
            features_by_rollout.get(member.trajectory.rollout_id, vector)
            for member, vector in zip(members, fallback, strict=True)
        ]
        member_rewards = [
            director_outcome_reward(member.trajectory, params.cost_lambda) for member in members
        ]
        group_mean = statistics.fmean(member_rewards)
        for member, vector in zip(members, group_features, strict=True):
            features[member.trajectory.rollout_id] = vector
            baseline[member.trajectory.rollout_id] = group_mean
        if len(members) < 2:
            continue
        centered = [reward - group_mean for reward in member_rewards]
        for member, raw in zip(members, centered, strict=True):
            centered_by_id[member.trajectory.rollout_id] = raw

    # Low-signal pre-selection.  A group whose members share one outcome has outcome
    # advantage 0 for every member and only costs forward/backward tokens.  They are removed BEFORE the
    # whitening so the batch that is trained keeps the invariant of masked whitening (token-weighted
    # mean 0, unit variance): whitening first and filtering afterwards left the kept samples (mostly
    # the long, failed ones) with a negative mean advantage, i.e. a push-down of every sampled token.

    def _stays(rid: str) -> bool:
        if params.low_signal_abs_advantage <= 0.0 or abs(centered_by_id[rid]) >= params.low_signal_abs_advantage:
            return True
        return _keep_low_signal(rid, params.low_signal_keep_fraction)

    whiten_ids = [rid for rid in centered_by_id if _stays(rid)] or list(centered_by_id)
    mean, variance, tokens = masked_whiten_stats(
        [centered_by_id[rid] for rid in whiten_ids],
        [_action_token_count(by_id[rid].trajectory) for rid in whiten_ids],
    )
    scale = 1.0 / math.sqrt(variance + params.whiten_eps)
    whitened = {rid: (centered_by_id[rid] - mean) * scale for rid in whiten_ids}

    samples: list[TrainingSample] = []
    low_signal_dropped: list[str] = []
    for rollout in eligible:
        trajectory = rollout.trajectory
        rid = trajectory.rollout_id
        r2orch = trajectory.metadata["r2orch"]
        condition = str(r2orch["scaffold_condition"])
        group_size = len(groups[(trajectory.task_id, condition)])
        if group_size == 1:
            continue
        if rid not in whitened:
            low_signal_dropped.append(rid)
            continue
        advantage = whitened[rid]
        graph = rollout.graph
        vector = features[rid]
        samples.append(
            TrainingSample(
                rollout_id=rid,
                task_id=trajectory.task_id,
                token_ids=trajectory.token_ids,
                action_mask=trajectory.action_mask,
                reward=float(trajectory.reward),
                advantage=advantage,
                canonical_graph_key=canonical_graph_key(
                    graph, include_prompts=bool(graph.runtime_routes)
                ),
                graph_features=vector.values,
                policy_calls=tuple(trajectory.policy_calls),
                metadata={
                    **trajectory.metadata,
                    "group_reward_mean": baseline[rid],
                    "base_group_advantage": centered_by_id[rid],
                    "final_group_advantage": advantage,
                    "seed": trajectory.seed,
                    "executor_version": trajectory.executor_version,
                    "director_reward_version": trajectory.metadata.get("director_reward_version"),
                    "graph_feature_schema_id": vector.schema_id,
                    "training_selection_schema": SELECTION_SCHEMA,
                    "r2orch": {
                        **r2orch,
                        "condition_group_key": f"{trajectory.task_id}|{condition}",
                        "condition_group_size": group_size,
                        "condition_mean": baseline[rid],
                        "outcome_advantage": advantage,
                        **(
                            {
                                "cost": {
                                    "lambda": float(params.cost_lambda),
                                    "worker_cost": worker_cost_of(trajectory),
                                    "reward": float(trajectory.reward),
                                    "adjusted_reward": director_outcome_reward(
                                        trajectory, params.cost_lambda
                                    ),
                                }
                            }
                            if params.cost_lambda
                            else {}
                        ),
                        "kl_coefficient": params.kl,
                    },
                },
            )
        )
    return TrainingBatch(
        role="solver",
        samples=tuple(samples),
        objective=DIRECTOR_OBJECTIVE,
        metadata={
            "training_selection_schema": SELECTION_SCHEMA,
            "r2orch_whiten": {"mean": mean, "std": math.sqrt(variance), "tokens": tokens},
            **(
                {
                    "r2orch_low_signal": {
                        "abs_advantage": params.low_signal_abs_advantage,
                        "keep_fraction": params.low_signal_keep_fraction,
                        "dropped": len(low_signal_dropped),
                        "kept": len(samples),
                    }
                }
                if params.low_signal_abs_advantage > 0.0
                else {}
            ),
            **(
                {
                    "r2orch_cost": {
                        "lambda": float(params.cost_lambda),
                        # Rollouts with no recorded cost keep their raw reward (a visible count,
                        # not a silent skip of the penalty).
                        "rollouts_without_cost": sorted(
                            rollout.trajectory.rollout_id
                            for rollout in eligible
                            if worker_cost_of(rollout.trajectory) is None
                        ),
                    }
                }
                if params.cost_lambda
                else {}
            ),
        },
    )
