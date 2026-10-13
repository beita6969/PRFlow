"""Frontier evidence admission.

No model calls live here. Solver rewards are consumed unchanged.
"""

from __future__ import annotations

import math


def frontier_evidence_exclusions(rollout) -> list[str]:
    """Policy token/probability failures do not invalidate independently known outcomes."""
    if rollout is None:
        return ["missing_trajectory"]
    trajectory = rollout.trajectory
    metadata = trajectory.metadata
    reasons = []
    reward = float(trajectory.reward)
    if metadata.get("reward_known") is not True or not math.isfinite(reward):
        reasons.append("untrusted_reward")
    if (
        metadata.get("uncertain_attribution_zero")
        or metadata.get("reward_admission_reason") == "uncertain_attribution_zero"
    ):
        reasons.append("statistical_zero_without_trusted_outcome")
    if any(
        metadata.get(key)
        for key in (
            "worker_backend_failure",
            "infrastructure_failure",
        )
    ):
        reasons.append("infrastructure_or_invalid_evaluation")
    if not 0.0 <= reward <= 1.0:
        reasons.append("invalid_task_reward")
    if metadata.get("task_reward", reward) != reward:
        reasons.append("task_reward_mismatch")
    if metadata.get("terminal_graph_status") == "unsafe_partial":
        reasons.append("unsafe_graph_record")
    # Graph.from_dict rebuilds the mutation counter. Compare actual contents,
    # not that counter, while retaining node prompts, routes and all relations.
    stored_graph = dict(trajectory.graph)
    actual_graph = rollout.graph.to_dict()
    stored_graph.pop("version", None)
    actual_graph.pop("version", None)
    if (
        stored_graph != actual_graph
        or rollout.graph.validate()
        or metadata.get("frontier_evidence_eligible") is False
    ):
        reasons.append("invalid_frontier_evidence")
    return sorted(set(reasons))
