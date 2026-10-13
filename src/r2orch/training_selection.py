"""Deterministic, sample-local admission for partial sibling groups."""

from __future__ import annotations

import math
from typing import Any


def trajectory_exclusions(trajectory: Any) -> list[str]:
    metadata = trajectory.metadata
    reasons = list(metadata.get("training_exclusion_reasons", ()))
    if "training_eligible" in metadata and not metadata["training_eligible"]:
        reasons.append("training_ineligible")
    if metadata.get("terminal_graph_status") == "unsafe_partial":
        reasons.append("unsafe_partial")
    if metadata.get("reward_known") is False or not math.isfinite(trajectory.reward):
        reasons.append("untrusted_reward")
    calls = trajectory.policy_calls
    if calls:
        # Offline mock Director text carries no provider log probabilities (the same
        # exemption as selfplay_runtime's call admission); real calls always need them.
        sampled = [
            call
            for call in calls
            if call.metadata.get("token_provenance") != "mock_text"
        ]
        if any(not call.behavior_log_probs for call in sampled):
            reasons.append("missing_behavior_log_probs")
        if any(not all(math.isfinite(x) for x in call.behavior_log_probs) for call in sampled):
            reasons.append("nonfinite_behavior_log_probs")
        if not any(len(c.token_ids) >= 2 and any(c.action_mask[1:]) for c in calls):
            reasons.append("missing_action_targets")
    elif len(trajectory.token_ids) < 2 or not any(trajectory.action_mask[1:]):
        reasons.append("missing_action_targets")
    return sorted(set(reasons))
