"""Worker cost of one rollout: Worker tokens normalised by a per-dataset reference.

The Canvas keeps every Worker execution as an ``ExecutionReport`` whose ``execution_events`` carry
the token usage of each attempt (``token_in`` / ``token_out`` are the sum of all requests the node
made: main call, tool rounds, truncation retries, finalization).  This module adds them up once at
the end of a rollout:

    worker_cost = (sum token_in * c_in + sum token_out * c_out) / reference(dataset)

with the weights and the per-dataset reference of ``CostConfig`` (``run_config``).  Only
Worker usage counts; the Director's own tokens do not.

The record is written for every rollout, whatever ``lambda`` is.  Only the Director's batch builder
reads it to adjust the reward (``director_update``), and only when ``lambda > 0``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

WORKER_COST_SCHEMA = "worker_cost"


def worker_usage_from_history(history: Iterable[Any]) -> dict[str, int]:
    """Add up the executed (non cache-hit) Worker attempts of a Canvas history.

    A report can be attached to several steps (a final step re-exposes the last report), so an
    attempt is counted once per report object and once per artifact id.  Returns
    ``{"executions", "token_in", "token_out"}``.
    """

    totals = {"executions": 0, "token_in": 0, "token_out": 0}
    seen_reports: set[int] = set()
    seen_artifacts: set[str] = set()
    for step in history:
        report = step.execution
        if report is None or id(report) in seen_reports:
            continue
        seen_reports.add(id(report))
        for event in report.execution_events:
            if event.get("cache_hit"):
                continue
            artifact_id = str(event.get("artifact_id") or "")
            if artifact_id:
                if artifact_id in seen_artifacts:
                    continue
                seen_artifacts.add(artifact_id)
            totals["executions"] += 1
            totals["token_in"] += int(event.get("token_in", 0) or 0)
            totals["token_out"] += int(event.get("token_out", 0) or 0)
    return totals


def worker_cost_record(
    usage: dict[str, int],
    *,
    dataset: str,
    cost: Any,
    canvas_total_tokens: int | None = None,
) -> dict[str, Any]:
    """The rollout's ``worker_cost_record`` (``cost`` is a ``CostConfig``)."""

    reference_in, reference_out = cost.reference_for(dataset)
    return {
        "schema": WORKER_COST_SCHEMA,
        "dataset": str(dataset),
        "worker_cost": cost.cost(usage["token_in"], usage["token_out"], dataset),
        "reference_cost": cost.reference_cost(dataset),
        "reference_tokens": [reference_in, reference_out],
        "c_in": float(cost.c_in),
        "c_out": float(cost.c_out),
        "lambda": float(cost.lambda_),
        "token_in": int(usage["token_in"]),
        "token_out": int(usage["token_out"]),
        "executions": int(usage["executions"]),
        # The Canvas's own running total (budget accounting); a mismatch with token_in + token_out
        # means an execution report was not reachable from the history.
        "canvas_total_tokens": None if canvas_total_tokens is None else int(canvas_total_tokens),
    }
