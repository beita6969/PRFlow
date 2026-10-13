from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .features import structural_features
from .graph import MultiAgentGraph
from .rollouts import TrainingBatch
from .selfplay import FrontierScore
from .training import PolicyUpdateResult

CORE_COLUMNS = (
    "cycle",
    "timestamp_utc",
    "proposer_step",
    "solver_step",
    "proposer_reward",
    "solver_reward",
    "answer_correctness",
    "proposer_loss",
    "solver_loss",
    "proposer_kl",
    "solver_kl",
    "proposer_entropy",
    "solver_entropy",
    "mean_interactive_turns",
    "mean_trajectory_tokens",
    "mean_action_tokens",
    "mean_agent_count",
    "mean_directed_edges",
    "mean_bidirectional_edges",
    "mean_graph_depth",
    "mean_graph_local_frontier",
    "unique_graph_ratio",
    "within_task_unique_graph_ratio",
    "single_agent_rate",
    "relation_graph_rate",
    "disconnected_multi_agent_rate",
    "mean_graph_novelty",
    "mean_graph_diversity_bonus",
    "answer_reward_release_rate",
    "mean_protocol_score",
    "protocol_qualified_rate",
    "mean_rollout_duration_s",
    "mean_react_action_calls",
    "action_success_rate",
    "runtime_error_count",
    "total_token_in",
    "total_token_out",
    "skillbank_size",
    "problem_extraction_success_rate",
    "active_skill_count",
)

STEP_COLUMNS = (
    "training_step",
    "cycle",
    "role",
    "role_optimizer_step",
    "alternating_update_step",
    "epoch",
    "microbatch_count",
    "reward_mean",
    "reward_std",
    "loss",
    "policy_loss",
    "kl",
    "entropy",
    "clip_fraction",
    "grad_norm",
    "learning_rate",
    "masked_tokens",
    "duration_seconds",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "timestamp_utc",
    "problem_extraction_success_rate",
    "active_skill_count",
)


class TrainingMetricsStore:
    """Append-only detailed telemetry plus a plotting-friendly core CSV."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.jsonl_path = self.root / "training_metrics.jsonl"
        self.latest_path = self.root / "training_metrics_latest.json"
        self.csv_path = self.root / "training_curves.csv"
        self.steps_path = self.root / "training_steps.jsonl"
        self.steps_latest_path = self.root / "training_steps_latest.json"
        self.steps_csv_path = self.root / "training_step_curves.csv"

    def append(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        previous = self._latest()
        _add_skillbank_delta(record, previous)
        self._append_steps(record)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _atomic_json(self.latest_path, record)
        row = _core_row(record)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CORE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _latest(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        return json.loads(self.latest_path.read_text(encoding="utf-8"))

    def _append_steps(self, record: dict[str, Any]) -> None:
        last_step = 0
        if self.steps_latest_path.exists():
            last_step = int(
                json.loads(self.steps_latest_path.read_text(encoding="utf-8")).get(
                    "training_step", 0
                )
            )
        rows: list[dict[str, Any]] = []
        for role in ("proposer", "solver"):
            for raw in record.get("policies", {}).get(role, {}).get("step_metrics", []):
                last_step += 1
                rows.append(
                    {
                        **raw,
                        "training_step": last_step,
                        "cycle": record.get("cycle"),
                        "role": role,
                        "problem_extraction_success_rate": record.get("rollout", {})
                        .get("proposal_extraction", {})
                        .get("success_rate"),
                        "active_skill_count": record.get("skillbank", {}).get("active_skill_count"),
                    }
                )
        if not rows:
            return
        with self.steps_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_header = not self.steps_csv_path.exists()
        with self.steps_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STEP_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        _atomic_json(self.steps_latest_path, rows[-1])


def collect_cycle_metrics(
    *,
    cycle: int,
    proposer_batch: TrainingBatch,
    solver_batch: TrainingBatch,
    updates: Iterable[PolicyUpdateResult],
    frontier_scores: Iterable[FrontierScore] = (),
    skillbank_path: str | Path | None = None,
    cycle_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
    proposal_extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updates = tuple(updates)
    update_map = {update.role: asdict(update) for update in updates}
    frontiers = list(frontier_scores)
    active_skill_ids = sorted(
        {
            str(skill_id)
            for sample in solver_batch.samples
            for skill_id in sample.metadata.get("skills_used", [])
        }
    )
    experiment_context = dict(context or {})
    solver_summary = summarize_training_batch(solver_batch)
    record = {
        "schema_version": 1,
        "cycle": int(cycle),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "experiment": experiment_context,
        "policies": update_map,
        "batches": {
            "proposer": summarize_training_batch(proposer_batch),
            "solver": solver_summary,
        },
        "rollout": {
            "task_count": len({sample.task_id for sample in solver_batch.samples}),
            "answer_correctness": _answer_correctness(solver_batch),
            "interactive_turns": _describe(
                [_interactive_turns(sample.metadata) for sample in solver_batch.samples]
            ),
            "accepted_turns": _metadata_distribution(solver_batch, "accepted_turns"),
            "rejected_turns": _metadata_distribution(solver_batch, "rejected_turns"),
            "finished_rate": _metadata_boolean_rate(solver_batch, "finished"),
            "proposal_extraction": dict(proposal_extraction or {}),
            "frontier": {
                "graph_local": _describe([item.graph_local for item in frontiers]),
                "validity": _describe([item.validity for item in frontiers]),
            },
        },
        "graph": _graph_summary(solver_batch),
        "topology_policy": _topology_policy_summary(solver_batch),
        "answer_reward_release_rate": _metadata_boolean_rate(solver_batch, "answer_reward_released"),
        "protocol_reward": _protocol_reward_summary(solver_batch),
        "runtime": _runtime_summary(solver_batch),
        "skillbank": _skillbank_snapshot(
            skillbank_path,
            cycle_dir=cycle_dir,
            active_skill_ids=active_skill_ids,
        ),
    }
    if cycle_dir is not None and (Path(cycle_dir) / "proposer_lists.jsonl").exists():
        record["r2orch"] = _r2orch_cycle_record(
            Path(cycle_dir), updates, experiment_context
        )
    return record


def _r2orch_cycle_record(
    cycle_dir: Path, updates: tuple[PolicyUpdateResult, ...], context: dict[str, Any]
) -> dict[str, Any]:
    """Per-cycle metrics, their json/jsonl copies and the dashboard."""
    from .cycle_metrics import (
        r2orch_cycle_metrics,
        serving_gpu_counts,
        write_cycle_metrics,
        write_dashboard,
    )

    timing = {
        "collection_wall_s": float(context.get("collection_elapsed_s") or 0.0),
        "training_wall_s": float(context.get("training_compute_elapsed_s") or 0.0),
        "cycle_wall_s": float(context.get("cycle_elapsed_s") or 0.0),
        **serving_gpu_counts(context.get("config")),
    }
    metrics = r2orch_cycle_metrics(cycle_dir, updates=updates, timing=timing)
    write_cycle_metrics(cycle_dir, metrics)
    write_dashboard(cycle_dir.parent)
    return metrics


def summarize_training_batch(batch: TrainingBatch) -> dict[str, Any]:
    samples = list(batch.samples)
    token_lengths = [len(sample.token_ids) for sample in samples]
    action_lengths = [sum(sample.action_mask) for sample in samples]
    graph_keys = [sample.canonical_graph_key for sample in samples if sample.canonical_graph_key]
    grouped_graph_keys: dict[str, list[str]] = {}
    for sample in samples:
        grouped_graph_keys.setdefault(sample.task_id, []).append(
            sample.canonical_graph_key or "<missing>"
        )
    within_task_ratios = [
        len(set(keys)) / len(keys) for keys in grouped_graph_keys.values() if keys
    ]
    return {
        "role": batch.role,
        "objective": batch.objective,
        "sample_count": len(samples),
        "task_count": len({sample.task_id for sample in samples}),
        "reward": _describe([sample.reward for sample in samples]),
        "advantage": _describe([sample.advantage for sample in samples]),
        "trajectory_tokens": _describe(token_lengths),
        "action_tokens": _describe(action_lengths),
        "total_tokens": sum(token_lengths),
        "total_action_tokens": sum(action_lengths),
        "action_token_ratio": sum(action_lengths) / max(1, sum(token_lengths)),
        "unique_graphs": len(set(graph_keys)),
        "unique_graph_ratio": len(set(graph_keys)) / max(1, len(graph_keys)),
        "within_task_unique_graph_ratio": (
            statistics.fmean(within_task_ratios) if within_task_ratios else None
        ),
        "within_task_unique_graph_ratio_distribution": _describe(within_task_ratios),
    }


def _graph_summary(batch: TrainingBatch) -> dict[str, Any]:
    rows = [sample.graph_features for sample in batch.samples if sample.graph_features]
    if not rows:
        return {"sample_count": 0, "features": {}}
    dimensions = [len(row) for row in rows]
    schemas = [
        str(sample.metadata.get("graph_feature_schema_id", "structure_only"))
        for sample in batch.samples
        if sample.graph_features
    ]
    feature_names = structural_features(MultiAgentGraph()).names
    if set(dimensions) != {len(feature_names)}:
        # Semantic graph vectors can contain thousands of coordinates.  They do
        # not share the 15 names of the structure-only ablation, and emitting a
        # metric per embedding coordinate would both mislabel the data and make
        # the telemetry needlessly huge.  Record their schema, dimension, and
        # norm instead; topology statistics are recovered from the canonical
        # graph key below and therefore remain interpretable.
        return {
            "sample_count": len(rows),
            "schema_ids": dict(sorted(Counter(schemas).items())),
            "dimensions": _describe(dimensions),
            "vector_l2_norm": _describe(
                [sum(value * value for value in row) ** 0.5 for row in rows]
            ),
            "features": {},
        }
    columns = zip(*rows, strict=True)
    return {
        "sample_count": len(rows),
        "schema_ids": dict(sorted(Counter(schemas).items())),
        "dimensions": _describe(dimensions),
        "features": {
            name: _describe(list(values))
            for name, values in zip(feature_names, columns, strict=True)
        },
    }


def _topology_policy_summary(batch: TrainingBatch) -> dict[str, Any]:
    graph_shapes = []
    for sample in batch.samples:
        shape = _canonical_graph_shape(sample.canonical_graph_key)
        if shape is None and len(sample.graph_features) == len(
            structural_features(MultiAgentGraph()).names
        ):
            row = sample.graph_features
            shape = (row[0], row[4], row[5], row[14])
        if shape is not None:
            graph_shapes.append(shape)
    agent_counts = [shape[0] for shape in graph_shapes]
    directed_counts = [shape[1] for shape in graph_shapes]
    bidirectional_counts = [shape[2] for shape in graph_shapes]
    graph_depths = [shape[3] for shape in graph_shapes]
    relation_counts = [shape[1] + shape[2] for shape in graph_shapes]
    disconnected_multi_agent_rate = (
        statistics.fmean(
            float(agent_count > 1.0 and relation_count == 0.0)
            for agent_count, relation_count in zip(agent_counts, relation_counts, strict=True)
        )
        if agent_counts
        else None
    )
    single_agent_rate = (
        statistics.fmean(float(value == 1.0) for value in agent_counts) if agent_counts else None
    )
    disconnected_collapse_alert = bool(
        disconnected_multi_agent_rate is not None and disconnected_multi_agent_rate > 0.0
    )
    return {
        "agent_count": _describe(agent_counts),
        "directed_edge_count": _describe(directed_counts),
        "bidirectional_edge_count": _describe(bidirectional_counts),
        "graph_depth": _describe(graph_depths),
        "single_agent_rate": single_agent_rate,
        "relation_graph_rate": (
            statistics.fmean(float(value > 0.0) for value in relation_counts)
            if relation_counts
            else None
        ),
        "disconnected_multi_agent_rate": disconnected_multi_agent_rate,
        "graph_novelty": _metadata_distribution(batch, "graph_novelty"),
        "graph_diversity_bonus": _metadata_distribution(batch, "graph_diversity_bonus"),
        "final_execution_round": _metadata_distribution(batch, "final_execution_round"),
        "final_execution_count": _metadata_distribution(batch, "final_execution_count"),
        "incremental_execution_count": _metadata_distribution(batch, "incremental_execution_count"),
        "worker_executed_agent_count": _metadata_distribution(batch, "worker_executed_agent_count"),
        "worker_reused_agent_count": _metadata_distribution(batch, "worker_reused_agent_count"),
        "worker_scheduled_agent_count": _metadata_distribution(
            batch, "worker_scheduled_agent_count"
        ),
        "worker_model_call_count": _metadata_distribution(batch, "worker_model_call_count"),
        "worker_initial_model_call_count": _metadata_distribution(
            batch, "worker_initial_model_call_count"
        ),
        "worker_revision_model_call_count": _metadata_distribution(
            batch, "worker_revision_model_call_count"
        ),
        "worker_cache_hit_count": _metadata_distribution(batch, "worker_cache_hit_count"),
        "worker_component_execution_count": _metadata_distribution(
            batch, "worker_component_execution_count"
        ),
        "worker_bidirectional_revision_gate_count": _metadata_distribution(
            batch, "worker_bidirectional_revision_gate_count"
        ),
        "worker_bidirectional_revision_required_count": _metadata_distribution(
            batch, "worker_bidirectional_revision_required_count"
        ),
        "worker_bidirectional_revision_skipped_agent_count": _metadata_distribution(
            batch, "worker_bidirectional_revision_skipped_agent_count"
        ),
        "worker_bidirectional_revision_wave_count": _metadata_distribution(
            batch, "worker_bidirectional_revision_wave_count"
        ),
        "worker_attempt_cache_hit_rate": _metadata_distribution(
            batch, "worker_attempt_cache_hit_rate"
        ),
        "prompt_revision_count": _metadata_distribution(batch, "prompt_revision_count"),
        "prompt_revision_target_model_call_count": _metadata_distribution(
            batch, "prompt_revision_target_model_call_count"
        ),
        "prompt_revision_worker_model_call_count": _metadata_distribution(
            batch, "prompt_revision_worker_model_call_count"
        ),
        "prompt_revision_evidence_rejection_count": _metadata_distribution(
            batch, "prompt_revision_evidence_rejection_count"
        ),
        "worker_cache_reuse_rate": _metadata_distribution(batch, "worker_cache_reuse_rate"),
        "feedback_truncation_count": _metadata_distribution(batch, "feedback_truncation_count"),
        "structural_repair_entry_count": _metadata_distribution(
            batch, "structural_repair_entry_count"
        ),
        "structural_repair_resolution_count": _metadata_distribution(
            batch, "structural_repair_resolution_count"
        ),
        "structural_repair_blocked_action_count": _metadata_distribution(
            batch, "structural_repair_blocked_action_count"
        ),
        "semantic_no_progress_recovery_count": _metadata_distribution(
            batch, "semantic_no_progress_recovery_count"
        ),
        "output_switch_without_progress_count": _metadata_distribution(
            batch, "output_switch_without_progress_count"
        ),
        "output_lifecycle_recovery_count": _metadata_distribution(
            batch, "output_lifecycle_recovery_count"
        ),
        "disconnected_multi_agent_step_count": _metadata_distribution(
            batch, "disconnected_multi_agent_step_count"
        ),
        "collapse_alert": disconnected_collapse_alert,
        "collapse_alert_reasons": (
            ["disconnected_multi_agent"] if disconnected_collapse_alert else []
        ),
        "collapse_thresholds": {
            "disconnected_multi_agent_rate": 0.0,
        },
    }


def _canonical_graph_shape(key: str) -> tuple[float, float, float, float] | None:
    """Return agent/edge counts and layer depth from a canonical graph key.

    The canonical key is the stable topology record shared by both structural
    and semantic feature schemas.  Metrics must not interpret normalized E5
    coordinates as raw graph counts.
    """

    try:
        nodes, directed, bidirectional = json.loads(key)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(isinstance(value, list) for value in (nodes, directed, bidirectional)):
        return None
    layers = {item[0] for item in nodes if isinstance(item, list) and item}
    return (
        float(len(nodes)),
        float(len(directed)),
        float(len(bidirectional)),
        float(len(layers)),
    )


def _protocol_reward_summary(batch: TrainingBatch) -> dict[str, Any]:
    rows = [
        sample.metadata.get("protocol_reward")
        for sample in batch.samples
        if isinstance(sample.metadata.get("protocol_reward"), dict)
    ]
    versions: dict[str, int] = {}
    for row in rows:
        version = str(row.get("version", "unknown"))
        versions[version] = versions.get(version, 0) + 1
    return {
        "sample_count": len(rows),
        "versions": versions,
        "protocol_score": _describe(float(row.get("protocol_score", 0.0)) for row in rows),
        "delegation_complete_rate": _mapping_boolean_rate(rows, "delegation_complete"),
        "delegation_fidelity_rate": _mapping_boolean_rate(rows, "delegation_fidelity"),
        "graph_complete_rate": _mapping_boolean_rate(rows, "graph_complete"),
        "finish_complete_rate": _mapping_boolean_rate(rows, "finish_complete"),
        "execution_complete_rate": _mapping_boolean_rate(rows, "execution_complete"),
        "qualified_rate": _mapping_boolean_rate(rows, "qualified"),
        "answer_reward_release_rate": _mapping_boolean_rate(rows, "answer_reward_released"),
    }


def _answer_correctness(batch: TrainingBatch) -> float | None:
    passed: list[float] = []
    for sample in batch.samples:
        metadata = sample.metadata
        if metadata.get("reward_known") is False:
            continue
        verification = sample.metadata.get("verification")
        if isinstance(verification, dict) and verification.get("passed") is not None:
            passed.append(float(bool(verification["passed"])))
    if passed:
        return statistics.fmean(passed)
    return None  # Continuous reward is not a binary accuracy fallback.


def _runtime_summary(batch: TrainingBatch) -> dict[str, Any]:
    """Aggregate Worker/ReAct telemetry retained in each Solver rollout trace."""

    route_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    action_status_counts: dict[str, int] = {}
    action_error_codes: dict[str, int] = {}
    react_calls_per_artifact: list[int] = []
    duration_values: list[float] = []
    token_in_values: list[int] = []
    token_out_values: list[int] = []
    execution_error_count = 0
    backend_failure_count = 0
    backend_failure_types: dict[str, int] = {}
    protocol_failure_count = 0
    responsibility_violations = 0
    duplicate_responsibility_rejections = 0
    duplicate_responsibility_detections = 0
    duplicate_responsibility_decisions: dict[str, int] = {}
    duplicate_responsibility_policies: dict[str, int] = {}
    read_overlap_audited_rollouts = 0
    duplicate_read_only_rollouts = 0
    duplicate_read_only_pairs = 0
    shared_exact_read_count = 0
    protocol_recoveries = 0
    director_action_repairs = 0
    director_action_repair_successes = 0
    director_discarded_output_chars = 0
    interactive_turn_count = 0
    recovered_rollout_count = 0
    artifact_count = 0
    seen_artifacts: set[tuple[str, str]] = set()

    for sample in batch.samples:
        metadata = sample.metadata
        duration = metadata.get("duration_s")
        if duration is not None:
            duration_values.append(float(duration))
        token_in = int(metadata.get("token_in", 0) or 0)
        token_out = int(metadata.get("token_out", 0) or 0)
        token_in_values.append(token_in)
        token_out_values.append(token_out)
        responsibility_violations += int(metadata.get("responsibility_violation_count", 0) or 0)
        duplicate_responsibility_rejections += int(
            metadata.get("duplicate_responsibility_rejection_count", 0) or 0
        )
        duplicate_responsibility_detections += int(
            metadata.get("duplicate_responsibility_detection_count", 0) or 0
        )
        decision_counts = metadata.get("duplicate_responsibility_decision_counts")
        if isinstance(decision_counts, dict):
            for decision, count in decision_counts.items():
                duplicate_responsibility_decisions[str(decision)] = (
                    duplicate_responsibility_decisions.get(str(decision), 0) + int(count or 0)
                )
        duplicate_policy = str(metadata.get("duplicate_responsibility_policy", "")).strip()
        if duplicate_policy:
            duplicate_responsibility_policies[duplicate_policy] = (
                duplicate_responsibility_policies.get(duplicate_policy, 0) + 1
            )
        read_overlap_audit = metadata.get("cross_agent_exact_read_overlap")
        if isinstance(read_overlap_audit, dict):
            read_overlap_audited_rollouts += 1
            duplicate_read_only_rollouts += int(
                bool(read_overlap_audit.get("duplicate_read_only_exploration"))
            )
            duplicate_read_only_pairs += int(read_overlap_audit.get("flagged_pair_count", 0) or 0)
            shared_exact_read_count += sum(
                int(pair.get("shared_exact_read_count", 0) or 0)
                for pair in read_overlap_audit.get("pairs", ())
                if isinstance(pair, dict)
            )
        director_action_repairs += int(metadata.get("director_action_repairs", 0) or 0)
        director_action_repair_successes += int(
            metadata.get("director_action_repair_successes", 0) or 0
        )
        director_discarded_output_chars += int(
            metadata.get("director_discarded_output_chars", 0) or 0
        )
        recovery_count = int(metadata.get("protocol_recovery_count", 0) or 0)
        protocol_recoveries += recovery_count
        recovered_rollout_count += int(recovery_count > 0)
        interactive_turn_count += int(metadata.get("interactive_turns", 0) or 0)
        trace = metadata.get("solver_trace")
        if not isinstance(trace, dict):
            continue
        for event in trace.get("events", []):
            if not isinstance(event, dict):
                continue
            payload = event.get("payload", {})
            execution = payload.get("execution") if isinstance(payload, dict) else None
            if not isinstance(execution, dict):
                continue
            errors = execution.get("errors", {})
            if isinstance(errors, dict):
                execution_error_count += len(errors)
            artifacts = execution.get("artifacts", {})
            if not isinstance(artifacts, dict):
                continue
            for artifact_id, artifact in artifacts.items():
                if not isinstance(artifact, dict):
                    continue
                identity = (sample.rollout_id, str(artifact_id))
                if identity in seen_artifacts:
                    continue
                seen_artifacts.add(identity)
                artifact_count += 1
                route = str(artifact.get("model_route", "")).strip()
                model = str(artifact.get("model", "")).strip()
                if route:
                    route_counts[route] = route_counts.get(route, 0) + 1
                if model:
                    model_counts[model] = model_counts.get(model, 0) + 1
                answer = str(artifact.get("answer", "")).strip()
                if answer == "WORKER_BACKEND_FAILURE":
                    backend_failure_count += 1
                    for issue in artifact.get("unresolved_issues", []):
                        issue_text = str(issue)
                        if issue_text.startswith("transient_backend_error:"):
                            failure_type = issue_text.partition(":")[2] or "unknown"
                            backend_failure_types[failure_type] = (
                                backend_failure_types.get(failure_type, 0) + 1
                            )
                elif answer == "WORKER_PROTOCOL_FAILURE":
                    protocol_failure_count += 1
                react_trace = artifact.get("react_trace", [])
                if not isinstance(react_trace, list):
                    continue
                react_calls_per_artifact.append(len(react_trace))
                for turn in react_trace:
                    if not isinstance(turn, dict):
                        continue
                    action = turn.get("action", {})
                    observation = turn.get("observation", {})
                    name = str(action.get("name", "")).strip() if isinstance(action, dict) else ""
                    if name:
                        action_counts[name] = action_counts.get(name, 0) + 1
                    status = (
                        str(observation.get("status", "unknown")).strip() or "unknown"
                        if isinstance(observation, dict)
                        else "unknown"
                    )
                    action_status_counts[status] = action_status_counts.get(status, 0) + 1
                    error = observation.get("error") if isinstance(observation, dict) else None
                    if isinstance(error, dict):
                        code = str(error.get("code", "unknown")).strip() or "unknown"
                        action_error_codes[code] = action_error_codes.get(code, 0) + 1

    total_actions = sum(action_status_counts.values())
    successful_actions = action_status_counts.get("ok", 0)
    return {
        "rollout_count": len(batch.samples),
        "artifact_count": artifact_count,
        "rollout_duration_s": _describe(duration_values),
        "token_in": {**_describe(token_in_values), "total": sum(token_in_values)},
        "token_out": {**_describe(token_out_values), "total": sum(token_out_values)},
        "react_calls_per_artifact": _describe(react_calls_per_artifact),
        "action_calls": total_actions,
        "successful_action_calls": successful_actions,
        "action_success_rate": successful_actions / total_actions if total_actions else None,
        "action_counts": dict(sorted(action_counts.items())),
        "action_status_counts": dict(sorted(action_status_counts.items())),
        "action_error_codes": dict(sorted(action_error_codes.items())),
        "model_route_counts": dict(sorted(route_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "execution_error_count": execution_error_count,
        "backend_failure_count": backend_failure_count,
        "backend_failure_types": dict(sorted(backend_failure_types.items())),
        "protocol_failure_count": protocol_failure_count,
        "responsibility_protocol": {
            "violation_count": responsibility_violations,
            "violation_rate_per_director_turn": (
                responsibility_violations / interactive_turn_count
                if interactive_turn_count
                else None
            ),
            "protocol_recovery_steps": protocol_recoveries,
            "recovered_rollout_count": recovered_rollout_count,
            "recovery_rate_per_rollout": (
                recovered_rollout_count / len(batch.samples) if batch.samples else None
            ),
        },
        "responsibility_overlap_audit": {
            "duplicate_responsibility_detection_count": (duplicate_responsibility_detections),
            "duplicate_responsibility_rejection_count": (duplicate_responsibility_rejections),
            "duplicate_responsibility_decision_counts": dict(
                sorted(duplicate_responsibility_decisions.items())
            ),
            "duplicate_responsibility_policy_rollout_counts": dict(
                sorted(duplicate_responsibility_policies.items())
            ),
            "read_overlap_audited_rollout_count": read_overlap_audited_rollouts,
            "duplicate_read_only_rollout_count": duplicate_read_only_rollouts,
            "duplicate_read_only_rollout_rate": (
                duplicate_read_only_rollouts / read_overlap_audited_rollouts
                if read_overlap_audited_rollouts
                else None
            ),
            "duplicate_read_only_pair_count": duplicate_read_only_pairs,
            "shared_exact_read_count": shared_exact_read_count,
        },
        "director_action_protocol": {
            "repair_attempts": director_action_repairs,
            "repair_successes": director_action_repair_successes,
            "repair_success_rate": (
                director_action_repair_successes / director_action_repairs
                if director_action_repairs
                else None
            ),
            "discarded_output_chars": director_discarded_output_chars,
            "discarded_output_chars_per_turn": (
                director_discarded_output_chars / interactive_turn_count
                if interactive_turn_count
                else None
            ),
        },
    }


def _interactive_turns(metadata: dict[str, Any]) -> float:
    if metadata.get("interactive_turns") is not None:
        return float(metadata["interactive_turns"])
    spans = metadata.get("action_token_spans")
    if isinstance(spans, (list, tuple)):
        return float(len(spans))
    prefixes = metadata.get("canvas_prefixes")
    return float(len(prefixes)) if isinstance(prefixes, (list, tuple)) else 0.0


def _metadata_distribution(batch: TrainingBatch, key: str) -> dict[str, float | int | None]:
    return _describe(
        float(sample.metadata[key])
        for sample in batch.samples
        if sample.metadata.get(key) is not None
    )


def _metadata_boolean_rate(batch: TrainingBatch, key: str) -> float | None:
    values = [
        float(bool(sample.metadata[key]))
        for sample in batch.samples
        if sample.metadata.get(key) is not None
    ]
    return statistics.fmean(values) if values else None


def _mapping_boolean_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(bool(row[key])) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _skillbank_snapshot(
    path: str | Path | None,
    *,
    cycle_dir: str | Path | None = None,
    active_skill_ids: list[str],
) -> dict[str, Any]:
    source = Path(path) if path else None
    directory = Path(cycle_dir) if cycle_dir is not None else None
    frozen_path = directory / "director_skill_snapshot.json" if directory else None
    if frozen_path is not None and frozen_path.exists():
        # The configured live JSON file need not exist. Collection uses
        # this immutable snapshot, not the state after a subsequent review.
        source = frozen_path
    if source is None or not source.exists():
        result = {
            "available": False,
            "size": 0,
            "skill_ids": [],
            "active_skill_ids_this_cycle": active_skill_ids,
            "active_skills_this_cycle": len(active_skill_ids),
        }
        _attach_pats_metrics(result, None, directory)
        return result
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [item for item in payload.get("cards", []) if item.get("status") in {"seed", "active"}]
    skills = [item["card"] for item in records]
    stats = [item.get("stats", {}) for item in skills]
    result = {
        "available": True,
        "size": len(skills),
        "skill_ids": [str(item.get("skill_id", "")) for item in skills],
        "active_skill_ids_this_cycle": active_skill_ids,
        "active_skills_this_cycle": len(active_skill_ids),
        "total_usage": sum(int(item.get("usage_count", 0)) for item in stats),
        "total_helpful": sum(int(item.get("helpful_count", 0)) for item in stats),
        "total_hurt": sum(int(item.get("hurt_count", 0)) for item in stats),
    }
    result.update(
        schema="director_skill",
        source_path=str(source),
        snapshot_id=payload.get("snapshot_id"),
        size_semantics="global_card_records_only; scoped copies are reported separately",
        global_card_count=len(records),
        global_unique_skill_id_count=len(set(result["skill_ids"])),
        global_seed_card_count=sum(
            item.get("status") == "seed"
            or item.get("provenance") == "human_seed"
            or item["card"].get("stats", {}).get("is_seed") is True
            for item in records
        ),
    )
    _attach_pats_metrics(result, payload.get("pats"), directory)
    return result


def _attach_pats_metrics(
    result: dict[str, Any], snapshot: dict[str, Any] | None, directory: Path | None
) -> None:
    """Keep frozen scoped membership distinct from after-collection review outcomes."""
    pats: dict[str, Any] = {}
    if snapshot is not None:
        scopes = {}
        unique_ids: set[str] = set()
        for key, value in sorted(snapshot.get("scopes", {}).items()):
            cards = [
                record
                for record in value.get("cards", [])
                if record.get("status") in {"seed", "active"}
            ]
            versions = [
                {"id": str(record["card"]["skill_id"]), "version": int(record["version"])}
                for record in cards
            ]
            unique_ids.update(item["id"] for item in versions)
            scopes[key] = {
                "card_count": len(cards),
                "skill_versions": versions,
                **{
                    field: value[field]
                    for field in ("policy_snapshot",)
                    if field in value
                },
            }
        pats.update(
            snapshot_available=True,
            snapshot_id=snapshot.get("snapshot_id"),
            snapshot_step=snapshot.get("step"),
            counts_semantics="scoped instances in the frozen collection view; not a global bank total",
            scope_count=len(scopes),
            scoped_card_instance_count=sum(value["card_count"] for value in scopes.values()),
            scoped_unique_skill_id_count=len(unique_ids),
            scopes=scopes,
        )
    if directory is not None:
        contract_path = directory / "skill_context_contract.json"
        if contract_path.exists():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if contract.get("pats_enabled") is True:
                pats["context_enabled"] = bool(contract.get("context_enabled"))
        review_path = directory / "pats_review.json"
        if review_path.exists():
            receipt = json.loads(review_path.read_text(encoding="utf-8"))
            reviews = receipt.get("reviews", [])
            statuses = Counter(str(review["status"]) for review in reviews)
            # A scope has up to two reviews per cycle (REVISE, DISTILL); scope counts are distinct.
            status_scopes: dict[str, set[str]] = {}
            for review in reviews:
                status_scopes.setdefault(str(review["status"]), set()).add(str(review["scope"]))
            # Rejected edits were never committed. Do not count their proposed
            # operations as additions/deletions to the scoped archive.
            operations = Counter(
                str(operation["op"])
                for review in reviews
                if review.get("status") == "updated"
                for operation in review.get("operations", [])
            )
            pats.update(
                review_available=True,
                review_source_path=str(review_path),
                review_step=receipt.get("step"),
                review_scope_count=len({review["scope"] for review in reviews}),
                review_status_counts=dict(sorted(statuses.items())),
                review_updated_scope_count=len(status_scopes.get("updated", ())),
                review_rejected_scope_count=len(status_scopes.get("rejected", ())),
                review_unchanged_scope_count=len(status_scopes.get("unchanged", ())),
                applied_operation_counts=dict(sorted(operations.items())),
            )
            for field in ("credible_groups", "refiner_calls"):
                if field in receipt:
                    pats[field] = receipt[field]
    if pats:
        # Missing files mean unavailable counts, never evidence for zero scopes
        # or zero reviews.
        pats.setdefault("snapshot_available", False)
        pats.setdefault("review_available", False)
        result["pats"] = pats


def _describe(values: Iterable[float | int]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "p50": _percentile(numbers, 0.50),
        "p95": _percentile(numbers, 0.95),
        "max": max(numbers),
    }


def _percentile(values: Iterable[float | int], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _add_skillbank_delta(record: dict[str, Any], previous: dict[str, Any] | None) -> None:
    current = record.get("skillbank", {})
    current_ids = set(current.get("skill_ids", []))
    old_ids = set((previous or {}).get("skillbank", {}).get("skill_ids", []))
    current["created_since_previous"] = sorted(current_ids - old_ids)
    current["removed_since_previous"] = sorted(old_ids - current_ids)
    current["size_delta"] = len(current_ids) - len(old_ids)
    previously_active = set((previous or {}).get("skillbank", {}).get("active_skill_ids", []))
    active_this_cycle = set(current.get("active_skill_ids_this_cycle", []))
    active = (previously_active | active_this_cycle) & current_ids
    current["active_skill_ids"] = sorted(active)
    current["active_skill_count"] = len(active)


def _core_row(record: dict[str, Any]) -> dict[str, Any]:
    policies = record.get("policies", {})
    proposer = policies.get("proposer", {})
    solver = policies.get("solver", {})
    batches = record.get("batches", {})
    solver_batch = batches.get("solver", {})
    graph = record.get("graph", {}).get("features", {})
    frontier = record.get("rollout", {}).get("frontier", {}).get("graph_local", {})
    runtime = record.get("runtime", {})
    protocol_reward = record.get("protocol_reward", {})
    topology = record.get("topology_policy", {})
    return {
        "cycle": record.get("cycle"),
        "timestamp_utc": record.get("timestamp_utc"),
        "proposer_step": proposer.get("step"),
        "solver_step": solver.get("step"),
        "proposer_reward": proposer.get("reward_mean"),
        "solver_reward": solver.get("reward_mean"),
        "answer_correctness": record.get("rollout", {}).get("answer_correctness"),
        "proposer_loss": proposer.get("loss"),
        "solver_loss": solver.get("loss"),
        "proposer_kl": proposer.get("kl"),
        "solver_kl": solver.get("kl"),
        "proposer_entropy": proposer.get("entropy"),
        "solver_entropy": solver.get("entropy"),
        "mean_interactive_turns": record.get("rollout", {})
        .get("interactive_turns", {})
        .get("mean"),
        "mean_trajectory_tokens": solver_batch.get("trajectory_tokens", {}).get("mean"),
        "mean_action_tokens": solver_batch.get("action_tokens", {}).get("mean"),
        "mean_agent_count": (
            graph.get("agent_count", {}).get("mean")
            if graph.get("agent_count")
            else topology.get("agent_count", {}).get("mean")
        ),
        "mean_directed_edges": (
            graph.get("directed_edge_count", {}).get("mean")
            if graph.get("directed_edge_count")
            else topology.get("directed_edge_count", {}).get("mean")
        ),
        "mean_bidirectional_edges": (
            graph.get("bidirectional_edge_count", {}).get("mean")
            if graph.get("bidirectional_edge_count")
            else topology.get("bidirectional_edge_count", {}).get("mean")
        ),
        "mean_graph_depth": (
            graph.get("graph_depth", {}).get("mean")
            if graph.get("graph_depth")
            else topology.get("graph_depth", {}).get("mean")
        ),
        "mean_graph_local_frontier": frontier.get("mean"),
        "unique_graph_ratio": solver_batch.get("unique_graph_ratio"),
        "within_task_unique_graph_ratio": solver_batch.get("within_task_unique_graph_ratio"),
        "single_agent_rate": topology.get("single_agent_rate"),
        "relation_graph_rate": topology.get("relation_graph_rate"),
        "disconnected_multi_agent_rate": topology.get("disconnected_multi_agent_rate"),
        "mean_graph_novelty": topology.get("graph_novelty", {}).get("mean"),
        "mean_graph_diversity_bonus": topology.get("graph_diversity_bonus", {}).get("mean"),
        "answer_reward_release_rate": record.get("answer_reward_release_rate"),
        "mean_protocol_score": protocol_reward.get("protocol_score", {}).get("mean"),
        "protocol_qualified_rate": protocol_reward.get("qualified_rate"),
        "mean_rollout_duration_s": runtime.get("rollout_duration_s", {}).get("mean"),
        "mean_react_action_calls": runtime.get("react_calls_per_artifact", {}).get("mean"),
        "action_success_rate": runtime.get("action_success_rate"),
        "runtime_error_count": runtime.get("execution_error_count"),
        "total_token_in": runtime.get("token_in", {}).get("total"),
        "total_token_out": runtime.get("token_out", {}).get("total"),
        "skillbank_size": record.get("skillbank", {}).get("size"),
        "problem_extraction_success_rate": record.get("rollout", {})
        .get("proposal_extraction", {})
        .get("success_rate"),
        "active_skill_count": record.get("skillbank", {}).get("active_skill_count"),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
