"""Primary-slot metrics, independent of training admission and extra graph runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .config import canonical_dataset_name


def number(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def describe(values):
    values = sorted(value for item in values if (value := number(item)) is not None)

    def quantile(q):
        if not values:
            return None
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
        "min": quantile(0),
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "max": quantile(1),
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def record_provenance(root, config_path, task_pools):
    source = Path(__file__).parent
    files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("*.py"))
    }
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    record = {
        "source_sha256": source_hash,
        "source_files": files,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "task_pool_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in task_pools
        },
    }
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    write_json(Path(root) / "provenance" / f"{fingerprint}.json", record)
    return fingerprint


def _ratio(n, d):
    return n / d if d else None


def summarize_requests(events):
    unique = {}
    for event in events:
        key = event.get("event_id") or json.dumps(event, sort_keys=True)
        unique[key] = event
    roles = defaultdict(list)
    for event in unique.values():
        roles[str(event.get("request_role") or "unknown")].append(event)
    result = {}
    for role, items in roles.items():
        successes = [e for e in items if e.get("event") == "backend_request_success"]
        failures = [e for e in items if e.get("event") == "backend_request_failure"]
        tokens = {}
        for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
            values = [number(e.get("provider_usage", {}).get(field)) for e in successes]
            known = [value for value in values if value is not None]
            tokens[field] = {
                "sum_reported": sum(known) if known else None,
                "reported_calls": len(known),
                "missing_calls": len(values) - len(known),
            }
        result[role] = {
            "success_calls": len(successes),
            "failure_attempts": len(failures),
            "request_attempts": len(successes) + len(failures),
            "success_attempt_rate": _ratio(len(successes), len(successes) + len(failures)),
            "route_counts": dict(
                Counter(str(e.get("route") or "unknown") for e in successes + failures)
            ),
            "provider_model_counts": dict(
                Counter(str(e.get("provider_model") or "unknown") for e in successes)
            ),
            "reasoning_effort_counts": dict(
                Counter(str(e.get("reasoning_effort") or "unspecified") for e in successes)
            ),
            "failure_kind_counts": dict(Counter(str(e.get("kind") or "unknown") for e in failures)),
            "http_status_counts": dict(
                Counter(str(e.get("status_code") or "unknown") for e in failures)
            ),
            "queue_wait_s": describe(e.get("queue_wait_s") for e in successes + failures),
            "upstream_elapsed_s": describe(
                e.get("upstream_elapsed_s") for e in successes + failures
            ),
            "retry_wait_s": describe(e.get("retry_delay_s", e.get("backoff_s")) for e in items),
            "tokens": tokens,
        }
    return result


def _dataset(task):
    return canonical_dataset_name(task.get("metadata", {}).get("dataset", "unknown"))


def _summary(rows, k):
    known = [row for row in rows if row["reward_known"]]
    binary = [row for row in known if row["task_outcome_passed"] is not None]
    eligible = [row for row in rows if row["training_eligible"]]
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    scored_groups = [
        group
        for group in groups.values()
        if len(group) == k and all(row["reward_known"] for row in group)
    ]
    binary_groups = [
        group
        for group in scored_groups
        if all(row["task_outcome_passed"] is not None for row in group)
    ]
    train_groups = [
        group
        for group in groups.values()
        if len(group) == k and all(row["training_eligible"] for row in group)
    ]
    successes = sum(row["task_outcome_passed"] is True for row in binary)
    solved = sum(any(row["task_outcome_passed"] for row in group) for group in binary_groups)
    all_success = sum(all(row["task_outcome_passed"] for row in group) for group in binary_groups)
    histogram = Counter(sum(row["task_outcome_passed"] for row in group) for group in binary_groups)
    return {
        "task_count": len(groups),
        "planned_rollout_count": len(rows),
        "recorded_rollout_count": sum(row["recorded"] for row in rows),
        "scored_count": len(known),
        "unscored_count": len(rows) - len(known),
        "scoring_coverage": _ratio(len(known), len(rows)),
        "binary_scored_count": len(binary),
        "success_count": successes,
        "failure_count": len(binary) - successes,
        "continuous_only_scored_count": len(known) - len(binary),
        "rollout_success_rate": _ratio(successes, len(binary)),
        "training_eligible_count": len(eligible),
        "training_eligible_rate": _ratio(len(eligible), len(rows)),
        "complete_scored_group_count": len(scored_groups),
        "complete_scored_group_rate": _ratio(len(scored_groups), len(groups)),
        "complete_training_group_count": len(train_groups),
        "complete_training_group_rate": _ratio(len(train_groups), len(groups)),
        "complete_binary_group_count": len(binary_groups),
        "task_pass_count": solved,
        "task_pass_at_k": _ratio(solved, len(binary_groups)),
        "k": k,
        "task_all_k_success_count": all_success,
        "task_all_k_success_rate": _ratio(all_success, len(binary_groups)),
        "task_success_histogram": {str(i): histogram[i] for i in range(k + 1)},
        "task_score_raw": describe(row["task_score_raw"] for row in rows),
        "training_reward_known": describe(row["training_reward"] for row in known),
        "training_reward_eligible": describe(row["training_reward"] for row in eligible),
        "policy_failure_zero_count": sum(row["policy_failure_zero"] for row in rows),
        "uncertain_attribution_zero_count": sum(row["uncertain_attribution_zero"] for row in rows),
        "policy_failure_zero_types": dict(
            Counter(
                row["policy_failure_code"] or "unknown"
                for row in rows
                if row["policy_failure_zero"]
            )
        ),
        "exclusion_reason_counts": dict(
            Counter(reason for row in rows for reason in row["exclusion_reasons"])
        ),
        "duration_s": describe(row["duration_s"] for row in rows),
        "qa_token_f1": describe(row["qa_token_f1"] for row in rows),
        "qa_answer_em": describe(row["qa_answer_em"] for row in rows),
        "qa_answer_f1": describe(row["qa_answer_f1"] for row in rows),
        "answer_submission_legal": describe(row["answer_submission_legal"] for row in rows),
        "environment_actions": describe(row["environment_actions"] for row in rows),
        "graph_agents": describe(row["graph_agents"] for row in rows),
        "single_agent_rate": _ratio(
            sum(row["graph_agents"] == 1 for row in rows),
            sum(row["graph_agents"] is not None for row in rows),
        ),
        "graph_relations": describe(row["graph_relations"] for row in rows),
        "model_selection_counts": dict(Counter(model for row in rows for model in row["models"])),
        "task_types": {
            kind: {
                "count": sum(r["task_type"] == kind for r in rows),
                "scored_count": sum(r["task_type"] == kind and r["reward_known"] for r in rows),
                "binary_scored_count": sum(
                    r["task_type"] == kind and r["task_outcome_passed"] is not None for r in rows
                ),
                "success_count": sum(
                    r["task_type"] == kind and r["task_outcome_passed"] is True for r in rows
                ),
                "success_rate": _ratio(
                    sum(r["task_type"] == kind and r["task_outcome_passed"] is True for r in rows),
                    sum(
                        r["task_type"] == kind and r["task_outcome_passed"] is not None
                        for r in rows
                    ),
                ),
            }
            for kind in sorted({r["task_type"] for r in rows})
        },
        "retry_attempt_count": sum(row["retry_attempt_count"] for row in rows),
        "retried_slot_count": sum(row["retry_attempt_count"] > 0 for row in rows),
        "retry_recovered_slot_count": sum(
            row["retry_attempt_count"] > 0 and row["reward_known"] for row in rows
        ),
        "retry_recovery_rate": _ratio(
            sum(row["retry_attempt_count"] > 0 and row["reward_known"] for row in rows),
            sum(row["retry_attempt_count"] > 0 for row in rows),
        ),
    }


def collect_outcome_metrics(
    cycle_dir, *, cycle, tasks, solver_batch, k, snapshots, admission_finalized=True
):
    cycle_dir = Path(cycle_dir)
    proposals = read_rows(cycle_dir / "tasks.jsonl")
    task_map = {row["task"]["task_id"]: row["task"] for row in proposals}
    for task in tasks:
        task_map.setdefault(
            task.task_id,
            {"task_id": task.task_id, "task_type": task.task_type, "metadata": dict(task.metadata)},
        )
    raw = {}
    for row in read_rows(cycle_dir / "solver_rollouts.jsonl"):
        key = row["rollout_id"]
        if key in raw and raw[key] != row:
            raise ValueError("conflicting primary slot records; do not select the best reward")
        raw[key] = row
    samples = {sample.rollout_id: sample for sample in solver_batch.samples}
    score_only = {
        row["rollout_id"]: row
        for row in read_rows(cycle_dir / "uncertain_failure_outcomes.jsonl")
        if row.get("reward_admission_reason") == "uncertain_attribution_zero"
    }
    attempts = defaultdict(list)
    for row in read_rows(cycle_dir / "rollout_attempts.jsonl"):
        attempts[row.get("rollout_id")].append(row)
    rows = []
    request_events = []
    for task_id, task in task_map.items():
        dataset = _dataset(task)
        for index in range(k):
            rid = f"{task_id}-r{index}"
            record = raw.get(rid, {})
            meta = dict(record.get("metadata", {}))
            sample = samples.get(rid)
            if sample is not None:
                meta.update(sample.metadata)
            if not record and rid in score_only:
                meta.update(score_only[rid])
            known = meta.get("reward_known") is True
            reward = number(meta.get("task_reward", record.get("reward")))
            known = known and reward is not None
            passed = meta.get("task_outcome_passed") if known else None
            if not isinstance(passed, bool):
                passed = None
            verification = meta.get("verification") or {}
            submission = meta.get("answer_submission") or {}
            environment = meta.get("environment_result_metrics") or {}
            for field in (
                "backend_request_events",
                "director_request_events",
            ):
                request_events.extend(meta.get(field) or [])
            graph = record.get("graph") or {}
            nodes = graph.get("nodes", [])
            if isinstance(nodes, dict):
                nodes = list(nodes.values())
            reasons = list(meta.get("training_exclusion_reasons") or [])
            if not record:
                reasons.append("missing_primary_record")
            elif not known:
                reasons.append("reward_unknown")
            elif rid not in samples and not reasons:
                reasons.append("not_in_training_batch")
            local_attempts = attempts[rid]
            retry_ids = {
                item.get("replacement_attempt")
                for item in local_attempts
                if isinstance(item.get("replacement_attempt"), int)
                and item["replacement_attempt"] > 0
            }
            official = meta.get("qa_official_metrics") or {}
            rows.append(
                {
                    "cycle": cycle,
                    "dataset": dataset,
                    "task_id": task_id,
                    "rollout_id": rid,
                    "attempt_id": max(
                        (item.get("replacement_attempt", 0) for item in local_attempts), default=0
                    ),
                    "seed": record.get("seed"),
                    "split": task.get("metadata", {}).get("source_split")
                    or task.get("metadata", {}).get("split"),
                    "difficulty": task.get("metadata", {}).get("difficulty"),
                    "qa_answer_precision": number(official.get("answer_precision"))
                    if known
                    else None,
                    "hotpot_evidence": official.get("evidence", {}),
                    "qa_answer_recall": number(official.get("answer_recall")) if known else None,
                    "qa_explicit_submission_em": number(official.get("explicit_submission_em"))
                    if known
                    else None,
                    "qa_explicit_submission_f1": number(official.get("explicit_submission_f1"))
                    if known
                    else None,
                    "task_type": str(
                        task.get("metadata", {}).get("alfworld_task_type")
                        or task.get("task_type")
                        or "unknown"
                    ),
                    "source_id": (
                        task.get("metadata", {}).get("source_task_id")
                        or task.get("metadata", {}).get("source_id")
                        or task.get("metadata", {}).get("pool_id")
                        or task_id
                    ),
                    "recorded": bool(record),
                    "submission_status": meta.get("submission_status"),
                    "submission_contract_version": meta.get("submission_contract_version"),
                    "outcome_status": (meta.get("outcome_decision") or {}).get("status"),
                    "diagnostic_qa_metrics": meta.get("diagnostic_qa_metrics"),
                    "reward_known": known,
                    "task_score_raw": number(verification.get("score"))
                    if known and meta.get("reward_admission_reason") == "trusted_task_result"
                    else None,
                    "training_reward": reward if known else None,
                    "training_eligible": (known and rid in samples)
                    if admission_finalized
                    else None,
                    "task_outcome_passed": passed,
                    "reward_source": meta.get("reward_admission_reason"),
                    "reward_version": meta.get("director_reward_version"),
                    "reward_breakdown": meta.get("task_reward_breakdown"),
                    "policy_failure_zero": bool(
                        known and reward == 0 and meta.get("typed_policy_failure")
                    ),
                    "uncertain_attribution_zero": bool(
                        known and reward == 0 and meta.get("uncertain_attribution_zero")
                    ),
                    "policy_failure_code": (meta.get("typed_policy_failure") or {}).get("code"),
                    "exclusion_reasons": sorted(set(reasons)),
                    "duration_s": number(meta.get("duration_s")),
                    "qa_token_f1": number(meta.get("qa_token_f1")) if known else None,
                    "qa_answer_em": number(official.get("answer_em", official.get("em")))
                    if known
                    else None,
                    "qa_answer_f1": number(official.get("answer_f1", official.get("f1")))
                    if known
                    else None,
                    "answer_submission_legal": submission.get("valid")
                    if isinstance(submission.get("valid"), bool)
                    else None,
                    "environment_actions": number(environment.get("steps")),
                    "graph_agents": len(nodes) if isinstance(record.get("graph"), dict) else None,
                    "graph_relations": len(graph.get("relations", []))
                    if isinstance(record.get("graph"), dict)
                    else None,
                    "models": [
                        str(n.get("metadata", {}).get("runtime_route"))
                        for n in nodes
                        if n.get("metadata", {}).get("runtime_route")
                    ],
                    "retry_attempt_count": len(retry_ids),
                    "recorded_total_token_in": number(meta.get("token_in")),
                    "recorded_total_token_out": number(meta.get("token_out")),
                }
            )
    dataset_rows = defaultdict(list)
    for row in rows:
        dataset_rows[row["dataset"]].append(row)
    datasets = {key: _summary(value, k) for key, value in sorted(dataset_rows.items())}
    for key, value in dataset_rows.items():
        for field in (
            "qa_answer_precision",
            "qa_answer_recall",
            "qa_explicit_submission_em",
            "qa_explicit_submission_f1",
        ):
            datasets[key][field] = describe(row.get(field) for row in value)
    overall = _summary(rows, k)
    for summary in datasets.values():
        summary["share_of_admitted_solver_samples"] = _ratio(
            summary["training_eligible_count"], len(samples)
        )
    # Raw scores and mixed-dataset rewards are not a common-unit task accuracy.
    overall.pop("task_score_raw")
    overall["macro_rollout_success_rate"] = describe(
        d["rollout_success_rate"] for d in datasets.values()
    )["mean"]
    overall["binary_datasets"] = [
        key for key, value in datasets.items() if value["binary_scored_count"]
    ]
    overall["selected_training_sample_count"] = len(samples)
    overall["selected_training_group_count"] = len({s.task_id for s in solver_batch.samples})
    task_rows = []
    for task_id in task_map:
        group = [row for row in rows if row["task_id"] == task_id]
        task_rows.append(
            {
                "task_id": task_id,
                "dataset": group[0]["dataset"],
                "source_id": group[0]["source_id"],
                "split": group[0]["split"],
                "rewards": [row["training_reward"] for row in group],
                "reward_mean_known": describe(row["training_reward"] for row in group)["mean"],
                "scoring_complete": all(row["reward_known"] for row in group),
            }
        )
    report = {
        "schema_version": "primary_outcomes",
        "cycle": cycle,
        "k": k,
        "admission_finalized": admission_finalized,
        "population": "planned_primary_slots_only",
        "datasets": datasets,
        "overall": overall,
        "generation_snapshots": snapshots,
        "tasks": task_rows,
        "primary_api": summarize_requests(request_events),
        "collection_incidents": {
            "record_count": len(read_rows(cycle_dir / "collection_incidents.jsonl")),
            "class_counts": dict(
                Counter(
                    str(e.get("incident_class") or "unknown")
                    for e in read_rows(cycle_dir / "collection_incidents.jsonl")
                )
            ),
        },
        "field_coverage": {
            key: sum(row.get(key) is not None for row in rows)
            for key in (
                "task_score_raw",
                "qa_token_f1",
                "qa_answer_em",
                "answer_submission_legal",
                "environment_actions",
            )
        },
    }
    advantages = defaultdict(list)
    for sample in solver_batch.samples:
        advantages[sample.task_id].append(sample.advantage)
    report["advantage"] = {
        "distribution": describe(s.advantage for s in solver_batch.samples),
        "task_groups": len(advantages),
        "zero_group_count": sum(
            all(value == 0 for value in values) for values in advantages.values()
        ),
        "zero_group_rate": _ratio(
            sum(all(value == 0 for value in values) for values in advantages.values()),
            len(advantages),
        ),
    }
    source_positions = defaultdict(list)
    for path in cycle_dir.parent.glob("cycle-*/primary_outcomes.json"):
        if path.parent == cycle_dir:
            continue
        previous = json.loads(path.read_text())
        if previous.get("cycle", cycle) >= cycle:
            continue
        for task in previous.get("tasks", []):
            source_positions[task["dataset"]].append(str(task.get("source_id", task["task_id"])))
    for task in task_rows:
        source_positions[task["dataset"]].append(str(task["source_id"]))
    report["source_coverage"] = {
        dataset: {
            "sampling_positions": len(ids),
            "unique_source_tasks": len(set(ids)),
            "repeat_sampling_rate": 1 - len(set(ids)) / len(ids),
        }
        for dataset, ids in source_positions.items()
    }
    vectors = defaultdict(list)
    for sample in solver_batch.samples:
        if getattr(sample, "graph_features", ()):
            vectors[sample.task_id].append(sample.graph_features)
    similarities = []
    for group in vectors.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if len(left) != len(right):
                    continue
                denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
                if denominator:
                    similarities.append(
                        sum(a * b for a, b in zip(left, right, strict=True)) / denominator
                    )
    report["graph_similarity"] = {
        "population": "within_task_training_graph_features_cosine",
        "cosine": describe(similarities),
    }
    if not admission_finalized:
        for summary in [report["overall"], *report["datasets"].values()]:
            for key in (
                "training_eligible_count",
                "training_eligible_rate",
                "complete_training_group_count",
                "complete_training_group_rate",
                "share_of_admitted_solver_samples",
            ):
                summary[key] = None
    write_json(cycle_dir / "primary_outcomes.json", report)
    target = cycle_dir / "primary_reward_rows.jsonl"
    temp = target.with_suffix(".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    )
    temp.replace(target)
    return report


def record_interrupted_collection(root, k):
    """Best-effort local diagnostics, never synthesize a finalized training batch."""
    from .rollouts import TrainingBatch

    candidates = sorted(
        path
        for path in Path(root).glob("cycle-*")
        if path.is_dir() and path.name.removeprefix("cycle-").isdigit()
    )
    if not candidates:
        return None
    path = candidates[-1]
    report_path = path / "primary_outcomes.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    if not (path / "tasks.jsonl").exists():
        return None
    report = collect_outcome_metrics(
        path,
        cycle=int(path.name.split("-")[-1]),
        tasks=(),
        solver_batch=TrainingBatch("solver", ()),
        k=k,
        snapshots={},
        admission_finalized=False,
    )
    report["collection_interrupted"] = True
    write_json(report_path, report)
    return report
