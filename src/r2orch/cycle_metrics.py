"""R2Orch per-cycle metrics and dashboard series.

Read-only: every value is computed from the artifacts a cycle already wrote plus the
trainer's ``PolicyUpdateResult`` objects. Nothing here feeds back into rewards, selection
or training.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name

SCHEMA = "r2orch_metrics@1"
DASHBOARD_SCHEMA = "r2orch_dashboard@1"
DASHBOARD_ORDER = ("aime", "hotpotqa", "nq", "medqa", "mbppplus", "alfworld")
_DASHBOARD_NAMES = {"nq_open": "nq"}
_EPS = 1e-9
DIRECTOR_LOSS_KEYS = (
    "outcome_adv_abs_mean", "kl_on", "kl_off", "tis_weight_mean", "tis_capped_fraction",
    "samples_on", "samples_off", "loss",
)
PROPOSER_LOSS_KEYS = (
    "pair_loss", "pairs", "skipped", "target_abs_mean", "logit_gap_mean",
    # Learning-signal monitoring (proposer_pair_training.pair_signal_stats / policy_diagnostics).
    # *_fresh values use only this cycle's own pairs (not trained on yet): the held-out view.
    "pairs_nonzero",
    "eff_participation", "eff_participation_all", "top_pair_share", "all_zero_steps", "steps",
    "rank_corr_fresh", "rank_corr_nonzero_fresh", "sign_agreement_fresh", "fresh_n",
    "fresh_gap_t_sum", "fresh_gap_sq_sum", "fresh_t_sq_sum",
    "kl_to_ref_mean", "kl_to_ref_max", "entropy_policy_mean", "same_choice_expected",
    "p_ads_policy", "p_ads_ref", "p_ads_drift", "proxy_rank_corr", "proxy_lists",
    "max_grad_norm", "grad_norm_pre_clip_mean", "grad_norm_pre_clip_max", "clip_fraction",
)


def dashboard_dataset_name(dataset: str) -> str:
    canonical = canonical_dataset_name(dataset)
    return _DASHBOARD_NAMES.get(canonical, canonical)


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _mean(values: Iterable[float | None]) -> float | None:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(items) / len(items) if items else None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _cycle_index(cycle_dir: Path) -> int:
    match = re.fullmatch(r"cycle-(\d+)", Path(cycle_dir).name)
    if match is None:
        raise ValueError(f"not a cycle directory: {cycle_dir}")
    return int(match.group(1))


def _r2orch(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    value = (metadata or {}).get("r2orch")
    return value if isinstance(value, dict) else {}


def _trusted_reward(row: Mapping[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    reward = row.get("reward")
    if metadata.get("reward_known") is not True or reward is None:
        return None
    reward = float(reward)
    return reward if math.isfinite(reward) else None


def _has_spread(rewards: Sequence[float]) -> bool:
    return len(rewards) >= 2 and max(rewards) - min(rewards) > _EPS


def _outcome_bucket(rewards: Sequence[float]) -> str:
    if all(r >= 1.0 - _EPS for r in rewards):
        return "all_correct"
    if all(r <= _EPS for r in rewards):
        return "all_wrong"
    return "mixed"


def _cluster_dataset(cluster_id: str) -> str:
    return canonical_dataset_name(re.split(r"[:/]", str(cluster_id), maxsplit=1)[0])


def ads_boundary_summary(curriculum_state: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per-dataset ADS scheduler summary from ``curriculum_state.json``.

    ``position_mean`` is the probability-weighted mean mini-cluster index inside the
    difficulty-sorted clusters (larger = harder tasks); ``success_mean`` is the
    probability-weighted cluster success EMA.
    """

    state = dict(curriculum_state or {})
    state = state.get("scheduler", state) if isinstance(state.get("scheduler"), dict) else state
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for cluster_id, values in (state.get("cluster_states") or {}).items():
        grouped[_cluster_dataset(cluster_id)].append((str(cluster_id), values))
    summary = {}
    for dataset, clusters in grouped.items():
        raw = [max(0.0, float(values.get("prob", 0.0))) for _cid, values in clusters]
        total = sum(raw)
        weights = [w / total for w in raw] if total > 0 else [1.0 / len(clusters)] * len(clusters)
        positions = [_mean(values.get("mini_positions") or [0]) for _cid, values in clusters]
        summary[dataset] = {
            "clusters": len(clusters),
            "position_mean": sum(w * p for w, p in zip(weights, positions, strict=True)),
            "success_mean": sum(
                w * float(values.get("success_rate", 0.5))
                for w, (_cid, values) in zip(weights, clusters, strict=True)
            ),
            "top_cluster": max(clusters, key=lambda item: float(item[1].get("prob", 0.0)))[0],
        }
    return summary


def _update_dict(update: Any) -> dict[str, Any]:
    if update is None:
        return {}
    return asdict(update) if is_dataclass(update) else dict(update)


def _last_step(update: Mapping[str, Any]) -> dict[str, Any]:
    steps = update.get("step_metrics") or ()
    return dict(steps[-1]) if steps else {}


def _director_losses(update: Mapping[str, Any]) -> dict[str, Any]:
    step = _last_step(update)
    losses: dict[str, Any] = {key: step.get(f"r2orch_{key}") for key in DIRECTOR_LOSS_KEYS}
    if losses["loss"] is None and update:
        losses["loss"] = update.get("loss")
    # Extra series for the dashboard's trajectory and KL curves.
    losses["policy_loss"] = update.get("policy_loss") if update else None
    losses["kl"] = update.get("kl") if update else None
    return losses


def _proposer_losses(update: Mapping[str, Any]) -> dict[str, Any]:
    step = _last_step(update)
    losses = {key: step.get(f"r2orch_{key}") for key in PROPOSER_LOSS_KEYS}
    if losses["pair_loss"] is None and update and update.get("status", "updated") == "updated":
        losses["pair_loss"] = update.get("loss")
    return losses


def _outer_metrics(snapshot: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    scopes = ((snapshot or {}).get("pats") or {}).get("scopes") or {}
    review_scopes = (review or {}).get("scopes") or {}
    reviews_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in (review or {}).get("reviews") or []:
        reviews_by_scope[str(item.get("scope"))].append(item)
    outer = {}
    for scope in sorted(set(scopes) | set(review_scopes)):
        state = scopes.get(scope) or {}
        scope_review = review_scopes.get(scope) or {}
        items = reviews_by_scope.get(scope, [])
        cards = [c for c in state.get("cards", []) if c.get("status", "active") in {"seed", "active"}]
        statuses = {item.get("step"): item.get("status") for item in items}
        outer[scope] = {
            # Scaffold size (card_count, card_tokens) the cycle's rollouts ran under; ``after`` is
            # the state after this cycle's ledger decisions and editor steps (what the next cycle
            # starts from).
            "after": {
                "card_count": scope_review.get("card_count"),
                "card_tokens": scope_review.get("card_tokens"),
            },
            "card_count": int(state.get("card_count", len(cards))),
            "card_tokens": int(state.get("card_tokens", 0) or 0),
            "edits_applied": int(scope_review.get("edits_applied", sum(
                len(item.get("operations") or []) for item in items if item.get("status") == "updated"
            ))),
            "refiner_calls": int(scope_review.get("refiner_calls", sum(
                "structured_output" in item for item in items
            ))),
            "checker_calls": int(scope_review.get("checker_calls", 0) or 0),
            "revise_status": str(
                scope_review.get("revise_status") or statuses.get("REVISE") or "none"
            ),
            "distill_status": str(
                scope_review.get("distill_status") or statuses.get("DISTILL") or "none"
            ),
        }
    return outer


def _review_tokens(review: Mapping[str, Any]) -> dict[str, int]:
    tokens = {"input": 0, "output": 0}
    for item in (review or {}).get("reviews") or []:
        attempts = item.get("generation_attempts") or []
        if attempts:
            tokens["input"] += sum(int(a.get("token_in", 0) or 0) for a in attempts)
            tokens["output"] += sum(int(a.get("token_out", 0) or 0) for a in attempts)
        else:
            tokens["output"] += int(item.get("response_tokens") or 0)
    return tokens


def _director_call_tokens(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {"input": 0, "output": 0}
    for row in rows:
        for call in row.get("policy_calls") or []:
            mask = call.get("action_mask") or []
            completion = sum(int(value) for value in mask)
            tokens["input"] += len(call.get("token_ids") or mask) - completion
            tokens["output"] += completion
    return tokens


def r2orch_cycle_metrics(
    cycle_dir: Path,
    *,
    updates: Sequence[Any],
    timing: Mapping[str, float],
) -> dict[str, Any]:
    """Build the metrics dictionary for one ``cycle-XXXX`` directory.

    ``timing`` keys: ``collection_wall_s``, ``training_wall_s``, ``cycle_wall_s`` and the
    resident service footprint ``vllm_gpus`` / ``retrieval_gpus`` (GPU counts).
    """

    cycle_dir = Path(cycle_dir)
    cycle = _cycle_index(cycle_dir)
    update_map = {u["role"]: u for u in (_update_dict(item) for item in updates) if u}
    tasks = [row["task"] for row in _read_jsonl(cycle_dir / "tasks.jsonl")]
    rollouts = _read_jsonl(cycle_dir / "solver_rollouts.jsonl")
    failures = _read_jsonl(cycle_dir / "rollout_errors.jsonl")
    lists = _read_jsonl(cycle_dir / "proposer_lists.jsonl")
    frontier = {
        item["task_id"]: _r2orch(item.get("metadata"))
        for item in _read_json(cycle_dir / "frontier_scores.json", [])
    }
    review = _read_json(cycle_dir / "pats_review.json", {})
    snapshot = _read_json(cycle_dir / "director_skill_snapshot.json", {})
    ads = ads_boundary_summary(_read_json(cycle_dir / "curriculum_state.json", {}))

    task_info: dict[str, dict[str, Any]] = {}
    for task in tasks:
        meta = task.get("metadata") or {}
        r2orch = _r2orch(meta)
        task_info[task["task_id"]] = {
            "dataset": canonical_dataset_name(r2orch.get("dataset") or meta.get("dataset", "unknown")),
        }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        r2orch = _r2orch(row.get("metadata"))
        by_task[row["task_id"]].append(row)
        task_info.setdefault(
            row["task_id"],
            {
                "dataset": canonical_dataset_name(r2orch.get("dataset") or row["metadata"].get("dataset")),
            },
        )

    datasets: dict[str, dict[str, Any]] = {}
    dataset_names = sorted({info["dataset"] for info in task_info.values()} | {
        canonical_dataset_name(row.get("dataset")) for row in lists
    })
    for dataset in dataset_names:
        task_ids = [tid for tid, info in task_info.items() if info["dataset"] == dataset]
        on, off, independent = [], [], []
        valid = valid_merged = scored_tasks = 0
        outcome_counts = Counter({"all_correct": 0, "all_wrong": 0, "mixed": 0})
        distances = []
        for task_id in task_ids:
            conditions: dict[str, list[float]] = defaultdict(list)
            for row in by_task.get(task_id, []):
                reward = _trusted_reward(row)
                if reward is None:
                    continue
                condition = _r2orch(row.get("metadata")).get("scaffold_condition", "on")
                conditions[condition].append(reward)
                (on if condition == "on" else off).append(reward)
                independent.append(reward)
            merged = [r for values in conditions.values() for r in values]
            if not merged:
                continue
            scored_tasks += 1
            valid += any(_has_spread(values) for values in conditions.values())
            valid_merged += _has_spread(merged)
            outcome_counts[_outcome_bucket(merged)] += 1
            distances.append(abs(sum(merged) / len(merged) - 0.5))
        structure = Counter()
        for task_id in task_ids:
            for row in by_task.get(task_id, []):
                structure["siblings"] += 1
                structure["multi_agent"] += len((row.get("graph") or {}).get("nodes") or []) > 1
        scores = [frontier[tid] for tid in task_ids if tid in frontier]
        f_tilde = [s["F_tilde"] for s in scores if s.get("F_tilde") is not None]
        dataset_lists = [row for row in lists if canonical_dataset_name(row.get("dataset")) == dataset]
        ok_lists = [row for row in dataset_lists if row.get("status") == "ok"]
        sources: Counter[str] = Counter({"ads": 0, "tsds": 0})
        for row in dataset_lists:
            by_pool = {c["pool_id"]: c for c in row.get("candidates") or []}
            chosen = {c["pool_id"] for c in row.get("choices") or [] if c.get("valid") and c.get("pool_id")}
            for pool_id in chosen:
                sources[str(by_pool.get(pool_id, {}).get("source", "ads"))] += 1
        datasets[dataset] = {
            "tasks": len(task_ids),
            "acc_on": _mean(on),
            "acc_off": _mean(off),
            "n_on": len(on),
            "n_off": len(off),
            "acc_independent": _mean(independent),
            "valid_group_fraction": _ratio(valid, scored_tasks) or 0.0,
            "valid_group_fraction_merged": _ratio(valid_merged, scored_tasks) or 0.0,
            "group_outcome_counts": dict(outcome_counts),
            "multi_agent_fraction": _ratio(structure["multi_agent"], structure["siblings"]),
            "F_tilde_mean": _mean(f_tilde),
            "F_tilde_negative_fraction": _ratio(sum(v < 0 for v in f_tilde), len(f_tilde)),
            "selected_success_distance_from_half": _mean(distances),
            "pair_same_choice_rate": _ratio(sum(bool(r.get("same_choice")) for r in ok_lists), len(ok_lists)),
            "chosen_source_counts": dict(sources),
            "ads_boundary": ads.get(dataset, {}),
            "slot_failures": sum(canonical_dataset_name(row.get("dataset")) == dataset for row in failures),
        }

    worker_primary = {
        "input": sum(int((row.get("metadata") or {}).get("token_in", 0) or 0) for row in rollouts),
        "output": sum(int((row.get("metadata") or {}).get("token_out", 0) or 0) for row in rollouts),
    }
    proposer_tokens = {"input": 0, "output": 0}
    for row in lists:
        attempted = len(row.get("choices") or [])
        proposer_tokens["input"] += attempted * len(row.get("prompt_token_ids") or [])
        proposer_tokens["output"] += sum(c.get("label") is not None for c in row.get("choices") or [])
    cycle_wall = float(timing.get("cycle_wall_s", 0.0) or 0.0)
    gpu_seconds = {
        "director_train": float(update_map.get("solver", {}).get("duration_seconds", 0.0) or 0.0),
        "proposer_train": float(update_map.get("proposer", {}).get("duration_seconds", 0.0) or 0.0),
        "vllm": cycle_wall * float(timing.get("vllm_gpus", 0) or 0),
        "retrieval": cycle_wall * float(timing.get("retrieval_gpus", 0) or 0),
    }
    return {
        "schema": SCHEMA,
        "cycle": cycle,
        "datasets": datasets,
        "losses": {
            "director": _director_losses(update_map.get("solver", {})),
            "proposer": _proposer_losses(update_map.get("proposer", {})),
        },
        "outer": _outer_metrics(snapshot, review),
        "cost": {
            "collection_wall_s": float(timing.get("collection_wall_s", 0.0) or 0.0),
            "training_wall_s": float(timing.get("training_wall_s", 0.0) or 0.0),
            "cycle_wall_s": cycle_wall,
            "gpu_seconds": gpu_seconds,
            "tokens": {
                "primary": {"worker": worker_primary, "director": _director_call_tokens(rollouts)},
                "review": {"refiner": _review_tokens(review)},
                "proposer": {"proposer": proposer_tokens},
            },
            "executions": {
                "independent": len(rollouts),
            },
        },
    }


def write_cycle_metrics(cycle_dir: Path, metrics: Mapping[str, Any]) -> None:
    """Write ``cycle_dir/r2orch_metrics.json`` and append ``<output>/r2orch_metrics.jsonl``."""

    cycle_dir = Path(cycle_dir)
    _atomic_json(cycle_dir / "r2orch_metrics.json", dict(metrics))
    with (cycle_dir.parent / "r2orch_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def _across(datasets: Mapping[str, Mapping[str, Any]], key: str) -> float | None:
    return _mean(values.get(key) for values in datasets.values())


def dashboard_round(metrics: Mapping[str, Any]) -> dict:
    """One ``rounds[]`` record of ``dashboard.json``."""

    datasets = metrics.get("datasets") or {}
    director = (metrics.get("losses") or {}).get("director") or {}
    proposer = (metrics.get("losses") or {}).get("proposer") or {}
    outer = metrics.get("outer") or {}
    cost = metrics.get("cost") or {}
    sources = Counter()
    for values in datasets.values():
        sources.update(values.get("chosen_source_counts") or {})
    total_tasks = sum(int(values.get("tasks", 0)) for values in datasets.values())
    tokens = (cost.get("tokens") or {})
    worker_tokens = sum(
        int(side.get("input", 0)) + int(side.get("output", 0))
        for phase in ("primary",)
        for side in [((tokens.get(phase) or {}).get("worker") or {})]
    )
    record = {
        "round": int(metrics["cycle"]),
        "acc": {
            dashboard_dataset_name(name): {"on": values.get("acc_on"), "off": values.get("acc_off")}
            for name, values in datasets.items()
        },
        "loss": {
            "director_traj": director.get("policy_loss"),
            "kl": director.get("kl"),
            "proposer": proposer.get("pair_loss"),
        },
        "proposer": {
            "f_tilde_mean": _across(datasets, "F_tilde_mean"),
            "valid_group_frac": _ratio(
                sum(float(v.get("valid_group_fraction", 0.0)) * int(v.get("tasks", 0)) for v in datasets.values()),
                total_tasks,
            ),
            "dist_half": _across(datasets, "selected_success_distance_from_half"),
            "dup_rate": _across(datasets, "pair_same_choice_rate"),
            "src_ads_frac": _ratio(sources["ads"], sources["ads"] + sources["tsds"]),
            "signal": {
                key: proposer.get(source)
                for key, source in (
                    ("pairs_nonzero", "pairs_nonzero"),
                    ("eff_participation", "eff_participation"),
                    ("rank_corr_fresh", "rank_corr_fresh"),
                    ("kl_to_ref", "kl_to_ref_mean"),
                    ("p_ads_policy", "p_ads_policy"),
                    ("p_ads_drift", "p_ads_drift"),
                    ("same_choice_expected", "same_choice_expected"),
                    ("clip_fraction", "clip_fraction"),
                )
            },
            "ads_boundary": {
                dashboard_dataset_name(name): (values.get("ads_boundary") or {}).get("position_mean")
                for name, values in datasets.items()
            },
        },
        "outer": {
            "scopes": {
                scope: {
                    "cards": values.get("card_count"),
                    "tokens": values.get("card_tokens"),
                }
                for scope, values in outer.items()
            },
            "total_cards": sum(int(values.get("card_count", 0) or 0) for values in outer.values()),
            "total_card_tokens": sum(int(values.get("card_tokens", 0) or 0) for values in outer.values()),
            "edits": sum(int(values.get("edits_applied", 0) or 0) for values in outer.values()),
        },
        "cost": {
            "wall_seconds": cost.get("cycle_wall_s"),
            "gpu_seconds": sum(float(v or 0.0) for v in (cost.get("gpu_seconds") or {}).values()),
            "worker_tokens": worker_tokens,
        },
    }
    return record


def write_dashboard(output_dir: Path) -> Path:
    """Rebuild ``<output>/dashboard.json`` from every ``cycle-*/r2orch_metrics.json``."""

    output_dir = Path(output_dir)
    rounds = []
    for path in sorted(output_dir.glob("cycle-*/r2orch_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rounds.append(dashboard_round(metrics))
    seen = {name for record in rounds for name in record["acc"]}
    datasets = [name for name in DASHBOARD_ORDER if name in seen] + sorted(seen - set(DASHBOARD_ORDER))
    destination = output_dir / "dashboard.json"
    _atomic_json(
        destination,
        {"schema": DASHBOARD_SCHEMA, "datasets": datasets or list(DASHBOARD_ORDER), "rounds": rounds},
    )
    return destination


def serving_gpu_counts(config_path: str | Path | None) -> dict[str, int]:
    """Resident GPU footprint (vLLM, retrieval) read from the rendered config."""

    if not config_path or not Path(config_path).is_file():
        return {"vllm_gpus": 0, "retrieval_gpus": 0}
    import tomllib

    with Path(config_path).open("rb") as handle:
        serving = (tomllib.load(handle).get("r2orch") or {}).get("serving") or {}
    return {
        "vllm_gpus": len(serving.get("vllm_gpu_ids") or ()),
        "retrieval_gpus": int(str(serving.get("retrieval_device", "cpu")).startswith("cuda")),
    }
