# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TSDS probability assignment: Copyright (c) 2024 Zifan Liu, MIT License.
# The full MIT text is retained at ``third_party/notices/TSDS-LICENSE``.
"""Fixed-pool curriculum adapted from ADS and TSDS.

The ADS cluster state, boundary band, neighbour movement and smoothed cluster
distribution are migrated from ADS.  The TSDS KDE-weighted probability assignment
is migrated from TSDS.  FAISS is replaced by exact NumPy distances because the
local pools contain only hundreds of tasks per dataset.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .observability import TaskSpec


CLUSTER_WEIGHTINGS = ("learnability",)
ANCHOR_MODES = ("sequential", "coverage_random")
MINI_INITS = ("spread",)
# A group of at least this many rollouts is needed before an all-correct / all-wrong outcome
# counts as evidence that a task is saturated / hopeless (a single rollout says almost nothing).
MIN_EVIDENCE_ROLLOUTS = 2
HISTORY_KEEP = 16
# Fewer boundary candidates than this: bring persistently saturated/hopeless tasks back.
THIN_CANDIDATES = 4


def task_learnability(rho: float) -> float:
    """1 at rho = 0.5, 0 for an all-correct or all-wrong group (4 rho (1 - rho))."""
    rho = min(1.0, max(0.0, float(rho)))
    return 4.0 * rho * (1.0 - rho)


class TextEmbedder(Protocol):
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class CurriculumProfile:
    """Runtime sizing for one fixed-pool ADS+TSDS experiment regime."""

    name: str
    expected_datasets: int
    tasks_per_cycle: int
    num_clusters_per_dataset: int
    active_clusters_per_dataset: int
    mini_cluster_size: int
    candidate_count: int
    task_window: int
    rollout_workers: int
    rollouts_per_task: int
    boundary_eps: float = 0.17
    alpha: float = 0.3
    cooldown_per_dataset: int = 64
    nominal_epoch_cycles: int = 0
    # Learnability-driven selection; see ADSBoundaryScheduler for what each key does.
    cluster_weighting: str = "learnability"
    cluster_floor: float = 0.2
    cluster_ema: float = 1.0
    anchor_mode: str = "sequential"
    mini_init: str = "spread"
    saturated_min_obs: int = 0
    hopeless_min_obs: int = 0
    history_window: int = 48
    learnable_first: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> CurriculumProfile:
        with Path(path).open("rb") as handle:
            payload = tomllib.load(handle)
        values = payload.get("curriculum", payload)
        if not isinstance(values, dict):
            raise ValueError("curriculum profile must contain a [curriculum] table")
        profile = cls(
            name=str(values.get("name", Path(path).stem)),
            expected_datasets=int(values["expected_datasets"]),
            tasks_per_cycle=int(values["tasks_per_cycle"]),
            num_clusters_per_dataset=int(values["num_clusters_per_dataset"]),
            active_clusters_per_dataset=int(values["active_clusters_per_dataset"]),
            mini_cluster_size=int(values["mini_cluster_size"]),
            candidate_count=int(values["candidate_count"]),
            task_window=int(values["task_window"]),
            rollout_workers=int(values["rollout_workers"]),
            rollouts_per_task=int(values["rollouts_per_task"]),
            boundary_eps=float(values.get("boundary_eps", 0.17)),
            alpha=float(values.get("alpha", 0.3)),
            cooldown_per_dataset=int(values.get("cooldown_per_dataset", 64)),
            nominal_epoch_cycles=int(values.get("nominal_epoch_cycles", 0)),
            cluster_weighting=str(values.get("cluster_weighting", "learnability")),
            cluster_floor=float(values.get("cluster_floor", 0.2)),
            cluster_ema=float(values.get("cluster_ema", 1.0)),
            anchor_mode=str(values.get("anchor_mode", "sequential")),
            mini_init=str(values.get("mini_init", "spread")),
            saturated_min_obs=int(values.get("saturated_min_obs", 0)),
            hopeless_min_obs=int(values.get("hopeless_min_obs", 0)),
            history_window=int(values.get("history_window", 48)),
            learnable_first=bool(values.get("learnable_first", False)),
        )
        profile.validate()
        return profile

    def scheduler_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of :class:`ADSBoundaryScheduler` that this profile controls."""
        return {
            "active_clusters": self.active_clusters_per_dataset,
            "mini_cluster_size": self.mini_cluster_size,
            "boundary_eps": self.boundary_eps,
            "alpha": self.alpha,
            "cooldown": self.cooldown_per_dataset,
            "cluster_weighting": self.cluster_weighting,
            "cluster_floor": self.cluster_floor,
            "cluster_ema": self.cluster_ema,
            "anchor_mode": self.anchor_mode,
            "mini_init": self.mini_init,
            "saturated_min_obs": self.saturated_min_obs,
            "hopeless_min_obs": self.hopeless_min_obs,
            "history_window": self.history_window,
            "learnable_first": self.learnable_first,
        }

    def validate(self) -> None:
        positive = (
            self.expected_datasets,
            self.tasks_per_cycle,
            self.num_clusters_per_dataset,
            self.active_clusters_per_dataset,
            self.mini_cluster_size,
            self.candidate_count,
            self.task_window,
            self.rollout_workers,
            self.rollouts_per_task,
        )
        if min(positive) <= 0:
            raise ValueError("curriculum profile sizes must be positive")
        if not 0.0 < self.boundary_eps < 0.5:
            raise ValueError("curriculum boundary_eps must be in (0, 0.5)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("curriculum alpha must be in (0, 1]")
        if self.cooldown_per_dataset < 0 or self.nominal_epoch_cycles < 0:
            raise ValueError("curriculum cooldown/epoch values cannot be negative")
        if self.cluster_weighting not in CLUSTER_WEIGHTINGS:
            raise ValueError(f"curriculum cluster_weighting must be one of {CLUSTER_WEIGHTINGS}")
        if not 0.0 <= self.cluster_floor < 1.0:
            raise ValueError("curriculum cluster_floor must be in [0, 1)")
        if not 0.0 < self.cluster_ema <= 1.0:
            raise ValueError("curriculum cluster_ema must be in (0, 1]")
        if self.anchor_mode not in ANCHOR_MODES:
            raise ValueError(f"curriculum anchor_mode must be one of {ANCHOR_MODES}")
        if self.mini_init not in MINI_INITS:
            raise ValueError(f"curriculum mini_init must be one of {MINI_INITS}")
        if min(self.saturated_min_obs, self.hopeless_min_obs) < 0 or self.history_window <= 0:
            raise ValueError("curriculum saturation counts must be >= 0 and history_window > 0")


@dataclass(frozen=True)
class PoolTask:
    pool_id: str
    task: TaskSpec
    dataset: str
    cluster_id: str
    difficulty_score: float
    embedding: tuple[float, ...]


class FixedTaskPool:
    """Trusted train-only task pool; the Proposer can select but cannot edit it."""

    def __init__(self, tasks: Iterable[PoolTask]) -> None:
        items = list(tasks)
        if not items:
            raise ValueError("fixed task pool cannot be empty")
        ids = [item.pool_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("fixed task pool ids must be unique")
        dimensions = {len(item.embedding) for item in items}
        if dimensions == {0} or len(dimensions) != 1:
            raise ValueError("all fixed-pool tasks need equal non-empty embeddings")
        self.tasks = {item.pool_id: item for item in items}
        self.ids = tuple(ids)

    @classmethod
    def from_jsonl(
        cls,
        paths: Iterable[str | Path],
        *,
        embedder: TextEmbedder | None = None,
        require_ads_metadata: bool = False,
        require_validation_manifest: bool = False,
    ) -> FixedTaskPool:
        rows: list[dict[str, Any]] = []
        for raw_path in paths:
            path = Path(raw_path)
            attestation: dict[str, Any] = {}
            if require_validation_manifest:
                manifest_path = path.with_name(f"{path.stem}_manifest.json")
                if not manifest_path.exists():
                    raise ValueError(f"validated task-pool manifest is missing: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("validator_version") != "fixed_task_pool_validator":
                    raise ValueError("fixed task-pool validator version mismatch")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != manifest.get("validated_pool_sha256"):
                    raise ValueError("validated task-pool hash does not match its manifest")
                attestation = {
                    "validated_pool_entry": True,
                    "validated_pool_version": manifest["validator_version"],
                    "validated_pool_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "validated_pool_sha256": digest,
                    "verifier_contract_version": manifest.get("verifier_contract_version"),
                    "adapter_contract_version": manifest.get("adapter_contract_version"),
                }
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["__validation_attestation__"] = dict(attestation)
                    rows.append(row)
        if any(str(row.get("split", "train")).casefold() != "train" for row in rows):
            raise ValueError("fixed task pool accepts train split rows only")
        if require_ads_metadata:
            required = {
                "embedding",
                "cluster_id",
                "difficulty_score",
                "rank_in_cluster",
                "cluster_size",
            }
            for index, row in enumerate(rows):
                missing = sorted(required - row.keys())
                if missing:
                    raise ValueError(
                        "formal ADS pool row "
                        f"{row.get('id', index)!r} is missing {', '.join(missing)}; "
                        "run `python -m r2orch prepare-ads-pool` first"
                    )
            grouped_ads_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped_ads_rows[str(row["cluster_id"])].append(row)
            for cluster_id, members in grouped_ads_rows.items():
                declared_sizes = {int(row["cluster_size"]) for row in members}
                ranks = sorted(int(row["rank_in_cluster"]) for row in members)
                expected_ranks = list(range(len(members)))
                if declared_sizes != {len(members)} or ranks != expected_ranks:
                    raise ValueError(
                        f"inconsistent ADS cluster metadata for {cluster_id!r}: "
                        "cluster_size must match the loaded cluster and ranks must be contiguous"
                    )

        prompts = [_task_prompt(row) for row in rows]
        missing_embeddings = ["embedding" not in row for row in rows]
        if any(missing_embeddings):
            if embedder is None:
                raise ValueError("task rows without embedding require a text embedder")
            encoded = embedder.encode(prompts, query=False)
        else:
            encoded = [tuple(float(value) for value in row["embedding"]) for row in rows]

        occurrences: Counter[str] = Counter()
        tasks: list[PoolTask] = []
        for index, (row, prompt, vector) in enumerate(zip(rows, prompts, encoded, strict=True)):
            source_id = str(row.get("id", row.get("source_id", index)))
            duplicate_index = occurrences[source_id]
            occurrences[source_id] += 1
            pool_id = source_id if duplicate_index == 0 else f"{source_id}#row-{duplicate_index}"
            metadata = {
                **dict(row.get("metadata") or {}),
                **dict(row.get("__validation_attestation__") or {}),
                **{
                    key: row[key]
                    for key in (
                        "dataset",
                        "split",
                        "mode",
                        "task_type",
                        "verifier",
                        "context_documents",
                        "cluster_id",
                        "difficulty_score",
                        "rank_in_cluster",
                        "cluster_size",
                        "ads_sample_id",
                    )
                    if key in row
                },
            }
            dataset = str(row.get("dataset", metadata.get("dataset", "unknown")))
            cluster_id = str(row.get("cluster_id", metadata.get("cluster_id", dataset)))
            difficulty = float(
                row.get(
                    "difficulty_score",
                    metadata.get("difficulty_score", row.get("rank_in_cluster", index)),
                )
            )
            rank_in_cluster = int(row.get("rank_in_cluster", index))
            cluster_size = int(row.get("cluster_size", len(rows)))
            vector_tuple = tuple(float(value) for value in vector)
            if require_ads_metadata and (
                not np.isfinite(difficulty)
                or rank_in_cluster < 0
                or cluster_size <= rank_in_cluster
                or not np.isfinite(np.asarray(vector_tuple, dtype=np.float64)).all()
            ):
                raise ValueError(f"invalid ADS metadata for fixed-pool row {source_id!r}")
            reference = _task_reference(row)
            tasks.append(
                PoolTask(
                    pool_id=pool_id,
                    task=TaskSpec(
                        task_id=pool_id,
                        prompt=prompt,
                        reference=reference,
                        task_type=str(row.get("task_type", metadata.get("task_type", "general"))),
                        metadata={
                            **metadata,
                            "pool_id": pool_id,
                            "source_task_id": source_id,
                            "required_reasoning_hops": None,
                        },
                    ),
                    dataset=dataset,
                    cluster_id=cluster_id,
                    difficulty_score=difficulty,
                    embedding=vector_tuple,
                )
            )
        return cls(tasks)

    def balanced_selection_seeds(self, limit: int, *, offset_per_dataset: int = 0) -> list[Any]:
        """Round-robin distinct source tasks across datasets.

        A fixed pool may contain several deterministic rows for one underlying
        source task.  A frozen evaluation/training anchor schedule must never
        count those replicas as different questions, even though the adaptive
        scheduler still needs to retain them as individual pool rows internally.
        """

        from .selfplay import SelfPlaySeed

        if limit <= 0:
            return []
        by_dataset: dict[str, list[str]] = defaultdict(list)
        seen_sources: dict[str, set[str]] = defaultdict(set)
        for pool_id in self.ids:
            task = self.tasks[pool_id]
            source_id = self._source_id(task)
            if source_id in seen_sources[task.dataset]:
                continue
            seen_sources[task.dataset].add(source_id)
            by_dataset[task.dataset].append(pool_id)
        datasets = sorted(by_dataset)
        selected: list[str] = []
        offset = max(0, int(offset_per_dataset))
        while len(selected) < limit:
            added = False
            for dataset in datasets:
                ids = by_dataset[dataset]
                if offset < len(ids):
                    selected.append(ids[offset])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            offset += 1
        return [
            SelfPlaySeed(
                self.tasks[pool_id].task.prompt,
                seed_id=pool_id,
                metadata={
                    "pool_id": pool_id,
                    "dataset": self.tasks[pool_id].dataset,
                    "cluster_id": self.tasks[pool_id].cluster_id,
                },
            )
            for pool_id in selected
        ]

    def seeds_for_ids(self, pool_ids: Iterable[str]) -> list[Any]:
        """Selection seeds (Proposer anchors) for explicit pool ids, in the given order."""

        from .selfplay import SelfPlaySeed

        return [
            SelfPlaySeed(
                self.tasks[pool_id].task.prompt,
                seed_id=pool_id,
                metadata={
                    "pool_id": pool_id,
                    "dataset": self.tasks[pool_id].dataset,
                    "cluster_id": self.tasks[pool_id].cluster_id,
                },
            )
            for pool_id in pool_ids
        ]

    @staticmethod
    def _source_id(task: PoolTask) -> str:
        """Return the identity used to de-duplicate fixed schedule anchors."""

        return str(task.task.metadata.get("source_task_id") or task.pool_id)


@dataclass
class ADSClusterState:
    cluster_id: str
    cluster_size: int
    success_rate: float = 0.5
    prob: float = 0.0
    mini_positions: set[int] = field(default_factory=set)
    # Learnability-weighting state: EMA of the batch-mean task learnability 4 rho (1 - rho)
    # (1.0 = optimistic prior for a cluster that was never observed) and the number of
    # batches that updated this cluster.
    learnability: float = 1.0
    observations: int = 0


def _spread_positions(size: int, count: int) -> set[int]:
    """``count`` positions spread evenly over ``range(size)`` (all of them when count >= size)."""
    count = min(int(count), int(size))
    if count >= size:
        return set(range(size))
    if count == 1:
        return {size // 2}
    return {round(i * (size - 1) / (count - 1)) for i in range(count)}


class ADSBoundaryScheduler:
    """ADS boundary mini-clusters over a trusted fixed task pool.

    The keyword options below are learnability refinements of ADS:

    * ``cluster_weighting="learnability"``: the inter-cluster distribution follows the
      EMA of each cluster's mean task learnability 4 rho (1 - rho) instead of being
      proportional to the success rate (which gave all-correct clusters the largest
      weight and starved all-wrong ones). ``cluster_floor`` is the share of probability
      mass spread uniformly so that no cluster starves; ``cluster_ema`` is the EMA
      coefficient of the cluster statistics (1.0 = overwrite by the batch mean, as ADS).
    * ``anchor_mode="coverage_random"``: :meth:`anchor_ids` draws the Proposer anchors
      pseudo-randomly over the whole pool (seeded, cluster-weighted, least-selected
      first) instead of reading pool rows in file order.
    * ``mini_init="spread"``: the boundary mini-clusters start spread over the whole
      difficulty range instead of at the easiest end of each cluster (where movement
      is one step per observation and the boundary stayed saturated).
    * ``learnable_first``: among the boundary candidates, tasks whose last group was mixed
      (0 < rho < 1) come before never-observed ones, which come before the rest, and a
      "revisit" bucket of such tasks from any cluster takes the first slot of every round, so
      a learnable task found once is revisited after its cooldown instead of being forgotten.
      Anchors ignore this tier (coverage only), or one found task would anchor every list.
    * ``saturated_min_obs`` / ``hopeless_min_obs``: a task with at least that many
      all-correct (all-wrong) observations within the last ``history_window`` updates is
      left out of the candidates and anchors unless the pool is thin.
    """

    def __init__(
        self,
        pool: FixedTaskPool,
        *,
        active_clusters: int = 4,
        mini_cluster_size: int = 32,
        boundary_eps: float = 0.17,
        alpha: float = 0.3,
        seed: int = 42,
        cooldown: int = 64,
        cluster_weighting: str = "learnability",
        cluster_floor: float = 0.2,
        cluster_ema: float = 1.0,
        anchor_mode: str = "sequential",
        mini_init: str = "spread",
        saturated_min_obs: int = 0,
        hopeless_min_obs: int = 0,
        history_window: int = 48,
        learnable_first: bool = False,
    ) -> None:
        if cluster_weighting not in CLUSTER_WEIGHTINGS:
            raise ValueError(f"cluster_weighting must be one of {CLUSTER_WEIGHTINGS}")
        if anchor_mode not in ANCHOR_MODES or mini_init not in MINI_INITS:
            raise ValueError("unknown anchor_mode or mini_init")
        if not 0.0 <= cluster_floor < 1.0 or not 0.0 < cluster_ema <= 1.0:
            raise ValueError("cluster_floor must be in [0, 1) and cluster_ema in (0, 1]")
        self.cluster_weighting = cluster_weighting
        self.cluster_floor = float(cluster_floor)
        self.cluster_ema = float(cluster_ema)
        self.anchor_mode = anchor_mode
        self.mini_init = mini_init
        self.saturated_min_obs = max(0, int(saturated_min_obs))
        self.hopeless_min_obs = max(0, int(hopeless_min_obs))
        self.history_window = max(1, int(history_window))
        self.learnable_first = bool(learnable_first)
        self.update_index = 0
        self._excluded_cache: dict[str | None, set[str]] = {}
        # pool_id -> [[update_index, rho, n_rollouts], ...], the last HISTORY_KEEP observations.
        self.task_history: dict[str, list[list[float]]] = {}
        self.pool = pool
        self.active_clusters = min(
            int(active_clusters), len({t.cluster_id for t in pool.tasks.values()})
        )
        self.mini_cluster_size = int(mini_cluster_size)
        self.boundary_eps = float(boundary_eps)
        self.alpha = float(alpha)
        self.band_low = 0.5 - self.boundary_eps
        self.band_high = 0.5 + self.boundary_eps
        self.rng = np.random.default_rng(seed)
        self.cooldown = max(0, int(cooldown))
        self.recent: list[str] = []
        self.reserved: set[str] = set()
        self.selection_count: Counter[str] = Counter()
        self.task_success: dict[str, float] = {}
        self.source_to_pool_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
        for task in pool.tasks.values():
            source_id = self._source_id(task)
            self.source_to_pool_ids[(task.dataset, source_id)].add(task.pool_id)

        grouped: dict[str, list[PoolTask]] = defaultdict(list)
        for task in pool.tasks.values():
            grouped[task.cluster_id].append(task)
        self.cluster_ids = sorted(grouped)
        self.cluster_dataset: dict[str, str] = {}
        for cluster_id, tasks in grouped.items():
            datasets = {task.dataset for task in tasks}
            if len(datasets) != 1:
                raise ValueError(f"ADS cluster {cluster_id!r} mixes multiple datasets")
            self.cluster_dataset[cluster_id] = next(iter(datasets))
        self.dataset_cluster_ids = {
            dataset: tuple(
                cluster_id
                for cluster_id in self.cluster_ids
                if self.cluster_dataset[cluster_id] == dataset
            )
            for dataset in sorted(set(self.cluster_dataset.values()))
        }
        self.cluster_to_sorted_ids = {
            cluster_id: tuple(
                task.pool_id
                for task in sorted(tasks, key=lambda item: (item.difficulty_score, item.pool_id))
            )
            for cluster_id, tasks in grouped.items()
        }
        self.cluster_states = {
            cluster_id: ADSClusterState(
                cluster_id=cluster_id,
                cluster_size=len(task_ids),
                prob=1.0 / len(self.dataset_cluster_ids[self.cluster_dataset[cluster_id]]),
                mini_positions=self._initial_positions(len(task_ids)),
            )
            for cluster_id, task_ids in self.cluster_to_sorted_ids.items()
        }

    def _order_key(self, pool_id: str) -> tuple[int, int, str]:
        """Preference among candidates (smaller first): learnability tier, selection count, id.

        Without ``learnable_first`` this is the original ADS order, least selected first with the
        id as tie-break.
        """
        tier = 0
        if self.learnable_first:
            history = self.task_history.get(pool_id)
            if not history:
                tier = 1
            else:
                _, rho, count = history[-1]
                tier = 0 if count >= MIN_EVIDENCE_ROLLOUTS and 0.0 < rho < 1.0 else 2
        return (tier, self.selection_count[pool_id], pool_id)

    def _initial_positions(self, size: int) -> set[int]:
        return _spread_positions(size, self.mini_cluster_size)

    def candidates(self, *, dataset: str | None = None) -> list[str]:
        cluster_ids = (
            list(self.dataset_cluster_ids.get(dataset, ())) if dataset else self.cluster_ids
        )
        if not cluster_ids:
            raise ValueError(f"fixed task pool has no ADS clusters for dataset {dataset!r}")
        probs = np.asarray([self.cluster_states[cid].prob for cid in cluster_ids], dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(probs), 1 / len(probs))
        chosen = self.rng.choice(
            len(cluster_ids),
            size=min(self.active_clusters, len(cluster_ids)),
            replace=False,
            p=probs,
        )
        recent = self._recent_ids(dataset)
        skipped = self.excluded_ids(dataset)
        buckets: list[list[str]] = []
        held_back: list[str] = []
        for cluster_index in chosen.tolist():
            cluster_id = cluster_ids[int(cluster_index)]
            task_ids = self.cluster_to_sorted_ids[cluster_id]
            positions = sorted(self.cluster_states[cluster_id].mini_positions)
            eligible = [
                task_ids[position]
                for position in positions
                if task_ids[position] not in recent and task_ids[position] not in self.reserved
            ]
            held_back.extend(pool_id for pool_id in eligible if pool_id in skipped)
            buckets.append(
                sorted(
                    (pool_id for pool_id in eligible if pool_id not in skipped),
                    key=self._order_key,
                )
            )
        if self.learnable_first:
            # Revisit bucket: learnable tasks found earlier, wherever they sit in their cluster,
            # take the first slot of every round (the boundary movement alone only keeps those
            # that happen to sit inside a drawn cluster's mini-cluster).
            in_buckets = {pool_id for bucket in buckets for pool_id in bucket}
            revisit = sorted(
                (
                    pool_id
                    for pool_id in self.known_learnable_ids(dataset)
                    if pool_id not in recent
                    and pool_id not in self.reserved
                    and pool_id not in skipped
                    and pool_id not in in_buckets
                ),
                key=self._order_key,
            )
            buckets.insert(0, revisit)
        selected = [
            bucket[index]
            for index in range(max((len(bucket) for bucket in buckets), default=0))
            for bucket in buckets
            if index < len(bucket)
        ]
        if len(selected) < THIN_CANDIDATES and held_back:
            # Thin boundary: persistently saturated / hopeless tasks come back, least selected first.
            selected += sorted(held_back, key=self._order_key)
        if not selected:
            selected = [
                pool_id
                for pool_id in self.pool.ids
                if pool_id not in recent
                and pool_id not in self.reserved
                and (dataset is None or self.pool.tasks[pool_id].dataset == dataset)
            ]
        return selected

    def reserve(self, pool_id: str) -> None:
        if pool_id not in self.pool.tasks:
            raise KeyError(f"unknown fixed-pool task: {pool_id}")
        if pool_id not in self.task_success:
            self.reserved.update(self._source_equivalents(pool_id))

    def record_batch(self, observations: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Apply one deterministic ADS update from a task execution window."""

        cluster_rhos: dict[str, list[float]] = defaultdict(list)
        movements: dict[str, dict[int, int]] = defaultdict(dict)
        affected_datasets: set[str] = set()
        observed_any = False
        self._excluded_cache = {}
        for pool_id, rewards in observations:
            if pool_id not in self.pool.tasks:
                raise KeyError(f"unknown fixed-pool task: {pool_id}")
            if not rewards:
                continue
            observed_any = True
            rho = float(np.mean(np.clip(np.asarray(rewards, dtype=np.float64), 0.0, 1.0)))
            self.task_success[pool_id] = rho
            self.selection_count[pool_id] += 1
            history = self.task_history.setdefault(pool_id, [])
            history.append([float(self.update_index), rho, float(len(rewards))])
            del history[:-HISTORY_KEEP]
            self.reserved.difference_update(self._source_equivalents(pool_id))
            self.recent.append(pool_id)
            cluster_id = self.pool.tasks[pool_id].cluster_id
            affected_datasets.add(self.pool.tasks[pool_id].dataset)
            cluster_rhos[cluster_id].append(rho)
            task_ids = self.cluster_to_sorted_ids[cluster_id]
            position = task_ids.index(pool_id)
            if position in self.cluster_states[cluster_id].mini_positions:
                movements[cluster_id][position] = (
                    -1 if rho < self.band_low else 1 if rho > self.band_high else 0
                )

        retained = self.cooldown * len(self.dataset_cluster_ids)
        if retained and len(self.recent) > retained:
            self.recent = self.recent[-retained:]
        for cluster_id, values in cluster_rhos.items():
            state = self.cluster_states[cluster_id]
            batch_success = float(np.mean(values))
            batch_learnability = float(np.mean([task_learnability(value) for value in values]))
            # ema == 1 overwrites by the batch mean (ADS); a cluster's first batch always does.
            weight = 1.0 if state.observations == 0 else self.cluster_ema
            state.success_rate = (1.0 - weight) * state.success_rate + weight * batch_success
            state.learnability = (1.0 - weight) * state.learnability + weight * batch_learnability
            state.observations += 1
            self._update_mini_positions(cluster_id, movements[cluster_id])
        for dataset in sorted(affected_datasets):
            self._update_inter_cluster_distribution(dataset)
        if observed_any:
            self.update_index += 1

    def _update_inter_cluster_distribution(self, dataset: str) -> None:
        cluster_ids = self.dataset_cluster_ids[dataset]
        # Weight clusters by how much learnable (mixed-outcome) signal they yield, with a
        # uniform floor so that a cluster that looks saturated or hopeless now is not starved.
        scores = np.asarray(
            [self.cluster_states[cid].learnability for cid in cluster_ids], dtype=np.float64
        )
        shares = scores / scores.sum() if scores.sum() > 0 else np.full(len(scores), 1 / len(scores))
        target = self.cluster_floor / len(cluster_ids) + (1.0 - self.cluster_floor) * shares
        for cluster_id, target_prob in zip(cluster_ids, target.tolist(), strict=True):
            state = self.cluster_states[cluster_id]
            state.prob = (1.0 - self.alpha) * state.prob + self.alpha * float(target_prob)
        total = sum(self.cluster_states[cluster_id].prob for cluster_id in cluster_ids)
        for cluster_id in cluster_ids:
            self.cluster_states[cluster_id].prob /= total

    # ---- persistent saturation / hopelessness -----------------------------------------
    def _count_recent(self, pool_id: str, *, saturated: bool) -> int:
        horizon = self.update_index - self.history_window
        return sum(
            1
            for index, rho, count in self.task_history.get(pool_id, ())
            if index >= horizon
            and count >= MIN_EVIDENCE_ROLLOUTS
            and (rho >= 1.0 if saturated else rho <= 0.0)
        )

    def is_saturated(self, pool_id: str) -> bool:
        """At least ``saturated_min_obs`` all-correct groups within the last ``history_window`` updates."""
        return (
            self.saturated_min_obs > 0
            and self._count_recent(pool_id, saturated=True) >= self.saturated_min_obs
        )

    def is_hopeless(self, pool_id: str) -> bool:
        """At least ``hopeless_min_obs`` all-wrong groups within the last ``history_window`` updates."""
        return (
            self.hopeless_min_obs > 0
            and self._count_recent(pool_id, saturated=False) >= self.hopeless_min_obs
        )

    def known_learnable_ids(self, dataset: str | None = None) -> list[str]:
        """Tasks whose last group (>= 2 rollouts) was mixed: some rollouts right, some wrong."""
        return [
            pool_id
            for pool_id, history in self.task_history.items()
            if history
            and history[-1][2] >= MIN_EVIDENCE_ROLLOUTS
            and 0.0 < history[-1][1] < 1.0
            and (dataset is None or self.pool.tasks[pool_id].dataset == dataset)
        ]

    def is_excluded(self, pool_id: str) -> bool:
        return self.is_saturated(pool_id) or self.is_hopeless(pool_id)

    def excluded_ids(self, dataset: str | None = None) -> set[str]:
        """Pool ids that are persistently saturated or hopeless (source-equivalent rows included)."""
        if not (self.saturated_min_obs or self.hopeless_min_obs):
            return set()
        if dataset in self._excluded_cache:
            return set(self._excluded_cache[dataset])
        excluded: set[str] = set()
        for pool_id in self.task_history:
            if dataset is not None and self.pool.tasks[pool_id].dataset != dataset:
                continue
            if self.is_excluded(pool_id):
                excluded.update(self._source_equivalents(pool_id))
        self._excluded_cache[dataset] = set(excluded)
        return excluded

    def filter_candidates(self, pool_ids: Sequence[str], *, minimum: int) -> list[str]:
        """Drop persistently saturated/hopeless ids, order kept, unless that leaves fewer than ``minimum``.

        A thin pool gets the excluded ids back (least selected first) up to ``minimum``.
        """
        ids = list(pool_ids)
        if not (self.saturated_min_obs or self.hopeless_min_obs) or not ids:
            return ids
        excluded = self.excluded_ids()
        kept = [pool_id for pool_id in ids if pool_id not in excluded]
        if len(kept) >= minimum:
            return kept
        returned = sorted((pool_id for pool_id in ids if pool_id in excluded), key=self._order_key)
        return kept + returned[: max(0, minimum - len(kept))]

    # ---- anchors -----------------------------------------------------------------------
    def anchor_ids(self, *, per_dataset: int, cycle: int, seed: int) -> list[str]:
        """Seeded anchors over the whole pool, ``per_dataset`` per dataset, datasets interleaved.

        For every slot a cluster is drawn with the cluster probability (which has a
        uniform floor under ``cluster_weighting="learnability"``), then the
        least-selected eligible task of that cluster (random among ties). Eligible means
        not recently selected, not reserved, not a source duplicate of an anchor already
        drawn and not persistently saturated/hopeless; the last three conditions are
        relaxed one by one when they would leave nothing. The result depends only on
        ``(seed, cycle)`` and the scheduler state, so a resumed cycle repeats it.
        """
        drawn: dict[str, list[str]] = {}
        for dataset in sorted(self.dataset_cluster_ids):
            rng = random.Random(f"anchors:{int(seed)}:{int(cycle)}:{dataset}")
            clusters = list(self.dataset_cluster_ids[dataset])
            weights = [max(self.cluster_states[cid].prob, 0.0) for cid in clusters]
            recent = self._recent_ids(dataset)
            skipped = self.excluded_ids(dataset)
            used: set[str] = set()
            picks: list[str] = []
            for _slot in range(per_dataset):
                pick = self._draw_anchor(rng, clusters, weights, recent, skipped, used)
                if pick is None:
                    break
                picks.append(pick)
                used.update(self._source_equivalents(pick))
            drawn[dataset] = picks
        return [
            drawn[dataset][slot]
            for slot in range(per_dataset)
            for dataset in sorted(drawn)
            if slot < len(drawn[dataset])
        ]

    def _draw_anchor(
        self,
        rng: random.Random,
        clusters: Sequence[str],
        weights: Sequence[float],
        recent: set[str],
        skipped: set[str],
        used: set[str],
    ) -> str | None:
        remaining = list(range(len(clusters)))
        while remaining:
            total = sum(weights[index] for index in remaining)
            if total <= 0:
                index = rng.choice(remaining)
            else:
                point, acc, index = rng.random() * total, 0.0, remaining[-1]
                for candidate in remaining:
                    acc += weights[candidate]
                    if point < acc:
                        index = candidate
                        break
            remaining.remove(index)
            members = self.cluster_to_sorted_ids[clusters[index]]
            for relaxed in (0, 1, 2):
                eligible = [
                    pool_id
                    for pool_id in members
                    if pool_id not in used
                    and (relaxed >= 2 or pool_id not in recent)
                    and (relaxed >= 1 or pool_id not in skipped)
                    and (relaxed >= 2 or pool_id not in self.reserved)
                ]
                if eligible:
                    # Coverage only (selection count / weight): the learnability tier is not used,
                    # or one found task would anchor every list and the neighbourhoods would repeat.
                    best = min(self._order_key(pool_id)[1] for pool_id in eligible)
                    return rng.choice([p for p in eligible if self._order_key(p)[1] == best])
        return None

    def _recent_ids(self, dataset: str | None) -> set[str]:
        if not self.cooldown:
            return set()
        selected: list[str] = []
        for pool_id in reversed(self.recent):
            if dataset is None or self.pool.tasks[pool_id].dataset == dataset:
                selected.append(pool_id)
                if len(selected) == self.cooldown:
                    break
        return {
            equivalent for pool_id in selected for equivalent in self._source_equivalents(pool_id)
        }

    def _source_equivalents(self, pool_id: str) -> set[str]:
        task = self.pool.tasks[pool_id]
        source_id = self._source_id(task)
        return self.source_to_pool_ids[(task.dataset, source_id)]

    @staticmethod
    def _source_id(task: PoolTask) -> str:
        # The source task identity, rather than the row id, defines whether two
        # rows are the same underlying task.
        return str(task.task.metadata.get("source_task_id") or task.pool_id)

    def _update_mini_positions(self, cluster_id: str, directions: dict[int, int]) -> None:
        state = self.cluster_states[cluster_id]
        size = state.cluster_size
        old_positions = sorted(state.mini_positions)
        if size <= len(old_positions):
            return
        new_positions: set[int] = set()
        movers: list[tuple[int, int]] = []
        for position in old_positions:
            direction = directions.get(position, 0)
            (
                new_positions.add(position)
                if direction == 0
                else movers.append((position, direction))
            )
        easier = sorted(position for position, direction in movers if direction < 0)
        harder = sorted((position for position, direction in movers if direction > 0), reverse=True)
        for position in easier:
            new_positions.add(self._nearest_free(position - 1, -1, new_positions, size, position))
        for position in harder:
            new_positions.add(self._nearest_free(position + 1, 1, new_positions, size, position))
        state.mini_positions = new_positions

    @staticmethod
    def _nearest_free(
        desired: int, direction: int, occupied: set[int], size: int, fallback: int
    ) -> int:
        desired = min(max(desired, 0), size - 1)
        if desired not in occupied:
            return desired
        for radius in range(1, size):
            for candidate in (desired + direction * radius, desired - direction * radius):
                if 0 <= candidate < size and candidate not in occupied:
                    return candidate
        return fallback

    def state_dict(self) -> dict[str, Any]:
        return {
            "cluster_states": {
                cluster_id: {
                    "success_rate": state.success_rate,
                    "prob": state.prob,
                    "mini_positions": sorted(state.mini_positions),
                    "learnability": state.learnability,
                    "observations": state.observations,
                }
                for cluster_id, state in self.cluster_states.items()
            },
            "task_success": self.task_success,
            "task_history": self.task_history,
            "update_index": self.update_index,
            "selection_count": dict(self.selection_count),
            "recent": self.recent,
            "reserved": sorted(self.reserved),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        for cluster_id, values in payload.get("cluster_states", {}).items():
            if cluster_id not in self.cluster_states:
                continue
            state = self.cluster_states[cluster_id]
            state.success_rate = float(values["success_rate"])
            state.prob = float(values["prob"])
            state.mini_positions = {int(value) for value in values["mini_positions"]}
            state.learnability = float(values.get("learnability", 1.0))
            state.observations = int(values.get("observations", 0))
        self.task_history = {
            str(key): [[float(a), float(b), float(c)] for a, b, c in value]
            for key, value in payload.get("task_history", {}).items()
        }
        self.update_index = int(payload.get("update_index", 0))
        self._excluded_cache = {}
        self.task_success = {
            str(key): float(value) for key, value in payload.get("task_success", {}).items()
        }
        self.selection_count = Counter(
            {str(key): int(value) for key, value in payload.get("selection_count", {}).items()}
        )
        self.recent = [str(value) for value in payload.get("recent", [])]
        self.reserved = {str(value) for value in payload.get("reserved", [])}
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = payload["rng_state"]


class TSDSRetriever:
    """KNN-KDE candidate distribution migrated from the TSDS implementation."""

    def __init__(
        self,
        pool: FixedTaskPool,
        *,
        max_k: int = 32,
        kde_k: int = 16,
        sigma: float = 0.1,
        alpha: float = 0.5,
        c: float = 1.0,
    ) -> None:
        self.pool = pool
        self.max_k = int(max_k)
        self.kde_k = int(kde_k)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.c = float(c)

    def rank(
        self,
        candidate_ids: Sequence[str],
        query_vectors: Sequence[Sequence[float]],
        *,
        limit: int,
    ) -> list[str]:
        ids = list(dict.fromkeys(candidate_ids))
        if not ids:
            return []
        if not query_vectors:
            return ids[:limit]
        candidates = np.asarray(
            [self.pool.tasks[pool_id].embedding for pool_id in ids], dtype=np.float32
        )
        queries = np.asarray(query_vectors, dtype=np.float32)
        probabilities = _tsds_probabilities(
            queries,
            candidates,
            max_k=min(self.max_k, len(ids)),
            kde_k=min(self.kde_k, len(ids)),
            sigma=self.sigma,
            alpha=self.alpha,
            c=self.c,
        )
        ranked = np.argsort(-probabilities, kind="stable")
        return [ids[int(index)] for index in ranked[:limit] if probabilities[int(index)] > 0]


def _tsds_probabilities(
    query: np.ndarray,
    candidates: np.ndarray,
    *,
    max_k: int,
    kde_k: int,
    sigma: float,
    alpha: float,
    c: float,
) -> np.ndarray:
    """TSDS probability assignment; only nearest-neighbour search is NumPy based."""

    max_k = max(1, min(int(max_k), len(candidates)))
    kde_k = max(1, min(int(kde_k), len(candidates)))
    squared = ((query[:, None, :] - candidates[None, :, :]) ** 2).sum(axis=-1)
    top_indices = np.argsort(squared, axis=-1)[:, :max_k]
    top_dists = np.sqrt(np.take_along_axis(squared, top_indices, axis=1))

    if sigma == 0:
        top_kdes = np.ones_like(top_dists)
    else:
        unique_indices = np.unique(top_indices)
        features = candidates[unique_indices]
        kde_squared = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.sort(kde_squared, axis=-1)[:, :kde_k]
        kernel = np.maximum(1.0 - nearest / (sigma**2), 0.0)
        kde = np.maximum(kernel.sum(axis=-1), 1e-12)
        kde_map = {int(index): float(kde[pos]) for pos, index in enumerate(unique_indices)}
        top_kdes = np.asarray(
            [[kde_map[int(index)] for index in row] for row in top_indices], dtype=np.float64
        )

    query_count, candidate_count = top_indices.shape[0], candidates.shape[0]
    if max_k == 1:
        probabilities = np.zeros(candidate_count)
        for row in top_indices:
            probabilities[int(row[0])] += 1.0 / query_count
        return probabilities / probabilities.sum()

    last_k = [0] * query_count
    heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(query_count)]
    heapq.heapify(heap)
    weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(query_count)]
    mass = 1.0
    costs = np.zeros(query_count)
    total_cost = 0.0
    while heap:
        count, current_k, current_query = heapq.heappop(heap)
        mass = count
        total_cost -= costs[current_query]
        costs[current_query] = (
            top_dists[current_query][current_k + 1] * count - weighted_sum[current_query]
        )
        total_cost += costs[current_query]
        if alpha / c * total_cost >= (1 - alpha) * query_count:
            break
        last_k[current_query] = current_k
        if current_k < max_k - 2:
            count += 1.0 / top_kdes[current_query][current_k + 1]
            heapq.heappush(heap, (count, current_k + 1, current_query))
            weighted_sum[current_query] += (
                top_dists[current_query][current_k + 1] / top_kdes[current_query][current_k + 1]
            )

    probabilities = np.zeros(candidate_count)
    for query_index in range(query_count):
        assigned = 0.0
        for neighbor in range(last_k[query_index] + 1):
            value = 1.0 / query_count / mass / top_kdes[query_index][neighbor]
            probabilities[top_indices[query_index][neighbor]] += value
            assigned += value
        remainder = max(1.0 / query_count - assigned, 0.0)
        probabilities[top_indices[query_index][last_k[query_index] + 1]] += remainder
    total = probabilities.sum()
    return probabilities / total if total > 0 else np.full(candidate_count, 1 / candidate_count)


def _task_prompt(row: dict[str, Any]) -> str:
    prompt = str(
        row.get("prompt", row.get("task", row.get("problem", row.get("question", ""))))
    ).strip()
    if not prompt:
        raise ValueError("fixed-pool row has no prompt")
    return prompt


def _task_reference(row: dict[str, Any]) -> Any:
    answers = row.get("target_answers")
    if isinstance(answers, list) and answers:
        return answers[0] if len(answers) == 1 else answers
    return row.get("reference", row.get("answer"))
