"""Versioned Director-only skills. SQLite is authoritative; collection views are immutable.

No model or environment is started by importing this module.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .skills import (
    SkillCard,
    _dot,
    _jaccard,
    _token_set,
)

SCHEMA = "director_skill"
_PATS_ACCOUNTING_FIELDS = ("pats_scope", "pats_snapshot_id", "selection_revision")


def _canonical_seed_skills_path(value: Any) -> str:
    """Keep the immutable collection contract stable after config rendering."""

    raw = str(value or "")
    parts = Path(raw).parts
    if "configs" in parts:
        return Path(*parts[parts.index("configs"):]).as_posix()
    return raw


def _pats_accounting_metadata(manifest: dict[str, Any]) -> dict[str, str]:
    # Only the original collection manifest can identify a historical scoped card.
    # Never infer missing scope/snapshot metadata from the current PATS state.
    if manifest.get("pats_enabled") is not True:
        return {}
    return {
        key: manifest[key]
        for key in _PATS_ACCOUNTING_FIELDS
        if isinstance(manifest.get(key), str) and manifest[key]
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class SkillStore:
    def __init__(
        self,
        path: Path | str,
        *,
        seeds_path: Path | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.seeds_path = Path(seeds_path) if seeds_path else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cards(
                    id TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(id, version));
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage_outcomes(
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (SCHEMA,))
            if db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] != SCHEMA:
                raise ValueError("skill store schema mismatch")

    def connect(self):
        # One connection per transaction.
        @contextlib.contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            try:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    yield db
            finally:
                db.close()

        return connection()

    def initialize_seeds(self) -> None:
        if self.seeds_path is None:
            raise ValueError("[pats].seed_skills_path must name the seed skill cards")
        with self.connect() as db:
            for item in json.loads(self.seeds_path.read_text(encoding="utf-8")):
                card = SkillCard.from_dict(item["card"])
                card.validate()
                if db.execute(
                    "SELECT 1 FROM cards WHERE id=? AND version=1", (card.skill_id,)
                ).fetchone():
                    continue
                payload = dict(
                    item,
                    version=1,
                    provenance="human_seed",
                    parent_version=None,
                    source_cases=[],
                )
                db.execute(
                    "INSERT INTO cards VALUES(?,1,'seed',?)",
                    (card.skill_id, json.dumps(payload, ensure_ascii=False)),
                )
            db.execute("INSERT OR IGNORE INTO meta VALUES('seeds_initialized','true')")

    def cards(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(json.loads(row["payload"]), status=row["status"])
                for row in db.execute("SELECT * FROM cards ORDER BY id,version")
            ]

    def snapshot(self, destination: Path) -> dict[str, Any]:
        if destination.exists():
            return json.loads(destination.read_text(encoding="utf-8"))
        # Single read transaction gives a complete version even during publication.
        with self.connect() as db:
            cards = [
                dict(json.loads(row["payload"]), status=row["status"])
                for row in db.execute(
                    "SELECT * FROM cards WHERE status IN ('seed','active') ORDER BY id"
                )
            ]
            value = {"schema": SCHEMA, "snapshot_id": uuid.uuid4().hex, "cards": cards}
        atomic_json(destination, value)
        return value

    def record_event(self, event: dict[str, Any]) -> None:
        key = json.dumps([event[k] for k in ("run", "cycle", "rollout", "skill", "version")])
        with self.connect() as db:
            previous = db.execute("SELECT payload FROM events WHERE id=?", (key,)).fetchone()
            payload = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if previous and previous[0] != payload:
                raise ValueError("conflicting repeated skill outcome event")
            db.execute("INSERT OR IGNORE INTO events VALUES(?,?)", (key, payload))

    def record_usage_outcome(self, event: dict[str, Any]) -> None:
        key = json.dumps([event[k] for k in ("run", "cycle", "rollout", "skill", "version")])
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.connect() as db:
            old = db.execute("SELECT payload FROM usage_outcomes WHERE id=?", (key,)).fetchone()
            if old and old[0] != payload:
                raise ValueError("conflicting repeated skill usage outcome")
            db.execute("INSERT OR IGNORE INTO usage_outcomes VALUES(?,?)", (key, payload))

    def usage_summary(self) -> list[dict[str, Any]]:
        # Rebuild from idempotent records; rereading a cycle never increments counters.
        groups = {}
        with self.connect() as db:
            events = [
                json.loads(row[0]) for row in db.execute("SELECT payload FROM usage_outcomes")
            ]
        for event in events:
            key = (event["skill"], event["version"], event["dataset"], event.get("pats_scope"))
            record = groups.setdefault(
                key,
                dict(
                    skill_id=key[0],
                    version=key[1],
                    dataset=key[2],
                    mode=event["mode"],
                    usage_count=0,
                    valid_count=0,
                    reward_sum=0.0,
                    helpful_count=0,
                    hurt_count=0,
                    infrastructure_failure_count=0,
                    unscored_count=0,
                    skipped_feedback_count=0,
                    **({"pats_scope": key[3]} if key[3] is not None else {}),
                ),
            )
            record["usage_count"] += 1
            if event["infrastructure_failure"]:
                record["infrastructure_failure_count"] += 1
                continue
            if event["reward"] is None:
                record["unscored_count"] += 1
                continue
            record["valid_count"] += 1
            record["reward_sum"] += event["reward"]
            if event["feedback"] == "helpful":
                record["helpful_count"] += 1
            elif event["feedback"] == "hurt":
                record["hurt_count"] += 1
            else:
                record["skipped_feedback_count"] += 1
        for record in groups.values():
            record["mean_reward"] = (
                record["reward_sum"] / record["valid_count"] if record["valid_count"] else None
            )
            record["net_score"] = record["helpful_count"] - record["hurt_count"]
        return sorted(
            groups.values(),
            key=lambda r: (r["skill_id"], r["version"], r["dataset"], r.get("pats_scope", "")),
        )


def render_skill_context(entries) -> str:
    """Director SkillBank prompt context of ``(card, version)`` pairs, in order ("" when empty).

    ``DirectorSkillBank.format_prompt_context`` renders its retrieved cards with it; the
    ``[r2orch.arms]`` audit renders the retrieved cards plus one ledger-retired card (which
    the scoped view no longer holds) with the same format.
    """
    if not entries:
        return ""
    return "## Director SkillBank — optional orchestration guidance\n" + "\n\n".join(
        f"[{card.skill_id}@{version}] {card.name}\n"
        f"When: {card.trigger}\nPlan: {card.plan}\nPitfall: {card.pitfall}\nConstraint: {card.constraint}"
        for card, version in entries
    )


class DirectorSkillBank:
    """Read-only, scoped view. No per-rollout mutable outcome counters."""

    _vector_cache: dict[tuple[str, str], tuple[float, ...]] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        snapshot: dict[str, Any],
        *,
        embedder=None,
        top_k=3,
        prompt_token_budget=1024,
        retrieval_min_score=None,
        selection_revision="e5_only",
    ):
        if snapshot.get("schema") != SCHEMA:
            raise ValueError("expected a frozen director_skill snapshot")
        self.snapshot_id = snapshot["snapshot_id"]
        self.records = {
            p["card"]["skill_id"]: p for p in snapshot["cards"] if p["status"] in {"seed", "active"}
        }
        self.skills = {key: SkillCard.from_dict(p["card"]) for key, p in self.records.items()}
        self.embedder = embedder
        self.retrieve_top_k = min(3, top_k)
        self.prompt_token_budget = prompt_token_budget
        self.retrieval_min_score = retrieval_min_score
        self._pats = snapshot.get("pats")
        _validate_semantic_snapshot(self._pats)
        self._selection_revision = (
            self._pats["selection_revision"]
            if self._pats is not None
            else selection_revision
        )
        if self._selection_revision not in {"e5_only", "learned_first"}:
            raise ValueError(f"unknown PATS selection revision: {self._selection_revision!r}")
        self._pats_views: dict[str, DirectorSkillBank] = {}
        self._embeddings = {}
        if embedder:
            namespace = str(getattr(embedder, "model_path", type(embedder).__qualname__))
            with self._cache_lock:
                texts = {
                    key: f"{card.description} || {card.trigger} || {card.plan}"
                    for key, card in self.skills.items()
                }
                missing = list(
                    dict.fromkeys(
                        text
                        for text in texts.values()
                        if (namespace, text) not in self._vector_cache
                    )
                )
                if missing:
                    vectors = embedder.encode(missing, query=False)
                    for text, vector in zip(missing, vectors, strict=True):
                        self._vector_cache[namespace, text] = vector
                self._embeddings = {
                    key: self._vector_cache[namespace, text] for key, text in texts.items()
                }
                while len(self._vector_cache) > 2400:
                    self._vector_cache.pop(next(iter(self._vector_cache)))

    def _rank_candidates(self, query, *, task_type="", tools=()):
        if not query.strip():
            return []
        vector = self.embedder.encode([query], query=True)[0] if self.embedder else None
        scores = []
        for key, card in self.skills.items():
            record = self.records[key]
            if (
                card.task_types
                and task_type not in card.task_types
                or task_type in record.get("excluded_task_types", [])
                or not set(record.get("required_tools", [])) <= set(tools)
            ):
                continue
            score = (
                _dot(vector, self._embeddings[key])
                if vector is not None
                else _jaccard(
                    _token_set(query),
                    _token_set(card.description + " " + card.trigger + " " + card.plan),
                )
            )
            # No universal E5 cutoff is assumed. Optional cutoff must be calibrated;
            # without one, nonpositive/no lexical overlap can explicitly yield NONE.
            if (
                score <= 0
                or self.retrieval_min_score is not None
                and score < self.retrieval_min_score
            ):
                continue
            scores.append((score, key, card))
        if self._selection_revision == "learned_first":
            # Scoped ADD and UPDATE cards carry this provenance, including updated
            # seed IDs. Eligibility and calibrated score cutoffs still apply first.
            scores.sort(
                key=lambda row: (
                    self.records[row[1]].get("provenance") != "pats_scoped_unvalidated",
                    -row[0],
                    row[1],
                )
            )
        else:
            scores.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scores]

    def retrieve(self, query, *, task_type="", top_k=None, tools=()):
        candidates = self._rank_candidates(query, task_type=task_type, tools=tools)
        return candidates[: min(self.retrieve_top_k, top_k if top_k is not None else 3)]

    def format_prompt_context(self, selected):
        return render_skill_context(
            [(card, self.records[card.skill_id]["version"]) for card in selected]
        )

    def select_context(self, query, *, task_type, tokenizer, tools=(), task_metadata=None):
        if self._pats is not None:
            from .pats import resolve_scope

            scope = resolve_scope(task_type, task_metadata)
            if scope not in self._pats_views:
                scoped = self._pats.get("scopes", {}).get(scope)
                # An explicitly empty view represents withdrawal, not a retrieval miss.
                records = scoped["cards"] if scoped is not None else list(self.records.values())
                self._pats_views[scope] = DirectorSkillBank(
                    {"schema": SCHEMA, "snapshot_id": self.snapshot_id, "cards": records},
                    embedder=self.embedder,
                    top_k=self.retrieve_top_k,
                    prompt_token_budget=self.prompt_token_budget,
                    retrieval_min_score=self.retrieval_min_score,
                    selection_revision=self._selection_revision,
                )
            selected, context, manifest = self._pats_views[scope].select_context(
                query,
                task_type=task_type,
                tokenizer=tokenizer,
                tools=tools,
            )
            import hashlib

            manifest.update(
                pats_enabled=True,
                pats_scope=scope,
                pats_snapshot_id=self._pats["snapshot_id"],
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
            )
            manifest.update(
                semantic_gate_revision=self._pats["semantic_gate_revision"],
                semantic_contract_sha256=self._pats["semantic_contract_sha256_by_dataset"].get(
                    json.loads(scope)[0], self._pats["semantic_default_contract_sha256"]
                ),
            )
            return selected, context, manifest
        if tokenizer is None:
            raise ValueError("skill token budget requires the actual Director tokenizer")
        chosen = []
        candidates = (
            self._rank_candidates(query, task_type=task_type, tools=tools)
            if self._selection_revision == "learned_first"
            else self.retrieve(query, task_type=task_type, tools=tools)
        )
        for card in candidates:
            if (
                self._selection_revision == "learned_first"
                and len(chosen) >= self.retrieve_top_k
            ):
                break
            context = self.format_prompt_context([*chosen, card])
            if len(tokenizer.encode(context, add_special_tokens=False)) <= self.prompt_token_budget:
                chosen.append(card)
        context = self.format_prompt_context(chosen)
        manifest = {
            "schema": SCHEMA,
            "snapshot_id": self.snapshot_id,
            "selected": [
                {"id": c.skill_id, "version": self.records[c.skill_id]["version"]} for c in chosen
            ],
            "prompt_tokens": len(tokenizer.encode(context, add_special_tokens=False))
            if context
            else 0,
            "context": context,
        }
        # Only the learned_first selector records its revision in the manifest.
        if self._selection_revision == "learned_first":
            manifest["selection_revision"] = self._selection_revision
        return chosen, context, manifest


def _seed_skills_path(config) -> Path | None:
    raw = config.pats.seed_skills_path
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    # Configured relative to the repository root (configs/skills/...).
    return Path(__file__).resolve().parents[2] / path


def store_for_config(config):
    return SkillStore(
        config.skill_cases_path.with_suffix(".sqlite3"),
        seeds_path=_seed_skills_path(config),
    )


def pats_controller(config, store, token_counter):
    """PatsController with the R2Orch scope list."""
    from .pats import PatsController
    from .skill_ledger import SkillLedger, skill_ledger_path

    pats = config.pats
    r2orch = config.r2orch
    # [r2orch.arms]: the end-of-cycle card lifecycle reads the run's skill ledger.
    skill_ledger = SkillLedger.from_arms(skill_ledger_path(r2orch), r2orch.arms, base_seed=int(r2orch.base_seed))
    return PatsController(
        store,
        pats,
        token_counter,
        # R2OrchConfig.scopes: the six (dataset, task_type) scope keys.
        scopes=tuple(r2orch.scopes),
        skill_ledger=skill_ledger,
    )


def _validate_semantic_snapshot(pats_view):
    if pats_view is None:
        return
    from .pats_semantics import SEMANTIC_REVISION

    if pats_view["semantic_gate_revision"] != SEMANTIC_REVISION:
        raise ValueError("unknown PATS semantic gate revision in frozen collection")
    digest = pats_view.get("semantic_contract_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("frozen PATS semantic gate lacks its contract hash")


def freeze_collection(config, cycle_dir: Path, *, step: int | None = None):
    from .pats import config_payload

    enabled = config.skillbank_enabled
    pats_enabled = config.pats.enabled
    contract_path = cycle_dir / "skill_context_contract.json"
    if pats_enabled or contract_path.exists():
        contract = {
            "schema": "skill_context_contract",
            "context_enabled": bool(enabled),
            "pats_enabled": pats_enabled,
            "pats_config": config_payload(config.pats),
            "prompt_token_budget": config.skillbank_prompt_token_budget,
            "retrieve_top_k": config.skillbank_retrieve_top_k,
            "retrieval_min_score": config.skillbank_retrieval_min_score,
            "embedding_model_path": str(config.skillbank_embedding_model_path),
        }
        contract["pats_config"]["seed_skills_path"] = _canonical_seed_skills_path(
            contract["pats_config"].get("seed_skills_path")
        )
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise ValueError(
                    "collection skill context contract changed; use a fresh output cycle"
                )
        elif (cycle_dir / "solver_rollouts.jsonl").exists():
            raise ValueError(
                "existing PATS rollouts lack a context contract; use a fresh output cycle"
            )
        else:
            atomic_json(contract_path, contract)
    if not enabled:
        return config
    path = cycle_dir / "director_skill_snapshot.json"
    if not path.exists() and (cycle_dir / "solver_rollouts.jsonl").exists():
        raise ValueError(
            "existing rollouts lack a skill snapshot; use a fresh output cycle"
        )
    store = store_for_config(config)
    store.initialize_seeds()
    with contextlib.ExitStack() as stack:
        pats_view = None
        if pats_enabled:
            # Serialize with normal maintenance through the final snapshot write.
            lock = stack.enter_context(store.path.with_suffix(".pats.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            controller = pats_controller(config, store, len)
            pats_view = controller.snapshot(
                run=str(cycle_dir.parent.resolve()),
                next_step=step if not path.exists() else None,
            )
        if path.exists():
            _validate_pats_snapshot(config, json.loads(path.read_text(encoding="utf-8")))
        if not path.exists() and pats_enabled:
            # Existing snapshots are never rewritten, including on resume. The
            # token counter is unused by this read-only state snapshot.
            snapshot = {
                "schema": SCHEMA,
                "snapshot_id": uuid.uuid4().hex,
                "collection_frozen": True,
                "cards": [r for r in store.cards() if r["status"] in {"seed", "active"}],
                "pats": pats_view,
            }
            atomic_json(path, snapshot)
        else:
            store.snapshot(path)
    return replace(config, skillbank_path=path)


def _validate_pats_snapshot(config, snapshot):
    from .pats import config_matches

    enabled = config.pats.enabled
    frozen = snapshot.get("pats")
    if enabled != (frozen is not None):
        raise ValueError(
            "frozen collection PATS mode differs from configuration; use a fresh cycle"
        )
    if enabled and not config_matches(frozen.get("config"), config.pats):
        raise ValueError(
            "frozen collection PATS configuration differs; resume original configuration"
        )
    if enabled and frozen.get("selection_revision") not in {
        "e5_only",
        "learned_first",
    }:
        raise ValueError("unknown PATS selection revision in frozen collection")
    _validate_semantic_snapshot(frozen)
    if enabled and frozen.get("semantic_contract_sha256") != config.pats.semantic_contract_hash():
        raise ValueError("frozen PATS runtime contract changed; use a fresh collection")


def load_bank(config, *, embedder=None):
    """The skill bank of one collection: its frozen snapshot (``freeze_collection``)."""
    snapshot = json.loads(config.skillbank_path.read_text(encoding="utf-8"))
    _validate_pats_snapshot(config, snapshot)
    return DirectorSkillBank(
        snapshot,
        embedder=embedder,
        top_k=config.skillbank_retrieve_top_k,
        prompt_token_budget=config.skillbank_prompt_token_budget,
        retrieval_min_score=config.skillbank_retrieval_min_score,
    )


def ingest_rollouts(store: SkillStore, tasks, rows, *, run: str, step: int):
    for row in rows:
        task = tasks.get(row.get("task_id"))
        if task and (
            task.metadata.get("skill_evaluation_split") == "final_test"
            or task.metadata.get("is_final_test") is True
        ):
            continue
        metadata = row.get("metadata", {})
        reason = " ".join(
            str(metadata.get(key, ""))
            for key in (
                "failure_mode",
                "training_exclusion_reasons",
                "error_type",
                "error",
                "terminal_reason",
            )
        )
        infra = any(
            bool(metadata.get(key))
            for key in (
                "infrastructure_failure",
                "worker_backend_failure",
                "worker_artifact_integrity_failure",
                "unresolved_tool_failure",
            )
        ) or bool(
            re.search(
                r"api[_ -]?error|backend|route|upstream|connection|judge.*fail|environment.*timeout|attribution_unresolved",
                reason,
                re.I,
            )
        )
        known = metadata.get("reward_known") is True
        try:
            reward = float(row["reward"])
        except (KeyError, TypeError, ValueError, OverflowError):
            reward = 0.0
            known = False
        known = known and math.isfinite(reward)
        manifest = metadata.get("skill_context", {})
        pats_metadata = _pats_accounting_metadata(manifest)
        for used in manifest.get("selected", []):
            store.record_event(
                {
                    "run": run,
                    "cycle": step,
                    "rollout": row["rollout_id"],
                    "skill": used["id"],
                    "version": used["version"],
                    "retrieved": True,
                    "observed_following": None,
                    "task_reward": reward if known else None,
                    "infrastructure_failure": infra,
                    "snapshot_id": manifest.get("snapshot_id"),
                    "dataset": str(tasks[row["task_id"]].metadata.get("dataset", ""))
                    if row["task_id"] in tasks
                    else "",
                    "policy_stage": step,
                    **pats_metadata,
                }
            )
        # Keep these outcome records separate from the observation events above.
        # No answer is copied into the usage ledger; zero-score feedback requires
        # a substantive deliverable, analogous to SESA's successful extraction gate.
        dataset = str(task.metadata.get("dataset", task.task_type)) if task else "unknown"
        answer = metadata.get("solver_answer", "")
        substantive = bool(str(answer).strip()) if answer is not None else False
        format_failure = bool(metadata.get("extraction_failed")) or bool(
            re.search(
                r"extraction_failed|format_failure|protocol_failure|typed_policy_failure",
                reason,
                re.I,
            )
        )
        feedback = None
        if known and not infra:
            if reward > 0:
                feedback = "helpful"
            elif reward == 0 and substantive and not format_failure:
                feedback = "hurt"
        for used in manifest.get("selected", []):
            store.record_usage_outcome(
                dict(
                    run=run,
                    cycle=step,
                    rollout=row["rollout_id"],
                    skill=used["id"],
                    version=used["version"],
                    dataset=dataset,
                    mode="sesa_outcome",
                    reward=reward if known else None,
                    infrastructure_failure=infra,
                    feedback=feedback,
                    substantive_output=substantive and not format_failure,
                    **pats_metadata,
                )
            )


def consolidate(config, result, *, step: int, mock: bool, cycle_dir: Path | None, tokenizer=None):
    if not config.skillbank_enabled:
        return
    store = store_for_config(config)
    store.initialize_seeds()
    if cycle_dir is None:
        # Raw evidence is essential: silently falling back to PPO-filtered samples is wrong.
        return
    source = cycle_dir / "solver_rollouts.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if config.pats.enabled:
        from .pats import apply_metadata_updates

        updates = cycle_dir / "rollout_metadata_updates.jsonl"
        if updates.exists():
            rows = apply_metadata_updates(
                rows,
                [json.loads(line) for line in updates.read_text().splitlines() if line.strip()],
            )

        if tokenizer is None:
            from .selfplay_runtime import ByteTokenizer, HuggingFaceTokenizer

            tokenizer = (
                ByteTokenizer()
                if mock
                else HuggingFaceTokenizer(config.solver_model.base_model_path)
            )
        if not mock and getattr(tokenizer, "approximate_token_count", False):
            raise ValueError("PATS real review requires the actual Director tokenizer")
        controller = pats_controller(
            config, store, lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        )
        backend = None
        try:
            if not mock:
                from .application import _create_runtime_backend

                backend = _create_runtime_backend(
                    config.runtime_pool()[config.skill_distiller_runtime],
                    route_name=config.skill_distiller_runtime,
                )
            receipt = controller.maintain(
                {task.task_id: task for task in result.tasks},
                rows,
                run=str(cycle_dir.parent.resolve()),
                step=step,
                policy_snapshot=result.snapshots.get("solver"),
                backend=backend,
                mock=mock,
            )
            atomic_json(cycle_dir / "pats_review.json", receipt)
            if isinstance(receipt.get("skill_ledger"), dict):
                # [r2orch.arms]: per-card posterior, decision and status after this cycle.
                atomic_json(cycle_dir / "skill_ledger_summary.json", receipt["skill_ledger"])
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    ingest_rollouts(
        store,
        {task.task_id: task for task in result.tasks},
        rows,
        run=str(cycle_dir.parent.resolve()),
        step=step,
    )
    atomic_json(cycle_dir / "skill_usage_summary.json", store.usage_summary())
