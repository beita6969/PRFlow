"""Policy-aware Director scaffolding, adapted from PATS (Apache-2.0).

Adapted from shi-yipeng/PATS (Apache-2.0). The grouped rollout evidence, the rule checker
and the bounded atomic edits follow the official ``experience_bank`` / evidence-bank design
at upstream commit bad468b5c73081c2f5aa74c4e0011c6fb2872dbf. This integration uses existing
Director cards, audited task rewards (continuous where applicable), strict
atomic edit validation, actual tokenizer counts and immutable scoped views.

Once per cycle, after the decisions of the skill ledger (withdraw, admit, restore, discard), an
LLM editor runs two steps on every reviewed scope: REVISE updates or deletes the active card with
the lowest posterior mean effect, and DISTILL adds cards from the task groups in which both parts
fail. Every card an edit adds or rewrites enters the scaffold on trial under a new ledger key.
See THIRD_PARTY_NOTICES.md and third_party/notices for attribution. No PPO data is edited.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name
from .pats_evidence import (
    card_gaps,
    condition_contrast,
    graph_structure,
    public_process_trace,
    public_text,
    selected_skill_context,
    unhelpful_cards,
    within_condition,
)
from .skill_ledger import RETIRED_VIEW_KEY as LEDGER_RETIRED_VIEW_KEY
from .skill_ledger import SUMMARY_SCHEMA as LEDGER_SUMMARY_SCHEMA
from .skill_ledger import record_card_key
from .skills import (
    SKILL_KINDS,
    SkillCard,
    SkillStats,
    SolverFailureCase,
    _jaccard,
    _public_failure_case,
    _token_set,
)

SCHEMA = "policy_aware_director_scaffold"
SCOPE_SCHEMA = "r2orch_scope@2"
CARD_FIELDS = ("name", "description", "trigger", "plan", "pitfall", "constraint", "kind")
SCOPE_FIELDS = ("schema", "groups_seen", "card_count", "card_tokens", "archive_count")
# Copied from PATS experience_bank.VAGUE_PHRASE_BLOCKLIST (Apache-2.0).
VAGUE_PHRASE_BLOCKLIST = frozenset(
    {
        "observe the environment",
        "observe your surroundings",
        "check your surroundings",
        "check the surroundings",
        "look around",
        "explore the environment",
        "retry carefully",
        "try again",
        "be careful",
        "pay attention",
        "think step by step",
    }
)
_PROMPT_ECHO_MARKERS = (
    "<action>",
    "director skillbank",
    "optional orchestration knowledge",
    "[task skills]",
    "[experience bank",
)


@dataclass(frozen=True)
class PatsConfig:
    enabled: bool = False
    director_prompt_variant: str = "r2orch"
    director_prompt_variant_by_dataset: dict[str, str] = field(default_factory=dict)
    max_skills: int = 30
    max_tokens: int = 2000
    min_groups: int = 2
    min_evidence_cards: int = 2
    review_interval: int = 1
    max_edits: int = 4
    max_reviews_per_cycle: int = 2
    max_evidence_groups: int = 32
    max_policy_lag: int = 1
    max_review_input_tokens: int = 20480
    # Up to two editor calls (REVISE, DISTILL) for each of the max_reviews_per_cycle scopes.
    max_checker_calls_per_cycle: int = 6
    seed_skills_path: str = ""
    field_char_limit: int = 300
    dedup_jaccard: float = 0.7

    def validate(self) -> None:
        from .director import DIRECTOR_PROMPT_VARIANTS

        if self.director_prompt_variant not in DIRECTOR_PROMPT_VARIANTS:
            raise ValueError("unknown PATS Director prompt variant")
        if any(not canonical_dataset_name(dataset) or variant not in DIRECTOR_PROMPT_VARIANTS
               for dataset, variant in self.director_prompt_variant_by_dataset.items()):
            raise ValueError("unknown dataset PATS Director prompt variant")
        if type(self.enabled) is not bool:
            raise ValueError("PATS enabled must be boolean")
        for name in (
            "max_skills",
            "max_tokens",
            "min_groups",
            "min_evidence_cards",
            "review_interval",
            "max_edits",
            "max_reviews_per_cycle",
            "max_evidence_groups",
            "max_review_input_tokens",
            "max_checker_calls_per_cycle",
            "field_char_limit",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"PATS {name} must be a positive integer")
        if self.min_groups < 2 or self.min_evidence_cards < 2:
            raise ValueError("PATS requires at least two distinct evidence groups")
        if self.max_evidence_groups < max(self.min_groups, self.min_evidence_cards):
            raise ValueError("PATS evidence capacity is smaller than admission requirements")
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise ValueError("PATS max_policy_lag must be a nonnegative integer")

    def variant_for_scope(self, scope: str) -> str:
        if not self.director_prompt_variant_by_dataset:
            return self.director_prompt_variant
        dataset = canonical_dataset_name(json.loads(scope)[0])
        return self.director_prompt_variant_by_dataset.get(dataset, self.director_prompt_variant)

    def for_scope(self, scope: str) -> PatsConfig:
        return replace(self, director_prompt_variant=self.variant_for_scope(scope),
                       director_prompt_variant_by_dataset={})

    def contract_hashes_by_dataset(self) -> dict[str, str]:
        from .pats_semantics import contract_hash
        return {dataset: contract_hash(variant)
                for dataset, variant in sorted(self.director_prompt_variant_by_dataset.items())}

    def semantic_contract_hash(self) -> str:
        from .pats_semantics import contract_hash
        default = contract_hash(self.director_prompt_variant)
        scoped = self.contract_hashes_by_dataset()
        return _digest({"default": default, "by_dataset": scoped}) if scoped else default


def _canonical_seed_skills_path(value: Any) -> str:
    raw = str(value or "")
    parts = Path(raw).parts
    if "configs" in parts:
        return Path(*parts[parts.index("configs"):]).as_posix()
    return raw


def _comparable_config(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    if "seed_skills_path" in value:
        value["seed_skills_path"] = _canonical_seed_skills_path(value["seed_skills_path"])
    return value


def config_payload(config: PatsConfig) -> dict[str, Any]:
    """Persisted/hashed configuration."""
    return _comparable_config(asdict(config))


def config_matches(stored: Any, config: PatsConfig) -> bool:
    """A stored PATS configuration equals ``config``."""
    return isinstance(stored, dict) and _comparable_config(stored) == config_payload(config)


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def resolve_scope(task_type: str, task_metadata: dict[str, Any] | None = None) -> str:
    """Only public, fixed task metadata defines a scope; never use observed reward."""
    metadata = task_metadata or {}
    dataset = canonical_dataset_name(str(metadata.get("dataset") or task_type or "general"))
    difficulty = metadata.get("difficulty_bucket", metadata.get("difficulty", "unspecified"))
    if not isinstance(difficulty, (str, int, float)) or isinstance(difficulty, bool):
        difficulty = "unspecified"
    if isinstance(difficulty, float) and not math.isfinite(difficulty):
        difficulty = "unspecified"
    return _json([dataset, str(task_type or "general"), str(difficulty)])


@dataclass(frozen=True)
class ScopeState:
    """The size of one scope's scaffold in the frozen collection snapshot."""

    scope: str
    dataset: str
    task_type: str
    card_count: int
    card_tokens: int


def scope_view(skill_snapshot: Mapping[str, Any]) -> dict[str, ScopeState]:
    """Read the frozen collection snapshot; scopes absent from it are absent here."""
    scopes = (skill_snapshot.get("pats") or {}).get("scopes", {})
    view = {}
    for key, value in sorted(scopes.items()):
        dataset, task_type = json.loads(key)[:2]
        view[key] = ScopeState(
            scope=key,
            dataset=dataset,
            task_type=task_type,
            card_count=int(value.get("card_count", len(value.get("cards", [])))),
            card_tokens=int(value.get("card_tokens", 0)),
        )
    return view


def render_cards(records: list[dict[str, Any]]) -> str:
    """Render cards in the text format of ``skill_store.render_skill_context``."""
    if not records:
        return ""
    return "## Director SkillBank — optional orchestration guidance\n" + "\n\n".join(
        f"[{r['card']['skill_id']}@{r['version']}] {r['card']['name']}\n"
        f"When: {r['card']['trigger']}\nPlan: {r['card']['plan']}\n"
        f"Pitfall: {r['card']['pitfall']}\nConstraint: {r['card']['constraint']}"
        for r in records
    )


def _credible(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    if metadata.get("reward_known") is not True:
        return False
    if any(
        metadata.get(key)
        for key in (
            "infrastructure_failure",
            "worker_backend_failure",
            "worker_artifact_integrity_failure",
            "unresolved_tool_failure",
            "uncertain_attribution_zero",
        )
    ):
        return False
    if (
        "frozen_policy_probability_mismatch" in metadata.get("training_exclusion_reasons", [])
        or metadata.get("reward_admission_reason") == "uncertain_attribution_zero"
        or metadata.get("terminal_graph_status") == "unsafe_partial"
    ):
        return False
    reason = " ".join(
        str(metadata.get(k, ""))
        for k in (
            "failure_mode",
            "training_exclusion_reasons",
            "error_type",
            "error",
            "terminal_reason",
        )
    )
    if re.search(
        r"api[_ -]?error|backend|route|upstream|connection|judge.*fail|"
        r"environment.*timeout|attribution_unresolved",
        reason,
        re.I,
    ):
        return False
    try:
        reward = float(row["reward"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(reward) and 0 <= reward <= 1 and metadata.get("task_reward", reward) == reward
    )


def apply_metadata_updates(
    rows: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the collector's authoritative post-collection audits without rewriting raw logs."""
    by_id = {}
    for row in rows:
        key = str(row.get("rollout_id", ""))
        if key in by_id and by_id[key] != row:
            raise ValueError("conflicting duplicate PATS rollout")
        by_id[key] = copy.deepcopy(row)
    for update in updates:
        key = str(update.get("rollout_id", ""))
        if key in by_id and "metadata" in update:
            if not isinstance(update["metadata"], dict):
                raise ValueError("PATS rollout metadata audit must be an object")
            by_id[key]["metadata"] = copy.deepcopy(update["metadata"])
    return list(by_id.values())


def _credible_groups(tasks: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, str] = {}
    for row in rows:
        task = tasks.get(row.get("task_id"))
        if (
            task is None
            or task.metadata.get("is_final_test") is True
            or (task.metadata.get("skill_evaluation_split") == "final_test")
        ):
            continue
        identity = str(row.get("rollout_id", ""))
        if not identity:
            raise ValueError("PATS evidence requires rollout identity")
        signature = _digest(row)
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError("conflicting duplicate PATS rollout")
            continue
        seen[identity] = signature
        if _credible(row):
            groups[str(row["task_id"])].append(row)
    return groups


def _attempt(task: Any, row: dict[str, Any], skill_context: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    public = _public_failure_case(
        SolverFailureCase(
            task=task.prompt,
            task_type=task.task_type,
            failure_trace=json.dumps(metadata.get("solver_trace", {})),
            failure_mode=str(metadata.get("failure_mode", "trusted_outcome")),
            solver_answer="",
            used_skill_ids=tuple(item["id"] for item in skill_context["selected"]),
        )
    )
    return {
        "rollout_id": str(row["rollout_id"]),
        "reward": float(row["reward"]),
        "failure_mode": public["failure_mode"],
        "trace": public_process_trace(public["failure_trace"]),
    }


def _representatives(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Use every sibling for contrast. Representatives favor the longest
    # low-reward attempt and shortest high-reward attempt, with stable ties.
    ordered = sorted(
        attempts,
        key=lambda item: (
            item["reward"],
            -item["trace"]["summary"]["steps"],
            item["rollout_id"],
        ),
    )
    return list({item["rollout_id"]: item for item in (ordered[0], ordered[-1])}.values())


def scaffold_condition(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("r2orch", {}).get("scaffold_condition", "on"))


def _arm_evidence(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Hidden / visible outcomes of a ``[r2orch.arms]`` task group (None without arm rollouts)."""
    armed = [
        (float(row["reward"]), row.get("metadata", {}).get("r2orch") or {})
        for row in group
        if (row.get("metadata", {}).get("r2orch") or {}).get("skill_arm") in ("hidden", "visible")
    ]
    if not armed:
        return None
    hidden = [reward for reward, meta in armed if meta["skill_arm"] == "hidden"]
    visible = [reward for reward, meta in armed if meta["skill_arm"] == "visible"]
    keys = sorted({str((meta.get("arm_card") or {}).get("key", "")) for _reward, meta in armed})
    return {
        "card": keys[0] if len(keys) == 1 else None,
        "hidden": {"n": len(hidden), "successes": sum(hidden)},
        "visible": {"n": len(visible), "successes": sum(visible)},
        "both_failed": bool(hidden and visible and not any(hidden) and not any(visible)),
    }


def collect_evidence(
    tasks: dict[str, Any], rows: list[dict[str, Any]], *, run: str, step: int, policy_snapshot: str
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """R2Orch: a task group mixes scaffold conditions by design.

    Each task yields within-condition contrasts (success against failure graphs)
    and a cross-condition contrast (the same task with and without cards).
    Returns the evidence and the admitted credible rows of each scope.
    """
    evidence, admitted = [], defaultdict(list)
    for task_id, group in sorted(_credible_groups(tasks, rows).items()):
        task = tasks[task_id]
        policies = {
            str(r.get("metadata", {}).get("solver_snapshot", policy_snapshot)) for r in group
        }
        if policies != {policy_snapshot}:
            continue
        on_rows = [row for row in group if scaffold_condition(row) == "on"]
        skill_context = selected_skill_context(
            (on_rows or group)[0].get("metadata", {}).get("skill_context")
        )
        attempts = []
        for row in group:
            metadata = row.get("metadata", {})
            graph = row.get("graph") or metadata.get("solver_trace", {}).get("final_graph")
            attempts.append(
                {
                    **_attempt(task, row, skill_context),
                    "condition": scaffold_condition(row),
                    "structure": graph_structure(graph),
                }
            )
        rewards = [item["reward"] for item in attempts]
        scope = resolve_scope(task.task_type, task.metadata)
        admitted[scope].extend(group)
        arms = _arm_evidence(group)
        evidence.append(
            {
                **({"arms": arms} if arms is not None else {}),
                "id": _digest([run, step, scope, task_id]),
                "task_id": str(task.metadata.get("source_task_id", task_id)),
                "scope": scope,
                "step": step,
                "policy_snapshot": policy_snapshot,
                "context_hash": _digest(
                    sorted({_digest(r.get("metadata", {}).get("skill_context", {})) for r in group})
                ),
                "skill_context": skill_context,
                "task_goal": public_text(task.prompt, 1200),
                "rollout_ids": sorted(r["rollout_id"] for r in group),
                "reward_mean": sum(rewards) / len(rewards),
                "valid_rollouts": len(rewards),
                "all_failed": all(r == 0 for r in rewards),
                "all_passed": all(r == 1 for r in rewards),
                "within_condition": {
                    condition: within_condition([a for a in attempts if a["condition"] == condition])
                    for condition in ("on", "off")
                    if any(a["condition"] == condition for a in attempts)
                },
                "condition_contrast": condition_contrast(attempts),
                "examples": [
                    {key: item[key] for key in ("rollout_id", "reward", "condition", "failure_mode", "trace")}
                    for item in _representatives(attempts)
                ],
            }
        )
    return evidence, dict(admitted)


def _contrast_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-condition contrast over the retained evidence window.

    One number per task group that has both conditions: scaffolded minus
    scaffold-free reward. It tells the reviewer whether the cards help at all
    (``mean_reward_gap`` > 0, ``on_better`` > ``off_better``) or have stopped mattering.
    """
    gaps = [
        float(item["condition_contrast"]["reward_gap"])
        for item in evidence
        if item.get("condition_contrast")
    ]
    return {
        "groups": len(gaps),
        "mean_reward_gap": sum(gaps) / len(gaps) if gaps else None,
        "on_better": sum(gap > 0 for gap in gaps),
        "off_better": sum(gap < 0 for gap in gaps),
        "tied": sum(gap == 0 for gap in gaps),
    }


def _validated_operation(
    operation: Any,
    *,
    index: int,
    by_id: dict[str, dict[str, Any]],
    editor_step: str,
    evidence_by_id: dict[str, dict[str, Any]],
    config: PatsConfig,
    scope: str,
    step: int,
    touched: set[str],
    additions: int,
) -> tuple[str, dict[str, Any] | None]:
    """Check one edit against the transaction so far; returns (skill_id, new record or None)."""
    if not isinstance(operation, dict) or set(operation) - {
        "op",
        "skill_id",
        "card",
        "evidence_ids",
    }:
        raise ValueError("invalid PATS operation fields")
    op = operation.get("op")
    if op not in {"ADD", "UPDATE", "DELETE"}:
        raise ValueError("unknown PATS operation")
    refs = operation.get("evidence_ids")
    if not isinstance(refs, list) or any(
        not isinstance(r, str) or r not in evidence_by_id for r in refs
    ):
        raise ValueError("PATS edit references unknown evidence")
    if len({evidence_by_id[r]["task_id"] for r in refs}) < config.min_evidence_cards:
        raise ValueError("PATS edits require distinct-task supporting evidence")
    skill_id = operation.get("skill_id")
    if op == "ADD":
        if skill_id is not None or editor_step != "DISTILL":
            raise ValueError("PATS addition is forbidden in this editor step")
        if additions + 1 > 2:
            raise ValueError("PATS editor step addition budget exceeded")
        skill_id = "pats_" + _digest([scope, step, index, operation])[:24]
        if skill_id in by_id:
            raise ValueError("PATS duplicate addition")
    elif editor_step == "DISTILL":
        raise ValueError("PATS DISTILL permits ADD only")
    elif not isinstance(skill_id, str) or skill_id not in by_id:
        raise ValueError("PATS edit references an unknown skill")
    if skill_id in touched:
        raise ValueError("PATS cannot edit a skill twice in one transaction")
    if op == "DELETE":
        if "card" in operation:
            raise ValueError("PATS DELETE cannot carry a card")
        return skill_id, None
    card = operation.get("card")
    if not isinstance(card, dict) or set(card) != set(CARD_FIELDS):
        raise ValueError("PATS ADD/UPDATE requires the complete card schema")
    if any(not isinstance(value, str) or not value.strip() for value in card.values()):
        raise ValueError("PATS card fields must be nonempty text")
    text = " ".join(card.values())
    if (
        card["kind"] not in SKILL_KINDS
        or len(text) > 6000
        or re.search(
            r"gold_answer|reference_answer|rubric_items|api[_ -]?key|ignore .*system",
            text,
            re.I,
        )
    ):
        raise ValueError("PATS card violates static content boundary")
    previous = by_id.get(skill_id)
    version = int(previous["version"]) + 1 if previous else 1
    record = {
        "card": SkillCard(
            skill_id=skill_id,
            **card,
            task_types=[json.loads(scope)[1]],
            evidence=sorted(set(refs)),
            stats=SkillStats(creation_step=step),
        ).to_dict(),
        "version": version,
        "parent_version": previous["version"] if previous else None,
        "status": "active",
        "provenance": "pats_scoped_unvalidated",
        "required_tools": copy.deepcopy(previous.get("required_tools", [])) if previous else [],
        "excluded_task_types": [],
        "source_cases": sorted(set(refs)),
        "pats_scope": scope,
        "pats_editor_step": editor_step,
    }
    record["card"]["evidence"] = sorted(set(refs))
    SkillCard.from_dict(record["card"]).validate()
    return skill_id, record


def _normalize_phrase(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def rule_check_card(card: dict[str, str], *, field_char_limit: int) -> str | None:
    """Rule checker (PATS experience_bank validation); returns a reason or None."""
    for key in CARD_FIELDS:
        if key == "kind":
            continue
        value = card[key]
        if len(value) > field_char_limit:
            return f"{key} exceeds {field_char_limit} characters"
        if _normalize_phrase(value) in VAGUE_PHRASE_BLOCKLIST:
            return f"{key} is a vague platitude"
        if any(marker in value.casefold() for marker in _PROMPT_ECHO_MARKERS):
            return f"{key} echoes prompt markup"
    return None


def _card_words(card: dict[str, Any]) -> set[str]:
    return _token_set(f"{card.get('name', '')} {card.get('plan', '')}")


def apply_operations(
    records: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    editor_step: str,
    evidence: list[dict[str, Any]],
    config: PatsConfig,
    token_counter: Callable[[str], int],
    scope: str,
    step: int,
    editable: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check each edit, drop invalid ones and commit the rest atomically.

    DISTILL permits ADD only (at most 2); an ADD whose name+plan words overlap an existing card at
    Jaccard >= dedup_jaccard is dropped as a duplicate. REVISE permits UPDATE or DELETE and no ADD;
    ``editable`` (the ledger-guided REVISE target) drops UPDATE/DELETE of any other card. An edit
    overlapping an earlier kept edit of the batch is dropped.
    """
    if (
        editor_step not in {"REVISE", "DISTILL"}
        or not isinstance(payload, dict)
        or set(payload) != {"operations"}
    ):
        raise ValueError("PATS requires a JSON object containing only operations")
    if not isinstance(payload["operations"], list):
        raise ValueError("PATS operations must be an array")
    by_id = {record["card"]["skill_id"]: copy.deepcopy(record) for record in records}
    existing = {record["card"]["skill_id"]: _card_words(record["card"]) for record in records}
    evidence_by_id = {e["id"]: e for e in evidence if e.get("scope") == scope}
    touched: set[str] = set()
    additions = 0
    applied, dropped, batch_words = [], [], []
    for index, operation in enumerate(payload["operations"]):
        if len(applied) >= config.max_edits:
            dropped.append({"index": index, "reason": "PATS edit budget exceeded"})
            continue
        card = operation.get("card") if isinstance(operation, dict) else None
        words = _card_words(card) if isinstance(card, dict) else None
        if words is not None and editor_step == "DISTILL" and operation.get("op") == "ADD" and existing:
            if max(_jaccard(words, value) for value in existing.values()) >= config.dedup_jaccard:
                dropped.append({"index": index, "reason": "duplicate of an existing card"})
                continue
        if words is not None and any(
            _jaccard(words, other) >= config.dedup_jaccard for other in batch_words
        ):
            dropped.append({"index": index, "reason": "duplicate of an earlier edit in this batch"})
            continue
        if (
            editable is not None
            and isinstance(operation, dict)
            and operation.get("op") in {"UPDATE", "DELETE"}
            and operation.get("skill_id") not in editable
        ):
            dropped.append(
                {"index": index, "reason": "ledger-guided REVISE edits only " + ", ".join(sorted(editable))}
            )
            continue
        try:
            skill_id, record = _validated_operation(
                operation,
                index=index,
                by_id=by_id,
                editor_step=editor_step,
                evidence_by_id=evidence_by_id,
                config=config,
                scope=scope,
                step=step,
                touched=touched,
                additions=additions,
            )
            if record is not None:
                reason = rule_check_card(operation["card"], field_char_limit=config.field_char_limit)
                if reason is not None:
                    raise ValueError(reason)
                candidate = [record if key == skill_id else value for key, value in by_id.items()]
                if skill_id not in by_id:
                    candidate.append(record)
                if (
                    len(candidate) > config.max_skills
                    or token_counter(render_cards(sorted(candidate, key=lambda r: r["card"]["skill_id"])))
                    > config.max_tokens
                ):
                    raise ValueError("PATS resulting view exceeds capacity")
        except ValueError as exc:
            dropped.append({"index": index, "reason": str(exc)[:300]})
            continue
        additions += operation["op"] == "ADD"
        touched.add(skill_id)
        if record is None:
            del by_id[skill_id]
        else:
            by_id[skill_id] = record
            batch_words.append(words)
        applied.append(operation)
    result = [by_id[key] for key in sorted(by_id)]
    return result, {"applied": applied, "dropped": dropped}


def _unsolved_failure(evidence: dict[str, Any]) -> bool:
    """A task group no arm solved: both arms present and failed, or (no arm) every sibling failed."""
    arms = evidence.get("arms")
    return bool(arms["both_failed"]) if arms else bool(evidence.get("all_failed"))


def _ledger_row(key: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    row = rows.get(key)
    return row if row is not None and row.get("status") != "retired" else None


def _revise_target(
    cards: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """The active card with the lowest posterior mean effect among cards with ledger records."""
    candidates = []
    for card in cards:
        key = record_card_key(card)
        row = _ledger_row(key, rows)
        if row is not None and int(row.get("n", 0)) >= 1 and row.get("mean") is not None:
            candidates.append((float(row["mean"]), key, card, row))
    if not candidates:
        return None
    mean, key, card, row = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "id": str(card["card"]["skill_id"]),
        "version": int(card["version"]),
        "key": key,
        "n": int(row["n"]),
        "mean": mean,
        "E_delta_plus": row.get("E_delta_plus"),
        "q05": row.get("q05"),
        "status": row.get("status"),
    }


def _scope_card_ledger(
    cards: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per-card ledger posterior of a scope's active cards, for the refiner's evidence."""
    entries = []
    for card in cards:
        row = _ledger_row(record_card_key(card), rows) or {}
        entries.append(
            {
                "id": str(card["card"]["skill_id"]),
                "version": int(card["version"]),
                "n": int(row.get("n", 0)),
                "mean": row.get("mean"),
                "E_delta_plus": row.get("E_delta_plus"),
                "q05": row.get("q05"),
                "status": row.get("status", card.get("ledger_status", "incumbent")),
            }
        )
    return {
        "note": (
            "Arm-coupled evidence per card: on each of n tasks some siblings saw the card "
            "(visible) and some saw the same context without it (hidden); mean is the posterior "
            "mean of the success-rate difference visible - hidden, E_delta_plus its expected "
            "positive part, q05 its 5% quantile. Negative mean: the card hurt."
        ),
        "cards": entries,
    }


def _summarize_reviewed_cards(summary: dict[str, Any], state: dict[str, Any]) -> None:
    """Add cards the reviews created (on trial, no records yet) and mark replaced versions."""
    present: dict[str, list[str]] = defaultdict(list)
    for scope, record in sorted(state["scopes"].items()):
        for card in record["cards"]:
            present[record_card_key(card)].append(scope)
            key = record_card_key(card)
            if key not in summary["cards"]:
                summary["cards"][key] = {
                    "id": str(card["card"]["skill_id"]),
                    "version": int(card["version"]),
                    "scopes": [],
                    "n": 0,
                    "mean": None,
                    "E_delta_plus": None,
                    "q05": None,
                    "q95": None,
                    "q_admit": None,
                    "decision": "undecided",
                    "status": card.get("ledger_status", "trial"),
                }
    for key, row in summary["cards"].items():
        if row.get("status") == "retired":
            continue
        if key in present:
            row["scopes"] = present[key]
        else:
            row["status"] = "replaced"  # updated or deleted by this cycle's review


def _ledger_retired_keys(state: dict[str, Any] | None) -> set[str]:
    """Card keys the skill ledger retired (``[r2orch.arms]``); empty before the first retirement."""
    return set(((state or {}).get("skill_ledger") or {}).get("retired") or {})


def _ledger_retired_archive(record: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The archive entries of a scope that hold card ``key`` retired by the skill ledger."""
    return [
        entry for entry in record.get("archive", []) if record_card_key(entry["record"]) == key
    ]


def _ledger_retired_view(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Each scope's ledger-retired cards with their archived records (``audit_share``).

    Only cards that are still retired; the record is the exact version that was retired, so an
    audit sibling sees the card the ledger took away (retrieval no longer shows it).
    """
    retired = _ledger_retired_keys(state)
    view: dict[str, list[dict[str, Any]]] = {}
    for scope, record in sorted((state.get("scopes") or {}).items()):
        entries: dict[str, dict[str, Any]] = {}
        for entry in record.get("archive", []):
            key = record_card_key(entry["record"])
            if key in retired:
                entries[key] = {
                    "key": key,
                    "id": str(entry["record"]["card"]["skill_id"]),
                    "version": int(entry["record"]["version"]),
                    "reason": entry.get("reason"),
                    "retired_step": entry.get("archived_step"),
                    "record": copy.deepcopy(entry["record"]),
                }
        if entries:
            view[scope] = [entries[key] for key in sorted(entries)]
    return view


def _mark_trial_cards(before: list[dict[str, Any]], after: list[dict[str, Any]], step: int) -> None:
    """Flag the records an edit transaction created (ADD) or rewrote (UPDATE) as ledger trials."""
    previous = {(r["card"]["skill_id"], int(r["version"])) for r in before}
    for record in after:
        if (record["card"]["skill_id"], int(record["version"])) not in previous:
            record["ledger_status"] = "trial"
            record["ledger_status_step"] = step


class PatsController:
    """Bounded synchronous reviews per admitted scope (REVISE, then DISTILL); scoped state never
    edits global cards."""

    def __init__(
        self,
        store,
        config: PatsConfig,
        token_counter: Callable[[str], int],
        *,
        scopes: tuple[str, ...],
        skill_ledger: Any,
    ):
        config.validate()
        self.store, self.config, self.token_counter = store, config, token_counter
        # Every configured scope appears in each frozen view.
        self.scopes = tuple(scopes)
        # ``[r2orch.arms]``: the skill ledger whose decisions retire/admit cards at the end of
        # each cycle (``_apply_skill_ledger``).
        self.skill_ledger = skill_ledger
        with store.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS pats_state(id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS pats_cycles(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def _base_cards(self, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        cards = [r for r in self.store.cards() if r["status"] in {"seed", "active"}]
        retired = _ledger_retired_keys(state)
        if retired:
            # A card the skill ledger retired stays out of every scope, new ones included, until
            # the ledger reinstates it (``_reinstate_retired``).
            cards = [r for r in cards if record_card_key(r) not in retired]
        return cards

    def _new_scope_record(
        self, scope: str, base: list[dict[str, Any]], policy_snapshot: str | None
    ) -> dict[str, Any]:
        task_type = json.loads(scope)[1]
        record = {
            "schema": SCOPE_SCHEMA,
            "groups_seen": 0,
            "evidence": [],
            "policy_snapshot": policy_snapshot,
            "cards": copy.deepcopy(
                [
                    r
                    for r in base
                    if (not r["card"].get("task_types") or task_type in r["card"]["task_types"])
                    and task_type not in r.get("excluded_task_types", [])
                ]
            ),
            "archive": [],
        }
        self._finalize_scope(record)
        return record

    def _finalize_scope(self, record: dict[str, Any]) -> None:
        """Card counts, computed after edits."""
        record.update(
            card_count=len(record["cards"]),
            card_tokens=self.token_counter(render_cards(record["cards"])),
            archive_count=len(record["archive"]),
        )

    def snapshot(self, *, run: str | None = None, next_step: int | None = None) -> dict[str, Any]:
        from .pats_semantics import SEMANTIC_REVISION, contract_hash

        with self.store.connect() as db:
            row = db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()
        state = json.loads(row[0]) if row else {"scopes": {}, "step": -1}
        if row and not config_matches(state["config"], self.config):
            raise ValueError("PATS configuration changed; use a fresh scoped state")
        if row and run is not None and state["run"] != run:
            raise ValueError(
                "PATS state belongs to another run; configure a separate skill cases path"
            )
        if row and next_step is not None and next_step <= state["step"]:
            raise ValueError("PATS cannot collect a new cycle at an already committed step")
        view = {
            "schema": SCHEMA,
            "selection_revision": "learned_first",
            "semantic_gate_revision": SEMANTIC_REVISION,
            "semantic_contract_sha256": self.config.semantic_contract_hash(),
            "semantic_default_contract_sha256": contract_hash(self.config.director_prompt_variant),
            "semantic_contract_sha256_by_dataset": self.config.contract_hashes_by_dataset(),
            "config": config_payload(self.config),
            "step": state["step"],
            "run": state.get("run"),
        }
        scopes = dict(state["scopes"])
        missing = [key for key in self.scopes if key not in scopes]
        if missing:
            base = self._base_cards(state)
            scopes.update({key: self._new_scope_record(key, base, None) for key in missing})
        view["scopes"] = {
            key: {
                "policy_snapshot": value["policy_snapshot"],
                # The rule checker admits edits before commit.
                "cards": copy.deepcopy(value["cards"]),
            }
            for key, value in sorted(scopes.items())
        }
        for scope, value in view["scopes"].items():
            value.update({key: scopes[scope][key] for key in SCOPE_FIELDS})
        # [r2orch.arms] audit_share: the frozen collection plans audits of retired cards.
        view[LEDGER_RETIRED_VIEW_KEY] = _ledger_retired_view(state)
        view["snapshot_id"] = _digest(view)
        return view

    def maintain(
        self,
        tasks,
        rows,
        *,
        run: str,
        step: int,
        policy_snapshot: str | None = None,
        backend=None,
        mock: bool = False,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}
        if not isinstance(step, int) or step < 0:
            raise ValueError("PATS cycle step must be nonnegative")
        if not policy_snapshot:
            raise ValueError("PATS evidence requires the behavior Solver snapshot")
        identity = _json([run, step])
        input_hash = _digest(
            {
                "config": config_payload(self.config),
                "policy": policy_snapshot,
                "rows": sorted(rows, key=lambda r: str(r.get("rollout_id", ""))),
                "tasks": {
                    k: {"prompt": t.prompt, "type": t.task_type, "metadata": t.metadata}
                    for k, t in sorted(tasks.items())
                },
                "mock": mock,
            }
        )
        # Do not hold a SQLite transaction across inference. This lock protects the
        # full read/review/commit cycle from another controller process.
        with self.store.path.with_suffix(".pats.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.store.connect() as db:
                old = db.execute(
                    "SELECT payload FROM pats_cycles WHERE id=?", (identity,)
                ).fetchone()
                if old:
                    receipt = json.loads(old[0])
                    if receipt["input_hash"] != input_hash:
                        raise ValueError("PATS cycle replay changed input/configuration")
                    return receipt
                current = db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()
            state = (
                json.loads(current[0])
                if current
                else {
                    "config": config_payload(self.config),
                    "run": run,
                    "step": -1,
                    "scopes": {},
                }
            )
            if not config_matches(state["config"], self.config) or state["run"] != run:
                raise ValueError("PATS configuration/run changed; use a fresh scoped state")
            if step <= state["step"]:
                raise ValueError("PATS cycles must commit in increasing order")
            receipt = self._maintain(
                state,
                tasks,
                rows,
                run=run,
                step=step,
                policy_snapshot=policy_snapshot,
                backend=backend,
                mock=mock,
            )
            receipt["input_hash"] = input_hash
            with self.store.connect() as db:
                db.execute("INSERT OR REPLACE INTO pats_state VALUES(1,?)", (_json(state),))
                db.execute("INSERT INTO pats_cycles VALUES(?,?)", (identity, _json(receipt)))
            return receipt

    def _maintain(
        self,
        state: dict[str, Any],
        tasks,
        rows,
        *,
        run: str,
        step: int,
        policy_snapshot: str,
        backend,
        mock: bool,
    ) -> dict[str, Any]:
        """R2Orch outer step: the ledger's decisions, then the editor's REVISE and DISTILL steps."""
        from .pats_semantics import SEMANTIC_REVISION

        evidence, scope_rows = collect_evidence(
            tasks, rows, run=run, step=step, policy_snapshot=policy_snapshot
        )
        grouped = defaultdict(list)
        for item in evidence:
            grouped[item["scope"]].append(item)
        base = self._base_cards(state)
        for scope in sorted(set(self.scopes) | set(grouped)):
            if scope not in state["scopes"]:
                state["scopes"][scope] = self._new_scope_record(scope, base, policy_snapshot)
        reports = {}
        for scope, record in sorted(state["scopes"].items()):
            record["groups_seen"] += len(grouped.get(scope, []))
            if scope_rows.get(scope):
                record["policy_snapshot"] = policy_snapshot
            reports[scope] = {
                "edits_applied": 0,
                "refiner_calls": 0,
                "checker_calls": 0,
                "revise_status": "no_evidence",
                "distill_status": "no_evidence",
            }
        # The ledger decisions come first, so the editor sees the surviving cards with their
        # posteriors (card_ledger): REVISE edits the card with the lowest posterior mean and
        # DISTILL reads the failures that no arm solved.
        ledger_summary = self._apply_skill_ledger(state, step, reports)
        guided = "cards" in ledger_summary
        required = max(self.config.min_groups, self.config.min_evidence_cards)
        reviews = []
        calls = checker_calls = 0
        reviewed: set[str] = set()  # scopes whose editor ran this cycle (max_reviews_per_cycle)
        ordered_scopes = sorted(
            grouped, key=lambda key: (state["scopes"][key].get("last_review_step", -1), key)
        )
        for scope in ordered_scopes:
            current_groups = grouped[scope]
            record = state["scopes"][scope]
            report = reports[scope]
            recent = [
                e for e in record["evidence"] if step - e["step"] <= self.config.max_policy_lag
            ]
            recent.extend(current_groups)
            recent.sort(key=lambda e: (-e["step"], e["id"]))
            record["evidence"] = recent[: self.config.max_evidence_groups]
            admitted = scope_rows.get(scope, [])
            # DISTILL sees the cards as REVISE left them.
            for editor_step in ("REVISE", "DISTILL"):
                review = {
                    "scope": scope,
                    "editor_step": editor_step,
                    "mean_reward": sum(e["reward_mean"] for e in current_groups) / len(current_groups),
                    "credible_groups": len(current_groups),
                    "bank_pressure": max(
                        record["card_count"] / self.config.max_skills,
                        record["card_tokens"] / self.config.max_tokens,
                    ),
                    "policy_snapshot": policy_snapshot,
                    "status": "evidence_only",
                }
                reviews.append(review)
                target = editable = None
                review_evidence = record["evidence"]
                if editor_step == "REVISE":
                    if guided:
                        target = _revise_target(record["cards"], ledger_summary["cards"])
                    if target is not None:
                        editable = {target["id"]}
                        review["revise_target"] = target["id"]
                else:
                    review_evidence = [e for e in record["evidence"] if _unsolved_failure(e)]
                if len({e["task_id"] for e in record["evidence"]}) < required:
                    review["status"] = "insufficient_evidence"
                elif editor_step == "REVISE" and target is None:
                    review["status"] = "no_revision_candidate"
                elif (
                    editor_step == "DISTILL"
                    and len({e["task_id"] for e in review_evidence}) < required
                ):
                    review["status"] = "insufficient_unsolved_failure_evidence"
                elif step % self.config.review_interval:
                    review["status"] = "interval_not_due"
                elif editor_step == "DISTILL" and (
                    record["card_count"] >= self.config.max_skills
                    or record["card_tokens"] >= self.config.max_tokens
                ):
                    review["status"] = "scaffold_full"
                elif scope not in reviewed and len(reviewed) >= self.config.max_reviews_per_cycle:
                    review["status"] = "cycle_review_budget"
                elif checker_calls >= self.config.max_checker_calls_per_cycle:
                    review["status"] = "checker_budget"
                elif mock or backend is None:
                    review["status"] = "mock_no_refiner" if mock else "refiner_unavailable"
                else:
                    record["last_review_step"] = step
                    extra = {
                        "contrast": _contrast_summary(record["evidence"]),
                        "card_gaps": card_gaps(admitted),
                        "unhelpful_cards": unhelpful_cards(admitted),
                    }
                    if guided:
                        extra["card_ledger"] = _scope_card_ledger(
                            record["cards"], ledger_summary["cards"]
                        )
                    if editor_step == "DISTILL":
                        extra["evidence_selection"] = (
                            "DISTILL evidence is restricted to task groups that no arm solved: "
                            "every sibling failed, both without card k (hidden) and with it "
                            "(visible), or no card was retrieved and every sibling failed."
                        )
                    else:
                        extra["revise_target"] = {
                            **target,
                            "note": "REVISE may UPDATE or DELETE only this card (the active card "
                            "with the lowest posterior mean effect); no card is added in this step.",
                        }
                    called, checked = self._edit_scope(
                        editor_step=editor_step,
                        scope=scope,
                        record=record,
                        review=review,
                        report=report,
                        extra=extra,
                        review_evidence=review_evidence,
                        editable=editable,
                        backend=backend,
                        policy_snapshot=policy_snapshot,
                        step=step,
                    )
                    calls += called
                    checker_calls += checked
                    if called:
                        reviewed.add(scope)
                report[f"{editor_step.lower()}_status"] = review["status"]
        if guided:
            _summarize_reviewed_cards(ledger_summary, state)
        state["step"] = step
        receipt = {
            "schema": SCHEMA,
            "step": step,
            "run": run,
            "policy_snapshot": policy_snapshot,
            "credible_groups": len(evidence),
            "refiner_calls": calls,
            "checker_calls": checker_calls,
            "semantic_validator_revision": SEMANTIC_REVISION,
            "semantic_contract_sha256": self.config.semantic_contract_hash(),
            "semantic_contract_sha256_by_dataset": self.config.contract_hashes_by_dataset(),
            "reviews": reviews,
            "scopes": {
                scope: {
                    **{key: record[key] for key in SCOPE_FIELDS if key != "schema"},
                    **reports[scope],
                }
                for scope, record in sorted(state["scopes"].items())
            },
        }
        receipt["skill_ledger"] = ledger_summary
        return receipt

    def _edit_scope(
        self,
        *,
        editor_step: str,
        scope: str,
        record: dict[str, Any],
        review: dict[str, Any],
        report: dict[str, Any],
        extra: dict[str, Any],
        review_evidence: list[dict[str, Any]],
        editable: set[str] | None,
        backend,
        policy_snapshot: str,
        step: int,
    ) -> tuple[bool, bool]:
        """One editor call for one scope and step: request, rule check, atomic commit.

        Returns (whether the editor was called, whether the rule checker had edits to check). A
        rejected reply leaves ``record["cards"]`` untouched; the cards an accepted edit adds or
        rewrites start on trial.
        """
        review_stage = "input_validation"
        called = checked = False
        try:
            from .pats_refiner import (
                build_review_messages,
                normalize_evidence_references,
                parse_review_response,
                public_response_audit,
                response_generation_audit,
                returned_evidence_references,
                review_json_schema,
            )

            messages, supplied_evidence, request_audit = build_review_messages(
                scope=scope,
                editor_step=editor_step,
                policy_snapshot=policy_snapshot,
                record=(
                    record
                    if review_evidence is record["evidence"]
                    else {**record, "evidence": review_evidence}
                ),
                config=self.config,
                token_counter=self.token_counter,
                extra=extra,
            )
            review.update(request_audit, request_sha256=_digest(messages))
            schema = review_json_schema(
                config=self.config,
                editor_step=editor_step,
                records=(
                    record["cards"]
                    if editable is None
                    else [r for r in record["cards"] if r["card"]["skill_id"] in editable]
                ),
                evidence_aliases=request_audit["evidence_aliases"],
            )
            review["response_schema_sha256"] = _digest(schema)
            review_stage = "refiner_request"
            generate_json = getattr(backend, "generate_json", None)
            called = True
            if callable(generate_json):
                review["structured_output"] = "json_schema"
                response = generate_json(messages, role="skill-distiller", schema=schema)
            else:
                review["structured_output"] = "unavailable_fallback"
                response = backend.generate(messages, role="skill-distiller")
            review_stage = "output_validation"
            raw = response.raw_action_text or response.text
            review.update(public_response_audit(raw))
            review.update(response_generation_audit(response))
            if isinstance(raw, str):
                review["response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            review["response_tokens"] = getattr(response, "token_out", None)
            payload = parse_review_response(raw)
            review["returned_evidence_ids"] = returned_evidence_references(payload)
            payload = normalize_evidence_references(payload, request_audit["evidence_aliases"])
            review_stage = "rule_validation"
            if payload["operations"]:
                checked = True
                report["checker_calls"] += 1
            revised, audit = apply_operations(
                record["cards"],
                payload,
                editor_step=editor_step,
                evidence=supplied_evidence,
                config=self.config,
                token_counter=self.token_counter,
                scope=scope,
                step=step,
                editable=editable,
            )
            review["before_tokens"] = self.token_counter(render_cards(record["cards"]))
            review["after_tokens"] = self.token_counter(render_cards(revised))
            review["operations"] = audit["applied"]
            review["dropped_operations"] = audit["dropped"]
            review["status"] = "updated" if revised != record["cards"] else "unchanged"
            report["edits_applied"] += len(audit["applied"])
            # Cards the editor created or rewrote this cycle start on trial.
            _mark_trial_cards(record["cards"], revised, step)
            record["cards"] = revised
        except Exception as exc:
            review.update(status="rejected", error_type=type(exc).__name__, error_stage=review_stage)
            if isinstance(exc, ValueError) and review_stage != "refiner_request":
                review["validation_error"] = str(exc)[:300]
        report["refiner_calls"] += int(called)
        self._finalize_scope(record)
        return called, checked

    def _apply_skill_ledger(
        self, state: dict[str, Any], step: int, reports: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Slow-timescale card lifecycle of ``[r2orch.arms]`` (end of the PATS cycle).

        Runs before this step's editor calls.  Every active card gets
        its ledger posterior from the records of cycles <= step - 1 (the cycle whose evidence
        this PATS step consumes).  ``retire`` removes the card from every scope that holds it
        (archived with reason ``ledger_retire`` and the posterior).  A ``trial``
        card (created or rewritten by the editor) becomes ``admitted`` on ``admit`` and is
        retired with ``ledger_not_admitted`` once it has ``3 * min_tasks`` records without
        admission; cards present before the lifecycle are ``incumbent``.  The decisions are
        applied to a copy of the state; on a failure PATS maintenance still commits, REVISE finds no
        card with a posterior and the summary records the error.

        The posteriors read the window (``window_cycles``), and before the decisions a retired card
        whose posterior would admit it at ``reinstate_quantile`` is reinstated
        (``_reinstate_retired``).
        """
        ledger = self.skill_ledger
        through = int(step) - 1
        draft = copy.deepcopy(state)
        try:
            summary = self._ledger_decisions(draft, step, through)
        except Exception as exc:  # the scaffold maintenance must not be lost with it
            return {
                "schema": LEDGER_SUMMARY_SCHEMA,
                "step": step,
                "through_cycle": through,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
        state.clear()
        state.update(draft)
        for scope, count in summary.pop("_retired_by_scope").items():
            reports.setdefault(scope, {})["ledger_retired"] = count
        for scope, count in summary.pop("_reinstated_by_scope", {}).items():
            reports.setdefault(scope, {})["ledger_reinstated"] = count
        summary.update(
            epsilon=ledger.epsilon,
            admit_quantile=ledger.admit_quantile,
            min_tasks=ledger.min_tasks,
            bootstrap_draws=ledger.bootstrap_draws,
        )
        summary.update(
            window_cycles=ledger.window_cycles,
            audit_share=ledger.audit_share,
            reinstate_quantile=ledger.reinstate_quantile,
        )
        return summary

    def _reinstate_retired(
        self, state: dict[str, Any], step: int, through: int, contrasts: dict[str, list[float]]
    ) -> tuple[list[str], dict[str, int]]:
        """Reinstate the retired cards whose recent evidence shows that they help again.

        Delta_k depends on the current policy: a card the Director had internalised (retired at
        E[Delta^+] <= epsilon) can matter again after the policy drifts.  Audits keep measuring a
        retired card; once its (windowed) posterior would admit it at ``reinstate_quantile``
        (n >= min_tasks, E[Delta^+] > epsilon so the retire rule no longer holds, Q_q > 0) its
        archived record returns to every scope it was retired from, with status ``admitted``, and
        its key leaves the retired book, so ``_base_cards`` (new scopes) and the Thompson
        candidates include it again.  A scope that meanwhile holds another card with the same id
        keeps the retired copy archived; a card no scope can take back stays retired.
        Retirement never changes the SkillStore status of a card (the store keeps it seed /
        active), so nothing there needs restoring.
        """
        ledger = self.skill_ledger
        book = state["skill_ledger"]
        retired_book = book["retired"]
        reinstated: list[str] = []
        by_scope: dict[str, int] = defaultdict(int)
        for key in sorted(retired_book):
            posterior = ledger.posterior(
                key,
                cycle=through,
                contrasts=contrasts.get(key, []),
                admit_quantile=ledger.reinstate_quantile,
            )
            if ledger.decide(posterior) != "admit":
                continue
            scopes = []
            for scope, record in sorted(state["scopes"].items()):
                entries = _ledger_retired_archive(record, key)
                if not entries:
                    continue
                card = copy.deepcopy(entries[-1]["record"])
                if any(item["card"]["skill_id"] == card["card"]["skill_id"] for item in record["cards"]):
                    continue
                card["ledger_status"] = "admitted"
                card["ledger_status_step"] = step
                record["cards"] = sorted(
                    [*record["cards"], card], key=lambda item: item["card"]["skill_id"]
                )
                taken = {id(entry) for entry in entries}
                record["archive"] = [entry for entry in record["archive"] if id(entry) not in taken]
                scopes.append(scope)
                by_scope[scope] += 1
            if not scopes:
                continue
            retired = retired_book.pop(key)
            book.setdefault("reinstated", []).append(
                {
                    "key": key,
                    "id": retired["id"],
                    "version": retired["version"],
                    "step": step,
                    "through_cycle": through,
                    "scopes": scopes,
                    "retired_step": retired.get("step"),
                    "retired_reason": retired.get("reason"),
                    "reinstate_quantile": ledger.reinstate_quantile,
                    "posterior": posterior.to_dict(),
                }
            )
            reinstated.append(key)
        for scope in sorted(by_scope):
            self._finalize_scope(state["scopes"][scope])
        return reinstated, dict(by_scope)

    def _ledger_decisions(self, state: dict[str, Any], step: int, through: int) -> dict[str, Any]:
        ledger = self.skill_ledger
        book = state.setdefault("skill_ledger", {})
        retired_book = book.setdefault("retired", {})
        # The window (``window_cycles``) applies to every posterior below.
        contrasts = ledger.decision_contrasts(through_cycle=through)
        reinstated_now, reinstated_by_scope = self._reinstate_retired(state, step, through, contrasts)
        holders: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for scope, record in sorted(state["scopes"].items()):
            for card in record["cards"]:
                card.setdefault("ledger_status", "incumbent")
                holders[record_card_key(card)].append((scope, card))
        cards: dict[str, dict[str, Any]] = {}
        retired_now, admitted_now = [], []
        retired_by_scope: dict[str, int] = defaultdict(int)
        touched: set[str] = set()
        for key in sorted(holders):
            entries = holders[key]
            statuses = {card["ledger_status"] for _scope, card in entries}
            status = (
                "trial" if "trial" in statuses else "admitted" if "admitted" in statuses else "incumbent"
            )
            posterior = ledger.posterior(key, cycle=through, contrasts=contrasts.get(key, []))
            decision = ledger.decide(posterior)
            reason = None
            if decision == "retire":
                reason = "ledger_retire"
            elif decision == "admit" and status != "admitted":
                for _scope, card in entries:
                    card["ledger_status"] = "admitted"
                    card["ledger_status_step"] = step
                status = "admitted"
                admitted_now.append(key)
            elif (
                decision != "admit"
                and status == "trial"
                and posterior.n >= 3 * int(ledger.min_tasks)
            ):
                reason = "ledger_not_admitted"
            first = entries[0][1]
            row = {
                "id": str(first["card"]["skill_id"]),
                "version": int(first["version"]),
                "scopes": [scope for scope, _card in entries],
                **posterior.to_dict(),
                "decision": decision,
                "status": status,
            }
            if reason is not None:
                for scope, card in entries:
                    record = state["scopes"][scope]
                    record["cards"] = [item for item in record["cards"] if item is not card]
                    record.setdefault("archive", []).append(
                        {
                            "record": copy.deepcopy(card),
                            "archived_step": step,
                            "reason": reason,
                            "ledger": posterior.to_dict(),
                        }
                    )
                    retired_by_scope[scope] += 1
                    touched.add(scope)
                retired_book[key] = {
                    "id": row["id"],
                    "version": row["version"],
                    "reason": reason,
                    "step": step,
                    "through_cycle": through,
                    "scopes": row["scopes"],
                    "status_before": status,
                    "posterior": posterior.to_dict(),
                }
                row.update(status="retired", reason=reason)
                retired_now.append(key)
            if key in reinstated_now:
                row["reinstated"] = True
            cards[key] = row
        for scope in sorted(touched):
            self._finalize_scope(state["scopes"][scope])
        summary = {
            "schema": LEDGER_SUMMARY_SCHEMA,
            "step": step,
            "through_cycle": through,
            "records": sum(len(values) for values in contrasts.values()),
            "cards": cards,
            "retired_this_cycle": retired_now,
            "admitted_this_cycle": admitted_now,
            "retired_total": len(retired_book),
            "retired": {
                key: {name: value for name, value in item.items() if name != "posterior"}
                for key, item in sorted(retired_book.items())
            },
            "_retired_by_scope": dict(retired_by_scope),
        }
        log = book.get("reinstated") or []
        summary.update(
            window_first_cycle=ledger.window_start(through),
            audits_this_cycle=len(ledger.audit_records(cycle=through)),
            reinstated_this_cycle=reinstated_now,
            reinstated_total=len(log),
            reinstated=[
                {name: value for name, value in item.items() if name != "posterior"}
                for item in log
            ],
            _reinstated_by_scope=reinstated_by_scope,
        )
        return summary
