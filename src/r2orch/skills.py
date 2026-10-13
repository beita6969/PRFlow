from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

SKILL_KINDS = frozenset(
    {
        "orchestration",
        "decomposition",
        "communication",
        "verification",
        "tool-use",
        "bug-repair",
    }
)


@dataclass
class SkillStats:
    usage_count: int = 0
    helpful_count: int = 0
    hurt_count: int = 0
    last_used_step: int = -1
    creation_step: int = 0
    is_seed: bool = False


@dataclass
class SkillCard:
    skill_id: str
    name: str
    description: str
    trigger: str
    plan: str
    pitfall: str = ""
    constraint: str = ""
    kind: str = "bug-repair"
    task_types: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    stats: SkillStats = field(default_factory=SkillStats)

    def validate(self) -> None:
        if self.kind not in SKILL_KINDS:
            raise ValueError(f"unknown Solver skill kind: {self.kind}")
        for name, value in (
            ("skill_id", self.skill_id),
            ("name", self.name),
            ("description", self.description),
            ("trigger", self.trigger),
            ("plan", self.plan),
        ):
            if not str(value).strip():
                raise ValueError(f"skill {name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillCard:
        raw = dict(payload)
        raw["stats"] = SkillStats(**raw.get("stats", {}))
        return cls(**raw)


@dataclass(frozen=True)
class SolverFailureCase:
    task: str
    task_type: str
    failure_trace: str
    failure_mode: str
    solver_answer: str = ""
    used_skill_ids: tuple[str, ...] = ()


class E5SkillEmbedder:
    """CPU E5 encoder migrated from SESA's SkillBank retrieval path."""

    _cache: ClassVar[dict[str, tuple[Any, Any]]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        source = str(model_path)
        self.model_path = source
        with self._cache_lock:
            cached = self._cache.get(source)
            if cached is None:
                cached = (
                    AutoTokenizer.from_pretrained(source),
                    AutoModel.from_pretrained(source).eval().cpu(),
                )
                self._cache[source] = cached
        self.tokenizer, self.model = cached

    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]:
        if not texts:
            return []
        prefix = "query: " if query else "passage: "
        vectors: list[tuple[float, ...]] = []
        with self.torch.no_grad():
            for offset in range(0, len(texts), 32):
                batch = [prefix + text for text in texts[offset : offset + 32]]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
                )
                output = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(tuple(float(value) for value in row) for row in pooled)
        return vectors


def _public_failure_case(case: SolverFailureCase) -> dict[str, Any]:
    """Only pre-verification process evidence may reach a future policy skill.

    Archived cases still retain labels for offline analysis. Do not forward them
    wholesale or trust a prompt instruction to hide labels from the distiller.
    """
    private_keys = {
        "reference",
        "references",
        "reference_answer",
        "reference_answers",
        "gold",
        "gold_answer",
        "gold_answers",
        "target_answer",
        "target_answers",
        "private_verifier_payload",
        "verification",
        "verifier_result",
        "verification_detail",
        "rubric_items",
        "rubrics",
        "correct_answer",
        "correct_answers",
        "ground_truth",
        "solution",
        "solutions",
        "hidden_goal",
        "goal_contract",
        "goal_id",
        "target_object",
        "target_receptacle",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if str(key).casefold() not in private_keys
                and not any(word in str(key).casefold() for word in ("canary", "private_", "gold_"))
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str) and (
            value.lstrip().startswith("{")
            or re.match(r'\s*\[\s*(?:[\[\]"{]|true|false|null|-?\d)', value)
        ):
            try:
                return json.dumps(scrub(json.loads(value)), ensure_ascii=False)
            except ValueError:
                # Opaque serialized traces cannot establish a public-data boundary.
                return "[unparseable structured trace omitted]"
        return value

    try:
        trace = json.loads(case.failure_trace)
    except ValueError:
        trace = {}
    if not isinstance(trace, dict):
        trace = {}
    # Exclude post-verification summaries and arbitrary metadata at trace root.
    events = trace.get("events", [])
    events = events if isinstance(events, list) else []
    process = {
        "final_graph": trace.get("final_graph", {}),
        "events": [
            event
            for event in events
            if isinstance(event, dict) and event.get("kind") == "canvas_step"
        ],
    }
    return {
        "schema": "public_failure_case",
        "task": case.task,
        "task_type": case.task_type,
        "failure_trace": scrub(process),
        "failure_mode": case.failure_mode
        if re.fullmatch(r"[a-z_]+", case.failure_mode)
        else "task_verification_failure",
        "solver_answer": case.solver_answer,
        "used_skill_ids": list(case.used_skill_ids),
    }


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal length")
    return sum(a * b for a, b in zip(left, right, strict=True))
