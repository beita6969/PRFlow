"""R2Orch Proposer: one frozen candidate list per anchor, two label choices.

Each list holds up to 8 fixed-pool candidates (4 ADS boundary, 4 TSDS neighbours of
the anchor) with their scope's scaffold state. The Proposer answers with one label
token (thinking off, token-ID constrained, T=1); the same list is sampled twice with
independent per-call seeds. A failed call yields no task and no retry; its list
contributes no gradient this cycle.

``second_pick = "distinct"`` (opt-in) redraws a second pick that lands on the first
candidate from the policy conditioned on not being that candidate.  Two picks on the same
candidate run one task and give no pair (a == b has zero gradient), and their share grows as
the policy sharpens, so exactly when the policy is most confident the list stops teaching.  The
redraw uses the policy's own label probabilities from the second call (no extra request) and a
seed-derived generator, so it is deterministic across resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import canonical_dataset_name
from .observability import TaskSpec
from .selfplay import ProposedTask, SelfPlaySeed, _selection_prompt_preview

LIST_SCHEMA = "r2orch_proposer_list@1"
SECOND_PICK_MODES = ("independent", "distinct")

SYSTEM_PROMPT = (
    "You choose the next training task for a graph-orchestration Director. Each lettered "
    "candidate is a fixed-pool task prepared by ADS boundary scheduling or TSDS retrieval "
    "near the anchor. Prefer a task whose outcome is still uncertain for the Director: "
    "neither reliably solved nor hopeless. Each candidate shows its skill-scaffold scope: "
    "scope_cards is the number of skill cards in the scope's scaffold and scope_card_tokens "
    "their rendered size (null when unknown), previous_success is the task's last observed "
    "success rate. Answer with exactly one candidate letter."
)


def label_token_ids(tokenizer: Any, labels: Sequence[str]) -> dict[str, int]:
    """Attest that every label is exactly one distinct policy token (as for off/on)."""
    resolved = {}
    for label in labels:
        ids = list(tokenizer.encode(label, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"Proposer label {label!r} is not a single tokenizer token")
        resolved[label] = int(ids[0])
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Proposer labels must map to distinct tokenizer tokens")
    return resolved


def _scope_value(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, Mapping) else getattr(state, name)


@dataclass(frozen=True)
class CandidateList:
    list_id: str
    cycle: int
    dataset: str
    scope: str
    anchor_pool_id: str
    candidates: tuple[dict, ...]
    messages: tuple[dict, ...]
    label_token_ids: dict[str, int]

    @property
    def labels(self) -> list[str]:
        return [candidate["label"] for candidate in self.candidates]


@dataclass(frozen=True)
class PairSelection:
    list: CandidateList
    choices: tuple[dict, ...]
    same_choice: bool
    status: str
    failure: str | None
    prompt_token_ids: tuple[int, ...]
    messages_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIST_SCHEMA,
            "cycle": self.list.cycle,
            "list_id": self.list.list_id,
            "dataset": self.list.dataset,
            "scope": self.list.scope,
            "anchor_pool_id": self.list.anchor_pool_id,
            "candidates": [dict(candidate) for candidate in self.list.candidates],
            "labels": self.list.labels,
            "label_token_ids": dict(self.list.label_token_ids),
            "messages_sha256": self.messages_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "choices": [dict(choice) for choice in self.choices],
            "same_choice": self.same_choice,
            "status": self.status,
            "failure": self.failure,
        }


def redraw_distinct_label(
    behavior_logprobs: Mapping[str, float],
    exclude: str,
    *,
    seed: int,
    temperature: float = 1.0,
) -> str | None:
    """Sample a label from the (temperature-scaled) policy restricted to labels != ``exclude``.

    ``behavior_logprobs`` is the policy's own log-softmax over the list's labels as returned by
    the label call.  Returns None when no other label has positive probability.
    """

    others = [
        (label, float(value))
        for label, value in behavior_logprobs.items()
        if label != exclude and math.isfinite(float(value))
    ]
    if not others:
        return None
    scale = 1.0 / max(float(temperature), 1e-6)
    top = max(value for _label, value in others)
    weights = [math.exp((value - top) * scale) for _label, value in others]
    digest = hashlib.sha256(f"{int(seed)}\0distinct_second_pick".encode()).digest()
    draw = random.Random(int.from_bytes(digest[:8], "big")).random() * sum(weights)
    running = 0.0
    for (label, _value), weight in zip(others, weights, strict=True):
        running += weight
        if draw < running:
            return label
    return others[-1][0]


def _messages_sha256(messages: Sequence[dict]) -> str:
    encoded = json.dumps(list(messages), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PairProposer:
    def __init__(
        self,
        backend: Any,
        pool: Any,
        scheduler: Any,
        retriever: Any,
        tokenizer: Any,
        *,
        candidates_per_list: int = 8,
        ads_candidates: int = 4,
        tsds_candidates: int = 4,
        label_alphabet: str = "ABCDEFGH",
        temperature: float = 1.0,
        prompt_preview_chars: int = 500,
        second_pick: str = "independent",
    ) -> None:
        if len(label_alphabet) < candidates_per_list:
            raise ValueError("label_alphabet must provide one label per candidate")
        if second_pick not in SECOND_PICK_MODES:
            raise ValueError(f"second_pick must be one of {SECOND_PICK_MODES}")
        self.backend = backend
        self.pool = pool
        self.scheduler = scheduler
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.candidates_per_list = int(candidates_per_list)
        self.ads_candidates = int(ads_candidates)
        self.tsds_candidates = int(tsds_candidates)
        self.labels = tuple(label_alphabet[: self.candidates_per_list])
        self.temperature = float(temperature)
        self.prompt_preview_chars = int(prompt_preview_chars)
        self.second_pick = second_pick
        self.label_token_ids = label_token_ids(tokenizer, self.labels)

    # ---- candidate list -------------------------------------------------
    def _ranked_candidates(self, anchor_id: str, dataset: str) -> list[tuple[str, str]]:
        scheduler = self.scheduler
        boundary_ids = scheduler.candidates(dataset=dataset)
        recent = scheduler._recent_ids(dataset)
        repository = [
            pool_id
            for pool_id in self.pool.ids
            if self.pool.tasks[pool_id].dataset == dataset
            and pool_id not in recent
            and pool_id not in scheduler.reserved
        ]
        anchor_vector = np.asarray(self.pool.tasks[anchor_id].embedding, dtype=np.float64)
        # Persistently saturated / hopeless tasks leave the TSDS pool too (unless it is thin).
        repository = scheduler.filter_candidates(repository, minimum=self.candidates_per_list)
        # Source identity (not row id) defines "the same task" for a=b and reservation.
        taken: set[str] = set(scheduler._source_equivalents(anchor_id))
        ranked: list[tuple[str, str]] = []

        def take(pool_ids: Sequence[str], source: str, limit: int) -> None:
            for pool_id in pool_ids:
                if limit <= 0 or len(ranked) >= self.candidates_per_list:
                    return
                if pool_id not in taken:
                    taken.update(scheduler._source_equivalents(pool_id))
                    ranked.append((pool_id, source))
                    limit -= 1

        take(boundary_ids, "ads", self.ads_candidates)
        # TSDS neighbours of the anchor (dataset-wide) fill the tsds slots plus any ADS
        # shortfall. TSDS truncates its KDE support, so plain nearest neighbours pad it.
        rest = [pool_id for pool_id in repository if pool_id not in taken]
        nearest = sorted(
            rest,
            key=lambda pool_id: float(
                np.square(np.asarray(self.pool.tasks[pool_id].embedding) - anchor_vector).sum()
            ),
        )
        take(
            [
                *self.retriever.rank(
                    rest, [anchor_vector], limit=self.candidates_per_list - len(ranked)
                ),
                *nearest,
            ],
            "tsds",
            self.candidates_per_list,
        )
        take(boundary_ids, "ads", self.candidates_per_list)  # thin pools only
        if not ranked and anchor_id in repository:
            ranked.append((anchor_id, "ads"))
        return ranked

    def build_candidate_list(
        self, anchor: SelfPlaySeed, *, cycle: int, scope_view: Mapping[str, Any]
    ) -> CandidateList:
        from .pats import resolve_scope

        anchor_id = anchor.seed_id
        anchor_task = self.pool.tasks[anchor_id]
        dataset = canonical_dataset_name(anchor_task.dataset)
        # A cycle may contain several anchors from the same dataset. Their
        # candidates, label meanings and outcome keys must remain distinct.
        anchor_key = hashlib.sha256(anchor_id.encode("utf-8")).hexdigest()[:16]
        list_id = f"c{int(cycle):04d}-{dataset}-a{anchor_key}"
        ranked = self._ranked_candidates(anchor_id, anchor_task.dataset)
        candidates = []
        for label, (pool_id, source) in zip(self.labels, ranked, strict=False):
            entry = self.pool.tasks[pool_id]
            scope = resolve_scope(entry.task.task_type, entry.task.metadata)
            state = scope_view.get(scope)
            candidates.append(
                {
                    "label": label,
                    "pool_id": pool_id,
                    "source": source,
                    "dataset": canonical_dataset_name(entry.dataset),
                    "task_type": entry.task.task_type,
                    "scope": scope,
                    "scope_cards": _scope_value(state, "card_count") if state is not None else None,
                    "scope_card_tokens": (
                        _scope_value(state, "card_tokens") if state is not None else None
                    ),
                    "previous_success": self.scheduler.task_success.get(pool_id),
                    "selection_count": int(self.scheduler.selection_count[pool_id]),
                    "prompt_preview": _selection_prompt_preview(
                        entry.task.prompt,
                        entry.task.metadata,
                        dataset=entry.dataset,
                        max_chars=self.prompt_preview_chars,
                    ),
                }
            )
        public = [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"pool_id", "source", "scope"}
            }
            for candidate in candidates
        ]
        messages = (
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "anchor": _selection_prompt_preview(
                            anchor_task.task.prompt,
                            anchor_task.task.metadata,
                            dataset=anchor_task.dataset,
                            max_chars=self.prompt_preview_chars,
                        ),
                        "candidates": public,
                        "answer_with_one_of": [candidate["label"] for candidate in candidates],
                    },
                    ensure_ascii=False,
                ),
            },
        )
        return CandidateList(
            list_id=list_id,
            cycle=int(cycle),
            dataset=dataset,
            scope=resolve_scope(anchor_task.task.task_type, anchor_task.task.metadata),
            anchor_pool_id=anchor_id,
            candidates=tuple(candidates),
            messages=messages,
            label_token_ids={
                candidate["label"]: self.label_token_ids[candidate["label"]]
                for candidate in candidates
            },
        )

    # ---- two choices ----------------------------------------------------
    def _fallback_prompt_ids(self, messages: Sequence[dict]) -> tuple[int, ...]:
        prompt = (
            "\n".join(f"{message['role'].title()}:\n{message['content']}" for message in messages)
            + "\nAssistant:\n"
        )
        return tuple(int(value) for value in self.tokenizer.encode(prompt, add_special_tokens=False))

    def propose_pair(self, cand: CandidateList, *, seeds: tuple[int, int]) -> PairSelection:
        messages_sha256 = _messages_sha256(cand.messages)
        if len(cand.candidates) == 0:
            return PairSelection(cand, (), False, "failed", "no_candidates", (), messages_sha256)
        by_label = {candidate["label"]: candidate for candidate in cand.candidates}
        choices: list[dict] = []
        prompt_ids: list[tuple[int, ...]] = []
        for k, seed in enumerate(seeds):
            choice: dict[str, Any] = {
                "k": k,
                "seed": int(seed),
                "label": None,
                "pool_id": None,
                "task_id": f"{cand.list_id}-k{k}",
                "behavior_logprobs": {},
                "sampled_logprob": None,
                "valid": False,
                "failure": None,
            }
            try:
                response = self.backend.choose_label(
                    list(cand.messages),
                    role="proposer",
                    token_ids=dict(cand.label_token_ids),
                    seed=int(seed),
                    temperature=self.temperature,
                )
            except Exception as exc:  # no retry: this choice yields no task, the list no gradient
                choice["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                choice.update(
                    label=response.label,
                    pool_id=by_label[response.label]["pool_id"],
                    behavior_logprobs=dict(response.log_probabilities),
                    sampled_logprob=float(response.log_probabilities[response.label]),
                    valid=True,
                )
                prompt_ids.append(
                    tuple(response.prompt_token_ids) or self._fallback_prompt_ids(cand.messages)
                )
            choices.append(choice)
        valid = [choice for choice in choices if choice["valid"]]
        if (
            self.second_pick == "distinct"
            and len(valid) == 2
            and valid[0]["pool_id"] == valid[1]["pool_id"]
        ):
            self._redraw_second(valid[1], valid[0]["label"], by_label)
        same_choice = len(valid) == 2 and valid[0]["pool_id"] == valid[1]["pool_id"]
        if same_choice:
            choices[1]["task_id"] = choices[0]["task_id"]
        failure = next((choice["failure"] for choice in choices if choice["failure"]), None)
        if failure is None and len(set(prompt_ids)) > 1:
            failure = "prompt_token_mismatch"
        return PairSelection(
            list=cand,
            choices=tuple(choices),
            same_choice=same_choice,
            status="ok" if failure is None else "failed",
            failure=failure,
            prompt_token_ids=prompt_ids[0] if prompt_ids else (),
            messages_sha256=messages_sha256,
        )

    def _redraw_second(self, choice: dict, first_label: str, by_label: Mapping[str, dict]) -> None:
        """Replace a second pick equal to the first by a draw from pi(. | label != first)."""
        label = redraw_distinct_label(
            choice["behavior_logprobs"],
            first_label,
            seed=int(choice["seed"]),
            temperature=self.temperature,
        )
        if label is None or label not in by_label:
            return  # degenerate policy: keep the duplicate, the list simply carries no pair
        choice.update(
            redrawn_from=choice["label"],
            label=label,
            pool_id=by_label[label]["pool_id"],
            sampled_logprob=float(choice["behavior_logprobs"][label]),
        )

    def reserve_choices(self, sel: PairSelection) -> None:
        for pool_id in dict.fromkeys(c["pool_id"] for c in sel.choices if c["valid"]):
            self.scheduler.reserve(pool_id)

    def task_for_choice(self, sel: PairSelection, k: int) -> ProposedTask | None:
        choice = sel.choices[k] if k < len(sel.choices) else None
        if choice is None or not choice["valid"] or (k == 1 and sel.same_choice):
            return None
        pool_id = choice["pool_id"]
        entry = self.pool.tasks[pool_id]
        candidate = next(c for c in sel.list.candidates if c["pool_id"] == pool_id)
        r2orch = {
            "list_id": sel.list.list_id,
            "choice_index": k,
            "dataset": candidate["dataset"],
            "scope": candidate["scope"],
            "label": choice["label"],
            "candidate_source": candidate["source"],
        }
        candidate_ids = [c["pool_id"] for c in sel.list.candidates]
        validated = bool(entry.task.metadata.get("validated_pool_entry", False))
        return ProposedTask(
            task=TaskSpec(
                task_id=choice["task_id"],
                prompt=entry.task.prompt,
                reference=entry.task.reference,
                task_type=entry.task.task_type,
                metadata={
                    **entry.task.metadata,
                    "pool_id": pool_id,
                    "proposer_mode": "r2orch_pair",
                    "candidate_ids": candidate_ids,
                    "anchor_pool_id": sel.list.anchor_pool_id,
                    "required_reasoning_hops": None,
                    "validated_pool_entry": validated,
                    "r2orch": r2orch,
                },
                private_verifier_payload=entry.task.private_verifier_payload,
            ),
            response=str(choice["label"]),
            metadata={
                "pool_id": pool_id,
                "candidate_ids": candidate_ids,
                "list_id": sel.list.list_id,
                "choice_index": k,
                "selection_group": f"{entry.dataset}:{entry.cluster_id}",
                "validated_pool_entry": validated,
                "validated_pool_manifest_sha256": entry.task.metadata.get(
                    "validated_pool_manifest_sha256"
                ),
                "validated_pool_version": entry.task.metadata.get("validated_pool_version"),
                "r2orch": r2orch,
            },
        )

    # ---- ADS state ------------------------------------------------------
    def observe_many(self, observations: Sequence[tuple[str, Sequence[float]]]) -> None:
        self.scheduler.record_batch([(pool_id, list(rewards)) for pool_id, rewards in observations])

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, payload: dict) -> None:
        self.scheduler.load_state_dict(payload)


def create_pair_proposer(
    config: Any,
    pool: Any,
    scheduler: Any,
    retriever: Any,
    *,
    tokenizer: Any = None,
    mock: bool = False,
) -> PairProposer:
    """Proposer on its own LoRA route; only the r2orch.proposer table is read."""
    from .application import _gateway_config
    from .llm import MockBackend, OpenAICompatibleBackend

    settings = config.r2orch.proposer
    backend = (
        MockBackend()
        if mock
        else OpenAICompatibleBackend(
            _gateway_config(
                config.proposer_model,
                {"proposer": float(settings.temperature)},
                sampling_seed=config.seed,
            )
        )
    )
    return PairProposer(
        backend,
        pool,
        scheduler,
        retriever,
        tokenizer,
        candidates_per_list=settings.candidates_per_list,
        ads_candidates=settings.ads_candidates,
        tsds_candidates=settings.tsds_candidates,
        label_alphabet=settings.label_alphabet,
        temperature=settings.temperature,
        second_pick=str(settings.second_pick),
    )
