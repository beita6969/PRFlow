from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class DirectorTurn:
    """One FlowSteer-style model action followed by immutable environment feedback."""

    model_response: str
    feedback: str = ""
    accepted: bool = False
    graph_prefix: dict[str, Any] = field(default_factory=dict)
    prompt_messages: tuple[dict[str, str], ...] = ()
    trainable: bool = True
    call_id: str = ""
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    raw_reasoning_text: str = ""
    raw_action_text: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    action_character_span: tuple[int, int] | None = None
    turn_kind: str = "graph_action"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedPolicyCall:
    call_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    behavior_log_probs: tuple[float, ...] = ()
    action_token_span: tuple[int, int] | None = None
    relation_token_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.action_mask):
            raise ValueError("policy-call token IDs and mask must have equal length")
        if self.behavior_log_probs and len(self.behavior_log_probs) != len(self.token_ids) - 1:
            raise ValueError("behavior log probabilities must align with next-token targets")


@dataclass(frozen=True)
class TokenizedDirectorTrajectory:
    rollout_id: str
    task_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    reward: float
    graph: dict[str, Any]
    seed: int = 0
    executor_version: str = "adaptive"
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()

    def __post_init__(self) -> None:
        from .submission_contract import validate_primary_training_outcome

        validate_primary_training_outcome(self.metadata)
        if len(self.token_ids) != len(self.action_mask):
            raise ValueError("token_ids and action_mask must have equal length")
        if any(value not in (0, 1) for value in self.action_mask):
            raise ValueError("action_mask values must be binary")
        if not any(self.action_mask) and self.metadata.get("training_eligible") is not False:
            raise ValueError("trajectory has no trainable Director action tokens")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["action_mask"] = list(self.action_mask)
        payload["policy_calls"] = [asdict(call) for call in self.policy_calls]
        return payload


def tokenize_director_policy_calls(
    turns: list[DirectorTurn],
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 4096,
) -> tuple[TokenizedPolicyCall, ...]:
    """Tokenize each actual policy call independently; never splice long contexts."""

    calls: list[TokenizedPolicyCall] = []
    chat_encoder = getattr(tokenizer, "encode_chat_trajectory", None)
    for index, turn in enumerate(turns):
        raw_completion = turn.raw_reasoning_text + turn.raw_action_text
        if not raw_completion:
            raw_completion = turn.model_response
        messages = [
            *[dict(message) for message in turn.prompt_messages],
            {"role": "assistant", "content": raw_completion},
        ]
        if turn.prompt_token_ids and turn.completion_token_ids:
            # Both sides of this actual request are already provider-attested.
            # Re-rendering a multi-turn chat can intentionally remove historical
            # reasoning and is not a valid prerequisite for using these exact IDs.
            token_ids = [*turn.prompt_token_ids, *turn.completion_token_ids]
            completion_span = (len(turn.prompt_token_ids), len(token_ids))
            action_mask = [0] * len(turn.prompt_token_ids) + [int(turn.trainable)] * len(
                turn.completion_token_ids
            )
        elif callable(chat_encoder):
            token_ids, _mask, assistant_spans = chat_encoder(messages)
            if not assistant_spans:
                raise ValueError("chat tokenizer did not expose the policy completion span")
            reconstructed_span = assistant_spans[-1]
            if turn.completion_token_ids:
                # The provider's sampled IDs are authoritative. Decoding then
                # re-tokenizing generated text is not lossless for every tokenizer
                # and can also synthesize a chat-template terminal token that the
                # policy did not sample.
                token_ids = [
                    *token_ids[: reconstructed_span[0]],
                    *(int(value) for value in turn.completion_token_ids),
                ]
                completion_span = (reconstructed_span[0], len(token_ids))
            else:
                completion_span = reconstructed_span
            action_mask = [0] * len(token_ids)
            if turn.trainable:
                action_mask[completion_span[0] : completion_span[1]] = [1] * (
                    completion_span[1] - completion_span[0]
                )
        else:
            prompt_ids: list[int] = []
            for message in turn.prompt_messages:
                prompt_ids.extend(tokenizer.encode(message["content"], add_special_tokens=False))
            completion = list(
                turn.completion_token_ids
                or tokenizer.encode(raw_completion, add_special_tokens=False)
            )
            token_ids = prompt_ids + completion
            completion_span = (len(prompt_ids), len(token_ids))
            action_mask = [0] * len(prompt_ids) + [int(turn.trainable)] * len(completion)
        effective_limit = (
            max_tokens * 64 if getattr(tokenizer, "approximate_token_count", False) else max_tokens
        )
        if len(token_ids) > effective_limit:
            raise ValueError(
                f"Director policy call {turn.call_id or index} has {len(token_ids)} tokens; "
                f"limit is {effective_limit}; discontinuous truncation is forbidden"
            )
        completion_length = completion_span[1] - completion_span[0]
        if turn.completion_token_ids and tuple(token_ids[slice(*completion_span)]) != tuple(
            turn.completion_token_ids
        ):
            raise ValueError(
                f"policy call {turn.call_id or index} completion tokens do not match the "
                "rollout-time tokenizer record"
            )
        if turn.prompt_token_ids and tuple(token_ids[: completion_span[0]]) != tuple(
            turn.prompt_token_ids
        ):
            raise ValueError(
                f"policy call {turn.call_id or index} prompt tokens do not match the "
                "rollout-time provider record"
            )
        behavior = [0.0] * max(0, len(token_ids) - 1)
        if turn.behavior_log_probs:
            if len(turn.behavior_log_probs) != completion_length:
                raise ValueError(
                    f"policy call {turn.call_id or index} rollout behavior log probabilities "
                    f"({len(turn.behavior_log_probs)}) do not match completion span "
                    f"({completion_length})"
                )
            for offset, value in enumerate(turn.behavior_log_probs):
                target_position = completion_span[0] + offset - 1
                if target_position < 0:
                    raise ValueError("policy completion cannot begin before a next-token context")
                behavior[target_position] = float(value)
        action_span = completion_span
        # All tokens actually generated by an ordinary Director call receive the
        # graph-level advantage.  Mapping parsed JSON character offsets back onto
        # provider token IDs is neither required nor generally reversible.  The
        # only local-credit span is the separately sampled one-token relation call.
        relation_span = completion_span if turn.turn_kind == "relation_choice" else None
        metadata = dict(turn.metadata)
        if "action_type" in metadata:
            # R2Orch call metadata: E_i is the accepted graph edit after the
            # final </think>; FINISH/RUN/relation tokens are excluded.
            metadata["accepted"] = bool(metadata.get("accepted", turn.accepted))
            metadata.setdefault("rejection_code", None)
            metadata["edit_token_span"] = (
                edit_token_span(token_ids, completion_span, tokenizer)
                if metadata["accepted"]
                and turn.turn_kind == "graph_action"
                and metadata["action_type"] not in NON_EDIT_ACTION_TYPES
                else None
            )
        calls.append(
            TokenizedPolicyCall(
                call_id=turn.call_id or f"call-{index}",
                token_ids=tuple(int(value) for value in token_ids),
                action_mask=tuple(action_mask),
                behavior_log_probs=(
                    tuple(behavior) if turn.behavior_log_probs else ()
                ),
                action_token_span=action_span,
                relation_token_span=relation_span,
                metadata=metadata,
            )
        )
    return tuple(calls)


NON_EDIT_ACTION_TYPES = frozenset({"finish", "run_agent", "invalid", "relation_choice"})


def edit_token_span(
    token_ids: list[int] | tuple[int, ...],
    completion_span: tuple[int, int],
    tokenizer: Tokenizer,
) -> list[int] | None:
    """Absolute [start, end) of completion tokens after the last ``</think>``."""

    start, end = completion_span
    marker = list(tokenizer.encode("</think>", add_special_tokens=False))
    width = len(marker)
    for position in range(end - width, start - 1, -1):
        if list(token_ids[position : position + width]) == marker:
            start = position + width
            break
    return [start, end] if start < end else None


@dataclass(frozen=True)
class TrainingSample:
    rollout_id: str
    task_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    reward: float
    advantage: float
    canonical_graph_key: str = ""
    graph_features: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()


@dataclass(frozen=True)
class TrainingBatch:
    """Trainer-ready data boundary. Optimizer/model references are intentionally absent."""

    role: str
    samples: tuple[TrainingSample, ...]
    objective: str = "masked_graph_level_grpo"
    optimizer_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optimizer_steps != 0:
            raise ValueError("rollout batches must stop before optimizer execution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "objective": self.objective,
            "optimizer_steps": self.optimizer_steps,
            "samples": [asdict(sample) for sample in self.samples],
            "metadata": self.metadata,
        }
