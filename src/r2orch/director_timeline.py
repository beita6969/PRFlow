"""Incremental Director history and exact token-prefix audits.

Online history retains the sampled thinking and actions. Exact token-prefix
checks gate timeline merging without changing the masked GRPO objective.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_CONTEXT_MODE = "snapshot_dedup"
APPEND_CONTEXT_MODE = "append_only"
TIMELINE_CONTEXT_MODES = frozenset({APPEND_CONTEXT_MODE})
HISTORY_THINKING_VISIBILITY = "online_and_training"


def director_context_mode() -> str:
    mode = os.environ.get("R2ORCH_DIRECTOR_CONTEXT_MODE", DEFAULT_CONTEXT_MODE)
    if mode not in {DEFAULT_CONTEXT_MODE, *TIMELINE_CONTEXT_MODES}:
        raise ValueError(
            f"unsupported R2ORCH_DIRECTOR_CONTEXT_MODE: {mode!r}; "
            f"use {DEFAULT_CONTEXT_MODE!r} or {APPEND_CONTEXT_MODE!r}"
        )
    return mode


def timeline_assistant_content(tokenizer: Any, prompt_ids, completion_ids) -> str:
    """Keep the actual prefill and sampled bytes, including thinking and stops."""
    if tokenizer is None or not prompt_ids or not completion_ids:
        raise ValueError("append-only Director requires provider token IDs and a tokenizer")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    decode = lambda ids: tokenizer.decode(  # noqa: E731
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    prompt = decode(prompt_ids)
    delimiter = "<|im_start|>assistant\n"
    if delimiter not in prompt:
        raise ValueError("append-only Director prompt lacks an assistant generation prefix")
    prefill = prompt.rsplit(delimiter, 1)[1]
    return prefill + decode(completion_ids)


def timeline_prefix_audit(previous_ids, prompt_ids, completion_ids) -> dict[str, Any]:
    available = bool(prompt_ids and completion_ids)
    matches = (
        tuple(prompt_ids[: len(previous_ids)]) == tuple(previous_ids)
        if available and previous_ids
        else None
    )
    return {
        "provider_tokens_available": available,
        "previous_policy_token_count": len(previous_ids),
        "previous_policy_is_exact_prefix": matches,
        "timeline_merge_candidate": available and (not previous_ids or matches is True),
    }
