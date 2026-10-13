from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .director_timeline import TIMELINE_CONTEXT_MODES, director_context_mode


def _chat_template_ids(rendered: Any) -> list[int]:
    """Normalize Transformers chat-template return types across releases."""

    if hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("expected one chat trajectory, received a token batch")
        rendered = rendered[0]
    return [int(token_id) for token_id in rendered]


def qwen_request_extra(
    *, enable_thinking: bool = False, director_timeline: bool = False
) -> dict[str, Any]:
    """Request options for Qwen3.5 chat servers."""

    options: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    if director_timeline:
        options["director_append_only"] = True
    if enable_thinking:
        options["thinking_budget"] = 512
    return {"chat_template_kwargs": options}


def vllm_enforce_eager() -> bool:
    """Whether to force vLLM eager execution (``R2ORCH_VLLM_ENFORCE_EAGER=1``).

    Eager mode was a workaround for FlashInfer's compilation passes on Python
    3.11. Measured on A100-64GB (vLLM 0.27, Qwen3.5-9B, LoRA enabled,
    32 concurrent requests) CUDA graphs run without that problem and decode
    2.7x (base) to 3.3x (LoRA) faster: 1430 vs 434 tok/s with a LoRA adapter.
    CUDA graphs are therefore the default; set the variable to 1 to go back.
    """

    return os.environ.get("R2ORCH_VLLM_ENFORCE_EAGER", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def qwen_vllm_server_args(model_name: str) -> tuple[str, ...]:
    """Select the Python 3.11-compatible vLLM path for Qwen3.5.

    The Triton GDN prefill backend supplies the Qwen GDN prefill kernel and the
    native sampler is selected by :func:`qwen_vllm_server_env`. ``--enforce-eager``
    is opt-in (see :func:`vllm_enforce_eager`).
    """

    normalized = "".join(character for character in model_name.casefold() if character.isalnum())
    if "qwen35" not in normalized:
        return ()
    timeline_args = (
        (
            "--chat-template",
            str(
                Path(__file__).resolve().parents[2] / "configs/templates/director_append_only.jinja"
            ),
        )
        if director_context_mode() in TIMELINE_CONTEXT_MODES
        else ()
    )
    return (
        (("--enforce-eager",) if vllm_enforce_eager() else ())
        + (
            "--max-model-len",
            "32768",
            # Match FlowSteer's serving path: repeated Director prompts share a
            # large stable system/task prefix, while chunked prefill prevents one
            # long Canvas prompt from monopolizing the scheduler.  These only
            # change vLLM scheduling/cache behavior, not sampling semantics.
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--gdn-prefill-backend",
            "triton",
        )
        + timeline_args
    )


def qwen_vllm_server_env(model_name: str) -> tuple[tuple[str, str], ...]:
    """Use vLLM's native sampler when FlashInfer cannot JIT without CUDA_HOME."""

    normalized = "".join(character for character in model_name.casefold() if character.isalnum())
    if "qwen35" not in normalized:
        return ()
    return (("VLLM_USE_FLASHINFER_SAMPLER", "0"),)


def response_content(
    message: Any,
    *,
    enable_thinking: bool = False,
    reasoning_fallback: bool = False,
) -> str:
    """Read Qwen content while tolerating servers that return reasoning separately."""

    content = str(getattr(message, "content", "") or "").strip()
    if content:
        return content
    reasoning = str(
        getattr(message, "reasoning_content", "")
        or getattr(message, "reasoning", "")
        or ""
    ).strip()
    if enable_thinking and reasoning:
        lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
        return lines[-1] if lines else reasoning
    if reasoning_fallback and reasoning:
        return reasoning
    return ""


def response_policy_parts(message: Any, *, thinking_prefilled: bool = False) -> tuple[str, str]:
    """Return provider-authored reasoning and content without cleaning either string.

    Policy trajectory persistence must happen before presentation-oriented trimming or
    fallback selection.  Keeping this separate from :func:`response_content` preserves
    compatibility for non-policy callers while giving Graph Director an auditable raw
    boundary.
    """

    # vLLM/Qwen deployments have used both ``reasoning_content`` and
    # ``reasoning`` for the separate private channel.  Treat them identically
    # so a one-request reasoning+action response is parsed consistently.
    reasoning = str(
        getattr(message, "reasoning_content", "")
        or getattr(message, "reasoning", "")
        or ""
    )
    content = str(getattr(message, "content", "") or "")
    # Some Qwen/vLLM combinations return the reasoning channel inline rather
    # than in ``reasoning_content``. Preserve every provider-authored character,
    # but expose the suffix after the explicit closing tag as the typed-action
    # parser input. Concatenating the two parts exactly reconstructs content.
    if not reasoning:
        opening = content.find("<think>")
        closing = content.find("</think>", opening + len("<think>") if opening >= 0 else 0)
        # Qwen's generation prefix already contains <think> when thinking is
        # enabled. In that case only its closing delimiter is generated. The
        # caller must attest the request mode; do not infer this from arbitrary
        # non-thinking content containing a literal closing tag.
        if (opening >= 0 or thinking_prefilled) and closing >= 0:
            boundary = closing + len("</think>")
            reasoning, content = content[:boundary], content[boundary:]
        elif thinking_prefilled:
            # A cut-off reasoning channel is not an action even if it quotes
            # exactly one syntactically valid JSON example.
            reasoning, content = content, ""
    return reasoning, content


def recover_qwen_policy_parts(
    message: Any,
    *,
    completion_token_ids: Sequence[int],
    content_logprobs: Sequence[Any],
) -> tuple[str, str] | None:
    """Recover a tool-parser-stripped thinking boundary from the same sample.

    Some vLLM tool parsers remove ``</think>`` from message.content even when
    no reasoning parser is configured. The per-token logprob bytes still
    contain that sampled delimiter. Accept them only when all sampled tokens
    are present and removing that one delimiter reproduces message.content
    exactly. Never search reasoning for a JSON action or infer a boundary.
    """

    if getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None):
        return None
    content = str(getattr(message, "content", "") or "")
    if "</think>" in content or not completion_token_ids:
        return None
    if len(completion_token_ids) != len(content_logprobs):
        return None
    pieces = []
    for item in content_logprobs:
        token_bytes = getattr(item, "bytes", None)
        if token_bytes is None or getattr(item, "logprob", None) is None:
            return None
        try:
            pieces.append(bytes(token_bytes))
        except (TypeError, ValueError):
            return None
    # Require the delimiter to be a sampled token, not a literal string
    # assembled from ordinary tokens inside a quoted example.
    if pieces.count(b"</think>") != 1:
        return None
    # vLLM includes the sampled EOS in token IDs/logprobs but omits it from
    # message.content. Remove only a known, complete, final Qwen stop token
    # for this comparison; retain the original IDs/logprobs in the response.
    if pieces[-1] in {b"<|im_end|>", b"<|endoftext|>"}:
        pieces = pieces[:-1]
    try:
        raw_content = b"".join(pieces).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if raw_content.count("</think>") != 1:
        return None
    if raw_content.replace("</think>", "", 1) != content:
        return None
    boundary = raw_content.index("</think>") + len("</think>")
    return raw_content[:boundary], raw_content[boundary:]


def encode_chat_trajectory(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Tokenize the exact chat history and mark assistant spans.

    Qwen3.5's bundled template does not expose a ``generation`` block, so
    ``return_assistant_tokens_mask`` is all-zero.  It also renders the final
    assistant turn differently from earlier turns.  We therefore locate each
    assistant payload in the fully rendered template and map its character
    range back to tokens.  This keeps rollout and training tokenization aligned.
    """

    items = [dict(message) for message in messages]
    kwargs = {"enable_thinking": bool(enable_thinking)}
    rendered = tokenizer.apply_chat_template(
        items,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = _chat_template_ids(encoded)
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and offsets and isinstance(offsets[0][0], list):
        offsets = offsets[0]

    mask = [0] * len(token_ids)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in items:
        # Qwen3.5's template applies Jinja ``trim`` to textual message content.
        # Locate that rendered form so harmless boundary whitespace does not
        # make an otherwise valid real rollout impossible to train.
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        content_start = rendered.find(content, cursor)
        if content_start < 0:
            raise ValueError("message content was transformed by the Qwen chat template")
        content_end = content_start + len(content)
        cursor = content_end
        if message.get("role") != "assistant":
            continue
        terminal_start = rendered.find("<|im_end|>", content_end)
        char_end = terminal_start + len("<|im_end|>") if terminal_start >= 0 else content_end
        positions = [
            position
            for position, (start, end) in enumerate(offsets)
            if end > content_start and start < char_end
        ]
        if not positions:
            raise ValueError("assistant payload produced no trainable Qwen tokens")
        start, end = positions[0], positions[-1] + 1
        for position in range(start, end):
            mask[position] = 1
        spans.append((start, end))
    return token_ids, mask, spans


def load_training_model(model_path: str | Path, *, dtype: Any) -> Any:
    """Load either a text-only or multimodal Qwen3.5 checkpoint correctly."""

    from transformers import AutoConfig, AutoModelForCausalLM

    source = str(model_path)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    architectures = set(getattr(config, "architectures", ()) or ())
    if any(name.endswith("ForConditionalGeneration") for name in architectures):
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                source,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except (ImportError, ValueError):
            pass
    return AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
