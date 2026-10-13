"""Reference-free, conservative parsing of explicitly submitted AIME answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class UnscorableAnswerError(ValueError):
    """An intermediate artifact cannot be scored as a final task answer."""


@dataclass(frozen=True)
class AIMEAnswer:
    answer: str = ""
    reason: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.answer)


def is_aime_dataset(dataset: object) -> bool:
    return "aime" in str(dataset).strip().casefold()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _integer(value: str) -> AIMEAnswer:
    if not re.fullmatch(r"[0-9]{1,3}", value):
        return AIMEAnswer(reason="not_an_aime_integer")
    return AIMEAnswer(answer=value)


_BOX = r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}"
_LABEL = r"(?:(?:final|short)\s+)?answer\s*(?:is\b|[=:：])|答案\s*(?:是|为|[=:：])"
_AMBIGUOUS = (
    r"\b(?:or|maybe|possibly|either|actually|instead|correction|corrected|retract|"
    r"not|example|sample|candidate|draft|quoted)\b|可能|或者|或|改口|更正|修正|其实|不是|示例|例如"
)


def _presentation(text: str) -> str:
    """Unwrap only a balanced whole-value presentation, never strip arbitrary markup."""
    for pattern in (
        r"\$\$([^$]*)\$\$",
        r"\$([^$]*)\$",
        r"\\\((.*?)\\\)",
        r"\\\[(.*?)\\\]",
        r"\*\*([^*]*)\*\*",
    ):
        match = re.fullmatch(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def _safe_tail(tail: str) -> bool:
    if tail.strip() in {"", ".", "。"}:
        return True
    # Do not silently ignore a second line or a revised answer. Extra prose is
    # supported only as a labelled explanation, not an arbitrary continuation.
    explanation = re.fullmatch(
        r"(?:\.\s+|。\s*|\s*\n\s*)"
        r"((?:checks?\b|verification\b|explanation\b|reasoning\b|evidence\b|"
        r"检查|验证|解释|推导|说明)[\s\S]*)",
        tail.lstrip(" \t"),
        re.IGNORECASE,
    )
    return bool(
        explanation
        and not re.search(
            rf"{_AMBIGUOUS}|{_LABEL}|\\boxed|```|[<>]",
            explanation.group(1),
            re.IGNORECASE,
        )
    )


def _marked_value(value: str) -> AIMEAnswer:
    value = _presentation(value.strip())
    match = re.match(rf"(?:{_BOX}|([0-9]{{1,3}}))", value)
    if match is None or not _safe_tail(value[match.end() :]):
        return AIMEAnswer(reason="ambiguous_or_invalid_final_answer")
    return _integer(match.group(1) or match.group(2))


def _reject_json_constant(value: str):
    raise ValueError(f"nonstandard JSON constant: {value}")


def parse_aime_answer(raw: str) -> AIMEAnswer:
    """Accept a scalar or explicit final declaration; never scan for last digits.

    This checks submission syntax only, not mathematical correctness. No task,
    reference, summary or evidence is accepted by this API.
    """
    if not isinstance(raw, str):
        return AIMEAnswer(reason="non_text_answer")
    if len(raw) > 65536:
        return AIMEAnswer(reason="answer_too_long")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if not text:
        return AIMEAnswer(reason="missing_final_answer")
    if text.startswith(("{", "[")):
        try:
            payload = json.loads(
                text, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant
            )
        except (ValueError, TypeError, RecursionError):
            return AIMEAnswer(reason="invalid_answer_json")
        if not isinstance(payload, dict):
            return AIMEAnswer(reason="non_scalar_answer")
        if "action_calls" in payload or "tool_calls" in payload:
            # A proposed tool request is not an observed final answer. Keep
            # its context intact instead of freezing a nested answer literal.
            return AIMEAnswer(reason="unexecuted_action_envelope")
        values = [payload[key] for key in ("answer", "final_answer") if key in payload]
        if not values:
            return AIMEAnswer(reason="missing_final_answer")
        parsed = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                return AIMEAnswer(reason="non_scalar_answer")
            # Explicit JSON answer fields must themselves contain only an integer
            # or a boxed integer, not another document or an inferred answer.
            candidate = _presentation(str(value).strip())
            boxed = re.fullmatch(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}", candidate)
            parsed.append(_integer(boxed.group(1) if boxed else candidate))
        return _consistent(parsed)
    if "```" in text or re.search(
        r"[<>\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", text
    ):
        return AIMEAnswer(reason="unsupported_markup_or_control_character")
    text = _presentation(text)
    bold_label = re.fullmatch(rf"\*\*({_LABEL})\*\*\s*(.*)", text, re.IGNORECASE)
    if bold_label:
        text = bold_label.group(1) + " " + bold_label.group(2)
    if re.fullmatch(r"[0-9]+", text):
        return _integer(text)
    marked = list(
        re.finditer(
            rf"(?:^|(?<=[.。!?！？]))[ \t]*(?:the\s+)?(?:{_LABEL})[ \t]*",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    candidates = []
    if marked:
        prefix = text[: marked[0].start()]
        if re.search(rf"{_LABEL}|\\boxed|{_AMBIGUOUS}", prefix, re.IGNORECASE):
            return AIMEAnswer(reason="ambiguous_answer_context")
        for index, match in enumerate(marked):
            end = marked[index + 1].start() if index + 1 < len(marked) else len(text)
            candidates.append(_marked_value(text[match.end() : end]))
    else:
        boxed = list(re.finditer(_BOX, text))
        if len(boxed) != len(re.findall(r"\\boxed", text)):
            return AIMEAnswer(reason="invalid_or_truncated_box")
        if boxed:
            prefix = text[: boxed[0].start()].strip()
            if prefix and (
                re.search(rf"{_AMBIGUOUS}|{_LABEL}", prefix, re.IGNORECASE)
                or not re.search(
                    r"(?:[.。]|\b(?:therefore|thus|hence)[,:]?|\bfinal\s+is)$",
                    prefix,
                    re.IGNORECASE,
                )
            ):
                return AIMEAnswer(reason="ambiguous_answer_context")
            for index, match in enumerate(boxed):
                end = boxed[index + 1].start() if index + 1 < len(boxed) else len(text)
                if not _safe_tail(text[match.end() : end]):
                    return AIMEAnswer(reason="ambiguous_or_invalid_final_answer")
                candidates.append(_integer(match.group(1)))
    if not candidates:
        return AIMEAnswer(reason="missing_explicit_final_answer")
    return _consistent(candidates)


def _consistent(candidates: list[AIMEAnswer]) -> AIMEAnswer:
    for candidate in candidates:
        if not candidate.valid:
            return candidate
    if len({int(candidate.answer) for candidate in candidates}) != 1:
        return AIMEAnswer(reason="conflicting_final_answers")
    return candidates[0]
