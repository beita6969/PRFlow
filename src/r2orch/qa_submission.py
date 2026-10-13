"""Reference-blind QA extraction: one committed answer, never gold-selected guesses."""

from __future__ import annotations

import json
import re
import string
import unicodedata


def is_short_qa_dataset(value: object) -> bool:
    dataset = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return dataset in {
        "nq",
        "nqopen",
        "naturalquestions",
        "naturalquestionsopen",
        "hotpot",
        "hotpotqa",
    }


def normalize_qa_answer(value: str) -> str:
    """Preserve the project's existing Unicode/whitespace normalization."""
    lowered = unicodedata.normalize("NFKC", value).casefold()
    plain = "".join(
        " " if c in string.punctuation or unicodedata.category(c).startswith("P") else c
        for c in lowered
    )
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", plain).split())


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def extract_qa_answer(value: str) -> str:
    """Unwrap explicit declarations, rejecting conflicting or structured answers.

    No reference, date-range widening, semantic answer selection, or last-answer
    preference is involved. Unmarked natural language remains unchanged for EM.
    """
    raw = str(value).strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json|text)?\s*\n?", "", raw, flags=re.I)[:-3].strip()
    if raw.startswith("["):
        return ""
    declarations: list[str] = []
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    # Consume complete JSON objects before label parsing, including contradictions
    # across multiple objects. Never interpret metadata fields as answer text.
    remaining = raw
    offset = 0
    while offset < len(remaining):
        start = remaining.find("{", offset)
        if start < 0:
            break
        try:
            payload, length = decoder.raw_decode(remaining[start:])
        except ValueError:
            if re.match(r'\{\s*"(?:answer|final_answer)"\s*:', remaining[start:]):
                return ""
            offset = start + 1
            continue
        found = False
        if isinstance(payload, dict):
            for key in ("answer", "final_answer"):
                if key not in payload:
                    continue
                answer = payload[key]
                if not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
                    return ""
                answer = str(answer).strip()
                if not answer:
                    return ""
                declarations.append(answer)
                found = True
        if found:
            remaining = remaining[:start] + " " * length + remaining[start + length :]
        offset = start + length
    boxed = re.compile(r"\\boxed\s*\{([^{}]+)\}")
    declarations.extend(m.group(1).strip() for m in boxed.finditer(remaining))
    # Replacing a box by its content makes `Final answer: \\boxed{X}` and X agree.
    remaining = boxed.sub(lambda m: m.group(1), remaining)
    label = re.compile(
        r"(?<!\w)(?:final\s+answer|short\s+answer|answer|答案)\s*"
        r"(?:is\b|=|:|：)\s*([^\n\r]+)",
        re.I,
    )
    declarations.extend(m.group(1).strip() for m in label.finditer(remaining))
    if not declarations:
        # A JSON array is not a committed scalar QA answer.
        return raw
    normalized = {normalize_qa_answer(answer) for answer in declarations}
    if len(normalized) != 1 or not next(iter(normalized)):
        return ""
    return declarations[0]
