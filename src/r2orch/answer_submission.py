from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .aime_submission import is_aime_dataset, parse_aime_answer
from .hotpot_answer_contract import hotpot_submission_instruction
from .observability import AutoVerifier, TaskSpec, Verifier
from .qa_submission import extract_qa_answer, is_short_qa_dataset


@dataclass(frozen=True)
class AnswerSubmission:
    """Submission payload; valid means nonempty, not correct or well-formatted.

    Runtime integrity is checked separately before FINISH. Dataset answer-format
    and correctness checks belong to the verifier, including for raw fallbacks.
    """

    raw_answer: str
    submitted_answer: str
    method: str
    changed: bool
    valid: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerFinalizer:
    """Deterministically extract the selected output without model calls.

    Summaries, reference answers and evidence passages never supply replacement
    candidates. Ambiguous nonempty answers are preserved for grading.
    """

    def finalize(
        self,
        task: TaskSpec,
        raw_answer: str,
        *,
        raw_summary: str = "",
        verifier: Verifier | None = None,
    ) -> AnswerSubmission:
        raw = str(raw_answer or "").strip()
        # Normalize an unambiguous valid scalar, otherwise preserve the entire
        # nonempty answer for grading. Never repair, truncate or choose a number.
        # AIME never reads the summary/reference.
        if is_aime_dataset(task.metadata.get("dataset", "")):
            concrete_verifier = verifier.resolve(task) if isinstance(verifier, AutoVerifier) else verifier
            if getattr(concrete_verifier, "name", "") == "integer_answer":
                from .benchmark_scoring import _math_prediction

                prediction = _math_prediction(raw)
                submitted = format(prediction, "f") if prediction is not None else raw
                return AnswerSubmission(
                    raw_answer=raw,
                    submitted_answer=submitted,
                    method=("aime_integer_scalar_submission" if prediction is not None
                            else "aime_integer_raw_submission"),
                    changed=submitted != raw,
                    valid=bool(submitted),
                    detail="" if prediction is not None else "missing_trusted_final_math_answer",
                )
            parsed = parse_aime_answer(raw)
            submitted = parsed.answer if parsed.valid else raw
            return AnswerSubmission(
                raw_answer=raw,
                submitted_answer=submitted,
                method="aime_parsed_scalar_submission" if parsed.valid else "aime_parsed_raw_submission",
                changed=submitted != raw,
                valid=bool(submitted),
                detail=parsed.reason,
            )

        kind = submission_kind(task)
        fallback, fallback_method = _deterministic_submission(kind, raw)
        if kind == "qa" and not fallback:
            # Conflicting declarations are a gradable answer failure, not a
            # missing runtime artifact. The QA verifier must see all of them;
            # selecting one here could turn an ambiguous answer into a success.
            return AnswerSubmission(
                raw_answer=raw, submitted_answer=raw, method="qa_raw_submission",
                changed=False, valid=bool(raw), detail="ambiguous_or_unextractable_answer",
            )
        return _submission(raw, fallback, method=fallback_method)


def submission_kind(task: TaskSpec) -> str:
    if is_short_answer_qa(task):
        return "qa"
    task_type = task.task_type.casefold()
    if any(value in task_type for value in ("multiple_choice", "choice", "mcq")):
        return "multiple_choice"
    return "passthrough"


def is_short_answer_qa(task: TaskSpec) -> bool:
    """Dataset identity only: never infer an answer type from private labels."""
    return is_short_qa_dataset(task.metadata.get("dataset", ""))


def submission_contract(task: TaskSpec) -> str:
    dataset = str(task.metadata.get("dataset", "")).strip().casefold()
    if dataset == "hotpotqa":
        return hotpot_submission_instruction()
    if dataset == "mbppplus":
        return (
            "Submission contract: The selected output Agent puts the complete Python code in "
            "the answer field: the function with the task's function name, plus the imports and "
            "helpers it needs, without tests or prints. Put explanations in summary or evidence."
        )
    if is_aime_dataset(task.metadata.get("dataset", "")):
        return (
            "Submission contract: The selected output Agent must put one final integer "
            "from 0 to 999 in the answer field. Put derivations in summary or evidence. "
            "Intermediate Agents may retain local findings. Out-of-range, non-integer, "
            "list or ambiguous final answers receive zero from the AIME verifier; "
            "FINISH does not repair them or certify their format."
        )
    kind = submission_kind(task)
    if kind == "multiple_choice":
        return (
            "Submission contract: Put only the final option label in the answer field. Put all "
            "reasoning in summary or evidence."
        )
    if kind == "qa":
        return (
            "Submission contract: The selected output Agent must put only the shortest entity, "
            "date, location, title, or phrase that directly answers the question in the answer "
            "field. Do not add explanations, "
            "appositives or related facts there; put them in summary or evidence. Preserve units "
            "and qualifications needed to express the answer without changing its meaning. "
            "Intermediate Agents may retain local findings for their assigned responsibilities; "
            "they are not required to answer the entire question."
        )
    return (
        "Submission contract: Put only the direct task result in the answer field and place all "
        "supporting explanation in summary or evidence."
    )


def qa_token_f1(task: TaskSpec, prediction: str) -> float | None:
    if submission_kind(task) != "qa" or task.reference is None:
        return None
    references = (
        list(task.reference) if isinstance(task.reference, (list, tuple, set)) else [task.reference]
    )
    predicted = _qa_tokens(prediction)
    if not predicted:
        return 0.0
    best = 0.0
    for reference in references:
        expected = _qa_tokens(str(reference))
        if not expected:
            continue
        remaining = list(expected)
        overlap = 0
        for token in predicted:
            if token in remaining:
                remaining.remove(token)
                overlap += 1
        if not overlap:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def _deterministic_submission(kind: str, raw: str) -> tuple[str, str]:
    if kind == "qa":
        return extract_qa_answer(raw), "qa_deterministic_extraction"
    explicit = _explicit_answer(raw)
    if kind == "multiple_choice":
        direct = re.fullmatch(r"\s*\(?([A-Z])\)?[.)]?\s*", explicit, flags=re.IGNORECASE)
        marked = re.findall(
            r"(?i)(?:option|choice|answer)\s*(?:is|=|:)?\s*\(?([A-Z])\)?",
            raw,
        )
        choice = direct.group(1) if direct else (marked[-1] if marked else "")
        return (choice.upper() if choice else explicit), "choice_extraction"
    return raw, "passthrough"


def _explicit_answer(raw: str) -> str:
    payload = _json_payload(raw)
    if isinstance(payload, dict):
        value = payload.get("answer", payload.get("final_answer"))
        if value not in (None, ""):
            return str(value).strip()
    matches = re.findall(
        r"(?i)(?:final\s+answer|short\s+answer|answer|答案)\s*(?:is|=|:|：)?\s*([^\n\r]+)",
        raw,
    )
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    return boxed[-1].strip() if boxed else raw.strip()


def _json_payload(text: str) -> Any:
    stripped = str(text).strip()
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except ValueError:
            continue
        return payload
    return None


def _qa_tokens(value: str) -> list[str]:
    lowered = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if char in string.punctuation or unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return re.sub(r"\s+", " ", without_articles).strip().split()


def _submission(raw: str, submitted: str, *, method: str) -> AnswerSubmission:
    return AnswerSubmission(
        raw_answer=raw,
        submitted_answer=submitted,
        method=method,
        changed=submitted != raw,
        valid=bool(submitted),
    )
