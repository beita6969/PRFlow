"""Answer scoring of the six benchmarks: integer answers (AIME), exact match (HotpotQA, NQ-Open), letter choice
(MedQA) and the MBPP+ tests.

``_code_score``/``_passes`` take the Python executable that runs MBPP+ tests
(``r2orch.paths.evalplus_python``). Math extraction also requires a
declared final candidate instead of guessing from the last digits of an
unfinished derivation. The verifier classes at the end adapt the scores to
this project's ``Verifier`` protocol and ``TaskSpec`` rows.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import re
import subprocess
import symtable
import sys
import tempfile
import textwrap
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

from .aime_submission import UnscorableAnswerError
from .observability import TaskSpec, VerificationResult

# Inline pairing, used only when a reply has no fence line (``` on a line of its own).
_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


# A fence line: three or more backticks, then an optional info string that
# (CommonMark) holds no backtick; its first word is the language tag.
_FENCE_LINE = re.compile(r"^\s*(`{3,})([^`]*)$")


_PYTHON_TAGS = frozenset({"", "python", "py", "python3", "py3"})


def _hotpot_norm(text: str) -> str:
    """SQuAD-style answer normalization: lower case, punctuation and articles removed.

    Unlike the official SQuAD script it also removes non-ASCII punctuation.
    """
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    normalized_prediction, normalized_answer = _hotpot_norm(prediction), _hotpot_norm(answer)
    # Official HotpotQA rule: yes/no/noanswer only match exactly.
    special = {"yes", "no", "noanswer"}
    if (normalized_prediction in special or normalized_answer in special) and (
        normalized_prediction != normalized_answer
    ):
        return 0.0
    p, g = normalized_prediction.split(), normalized_answer.split()
    if not p or not g:
        return float(p == g)
    overlap = sum((__import__("collections").Counter(p) & __import__("collections").Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def _last_number(text: str) -> Decimal | None:
    numbers = _NUMBER.findall(text.replace("\\!", "").replace("{,}", ","))
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


_MATH_CUE = re.compile(
    r"\b(?:final\s+answer|answer)[*_\s]*(?:is\b[*_\s]*[:：]?|[:：=])"
    r"|(?:最终)?答案\s*(?:是|为|[:：=])",
    re.IGNORECASE,
)
_MATH_CONCLUSION = re.compile(
    r"\b(?:(?:therefore|thus|hence|so)[,:]?\s+(?:the\s+)?"
    r"(?:(?:required|requested|final)\s+)?(?:answer|result|value|sum|total)"
    r"|(?:the\s+)?(?:final|required|requested)\s+(?:result|value|sum|total))"
    r"\s*(?:is\b|equals\b|[:=])",
    re.IGNORECASE,
)
_MATH_UNCERTAIN = re.compile(
    r"\b(?:or|maybe|possible|possibly|either|candidate|example|sample|quoted|not"
    r"|unknown|undetermined|unsolved|unresolved|incomplete|intermediate|draft"
    r"|cannot|unable|never)\b"
    r"|可能|或者|或许|示例|例如|不是|未知|无法",
    re.IGNORECASE,
)
_MATH_BOX_UNCERTAIN = re.compile(
    r"\b(?:or|maybe|possible|possibly|either|candidate|draft|intermediate"
    r"|unknown|undetermined|unresolved|incomplete)\b|未知|可能|或者|或许",
    re.IGNORECASE,
)
_MATH_PROGRAM_LINE = re.compile(
    r"(?m)^\s*(?:import\s|from\s\S+\s+import\s|(?:async\s+)?def\s|class\s"
    r"|return\b|(?:print|exec|eval)\s*\(|[A-Za-z_]\w*\s*=\s*[^=]).*$"
)


def _math_scalar(text: str) -> Decimal | None:
    """Read a whole presented number, never a constant embedded in an expression."""
    value = text.strip()
    for pattern in (r"\$\$([^$]+)\$\$", r"\$([^$]+)\$", r"\\\((.*?)\\\)",
                    r"\\\[(.*?)\\\]", r"\*\*([^*]+)\*\*", r"`([^`]+)`"):
        wrapped = re.fullmatch(pattern, value, re.DOTALL)
        if wrapped:
            value = wrapped.group(1).strip()
            break
    if value.endswith((".", "。")) and _NUMBER.fullmatch(value[:-1]):
        value = value[:-1]
    if not _NUMBER.fullmatch(value):
        return None
    return _last_number(value)


def _math_json_values(text: str) -> tuple[list[object], str]:
    """Read complete root answer fields, including a missing outer closing brace.

    Parsing values rather than searching for an ``answer`` substring prevents
    source strings and nested tool arguments from supplying answer candidates.
    No malformed or incomplete field value is repaired.
    """
    decoder = json.JSONDecoder()
    position = 1  # The caller has already checked the opening object brace.
    values: list[object] = []
    keys: set[str] = set()
    while position < len(text):
        position += len(text[position:]) - len(text[position:].lstrip())
        key = None
        try:
            key, position = decoder.raw_decode(text, position)
            if not isinstance(key, str) or key in keys:
                return [], text[position:]
            keys.add(key)
            position += len(text[position:]) - len(text[position:].lstrip())
            if text[position:position + 1] != ":":
                break
            position += 1
            position += len(text[position:]) - len(text[position:].lstrip())
            value, position = decoder.raw_decode(text, position)
        except (ValueError, RecursionError):
            if key in {"answer", "final_answer"}:
                # A second declared final field that cuts off cannot certify
                # the earlier candidate as the final answer.
                return [], text[position:]
            break
        if key in {"action_calls", "tool_calls"}:
            return [], text[position:]
        if key in {"answer", "final_answer"}:
            values.append(value)
        position += len(text[position:]) - len(text[position:].lstrip())
        if text[position:position + 1] != ",":
            break
        position += 1
    tail = text[position:]
    if re.search(r'(?<!\\)"(?:answer|final_answer)"\s*:', tail):
        # Preserve a complete field in a damaged outer wrapper, but never hide
        # another explicit candidate behind the damaged syntax or a second object.
        return [], tail
    return values, tail


def _math_terminal_declaration(prose: str) -> Decimal | None:
    """Read a clear numeric conclusion at the end, not an arbitrary last digit."""
    connectors = list(re.finditer(r"\b(?:therefore|thus|hence|so)[,:]?\s+", prose, re.I))
    if connectors:
        candidate = _math_scalar(prose[connectors[-1].end():])
        if candidate is not None:
            return candidate
    postfix = re.search(
        r"(?<![\w.+*/=\\-])(" + _NUMBER.pattern + r")\s+is\s+(?:the\s+)?"
        r"(?:final\s+)?(?:answer|result|requested\s+(?:sum|value|total))\s*[.!。]?\s*$",
        prose, re.IGNORECASE,
    )
    return _math_scalar(postfix.group(1)) if postfix else None


def _math_prose(text: str) -> str:
    # Fenced source, including an unfinished fence, is a proposal to execute a
    # program. Its literals and printed labels are not observed final answers.
    text = re.sub(r"```.*?(?:```|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"`(" + _NUMBER.pattern + r")`", r"\1", text)
    text = re.sub(r"`[^`\n]*`", "", text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        tree = None
    if tree is not None and tree.body:
        # A lone "Answer: 193" / "Answer = 193" or "Therefore, 193."
        # happens to parse as Python. It is still an answer declaration.
        declared = _MATH_CUE.match(text.strip()) or _math_terminal_declaration(text) is not None
        if len(tree.body) != 1 or not declared:
            return ""
        return text
    return _MATH_PROGRAM_LINE.sub("", text)


def _math_declared_number(span: str) -> Decimal | None:
    line = next((line.strip() for line in span.splitlines() if line.strip()), "")
    line = line.replace("{,}", ",")
    if not line or _MATH_UNCERTAIN.search(line):
        return None
    match = _NUMBER.search(line[:200])
    if match is None:
        return None
    # Natural phrasing ("Answer: Tom spends $45 in 10 days") remains readable,
    # but "Answer: x = 7r+16" or "Answer: 16/2" is an intermediate expression.
    prefix, tail = line[:match.start()], line[match.end():]
    if re.search(r"[=+*/{}\\]|\b(?:print|return|code)\b", prefix, re.IGNORECASE):
        return None
    if re.match(r"[A-Za-z_]|\s*[-+*/=\\]", tail):
        return None
    return _last_number(match.group(0))


_EMPTY_MATH_ACTION_LINE = re.compile(
    r'(?m)^[ \t]*(\{\s*"(?:action_calls|tool_calls)"\s*:\s*\[\s*\]\s*'
    r'(?:,\s*"(?:action_calls|tool_calls)"\s*:\s*\[\s*\]\s*)*\})'
    r'[ \t]*(?:\r?\n|$)'
)
_QUOTED_MATH_ACTION_TEMPLATE = re.compile(
    r'''(?P<quote>["'`“‘])(?P<lead>[^"'`“”‘’\r\n]*?)'''
    r'''(?P<template>\{[ \t]*"(?:action_calls|tool_calls)"[ \t]*:[ \t]*'''
    r'''(?:\[[ \t]*\.{3}[ \t]*\]|\.{3})[ \t]*\})'''
    r'''[^"'`“”‘’\r\n]*(?P<closing>["'`”’])'''
)
_MATH_TEMPLATE_QUOTE_PAIRS = {'"': '"', "'": "'", "`": "`", "“": "”", "‘": "’"}
_MATH_ACTION_TEMPLATE_CONTEXT = re.compile(
    r"\b(?:prompt|instructions?|format|schema|examples?|templates?)\b|提示|指令|格式|示例|模板",
    re.IGNORECASE,
)
_MATH_TEMPLATE_SOURCE_PREFIX = re.compile(
    r"^\s*(?:(?:await|raise|yield)\s+)?[A-Za-z_]\w*"
    r"(?:\.[A-Za-z_]\w*|\[[^\]\n]*\])*\s*"
    r"(?:\(|(?::[^=\n]+)?(?:=(?!=)|\+=|-=|\*=|/=|:=))"
)
_MATH_TEMPLATE_EXECUTION_REQUEST = re.compile(
    r"\b(?:(?:now|please)\s+(?:execute|run|invoke|call|dispatch)"
    r"|(?:I|we)\s+(?:will|shall|must|should)\s+(?:execute|run|invoke|call|dispatch)"
    r"|use\s+(?:this|the)\s+template\s+now|(?:next|pending)\s+tool\s+request)\b"
    r"|现在(?:运行|执行)|请(?:运行|执行)|(?:我|我们)(?:会|将|要)(?:运行|执行)",
    re.IGNORECASE,
)


def _math_template_is_source_line(line: str, quote_span: tuple[int, int]) -> bool:
    """Keep proposed program strings, including unfinished call/assignment lines."""
    if _MATH_PROGRAM_LINE.match(line) or _MATH_TEMPLATE_SOURCE_PREFIX.match(line):
        return True
    left, right = quote_span
    for source in (line, line[:left] + '"displayed template"' + line[right:]):
        try:
            tree = ast.parse(source.strip())
        except (SyntaxError, ValueError, RecursionError):
            continue
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.AnnAssign):
            statement = tree.body[0]
            # ``Example: '...'`` is also valid Python syntax, but presents a
            # format label rather than assigning a program value.
            if (statement.value is None and isinstance(statement.target, ast.Name)
                    and statement.target.id.lower() in {"example", "format", "schema", "template", "prompt"}
                    and isinstance(statement.annotation, ast.Constant)
                    and isinstance(statement.annotation.value, str)):
                continue
        if tree.body:
            return True
    return False


def _math_empty_action_envelope(text: str) -> bool:
    """Only an entire empty transport object has no execution or answer content."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return False
    return (
        isinstance(value, dict)
        and bool(value)
        and set(value) <= {"action_calls", "tool_calls"}
        and all(isinstance(calls, list) and not calls for calls in value.values())
    )


def _math_without_empty_action_envelopes(text: str) -> str:
    """Ignore standalone empty transport wrappers while preserving source fences.

    A Worker may append ``{"action_calls":[]}`` to its completed prose answer,
    or quote the prompt's ``{"action_calls":[...]}`` format template. Neither
    requests execution. Actual calls and unquoted or incomplete action objects
    retain the action-key guard in ``_math_prediction`` below.
    """
    # A standalone-looking object can still be a literal inside a proposed
    # multiline program string. Keep only that quoted range protected; an
    # unrelated triple-quoted explanation must not reject the whole reply.
    quoted_ranges: list[tuple[int, int]] = []
    for opening in re.finditer(r"(?<!\\)(?:\"\"\"|''')", text):
        if quoted_ranges and opening.start() < quoted_ranges[-1][1]:
            continue
        closing = re.search(r"(?<!\\)" + re.escape(opening.group()), text[opening.end():])
        end = opening.end() + closing.end() if closing else len(text)
        quoted_ranges.append((opening.start(), end))

    def overlaps_quote(start: int, end: int) -> bool:
        return any(start < right and end > left for left, right in quoted_ranges)

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result: list[str] = []
    ordinary: list[str] = []

    def flush_ordinary() -> None:
        if not ordinary:
            return
        span = "".join(ordinary)
        start = offsets[position - len(ordinary)]
        removals: list[tuple[int, int, str]] = []
        for match in _EMPTY_MATH_ACTION_LINE.finditer(span):
            if (not overlaps_quote(start + match.start(), start + match.end())
                    and _math_empty_action_envelope(match.group(1))):
                removals.append((match.start(), match.end(), "\n"))
        for match in _QUOTED_MATH_ACTION_TEMPLATE.finditer(span):
            line_start = span.rfind("\n", 0, match.start()) + 1
            line_end = span.find("\n", match.end())
            line_end = len(span) if line_end < 0 else line_end
            line = span[line_start:line_end]
            prefix = span[line_start:match.start()]
            outside_quote = prefix + span[match.end():line_end]
            display_label = bool(re.fullmatch(
                r"\s*(?:example|format|schema|template|prompt)\s*:\s*", prefix, re.I,
            )) and not span[match.end():line_end].strip(" \t.;。")
            preceding_lines = span[:line_start].rstrip().splitlines()
            source_heading = any(re.search(
                r"\b(?:source|code|program)(?:\s+snippet)?\s*:\s*$", earlier_line, re.I,
            ) for earlier_line in preceding_lines)
            if (not overlaps_quote(start + match.start(), start + match.end())
                    and _MATH_TEMPLATE_QUOTE_PAIRS[match.group("quote")] == match.group("closing")
                    and _MATH_ACTION_TEMPLATE_CONTEXT.search(prefix + match.group("lead"))
                    and not prefix.lstrip().startswith(("{", "[", "#"))
                    and not _math_template_is_source_line(
                        line, (match.start() - line_start, match.end() - line_start),
                    )
                    and not source_heading
                    and not _MATH_TEMPLATE_EXECUTION_REQUEST.search(outside_quote)):
                if display_label:
                    removals.append((line_start, line_end, "Quoted action format."))
                else:
                    removals.append((*match.span("template"), "[quoted action format]"))
        for left, right, replacement in sorted(removals, reverse=True):
            span = span[:left] + replacement + span[right:]
        result.append(span)
        ordinary.clear()

    position = 0
    while position < len(lines):
        opening = _FENCE_LINE.match(lines[position].rstrip("\r\n"))
        if opening is None:
            ordinary.append(lines[position])
            position += 1
            continue
        flush_ordinary()
        end = position + 1
        while end < len(lines):
            closing = _FENCE_LINE.match(lines[end].rstrip("\r\n"))
            if closing and len(closing.group(1)) >= len(opening.group(1)) and not closing.group(2).strip():
                break
            end += 1
        body = "".join(lines[position + 1:end])
        if (end < len(lines) and not overlaps_quote(offsets[position], offsets[end + 1])
                and opening.group(2).strip().lower() in {"", "json"}
                and _math_empty_action_envelope(body)):
            result.append("\n")
        else:
            result.extend(lines[position:min(end + 1, len(lines))])
        position = end + 1
    flush_ordinary()
    return "".join(result).strip()


def _math_prediction(output: str, *, _depth: int = 0) -> Decimal | None:
    """A reference-free final candidate shared by Worker extraction and scoring.

    Accept a complete answer field, boxed scalar, bare number, numeric ``####``
    line, or a clear answer/conclusion in natural text. Never guess from the
    last number of a derivation, source program or unexecuted action envelope.
    """
    if _depth > 4 or not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None
    scalar = _math_scalar(text)
    if scalar is not None:
        return scalar
    text = _math_without_empty_action_envelopes(text)
    if not text:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if fenced and fenced.group(1).lstrip().startswith("{"):
        text = fenced.group(1).strip()
    if re.search(r'"(?:action_calls|tool_calls)"\s*:', text):
        return None
    if text.startswith("{"):
        values, tail = _math_json_values(text)
        predictions = [
            _math_prediction(str(value), _depth=_depth + 1)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None
            for value in values
        ]
        if not predictions or any(value is None for value in predictions):
            return None
        if len(set(predictions)) != 1:
            return None
        suffix = _math_prediction(tail.lstrip("} \t\r\n"), _depth=_depth + 1)
        return predictions[0] if suffix is None or suffix == predictions[0] else None
    if text.startswith("["):
        return None
    prose = _math_prose(text)
    prose = re.sub(r"\*\*(" + _NUMBER.pattern + r")\*\*", r"\1", prose)
    boxed = list(re.finditer(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", prose))
    if boxed:
        final_box = boxed[-1]
        candidate = _math_scalar(final_box.group(1))
        # A boxed intermediate or alternative is not certified by its styling.
        # Limit this guard to its sentence so earlier failed reasoning does not
        # hide a later clear final answer.
        start = max(prose.rfind(boundary, 0, final_box.start()) for boundary in "\n.;。") + 1
        tail_end = re.search(r"[\n.;。]", prose[final_box.end():])
        end = final_box.end() + tail_end.start() if tail_end else len(prose)
        if _MATH_BOX_UNCERTAIN.search(prose[start:final_box.start()] + prose[final_box.end():end]):
            return None
        after = prose[final_box.end():]
        later_cues = list(_MATH_CUE.finditer(after)) or list(_MATH_CONCLUSION.finditer(after))
        if later_cues and _math_declared_number(after[later_cues[-1].end():]) != candidate:
            return None
        return candidate
    headings = re.findall(r"(?m)^\s*####\s*([^\n]+)", prose)
    if headings:
        candidate = _math_scalar(headings[-1])
        if candidate is not None:
            return candidate
    cues = list(_MATH_CUE.finditer(prose)) or list(_MATH_CONCLUSION.finditer(prose))
    if not cues:
        return _math_terminal_declaration(prose)
    cue = cues[-1]
    sentence_start = max(prose.rfind(boundary, 0, cue.start()) for boundary in "\n.!?;。") + 1
    cue_prefix = prose[sentence_start:cue.start()]
    if _MATH_UNCERTAIN.search(cue_prefix):
        return None
    return _math_declared_number(prose[cue.end():])


_ANSWER_CUE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)


def _short_answer(output: str) -> str:
    """The span a reader would take as the answer; formatting and lead-ins removed.

    A cue whose own line holds only markup ("**Answer:**" with the answer on the
    next line) takes the next non-empty line after it: reading the markup as the
    answer would score correct replies as empty.
    """
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    cues = list(_ANSWER_CUE.finditer(output))
    span = cues[-1].group(1) if cues else lines[-1]
    span = re.sub(r"[*_`]+", "", span.splitlines()[0])
    if cues and not span.strip("\"':. \t"):
        after = [
            re.sub(r"[*_`]+", "", line).strip()
            for line in output[cues[-1].end() :].splitlines()
        ]
        span = next((line for line in after if line.strip("\"':. \t")), "")
    return span.strip().strip("\"'").rstrip(".").strip()


def _hotpot_score(output: str, answer: str) -> float:
    # Token F1 is the HotpotQA metric; it is taken on the extracted answer span
    # so a correct answer phrased as a sentence is not scored as a near miss.
    short = _short_answer(output)
    gold = _hotpot_norm(answer)
    if gold in {"yes", "no"}:
        # "Yes, both are American." answers yes; the span's first word decides.
        return float((_hotpot_norm(short).split() or [""])[0] == gold)
    return max(_f1(short, answer), _f1(output, answer))


def _answer_em(output: str, answer: str) -> float:
    """Answer exact match: the extracted answer span equals the gold after normalization.

    HotpotQA's yes/no rule is kept: the span's first word decides ("Yes, both are
    American." answers yes). An empty span never matches.
    """
    short = _hotpot_norm(_short_answer(output))
    gold = _hotpot_norm(answer)
    if gold in {"yes", "no"}:
        return float((short.split() or [""])[0] == gold)
    return float(bool(short) and short == gold)


def _qa_em(output: str, answers: list[str]) -> float:
    """Ans EM: the maximum over the reference aliases (the reward of NQ-Open and HotpotQA)."""
    return max((_answer_em(output, answer) for answer in answers), default=0.0)


# Cue words are case-insensitive; the option letter must be a capital so that
# "answer, a patient ..." is never read as option A.
_CHOICE_CUES = (
    re.compile(r"(?i:answer)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
    re.compile(r"(?i:option|choice)[*_\s]*(?i:is)?[*_\s]*[:：]?[*_\s(\[]*([A-J])(?![A-Za-z])"),
)


def _choice_letter(output: str, letters: list[str]) -> str | None:
    """Option letter: last 'Answer: X' cue, else 'option X', else a reply that is only a letter."""
    for cue in _CHOICE_CUES:
        found = [letter for letter in cue.findall(output) if letter in letters]
        if found:
            return found[-1]
    bare = re.fullmatch(r"\s*[*_(\[]*\s*([A-J])\s*[)\].:*_]*\s*", output)
    if bare and bare.group(1) in letters:
        return bare.group(1)
    return None


def _integer_score(output: str, answer: str) -> float:
    prediction = _math_prediction(output)
    return float(prediction is not None and prediction == Decimal(answer))


def _mbpp_tests(row: dict[str, Any]) -> list[str]:
    tests = row.get("test_list", []) or []
    if isinstance(tests, str):
        # The manifest stores MBPP's test list as a Python literal.
        try:
            tests = ast.literal_eval(tests)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(tests, list) or any(not isinstance(t, str) for t in tests):
        return []
    return tests


_ASSERTION_CALL = re.compile(r"assertion\(\s*([A-Za-z_]\w*)\(\*inp\)")


_BUILTIN_NAMES = frozenset(dir(builtins))


def _entry_point(row: dict[str, Any]) -> str | None:
    """The function the verifier calls.

    EvalPlus tests call ``assertion(<entry>(*inp), exp, atol)``; otherwise the
    first call of a non-builtin name in the first MBPP assertion
    (``assert set(f(x)) == ...`` names ``f``).
    """
    found = _ASSERTION_CALL.findall(row.get("plus_test") or "")
    if found:
        return found[-1]
    tests = _mbpp_tests(row)
    tree = _parse(tests[0]) if tests else None
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _BUILTIN_NAMES:
                return node.func.id
    return None


class _Block(NamedTuple):
    source: str
    python: bool
    closed: bool


def _fenced_blocks(output: str) -> list[_Block]:
    """Fenced blocks, read line by line: only a line of its own is a fence.

    A bare fence at least as long as the opener closes the open block. A tagged
    fence inside an open block starts a new one (the model never closed the
    previous block). An unclosed block at the end is kept (a cut-off reply).
    """
    blocks: list[_Block] = []

    def emit(tag: str, lines: list[str], closed: bool) -> None:
        source = textwrap.dedent("\n".join(lines)).strip()
        if source:
            blocks.append(_Block(source, tag in _PYTHON_TAGS, closed))

    current: tuple[str, int, list[str]] | None = None  # (tag, fence length, lines)
    for line in _lines(output):
        fence = _FENCE_LINE.match(line)
        if fence is not None and (current is None or len(fence.group(1)) >= current[1]):
            info = fence.group(2).strip()
            tag = info.split()[0].lower() if info else ""
            if current is None:
                current = (tag, len(fence.group(1)), [])
            elif not info:
                emit(current[0], current[2], closed=True)
                current = None
            else:
                emit(current[0], current[2], closed=False)
                current = (tag, len(fence.group(1)), [])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        emit(current[0], current[2], closed=False)
    return blocks


def _lines(text: str) -> list[str]:
    # Python's own line breaks: str.splitlines also splits on form feeds, the
    # Unicode line separator and other characters a string literal may hold.
    return re.split(r"\r\n|\r|\n", text)


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _defines(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if isinstance(node, _DEFINITIONS) and node.name == name:
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return True
    return False


def _extract_code(output: str, entry: str | None = None) -> str:
    """The candidate program in a reply: one fenced block, then ``_sanitize``.

    Pairing fences by position and keeping the longest block would score a
    superseded draft (an earlier block with an unterminated docstring, or a buggy
    first version the reply then corrected). The chosen block is the last closed
    python block that parses and defines ``entry``;
    else the last closed block of any tag, then the last cut-off block, that does;
    else the longest block that parses and defines a function; else the longest
    block. A reply without a fence line falls back to inline pairing, then to
    the whole reply.
    """
    blocks = _fenced_blocks(output)
    parsed = [(block, _parse(block.source)) for block in blocks]
    usable = [(block, tree) for block, tree in parsed if tree is not None]
    if entry:
        defining = [block for block, tree in usable if _defines(tree, entry)]
        for picks in (
            [block for block in defining if block.closed and block.python],
            [block for block in defining if block.closed],
            defining,
        ):
            if picks:
                return _sanitize(picks[-1].source, entry)
    with_function = [
        block
        for block, tree in usable
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
    ]
    if with_function:
        source = max(with_function, key=lambda block: len(block.source)).source
    elif blocks:
        source = max(blocks, key=lambda block: len(block.source)).source
    else:
        inline = [match.group(1).strip() for match in _FENCE.finditer(output)]
        source = max(inline, key=len) if inline else output.strip()
        if not inline and _parse(source) is None:
            source = _unfenced_code(source, entry) or source
    return _sanitize(source, entry)


_CODE_START = re.compile(r"\b(?:async\s+def|def|class|import|from)\s+\w")


def _unfenced_code(text: str, entry: str | None) -> str | None:
    """Code written without a fence after prose, e.g. "Answer: def f(x):\n    return x".

    Verifier replies can put the function after an answer cue instead of in a code
    block. Start at the first line that opens a definition or an
    import (the one defining ``entry`` when there is one), drop any lead-in before the
    keyword on that line, and cut trailing prose until the rest parses.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _CODE_START.search(line)]
    if entry:
        defining = [i for i in starts if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", lines[i])]
        # Imports just above the defining line belong to it.
        if defining:
            first = defining[0]
            while first - 1 in starts and re.search(r"\b(?:import|from)\s+\w", lines[first - 1]):
                first -= 1
            starts = [first]
    for start in starts[:1]:
        head = lines[start]
        head = head[_CODE_START.search(head).start():]
        body = [head, *lines[start + 1 :]]
        for end in range(len(body), 0, -1):
            candidate = "\n".join(body[:end]).rstrip()
            if candidate and _parse(candidate) is not None:
                return candidate
    return None


# Calls that make a module-level statement test or driver code.
_DRIVER_CALLS = frozenset(
    {
        "print",
        "input",
        "open",
        "exit",
        "quit",
        "breakpoint",
        "help",
        "sys.exit",
        "os._exit",
        "unittest.main",
        "doctest.testmod",
        "pytest.main",
    }
)


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") in {
        "__name__ == '__main__'",
        "'__main__' == __name__",
    }


def _drives(node: ast.stmt, defined: set[str]) -> bool:
    """Test or driver code: an assert, I/O, exiting, or a call of the program's own definitions."""
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.Call):
            name = _dotted(child.func)
            if name in _DRIVER_CALLS or name in defined:
                return True
        if isinstance(child, ast.Attribute) and _dotted(child) == "sys.stdin":
            return True
    return False


def _binds(node: ast.stmt) -> set[str]:
    """Module-level names a statement binds or may mutate (``t[i] = x``, ``t.append(x)``)."""
    names: set[str] = set()
    for child in ast.walk(node):
        stores = isinstance(getattr(child, "ctx", None), (ast.Store, ast.Del))
        if isinstance(child, ast.Name) and stores:
            names.add(child.id)
        elif isinstance(child, (ast.Attribute, ast.Subscript)) and stores:
            names.add(_root_name(child) or "")
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.add(_root_name(child.func.value) or "")
        elif isinstance(child, _DEFINITIONS):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in child.names)
    return names - {""}


def _reads(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _definition_reads(node: ast.stmt) -> set[str]:
    """Module-level names a function or class reads: its header, and its bodies' global names."""
    try:
        table = symtable.symtable(ast.unparse(node), "<candidate>", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return _reads(node)
    names = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_referenced()}
    scopes = list(table.get_children())
    while scopes:
        scope = scopes.pop()
        names.update(
            symbol.get_name()
            for symbol in scope.get_symbols()
            if symbol.is_global() and (symbol.is_referenced() or symbol.is_declared_global())
        )
        scopes.extend(scope.get_children())
    return names


def _sanitize(source: str, entry: str | None = None) -> str:
    """Drop module-level test and driver code, EvalPlus-style.

    Kept: imports, functions, classes, and every other statement that is not
    test or driver code (constants, lookup tables, ``sys.setrecursionlimit``).
    Dropped: top-level asserts, ``if __name__ == "__main__":`` blocks (their own
    imports are kept) and statements that print, read input, exit or call the
    program's own definitions, unless they bind a name the kept code or the
    verifier reads (``PRIMES = sieve(100)``, ``f = cache(f)``). A module-level
    test that used ``math``, imported only inside the correct function, would
    otherwise fail before the verifier ran.
    """
    tree = _parse(source)
    if tree is None:
        return source
    try:
        kept = _kept_statements(tree, entry)
        if len(kept) == len(tree.body):
            return source
        sanitized = _splice(source, tree, kept)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return source
    return sanitized if _parse(sanitized) is not None else source


def _kept_statements(tree: ast.Module, entry: str | None) -> set[int]:
    """Indices of the top-level statements ``_sanitize`` keeps."""
    defined = {node.name for node in tree.body if isinstance(node, _DEFINITIONS)}
    # A driver's module calls (``sys.stdin.readline()``) do not make it setup.
    imported: set[str] = set()
    kept: set[int] = set()
    pending: list[int] = []
    reads: set[str] = {entry} if entry else set()
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.add(index)
            imported |= _binds(node)
        elif isinstance(node, _DEFINITIONS):
            kept.add(index)
            reads |= _definition_reads(node)
        elif isinstance(node, ast.Assert) or _is_main_guard(node):
            continue
        elif _drives(node, defined):
            pending.append(index)
        else:
            kept.add(index)
            reads |= _reads(node)
    grown = True
    while grown:
        grown = False
        for index in [index for index in pending if (_binds(tree.body[index]) - imported) & reads]:
            pending.remove(index)
            kept.add(index)
            reads |= _reads(tree.body[index])
            grown = True
    return kept


def _splice(source: str, tree: ast.Module, kept: set[int]) -> str:
    """``source`` without the dropped top-level statements, comments and layout kept.

    A dropped ``__main__`` block leaves its own imports in its place. Falls back
    to unparsing the kept statements when a dropped one shares a line.
    """

    def left(node: ast.stmt) -> list[str]:
        # What a dropped statement leaves: a __main__ block's own imports.
        if not (isinstance(node, ast.If) and _is_main_guard(node)):
            return []
        imports = (ast.Import, ast.ImportFrom)
        return [ast.unparse(child) for child in node.body if isinstance(child, imports)]

    lines = _lines(source)
    replace: dict[int, list[str]] = {}
    for index, node in enumerate(tree.body):
        if index in kept:
            continue
        first, last = node.lineno - 1, (node.end_lineno or node.lineno) - 1
        # Column offsets count UTF-8 bytes.
        before = lines[first].encode("utf-8")[: node.col_offset].decode("utf-8", "replace")
        after = lines[last].encode("utf-8")[node.end_col_offset :].decode("utf-8", "replace")
        before, after = before.strip(), after.strip()
        if before or (after and not after.startswith("#")):
            unparsed: list[str] = []
            for number, item in enumerate(tree.body):
                unparsed.extend([ast.unparse(item)] if number in kept else left(item))
            return "\n".join(unparsed)
        replace[first] = left(node)
        for number in range(first + 1, last + 1):
            replace[number] = []
    out: list[str] = []
    for number, line in enumerate(lines):
        out.extend(replace.get(number, [line]))
    return "\n".join(out).strip()


def _code_score(output: str, row: dict[str, Any], *, python: str = sys.executable) -> float:
    source = _extract_code(output, _entry_point(row))
    if not source:
        return 0.0
    try:
        ast.parse(source)
    except SyntaxError:
        return 0.0
    # EvalPlus: base and plus inputs are checked by the dataset's own script,
    # which imports numpy; see _passes for the 30 s.
    imports = "\n".join(row.get("test_imports", []) or [])
    return float(_passes(source + "\n\n" + imports + "\n" + row["plus_test"], cpu_seconds=30, python=python))


_ONE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _passes(program: str, *, cpu_seconds: int, python: str = sys.executable) -> bool:
    """Run a candidate plus its assertions in a child process.

    Assertions stay verifier-side. A per-run sentinel printed after the last
    assertion rejects programs that exit early with status 0. Limits are set by
    the child itself: preexec_fn is unsafe in this multithreaded trainer.

    The child runs single-threaded BLAS/OpenMP: numpy's 64-thread OpenBLAS pool
    took 2.7 GB of the 4 GB address space and 8.5 s of the 20 s CPU limit (CPU
    time counts every thread), and 2 of the 378 canonical MBPP+
    solutions failed (599, 255). One thread and 30 s: all 378 pass.
    """
    sentinel = f"mbppplus-tests-passed-{uuid.uuid4().hex}"
    limits = (
        "import resource as _sandbox_resource\n"
        f"_sandbox_resource.setrlimit(_sandbox_resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "_sandbox_resource.setrlimit(_sandbox_resource.RLIMIT_AS, (4 << 30, 4 << 30))\n"
        "_sandbox_resource.setrlimit(_sandbox_resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))\n"
        if sys.platform.startswith("linux")
        else ""
    )
    with tempfile.TemporaryDirectory(prefix="mbppplus-") as temp:
        path = Path(temp) / "candidate.py"
        path.write_text(limits + program + f"\nprint('\\n' + {sentinel!r})\n", encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", **_ONE_THREAD}
        result = subprocess.run(
            [python, "-I", str(path)],
            cwd=temp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=cpu_seconds + 10,
            check=False,
        )
    return result.returncode == 0 and result.stdout.decode("utf-8", "replace").splitlines()[-1:] == [sentinel]


# --------------------------------------------------------------------------- adapters

BENCHMARK_VERIFIERS = (
    "integer_answer",
    "qa_exact_match",
    "letter_choice",
    "mbppplus_tests",
)


def _answers(reference: Any) -> list[str]:
    values = reference if isinstance(reference, (list, tuple)) else [reference]
    return [str(value) for value in values if value is not None]


def benchmark_score(verifier: str, output: str, task: TaskSpec, *, python: str = sys.executable) -> float:
    """Score of one task row (reference and metadata as in data/r2orch/)."""

    if verifier == "qa_exact_match":
        return _qa_em(output, _answers(task.reference))
    if verifier == "letter_choice":
        return float(_choice_letter(output, list(task.metadata["options"])) == str(task.reference))
    if verifier == "integer_answer":
        try:
            return _integer_score(output, _answers(task.reference)[0])
        except InvalidOperation:
            return 0.0
    if verifier == "mbppplus_tests":
        try:
            return _code_score(output, dict(task.reference), python=python)
        except (OSError, subprocess.SubprocessError, ValueError, MemoryError):
            return 0.0
    raise ValueError(f"unsupported verifier: {verifier}")


class BenchmarkVerifier:
    """One binary benchmark reward behind the project's Verifier protocol."""

    def __init__(self, name: str, *, python: str | None = None) -> None:
        if name not in BENCHMARK_VERIFIERS:
            raise ValueError(f"unsupported verifier: {name}")
        self.name = name
        self.python = python or sys.executable

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        if self.name == "integer_answer" and _math_prediction(str(prediction or "")) is None:
            # Benchmark accounting still includes a no-answer trial as zero via
            # benchmark_score. Training must retain its unknown status instead
            # of manufacturing a trusted negative from an intermediate digit.
            raise UnscorableAnswerError("missing_trusted_final_math_answer")
        score = benchmark_score(self.name, str(prediction or ""), task, python=self.python)
        detail = ""
        if self.name == "qa_exact_match":
            f1 = max((_hotpot_score(str(prediction or ""), answer)
                      for answer in _answers(task.reference)), default=0.0)
            detail = f"em={score:.12g}; f1={f1:.12g}"
        return VerificationResult(score, score == 1.0, self.name, detail)


class MockHashVerifier:
    """Deterministic mock reward: a bit of sha256(task_id|execution_seed)."""

    name = "mock_hash"

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        r2orch = task.metadata.get("r2orch") or {}
        key = f"{r2orch.get('task_id', task.task_id)}|{r2orch.get('execution_seed', 0)}"
        score = float(hashlib.sha256(key.encode()).digest()[0] & 1)
        return VerificationResult(score, score == 1.0, self.name, f"mock_hash={key}")


def benchmark_verifier_adapters(*, python: str | None = None) -> dict[str, Any]:
    return {
        **{name: BenchmarkVerifier(name, python=python) for name in BENCHMARK_VERIFIERS},
        "mock_hash": MockHashVerifier(),
    }
