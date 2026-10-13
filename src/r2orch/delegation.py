from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_ANSWER_OR_SOLUTION_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*(?:is|=|:|：)\s*\S+|"
    r"(?:candidate\s+answer|intermediate\s+calculation|solution\s+steps?)",
    flags=re.IGNORECASE,
)
_OUTPUT_LETTER_TEMPLATE_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*[:：]\s*['\"`]?<letter>['\"`]*[.,;!?]*",
    flags=re.IGNORECASE,
)
_OUTPUT_LETTER_CONCRETE_SUFFIX_RE = re.compile(
    r"\s*(?:[.,;:：!?]\s*)*"
    r"(?:(?:is|means|choose|select)\b\s*|(?:=>|->|=|→)\s*)?"
    r"(?:[.,;:：!?]\s*)*(?:[('\"`\[]\s*)*"
    r"(?:[A-Z]|[+-]?\d+(?:\.\d+)?)(?!\w)",
    flags=re.IGNORECASE,
)
_PROCEDURAL_SOLUTION_RE = re.compile(
    r"\b(?:use|apply)\b.{0,160}\b(?:to\s+)?"
    r"(?:derive|calculate|compute|prove|solve|obtain)\b|"
    r"\b(?:first|then|next)\b.{0,100}\b"
    r"(?:derive|calculate|compute|prove|solve|obtain)\b|"
    r"\b(?:first|then|next)\s*[,;:]?\s*determine\s+(?:the\s+)?"
    r"(?:sample\s+space|favo[u]?rable\s+outcomes|intermediate\s+(?:state|value|quantity))\b",
    flags=re.IGNORECASE,
)
_SYMBOLIC_DELIVERABLE = r"[A-Za-z]\w*(?:\s*[+*/^-]\s*[A-Za-z]\w*)*"
_PUBLIC_SYMBOLIC_GOAL_RE = re.compile(
    r"\b(?:find|determine|compute|calculate)\s+(?:the\s+value\s+of\s+)?"
    rf"(?P<expression>{_SYMBOLIC_DELIVERABLE})(?!\w)", re.IGNORECASE,
)
_TYPED_SYMBOLIC_OUTPUT_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*(?:an?\s+)?integer\s+"
    rf"(?P<expression>{_SYMBOLIC_DELIVERABLE})(?!\w)", re.IGNORECASE,
)
_TERMINAL_SYMBOLIC_DELIVERABLE_RE = re.compile(
    r"[,;]?\s*\bthen\s+(?:compute|calculate|find|determine)\s+"
    rf"(?P<expression>{_SYMBOLIC_DELIVERABLE})\s*[.!]?$", re.IGNORECASE,
)
_TERMINAL_MODULO_DELIVERABLE_RE = re.compile(
    r",?\s*\bthen\s+(?:compute|calculate|find|determine)\s+"
    r"(?:(?:(?:this|that|the|its)\s+)?(?:remainder|count|value|number|result)|"
    r"(?P<variable>[a-z][a-z0-9_]*))\s+"
    r"(?:modulo|mod)\s+(?P<divisor>\d+)\s*[.!]?$",
    flags=re.IGNORECASE,
)
_ROUTING_RE = re.compile(
    r"\b(?:mace|runtime|model)\s*(?:router|routing|selection)\b|"
    r"\b(?:select|choose|rank|evaluate)\b.{0,24}\b(?:candidate\s+)?models?\b",
    flags=re.IGNORECASE,
)
_WORDS_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

_RESPONSIBILITY_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "implementation": re.compile(
        r"\b(?:implement|implementation|modify|modification|patch|fix|repair|"
        r"add|change|update|developer|code)\b|\b_cdf\b",
        flags=re.IGNORECASE,
    ),
    "diagnosis": re.compile(
        r"\b(?:diagnos(?:e|is)|investigat(?:e|ion)|root cause|locali[sz](?:e|ation)|"
        r"identify the (?:cause|failure|bug))\b",
        flags=re.IGNORECASE,
    ),
    "testing": re.compile(
        r"\b(?:test|tests|testing|validate|validation|verify|verification|regression|"
        r"differentiat(?:e|ion))\b",
        flags=re.IGNORECASE,
    ),
    "review": re.compile(
        r"\b(?:review|audit|critique|check correctness|falsif(?:y|ication))\b",
        flags=re.IGNORECASE,
    ),
    "synthesis": re.compile(
        r"\b(?:synthesi[sz](?:e|er|is)|aggregate|combine|merge)\b|"
        r"\bintegrat(?:e|ion)\s+(?:findings|artifacts|results)\b",
        flags=re.IGNORECASE,
    ),
    "environment_execution": re.compile(
        r"\b(?:execute|interact|navigate|configure|environment|stateful)\b",
        flags=re.IGNORECASE,
    ),
}

_DELIVERABLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "code_change": re.compile(
        r"\b(?:code|patch|implementation|implementations|modification|modifications|"
        r"fixed|fix|method|methods)\b|\b_cdf\b",
        flags=re.IGNORECASE,
    ),
    "tests": re.compile(
        r"\b(?:test|tests|testing|validation|validated|verified|regression|"
        r"differentiat(?:e|ion))\b",
        flags=re.IGNORECASE,
    ),
    "root_cause_report": re.compile(
        r"\b(?:root cause|diagnosis|diagnostic|findings|analysis|report)\b",
        flags=re.IGNORECASE,
    ),
    "final_answer": re.compile(
        r"\b(?:final answer|requested value|concise answer|answer span)\b",
        flags=re.IGNORECASE,
    ),
    "environment_completion": re.compile(
        r"\b(?:environment outcome|configured environment|"
        r"official outcome)\b",
        flags=re.IGNORECASE,
    ),
}

_RESPONSIBILITY_PRIMARY_ORDER = (
    "implementation",
    "environment_execution",
    "diagnosis",
    "testing",
    "review",
    "synthesis",
)


@dataclass(frozen=True)
class ResponsibilitySignature:
    """Deterministic semantic summary of one Director-authored responsibility."""

    types: tuple[str, ...]
    primary_type: str
    scope_entities: frozenset[str]
    objective_tokens: frozenset[str]
    expected_output_tokens: frozenset[str]
    deliverables: frozenset[str]


@dataclass(frozen=True)
class ResponsibilityComparison:
    """Pairwise responsibility overlap used for admission and offline audits."""

    same_primary_type: bool
    scope_overlap: float
    objective_overlap: float
    expected_output_overlap: float
    deliverable_overlap: float
    similarity: float
    shared_scope_entities: tuple[str, ...]
    high_confidence_duplicate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_primary_type": self.same_primary_type,
            "scope_overlap": self.scope_overlap,
            "objective_overlap": self.objective_overlap,
            "expected_output_overlap": self.expected_output_overlap,
            "deliverable_overlap": self.deliverable_overlap,
            "similarity": self.similarity,
            "shared_scope_entities": list(self.shared_scope_entities),
            "high_confidence_duplicate": self.high_confidence_duplicate,
        }


DELEGATION_CONTRACT_VERSION = "dataset-output-contract"
DELEGATION_FIELD_LIMITS = {
    "role": 80,
    "objective": 320,
    "scope": 240,
    "expected_output": 200,
}
DELEGATION_TOTAL_LIMIT = 700

_DATASET_OUTPUT_CONTRACTS: dict[str, tuple[tuple[str, str], ...]] = {
    "aime": (
        (
            "preserve_exact_forms",
            "Preserve exact integers, fractions, and radicals in the reported result.",
        ),
        (
            "identify_requested_value",
            "State the requested final value unambiguously with concise supporting evidence.",
        ),
    ),
    "nq_open": (
        (
            "concise_answer_span",
            "Return a concise answer span and evidence that directly supports it.",
        ),
        (
            "report_uncertainty",
            "Report unresolved uncertainty instead of inventing unsupported details.",
        ),
    ),
    "hotpotqa": (
        (
            "multi_fact_evidence",
            "Return a concise answer and the evidence connecting the relevant facts or entities.",
        ),
        (
            "report_uncertainty",
            "Report unresolved uncertainty instead of inventing unsupported details.",
        ),
    ),
    "alfworld": (
        (
            "execute_environment_actions",
            "Use alfworld_step with an action_id from the latest observation; naming or recommending a command without calling the Action does not change the environment.",
        ),
        (
            "continue_until_environment_done",
            "Continue one stateful Action at a time, re-observing after every call, until the official environment reports success or termination.",
        ),
        (
            "follow_live_action_ids",
            "Use only state-scoped action identifiers exposed by the latest observation; never reuse stale identifiers.",
        ),
    ),
}


@dataclass(frozen=True)
class DelegationIssue:
    """Machine-readable reason why a Director responsibility is unsafe."""

    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] | None = None


class DelegationValidationError(ValueError):
    """Raised with structured detail while preserving transactional rollback."""

    def __init__(self, issue: DelegationIssue) -> None:
        self.issue = issue
        suffix = f" (field={issue.field})" if issue.field else ""
        super().__init__(issue.message + suffix)


@dataclass(frozen=True)
class DelegationFieldRepair:
    """A deterministic normalization applied to one Director-authored field."""

    field: str
    reason: str
    original_length: int
    repaired_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reason": self.reason,
            "original_length": self.original_length,
            "repaired_length": self.repaired_length,
        }


@dataclass(frozen=True)
class DelegationCompilation:
    """Compiled Worker prompt with provenance kept separate from system rules."""

    prompt: str
    director_fields: dict[str, str]
    dataset: str
    contract_version: str | None
    contract_rule_ids: tuple[str, ...]
    field_repairs: tuple[DelegationFieldRepair, ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {
            "director_delegation": dict(self.director_fields),
            "system_managed_contract": {
                "version": self.contract_version,
                "dataset": self.dataset,
                "rule_ids": list(self.contract_rule_ids),
            },
            "delegation_field_repairs": [repair.to_dict() for repair in self.field_repairs],
        }


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "agent",
        "analyze",
        "analysis",
        "as",
        "at",
        "be",
        "best",
        "by",
        "concise",
        "direct",
        "for",
        "from",
        "given",
        "identify",
        "in",
        "independent",
        "independently",
        "information",
        "is",
        "of",
        "on",
        "or",
        "objective",
        "output",
        "provide",
        "provided",
        "relevant",
        "report",
        "result",
        "return",
        "role",
        "scope",
        "specialist",
        "task",
        "the",
        "to",
        "using",
        "was",
        "when",
        "which",
        "with",
        "expected",
    }
)

_RESPONSIBILITY_SCOPE_STOPWORDS = frozenset(
    {
        *_STOPWORDS,
        "add",
        "all",
        "code",
        "correct",
        "correctly",
        "ensure",
        "full",
        "internal",
        "known",
        "method",
        "methods",
        "missing",
        "performance",
        "poor",
        "respective",
        "specific",
        "working",
    }
)


def responsibility_signature(fields: dict[str, object]) -> ResponsibilitySignature:
    """Extract a bounded, deterministic signature from Director-authored fields."""

    role = str(fields.get("role") or "")
    objective = str(fields.get("objective") or "")
    scope = str(fields.get("scope") or "")
    expected_output = str(fields.get("expected_output") or "")
    responsibility_text = " ".join((role, objective, expected_output))
    types = tuple(
        name
        for name, pattern in _RESPONSIBILITY_TYPE_PATTERNS.items()
        if pattern.search(responsibility_text)
    )
    primary_type = next(
        (name for name in _RESPONSIBILITY_PRIMARY_ORDER if name in types),
        "general",
    )
    deliverables = frozenset(
        name for name, pattern in _DELIVERABLE_PATTERNS.items() if pattern.search(expected_output)
    )
    return ResponsibilitySignature(
        types=types,
        primary_type=primary_type,
        scope_entities=_responsibility_tokens(
            scope,
            stopwords=_RESPONSIBILITY_SCOPE_STOPWORDS,
        ),
        objective_tokens=_responsibility_tokens(objective, stopwords=_STOPWORDS),
        expected_output_tokens=_responsibility_tokens(
            expected_output,
            stopwords=_STOPWORDS,
        ),
        deliverables=deliverables,
    )


def compare_responsibilities(
    left_fields: dict[str, object],
    right_fields: dict[str, object],
) -> ResponsibilityComparison:
    """Compare two responsibilities without using system-managed prompt text."""

    left = responsibility_signature(left_fields)
    right = responsibility_signature(right_fields)
    scope_overlap = _containment_overlap(left.scope_entities, right.scope_entities)
    objective_overlap = _jaccard(left.objective_tokens, right.objective_tokens)
    expected_output_overlap = _jaccard(
        left.expected_output_tokens,
        right.expected_output_tokens,
    )
    deliverable_overlap = _containment_overlap(left.deliverables, right.deliverables)
    similarity = (
        0.50 * scope_overlap
        + 0.20 * objective_overlap
        + 0.10 * expected_output_overlap
        + 0.20 * deliverable_overlap
    )
    same_primary_type = left.primary_type == right.primary_type
    high_confidence_duplicate = bool(
        same_primary_type
        and left.primary_type == "implementation"
        and scope_overlap >= 0.80
        and deliverable_overlap >= 0.75
        and similarity >= 0.78
    )
    return ResponsibilityComparison(
        same_primary_type=same_primary_type,
        scope_overlap=round(scope_overlap, 6),
        objective_overlap=round(objective_overlap, 6),
        expected_output_overlap=round(expected_output_overlap, 6),
        deliverable_overlap=round(deliverable_overlap, 6),
        similarity=round(similarity, 6),
        shared_scope_entities=tuple(sorted(left.scope_entities & right.scope_entities)[:24]),
        high_confidence_duplicate=high_confidence_duplicate,
    )


def _responsibility_tokens(
    text: str,
    *,
    stopwords: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        token
        for token in (match.casefold() for match in _WORDS_RE.findall(str(text or "")))
        if len(token) >= 2 and token not in stopwords
    )


def _containment_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _public_symbolic_targets(public_task: str) -> set[str]:
    """Read only symbolic quantities explicitly requested in the public task."""
    public = str(public_task or "").replace("$", "")
    targets = set()
    for match in _PUBLIC_SYMBOLIC_GOAL_RE.finditer(public):
        expression = re.sub(r"\s+", "", match.group("expression"))
        if re.match(r"\s*[-+*/^=]", public[match.end():]):
            continue
        if re.search(r"[+*/^-]", expression) or re.fullmatch(r"[A-Za-z](?:_\w+|\d+)?", expression):
            targets.add(expression)
    return targets


def _public_symbolic_output_match(
    delegation: str, answer_match: re.Match[str], public_targets: set[str], public_task: str,
) -> bool:
    typed = _TYPED_SYMBOLIC_OUTPUT_RE.match(delegation, answer_match.start())
    if typed is None:
        return False
    expression = re.sub(r"\s+", "", typed.group("expression"))
    if expression not in public_targets:
        return False
    remainder = delegation[typed.end():]
    # A public symbol names what to return; it never supplies a value for it.
    # Keep appended numbers, altered expressions and concrete variable values
    # visible to the leakage boundary, including in the joined-field audit.
    if re.match(r"\s*[-+*/^=]", remainder) or re.search(r"\d", remainder):
        return False
    if remainder.strip(" \t\r\n.,;"):
        fraction_tail = re.fullmatch(
            r"[,;]?\s*where\s+the\s+probability(?:\s+of\s+(?:the\s+specified|this)\s+event)?"
            r"\s+is\s+(?P<numerator>[A-Za-z]\w*)\s*/\s*(?P<denominator>[A-Za-z]\w*)"
            r"\s+in\s+simplest\s+form\s*[.]?", remainder, re.IGNORECASE,
        )
        public = re.sub(
            r"\\(?:frac|dfrac|tfrac)\s*\{\s*(\w+)\s*\}\s*\{\s*(\w+)\s*\}",
            r"\1/\2", public_task.replace("$", ""),
        )
        if fraction_tail is None:
            return False
        ratio = (
            r"(?<!\w)" + re.escape(fraction_tail.group("numerator"))
            + r"\s*/\s*" + re.escape(fraction_tail.group("denominator")) + r"(?!\w)"
        )
        if not re.search(ratio, public):
            return False
    symbols = re.findall(r"[A-Za-z]\w*", expression)
    assignments = "|".join(re.escape(symbol) for symbol in symbols)
    return not re.search(
        rf"\b(?:{assignments})\s*(?:=|is\b|equals\b)\s*[+-]?\d", delegation,
        flags=re.IGNORECASE,
    )


def delegation_safety_issue(
    text: str,
    *,
    action_names: Iterable[str] = (),
    field: str | None = None,
    public_task: str = "",
) -> DelegationIssue | None:
    """Return a deterministic responsibility issue, or ``None`` when safe.

    Task entities and goals are deliberately allowed.  The hard boundary covers
    only answer/solution leakage, explicit Action control, and runtime routing.
    """

    delegation = str(text or "").strip()
    public_targets = _public_symbolic_targets(public_task)
    for answer_match in _ANSWER_OR_SOLUTION_RE.finditer(delegation):
        # An unfilled MCQ format slot does not choose an answer. Exempt only
        # this exact match, in individual fields and the joined graph audit;
        # continue inspecting the rest for real answers or solution content.
        # A value following the slot (including across field boundaries) is
        # still an answer declaration, not an unfilled format.
        if (
            _OUTPUT_LETTER_TEMPLATE_RE.fullmatch(answer_match.group())
            and not _OUTPUT_LETTER_CONCRETE_SUFFIX_RE.match(delegation[answer_match.end():])
        ):
            continue
        # A conditional abstention format copied from the public question is
        # an output rule, not a claim that this particular question is
        # unanswerable. Keep unconditional answer declarations blocked.
        public_abstention = bool(
            re.search(r"\b(?:respond|reply)\s+with\s+['\"]?unanswerable\b", public_task, re.I)
        )
        conditional_abstention = bool(
            re.fullmatch(r"answer\s+is\s+['\"]?unanswerable['\"]?", answer_match.group(), re.I)
            and re.search(r"\botherwise\b[^.!?]{0,100}$", delegation[:answer_match.start()], re.I)
        )
        if public_abstention and conditional_abstention:
            continue
        if _public_symbolic_output_match(delegation, answer_match, public_targets, public_task):
            continue
        return DelegationIssue(
            "answer_or_solution_leak",
            "SET_PROMPT contains an answer clue or solution content",
            field,
        )
    procedure_text = delegation
    # A public event's "first day" is not the Director's first solution step.
    # Only remove the non-imperative chronology word; commands such as
    # "First, determine the sample space" remain subject to the original guard.
    if re.search(r"\bfirst\s+day\b", public_task, re.IGNORECASE):
        procedure_text = re.sub(
            r"\bfirst\s+(?=day\b|selected\b|selects\b|occur(?:s)?\b)",
            "", procedure_text, flags=re.IGNORECASE,
        )
    symbolic_terminal = _TERMINAL_SYMBOLIC_DELIVERABLE_RE.search(procedure_text)
    if symbolic_terminal and field in {"objective", "expected_output", None}:
        expression = re.sub(r"\s+", "", symbolic_terminal.group("expression"))
        original_procedure = _PROCEDURAL_SOLUTION_RE.search(procedure_text)
        only_terminal = original_procedure and original_procedure.start() >= symbolic_terminal.start()
        if expression in public_targets and only_terminal:
            procedure_text = procedure_text[:symbolic_terminal.start()]
    # A final modulo representation explicitly requested by the public task
    # is an output requirement, not a Director-supplied solution method. Match
    # the requested divisor; never consult reference answers or strip methods
    # elsewhere in the delegation. Without public grounding the guard remains
    # conservative. This applies to any task, not a dataset/id/constant list.
    terminal = _TERMINAL_MODULO_DELIVERABLE_RE.search(delegation)
    if terminal and field in {"objective", "expected_output"} and public_task:
        public_text = re.sub(r"[$]", "", public_task)
        requested = re.search(
            r"\b(?:find|determine|compute|calculate|what\s+is)\s+(?:the\s+)?remainder\b"
            r"(?P<subject>[^.!?]{0,180}?)\b(?:divided\s+by|modulo|mod)\s+"
            + re.escape(terminal.group("divisor")) + r"\b",
            public_text, flags=re.IGNORECASE,
        )
        variable = terminal.group("variable")
        if requested and (
            variable is None
            or re.search(r"\b" + re.escape(variable) + r"\b", requested.group("subject"), re.I)
        ):
            procedure_text = delegation[:terminal.start()]
    if _PROCEDURAL_SOLUTION_RE.search(procedure_text):
        return DelegationIssue(
            "concrete_solution_procedure",
            "SET_PROMPT prescribes a concrete solution procedure",
            field,
        )
    if _ROUTING_RE.search(delegation):
        return DelegationIssue(
            "runtime_routing_control",
            "SET_PROMPT must not delegate runtime routing or model selection",
            field,
        )

    lowered = delegation.casefold()
    for raw_name in action_names:
        name = str(raw_name or "").strip().casefold()
        if not name:
            continue
        escaped = re.escape(name)
        code_like = "_" in name and re.search(rf"\b{escaped}\b", lowered)
        explicit_invocations = re.finditer(
            rf"\b(?:call|invoke|execute|run)\b.{{0,24}}\b{escaped}\b|"
            rf"\buse\b.{{0,16}}\b{escaped}\b"
            rf"(?!\s+(?:results?|output|evidence|findings|records?|documents?))|"
            rf"\b(?:use|select|choose)\b.{{0,16}}\b{escaped}\b"
            rf"\s+(?:action|tool|function|api)\b|"
            rf"\b{escaped}\b\s+(?:action|tool|function|api)\b",
            lowered,
        )
        explicit_invocation = False
        for invocation in explicit_invocations:
            # In an NQ assignment, plural "available search tools" names a
            # general retrieval capability, not the search Action.
            # Exempt only this local noun phrase; inspect every other match.
            generic_search_tools = bool(
                name == "search"
                and re.fullmatch(r"use\s+(?:the\s+)?available\s+search", invocation.group())
                and re.match(r"\s+tools\b", lowered[invocation.end():])
                and not re.search(r"\b(?:only|exclusively)\s*$", lowered[:invocation.start()])
                and not re.match(
                    r"\s+tools\s+(?:only|exclusively|once|twice|\d+\s+times)\b",
                    lowered[invocation.end():],
                )
                and not re.search(
                    r"\b(?:query|arguments?|parameters?|top_k|limit)\s*[:=]"
                    r"|\bsearch\s*[\[(]|`search`"
                    r"|\b(?:exact\s+)?query\s+['\"]"
                    r"|\b(?:at (?:most|least)|no more than|exactly)\s+(?:\d+|one|two)\s+(?:calls?|queries|query|times)\b"
                    r"|\buse\s+no\s+other\s+tools?\b",
                    lowered,
                )
            )
            if not generic_search_tools:
                explicit_invocation = True
                break
        if code_like or explicit_invocation:
            return DelegationIssue(
                "worker_action_control",
                "SET_PROMPT must not select or constrain Worker Actions",
                field,
            )
    return None


def compile_delegation(
    fields: dict[str, object],
    *,
    dataset: str = "",
    action_names: Iterable[str] = (),
    public_task: str = "",
) -> tuple[DelegationCompilation | None, DelegationIssue | None]:
    """Validate, locally compact, and deterministically compile a responsibility.

    Semantic safety is checked on the complete Director-authored text *before*
    any length repair.  Consequently compaction can never hide an answer clue,
    procedure, routing instruction, or Worker Action selection.  Only formatting
    and length are repaired automatically; semantic violations remain rejected.
    """

    normalized: dict[str, str] = {}
    repairs: list[DelegationFieldRepair] = []
    for name in DELEGATION_FIELD_LIMITS:
        original = str(fields.get(name) or "").strip()
        if not original:
            return None, DelegationIssue(
                "missing_field",
                f"SET_PROMPT field {name} is required",
                name,
            )
        one_line = " ".join(original.split())
        if one_line != original:
            repairs.append(
                DelegationFieldRepair(
                    name,
                    "whitespace_normalization",
                    len(original),
                    len(one_line),
                )
            )
        semantic_issue = delegation_safety_issue(
            one_line,
            action_names=action_names,
            field=name,
            public_task=public_task,
        )
        if semantic_issue is not None:
            return None, semantic_issue
        normalized[name] = one_line

    repaired = dict(normalized)
    dataset_key = str(dataset or "").strip().casefold()
    for name, limit in DELEGATION_FIELD_LIMITS.items():
        value = repaired[name]
        if len(value) <= limit:
            continue
        shortened = _shorten_at_boundary(value, limit)
        repaired[name] = shortened
        repairs.append(
            DelegationFieldRepair(
                name,
                "field_length",
                len(value),
                len(shortened),
            )
        )

    _fit_total_delegation_limit(repaired, repairs)
    director_prompt = "\n".join(
        (
            f"Role: {repaired['role']}",
            f"Objective: {repaired['objective']}",
            f"Scope: {repaired['scope']}",
            f"Expected output: {repaired['expected_output']}",
        )
    )
    contract = _DATASET_OUTPUT_CONTRACTS.get(dataset_key, ())
    rule_ids = tuple(rule_id for rule_id, _text in contract)
    if contract:
        managed_contract_version = DELEGATION_CONTRACT_VERSION
        managed = "\n".join(
            (
                f"System-managed output contract ({managed_contract_version}):",
                *(f"- {text}" for _rule_id, text in contract),
            )
        )
        prompt = director_prompt + "\n\n" + managed
        contract_version: str | None = managed_contract_version
    else:
        prompt = director_prompt
        contract_version = None
    return (
        DelegationCompilation(
            prompt=prompt,
            director_fields=repaired,
            dataset=dataset_key,
            contract_version=contract_version,
            contract_rule_ids=rule_ids,
            field_repairs=tuple(repairs),
        ),
        None,
    )


def _shorten_at_boundary(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    candidate = text[:limit].rstrip()
    boundary = max(candidate.rfind(" "), candidate.rfind(";"), candidate.rfind(","))
    if boundary >= max(16, limit // 2):
        candidate = candidate[:boundary].rstrip(" ,;")
    return candidate


def _fit_total_delegation_limit(
    fields: dict[str, str],
    repairs: list[DelegationFieldRepair],
) -> None:
    def total() -> int:
        return len(" ".join(fields.values()))

    minimums = {"scope": 48, "objective": 64, "expected_output": 40, "role": 24}
    for name in ("scope", "objective", "expected_output", "role"):
        excess = total() - DELEGATION_TOTAL_LIMIT
        if excess <= 0:
            return
        current = fields[name]
        target = max(minimums[name], len(current) - excess)
        if target >= len(current):
            continue
        shortened = _shorten_at_boundary(current, target)
        fields[name] = shortened
        repairs.append(
            DelegationFieldRepair(
                name,
                "total_length",
                len(current),
                len(shortened),
            )
        )
